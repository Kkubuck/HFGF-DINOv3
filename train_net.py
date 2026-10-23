#!/usr/bin/env python3
"""Train or evaluate HFGF-DINOv3."""

# Training utilities adapted from Mask2Former and FOCUS; see NOTICE.
import copy
import itertools
import json
import os
from pathlib import Path
import weakref

import torch
from torch import nn

from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    AMPTrainer, DefaultTrainer, SimpleTrainer, TrainerBase,
    default_argument_parser, default_setup, launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.utils import comm
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

from hfgf_dinov3 import HFGFDatasetMapper, load_config
from hfgf_dinov3.checkpoint import initialize_backbone, load_trained_weights
from hfgf_dinov3.evaluation import CODEvaluator


class HFGFTrainer(DefaultTrainer):
    """Detectron2 trainer with AdamW, per-module learning rates and full-model gradient clipping."""

    def __init__(self, cfg):
        TrainerBase.__init__(self)
        cfg = self.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        # DINOv3 contains parameters such as mask_token that are inactive for
        # unmasked dense prediction. Make their DDP handling explicit.
        model = create_ddp_model(
            model, broadcast_buffers=False, find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer,
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=HFGFDatasetMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return CODEvaluator(
            dataset_name,
            output_dir=output_folder or os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name),
            num_workers=cfg.TEST.METRIC_WORKERS,
        )

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        norm_types = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
            nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
            nn.InstanceNorm3d, nn.LayerNorm, nn.LocalResponseNorm,
        )
        groups, seen = [], set()
        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        for module_name, module in model.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad or parameter in seen:
                    continue
                seen.add(parameter)
                options = copy.copy(defaults)
                # The paper's recipe applies this multiplier to the entire
                # ViT-Split/HFGF backbone, including its adaptation modules.
                if "backbone" in module_name:
                    options["lr"] *= cfg.SOLVER.BACKBONE_MULTIPLIER
                if parameter_name in ("relative_position_bias_table", "absolute_pos_embed"):
                    options["weight_decay"] = 0.0
                if isinstance(module, norm_types):
                    options["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
                if isinstance(module, nn.Embedding):
                    options["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_EMBED
                groups.append({"params": [parameter], **options})

        clip = cfg.SOLVER.CLIP_GRADIENTS
        if clip.ENABLED and clip.CLIP_TYPE != "full_model":
            raise ValueError("This recipe uses full_model gradient clipping.")
        max_norm = clip.CLIP_VALUE if clip.ENABLED else 0.0
        norm_type = clip.NORM_TYPE

        class ClippedAdamW(torch.optim.AdamW):
            def step(self, closure=None):
                if max_norm > 0:
                    parameters = itertools.chain.from_iterable(group["params"] for group in self.param_groups)
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
                return super().step(closure=closure)

        return ClippedAdamW(groups, lr=cfg.SOLVER.BASE_LR)

    def resume_or_load(self, resume=True):
        model = self.model.module if hasattr(self.model, "module") else self.model
        if resume and self.checkpointer.has_checkpoint():
            # Validate the model strictly before restoring optimizer/scheduler/
            # scaler/iteration through Detectron2's normal resume mechanism.
            load_trained_weights(model, self.checkpointer.get_checkpoint_file())
            result = super().resume_or_load(resume=True)
            saved_max_iter = getattr(self.scheduler, "max_iters", self.max_iter)
            if saved_max_iter != self.max_iter:
                raise ValueError(
                    f"--resume requires the original SOLVER.MAX_ITER={saved_max_iter}; "
                    f"received {self.max_iter}. For a new training schedule, use "
                    "MODEL.WEIGHTS without --resume and choose a new OUTPUT_DIR."
                )
            return result
        if self.cfg.MODEL.WEIGHTS:
            load_trained_weights(model, self.cfg.MODEL.WEIGHTS)
        elif self.cfg.MODEL.BACKBONE_WEIGHTS:
            initialize_backbone(model, self.cfg.MODEL.BACKBONE_WEIGHTS)
        else:
            raise ValueError("Training requires MODEL.BACKBONE_WEIGHTS or MODEL.WEIGHTS.")
        self.start_iter = 0


def main(args):
    cfg = load_config(args.config_file, args.opts)
    if cfg.TEST.AUG.ENABLED:
        raise ValueError("The paper's evaluation protocol does not use test-time augmentation.")
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="hfgf_dinov3")
    if args.eval_only:
        if not cfg.MODEL.WEIGHTS:
            raise ValueError("Evaluation requires MODEL.WEIGHTS pointing to a trained checkpoint.")
        model = HFGFTrainer.build_model(cfg)
        load_trained_weights(model, cfg.MODEL.WEIGHTS)
        with EventStorage(), torch.no_grad():
            results = HFGFTrainer.test(cfg, model)
        if comm.is_main_process():
            output = Path(cfg.OUTPUT_DIR)
            output.mkdir(parents=True, exist_ok=True)
            (output / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
        return results
    trainer = HFGFTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main, args.num_gpus, num_machines=args.num_machines,
        machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,),
    )
