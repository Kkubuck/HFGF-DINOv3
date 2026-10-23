# ------------------------------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
# Modified by Zuyao You (https://github.com/geshang777)
# ------------------------------------------------------------------------------------------------------
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from .criterion import SetCriterion
from .matcher import HungarianMatcher


@META_ARCH_REGISTRY.register()
class HFGF_DINOv3(nn.Module):
    """HFGF-DINOv3 camouflaged object segmentation model."""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float, ...],
        pixel_std: Tuple[float, ...],
        baphf_weight: float,
        baphf_start: int,
        semantic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.size_divisibility = self.backbone.size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.test_topk_per_image = test_topk_per_image
        self.baphf_weight = baphf_weight
        self.baphf_start = baphf_start
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        loss_cfg = cfg.MODEL.MASK_FORMER
        matcher = HungarianMatcher(
            cost_class=loss_cfg.CLASS_WEIGHT,
            cost_mask=loss_cfg.MASK_WEIGHT,
            cost_dice=loss_cfg.DICE_WEIGHT,
            num_points=loss_cfg.TRAIN_NUM_POINTS,
        )
        weight_dict = {
            "loss_ce": loss_cfg.CLASS_WEIGHT,
            "loss_mask": loss_cfg.MASK_WEIGHT,
            "loss_dice": loss_cfg.DICE_WEIGHT,
            "loss_bbox": loss_cfg.BBOX_WEIGHT,
            "loss_giou": loss_cfg.GIOU_WEIGHT,
        }
        if loss_cfg.DEEP_SUPERVISION:
            aux_weights = {}
            for i in range(loss_cfg.DEC_LAYERS - 1):
                aux_weights.update({f"{key}_{i}": value for key, value in weight_dict.items()})
            weight_dict.update(aux_weights)
        losses = ["masks", "labels", "boxes"]
        if loss_cfg.BAPHF_WEIGHT > 0:
            weight_dict["loss_bap_hf"] = loss_cfg.BAPHF_WEIGHT
            losses.append("bap_hf")
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=loss_cfg.NO_OBJECT_WEIGHT,
            losses=losses,
            num_points=loss_cfg.TRAIN_NUM_POINTS,
            oversample_ratio=loss_cfg.OVERSAMPLE_RATIO,
            importance_sample_ratio=loss_cfg.IMPORTANCE_SAMPLE_RATIO,
            baphf_highpass=loss_cfg.BAPHF_HIGHPASS,
            baphf_band_kernel=loss_cfg.BAPHF_BAND_KERNEL,
        )
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": loss_cfg.NUM_OBJECT_QUERIES,
            "sem_seg_postprocess_before_inference": (
                loss_cfg.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE or loss_cfg.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": loss_cfg.TEST.SEMANTIC_ON,
            "instance_on": loss_cfg.TEST.INSTANCE_ON,
            "test_topk_per_image": loss_cfg.TEST.DETECTIONS_PER_IMAGE,
            "baphf_weight": loss_cfg.BAPHF_WEIGHT,
            "baphf_start": loss_cfg.BAPHF_START_ITER,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """Return weighted training losses or per-image segmentation predictions."""
        images = [sample["image"].to(self.device) for sample in batched_inputs]
        images = [(image - self.pixel_mean) / self.pixel_std for image in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features, feature_mask = self.backbone(images.tensor)
        training_context = {"feature_mask": feature_mask, "is_train": self.training}
        outputs = self.sem_seg_head(features, training_context=training_context)

        if self.training:
            gt_instances = [sample["instances"].to(self.device) for sample in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)
            losses = self.criterion(outputs, targets)
            try:
                current_iter = get_event_storage().iter
            except AssertionError:
                current_iter = 0
            weight_dict = self.criterion.weight_dict.copy()
            # BaP-HF is computed throughout training, with zero weight before warmup.
            weight_dict["loss_bap_hf"] = (
                self.baphf_weight if current_iter >= self.baphf_start else 0.0
            )
            for name in list(losses):
                if name in weight_dict:
                    losses[name] *= weight_dict[name]
                else:
                    losses.pop(name)
            return losses

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = F.interpolate(
            outputs["pred_masks"],
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del outputs
        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            result = {}
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)
            if self.semantic_on:
                prediction = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    prediction = retry_if_cuda_oom(sem_seg_postprocess)(
                        prediction, image_size, height, width
                    )
                result["sem_seg"] = prediction
            if self.instance_on:
                result["instances"] = retry_if_cuda_oom(self.instance_inference)(
                    mask_cls_result, mask_pred_result
                )
            processed_results.append(result)
        return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            gt_boxes = targets_per_image.gt_boxes.tensor
            ratio = torch.tensor([w_pad, h_pad, w_pad, h_pad]).to(gt_boxes.device)[None, :]
            gt_boxes = gt_boxes / ratio
            xc, yc, w, h = (
                (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2,
                gt_boxes[:, 2] - gt_boxes[:, 0],
                gt_boxes[:, 3] - gt_boxes[:, 1],
            )
            gt_boxes = torch.stack([xc, yc, w, h]).permute(1, 0)

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": gt_boxes,
                }
            )
        return new_targets

    def _merge_mask(self, instances):
        """Combine the foreground/background query masks into a binary mask."""
        selected_classes = []
        selected_masks = []
        previous_class = -1
        for index, class_id in enumerate(instances.pred_classes.tolist()):
            if len(selected_classes) < 2 and instances.scores[index] > 0.5:
                if class_id != previous_class:
                    selected_classes.append(class_id)
                    selected_masks.append(instances.pred_masks[index])
                    previous_class = class_id
        if len(selected_classes) < 2:
            scores = instances.scores.tolist()
            best_index = scores.index(max(scores))
            if instances.pred_classes[best_index] == 0:
                return (instances.pred_masks[best_index] > 0.5).int()
            return (instances.pred_masks[best_index] < 0.5).int()

        background_mask, foreground_mask = selected_masks
        if selected_classes[0] == 0:
            background_mask, foreground_mask = foreground_mask, background_mask
        return (foreground_mask > background_mask).int()

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (
            torch.arange(self.sem_seg_head.num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        # Query probabilities at the original image resolution.
        result.pred_masks = (mask_pred.sigmoid()).float()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(
            1
        ) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        merged_mask = self._merge_mask(result)
        result = Instances(image_size)
        result.pred_masks = merged_mask
        return result
