"""DINOv3 ViT-Split backbone with an HFGF feature pyramid (Sec. 3.1-3.2 of the paper)."""

import copy
import importlib
import itertools
import math
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from .hfgf import HFGFBlock, SpatialConv


class LayerSelector(nn.Module):
    """Learned sparse mixtures of intermediate transformer features."""

    def __init__(self, n_layers: int, k: int, out_num: int, temperature: float = 0.5):
        super().__init__()
        self.n = n_layers
        self.k = k
        self.out_num = out_num
        self.temperature = temperature
        self.scores = nn.Parameter(torch.rand(n_layers, out_num))

    def forward(self, x):
        batch, n_layers, _, _, _ = x.shape
        if n_layers != self.n:
            raise ValueError(f"Expected {self.n} transformer layers, received {n_layers}")
        scores = self.scores.unsqueeze(0).expand(batch, -1, -1)
        soft = F.softmax(scores / self.temperature, dim=1)
        topk_idx = torch.topk(soft, self.k, dim=1).indices
        sparse = torch.zeros_like(soft)
        batch_idx = torch.arange(batch, device=x.device).view(batch, 1, 1).expand_as(topk_idx)
        out_idx = torch.arange(self.out_num, device=x.device).view(1, 1, self.out_num).expand_as(topk_idx)
        sparse[batch_idx, topk_idx, out_idx] = soft[batch_idx, topk_idx, out_idx]
        gates = sparse - soft.detach() + soft
        return gates / (gates.sum(dim=1, keepdim=True) + 1e-6)


def _get_pad(size, multiple):
    pad_size = math.ceil(size / multiple) * multiple - size
    left = pad_size // 2
    return left, pad_size - left


def _center_pad(x, multiple):
    pads = list(itertools.chain.from_iterable(_get_pad(m, multiple) for m in x.shape[:1:-1]))
    return F.pad(x, pads)


def _load_dinov3(model_name):
    """Construct DINOv3 from the bundled runtime (or DINOV3_LOCAL_DIR); never download weights."""
    repo_root = Path(__file__).resolve().parents[3]
    local_dir = os.environ.get("DINOV3_LOCAL_DIR")
    candidates = [Path(local_dir).expanduser()] if local_dir else []
    candidates.append(repo_root / "third_party" / "dinov3")
    for directory in candidates:
        if (directory / "dinov3" / "hub" / "backbones.py").is_file():
            # Import the supported PyTorch modules explicitly. No global stubs or
            # replacement decorators are installed.
            importlib.import_module("torch._dynamo")
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
            backbones = importlib.import_module("dinov3.hub.backbones")
            return getattr(backbones, model_name)(pretrained=False)
    raise FileNotFoundError(
        "DINOv3 runtime not found in third_party/dinov3. Set "
        "DINOV3_LOCAL_DIR to a DINOv3 repository checkout."
    )


def _build_compression(in_channels, out_channels):
    layers = []
    channels = in_channels
    while channels > out_channels:
        next_channels = max(out_channels, math.ceil(channels / 2))
        layers.extend([
            nn.Conv2d(channels, next_channels, kernel_size=1),
            nn.GELU(),
            SpatialConv(next_channels, next_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ])
        channels = next_channels
    return nn.Sequential(*layers) if layers else nn.Sequential(nn.Identity())


def _projection(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.SyncBatchNorm(out_channels),
        nn.GELU(),
    )


@BACKBONE_REGISTRY.register()
class HFGFViTSplit(Backbone):
    """DINOv3 intermediate features, a split task head, and an HFGF pyramid.

    ``TUNING_TYPE`` controls backbone trainability. The split head contains
    independent block copies.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        options = cfg.MODEL.VITSPLIT
        self.select_layers = list(options.SELECT_LAYERS)
        self.ms_layers = list(options.MS_LAYERS)
        self.out_num = options.OUT_NUM
        self.channels = options.CHANNELS
        self.frozen_dim = options.FROZEN_DIM
        self.fusion_out_dim = options.get("FUSION_OUT_DIM", self.channels)
        self.fusion_mode = options.FUSION_MODE
        self.hf_use_mask = options.HFGF_USE_MASK
        self.hf_gain = options.HFGF_GAIN
        self.hf_enable_spatial = options.get("HFGF_ENABLE_SPATIAL", True)
        self.hf_enable_freq = options.get("HFGF_ENABLE_FREQ", True)
        self.use_split_checkpoint = options.get("USE_SPLIT_CHECKPOINT", False)
        self.split_checkpoint_segments = options.get("SPLIT_CHECKPOINT_SEGMENTS", 2)
        self.generate_feature_mask = cfg.MODEL.MASK_FORMER.get("USE_FEATURE_MASK_ATTN", True)
        if self.fusion_mode not in {"hfgf", "residual"}:
            raise ValueError("FUSION_MODE must be 'hfgf' (HFGF) or 'residual'")
        if len(self.ms_layers) != 4 or not self.select_layers:
            raise ValueError("MS_LAYERS must contain four indices and SELECT_LAYERS must be nonempty")
        if self.fusion_out_dim != self.channels:
            raise ValueError("HFGF requires FUSION_OUT_DIM to equal CHANNELS")
        if options.BACKBONE_SIZE != "v3-h+":
            raise ValueError("This implementation supports BACKBONE_SIZE='v3-h+' (DINOv3 ViT-H+) only")
        backbone = _load_dinov3("dinov3_vith16plus")
        n_layers = len(backbone.blocks)
        if not 0 < self.out_num <= n_layers:
            raise ValueError("OUT_NUM must be between one and the number of backbone layers")
        if any(index < 0 or index >= n_layers for index in self.select_layers + self.ms_layers):
            raise ValueError("SELECT_LAYERS and MS_LAYERS must reference existing backbone blocks")
        backbone.forward = partial(
            backbone.get_intermediate_layers,
            n=list(range(n_layers)),
            reshape=True,
            norm=True,
            return_class_token=False,
        )
        tuning_type = options.TUNING_TYPE
        if tuning_type in ("all", "frozen"):
            for parameter in backbone.parameters():
                parameter.requires_grad = tuning_type == "all"
        elif isinstance(tuning_type, (list, tuple)):
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            for index in tuning_type:
                for parameter in backbone.blocks[index].parameters():
                    parameter.requires_grad = True
        else:
            raise ValueError(f"Unsupported TUNING_TYPE: {tuning_type}")
        self.split_head = nn.Sequential(*[copy.deepcopy(backbone.blocks[i]) for i in self.select_layers])
        if options.get("FREEZE_GATE", False):
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        self.learnable_gate = LayerSelector(n_layers, self.out_num, self.out_num, temperature=0.5)
        self.embed_dim = backbone.blocks[0].attn.qkv.in_features
        self.layer_projection = _build_compression(self.embed_dim * self.out_num, self.frozen_dim)
        self.fusion_net = nn.Sequential(
            nn.Conv2d(self.frozen_dim + self.embed_dim, self.fusion_out_dim, kernel_size=1),
            nn.GELU(),
            SpatialConv(self.fusion_out_dim, self.fusion_out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.prior_projs = nn.ModuleList([
            _projection(self.embed_dim, self.fusion_out_dim) for _ in range(4)
        ])
        if self.fusion_mode == "hfgf":
            self.hfgf_blocks = nn.ModuleList([
                HFGFBlock(
                    self.channels,
                    self.channels,
                    hf_gamma_init=options.HFGF_GAMMA_INIT,
                    enable_spatial=self.hf_enable_spatial,
                    enable_freq=self.hf_enable_freq,
                )
                for _ in range(4)
            ])
            self.register_buffer("sobel_x", torch.tensor(
                [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32
            ))
            self.register_buffer("sobel_y", torch.tensor(
                [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32
            ))
        else:
            self.task_adapters = nn.ModuleList([
                _projection(self.fusion_out_dim, self.fusion_out_dim) for _ in range(4)
            ])
            self.fusion_heads = nn.ModuleList([
                _projection(self.fusion_out_dim * 2, self.fusion_out_dim) for _ in range(4)
            ])
        self.backbone = backbone
        self.patch_size = backbone.patch_embed.patch_size[0]
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_channels = {name: self.fusion_out_dim for name in self._out_features}
        self._out_feature_strides = dict(zip(self._out_features, [4, 8, 16, 32]))

    def _reshape_feat(self, feat, height, width):
        """Convert intermediate patch features to contiguous BCHW format."""
        if feat.dim() == 4:
            if feat.shape[1] == self.embed_dim:
                return feat.contiguous()
            if feat.shape[-1] == self.embed_dim:
                return feat.permute(0, 3, 1, 2).contiguous()
            raise ValueError(f"Unexpected feature shape: {feat.shape}")
        batch, length, channels = feat.shape
        if length < height * width:
            raise ValueError(f"Token length {length} is smaller than grid {height} x {width}")
        return feat[:, :height * width].view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _run_split_head(self, tokens):
        if not self.use_split_checkpoint:
            return self.split_head(tokens)
        return checkpoint_sequential(
            list(self.split_head.children()),
            self.split_checkpoint_segments,
            tokens,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def _grad_map(self, image, size):
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x.to(gray.device), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(gray.device), padding=1)
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return magnitude / (magnitude.mean(dim=(2, 3), keepdim=True) + 1e-6)

    @staticmethod
    def _feature_mask(last_feat):
        """PCA foreground mask that guides the first decoder attention layer."""
        batch, channels, height, width = last_feat.shape
        patches = last_feat.permute(0, 2, 3, 1).contiguous().view(batch, -1, channels)
        patches_np = patches.detach().cpu().numpy().reshape(-1, channels)
        try:
            reduced = PCA(n_components=1).fit_transform(patches_np)
            normalized = minmax_scale(reduced)
            return (normalized.reshape(batch, height, width) > 0.6).astype(np.uint8)
        except Exception:
            # Fall back to a feature-mean mask when PCA fails (e.g. nonfinite data).
            score = last_feat.mean(dim=1)
            mean = score.mean(dim=(1, 2), keepdim=True)
            std = score.std(dim=(1, 2), keepdim=True) + 1e-6
            return (torch.sigmoid((score - mean) / std) > 0.5).float().detach().cpu().numpy()

    def forward(self, x):
        padded = _center_pad(x, self.patch_size)
        tokens = self.backbone(padded)
        _, _, height, width = padded.shape
        h, w = math.ceil(height / self.patch_size), math.ceil(width / self.patch_size)
        features = torch.stack([self._reshape_feat(feature, h, w) for feature in tokens], dim=1)
        gates = self.learnable_gate(features)
        selected = torch.einsum("bndhw,bnk->bkhwd", features, gates)
        selected = selected.permute(0, 1, 4, 2, 3).contiguous()
        batch, n_selected, channels, fh, fw = selected.shape
        compressed = self.layer_projection(selected.reshape(batch, n_selected * channels, fh, fw))

        split_input = tokens[self.select_layers[0]]
        if split_input.dim() == 4:
            if split_input.shape[-1] == self.embed_dim:
                b, sh, sw, dim = split_input.shape
                split_input = split_input.view(b, sh * sw, dim)
            elif split_input.shape[1] == self.embed_dim:
                b, dim, sh, sw = split_input.shape
                split_input = split_input.permute(0, 2, 3, 1).contiguous().view(b, sh * sw, dim)
            else:
                raise ValueError(f"Unexpected split feature shape: {split_input.shape}")
        tuned_tokens = self._run_split_head(split_input)
        tuned = tuned_tokens[:, :h * w].reshape(batch, h, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        priors = [
            projection(self._reshape_feat(tokens[index], h, w))
            for index, projection in zip(self.ms_layers, self.prior_projs)
        ]
        fused = self.fusion_net(torch.cat([compressed, tuned], dim=1))
        target_sizes = [(math.ceil(height / stride), math.ceil(width / stride)) for stride in [4, 8, 16, 32]]
        prior_mean = torch.mean(torch.stack(priors, dim=0), dim=0)
        prior_scales = [F.interpolate(prior_mean, size=size, mode="bilinear", align_corners=False) for size in target_sizes]

        if self.fusion_mode == "hfgf":
            last_feat = self._reshape_feat(tokens[-1], h, w)
            gradients = [self._grad_map(padded, size) for size in target_sizes] if self.hf_enable_spatial else [None] * 4
            if self.hf_use_mask and self.hf_enable_spatial:
                score = last_feat.mean(dim=1, keepdim=True)
                mean = score.mean(dim=(2, 3), keepdim=True)
                std = score.std(dim=(2, 3), keepdim=True) + 1e-6
                soft_mask = torch.sigmoid((score - mean) / std).detach()
                masks = [F.interpolate(soft_mask, size=size, mode="bilinear", align_corners=False) for size in target_sizes]
                gradients = [gradient * mask for gradient, mask in zip(gradients, masks)]
            if self.hf_enable_spatial:
                gradients = [gradient * float(self.hf_gain) for gradient in gradients]
            outputs = {}
            feature = fused
            for index in [3, 2, 1, 0]:
                feature = F.interpolate(feature, size=target_sizes[index], mode="bilinear", align_corners=False)
                feature = self.hfgf_blocks[index](feature, prior_scales[index], gradients[index])
                outputs[self._out_features[index]] = feature
            outputs = {name: outputs[name] for name in self._out_features}
        else:
            task_scales = [
                adapter(F.interpolate(fused, size=size, mode="bilinear", align_corners=False))
                for size, adapter in zip(target_sizes, self.task_adapters)
            ]
            outputs = {
                name: head(torch.cat([prior, task], dim=1))
                for name, prior, task, head in zip(self._out_features, prior_scales, task_scales, self.fusion_heads)
            }
        outputs["fused"] = fused
        feature_mask = self._feature_mask(self._reshape_feat(tokens[-1], h, w)) if self.generate_feature_mask else None
        return outputs, feature_mask

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }
