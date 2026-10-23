# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted from Mask2Former and FOCUS:
# https://github.com/facebookresearch/Mask2Former
# Original FOCUS modifications by Zuyao You (https://github.com/geshang777).
"""Pixel and query decoding head for HFGF-DINOv3."""

from typing import Dict

from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .transformer_decoder import build_transformer_decoder


def build_pixel_decoder(cfg, input_shape):
    """Build the configured pixel decoder with a forward_features interface."""
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    if not callable(getattr(model, "forward_features", None)):
        raise ValueError(f"Pixel decoder {name} must implement forward_features")
    return model


@SEM_SEG_HEADS_REGISTRY.register()
class HFGFMaskFormerHead(nn.Module):
    """Fuse backbone features and produce foreground/background predictions."""

    _version = 2

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        transformer_predictor: nn.Module,
    ):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE != "multi_scale_pixel_decoder":
            raise ValueError("HFGF-DINOv3 requires multi_scale_pixel_decoder features.")
        return {
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "transformer_predictor": build_transformer_decoder(
                cfg, cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, mask_classification=True
            ),
        }

    def forward(self, features, training_context=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        return self.predictor(multi_scale_features, mask_features, training_context)
