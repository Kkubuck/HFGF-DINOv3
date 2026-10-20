# Copyright (c) Facebook, Inc. and its affiliates.
"""Pixel-decoder construction through the Detectron2 registry."""
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


def build_pixel_decoder(cfg, input_shape):
    """Build the configured pixel decoder with a forward_features interface."""
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    if not callable(getattr(model, "forward_features", None)):
        raise ValueError(f"Pixel decoder {name} must implement forward_features")
    return model
