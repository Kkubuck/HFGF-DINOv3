# Copyright (c) Facebook, Inc. and its affiliates.
"""Registry and constructor for the masked transformer decoder."""

from detectron2.utils.registry import Registry

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)
