# Copyright (c) Facebook, Inc. and its affiliates.
from .masked_transformer_decoder import (
    TRANSFORMER_DECODER_REGISTRY,
    MultiScaleMaskedTransformerDecoder,
    build_transformer_decoder,
)

__all__ = [
    "TRANSFORMER_DECODER_REGISTRY",
    "MultiScaleMaskedTransformerDecoder",
    "build_transformer_decoder",
]
