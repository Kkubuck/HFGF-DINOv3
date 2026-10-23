# Copyright (c) Facebook, Inc. and its affiliates.
"""Detectron2 modules of HFGF-DINOv3; importing this package registers them."""

from .backbone import HFGFViTSplit
from .hfgf_dinov3 import HFGF_DINOv3
from .mask_former_head import HFGFMaskFormerHead
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder import MultiScaleMaskedTransformerDecoder

__all__ = [
    "HFGFViTSplit",
    "HFGF_DINOv3",
    "HFGFMaskFormerHead",
    "MSDeformAttnPixelDecoder",
    "MultiScaleMaskedTransformerDecoder",
]
