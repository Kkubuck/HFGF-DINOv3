"""HFGF-DINOv3: High-Frequency Guided Fusion and phase-aware loss for camouflaged object detection."""

from . import data, modeling  # noqa: F401  (registers datasets and Detectron2 modules)
from .config import add_hfgf_dinov3_config, load_config
from .data import HFGFDatasetMapper
from .modeling import HFGF_DINOv3

__all__ = ["HFGF_DINOv3", "HFGFDatasetMapper", "add_hfgf_dinov3_config", "load_config"]
