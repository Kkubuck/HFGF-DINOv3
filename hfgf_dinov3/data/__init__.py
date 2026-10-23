"""COD benchmark registration and the training data mapper."""

from . import datasets  # noqa: F401  (registers CAMO, COD10K, NC4K and CHAMELEON)
from .dataset_mapper import HFGFDatasetMapper

__all__ = ["HFGFDatasetMapper"]
