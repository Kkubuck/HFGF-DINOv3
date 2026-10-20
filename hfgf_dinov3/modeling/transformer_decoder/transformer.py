# Copyright (c) Facebook, Inc. and its affiliates.
"""Shared transformer utilities adapted from DETR and Mask2Former."""

import copy

from torch import nn
from torch.nn import functional as F


def _get_clones(module, count):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise ValueError(f"Unsupported activation: {activation}")
