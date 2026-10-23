"""Strict model loading and DINOv3 initialization."""
import json
from pathlib import Path

import torch


def read_state_dict(path):
    """Read model tensors from a PyTorch checkpoint without loading Python objects."""
    path = Path(path)
    if path.suffix not in (".pth", ".pt"):
        raise ValueError("Expected a PyTorch .pth or .pt checkpoint")
    value = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    if isinstance(value, dict) and "model" in value:
        value = value["model"]
    elif isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict) or not value:
        raise ValueError("Checkpoint must contain a nonempty tensor state dictionary")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in value.items()):
        raise ValueError("Every state dictionary entry must be a named tensor")
    return value


def validate_state_dict(module, state):
    """Check all keys, shapes and dtypes before modifying the model."""
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatches = []
    for key in sorted(set(expected).intersection(state)):
        value, target = state[key], expected[key]
        if not isinstance(value, torch.Tensor):
            mismatches.append(key + ": checkpoint value is not a tensor")
        elif value.shape != target.shape or value.dtype != target.dtype:
            mismatches.append("{}: checkpoint {} {}, model {} {}".format(
                key, tuple(value.shape), value.dtype, tuple(target.shape), target.dtype
            ))
    if missing or unexpected or mismatches:
        raise ValueError("Checkpoint/model mismatch:\n" + json.dumps({
            "missing": missing, "unexpected": unexpected, "mismatches": mismatches
        }, indent=2))
    return {"tensors": len(expected), "parameters_and_buffers": sum(v.numel() for v in expected.values())}


def load_trained_weights(model, path):
    """Load an exact complete model; missing or unexpected tensors are errors."""
    state = read_state_dict(path)
    report = validate_state_dict(model, state)
    model.load_state_dict(state, strict=True)
    return report


def initialize_backbone(model, path):
    """Initialize only the DINOv3 backbone from the official pretrained weights.

    The split task head is built from the randomly initialized blocks before the
    backbone weights are loaded, so it keeps its own random initialization.
    This is initialization for a new run, not training-state resumption.
    """
    backbone = model.backbone.backbone
    state = read_state_dict(path)
    # Official H/16+ stores this fixed buffer in BF16. The paper's loader
    # copied it into the model's FP32 buffer without losing precision.
    target = backbone.state_dict().get("rope_embed.periods")
    periods = state.get("rope_embed.periods")
    if (
        periods is not None and target is not None
        and periods.dtype == torch.bfloat16 and target.dtype == torch.float32
    ):
        state["rope_embed.periods"] = periods.float()
    report = validate_state_dict(backbone, state)
    backbone.load_state_dict(state, strict=True)
    report["split_head_initialized_from_pretrained"] = False
    return report
