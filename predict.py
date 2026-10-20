#!/usr/bin/env python3
"""Generate binary camouflage masks for an image or a directory of images."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from detectron2.data import DatasetMapper
from detectron2.modeling import build_model
from detectron2.utils.env import seed_all_rng

from hfgf_dinov3 import load_config
from hfgf_dinov3.checkpoint import load_trained_weights


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(source):
    source = Path(source)
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return source.parent, [source]
    if source.is_dir():
        paths = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if paths:
            return source, paths
    raise ValueError(f"No supported images found at {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default="configs/hfgf_dinov3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", required=True, help="Output directory for binary PNG masks")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config_file, ["MODEL.DEVICE", args.device])
    seed_all_rng(cfg.SEED if cfg.SEED >= 0 else 42)
    base, paths = image_paths(args.input)
    output_dir = Path(args.output)
    if Path(args.input).is_dir():
        try:
            output_dir.resolve().relative_to(base.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("Place the output directory outside the input directory.")
    destinations = [output_dir / path.relative_to(base).with_suffix(".png") for path in paths]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Input images have duplicate stems; PNG outputs would overwrite each other.")
    if any(destination.resolve() == source.resolve() for source, destination in zip(paths, destinations)):
        raise ValueError("Choose an output directory different from the input images.")
    model = build_model(cfg)
    load_trained_weights(model, args.checkpoint)
    model.eval()
    mapper = DatasetMapper(cfg, is_train=False)
    with torch.no_grad():
        for source, destination in tqdm(zip(paths, destinations), total=len(paths), desc="Predict"):
            sample = mapper({"file_name": str(source)})
            prediction = model([sample])[0]["instances"].pred_masks
            if prediction.ndim != 2:
                raise ValueError(f"Expected a merged 2D mask, got {tuple(prediction.shape)}")
            mask = prediction.to(dtype=torch.uint8).cpu().numpy()
            if not np.isin(mask, (0, 1)).all():
                raise ValueError("The model did not return a binary foreground mask")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask * 255).save(destination)
    print(f"Saved {len(paths)} masks to {output_dir}")


if __name__ == "__main__":
    main()
