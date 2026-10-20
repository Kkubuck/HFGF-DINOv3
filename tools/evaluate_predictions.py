#!/usr/bin/env python3
"""Evaluate saved binary PNG predictions using the same protocol as train_net.py."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cod_val", "cod10k_val", "nc4k", "chameleon"))
    parser.add_argument("--predictions", required=True, type=Path, help="Directory of <image_stem>.png binary masks")
    parser.add_argument("--dataset-root", type=Path, default=Path(os.getenv("DETECTRON2_DATASETS", "datasets")))
    parser.add_argument("--output", type=Path, help="Metrics directory; defaults to the prediction directory")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    os.environ["DETECTRON2_DATASETS"] = str(args.dataset_root)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from detectron2.data import DatasetCatalog
    from detectron2.structures import Instances
    from hfgf_dinov3.evaluation import CODEvaluator

    records = DatasetCatalog.get(args.dataset)
    expected_names = {Path(record["file_name"]).stem + ".png" for record in records}
    actual_names = {path.name for path in args.predictions.glob("*.png")}
    if expected_names != actual_names:
        raise ValueError(
            f"Prediction files do not match {args.dataset}: "
            f"{len(expected_names - actual_names)} missing, "
            f"{len(actual_names - expected_names)} unexpected PNG files"
        )
    evaluator = CODEvaluator(
        args.dataset, distributed=False,
        output_dir=str(args.output or args.predictions), num_workers=args.num_workers,
    )
    for record in records:
        path = args.predictions / (Path(record["file_name"]).stem + ".png")
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))
        if not np.isin(mask, (0, 255)).all():
            raise ValueError(f"Expected a binary 0/255 prediction: {path}")
        instances = Instances(mask.shape)
        instances.pred_masks = torch.from_numpy(mask > 128)[None]
        instances.scores = torch.ones(1)
        evaluator.process([{"image_id": record["image_id"]}], [{"instances": instances}])
    results = evaluator.evaluate()["cod"]
    print(args.dataset + ": " + ", ".join(f"{name}={value:.6f}" for name, value in results.items()))


if __name__ == "__main__":
    main()
