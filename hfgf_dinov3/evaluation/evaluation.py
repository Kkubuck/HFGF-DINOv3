"""COD evaluation of the model's merged binary foreground mask."""

import contextlib
import csv
import io
import itertools
import json
import logging
import multiprocessing as mp
import os
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .utils import calc_metrics


METRIC_NAMES = ("SM", "EM", "WFM", "MAE", "FM")


def evaluate_mask_pair(task):
    """Decode and evaluate one binary prediction and foreground annotation."""
    prediction = np.asarray(mask_util.decode(task["pred_rle"]), dtype=np.float32)
    target = np.asarray(mask_util.decode(task["gt_rle"]), dtype=np.float32)
    if prediction.ndim == 3:
        prediction = prediction[:, :, 0]
    if target.ndim == 3:
        target = target[:, :, 0]
    if prediction.shape != target.shape:
        tensor = torch.from_numpy(prediction)[None, None]
        prediction = F.interpolate(tensor, size=target.shape, mode="nearest")[0, 0].numpy()
    values = calc_metrics(prediction, target)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite metrics for image {task['image_id']}: {values}")
    return {"image_id": task["image_id"], **dict(zip(METRIC_NAMES, map(float, values)))}


class CODEvaluator(DatasetEvaluator):
    """Evaluate CAMO, COD10K, NC4K, or CHAMELEON with the retained protocol.

    The model's merged binary foreground mask is evaluated against the first
    foreground COCO annotation. Metrics are averaged over images in float64.
    Missing annotations and computation errors fail explicitly so a partial
    dataset cannot silently produce a benchmark score.
    """

    def __init__(self, dataset_name, distributed=True, output_dir=None, num_workers=0):
        self.dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._num_workers = max(0, int(num_workers))
        self._logger = logging.getLogger(__name__)
        metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        # COD10K JSON also contains NonCAM images, excluded by registration.
        records = DatasetCatalog.get(dataset_name)
        self._expected_ids = {record["image_id"] for record in records}
        if len(self._expected_ids) != len(records):
            raise ValueError(f"Duplicate image IDs in {dataset_name} registration")
        if not self._expected_ids.issubset(self._coco_api.imgs):
            raise ValueError(f"Registered images are missing from {json_file}")
        self.reset()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for item, output in zip(inputs, outputs):
            instances = output["instances"]
            masks = instances.pred_masks
            if masks.ndim == 3:
                if masks.shape[0] == 0:
                    mask = torch.zeros(
                        item["image"].shape[-2:], dtype=torch.uint8, device=masks.device,
                    )
                else:
                    index = int(instances.scores.argmax().item()) if instances.has("scores") else 0
                    mask = masks[index]
            else:
                mask = masks
            mask = mask.detach().to(torch.uint8).cpu().numpy()
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            self._predictions.append({"image_id": item["image_id"], "pred_rle": rle})

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            gathered = comm.gather(self._predictions, dst=0)
            if not comm.is_main_process():
                return {}
            predictions = list(itertools.chain.from_iterable(gathered))
        else:
            predictions = self._predictions
        if not predictions:
            raise ValueError(f"No predictions for {self.dataset_name}")
        image_ids = [prediction["image_id"] for prediction in predictions]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"Duplicate predictions for {self.dataset_name}")
        actual_ids = set(image_ids)
        if actual_ids != self._expected_ids:
            raise ValueError(
                f"Incomplete {self.dataset_name} predictions: "
                f"{len(self._expected_ids - actual_ids)} missing, "
                f"{len(actual_ids - self._expected_ids)} unexpected"
            )

        tasks = []
        for prediction in predictions:
            image_id = prediction["image_id"]
            annotations = self._coco_api.loadAnns(self._coco_api.getAnnIds(imgIds=image_id))
            targets = [ann["segmentation"] for ann in annotations if ann.get("segmentation") is not None]
            if not targets:
                raise ValueError(f"Missing foreground annotation for image {image_id}")
            tasks.append({**prediction, "gt_rle": targets[0]})
        if self._num_workers and len(tasks) > 1:
            # CUDA inference has already initialized threads in the parent.
            # Spawn clean CPU workers instead of inheriting that runtime.
            with mp.get_context("spawn").Pool(self._num_workers) as pool:
                rows = list(pool.imap(evaluate_mask_pair, tasks, chunksize=16))
        else:
            rows = [evaluate_mask_pair(task) for task in tasks]
        results = {
            name: float(np.mean(np.asarray([row[name] for row in rows], dtype=np.float64)))
            for name in METRIC_NAMES
        }
        self._logger.info("%s (%d images): %s", self.dataset_name, len(rows), results)
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            with PathManager.open(os.path.join(self._output_dir, "metrics.json"), "w") as handle:
                json.dump({"dataset": self.dataset_name, "images": len(rows), **results}, handle, indent=2)
                handle.write("\n")
            with PathManager.open(os.path.join(self._output_dir, "per_image.csv"), "w") as handle:
                writer = csv.DictWriter(handle, fieldnames=("image_id", *METRIC_NAMES))
                writer.writeheader()
                writer.writerows(rows)
        return OrderedDict(cod=results)
