"""Register the four camouflaged object detection benchmarks."""

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


# image directory, mask directory, semantic JSON, training labels, instance JSON
DATASET_SPLITS = {
    "cod_train": (
        "CAMO-V.1.0-CVIU2019/Images/Train",
        "CAMO-V.1.0-CVIU2019/cod_semantic_train",
        "CAMO-V.1.0-CVIU2019/COD_SEMANTIC_Train.json",
        "CAMO-V.1.0-CVIU2019/annotations_detectron2/cod_semantic_train",
        "CAMO-V.1.0-CVIU2019/COD_Train.json",
    ),
    "cod_val": (
        "CAMO-V.1.0-CVIU2019/Images/Test",
        "CAMO-V.1.0-CVIU2019/cod_semantic_test",
        "CAMO-V.1.0-CVIU2019/COD_SEMANTIC_Test.json",
        "CAMO-V.1.0-CVIU2019/annotations_detectron2/cod_semantic_test",
        "CAMO-V.1.0-CVIU2019/COD_Test.json",
    ),
    "cod10k_train": (
        "COD10K-v3/Train/Image",
        "COD10K-v3/Train/GT_Object",
        "COD10K-v3/COD_SEMANTIC_Train.json",
        "COD10K-v3/annotations_detectron2/Train",
        "COD10K-v3/COD_Train.json",
    ),
    "cod10k_val": (
        "COD10K-v3/Test/Image",
        "COD10K-v3/Test/GT_Object",
        "COD10K-v3/COD_SEMANTIC_Test.json",
        "COD10K-v3/annotations_detectron2/Test",
        "COD10K-v3/COD_Test.json",
    ),
    "nc4k": (
        "NC4K/Imgs", "NC4K/GT", "NC4K/NC4K_SEMANTIC.json",
        "NC4K/annotations_detectron2/GT", "NC4K/NC4K_INSTANCE.json",
    ),
    "chameleon": (
        "CHAMELEON/Imgs", "CHAMELEON/GT", "CHAMELEON/CHAMELEON_SEMANTIC.json",
        "CHAMELEON/annotations_detectron2/GT", "CHAMELEON/CHAMELEON_INSTANCE.json",
    ),
}


def load_cod_dataset(json_file, image_root, mask_root, semantic_root, exclude_noncam=False):
    """Load prepared annotations with class 0 foreground and class 1 background."""
    with PathManager.open(json_file) as handle:
        annotations = json.load(handle)["annotations"]

    records = []
    for annotation in annotations:
        stem = os.path.splitext(annotation["file_name"])[0]
        if exclude_noncam and stem.split("-")[1:2] == ["NonCAM"]:
            continue
        segments = []
        for segment in annotation["segments_info"]:
            segment = dict(segment)
            category_id = segment["category_id"]
            if category_id not in (1, 2):
                raise ValueError(f"Unexpected COD category {category_id} in {json_file}")
            segment["category_id"] = category_id - 1
            segment["isthing"] = category_id == 1
            segments.append(segment)
        records.append({
            "file_name": os.path.join(image_root, stem + ".jpg"),
            "image_id": annotation["image_id"],
            "pan_seg_file_name": os.path.join(mask_root, stem + ".png"),
            "sem_seg_file_name": os.path.join(semantic_root, stem + ".png"),
            "segments_info": segments,
        })
    if not records:
        raise ValueError(f"No images found in {image_root}")
    for key in ("file_name", "pan_seg_file_name", "sem_seg_file_name"):
        if not PathManager.isfile(records[0][key]):
            raise FileNotFoundError(
                f"{records[0][key]}. Run tools/prepare_datasets.py before loading the dataset."
            )
    return records


def register_cod_datasets(root):
    """Register prepared datasets below ``root`` without reading their files."""
    for name, paths in DATASET_SPLITS.items():
        image_root, mask_root, semantic_json, semantic_root, instance_json = (
            os.path.join(root, path) for path in paths
        )
        DatasetCatalog.register(
            name,
            lambda image_root=image_root, mask_root=mask_root,
            semantic_json=semantic_json, semantic_root=semantic_root, name=name:
            load_cod_dataset(
                semantic_json, image_root, mask_root, semantic_root,
                exclude_noncam=name.startswith("cod10k"),
            ),
        )
        MetadataCatalog.get(name).set(
            image_root=image_root,
            panoptic_root=mask_root,
            panoptic_json=semantic_json,
            json_file=instance_json,
            evaluator_type="cod",
            ignore_label=255,
            label_divisor=1000,
            thing_classes=["foreground"],
            thing_colors=[[255, 255, 255]],
            stuff_classes=["foreground", "background"],
            stuff_colors=[[255, 255, 255], [0, 0, 0]],
            thing_dataset_id_to_contiguous_id={1: 0},
            stuff_dataset_id_to_contiguous_id={1: 0, 2: 1},
        )


register_cod_datasets(os.getenv("DETECTRON2_DATASETS", "datasets"))
