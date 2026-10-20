#!/usr/bin/env python3
"""Prepare COD benchmark annotations without modifying the downloaded masks.

Expected directories below --root:
  CAMO-V.1.0-CVIU2019/{Images/Train,Images/Test,GT}
  COD10K-v3/{Train,Test}/{Image,GT_Object}
  NC4K/{Imgs,GT}
  CHAMELEON/{Imgs,GT}

Training labels are 0=foreground and 1=background. The evaluation annotation
contains one foreground RLE per image. Image IDs follow sorted JPEG filenames.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pycocotools.mask as mask_util


DATASET_DIRECTORIES = {
    "camo": "CAMO-V.1.0-CVIU2019",
    "cod10k": "COD10K-v3",
    "nc4k": "NC4K",
    "chameleon": "CHAMELEON",
}


def encode_mask(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def prepare_split(root, dataset, split, overwrite=False):
    base = root / DATASET_DIRECTORIES[dataset]
    if dataset == "camo":
        image_dir, mask_dir = base / "Images" / split, base / "GT"
        label_dir = base / "annotations_detectron2" / f"cod_semantic_{split.lower()}"
        binary_dir = base / f"cod_semantic_{split.lower()}"
        instance_path, semantic_path = base / f"COD_{split}.json", base / f"COD_SEMANTIC_{split}.json"
    elif dataset == "cod10k":
        image_dir, mask_dir = base / split / "Image", base / split / "GT_Object"
        label_dir, binary_dir = base / "annotations_detectron2" / split, None
        instance_path, semantic_path = base / f"COD_{split}.json", base / f"COD_SEMANTIC_{split}.json"
    else:
        image_dir, mask_dir = base / "Imgs", base / "GT"
        label_dir, binary_dir = base / "annotations_detectron2" / "GT", None
        instance_path = base / f"{dataset.upper()}_INSTANCE.json"
        semantic_path = base / f"{dataset.upper()}_SEMANTIC.json"
    for output in (instance_path, semantic_path, label_dir, binary_dir):
        if output is not None and output.exists() and not overwrite:
            raise FileExistsError(f"{output} already exists; use --overwrite to regenerate derived annotations")
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPEG images in {image_dir}")
    missing = [mask_dir / f"{image.stem}.png" for image in image_files if not (mask_dir / f"{image.stem}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks; first missing file: {missing[0]}")
    label_dir.mkdir(parents=True, exist_ok=True)
    if binary_dir is not None:
        binary_dir.mkdir(parents=True, exist_ok=True)

    images, instance_annotations, semantic_annotations = [], [], []
    for image_id, image_path in enumerate(image_files, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(mask_dir / f"{image_path.stem}.png") as image:
            raw_mask = np.asarray(image.convert("L"))
        if dataset != "camo" and not np.isin(raw_mask, [0, 255]).all():
            raise ValueError(
                f"Expected binary 0/255 masks in {mask_dir}; invalid mask {image_path.stem}.png"
            )
        foreground = raw_mask > 128
        labels = (~foreground).astype(np.uint8)
        eval_foreground = foreground
        if dataset == "chameleon" and foreground.shape != (height, width):
            # Match the corrected evaluation annotations: retain the top-left
            # alignment, crop excess pixels, and pad the bottom/right with 0.
            eval_foreground = foreground[:height, :width]
            eval_foreground = np.pad(eval_foreground, (
                (0, max(0, height - eval_foreground.shape[0])),
                (0, max(0, width - eval_foreground.shape[1])),
            ))
            # The supplied semantic labels use nearest resizing. They are not
            # used to evaluate predictions, which use the foreground RLE above.
            labels = np.asarray(Image.fromarray(labels).resize(
                (width, height), resample=Image.Resampling.NEAREST,
            ))
        Image.fromarray(labels).save(label_dir / f"{image_path.stem}.png")
        if binary_dir is not None:
            Image.fromarray(foreground.astype(np.uint8) * 255).save(binary_dir / f"{image_path.stem}.png")

        foreground_rle, background_rle = encode_mask(foreground), encode_mask(~foreground)
        eval_rle = encode_mask(eval_foreground)
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        instance_annotations.append({
            "id": image_id,
            "image_id": image_id,
            "file_name": image_path.name,
            "category_id": 1,
            "segmentation": eval_rle,
            "area": mask_util.area(eval_rle).tolist(),
            "bbox": mask_util.toBbox(eval_rle).tolist(),
            "iscrowd": 0,
        })
        segments = []
        for category_id, rle in enumerate((foreground_rle, background_rle), start=1):
            segments.append({
                "id": 2 * image_id - 2 + category_id,
                "category_id": category_id,
                "area": mask_util.area(rle).tolist(),
                "bbox": mask_util.toBbox(rle).tolist(),
                "iscrowd": 0,
            })
        semantic_annotations.append({
            "image_id": image_id, "file_name": image_path.name, "segments_info": segments,
        })
    categories = [
        {"id": 1, "name": "foreground", "supercategory": "foreground", "is_thing": 1},
        {"id": 2, "name": "background", "supercategory": "background", "is_thing": 0},
    ]
    for path, annotations, selected_categories in (
        (instance_path, instance_annotations, [{key: value for key, value in categories[0].items() if key != "is_thing"}]),
        (semantic_path, semantic_annotations, categories),
    ):
        with path.open("w") as handle:
            json.dump({"images": images, "annotations": annotations, "categories": selected_categories}, handle)
            handle.write("\n")
    retained = sum(dataset != "cod10k" or image.stem.split("-")[1:2] != ["NonCAM"] for image in image_files)
    print(f"{dataset}/{split or 'Test'}: {retained} COD images ({len(image_files)} annotated images)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_DIRECTORIES), default=list(DATASET_DIRECTORIES))
    parser.add_argument("--overwrite", action="store_true", help="Regenerate derived annotations; raw masks remain unchanged")
    args = parser.parse_args()
    for dataset in args.datasets:
        for split in (("Train", "Test") if dataset in {"camo", "cod10k"} else (None,)):
            prepare_split(args.root, dataset, split, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
