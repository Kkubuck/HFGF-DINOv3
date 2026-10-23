# Adapted from Mask2Former and FOCUS; see NOTICE.
"""Image augmentation and foreground/background targets for HFGF-DINOv3."""

import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["HFGFDatasetMapper"]


class HFGFDatasetMapper:
    """Apply training transforms and construct two class-specific mask targets.

    Semantic labels are 0 for foreground, 1 for background, and 255 for ignored
    padding. ``size_divisibility`` is the fixed output side in the training
    training recipe: padding on the bottom/right also crops larger dimensions.
    """

    @configurable
    def __init__(
        self, is_train=True, *, augmentations, image_format, ignore_label,
        size_divisibility,
    ):
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        logging.getLogger(__name__).info("HFGF augmentations: %s", augmentations)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        if is_train:
            augmentations = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augmentations.append(T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                ))
            if cfg.INPUT.COLOR_AUG_SSD:
                augmentations.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augmentations.append(T.RandomFlip())
        else:
            augmentations = [T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice",
            )]
        names = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
        return {
            "is_train": is_train,
            "augmentations": augmentations,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": MetadataCatalog.get(names[0]).ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        semantic = None
        if self.is_train:
            if "sem_seg_file_name" not in dataset_dict:
                raise ValueError(f"Missing semantic labels for {dataset_dict['file_name']}")
            # Float64 keeps the nearest-neighbor transform path used in the paper.
            semantic = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        aug_input = T.AugInput(image, sem_seg=semantic)
        aug_input, _ = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1)))
        if aug_input.sem_seg is not None:
            semantic = torch.as_tensor(aug_input.sem_seg.astype("long"))

        if self.size_divisibility > 0:
            padding = [
                0, self.size_divisibility - image.shape[-1],
                0, self.size_divisibility - image.shape[-2],
            ]
            image = F.pad(image, padding, value=128).contiguous()
            if semantic is not None:
                semantic = F.pad(semantic, padding, value=self.ignore_label).contiguous()
        dataset_dict["image"] = image
        if not self.is_train:
            return dataset_dict
        if "annotations" in dataset_dict:
            raise ValueError("Expected semantic labels rather than instance annotations")

        dataset_dict["sem_seg"] = semantic.long()
        semantic = semantic.numpy()
        classes = np.unique(semantic)
        classes = classes[classes != self.ignore_label]
        instances = Instances(image.shape[-2:])
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        masks = [semantic == class_id for class_id in classes]
        if not masks:
            instances.gt_masks = torch.zeros((0, *semantic.shape[-2:]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
        else:
            masks = BitMasks(torch.stack([
                torch.from_numpy(np.ascontiguousarray(mask.copy())) for mask in masks
            ]))
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()
        dataset_dict["instances"] = instances
        return dataset_dict
