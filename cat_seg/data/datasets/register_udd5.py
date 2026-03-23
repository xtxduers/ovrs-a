# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 dataset registration for UDD5 (Urban Drone Dataset – 5 categories).

UDD5 is a drone-captured urban scene segmentation dataset with 5 semantic
categories.  Annotation masks are grayscale PNGs where pixel value = class
index (0-4); 255 is used as the ignore label.

Expected directory layout under ``DETECTRON2_DATASETS`` (default: ``datasets/``):

    datasets/
      UDD5/
        images/
          train/          *.jpg (or *.png)
          val/            *.jpg (or *.png)
        annotations/
          train/          *.png  (uint8 grayscale, values 0-4, 255 = ignore)
          val/            *.png
"""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# --------------------------------------------------------------------------- #
# Class definitions                                                            #
# --------------------------------------------------------------------------- #

UDD5_CLASSES = (
    "vegetation",
    "ground",
    "building",
    "vehicle",
    "human",
)

# BGR colours (approximate, based on common UDD visualisation palettes)
UDD5_COLORS = (
    (35, 142, 107),   # vegetation – green
    (128, 64, 128),   # ground     – grey-purple
    (70, 70, 70),     # building   – dark grey
    (142, 0, 0),      # vehicle    – dark blue (BGR representation of 0,0,142)
    (60, 20, 220),    # human      – red (BGR representation of 220,20,60)
)

# --------------------------------------------------------------------------- #
# Registration helpers                                                         #
# --------------------------------------------------------------------------- #

_SPLITS = {
    "UDD5_train_sem_seg": ("UDD5/images/train", "UDD5/annotations/train"),
    "UDD5_val_sem_seg":   ("UDD5/images/val",   "UDD5/annotations/val"),
}


def _register_udd5(root: str) -> None:
    for name, (image_dir, gt_dir) in _SPLITS.items():
        image_root = os.path.join(root, image_dir)
        gt_root = os.path.join(root, gt_dir)
        DatasetCatalog.register(
            name,
            lambda i=image_root, g=gt_root: load_sem_seg(g, i, gt_ext="png", image_ext="jpg"),
        )
        MetadataCatalog.get(name).set(
            image_root=image_root,
            sem_seg_root=gt_root,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_classes=list(UDD5_CLASSES),
            stuff_colors=list(UDD5_COLORS),
        )

    # "all" split – train + val combined for zero-shot evaluation
    all_name = "UDD5_all_sem_seg"
    train_img = os.path.join(root, "UDD5/images/train")
    train_gt = os.path.join(root, "UDD5/annotations/train")
    val_img = os.path.join(root, "UDD5/images/val")
    val_gt = os.path.join(root, "UDD5/annotations/val")
    DatasetCatalog.register(
        all_name,
        lambda ti=train_img, tg=train_gt, vi=val_img, vg=val_gt: (
            load_sem_seg(tg, ti, gt_ext="png", image_ext="jpg")
            + load_sem_seg(vg, vi, gt_ext="png", image_ext="jpg")
        ),
    )
    MetadataCatalog.get(all_name).set(
        evaluator_type="sem_seg",
        ignore_label=255,
        stuff_classes=list(UDD5_CLASSES),
        stuff_colors=list(UDD5_COLORS),
    )


# --------------------------------------------------------------------------- #
# Entry point – called from datasets/__init__.py                               #
# --------------------------------------------------------------------------- #

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_register_udd5(_root)
