# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 dataset registration for VDD (Varied Drone Dataset).

VDD is a drone-captured scene segmentation benchmark with 6 semantic
categories.  Annotation masks are grayscale PNGs where pixel value = class
index (0-5); 255 is used as the ignore label.

Expected directory layout under ``DETECTRON2_DATASETS`` (default: ``datasets/``):

    datasets/
      VDD/
        images/
          train/          *.jpg (or *.png)
          val/            *.jpg (or *.png)
        annotations/
          train/          *.png  (uint8 grayscale, values 0-5, 255 = ignore)
          val/            *.png
"""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# --------------------------------------------------------------------------- #
# Class definitions                                                            #
# --------------------------------------------------------------------------- #

VDD_CLASSES = (
    "background clutter",
    "building",
    "road",
    "vegetation",
    "vehicle",
    "human",
)

# BGR colours (approximate visualisation palette)
VDD_COLORS = (
    (0, 0, 0),        # background clutter – black
    (128, 0, 0),      # building           – dark red
    (128, 64, 128),   # road               – purple-grey
    (0, 128, 0),      # vegetation         – dark green
    (0, 0, 128),      # vehicle            – dark blue
    (60, 20, 220),    # human              – red
)

# --------------------------------------------------------------------------- #
# Registration helpers                                                         #
# --------------------------------------------------------------------------- #

_SPLITS = {
    "VDD_train_sem_seg": ("VDD/images/train", "VDD/annotations/train"),
    "VDD_val_sem_seg":   ("VDD/images/val",   "VDD/annotations/val"),
}


def _register_vdd(root: str) -> None:
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
            stuff_classes=list(VDD_CLASSES),
            stuff_colors=list(VDD_COLORS),
        )

    # "all" split – train + val combined for zero-shot evaluation
    all_name = "VDD_all_sem_seg"
    train_img = os.path.join(root, "VDD/images/train")
    train_gt = os.path.join(root, "VDD/annotations/train")
    val_img = os.path.join(root, "VDD/images/val")
    val_gt = os.path.join(root, "VDD/annotations/val")
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
        stuff_classes=list(VDD_CLASSES),
        stuff_colors=list(VDD_COLORS),
    )


# --------------------------------------------------------------------------- #
# Entry point – called from datasets/__init__.py                               #
# --------------------------------------------------------------------------- #

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_register_vdd(_root)
