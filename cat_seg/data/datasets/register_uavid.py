# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 dataset registration for UAVid.

UAVid: A Semantic Segmentation Dataset for UAV Imagery
  - 8 semantic categories
  - Grayscale annotation masks where pixel value = class index (0-7)
  - 255 = ignore

Expected directory layout under ``DETECTRON2_DATASETS`` (default: ``datasets/``):

    datasets/
      uavid/
        images/
          train/          *.png (or *.jpg)
          val/            *.png (or *.jpg)
        annotations/
          train/          *.png  (uint8 grayscale, values 0-7, 255 = ignore)
          val/            *.png
"""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# --------------------------------------------------------------------------- #
# Class definitions                                                            #
# --------------------------------------------------------------------------- #

UAVID_CLASSES = (
    "background clutter",
    "building",
    "road",
    "static car",
    "tree",
    "low vegetation",
    "human",
    "moving car",
)

# BGR colours from the UAVid official colour palette
UAVID_COLORS = (
    (0, 0, 0),        # background clutter – black
    (128, 0, 0),      # building           – dark red
    (128, 64, 128),   # road               – purple-grey
    (192, 0, 192),    # static car         – magenta
    (0, 128, 0),      # tree               – dark green
    (128, 128, 0),    # low vegetation     – olive
    (64, 64, 0),      # human              – dark olive
    (64, 0, 128),     # moving car         – dark purple
)

# --------------------------------------------------------------------------- #
# Registration helpers                                                         #
# --------------------------------------------------------------------------- #

_SPLITS = {
    "uavid_train_sem_seg": ("uavid/images/train", "uavid/annotations/train"),
    "uavid_val_sem_seg":   ("uavid/images/val",   "uavid/annotations/val"),
}


def _register_uavid(root: str) -> None:
    for name, (image_dir, gt_dir) in _SPLITS.items():
        image_root = os.path.join(root, image_dir)
        gt_root = os.path.join(root, gt_dir)
        DatasetCatalog.register(
            name,
            lambda i=image_root, g=gt_root: load_sem_seg(g, i, gt_ext="png", image_ext="png"),
        )
        MetadataCatalog.get(name).set(
            image_root=image_root,
            sem_seg_root=gt_root,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_classes=list(UAVID_CLASSES),
            stuff_colors=list(UAVID_COLORS),
        )

    # "all" split – train + val combined for zero-shot evaluation
    all_name = "uavid_all_sem_seg"
    train_img = os.path.join(root, "uavid/images/train")
    train_gt = os.path.join(root, "uavid/annotations/train")
    val_img = os.path.join(root, "uavid/images/val")
    val_gt = os.path.join(root, "uavid/annotations/val")
    DatasetCatalog.register(
        all_name,
        lambda ti=train_img, tg=train_gt, vi=val_img, vg=val_gt: (
            load_sem_seg(tg, ti, gt_ext="png", image_ext="png")
            + load_sem_seg(vg, vi, gt_ext="png", image_ext="png")
        ),
    )
    MetadataCatalog.get(all_name).set(
        evaluator_type="sem_seg",
        ignore_label=255,
        stuff_classes=list(UAVID_CLASSES),
        stuff_colors=list(UAVID_COLORS),
    )


# --------------------------------------------------------------------------- #
# Entry point – called from datasets/__init__.py                               #
# --------------------------------------------------------------------------- #

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_register_uavid(_root)
