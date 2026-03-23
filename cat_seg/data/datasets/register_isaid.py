# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 dataset registration for iSAID.

iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images
  - 15 semantic categories (background is excluded from evaluation)
  - Annotation masks are grayscale PNGs (pixel value = class index 0-14;
    255 = ignore / background)

Expected directory layout under ``DETECTRON2_DATASETS`` (default: ``datasets/``):

    datasets/
      iSAID/
        images/
          train/          *.png
          val/            *.png
        annotations/
          train/          *.png  (uint8 grayscale, values 0-14, 255 = ignore)
          val/            *.png
"""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# --------------------------------------------------------------------------- #
# Class definitions                                                            #
# --------------------------------------------------------------------------- #

ISAID_CLASSES = (
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "swimming pool",
    "roundabout",
    "soccer ball field",
    "plane",
    "harbor",
)

ISAID_COLORS = (
    (0, 0, 63),       # ship
    (0, 191, 127),    # storage tank
    (0, 63, 0),       # baseball diamond
    (0, 63, 127),     # tennis court
    (0, 63, 191),     # basketball court
    (0, 63, 255),     # ground track field
    (0, 127, 63),     # bridge
    (0, 127, 127),    # large vehicle
    (0, 0, 127),      # small vehicle
    (0, 0, 191),      # helicopter
    (0, 0, 255),      # swimming pool
    (0, 63, 63),      # roundabout
    (0, 127, 191),    # soccer ball field
    (0, 127, 255),    # plane
    (0, 100, 155),    # harbor
)

# --------------------------------------------------------------------------- #
# Registration helpers                                                         #
# --------------------------------------------------------------------------- #

_SPLITS = {
    "iSAID_train_sem_seg": ("iSAID/images/train", "iSAID/annotations/train"),
    "iSAID_val_sem_seg":   ("iSAID/images/val",   "iSAID/annotations/val"),
}


def _register_isaid(root: str) -> None:
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
            stuff_classes=list(ISAID_CLASSES),
            stuff_colors=list(ISAID_COLORS),
        )

    # "all" split – train + val combined for zero-shot evaluation
    all_name = "iSAID_all_sem_seg"
    train_img = os.path.join(root, "iSAID/images/train")
    train_gt = os.path.join(root, "iSAID/annotations/train")
    val_img = os.path.join(root, "iSAID/images/val")
    val_gt = os.path.join(root, "iSAID/annotations/val")
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
        stuff_classes=list(ISAID_CLASSES),
        stuff_colors=list(ISAID_COLORS),
    )


# --------------------------------------------------------------------------- #
# Entry point – called from datasets/__init__.py                               #
# --------------------------------------------------------------------------- #

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_register_isaid(_root)
