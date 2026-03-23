# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 dataset registration for DLRSD.

DLRSD (Deep Learning Remote Sensing Dataset):
  - 17 semantic categories
  - Annotation masks are grayscale PNGs (pixel value = class index 0-16;
    255 = ignore)

Expected directory layout under ``DETECTRON2_DATASETS`` (default: ``datasets/``):

    datasets/
      DLRSD/
        images/
          train/          *.jpg
          val/            *.jpg
        annotations/
          train/          *.png  (uint8 grayscale, values 0-16, 255 = ignore)
          val/            *.png
"""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# --------------------------------------------------------------------------- #
# Class definitions                                                            #
# --------------------------------------------------------------------------- #

DLRSD_CLASSES = (
    "airplane",
    "bare soil",
    "buildings",
    "cars",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile home",
    "pavement",
    "sand",
    "sea",
    "ships",
    "tanks",
    "trees",
    "water",
)

DLRSD_COLORS = (
    (0, 0, 255),      # airplane
    (128, 96, 0),     # bare soil
    (128, 0, 0),      # buildings
    (128, 128, 128),  # cars
    (0, 128, 0),      # chaparral
    (0, 0, 128),      # court
    (128, 0, 128),    # dock
    (0, 64, 0),       # field
    (0, 255, 0),      # grass
    (128, 0, 64),     # mobile home
    (64, 64, 64),     # pavement
    (255, 255, 128),  # sand
    (0, 128, 255),    # sea
    (0, 0, 64),       # ships
    (128, 128, 0),    # tanks
    (0, 64, 64),      # trees
    (0, 128, 128),    # water
)

# --------------------------------------------------------------------------- #
# Registration helpers                                                         #
# --------------------------------------------------------------------------- #

_SPLITS = {
    "DLRSD_train_sem_seg": ("DLRSD/images/train", "DLRSD/annotations/train"),
    "DLRSD_val_sem_seg":   ("DLRSD/images/val",   "DLRSD/annotations/val"),
}


def _register_dlrsd(root: str) -> None:
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
            stuff_classes=list(DLRSD_CLASSES),
            stuff_colors=list(DLRSD_COLORS),
        )

    # "all" split – train + val combined for zero-shot evaluation
    all_name = "DLRSD_all_sem_seg"
    train_img = os.path.join(root, "DLRSD/images/train")
    train_gt = os.path.join(root, "DLRSD/annotations/train")
    val_img = os.path.join(root, "DLRSD/images/val")
    val_gt = os.path.join(root, "DLRSD/annotations/val")
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
        stuff_classes=list(DLRSD_CLASSES),
        stuff_colors=list(DLRSD_COLORS),
    )


# --------------------------------------------------------------------------- #
# Entry point – called from datasets/__init__.py                               #
# --------------------------------------------------------------------------- #

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_register_dlrsd(_root)
