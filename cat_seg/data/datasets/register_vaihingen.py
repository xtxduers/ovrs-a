# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 dataset registration for ISPRS Vaihingen.

Vaihingen is an ISPRS benchmark for semantic segmentation of aerial imagery.
  - 6 semantic categories (same class set as Potsdam)
  - Annotation masks are grayscale PNGs (pixel value = class index 0-5;
    255 = ignore)

Expected directory layout under ``DETECTRON2_DATASETS`` (default: ``datasets/``):

    datasets/
      Vaihingen/
        images/
          train/          *.png
          val/            *.png
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

VAIHINGEN_CLASSES = (
    "impervious surfaces",
    "building",
    "low vegetation",
    "tree",
    "car",
    "background clutter",
)

VAIHINGEN_COLORS = (
    (255, 255, 255),  # impervious surfaces – white
    (0, 0, 255),      # building            – blue
    (0, 255, 255),    # low vegetation      – cyan
    (0, 255, 0),      # tree                – green
    (255, 255, 0),    # car                 – yellow
    (255, 0, 0),      # background clutter  – red
)

# --------------------------------------------------------------------------- #
# Registration helpers                                                         #
# --------------------------------------------------------------------------- #

_SPLITS = {
    "Vaihingen_train_sem_seg": ("Vaihingen/images/train", "Vaihingen/annotations/train"),
    "Vaihingen_val_sem_seg":   ("Vaihingen/images/val",   "Vaihingen/annotations/val"),
}


def _register_vaihingen(root: str) -> None:
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
            stuff_classes=list(VAIHINGEN_CLASSES),
            stuff_colors=list(VAIHINGEN_COLORS),
        )

    # "all" split – train + val combined for zero-shot evaluation
    all_name = "Vaihingen_all_sem_seg"
    train_img = os.path.join(root, "Vaihingen/images/train")
    train_gt = os.path.join(root, "Vaihingen/annotations/train")
    val_img = os.path.join(root, "Vaihingen/images/val")
    val_gt = os.path.join(root, "Vaihingen/annotations/val")
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
        stuff_classes=list(VAIHINGEN_CLASSES),
        stuff_colors=list(VAIHINGEN_COLORS),
    )


# --------------------------------------------------------------------------- #
# Entry point – called from datasets/__init__.py                               #
# --------------------------------------------------------------------------- #

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_register_vaihingen(_root)
