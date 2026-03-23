# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 dataset registration for LoveDA.

LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation
  - 7 semantic categories (0 = background, 1-6 = foreground)
  - Two domains: Urban and Rural
  - Official splits: Train, Val, Test (Test has no annotations)

Expected directory layout under ``DETECTRON2_DATASETS`` (default: ``datasets/``):

    datasets/
      LoveDA/
        images/
          train/          *.png
          val/            *.png
        annotations/
          train/          *.png  (uint8 grayscale, values 0-6, 255 = ignore)
          val/            *.png
"""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# --------------------------------------------------------------------------- #
# Class definitions                                                            #
# --------------------------------------------------------------------------- #

LOVEDA_CLASSES = (
    "background",
    "building",
    "road",
    "water",
    "barren land",
    "forest",
    "agriculture",
)

# BGR colours used for visualisation (follows common LoveDA palette)
LOVEDA_COLORS = (
    (255, 255, 255),  # background   – white
    (255, 0, 0),      # building     – red
    (255, 255, 0),    # road         – yellow
    (0, 0, 255),      # water        – blue
    (159, 129, 183),  # barren land  – purple
    (0, 255, 0),      # forest       – green
    (255, 195, 128),  # agriculture  – orange
)

# --------------------------------------------------------------------------- #
# Registration helpers                                                         #
# --------------------------------------------------------------------------- #

_SPLITS = {
    "LoveDA_train_sem_seg": ("LoveDA/images/train", "LoveDA/annotations/train"),
    "LoveDA_val_sem_seg":   ("LoveDA/images/val",   "LoveDA/annotations/val"),
}


def _register_loveda(root: str) -> None:
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
            stuff_classes=list(LOVEDA_CLASSES),
            stuff_colors=list(LOVEDA_COLORS),
        )

    # "all" split – train + val combined for zero-shot evaluation
    all_name = "LoveDA_all_sem_seg"
    train_img = os.path.join(root, "LoveDA/images/train")
    train_gt = os.path.join(root, "LoveDA/annotations/train")
    val_img = os.path.join(root, "LoveDA/images/val")
    val_gt = os.path.join(root, "LoveDA/annotations/val")
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
        stuff_classes=list(LOVEDA_CLASSES),
        stuff_colors=list(LOVEDA_COLORS),
    )


# --------------------------------------------------------------------------- #
# Entry point – called from datasets/__init__.py                               #
# --------------------------------------------------------------------------- #

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
_register_loveda(_root)
