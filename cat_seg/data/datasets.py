# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# ---------------------------------------------------------------------------
# DLRSD  (UC Merced Land Use with semantic segmentation labels, 17 classes)
# ---------------------------------------------------------------------------
DLRSD_CLASSES = [
    "airplane", "bare soil", "buildings", "cars", "chaparral", "court",
    "dock", "field", "grass", "mobile home", "pavement", "sand", "sea",
    "ship", "tanks", "trees", "water",
]

# ---------------------------------------------------------------------------
# iSAID  (Instance Segmentation in Aerial Images Dataset, 15 classes)
# ---------------------------------------------------------------------------
ISAID_CLASSES = [
    "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "bridge", "large vehicle",
    "small vehicle", "helicopter", "swimming pool", "roundabout",
    "soccer ball field", "plane", "harbor",
]

# ---------------------------------------------------------------------------
# ISPRS Potsdam  (6 classes)
# ---------------------------------------------------------------------------
POTSDAM_CLASSES = [
    "impervious surfaces", "buildings", "low vegetation", "trees", "cars",
    "clutter",
]

# ---------------------------------------------------------------------------
# ISPRS Vaihingen  (5 classes)
# ---------------------------------------------------------------------------
VAIHINGEN_CLASSES = [
    "impervious surfaces", "buildings", "low vegetation", "trees", "cars",
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _register_sem_seg(
    name,
    image_root,
    sem_seg_root,
    classes,
    ignore_label=255,
):
    DatasetCatalog.register(
        name,
        lambda gt=sem_seg_root, img=image_root: load_sem_seg(
            gt, img, gt_ext="png", image_ext="jpg"
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        sem_seg_root=sem_seg_root,
        evaluator_type="sem_seg",
        stuff_classes=classes,
        ignore_label=ignore_label,
    )


# ---------------------------------------------------------------------------
# DLRSD splits
# ---------------------------------------------------------------------------
_register_sem_seg(
    "DLRSD_train_sem_seg",
    "datasets/DLRSD/images/train",
    "datasets/DLRSD/annotations/train",
    DLRSD_CLASSES,
)

_register_sem_seg(
    "DLRSD_val_sem_seg",
    "datasets/DLRSD/images/val",
    "datasets/DLRSD/annotations/val",
    DLRSD_CLASSES,
)

_register_sem_seg(
    "DLRSD_all_sem_seg",
    "datasets/DLRSD/images/all",
    "datasets/DLRSD/annotations/all",
    DLRSD_CLASSES,
)

# ---------------------------------------------------------------------------
# iSAID splits
# ---------------------------------------------------------------------------
_register_sem_seg(
    "iSAID_train_sem_seg",
    "datasets/iSAID/images/train",
    "datasets/iSAID/annotations/train",
    ISAID_CLASSES,
)

_register_sem_seg(
    "iSAID_val_sem_seg",
    "datasets/iSAID/images/val",
    "datasets/iSAID/annotations/val",
    ISAID_CLASSES,
)

_register_sem_seg(
    "iSAID_all_sem_seg",
    "datasets/iSAID/images/all",
    "datasets/iSAID/annotations/all",
    ISAID_CLASSES,
)

# ---------------------------------------------------------------------------
# Potsdam
# ---------------------------------------------------------------------------
_register_sem_seg(
    "Potsdam_all_sem_seg",
    "datasets/Potsdam/images/all",
    "datasets/Potsdam/annotations/all",
    POTSDAM_CLASSES,
)

# ---------------------------------------------------------------------------
# Vaihingen
# ---------------------------------------------------------------------------
_register_sem_seg(
    "Vaihingen_all_sem_seg",
    "datasets/Vaihingen/images/all",
    "datasets/Vaihingen/annotations/all",
    VAIHINGEN_CLASSES,
)
