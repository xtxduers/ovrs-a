import os
from typing import Dict, List, Tuple

from detectron2.data import DatasetCatalog, MetadataCatalog


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _list_files_with_ext(root: str, extensions: Tuple[str, ...]) -> List[str]:
    if not os.path.isdir(root):
        return []
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(extensions):
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def _build_image_index(image_root: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for image_path in _list_files_with_ext(image_root, _IMAGE_EXTENSIONS):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        index[stem] = image_path
    return index


def _load_sem_seg_pairs(sem_seg_root: str, image_root: str) -> List[Dict[str, str]]:
    gt_files = _list_files_with_ext(sem_seg_root, (".png",))
    image_index = _build_image_index(image_root)
    dataset_dicts: List[Dict[str, str]] = []
    for gt_path in gt_files:
        stem = os.path.splitext(os.path.basename(gt_path))[0]
        image_path = image_index.get(stem)
        if image_path is None:
            continue
        dataset_dicts.append(
            {
                "file_name": image_path,
                "sem_seg_file_name": gt_path,
            }
        )
    return dataset_dicts


def _find_existing_pair(root: str, candidates: List[Tuple[str, str]]) -> Tuple[str, str]:
    for image_rel, gt_rel in candidates:
        image_dir = os.path.join(root, image_rel)
        gt_dir = os.path.join(root, gt_rel)
        if os.path.isdir(image_dir) and os.path.isdir(gt_dir):
            return image_dir, gt_dir
    image_rel, gt_rel = candidates[0]
    return os.path.join(root, image_rel), os.path.join(root, gt_rel)


def _register_sem_seg_dataset(
    dataset_name: str,
    image_root: str,
    sem_seg_root: str,
    evaluator_type: str,
    class_count: int,
) -> None:
    if dataset_name in DatasetCatalog.list():
        return
    DatasetCatalog.register(
        dataset_name,
        lambda image_root=image_root, sem_seg_root=sem_seg_root: _load_sem_seg_pairs(
            sem_seg_root, image_root
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        image_root=image_root,
        sem_seg_root=sem_seg_root,
        evaluator_type=evaluator_type,
        ignore_label=255,
        stuff_classes=[f"class_{i}" for i in range(class_count)],
    )


def register_remote_sensing_datasets() -> None:
    root = os.getenv("DETECTRON2_DATASETS", "datasets")
    dataset_specs = {
        "iSAID": {
            "class_count": 15,
            "splits": {
                "train": "iSAID_train_sem_seg",
                "val": "iSAID_val_sem_seg",
                "all": "iSAID_all_sem_seg",
            },
        },
        "DLRSD": {
            "class_count": 17,
            "splits": {
                "train": "DLRSD_train_sem_seg",
                "val": "DLRSD_val_sem_seg",
                "all": "DLRSD_all_sem_seg",
            },
        },
        "Potsdam": {
            "class_count": 6,
            "splits": {
                "train": "Potsdam_train_sem_seg",
                "val": "Potsdam_val_sem_seg",
                "all": "Potsdam_all_sem_seg",
            },
        },
        "Vaihingen": {
            "class_count": 6,
            "splits": {
                "train": "Vaihingen_train_sem_seg",
                "val": "Vaihingen_val_sem_seg",
                "all": "Vaihingen_all_sem_seg",
            },
        },
    }

    split_candidates = {
        "train": [
            ("images/train", "annotations/train"),
            ("images/training", "annotations/training"),
            ("img_dir/train", "ann_dir/train"),
            ("train/images", "train/annotations"),
            ("train/img", "train/label"),
            ("train/image", "train/label"),
        ],
        "val": [
            ("images/val", "annotations/val"),
            ("images/valid", "annotations/valid"),
            ("img_dir/val", "ann_dir/val"),
            ("val/images", "val/annotations"),
            ("val/img", "val/label"),
            ("valid/images", "valid/annotations"),
        ],
        "all": [
            ("images/all", "annotations/all"),
            ("img_dir/all", "ann_dir/all"),
            ("all/images", "all/annotations"),
            ("images/test", "annotations/test"),
            ("img_dir/test", "ann_dir/test"),
            ("test/images", "test/annotations"),
            ("images", "annotations"),
        ],
    }

    for dataset_folder, spec in dataset_specs.items():
        dataset_root = os.path.join(root, dataset_folder)
        for split_name, dataset_name in spec["splits"].items():
            image_root, sem_seg_root = _find_existing_pair(
                dataset_root, split_candidates[split_name]
            )
            _register_sem_seg_dataset(
                dataset_name=dataset_name,
                image_root=image_root,
                sem_seg_root=sem_seg_root,
                evaluator_type="sem_seg",
                class_count=spec["class_count"],
            )


register_remote_sensing_datasets()
