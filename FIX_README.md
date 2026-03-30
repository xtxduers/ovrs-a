# 问题修复说明

## 问题描述

执行 `sh train_DLRSD.sh` 时报错：

```
AttributeError: Attribute 'ignore_label' does not exist in the metadata of dataset 'DLRSD_train_sem_seg': metadata is empty.
```

## 根本原因

错误发生在 `cat_seg/data/dataset_mappers/mask_former_semantic_dataset_mapper.py` 第 87 行：

```python
meta = MetadataCatalog.get(dataset_names[0])
ignore_label = meta.ignore_label  # ← 此处抛出异常
```

该行代码尝试从 detectron2 的 `MetadataCatalog` 中读取数据集 `DLRSD_train_sem_seg` 的 `ignore_label` 属性，但该数据集从未被注册，导致 metadata 为空。

**调用链分析：**

1. `train_net.py` 在启动时 `import cat_seg`
2. `cat_seg/__init__.py` 执行 `from . import data`（注册所有数据集）
3. `cat_seg/data/__init__.py` 执行 `from . import datasets`
4. **`cat_seg/data/datasets.py` 文件不存在** → `ImportError`（或 `ModuleNotFoundError`）

由于导入失败，数据集注册代码从未执行，`MetadataCatalog` 中没有 `DLRSD_train_sem_seg` 的任何元数据，因此访问 `meta.ignore_label` 时抛出 `AttributeError`。

同时，模型配置中引用的类别定义 JSON 文件（如 `datasets/DLRSD.json`）也不存在。

## 修复内容

### 1. 新增 `cat_seg/data/datasets.py`

创建了缺失的数据集注册模块，使用 detectron2 的 `DatasetCatalog` 和 `MetadataCatalog` 注册以下数据集：

| 数据集名称 | 类别数 | splits |
|---|---|---|
| DLRSD | 17 | train / val / all |
| iSAID | 15 | train / val / all |
| Potsdam | 6 | all |
| Vaihingen | 5 | all |

每个数据集注册时设置了以下关键元数据：

- `stuff_classes`：类别名称列表
- `ignore_label`：255（忽略标签值，与配置中 `IGNORE_VALUE: 255` 一致）
- `evaluator_type`：`"sem_seg"`（告知评估器使用语义分割评估方式）
- `image_root` / `sem_seg_root`：图像和标注文件路径

### 2. 新增 `datasets/` 目录及 JSON 类别文件

创建了模型配置（YAML）中引用的类别定义文件：

| 文件 | 内容 |
|---|---|
| `datasets/DLRSD.json` | DLRSD 17 个类别名称列表 |
| `datasets/iSAID.json` | iSAID 15 个类别名称列表 |
| `datasets/Potsdam.json` | Potsdam 6 个类别名称列表 |
| `datasets/Vaihingen.json` | Vaihingen 5 个类别名称列表 |

这些 JSON 文件由模型（`CATSegPredictor`）加载，用于生成 CLIP 文本特征编码。

## 数据集目录结构要求

运行训练前，请按以下结构组织数据集文件（以 DLRSD 为例）：

```
datasets/
└── DLRSD/
    ├── images/
    │   ├── train/      # 训练集图像（.jpg）
    │   ├── val/        # 验证集图像（.jpg）
    │   └── all/        # 全量图像（.jpg），用于评估
    └── annotations/
        ├── train/      # 训练集语义分割标注（.png，像素值为类别 ID，255 表示忽略）
        ├── val/        # 验证集标注（.png）
        └── all/        # 全量标注（.png）
```

iSAID、Potsdam、Vaihingen 数据集的目录结构相同，对应根目录分别为 `datasets/iSAID/`、`datasets/Potsdam/`、`datasets/Vaihingen/`。

## 修改文件列表

```
cat_seg/data/datasets.py       ← 新增（核心修复）
datasets/DLRSD.json            ← 新增
datasets/iSAID.json            ← 新增
datasets/Potsdam.json          ← 新增
datasets/Vaihingen.json        ← 新增
FIX_README.md                  ← 新增（本文件）
```
