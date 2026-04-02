# CAT-Seg 运行修复说明

本次修复目标：解决项目无法运行的问题，并补齐按代码逻辑需要的数据集注册。

## 本次修改内容

1. **新增数据集注册模块**
   - 新增文件：`cat_seg/data/datasets.py`
   - 作用：注册以下语义分割数据集（含 train/val/all）：
     - `iSAID_train_sem_seg` / `iSAID_val_sem_seg` / `iSAID_all_sem_seg`
     - `DLRSD_train_sem_seg` / `DLRSD_val_sem_seg` / `DLRSD_all_sem_seg`
     - `Potsdam_train_sem_seg` / `Potsdam_val_sem_seg` / `Potsdam_all_sem_seg`
     - `Vaihingen_train_sem_seg` / `Vaihingen_val_sem_seg` / `Vaihingen_all_sem_seg`

2. **修复无法运行的根因**
   - `cat_seg/data/__init__.py` 原本会导入 `datasets`，但仓库中缺少该模块，导致导入失败。
   - 现在补齐 `cat_seg/data/datasets.py` 后，数据注册逻辑可正常被加载（依赖 detectron2 环境）。

3. **数据目录自动适配**
   - 默认根目录：`datasets/`（可通过环境变量 `DETECTRON2_DATASETS` 覆盖）。
   - 对每个数据集支持多种常见目录布局候选（如 `images/train + annotations/train`、`img_dir/train + ann_dir/train` 等），优先使用实际存在的路径。

## 使用方式

### 1）准备数据目录
将数据放在如下结构之一（每个数据集目录名对应 `iSAID`、`DLRSD`、`Potsdam`、`Vaihingen`）：

- `datasets/<DatasetName>/images/<split>` 与 `datasets/<DatasetName>/annotations/<split>`
- 或 `datasets/<DatasetName>/img_dir/<split>` 与 `datasets/<DatasetName>/ann_dir/<split>`

其中 `<split>` 支持 `train`、`val`、`all`（`all` 也兼容 `test`）。

### 2）运行训练/评估
按原脚本运行，例如：

```bash
sh run.sh configs/vitl_336_DLRSD.yaml 4 output_dir
sh eval.sh configs/vitl_336_DLRSD.yaml 4 output_dir
```

## 说明

- 本次修复是**最小改动**：只补齐缺失的数据集注册模块，不改动模型主体逻辑。
- 若运行时出现 `No module named detectron2`，请先按项目依赖安装 detectron2 与其环境。
