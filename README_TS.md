# Teacher-Student CATSeg (CATSegTS)

本文档描述了基于 NACLIP 教师-学生知识蒸馏框架对 ovrs-a 仓库所做的改动。

---

## 整体架构

```
┌───────────────────────────────────────────────────────┐
│  输入图像                                              │
│      │                                                 │
│   ┌──┴──────────────────────┐                          │
│   │  教师 CLIP (NAClipTeacher) │  ← 全部参数冻结       │
│   │  视觉编码器 (NACLIP 改造)  │                        │
│   │  • 前 L-1 个 Transformer 块：标准自注意力          │
│   │  • 最后一个块：kk^T + 高斯空间偏置，去除 FFN       │
│   └──────────────┬──────────┘                          │
│                  │ R^T  [B, N, C]                      │
│   ┌──────────────┴──────────┐                          │
│   │  蒸馏损失 (高置信区域)   │                         │
│   └──────────────┬──────────┘                          │
│                  │                                     │
│   ┌──┴──────────────────────┐                          │
│   │  学生 CLIP (Standard)   │  ← 视觉编码器可微调      │
│   │  文本编码器冻结          │                          │
│   └──────────────┬──────────┘                          │
│                  │ R^S  [B, N, C]  +  Aggregator输出   │
│   ┌──────────────┴──────────┐                          │
│   │  分割损失 (BCE)          │                         │
│   └─────────────────────────┘                          │
└───────────────────────────────────────────────────────┘
```

---

## 教师模型：NACLIP 注意力改造

### 注意力公式

教师模型在 CLIP 视觉编码器的**最后一个**编码器块中应用如下修改：

$$
A_s = \mathrm{softmax}\!\left(\frac{KK^\top}{\sqrt{d}} + \omega(\sigma)\right) V
$$

其中：

$$
\omega\!\left((i,j);\,\sigma\right)_{mn}
= \exp\!\left(-\frac{\left\|(m,n)-(i,j)\right\|^2}{2\sigma^2}\right)
$$

- $K$ 为键（Key）向量矩阵，使用 $KK^\top$ 替代标准的 $QK^\top$，以捕获图像块之间的语义相似度。
- $\omega((i,j);\sigma)$ 是以当前 patch 位置 $(i,j)$ 为中心的二维高斯函数，用于引入局部邻域偏置。
- 最后一个编码器块**移除前馈网络（FFN）**并**不使用残差连接**（reduced 架构），令 CLIP 更适合密集预测。

### 整体最终层

$$
Z^{(L)} = \mathrm{SA}_f\!\left(\mathrm{LN}\!\left(Z^{(L-1)}\right)\right)
$$

---

## 蒸馏损失

仅在教师**高置信度区域**进行蒸馏：

$$
\mathcal{L}_{\text{distill}} = \frac{1}{|\mathcal{M}| \cdot C}
\sum_{i,j,c}
\mathbb{I}\!\left(\max_c R_{ij}^T > \theta\right)
\cdot \left|R_{ijc}^S - R_{ijc}^T\right|
$$

其中 $R_{ijc}$ 为图像 patch $(i,j)$ 与类别 $c$ 文本嵌入的余弦相似度：

$$
R_{ijc} = \frac{f_{ij} \cdot t_c}{\|f_{ij}\|\,\|t_c\|}
$$

- $R^T$：教师响应图（NACLIP 编码器提取的 patch 特征与文本嵌入的相似度）。
- $R^S$：学生响应图（学生 CLIP 编码器提取的 patch 特征与文本嵌入的相似度）。
- $\theta$（`DISTILL_THETA`）：置信阈值，仅蒸馏教师置信度超过 $\theta$ 的区域。
- $|\mathcal{M}|$：满足阈值条件的空间位置数量。
- $C$：类别数。

---

## 总损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda \cdot \mathcal{L}_{\text{distill}}
$$

- $\mathcal{L}_{\text{seg}}$：学生分割头的二元交叉熵损失（与基础 CATSeg 一致）。
- $\lambda$（`DISTILL_LAMBDA`）：蒸馏损失的权重。

---

## 参数说明与修改位置

| 参数 | 默认值 | 含义 | 修改位置 |
|------|--------|------|----------|
| `MODEL.TS.DISTILL_LAMBDA` | `0.5` | 蒸馏损失权重 $\lambda$ | `configs/config_ts.yaml` 或命令行 `MODEL.TS.DISTILL_LAMBDA 0.5` |
| `MODEL.TS.DISTILL_THETA` | `0.1` | 蒸馏置信阈值 $\theta$ | `configs/config_ts.yaml` 或命令行 `MODEL.TS.DISTILL_THETA 0.1` |
| `MODEL.TS.GAUSSIAN_STD` | `5.0` | 教师 Gaussian 空间偏置标准差 $\sigma$ | `configs/config_ts.yaml` 或命令行 `MODEL.TS.GAUSSIAN_STD 5.0` |
| `MODEL.SEM_SEG_HEAD.CLIP_FINETUNE` | `"attention"` | 学生视觉编码器微调策略（`attention`/`full`/`prompt`/`none`） | `configs/config_ts.yaml` |
| `MODEL.META_ARCHITECTURE` | `"CATSegTS"` | 启用教师-学生模型 | `configs/config_ts.yaml` |

### 参数修改方式

**方法一：直接修改配置文件**

编辑 `configs/config_ts.yaml`，在 `MODEL.TS` 节下修改对应参数：

```yaml
MODEL:
  TS:
    DISTILL_LAMBDA: 0.5   # 修改为你需要的值
    DISTILL_THETA: 0.1
    GAUSSIAN_STD: 5.0
```

**方法二：命令行覆盖**

```bash
python train_net.py \
    --config-file configs/config_ts.yaml \
    MODEL.TS.DISTILL_LAMBDA 0.5 \
    MODEL.TS.DISTILL_THETA 0.1 \
    MODEL.TS.GAUSSIAN_STD 5.0
```

---

## 新增文件

| 文件 | 描述 |
|------|------|
| `cat_seg/modeling/naclip_teacher.py` | NACLIP 教师模块（kk^T+Gaussian 注意力，全参数冻结） |
| `cat_seg/cat_seg_model_ts.py` | 教师-学生 CATSegTS 模型（注册为 `CATSegTS`） |
| `configs/config_ts.yaml` | 教师-学生训练配置文件 |
| `README_TS.md` | 本文档 |

## 修改的文件

| 文件 | 改动内容 |
|------|----------|
| `cat_seg/config.py` | 新增 `MODEL.TS` 配置节（DISTILL_LAMBDA, DISTILL_THETA, GAUSSIAN_STD） |
| `cat_seg/__init__.py` | 注册 `CATSegTS` 模型类 |

---

## 训练方法

```bash
# 教师-学生蒸馏训练
python train_net.py \
    --config-file configs/config_ts.yaml \
    --num-gpus 4 \
    OUTPUT_DIR output/catseg_ts
```

---

## 数据集注册

本仓库通过 `cat_seg/data/datasets/` 包统一向 Detectron2 的 `DatasetCatalog` 和 `MetadataCatalog` 注册所有自定义数据集，启动训练/评估脚本时会自动完成注册。

### 支持的数据集

| 数据集 | 分割数 | 场景 | 注册名示例 |
|--------|--------|------|-----------|
| **iSAID** | 15 | 航空图像实例/语义分割 | `iSAID_train_sem_seg`, `iSAID_val_sem_seg`, `iSAID_all_sem_seg` |
| **DLRSD** | 17 | 遥感土地利用分类 | `DLRSD_train_sem_seg`, `DLRSD_val_sem_seg`, `DLRSD_all_sem_seg` |
| **Potsdam** | 6 | ISPRS 城市航空图像 | `Potsdam_train_sem_seg`, `Potsdam_val_sem_seg`, `Potsdam_all_sem_seg` |
| **Vaihingen** | 6 | ISPRS 城市航空图像 | `Vaihingen_train_sem_seg`, `Vaihingen_val_sem_seg`, `Vaihingen_all_sem_seg` |
| **LoveDA** | 7 | 遥感土地覆盖/域适应 | `LoveDA_train_sem_seg`, `LoveDA_val_sem_seg`, `LoveDA_all_sem_seg` |
| **uavid** | 8 | 无人机城市语义分割 | `uavid_train_sem_seg`, `uavid_val_sem_seg`, `uavid_all_sem_seg` |
| **UDD5** | 5 | 城市无人机数据集（5类）| `UDD5_train_sem_seg`, `UDD5_val_sem_seg`, `UDD5_all_sem_seg` |
| **VDD** | 6 | 多样化无人机数据集 | `VDD_train_sem_seg`, `VDD_val_sem_seg`, `VDD_all_sem_seg` |

### 类别定义

#### LoveDA（7 类）

| 索引 | 类别 | 颜色 (RGB) |
|------|------|-----------|
| 0 | background | (255, 255, 255) |
| 1 | building | (255, 0, 0) |
| 2 | road | (255, 255, 0) |
| 3 | water | (0, 0, 255) |
| 4 | barren land | (159, 129, 183) |
| 5 | forest | (0, 255, 0) |
| 6 | agriculture | (255, 195, 128) |

#### uavid（8 类）

| 索引 | 类别 | 颜色 (RGB) |
|------|------|-----------|
| 0 | background clutter | (0, 0, 0) |
| 1 | building | (128, 0, 0) |
| 2 | road | (128, 64, 128) |
| 3 | static car | (192, 0, 192) |
| 4 | tree | (0, 128, 0) |
| 5 | low vegetation | (128, 128, 0) |
| 6 | human | (64, 64, 0) |
| 7 | moving car | (64, 0, 128) |

#### UDD5（5 类）

| 索引 | 类别 |
|------|------|
| 0 | vegetation |
| 1 | ground |
| 2 | building |
| 3 | vehicle |
| 4 | human |

#### VDD（6 类）

| 索引 | 类别 |
|------|------|
| 0 | background clutter |
| 1 | building |
| 2 | road |
| 3 | vegetation |
| 4 | vehicle |
| 5 | human |

### 目录结构

每个数据集需按以下结构存放在 `DETECTRON2_DATASETS` 环境变量指定的根目录下（默认为 `datasets/`）：

```
datasets/
  LoveDA/
    images/
      train/          *.png
      val/            *.png
    annotations/
      train/          *.png   # uint8 灰度图，像素值 = 类别索引(0-6)，255 = 忽略
      val/            *.png
  uavid/
    images/
      train/          *.png
      val/            *.png
    annotations/
      train/          *.png   # uint8 灰度图，像素值 = 类别索引(0-7)，255 = 忽略
      val/            *.png
  UDD5/
    images/
      train/          *.jpg
      val/            *.jpg
    annotations/
      train/          *.png   # uint8 灰度图，像素值 = 类别索引(0-4)，255 = 忽略
      val/            *.png
  VDD/
    images/
      train/          *.jpg
      val/            *.jpg
    annotations/
      train/          *.png   # uint8 灰度图，像素值 = 类别索引(0-5)，255 = 忽略
      val/            *.png
```

> 如需更改数据集根目录，只需设置环境变量：  
> `export DETECTRON2_DATASETS=/path/to/your/datasets`

### 类别 JSON 文件

`datasets/*.json` 文件提供 CLIP 文本编码器所需的类别名称列表，用于在训练和推理时生成文本嵌入特征：

```
datasets/
  LoveDA.json    # ["background", "building", "road", ...]
  uavid.json     # ["background clutter", "building", ...]
  UDD5.json      # ["vegetation", "ground", ...]
  VDD.json       # ["background clutter", "building", ...]
  iSAID.json
  DLRSD.json
  Potsdam.json
  Vaihingen.json
```

### 各数据集训练配置文件

| 数据集 | 配置文件 |
|--------|---------|
| LoveDA | `configs/vitl_336_LoveDA.yaml` |
| uavid | `configs/vitl_336_uavid.yaml` |
| UDD5 | `configs/vitl_336_UDD5.yaml` |
| VDD | `configs/vitl_336_VDD.yaml` |

训练示例：

```bash
# 在 LoveDA 上进行教师-学生蒸馏训练
python train_net.py \
    --config-file configs/vitl_336_LoveDA.yaml \
    --num-gpus 4 \
    MODEL.META_ARCHITECTURE CATSegTS \
    MODEL.TS.DISTILL_LAMBDA 0.5 \
    OUTPUT_DIR output/catseg_ts_loveda

# 在 uavid 上训练
python train_net.py \
    --config-file configs/vitl_336_uavid.yaml \
    --num-gpus 4 \
    MODEL.META_ARCHITECTURE CATSegTS \
    OUTPUT_DIR output/catseg_ts_uavid
```

### 零样本评估

使用 `eval.sh` 一次性在所有数据集上进行零样本评估：

```bash
sh eval.sh configs/vitl_336_DLRSD.yaml 4 output/your_checkpoint
```

该脚本会依次在 iSAID、DLRSD、Potsdam、Vaihingen、LoveDA、UDD5、VDD、uavid 上运行评估，并汇总结果。

### 新增/修改文件一览

| 文件 | 描述 |
|------|------|
| `cat_seg/data/datasets/__init__.py` | 数据集注册包入口，导入所有注册模块 |
| `cat_seg/data/datasets/register_loveda.py` | LoveDA 数据集注册（7 类） |
| `cat_seg/data/datasets/register_uavid.py` | UAVid 数据集注册（8 类） |
| `cat_seg/data/datasets/register_udd5.py` | UDD5 数据集注册（5 类） |
| `cat_seg/data/datasets/register_vdd.py` | VDD 数据集注册（6 类） |
| `cat_seg/data/datasets/register_isaid.py` | iSAID 数据集注册（15 类） |
| `cat_seg/data/datasets/register_dlrsd.py` | DLRSD 数据集注册（17 类） |
| `cat_seg/data/datasets/register_potsdam.py` | Potsdam 数据集注册（6 类） |
| `cat_seg/data/datasets/register_vaihingen.py` | Vaihingen 数据集注册（6 类） |
| `datasets/LoveDA.json` | LoveDA 类别名称（供 CLIP 使用） |
| `datasets/uavid.json` | UAVid 类别名称 |
| `datasets/UDD5.json` | UDD5 类别名称 |
| `datasets/VDD.json` | VDD 类别名称 |
| `datasets/iSAID.json` | iSAID 类别名称 |
| `datasets/DLRSD.json` | DLRSD 类别名称 |
| `datasets/Potsdam.json` | Potsdam 类别名称 |
| `datasets/Vaihingen.json` | Vaihingen 类别名称 |
| `configs/vitl_336_LoveDA.yaml` | LoveDA 训练配置 |
| `configs/vitl_336_uavid.yaml` | UAVid 训练配置 |
| `configs/vitl_336_UDD5.yaml` | UDD5 训练配置 |
| `configs/vitl_336_VDD.yaml` | VDD 训练配置 |

---

## 设计说明

### 教师模型初始化

教师模型通过 `copy.deepcopy` 从学生模型的 CLIP 权重中复制，因此二者在训练开始时共享相同的初始权重。随后对教师的最后一个视觉 Transformer 块应用 NACLIP 改造（仅推理时生效，不更新权重）。

### 参数冻结策略

- **教师**：`NAClipTeacher` 在初始化时将所有参数的 `requires_grad` 设为 `False`。
- **学生文本编码器**：在 `CATSegTS.__init__` 中，所有不含 `"visual"` 的参数（即文本编码器相关参数）均被冻结。
- **学生视觉编码器**：根据 `CLIP_FINETUNE` 策略进行选择性微调：
  - `"attention"`：仅微调每个注意力块的 Q、V 投影权重及位置编码。
  - `"full"`：微调整个视觉 Transformer。
  - `"prompt"`：仅微调 prompt 参数（需配合 VPT）。
  - 其他/`"none"`：视觉编码器完全冻结。

### 高斯空间偏置缓存

`NAClipTeacher` 内部维护一个 `_addition_cache` 字典，以 `(H, W)` 为键缓存已计算的 Gaussian 空间偏置矩阵，避免在不同批次中重复计算。
