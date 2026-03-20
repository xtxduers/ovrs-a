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
