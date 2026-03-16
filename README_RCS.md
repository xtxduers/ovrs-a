# RCS (Residual Cross-correlation Self-attention) for OVRS-A

## 概述 / Overview

本分支（`add_RCS`）在 OVRS-A 的 CLIP 视觉编码器中集成了来自
[ResCLIP](https://github.com/xtxduers/ResCLIP) 的
**RCS（Residual Cross-correlation Self-attention）** 注意力机制。

This branch (`add_RCS`) integrates the **RCS (Residual Cross-correlation
Self-attention)** mechanism from [ResCLIP](https://github.com/xtxduers/ResCLIP)
into the CLIP visual encoder used in OVRS-A.

---

## 方法 / Method

### 跨层注意力聚合 / Cross-layer Attention Aggregation

设第 $i$ 层 Transformer Block 的 Q-K 互相关注意力为 $A_{qk}^{i}$，则聚合注意力为：

$$
A_c = \frac{1}{N} \sum_{i=s}^{e} A_{qk}^{i}, \quad N = e - s + 1
$$

其中起始层 $s$ 与终止层 $e$ 的取值（**0-based 索引**）如下：

| 视觉编码器 | 层数 | $s$（0-based） | $e$（0-based） | $N$ |
|-----------|------|--------------|--------------|-----|
| ViT-L/14@336px | 24 | 11（第12层）| 22（第23层）| 12 |
| ViT-B/16 | 12 |  5（第 6层）|  8（第 9层）|  4 |

> 参考：ResCLIP 代码中对 ViT-B 使用 `range(5, 9)`；问题说明对 ViT-L 若代码未指定则取 $s=12, e=23$（1-indexed），即 0-indexed 下 $s=11, e=22$。

### RCS 注意力融合 / RCS Attention Blending

基于聚合注意力 $A_c$，RCS 注意力定义为：

$$
A_{\text{rca}} = (1 - \lambda_{\text{rca}}) \cdot A_s + \lambda_{\text{rca}} \cdot A_c \tag{1}
$$

- $A_s$：最后一层的 Q-K 互相关注意力（`softmax(Q \cdot K^T / \sqrt{d})`），捕获局部 patch 结构信息。  
- $A_c$：各中间层注意力的均值，捕获跨特征对应关系。  
- $\lambda_{\text{rca}} = 0.5$：平衡参数。

RCS 同时融合了局部 patch 结构与跨特征对应关系，有效提升最后一层的特征重组质量，从而改善分割性能。

---

## 代码修改 / Code Changes

### `cat_seg/third_party/model.py`

#### `ResidualAttentionBlock` 新增方法

| 方法 | 说明 |
|------|------|
| `forward_with_attn(x)` | 标准前向传播，额外返回 Q-K 注意力权重 `(N×H, L, L)`；用于收集中间层注意力图。 |
| `forward_dense_rcs(x, A_c, lambda_rca=0.5)` | 最后一层的 RCS 前向：计算 $A_s$，按公式 (1) 与 $A_c$ 融合，用融合注意力加权 V 并经残差+MLP 得到输出。 |

#### `Transformer.forward` 新增参数

```python
Transformer.forward(x, dense=False, use_rcs=False,
                    rcs_start=11, rcs_end=22, lambda_rca=0.5)
```

- `use_rcs=True` 时启用 RCS：对第 `rcs_start` 至 `rcs_end` 层（含）收集注意力图，计算 $A_c$，并在最后一层调用 `forward_dense_rcs`。

#### `VisualTransformer.forward` 新增参数

```python
VisualTransformer.forward(x, dense=False, use_rcs=False,
                          rcs_start=None, rcs_end=None, lambda_rca=0.5)
```

- 根据编码器层数自动选择默认 `rcs_start`/`rcs_end`，并透传给 `Transformer.forward`。

#### `CLIP.encode_image` 新增参数

```python
CLIP.encode_image(image, masks=None, pool_mask=None, dense=False,
                  use_rcs=False, rcs_start=None, rcs_end=None, lambda_rca=0.5)
```

### `cat_seg/cat_seg_model_ours.py`

所有 `encode_image` 调用均添加 `use_rcs=True`，包括：

- 主训练/推理循环中对原始图像及 0°/90°/180°/270° 旋转图像的编码。
- `inference_sliding_window` 中的对应调用。

---

## 使用说明 / Usage

无需额外配置，修改对现有配置文件完全兼容。`use_rcs=True` 已默认启用于
`cat_seg_model_ours.py` 中的所有图像编码调用。

若要禁用 RCS（回退到原始行为），只需在调用 `encode_image` 时传入 `use_rcs=False`：

```python
clip_features = clip_model.encode_image(images, dense=True, use_rcs=False)
```

若要自定义 RCS 层范围或权重：

```python
clip_features = clip_model.encode_image(
    images, dense=True, use_rcs=True,
    rcs_start=11, rcs_end=22, lambda_rca=0.5
)
```

---

## 参考 / References

- ResCLIP: [https://github.com/xtxduers/ResCLIP](https://github.com/xtxduers/ResCLIP)
- CAT-Seg: [https://github.com/KU-CVLAB/CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg)
