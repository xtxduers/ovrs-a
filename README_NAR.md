# NAR：Neighbour-Aware + Residual Cross-correlation Self-attention

## 概述 / Overview

本分支（`add_NAR`）在 OVRS-A 的 CLIP 视觉编码器中集成了来自
[NACLIP](https://github.com/sinahmr/NACLIP) 的**邻域感知注意力**和
来自 [ResCLIP](https://github.com/xtxduers/ResCLIP) 的
**残差互相关自注意力（RCS）**，形成统一的 **NAR** 注意力机制。

This branch (`add_NAR`) integrates
**Neighbour-Aware Attention** from [NACLIP](https://github.com/sinahmr/NACLIP) and
**Residual Cross-correlation Self-attention (RCS)** from
[ResCLIP](https://github.com/xtxduers/ResCLIP) into the CLIP visual encoder
used in OVRS-A, forming a unified **NAR** attention mechanism.

---

## 方法 / Method

### 1. NACLIP 邻域感知注意力（$A_s$）

查阅 NACLIP 代码，最终编码器块的注意力替换为以 $kk^\top$ 相似度量加二维高斯偏置的形式：

$$
A_s = \mathrm{softmax}\!\left(\frac{kk^\top}{\sqrt{d}} + \omega\!\left((i,j);\sigma\right)\right) v
$$

$$
\omega\!\left((i,j);\sigma\right)_{mn}
= \exp\!\left(-\frac{\left\|(m,n) - (i,j)\right\|^2}{2\sigma^2}\right), \quad \sigma = 5
$$

- 使用 $kk^\top$ 而非标准的 $qk^\top$，侧重捕获 patch 自身的语义属性。  
- $\omega$ 是以当前 patch 位置为中心的二维高斯函数，引入局部空间先验。  
- 最终编码器块移除前馈网络（FFN），采用 NACLIP"精简"架构：

$$
Z^{(L)} = \mathrm{SA}_f\!\left(\mathrm{LN}\!\left(Z^{(L-1)}\right)\right)
$$

### 2. ResCLIP 跨层注意力聚合（$A_c$）

查阅 ResCLIP 代码，对第 $i$ 层的 Q-K 互相关注意力 $A_{qk}^i$ 按层范围均值聚合：

$$
A_c = \frac{1}{N} \sum_{i=s}^{e} A_{qk}^{i}, \quad N = e - s + 1
$$

层范围（**0-based 索引**）：

| 视觉编码器 | 层数 | $s$ | $e$ | $N$ |
|-----------|------|-----|-----|-----|
| ViT-L/14@336px | 24 | 11 | 19 | 9 |
| ViT-B/16       | 12 |  5 |  9 | 5 |

### 3. NAR 注意力融合

融合 $A_s$（局部 patch 结构）与 $A_c$（跨层特征对应关系）：

$$
A_{\text{rca}} = (1 - \lambda_{\text{rca}}) \cdot A_s + \lambda_{\text{rca}} \cdot A_c
\tag{6}
$$

$\lambda_{\text{rca}} = 0.5$（默认平衡参数）。

---

## 代码修改 / Code Changes

### 新增文件：`cat_seg/add_NA.py`

独立的 NACLIP 注意力模块，包含三个函数：

| 函数 | 说明 |
|------|------|
| `gaussian_window(dim1, dim2, std=5.0)` | 计算形如 `(2*dim1-1, 2*dim2-1)` 的二维高斯窗，用于构造 $\omega$。 |
| `get_attention_addition(dim1, dim2, window, adjust_for_cls=True)` | 将高斯窗展开为完整的 `(L, L)` 注意力偏置矩阵，可选为 `[CLS]` token 补零。 |
| `compute_naclip_attention(k, scale, n_patches, gaussian_std=5.0, addition_cache=None)` | 计算 $A_s = \mathrm{softmax}(kk^\top/\sqrt{d} + \omega)$，支持 omega 矩阵缓存。 |

### 修改文件：`cat_seg/third_party/model.py`

#### `ResidualAttentionBlock` 新增方法

| 方法 | 说明 |
|------|------|
| `forward_nar(x, n_patches, gaussian_std, A_c, lambda_rca, addition_cache)` | NAR 前向：调用 `compute_naclip_attention` 计算 $A_s$；若提供 $A_c$ 则按公式 (6) 融合；**不含残差连接和 FFN**（NACLIP 精简架构）。 |
| `forward_with_attn(x)` *(已有)* | 标准前向 + 返回 Q-K 注意力权重，用于收集中间层 $A_c$。 |
| `forward_dense_rcs(x, A_c, lambda_rca)` *(已有)* | 原 RCS 前向（标准 QK $A_s$，保留残差+FFN）。 |

#### `Transformer.forward` 新增参数

```python
Transformer.forward(
    x, dense=False,
    use_rcs=False,           # 原 RCS 路径（QK A_s，保留 FFN）
    use_nar=False,           # 新 NAR 路径（kk+Gaussian A_s，去除 FFN）
    n_patches=None,          # 空间 patch 网格 (h, w)，use_nar=True 时必须提供
    gaussian_std=5.0,        # 高斯 sigma
    rcs_start=11, rcs_end=19, lambda_rca=0.5,
    addition_cache=None      # omega 缓存字典
)
```

`use_nar=True` 优先级高于 `use_rcs=True`。

#### `VisualTransformer` 修改

- `__init__` 新增 `self._na_addition_cache = {}`，在实例内缓存 $\omega$ 矩阵。  
- `forward` 在 `conv1` 后自动计算 `n_patches = (grid_h, grid_w)`，传递给 `Transformer`。  
- 新增 `use_nar` 和 `gaussian_std` 参数，层范围默认值更新为：
  - ViT-L: `rcs_end=19`（对应论文 $e=19$）
  - ViT-B: `rcs_end=9`（对应论文 $e=9$）

#### `CLIP.encode_image` 新增参数

```python
CLIP.encode_image(
    image, masks=None, pool_mask=None, dense=False,
    use_rcs=False, use_nar=False, gaussian_std=5.0,
    rcs_start=None, rcs_end=None, lambda_rca=0.5
)
```

### 修改文件：`cat_seg/cat_seg_model_ours.py`

所有 `encode_image` 调用由 `use_rcs=True` 改为 `use_nar=True`：

```python
# 训练/推理主循环
clip_features = clip_model.encode_image(clip_images_resized, dense=True, use_nar=True)

# 推理滑窗
clip_features = clip_model.encode_image(clip_images, dense=True, use_nar=True)
```

---

## 使用说明 / Usage

### 启用 NAR（默认）

无需额外配置，`use_nar=True` 已在 `cat_seg_model_ours.py` 中启用。

### 仅用 RCS（原实现回退）

```python
clip_features = clip_model.encode_image(images, dense=True, use_rcs=True)
```

### 禁用所有增强

```python
clip_features = clip_model.encode_image(images, dense=True)
```

### 自定义参数

```python
clip_features = clip_model.encode_image(
    images, dense=True,
    use_nar=True,
    gaussian_std=5.0,    # NACLIP Gaussian sigma
    rcs_start=11,        # ViT-L 起始层（0-based）
    rcs_end=19,          # ViT-L 终止层（0-based）
    lambda_rca=0.5       # 融合权重
)
```

---

## 参考 / References

- NACLIP: <https://github.com/sinahmr/NACLIP>  
  Hajimiri et al., "Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation", WACV 2025.
- ResCLIP: <https://github.com/xtxduers/ResCLIP>  
  "Residual Cross-correlation Self-attention for Open-Vocabulary Semantic Segmentation"
- CAT-Seg: <https://github.com/KU-CVLAB/CAT-Seg>
