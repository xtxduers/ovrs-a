# RCS Attention Modification for OVRS-A

## Background

This document describes the **Residual Cross-correlation Self-attention (RCS)** modification
applied to the OVRS-A model's CLIP visual encoder, inspired by the
[ResCLIP](https://github.com/xtxduers/ResCLIP) method.

---

## Method Overview

### Original Dense Attention (CATSeg baseline)

In the original CATSeg codebase the last ViT layer is run in a **dense** mode
(`forward_dense`) that skips the standard attention computation and uses the
**value projection output directly** (i.e. treats the attended features as equal
to the linearly-projected values, ignoring the softmax-weighted aggregation).
While this improves spatial resolution of the feature map, it loses the
cross-patch correspondence information captured by Query–Key (QK) attention.

### RCS (Residual Cross-correlation Self-attention)

ResCLIP proposes to enrich the last-layer attention of a frozen ViT by
re-introducing cross-patch structure through an aggregated attention map
collected from intermediate layers.

Let `A_qk^i` denote the per-head QK softmax attention map of layer `i`
(the **C²SA** attention, i.e. `softmax(QK^T / sqrt(d))`). The aggregated map is:

```
A_c = (1/N) * sum_{i=s}^{e} A_qk^i,   N = e - s + 1
```

The **RCS attention** for the final layer is then:

```
A_rca = (1 - λ_rca) · A_s  +  λ_rca · A_c        (λ_rca = 0.5)
```

where `A_s` is the standard self-attention of the last layer.

The final dense output uses `A_rca @ V` (instead of `V` alone), restoring
global correspondence while retaining the local patch structure encoded in `A_s`.

### Layer range for aggregation

| Backbone        | ViT layers | `s` (1-indexed) | `e` (1-indexed) | `N` |
|-----------------|-----------|-----------------|-----------------|-----|
| ViT-B/16        | 12        | 6               | 9               | 4   |
| ViT-L/14@336px  | 24        | 12              | 23              | 12  |

---

## Files Modified

| File | Change |
|------|--------|
| `cat_seg/third_party/model.py` | Added `get_qk_attn_weights()` and `forward_dense_rcs()` to `ResidualAttentionBlock`; modified `Transformer.forward()`, `VisualTransformer.forward()`, and `CLIP.encode_image()` to accept `use_rcs` / `rcs_start` / `rcs_end` / `lambda_rca` parameters |
| `cat_seg/cat_seg_model_ours.py` | Set `rcs_start` / `rcs_end` from `clip_pretrained`; passes `use_rcs=True` to all four `encode_image()` calls (original image + 3 rotations) |
| `cat_seg/__init__.py` | Changed import from `cat_seg_model` to `cat_seg_model_ours` so the rotation-augmented OVRS-A model is registered |

---

## How It Works (code walk-through)

### `get_qk_attn_weights(x)` — `ResidualAttentionBlock`

Computes `softmax(QK^T / sqrt(head_dim))` from the block's input `x` **without
modifying the activations** (i.e. the normal residual forward still runs
afterwards). Returns a tensor of shape `[N * num_heads, L, L]`.

### `forward_dense_rcs(x, attn_c, lambda_rca)` — `ResidualAttentionBlock`

Replaces `forward_dense` in the **last** ViT block:

1. Compute `Q`, `K`, `V` from `in_proj`.
2. Compute `A_s = softmax(QK^T / sqrt(d))`.
3. If `attn_c` is provided: `A_rca = (1 - λ) A_s + λ A_c`.
4. Output = `A_rca @ V`, then `out_proj`, then residual + MLP.

### `Transformer.forward(use_rcs, rcs_start, rcs_end, lambda_rca)`

When `use_rcs=True` and `dense=True`:

- For layers `[rcs_start-1, rcs_end-1]` (0-indexed): calls
  `get_qk_attn_weights(x)` **before** the standard layer forward and collects
  the results.
- For the last layer: stacks collected maps, averages them into `A_c`, then
  calls `forward_dense_rcs`.

The default path (`use_rcs=False`) is unchanged.

---

## How to Pull and Run on a Remote Server

### 1. Pull the code

```bash
# If you don't have the repo yet
git clone https://github.com/xtxduers/ovrs-a.git
cd ovrs-a

# If you already have the repo
git fetch origin
git checkout copilot/modify-ovrs-a-attention-mechanism
git pull
```

### 2. Install dependencies

```bash
# Create a conda environment (Python 3.8 recommended)
conda create -n ovrs_a python=3.8 -y
conda activate ovrs_a

# Install PyTorch (CUDA 11.8 example; adjust to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install detectron2 (match your torch/CUDA versions)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3. Prepare datasets

Follow the original CATSeg / OVRS-A dataset preparation instructions.
The expected directory structure is:

```
datasets/
  coco.json
  iSAID.json
  DLRSD.json
  Potsdam.json
  Vaihingen.json
  ...
```

### 4. Download pre-trained weights

```bash
# Example: ViT-L/14@336px CATSeg checkpoint
mkdir -p OUTPUT_DIR
# Place your pre-trained model_final.pth into OUTPUT_DIR/
```

### 5. Evaluate

```bash
# Evaluate on all four remote-sensing benchmarks at once:
sh eval.sh configs/vitl_336.yaml 1 OUTPUT_DIR

# Or evaluate a single benchmark, e.g. iSAID:
python train_net.py \
  --config configs/vitl_336.yaml \
  --num-gpus 1 \
  --eval-only \
  OUTPUT_DIR OUTPUT_DIR/eval_iSAID \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON datasets/iSAID.json \
  DATASETS.TEST '("iSAID_all_sem_seg",)' \
  TEST.SLIDING_WINDOW True \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
  MODEL.WEIGHTS OUTPUT_DIR/model_final.pth
```

### 6. Train from scratch (optional)

```bash
sh run.sh configs/vitl_336.yaml 4 OUTPUT_DIR
```

The `run.sh` script runs training followed by evaluation on all benchmarks.

---

## Key Hyperparameters

| Parameter    | Default | Description |
|-------------|---------|-------------|
| `use_rcs`    | `True`  | Enable RCS attention in the final dense ViT layer |
| `lambda_rca` | `0.5`   | Blending coefficient between `A_s` and `A_c` |
| `rcs_start`  | 12 (ViT-L) / 6 (ViT-B) | First 1-indexed layer to aggregate |
| `rcs_end`    | 23 (ViT-L) / 9 (ViT-B) | Last 1-indexed layer to aggregate |

To disable RCS and revert to the original dense forward, simply pass
`use_rcs=False` to `encode_image()`, or remove the `use_rcs=True` arguments in
`cat_seg_model_ours.py`.
