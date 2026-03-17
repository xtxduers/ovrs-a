"""add_NA.py — Neighbour-Aware (NA) Attention Module for OVRS-A.

Implements the NACLIP-style kk-similarity + 2D Gaussian neighbourhood
attention for the final CLIP ViT encoder block, combined with ResCLIP's
Residual Cross-correlation Self-attention (RCS) from intermediate layers.

Mathematical summary
--------------------
NACLIP attention (A_s):
    A_s = softmax(kk^T / sqrt(d) + omega((i,j); sigma)) * V
    omega((i,j); sigma)_{mn} = exp(-||(m,n)-(i,j)||^2 / (2*sigma^2))

    - kk^T captures patch self-similarity (semantic attributes).
    - omega is a 2-D Gaussian centred at the current patch (sigma=5).
    - The last encoder block's FFN is removed (NACLIP "reduced" arch).

ResCLIP cross-layer aggregation (A_c):
    A_c = (1/N) * sum_{i=s}^{e} A_qk^i
    ViT-B (12 layers): s=5, e=9  (0-based, inclusive)
    ViT-L (24 layers): s=11, e=19 (0-based, inclusive)

RCS attention blending (A_rca):
    A_rca = (1 - lambda_rca) * A_s + lambda_rca * A_c
    lambda_rca = 0.5 (default)

References
----------
- NACLIP: https://github.com/sinahmr/NACLIP
- ResCLIP: https://github.com/xtxduers/ResCLIP
"""

import math

import torch
import torch.nn.functional as F


def gaussian_window(dim1: int, dim2: int, std: float = 5.0) -> torch.Tensor:
    """Compute a 2-D Gaussian kernel of shape (2*dim1-1, 2*dim2-1).

    The kernel is centred at the origin and is used as the convolution
    window to build the full neighbourhood attention addition matrix.

    Args:
        dim1: height of the patch grid.
        dim2: width  of the patch grid.
        std:  Gaussian standard deviation sigma; default 5.0 per NACLIP.

    Returns:
        FloatTensor of shape (2*dim1-1, 2*dim2-1).
    """
    constant = 1.0 / (std * math.sqrt(2.0))
    ks = []
    for dim in [dim1, dim2]:
        start = -(dim - 1) / 2.0
        k = torch.linspace(
            start=start * constant,
            end=(start + (dim - 1)) * constant,
            steps=dim,
            dtype=torch.float,
        )
        ks.append(k)
    dist_sq = (torch.stack(torch.meshgrid(*ks, indexing="ij")) ** 2).sum(0)
    return torch.exp(-dist_sq)


def get_attention_addition(
    dim1: int,
    dim2: int,
    window: torch.Tensor,
    adjust_for_cls: bool = True,
) -> torch.Tensor:
    """Build the full (L, L) Gaussian attention-addition matrix.

    For each patch position (i, j) in a (dim1 x dim2) grid, compute the
    Gaussian decay to every other position (m, n):
        omega((i,j); sigma)_{mn} = exp(-||(m,n)-(i,j)||^2 / (2*sigma^2))

    This is done efficiently by convolving identity matrices with the
    Gaussian window, following the NACLIP implementation.

    Args:
        dim1: height of the patch grid.
        dim2: width  of the patch grid.
        window: Gaussian window of shape (2*dim1-1, 2*dim2-1).
        adjust_for_cls: if True prepend a zero row/column for [CLS] token.

    Returns:
        FloatTensor of shape (L+1, L+1) if adjust_for_cls else (L, L),
        where L = dim1 * dim2.
    """
    m = torch.einsum("ij,kl->ijkl", torch.eye(dim1), torch.eye(dim2))
    m = m.permute((0, 3, 1, 2)).contiguous()
    out = F.conv2d(
        m.view(-1, dim1, dim2).unsqueeze(1),
        window.unsqueeze(0).unsqueeze(1),
        padding="same",
    ).squeeze(1)
    out = out.view(dim1 * dim2, dim1 * dim2)
    if adjust_for_cls:
        v_adjusted = torch.vstack([torch.zeros((1, dim1 * dim2)), out])
        out = torch.hstack([torch.zeros((dim1 * dim2 + 1, 1)), v_adjusted])
    return out


def compute_naclip_attention(
    k: torch.Tensor,
    scale: float,
    n_patches: tuple,
    gaussian_std: float = 5.0,
    addition_cache: dict = None,
) -> torch.Tensor:
    """Compute NACLIP attention weights A_s = softmax(kk^T/sqrt(d) + omega).

    Using kk^T (key self-correlation) instead of the standard qk^T focuses
    on patch semantic attributes, which is better suited for segmentation.
    The 2-D Gaussian bias omega enforces neighbourhood locality.

    Args:
        k:             Key tensor of shape (N*H, L, head_dim).
        scale:         Attention scale 1/sqrt(head_dim).
        n_patches:     Tuple (h, w) — spatial patch grid size (no [CLS]).
        gaussian_std:  Sigma for the Gaussian; default 5.0 per NACLIP.
        addition_cache: Optional dict for caching omega matrices.  The
                        cache stores CPU tensors keyed by n_patches.

    Returns:
        A_s: Softmax attention weights of shape (N*H, L, L),
             where L = h*w + 1 (includes [CLS] token).
    """
    cache_key = n_patches
    if addition_cache is not None and cache_key in addition_cache:
        omega = addition_cache[cache_key].to(dtype=k.dtype, device=k.device)
    else:
        h, w = n_patches
        window_size = (h * 2 - 1, w * 2 - 1)
        window = gaussian_window(*window_size, std=gaussian_std)
        omega = get_attention_addition(h, w, window)
        omega = omega.unsqueeze(0)  # (1, L, L) — broadcast over heads
        if addition_cache is not None:
            addition_cache[cache_key] = omega.cpu()
        omega = omega.to(dtype=k.dtype, device=k.device)

    attn_weights = torch.bmm(k * scale, k.transpose(1, 2)) + omega
    return F.softmax(attn_weights, dim=-1)
