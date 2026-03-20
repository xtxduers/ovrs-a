"""
NACLIP Teacher Module for Teacher-Student Knowledge Distillation.

Implements the NACLIP (Neighbour-Aware CLIP) visual encoder modification:
  - Last self-attention block uses kk^T similarity (instead of qk^T)
  - Gaussian spatial bias is added to the attention scores:
        omega((i,j); sigma)_{mn} = exp(-||(m,n) - (i,j)||^2 / (2*sigma^2))
  - The FFN is removed from the last encoder block (reduced architecture)
  - All parameters are fully frozen

Reference: "Pay Attention to Your Neighbours: Training-Free Open-Vocabulary
Semantic Segmentation" (WACV 2025), Hajimiri et al.
https://github.com/sinahmr/NACLIP
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class NAClipTeacher(nn.Module):
    """
    Teacher model based on NACLIP modifications to CLIP's VisualTransformer.

    The last transformer block is replaced with a neighbour-aware self-attention
    that uses kk^T similarity and a learnable-free Gaussian spatial bias.
    The feed-forward network (FFN) is omitted from that block (reduced arch).
    All parameters are frozen; no gradients flow through the teacher.

    Args:
        clip_model: a CLIP model loaded from cat_seg.third_party.clip.load().
                    The model must have a ``visual`` attribute that is a
                    VisualTransformer with attributes:
                      conv1, class_embedding, positional_embedding, ln_pre,
                      ln_post, proj, patch_size, input_resolution,
                      transformer.resblocks  (nn.Sequential of ResidualAttentionBlock).
        gaussian_std: standard deviation (sigma) of the Gaussian spatial bias.
                      Default: 5.0 (as used in the NACLIP paper).
    """

    def __init__(self, clip_model: nn.Module, gaussian_std: float = 5.0) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.gaussian_std = gaussian_std
        # Cache {(H, W): addition_tensor} to avoid recomputing for same grid size.
        self._addition_cache: dict = {}

        # Freeze every parameter.
        for param in self.clip_model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Static helpers (identical to NACLIP's VisionTransformer helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_window(dim1: int, dim2: int, std: float = 1.0) -> torch.Tensor:
        """2D Gaussian kernel used to build the spatial attention bias."""
        constant = 1.0 / (std * math.sqrt(2))
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

    @staticmethod
    def _get_attention_addition(
        dim1: int, dim2: int, window: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the (H*W+1) x (H*W+1) Gaussian spatial attention bias matrix.

        A [CLS] row and column of zeros is prepended so the bias can be added
        directly to the full attention weight matrix that includes the CLS token.

        Returns:
            Tensor of shape [1 + H*W, 1 + H*W].
        """
        m = torch.einsum("ij,kl->ijkl", torch.eye(dim1), torch.eye(dim2))
        m = m.permute(0, 3, 1, 2).contiguous()
        out = F.conv2d(
            m.view(-1, dim1, dim2).unsqueeze(1),
            window.unsqueeze(0).unsqueeze(1),
            padding="same",
        ).squeeze(1)
        out = out.view(dim1 * dim2, dim1 * dim2)
        # Prepend zeros for the CLS token (row + column).
        with_cls_row = torch.vstack([torch.zeros(1, dim1 * dim2), out])
        out = torch.hstack([torch.zeros(dim1 * dim2 + 1, 1), with_cls_row])
        return out  # [H*W+1, H*W+1]

    # ------------------------------------------------------------------
    # NACLIP attention
    # ------------------------------------------------------------------

    def _naclip_attn(
        self,
        attn_layer: nn.MultiheadAttention,
        x: torch.Tensor,
        n_patches: tuple,
    ) -> torch.Tensor:
        """
        NACLIP neighbour-aware self-attention for the last transformer block.

        Computes:
            A_s = softmax( kk^T / sqrt(d) + omega(sigma) ) * v

        where omega is the 2D Gaussian spatial bias centred at each patch.

        Args:
            attn_layer: the nn.MultiheadAttention of the last block.
            x: layer-normed input of shape [L, B, D]  (L = 1+H*W tokens).
            n_patches: (H, W) number of patches along each spatial dimension.

        Returns:
            Output tensor of shape [L, B, D].
        """
        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        # Project to Q, K, V.
        # The VPT variant uses separate q/k/v projection weights instead of a
        # single fused in_proj_weight (see cat_seg/third_party/model_vpt.py).
        if attn_layer.in_proj_weight is not None:
            qkv = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias)
        else:
            proj = torch.cat(
                [attn_layer.q_proj_weight, attn_layer.k_proj_weight, attn_layer.v_proj_weight],
                dim=0,
            )
            qkv = F.linear(x, proj, attn_layer.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # kk^T attention weights.
        attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale

        # Gaussian spatial bias (cached per patch-grid size).
        addition = self._addition_cache.get(n_patches)
        if addition is None:
            window_size = [side * 2 - 1 for side in n_patches]
            window = self._gaussian_window(*window_size, std=self.gaussian_std)
            addition = self._get_attention_addition(*n_patches, window)
            addition = addition.unsqueeze(0).to(dtype=x.dtype, device=x.device)
            self._addition_cache[n_patches] = addition

        attn_weights = attn_weights + addition
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of values.
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            -1, bsz, embed_dim
        )
        attn_output = attn_layer.out_proj(attn_output)
        return attn_output  # [L, B, D]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Dense image encoding with NACLIP modifications (reduced architecture).

        Processing:
          1. Patch embedding + CLS prepending + positional encoding.
          2. All transformer blocks except the last run in standard mode.
          3. Last block: NACLIP kk^T + Gaussian attention, **no FFN**.
             The standard residual-attention block is entirely replaced by
             ``_naclip_attn``; the block's own residual add-back is not used.
          4. ln_post + proj applied to all tokens.

        Args:
            image: input images of shape [B, 3, H, W].

        Returns:
            Dense features of shape [B, 1+N, D]  (1+N = CLS + patch tokens).
        """
        visual = self.clip_model.visual
        x = image.type(visual.conv1.weight.dtype)
        B, _nc, w, h = x.shape
        n_patches = (w // visual.patch_size, h // visual.patch_size)

        # Patch embedding.
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Prepend CLS token.
        cls_tokens = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+N, D]

        # Positional encoding (with bicubic interpolation if needed).
        if x.shape[1] != visual.positional_embedding.shape[0]:
            x = x + visual.resized_pos_embed(
                visual.input_resolution, x.shape[1]
            ).to(x.dtype)
        else:
            x = x + visual.positional_embedding.to(x.dtype)

        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Standard transformer blocks (all except the last).
        for blk in visual.transformer.resblocks[:-1]:
            x = blk(x)

        # Last block: NACLIP neighbour-aware attention, no FFN, no residual.
        last_blk = visual.transformer.resblocks[-1]
        x = self._naclip_attn(last_blk.attn, last_blk.ln_1(x), n_patches)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj

        return x  # [B, 1+N, D]

    @torch.no_grad()
    def get_similarity_map(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the teacher's class-response map R^T.

        R^T_{ijc} = cosine_similarity( image_feat_{ij}, text_feat_c )

        Args:
            image: input images of shape [B, 3, H, W].
            text_features: **normalised** text class embeddings of shape [C, D].

        Returns:
            R^T of shape [B, N, C]  where N = H*W / patch_size^2.
        """
        feats = self.encode_image(image)       # [B, 1+N, D]
        patch = feats[:, 1:, :]               # [B, N, D]  (exclude CLS)
        patch = F.normalize(patch.float(), dim=-1)
        text_features = text_features.float()
        return torch.einsum("bnd,cd->bnc", patch, text_features)  # [B, N, C]
