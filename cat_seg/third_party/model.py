from collections import OrderedDict
from typing import Tuple, Union
import warnings

import torch
import torch.nn.functional as F
from torch import nn

try:
    from cat_seg.add_NA import compute_naclip_attention
except ImportError:
    compute_naclip_attention = None


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.mask_pre_mlp = True

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_dense(self, x: torch.Tensor):
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        L, N, D = y.shape # L N 3D
        
        y = y.reshape(L, N, 3, D // 3).permute(2, 1, 0, 3).reshape(3 * N, L, D // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        
        q, k, v = y.tensor_split(3, dim=0)
        v = v.transpose(1, 0) + x # L N D

        v = v + self.mlp(self.ln_2(v))
        return v

    def forward_with_attn(self, x: torch.Tensor):
        """Normal forward pass that also returns Q-K cross-correlation attention weights.

        Returns:
            x: updated hidden states, same as forward()
            attn_weights: Q-K attention weights of shape (N*num_heads, L, L)
        """
        num_heads = self.attn.num_heads
        L, N, D = x.shape
        head_dim = D // num_heads
        scale = head_dim ** -0.5

        y = self.ln_1(x)
        qkv = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # each (L, N, D)

        # Reshape to (N*num_heads, L, head_dim)
        q = q.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)

        # Q-K cross-correlation attention weights
        attn_weights = torch.bmm(q * scale, k.transpose(1, 2))  # (N*H, L, L)
        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute attention output
        attn_output = torch.bmm(attn_weights, v)  # (N*H, L, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, D)
        attn_output = F.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)

        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights  # attn_weights: (N*H, L, L)

    def forward_nar(
        self,
        x: torch.Tensor,
        n_patches: tuple,
        gaussian_std: float = 5.0,
        A_c: torch.Tensor = None,
        lambda_rca: float = 0.5,
        addition_cache: dict = None,
    ) -> torch.Tensor:
        """NAR (Neighbour-Aware + RCS) forward for the final ViT block.

        Implements the combined NACLIP + ResCLIP attention mechanism:
            A_s  = softmax(kk^T / sqrt(d) + omega)   [NACLIP, eq. 1]
            A_rca = (1-lambda) * A_s + lambda * A_c  [RCS blending, eq. 6]

        Following NACLIP's "reduced" architecture (Z^(L) = SA_f(LN(Z^(L-1)))),
        both the residual connection and the FFN are omitted in the last block.

        Args:
            x:             Input tensor of shape (L, N, D).
            n_patches:     Spatial patch grid (h, w) excluding [CLS].
            gaussian_std:  Gaussian sigma for neighbourhood attention (default 5.0).
            A_c:           Aggregated cross-correlation attention (N*H, L, L)
                           from intermediate layers (ResCLIP). If None only A_s
                           is used.
            lambda_rca:    Blending weight lambda_rca (default 0.5).
            addition_cache: Optional dict for caching omega matrices.

        Returns:
            Output tensor of shape (L, N, D) — NO residual, NO FFN.
        """
        if compute_naclip_attention is None:
            raise ImportError(
                "cat_seg.add_NA is not importable; cannot use forward_nar."
            )

        num_heads = self.attn.num_heads
        L, N, D = x.shape
        head_dim = D // num_heads
        scale = head_dim ** -0.5

        y = self.ln_1(x)
        qkv = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        k = k.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)

        # A_s: NACLIP kk^T + Gaussian neighbourhood attention
        A_s = compute_naclip_attention(k, scale, n_patches, gaussian_std, addition_cache)

        if A_c is not None:
            # RCS blending: A_rca = (1 - lambda_rca) * A_s + lambda_rca * A_c
            A_rca = (1.0 - lambda_rca) * A_s + lambda_rca * A_c
        else:
            A_rca = A_s

        attn_output = torch.bmm(A_rca, v)  # (N*H, L, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, D)
        attn_output = F.linear(
            attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias
        )

        # NACLIP "reduced" arch: no residual addition, no FFN
        return attn_output

    def forward_dense_rcs(self, x: torch.Tensor, A_c: torch.Tensor, lambda_rca: float = 0.5):
        """Dense forward for the last ViT layer with RCS (Residual Cross-correlation Self-attention).

        Implements: A_rca = (1 - lambda_rca) * A_s + lambda_rca * A_c
        where A_s is the Q-K attention of this layer and A_c is the aggregated
        cross-correlation attention from intermediate layers.

        Args:
            x: input of shape (L, N, D)
            A_c: aggregated attention map of shape (N*num_heads, L, L)
            lambda_rca: blending weight (default 0.5)
        Returns:
            updated hidden states of shape (L, N, D)
        """
        num_heads = self.attn.num_heads
        L, N, D = x.shape
        head_dim = D // num_heads
        scale = head_dim ** -0.5

        y = self.ln_1(x)
        qkv = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # each (L, N, D)

        # Reshape to (N*num_heads, L, head_dim)
        q = q.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(L, N * num_heads, head_dim).transpose(0, 1)

        # Compute A_s: Q-K cross-correlation attention for the last layer
        A_s = torch.bmm(q * scale, k.transpose(1, 2))  # (N*H, L, L)
        A_s = F.softmax(A_s, dim=-1)

        # RCS formula: A_rca = (1 - lambda_rca) * A_s + lambda_rca * A_c
        A_rca = (1.0 - lambda_rca) * A_s + lambda_rca * A_c  # (N*H, L, L)

        # Compute attention output with blended weights
        attn_output = torch.bmm(A_rca, v)  # (N*H, L, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, D)
        attn_output = F.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)

        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, dense=False, use_rcs=False, use_nar=False,
                n_patches=None, gaussian_std=5.0,
                rcs_start=11, rcs_end=22, lambda_rca=0.5, addition_cache=None):
        """Forward pass with optional RCS or NAR attention.

        Args:
            x: input tensor (L, N, D)
            dense: if True the last block uses forward_dense / forward_dense_rcs
            use_rcs: if True collect intermediate Q-K attention maps and apply RCS
                     (standard QK A_s) in the final block
            use_nar: if True apply NAR (Neighbour-Aware + RCS): uses NACLIP
                     kk+Gaussian as A_s and removes FFN from the last block.
                     Takes precedence over use_rcs when both are True.
            n_patches: spatial patch grid (h, w) required when use_nar=True
            gaussian_std: Gaussian sigma for NACLIP omega (default 5.0)
            rcs_start: first intermediate layer index (0-based, inclusive)
            rcs_end:   last  intermediate layer index (0-based, inclusive)
            lambda_rca: blending weight for RCS/NAR (default 0.5)
            addition_cache: optional dict for caching Gaussian omega matrices
        """
        if not use_rcs and not use_nar:
            for i, resblock in enumerate(self.resblocks):
                if i == self.layers - 1 and dense:
                    x = resblock.forward_dense(x)
                else:
                    x = resblock(x)
            return x

        # --- RCS / NAR path: collect Q-K attention from intermediate layers ---
        attn_maps = []
        for i, resblock in enumerate(self.resblocks[:-1]):  # all except last
            if rcs_start <= i <= rcs_end:
                x, attn_w = resblock.forward_with_attn(x)
                attn_maps.append(attn_w)
            else:
                x = resblock(x)

        last_block = self.resblocks[-1]
        A_c = torch.stack(attn_maps).mean(dim=0) if attn_maps else None

        if A_c is None:
            warnings.warn(
                f"use_rcs/use_nar=True but no attention maps were collected "
                f"(rcs_start={rcs_start}, rcs_end={rcs_end}, layers={self.layers}). "
                "Falling back to standard forward pass.",
                RuntimeWarning,
                stacklevel=2,
            )
            if dense:
                x = last_block.forward_dense(x)
            else:
                x = last_block(x)
            return x

        if use_nar:
            # NAR: NACLIP kk+Gaussian A_s, no FFN, blend with A_c
            x = last_block.forward_nar(
                x, n_patches, gaussian_std, A_c, lambda_rca, addition_cache
            )
        else:
            # Original RCS: QK A_s, with residual+FFN, blend with A_c
            x = last_block.forward_dense_rcs(x, A_c, lambda_rca)
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.patch_size = patch_size
        self.input_resolution = input_resolution

        # Cache for NACLIP Gaussian omega matrices (keyed by n_patches tuple)
        self._na_addition_cache = {}

    def forward(self, x: torch.Tensor, dense=False, use_rcs=False, use_nar=False,
                gaussian_std=5.0, rcs_start=None, rcs_end=None, lambda_rca=0.5):
        """Forward pass with optional RCS or NAR attention.

        Args:
            x: input images
            dense: whether to use the dense (all-token) forward
            use_rcs: apply Residual Cross-correlation Self-attention
                     (standard QK A_s, with residual+FFN in last block)
            use_nar: apply Neighbour-Aware + RCS attention (NACLIP kk+Gaussian
                     A_s, no FFN in last block, blended with A_c from intermediate
                     layers). use_nar takes precedence over use_rcs.
            gaussian_std: Gaussian sigma for NACLIP omega (default 5.0)
            rcs_start: first intermediate layer (0-based) for aggregated A_c;
                       auto-set per encoder depth if None
            rcs_end:   last  intermediate layer (0-based) for aggregated A_c;
                       auto-set per encoder depth if None
            lambda_rca: blending weight lambda_rca (default 0.5)
        """
        x_conv = self.conv1(x)  # shape = [*, width, grid_h, grid_w]
        n_patches = (x_conv.shape[-2], x_conv.shape[-1])  # (grid_h, grid_w)

        x = x_conv.reshape(x_conv.shape[0], x_conv.shape[1], -1)  # [*, width, grid**2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if dense and (x.shape[1] != self.positional_embedding.shape[0]):
            x = x + self.resized_pos_embed(self.input_resolution, x.shape[1]).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        need_rcs_range = use_rcs or use_nar
        if need_rcs_range:
            num_layers = self.transformer.layers
            # Default A_c aggregation range (0-based, inclusive):
            # ViT-L (24 layers): s=11, e=19  (problem spec: s=11, e=19)
            # ViT-B (12 layers): s=5,  e=9   (problem spec: s=5, e=9)
            if rcs_start is None:
                if num_layers == 24:
                    rcs_start = 11
                elif num_layers == 12:
                    rcs_start = 5
                else:
                    rcs_start = num_layers // 2 - 1
            if rcs_end is None:
                if num_layers == 24:
                    rcs_end = 19
                elif num_layers == 12:
                    rcs_end = 9
                else:
                    rcs_end = num_layers - 2

        x = self.transformer(
            x, dense,
            use_rcs=use_rcs and not use_nar,
            use_nar=use_nar,
            n_patches=n_patches,
            gaussian_std=gaussian_std,
            rcs_start=rcs_start if need_rcs_range else 0,
            rcs_end=rcs_end if need_rcs_range else 0,
            lambda_rca=lambda_rca,
            addition_cache=self._na_addition_cache if use_nar else None,
        )
        x = x.permute(1, 0, 2)  # LND -> NLD

        if dense:
            x = self.ln_post(x[:, :, :])
        else:
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        
        return x

    def resized_pos_embed(self, in_res, tgt_res, mode="bicubic"):
        #assert L == (input_resolution // self.patch_size) ** 2 + 1
        L, D = self.positional_embedding.shape
        
        in_side = in_res // self.patch_size
        #tgt_side = tgt_res // self.patch_size
        tgt_side = int((tgt_res - 1) ** 0.5)
        
        cls_pos = self.positional_embedding[0].unsqueeze(0)  # 1 D
        pos_embed = self.positional_embedding[1:].reshape(1, in_side, in_side, D).permute(0, 3, 1, 2) # L-1 D -> 1 D S S
        resized_pos_embed = F.interpolate(pos_embed, size=(tgt_side, tgt_side), mode=mode, align_corners=False,) # 1 D S S -> 1 D S' S'
        resized_pos_embed = resized_pos_embed.squeeze(0).reshape(D, -1).T # L'-1 D

        return torch.cat((cls_pos, resized_pos_embed), dim=0)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        
        self.image_resolution = image_resolution


        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


    def encode_image(self, image, masks=None, pool_mask=None, dense=False,
                     use_rcs=False, use_nar=False, gaussian_std=5.0,
                     rcs_start=None, rcs_end=None, lambda_rca=0.5):
        """Encode an image with optional RCS or NAR attention.

        Args:
            image: input image tensor
            masks: optional attention masks
            pool_mask: optional pooling mask
            dense: use dense (patch-level) features
            use_rcs: apply RCS (QK A_s, residual+FFN kept in last block)
            use_nar: apply NAR — Neighbour-Aware + RCS (NACLIP kk+Gaussian A_s,
                     no FFN in last block, blended with A_c). use_nar takes
                     precedence over use_rcs when both are True.
            gaussian_std: Gaussian sigma for NACLIP omega (default 5.0)
            rcs_start: first intermediate layer (0-based) for aggregated A_c
            rcs_end:   last  intermediate layer (0-based)
            lambda_rca: blending weight (default 0.5)
        """
        if pool_mask is not None:
            return self.visual(image.type(self.dtype), mask=pool_mask, dense=dense,
                               use_rcs=use_rcs, use_nar=use_nar, gaussian_std=gaussian_std,
                               rcs_start=rcs_start, rcs_end=rcs_end, lambda_rca=lambda_rca)
        if masks is None:
            return self.visual(image.type(self.dtype), dense=dense,
                               use_rcs=use_rcs, use_nar=use_nar, gaussian_std=gaussian_std,
                               rcs_start=rcs_start, rcs_end=rcs_end, lambda_rca=lambda_rca)
        else:
            return self.visual(image.type(self.dtype), masks.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # import pdb; pdb.set_trace()
        # normalized features
        # image_features shape: [1, 1024]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_iamge = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_iamge, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
