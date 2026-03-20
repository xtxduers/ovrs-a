# Copyright (c) Facebook, Inc. and its affiliates.
"""
Teacher-Student CATSeg model (CATSegTS).

Architecture
------------
Teacher:
  - CLIP visual encoder with NACLIP modifications:
      * Last self-attention block uses kk^T + Gaussian spatial bias.
      * FFN removed from last block (reduced architecture).
  - Visual encoder is fully frozen.
  - Text encoder is fully frozen.

Student:
  - Standard CATSeg CLIP visual encoder (fine-tunable, see CLIP_FINETUNE).
  - Text encoder is fully frozen.
  - Aggregator head is fully trainable.

Total loss
----------
  L_total = L_seg + lambda_distill * L_distill

  L_seg      : binary cross-entropy segmentation loss (same as base CATSeg).
  L_distill  : high-confidence knowledge-distillation loss.

Distillation loss
-----------------
  L_distill = mean over masked positions of |R^S_{ijc} - R^T_{ijc}|_1

  where the confidence mask is I(max_c(R^T_{ij}) > theta).

  R_{ijc} = cosine_similarity(image_patch_feat_{ij}, text_feat_c).

Config keys (under MODEL.TS)
----------------------------
  DISTILL_LAMBDA  : weight lambda for the distillation loss.    Default: 0.5
  DISTILL_THETA   : confidence threshold theta.                 Default: 0.1
  GAUSSIAN_STD    : sigma for the teacher Gaussian spatial bias. Default: 5.0
"""

import copy
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from einops import rearrange

from cat_seg.modeling.naclip_teacher import NAClipTeacher


@META_ARCH_REGISTRY.register()
class CATSegTS(nn.Module):
    """
    Teacher-Student CATSeg model.

    Wraps the existing CATSeg student and attaches a frozen NACLIP teacher
    that provides high-confidence distillation targets during training.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        distill_lambda: float,
        distill_theta: float,
        gaussian_std: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head

        if size_divisibility < 0:
            size_divisibility = (
                self.backbone.size_divisibility if self.backbone is not None else 32
            )
        self.size_divisibility = size_divisibility

        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False
        )
        self.register_buffer(
            "clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False
        )

        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        self.distill_lambda = distill_lambda
        self.distill_theta = distill_theta

        # ------------------------------------------------------------------
        # Student CLIP parameter setup
        # ------------------------------------------------------------------
        self.clip_finetune = clip_finetune
        student_clip = self.sem_seg_head.predictor.clip_model

        for name, param in student_clip.named_parameters():
            # Freeze the text encoder unconditionally.
            if "visual" not in name:
                param.requires_grad = False
                continue
            # Visual encoder: fine-tune according to clip_finetune strategy.
            if "visual.transformer" in name:
                if clip_finetune == "prompt":
                    param.requires_grad = "prompt" in name
                elif clip_finetune == "attention":
                    if "attn" in name:
                        param.requires_grad = (
                            "q_proj" in name or "v_proj" in name
                        )
                    elif "position" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                elif clip_finetune == "full":
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                # Other visual params (conv1, class_embedding, ln_pre/post,
                # proj, positional_embedding) are frozen.
                param.requires_grad = False

        # ------------------------------------------------------------------
        # Teacher: deep-copy of student weights + NACLIP modifications,
        # fully frozen.  A deep copy gives the teacher its own independent
        # parameter tensors (not shared with the student), which is required
        # because NAClipTeacher freezes all parameters in __init__.
        # ------------------------------------------------------------------
        teacher_clip = copy.deepcopy(student_clip)
        self.teacher = NAClipTeacher(teacher_clip, gaussian_std=gaussian_std)

        # ------------------------------------------------------------------
        # CLIP resolution & intermediate feature hooks (same as base CATSeg)
        # ------------------------------------------------------------------
        self.sliding_window = sliding_window
        self.clip_resolution = (
            (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        )
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.upsample1 = nn.ConvTranspose2d(
            self.proj_dim, 256, kernel_size=2, stride=2
        )
        self.upsample2 = nn.ConvTranspose2d(
            self.proj_dim, 128, kernel_size=4, stride=4
        )
        self.layer_indexes = (
            [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15]
        )
        self.layers: list = []
        for li in self.layer_indexes:
            student_clip.visual.transformer.resblocks[li].register_forward_hook(
                lambda _m, _i, o: self.layers.append(o)
            )

    # ------------------------------------------------------------------
    # Detectron2 config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "distill_lambda": cfg.MODEL.TS.DISTILL_LAMBDA,
            "distill_theta": cfg.MODEL.TS.DISTILL_THETA,
            "gaussian_std": cfg.MODEL.TS.GAUSSIAN_STD,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def device(self):
        return self.pixel_mean.device

    def _get_text_features_norm(self) -> torch.Tensor:
        """
        Return L2-normalised text class embeddings of shape [C, D].

        ``predictor.text_features`` has shape [C, T, D] (C classes, T templates,
        D feature dimension).  Templates are mean-pooled then re-normalised.
        """
        predictor = self.sem_seg_head.predictor
        tf = predictor.text_features if self.training else predictor.text_features_test
        tf = tf.mean(dim=1)  # [C, D]
        tf = F.normalize(tf.float(), dim=-1)
        return tf

    def _compute_distill_loss(
        self, R_S: torch.Tensor, R_T: torch.Tensor
    ) -> torch.Tensor:
        """
        High-confidence distillation loss (L1).

        L_distill = mean_{masked positions} |R^S_{ijc} - R^T_{ijc}|

        Confidence mask: I(max_c R^T_{ij} > theta).

        Args:
            R_S: student response map  [B, N, C]
            R_T: teacher response map  [B, N, C]

        Returns:
            Scalar loss.
        """
        max_teacher = R_T.max(dim=-1).values           # [B, N]
        mask = (max_teacher > self.distill_theta).float().unsqueeze(-1)  # [B, N, 1]
        diff = (R_S - R_T).abs()                       # [B, N, C]
        loss = (mask * diff).sum()
        # n_active = number of active (i,j) positions (mask is [B,N,1] so
        # .sum() counts each position once). Dividing by (n_active * C) gives
        # the mean L1 distance per active element.
        n_active = mask.sum().clamp(min=1.0)
        loss = loss / (n_active * R_S.shape[-1])
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batched_inputs):
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        clip_images = [
            (x - self.clip_pixel_mean) / self.clip_pixel_std for x in images
        ]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        # Reset intermediate-layer buffer.
        self.layers = []

        clip_images_resized = F.interpolate(
            clip_images.tensor,
            size=self.clip_resolution,
            mode="bilinear",
            align_corners=False,
        )

        # ---- Student forward ----
        student_clip = self.sem_seg_head.predictor.clip_model
        clip_features = student_clip.encode_image(clip_images_resized, dense=True)
        image_features = clip_features[:, 1:, :]  # [B, N, D]  (patch tokens)

        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {"res5": res5, "res4": res4, "res3": res3}

        outputs = self.sem_seg_head(clip_features, features)

        if self.training:
            targets = torch.stack(
                [x["sem_seg"].to(self.device) for x in batched_inputs], dim=0
            )
            outputs = F.interpolate(
                outputs,
                size=(targets.shape[-2], targets.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            num_classes = outputs.shape[1]
            valid_mask = targets != self.sem_seg_head.ignore_value

            outputs_perm = outputs.permute(0, 2, 3, 1)
            _targets = torch.zeros(outputs_perm.shape, device=self.device)
            _onehot = F.one_hot(targets[valid_mask], num_classes=num_classes).float()
            _targets[valid_mask] = _onehot

            loss_seg = F.binary_cross_entropy_with_logits(outputs_perm, _targets)
            losses = {"loss_sem_seg": loss_seg}

            # ---- Distillation loss ----
            if self.distill_lambda > 0.0:
                text_feat_norm = self._get_text_features_norm()  # [C, D]

                # Student response map R^S  [B, N, C]
                patch_s = F.normalize(image_features.float(), dim=-1)
                R_S = torch.einsum("bnd,cd->bnc", patch_s, text_feat_norm)

                # Teacher response map R^T  [B, N, C]  (no grad)
                R_T = self.teacher.get_similarity_map(
                    clip_images_resized, text_feat_norm
                )

                loss_distill = self._compute_distill_loss(R_S, R_T)
                losses["loss_distill"] = self.distill_lambda * loss_distill

            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])
            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            return [{"sem_seg": output}]

    # ------------------------------------------------------------------
    # Sliding-window inference (same logic as base CATSeg)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference_sliding_window(
        self, batched_inputs, kernel=384, overlap=0.333, out_res=None
    ):
        if out_res is None:
            out_res = [640, 640]

        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(
            images[0].unsqueeze(0), size=out_res, mode="bilinear", align_corners=False
        ).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(
            images[0].unsqueeze(0),
            size=(kernel, kernel),
            mode="bilinear",
            align_corners=False,
        )
        image = torch.cat((image, global_image), dim=0)

        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(
            clip_images, size=self.clip_resolution, mode="bilinear", align_corners=False
        )

        self.layers = []
        student_clip = self.sem_seg_head.predictor.clip_model
        clip_features = student_clip.encode_image(clip_images, dense=True)

        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(
            rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        )
        res5 = self.upsample2(
            rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        )
        features = {"res5": res5, "res4": res4, "res3": res3}
        outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(
            outputs, size=kernel, mode="bilinear", align_corners=False
        )
        outputs = outputs.sigmoid()

        global_output = outputs[-1:]
        global_output = F.interpolate(
            global_output, size=out_res, mode="bilinear", align_corners=False
        )
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(
            unfold(torch.ones([1] + out_res, device=self.device))
        )
        outputs = (outputs + global_output) / 2.0

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{"sem_seg": output}]
