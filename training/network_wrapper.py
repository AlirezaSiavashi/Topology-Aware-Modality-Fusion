"""
network_wrapper.py
==================
Wraps a standard nnU-Net PlainConvUNet with two auxiliary heads:

  1. SDF head      — regresses signed distance field  (1 channel, MSE)
  2. Skeleton head — predicts binary centerline mask  (1 channel, BCE+Dice)

The wrapper taps into the final decoder feature map (full resolution,
before the segmentation conv) and attaches lightweight 1×1×1 conv heads.

Architecture
------------
                      ┌─────────────────────────────┐
  image ──► encoder ──► decoder ──► seg_head  ──► K-class logits
                                └─► feat_map
                                       ├─► sdf_head  ──► 1-ch SDF
                                       └─► skel_head ──► 1-ch skeleton prob

The nnU-Net PlainConvUNet stores the final decoder output in a hooked
variable; we splice in after the last decoder block.

Forward output (training mode)
-------------------------------
  dict with keys:
    'seg'   : list[Tensor] or Tensor  — deep-supervised segmentation logits
    'sdf'   : Tensor (B,1,D,H,W)     — predicted SDF at full resolution
    'skel'  : Tensor (B,1,D,H,W)     — predicted skeleton logits

Forward output (eval mode)
--------------------------
  dict with same keys; 'seg' is a single Tensor (no DS list).
"""

import torch
import torch.nn as nn
from typing import List, Union


class SDFHead(nn.Module):
    """Lightweight 3-layer head for SDF regression."""

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True),
            nn.Tanh(),   # output in (-1, 1); matches normalised SDF targets
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkeletonHead(nn.Module):
    """Lightweight 3-layer head for centerline/skeleton probability."""

    def __init__(self, in_channels: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True),
            # No sigmoid here — BCEWithLogitsLoss is more numerically stable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopologyVesselNet(nn.Module):
    """
    Wraps a nnU-Net backbone with SDF + skeleton auxiliary heads.

    Parameters
    ----------
    backbone      : PlainConvUNet (or any nnU-Net arch) with deep supervision.
    backbone_feat_channels : number of feature channels in the final decoder
                   layer (= features_per_stage[0] for PlainConvUNet, e.g. 32).
    enable_aux_heads : set False to disable aux heads (e.g., for inference-only).
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_feat_channels: int = 32,
        enable_aux_heads: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.enable_aux_heads = enable_aux_heads

        if enable_aux_heads:
            self.sdf_head  = SDFHead(backbone_feat_channels, hidden=32)
            self.skel_head = SkeletonHead(backbone_feat_channels, hidden=16)

        # Hook storage
        self._feat_map: torch.Tensor | None = None
        self._hook_handle = None
        if enable_aux_heads:
            self._register_feat_hook()

    # ── Hook: intercept decoder's final feature map ───────────────────────
    def _register_feat_hook(self):
        """
        nnU-Net's PlainConvUNet has a `decoder` attribute whose last stage
        outputs the full-resolution feature map before the final seg conv.

        We hook the last `ConvDropoutNormReLU` block in the decoder to
        capture the pre-logit features.
        """
        # Walk to the last decoder block
        target = self._find_last_decoder_block()
        if target is not None:
            self._hook_handle = target.register_forward_hook(self._feat_hook)
        else:
            # Fallback: hook the whole backbone and use the first output element
            self._hook_handle = self.backbone.register_forward_hook(self._backbone_hook)

    def _find_last_decoder_block(self):
        """Find the last conv block in the decoder."""
        # PlainConvUNet: backbone.decoder.stages[-1] (list of conv blocks)
        try:
            stages = self.backbone.decoder.stages
            return stages[-1]
        except AttributeError:
            pass
        # UNetDecoder variant
        try:
            stages = self.backbone.decoder.seg_layers
            return list(self.backbone.decoder.stages.children())[-1]
        except AttributeError:
            pass
        return None

    def _feat_hook(self, module, input, output):
        # output is the feature map from the last decoder block
        if isinstance(output, (list, tuple)):
            self._feat_map = output[0]
        else:
            self._feat_map = output

    def _backbone_hook(self, module, input, output):
        # Fallback: extract from backbone output
        # output is a list (DS) or tensor; capture input to seg_layers[-1]
        self._feat_map = None   # Will be None — aux heads skipped

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        self._feat_map = None
        seg_out = self.backbone(x)   # list[Tensor] (DS) or Tensor

        # Inference mode: nnUNet's predictor expects a plain tensor/list,
        # not a dict. Return seg_out directly so sliding-window inference works.
        if not self.training:
            return seg_out

        result = {'seg': seg_out}

        if self.enable_aux_heads and self._feat_map is not None:
            result['sdf']  = self.sdf_head(self._feat_map)
            result['skel'] = self.skel_head(self._feat_map)
        else:
            # Aux heads disabled or hook missed — return zeros as placeholder
            # so the trainer can always access these keys
            ref = seg_out[0] if isinstance(seg_out, list) else seg_out
            result['sdf']  = torch.zeros(
                ref.shape[0], 1, *ref.shape[2:], device=x.device, dtype=x.dtype)
            result['skel'] = torch.zeros_like(result['sdf'])

        return result

    # ── Delegate backbone attributes expected by nnUNet trainer ──────────
    # nnUNetTrainer.set_deep_supervision_enabled accesses self.network.decoder
    # and self.network.encoder directly.
    @property
    def decoder(self):
        return self.backbone.decoder

    @property
    def encoder(self):
        return self.backbone.encoder

    def __del__(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
