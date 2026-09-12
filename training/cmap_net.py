"""
cmap_net.py
===========
Wraps an nnU-Net backbone so that the cross-modal anatomical prior is fused
into the finest decoder feature map.

Where the fusion happens
------------------------
dynamic_network_architectures' UNetDecoder.forward runs

    for s in range(len(self.stages)):
        x = self.transpconvs[s](lres_input)
        x = torch.cat((x, skips[-(s+2)]), 1)
        x = self.stages[s](x)
        if self.deep_supervision:
            seg_outputs.append(self.seg_layers[s](x))
        ...
        lres_input = x

so a forward hook on `decoder.stages[-1]` that *returns* a tensor replaces the
x that feeds `seg_layers[-1]` -- the full-resolution segmentation head.  That
is exactly the tap point we want, and it leaves the coarser deep-supervision
outputs reading unmodified backbone features, so the prior cannot corrupt the
deep-supervision signal it is not responsible for.

The wrapper returns a plain tensor/list in eval mode so nnU-Net's sliding
window predictor works unchanged.  Note that the fusion itself stays active at
inference -- it is part of the model.  Only the *reliability matrix* is
training-time-only, which is what keeps the deployed model modality-agnostic:
it needs the image, never a modality label.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cmap import CrossModalPriorModule


class CMAPNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_channels: int,
        num_classes: int,
        embed_dim: int = 16,
        curvature: float = 1.0,
        tau: float = 0.5,
        prior_stride: int = 2,
        gate_hidden: int = 32,
        num_modalities: int = 2,
        use_hyperbolic: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.cmap = CrossModalPriorModule(
            feat_channels=feat_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            curvature=curvature,
            tau=tau,
            prior_stride=prior_stride,
            gate_hidden=gate_hidden,
            num_modalities=num_modalities,
            use_hyperbolic=use_hyperbolic,
        )
        # Toggled by the trainer for the delayed-fusion schedule.
        self.fuse_active: bool = True
        self._aux = None

        target = self._last_decoder_stage()
        if target is None:
            raise RuntimeError(
                "CMAPNet: could not locate backbone.decoder.stages[-1]; "
                "the fusion hook cannot be attached."
            )
        self._hook = target.register_forward_hook(self._fuse_hook)

    def _last_decoder_stage(self):
        try:
            return self.backbone.decoder.stages[-1]
        except (AttributeError, IndexError):
            return None

    def _fuse_hook(self, module, inputs, output):
        fused, aux = self.cmap(output, fuse_active=self.fuse_active)
        self._aux = aux
        return fused

    def forward(self, x: torch.Tensor):
        self._aux = None
        seg = self.backbone(x)
        if not self.training:
            # Sliding-window inference expects a tensor or list of tensors.
            return seg
        return {"seg": seg, "cmap": self._aux}

    # nnUNetTrainer reaches into these directly (deep supervision toggling,
    # optimiser construction).
    @property
    def decoder(self):
        return self.backbone.decoder

    @property
    def encoder(self):
        return self.backbone.encoder

    def __del__(self):
        h = getattr(self, "_hook", None)
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
