"""
mfssm_net.py
============
Injects MF-SSM blocks into the deepest encoder stages of an nnU-Net.

Placement. dynamic_network_architectures' PlainConvEncoder.forward is

    for s in self.stages:
        x = s(x); ret.append(x)

so a forward hook on stages[-k] that returns a tensor replaces both the running
x and the corresponding skip. The deepest stages are where a shared latent
state is meaningful: for Dataset104's 3d_fullres plan the last two stages carry
560 and 140 tokens at 320 channels, short enough that the scan is cheap and
coarse enough that "state" means global vessel-tree context rather than local
texture.

The U-Net encoder/decoder is left otherwise intact. That is deliberate: the
claim under test is about the factorisation (shared A and C vs free B and dt),
not about replacing convolutions with scans, so the backbone is held fixed as
the control.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mfssm import MFSSMBlock3D


class MFSSMNet(nn.Module):
    def __init__(self, backbone: nn.Module, stage_channels, n_stages_to_wrap: int = 2,
                 d_state: int = 16, expand: int = 2, n_modalities: int = 2,
                 share_A: bool = True, share_C: bool = True,
                 force_ref_scan: bool = False):
        super().__init__()
        self.backbone = backbone
        self.n_wrap = n_stages_to_wrap
        self._mod_ids: torch.Tensor | None = None

        stages = self.backbone.encoder.stages
        self.wrapped_idx = list(range(len(stages) - n_stages_to_wrap, len(stages)))
        self.blocks = nn.ModuleList([
            MFSSMBlock3D(channels=stage_channels[i], d_state=d_state, expand=expand,
                         n_modalities=n_modalities, share_A=share_A, share_C=share_C,
                         force_ref_scan=force_ref_scan)
            for i in self.wrapped_idx])

        self._hooks = []
        for slot, i in enumerate(self.wrapped_idx):
            self._hooks.append(stages[i].register_forward_hook(self._make_hook(slot)))

    def _make_hook(self, slot: int):
        def hook(module, inputs, output):
            mid = self._mod_ids
            if mid is None:
                # Sliding-window inference does not pass batch keys, so the
                # modality must come from the environment. It cannot be
                # inferred from the image: on Dataset104 the most promising
                # statistic (intensity skewness after z-scoring) leaves 35/50
                # cases ambiguous (CT range [0.46, 2.80], MR range [1.00,
                # 51.11]).
                #
                # Requiring it is not the limitation it would be for a
                # domain-conditioned method: modality is recorded in the DICOM
                # header and is always known at acquisition, unlike site or
                # scanner domain.
                m = int(os.environ.get("MFSSM_MODALITY", "0"))
                mid = torch.full((output.shape[0],), m, dtype=torch.long,
                                 device=output.device)
            return self.blocks[slot](output, mid)
        return hook

    def set_modality(self, mod_ids: torch.Tensor | None):
        self._mod_ids = mod_ids

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    @property
    def decoder(self):
        return self.backbone.decoder

    @property
    def encoder(self):
        return self.backbone.encoder

    def __del__(self):
        for h in getattr(self, "_hooks", []):
            try: h.remove()
            except Exception: pass
