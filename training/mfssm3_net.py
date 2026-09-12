"""
mfssm3_net.py
=============
Assembles M3-CoW: an nnU-Net backbone whose deepest encoder stages carry
Mamba-3 blocks, with a state-keyed anatomy transformer on top.

    input -> nnU-Net encoder
                stage 4 (stride 16, 7x10x8 = 560 tokens, 320 ch)  <- M3 block
                stage 5 (stride 32/16, 7x5x4 = 140 tokens, 320 ch) <- M3 block
          -> nnU-Net decoder ------------------> seg logits
                                                     +
             state (560 tokens) -> keys -> 13 vessel queries -> presence
                                                              + spatial prior

The backbone is held fixed as the control, exactly as in the original MF-SSM
study: the claim under test is about the state and its supervision, not about
replacing convolutions with scans. Both new paths (the SSM residual gamma and
the transformer fusion weights) are zero-initialised, so an untrained M3-CoW
is bit-for-bit a plain nnU-Net and every point of difference has to be earned.

The keys come from the SHALLOWER of the two wrapped stages (560 tokens, 8 mm)
rather than the deepest (140 tokens, 16 mm). A 1.5 mm Pcom is already
sub-token at 8 mm; at 16 mm the entire communicating complex collapses into
two or three tokens and the attention has nothing to localise.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mamba3_ssm import M3SSMBlock3D
from anatomy_transformer import AnatomyTransformer
from state_losses import N_CLASS


class MFSSM3Net(nn.Module):
    def __init__(self, backbone: nn.Module, stage_channels, n_stages_to_wrap: int = 2,
                 d_state: int = 8, expand: int = 2, n_modalities: int = 2,
                 n_groups: int = 1, share_A: bool = True, share_C: bool = True,
                 trapezoidal: bool = True, grad_checkpoint: bool = False,
                 d_bottleneck: Optional[int] = 128,
                 tr_dim: int = 256, tr_layers: int = 3, key_slot: int = 0,
                 use_transformer: bool = True, theta_scale: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.n_wrap = n_stages_to_wrap
        self.key_slot = key_slot
        self.use_transformer = use_transformer
        self._mod_ids: Optional[torch.Tensor] = None
        self._grids: Dict[int, tuple] = {}

        stages = self.backbone.encoder.stages
        self.wrapped_idx = list(range(len(stages) - n_stages_to_wrap, len(stages)))
        self.blocks = nn.ModuleList([
            M3SSMBlock3D(channels=stage_channels[i], d_state=d_state, expand=expand,
                         n_modalities=n_modalities, n_groups=n_groups,
                         share_A=share_A, share_C=share_C, trapezoidal=trapezoidal,
                         grad_checkpoint=grad_checkpoint, d_bottleneck=d_bottleneck,
                         theta_scale=theta_scale)
            for i in self.wrapped_idx])

        # The state lives at the block's bottleneck width, not the encoder
        # stage's width, so the transformer and probe must read that.
        key_ch = self.blocks[key_slot].d_ssm
        self.state_dim = key_ch
        self.anatomy = (AnatomyTransformer(key_ch, dim=tr_dim, n_layers=tr_layers,
                                           n_class=N_CLASS)
                        if use_transformer else None)
        # Linear probe on the state -- see state_losses.probe_loss. Training-time
        # only; it never touches the forward path, so it costs nothing at
        # inference and cannot leak into the prediction.
        self.state_probe = nn.Linear(key_ch, N_CLASS)

        self._hooks = []
        for slot, i in enumerate(self.wrapped_idx):
            self._hooks.append(stages[i].register_forward_hook(self._make_hook(slot)))

    def _make_hook(self, slot: int):
        def hook(module, inputs, output):
            mid = self._mod_ids
            if mid is None:
                # Sliding-window inference does not pass batch keys, so the
                # modality comes from the environment. It is not inferable from
                # the image: on Dataset104 the most promising statistic
                # (post-z-score skewness) leaves 35/50 cases ambiguous. Modality
                # is in the DICOM header and always known at acquisition, so
                # requiring it is not the limitation it would be for a
                # site/scanner domain indicator.
                m = int(os.environ.get("MFSSM_MODALITY", "0"))
                mid = torch.full((output.shape[0],), m, dtype=torch.long,
                                 device=output.device)
            self._grids[slot] = tuple(output.shape[2:])
            return self.blocks[slot](output, mid)
        return hook

    def set_modality(self, mod_ids: Optional[torch.Tensor]):
        self._mod_ids = mod_ids

    def forward(self, x: torch.Tensor):
        seg = self.backbone(x)
        state = self.blocks[self.key_slot].last_state
        grid = self._grids[self.key_slot]

        tr = None
        if self.anatomy is not None:
            tr = self.anatomy(state, grid)
            if isinstance(seg, (list, tuple)):
                # Fuse into the full-resolution head only. The coarse deep-
                # supervision heads keep reading unmodified backbone features,
                # so the anatomy path cannot corrupt a signal it is not
                # responsible for.
                seg = list(seg)
                seg[0] = self.anatomy.fuse(seg[0], tr, grid)
            else:
                seg = self.anatomy.fuse(seg, tr, grid)

        if not self.training:
            return seg
        return {"seg": seg, "state": state, "grid": grid, "tr": tr,
                "dyn": [b.last_dyn for b in self.blocks if b.last_dyn]}

    @property
    def decoder(self):
        return self.backbone.decoder

    @property
    def encoder(self):
        return self.backbone.encoder

    def __del__(self):
        for h in getattr(self, "_hooks", []):
            try:
                h.remove()
            except Exception:
                pass
