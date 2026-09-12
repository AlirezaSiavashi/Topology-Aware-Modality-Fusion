"""
tcmn.py  —  Topology-Conditioned Modality Normalization
========================================================

Core idea
---------
nnU-Net's decoder has 5 stages operating at different spatial resolutions:
  Stage 0 (finest,  full-res)  → processes small distal vessel features
  Stage 4 (coarsest, 1/16-res) → processes large proximal vessel features

In CTA: large proximal vessels dominate (contrast agent fills them).
In MRA (TOF): small fast-flow distal vessels are reliable; large slow-flow
vessels may drop out.

TCMN replaces the affine transform in each InstanceNorm3d of the DECODER
with a FiLM (Feature-wise Linear Modulation) transform conditioned on:
  - modality_id  : 0 = CTA, 1 = MRA   (from subject key, e.g. "topcow_ct_001")
  - depth_id     : 0..4 = finest..coarsest decoder stage

This gives the network independent γ/β per (modality × depth), so it can
amplify MRA small-vessel features at fine stages and CTA large-vessel
features at coarse stages — without any paired images, without any GAN,
and with modality label always available from DICOM metadata.

Parameters per TCMNLayer: 2 * embed_dim * 2 * num_features / num_features ≈ tiny.
Total added parameters: ~6 K (negligible vs 30M backbone).

Usage
-----
From the trainer, before each forward pass:

    with tcmn.modality_context(modality_ids):   # (B,) long tensor, 0=CTA 1=MRA
        output = network(data)

The TCMNLayer instances read the context via a module-level variable.
This avoids changing the call signature of every nn.Module in the decoder.

Injection
---------
    tcmn.inject_tcmn_into_decoder(network, num_decoder_stages=5, embed_dim=16)

This walks the backbone's decoder and replaces each InstanceNorm3d
(affine=True) with a TCMNLayer. The original affine weights are copied
into the FiLM MLP bias so training starts from the pretrained InstanceNorm
identity (γ=1, β=0 for the mean modality/depth).
"""

from __future__ import annotations

import contextlib
import threading
from typing import Optional

import torch
import torch.nn as nn

# ── Thread-local modality context ─────────────────────────────────────────────
# Stores the current batch's modality_ids tensor.
# Using thread-local so multi-worker data loading doesn't interfere.
_local = threading.local()


def _get_context() -> Optional[torch.Tensor]:
    return getattr(_local, 'modality_ids', None)


@contextlib.contextmanager
def modality_context(modality_ids: torch.Tensor):
    """
    Context manager that sets the current modality IDs for a forward pass.

    Parameters
    ----------
    modality_ids : (B,) long tensor — 0 = CTA, 1 = MRA
    """
    _local.modality_ids = modality_ids
    try:
        yield
    finally:
        _local.modality_ids = None


# ── TCMNLayer ──────────────────────────────────────────────────────────────────

class TCMNLayer(nn.Module):
    """
    Drop-in replacement for InstanceNorm3d(num_features, affine=True).

    When modality context is set: applies FiLM conditioned on
    (modality_id, depth_id).
    When modality context is absent (e.g. inference without modality info):
    falls back to standard InstanceNorm3d with learned affine.
    """

    NUM_MODALITIES = 2   # 0=CTA, 1=MRA

    def __init__(
        self,
        num_features: int,
        depth_id: int,
        num_depths: int = 5,
        embed_dim: int = 16,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.depth_id     = depth_id
        self.eps          = eps

        # Fallback standard InstanceNorm (used when no modality context)
        self.fallback_norm = nn.InstanceNorm3d(num_features, affine=True, eps=eps)

        # Embeddings
        self.modality_embed = nn.Embedding(self.NUM_MODALITIES, embed_dim)
        self.depth_embed    = nn.Embedding(num_depths, embed_dim)

        # FiLM generator: concat embeddings → 2 * num_features (γ and β)
        self.film_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, num_features * 2),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Residual FiLM init: network outputs Δγ and Δβ (deviations from identity).
        At init Δγ≈0, Δβ≈0 → behaves exactly like InstanceNorm with learned affine.
        Gradients flow freely from the first step because the final layer bias is 0
        (no saturation) and embeddings are non-zero.

        Strategy:
        - Embeddings: standard normal init — strong gradients from step 1
        - Hidden layer: standard Xavier + zero bias — clean start
        - Final layer: near-zero weights, zero bias → Δγ≈0, Δβ≈0 at init
        - Forward: uses fallback_norm affine (γ_base, β_base) + Δγ, Δβ residual
        """
        # Hidden layer — standard init
        nn.init.xavier_uniform_(self.film_mlp[0].weight)
        nn.init.zeros_(self.film_mlp[0].bias)

        # Final layer: near-zero weights so residual starts small
        nn.init.xavier_uniform_(self.film_mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # Embeddings: standard normal for healthy gradient flow
        nn.init.normal_(self.modality_embed.weight, 0.0, 1.0)
        nn.init.normal_(self.depth_embed.weight, 0.0, 1.0)

    def _film_forward(self, x_norm: torch.Tensor, mod_ids: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning given normalised features and modality ids."""
        B, C = x_norm.shape[:2]
        dep_ids = torch.full(
            (B,), self.depth_id, dtype=torch.long, device=x_norm.device)

        mod_emb = self.modality_embed(mod_ids)               # (B, embed_dim)
        dep_emb = self.depth_embed(dep_ids)                  # (B, embed_dim)

        film_in     = torch.cat([mod_emb, dep_emb], dim=-1)  # (B, 2*embed_dim)
        film_params = self.film_mlp(film_in)                  # (B, 2*C)

        delta_γ = film_params[:, :C].view(B, C, 1, 1, 1)
        delta_β = film_params[:, C:].view(B, C, 1, 1, 1)

        γ_base = self.fallback_norm.weight.view(1, C, 1, 1, 1)
        β_base = self.fallback_norm.bias.view(1, C, 1, 1, 1)

        return (γ_base + delta_γ) * x_norm + (β_base + delta_β)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, D, H, W)

        Residual FiLM: output = (γ_base + Δγ) * x_norm + (β_base + Δβ)
        where γ_base and β_base come from fallback_norm's learned affine weights,
        and Δγ, Δβ are the modality-conditioned residuals from the FiLM MLP.

        When no modality context is set (e.g. nnU-Net sliding-window inference),
        averages the FiLM output over all modalities — modality-agnostic but
        still uses the trained FiLM weights rather than the untrained fallback_norm.
        """
        modality_ids = _get_context()

        # ── Instance normalization (no affine, shared across both paths) ──
        B, C = x.shape[:2]
        x_flat = x.view(B, C, -1)
        mean   = x_flat.mean(dim=2, keepdim=True)
        var    = x_flat.var(dim=2, keepdim=True, unbiased=False)
        x_norm = ((x_flat - mean) / (var + self.eps).sqrt()).view_as(x)

        if modality_ids is None:
            # No modality context (inference without explicit modality label).
            # Average FiLM outputs over all modalities to remain modality-agnostic.
            out = torch.zeros_like(x_norm)
            for mid in range(self.NUM_MODALITIES):
                ids = torch.full((B,), mid, dtype=torch.long, device=x.device)
                out = out + self._film_forward(x_norm, ids)
            return out / self.NUM_MODALITIES

        # ── Normal training / validation_step path ────────────────────────
        return self._film_forward(x_norm, modality_ids.to(x.device))


# ── Injection utility ──────────────────────────────────────────────────────────

def inject_tcmn_into_decoder(
    network: nn.Module,
    num_depths: int = 5,
    embed_dim: int = 16,
) -> int:
    """
    Walk the backbone decoder and replace each InstanceNorm3d (affine=True)
    with a TCMNLayer.  Assigns depth_id based on which decoder stage the
    norm belongs to (stage 0 = finest resolution).

    Parameters
    ----------
    network   : TopologyVesselNet (or PlainConvUNet directly)
    num_depths: number of decoder stages (5 for nnUNet 3d_fullres default)
    embed_dim : embedding dimension for FiLM MLP

    Returns
    -------
    Number of InstanceNorm3d layers replaced.
    """
    # Reach the backbone's decoder
    backbone = getattr(network, 'backbone', network)
    try:
        decoder = backbone.decoder
    except AttributeError:
        raise RuntimeError(
            "Cannot find decoder in network. Expected network.backbone.decoder "
            "or network.decoder."
        )

    try:
        stages = decoder.stages   # nn.ModuleList of StackedConvBlocks
    except AttributeError:
        raise RuntimeError(
            "decoder.stages not found. Is this a PlainConvUNet decoder?"
        )

    replaced = 0
    for stage_idx, stage in enumerate(stages):
        depth_id = stage_idx   # 0 = finest, num_depths-1 = coarsest
        depth_id = min(depth_id, num_depths - 1)

        # Walk all submodules in this stage, replace InstanceNorm3d
        for name, module in list(stage.named_modules()):
            # Skip InstanceNorm3d that lives inside a TCMNLayer (already injected)
            if isinstance(module, TCMNLayer):
                continue
            if isinstance(module, nn.InstanceNorm3d) and module.affine:
                # Guard: don't replace if the parent is already a TCMNLayer
                parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                if parent_name:
                    parent_mod = stage
                    for p in parent_name.split('.'):
                        parent_mod = getattr(parent_mod, p)
                    if isinstance(parent_mod, TCMNLayer):
                        continue
                num_features = module.num_features
                eps          = module.eps

                tcmn = TCMNLayer(
                    num_features=num_features,
                    depth_id=depth_id,
                    num_depths=num_depths,
                    embed_dim=embed_dim,
                    eps=eps,
                )

                # Copy existing affine weights into fallback_norm
                # so the TCMN fallback path preserves pretrained stats
                with torch.no_grad():
                    if module.weight is not None:
                        tcmn.fallback_norm.weight.copy_(module.weight)
                    if module.bias is not None:
                        tcmn.fallback_norm.bias.copy_(module.bias)

                # Replace the module in its parent
                _replace_module(stage, name, tcmn)
                replaced += 1

                # Also replace the same InstanceNorm3d in all_modules[1] if it
                # exists. nnU-Net's ConvDropoutNormReLU stores [conv, norm, nonlin]
                # in all_modules (Sequential) AND as .conv/.norm/.nonlin attributes.
                # Replacing .norm doesn't update all_modules[1], leading to a stale
                # InstanceNorm3d that gets double-injected on the next call.
                if name.endswith('.norm'):
                    try:
                        # Navigate to the ConvDropoutNormReLU block
                        parent_path = name.rsplit('.norm', 1)[0]
                        block = stage
                        for p in parent_path.split('.'):
                            block = getattr(block, p)
                        if hasattr(block, 'all_modules') and len(block.all_modules) > 1:
                            if isinstance(block.all_modules[1], nn.InstanceNorm3d):
                                block.all_modules[1] = tcmn
                    except (AttributeError, IndexError):
                        pass  # different block structure — skip

    return replaced


def _replace_module(parent: nn.Module, dotted_name: str, new_module: nn.Module):
    """Replace a submodule by its dotted name path within parent."""
    parts = dotted_name.split('.')
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)
