"""
nnUNetTrainerTCMN.py
====================
Trainer that extends nnUNetTrainerTopologyVessel with Topology-Conditioned
Modality Normalization (TCMN).

What changes vs the parent trainer
------------------------------------
1. After building the network, injects TCMNLayer into all decoder
   InstanceNorm3d layers (one call to tcmn.inject_tcmn_into_decoder).

2. Before every forward pass (train + validation), extracts modality IDs
   from the batch subject keys ("topcow_ct_*" → 0, "topcow_mr_*" → 1) and
   sets the modality context so TCMNLayer instances can read it.

3. Everything else (cbDice, SDF head, skeleton head, checkpoint saving) is
   inherited unchanged from nnUNetTrainerTopologyVessel.

Dataset compatibility
---------------------
- DS102 (TopCoW CT only): all subjects are CTA → modality_id=0 for all.
  TCMN degenerates to per-depth normalization only. Still useful as it
  learns depth-specific affine transforms.

- DS103 (TopCoW MR only): all MRA → modality_id=1. Same as above.

- DS104 (joint CT+MR, 250 subjects): mixed batches. TCMN is fully active.
  This is the primary use case.

Training script
---------------
  nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerTCMN
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnUNetTrainerTopologyVessel import nnUNetTrainerTopologyVessel
import tcmn


# ── Modality ID extraction ────────────────────────────────────────────────────

def _modality_ids_from_keys(keys: List[str], device: torch.device) -> torch.Tensor:
    """
    Extract modality IDs from nnU-Net batch keys.

    Convention:
      key contains "_ct_"  → CTA → 0
      key contains "_mr_"  → MRA → 1
      key contains "_mra_" → MRA → 1
      unknown              → 0   (fallback to CTA)

    Parameters
    ----------
    keys   : list of subject keys (e.g. ['topcow_ct_001', 'topcow_mr_005'])
    device : torch device

    Returns
    -------
    (B,) long tensor of modality IDs
    """
    ids = []
    for key in keys:
        key_lower = key.lower()
        if '_mr_' in key_lower or '_mra_' in key_lower or key_lower.endswith('_mr'):
            ids.append(1)
        else:
            ids.append(0)   # Default: CTA
    return torch.tensor(ids, dtype=torch.long, device=device)


# ── Trainer ────────────────────────────────────────────────────────────────────

class nnUNetTrainerTCMN(nnUNetTrainerTopologyVessel):
    """
    Topology-Conditioned Modality Normalization trainer.

    Inherits all topology losses (cbDice, SDF, skeleton) from the parent
    and adds TCMN injection into the decoder normalization layers.
    """

    # TCMN hyperparameters — easy to override in subclasses
    TCMN_EMBED_DIM:  int = 16
    TCMN_NUM_DEPTHS: int = 5   # = number of decoder stages in 3d_fullres

    # Extended schedule: TCMN needs more epochs to converge due to FiLM warmup
    num_epochs: int = 2000

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 2000
        self._tcmn_injected: bool = False
        self._tcmn_replaced_count: int = 0

    # ── build_network_architecture: inject TCMN so inference works too ───
    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        # Build via parent (TopologyVesselNet wrapper)
        network = nnUNetTrainerTopologyVessel.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        # Inject TCMN immediately so state_dict keys match checkpoint
        tcmn.inject_tcmn_into_decoder(
            network=network,
            num_depths=5,
            embed_dim=16,
        )
        return network

    # ── Network building: inject TCMN after backbone is built ────────────
    def initialize(self):
        """
        Called by nnUNetTrainer before training starts.
        We override to inject TCMN into the decoder after the network is built.
        """
        super().initialize()
        self._inject_tcmn()

    def _inject_tcmn(self):
        if self._tcmn_injected:
            return

        n = tcmn.inject_tcmn_into_decoder(
            network=self.network,
            num_depths=self.TCMN_NUM_DEPTHS,
            embed_dim=self.TCMN_EMBED_DIM,
        )
        self._tcmn_replaced_count = n
        self._tcmn_injected = True
        self.print_to_log_file(
            f"[TCMN] Injected TCMNLayer into {n} decoder InstanceNorm3d layers "
            f"(embed_dim={self.TCMN_EMBED_DIM}, num_depths={self.TCMN_NUM_DEPTHS})"
        )

    # ── Checkpoint loading: strict=False for TCMN key compatibility ──────
    def load_checkpoint(self, filename_or_checkpoint):
        """
        Override to use strict=False — the TCMN injection may create keys
        (fallback_norm, modality_embed, etc.) that are absent in older
        checkpoints, or the double-injection guard may leave minor mismatches.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint,
                                    map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        new_state_dict = {}
        model_keys = set(self.network.state_dict().keys())
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in model_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        missing, unexpected = self.network.load_state_dict(new_state_dict, strict=False)
        if missing:
            self.print_to_log_file(
                f"[TCMN] load_checkpoint: {len(missing)} missing keys (random init). "
                f"First 3: {missing[:3]}"
            )
        if unexpected:
            self.print_to_log_file(
                f"[TCMN] load_checkpoint: {len(unexpected)} unexpected keys (ignored)."
            )

        self.current_epoch = checkpoint['current_epoch']
        self._best_ema = checkpoint.get('_best_ema', 0.0)
        self.inference_allowed_mirroring_axes = checkpoint.get(
            'inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)

        if checkpoint.get('logging') is not None:
            try:
                self.logger.load_checkpoint(checkpoint['logging'])
            except Exception:
                pass

        if checkpoint.get('optimizer_state') is not None:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            except Exception:
                self.print_to_log_file(
                    "[TCMN] Could not restore optimizer state — starting fresh."
                )

        if self.grad_scaler is not None and checkpoint.get('grad_scaler_state') is not None:
            try:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
            except Exception:
                pass

    # ── Train step: set modality context for forward pass ────────────────
    def train_step(self, batch: dict) -> dict:
        keys = batch.get('keys', [])
        modality_ids = _modality_ids_from_keys(keys, self.device)

        with tcmn.modality_context(modality_ids):
            result = super().train_step(batch)

        return result

    # ── Validation step: set modality context ────────────────────────────
    def validation_step(self, batch: dict) -> dict:
        keys = batch.get('keys', [])
        modality_ids = _modality_ids_from_keys(keys, self.device)

        with tcmn.modality_context(modality_ids):
            result = super().validation_step(batch)

        return result


# ── Ablation: TCMN without topology losses (pure domain norm study) ────────────

class nnUNetTrainerTCMN_NoTopo(nnUNetTrainerTCMN):
    """
    TCMN domain normalization only — no cbDice, no SDF, no skeleton.
    Used for ablation: measures domain norm contribution independently
    of topology loss contribution.
    """
    λ_topo: float = 0.0
    λ_sdf:  float = 0.0
    λ_skel: float = 0.0
