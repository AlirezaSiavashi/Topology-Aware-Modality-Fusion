"""
nnUNetTrainerCMTC.py
====================
Trainer extending nnUNetTrainerTCMN with the Cross-Modal Topology Consistency
(CMTC) loss.

What changes vs nnUNetTrainerTCMN
-----------------------------------
1. Adds ``CmtcLoss`` as a training objective.
   After the standard TCMN losses (CE+Dice, cbDice, SDF, skeleton) are
   computed, we compute:

       l_cmtc = CmtcLoss(seg_output, modality_ids)

   and add ``λ_cmtc * l_cmtc`` to the total loss.

2. ``λ_cmtc`` defaults to 0.1 and can be overridden in subclasses.

3. ``l_cmtc`` is logged in the returned training-step dict so it appears in
   nnU-Net's progress.png.

Everything else (TCMN injection, modality context, cbDice, SDF head, skeleton
head, deep supervision, checkpoint saving) is inherited unchanged from the
parent trainers.

Inheritance chain
-----------------
  nnUNetTrainer
    └─ nnUNetTrainerTopologyVessel   (cbDice + SDF + skeleton)
         └─ nnUNetTrainerTCMN        (FiLM modality norm in decoder)
              └─ nnUNetTrainerCMTC   (+ CMTC cross-modal topology loss)

Dataset compatibility
---------------------
- DS104 (joint CT+MR, 250 subjects): primary use-case. Each mini-batch contains
  both CTA and MRA cases, so CMTC is fully active.
- DS102/DS103 (single-modality): CMTC returns 0 (only one modality present in
  batch), so training reduces to nnUNetTrainerTCMN.

Training
--------
  nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerCMTC
"""

from __future__ import annotations

import sys as _sys

# ── Make our training/ directory importable when running from the nnU-Net ──────
# installed copy in site-packages/nnunetv2/... (see the stub file).
_TRAINING_DIR = (
    "/scratch/siyavash/Alireza_thesis/external_dataset"
    "/rsna-intracranial-aneurysm-detection"
    "/brain_external_dataset/training"
)
if _TRAINING_DIR not in _sys.path:
    _sys.path.insert(0, _TRAINING_DIR)

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from nnUNetTrainerTCMN import nnUNetTrainerTCMN, _modality_ids_from_keys
import tcmn

from loss_cmtc import CmtcLoss


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerCMTC(nnUNetTrainerTCMN):
    """
    TCMN trainer augmented with Cross-Modal Topology Consistency (CMTC) loss.

    Inherits:
      - cbDice topology loss (λ_topo = 0.5)
      - SDF regression head  (λ_sdf  = 0.2)
      - Skeleton head        (λ_skel = 0.1)
      - TCMN modality normalization

    Adds:
      - CMTC loss            (λ_cmtc = 0.1, class variable — override in subclass)
    """

    # ── CMTC hyperparameters ─────────────────────────────────────────────────
    # Override in a subclass to run ablations without changing the source file.
    lambda_cmtc:          float = 0.05   # small weight — CMTC supplements, not dominates
    cmtc_vessel_classes:  int   = 13
    cmtc_lambda_b1:       float = 0.5
    cmtc_target_size:     int   = 32
    cmtc_sw_directions:   int   = 5
    # Persistent homology is expensive (gudhi CubicalComplex on 32³ grid).
    # We compute CMTC only every N steps to keep epoch time acceptable.
    # At 250 steps/epoch, every_n=25 means 10 PH calls/epoch.
    cmtc_every_n_steps:   int   = 25
    # Warm-up: delay CMTC until segmentation is learning (epoch ≥ warm_up_epochs).
    # Without warm-up the CMTC JS-divergence gradient can destabilise early
    # training when softmax outputs are still near-uniform.
    cmtc_warmup_epochs:   int   = 50

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        # Force skip unpacking — data already unpacked as .npy, and the
        # multiprocessing.spawn-based unpack segfaults under SLURM.
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=False, device=device)
        self.cmtc_loss = CmtcLoss(
            lambda_b1=self.cmtc_lambda_b1,
            vessel_classes=self.cmtc_vessel_classes,
            target_size=self.cmtc_target_size,
            num_sw_directions=self.cmtc_sw_directions,
        )
        self._cmtc_step_counter: int = 0

    def initialize(self):
        """
        Ensure all TCMN modules (embedding tables, FiLM MLP) are on the
        correct device after TCMN injection in super().initialize().
        nnUNetTrainerTCMN._inject_tcmn() creates fresh TCMNLayer modules
        AFTER the backbone is moved to device; this moves them explicitly.
        """
        super().initialize()
        if self.network is not None:
            self.network.to(self.device)

    def load_checkpoint(self, filename_or_checkpoint):
        """
        Override to use strict=False so that loading a plain nnUNetTrainer
        checkpoint (which lacks TCMN parameters) doesn't crash.
        Missing TCMN parameters are left at their random-init values.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint,
                                    map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        model_keys = set(self.network.state_dict().keys())

        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            # Strip 'module.' prefix (DataParallel wrapping)
            if key not in model_keys and key.startswith('module.'):
                key = key[7:]
            # Remap baseline nnUNetTrainer keys to TCMN-wrapped backbone:
            # baseline uses encoder.*/decoder.*, TCMN wraps as backbone.encoder.*/backbone.decoder.*
            if key not in model_keys and not key.startswith('backbone.'):
                remapped = 'backbone.' + key
                if remapped in model_keys:
                    key = remapped
            # Fix double-TCMN-injection keys from old checkpoints:
            # Old: ...norm.fallback_norm.fallback_norm.{weight,bias}  (InstanceNorm3d inside double-TCMNLayer)
            # New: ...norm.fallback_norm.{weight,bias}                (InstanceNorm3d inside single TCMNLayer)
            if key not in model_keys and '.fallback_norm.fallback_norm.' in key:
                remapped = key.replace('.fallback_norm.fallback_norm.', '.fallback_norm.')
                if remapped in model_keys:
                    key = remapped
            # Old checkpoint also has TCMN params inside fallback_norm (the double layer):
            # ...norm.fallback_norm.film_mlp.*, ...norm.fallback_norm.modality_embed.*, etc.
            # These are extra TCMN modules that don't exist in the fixed model — skip them.
            new_state_dict[key] = value

        # Use strict=False: TCMN params absent in baseline ckpt stay random-init
        missing, unexpected = self.network.load_state_dict(new_state_dict, strict=False)
        if missing:
            self.print_to_log_file(
                f"[CMTC] load_checkpoint strict=False: {len(missing)} missing keys "
                f"(TCMN params, will use random init). First 3: {missing[:3]}"
            )
        if unexpected:
            self.print_to_log_file(
                f"[CMTC] load_checkpoint: {len(unexpected)} unexpected keys (ignored). "
                f"First 3: {list(unexpected)[:3]}"
            )

        self.current_epoch = checkpoint['current_epoch']
        self._best_ema = checkpoint.get('_best_ema', 0.0)
        self.inference_allowed_mirroring_axes = checkpoint.get(
            'inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)

        # Restore logger if available (may be None for cross-trainer warm-start).
        # When logging is None, pad the logger lists so the EMA computation
        # at epoch end doesn't crash with IndexError (it expects N-1 entries).
        if checkpoint.get('logging') is not None:
            try:
                self.logger.load_checkpoint(checkpoint['logging'])
            except Exception:
                pass  # logger format mismatch — start fresh

        # Pad logger lists if they're shorter than current_epoch.
        # The nnU-Net logger accesses [epoch-1] for EMA computation at epoch end;
        # if logging was cleared (None) or from a different trainer, the lists
        # will be empty while current_epoch > 0, causing IndexError.
        n_epochs_done = self.current_epoch
        if n_epochs_done > 0 and hasattr(self, 'logger'):
            import time
            log = self.logger.my_fantastic_logging
            defaults = {
                'mean_fg_dice': 0.0,
                'ema_fg_dice': 1e-3,
                'dice_per_class_or_region': [],
                'train_losses': 1.0,
                'val_losses': 1.0,
                'lrs': 0.01,
                'epoch_start_timestamps': time.time(),
                'epoch_end_timestamps': time.time(),
            }
            for key in log:
                while len(log[key]) < n_epochs_done:
                    log[key].append(defaults.get(key, 0.0))

        # Restore optimizer only if param count matches (architecture unchanged).
        # If there's a mismatch (e.g. loading a double-TCMN checkpoint into
        # fixed single-TCMN model), start with fresh optimizer to avoid
        # corrupted momentum buffers.
        if checkpoint.get('optimizer_state') is not None:
            try:
                old_n_params = sum(
                    len(g['params']) for g in checkpoint['optimizer_state']['param_groups']
                )
                new_n_params = sum(
                    len(g['params']) for g in self.optimizer.param_groups
                )
                if old_n_params == new_n_params:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                else:
                    self.print_to_log_file(
                        f"[CMTC] Optimizer param count mismatch ({old_n_params} → {new_n_params}) "
                        "— starting with fresh optimizer."
                    )
            except Exception:
                self.print_to_log_file(
                    "[CMTC] Could not restore optimizer state (cross-trainer init) — "
                    "starting with fresh optimizer."
                )

        if self.grad_scaler is not None and checkpoint.get('grad_scaler_state') is not None:
            try:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
            except Exception:
                pass

    # ── Train step ────────────────────────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        """
        Extends the parent train_step with the CMTC loss term.

        Strategy
        --------
        The parent (nnUNetTrainerTCMN) already:
          1. Extracts modality_ids.
          2. Sets the TCMN modality context.
          3. Runs the full forward+backward pass with CE+Dice+cbDice+SDF+skel.
          4. Returns a dict with keys {loss, l_seg, l_topo, l_sdf, l_skel}.

        We cannot simply call super().train_step() because the backward() is
        called inside it — we would need to add CMTC *before* the backward.

        Therefore we re-implement the train_step here, duplicating the parent
        logic but inserting the CMTC term into the loss before backward().
        This is the same pattern used in other nnU-Net multi-loss trainers.

        To avoid code rot when the parent changes, we import the aux helpers
        directly from the parent class rather than copy them.
        """
        from torch.amp import autocast
        from contextlib import contextmanager

        @contextmanager
        def dummy_context():
            yield

        data   = batch['data']
        target = batch['target']
        keys   = batch.get('keys', [])

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # ── Modality IDs for TCMN context and CMTC loss ──────────────────────
        modality_ids = _modality_ids_from_keys(keys, self.device)

        # ── Determine patch size for aux map loading ──────────────────────────
        ref = target[0] if isinstance(target, list) else target
        patch_size = tuple(ref.shape[2:])  # (D, H, W)

        # ── Load aux maps (vessel_union, sdf, skeleton) ───────────────────────
        # _load_aux_batch is defined in nnUNetTrainerTopologyVessel (grandparent).
        aux = self._load_aux_batch(keys, crop_bboxes=None, patch_size=patch_size)

        # ── Forward pass under TCMN modality context ──────────────────────────
        self.optimizer.zero_grad(set_to_none=True)

        ctx = (autocast(self.device.type)
               if self.device.type == 'cuda' else dummy_context())

        with tcmn.modality_context(modality_ids):
            with ctx:
                net_out = self.network(data)   # dict: {seg, sdf, skel}

                seg_out  = net_out['seg']
                sdf_out  = net_out['sdf']
                skel_out = net_out['skel']

                # ── Segmentation loss (CE + Dice, deep supervision) ───────────
                l_seg = self.loss(seg_out, target)

                # ── cbDice topology loss ──────────────────────────────────────
                l_topo = torch.tensor(0.0, device=self.device)
                if aux['vessel_union'] is not None and self.λ_topo > 0:
                    seg_full = seg_out[0] if isinstance(seg_out, list) else seg_out
                    seg_half = F.interpolate(seg_full, scale_factor=0.5,
                                             mode='trilinear', align_corners=False)
                    vu_half  = F.interpolate(aux['vessel_union'], scale_factor=0.5,
                                             mode='nearest')
                    l_topo = self._cbdice_loss(seg_half, vu_half)

                # ── SDF regression loss ───────────────────────────────────────
                l_sdf = torch.tensor(0.0, device=self.device)
                if aux['sdf'] is not None and self.λ_sdf > 0:
                    sdf_target = aux['sdf']
                    if sdf_out.shape != sdf_target.shape:
                        sdf_out_rs = F.interpolate(
                            sdf_out, size=sdf_target.shape[2:],
                            mode='trilinear', align_corners=False)
                    else:
                        sdf_out_rs = sdf_out
                    l_sdf = self._sdf_criterion(sdf_out_rs, sdf_target)

                # ── Skeleton head loss ────────────────────────────────────────
                l_skel = torch.tensor(0.0, device=self.device)
                if aux['skeleton'] is not None and self.λ_skel > 0:
                    skel_target = aux['skeleton']
                    if skel_out.shape != skel_target.shape:
                        skel_out_rs = F.interpolate(
                            skel_out, size=skel_target.shape[2:],
                            mode='trilinear', align_corners=False)
                    else:
                        skel_out_rs = skel_out
                    l_skel = self._skel_criterion(skel_out_rs, skel_target)

                # ── CMTC cross-modal topology consistency loss ────────────────
                # PH is expensive: compute every cmtc_every_n_steps steps only.
                # Also respect warm-up: skip CMTC for the first N epochs.
                l_cmtc = torch.tensor(0.0, device=self.device)
                self._cmtc_step_counter += 1
                cmtc_ready = (
                    self.lambda_cmtc > 0
                    and self._cmtc_step_counter % self.cmtc_every_n_steps == 0
                    and self.current_epoch >= self.cmtc_warmup_epochs
                )
                if cmtc_ready:
                    try:
                        l_cmtc = self.cmtc_loss(seg_out, modality_ids)
                        # Guard against NaN/inf from PH computation
                        if not torch.isfinite(l_cmtc):
                            self.print_to_log_file(
                                f"[CMTC] WARNING: CmtcLoss returned {l_cmtc.item()}. "
                                "Replacing with 0."
                            )
                            l_cmtc = torch.tensor(0.0, device=self.device)
                    except Exception as exc:
                        # Safety net: CMTC uses external PH library; if it fails
                        # (e.g. first epoch, degenerate maps) log and continue.
                        self.print_to_log_file(
                            f"[CMTC] WARNING: CmtcLoss raised exception: {exc}. "
                            "Skipping CMTC term this step."
                        )
                        l_cmtc = torch.tensor(0.0, device=self.device)

                # ── Total weighted loss ───────────────────────────────────────
                loss = (l_seg
                        + self.λ_topo  * l_topo
                        + self.λ_sdf   * l_sdf
                        + self.λ_skel  * l_skel
                        + self.lambda_cmtc * l_cmtc)

        # ── NaN guard: skip backward entirely if loss is non-finite ───────────
        # Once NaN enters the optimizer state, the model is irrecoverable.
        # Skipping the step is always safer than propagating NaN gradients.
        if not torch.isfinite(loss):
            self.print_to_log_file(
                f"[CMTC] WARNING: total loss is {loss.item():.4f} — skipping "
                f"backward (l_seg={l_seg.item():.4f}, l_topo={l_topo.item():.4f}, "
                f"l_sdf={l_sdf.item():.4f}, l_skel={l_skel.item():.4f}, "
                f"l_cmtc={l_cmtc.item():.4f})"
            )
            self.optimizer.zero_grad(set_to_none=True)
            return {
                'loss':    float('nan'),
                'l_seg':   l_seg.detach().cpu().numpy(),
                'l_topo':  l_topo.detach().cpu().numpy(),
                'l_sdf':   l_sdf.detach().cpu().numpy(),
                'l_skel':  l_skel.detach().cpu().numpy(),
                'l_cmtc':  l_cmtc.detach().cpu().numpy(),
            }

        # ── Backward + optimiser step ─────────────────────────────────────────
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)

            # grad_scaler.step() skips the optimizer step if NaN/inf grads
            # were detected during unscale_.
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # Cap the grad_scaler AFTER update to prevent growth beyond safe
            # fp16 range.  AMP's default growth can push scale to 2^20+,
            # causing fp16 gradients to overflow on multi-head losses.
            MAX_SCALE = 65536.0   # 2^16
            if (self.grad_scaler._scale is not None
                    and self.grad_scaler.get_scale() > MAX_SCALE):
                self.grad_scaler._scale.fill_(MAX_SCALE)

            # Belt-and-suspenders: verify no NaN crept into weights.
            # If it did, the model is irrecoverable in this step — log loudly.
            if any(torch.isnan(p).any() for p in self.network.parameters()
                   if p is not None):
                self.print_to_log_file(
                    "[CMTC] CRITICAL: NaN detected in network weights after "
                    "optimizer step! Model may be corrupted."
                )
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss':    loss.detach().cpu().numpy(),
            'l_seg':   l_seg.detach().cpu().numpy(),
            'l_topo':  l_topo.detach().cpu().numpy(),
            'l_sdf':   l_sdf.detach().cpu().numpy(),
            'l_skel':  l_skel.detach().cpu().numpy(),
            'l_cmtc':  l_cmtc.detach().cpu().numpy(),
        }

    # ── Validation step inherits from nnUNetTrainerTCMN (seg loss only) ───────
    # No CMTC at validation: we measure topology consistency only during training;
    # the validation metric is the standard Dice, which is already informative.


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerCMTC_NoTopo(nnUNetTrainerCMTC):
    """
    CMTC + TCMN domain norm, no cbDice / SDF / skeleton.

    Ablation: isolates the contribution of CMTC vs all topology losses.
    """
    λ_topo: float = 0.0
    λ_sdf:  float = 0.0
    λ_skel: float = 0.0


class nnUNetTrainerCMTC_Strong(nnUNetTrainerCMTC):
    """
    CMTC with a stronger cross-modal consistency weight (λ_cmtc = 0.3).

    Use when cross-domain generalisation is the primary concern over
    in-domain accuracy.
    """
    lambda_cmtc: float = 0.3


class nnUNetTrainerCMTC_WeakB1(nnUNetTrainerCMTC):
    """
    CMTC with Betti-1 contribution disabled (λ_b1 = 0.0).

    Ablation: measures whether loop topology consistency adds anything
    beyond connected-component consistency for brain vessel segmentation.
    """
    cmtc_lambda_b1: float = 0.0
