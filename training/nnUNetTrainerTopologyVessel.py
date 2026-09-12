"""
nnUNetTrainerTopologyVessel.py
==============================
Custom nnU-Net v2 trainer implementing the topology-aware vessel
segmentation framework described in the TMI proposal.

Ablation ladder implemented here
---------------------------------
  E0  nnUNetTrainer baseline            (standard CE + Dice)
  E1  + CbDiceLoss on vessel union      (topology + radius balance)
  E2  + SDF auxiliary regression head   (shape scaffold)

This file implements E1+E2.  The baseline E0 is the stock nnUNetTrainer.

Design decisions
----------------
1. The trainer subclasses nnUNetTrainer and overrides only:
     build_network_architecture  — wraps backbone with aux heads
     _build_loss                 — noop (losses computed in train_step)
     train_step                  — custom multi-loss forward pass
     validation_step             — standard seg loss only (fast eval)
     on_train_start              — load vessel-union aux maps into memory

2. Vessel-union binary masks are stored in
     {nnUNet_preprocessed}/../aux/<DatasetName>/vessel_union/<subj>.nii.gz
   They are loaded lazily per batch using subject keys from batch['keys'].

3. SDF targets are stored in
     {nnUNet_preprocessed}/../aux/<DatasetName>/sdf/<subj>.nii.gz

4. Patch-level alignment: nnU-Net crops random patches from the full
   volume.  We need the *same* crop from the aux maps.  We achieve this
   by loading the full aux volume and applying the same crop bounding-box
   that nnU-Net stored in the .pkl properties file.
   Fallback: centre-crop to patch size.

Usage
-----
  bash nnunet_env.sh train 100 0 nnUNetTrainerTopologyVessel
  # or equivalently:
  source nnunet_env.sh
  nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerTopologyVessel
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk

# ── Make sure our training/ directory is importable ──────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

from loss_cbdice import CbDiceLoss
from network_wrapper import TopologyVesselNet


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton BCE+Dice loss (sparse — skeleton voxels are extremely rare)
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonLoss(nn.Module):
    """BCE + Dice loss masked to skeleton voxels."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred_logits : (B, 1, D, H, W)
        target      : (B, 1, D, H, W) binary {0, 1}
        """
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')

        pred_prob = torch.sigmoid(pred_logits)
        inter = (pred_prob * target).sum()
        denom = pred_prob.sum() + target.sum() + self.smooth
        dice_loss = 1.0 - (2 * inter + self.smooth) / denom

        return bce + dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# Aux-map loader: reads vessel_union, sdf, skeleton patches
# ─────────────────────────────────────────────────────────────────────────────

class AuxMapCache:
    """
    Lazy per-volume cache for auxiliary maps (vessel union, SDF, skeleton).

    Volumes are loaded on first access, cached in a dict bounded by
    `max_cached` entries (LRU-style eviction).
    """

    def __init__(self, aux_root: Path, map_types: List[str], max_cached: int = 60):
        self.aux_root   = aux_root
        self.map_types  = map_types    # e.g. ['vessel_union', 'sdf', 'skeleton']
        self.max_cached = max_cached
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}   # subj_id -> {map_type -> arr}
        self._order: List[str] = []

    def _load(self, subj_id: str) -> Dict[str, np.ndarray]:
        maps = {}
        for mt in self.map_types:
            p = self.aux_root / mt / f"{subj_id}.nii.gz"
            if p.exists():
                arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
                maps[mt] = arr
            else:
                maps[mt] = None
        return maps

    def get(self, subj_id: str) -> Dict[str, Optional[np.ndarray]]:
        if subj_id not in self._cache:
            # Evict oldest if full
            if len(self._cache) >= self.max_cached:
                old = self._order.pop(0)
                del self._cache[old]
            self._cache[subj_id] = self._load(subj_id)
            self._order.append(subj_id)
        return self._cache[subj_id]


def _crop_patch(
    vol: np.ndarray,
    crop_bbox: Optional[Tuple],
    patch_size: Tuple[int, int, int],
) -> np.ndarray:
    """
    Extract a patch from vol matching nnU-Net's crop bounding box.

    crop_bbox : ((d0,d1),(h0,h1),(w0,w1)) or None
    patch_size: (D, H, W)

    If bbox is None or mismatched, falls back to centre crop.
    """
    D, H, W = vol.shape[-3], vol.shape[-2], vol.shape[-1]
    pD, pH, pW = patch_size

    if crop_bbox is not None:
        try:
            (d0, d1), (h0, h1), (w0, w1) = crop_bbox
            patch = vol[..., d0:d1, h0:h1, w0:w1]
            # Resize if off by 1 due to rounding
            if patch.shape[-3:] != (pD, pH, pW):
                t = torch.from_numpy(patch[np.newaxis, np.newaxis].astype(np.float32))
                t = F.interpolate(t, size=(pD, pH, pW), mode='nearest')
                patch = t.squeeze().numpy()
            return patch
        except Exception:
            pass

    # Centre crop fallback
    d0 = max(0, (D - pD) // 2); d1 = d0 + pD
    h0 = max(0, (H - pH) // 2); h1 = h0 + pH
    w0 = max(0, (W - pW) // 2); w1 = w0 + pW
    patch = vol[..., d0:d1, h0:h1, w0:w1]
    if patch.shape[-3:] != (pD, pH, pW):
        t = torch.from_numpy(patch[np.newaxis, np.newaxis].astype(np.float32))
        t = F.interpolate(t, size=(pD, pH, pW), mode='nearest')
        patch = t.squeeze().numpy()
    return patch


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerTopologyVessel(nnUNetTrainer):
    """
    Topology-aware vessel segmentation trainer.

    Adds to the baseline nnUNet loss:
      L_cbDice  : radius-weighted topology loss on vessel union  (weight λ_topo)
      L_SDF     : MSE regression of signed distance field        (weight λ_sdf)
      L_skel    : BCE+Dice on centerline skeleton                (weight λ_skel)

    Loss weights (class variables, easy to override in subclasses):
    """

    # ── Hyperparameters (override in subclass for ablations) ─────────────
    # Reduced from (0.5, 0.2, 0.1) — previous values caused the cbDice topology
    # loss to dominate, pushing the model toward over-segmentation. The seg loss
    # must remain the primary training signal.
    λ_topo: float = 0.1     # cbDice weight (was 0.5)
    λ_sdf:  float = 0.05    # SDF regression weight (was 0.2)
    λ_skel: float = 0.05    # Skeleton head weight (was 0.1)

    # Soft skeletonisation iterations. Reduced to 3 to fit 20GB GPU.
    # cbDice is computed at half-resolution (see train_step) to cut memory 8x.
    SKEL_ITER: int = 3

    # Aux map types to load
    AUX_MAPS: List[str] = ['vessel_union', 'sdf', 'skeleton']

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self._aux_cache: Optional[AuxMapCache] = None
        self._aux_root:  Optional[Path] = None

    def on_epoch_end(self):
        """Save checkpoint_latest.pth every epoch so resumption never loses progress.

        Must save BEFORE calling super() because super() increments self.current_epoch
        at its end, which would make the saved epoch counter ahead of the logger lists.
        """
        from batchgenerators.utilities.file_and_folder_operations import join
        super().on_epoch_end()
        # Temporarily step current_epoch back to match the logger list length,
        # then restore — ensures checkpoint_latest epoch == logger length.
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
        self.current_epoch += 1

    def _do_i_compile(self) -> bool:
        # torch.compile cannot trace our custom forward hook + cbDice loop.
        # Disable it unconditionally for this trainer family.
        return False

    # ── Dataset name helper ───────────────────────────────────────────────
    def _get_dataset_name(self) -> str:
        """Extract DatasetXXX_Name from the nnUNet_preprocessed path."""
        prep_dir = Path(self.preprocessed_dataset_folder_base)
        return prep_dir.name   # e.g. "Dataset100_TopBrain_CT"

    def _get_aux_root(self) -> Optional[Path]:
        """
        Locate the aux/ directory for this dataset.
        Layout: {nnUNet_preprocessed}/../../../aux/<DatasetName_short>/
        e.g.:
          nnUNet_preprocessed = .../preprocessed/nnUNet_preprocessed/Dataset100_TopBrain_CT
          aux_root            = .../preprocessed/aux/TopBrain_CT
        """
        ds_full = self._get_dataset_name()  # "Dataset100_TopBrain_CT"
        # Strip "DatasetXXX_" prefix
        parts = ds_full.split('_', 1)
        ds_short = parts[1] if len(parts) > 1 else ds_full

        # Navigate up from nnUNet_preprocessed/Dataset.../
        prep_base = Path(self.preprocessed_dataset_folder_base)
        # .../preprocessed/nnUNet_preprocessed  -> .../preprocessed -> .../aux
        aux_root = prep_base.parent.parent / 'aux' / ds_short
        if aux_root.exists():
            return aux_root

        # Fallback: try full dataset name
        aux_root2 = prep_base.parent.parent / 'aux' / ds_full
        if aux_root2.exists():
            return aux_root2

        self.print_to_log_file(
            f"[TopologyVessel] WARNING: aux root not found for {ds_full}. "
            f"Tried: {aux_root}, {aux_root2}. "
            f"cbDice and aux losses will be skipped."
        )
        return None

    # ── on_train_start: initialise aux map cache ──────────────────────────
    def on_train_start(self):
        super().on_train_start()
        self._aux_root = self._get_aux_root()
        if self._aux_root is not None:
            self._aux_cache = AuxMapCache(
                aux_root=self._aux_root,
                map_types=self.AUX_MAPS,
                max_cached=80,
            )
            self.print_to_log_file(
                f"[TopologyVessel] Aux map cache initialised at: {self._aux_root}"
            )
        else:
            self._aux_cache = None

    # ── Network: wrap backbone with aux heads ─────────────────────────────
    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

        # Infer feature channels from plans (first stage = finest resolution)
        feat_ch = arch_init_kwargs.get('features_per_stage', [32])[0]

        return TopologyVesselNet(
            backbone=backbone,
            backbone_feat_channels=feat_ch,
            enable_aux_heads=True,
        )

    # ── Loss (base seg loss only, aux losses computed in train_step) ──────
    def _build_loss(self):
        """
        Returns the segmentation loss (CE + Dice) with deep supervision.
        Topology (cbDice), SDF, and skeleton losses are added in train_step.
        """
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    # ── Aux loss modules (built lazily on first use) ──────────────────────
    @property
    def _cbdice_loss(self) -> CbDiceLoss:
        if not hasattr(self, '_cbdice_loss_module'):
            self._cbdice_loss_module = CbDiceLoss(
                smooth=1e-5, n_skel_iter=self.SKEL_ITER, use_radius_weight=True
            ).to(self.device)
        return self._cbdice_loss_module

    @property
    def _sdf_criterion(self) -> nn.MSELoss:
        if not hasattr(self, '_sdf_mse'):
            self._sdf_mse = nn.MSELoss()
        return self._sdf_mse

    @property
    def _skel_criterion(self) -> SkeletonLoss:
        if not hasattr(self, '_skel_loss'):
            self._skel_loss = SkeletonLoss().to(self.device)
        return self._skel_loss

    # ── Aux map batch loader ──────────────────────────────────────────────
    def _load_aux_batch(
        self,
        keys: List[str],
        crop_bboxes: Optional[List],
        patch_size: Tuple[int, int, int],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Load vessel_union, sdf, skeleton patches for the current batch.

        Returns a dict: {'vessel_union': Tensor|None, 'sdf': ..., 'skeleton': ...}
        Shape of each tensor: (B, 1, D, H, W)
        """
        if self._aux_cache is None:
            return {mt: None for mt in self.AUX_MAPS}

        result = {mt: [] for mt in self.AUX_MAPS}

        for i, key in enumerate(keys):
            # nnU-Net keys can be like 'topcow_ct_001' or full path; take stem
            subj_id = Path(key).stem if '/' in key or '\\' in key else key
            bbox = crop_bboxes[i] if crop_bboxes is not None else None

            maps = self._aux_cache.get(subj_id)
            for mt in self.AUX_MAPS:
                arr = maps.get(mt)
                if arr is not None:
                    patch = _crop_patch(arr, bbox, patch_size)
                    result[mt].append(torch.from_numpy(patch[np.newaxis]).float())
                else:
                    result[mt].append(None)

        # Stack — if any subject is missing a map, skip the whole map for this batch
        out = {}
        for mt in self.AUX_MAPS:
            items = result[mt]
            if all(t is not None for t in items):
                out[mt] = torch.stack(items, dim=0).to(self.device, non_blocking=True)
            else:
                out[mt] = None
        return out

    # ── On-the-fly aux target computation from seg target ────────────────
    @staticmethod
    def _compute_vessel_union(seg_target: torch.Tensor) -> torch.Tensor:
        """
        Derive vessel union mask from the segmentation target.
        seg_target : (B, 1, D, H, W) integer label map
        Returns    : (B, 1, D, H, W) float binary vessel mask
        """
        return (seg_target > 0).float()

    @staticmethod
    def _compute_sdf_gpu(vessel_union: torch.Tensor, d_max: float = 5.0) -> torch.Tensor:
        """
        Approximate SDF on GPU using iterative erosion depth.
        vessel_union : (B, 1, D, H, W) float binary mask
        Returns      : (B, 1, D, H, W) approximate SDF in [-1, 1]
        """
        fg = vessel_union
        bg = 1.0 - fg
        n_iter = int(d_max)

        fg_dist_val = torch.zeros_like(fg)
        bg_dist_val = torch.zeros_like(bg)
        fg_rem = fg.clone()
        bg_rem = bg.clone()

        for i in range(1, n_iter + 1):
            fg_eroded = -F.max_pool3d(-fg_rem, kernel_size=3, stride=1, padding=1)
            fg_dist_val += (fg_rem - fg_eroded) * i
            fg_rem = fg_eroded

            bg_eroded = -F.max_pool3d(-bg_rem, kernel_size=3, stride=1, padding=1)
            bg_dist_val += (bg_rem - bg_eroded) * i
            bg_rem = bg_eroded

        # SDF convention: positive outside (bg), negative inside (fg)
        sdf = bg_dist_val * bg - fg_dist_val * fg
        sdf = sdf.clamp(-d_max, d_max) / d_max
        return sdf

    # ── Train step ────────────────────────────────────────────────────────
    def train_step(self, batch: dict) -> dict:
        from torch.amp import autocast
        from contextlib import contextmanager

        @contextmanager
        def dummy_context():
            yield

        data   = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # ── Compute aux targets on-the-fly from the correctly-cropped seg target ──
        # All derived from batch['target'] — guaranteed aligned with batch['data']
        ref = target[0] if isinstance(target, list) else target
        vessel_union = self._compute_vessel_union(ref)
        sdf_target = self._compute_sdf_gpu(vessel_union)
        # Skeleton target: use soft_skeletonize_3d from cbDice (GPU, differentiable)
        from loss_cbdice import soft_skeletonize_3d
        with torch.no_grad():
            skel_target = soft_skeletonize_3d(vessel_union, n_iter=self.SKEL_ITER)
            skel_target = (skel_target > 0.5).float()  # binarize for BCE+Dice target

        # ── Forward pass ──────────────────────────────────────────────────
        self.optimizer.zero_grad(set_to_none=True)

        ctx = autocast(self.device.type) \
            if self.device.type == 'cuda' else dummy_context()

        with ctx:
            net_out = self.network(data)   # dict with 'seg', 'sdf', 'skel'

            seg_out  = net_out['seg']
            sdf_out  = net_out['sdf']
            skel_out = net_out['skel']

            # ── Segmentation loss (CE + Dice, with DS) ────────────────────
            l_seg = self.loss(seg_out, target)

            # ── cbDice topology loss ──────────────────────────────────────
            l_topo = torch.tensor(0.0, device=self.device)
            if self.λ_topo > 0:
                seg_full = seg_out[0] if isinstance(seg_out, list) else seg_out
                # Downsample to half-res before skeletonization — cuts memory 8x
                seg_half = F.interpolate(seg_full, scale_factor=0.5, mode='trilinear',
                                         align_corners=False)
                vu_half  = F.interpolate(vessel_union, scale_factor=0.5,
                                         mode='nearest')
                l_topo = self._cbdice_loss(seg_half, vu_half)

            # ── SDF regression loss ───────────────────────────────────────
            l_sdf = torch.tensor(0.0, device=self.device)
            if self.λ_sdf > 0:
                if sdf_out.shape != sdf_target.shape:
                    sdf_out_rs = F.interpolate(
                        sdf_out, size=sdf_target.shape[2:], mode='trilinear',
                        align_corners=False)
                else:
                    sdf_out_rs = sdf_out
                l_sdf = self._sdf_criterion(sdf_out_rs, sdf_target)

            # ── Skeleton head loss ────────────────────────────────────────
            l_skel = torch.tensor(0.0, device=self.device)
            if self.λ_skel > 0:
                if skel_out.shape != skel_target.shape:
                    skel_out_rs = F.interpolate(
                        skel_out, size=skel_target.shape[2:], mode='trilinear',
                        align_corners=False)
                else:
                    skel_out_rs = skel_out
                l_skel = self._skel_criterion(skel_out_rs, skel_target)

            # ── Total loss ────────────────────────────────────────────────
            loss = (l_seg
                    + self.λ_topo * l_topo
                    + self.λ_sdf  * l_sdf
                    + self.λ_skel * l_skel)

        # ── Backward ─────────────────────────────────────────────────────
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss':   loss.detach().cpu().numpy(),
            'l_seg':  l_seg.detach().cpu().numpy(),
            'l_topo': l_topo.detach().cpu().numpy(),
            'l_sdf':  l_sdf.detach().cpu().numpy(),
            'l_skel': l_skel.detach().cpu().numpy(),
        }

    # ── Validation step (seg only — delegates to base after unwrapping) ───
    def validation_step(self, batch: dict) -> dict:
        """
        Override network call to unwrap the dict output, then delegate
        all metric computation to the base nnUNetTrainer.validation_step
        by temporarily monkey-patching the network forward.
        """
        from torch.amp import autocast
        from contextlib import contextmanager

        @contextmanager
        def dummy_context():
            yield

        data   = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type) \
                if self.device.type == 'cuda' else dummy_context():
            net_out = self.network(data)
            seg_out = net_out['seg'] if isinstance(net_out, dict) else net_out
            del data
            l = self.loss(seg_out, target)

        # Replicate base trainer metric computation exactly
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        output = seg_out[0] if self.enable_deep_supervision else seg_out
        tgt    = target[0]  if self.enable_deep_supervision else target

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (tgt != self.label_manager.ignore_label).float()
                tgt[tgt == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - tgt[:, -1:] if tgt.dtype != torch.bool else ~tgt[:, -1:]
                tgt  = tgt[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, tgt, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
        }



# ─────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerTopologyVessel_cbDiceOnly(nnUNetTrainerTopologyVessel):
    """E1 only: cbDice, no SDF head, no skeleton head."""
    λ_topo: float = 0.5
    λ_sdf:  float = 0.0
    λ_skel: float = 0.0

    @staticmethod
    def build_network_architecture(
        architecture_class_name, arch_init_kwargs,
        arch_init_kwargs_req_import, num_input_channels,
        num_output_channels, enable_deep_supervision=True,
    ) -> nn.Module:
        # Standard backbone — no aux heads needed when sdf+skel disabled
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        backbone = get_network_from_plans(
            architecture_class_name, arch_init_kwargs,
            arch_init_kwargs_req_import, num_input_channels,
            num_output_channels, allow_init=True,
            deep_supervision=enable_deep_supervision)
        feat_ch = arch_init_kwargs.get('features_per_stage', [32])[0]
        # Wrap but disable aux heads (cbDice only uses seg output)
        return TopologyVesselNet(backbone, feat_ch, enable_aux_heads=False)


class nnUNetTrainerTopologyVessel_SDFOnly(nnUNetTrainerTopologyVessel):
    """E2 only: SDF head, no cbDice."""
    λ_topo: float = 0.0
    λ_sdf:  float = 0.2
    λ_skel: float = 0.0
