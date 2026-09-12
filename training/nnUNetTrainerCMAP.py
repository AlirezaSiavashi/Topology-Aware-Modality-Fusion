"""
nnUNetTrainerCMAP.py
====================
Cross-Modal Anatomical Priors trainer for joint CTA + MRA Circle-of-Willis
segmentation (Dataset104).

Objective
---------
    L = L_seg(CE+Dice, deep supervised)
      + lambda_prior * L_prior      (reliability- and radius-weighted
                                     prototype cross-entropy in hyperbolic space)
      + lambda_gate  * L_gate       (identity regulariser on the fusion gate)

Schedule
--------
- Epoch 0 .. T_gate-1 : gate inactive, network is a plain nnU-Net.  The prior
  loss still runs, so the prototype bank organises itself before it is allowed
  to touch the segmentation path.  Injecting an under-trained prior early is
  the failure mode this avoids.
- Epoch T_gate ..     : gated fusion active.

The reliability matrix is refreshed at the end of every validation epoch from
held-out per-class Dice, split by modality.  It is uniform until the first
validation epoch has completed.

Modality labels are needed for the *loss* only.  The forward pass never sees
them, so inference is modality-agnostic and the trained model runs on a CTA or
an MRA with no domain indicator -- unlike domain-specific normalisation
schemes, which need to be told which domain they are in.

Run
---
  nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerCMAP
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from cmap import (
    RadiusStats,
    gate_regulariser,
    prior_loss,
    radius_regulariser,
    tree_loss,
)
from cmap_net import CMAPNet


CT, MR = 0, 1


def modality_ids_from_keys(keys: List[str], device: torch.device) -> torch.Tensor:
    """'topcow_ct_001' -> 0 (CTA), 'topcow_mr_001' -> 1 (MRA)."""
    ids = []
    for key in keys:
        k = str(key).lower()
        ids.append(MR if ("_mr_" in k or "_mra_" in k or k.endswith("_mr")) else CT)
    return torch.tensor(ids, dtype=torch.long, device=device)


class nnUNetTrainerCMAP(nnUNetTrainer):

    # ── Objective weights ────────────────────────────────────────────────
    lambda_prior: float = 0.3
    lambda_gate: float = 0.05
    # Supervises the known CoW arterial tree on the prototype bank. This is
    # what makes the radial coordinate encode branch order; left to itself the
    # embedding collapses to a single radius (measured spread 0.009 across all
    # 14 classes).
    lambda_tree: float = 0.5
    # Pulls each voxel to the radius of its own class prototype. Required:
    # the prototype cross-entropy is scale-free and will not do this on its
    # own (voxels drift to r~0.95 whatever their prototype's radius).
    lambda_match: float = 1.0
    # Inward radial pull. Superseded by lambda_tree and off by default: on its
    # own it trades boundary collapse for origin collapse, since mean(r) is
    # minimised by making every radius equal and small.
    lambda_radius: float = 0.0

    # ── Geometry / bank ──────────────────────────────────────────────────
    EMBED_DIM: int = 16
    CURVATURE: float = 1.0      # c; ablated in the paper
    TAU: float = 0.5            # assignment temperature
    PRIOR_STRIDE: int = 2       # prior lattice = 2x decoder stride
    GATE_HIDDEN: int = 32
    USE_HYPERBOLIC: bool = True   # False = Euclidean control
    MAX_FG_VOXELS: int = 8192   # sampled foreground voxels per step

    # ── Schedule ─────────────────────────────────────────────────────────
    T_gate: int = 100           # 10% of the default 1000-epoch schedule

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self._radius_stats: Optional[RadiusStats] = None
        self._class_names: List[str] = []

    # Hooks that mutate module outputs do not survive torch.compile cleanly.
    def _do_i_compile(self) -> bool:
        return False

    # ── Network ──────────────────────────────────────────────────────────
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
        feat_ch = arch_init_kwargs.get("features_per_stage", [32])[0]

        return CMAPNet(
            backbone=backbone,
            feat_channels=feat_ch,
            num_classes=num_output_channels,
            embed_dim=nnUNetTrainerCMAP.EMBED_DIM,
            curvature=nnUNetTrainerCMAP.CURVATURE,
            tau=nnUNetTrainerCMAP.TAU,
            prior_stride=nnUNetTrainerCMAP.PRIOR_STRIDE,
            gate_hidden=nnUNetTrainerCMAP.GATE_HIDDEN,
            num_modalities=2,
            use_hyperbolic=nnUNetTrainerCMAP.USE_HYPERBOLIC,
        )

    def initialize(self):
        super().initialize()
        k = self.label_manager.num_segmentation_heads
        self._radius_stats = RadiusStats(num_classes=k, num_modalities=2)
        # Index by label VALUE, not by dict insertion order. Dataset104 lists
        # "background": 0 LAST, so keys()[k] is off by one against prototype k
        # (prototypes are indexed by label value, as the seg target is).
        self._class_names = [str(i) for i in range(k)]
        for name, val in self.dataset_json["labels"].items():
            v = int(val)
            if 0 <= v < k:
                self._class_names[v] = str(name)
        self.print_to_log_file(
            f"[CMAP] prototypes={k} embed_dim={self.EMBED_DIM} c={self.CURVATURE} "
            f"tau={self.TAU} prior_stride={self.PRIOR_STRIDE} "
            f"T_gate={self.T_gate} lambda_prior={self.lambda_prior} "
            f"lambda_gate={self.lambda_gate}"
        )

    # ── Delayed fusion ───────────────────────────────────────────────────
    def _net(self) -> CMAPNet:
        net = self.network
        return net.module if hasattr(net, "module") else net

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        active = self.current_epoch >= self.T_gate
        net = self._net()
        if net.fuse_active != active:
            self.print_to_log_file(
                f"[CMAP] gated fusion {'ENABLED' if active else 'disabled'} "
                f"at epoch {self.current_epoch}"
            )
        net.fuse_active = active
        if self._radius_stats is not None:
            self._radius_stats.reset()

    # ── Train step ───────────────────────────────────────────────────────
    def train_step(self, batch: dict) -> dict:
        from contextlib import contextmanager

        from torch.amp import autocast

        @contextmanager
        def dummy_context():
            yield

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        mod_ids = modality_ids_from_keys(batch.get("keys", []), self.device)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type) if self.device.type == "cuda" else dummy_context()

        with ctx:
            net_out = self.network(data)
            seg_out = net_out["seg"]
            aux = net_out["cmap"]

            l_seg = self.loss(seg_out, target)

            l_prior = torch.zeros((), device=self.device)
            l_match = torch.zeros((), device=self.device)
            l_gate = torch.zeros((), device=self.device)
            # Fixed key set for the same collate_outputs reason as in
            # validation_step: every step must report the same fields.
            pstats: Dict[str, float] = {"n_vox": 0.0, "mean_radius": 0.0,
                                        "mean_rho": 0.0, "tree_order": 0.0,
                                        "tree_anchor": 0.0, "tree_violations": 0.0,
                                        "r_match": 0.0}

            if aux is not None and self.lambda_prior > 0:
                ref = target[0] if isinstance(target, list) else target
                # (B,1,D,H,W) -> prior lattice, nearest so labels stay valid
                tgt_low = F.interpolate(
                    ref.float(), size=aux["z"].shape[1:4], mode="nearest"
                ).squeeze(1).long()

                l_prior, l_match, _stats = prior_loss(
                    z=aux["z"],
                    dist=aux["dist"],
                    target_low=tgt_low,
                    reliability=self._net().cmap.reliability,
                    modality_ids=mod_ids,
                    tau=self.TAU,
                    max_fg=self.MAX_FG_VOXELS,
                    radius_stats=self._radius_stats,
                    proto_radii=self._net().cmap.bank.prototypes()
                                    .norm(dim=-1).detach(),
                )
                pstats.update(_stats)

            if aux is not None and self.lambda_gate > 0 and aux.get("gate") is not None:
                l_gate = gate_regulariser(aux["gate"]).to(self.device)

            l_rad = torch.zeros((), device=self.device)
            if aux is not None and self.lambda_radius > 0:
                l_rad = radius_regulariser(aux["z"]).to(self.device)

            l_tree = torch.zeros((), device=self.device)
            if self.lambda_tree > 0:
                l_tree, tstats = tree_loss(
                    self._net().cmap.bank, c=self.CURVATURE)
                l_tree = l_tree.to(self.device)
                pstats.update(tstats)

            loss = (l_seg
                    + self.lambda_prior * l_prior
                    + self.lambda_gate * l_gate
                    + self.lambda_radius * l_rad
                    + self.lambda_tree * l_tree
                    + self.lambda_match * l_match)

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

        out = {
            "loss": loss.detach().cpu().numpy(),
            "l_seg": l_seg.detach().cpu().numpy(),
            "l_prior": l_prior.detach().cpu().numpy(),
            "l_gate": l_gate.detach().cpu().numpy(),
            "l_rad": l_rad.detach().cpu().numpy(),
            "l_tree": l_tree.detach().cpu().numpy(),
            "l_match": l_match.detach().cpu().numpy(),
        }
        out.update(pstats)
        return out

    # ── Validation step: per-modality confusion counts ───────────────────
    @staticmethod
    def _hard_counts(output: torch.Tensor, tgt: torch.Tensor, label_manager):
        axes = [0] + list(range(2, output.ndim))
        if label_manager.has_regions:
            pred_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            seg = output.argmax(1)[:, None]
            pred_onehot = torch.zeros(output.shape, device=output.device,
                                      dtype=torch.float32)
            pred_onehot.scatter_(1, seg, 1)

        if label_manager.has_ignore_label:
            if not label_manager.has_regions:
                mask = (tgt != label_manager.ignore_label).float()
                tgt = tgt.clone()
                tgt[tgt == label_manager.ignore_label] = 0
            else:
                mask = 1 - tgt[:, -1:] if tgt.dtype != torch.bool else ~tgt[:, -1:]
                tgt = tgt[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, tgt, axes=axes, mask=mask)
        tp, fp, fn = (x.detach().cpu().numpy() for x in (tp, fp, fn))
        if not label_manager.has_regions:
            tp, fp, fn = tp[1:], fp[1:], fn[1:]
        return tp, fp, fn

    def validation_step(self, batch: dict) -> dict:
        from contextlib import contextmanager

        from torch.amp import autocast

        @contextmanager
        def dummy_context():
            yield

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        mod_ids = modality_ids_from_keys(batch.get("keys", []), self.device)

        with autocast(self.device.type) if self.device.type == "cuda" else dummy_context():
            net_out = self.network(data)
            seg_out = net_out["seg"] if isinstance(net_out, dict) else net_out
            del data
            l = self.loss(seg_out, target)

        output = seg_out[0] if self.enable_deep_supervision else seg_out
        tgt = target[0] if self.enable_deep_supervision else target

        tp, fp, fn = self._hard_counts(output, tgt, self.label_manager)
        res = {"loss": l.detach().cpu().numpy(),
               "tp_hard": tp, "fp_hard": fp, "fn_hard": fn}

        # Same counts again, but split by modality, for the reliability matrix.
        #
        # These keys must be emitted on EVERY step, even when a batch contains
        # only one modality: nnUNetTrainer collates validation outputs with
        # collate_outputs(), which reads the key set from outputs[0] and then
        # indexes every other output with it. A missing key on any later batch
        # raises KeyError. Zeros are the correct filler -- they contribute
        # nothing to the accumulated confusion counts.
        zero = np.zeros_like(tp)
        for m, name in ((CT, "ct"), (MR, "mr")):
            sel = (mod_ids == m)
            if sel.any():
                t, f, n = self._hard_counts(output[sel], tgt[sel], self.label_manager)
            else:
                t, f, n = zero.copy(), zero.copy(), zero.copy()
            res[f"tp_{name}"], res[f"fp_{name}"], res[f"fn_{name}"] = t, f, n
        return res

    # ── Reliability refresh ──────────────────────────────────────────────
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)

        n_fg = self.label_manager.num_segmentation_heads - 1
        dice = torch.zeros(2, self.label_manager.num_segmentation_heads)
        # Background prototype gets no modality preference.
        dice[:, 0] = 1.0

        per_mod = {}
        for m, name in ((CT, "ct"), (MR, "mr")):
            tp = np.zeros(n_fg)
            fp = np.zeros(n_fg)
            fn = np.zeros(n_fg)
            for o in val_outputs:
                tp = tp + o[f"tp_{name}"]
                fp = fp + o[f"fp_{name}"]
                fn = fn + o[f"fn_{name}"]
            # A modality that never appeared in this validation epoch has all
            # counts zero. That is "no evidence", not "Dice 0" -- scoring it as
            # 0 would hand the whole prior to the other modality. Matters on the
            # single-modality datasets (102 CT-only, 103 MR-only).
            present = float((tp + fp + fn).sum()) > 0
            per_mod[m] = (2 * tp / np.clip(2 * tp + fp + fn, 1e-8, None), present)

        present_mods = [m for m, (_, p) in per_mod.items() if p]
        if len(present_mods) == 2:
            for m in (CT, MR):
                dice[m, 1:] = torch.from_numpy(per_mod[m][0]).float()
        elif len(present_mods) == 1:
            # Single-modality training: keep alpha uniform rather than degenerate.
            d = torch.from_numpy(per_mod[present_mods[0]][0]).float()
            dice[CT, 1:] = d
            dice[MR, 1:] = d
        else:
            self.print_to_log_file("[CMAP] no validation counts; alpha unchanged")
            return

        self._net().cmap.reliability.update_from_dice(dice)

        alpha = self._net().cmap.reliability.alpha
        pretty = ", ".join(
            f"{self._class_names[k]}:{alpha[CT, k]:.2f}/{alpha[MR, k]:.2f}"
            for k in range(1, n_fg + 1)
        )
        self.print_to_log_file(f"[CMAP] reliability alpha CT/MR per class -> {pretty}")

    # ── Radius statistics (vessel-hierarchy evidence) ────────────────────
    def on_epoch_end(self):
        if self._radius_stats is not None and self.current_epoch % 10 == 0:
            r = self._radius_stats.mean_radius()
            names = self._class_names
            parts = []
            for k in range(r.shape[1]):
                nm = names[k] if k < len(names) else str(k)
                vals = r[:, k]
                if not torch.isnan(vals).all():
                    parts.append(f"{nm}:{np.nanmean(vals.numpy()):.3f}")
            if parts:
                # Foreground spread is the number that decides whether the
                # radius encodes anatomy at all. If fg_spread stays near zero
                # while background sits inside the same band, the embedding has
                # collapsed onto the boundary and the hierarchy claim is dead
                # regardless of what Dice does.
                fg = np.nanmean(r[:, 1:].numpy(), axis=0)
                bg = float(np.nanmean(r[:, 0].numpy()))
                if np.isfinite(fg).any():
                    self.print_to_log_file(
                        f"[CMAP] radius fg_spread={np.nanmax(fg) - np.nanmin(fg):.4f} "
                        f"fg_mean={np.nanmean(fg):.4f} bg={bg:.4f} "
                        f"(spread << 0.05 => boundary collapse)"
                    )
                self.print_to_log_file(
                    "[CMAP] mean Poincare radius per class -> " + ", ".join(parts)
                )
        super().on_epoch_end()


# ── Ablations ────────────────────────────────────────────────────────────────

class nnUNetTrainerCMAP_NoHyperbolic(nnUNetTrainerCMAP):
    """
    Euclidean control: identical polar parameterisation, depth anchors and
    radial matching, with only the voxel-to-prototype distance swapped from
    the Poincare geodesic to L2. Any difference is therefore attributable to
    the geometry alone.

    Lowering the curvature instead would also move the radial scale
    (sigmoid(0) -> 0.495 at c=1 vs 0.543 at c=1e-4) and confound the two.
    """
    USE_HYPERBOLIC: bool = False


class nnUNetTrainerCMAP_NoReliability(nnUNetTrainerCMAP):
    """Uniform alpha: ablates the per-class x per-modality weighting."""

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        # Skip the reliability refresh entirely; alpha stays at its 0.5 init.
        nnUNetTrainer.on_validation_epoch_end(self, val_outputs)


class nnUNetTrainerCMAP_NoGate(nnUNetTrainerCMAP):
    """Prototype learning only -- the prior never touches the seg path."""
    T_gate: int = 10 ** 9
    lambda_gate: float = 0.0


class nnUNetTrainerCMAP_NoPrior(nnUNetTrainerCMAP):
    """Gate with an unlearned bank: isolates the contribution of L_prior."""
    lambda_prior: float = 0.0


class nnUNetTrainerCMAP_NoMatch(nnUNetTrainerCMAP):
    """No voxel-to-prototype radial matching: voxels ignore the tree."""
    lambda_match: float = 0.0


class nnUNetTrainerCMAP_NoTree(nnUNetTrainerCMAP):
    """
    No anatomical-tree supervision on the prototypes.

    This is the control for the central claim. Without it the radial
    coordinate collapses (all 14 classes within 0.009 of one another), so this
    run is what establishes that the hierarchy is doing work rather than being
    read post hoc off an arbitrary embedding.
    """
    lambda_tree: float = 0.0


class nnUNetTrainerCMAP_NoRadiusReg(nnUNetTrainerCMAP):
    """
    No inward radial pull. Expected to reproduce boundary collapse -- this is
    the ablation that demonstrates the radial coordinate is doing work rather
    than being a post-hoc reading of an arbitrary embedding.
    """
    lambda_radius: float = 0.0


class nnUNetTrainerCMAP_2epochs(nnUNetTrainerCMAP):
    """
    Integration smoke test. T_gate=1 so a single run exercises both the
    fusion-off and fusion-on code paths, plus the reliability refresh and the
    sliding-window inference path.

    NOTE: the explicit signature is mandatory. nnUNetTrainer.__init__ does
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]
    so a (*args, **kwargs) override raises KeyError: 'args' -- the parameter
    names are looked up in the *parent's* locals().
    """
    T_gate: int = 1

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 2
