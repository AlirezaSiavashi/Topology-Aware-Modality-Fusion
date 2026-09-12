"""
nnUNetTrainerM3CoW.py
=====================
M3-CoW: Mamba-3 rotational state, supervised as a state space, read as keys.

Three changes from nnUNetTrainerMFSSM, in the order they matter:

1. THE LOSS NO LONGER STOPS AT DICE.
   In MF-SSM the only gradient reaching the state came from the segmentation
   head, so the state was a free intermediate. The measured consequence was
   that the factorisation did nothing (+0.003 Dice over its own no-factorisation
   control, against a 0.013 seed-noise floor) and that re-fitting B^MR and
   dt^MR on 5 MRA volumes moved zero-shot Dice from 0.3957 to 0.3957 -- the
   adapter had no organised state geometry to adapt into. Six objectives now
   attach directly to the state, the realised dynamics, and the prototype
   geometry. See state_losses.py.

2. THE STATE IS ROTATIONAL.
   A real diagonal A only decays, so its class prototypes are ordered by
   recency -- a line. The Circle of Willis is a ring and half its classes are
   mirror pairs; a line cannot embed a cycle without tearing it, and the tear
   lands exactly where this model fails today (Acom 0.594, R-Pcom 0.472,
   L-Pcom 0.464, versus 0.851 for R-ICA). A complex A adds phase, which is
   periodic, and L_ring/L_mirror are what spend it. See mamba3_ssm.py.

3. THE STATE IS USED AS KEYS.
   Thirteen anatomy-anchored queries cross-attend with K = W_K h. L_probe makes
   the state linearly decodable, which is the precondition for a dot product to
   retrieve from it, and L_attn trains the retrieval. See anatomy_transformer.py.

Objective
---------
    L = L_seg (CE+Dice, deep supervised)
      + a1 L_align    cross-modal anatomical alignment on h
      + a2 L_dyn      realised-dynamics agreement, dt*lambda and dt*theta
      + a3 L_probe    linear decodability of the state
      + a4 L_vc       variance + decorrelation (anti-collapse)
      + a5 L_ring     circular geometry of the CoW ring
      + a6 L_mirror   bilateral symmetry as a single offset
      + a7 L_attn     query c attends to tokens labelled c      [from T_tr]
      + a8 L_pres     per-vessel presence                       [from T_tr]

Schedule. State losses run from epoch 0; the transformer terms wait until 5% of
training has passed. Same reasoning as CMAP's delayed gate: attention alignment
computed against untrained queries would drag the state toward noise, and the
transformer's fusion weights are zero-initialised anyway, so nothing is lost by
letting the state organise first.

Run
---
  nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerM3CoW
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from mfssm3_net import MFSSM3Net
from anatomy_transformer import attention_alignment_loss, presence_loss
from state_losses import (
    N_CLASS,
    fp32,
    StatePrototypeBank,
    align_loss,
    dynamics_loss,
    mirror_loss,
    probe_loss,
    ring_loss,
    token_labels,
    variance_covariance_loss,
)

CT, MR = 0, 1


def modality_ids_from_keys(keys: List[str], device: torch.device) -> torch.Tensor:
    ids = []
    for key in keys:
        k = str(key).lower()
        ids.append(MR if ("_mr_" in k or "_mra_" in k or k.endswith("_mr")) else CT)
    return torch.tensor(ids, dtype=torch.long, device=device)


class nnUNetTrainerM3CoW(nnUNetTrainer):

    # architecture
    N_STAGES_TO_WRAP: int = 2
    D_STATE: int = 8            # complex; ~= 16 real states, half the memory
    EXPAND: int = 2
    N_GROUPS: int = 1
    D_BOTTLENECK: int = 128    # channel width the scan runs at; None = full 320
    SHARE_A: bool = True
    SHARE_C: bool = True
    TRAPEZOIDAL: bool = True
    COMPLEX_STATE: bool = True
    THETA_SCALE: float = 1.0
    USE_TRANSFORMER: bool = True
    TR_DIM: int = 256
    TR_LAYERS: int = 3

    # loss weights
    L_ALIGN: float = 0.5
    L_DYN: float = 0.1
    L_PROBE: float = 0.2
    L_VC: float = 0.05
    L_RING: float = 0.2
    L_MIRROR: float = 0.1
    L_ATTN: float = 0.3
    L_PRES: float = 0.2
    TR_WARMUP_FRAC: float = 0.05

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.bank: StatePrototypeBank | None = None

    # Forward hooks that replace module outputs do not survive torch.compile.
    def _do_i_compile(self) -> bool:
        return False

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True) -> nn.Module:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        cls = nnUNetTrainerM3CoW
        backbone = get_network_from_plans(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, allow_init=True,
            deep_supervision=enable_deep_supervision)
        net = MFSSM3Net(
            backbone=backbone,
            stage_channels=arch_init_kwargs["features_per_stage"],
            n_stages_to_wrap=cls.N_STAGES_TO_WRAP,
            d_state=cls.D_STATE, expand=cls.EXPAND, n_modalities=2,
            n_groups=cls.N_GROUPS, share_A=cls.SHARE_A, share_C=cls.SHARE_C,
            trapezoidal=cls.TRAPEZOIDAL, d_bottleneck=cls.D_BOTTLENECK,
            tr_dim=cls.TR_DIM,
            tr_layers=cls.TR_LAYERS, use_transformer=cls.USE_TRANSFORMER,
            theta_scale=cls.THETA_SCALE)
        if not cls.COMPLEX_STATE:
            # Real-diagonal ablation: zero the phase and freeze it. This is the
            # Mamba-2-equivalent state and the control for every claim that
            # rests on rotation.
            for b in net.blocks:
                with torch.no_grad():
                    b.ssm.A_theta.zero_()
                b.ssm.A_theta.requires_grad_(False)
        return net

    def _net(self) -> MFSSM3Net:
        n = self.network
        return n.module if hasattr(n, "module") else n

    def initialize(self):
        super().initialize()
        net = self._net()
        self.bank = StatePrototypeBank(net.state_dim, N_CLASS, 2).to(self.device)
        ssm = sum(p.numel() for p in net.blocks.parameters())
        tr = sum(p.numel() for p in net.anatomy.parameters()) if net.anatomy else 0
        total = sum(p.numel() for p in net.parameters())
        self.print_to_log_file(
            f"[M3CoW] wrapped={net.wrapped_idx} d_state={self.D_STATE} "
            f"complex={self.COMPLEX_STATE} trapezoid={self.TRAPEZOIDAL} "
            f"groups={self.N_GROUPS} share_A={self.SHARE_A} share_C={self.SHARE_C}")
        self.print_to_log_file(
            f"[M3CoW] params: SSM {ssm/1e6:.2f}M + anatomy-TR {tr/1e6:.2f}M "
            f"of {total/1e6:.2f}M total ({100*(ssm+tr)/total:.1f}%)")
        self.print_to_log_file(
            f"[M3CoW] lambdas align={self.L_ALIGN} dyn={self.L_DYN} "
            f"probe={self.L_PROBE} vc={self.L_VC} ring={self.L_RING} "
            f"mirror={self.L_MIRROR} attn={self.L_ATTN} pres={self.L_PRES}")

    # -- the objective --------------------------------------------------------
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
        self._net().set_modality(mod_ids)

        tr_on = self.current_epoch >= int(self.TR_WARMUP_FRAC * self.num_epochs)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type) if self.device.type == "cuda" else dummy_context()

        # Every step must report the same keys -- collate_outputs stacks them.
        stats: Dict[str, float] = {"align_n": 0.0, "dyn_blocks": 0.0,
                                   "ring_n": 0.0, "mirror_n": 0.0,
                                   "attn_q": 0.0, "nonfinite": 0.0,
                                   "skipped": 0.0}
        z = lambda: torch.zeros((), device=self.device)
        l_align = l_dyn = l_probe = l_vc = l_ring = l_mirror = l_attn = l_pres = None

        try:
            with ctx:
                out = self.network(data)
                seg = out["seg"]
                state = out["state"].float()
                grid = out["grid"]

                l_seg = self.loss(seg, target)

                ref = target[0] if isinstance(target, list) else target
                tok = token_labels(ref, grid)                   # (B, L)

                # Every state loss runs OUTSIDE autocast. `.float()` on the
                # inputs is not sufficient -- autocast forces matmul to fp16
                # regardless, which is what made the covariance overflow and
                # took the first run to NaN at epoch 2.
                with fp32(self.device.type):
                    l_align, s = align_loss(state, tok, mod_ids, self.bank)
                    stats.update(s)
                    l_dyn, s = dynamics_loss(out["dyn"])
                    l_dyn = l_dyn.to(self.device)
                    stats.update(s)
                    l_probe = probe_loss(state, tok, self._net().state_probe)
                    l_vc = variance_covariance_loss(state)
                    l_ring, s = ring_loss(state, tok)
                    stats.update(s)
                    l_mirror, s = mirror_loss(state, tok)
                    stats.update(s)

                    l_attn, l_pres = z(), z()
                    if tr_on and out["tr"] is not None:
                        l_attn, s = attention_alignment_loss(
                            out["tr"]["attn"].float(), tok)
                        stats.update(s)
                        l_pres = presence_loss(out["tr"]["presence"].float(), tok)

                # A single non-finite auxiliary term would otherwise poison
                # every parameter through the shared backward. Drop it, count
                # it, and keep training -- but make it visible, because a
                # nonzero count means something upstream is still wrong.
                aux = {"l_align": l_align, "l_dyn": l_dyn, "l_probe": l_probe,
                       "l_vc": l_vc, "l_ring": l_ring, "l_mirror": l_mirror,
                       "l_attn": l_attn, "l_pres": l_pres}
                n_bad = 0
                for k, v in aux.items():
                    if not torch.isfinite(v):
                        aux[k] = z()
                        n_bad += 1
                stats["nonfinite"] = float(n_bad)
                (l_align, l_dyn, l_probe, l_vc, l_ring, l_mirror,
                 l_attn, l_pres) = (aux["l_align"], aux["l_dyn"], aux["l_probe"],
                                    aux["l_vc"], aux["l_ring"], aux["l_mirror"],
                                    aux["l_attn"], aux["l_pres"])

                loss = (l_seg
                        + self.L_ALIGN * l_align
                        + self.L_DYN * l_dyn
                        + self.L_PROBE * l_probe
                        + self.L_VC * l_vc
                        + self.L_RING * l_ring
                        + self.L_MIRROR * l_mirror
                        + self.L_ATTN * l_attn
                        + self.L_PRES * l_pres)

            # Skip the step if ANY gradient is non-finite, before clipping.
            # clip_grad_norm_ computes clip_coef = max_norm / total_norm; with
            # total_norm = inf that is 0, and inf * 0 = NaN -- so clipping
            # actively converts one overflowed gradient into NaN across every
            # parameter. That is how a single bad step poisoned the whole
            # network in the 2026-08-29 run. Check first, skip, and count.
            def _grads_finite():
                for p_ in self.network.parameters():
                    if p_.grad is not None and not torch.isfinite(p_.grad).all():
                        return False
                return True

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                if _grads_finite():
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.grad_scaler.step(self.optimizer)
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    stats["skipped"] = 1.0
                self.grad_scaler.update()
            else:
                loss.backward()
                if _grads_finite():
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.optimizer.step()
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    stats["skipped"] = 1.0
        finally:
            self._net().set_modality(None)

        res = {"loss": loss.detach().cpu().numpy(),
               "l_seg": l_seg.detach().cpu().numpy(),
               "l_align": l_align.detach().cpu().numpy(),
               "l_dyn": l_dyn.detach().cpu().numpy(),
               "l_probe": l_probe.detach().cpu().numpy(),
               "l_vc": l_vc.detach().cpu().numpy(),
               "l_ring": l_ring.detach().cpu().numpy(),
               "l_mirror": l_mirror.detach().cpu().numpy(),
               "l_attn": l_attn.detach().cpu().numpy(),
               "l_pres": l_pres.detach().cpu().numpy()}
        res.update(stats)
        return res

    def on_train_epoch_end(self, train_outputs: List[dict]):
        # nnU-Net only prints the aggregate loss, which makes a nine-term
        # objective impossible to debug: a term that quietly dominates or dies
        # is invisible until the run is over. Print the balance every epoch.
        super().on_train_epoch_end(train_outputs)
        keys = ["l_seg", "l_align", "l_dyn", "l_probe", "l_vc", "l_ring",
                "l_mirror", "l_attn", "l_pres"]
        w = {"l_seg": 1.0, "l_align": self.L_ALIGN, "l_dyn": self.L_DYN,
             "l_probe": self.L_PROBE, "l_vc": self.L_VC, "l_ring": self.L_RING,
             "l_mirror": self.L_MIRROR, "l_attn": self.L_ATTN,
             "l_pres": self.L_PRES}
        parts = []
        for k in keys:
            v = [float(o[k]) for o in train_outputs if k in o]
            if v:
                m = float(np.mean(v))
                parts.append(f"{k[2:]} {m:.3f}({w[k]*m:.3f})")
        n = [f"{k} {np.mean([float(o[k]) for o in train_outputs if k in o]):.2f}"
             for k in ("align_n", "ring_n", "mirror_n", "attn_q", "nonfinite", "skipped")
             if any(k in o for o in train_outputs)]
        self.print_to_log_file("[M3CoW] raw(weighted): " + "  ".join(parts))
        self.print_to_log_file("[M3CoW] counts: " + "  ".join(n))

    def validation_step(self, batch: dict) -> dict:
        # network.eval() is already set by on_validation_epoch_start, so the
        # forward returns plain logits and the base implementation applies.
        self._net().set_modality(
            modality_ids_from_keys(batch.get("keys", []), self.device))
        try:
            return super().validation_step(batch)
        finally:
            self._net().set_modality(None)


# ── Ablations. These ARE the experiment. ─────────────────────────────────────

class nnUNetTrainerM3CoW_NoStateLoss(nnUNetTrainerM3CoW):
    """
    Identical architecture, Dice/CE only.

    The direct test of the central claim: if this matches the full model, the
    state-space supervision is doing nothing and the gain is architectural. It
    is the control that the original MF-SSM study lacked.
    """
    L_ALIGN = L_DYN = L_PROBE = L_VC = L_RING = L_MIRROR = 0.0
    L_ATTN = L_PRES = 0.0


class nnUNetTrainerM3CoW_NoComplex(nnUNetTrainerM3CoW):
    """Real diagonal A (phase frozen at zero). Isolates the rotational state."""
    COMPLEX_STATE = False
    # Ring and mirror geometry are the terms rotation exists to satisfy; leaving
    # them on against a frozen phase would measure how badly a line fails to be
    # a circle, which is a different question. Kept on deliberately -- that IS
    # the question -- but reported separately from the no-phase-no-geometry run.


class nnUNetTrainerM3CoW_NoComplexNoGeom(nnUNetTrainerM3CoW_NoComplex):
    """Real diagonal AND no ring/mirror terms: the closest thing to Mamba-2."""
    L_RING = L_MIRROR = 0.0


class nnUNetTrainerM3CoW_NoTrapezoid(nnUNetTrainerM3CoW):
    """Euler/ZOH input rule. Isolates the second-order discretisation."""
    TRAPEZOIDAL = False


class nnUNetTrainerM3CoW_NoAlign(nnUNetTrainerM3CoW):
    """No cross-modal alignment. Predicts: transfer collapses, in-domain holds."""
    L_ALIGN = 0.0


class nnUNetTrainerM3CoW_NoDyn(nnUNetTrainerM3CoW):
    """
    No realised-dynamics constraint, so `share_A` reverts to nominal.
    Quantifies how much of the factorisation's value was never actually enforced
    in the original MF-SSM.
    """
    L_DYN = 0.0


class nnUNetTrainerM3CoW_NoTransformer(nnUNetTrainerM3CoW):
    """State not used as keys. Isolates the query decoder from the state losses."""
    USE_TRANSFORMER = False
    L_ATTN = L_PRES = 0.0


class nnUNetTrainerM3CoW_2epochs(nnUNetTrainerM3CoW):
    """Integration smoke test. Explicit signature required (see CMAP notes)."""
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 2
        self.TR_WARMUP_FRAC = 0.0


# ── Single-modality training + few-shot adaptation ──────────────────────────

class nnUNetTrainerM3CoW_CTonly(nnUNetTrainerM3CoW):
    """
    CTA only, on Dataset104's plans unchanged.

    Filtering the split rather than switching to Dataset102 keeps patch size,
    spacing, architecture and held-out subjects byte-identical to every other
    run, so the only variable is which modality the model saw. B^MR and dt^MR
    receive no gradient and stay at initialisation -- which is the starting
    point for the adaptation study.

    Note L_ALIGN and L_DYN are inert here: both compare two modalities and only
    one is present. They are left switched on so the config matches the joint
    runs; the reported values will be exactly zero.
    """
    TRAIN_MODALITY = "_ct_"

    def do_split(self):
        tr, val = super().do_split()
        tr_f = [k for k in tr if self.TRAIN_MODALITY in k]
        val_f = [k for k in val if self.TRAIN_MODALITY in k] or val
        self.print_to_log_file(
            f"[M3CoW] single-modality '{self.TRAIN_MODALITY}': "
            f"train {len(tr)}->{len(tr_f)}, val {len(val)}->{len(val_f)}")
        return tr_f, val_f


class nnUNetTrainerM3CoW_MRonly(nnUNetTrainerM3CoW_CTonly):
    """MRA only. The harder direction: plain nnU-Net MR->CTA collapses to 0.002."""
    TRAIN_MODALITY = "_mr_"


class nnUNetTrainerM3CoW_Adapt(nnUNetTrainerM3CoW):
    """
    Adapt across modality by re-fitting ONLY the measurement operator: B^target
    and dt^target. A, C, the encoder, the decoder and the anatomy transformer
    stay frozen.

    This is the claim the state losses license and the one MF-SSM could not
    make: with L_align and L_dyn, the shared state is trained to be
    modality-independent, so a new instrument should need only a new way of
    writing into it. Without them the same adapter gained 0.0000 Dice.

    ADAPTER_ONLY=False fine-tunes all ~34M parameters from the same checkpoint
    on the same n scans -- the fair comparison, isolating parameter efficiency
    from initialisation and data.
    """
    SOURCE_TRAINER: str = "nnUNetTrainerM3CoW_CTonly"
    TARGET_MODALITY: str = "_mr_"
    N_SHOT: int = 5
    ADAPTER_ONLY: bool = True

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 50

    def do_split(self):
        tr, val = super().do_split()
        tgt = sorted([k for k in tr if self.TARGET_MODALITY in k])
        tr_f = tgt[:self.N_SHOT]
        val_f = [k for k in val if self.TARGET_MODALITY in k] or val
        self.print_to_log_file(
            f"[ADAPT] n_shot={self.N_SHOT} target='{self.TARGET_MODALITY}' "
            f"train={tr_f} val={len(val_f)}")
        return tr_f, val_f

    def initialize(self):
        super().initialize()
        ck = os.path.normpath(os.path.join(
            self.output_folder_base, "..",
            f"{self.SOURCE_TRAINER}__{self.plans_manager.plans_name}__{self.configuration_name}",
            f"fold_{self.fold}", "checkpoint_final.pth"))
        sd = torch.load(ck, map_location=self.device, weights_only=False)["network_weights"]
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = self._net().load_state_dict(sd, strict=False)
        self.print_to_log_file(
            f"[ADAPT] loaded {ck} ({len(missing)} missing, {len(unexpected)} unexpected)")

        if not self.ADAPTER_ONLY:
            n_all = sum(p.numel() for p in self._net().parameters())
            self.print_to_log_file(f"[ADAPT] full fine-tune: {n_all:,} trainable")
            return

        tgt_idx = MR if self.TARGET_MODALITY == "_mr_" else CT
        trainable = []
        for p in self._net().parameters():
            p.requires_grad_(False)
        for blk in self._net().blocks:
            for mod in (blk.ssm.x_proj_B[tgt_idx], blk.ssm.x_proj_dt[tgt_idx],
                        blk.ssm.dt_proj[tgt_idx]):
                for p in mod.parameters():
                    p.requires_grad_(True)
                    trainable.append(p)
        n_tr = sum(p.numel() for p in trainable)
        n_all = sum(p.numel() for p in self._net().parameters())
        self.print_to_log_file(
            f"[ADAPT] adapter-only: {n_tr:,} of {n_all:,} ({100*n_tr/n_all:.3f}%)")
        self.optimizer = torch.optim.SGD(trainable, lr=self.initial_lr,
                                         weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr,
                                            self.num_epochs)


def _mk(n, adapter):
    name = f"nnUNetTrainerM3CoW_Adapt_n{n}" + ("" if adapter else "_full")
    return name, type(name, (nnUNetTrainerM3CoW_Adapt,),
                      {"N_SHOT": n, "ADAPTER_ONLY": adapter})


for _n in (1, 2, 5, 10):
    for _a in (True, False):
        _nm, _cls = _mk(_n, _a)
        globals()[_nm] = _cls


# ---------------------------------------------------------------------------
# Subject-disjoint modality split
# ---------------------------------------------------------------------------
# TopCoW is FULLY paired: topcow_ct_i and topcow_mr_i are the same patient, and
# all 100 training subjects appear in both modalities. So `L_align` learns to
# match "CT anatomy" to "MR anatomy" when those are literally the same people.
# That is easier than deployment, where CT and MR come from different cohorts,
# and it is the most obvious reviewer attack on the modality-blind result.
#
# `_Unpaired` removes the pairing: every training subject contributes exactly
# ONE modality, so the model never sees the same head twice. Validation is left
# untouched, so its numbers compare directly against the full model.
#
# The unavoidable confound: unpaired means at most one volume per subject, so
# training data halves (100 volumes instead of 200). `_PairedHalf` is the
# size-matched control -- also 100 volumes, but 50 subjects x BOTH modalities.
# Comparing the two isolates pairing from data quantity:
#
#     full model     200 vol, 100 subj, paired
#     _PairedHalf    100 vol,  50 subj, paired      <- controls for data size
#     _Unpaired      100 vol, 100 subj, UNPAIRED    <- the question
class nnUNetTrainerM3CoW_Unpaired(nnUNetTrainerM3CoW):
    """One modality per training subject; no patient seen in both."""

    SPLIT_SEED: int = 1337

    def _partition(self):
        tr, val = nnUNetTrainer.do_split(self)
        subs = sorted({k.split("_")[-1] for k in tr})
        rng = np.random.RandomState(self.SPLIT_SEED)
        order = list(subs)
        rng.shuffle(order)
        return tr, val, order

    def do_split(self):
        tr, val, order = self._partition()
        half = len(order) // 2
        want = {s: "ct" for s in order[:half]}
        want.update({s: "mr" for s in order[half:]})
        keep = [k for k in tr if f"_{want[k.split('_')[-1]]}_" in k]
        self._log_split(keep, val, tr)
        return keep, val

    def _log_split(self, keep, val, tr):
        ct = [k for k in keep if "_ct_" in k]
        mr = [k for k in keep if "_mr_" in k]
        subs = {k.split("_")[-1] for k in keep}
        both = {s for s in subs
                if any(f"_ct_" in k and k.split("_")[-1] == s for k in keep)
                and any(f"_mr_" in k and k.split("_")[-1] == s for k in keep)}
        self.print_to_log_file(
            f"[SPLIT] {type(self).__name__}: train {len(keep)} of {len(tr)} vol "
            f"(CT {len(ct)} / MR {len(mr)}), {len(subs)} subjects, "
            f"{len(both)} in BOTH modalities; val {len(val)} untouched")
        assert not (set(keep) & set(val)), "train/val overlap"


class nnUNetTrainerM3CoW_PairedHalf(nnUNetTrainerM3CoW_Unpaired):
    """Size-matched control: half the subjects, but BOTH modalities each."""

    def do_split(self):
        tr, val, order = self._partition()
        keep_subs = set(order[: len(order) // 2])
        keep = [k for k in tr if k.split("_")[-1] in keep_subs]
        self._log_split(keep, val, tr)
        return keep, val


class nnUNetTrainerM3CoW_NoComplexNoState(nnUNetTrainerM3CoW_NoComplex):
    """
    Phase frozen AND every state loss off: Dice/CE only on a real-diagonal SSM.

    Diagnostic, not an ablation for the paper. Both phase-frozen runs so far
    diverged (_NoComplex at epoch 9, _NoComplexNoGeom at epoch 20) and there are
    two incompatible readings: either the state objectives are genuinely
    unsatisfiable without rotation -- which is a result -- or the complex=False
    forward/backward path is numerically broken, which is a bug we must fix
    before claiming anything.

    This run separates them. Dice/CE cannot be "unsatisfiable", so:
        trains fine  -> the divergence came from the state losses     -> (A) science
        also diverges -> the real-diagonal path itself is broken      -> (B) bug

    The answer shows up within ~30 epochs, so this does not need 1000.
    """
    L_ALIGN = L_DYN = L_PROBE = L_VC = L_RING = L_MIRROR = 0.0
    L_ATTN = L_PRES = 0.0


class nnUNetTrainerM3CoW_NoGeom(nnUNetTrainerM3CoW):
    """
    Rotation KEPT, ring/mirror dropped. The arm that was missing.

    _NoComplexNoGeom moves two variables at once relative to the full model
    (phase off AND geometry off), so it cannot attribute the rotation on its own.
    This is the midpoint that makes the chain single-variable:

        full          phase on,  ring/mirror on
        _NoGeom       phase on,  ring/mirror OFF   <- full vs this = ring/mirror
        _NoComplexNoGeom phase OFF, ring/mirror off <- _NoGeom vs this = phase

    Also the only way to tell whether ring/mirror earn their weight, or whether
    align/dyn/probe/vc were carrying the +0.0356 on their own.
    """
    L_RING = 0.0
    L_MIRROR = 0.0


class nnUNetTrainerM3CoW_ThetaLow(nnUNetTrainerM3CoW):
    """
    S4D-Lin frequency bank rescaled to the band the model actually converges to.

    Inspecting the trained full model showed A_theta had been driven from its
    init [0, 3.14, ..., 21.99] down to [-0.01, 0.09, ..., 0.51] -- a mean drift
    of 10.76. The default init is above the Nyquist guard: theta up to 22 with
    dt up to 0.5 gives dt*theta ~ 11 rad, and `dt_the = pi*tanh(dt_the/pi)`
    saturates, so the top channels start aliased and contribute nothing until
    training walks them down.

    theta_scale 0.05 puts the bank at [0, 0.16, ..., 1.10]: the right regime from
    step 0, still ~2x above the observed optimum so there is room to explore, and
    far below the aliasing ceiling. Tests whether the walk-down was costing us
    accuracy or merely time.
    """
    THETA_SCALE = 0.05
