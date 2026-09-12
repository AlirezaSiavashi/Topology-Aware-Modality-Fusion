"""
nnUNetTrainerMFSSM.py
=====================
Modality-Factorised SSM trainer.

    A   state dynamics    SHARED           (one anatomy)
    B   measurement -> state   PER MODALITY (CT and TOF invert differently)
    C   state -> vessel class  SHARED       (one taxonomy)
    dt  timescale         PER MODALITY      (spacing, noise)

What this is being tested against
---------------------------------
On Dataset104 the *joint-training* disparity (CTA 0.646 vs MRA 0.762) proved
immovable: four interventions and a preprocessing change all landed inside the
measured seed-noise floor of 0.013. That disparity is bounded by image content
-- vessel/background ambiguity is 13.9% in CTA vs 0.10% in MRA and is provably
invariant to any monotone preprocessing.

The *cross-modality transfer* gap is a different and much larger target:
CT-trained -> MRA loses 0.312 Dice, MR-trained -> CTA loses 0.644 (collapse to
0.002). That is 25-50x the noise floor and untouched by anything tried so far.
This trainer targets that, not the joint disparity.

The control that matters is not plain nnU-Net -- a joint nnU-Net already sees
both modalities. It is `_SharedAll` below: an identical SSM with no
factorisation. The claim is that STRUCTURED sharing (A, C tied; B, dt free)
beats unstructured sharing, and `_SeparateAll` bounds the other end by giving
the modalities nothing in common.

Run
---
  nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerMFSSM
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from mfssm import HAVE_CUDA_SCAN
from mfssm_net import MFSSMNet

CT, MR = 0, 1


def modality_ids_from_keys(keys: List[str], device: torch.device) -> torch.Tensor:
    ids = []
    for key in keys:
        k = str(key).lower()
        ids.append(MR if ("_mr_" in k or "_mra_" in k or k.endswith("_mr")) else CT)
    return torch.tensor(ids, dtype=torch.long, device=device)


class nnUNetTrainerMFSSM(nnUNetTrainer):

    N_STAGES_TO_WRAP: int = 2
    D_STATE: int = 16
    EXPAND: int = 2
    SHARE_A: bool = True
    SHARE_C: bool = True
    FORCE_REF_SCAN: bool = False

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)

    # Hooks that replace module outputs do not survive torch.compile cleanly.
    def _do_i_compile(self) -> bool:
        return False

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True) -> nn.Module:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        cls = nnUNetTrainerMFSSM
        backbone = get_network_from_plans(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, allow_init=True,
            deep_supervision=enable_deep_supervision)
        return MFSSMNet(
            backbone=backbone,
            stage_channels=arch_init_kwargs["features_per_stage"],
            n_stages_to_wrap=cls.N_STAGES_TO_WRAP,
            d_state=cls.D_STATE, expand=cls.EXPAND, n_modalities=2,
            share_A=cls.SHARE_A, share_C=cls.SHARE_C,
            force_ref_scan=cls.FORCE_REF_SCAN)

    def _net(self) -> MFSSMNet:
        n = self.network
        return n.module if hasattr(n, "module") else n

    def initialize(self):
        super().initialize()
        net = self._net()
        extra = sum(p.numel() for p in net.blocks.parameters())
        total = sum(p.numel() for p in net.parameters())
        self.print_to_log_file(
            f"[MFSSM] stages wrapped={net.wrapped_idx} d_state={self.D_STATE} "
            f"expand={self.EXPAND} share_A={self.SHARE_A} share_C={self.SHARE_C} "
            f"cuda_scan={HAVE_CUDA_SCAN and not self.FORCE_REF_SCAN}")
        self.print_to_log_file(
            f"[MFSSM] SSM params {extra/1e6:.2f}M of {total/1e6:.2f}M total "
            f"({100*extra/total:.1f}%)")

    def train_step(self, batch: dict) -> dict:
        self._net().set_modality(
            modality_ids_from_keys(batch.get("keys", []), self.device))
        try:
            return super().train_step(batch)
        finally:
            self._net().set_modality(None)

    def validation_step(self, batch: dict) -> dict:
        self._net().set_modality(
            modality_ids_from_keys(batch.get("keys", []), self.device))
        try:
            return super().validation_step(batch)
        finally:
            self._net().set_modality(None)


# ── Ablations: these are the experiment, not extras ──────────────────────────

class nnUNetTrainerMFSSM_SharedAll(nnUNetTrainerMFSSM):
    """
    Unstructured sharing: one SSM, no modality factorisation at all.

    This is the primary control. It isolates the factorisation from the mere
    presence of a state-space block -- without it, any gain could be attributed
    to added capacity and long-range scanning.
    """
    SHARE_A = True
    SHARE_C = True

    @staticmethod
    def build_network_architecture(*a, **kw):
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        (acn, aik, aikri, nic, noc) = a[:5]
        eds = kw.get("enable_deep_supervision", a[5] if len(a) > 5 else True)
        backbone = get_network_from_plans(acn, aik, aikri, nic, noc,
                                          allow_init=True, deep_supervision=eds)
        return MFSSMNet(backbone=backbone,
                        stage_channels=aik["features_per_stage"],
                        n_stages_to_wrap=nnUNetTrainerMFSSM.N_STAGES_TO_WRAP,
                        d_state=nnUNetTrainerMFSSM.D_STATE,
                        expand=nnUNetTrainerMFSSM.EXPAND,
                        n_modalities=1,          # <- one modality => no factorisation
                        share_A=True, share_C=True,
                        force_ref_scan=nnUNetTrainerMFSSM.FORCE_REF_SCAN)

    def train_step(self, batch):
        return nnUNetTrainer.train_step(self, batch)

    def validation_step(self, batch):
        return nnUNetTrainer.validation_step(self, batch)


class nnUNetTrainerMFSSM_SeparateAll(nnUNetTrainerMFSSM):
    """
    Nothing shared: A, B, C and dt all modality-specific.

    Bounds the other end. If this matches the factorised model, the shared
    latent state is doing no work and the result is just two encoders.
    """
    SHARE_A = False
    SHARE_C = False


class nnUNetTrainerMFSSM_ShareAOnly(nnUNetTrainerMFSSM):
    """Shared dynamics, per-modality readout. Tests whether tying C matters."""
    SHARE_A = True
    SHARE_C = False


class nnUNetTrainerMFSSM_2epochs(nnUNetTrainerMFSSM):
    """Integration smoke test. Explicit signature is required (see CMAP notes)."""
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 2


# ── Single-modality training, for the transfer / few-shot adaptation study ───

class nnUNetTrainerMFSSM_CTonly(nnUNetTrainerMFSSM):
    """
    Train on CTA only, using Dataset104's plans unchanged.

    Filtering the split rather than training on Dataset102 keeps patch size
    ([112,160,128] vs DS102's [128,160,112]), spacing, architecture and the
    held-out subjects byte-identical to every other run, so the only variable
    is which modality the model saw. It also leaves the CT/MR held-out sets
    exactly as used everywhere else in this study.

    Because no MRA sample is ever seen, B^MR and dt^MR receive no gradient and
    stay at initialisation -- verified in test_mfssm.py ("CT-only batch leaves
    B[MR] untouched"). That is precisely the starting point for adapting to
    MRA by fitting only those 37K parameters (0.1% of the network).
    """
    TRAIN_MODALITY = "_ct_"

    def do_split(self):
        tr, val = super().do_split()
        tr_f = [k for k in tr if self.TRAIN_MODALITY in k]
        val_f = [k for k in val if self.TRAIN_MODALITY in k] or val
        self.print_to_log_file(
            f"[MFSSM] single-modality split '{self.TRAIN_MODALITY}': "
            f"train {len(tr)}->{len(tr_f)}, val {len(val)}->{len(val_f)}")
        return tr_f, val_f


class nnUNetTrainerMFSSM_MRonly(nnUNetTrainerMFSSM_CTonly):
    """Train on MRA only. The harder transfer direction: plain nnU-Net trained
    MR-only collapses to 0.002 Dice on CTA."""
    TRAIN_MODALITY = "_mr_"


# ── Few-shot modality adaptation ─────────────────────────────────────────────

class nnUNetTrainerMFSSM_Adapt(nnUNetTrainerMFSSM):
    """
    Adapt a CTA-trained model to MRA by re-fitting ONLY the measurement
    operator: B^MR and dt^MR (37K parameters, 0.1% of the network). A, C, the
    U-Net encoder and the decoder stay frozen.

    This is the claim the state-overlap diagnostic licenses. Within the shared
    state, vessel identity explains more of the distance structure than
    modality does (R = 0.56), so A and C encode anatomy in a form that should
    not need relearning for a new instrument -- only the operator that writes
    that instrument's measurements into the state.

    ADAPTER_ONLY=False fine-tunes the whole 32M-parameter network from the same
    checkpoint. That is the fair comparison: both start identically and see the
    same n scans, so any difference is parameter efficiency, not initialisation
    or data. At n = 1-5 volumes the expectation is that 32M parameters overfit
    where 37K cannot.
    """
    SOURCE_TRAINER: str = "nnUNetTrainerMFSSM_CTonly"
    TARGET_MODALITY: str = "_mr_"
    N_SHOT: int = 5
    ADAPTER_ONLY: bool = True

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 50          # 37K params converge fast

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
        import os
        ck = os.path.join(self.output_folder_base, "..",
                          f"{self.SOURCE_TRAINER}__{self.plans_manager.plans_name}__{self.configuration_name}",
                          f"fold_{self.fold}", "checkpoint_final.pth")
        ck = os.path.normpath(ck)
        sd = torch.load(ck, map_location=self.device, weights_only=False)["network_weights"]
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = self._net().load_state_dict(sd, strict=False)
        self.print_to_log_file(f"[ADAPT] loaded {ck} ({len(missing)} missing, {len(unexpected)} unexpected)")

        if self.ADAPTER_ONLY:
            tgt_idx = 1 if self.TARGET_MODALITY == "_mr_" else 0
            trainable = []
            for p in self._net().parameters():
                p.requires_grad_(False)
            for blk in self._net().blocks:
                for mod in (blk.ssm.x_proj_B[tgt_idx], blk.ssm.x_proj_dt[tgt_idx],
                            blk.ssm.dt_proj[tgt_idx]):
                    for p in mod.parameters():
                        p.requires_grad_(True); trainable.append(p)
            n_tr = sum(p.numel() for p in trainable)
            n_all = sum(p.numel() for p in self._net().parameters())
            self.print_to_log_file(
                f"[ADAPT] adapter-only: {n_tr:,} trainable of {n_all:,} "
                f"({100*n_tr/n_all:.3f}%)")
            # rebuild optimiser over the trainable subset only
            self.optimizer = torch.optim.SGD(trainable, lr=self.initial_lr,
                                             weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)
            from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
            self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr,
                                                self.num_epochs)
        else:
            n_all = sum(p.numel() for p in self._net().parameters())
            self.print_to_log_file(f"[ADAPT] full fine-tune: {n_all:,} trainable (100%)")


def _mk(n, adapter):
    name = f"nnUNetTrainerMFSSM_Adapt_n{n}" + ("" if adapter else "_full")
    return name, type(name, (nnUNetTrainerMFSSM_Adapt,),
                      {"N_SHOT": n, "ADAPTER_ONLY": adapter})

for _n in (1, 2, 5, 10):
    for _a in (True, False):
        _nm, _cls = _mk(_n, _a)
        globals()[_nm] = _cls


class nnUNetTrainerPlain_MRonly(nnUNetTrainer):
    """
    Plain nnU-Net trained on MRA only, on Dataset104's plans.

    The missing control for the one above-noise positive in this study:
    MF-SSM MR-only reaches 0.786 on MRA versus 0.762 for joint nnU-Net. Without
    this run that gap cannot be attributed -- it may simply be that avoiding
    joint-training interference helps ANY architecture, with the SSM
    contributing nothing.
    """
    def do_split(self):
        tr, val = super().do_split()
        tr_f = [k for k in tr if "_mr_" in k]
        val_f = [k for k in val if "_mr_" in k] or val
        self.print_to_log_file(f"[CTRL] MR-only: train {len(tr)}->{len(tr_f)}, val {len(val)}->{len(val_f)}")
        return tr_f, val_f


class nnUNetTrainerPlain_CTonly(nnUNetTrainerPlain_MRonly):
    """Plain nnU-Net, CTA only. Matched control for MF-SSM CT-only (0.645)."""
    def do_split(self):
        tr, val = nnUNetTrainer.do_split(self)
        tr_f = [k for k in tr if "_ct_" in k]
        val_f = [k for k in val if "_ct_" in k] or val
        self.print_to_log_file(f"[CTRL] CT-only: train {len(tr)}->{len(tr_f)}, val {len(val)}->{len(val_f)}")
        return tr_f, val_f
