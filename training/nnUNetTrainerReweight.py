"""
nnUNetTrainerReweight.py
========================
Per-class x per-modality loss reweighting -- the simple control.

This is deliberately NOT presented as a contribution. Per-class loss weighting
(inverse frequency, focal, class-balanced) and per-modality loss balancing
(GradNorm, uncertainty weighting, OGM-GE) are both standard; their product is
an obvious extension. It is here for two reasons:

1. It closes an obvious rejection path. Any reviewer will ask whether a simple
   reweighting achieves what the prior-based method was built for.

2. It is diagnostic. The measured CTA/MRA gap on TopCoW is 0.115 macro, but
   ranges from -0.205 (R-Pcom) to +0.012 (Acom). Two explanations are
   consistent with that:

     (a) optimisation neglect -- the model *could* do better on CTA R-Pcom but
         gains more total loss reduction elsewhere, so it does not bother.
         Reweighting should then close the gap at no cost to the mean.

     (b) information limit -- CTA genuinely carries less signal for small
         communicating arteries (bone at the skull base, 1-2 mm calibre, veins
         also opacified). Reweighting can then only trade: the gap narrows and
         the mean falls.

   The Euclidean control already hints at (b): it narrowed the gap 0.115 ->
   0.096 but lowered CTA by 0.030 AND MRA by 0.050, i.e. levelled down. This
   trainer logs mean-vs-gap explicitly so (a) and (b) can be told apart rather
   than read off a single headline number.

Weighting rule
--------------
    w[m,k] proportional to (1 - Dice[m,k]),  normalised to mean 1 per modality,
    clipped to [W_MIN, W_MAX] for stability.

Weights are applied to the cross-entropy term only (per-class weights are
natively supported there); the Dice term is left alone. Batches are split by
modality so each half is scored with its own weight vector.

Run
---
  nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerReweight
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn

CT, MR = 0, 1


def modality_ids_from_keys(keys: List[str], device: torch.device) -> torch.Tensor:
    ids = []
    for key in keys:
        k = str(key).lower()
        ids.append(MR if ("_mr_" in k or "_mra_" in k or k.endswith("_mr")) else CT)
    return torch.tensor(ids, dtype=torch.long, device=device)


class nnUNetTrainerReweight(nnUNetTrainer):

    W_MIN: float = 0.5
    W_MAX: float = 3.0
    MOMENTUM: float = 0.5      # EMA on the weights, as Dice is noisy per epoch

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self._w: Optional[torch.Tensor] = None       # (2, K)
        self._class_names: List[str] = []
        self._loss_mod = {}

    def initialize(self):
        super().initialize()
        k = self.label_manager.num_segmentation_heads
        self._w = torch.ones(2, k, device=self.device)
        self._class_names = [str(i) for i in range(k)]
        for name, val in self.dataset_json["labels"].items():
            v = int(val)
            if 0 <= v < k:
                self._class_names[v] = str(name)
        # One loss object per modality; only their CE weight vectors differ.
        for m in (CT, MR):
            self._loss_mod[m] = self._build_loss()
        self.print_to_log_file(
            f"[RW] per-class x per-modality CE reweighting, K={k}, "
            f"clip=[{self.W_MIN},{self.W_MAX}], ema={self.MOMENTUM}")

    # ── weighted loss ────────────────────────────────────────────────────
    def _set_ce_weight(self, m: int) -> None:
        loss = self._loss_mod[m]
        inner = loss.loss if isinstance(loss, DeepSupervisionWrapper) else loss
        inner.ce.weight = self._w[m].detach()

    def _weighted_loss(self, output, target, mod_ids):
        total = None
        n = 0
        for m in (CT, MR):
            sel = mod_ids == m
            cnt = int(sel.sum())
            if cnt == 0:
                continue
            self._set_ce_weight(m)
            out_m = [o[sel] for o in output] if isinstance(output, list) else output[sel]
            tgt_m = [t[sel] for t in target] if isinstance(target, list) else target[sel]
            l = self._loss_mod[m](out_m, tgt_m) * cnt
            total = l if total is None else total + l
            n += cnt
        return total / max(n, 1)

    # ── train / validation steps ─────────────────────────────────────────
    def train_step(self, batch: dict) -> dict:
        from contextlib import contextmanager

        from torch.amp import autocast

        @contextmanager
        def dummy():
            yield

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([t.to(self.device, non_blocking=True) for t in target]
                  if isinstance(target, list)
                  else target.to(self.device, non_blocking=True))
        mod = modality_ids_from_keys(batch.get("keys", []), self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type) if self.device.type == "cuda" else dummy():
            out = self.network(data)
            loss = self._weighted_loss(out, target, mod)

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
        return {"loss": loss.detach().cpu().numpy()}

    @staticmethod
    def _hard_counts(output, tgt, lm):
        axes = [0] + list(range(2, output.ndim))
        if lm.has_regions:
            pred = (torch.sigmoid(output) > 0.5).long()
        else:
            seg = output.argmax(1)[:, None]
            pred = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            pred.scatter_(1, seg, 1)
        mask = None
        if lm.has_ignore_label and not lm.has_regions:
            mask = (tgt != lm.ignore_label).float()
            tgt = tgt.clone()
            tgt[tgt == lm.ignore_label] = 0
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred, tgt, axes=axes, mask=mask)
        tp, fp, fn = (x.detach().cpu().numpy() for x in (tp, fp, fn))
        return (tp[1:], fp[1:], fn[1:]) if not lm.has_regions else (tp, fp, fn)

    def validation_step(self, batch: dict) -> dict:
        from contextlib import contextmanager

        from torch.amp import autocast

        @contextmanager
        def dummy():
            yield

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([t.to(self.device, non_blocking=True) for t in target]
                  if isinstance(target, list)
                  else target.to(self.device, non_blocking=True))
        mod = modality_ids_from_keys(batch.get("keys", []), self.device)

        with autocast(self.device.type) if self.device.type == "cuda" else dummy():
            out = self.network(data)
            del data
            l = self._weighted_loss(out, target, mod)

        o = out[0] if self.enable_deep_supervision else out
        t = target[0] if self.enable_deep_supervision else target
        tp, fp, fn = self._hard_counts(o, t, self.label_manager)
        res = {"loss": l.detach().cpu().numpy(),
               "tp_hard": tp, "fp_hard": fp, "fn_hard": fn}
        zero = np.zeros_like(tp)
        for m, nm in ((CT, "ct"), (MR, "mr")):
            sel = mod == m
            if sel.any():
                a, b, c = self._hard_counts(o[sel], t[sel], self.label_manager)
            else:
                a, b, c = zero.copy(), zero.copy(), zero.copy()
            res[f"tp_{nm}"], res[f"fp_{nm}"], res[f"fn_{nm}"] = a, b, c
        return res

    # ── weight refresh ───────────────────────────────────────────────────
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        n_fg = self.label_manager.num_segmentation_heads - 1
        dice = {}
        for m, nm in ((CT, "ct"), (MR, "mr")):
            tp = np.zeros(n_fg); fp = np.zeros(n_fg); fn = np.zeros(n_fg)
            for o in val_outputs:
                tp = tp + o[f"tp_{nm}"]; fp = fp + o[f"fp_{nm}"]; fn = fn + o[f"fn_{nm}"]
            present = float((tp + fp + fn).sum()) > 0
            dice[m] = (2 * tp / np.clip(2 * tp + fp + fn, 1e-8, None)) if present else None

        if dice[CT] is None or dice[MR] is None:
            return

        new = torch.ones_like(self._w)
        for m in (CT, MR):
            err = 1.0 - np.clip(dice[m], 0.0, 1.0)          # weight ~ remaining error
            err = err / max(err.mean(), 1e-6)               # mean 1
            err = np.clip(err, self.W_MIN, self.W_MAX)
            new[m, 1:] = torch.from_numpy(err).float().to(self._w.device)
            new[m, 0] = 1.0                                  # background unweighted
        self._w.mul_(1 - self.MOMENTUM).add_(self.MOMENTUM * new)

        # Log mean and gap together: the whole point is to see whether the gap
        # closes because CTA improves, or because MRA is dragged down.
        mct, mmr = float(np.mean(dice[CT])), float(np.mean(dice[MR]))
        self.print_to_log_file(
            f"[RW] val CTA={mct:.4f} MRA={mmr:.4f} mean={(mct+mmr)/2:.4f} "
            f"gap={mmr-mct:+.4f}")
        self.print_to_log_file(
            "[RW] weights CT/MR -> " + ", ".join(
                f"{self._class_names[k]}:{self._w[CT,k]:.2f}/{self._w[MR,k]:.2f}"
                for k in range(1, n_fg + 1)))
