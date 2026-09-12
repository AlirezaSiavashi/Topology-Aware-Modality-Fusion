#!/usr/bin/env python3
"""
Is the M3-CoW state organised by anatomy, and is its geometry actually circular?

Extends evaluation/state_overlap.py with the two measurements the new losses
are supposed to move, so the claim is checkable rather than asserted:

  1. Linear-probe MODALITY accuracy on the state.
     ~0.50 -> states are modality-mixed (what L_align is for).
     ~1.00 -> the two modalities occupy disjoint regions and the "shared state
              space" is nominal only.

  2. Distance ratio R = d(same class, across modality) / d(diff class, within
     modality).
     R < 1 -> geometry dominated by ANATOMY (a CT ICA state sits nearer an MR
              ICA state than a CT MCA state). R > 1 -> dominated by MODALITY.
     MF-SSM measured R = 0.56 with no loss acting on it; L_align optimises it
     directly, so this is where the effect must show.

  3. RING error: mean |<P_u,P_v> - cos(phi_u - phi_v)| over the ten CoW ring
     classes. This is what L_ring minimises and what a real-diagonal state
     cannot minimise, so `_NoComplex` should stay high here while the full
     model drops.

  4. MIRROR spread: variance of P[R-x] - P[L-x] across the five mirror pairs.

  5. LINEAR DECODABILITY: 14-way class accuracy from a linear probe on the
     state. This is the precondition for the state to work as attention keys,
     so it should track how much the anatomy transformer contributes.

Usage:
  nnunet_venv/bin/python3.9 evaluation/state_geometry_m3.py \
      --trainer nnUNetTrainerM3CoW [--trainer nnUNetTrainerM3CoW_NoStateLoss ...]
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "training"))
from mfssm3_net import MFSSM3Net
from state_losses import COW_RING, COW_MIRROR, N_CLASS
from dynamic_network_architectures.architectures.unet import PlainConvUNet

DS = "Dataset104_TopCoW_joint"
PRE = f"{BASE}/preprocessed/nnUNet_preprocessed/{DS}"


def build(trainer, **kw):
    p = json.load(open(f"{PRE}/nnUNetPlans.json"))
    a = p["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    bb = PlainConvUNet(input_channels=1, n_stages=a["n_stages"],
        features_per_stage=a["features_per_stage"], conv_op=nn.Conv3d,
        kernel_sizes=a["kernel_sizes"], strides=a["strides"], num_classes=N_CLASS,
        n_conv_per_stage=a["n_conv_per_stage"],
        n_conv_per_stage_decoder=a["n_conv_per_stage_decoder"], conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=True)
    net = MFSSM3Net(bb, stage_channels=a["features_per_stage"], n_modalities=2, **kw)
    ck = (f"{BASE}/results/nnUNet/{DS}/{trainer}__nnUNetPlans__3d_fullres/"
          f"fold_0/checkpoint_final.pth")
    sd = torch.load(ck, map_location="cpu", weights_only=False)["network_weights"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"  loaded {trainer}: {len(missing)} missing, {len(unexpected)} unexpected")
    return net.cuda().eval(), p["configurations"]["3d_fullres"]["patch_size"]


@torch.no_grad()
def collect(net, patch, cases, mod_id, n_patch=14):
    """Sample token states + labels from the key-slot SSM block."""
    P = f"{PRE}/nnUNetPlans_3d_fullres"
    S, L = [], []
    for c in cases:
        d = np.load(f"{P}/{c}.npy", mmap_mode="r")
        s = np.load(f"{P}/{c}_seg.npy", mmap_mode="r")
        for _ in range(n_patch):
            fg = np.argwhere(np.asarray(s[0][::4, ::4, ::4]) > 0)
            if len(fg) == 0:
                continue
            ctr = fg[np.random.randint(len(fg))] * 4
            sl = tuple(slice(max(0, int(ctr[i]) - patch[i] // 2),
                             max(0, int(ctr[i]) - patch[i] // 2) + patch[i])
                       for i in range(3))
            x = np.asarray(d[(slice(None),) + sl], dtype=np.float32)
            y = np.asarray(s[(slice(None),) + sl])
            if x.shape[1:] != tuple(patch):
                continue
            net.set_modality(torch.tensor([mod_id]).cuda())
            with torch.autocast("cuda"):
                net(torch.from_numpy(x)[None].cuda())
            st = net.blocks[net.key_slot].last_state.float()[0]     # (L, C)
            grid = net._grids[net.key_slot]
            # Max-pool the one-hot: nearest-downsampling a 1-2 mm vessel onto an
            # 8 mm token grid returns background almost everywhere. -1 is
            # nnU-Net's ignore label outside the crop; one_hot rejects it.
            lab0 = torch.from_numpy(np.ascontiguousarray(y[0])).long().cuda().clamp_(0, N_CLASS - 1)
            oh = F.one_hot(lab0, N_CLASS).permute(3, 0, 1, 2).float()[None]
            oh = F.adaptive_max_pool3d(oh, grid)[0]
            fgmax, fgarg = oh[1:].max(0)
            lab = torch.where(fgmax.flatten() > 0, fgarg.flatten() + 1,
                              torch.zeros_like(fgarg.flatten())).long()
            keep = lab > 0
            if keep.sum() == 0:
                continue
            S.append(st[keep].cpu()); L.append(lab[keep].cpu())
    net.set_modality(None)
    return torch.cat(S), torch.cat(L)


def analyse(tag, net, patch, ct, mr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    Sc, Lc = collect(net, patch, ct, 0)
    Sm, Lm = collect(net, patch, mr, 1)
    X = torch.cat([Sc, Sm]).numpy()
    y = np.concatenate([Lc.numpy(), Lm.numpy()])
    m = np.concatenate([np.zeros(len(Sc)), np.ones(len(Sm))])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    idx = np.random.permutation(len(X))[:6000]
    acc_mod = cross_val_score(LogisticRegression(max_iter=300), X[idx], m[idx], cv=3).mean()
    acc_cls = cross_val_score(LogisticRegression(max_iter=300), X[idx], y[idx], cv=3).mean()

    Xc, Xm = X[m == 0], X[m == 1]
    cls = sorted(set(Lc.tolist()) & set(Lm.tolist()))
    cc = {k: Xc[Lc.numpy() == k].mean(0) for k in cls if (Lc.numpy() == k).sum() > 30}
    cm = {k: Xm[Lm.numpy() == k].mean(0) for k in cls if (Lm.numpy() == k).sum() > 30}
    common = sorted(set(cc) & set(cm))
    cross = [np.linalg.norm(cc[k] - cm[k]) for k in common]
    within = [np.linalg.norm(cc[i] - cc[j]) for i in common for j in common if i < j]
    R = np.mean(cross) / (np.mean(within) + 1e-9)

    # ring + mirror geometry, on modality-pooled unit-norm prototypes
    proto = {}
    for k in set(cc) | set(cm):
        v = np.mean([a[k] for a in (cc, cm) if k in a], axis=0)
        proto[k] = v / (np.linalg.norm(v) + 1e-9)
    ring = [c for c in COW_RING if c in proto]
    if len(ring) >= 3:
        phi = {c: 2 * math.pi * COW_RING.index(c) / len(COW_RING) for c in ring}
        errs = [abs(float(proto[u] @ proto[v]) - math.cos(phi[u] - phi[v]))
                for u in ring for v in ring if u < v]
        ring_err = float(np.mean(errs))
    else:
        ring_err = float("nan")
    diffs = [proto[r] - proto[l] for r, l in COW_MIRROR if r in proto and l in proto]
    mir = float(np.mean(np.var(np.stack(diffs), axis=0).sum())) if len(diffs) >= 2 else float("nan")

    print(f"\n=== {tag} ===")
    print(f"  tokens: CT {len(Sc)}, MR {len(Sm)};  classes compared: {len(common)}")
    print(f"  linear-probe MODALITY acc : {acc_mod:.3f}   (0.50 mixed, 1.00 disjoint)")
    print(f"  linear-probe CLASS acc    : {acc_cls:.3f}   (state decodability -> key quality)")
    print(f"  d(same class, cross mod)  : {np.mean(cross):.3f}")
    print(f"  d(diff class, same mod)   : {np.mean(within):.3f}")
    print(f"  ratio R                   : {R:.3f}   ({'MODALITY dominates' if R > 1 else 'ANATOMY dominates'})")
    print(f"  RING gram error           : {ring_err:.4f}   (0 = perfectly circular)")
    print(f"  MIRROR offset variance    : {mir:.4f}   (0 = one consistent L/R offset)")
    return dict(tag=tag, mod_acc=acc_mod, cls_acc=acc_cls, R=R,
                ring_err=ring_err, mirror_var=mir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", action="append", required=True)
    ap.add_argument("--n-cases", type=int, default=10)
    ap.add_argument("--out", default=f"{BASE}/evaluation/results/state_geometry_m3.json")
    args = ap.parse_args()

    np.random.seed(0); torch.manual_seed(0)
    sp = json.load(open(f"{PRE}/splits_final.json"))[0]
    ct = [c for c in sp["val"] if "_ct_" in c][:args.n_cases]
    mr = [c for c in sp["val"] if "_mr_" in c][:args.n_cases]

    rows = []
    for t in args.trainer:
        kw = {}
        if "NoComplex" in t:
            pass          # phase is frozen at zero in the checkpoint itself
        net, patch = build(t, **kw)
        rows.append(analyse(t, net, patch, ct, mr))
        del net; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
