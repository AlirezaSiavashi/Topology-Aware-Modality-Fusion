#!/usr/bin/env python3
"""
Is the MF-SSM shared state organised by anatomy or by modality?

MF-SSM ties A and C so both modalities write into one state space. But nothing
forces them to write into the SAME REGION of it: B^CT and B^MR are free to map
into disjoint sub-spaces, and a shared A and C would still work on each. The
"common state space" is an assumption of the architecture, not a guarantee.

Two measurements decide whether a cross-modal contrastive term has any job:

1. Linear-probe modality accuracy on the state.
   ~50%  -> states are already modality-mixed; contrastive has nothing to fix.
   ~100% -> states are trivially modality-separable; the two modalities occupy
            disjoint regions and the shared space is nominal only.

2. Distance ratio R = d(same class, across modality) / d(different class,
   within modality).
   R < 1 -> geometry is dominated by ANATOMY (desired: a CT ICA state sits
            closer to an MR ICA state than to a CT MCA state).
   R > 1 -> geometry is dominated by MODALITY, which is exactly the failure a
            contrastive loss on h would target.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "training"))
from mfssm_net import MFSSMNet
from dynamic_network_architectures.architectures.unet import PlainConvUNet


def build(trainer, n_mod, share_A, share_C):
    p = json.load(open(f"{BASE}/preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/nnUNetPlans.json"))
    a = p["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    bb = PlainConvUNet(input_channels=1, n_stages=a["n_stages"],
        features_per_stage=a["features_per_stage"], conv_op=nn.Conv3d,
        kernel_sizes=a["kernel_sizes"], strides=a["strides"],
        n_conv_per_stage=a["n_conv_per_stage"], num_classes=14,
        n_conv_per_stage_decoder=a["n_conv_per_stage_decoder"], conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps":1e-5,"affine":True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace":True}, deep_supervision=True)
    net = MFSSMNet(bb, stage_channels=a["features_per_stage"], n_stages_to_wrap=2,
                   d_state=16, n_modalities=n_mod, share_A=share_A, share_C=share_C,
                   force_ref_scan=False)
    ck = torch.load(f"{BASE}/results/nnUNet/Dataset104_TopCoW_joint/{trainer}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                    map_location="cpu", weights_only=False)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ck["network_weights"].items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"  loaded {trainer}: {len(missing)} missing, {len(unexpected)} unexpected")
    return net.cuda().eval(), p["configurations"]["3d_fullres"]["patch_size"]


@torch.no_grad()
def collect(net, patch, cases, mod_id, n_patch=14):
    """Return (states, labels) sampled from the deepest wrapped SSM block."""
    grab = {}
    h = net.blocks[0].register_forward_hook(lambda m, i, o: grab.__setitem__("y", o))
    P = f"{BASE}/preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/nnUNetPlans_3d_fullres"
    S, L = [], []
    for c in cases:
        d = np.load(f"{P}/{c}.npy", mmap_mode="r")
        s = np.load(f"{P}/{c}_seg.npy", mmap_mode="r")
        for _ in range(n_patch):
            fg = np.argwhere(np.asarray(s[0][::4, ::4, ::4]) > 0)
            if len(fg) == 0: continue
            ctr = fg[np.random.randint(len(fg))] * 4
            sl = tuple(slice(max(0, int(ctr[i]) - patch[i]//2),
                             max(0, int(ctr[i]) - patch[i]//2) + patch[i]) for i in range(3))
            x = np.asarray(d[(slice(None),) + sl], dtype=np.float32)
            y = np.asarray(s[(slice(None),) + sl])
            if x.shape[1:] != tuple(patch): continue
            xt = torch.from_numpy(x)[None].cuda()
            net.set_modality(torch.tensor([mod_id]).cuda())
            with torch.autocast("cuda"):
                net(xt)
            st = grab["y"].float()                            # (1,C,d,h,w)
            # Nearest-downsampling a 1-2 mm vessel onto an 8 mm token grid
            # returns background almost everywhere (measured: ~3 labelled
            # tokens per patch). Max-pool the one-hot instead, so a token
            # inherits whichever vessel occupies its receptive field.
            # nnU-Net seg arrays use -1 as the ignore label outside the crop;
            # one_hot asserts on negatives, so clamp first.
            lab0 = torch.from_numpy(np.ascontiguousarray(y[0])).long().cuda().clamp_(0, 13)
            oh = torch.nn.functional.one_hot(lab0, 14).permute(3,0,1,2).float()[None]
            oh = torch.nn.functional.adaptive_max_pool3d(oh, st.shape[2:])[0]
            fgmax, fgarg = oh[1:].max(0)
            st = st[0].flatten(1).T                           # (n_tok, C)
            lab = torch.where(fgmax.flatten() > 0, fgarg.flatten() + 1,
                              torch.zeros_like(fgarg.flatten())).long()
            keep = lab > 0
            if keep.sum() == 0: continue
            S.append(st[keep].cpu()); L.append(lab[keep].cpu())
    h.remove()
    net.set_modality(None)
    return torch.cat(S), torch.cat(L)


def analyse(tag, net, patch, ct_cases, mr_cases):
    Sc, Lc = collect(net, patch, ct_cases, 0)
    Sm, Lm = collect(net, patch, mr_cases, 1)
    X = torch.cat([Sc, Sm]).numpy()
    m = np.concatenate([np.zeros(len(Sc)), np.ones(len(Sm))])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    idx = np.random.permutation(len(X))[:6000]
    acc = cross_val_score(LogisticRegression(max_iter=300), X[idx], m[idx], cv=3).mean()

    # distance ratio
    cls = sorted(set(Lc.tolist()) & set(Lm.tolist()))
    Xc, Xm = X[m == 0], X[m == 1]
    cc = {k: Xc[Lc.numpy() == k].mean(0) for k in cls if (Lc.numpy() == k).sum() > 30}
    cm = {k: Xm[Lm.numpy() == k].mean(0) for k in cls if (Lm.numpy() == k).sum() > 30}
    common = sorted(set(cc) & set(cm))
    cross = [np.linalg.norm(cc[k] - cm[k]) for k in common]
    within = [np.linalg.norm(cc[i] - cc[j]) for i in common for j in common if i < j]
    R = np.mean(cross) / (np.mean(within) + 1e-9)

    print(f"\n=== {tag} ===")
    print(f"  tokens: CT {len(Sc)}, MR {len(Sm)};  classes compared: {len(common)}")
    print(f"  linear-probe MODALITY accuracy : {acc:.3f}   (0.50 = mixed, 1.00 = disjoint)")
    print(f"  d(same class, cross modality)  : {np.mean(cross):.3f}")
    print(f"  d(diff class, same modality)   : {np.mean(within):.3f}")
    print(f"  ratio R                        : {R:.3f}   ({'MODALITY dominates' if R>1 else 'ANATOMY dominates'})")
    return acc, R


if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)
    sp = json.load(open(f"{BASE}/preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/splits_final.json"))[0]
    ct = [c for c in sp["val"] if "_ct_" in c][:10]
    mr = [c for c in sp["val"] if "_mr_" in c][:10]
    for tag, tr, nm, sA, sC in (("MF-SSM (A,C shared; B,dt per-modality)", "nnUNetTrainerMFSSM", 2, True, True),):
        net, patch = build(tr, nm, sA, sC)
        analyse(tag, net, patch, ct, mr)
