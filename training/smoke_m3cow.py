"""End-to-end smoke on real plans. Usage: smoke_m3cow.py [dataset_id]"""
import os, sys, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "training"))
import torch, torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from mfssm3_net import MFSSM3Net
from state_losses import (token_labels, StatePrototypeBank, align_loss, dynamics_loss,
                          probe_loss, variance_covariance_loss, ring_loss, mirror_loss)
from anatomy_transformer import attention_alignment_loss, presence_loss

DS = sys.argv[1] if len(sys.argv) > 1 else "104"
import glob as _g
DSDIR = _g.glob(f"{BASE}/preprocessed/nnUNet_preprocessed/Dataset{DS}_*")[0]
print(f"=== {os.path.basename(DSDIR)} ===")
p = json.load(open(f"{DSDIR}/nnUNetPlans.json"))
cfg = p["configurations"]["3d_fullres"]; a = cfg["architecture"]["arch_kwargs"]
patch, bs = cfg["patch_size"], cfg["batch_size"]
print(f"patch={patch} batch={bs} feats={a['features_per_stage']}")

def build(**kw):
    bb = PlainConvUNet(input_channels=1, n_stages=a["n_stages"],
        features_per_stage=a["features_per_stage"], conv_op=nn.Conv3d,
        kernel_sizes=a["kernel_sizes"], strides=a["strides"], num_classes=14,
        n_conv_per_stage=a["n_conv_per_stage"],
        n_conv_per_stage_decoder=a["n_conv_per_stage_decoder"], conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps":1e-5,"affine":True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace":True}, deep_supervision=True)
    return bb, MFSSM3Net(bb, stage_channels=a["features_per_stage"], **kw)

bb_ref, net = build(n_stages_to_wrap=2, d_state=8, n_modalities=2)
n_bb = sum(q.numel() for q in bb_ref.parameters())
n_ssm = sum(q.numel() for q in net.blocks.parameters())
n_tr = sum(q.numel() for q in net.anatomy.parameters())
print(f"backbone {n_bb/1e6:.2f}M | SSM {n_ssm/1e6:.2f}M | anatomy-TR {n_tr/1e6:.2f}M "
      f"| total {sum(q.numel() for q in net.parameters())/1e6:.2f}M")

net = net.cuda().train()
x = torch.randn(bs, 1, *patch, device="cuda")
tgt = torch.randint(0, 14, (bs, 1, *patch), device="cuda")
mod = torch.tensor([0, 1], device="cuda")[:bs]
bank = StatePrototypeBank(net.state_dim, 14, 2).cuda()
opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.99, nesterov=True)
scaler = torch.amp.GradScaler("cuda")

print("\n--- zero-init check: does the untrained net equal the plain nnU-Net? ---")
net.eval(); net.set_modality(mod)
with torch.no_grad(), torch.autocast("cuda"):
    y_m3 = net(x)[0] if isinstance(net(x), list) else net(x)
    for b in net.blocks: b.gamma.data.fill_(0.)
    net.anatomy = None
    y_bb = bb_ref.cuda()(x)[0]
print(f"max|M3CoW - nnU-Net| at init = {(y_m3-y_bb).abs().max().item():.3e}  (expect ~0)")
bb_ref2, net = build(n_stages_to_wrap=2, d_state=8, n_modalities=2)
net = net.cuda().train(); opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.99)

print("\n--- training step: memory + wall time ---")
for it in range(4):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    net.set_modality(mod); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda"):
        out = net(x)
        seg, state, grid = out["seg"], out["state"].float(), out["grid"]
        tok = token_labels(tgt, grid)
        l_seg = sum(nn.functional.cross_entropy(s, nn.functional.interpolate(
                        tgt.float(), size=s.shape[2:], mode="nearest").squeeze(1).long())
                    for s in seg) / len(seg)
        la,_ = align_loss(state, tok, mod, bank)
        ld,_ = dynamics_loss(out["dyn"])
        lp = probe_loss(state, tok, net.state_probe)
        lv = variance_covariance_loss(state)
        lr_,_ = ring_loss(state, tok); lm,_ = mirror_loss(state, tok)
        lat,_ = attention_alignment_loss(out["tr"]["attn"].float(), tok)
        lpr = presence_loss(out["tr"]["presence"].float(), tok)
        loss = l_seg + .5*la + .1*ld.cuda() + .2*lp + .05*lv + .2*lr_ + .1*lm + .3*lat + .2*lpr
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
    scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    if it:  # skip warm-up iteration
        print(f"  iter {it}: {time.time()-t0:5.2f}s  peak {torch.cuda.max_memory_allocated()/2**30:5.2f} GiB  "
              f"loss={loss.item():.3f} (seg {l_seg.item():.3f} align {la.item():.3f} "
              f"dyn {ld.item():.4f} probe {lp.item():.3f} vc {lv.item():.3f} "
              f"ring {lr_.item():.3f} mirror {lm.item():.4f} attn {lat.item():.3f})")
print(f"\ntoken grids: {net._grids}")
print("state grid used for keys:", net._grids[net.key_slot],
      "=", int(torch.tensor(net._grids[net.key_slot]).prod()), "tokens")
