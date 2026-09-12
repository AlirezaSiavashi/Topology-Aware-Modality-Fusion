"""
test_cmap.py
============
Smoke tests for the CMAP components. Run before committing GPU hours:

    python3.9 training/test_cmap.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hyperbolic as H
from cmap import (
    ClasswisePriorBank,
    CrossModalPriorModule,
    RadiusStats,
    ReliabilityMatrix,
    gate_regulariser,
    prior_loss,
    sample_voxels,
)
from cmap_net import CMAPNet

OK, FAIL = "  ok  ", " FAIL "
failures = []


def check(name, cond, extra=""):
    print(f"[{OK if cond else FAIL}] {name} {extra}")
    if not cond:
        failures.append(name)


# ── 1. Hyperbolic ops ────────────────────────────────────────────────────────
print("\n--- hyperbolic ops ---")
torch.manual_seed(0)
v = torch.randn(500, 16) * 3.0
x = H.expmap0(v, 1.0)
check("expmap0 lands inside the ball", bool((x.norm(dim=-1) < 1.0).all()),
      f"max||x||={x.norm(dim=-1).max():.6f}")

# Round-trip only holds below tanh saturation; above ~4 the map is genuinely
# lossy, which is why the module clips the tangent norm before expmap0.
v_small = torch.randn(500, 16) * 0.15          # ||v|| ~ 0.6
err = (v_small - H.logmap0(H.expmap0(v_small, 1.0), 1.0)).abs().max().item()
check("logmap0(expmap0(v)) == v in the operating range", err < 1e-3,
      f"max err={err:.2e}")

# The property the method actually relies on: radius is strictly monotone in
# ||v|| across the clipped range, so it can encode a hierarchy.
norms = torch.linspace(0.05, 2.5, 40).unsqueeze(1) * torch.ones(1, 16) / (16 ** 0.5)
radii = H.expmap0(norms, 1.0).norm(dim=-1)
check("radius strictly increasing over the clipped range",
      bool((radii[1:] > radii[:-1]).all()),
      f"r in [{radii[0]:.3f}, {radii[-1]:.3f}]")

d_self = H.dist(x, x, 1.0)
check("d(x,x) == 0", float(d_self.abs().max()) < 1e-4, f"max={d_self.abs().max():.2e}")

d_ab = H.dist(x[:100], x[100:200], 1.0)
d_ba = H.dist(x[100:200], x[:100], 1.0)
check("d symmetric", torch.allclose(d_ab, d_ba, atol=1e-4))
check("d non-negative", bool((d_ab >= 0).all()))

# radius grows monotonically with tangent norm -> the anatomical reading holds
small = H.expmap0(torch.randn(200, 16) * 0.1, 1.0).norm(dim=-1).mean()
large = H.expmap0(torch.randn(200, 16) * 5.0, 1.0).norm(dim=-1).mean()
check("larger tangent norm -> larger radius", float(small) < float(large),
      f"{small:.4f} < {large:.4f}")

protos = H.expmap0(torch.randn(14, 16) * 0.5, 1.0)
z = H.expmap0(torch.randn(2, 4, 5, 6, 16), 1.0)
dp = H.dist_to_prototypes(z, protos, 1.0)
check("dist_to_prototypes shape", tuple(dp.shape) == (2, 4, 5, 6, 14), str(tuple(dp.shape)))

# fp16 input must not produce NaN (autocast safety)
zh = H.expmap0((torch.randn(100, 16) * 8).half(), 1.0)
check("no NaN from fp16 input", not bool(torch.isnan(zh).any()))


# ── 2. Prototype bank + reliability ──────────────────────────────────────────
print("\n--- bank / reliability ---")
bank = ClasswisePriorBank(14, 16)
p = bank.prototypes()
check("bank prototypes on ball", bool((p.norm(dim=-1) < 1.0).all()))
check("bank prototypes shape", tuple(p.shape) == (14, 16))

rel = ReliabilityMatrix(2, 14)
check("alpha starts uniform", torch.allclose(rel.alpha, torch.full((2, 14), 0.5)))

dice = torch.zeros(2, 14)
dice[:, 0] = 1.0
dice[0, 1:] = 0.80          # CT good
dice[1, 1:] = 0.20          # MR poor
rel.update_from_dice(dice)
check("alpha favours the reliable modality",
      bool(rel.alpha[0, 1] > rel.alpha[1, 1]),
      f"CT={rel.alpha[0,1]:.3f} MR={rel.alpha[1,1]:.3f}")
check("alpha sums to 1 per class",
      torch.allclose(rel.alpha.sum(0), torch.ones(14), atol=1e-5))

# class-dependent asymmetry: the thing a scalar cannot express
dice2 = torch.zeros(2, 14)
dice2[:, 0] = 1.0
dice2[0, 1:] = 0.5
dice2[1, 1:] = 0.5
dice2[0, 4] = 0.9           # CT much better on class 4
dice2[1, 4] = 0.1
rel2 = ReliabilityMatrix(2, 14)
rel2.update_from_dice(dice2)
check("per-class asymmetry is representable",
      bool(rel2.alpha[0, 4] > 0.8 and abs(rel2.alpha[0, 3] - 0.5) < 1e-3),
      f"cls4 CT={rel2.alpha[0,4]:.2f}, cls3 CT={rel2.alpha[0,3]:.2f}")


# ── 3. Sampling + prior loss ─────────────────────────────────────────────────
print("\n--- prior loss ---")
tgt = torch.zeros(2, 8, 10, 12, dtype=torch.long)
tgt[0, 2:4, 3:6, 4:7] = 3
tgt[1, 1:3, 2:4, 5:8] = 7
idx = sample_voxels(tgt, max_fg=1000)
check("sampler returns foreground", bool((tgt[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] > 0).any()))
check("sampler index shape", idx.shape[1] == 4)

mod = CrossModalPriorModule(feat_channels=8, num_classes=14, embed_dim=16,
                            prior_stride=2)
feat = torch.randn(2, 8, 16, 20, 24, requires_grad=True)
fused, aux = mod(feat, fuse_active=True)
check("fused shape == feature shape", fused.shape == feat.shape,
      f"{tuple(fused.shape)}")
check("gate in [0,1]", bool((aux["gate"] >= 0).all() and (aux["gate"] <= 1).all()))
check("z on ball", bool((aux["z"].norm(dim=-1) < 1.0).all()))

tgt_low = torch.zeros(2, 8, 10, 12, dtype=torch.long)
tgt_low[0, 2:5, 3:6, 4:8] = 3
tgt_low[1, 1:4, 2:5, 5:9] = 7
rstats = RadiusStats(14, 2)
lp, lm, st = prior_loss(aux["z"], aux["dist"], tgt_low, mod.reliability,
                        torch.tensor([0, 1]), tau=0.5, radius_stats=rstats,
                        proto_radii=mod.bank.prototypes().norm(dim=-1).detach())
check("radial matching term finite", bool(torch.isfinite(lm)), f"={lm.item():.4f}")
check("prior loss finite", bool(torch.isfinite(lp)), f"L={lp.item():.4f}")
check("prior loss stats populated", "mean_radius" in st, str(st))

mr = rstats.mean_radius()
check("radius stats recorded per class", bool((~torch.isnan(mr)).any()))

lg = gate_regulariser(aux["gate"])
check("gate regulariser near 0 at init (gate ~ 1)",
      float(lg.detach()) < 0.2, f"={lg.detach():.4f}")

# Saturation guard: the clip must keep voxel radii off the boundary shell so
# they stay distinguishable.
r_all = aux["z"].norm(dim=-1)
check("voxel radii not collapsed onto the boundary",
      float(r_all.max()) < 0.999 and float(r_all.std()) > 1e-3,
      f"max={r_all.max():.4f} std={r_all.std():.4f}")

total = lp + lg + lm
total.backward()
check("grad reaches prototype bank", mod.bank.radius_logit.grad is not None
      and bool(torch.isfinite(mod.bank.radius_logit.grad).all()))
check("grad reaches input features", feat.grad is not None
      and bool(torch.isfinite(feat.grad).all()))
check("prototype grad is non-zero", float(mod.bank.radius_logit.grad.abs().sum()) > 0)


# ── 4. CMAPNet end to end ────────────────────────────────────────────────────
print("\n--- CMAPNet on a real PlainConvUNet ---")
from dynamic_network_architectures.architectures.unet import PlainConvUNet

backbone = PlainConvUNet(
    input_channels=1, n_stages=3, features_per_stage=(8, 16, 32),
    conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2),
    n_conv_per_stage=2, num_classes=14, n_conv_per_stage_decoder=2,
    conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={"affine": True},
    nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=True,
)
net = CMAPNet(backbone, feat_channels=8, num_classes=14, embed_dim=16, prior_stride=2)

x_in = torch.randn(2, 1, 32, 32, 32)

# Reference: what the backbone alone predicts, gate off
net.train()
net.fuse_active = False
out_off = net(x_in)
check("train mode returns dict", isinstance(out_off, dict) and "seg" in out_off)
check("deep supervision list preserved", isinstance(out_off["seg"], list),
      f"n_ds={len(out_off['seg'])}")
check("finest seg is full resolution",
      tuple(out_off["seg"][0].shape[2:]) == (32, 32, 32),
      str(tuple(out_off["seg"][0].shape)))

net.fuse_active = True
out_on = net(x_in)
delta = (out_on["seg"][0] - out_off["seg"][0]).abs().max().item()
check("fusion hook actually changes the finest output", delta > 0,
      f"max|delta|={delta:.3e}")
coarse_delta = (out_on["seg"][1] - out_off["seg"][1]).abs().max().item()
check("coarser DS outputs left untouched", coarse_delta == 0,
      f"max|delta|={coarse_delta:.3e}")

loss = out_on["seg"][0].mean() + 0.3 * out_on["cmap"]["z"].sum() * 0.0
loss.backward()
gn = sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
check("backward through the hook works", gn > 0)

net.eval()
with torch.no_grad():
    ev = net(x_in)
check("eval mode returns tensor/list, not dict", not isinstance(ev, dict))

# state_dict must round-trip (checkpoint compatibility)
sd = net.state_dict()
net2 = CMAPNet(
    PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(8, 16, 32),
        conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2),
        n_conv_per_stage=2, num_classes=14, n_conv_per_stage_decoder=2,
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={"affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=True),
    feat_channels=8, num_classes=14, embed_dim=16, prior_stride=2)
missing, unexpected = net2.load_state_dict(sd, strict=True)
check("state_dict round-trips strictly", True)
check("reliability matrix is checkpointed", "cmap.reliability.alpha" in sd)

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("all checks passed")
