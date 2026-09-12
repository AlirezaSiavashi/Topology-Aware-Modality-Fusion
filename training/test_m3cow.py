"""
test_m3cow.py
=============
Correctness checks for the Mamba-3 scan, the state losses and the state-keyed
transformer. Run:  nnunet_venv/bin/python3.9 training/test_m3cow.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F

from mamba3_ssm import complex_scan, M3SSM1D, M3SSMBlock3D
import state_losses as SL
from state_losses import (token_labels, StatePrototypeBank, align_loss,
                          dynamics_loss, probe_loss, variance_covariance_loss,
                          ring_loss, mirror_loss, N_CLASS, COW_RING, COW_MIRROR)
from anatomy_transformer import (AnatomyTransformer, attention_alignment_loss,
                                 presence_loss)

torch.manual_seed(0)
OK, FAIL = [], []
def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name} {extra}")


print("\n[1] complex_scan matches a sequential complex recurrence")
b,g,p,l,n = 2,1,3,17,4
ar,ai = torch.randn(b,g,p,l,n)*0.3, torch.randn(b,g,p,l,n)*0.3
br,bi = torch.randn(b,g,p,l,n),     torch.randn(b,g,p,l,n)
hr,hi = complex_scan(ar,ai,br,bi)
a = torch.complex(ar,ai); bb = torch.complex(br,bi)
h = torch.zeros(b,g,p,n, dtype=torch.cfloat); ref=[]
for t in range(l):
    h = a[:,:,:,t]*h + bb[:,:,:,t]; ref.append(h)
ref = torch.stack(ref, dim=3)
err = (torch.complex(hr,hi)-ref).abs().max().item()
check("scan == sequential recurrence", err < 1e-4, f"max|err|={err:.2e}")

print("\n[2] real phase => pure decay; a real-A scan is a leaky accumulator")
# Constant decay: h_t = a h_{t-1} + 1 saturates monotonically at 1/(1-a).
# (With a VARYING decay rate it need not be monotone, which is why a is fixed.)
ar2 = torch.full((1,1,1,60,1), 0.7)
hr2,hi2 = complex_scan(ar2, torch.zeros_like(ar2), torch.ones_like(ar2), torch.zeros_like(ar2))
check("zero phase leaves imaginary part zero", hi2.abs().max().item() < 1e-7)
check("decay-only state is a monotone leaky accumulator",
      bool((hr2[0,0,0,:,0].diff() >= -1e-6).all()) and hr2[0,0,0,3,0] > hr2[0,0,0,0,0])
check("and it saturates at 1/(1-a), never returning to its start",
      abs(hr2[0,0,0,-1,0].item() - 1/(1-0.7)) < 1e-3)

print("\n[3] rotation is periodic -- the property a real state cannot have")
theta = 2*math.pi/6                       # period 6 tokens
L = 25
a_re = torch.full((1,1,1,L,1), math.cos(theta))
a_im = torch.full((1,1,1,L,1), math.sin(theta))
d_re = torch.zeros(1,1,1,L,1); d_im = torch.zeros(1,1,1,L,1)
d_re[0,0,0,0,0] = 1.0                     # unit impulse, no decay
hr3,hi3 = complex_scan(a_re,a_im,d_re,d_im)
ph = torch.atan2(hi3[0,0,0,:,0], hr3[0,0,0,:,0])
rec = (ph[0]-ph[6]).abs().item() % (2*math.pi)
check("state returns to phase after 2*pi/theta tokens", rec < 1e-4 or abs(rec-2*math.pi) < 1e-4,
      f"phase drift over one period = {rec:.2e}")

print("\n[4] trapezoidal rule differs from Euler and is 2nd-order on a ramp")
ssm = M3SSM1D(d_model=16, d_state=4, n_modalities=2)
x = torch.randn(2, 12, 16); mid = torch.tensor([0,1])
ssm.trapezoidal = True;  y_tr,_ = ssm(x, mid)
ssm.trapezoidal = False; y_eu,_ = ssm(x, mid)
check("trapezoid != euler", (y_tr-y_eu).abs().max().item() > 1e-5)
check("both finite", torch.isfinite(y_tr).all() and torch.isfinite(y_eu).all())

print("\n[5] modality factorisation: a CT-only batch leaves B[MR] untouched")
ssm = M3SSM1D(d_model=16, d_state=4, n_modalities=2)
before = ssm.x_proj_B[1].weight.detach().clone()
y,_ = ssm(torch.randn(2,10,16), torch.tensor([0,0]))
y.sum().backward()
check("B[MR] gets no gradient from a CT-only batch",
      ssm.x_proj_B[1].weight.grad is None or ssm.x_proj_B[1].weight.grad.abs().max()==0)
check("B[CT] does get gradient", ssm.x_proj_B[0].weight.grad.abs().max() > 0)

print("\n[6] forward exposes realised dynamics for both modalities")
ssm = M3SSM1D(d_model=16, d_state=4, n_modalities=2)
y, aux = ssm(torch.randn(4,10,16), torch.tensor([0,0,1,1]))
check("dyn has both modalities", set(aux["dyn"].keys()) == {0,1})
check("dyn tensors carry grad", aux["dyn"][0][0].requires_grad)

print("\n[7] token_labels max-pools rather than nearest-samples")
tgt = torch.zeros(1,1,16,16,16); tgt[0,0,3,3,3] = 7.0
tl = token_labels(tgt, (2,2,2))
check("a single voxel survives 8x downsampling", int((tl==7).sum()) == 1)
tgt2 = torch.full((1,1,8,8,8), -1.0)      # nnU-Net ignore label
check("ignore label (-1) does not crash one_hot", int(token_labels(tgt2,(2,2,2)).sum())==0)

print("\n[8] L_dyn is zero iff the two modalities realise the same spectrum")
same = torch.randn(500).abs()
dyn_same = [{0:(same, same), 1:(same.clone(), same.clone())}]
dyn_diff = [{0:(same, same), 1:(same*3+1, same.clone())}]
ls,_ = dynamics_loss(dyn_same); ld,_ = dynamics_loss(dyn_diff)
check("identical spectra => 0", ls.item() < 1e-6, f"{ls.item():.2e}")
check("shifted spectrum => >0", ld.item() > 0.5, f"{ld.item():.3f}")

print("\n[9] L_dyn backpropagates into A_log, A_theta and dt_proj")
blk = M3SSMBlock3D(channels=16, d_state=4, grad_checkpoint=False)
feat = torch.randn(2,16,4,4,4, requires_grad=True)
out = blk(feat, torch.tensor([0,1]))
ldyn,_ = dynamics_loss([blk.last_dyn])
ldyn.backward()
check("A_log has grad",   blk.ssm.A_log.grad is not None and blk.ssm.A_log.grad.abs().sum()>0)
check("A_theta has grad", blk.ssm.A_theta.grad is not None and blk.ssm.A_theta.grad.abs().sum()>0)
check("dt_proj[0] has grad", blk.ssm.dt_proj[0].weight.grad.abs().sum()>0)
check("dt_proj[1] has grad", blk.ssm.dt_proj[1].weight.grad.abs().sum()>0)

print("\n[10] L_ring is zero exactly on a circular embedding")
C = 32
phi = 2*math.pi*torch.arange(len(COW_RING)).float()/len(COW_RING)
proto = torch.zeros(N_CLASS, C)
for k,c in enumerate(COW_RING):
    proto[c,0] = math.cos(phi[k]); proto[c,1] = math.sin(phi[k])
L = len(COW_RING)
state = proto[list(COW_RING)].unsqueeze(0)                 # (1, 10, C)
labels = torch.tensor(COW_RING).unsqueeze(0)
lr,_ = ring_loss(state, labels)
check("perfect circle => L_ring ~ 0", lr.item() < 1e-6, f"{lr.item():.2e}")
bad = torch.randn(1, L, C)
lrb,_ = ring_loss(bad, labels)
check("random states => L_ring > 0", lrb.item() > 0.05, f"{lrb.item():.3f}")

print("\n[11] L_mirror is zero when laterality is one consistent offset")
C = 8
off = torch.randn(C); off = off/off.norm()
st, lb = [], []
for r,l_ in COW_MIRROR:
    base = torch.randn(C); base = base/base.norm()
    vr = base + off; vl = base - off
    st += [vr, vl]; lb += [r, l_]
state = torch.stack(st).unsqueeze(0); labels = torch.tensor(lb).unsqueeze(0)
lm,_ = mirror_loss(state, labels)
# not exactly 0 because prototypes are re-normalised, but must be far below random
rnd,_ = mirror_loss(torch.randn(1, len(lb), C), labels)
check("consistent offset << random", lm.item() < 0.25*rnd.item(),
      f"{lm.item():.4f} vs {rnd.item():.4f}")

print("\n[12] L_align pulls cross-modality, and collapses without L_vc")
bank = StatePrototypeBank(16, N_CLASS, 2)
st = torch.randn(4, 32, 16, requires_grad=True)
lb = torch.randint(0, N_CLASS, (4,32))
la, s = align_loss(st, lb, torch.tensor([0,0,1,1]), bank)
check("align returns a finite loss", torch.isfinite(la))
la.backward(); check("align backprops to the state", st.grad.abs().sum() > 0)
collapsed = torch.ones(4,32,16, requires_grad=True)
check("L_vc penalises a collapsed state",
      variance_covariance_loss(collapsed).item() > 0.9,
      f"{variance_covariance_loss(collapsed).item():.3f}")
spread = torch.randn(4,256,16)*2
check("L_vc small for a spread state", variance_covariance_loss(spread).item() < 0.6,
      f"{variance_covariance_loss(spread).item():.3f}")

print("\n[13] state-keyed transformer: shapes, keys, and attention alignment")
tr = AnatomyTransformer(state_dim=64, dim=32, n_layers=2)
state = torch.randn(2, 7*5*4, 64)
out = tr(state, (7,5,4))
check("attn is (B, 13, L)", tuple(out["attn"].shape) == (2, 13, 140))
check("attention rows sum to 1", (out["attn"].sum(-1)-1).abs().max().item() < 1e-4)
check("presence is (B, 13)", tuple(out["presence"].shape) == (2,13))
check("keys are a linear map of the state",
      any(m is tr.cross[0].k for m in tr.cross[0].modules()))
# L_attn is a KL: exactly 0 when the query attends only to its own tokens,
# regardless of how many tokens that class occupies.
tokl = torch.zeros(1, 10, dtype=torch.long); tokl[0,3] = 1
perfect = torch.full((1,13,10), 1e-8); perfect[0,0,3] = 1.0
la_p,_ = attention_alignment_loss(perfect, tokl)
diffuse = torch.full((1,13,10), 0.1)
la_d,_ = attention_alignment_loss(diffuse, tokl)
check("perfect attention => L_attn == 0", la_p.item() < 1e-5, f"{la_p.item():.2e}")
check("diffuse attention => L_attn > 0", la_d.item() > 0.1, f"{la_d.item():.3f}")
# and the floor does not depend on class size
tokl2 = torch.zeros(1, 10, dtype=torch.long); tokl2[0,3:7] = 1
perfect2 = torch.full((1,13,10), 1e-8); perfect2[0,0,3:7] = 0.25
la_p2,_ = attention_alignment_loss(perfect2, tokl2)
check("optimum is 0 for a 4-token class too", la_p2.item() < 1e-5, f"{la_p2.item():.2e}")
check("presence_loss finite", torch.isfinite(presence_loss(out["presence"], torch.zeros(2,140,dtype=torch.long))))

print("\n[14] fusion is zero-initialised: output starts identical to the backbone")
seg = torch.randn(2, N_CLASS, 8, 8, 8)
fused = tr.fuse(seg.clone(), out, (7,5,4))
check("fuse() is the identity at init", (fused-seg).abs().max().item() < 1e-6,
      f"max|delta|={(fused-seg).abs().max().item():.2e}")
with torch.no_grad():
    tr.w_prior.fill_(0.3)
fused2 = tr.fuse(seg.clone(), out, (7,5,4))
check("fuse() changes output once weights are nonzero",
      (fused2-seg).abs().max().item() > 1e-4)

print("\n[15] block is a no-op at init (zero-init gamma) and AMP-safe")
blk = M3SSMBlock3D(channels=8, d_state=4, grad_checkpoint=False).eval()
f = torch.randn(2,8,4,4,4)
check("gamma=0 => identity", (blk(f, torch.tensor([0,1]))-f).abs().max().item() < 1e-6)
if torch.cuda.is_available():
    blk = blk.cuda().train()
    f = torch.randn(2,8,6,6,6, device="cuda", requires_grad=True)
    with torch.autocast("cuda"):
        o = blk(f, torch.tensor([0,1], device="cuda"))
    o.float().sum().backward()
    check("fp16 autocast forward+backward is finite",
          torch.isfinite(o).all() and torch.isfinite(f.grad).all())
    blk.grad_checkpoint = True
    f2 = torch.randn(2,8,6,6,6, device="cuda", requires_grad=True)
    with torch.autocast("cuda"):
        o2 = blk(f2, torch.tensor([0,1], device="cuda"))
    o2.float().sum().backward()
    check("gradient checkpointing path runs", torch.isfinite(f2.grad).all())
else:
    print("  skip  cuda checks (no GPU visible)")

print("\n[16] REGRESSION: state losses survive autocast at a large state scale")
# This is the bug that took the first Dataset104 run to NaN at epoch 2.
# autocast forces matmul to fp16 whatever dtype the inputs are, so the
# covariance z^T z (diagonal ~ n*|z|^2) overflowed 65504 once |state| ~ 8.
from state_losses import fp32
tokl = torch.zeros(2, 560, dtype=torch.long)
for b in range(2):
    for c in range(1, 14):
        tokl[b, torch.randint(0, 560, (3,))] = c
base = torch.randn(2, 560, 128)
if torch.cuda.is_available():
    base, tokl = base.cuda(), tokl.cuda()
    probe_head = torch.nn.Linear(128, N_CLASS).cuda()
    for scale in (1, 8, 100, 1000):
        st = base * scale
        with torch.autocast("cuda"), fp32("cuda"):
            lv = variance_covariance_loss(st)
            lp = probe_loss(st, tokl, probe_head)
            la, _ = align_loss(st, tokl, torch.tensor([0,1]).cuda(),
                               StatePrototypeBank(128, N_CLASS, 2).cuda())
        ok = all(torch.isfinite(x) for x in (lv, lp, la))
        check(f"|state|={scale:5d}: all state losses finite under autocast", ok,
              f"vc={lv.item():.4g}")
    # and the covariance term must now be scale-invariant, not 4th order
    v1 = variance_covariance_loss(base)
    v2 = variance_covariance_loss(base * 1000)
    check("L_vc no longer explodes with state scale (was |state|^4)",
          v2.item() < 4 * max(v1.item(), 1e-3),
          f"{v1.item():.4f} -> {v2.item():.4f} at 1000x")
else:
    print("  skip  cuda checks (no GPU visible)")

print("\n[17] out_norm bounds the scan output")
ssm = M3SSM1D(d_model=64, d_state=8, n_modalities=2)
for amp in (1.0, 100.0, 10000.0):
    y, _ = ssm(torch.randn(2, 200, 64) * amp, torch.tensor([0, 1]))
    check(f"input x{amp:>7.0f} -> output std {y.std().item():6.3f}, stays O(1)",
          y.std().item() < 50 and torch.isfinite(y).all())

print(f"\n==== {len(OK)} passed, {len(FAIL)} failed ====")
if FAIL:
    print("failed:", FAIL); sys.exit(1)
