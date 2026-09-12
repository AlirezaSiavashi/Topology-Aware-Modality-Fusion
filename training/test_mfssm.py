"""Correctness tests for the modality-factorised SSM. Run before GPU hours."""
import os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mfssm import MFSSM1D, MFSSMBlock3D, selective_scan_ref, HAVE_CUDA_SCAN

ok, fail, failures = "  ok  ", " FAIL ", []
def check(name, cond, extra=""):
    print(f"[{ok if cond else fail}] {name} {extra}")
    if not cond: failures.append(name)

print(f"\n--- environment ---\nCUDA selective_scan available: {HAVE_CUDA_SCAN}")

# 1. reference scan vs an explicit recurrence
print("\n--- reference scan ---")
torch.manual_seed(0)
b,d,l,n = 2,4,7,3
u=torch.randn(b,d,l); delta=torch.rand(b,d,l)+.5
A=-torch.rand(d,n)-.5; B=torch.randn(b,n,l); C=torch.randn(b,n,l); D=torch.randn(d)
y=selective_scan_ref(u,delta,A,B,C,D,None,None,False)
h=torch.zeros(b,d,n); ys=[]
for t in range(l):
    dA=torch.exp(delta[:,:,t:t+1]*A[None]); dB=delta[:,:,t,None]*B[:,None,:,t]*u[:,:,t,None]
    h=dA*h+dB; ys.append((h*C[:,None,:,t]).sum(-1))
ref=torch.stack(ys,2)+u*D[None,:,None]
check("selective_scan_ref matches explicit recurrence",
      torch.allclose(y,ref,atol=1e-5), f"max err={(y-ref).abs().max():.2e}")

# 2. factorisation: what is shared and what is not
print("\n--- factorisation ---")
m = MFSSM1D(d_model=16, d_state=8, n_modalities=2, force_ref_scan=True)
check("A has ONE copy when share_A=True", m.A_log.shape[0]==1, str(tuple(m.A_log.shape)))
check("C has ONE projection when share_C=True", len(m.x_proj_C)==1)
check("B has one projection PER modality", len(m.x_proj_B)==2)
check("dt has one projection PER modality", len(m.dt_proj)==2)
sep = MFSSM1D(d_model=16, d_state=8, n_modalities=2, share_A=False, share_C=False, force_ref_scan=True)
check("share_A=False gives per-modality A", sep.A_log.shape[0]==2, str(tuple(sep.A_log.shape)))
check("share_C=False gives per-modality C", len(sep.x_proj_C)==2)

# 3. modality actually changes the output
print("\n--- modality conditioning ---")
x = torch.randn(4, 12, 16)
ct = torch.zeros(4, dtype=torch.long); mr = torch.ones(4, dtype=torch.long)
y_ct, y_mr = m(x, ct), m(x, mr)
check("same input, different modality -> different output",
      (y_ct-y_mr).abs().max().item() > 1e-4, f"max|d|={(y_ct-y_mr).abs().max():.3e}")

# batch-splitting must equal per-modality forward on the same rows
mixed = torch.tensor([0,1,0,1])
y_mix = m(x, mixed)
check("mixed batch row0 == pure-CT row0", torch.allclose(y_mix[0], y_ct[0], atol=1e-5),
      f"max err={(y_mix[0]-y_ct[0]).abs().max():.2e}")
check("mixed batch row1 == pure-MR row1", torch.allclose(y_mix[1], y_mr[1], atol=1e-5),
      f"max err={(y_mix[1]-y_mr[1]).abs().max():.2e}")

# 4. gradients reach the right parameters
print("\n--- gradients ---")
m.zero_grad(); m(x, mixed).sum().backward()
check("shared A receives gradient", m.A_log.grad is not None and m.A_log.grad.abs().sum()>0)
check("shared C receives gradient", m.x_proj_C[0].weight.grad.abs().sum()>0)
check("B[CT] receives gradient", m.x_proj_B[0].weight.grad.abs().sum()>0)
check("B[MR] receives gradient", m.x_proj_B[1].weight.grad.abs().sum()>0)

m.zero_grad(); m(x, torch.zeros(4,dtype=torch.long)).sum().backward()
gb1 = m.x_proj_B[1].weight.grad
check("CT-only batch leaves B[MR] untouched", gb1 is None or gb1.abs().sum()==0,
      "(modality-specific paths are genuinely separate)")
check("CT-only batch still trains shared A", m.A_log.grad.abs().sum()>0,
      "(shared dynamics learn from every sample)")

# 5. 3D block
print("\n--- 3D block ---")
blk = MFSSMBlock3D(channels=8, d_state=8, n_modalities=2, force_ref_scan=True)
f = torch.randn(2, 8, 4, 5, 4, requires_grad=True)
out = blk(f, torch.tensor([0,1]))
check("3D block preserves shape", out.shape==f.shape, str(tuple(out.shape)))
check("zero-init gamma => starts as identity", torch.allclose(out, f, atol=1e-6),
      "(safe to insert into a trained backbone)")
with torch.no_grad(): blk.gamma.fill_(1.0)
out2 = blk(f, torch.tensor([0,1]))
check("non-zero gamma changes output", (out2-f).abs().max().item()>1e-5)
out2.sum().backward()
check("gradient flows to input", f.grad is not None and torch.isfinite(f.grad).all())

print("\n"+"="*58)
if failures: print(f"FAILED ({len(failures)}): "+", ".join(failures)); sys.exit(1)
print("all checks passed")
