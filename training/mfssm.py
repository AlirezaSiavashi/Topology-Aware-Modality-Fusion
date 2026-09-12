"""
mfssm.py
========
Modality-Factorised Selective State Space Model (MF-SSM).

A vessel tree is one latent object observed through two different instruments.
The anatomy is modality-independent; only the measurement operator differs.
A selective SSM factorises exactly along that seam:

    h_t = exp(dt^(m) A) h_{t-1} + dt^(m) B^(m) x_t          state update
    y_t = C h_t + D x_t                                      readout

    A   state dynamics   -- how vessel context propagates    SHARED
    B   writes the measurement into state                    MODALITY-SPECIFIC
    C   reads state out to vessel class                      SHARED
    dt  timescale / physical step                            MODALITY-SPECIFIC
    D   local skip                                           SHARED

Why the sharing constraint is the whole idea
--------------------------------------------
Modality-specific B alone is just two encoders with a shared decoder -- the two
modalities can occupy disjoint regions of state space and never interact. Tying
A and C forces both measurement operators to write into ONE state space with
ONE set of dynamics and ONE readout. That is the constraint that makes this a
hypothesis rather than a re-parameterisation, and it needs no paired data.

A note on the direction of the analogy
--------------------------------------
In classical state-space form the observation model runs forward, y = Cx: the
image is the observation. Here the image is the *input*, so B is the INVERSE
observation operator (measurement -> latent anatomy) and C is a label readout
with no counterpart in the physics. That still supports making B
modality-specific -- CT and TOF forward operators differ, so their inverses
must too -- but it should be stated as "inverse observation operator" rather
than "observation model".

Implementation
--------------
Uses the official selective_scan_fn from mamba_ssm when available. That kernel
takes delta, B and C as explicit arguments rather than deriving them
internally, so the factorisation above is expressible without touching the CUDA
code. A pure-PyTorch reference scan is used as a fallback (correct, slower).

Mixed batches (some CT, some MRA) are handled by evaluating every modality's
projection and gathering per sample. The projections are small relative to the
scan, so this costs little and keeps the op fully batched.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
    HAVE_CUDA_SCAN = True
except Exception:                                     # pragma: no cover
    _cuda_scan = None
    HAVE_CUDA_SCAN = False


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False):
    """
    Reference selective scan. Shapes follow mamba_ssm:
      u, delta : (b, d, l)      A : (d, n)
      B, C     : (b, n, l)      D : (d,)      z : (b, d, l)
    Returns (b, d, l). Used only when the CUDA kernel is unavailable.
    """
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    b, d, l = u.shape
    n = A.shape[1]

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    x = torch.zeros(b, d, n, device=u.device, dtype=torch.float32)
    ys = []
    for i in range(l):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)                        # (b, d, l)

    if D is not None:
        y = y + u * D[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(dtype_in)


def selective_scan_parallel(u, delta, A, B, C, D=None, z=None,
                            delta_bias=None, delta_softplus=False):
    """
    Log-depth associative scan. Same contract as selective_scan_ref.

    The recurrence h_t = a_t h_{t-1} + b_t composes associatively:
        (a2,b2) . (a1,b1) = (a2 a1,  a2 b1 + b2)
    so a Hillis-Steele scan computes it in ceil(log2 L) vectorised steps
    instead of L sequential ones. For L=560 that is 10 steps rather than 560,
    which is the difference between a usable fallback and an 8-day run.

    Numerically safe here: a_t = exp(dt_t A) with A < 0, so a_t in (0,1) and
    repeated products underflow toward zero (correct decay) rather than
    overflowing. That is why this is done on the decay directly and not via
    exp(-cumsum), which does overflow.
    """
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))          # decay
    b = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)             # input drive

    L = a.shape[2]
    step = 1
    while step < L:
        a_sh = F.pad(a[:, :, :-step], (0, 0, step, 0), value=1.0)
        b_sh = F.pad(b[:, :, :-step], (0, 0, step, 0), value=0.0)
        b = b + a * b_sh
        a = a * a_sh
        step *= 2

    y = torch.einsum("bdln,bnl->bdl", b, C)
    if D is not None:
        y = y + u * D[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(dtype_in)


class MFSSM1D(nn.Module):
    """One modality-factorised selective-SSM layer over a token sequence."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: Optional[int] = None,
                 n_modalities: int = 2, bidirectional: bool = True,
                 share_A: bool = True, share_C: bool = True,
                 force_ref_scan: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))
        self.n_mod = n_modalities
        self.bidirectional = bidirectional
        self.share_A = share_A
        self.share_C = share_C
        self.force_ref_scan = force_ref_scan

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # ---- A: state dynamics. Shared => one anatomy propagating. ----------
        n_A = 1 if share_A else n_modalities
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.repeat(self.d_inner, 1)                       # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).repeat(n_A, 1, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ---- C: readout to vessel class. Shared => one taxonomy. -----------
        n_C = 1 if share_C else n_modalities
        self.x_proj_C = nn.ModuleList(
            [nn.Linear(self.d_inner, d_state, bias=False) for _ in range(n_C)])

        # ---- B and dt: modality-specific measurement operator + timescale --
        self.x_proj_B = nn.ModuleList(
            [nn.Linear(self.d_inner, d_state, bias=False) for _ in range(n_modalities)])
        self.x_proj_dt = nn.ModuleList(
            [nn.Linear(self.d_inner, self.dt_rank, bias=False) for _ in range(n_modalities)])
        self.dt_proj = nn.ModuleList(
            [nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(n_modalities)])
        for lin in self.dt_proj:
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3))
                           + math.log(1e-3)).clamp(min=1e-4)
            with torch.no_grad():
                lin.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _scan(self, u, delta, A, B, C, D, z):
        use_cuda = (HAVE_CUDA_SCAN and not self.force_ref_scan and u.is_cuda
                    and u.dtype in (torch.float16, torch.bfloat16, torch.float32))
        if use_cuda:
            return _cuda_scan(u, delta, A, B, C, D, z,
                              delta_bias=None, delta_softplus=True)
        return selective_scan_parallel(u, delta, A, B, C, D, z,
                                       delta_bias=None, delta_softplus=True)

    def _one_direction(self, xc, z, mod_ids):
        """
        xc: (b, d_inner, l) post-conv -> (b, d_inner, l).

        The batch is split by modality and scanned once per modality rather
        than gathering per sample. selective_scan_fn takes A with shape
        (d_inner, d_state) and has no batch axis for it, so a per-sample A is
        not expressible -- splitting is what makes the non-shared-A ablation
        implementable at all, and it keeps the shared case identical in form.
        """
        out = torch.zeros_like(xc)
        for m in range(self.n_mod):
            idx = (mod_ids == m).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            xm, zm = xc[idx], z[idx]
            xt = xm.transpose(1, 2)                          # (bm, l, d_inner)

            B = self.x_proj_B[m](xt)                         # modality-specific
            C = self.x_proj_C[0 if self.share_C else m](xt)  # shared by default
            dt = self.dt_proj[m](self.x_proj_dt[m](xt))      # modality-specific
            A = -torch.exp(self.A_log[0 if self.share_A else m].float())

            y = self._scan(xm.contiguous(), dt.transpose(1, 2).contiguous(), A,
                           B.transpose(1, 2).contiguous(),
                           C.transpose(1, 2).contiguous(),
                           self.D.float(), zm.contiguous())
            out = out.index_copy(0, idx, y.to(out.dtype))
        return out

    def forward(self, x: torch.Tensor, mod_ids: torch.Tensor) -> torch.Tensor:
        """x: (b, l, d_model); mod_ids: (b,) long. Returns (b, l, d_model)."""
        b, l, _ = x.shape
        xz = self.in_proj(x)                                 # (b, l, 2*d_inner)
        xs, z = xz.chunk(2, dim=-1)
        xs = xs.transpose(1, 2)                              # (b, d_inner, l)
        xs = F.silu(self.conv1d(xs)[..., :l])
        zt = z.transpose(1, 2).contiguous()

        y = self._one_direction(xs.contiguous(), zt, mod_ids)
        if self.bidirectional:
            y_rev = self._one_direction(xs.flip(-1).contiguous(),
                                        zt.flip(-1).contiguous(), mod_ids)
            y = y + y_rev.flip(-1)

        return self.out_proj(y.transpose(1, 2))              # (b, l, d_model)


class MFSSMBlock3D(nn.Module):
    """
    Residual MF-SSM block over a 3D feature map.

    Flattens (D,H,W) to a token sequence, scans, and restores the grid. A
    depthwise 3D conv before flattening preserves the spatial adjacency that a
    1D raster order destroys (as in UlikeMamba/VMamba 3D variants).
    """

    def __init__(self, channels: int, d_state: int = 16, expand: int = 2,
                 n_modalities: int = 2, share_A: bool = True,
                 share_C: bool = True, force_ref_scan: bool = False):
        super().__init__()
        self.dwconv = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.ssm = MFSSM1D(channels, d_state=d_state, expand=expand,
                           n_modalities=n_modalities, share_A=share_A,
                           share_C=share_C, force_ref_scan=force_ref_scan)
        self.gamma = nn.Parameter(torch.zeros(1))   # zero-init: starts as identity

    def forward(self, feat: torch.Tensor, mod_ids: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = feat.shape
        x = self.dwconv(feat)
        x = x.flatten(2).transpose(1, 2)                     # (b, l, c)
        x = self.ssm(self.norm(x), mod_ids)
        x = x.transpose(1, 2).reshape(b, c, d, h, w)
        return feat + self.gamma * x
