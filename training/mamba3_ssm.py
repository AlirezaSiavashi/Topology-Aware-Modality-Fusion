"""
mamba3_ssm.py
=============
Mamba-3 style modality-factorised selective SSM with a COMPLEX (rotational)
state, trapezoidal discretisation, and an explicitly exposed state trajectory.

Why complex state, for vessels specifically
-------------------------------------------
A real diagonal A can only DECAY along the scan: a_t = exp(dt*A) with A<0 lies
in (0,1), so the state is a leaky accumulator and every token monotonically
forgets. That is the wrong inductive bias for the Circle of Willis, because

  1. The CoW is literally a RING. Acom - A1 - ICA - Pcom - PCA - BA closes a
     cycle. Under any raster order the scan leaves a structure and comes back
     to its contralateral / downstream partner hundreds of tokens later. A
     decay-only state cannot express "return to what I saw N steps ago"; it can
     only express "forget at rate lambda".
  2. Six of the thirteen classes are MIRROR PAIRS (ICA, MCA, ACA, PCA, Pcom,
     A2 left/right). Bilateral symmetry is a translation in the scan order,
     and representing "the same structure, displaced" is what a phase does.

A complex diagonal a_t = exp(-dt*lambda) * (cos(dt*theta) + i sin(dt*theta))
adds rotation to decay. Rotation is periodic: after 2*pi/(dt*theta) tokens the
state returns to phase. That is the same mechanism that lets Mamba-3 solve
state-tracking/parity tasks a real-diagonal SSM provably cannot, and here it
buys a representation of ring closure and mirror symmetry.

Trapezoidal discretisation
--------------------------
Euler/ZOH puts the whole input mass at the right endpoint:
    h_t = a_t h_{t-1} + dt_t B_t x_t
Mamba-3 uses the trapezoid rule on the input integral:
    h_t = a_t h_{t-1} + (dt_t/2) B_t x_t + a_t (dt_{t-1}/2) B_{t-1} x_{t-1}
which is second-order accurate rather than first-order. It stays an AFFINE
recurrence h_t = a_t h_{t-1} + b_t, so the same associative scan applies -- only
b_t is redefined. For thin vessels sampled at 8 mm token spacing the
discretisation error of the first-order rule is not negligible, which is the
concrete reason to care.

What this module exposes that mamba_ssm does not
------------------------------------------------
`forward` returns (y, aux) where aux carries the per-token state readout and
the REALISED dynamics (dt*lambda, dt*theta). Every state-space loss in
state_losses.py needs those, and the fused CUDA kernel does not return them.
That is why this is a custom scan and not a call into mamba_ssm.

Factorisation (unchanged in intent from mfssm.py, now enforceable)
------------------------------------------------------------------
    lambda, theta  state dynamics       SHARED     (one anatomy)
    C              state -> class       SHARED     (one taxonomy)
    B, dt          measurement + scale  PER MODALITY
Sharing lambda/theta is NOMINAL on its own, because the update only ever sees
the product dt^m * A and dt^m is free. `L_dyn` in state_losses.py is what turns
the nominal sharing into an actual constraint.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ---------------------------------------------------------------------------
# Complex associative scan
# ---------------------------------------------------------------------------
def complex_scan(a_re, a_im, b_re, b_im):
    """
    Solve h_t = a_t * h_{t-1} + b_t over the token axis, in complex arithmetic.

    Shapes: all inputs (B, G, P, L, N); token axis is dim 3. Returns (h_re,
    h_im) of the same shape.

    The recurrence composes associatively,
        (a2,b2) . (a1,b1) = (a2 a1,  a2 b1 + b2)
    so a Hillis-Steele scan resolves it in ceil(log2 L) vectorised steps rather
    than L sequential ones (10 vs 560 for our deepest stage).

    Numerically this is safe because |a_t| = exp(-dt*lambda) < 1: repeated
    products underflow toward zero, which is correct decay, rather than
    overflowing. Doing the scan on the decay directly -- not via exp(-cumsum) --
    is what keeps that true once a phase is present, since the cumulative phase
    is unbounded while the cumulative decay is not.
    """
    L = a_re.shape[3]
    step = 1
    while step < L:
        # shift right along the token axis; identity (1+0i) / zero padding
        pa_re = F.pad(a_re[:, :, :, :-step], (0, 0, step, 0), value=1.0)
        pa_im = F.pad(a_im[:, :, :, :-step], (0, 0, step, 0), value=0.0)
        pb_re = F.pad(b_re[:, :, :, :-step], (0, 0, step, 0), value=0.0)
        pb_im = F.pad(b_im[:, :, :, :-step], (0, 0, step, 0), value=0.0)

        # b <- b + a * b_shifted   (complex)
        nb_re = b_re + a_re * pb_re - a_im * pb_im
        nb_im = b_im + a_re * pb_im + a_im * pb_re
        # a <- a * a_shifted       (complex)
        na_re = a_re * pa_re - a_im * pa_im
        na_im = a_re * pa_im + a_im * pa_re

        a_re, a_im, b_re, b_im = na_re, na_im, nb_re, nb_im
        step *= 2

    return b_re, b_im


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------
class M3SSM1D(nn.Module):
    """One Mamba-3 style modality-factorised complex SSM over a token sequence."""

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4,
                 expand: int = 2, dt_rank: Optional[int] = None,
                 n_modalities: int = 2, n_groups: int = 1,
                 bidirectional: bool = True, share_A: bool = True,
                 share_C: bool = True, trapezoidal: bool = True,
                 theta_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        assert self.d_inner % n_groups == 0, "d_inner must divide by n_groups"
        self.n_groups = n_groups
        self.d_head = self.d_inner // n_groups
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))
        self.n_mod = n_modalities
        self.bidirectional = bidirectional
        self.share_A = share_A
        self.share_C = share_C
        self.trapezoidal = trapezoidal

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        # Normalise the scan output before projecting back. VMamba and Mamba-2
        # both carry this `out_norm` and it is not cosmetic: the scan sums up to
        # L = 560 input drives with |a| < 1 but no lower bound on the decay, so
        # nothing bounds |y|. Leaving it out let the state drift past |z| ~ 8
        # during epoch 2 of the first Dataset104 run, which is where the fp16
        # covariance overflowed. It also puts the state on a sane scale for the
        # losses that read it raw and for use as attention keys.
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # ---- A = -exp(A_log) + i*theta : shared decay AND shared rotation ---
        n_A = 1 if share_A else n_modalities
        # S4D-Lin initialisation: real part -1/2, imaginary part pi*n. This
        # spreads the state channels over a bank of frequencies from DC to
        # Nyquist, so the layer starts with a range of periods available rather
        # than having to discover them.
        A_re = torch.full((self.n_groups, self.d_head, d_state), 0.5)
        # theta_scale rescales the S4D-Lin frequency bank. The default 1.0 puts
        # theta up to pi*(d_state-1) = 22 rad; with dt <= 0.5 that is dt*theta ~ 11,
        # far above the Nyquist guard's pi ceiling, so the top channels START
        # aliased and training spends its budget walking them down. The trained
        # model settled at theta in [0, 0.51], i.e. a scale of ~0.02.
        A_th = theta_scale * math.pi * torch.arange(d_state, dtype=torch.float32)
        A_th = A_th.view(1, 1, d_state).repeat(self.n_groups, self.d_head, 1)
        self.A_log = nn.Parameter(torch.log(A_re).unsqueeze(0).repeat(n_A, 1, 1, 1))
        self.A_theta = nn.Parameter(A_th.unsqueeze(0).repeat(n_A, 1, 1, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ---- C: state -> class readout. Complex, shared. --------------------
        n_C = 1 if share_C else n_modalities
        self.x_proj_C = nn.ModuleList([
            nn.Linear(self.d_inner, 2 * n_groups * d_state, bias=False)
            for _ in range(n_C)])

        # ---- B and dt: modality-specific inverse-observation operator -------
        self.x_proj_B = nn.ModuleList([
            nn.Linear(self.d_inner, 2 * n_groups * d_state, bias=False)
            for _ in range(n_modalities)])
        self.x_proj_dt = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank, bias=False)
            for _ in range(n_modalities)])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(n_modalities)])
        for lin in self.dt_proj:
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3))
                           + math.log(1e-3)).clamp(min=1e-4)
            with torch.no_grad():
                lin.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    # -- one modality, one direction -----------------------------------------
    def _scan_one(self, x, z, m):
        """
        x, z: (b, d_inner, l) for a single modality m. Returns
        (y (b,d_inner,l), dt_lambda, dt_theta) where the last two are the
        REALISED per-token dynamics that L_dyn constrains.
        """
        b, d, l = x.shape
        g, p, n = self.n_groups, self.d_head, self.d_state
        xt = x.transpose(1, 2)                                   # (b, l, d)

        Bc = self.x_proj_B[m](xt).view(b, l, 2, g, n)
        Cc = self.x_proj_C[0 if self.share_C else m](xt).view(b, l, 2, g, n)
        B_re = Bc[:, :, 0].permute(0, 2, 3, 1)                   # (b,g,n,l)
        B_im = Bc[:, :, 1].permute(0, 2, 3, 1)
        C_re = Cc[:, :, 0].permute(0, 2, 3, 1)
        C_im = Cc[:, :, 1].permute(0, 2, 3, 1)

        dt = F.softplus(self.dt_proj[m](self.x_proj_dt[m](xt)))   # (b, l, d)
        # Bound the timestep. As dt -> 0 the decay exp(-dt*lambda) -> 1 and the
        # 560-step scan degenerates into a pure cumulative sum, whose gradient
        # w.r.t. dt accumulates over every position and overflows -- observed
        # as a non-finite grad on dt_proj at step 319 of a real run. The bounds
        # are the range Mamba initialises dt into anyway, so this constrains the
        # effective memory length to a physically sensible window rather than
        # changing what the layer can represent.
        dt = dt.clamp(min=1e-4, max=0.5)
        dt = dt.transpose(1, 2).view(b, g, p, l).float()          # (b,g,p,l)

        ai = 0 if self.share_A else m
        lam = torch.exp(self.A_log[ai].float())                   # (g,p,n) > 0
        the = self.A_theta[ai].float()                            # (g,p,n)

        dt_lam = torch.einsum("bgpl,gpn->bgpln", dt, lam)         # decay rate
        dt_the = torch.einsum("bgpl,gpn->bgpln", dt, the)         # phase advance
        # Nyquist guard: more than pi radians of rotation per token aliases,
        # and an aliased phase is indistinguishable from a different frequency.
        # The tanh keeps |dt*theta| < pi smoothly instead of clipping gradients.
        dt_the = math.pi * torch.tanh(dt_the / math.pi)

        decay = torch.exp(-dt_lam)
        a_re = decay * torch.cos(dt_the)
        a_im = decay * torch.sin(dt_the)

        xg = x.view(b, g, p, l).float()
        u_re = torch.einsum("bgpl,bgnl,bgpl->bgpln", dt, B_re.float(), xg)
        u_im = torch.einsum("bgpl,bgnl,bgpl->bgpln", dt, B_im.float(), xg)

        if self.trapezoidal:
            # b_t = (1/2) u_t + (1/2) a_t u_{t-1}   -- second-order input rule
            pu_re = F.pad(u_re[:, :, :, :-1], (0, 0, 1, 0), value=0.0)
            pu_im = F.pad(u_im[:, :, :, :-1], (0, 0, 1, 0), value=0.0)
            b_re = 0.5 * u_re + 0.5 * (a_re * pu_re - a_im * pu_im)
            b_im = 0.5 * u_im + 0.5 * (a_re * pu_im + a_im * pu_re)
        else:
            b_re, b_im = u_re, u_im

        h_re, h_im = complex_scan(a_re, a_im, b_re, b_im)

        # y_t = 2 Re(conj(C_t) . h_t) -- the real readout of a complex state
        y = 2.0 * (torch.einsum("bgpln,bgnl->bgpl", h_re, C_re.float())
                   + torch.einsum("bgpln,bgnl->bgpl", h_im, C_im.float()))
        y = y.reshape(b, d, l)
        y = y + x.float() * self.D.float()[None, :, None]
        y = y * F.silu(z.float())
        return y, dt_lam, dt_the

    def _one_direction(self, xc, z, mod_ids):
        """
        Split the batch by modality, scan each, reassemble.

        The scan runs with autocast DISABLED. Calling `.float()` on the inputs
        is not sufficient -- torch.autocast intercepts einsum and matmul and
        runs them in fp16 whatever dtype the inputs are. A chain of up to 560
        multiply-accumulates has neither the range nor the precision for fp16,
        and the failure shows up in the BACKWARD pass (the forward looked
        healthy right up to the step that died).
        """
        out = torch.zeros_like(xc, dtype=torch.float32)
        dyn = {}
        for m in range(self.n_mod):
            idx = (mod_ids == m).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            with torch.autocast(xc.device.type, enabled=False):
                y, dl, dth = self._scan_one(xc[idx].float().contiguous(),
                                            z[idx].float().contiguous(), m)
            out = out.index_copy(0, idx, y)
            # Subsample the realised dynamics: L_dyn compares DISTRIBUTIONS, so
            # a few thousand samples are as informative as all b*g*p*l*n of them.
            # These stay attached to the graph on purpose -- L_dyn's whole job is
            # to push gradient back into A_log, A_theta and dt_proj.
            dyn[m] = (dl.flatten()[::97], dth.flatten()[::97])
        return out, dyn

    def forward(self, x: torch.Tensor, mod_ids: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict]:
        """x: (b, l, d_model); mod_ids: (b,). Returns (y (b,l,d_model), aux)."""
        b, l, _ = x.shape
        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)
        xs = xs.transpose(1, 2)
        xs = F.silu(self.conv1d(xs)[..., :l])
        zt = z.transpose(1, 2).contiguous()

        y, dyn = self._one_direction(xs.contiguous(), zt, mod_ids)
        if self.bidirectional:
            y_rev, dyn_r = self._one_direction(xs.flip(-1).contiguous(),
                                               zt.flip(-1).contiguous(), mod_ids)
            y = y + y_rev.flip(-1)
            for m, v in dyn_r.items():
                if m in dyn:
                    dyn[m] = (torch.cat([dyn[m][0], v[0]]),
                              torch.cat([dyn[m][1], v[1]]))
                else:
                    dyn[m] = v

        y = self.out_norm(y.transpose(1, 2))                     # (b, l, d_inner)
        out = self.out_proj(y.to(x.dtype))                       # (b, l, d_model)
        return out, {"dyn": dyn}


class M3SSMBlock3D(nn.Module):
    """
    Residual Mamba-3 block over a 3D feature map.

    Flattens (D,H,W) to a token sequence, scans, restores the grid. A depthwise
    3D conv before flattening restores the spatial adjacency a 1D raster order
    destroys. `last_state` -- the per-token state readout BEFORE the residual
    add -- is what the state losses supervise and what the anatomy transformer
    uses as keys; it is kept rather than discarded.

    Channel bottleneck
    ------------------
    The scan is memory-bandwidth bound on (b, d_inner, L, N) tensors, and
    d_inner = expand * channels dominates: at the wrapped stages channels = 320,
    so a plain block scans 640 channels and costs 2.15x the whole nnU-Net
    backbone per step. Projecting 320 -> d_bottleneck before the SSM and back
    after cuts that axis directly while keeping the gated Mamba structure
    intact -- unlike dropping `expand` to 1, which removes the gate's headroom.

    It also makes the state a better key: 128 dims of supervised state is a
    saner attention key space than 320 dims dominated by whatever the encoder
    stage happened to carry.
    """

    def __init__(self, channels: int, d_state: int = 8, expand: int = 2,
                 n_modalities: int = 2, n_groups: int = 1, share_A: bool = True,
                 share_C: bool = True, trapezoidal: bool = True,
                 grad_checkpoint: bool = False, d_bottleneck: Optional[int] = 128,
                 theta_scale: float = 1.0):
        super().__init__()
        self.dwconv = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        d_ssm = d_bottleneck if d_bottleneck else channels
        self.d_ssm = d_ssm
        self.proj_in = (nn.Linear(channels, d_ssm, bias=False)
                        if d_ssm != channels else nn.Identity())
        self.proj_out = (nn.Linear(d_ssm, channels, bias=False)
                         if d_ssm != channels else nn.Identity())
        self.norm = nn.LayerNorm(channels)
        self.ssm = M3SSM1D(d_ssm, d_state=d_state, expand=expand,
                           n_modalities=n_modalities, n_groups=n_groups,
                           share_A=share_A, share_C=share_C,
                           trapezoidal=trapezoidal, theta_scale=theta_scale)
        self.gamma = nn.Parameter(torch.zeros(1))   # zero-init => starts identity
        self.grad_checkpoint = grad_checkpoint
        self.last_state: Optional[torch.Tensor] = None
        self.last_dyn: Optional[Dict] = None

    def forward(self, feat: torch.Tensor, mod_ids: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = feat.shape
        x = self.dwconv(feat).flatten(2).transpose(1, 2)          # (b, l, c)
        x = self.proj_in(self.norm(x))                            # (b, l, d_ssm)

        if self.grad_checkpoint and self.training and x.requires_grad:
            # The scan materialises (b,g,p,l,n) tensors at every one of the
            # log2(L) steps. Trades a full recompute for that memory -- only
            # worth it when the card is contended; the bottleneck above is the
            # cheaper lever, so this defaults off.
            s, aux = cp.checkpoint(self.ssm, x, mod_ids, use_reentrant=False)
        else:
            s, aux = self.ssm(x, mod_ids)

        self.last_state = s                                       # (b, l, d_ssm)
        self.last_dyn = aux["dyn"]
        y = self.proj_out(s).transpose(1, 2).reshape(b, c, d, h, w)
        return feat + self.gamma * y
