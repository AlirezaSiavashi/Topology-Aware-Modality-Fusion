"""
state_losses.py
===============
Objectives defined on the SSM state itself -- on h, on the realised dynamics
dt*A, and on the geometry of the class prototypes in state space -- rather than
on the segmentation logits.

Why the segmentation loss alone is the wrong supervision for a state
--------------------------------------------------------------------
With Dice/CE at the output, the state is a free intermediate: any h that the
decoder happens to be able to read is optimal. Nothing makes h modality-
agnostic, nothing makes it linearly decodable, and nothing stops it from
collapsing onto whatever few directions the decoder currently uses. The
consequences were measured on this dataset:

  * MF-SSM (A,C tied) beat its no-factorisation control by +0.003 Dice, inside
    the 0.013 seed-noise floor -- the factorisation was doing no work.
  * Re-fitting only B^MR and dt^MR on 5 MRA volumes moved zero-shot Dice from
    0.3957 to 0.3957. The adapter had nothing to adapt INTO, because the state
    geometry A and C rely on was never made modality-independent.

Every loss here attaches to a specific piece of the state space:

  L_align   on h        cross-modal anatomical alignment  -> makes B^m adaptable
  L_dyn     on dt*A     realised-dynamics agreement       -> makes shared A real
  L_probe   on h, C     linear decodability of the state  -> makes h a good key
  L_vc      on h        variance + decorrelation          -> stops collapse
  L_ring    on protos   circular geometry of the CoW ring -> uses the phase
  L_mirror  on protos   bilateral symmetry as one offset  -> uses the phase

L_ring and L_mirror are the two that only a ROTATIONAL state can satisfy; they
are the reason the complex A in mamba3_ssm.py is not decoration.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# fp32 guard
# ---------------------------------------------------------------------------
class fp32:
    """
    Disable autocast for a block of loss arithmetic.

    Calling `.float()` on a tensor is NOT enough: torch.autocast intercepts
    matmul (and friends) and runs them in fp16 whatever dtype the inputs are.
    The covariance in `variance_covariance_loss` is z^T z, whose diagonal is
    ~n*|z|^2 = 1120*|z|^2 here, so it overflows fp16's 65504 ceiling once the
    state reaches |z| ~ 8 -- and then `cov - diag(cov)` is inf - inf = NaN.

    That is not hypothetical: it is what killed the first Dataset104 run.
    Epochs 0-1 were healthy (seg 0.210, probe 1.67, vc 1.06), the state drifted
    past 8 during epoch 2, and every term that touches the state RAW (probe, vc)
    went NaN while every term that L2-normalises first (align, ring, mirror)
    stayed finite. The losses run in fp32 from here on.
    """

    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type

    def __enter__(self):
        self.ctx = torch.autocast(self.device_type, enabled=False)
        self.ctx.__enter__()
        return self

    def __exit__(self, *a):
        return self.ctx.__exit__(*a)


# --- CoW ontology, in Dataset104 label order (0 = background) ---------------
COW_NAMES = ("BA", "R-PCA", "L-PCA", "R-ICA", "R-MCA", "L-ICA", "L-MCA",
             "R-Pcom", "L-Pcom", "Acom", "R-ACA", "L-ACA", "3rd-A2")
N_CLASS = len(COW_NAMES) + 1                                    # 14 with bg

# The principal Circle of Willis cycle, as label indices. This is the ring the
# vessel tree actually closes: basilar -> right posterior -> right carotid ->
# anterior communicating -> left carotid -> left posterior -> back to basilar.
# R-MCA / L-MCA / 3rd-A2 are branches off the ring, not on it, so they are
# excluded from the circular constraint.
COW_RING: Tuple[int, ...] = (1, 2, 8, 4, 11, 10, 12, 6, 9, 3)
#                            BA R-PCA R-Pcom R-ICA R-ACA Acom L-ACA L-ICA L-Pcom L-PCA

# Bilateral mirror pairs (right, left).
COW_MIRROR: Tuple[Tuple[int, int], ...] = ((2, 3), (4, 6), (5, 7),
                                           (8, 9), (11, 12))
#                                          PCA   ICA   MCA   Pcom  ACA


# ---------------------------------------------------------------------------
# Token labels
# ---------------------------------------------------------------------------
def token_labels(target: torch.Tensor, grid: Sequence[int]) -> torch.Tensor:
    """
    (B,1,D,H,W) voxel labels -> (B, prod(grid)) token labels.

    Nearest-downsampling a 1-2 mm vessel onto an 8 mm token grid returns
    background almost everywhere (~3 labelled tokens per patch, measured). Max-
    pooling the one-hot instead lets a token inherit whichever vessel occupies
    its receptive field, which is the semantics we want for a token-level loss.
    nnU-Net writes -1 outside the crop and one_hot rejects negatives, so clamp.
    """
    lab = target[:, 0].long().clamp_(0, N_CLASS - 1)
    oh = F.one_hot(lab, N_CLASS).permute(0, 4, 1, 2, 3).float()
    oh = F.adaptive_max_pool3d(oh, tuple(grid))                 # (B,14,d,h,w)
    fg_max, fg_arg = oh[:, 1:].max(dim=1)
    out = torch.where(fg_max.flatten(1) > 0,
                      fg_arg.flatten(1) + 1,
                      torch.zeros_like(fg_arg.flatten(1)))
    return out.long()                                            # (B, L)


# ---------------------------------------------------------------------------
# Prototype bank
# ---------------------------------------------------------------------------
class StatePrototypeBank(nn.Module):
    """
    EMA class prototypes in state space, kept SEPARATELY per modality.

    Per-modality is the point: a single shared bank would define away the
    question. Holding CT and MR prototypes apart is what lets L_align ask
    whether a CT ICA state sits near an MR ICA state, and lets state_overlap.py
    keep measuring R after the loss is added.
    """

    def __init__(self, dim: int, n_class: int = N_CLASS, n_mod: int = 2,
                 momentum: float = 0.95):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("proto", torch.zeros(n_mod, n_class, dim))
        self.register_buffer("seen", torch.zeros(n_mod, n_class))

    @torch.no_grad()
    def update(self, z: torch.Tensor, labels: torch.Tensor, mod: int):
        """z: (n, C) L2-normalised; labels: (n,)."""
        for c in labels.unique().tolist():
            v = z[labels == c].mean(0)
            v = v / (v.norm() + 1e-6)
            if self.seen[mod, c] == 0:
                self.proto[mod, c] = v
            else:
                p = self.momentum * self.proto[mod, c] + (1 - self.momentum) * v
                self.proto[mod, c] = p / (p.norm() + 1e-6)
            self.seen[mod, c] += 1

    def ready(self, mod: int) -> torch.Tensor:
        return self.seen[mod] > 0


# ---------------------------------------------------------------------------
# L_align : cross-modal anatomical alignment on h
# ---------------------------------------------------------------------------
def align_loss(state: torch.Tensor, labels: torch.Tensor,
               mod_ids: torch.Tensor, bank: StatePrototypeBank,
               tau: float = 0.1, max_tokens: int = 4096
               ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Pull a token's state toward the OTHER modality's prototype of its own class;
    push it from that modality's other class prototypes.

    This is the training-time form of the diagnostic in evaluation/state_overlap.py,
    R = d(same class, cross modality) / d(diff class, within modality). The
    cross-modality term shrinks R's numerator and the within-modality term grows
    its denominator, so the statistic that was only ever measured becomes the
    thing being optimised. No paired CT/MR scans are needed: the coupling runs
    through the prototypes, not through registered voxels.

    state (B,L,C) : token states.  labels (B,L) : token classes.  mod_ids (B,).
    """
    dev = state.device
    z = F.normalize(state.flatten(0, 1).float(), dim=-1)          # (B*L, C)
    lab = labels.flatten()
    mod = mod_ids[:, None].expand(-1, state.shape[1]).flatten()

    fg = lab > 0
    if fg.sum() == 0:
        return torch.zeros((), device=dev), {"align_n": 0.0}
    idx = fg.nonzero(as_tuple=True)[0]
    if idx.numel() > max_tokens:
        idx = idx[torch.randperm(idx.numel(), device=dev)[:max_tokens]]
    z, lab, mod = z[idx], lab[idx], mod[idx]

    # refresh the bank from this batch before using it as negatives
    with torch.no_grad():
        for m in mod.unique().tolist():
            s = mod == m
            bank.update(z[s].detach(), lab[s], m)

    loss, n_used = torch.zeros((), device=dev), 0
    for m in mod.unique().tolist():
        s = mod == m
        other = 1 - m
        P_x = bank.proto[other]                                   # (14, C)
        P_s = bank.proto[m]
        ok_x, ok_s = bank.ready(other), bank.ready(m)
        # only score against classes the bank has actually seen
        keep = ok_x & ok_s
        if keep.sum() < 2:
            continue
        cls = keep.nonzero(as_tuple=True)[0]
        remap = torch.full((N_CLASS,), -1, dtype=torch.long, device=dev)
        remap[cls] = torch.arange(cls.numel(), device=dev)
        tgt = remap[lab[s]]
        valid = tgt >= 0
        if valid.sum() == 0:
            continue
        zs, tgt = z[s][valid], tgt[valid]
        l_x = F.cross_entropy(zs @ P_x[cls].t() / tau, tgt)        # cross-modal
        l_s = F.cross_entropy(zs @ P_s[cls].t() / tau, tgt)        # within-modal
        loss = loss + 0.5 * (l_x + l_s)
        n_used += 1

    if n_used == 0:
        return torch.zeros((), device=dev), {"align_n": 0.0}
    return loss / n_used, {"align_n": float(idx.numel())}


# ---------------------------------------------------------------------------
# L_dyn : realised-dynamics agreement across modalities
# ---------------------------------------------------------------------------
def dynamics_loss(dyn_list: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Make "shared A" mean something.

    The state update only ever sees the PRODUCT dt^m * A. Tying A while leaving
    dt_proj modality-specific therefore constrains nothing: exp(dt^CT A) and
    exp(dt^MR A) are independent functions. That is the flaw in the original
    MF-SSM parameterisation, and it cannot be fixed structurally without giving
    up the modality-specific timescale that handles differing voxel spacing and
    noise -- which is the part that is physically justified.

    So constrain the realisation instead: require the DISTRIBUTIONS of the
    per-token decay rates dt*lambda and phase advances dt*theta to agree across
    modalities. Sorting both samples and taking the mean absolute difference is
    the 1-Wasserstein distance between the empirical distributions, and sorting
    is a permutation, so gradient flows into A_log, A_theta and dt_proj alike.

    Distribution-level, not pointwise: a CT token and an MR token are allowed to
    sit at different points on the same spectrum. What is forbidden is the two
    modalities using different spectra.
    """
    dev = None
    total, n = torch.zeros(()), 0
    for dyn in dyn_list:
        if len(dyn) < 2:
            continue
        (l0, t0), (l1, t1) = dyn[0], dyn[1]
        dev = l0.device
        if total.device != dev:
            total = total.to(dev)
        k = min(l0.numel(), l1.numel())
        if k < 32:
            continue
        a = torch.sort(l0[:k].float())[0]
        b = torch.sort(l1[:k].float())[0]
        p = torch.sort(t0[:k].float())[0]
        q = torch.sort(t1[:k].float())[0]
        total = total + (a - b).abs().mean() + (p - q).abs().mean()
        n += 1
    if n == 0:
        return torch.zeros(()) if dev is None else torch.zeros((), device=dev), {}
    return total / n, {"dyn_blocks": float(n)}


# ---------------------------------------------------------------------------
# L_probe + L_vc : linear decodability and anti-collapse
# ---------------------------------------------------------------------------
def probe_loss(state: torch.Tensor, labels: torch.Tensor, head: nn.Module,
               fg_weight: float = 10.0) -> torch.Tensor:
    """
    A single linear layer must recover the vessel class from the state alone.

    This is what makes the state usable as attention KEYS. Keys are consumed by
    a dot product, so any class information that is only recoverable through the
    decoder's nonlinearities is information the attention cannot use. Forcing
    linear separability puts the taxonomy where a dot product can find it.
    Background is downweighted 10:1 -- it is ~95% of tokens even after max-pool.
    """
    logits = head(state.flatten(0, 1).float())
    w = torch.full((N_CLASS,), fg_weight, device=state.device)
    w[0] = 1.0
    return F.cross_entropy(logits, labels.flatten(), weight=w)


def variance_covariance_loss(state: torch.Tensor, gamma: float = 1.0
                             ) -> torch.Tensor:
    """
    VICReg variance + covariance terms.

    Without this the alignment term has a trivial optimum: map every token to
    one point, and every cross-modal distance is zero. The variance hinge keeps
    each state dimension above unit std; the decorrelation term stops the state
    collapsing onto one direction. Non-optional -- L_align alone collapses.

    The decorrelation term uses the CORRELATION matrix, not the covariance one.
    Covariance is quadratic in the state and the term squares it, making the
    loss 4th order in |state|: a state that drifts by 10x moves this term by
    10000x, which swamps every other term and overflows long before anything
    else does. Standardising first bounds every entry to [-1, 1] and makes the
    term scale-invariant, so it measures what it is supposed to measure
    (redundancy between dimensions) rather than how large the state happens to
    have grown. The variance hinge still sees the true scale, which is where
    scale genuinely matters.
    """
    z = state.flatten(0, 1).float()
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + 1e-4)
    l_var = F.relu(gamma - std).mean()
    n, c = z.shape
    zn = z / std                                   # unit-variance per dimension
    corr = (zn.t() @ zn) / max(1, n - 1)           # correlation, entries in [-1,1]
    off = corr - torch.diag_embed(torch.diagonal(corr))
    l_cov = off.pow(2).sum() / c
    return l_var + l_cov


# ---------------------------------------------------------------------------
# L_ring + L_mirror : the geometry only a rotational state can hold
# ---------------------------------------------------------------------------
def _batch_prototypes(state: torch.Tensor, labels: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grad-carrying per-class means over the batch. Returns (14,C), (14,) mask."""
    z = F.normalize(state.flatten(0, 1).float(), dim=-1)
    lab = labels.flatten()
    c = z.shape[1]
    sums = torch.zeros(N_CLASS, c, device=z.device, dtype=z.dtype)
    cnts = torch.zeros(N_CLASS, device=z.device, dtype=z.dtype)
    sums.index_add_(0, lab, z)
    cnts.index_add_(0, lab, torch.ones_like(lab, dtype=z.dtype))
    have = cnts > 0
    proto = sums / cnts.clamp(min=1).unsqueeze(1)
    return F.normalize(proto, dim=-1), have


def ring_loss(state: torch.Tensor, labels: torch.Tensor
              ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Make the state geometry of the CoW ring actually circular.

    Assign each of the ten ring vessels its anatomical angle phi_c = 2*pi*k/10
    by position in COW_RING, then require the prototype Gram matrix to match the
    circle's:  <P_u, P_v> = cos(phi_u - phi_v).

    A decay-only real state cannot hold this. Its class prototypes are ordered
    by how recently the scan visited them, which is a line, not a loop; a line
    cannot embed a cycle without tearing it, and the tear lands between the two
    ring classes the raster order separates most -- which is exactly where the
    current model fails (Acom 0.594, R-Pcom 0.472, L-Pcom 0.464 against 0.85 for
    ICA). A rotational state has the loop natively: rotation is periodic.
    """
    proto, have = _batch_prototypes(state, labels)
    ring = torch.tensor(COW_RING, device=state.device)
    present = have[ring]
    if present.sum() < 3:
        return torch.zeros((), device=state.device), {"ring_n": 0.0}
    k = torch.arange(len(COW_RING), device=state.device, dtype=torch.float32)
    phi = 2 * math.pi * k / len(COW_RING)
    tgt = torch.cos(phi[:, None] - phi[None, :])
    gram = proto[ring] @ proto[ring].t()
    m = present[:, None] & present[None, :]
    return ((gram - tgt).pow(2) * m).sum() / m.sum(), {"ring_n": float(present.sum())}


def mirror_loss(state: torch.Tensor, labels: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Bilateral symmetry must be ONE displacement in state space.

    Six of thirteen classes are left/right mirror pairs. If the state encodes
    laterality as a consistent offset, then P[R-x] - P[L-x] is the same vector
    for every pair x, and the loss is just the variance of those difference
    vectors -- no extra parameter to learn. This is the translation-in-phase
    that a rotational state expresses naturally and a leaky accumulator, whose
    only free variable is elapsed decay, cannot.
    """
    proto, have = _batch_prototypes(state, labels)
    diffs = [proto[r] - proto[l] for r, l in COW_MIRROR if have[r] and have[l]]
    if len(diffs) < 2:
        return torch.zeros((), device=state.device), {"mirror_n": 0.0}
    D = torch.stack(diffs)
    return (D - D.mean(0, keepdim=True)).pow(2).sum(-1).mean(), \
           {"mirror_n": float(len(diffs))}
