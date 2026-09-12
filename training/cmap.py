"""
cmap.py
=======
Cross-Modal Anatomical Priors (CMAP) for modality-agnostic Circle-of-Willis
segmentation.

Three components, each a deliberate departure from the scalar/global
formulation used for audio-visual classification:

1. ClasswisePriorBank
   K prototypes on the Poincare ball, one per output class, instead of a single
   global cross-modal vector.  Segmentation is per-class, so a single prior
   cannot express "CTA is authoritative for ICA, MRA for Pcom".  Prototypes are
   stored as tangent vectors at the origin and mapped to the ball on access, so
   the optimiser only ever sees an unconstrained Euclidean parameter.

2. ReliabilityMatrix
   An (n_modalities x K) matrix, not a scalar per modality.  Updated from
   held-out per-class Dice at the end of each validation epoch rather than from
   logit confidence -- for dense prediction, mean logit confidence is dominated
   by background voxels and is nearly uninformative about a 0.5%-volume vessel
   class.  Dice is the quantity we actually care about and is already computed.

3. GatedPriorFusion + radius weighting
   Per-voxel element-wise gate over the fused feature, plus the radial weight
   rho = 1/(r + eps).  On the Poincare ball the radius has a concrete
   anatomical reading here: the origin is where CTA and MRA agree (proximal,
   large-calibre ICA/BA), the boundary is where they diverge (distal, small
   Acom/Pcom).  rho therefore drives prior learning from the shared trunk of
   the vessel tree rather than from modality-specific distal detail.

Voxel assignment to prototypes is done by hyperbolic distance
(softmax(-d_H/tau)), so the bank doubles as a classifier and the geometry is
directly testable -- see `RadiusStats`, which accumulates the mean radius per
anatomical class for the hierarchy analysis.

Memory
------
Distance and assignment tensors are (B, D, H, W, K).  At full 112x160x128
resolution with K=14 that is ~240 MB per tensor per copy, which will not
survive alongside nnU-Net's activations.  The projection/assignment therefore
runs at `prior_stride` (default 2), and only the prior mixture is upsampled
back to full resolution for the gate.  At 0.5 mm spacing that is a 1 mm prior
lattice, still finer than the calibre of the smallest CoW segments.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# nnU-Net imports trainers by package path, so this module's directory is not
# necessarily on sys.path when it is loaded. Make the flat imports below work
# regardless of where the file has been copied to.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperbolic import (
    dist,
    dist_to_prototypes,
    expmap0,
    logmap0,
    project,
)


# ── Prototype bank ────────────────────────────────────────────────────────────

def clip_tangent(v: torch.Tensor, max_norm: torch.Tensor,
                 dim: int = -1) -> torch.Tensor:
    """
    Radially clip a tangent vector to ||v|| <= max_norm, leaving shorter
    vectors untouched.

    This matters more than it looks.  expmap0 applies tanh to the tangent
    norm, and tanh(4) = 0.9993: every voxel whose projection has norm above
    ~4 lands on the same boundary shell and becomes radially indistinguishable
    from every other one.  Since we read the radius as an anatomical coordinate
    (proximal vs distal), letting the projection saturate would silently
    collapse the very signal the prior is built on.  Clipping to ~2.5 keeps the
    usable radial range at roughly [0, 0.99] while staying monotone in ||v||.
    """
    norm = v.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-6)
    scale = torch.clamp(max_norm.abs() / norm, max=1.0)
    return v * scale


R_MAX = 0.99   # atanh(1) is infinite; keep the attainable radius just inside


def polar_to_ball(direction: torch.Tensor, radius_logit: torch.Tensor,
                  c: float = 1.0) -> torch.Tensor:
    """
    Build a ball point from a direction and a radius logit.

    r = R_MAX * sigmoid(radius_logit), and the tangent vector is
    direction_unit * atanh(r), so expmap0 returns a point whose norm is
    *exactly* r. Radius therefore becomes an explicit, directly supervisable
    coordinate rather than a by-product of the projection's magnitude.

    This replaces "conv output, then clip the tangent norm", which failed twice
    for different reasons. Clipping pins every voxel to the clip once the
    conv's norms exceed it (measured radial spread 3e-4 across all classes).
    Raising the clip does not fix the conditioning either: with clip 3.0,
    sigmoid(0)=0.5 already maps to radius 0.905, so almost the whole sigmoid
    range lands above 0.9 and the depth targets are crammed into its lower
    tail. Here the depth targets 0.35/0.60/0.85 sit at sigmoid 0.354/0.606/
    0.859 -- the well-conditioned middle.
    """
    dirn = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    r = (R_MAX * torch.sigmoid(radius_logit)).clamp(1e-4, R_MAX)
    return expmap0(dirn * torch.atanh(r), c)


class ClasswisePriorBank(nn.Module):
    """K learnable prototypes living on the Poincare ball."""

    def __init__(self, num_classes: int, embed_dim: int, c: float = 1.0,
                 init_scale: float = 1.0, tangent_clip: float = 1.5):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.c = c
        # Direction and radius are separate parameters so the tree anchor acts
        # on a dedicated scalar per class instead of competing with the
        # discriminative objective over the whole embedding vector.
        self.direction = nn.Parameter(torch.randn(num_classes, embed_dim) * init_scale)
        self.radius_logit = nn.Parameter(torch.zeros(num_classes))

    def prototypes(self) -> torch.Tensor:
        """(K, E) points on the ball."""
        return polar_to_ball(self.direction, self.radius_logit.unsqueeze(-1), self.c)

    def prototypes_tangent(self) -> torch.Tensor:
        """(K, E) the same prototypes read back in the tangent space."""
        return logmap0(self.prototypes(), self.c)


# ── Reliability ───────────────────────────────────────────────────────────────

class ReliabilityMatrix(nn.Module):
    """
    Per-class, per-modality reliability, driven by held-out Dice.

    alpha[m, k] = Dice[m, k] / sum_m' Dice[m', k]

    so for each anatomical class the modalities' weights sum to 1.  Before the
    first validation epoch alpha is uniform, mirroring the "t > 0" guard in
    epoch-wise rebalancing schemes.
    """

    def __init__(self, num_modalities: int = 2, num_classes: int = 14,
                 momentum: float = 0.5, floor: float = 1e-3):
        super().__init__()
        self.momentum = momentum
        self.floor = floor
        self.register_buffer(
            "alpha", torch.full((num_modalities, num_classes), 1.0 / num_modalities))
        self.register_buffer("dice", torch.zeros(num_modalities, num_classes))
        self.register_buffer("initialised", torch.zeros(1))

    @torch.no_grad()
    def update_from_dice(self, dice: torch.Tensor) -> None:
        """dice: (n_modalities, K) held-out Dice, NaN where a class is absent."""
        dice = torch.nan_to_num(dice.to(self.alpha.device).float(), nan=0.0)
        self.dice.copy_(dice)
        s = dice.clamp_min(self.floor)
        target = s / s.sum(dim=0, keepdim=True).clamp_min(self.floor)
        if self.initialised.item() == 0:
            self.alpha.copy_(target)
            self.initialised.fill_(1)
        else:
            self.alpha.mul_(1.0 - self.momentum).add_(self.momentum * target)

    def weights_for(self, modality_ids: torch.Tensor,
                    class_ids: torch.Tensor) -> torch.Tensor:
        """Gather alpha[m_i, y_i] for a flat set of sampled voxels."""
        return self.alpha[modality_ids, class_ids]


# ── Gated fusion ──────────────────────────────────────────────────────────────

class GatedPriorFusion(nn.Module):
    """
    f_out = w * f + (1 - w) * p,  w = sigmoid(MLP([f; p])) elementwise in [0,1].

    The final gate conv is biased positive at init so w starts near 1, i.e. the
    module begins as a near-identity on the backbone features and has to earn
    any deviation.  Combined with delayed activation this is what keeps the
    prior from overwriting a well-trained modality's feature space.
    """

    def __init__(self, feat_channels: int, embed_dim: int, hidden: int = 32,
                 init_gate_bias: float = 3.0):
        super().__init__()
        self.to_feat = nn.Conv3d(embed_dim, feat_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv3d(feat_channels * 2, hidden, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, feat_channels, kernel_size=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_gate_bias)

    def forward(self, feat: torch.Tensor,
                prior_tangent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.to_feat(prior_tangent)
        w = torch.sigmoid(self.gate(torch.cat([feat, p], dim=1)))
        return w * feat + (1.0 - w) * p, w


# ── Radius statistics (the hierarchy evidence) ────────────────────────────────

class RadiusStats:
    """Accumulates mean Poincare radius per class, per modality."""

    def __init__(self, num_classes: int, num_modalities: int = 2):
        self.n_cls = num_classes
        self.n_mod = num_modalities
        self.reset()

    def reset(self) -> None:
        self._sum = torch.zeros(self.n_mod, self.n_cls, dtype=torch.float64)
        self._cnt = torch.zeros(self.n_mod, self.n_cls, dtype=torch.float64)

    @torch.no_grad()
    def update(self, radii: torch.Tensor, class_ids: torch.Tensor,
               modality_ids: torch.Tensor) -> None:
        idx = (modality_ids * self.n_cls + class_ids).cpu()
        self._sum.view(-1).index_add_(0, idx, radii.double().cpu())
        self._cnt.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))

    def mean_radius(self) -> torch.Tensor:
        """(n_modalities, K), NaN where a class was never observed."""
        out = self._sum / self._cnt.clamp_min(1e-9)
        return torch.where(self._cnt > 0, out, torch.full_like(out, float("nan")))


# ── The module ────────────────────────────────────────────────────────────────

class CrossModalPriorModule(nn.Module):
    """
    Hyperbolic projection + prototype bank + gated fusion.

    forward() returns the fused feature map and a dict of intermediates that
    the trainer turns into the prior loss.  Nothing here depends on modality
    identity, so inference needs only the image -- the reliability matrix is a
    training-time quantity only.
    """

    def __init__(
        self,
        feat_channels: int,
        num_classes: int,
        embed_dim: int = 16,
        curvature: float = 1.0,
        tau: float = 0.5,
        prior_stride: int = 2,
        gate_hidden: int = 32,
        num_modalities: int = 2,
        tangent_clip: float = 1.5,
        use_hyperbolic: bool = True,
    ):
        super().__init__()
        self.c = curvature
        self.tau = tau
        self.prior_stride = prior_stride
        self.num_classes = num_classes
        # Euclidean control. Swapping the distance function alone is the clean
        # ablation: the polar parameterisation, the depth anchors and the
        # radial matching all stay identical, so any difference is
        # attributable to the geometry rather than to a change of radial
        # scale. (Lowering the curvature instead would also shift the radius
        # scale -- sigmoid(0) maps to 0.495 at c=1 but 0.543 at c=1e-4 -- and
        # would confound the two.)
        self.use_hyperbolic = use_hyperbolic

        # Direction and radius are predicted separately. A single conv followed
        # by a norm clip does not work: the conv's output norms sit above the
        # clip, so every voxel is scaled to exactly the clip and the radial
        # coordinate becomes a constant (measured spread 3e-4 across all
        # classes, background included). Splitting them makes radius a free,
        # directly-supervisable quantity that the prior loss can drag toward
        # the prototype's radius.
        self.proj = nn.Conv3d(feat_channels, embed_dim, kernel_size=1)
        self.radius_head = nn.Conv3d(feat_channels, 1, kernel_size=1)
        self.bank = ClasswisePriorBank(num_classes, embed_dim, c=curvature,
                                       tangent_clip=tangent_clip)
        self.reliability = ReliabilityMatrix(num_modalities, num_classes)
        self.fusion = GatedPriorFusion(feat_channels, embed_dim, hidden=gate_hidden)

    def forward(self, feat: torch.Tensor, fuse_active: bool = True
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s = self.prior_stride
        f_low = F.avg_pool3d(feat, kernel_size=s, stride=s) if s > 1 else feat

        v = self.proj(f_low)                                   # (B,E,d,h,w)
        v = v.permute(0, 2, 3, 4, 1).contiguous().float()      # (B,d,h,w,E)
        rl = self.radius_head(f_low)                           # (B,1,d,h,w)
        rl = rl.permute(0, 2, 3, 4, 1).contiguous().float()    # (B,d,h,w,1)

        # Direction from proj, radius from radius_head, combined so that
        # ||z|| == R_MAX * sigmoid(rl) exactly. Same parameterisation as the
        # prototypes, so the two populations share one radial scale and the
        # matching term is always satisfiable -- previously voxels topped out
        # at 0.952 while depth-2 prototypes sat at 0.987, i.e. the target was
        # outside the attainable range.
        z = polar_to_ball(v, rl, self.c)                       # on the ball

        protos = self.bank.prototypes()                        # (K,E)
        if self.use_hyperbolic:
            d = dist_to_prototypes(z, protos, self.c)          # (B,d,h,w,K)
        else:
            d = torch.cdist(z.reshape(-1, z.shape[-1]), protos).reshape(
                *z.shape[:-1], protos.shape[0])                # Euclidean control
        assign = torch.softmax(-d / self.tau, dim=-1)

        # Prior mixture: convex combination in the tangent space at the origin,
        # then re-projected.  Exact Frechet means on the ball need an inner
        # iteration per voxel, which is not affordable at this density; the
        # tangent-space mean is the standard first-order surrogate.
        p_tan = torch.matmul(assign, self.bank.prototypes_tangent())   # (B,d,h,w,E)
        p_tan = p_tan.permute(0, 4, 1, 2, 3).contiguous()              # (B,E,d,h,w)
        if s > 1:
            p_tan = F.interpolate(p_tan, size=feat.shape[2:],
                                  mode="trilinear", align_corners=False)

        if fuse_active:
            out, gate = self.fusion(feat, p_tan.to(feat.dtype))
        else:
            out, gate = feat, None

        return out, {"z": z, "dist": d, "assign": assign, "gate": gate}


# ── Losses ────────────────────────────────────────────────────────────────────

def sample_voxels(target: torch.Tensor, max_fg: int = 8192,
                  bg_ratio: float = 0.25) -> torch.Tensor:
    """
    Pick voxel indices for the prior loss.

    target : (B, d, h, w) long
    returns: (N, 4) index tensor [b, d, h, w]

    Vessels occupy well under 1% of a patch, so uniform sampling would return
    almost pure background.  We take all foreground up to a cap plus a small
    background quota to keep the background prototype grounded.
    """
    fg = (target > 0).nonzero(as_tuple=False)
    bg = (target == 0).nonzero(as_tuple=False)

    if fg.numel() and fg.shape[0] > max_fg:
        fg = fg[torch.randperm(fg.shape[0], device=fg.device)[:max_fg]]

    n_bg = int(max(1, fg.shape[0] * bg_ratio)) if fg.numel() else max_fg
    if bg.numel():
        n_bg = min(n_bg, bg.shape[0])
        bg = bg[torch.randperm(bg.shape[0], device=bg.device)[:n_bg]]
        return torch.cat([fg, bg], dim=0) if fg.numel() else bg
    return fg


def prior_loss(
    z: torch.Tensor,
    dist: torch.Tensor,
    target_low: torch.Tensor,
    reliability: ReliabilityMatrix,
    modality_ids: torch.Tensor,
    tau: float = 0.5,
    eps: float = 1e-2,
    max_fg: int = 8192,
    radius_stats: Optional[RadiusStats] = None,
    proto_radii: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Radius- and reliability-weighted prototype cross-entropy.

        L = mean_i  alpha[m_i, y_i] * rho_i * CE( -d_H(z_i, .)/tau , y_i )

    with rho_i = 1/(||z_i|| + eps), batch-normalised to mean 1.

    Using a full softmax over hyperbolic distances (rather than a per-class
    sigmoid on a dot product) makes the same quantity serve as both the
    training signal and the voxel-to-prototype assignment used at fusion time,
    so the learned geometry is the thing actually being used downstream.
    """
    idx = sample_voxels(target_low, max_fg=max_fg)
    if idx.numel() == 0:
        zero = z.sum() * 0.0
        return zero, zero, {}

    b, dd, hh, ww = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    z_sel = z[b, dd, hh, ww]                       # (N,E)
    d_sel = dist[b, dd, hh, ww]                    # (N,K)
    y_sel = target_low[b, dd, hh, ww].long()       # (N,)
    m_sel = modality_ids[b].long()                 # (N,)

    logp = torch.log_softmax(-d_sel / tau, dim=-1)
    ce = -logp.gather(1, y_sel.unsqueeze(1)).squeeze(1)

    r = z_sel.norm(dim=-1)
    rho = 1.0 / (r + eps)
    rho = rho / rho.mean().clamp_min(1e-6)

    w = reliability.weights_for(m_sel, y_sel)

    loss = (w * rho * ce).mean()

    # Radial matching: pull each voxel to the radius of its own class
    # prototype.
    #
    # This term is not optional, and the reason is structural. The
    # cross-entropy above depends only on *relative* distances, so it is
    # minimised just as well by aligning direction and inflating radius --
    # larger radius buys larger margins for free. Measured, voxels drift to
    # r~0.95 regardless of whether their prototype sits at 0.67 or 0.92, and
    # the depth ordering in the voxel field stays flat (0.949/0.949/0.964)
    # even when the prototypes are correctly ordered. Nothing in a
    # distance-based classification objective ever ties a point's radius to
    # its class, so the correspondence has to be stated.
    match = z.sum() * 0.0
    if proto_radii is not None:
        match = ((r - proto_radii.to(r.device)[y_sel]) ** 2).mean()

    if radius_stats is not None:
        radius_stats.update(r.detach(), y_sel.detach(), m_sel.detach())

    with torch.no_grad():
        stats = {
            "n_vox": float(idx.shape[0]),
            "mean_radius": float(r.mean()),
            "mean_rho": float(rho.mean()),
            "r_match": float(match),
        }
    return loss, match, stats


# ── Circle-of-Willis anatomical tree ─────────────────────────────────────────
#
# Label values are Dataset104's (background = 0):
#   1 BA   2 R-PCA  3 L-PCA  4 R-ICA  5 R-MCA   6 L-ICA   7 L-MCA
#   8 R-Pcom  9 L-Pcom  10 Acom  11 R-ACA  12 L-ACA  13 3rd-A2
#
# (parent, child) with parent strictly proximal to child in the arterial tree.
COW_TREE_EDGES = (
    (1, 2), (1, 3),        # basilar -> posterior cerebrals
    (4, 5), (4, 11),       # right ICA -> right MCA, right ACA
    (6, 7), (6, 12),       # left  ICA -> left  MCA, left  ACA
    (4, 8), (6, 9),        # ICAs -> posterior communicating arteries
    (11, 10), (12, 10),    # ACAs -> anterior communicating artery
    (11, 13), (12, 13),    # ACA complex -> third A2 variant
)

# Depth is informational (logging / analysis); the loss uses edges only.
COW_DEPTH = {1: 0, 4: 0, 6: 0,
             2: 1, 3: 1, 5: 1, 7: 1, 11: 1, 12: 1,
             8: 2, 9: 2, 10: 2, 13: 2}


# Target Poincare radius per tree depth. These are anchors, not free
# parameters: prototypes and voxel embeddings must occupy the same radial band
# or every voxel-to-prototype distance is dominated by the radial gap between
# the two populations, the assignment softmax goes flat, and no gradient
# differentiates the classes. An earlier version used only an ordering margin
# plus a parent-child proximity term; proximity is minimised by collapsing all
# prototypes to the origin, which put prototypes at r~0.15 against voxels at
# r~0.90 and killed the signal entirely.
COW_DEPTH_RADIUS = {0: 0.35, 1: 0.60, 2: 0.85}


def tree_loss(bank: "ClasswisePriorBank", edges=COW_TREE_EDGES,
              margin: float = 0.05, beta: float = 1.0,
              c: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Impose the known Circle-of-Willis hierarchy on the prototype bank.

    Two terms, both over prototypes only (14 x E parameters -- negligible cost):

      ordering   mean relu(r_parent - r_child + margin)
                 a parent must sit strictly closer to the origin than its
                 child, so the radial coordinate encodes branch order.

      proximity  mean d_H(p_parent, p_child)
                 keeps each subtree coherent rather than scattering children
                 around the shell at the right radius.

    Why this is supervised rather than emergent: an earlier version left the
    hierarchy to arise on its own from per-voxel prototype cross-entropy. It
    does not, and cannot -- that objective depends only on relative distances
    and is invariant to rotation, so it has no reason to order classes
    radially. Measured, all 14 classes converged to radius 0.631-0.640
    (spread 0.009, background indistinguishable from vessels). Hyperbolic tree
    embeddings in the literature are trained on explicit parent-child pairs for
    exactly this reason.

    Only the *partial order* is imposed, not target radii, so the absolute
    scale stays free and the claim under test remains falsifiable: whether
    embedding the known tree in hyperbolic space beats the Euclidean control.
    """
    p = bank.prototypes()                       # (K, E) on the ball
    r = p.norm(dim=-1)                          # (K,)

    par = torch.tensor([e[0] for e in edges], device=p.device, dtype=torch.long)
    chi = torch.tensor([e[1] for e in edges], device=p.device, dtype=torch.long)

    ordering = torch.relu(r[par] - r[chi] + margin).mean()

    # Radial anchoring to the depth targets. Replaces the parent-child
    # proximity term, which was degenerate (minimised at the origin).
    idx = torch.tensor(sorted(COW_DEPTH), device=p.device, dtype=torch.long)
    tgt = torch.tensor([COW_DEPTH_RADIUS[COW_DEPTH[int(k)]] for k in idx],
                       device=p.device, dtype=r.dtype)
    anchor = ((r[idx] - tgt) ** 2).mean()

    loss = ordering + beta * anchor
    with torch.no_grad():
        viol = float((r[par] >= r[chi]).float().mean())
        stats = {
            "tree_order": float(ordering),
            "tree_anchor": float(anchor),
            "tree_violations": viol,
        }
    return loss, stats


def radius_regulariser(z: torch.Tensor) -> torch.Tensor:
    """
    Mean Poincare radius of the embedded field -- a mild inward pull.

    Why this is needed rather than optional: geodesic distance on the ball
    diverges as ||z|| -> 1, so a distance-based cross-entropy can widen every
    class margin for free simply by inflating the radius. Left alone it does
    exactly that -- an untuned run puts all 14 classes in [0.913, 0.961],
    background included, and the radial coordinate carries no anatomy at all.
    That is the standard boundary-crowding pathology of hyperbolic embeddings,
    and here it would silently void the one claim the radius is supposed to
    support.

    With an inward pull, radius becomes a contested resource: a class only
    moves outward when the separation it buys outweighs the penalty. The
    proximal/distal ordering then has to be *earned* by the optimisation
    instead of assumed by the architecture, which is also what makes it a
    testable prediction rather than a design decision.
    """
    return z.float().norm(dim=-1).mean()


def gate_regulariser(gate: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Penalise how far the gate departs from the identity, i.e. mean(1 - w).

    This is the "do no harm" term: the prior has to buy its way in against a
    constant cost, which is what stops a well-trained modality's features from
    being diluted by an under-trained prior.  It replaces the batch-similarity
    consistency objective used in the classification setting, which assumes
    paired samples inside a mini-batch -- not available here, since patches are
    drawn from unregistered volumes of different subjects.
    """
    if gate is None:
        return torch.tensor(0.0)
    return (1.0 - gate).mean()
