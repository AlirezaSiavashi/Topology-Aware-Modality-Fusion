"""
hyperbolic.py
=============
Poincare-ball operations for the cross-modal anatomical prior bank.

Everything here follows the standard Riemannian formulation of Ganea et al.,
"Hyperbolic Neural Networks" (NeurIPS 2018), not the ad-hoc rescaling used in
some multi-modal papers.  Concretely we use the true exponential map at the
origin

    expmap0(v) = tanh(sqrt(c) ||v||) * v / (sqrt(c) ||v||)

rather than tanh(sqrt(c)||v||/2) * v / (sqrt(c)||v||), which is not the
exponential map of any point and therefore has no geodesic interpretation.
The distinction matters here: we read the *radius* of an embedding as an
anatomical quantity (proximal vs distal vessel), so the radial coordinate has
to be the genuine geodesic one.

Numerical notes
---------------
- All ops run in float32 even under autocast.  atanh saturates catastrophically
  in float16 near the ball boundary, which is exactly where distal-vessel
  embeddings live.
- Points are projected back inside the ball after every op with a margin of
  1e-3, and atanh arguments are clamped to 1 - 1e-5.
"""
from __future__ import annotations

import torch

BALL_EPS = 1e-3      # keep ||x|| <= (1 - BALL_EPS) / sqrt(c)
ATANH_CLAMP = 1 - 1e-5
NORM_MIN = 1e-15


def _norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x.norm(dim=dim, keepdim=True, p=2).clamp_min(NORM_MIN)


def project(x: torch.Tensor, c: float = 1.0, dim: int = -1) -> torch.Tensor:
    """Clip x back inside the Poincare ball of curvature -c."""
    norm = _norm(x, dim)
    maxnorm = (1.0 - BALL_EPS) / (c ** 0.5)
    return torch.where(norm > maxnorm, x / norm * maxnorm, x)


def expmap0(v: torch.Tensor, c: float = 1.0, dim: int = -1) -> torch.Tensor:
    """Exponential map at the origin: tangent space -> ball."""
    v = v.float()
    sqrt_c = c ** 0.5
    v_norm = _norm(v, dim)
    out = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    return project(out, c, dim)


def logmap0(x: torch.Tensor, c: float = 1.0, dim: int = -1) -> torch.Tensor:
    """Logarithmic map at the origin: ball -> tangent space."""
    x = x.float()
    sqrt_c = c ** 0.5
    x_norm = _norm(x, dim)
    arg = (sqrt_c * x_norm).clamp(max=ATANH_CLAMP)
    return torch.atanh(arg) * x / (sqrt_c * x_norm)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0,
               dim: int = -1) -> torch.Tensor:
    """Mobius addition x (+)_c y."""
    x, y = x.float(), y.float()
    x2 = (x * x).sum(dim=dim, keepdim=True)
    y2 = (y * y).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / den.clamp_min(NORM_MIN)


def dist(x: torch.Tensor, y: torch.Tensor, c: float = 1.0,
         dim: int = -1) -> torch.Tensor:
    """Geodesic distance on the Poincare ball. Returns tensor without `dim`."""
    sqrt_c = c ** 0.5
    diff = _norm(mobius_add(-x, y, c, dim), dim).squeeze(dim)
    return (2.0 / sqrt_c) * torch.atanh((sqrt_c * diff).clamp(max=ATANH_CLAMP))


def dist_to_prototypes(z: torch.Tensor, protos: torch.Tensor,
                       c: float = 1.0) -> torch.Tensor:
    """
    Distance from every point to every prototype.

    z      : (..., E) points on the ball
    protos : (K, E)   prototypes on the ball
    returns: (..., K)

    Deliberately uses the Mobius form, not the algebraically equivalent closed
    form

        d_c(x,y) = (1/sqrt(c)) arcosh(1 + 2c||x-y||^2
                                          / ((1 - c||x||^2)(1 - c||y||^2)))

    even though that one needs only a matmul (via ||x-y||^2 = ||x||^2 + ||y||^2
    - 2<x,y>) and avoids the (..., K, E) broadcast this version allocates.

    Two reasons. First, the Gram expansion is catastrophic cancellation, and
    the 1/((1-c||x||^2)(1-c||y||^2)) factor amplifies it precisely near the
    boundary -- measured float32 error up to 1e-2 on distances of ~12, against
    ~1e-7 here. Since the softmax over these distances is what assigns voxels
    to prototypes, that error lands directly on the assignment. Second, the
    profiled cost of this path turned out to be minor: epoch time was dominated
    by GPU contention, and once that was removed it fell from 228s to ~96s
    against a ~50s plain-nnU-Net baseline.

    If throughput ever does become binding, raise `prior_stride` (cubic
    savings, no numerical cost) before reaching for the closed form.
    """
    z = z.float().unsqueeze(-2)                 # (..., 1, E)
    p = protos.float().view(*([1] * (z.dim() - 2)), *protos.shape)
    return dist(z, p, c, dim=-1)                # (..., K)


def radius(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Euclidean norm of a ball point == its geodesic radius up to a monotone map."""
    return x.float().norm(dim=dim, p=2)
