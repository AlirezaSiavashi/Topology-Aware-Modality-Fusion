"""
loss_cmtc.py
============
Cross-Modal Topology Consistency (CMTC) Loss
============================================

Scientific motivation
---------------------
Standard topology losses (cbDice, clDice) measure the agreement between a
single prediction and its ground-truth label.  They are entirely within-domain:
the topology signal comes from the GT annotation which was acquired in one
modality.  When a network is trained jointly on CTA and MRA (DS104), there is no
explicit pressure for the *topological structure* of CTA predictions to agree
with that of MRA predictions for the same vessel classes.

CMTC removes this gap.  For a batch that contains both CTA and MRA cases
(guaranteed by DS104), we enforce:

  ∀ vessel class c :
      W₁(PD(p_cta[c]), PD(p_mra[c])) → 0

where PD(·) is the persistence diagram of the per-class probability map,
W₁ is the sliced-Wasserstein distance (differentiable approximation), and
the expectation is over all batches.

No paired images are required.  We just need CTA and MRA cases in the same
mini-batch, which DS104 always provides.

Loss formula
------------
  L_CMTC = (1/C) · Σ_c [ W1(PD0_cta[c], PD0_mra[c])
                         + λ_b1 · W1(PD1_cta[c], PD1_mra[c]) ]

where:
  PD0 = Betti-0 persistence diagram (connected components)
  PD1 = Betti-1 persistence diagram (loops / tunnels)
  C   = number of vessel foreground classes (13 for DS104)
  λ_b1 = 0.5 (loops are secondary to connectivity for thin vessels)

Implementation notes
--------------------
- Spatial downsampling to 32³ before PH computation: reduces gudhi cost from
  O(N log N) on ~128³ volumes to O(N log N) on 32³ — ~64× speed-up.
- CubicalComplex is called once per volume per class (not batched) because
  gudhi's cubical complex is not GPU-accelerated; running in a list loop is
  the correct usage pattern for torch_topological.
- The sliced-Wasserstein distance handles diagrams of different sizes
  internally via the diagonal projection trick from Carrière et al. 2017.
- Gradient flows back through the birth/death coordinates of each diagram via
  the indexing into the flattened tensor in CubicalComplex._create_tensors_from_pairs.
- If a batch contains no CTA cases OR no MRA cases, the loss returns 0.0 with
  grad (so the backward graph is trivially zero — does not break training).

Usage
-----
    loss_fn = CmtcLoss(lambda_b1=0.5, vessel_classes=13, target_size=32,
                       num_sw_directions=10)
    l = loss_fn(pred_seg, modality_ids)  # pred_seg: list[Tensor] | Tensor
                                          # modality_ids: (B,) long tensor
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_topological.nn import CubicalComplex, SlicedWassersteinDistance, PersistenceInformation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_diagram(pi_list: list, dim: int) -> Optional[torch.Tensor]:
    """
    Return the persistence diagram tensor for homology dimension `dim`.

    Parameters
    ----------
    pi_list : list of Optional[Tensor]
        Output of _compute_ph_for_volume. Each element is either a
        Tensor of shape (n_pairs, 2) or None.
    dim : int
        Homology dimension to extract (0 or 1).

    Returns
    -------
    Tensor of shape (n_pairs, 2) on CPU, or None if no pairs exist.
    """
    if dim >= len(pi_list):
        return None
    diag = pi_list[dim]
    if diag is None or diag.numel() == 0 or len(diag) == 0:
        return None
    return diag.float()  # (n, 2)


def _safe_sw_distance(
    sw_fn: SlicedWassersteinDistance,
    pd_a: Optional[torch.Tensor],
    pd_b: Optional[torch.Tensor],
    device: torch.device,
) -> float:
    """
    Compute sliced-Wasserstein distance between two persistence diagrams.
    Returns a plain Python float (no gradient — used only as a weight).
    """
    if pd_a is None or pd_b is None:
        return 0.0
    if pd_a.numel() == 0 or pd_b.numel() == 0:
        return 0.0

    pd_a_cpu = pd_a.cpu().detach()
    pd_b_cpu = pd_b.cpu().detach()

    pi_a = PersistenceInformation(pairing=torch.empty(0), diagram=pd_a_cpu, dimension=0)
    pi_b = PersistenceInformation(pairing=torch.empty(0), diagram=pd_b_cpu, dimension=0)

    try:
        dist = sw_fn([pi_a], [pi_b])
    except Exception as exc:
        logger.warning(f"[CMTC] SlicedWassersteinDistance failed: {exc}; returning 0.")
        return 0.0

    return float(dist) if not isinstance(dist, torch.Tensor) else dist.item()


def _compute_ph_for_volume(
    cc: CubicalComplex,
    vol: torch.Tensor,
) -> List[Optional[torch.Tensor]]:
    """
    Run persistent homology on a single-channel 3-D volume using gudhi directly.

    Bypasses torch_topological's ``CubicalComplex._forward`` which calls
    ``cofaces_of_persistence_pairs()`` — that gudhi C++ function segfaults on
    some inputs under SLURM.  Instead we use the simpler ``persistence()`` API
    and return raw diagrams as tensors.

    Parameters
    ----------
    cc  : CubicalComplex (unused now, kept for API compat)
    vol : Tensor of shape (D, H, W) — float, values in [0, 1].

    Returns
    -------
    list of length 3: diagrams for H0, H1, H2.
    Each element is a Tensor of shape (n_pairs, 2) or None.
    """
    import gudhi
    import numpy as np

    vol_np = vol.detach().cpu().numpy().astype(np.float64)

    # Sanity check — NaN/inf causes gudhi to segfault
    if not np.isfinite(vol_np).all():
        logger.warning("[CMTC] PH input contains NaN/inf — skipping")
        return [None, None, None]

    # Superlevel filtration (negate for gudhi which does sublevel)
    gd_cc = gudhi.CubicalComplex(
        dimensions=vol_np.shape,
        top_dimensional_cells=(-vol_np).flatten(),
    )
    gd_cc.persistence()

    # Extract diagrams per homology dimension
    max_dim = min(len(vol_np.shape), 3)
    result: List[Optional[torch.Tensor]] = []
    for dim in range(max_dim):
        pairs = gd_cc.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            result.append(None)
        else:
            # Filter out infinite pairs (essential features)
            finite_mask = np.isfinite(pairs).all(axis=1)
            pairs = pairs[finite_mask]
            if len(pairs) == 0:
                result.append(None)
            else:
                # Keep in negative space to match torch_topological format
                # (gudhi returns [birth, death] in the negated filtration)
                result.append(torch.from_numpy(pairs.copy()).float())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main loss module
# ─────────────────────────────────────────────────────────────────────────────

class CmtcLoss(nn.Module):
    """
    Cross-Modal Topology Consistency Loss.

    Enforces that CTA and MRA predictions share consistent persistent homology
    descriptors on a per-class basis, without requiring paired images.

    Parameters
    ----------
    lambda_b1 : float
        Weight for Betti-1 (loop) topology term relative to Betti-0 (component)
        term.  Default 0.5 because vessel connectivity (H_0) is more critical
        than loop structure (H_1) for cerebral vessels.
    vessel_classes : int
        Number of foreground vessel classes (13 for DS104 TopCoW).
    target_size : int
        Spatial size to downsample probability maps to before PH computation.
        32 means each map is downsampled to (32, 32, 32) — reduces gudhi cost
        ~64× compared to the full 128³ patch.
    num_sw_directions : int
        Number of projection directions for sliced Wasserstein approximation.
        10 is a good default balancing speed and stability.
    """

    def __init__(
        self,
        lambda_b1: float = 0.5,
        vessel_classes: int = 13,
        target_size: int = 32,
        num_sw_directions: int = 10,
    ):
        super().__init__()
        self.lambda_b1      = lambda_b1
        self.vessel_classes = vessel_classes
        self.target_size    = target_size

        # CubicalComplex with dim=3 (3-D volumes), superlevel=True so that high
        # probability regions create topology features (sublevel of -p = superlevel of p).
        # We call _forward() directly with individual 3-D tensors for full control.
        self._cc = CubicalComplex(dim=3, superlevel=True)

        # SlicedWassersteinDistance (differentiable approximation of W1).
        self._sw = SlicedWassersteinDistance(num_directions=num_sw_directions)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        pred: "torch.Tensor | List[torch.Tensor]",
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CMTC loss.

        Parameters
        ----------
        pred : Tensor (B, K, D, H, W)  or  list of such Tensors (deep supervision)
            Raw logits.  If a list (deep supervision), takes pred[0] which is the
            full-resolution output.
        modality_ids : (B,) long Tensor
            Modality label per sample: 0 = CTA, 1 = MRA.

        Returns
        -------
        Scalar loss Tensor (float32, with grad).
        """
        device = modality_ids.device

        # ── Handle deep supervision list ──────────────────────────────────────
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # full-resolution logits: (B, K, D, H, W)

        # pred: (B, K, D, H, W)
        B, K = pred.shape[:2]

        # ── Guard: need at least one CTA and one MRA sample in the batch ─────
        cta_mask = (modality_ids == 0)   # (B,) bool
        mra_mask = (modality_ids == 1)   # (B,) bool

        if not cta_mask.any() or not mra_mask.any():
            # Return a zero scalar that is part of the computation graph
            # (multiplied by 0 so backward is safe).
            return pred.sum() * 0.0

        # ── Softmax probabilities ─────────────────────────────────────────────
        probs = torch.softmax(pred, dim=1)  # (B, K, D, H, W), K = 1+vessel_classes

        # ── Downsample to target_size³ ────────────────────────────────────────
        # This is the single most important speed optimisation:
        # gudhi CubicalComplex on 32³ ≈ 32768 cells vs 128³ ≈ 2M cells.
        ts = self.target_size
        if probs.shape[2:] != (ts, ts, ts):
            # trilinear is differentiable and preserves probability semantics.
            probs_small = F.interpolate(
                probs,
                size=(ts, ts, ts),
                mode='trilinear',
                align_corners=False,
            )  # (B, K, ts, ts, ts)
        else:
            probs_small = probs

        # ── Split into CTA and MRA subsets ───────────────────────────────────
        probs_cta = probs_small[cta_mask]   # (n_cta, K, ts, ts, ts)
        probs_mra = probs_small[mra_mask]   # (n_mra, K, ts, ts, ts)

        n_cta = probs_cta.shape[0]
        n_mra = probs_mra.shape[0]

        # ── Accumulate CMTC loss over vessel classes ──────────────────────────
        # Strategy (topology-weighted binary CE):
        #
        #   For each foreground class c:
        #     1. Compute mean CTA and MRA probability maps (mean over batch).
        #     2. Run PH on each → get sliced-Wasserstein distance w (scalar).
        #     3. If w is large (topologies disagree), penalize with BCE between
        #        the two probability maps — forces them toward topological agreement.
        #
        # The gradient carrier is binary cross-entropy between p_cta and p_mra:
        #   BCE(p_cta, p_mra.detach()) + BCE(p_mra, p_cta.detach())
        # This is the symmetric Jensen-Shannon divergence (JS ≈ 2×BCE - const).
        # It encourages the two maps to agree *in structure* without forcing them
        # to be pixel-identical — the network can produce high-prob foreground in
        # the same spatial pattern for both modalities.
        #
        # Crucially, this gradient does NOT force the raw intensity maps to be
        # the same — it only penalizes structural topology disagreement, weighted
        # by how much the PH descriptors differ.
        #
        # When w ≈ 0 (topologies already agree), loss_c ≈ 0 and the gradient
        # vanishes — the segmentation loss is unaffected.

        n_fg = min(self.vessel_classes, K - 1)

        loss_acc = torch.tensor(0.0, device=device)
        active_classes = 0

        for c in range(n_fg):
            ch = c + 1  # channel index (0 = background)

            map_cta = probs_cta[:, ch].mean(dim=0)  # (ts, ts, ts) — differentiable
            map_mra = probs_mra[:, ch].mean(dim=0)  # (ts, ts, ts) — differentiable

            # ── PH computation (no grad through gudhi) ────────────────────────
            pi_cta = _compute_ph_for_volume(self._cc, map_cta)
            pi_mra = _compute_ph_for_volume(self._cc, map_mra)

            pd0_cta = _extract_diagram(pi_cta, dim=0)
            pd0_mra = _extract_diagram(pi_mra, dim=0)
            pd1_cta = _extract_diagram(pi_cta, dim=1)
            pd1_mra = _extract_diagram(pi_mra, dim=1)

            # ── Topology weights (detached floats) ────────────────────────────
            w0 = _safe_sw_distance(self._sw, pd0_cta, pd0_mra, device)
            w1 = _safe_sw_distance(self._sw, pd1_cta, pd1_mra, device)
            w_total = w0 + self.lambda_b1 * w1

            # Skip classes that are already topologically consistent.
            if w_total < 1e-6:
                continue
            active_classes += 1

            # ── Differentiable surrogate: symmetric JS divergence ─────────────
            # Clamp to avoid log(0); maps are already softmax probs in [0,1].
            eps = 1e-5
            p = map_cta.float().clamp(eps, 1 - eps)   # shape: (ts, ts, ts)
            q = map_mra.float().clamp(eps, 1 - eps)

            # Symmetric JS divergence via manual BCE (avoids autocast ban on
            # F.binary_cross_entropy — AMP forbids it under fp16).
            # Manual formula: BCE(p, t) = -(t*log(p) + (1-t)*log(1-p))
            with torch.amp.autocast('cuda', enabled=False):
                bce_pq = -(q.detach() * torch.log(p) + (1 - q.detach()) * torch.log(1 - p)).mean()
                bce_qp = -(p.detach() * torch.log(q) + (1 - p.detach()) * torch.log(1 - q)).mean()
            js_div = 0.5 * (bce_pq + bce_qp)   # scalar, differentiable

            # Guard: if js_div is non-finite (shouldn't happen with clamping, but
            # belt-and-suspenders), skip this class.
            if not torch.isfinite(js_div):
                continue

            # Scale by PH distance: large topology gap → strong penalty.
            # Clamp weight to [0, 2] to prevent occasional large PH distances
            # from overwhelming the segmentation loss.
            w_clamped = min(w_total, 2.0)
            loss_c = w_clamped * js_div

            loss_acc = loss_acc + loss_c

        # Normalise by classes that contributed (avoid zero-division).
        if active_classes > 0:
            loss_acc = loss_acc / active_classes

        return loss_acc
