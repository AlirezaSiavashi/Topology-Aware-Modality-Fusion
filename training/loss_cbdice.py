"""
loss_cbdice.py
==============
cbDice loss for topology-aware vessel segmentation.

cbDice (Centre-line Dice) extends clDice by weighting the skeleton/mask
intersection by vessel radius, reducing the diameter-imbalance problem
for thin vs thick vessels.

Reference:
  Shit et al. "clDice - a Novel Topology-Preserving Loss Function for
  Tubular Structure Segmentation." CVPR 2021.
  cbDice extension: radius-weighted topology precision/sensitivity.

Interface
---------
  loss_fn = CbDiceLoss(alpha=0.5, smooth=1e-5, use_soft_skel=True)
  loss = loss_fn(pred_logits, target_binary)
  # pred_logits : (B, K, D, H, W)  — raw logits, K classes
  # target_binary: (B, 1, D, H, W) — binary vessel union mask {0,1}

The loss collapses multi-class predictions to a foreground probability
  P_union = max_k(softmax(logits)_k)   for k in foreground classes
before computing topology overlap — consistent with the proposal's
implementation note (Footnote 24).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Soft skeletonisation (morphological, differentiable)
# Based on clDice reference implementation (iterative erosion via min-pooling)
# ─────────────────────────────────────────────────────────────────────────────

def _soft_erode_3d(img: torch.Tensor) -> torch.Tensor:
    """Morphological soft erosion via 3×3×3 min-pooling."""
    # img: (B, 1, D, H, W) in [0,1]
    # Use -max(-img) = min(img) over a local neighbourhood
    p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate_3d(img: torch.Tensor) -> torch.Tensor:
    """Morphological soft dilation via 3×3×3 max-pooling."""
    return F.max_pool3d(img, kernel_size=(3, 3, 3), stride=1, padding=1)


def _soft_open_3d(img: torch.Tensor) -> torch.Tensor:
    """Morphological opening = erode then dilate."""
    return _soft_dilate_3d(_soft_erode_3d(img))


def soft_skeletonize_3d(img: torch.Tensor, n_iter: int = 40) -> torch.Tensor:
    """
    Iterative soft skeletonisation (clDice style).

    Repeatedly removes the outermost layer (erosion) unless it would
    disconnect the structure (= preserved by opening).

    img     : (B, 1, D, H, W) in [0, 1], differentiable
    n_iter  : number of erosion iterations (40 ≈ handles vessels up to
              radius 20 voxels; reduce for speed)
    Returns : skeleton probability map, same shape as img
    """
    skel = torch.zeros_like(img)
    delta = img
    for _ in range(n_iter):
        eroded = _soft_erode_3d(delta)
        opened = _soft_open_3d(delta)
        # "Topological skeleton contribution" at this layer
        layer = F.relu(delta - opened)
        skel = skel + layer * delta
        delta = eroded
        if delta.max() < 1e-4:
            break
    return skel


# ─────────────────────────────────────────────────────────────────────────────
# Radius weighting  (the "cb" part of cbDice)
# ─────────────────────────────────────────────────────────────────────────────

def _radius_weight_map(binary_mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Approximate per-voxel vessel radius via distance transform proxy.
    Since we want a differentiable approximation we use the product of
    successive erosions to estimate depth — alternatively we use the
    soft EDT approximation via iterative erosion depth.

    A simpler and robust approach used here:
      weight(x) = EDT(mask)[x] / (max_EDT + smooth)
    computed with scipy on the CPU, then converted back to tensor.

    Returns a weight map in [0, 1], shape (B, 1, D, H, W).
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np

    w = torch.zeros_like(binary_mask)
    b = binary_mask.shape[0]
    for i in range(b):
        m = binary_mask[i, 0].detach().cpu().numpy().astype(bool)
        dt = distance_transform_edt(m).astype(np.float32)
        dt_max = dt.max()
        if dt_max > 0:
            dt /= dt_max
        w[i, 0] = torch.from_numpy(dt)
    return w.to(binary_mask.device)


# ─────────────────────────────────────────────────────────────────────────────
# clDice / cbDice core
# ─────────────────────────────────────────────────────────────────────────────

def _clDice_precision_sensitivity(
    skel_pred: torch.Tensor,
    skel_true: torch.Tensor,
    mask_pred: torch.Tensor,
    mask_true: torch.Tensor,
    weight: torch.Tensor,
    smooth: float,
) -> tuple:
    """
    Topology precision / sensitivity (weighted).

    T_prec = |skel_pred ∩ mask_true| / |skel_pred|
    T_sens = |skel_true ∩ mask_pred| / |skel_true|

    cbDice weights each skeleton voxel by the local vessel radius.
    """
    # Flatten spatial dims
    sp = skel_pred.flatten(2)    # (B, 1, N)
    st = skel_true.flatten(2)
    mp = mask_pred.flatten(2)
    mt = mask_true.flatten(2)
    w  = weight.flatten(2)

    # Weighted intersections
    wT_prec_num = (sp * mt * w).sum(2)
    wT_prec_den = (sp * w).sum(2) + smooth

    wT_sens_num = (st * mp * w).sum(2)
    wT_sens_den = (st * w).sum(2) + smooth

    wT_prec = wT_prec_num / wT_prec_den
    wT_sens = wT_sens_num / wT_sens_den
    return wT_prec, wT_sens


class CbDiceLoss(nn.Module):
    """
    cbDice loss = 1 - cbDice(P_union, L_union).

    Combines radius-weighted topology precision + sensitivity on the
    soft skeleton of the predicted and ground-truth vessel union masks.

    Parameters
    ----------
    alpha       : weight balancing cbDice vs standard Dice (not used in
                  standalone mode; set to 1.0 for pure cbDice).
    smooth      : numerical stability constant.
    n_skel_iter : soft-skeletonisation iterations (reduce for speed).
    use_radius_weight : if True use radius-based cbDice; else plain clDice.
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        n_skel_iter: int = 10,
        use_radius_weight: bool = True,
    ):
        super().__init__()
        self.smooth         = smooth
        self.n_skel_iter    = n_skel_iter
        self.use_radius_weight = use_radius_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_union: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_logits  : (B, K, D, H, W) — raw logits for K classes
                       (0 = background, 1..K-1 = foreground vessels)
        target_union : (B, 1, D, H, W) — binary vessel union mask {0,1}

        Returns
        -------
        scalar loss in [0, 2]  (typically ≈ [0, 1])
        """
        # ── collapse to foreground probability ────────────────────────────
        probs = torch.softmax(pred_logits, dim=1)          # (B, K, ...)
        # Union: max over all foreground channels
        if probs.shape[1] > 1:
            p_union = probs[:, 1:].max(dim=1, keepdim=True)[0]  # (B, 1, ...)
        else:
            p_union = probs                                  # binary case

        l_union = target_union.float()

        # ── skeleton maps ─────────────────────────────────────────────────
        skel_pred = soft_skeletonize_3d(p_union, self.n_skel_iter)
        skel_true = soft_skeletonize_3d(l_union, self.n_skel_iter)

        # ── radius weighting ──────────────────────────────────────────────
        if self.use_radius_weight:
            w = _radius_weight_map(l_union, smooth=1.0) + self.smooth
        else:
            w = torch.ones_like(l_union)

        # ── cbDice ────────────────────────────────────────────────────────
        wT_prec, wT_sens = _clDice_precision_sensitivity(
            skel_pred, skel_true, p_union, l_union, w, self.smooth
        )
        cbdice = (2 * wT_prec * wT_sens) / (wT_prec + wT_sens + self.smooth)

        return 1.0 - cbdice.mean()
