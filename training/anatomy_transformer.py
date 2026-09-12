"""
anatomy_transformer.py
======================
Thirteen anatomy-anchored vessel queries that cross-attend over the SSM state,
with the state supplying the KEYS.

The point of keying on the state
--------------------------------
Attention retrieves by dot product against keys. If the vessel identity in the
state is only recoverable through the decoder's nonlinearities, the attention
cannot use it -- which is why `probe_loss` (linear decodability) and this module
are two halves of one design, not two independent additions. Keys are
K = W_K h, so the property "h is a good key" is exactly the property
`probe_loss` enforces and `attention_alignment_loss` below verifies and trains.

What the queries add that the U-Net cannot
------------------------------------------
Each query owns one CoW segment for the whole patch, so it can answer a
question the voxel decoder has no mechanism for: *is this vessel present at all
in this subject?* That matters here specifically -- R-Pcom is absent in 64/125
subjects, L-Pcom in 69/125, 3rd-A2 in 109/125 -- and those are precisely the
three worst classes (0.472 / 0.464 / 0.314 Dice). A per-voxel softmax has to
rediscover absence from scratch in every patch; a presence head states it once.

Query self-attention carries an anatomical adjacency bias from the CoW edge
list, so a query for Acom is primed to attend to the R-ACA and L-ACA queries it
physically connects to.

Fusion is zero-initialised
--------------------------
The fusion weights start at zero, so at epoch 0 the network's output is exactly
the nnU-Net baseline's and the transformer earns its contribution rather than
being handed it. Same reasoning as the zero-init gamma on the SSM residual: the
backbone is the control and must not be perturbed before the new path is
trained.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# nnU-Net's recursive_find_python_class imports every module in this folder
# directly, so a sibling import only resolves if this directory is on the path
# before the import runs. Same guard as mfssm_net.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state_losses import COW_NAMES, N_CLASS

# CoW adjacency over the 13 query slots (0-indexed over COW_NAMES).
COW_EDGES = ((0, 1), (0, 2), (1, 7), (2, 8), (3, 7), (5, 8), (3, 4), (5, 6),
             (3, 10), (5, 11), (10, 9), (11, 9), (10, 12), (11, 12),
             (1, 2), (4, 6), (7, 8), (10, 11), (3, 5))


class StateKeyedAttention(nn.Module):
    """Queries attend over state tokens. Returns context and the attention map."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.h = n_heads
        self.dh = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)          # <- keys from state
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q_in (B,Nq,C), kv (B,L,C) -> (context (B,Nq,C), attn (B,Nq,L))."""
        b, nq, c = q_in.shape
        l = kv.shape[1]
        q = self.q(q_in).view(b, nq, self.h, self.dh).transpose(1, 2)
        k = self.k(kv).view(b, l, self.h, self.dh).transpose(1, 2)
        v = self.v(kv).view(b, l, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)      # (b,h,nq,l)
        att = att.softmax(-1)
        ctx = (self.drop(att) @ v).transpose(1, 2).reshape(b, nq, c)
        return self.o(ctx), att.mean(1)                            # attn head-avg


class AnatomyTransformer(nn.Module):
    """
    n_layers of [cross-attend over state] -> [query self-attention with
    anatomical bias] -> [FFN], then presence and spatial-prior heads.
    """

    def __init__(self, state_dim: int, dim: int = 256, n_layers: int = 3,
                 n_heads: int = 8, n_class: int = N_CLASS, dropout: float = 0.1):
        super().__init__()
        self.n_q = n_class - 1                                     # 13 vessels
        self.n_class = n_class
        self.dim = dim

        self.query = nn.Parameter(torch.randn(self.n_q, dim) * 0.02)
        self.kv_proj = nn.Linear(state_dim, dim, bias=False)
        # Factored 3D positional encoding, interpolated to whatever token grid
        # the patch produces, so the module is patch-size agnostic.
        self.pos_d = nn.Parameter(torch.randn(1, dim, 16) * 0.02)
        self.pos_h = nn.Parameter(torch.randn(1, dim, 16) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(1, dim, 16) * 0.02)

        self.cross = nn.ModuleList([StateKeyedAttention(dim, n_heads, dropout)
                                    for _ in range(n_layers)])
        self.selfat = nn.ModuleList([nn.MultiheadAttention(dim, n_heads, dropout,
                                                           batch_first=True)
                                     for _ in range(n_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim)) for _ in range(n_layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.n3 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

        # anatomical adjacency bias for query self-attention
        adj = torch.full((self.n_q, self.n_q), float("-inf"))
        adj.fill_diagonal_(0.0)
        for u, v in COW_EDGES:
            adj[u, v] = 0.0
            adj[v, u] = 0.0
        # -inf would hard-mask; a learnable scale lets the model soften it
        self.register_buffer("adj_mask", (adj == 0).float())
        self.adj_scale = nn.Parameter(torch.tensor(0.0))

        self.presence = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        # Zero-initialised fusion: output starts identical to plain nnU-Net.
        self.w_prior = nn.Parameter(torch.zeros(self.n_q))
        self.w_presence = nn.Parameter(torch.zeros(self.n_q))

    def _posenc(self, grid, device, dtype):
        d, h, w = grid
        pd = F.interpolate(self.pos_d, size=d, mode="linear", align_corners=False)
        ph = F.interpolate(self.pos_h, size=h, mode="linear", align_corners=False)
        pw = F.interpolate(self.pos_w, size=w, mode="linear", align_corners=False)
        p = (pd[..., :, None, None] + ph[..., None, :, None] + pw[..., None, None, :])
        return p.flatten(2).transpose(1, 2).to(dtype)              # (1, L, dim)

    def forward(self, state: torch.Tensor, grid) -> Dict[str, torch.Tensor]:
        """state (B,L,state_dim) -> dict with attn, presence, prior."""
        b = state.shape[0]
        kv = self.kv_proj(state) + self._posenc(grid, state.device, state.dtype)
        q = self.query.unsqueeze(0).expand(b, -1, -1)

        bias = self.adj_scale * (self.adj_mask - 1.0) * 10.0       # 0 on edges
        attn = None
        for i in range(len(self.cross)):
            ctx, attn = self.cross[i](self.n1[i](q), kv)
            q = q + ctx
            sa, _ = self.selfat[i](self.n2[i](q), self.n2[i](q), self.n2[i](q),
                                   attn_mask=bias)
            q = q + sa
            q = q + self.ffn[i](self.n3[i](q))

        pres = self.presence(q).squeeze(-1)                        # (B, 13)
        return {"attn": attn, "presence": pres, "queries": q}

    def fuse(self, seg_logits: torch.Tensor, out: Dict[str, torch.Tensor],
             grid) -> torch.Tensor:
        """
        Add the anatomy path to the voxel logits.

        Two channels of information, both per-class and both zero-gated at init:
          * a spatial prior -- the query's attention map, upsampled from the
            token grid, saying roughly where this vessel is;
          * a presence bias -- a constant logit shift saying whether the vessel
            exists in this subject at all.
        The 8 mm token grid is far too coarse to segment a 1.5 mm Pcom, so the
        prior is deliberately additive and low-frequency: precision stays with
        the U-Net, existence and rough location come from the queries.
        """
        b, k = seg_logits.shape[0], self.n_q
        attn = out["attn"]                                          # (B,13,L)
        prior = attn.reshape(b, k, *grid)
        prior = prior / (prior.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
        prior = F.interpolate(prior, size=seg_logits.shape[2:],
                              mode="trilinear", align_corners=False)
        wp = self.w_prior.view(1, k, 1, 1, 1)
        wq = self.w_presence.view(1, k, 1, 1, 1)
        pres = out["presence"].view(b, k, 1, 1, 1)
        seg_logits = seg_logits.clone()
        seg_logits[:, 1:] = seg_logits[:, 1:] + wp * prior + wq * pres
        return seg_logits


# ---------------------------------------------------------------------------
# Losses that train the state to BE a good key
# ---------------------------------------------------------------------------
def attention_alignment_loss(attn: torch.Tensor, tok_labels: torch.Tensor
                             ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Query c's attention mass must land on tokens whose label is c.

    Without this, "the state makes good keys" is a hope. With it, it is a
    trained property with a number attached.

    Reported as the KL divergence, not the raw cross-entropy. The target is
    uniform over the tokens labelled c, so its entropy log(count) is an
    irreducible offset that depends only on how many tokens the vessel happens
    to occupy -- typically 2 to 5 out of 560. Leaving it in would start the term
    near log(560) = 6.3, which at weight 0.3 contributes 1.9 against a
    segmentation loss of ~2.7 and would dominate early training for reasons
    that have nothing to do with attention quality. Subtracting it makes the
    term exactly zero at the optimum, so its weight means what it says.

    Only queries whose class appears in the patch are scored -- a query for an
    absent vessel has no correct place to look, and its supervision is the
    presence head instead.
    """
    b, k, l = attn.shape
    logp = torch.log(attn.clamp_min(1e-8))
    loss, n = torch.zeros((), device=attn.device), 0
    for c in range(k):
        m = (tok_labels == (c + 1)).float()                        # (B, L)
        cnt = m.sum(1)
        sel = cnt > 0
        if sel.sum() == 0:
            continue
        tgt = m[sel] / cnt[sel].unsqueeze(1)
        ce = -(tgt * logp[sel, c]).sum(1)                          # H(t) + KL
        ent = torch.log(cnt[sel])                                  # H(t)
        loss = loss + (ce - ent).clamp_min(0).mean()               # KL >= 0
        n += 1
    if n == 0:
        return torch.zeros((), device=attn.device), {"attn_q": 0.0}
    return loss / n, {"attn_q": float(n)}


def presence_loss(pres_logits: torch.Tensor, tok_labels: torch.Tensor
                  ) -> torch.Tensor:
    """
    BCE on "does this vessel occur in this patch".

    Absence is a real anatomical fact here, not a sampling artefact: Pcom and
    the third A2 are genuinely missing in most subjects, and a model that cannot
    say so pays for it in false positives on exactly the rare classes whose Dice
    is worst.
    """
    b, k = pres_logits.shape
    tgt = torch.stack([(tok_labels == (c + 1)).any(1).float() for c in range(k)],
                      dim=1)
    return F.binary_cross_entropy_with_logits(pres_logits, tgt)
