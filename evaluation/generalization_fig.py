#!/usr/bin/env python3
"""
Modality-robustness bars: each model's performance on BOTH modalities.

Only models that can be evaluated on both modalities appear here. The other
joint-trained baselines (SharedAll, CMAP, MF-SSM) saw both modalities during
training, so they have no degradation to show -- their per-modality numbers are
in the per-modality figure, not this one.

Bars are zero-based and share one axis. The single-modality models were trained
on one modality only; the bar for the other modality is therefore a transfer
result, which the caption states explicitly.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

# name, CTA Dice, MRA Dice, trained on
ROWS = [
    ("M3CoW\n(ours, joint)",   0.739, 0.814, "both"),
    ("nnU-Net\nCT-only",       0.681, 0.429, "ct"),
    ("nnU-Net\nMR-only",       0.002, 0.786, "mr"),
]
x = np.arange(len(ROWS)); W = 0.34

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.axvspan(-0.5, 0.5, color="#f0f0ee", zorder=0)
for i, (key, colour, lab) in enumerate([(1, BLUE, "CTA"), (2, ORANGE, "MRA")]):
    vals = [r[key] for r in ROWS]
    b = ax.bar(x + (i - 0.5) * W, vals, W, color=colour, edgecolor="white",
               linewidth=1.0, zorder=3, label=lab)
    for r, v, row in zip(b, vals, ROWS):
        trained = row[3]
        hatch = (trained != "both") and (trained != ("ct" if key == 1 else "mr"))
        if hatch: r.set_hatch("///"); r.set_edgecolor("white")
        ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                va="bottom", fontsize=8, color=MUTED, zorder=4)

# annotate the two transfer failures
for xi, (dx, txt) in enumerate([(None, None), (0.17, "-0.252"), (-0.17, "-0.784")]):
    if dx is None: continue
    ax.annotate(txt, xy=(xi + dx, 0.52), fontsize=8.5, color=INK,
                ha="center", fontweight="bold")

ax.set_ylim(0, 0.96); ax.set_yticks(np.arange(0, 0.81, 0.2))
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in ROWS], fontsize=8.5)
ax.get_xticklabels()[0].set_fontweight("bold"); ax.get_xticklabels()[0].set_color(INK)
ax.set_ylabel("class-average Dice", fontsize=9.5, color=INK)
ax.spines[["top", "right"]].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("0.75"); ax.spines[s].set_linewidth(0.6)
ax.yaxis.grid(True, color="0.92", linewidth=0.6); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8.5, length=0)

h, l = ax.get_legend_handles_labels()
h.append(plt.Rectangle((0, 0), 1, 1, fc="0.6", hatch="///", ec="white"))
l.append("modality not seen in training")
ax.legend(h, l, fontsize=8, frameon=False, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.13))
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/generalization.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/generalization.pdf", bbox_inches="tight")
print("wrote figures/generalization.{png,pdf}")
