#!/usr/bin/env python3
"""
Per-modality comparison bars, all three metrics on one shared axis.

Bars start at ZERO and share a single y-axis -- there is no second scale, so
nothing is magnified relative to anything else. Note that Betti-0 is
lower-better while Dice and clDice are higher-better; the legend carries
explicit arrows because a short Betti-0 bar means good and a short Dice bar
means bad, which is the one thing a reader can misread here.
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

D = json.load(open("evaluation/results/m3cow/per_modality.json"))
ORDER = ["M3CoW (ours)", "MF-SSM SharedAll", "nnU-Net", "CMAP", "MF-SSM"]
SHORT = ["M3CoW\n(ours)", "SharedAll", "nnU-Net", "CMAP", "MF-SSM"]
SERIES = [("Dice", "Dice", BLUE), ("clDice", "clDice", ORANGE), ("Betti0", "Betti-0", AQUA)]
x = np.arange(len(ORDER))
W = 0.26

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
for c, mod in enumerate(("CTA", "MRA")):
    ax = axes[c]
    ax.axvspan(-0.5, 0.5, color="#f0f0ee", zorder=0)
    for i, (key, _, colour) in enumerate(SERIES):
        vals = [D[m][f"{mod}_{key}"] for m in ORDER]
        b = ax.bar(x + (i - 1) * W, vals, W, color=colour,
                   edgecolor="white", linewidth=1.0, zorder=3)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.2, color=MUTED,
                    rotation=90, zorder=4)
    ax.set_ylim(0, 1.16); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=8)
    for t, m in zip(ax.get_xticklabels(), ORDER):
        if m.startswith("M3CoW"): t.set_color(INK); t.set_fontweight("bold")
    ax.set_title(mod, fontsize=10, color=INK, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.75"); ax.spines[s].set_linewidth(0.6)
    ax.yaxis.grid(True, color="0.90", linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
axes[0].set_ylabel("score", fontsize=9, color=INK)

h = [plt.Rectangle((0, 0), 1, 1, fc=c) for _, _, c in SERIES]
fig.legend(h, ["Dice  $\\uparrow$", "clDice  $\\uparrow$", "Betti-0 error  $\\downarrow$"],
           loc="lower center", ncol=3, fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=[0, 0.06, 1, 1])
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/per_modality.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/per_modality.pdf", bbox_inches="tight")
print("wrote figures/per_modality.{png,pdf}")
