#!/usr/bin/env python3
"""Parse the CMAP training log and report voxel radius by CoW tree depth.

The per-class radius line is only useful once grouped by anatomical depth --
that grouping is the actual claim under test.
"""
import re, sys, glob
sys.path.insert(0, "training")
from cmap import COW_DEPTH

logs = sorted(glob.glob("results/nnUNet/Dataset104_TopCoW_joint/"
                        "nnUNetTrainerCMAP__nnUNetPlans__3d_fullres/fold_0/"
                        "training_log_*.txt"))
if not logs:
    sys.exit("no training log yet")
name2val = {"BA":1,"R-PCA":2,"L-PCA":3,"R-ICA":4,"R-MCA":5,"L-ICA":6,"L-MCA":7,
            "R-Pcom":8,"L-Pcom":9,"Acom":10,"R-ACA":11,"L-ACA":12,"3rd-A2":13}
rows = []
for line in open(logs[-1]):
    if "radius per class" not in line:
        continue
    per = {}
    for m in re.finditer(r"([A-Za-z0-9\-]+):([0-9.]+)", line.split("->")[1]):
        per[m.group(1)] = float(m.group(2))
    d = {0: [], 1: [], 2: []}
    for n, v in per.items():
        if n in name2val and name2val[n] in COW_DEPTH:
            d[COW_DEPTH[name2val[n]]].append(v)
    if all(d[k] for k in d):
        means = [sum(d[k]) / len(d[k]) for k in (0, 1, 2)]
        fg = [v for n, v in per.items() if n != "background"]
        rows.append((means, max(fg) - min(fg), per.get("background", float("nan"))))

print(f"{'#':>4} {'d0':>7} {'d1':>7} {'d2':>7} {'spread':>8} {'bg':>7}  monotone")
for i, (m, sp, bg) in enumerate(rows):
    mono = "YES" if m[0] < m[1] < m[2] else "no"
    print(f"{i*10:>4} {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f} {sp:>8.4f} {bg:>7.3f}  {mono}")
