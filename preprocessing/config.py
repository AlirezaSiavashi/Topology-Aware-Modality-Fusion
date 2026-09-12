"""
config.py
=========
Central configuration: all dataset paths, label definitions, and
preprocessing hyperparameters.  Edit ONLY this file to adapt to a
different machine layout.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Root paths  (override via env vars for portability)
# ─────────────────────────────────────────────
DATASET_ROOT = Path(
    os.environ.get(
        "VESSEL_DATASET_ROOT",
        "/scratch/siyavash/Alireza_thesis/external_dataset/"
        "rsna-intracranial-aneurysm-detection/brain_external_dataset/dataset",
    )
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "VESSEL_OUTPUT_ROOT",
        "/scratch/siyavash/Alireza_thesis/external_dataset/"
        "rsna-intracranial-aneurysm-detection/brain_external_dataset/preprocessed",
    )
)

# ─────────────────────────────────────────────
# Per-dataset source paths
# ─────────────────────────────────────────────
TOPBRAIN_ROOT = (
    DATASET_ROOT / "16878417" / "TopBrain_Data_Release_Batches1n2_081425"
)
TOPCOW_ROOT = (
    DATASET_ROOT / "15692630" / "TopCoW2024_Data_Release"
)

# Cross-domain brain subsets (TopCoW-style labels, different scanner/cohort)
CROSS_DOMAIN_BRAIN = {
    "CTA_ISLES2024":  DATASET_ROOT / "15692630" / "CTA_ISLES2024_TUM",
    "MRA_IXI_HH":     DATASET_ROOT / "15692630" / "MRA_IXI_HH",
    "MRA_Lausanne":   DATASET_ROOT / "15692630" / "MRA_Lausanne",
    # CTA_LargeIA has only labels (no images) – kept for label-only use
    "CTA_LargeIA":    DATASET_ROOT / "15692630" / "CTA_LargeIA",
}

# ITKTubeTK – unlabelled normal MRA; used for tortuosity agreement test only
ITKTUBETK_ROOT = DATASET_ROOT / "ITKTubeTK-MRA-20260222T163002Z-1-001" / "ITKTubeTK-MRA"
ITKTUBETK_ROOT2 = DATASET_ROOT / "ITKTubeTK-MRA-20260222T163002Z-1-002" / "ITKTubeTK-MRA"

# External (non-brain) vessel datasets – paths filled in when data is downloaded
EXTERNAL_DATASETS = {
    # MSD Task08 – Hepatic Vessels (liver CT, binary: vessels + tumours)
    # Download: http://medicaldecathlon.com/  Task08_HepaticVessel.tar
    "MSD_hepatic":    Path(os.environ.get("MSD_HEPATIC_ROOT", "/data/external/MSD_Task08")),

    # KiPA22 – Renal CTA (kidney, tumour, arteries, veins)
    # Download: https://zenodo.org/record/6361938
    "KiPA22":         Path(os.environ.get("KIPA22_ROOT", "/data/external/KiPA22")),

    # DRIVE / STARE / CHASE_DB1 – Retinal vessel segmentation (2D fundus)
    # Download: https://drive.grand-challenge.org  (DRIVE)
    "DRIVE_retinal":  Path(os.environ.get("DRIVE_ROOT", "/data/external/DRIVE")),

    # ASOCA – Coronary artery segmentation from cardiac CTA
    # Download: https://asoca.grand-challenge.org
    "ASOCA_coronary": Path(os.environ.get("ASOCA_ROOT", "/data/external/ASOCA")),
}

# ─────────────────────────────────────────────
# Target preprocessing spacing (mm)
# ─────────────────────────────────────────────
# Brain CTA:  ~0.4–0.5 mm in-plane, 0.6–0.75 mm slice → resample to 0.5 isotropic
# Brain MRA:  ~0.3–0.5 mm in-plane, 0.6–0.8 mm slice → resample to 0.5 isotropic
#   Rationale: cross-dataset MRA varies from 0.3 to 0.5 mm in-plane and up to
#   0.8 mm slice (Lausanne anisotropy=2.56).  Forcing 0.35 mm isotropic would
#   upsample the slice axis 2.3× with no new information and inflate volume size.
#   0.5 mm isotropic is the safe shared target across all brain modalities.
# Non-brain:  dataset-specific, defined per dataset below
TARGET_SPACING = {
    "brain_ct":  (0.50, 0.50, 0.50),
    "brain_mr":  (0.50, 0.50, 0.50),
    "liver_ct":  (0.80, 0.80, 0.80),
    "renal_cta": (0.60, 0.60, 0.60),
    "retinal_2d": (None, None, None),   # keep native; 2-D dataset
    "coronary_ct": (0.50, 0.50, 0.50),
}

# ─────────────────────────────────────────────
# Intensity normalisation windows (CT only; MRI uses z-score)
# ─────────────────────────────────────────────
CT_WINDOW = {
    "brain_ct":    (-200, 700),   # brain angio: soft-tissue + vessel enhancement
    "liver_ct":    (-50, 250),    # hepatic vessels
    "renal_cta":   (-50, 450),    # renal arteries
    "coronary_ct": (-200, 700),   # cardiac CTA
}

# ─────────────────────────────────────────────
# TopBrain label definitions
# ─────────────────────────────────────────────
# CTA has 40 labels; MRA has 42 labels; 34 labels overlap.
# Full list sourced from the ITK-Snap label map files.

TOPBRAIN_CTA_LABELS = {
    0: "Background",
    1: "BA", 2: "R-P1P2", 3: "L-P1P2", 4: "R-ICA", 5: "R-M1",
    6: "L-ICA", 7: "L-M1", 8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-A1A2", 12: "L-A1A2", 13: "R-A3", 14: "L-A3", 15: "3rd-A2",
    16: "3rd-A3", 17: "R-M2", 18: "R-M3", 19: "L-M2", 20: "L-M3",
    21: "R-P3P4", 22: "L-P3P4", 23: "R-VA", 24: "L-VA",
    25: "R-SCA", 26: "L-SCA", 27: "R-AICA", 28: "L-AICA",
    29: "R-PICA", 30: "L-PICA", 31: "R-AChA", 32: "L-AChA",
    33: "R-OA", 34: "L-OA", 35: "R-ECA", 36: "L-ECA",
    37: "R-STA", 38: "L-STA", 39: "R-MaxA", 40: "L-MaxA",
}

TOPBRAIN_MRA_LABELS = {
    **TOPBRAIN_CTA_LABELS,
    41: "R-MMA",
    42: "L-MMA",
}

# CoW subset (shared with TopCoW): labels present in both CTA and MRA
TOPCOW_LABEL_SUBSET = {
    0: "Background",
    1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA",
    6: "L-ICA", 7: "L-MCA", 8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 15: "3rd-A2",
}

# Arterial labels only (exclude veins for topology-focused training)
TOPBRAIN_ARTERIAL = {k: v for k, v in TOPBRAIN_MRA_LABELS.items()
                     if k not in {35, 36, 37, 38, 39, 40, 41, 42}  # VoG/StS/ICVs/SSS/BVR
                     and k != 0}

# ─────────────────────────────────────────────
# SDF generation parameters
# ─────────────────────────────────────────────
SDF_CLIP_VALUE = 50.0   # mm; clip SDF to [-clip, clip] for stability
SDF_NORMALIZE  = True   # divide by clip_value → range [-1, 1]

# ─────────────────────────────────────────────
# Skeletonisation / curvature parameters
# ─────────────────────────────────────────────
SKEL_METHOD = "lee"          # "lee" (3-D thinning, fast) | "zhang" (slower)
SPLINE_SMOOTHING = 0.5       # smoothing factor for scipy.interpolate.splprep
CURVATURE_CLIP = 5.0         # mm^-1; clip extreme curvature values
MIN_BRANCH_LEN = 5           # voxels; prune skeleton branches shorter than this

# ─────────────────────────────────────────────
# nnU-Net dataset IDs
# ─────────────────────────────────────────────
NNUNET_DATASET_IDS = {
    "TopBrain_CT":    100,
    "TopBrain_MR":    101,
    "TopCoW_CT":      102,
    "TopCoW_MR":      103,
    "TopCoW_joint":   104,   # CT+MR combined, domain label stored in channel-1 filename suffix
    "MSD_hepatic":    110,
    "KiPA22":         111,
    "ASOCA_coronary": 112,
    "AllVessel":      120,   # combined multi-domain training dataset
}

# ─────────────────────────────────────────────
# Parallelism
# ─────────────────────────────────────────────
# IMPORTANT: The SLURM job has a 16 GB memory cgroup limit.
# Each worker processing an MRA with aux maps (SDF + skeleton + curvature)
# peaks at ~3–4 GB (volume array + graph + spline buffers).
# Safe maximum: 3 workers for aux-map tasks, more for image-only tasks.
# Override with --workers N on the command line.
NUM_WORKERS = int(os.environ.get("VESSEL_WORKERS", "3"))
