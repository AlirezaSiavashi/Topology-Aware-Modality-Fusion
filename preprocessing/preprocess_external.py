"""
preprocess_external.py
======================
Preprocesses EXTERNAL (non-brain) vessel datasets for multi-domain training.

Included datasets:
  ─ MSD Task08   Hepatic Vessels (liver CT, binary: vessels + tumours)
  ─ KiPA22       Renal CTA (kidney, tumour, renal arteries, veins)
  ─ ASOCA        Coronary artery segmentation (cardiac CTA)
  ─ DRIVE        Retinal vessel segmentation (2-D fundus, RGB)

For each dataset:
  1. Resample to target spacing
  2. Normalise intensity
  3. Save nnU-Net imagesTr / labelsTr
  4. Generate auxiliary maps (vessel union, SDF, skeleton, curvature)

Download instructions (print with --show-downloads):
  MSD Task08 : http://medicaldecathlon.com/  → Task08_HepaticVessel.tar
  KiPA22     : https://zenodo.org/record/6361938
  ASOCA      : https://asoca.grand-challenge.org  (registration required)
  DRIVE      : https://drive.grand-challenge.org  (registration required)

Usage
-----
  python preprocess_external.py [--tasks all|msd|kipa|asoca|drive]
                                [--workers N]
                                [--skip-aux]
                                [--show-downloads]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from utils import (
    load_sitk, save_sitk, sitk_to_numpy, numpy_to_sitk,
    resample_image, resample_label, reorient_to_lps,
    normalize_ct, normalize_mri,
    make_vessel_union, compute_sdf, compute_vesselness,
    write_nnunet_dataset_json, nnunet_dirs, aux_dirs,
    run_parallel,
)
from skeleton_curvature import run_skeleton_curvature_pipeline

import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NNUNET_RAW_ROOT = cfg.OUTPUT_ROOT / "nnUNet_raw"
AUX_ROOT        = cfg.OUTPUT_ROOT / "aux"

DOWNLOAD_INSTRUCTIONS = """
════════════════════════════════════════════════════════
 External Dataset Download Instructions
════════════════════════════════════════════════════════

1.  MSD Task08 — Hepatic Vessels (liver CT)
    URL    : http://medicaldecathlon.com/
    File   : Task08_HepaticVessel.tar  (~5 GB)
    After  : set env var  MSD_HEPATIC_ROOT=/path/to/Task08_HepaticVessel
    Labels : 0=background, 1=vessel, 2=tumour
    Note   : use label=1 as vessel foreground; optionally exclude tumour

2.  KiPA22 — Renal CTA multi-structure
    URL    : https://zenodo.org/record/6361938
    After  : set env var  KIPA22_ROOT=/path/to/kipa22
    Labels : 0=background, 1=kidney, 2=tumour, 3=renal_artery, 4=renal_vein
    Note   : for vessel generalisation use labels 3 and 4

3.  ASOCA — Coronary Artery Segmentation (cardiac CTA)
    URL    : https://asoca.grand-challenge.org  (free registration)
    After  : set env var  ASOCA_ROOT=/path/to/ASOCA
    Labels : binary (0=background, 1=coronary_artery)

4.  DRIVE — Retinal vessel segmentation (2-D fundus)
    URL    : https://drive.grand-challenge.org  (free registration)
    After  : set env var  DRIVE_ROOT=/path/to/DRIVE
    Note   : 2-D images; SDF/skeleton are computed in 2-D

════════════════════════════════════════════════════════
For zero-shot generalisation claim in the TMI paper you need at least:
  ✓  MSD Task08  (widely used, easy access)
  ✓  KiPA22      (renal arteries — strong contrast to brain vessels)
  ✓  ASOCA       (cardiac CT — good cross-organ test)
════════════════════════════════════════════════════════
"""


# ──────────────────────────────────────────────────────────────────────────────
# Generic 3-D subject worker (same interface as brain script)
# ──────────────────────────────────────────────────────────────────────────────

def _process_subject_3d(args: Dict) -> str:
    img_path    = Path(args["img_path"])
    lbl_path    = Path(args.get("lbl_path", ""))
    out_img     = Path(args["out_img"])
    out_lbl     = Path(args.get("out_lbl", ""))
    modality    = args["modality"]
    spacing_key = args["spacing_key"]
    ct_window   = args.get("ct_window")
    do_aux      = args.get("do_aux", True)
    aux_base    = Path(args.get("aux_base", ""))
    subject_id  = args["subject_id"]
    vessel_classes = args.get("vessel_classes", None)  # which label ids = vessel

    target_spacing = cfg.TARGET_SPACING[spacing_key]
    spacing_zyx    = (target_spacing[2], target_spacing[1], target_spacing[0])
    has_label      = bool(args.get("lbl_path")) and lbl_path.is_file()

    # ── Resume: skip if all outputs already exist ─────────────────────────────
    need_aux = do_aux and has_label and aux_base
    already_done = (
        out_img.exists()
        and (not out_lbl or not has_label or out_lbl.exists())
        and (not need_aux or (
            (aux_base / "skeleton"  / f"{subject_id}.nii.gz").exists() and
            (aux_base / "curvature" / f"{subject_id}.nii.gz").exists() and
            (aux_base / "sdf"       / f"{subject_id}.nii.gz").exists()
        ))
    )
    if already_done:
        return f"SKIP {subject_id}"

    try:
        img_sitk = load_sitk(img_path)
        img_sitk = reorient_to_lps(img_sitk)

        lbl_sitk = None
        if has_label:
            lbl_sitk = load_sitk(lbl_path)
            lbl_sitk = reorient_to_lps(lbl_sitk)

        img_rs = resample_image(img_sitk, target_spacing)
        if lbl_sitk is not None:
            lbl_rs = resample_label(lbl_sitk, target_spacing, reference_image=img_rs)

        img_arr = sitk_to_numpy(img_rs).astype(np.float32)

        if modality == "ct":
            img_arr = normalize_ct(img_arr, ct_window[0], ct_window[1])
        else:
            img_arr = normalize_mri(img_arr)

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(img_rs)
        save_sitk(img_out, out_img)

        if lbl_sitk is not None and out_lbl:
            save_sitk(lbl_rs, out_lbl)

        if do_aux and has_label and aux_base:
            lbl_arr = sitk_to_numpy(lbl_rs).astype(np.int16)
            union   = make_vessel_union(lbl_arr, foreground_classes=vessel_classes)

            union_sitk = numpy_to_sitk(union, img_rs, is_label=True)
            save_sitk(union_sitk, aux_base / "vessel_union" / f"{subject_id}.nii.gz")

            sdf = compute_sdf(union, spacing_zyx,
                              clip_value=cfg.SDF_CLIP_VALUE,
                              normalize=cfg.SDF_NORMALIZE)
            sdf_sitk = sitk.GetImageFromArray(sdf)
            sdf_sitk.CopyInformation(img_rs)
            save_sitk(sdf_sitk, aux_base / "sdf" / f"{subject_id}.nii.gz")

            skel_result = run_skeleton_curvature_pipeline(
                vessel_union=union,
                spacing=spacing_zyx,
                min_branch_len=cfg.MIN_BRANCH_LEN,
                smoothing=cfg.SPLINE_SMOOTHING,
                curvature_clip=cfg.CURVATURE_CLIP,
            )
            skel_sitk = numpy_to_sitk(skel_result["skeleton"], img_rs, is_label=True)
            save_sitk(skel_sitk, aux_base / "skeleton" / f"{subject_id}.nii.gz")

            curv_sitk = sitk.GetImageFromArray(skel_result["curvature"])
            curv_sitk.CopyInformation(img_rs)
            save_sitk(curv_sitk, aux_base / "curvature" / f"{subject_id}.nii.gz")

            metrics_path = aux_base / "tortuosity_metrics" / f"{subject_id}.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump({
                    "subject_id":     subject_id,
                    "num_branches":   skel_result["num_branches"],
                    "branch_metrics": skel_result["branch_metrics"],
                }, f, indent=2)

        return f"OK  {subject_id}"

    except Exception as exc:
        logger.error("FAILED  %s  —  %s", subject_id, exc, exc_info=True)
        return f"FAIL {subject_id}: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# MSD Task08  –  Hepatic Vessels
# ──────────────────────────────────────────────────────────────────────────────

def process_msd_hepatic(do_aux: bool, num_workers: int) -> None:
    """
    MSD Task08 hepatic vessel segmentation.
    Expected structure:
        Task08_HepaticVessel/
            imagesTr/   hepaticvessel_NNN.nii.gz
            labelsTr/   hepaticvessel_NNN.nii.gz  (1=vessel, 2=tumour)
    """
    root = cfg.EXTERNAL_DATASETS["MSD_hepatic"]
    if not root.exists():
        logger.warning("MSD hepatic root not found: %s  –  skipping.", root)
        return

    logger.info("=== MSD Task08 Hepatic Vessels ===")

    imgs_src = root / "imagesTr"
    lbls_src = root / "labelsTr"
    img_files = sorted(f for f in imgs_src.glob("*.nii.gz") if not f.name.startswith("._"))

    if not img_files:
        # MSD archives often have one more level
        imgs_src = root / "Task08_HepaticVessel" / "imagesTr"
        lbls_src = root / "Task08_HepaticVessel" / "labelsTr"
        img_files = sorted(f for f in imgs_src.glob("*.nii.gz") if not f.name.startswith("._"))

    if not img_files:
        logger.warning("No images found in %s  –  skipping.", root)
        return

    ds_id   = cfg.NNUNET_DATASET_IDS["MSD_hepatic"]
    ds_name = "MSD_HepaticVessel"
    imgs_out, lbls_out, ds_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)

    job_list = []
    for img_path in img_files:
        subj_id = img_path.name.replace(".nii.gz", "")
        lbl_path = lbls_src / img_path.name

        job_list.append({
            "img_path":     str(img_path),
            "lbl_path":     str(lbl_path),
            "out_img":      str(imgs_out / f"{subj_id}_0000.nii.gz"),
            "out_lbl":      str(lbls_out / f"{subj_id}.nii.gz"),
            "modality":     "ct",
            "spacing_key":  "liver_ct",
            "ct_window":    cfg.CT_WINDOW["liver_ct"],
            "do_aux":       do_aux,
            "aux_base":     str(AUX_ROOT / ds_name) if do_aux else "",
            "subject_id":   subj_id,
            "vessel_classes": [1],   # label 1 = vessel only; exclude tumour (label 2)
        })

    write_nnunet_dataset_json(
        output_dir=ds_root,
        dataset_name=ds_name,
        dataset_id=ds_id,
        channel_names={0: "CT"},
        labels={"background": 0, "hepatic_vessel": 1, "tumour": 2},
        num_training=len(job_list),
        description="MSD Task08 — Hepatic Vessels & Tumours (liver CT)",
        reference="https://www.nature.com/articles/s41467-022-30695-9",
    )

    logger.info("Processing %d subjects (workers=%d) ...", len(job_list), num_workers)
    results = run_parallel(_process_subject_3d, job_list, num_workers, desc="MSD-Hepatic")
    _log_results(results)


# ──────────────────────────────────────────────────────────────────────────────
# KiPA22  –  Renal CTA
# ──────────────────────────────────────────────────────────────────────────────

def process_kipa22(do_aux: bool, num_workers: int) -> None:
    """
    KiPA22 multi-structure renal CTA.
    Expected structure:
        kipa22/
            train/
                images/  case_NNN_0000.nii.gz
                masks/   case_NNN.nii.gz       (1=kidney, 2=tumour, 3=artery, 4=vein)
    """
    root = cfg.EXTERNAL_DATASETS["KiPA22"]
    if not root.exists():
        logger.warning("KiPA22 root not found: %s  –  skipping.", root)
        return

    logger.info("=== KiPA22 Renal CTA ===")

    # Try common directory layouts
    for imgs_src, lbls_src in [
        (root / "train" / "images",  root / "train" / "masks"),
        (root / "imagesTr",          root / "labelsTr"),
        (root / "images",            root / "masks"),
    ]:
        if imgs_src.exists():
            break

    img_files = sorted(imgs_src.glob("*.nii.gz"))
    if not img_files:
        logger.warning("No images found under %s  –  skipping.", root)
        return

    ds_id   = cfg.NNUNET_DATASET_IDS["KiPA22"]
    ds_name = "KiPA22_RenalCTA"
    imgs_out, lbls_out, ds_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)

    job_list = []
    for img_path in img_files:
        # Normalise filename: ensure _0000 suffix for nnU-Net
        stem     = img_path.name.replace(".nii.gz", "").replace("_0000", "")
        subj_id  = stem
        out_name = f"{subj_id}_0000.nii.gz"

        lbl_path = lbls_src / f"{subj_id}.nii.gz"
        if not lbl_path.exists():
            lbl_path = lbls_src / img_path.name.replace("_0000", "")

        job_list.append({
            "img_path":     str(img_path),
            "lbl_path":     str(lbl_path),
            "out_img":      str(imgs_out / out_name),
            "out_lbl":      str(lbls_out / f"{subj_id}.nii.gz"),
            "modality":     "ct",
            "spacing_key":  "renal_cta",
            "ct_window":    cfg.CT_WINDOW["renal_cta"],
            "do_aux":       do_aux,
            "aux_base":     str(AUX_ROOT / ds_name) if do_aux else "",
            "subject_id":   subj_id,
            "vessel_classes": [3, 4],   # renal artery + renal vein only
        })

    write_nnunet_dataset_json(
        output_dir=ds_root,
        dataset_name=ds_name,
        dataset_id=ds_id,
        channel_names={0: "CTA"},
        labels={"background": 0, "kidney": 1, "tumour": 2,
                "renal_artery": 3, "renal_vein": 4},
        num_training=len(job_list),
        description="KiPA22 — Multi-structure Renal CTA",
        reference="https://zenodo.org/record/6361938",
    )

    logger.info("Processing %d subjects (workers=%d) ...", len(job_list), num_workers)
    results = run_parallel(_process_subject_3d, job_list, num_workers, desc="KiPA22")
    _log_results(results)


# ──────────────────────────────────────────────────────────────────────────────
# ASOCA  –  Coronary Artery Segmentation
# ──────────────────────────────────────────────────────────────────────────────

def process_asoca(do_aux: bool, num_workers: int) -> None:
    """
    ASOCA coronary artery segmentation from cardiac CTA.
    Expected structure:
        ASOCA/
            Normal/
                Normal_N.nrrd  or  Normal_N.nii.gz
            Diseased/
                Diseased_N.nrrd  or  Diseased_N.nii.gz
            Annotations/
                Normal/    Normal_N.nrrd
                Diseased/  Diseased_N.nrrd
    """
    root = cfg.EXTERNAL_DATASETS["ASOCA_coronary"]
    if not root.exists():
        logger.warning("ASOCA root not found: %s  –  skipping.", root)
        return

    logger.info("=== ASOCA Coronary Arteries ===")

    ds_id   = cfg.NNUNET_DATASET_IDS["ASOCA_coronary"]
    ds_name = "ASOCA_Coronary"
    imgs_out, lbls_out, ds_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)

    job_list = []
    for subtype in ["Normal", "Diseased"]:
        img_dir = root / subtype
        lbl_dir = root / "Annotations" / subtype
        if not img_dir.exists():
            continue

        for ext in ["*.nii.gz", "*.nrrd", "*.mhd"]:
            img_files = sorted(img_dir.glob(ext))
            if img_files:
                break

        for img_path in img_files:
            stem    = img_path.name.split(".")[0]
            subj_id = f"{subtype}_{stem}"

            # Try matching label
            lbl_path = None
            for ext in [".nii.gz", ".nrrd", ".mhd"]:
                candidate = lbl_dir / (stem + ext)
                if candidate.exists():
                    lbl_path = candidate
                    break

            job_list.append({
                "img_path":     str(img_path),
                "lbl_path":     str(lbl_path) if lbl_path else "",
                "out_img":      str(imgs_out / f"{subj_id}_0000.nii.gz"),
                "out_lbl":      str(lbls_out / f"{subj_id}.nii.gz"),
                "modality":     "ct",
                "spacing_key":  "coronary_ct",
                "ct_window":    cfg.CT_WINDOW["coronary_ct"],
                "do_aux":       do_aux,
                "aux_base":     str(AUX_ROOT / ds_name) if do_aux else "",
                "subject_id":   subj_id,
                "vessel_classes": None,   # binary: all foreground = coronary
            })

    if not job_list:
        logger.warning("No ASOCA images found under %s  –  skipping.", root)
        return

    write_nnunet_dataset_json(
        output_dir=ds_root,
        dataset_name=ds_name,
        dataset_id=ds_id,
        channel_names={0: "CT"},
        labels={"background": 0, "coronary_artery": 1},
        num_training=len(job_list),
        description="ASOCA — Coronary Artery Segmentation from Cardiac CTA",
        reference="https://asoca.grand-challenge.org",
    )

    logger.info("Processing %d subjects (workers=%d) ...", len(job_list), num_workers)
    results = run_parallel(_process_subject_3d, job_list, num_workers, desc="ASOCA")
    _log_results(results)


# ──────────────────────────────────────────────────────────────────────────────
# DRIVE  –  Retinal Vessel Segmentation (2-D)
# ──────────────────────────────────────────────────────────────────────────────

def process_drive(do_aux: bool, num_workers: int) -> None:
    """
    DRIVE retinal vessel segmentation (2-D fundus images).
    Expected structure:
        DRIVE/
            training/
                images/   NNN_training.tif
                1st_manual/  NNN_manual1.gif   (ground truth)
                mask/     NNN_training_mask.gif
            test/
                images/   NNN_test.tif
                1st_manual/ NNN_manual1.gif

    Note: 2-D images are converted to NIfTI with unit z-spacing for nnU-Net.
    SDF and skeleton are computed in 2-D (single-slice volumes).
    """
    root = cfg.EXTERNAL_DATASETS["DRIVE_retinal"]
    if not root.exists():
        logger.warning("DRIVE root not found: %s  –  skipping.", root)
        return

    logger.info("=== DRIVE Retinal Vessels (2-D) ===")

    try:
        from PIL import Image as PILImage
    except ImportError:
        logger.error("Pillow not installed — cannot process DRIVE (pip install Pillow).")
        return

    ds_id   = 205
    ds_name = "DRIVE_Retinal"
    imgs_out, lbls_out, ds_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)

    job_list = []

    for split in ["training", "test"]:
        img_dir = root / split / "images"
        lbl_dir = root / split / "1st_manual"
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.tif")):
            subj_id = img_path.stem   # e.g. "21_training"

            # Find matching label (21_manual1.gif)
            num = subj_id.split("_")[0]
            lbl_path = None
            for lbl_name in [f"{num}_manual1.gif", f"{num}_manual1.png",
                              f"{num}_manual1.tif"]:
                c = lbl_dir / lbl_name
                if c.exists():
                    lbl_path = c
                    break

            if lbl_path is None:
                logger.warning("  No label for %s  –  image-only.", subj_id)

            job_list.append({
                "img_path":   str(img_path),
                "lbl_path":   str(lbl_path) if lbl_path else "",
                "out_img":    str(imgs_out / f"{subj_id}_0000.nii.gz"),
                "out_lbl":    str(lbls_out / f"{subj_id}.nii.gz"),
                "split":      split,
                "do_aux":     do_aux,
                "aux_base":   str(AUX_ROOT / ds_name) if do_aux else "",
                "subject_id": subj_id,
            })

    if not job_list:
        logger.warning("No DRIVE images found  –  skipping.")
        return

    write_nnunet_dataset_json(
        output_dir=ds_root,
        dataset_name=ds_name,
        dataset_id=ds_id,
        channel_names={0: "R", 1: "G", 2: "B"},
        labels={"background": 0, "retinal_vessel": 1},
        num_training=sum(1 for j in job_list if j["split"] == "training"),
        description="DRIVE — Retinal Vessel Segmentation (2-D fundus)",
        reference="https://drive.grand-challenge.org",
    )

    logger.info("Processing %d subjects (workers=%d) ...", len(job_list), num_workers)
    results = run_parallel(_process_drive_subject, job_list, num_workers, desc="DRIVE")
    _log_results(results)


def _process_drive_subject(args: Dict) -> str:
    """2-D DRIVE subject: TIF → single-slice NIfTI."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        return f"FAIL {args['subject_id']}: Pillow not installed"

    try:
        img_path  = Path(args["img_path"])
        lbl_path  = Path(args.get("lbl_path", ""))
        out_img   = Path(args["out_img"])
        out_lbl   = Path(args.get("out_lbl", ""))
        subject_id = args["subject_id"]
        do_aux    = args.get("do_aux", True)
        aux_base  = Path(args.get("aux_base", ""))

        # Load RGB fundus image
        pil_img = PILImage.open(img_path).convert("RGB")
        rgb = np.array(pil_img, dtype=np.float32)   # (H, W, 3)

        # Normalise each channel independently
        for c in range(3):
            ch = rgb[..., c]
            ch = (ch - ch.mean()) / (ch.std() + 1e-8)
            rgb[..., c] = ch

        # Convert to 3-channel NIfTI: shape (3, H, W) → store as (1, H, W) × 3
        # nnU-Net multi-channel: save each channel as _000N.nii.gz
        H, W = rgb.shape[:2]
        spacing = (1.0, 1.0, 1.0)

        for c_idx in range(3):
            ch_arr = rgb[..., c_idx][np.newaxis, :, :]   # (1, H, W)
            ch_sitk = sitk.GetImageFromArray(ch_arr)
            ch_sitk.SetSpacing(spacing)
            out_c = out_img.parent / out_img.name.replace("_0000", f"_000{c_idx}")
            save_sitk(ch_sitk, out_c)

        # Label
        has_label = bool(args.get("lbl_path")) and lbl_path.is_file()
        if has_label and out_lbl:
            pil_lbl = PILImage.open(lbl_path).convert("L")
            lbl_arr = (np.array(pil_lbl) > 127).astype(np.uint8)[np.newaxis, :, :]
            lbl_sitk = sitk.GetImageFromArray(lbl_arr)
            lbl_sitk.SetSpacing(spacing)
            save_sitk(lbl_sitk, out_lbl)

            if do_aux and aux_base:
                union = lbl_arr[0]   # (H, W)
                # 2-D SDF
                from scipy.ndimage import distance_transform_edt
                edt_fg = distance_transform_edt(union.astype(bool)).astype(np.float32)
                edt_bg = distance_transform_edt(~union.astype(bool)).astype(np.float32)
                sdf_2d = np.clip(edt_bg - edt_fg, -cfg.SDF_CLIP_VALUE, cfg.SDF_CLIP_VALUE)
                if cfg.SDF_NORMALIZE:
                    sdf_2d /= cfg.SDF_CLIP_VALUE

                sdf_sitk = sitk.GetImageFromArray(sdf_2d[np.newaxis])
                sdf_sitk.SetSpacing(spacing)
                save_sitk(sdf_sitk, aux_base / "sdf" / f"{subject_id}.nii.gz")

                union_sitk = sitk.GetImageFromArray(union[np.newaxis])
                union_sitk.SetSpacing(spacing)
                save_sitk(union_sitk, aux_base / "vessel_union" / f"{subject_id}.nii.gz")

        return f"OK  {subject_id}"
    except Exception as exc:
        return f"FAIL {args.get('subject_id', '?')}: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _log_results(results: List[str]) -> None:
    ok   = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    fail = sum(1 for r in results if r.startswith("FAIL"))
    logger.info("  Done: %d OK, %d skipped (already exist), %d FAILED", ok, skip, fail)
    for r in results:
        if r.startswith("FAIL"):
            logger.error("  %s", r)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess external (non-brain) vessel datasets."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "msd", "kipa", "asoca", "drive"],
    )
    parser.add_argument("--workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--skip-aux", action="store_true")
    parser.add_argument(
        "--show-downloads", action="store_true",
        help="Print download instructions and exit.",
    )
    args = parser.parse_args()

    if args.show_downloads:
        print(DOWNLOAD_INSTRUCTIONS)
        return

    do_all = "all" in args.tasks
    do_aux = not args.skip_aux
    w      = args.workers

    logger.info("Output root: %s", cfg.OUTPUT_ROOT)

    if do_all or "msd" in args.tasks:
        process_msd_hepatic(do_aux, w)

    if do_all or "kipa" in args.tasks:
        process_kipa22(do_aux, w)

    if do_all or "asoca" in args.tasks:
        process_asoca(do_aux, w)

    if do_all or "drive" in args.tasks:
        process_drive(do_aux, w)

    logger.info("External preprocessing complete.")


if __name__ == "__main__":
    main()
