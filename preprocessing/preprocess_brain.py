"""
preprocess_brain.py
===================
Preprocesses all BRAIN vascular datasets into nnU-Net v2 format and generates
auxiliary supervision maps (vessel-union, SDF, skeleton, curvature).

Datasets handled:
  1. TopBrain 2025   – 25 CT + 25 MR  (40/42-class brain vessel anatomy)
  2. TopCoW 2024     – 125 CT + 125 MR (13-class Circle of Willis)
  3. Cross-domain subsets (zero-shot test sets):
       CTA_ISLES2024, MRA_IXI_HH, MRA_Lausanne
  4. ITKTubeTK       – unlabelled normal MRA (tortuosity agreement test only)

Output layout (under OUTPUT_ROOT):
  nnUNet_raw/
    Dataset100_TopBrain_CT/   imagesTr/ labelsTr/ dataset.json
    Dataset101_TopBrain_MR/   ...
    Dataset102_TopCoW_CT/     ...
    Dataset103_TopCoW_MR/     ...
    Dataset104_TopCoW_joint/  ...  (CT + MR combined, domain tag in filename)
  aux/
    TopBrain_CT/  vessel_union/  sdf/  skeleton/  curvature/
    TopBrain_MR/  ...
    TopCoW_CT/    ...
    ...

Usage
-----
  python preprocess_brain.py [--tasks all|topbrain|topcow|crossdomain|itktubetk]
                             [--workers N]
                             [--skip-aux]
                             [--skip-vesselness]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from utils import (
    load_sitk, save_sitk, sitk_to_numpy, numpy_to_sitk,
    resample_image, resample_label, reorient_to_lps,
    normalize_ct, normalize_mri, compute_brain_mask,
    make_vessel_union, compute_sdf, compute_vesselness,
    write_nnunet_dataset_json, nnunet_dirs, aux_dirs,
    run_parallel,
)
from skeleton_curvature import run_skeleton_curvature_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NNUNET_RAW_ROOT = cfg.OUTPUT_ROOT / "nnUNet_raw"
AUX_ROOT        = cfg.OUTPUT_ROOT / "aux"


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject preprocessing worker
# ──────────────────────────────────────────────────────────────────────────────

def _process_subject(args: Dict) -> str:
    """
    Process one subject: resample → normalise → save image + label.
    Optionally compute and save aux maps.

    This function is designed to be called from a process pool, so it is
    self-contained (no shared state).
    """
    img_path    = Path(args["img_path"])
    lbl_path    = Path(args.get("lbl_path", ""))
    out_img     = Path(args["out_img"])
    out_lbl     = Path(args.get("out_lbl", ""))
    modality    = args["modality"]          # "ct" | "mr"
    spacing_key = args["spacing_key"]       # key into cfg.TARGET_SPACING
    ct_window   = args.get("ct_window", cfg.CT_WINDOW.get(spacing_key))
    do_aux      = args.get("do_aux", True)
    do_vess     = args.get("do_vesselness", False)
    aux_base    = Path(args.get("aux_base", ""))
    subject_id  = args["subject_id"]
    has_label   = bool(args.get("lbl_path")) and lbl_path.is_file()

    target_spacing = cfg.TARGET_SPACING[spacing_key]  # (x, y, z)
    # SimpleITK spacing is (x,y,z); scipy EDT needs (z,y,x)
    spacing_zyx = (target_spacing[2], target_spacing[1], target_spacing[0])

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
        # ── Load ──────────────────────────────────────────────────────────────
        img_sitk = load_sitk(img_path)
        img_sitk = reorient_to_lps(img_sitk)

        lbl_sitk = None
        if has_label:
            lbl_sitk = load_sitk(lbl_path)
            lbl_sitk = reorient_to_lps(lbl_sitk)

        # ── Resample ──────────────────────────────────────────────────────────
        img_rs = resample_image(img_sitk, target_spacing)
        if lbl_sitk is not None:
            lbl_rs = resample_label(lbl_sitk, target_spacing, reference_image=img_rs)

        # ── Intensity normalise ───────────────────────────────────────────────
        img_arr = sitk_to_numpy(img_rs).astype(np.float32)

        if modality == "ct":
            img_arr = normalize_ct(img_arr, ct_window[0], ct_window[1])
        else:
            brain_mask = compute_brain_mask(img_arr)
            img_arr = normalize_mri(img_arr, brain_mask)

        # ── Save image (float32 → int16 for nnU-Net compatibility) ────────────
        # nnU-Net v2 accepts float images; store as float32 to keep full precision
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(img_rs)
        save_sitk(img_out, out_img)

        # ── Save label ────────────────────────────────────────────────────────
        if lbl_sitk is not None and out_lbl:
            save_sitk(lbl_rs, out_lbl)

        # ── Auxiliary maps ────────────────────────────────────────────────────
        if do_aux and has_label and aux_base:
            lbl_arr = sitk_to_numpy(lbl_rs).astype(np.int16)

            # Vessel union
            union = make_vessel_union(lbl_arr)
            union_sitk = numpy_to_sitk(union, img_rs, is_label=True)
            save_sitk(union_sitk, aux_base / "vessel_union" / f"{subject_id}.nii.gz")

            # SDF from vessel union
            sdf = compute_sdf(union, spacing_zyx,
                              clip_value=cfg.SDF_CLIP_VALUE,
                              normalize=cfg.SDF_NORMALIZE)
            sdf_sitk = sitk.GetImageFromArray(sdf)
            sdf_sitk.CopyInformation(img_rs)
            save_sitk(sdf_sitk, aux_base / "sdf" / f"{subject_id}.nii.gz")

            # Skeleton + curvature
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

            # Save branch tortuosity metrics as JSON
            metrics_path = aux_base / "tortuosity_metrics" / f"{subject_id}.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump({
                    "subject_id":     subject_id,
                    "num_branches":   skel_result["num_branches"],
                    "branch_metrics": skel_result["branch_metrics"],
                }, f, indent=2)

        # ── Optional vesselness channel ───────────────────────────────────────
        if do_vess and aux_base:
            # sigmas in mm covering expected vessel diameters 0.5–2 mm radius
            vess = compute_vesselness(
                img_arr, spacing_zyx,
                sigmas=[0.5, 1.0, 1.5, 2.0],
                black_ridges=(modality == "mr"),  # MRA: vessels bright; some sequences dark
            )
            vess_sitk = sitk.GetImageFromArray(vess)
            vess_sitk.CopyInformation(img_rs)
            vess_dir = aux_base / "vesselness"
            vess_dir.mkdir(parents=True, exist_ok=True)
            save_sitk(vess_sitk, vess_dir / f"{subject_id}.nii.gz")

        return f"OK  {subject_id}"

    except Exception as exc:
        logger.error("FAILED  %s  —  %s", subject_id, exc, exc_info=True)
        return f"FAIL {subject_id}: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset processors
# ──────────────────────────────────────────────────────────────────────────────

def process_topbrain(do_aux: bool, do_vesselness: bool, num_workers: int) -> None:
    """
    TopBrain 2025: 25 CT + 25 MR  →  Dataset100 and Dataset101
    """
    tb_root = cfg.TOPBRAIN_ROOT

    for modality, ds_id, ds_name, spacing_key, ct_win in [
        ("ct", cfg.NNUNET_DATASET_IDS["TopBrain_CT"], "TopBrain_CT",
         "brain_ct", cfg.CT_WINDOW["brain_ct"]),
        ("mr", cfg.NNUNET_DATASET_IDS["TopBrain_MR"], "TopBrain_MR",
         "brain_mr", None),
    ]:
        logger.info("=== TopBrain %s (%s) ===", modality.upper(), ds_name)

        imgs_dir  = tb_root / f"imagesTr_topbrain_{modality}"
        lbls_dir  = tb_root / f"labelsTr_topbrain_{modality}"
        img_files = sorted(imgs_dir.glob("*.nii.gz"))

        if not img_files:
            logger.warning("No images found in %s – skipping.", imgs_dir)
            continue

        imgs_out, lbls_out, ds_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)
        aux_d = aux_dirs(AUX_ROOT, ds_name) if do_aux else {}

        job_list = []
        for img_path in img_files:
            # Filename: topcow_ct_001_0000.nii.gz  → subject_id = topcow_ct_001
            stem = img_path.name.replace(".nii.gz", "").replace("_0000", "")
            subj_id = stem

            # Label filename: topcow_ct_001.nii.gz
            lbl_path = lbls_dir / f"{subj_id}.nii.gz"

            job_list.append({
                "img_path":    str(img_path),
                "lbl_path":    str(lbl_path),
                "out_img":     str(imgs_out / img_path.name),
                "out_lbl":     str(lbls_out / f"{subj_id}.nii.gz"),
                "modality":    modality,
                "spacing_key": spacing_key,
                "ct_window":   ct_win,
                "do_aux":      do_aux,
                "do_vesselness": do_vesselness,
                "aux_base":    str(AUX_ROOT / ds_name) if do_aux else "",
                "subject_id":  subj_id,
            })

        # Write dataset.json
        num_classes = 40 if modality == "ct" else 42
        label_names = cfg.TOPBRAIN_CTA_LABELS if modality == "ct" else cfg.TOPBRAIN_MRA_LABELS
        labels_json = {v: k for k, v in label_names.items()}  # name → id

        write_nnunet_dataset_json(
            output_dir=ds_root,
            dataset_name=ds_name,
            dataset_id=ds_id,
            channel_names={0: "CTA" if modality == "ct" else "MRA"},
            labels=labels_json,
            num_training=len(job_list),
            description=f"TopBrain 2025 {modality.upper()} — {num_classes}-class brain vessels",
            reference="https://zenodo.org/records/16878417",
        )

        logger.info("Processing %d subjects (workers=%d) ...", len(job_list), num_workers)
        results = run_parallel(_process_subject, job_list, num_workers,
                               desc=f"TopBrain-{modality.upper()}")
        _log_results(results)


def process_topcow(do_aux: bool, do_vesselness: bool, num_workers: int) -> None:
    """
    TopCoW 2024: 125 CT + 125 MR paired data + joint dataset.
    Also builds Dataset104_TopCoW_joint (CT+MR combined with domain indicator).
    """
    cow_root = cfg.TOPCOW_ROOT

    imgs_dir   = cow_root / "imagesTr"
    lbls_dir   = cow_root / "cow_seg_labelsTr"
    all_imgs   = sorted(imgs_dir.glob("*.nii.gz"))

    ct_imgs = [f for f in all_imgs if "_ct_" in f.name]
    mr_imgs = [f for f in all_imgs if "_mr_" in f.name]

    logger.info("=== TopCoW: %d CT + %d MR ===", len(ct_imgs), len(mr_imgs))

    for modality, img_files, ds_id, ds_name, spacing_key, ct_win in [
        ("ct", ct_imgs, cfg.NNUNET_DATASET_IDS["TopCoW_CT"], "TopCoW_CT",
         "brain_ct", cfg.CT_WINDOW["brain_ct"]),
        ("mr", mr_imgs, cfg.NNUNET_DATASET_IDS["TopCoW_MR"], "TopCoW_MR",
         "brain_mr", None),
    ]:
        imgs_out, lbls_out, ds_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)

        job_list = []
        for img_path in img_files:
            stem    = img_path.name.replace(".nii.gz", "").replace("_0000", "")
            subj_id = stem
            lbl_path = lbls_dir / f"{subj_id}.nii.gz"

            job_list.append({
                "img_path":    str(img_path),
                "lbl_path":    str(lbl_path),
                "out_img":     str(imgs_out / img_path.name),
                "out_lbl":     str(lbls_out / f"{subj_id}.nii.gz"),
                "modality":    modality,
                "spacing_key": spacing_key,
                "ct_window":   ct_win,
                "do_aux":      do_aux,
                "do_vesselness": do_vesselness,
                "aux_base":    str(AUX_ROOT / ds_name) if do_aux else "",
                "subject_id":  subj_id,
            })

        cow_labels = {v: k for k, v in cfg.TOPCOW_LABEL_SUBSET.items()}
        write_nnunet_dataset_json(
            output_dir=ds_root,
            dataset_name=ds_name,
            dataset_id=ds_id,
            channel_names={0: "CTA" if modality == "ct" else "MRA"},
            labels=cow_labels,
            num_training=len(job_list),
            description=f"TopCoW 2024 {modality.upper()} — 13-class Circle of Willis",
            reference="https://arxiv.org/abs/2312.17670",
        )

        logger.info("Processing %d subjects (workers=%d) ...", len(job_list), num_workers)
        results = run_parallel(_process_subject, job_list, num_workers,
                               desc=f"TopCoW-{modality.upper()}")
        _log_results(results)

    # ── Joint dataset: CT + MR in one folder ──────────────────────────────────
    # Domain is encoded in the subject_id prefix (topcow_ct_* vs topcow_mr_*)
    # nnU-Net sees them as the same task with a domain_id metadata field.
    _build_joint_topcow(do_aux, do_vesselness, num_workers)


def _build_joint_topcow(do_aux: bool, do_vesselness: bool, num_workers: int) -> None:
    """
    Build Dataset104_TopCoW_joint: CT and MR in one training set.
    Images from both modalities are included; their preprocessed versions
    are symlinked / re-used from individual CT/MR datasets to avoid re-computation.
    A split_domain.json is written listing which subjects are CT vs MR.
    """
    ds_id   = cfg.NNUNET_DATASET_IDS["TopCoW_joint"]
    ds_name = "TopCoW_joint"
    imgs_out, lbls_out, ds_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)

    ct_src  = NNUNET_RAW_ROOT / "Dataset102_TopCoW_CT"
    mr_src  = NNUNET_RAW_ROOT / "Dataset103_TopCoW_MR"

    domain_map = {}
    count = 0

    for src_imgs, src_lbls, domain_tag in [
        (ct_src / "imagesTr", ct_src / "labelsTr", "ct"),
        (mr_src / "imagesTr", mr_src / "labelsTr", "mr"),
    ]:
        if not src_imgs.exists():
            logger.warning("Joint: source %s not found – run topcow first.", src_imgs)
            continue
        for img_file in sorted(src_imgs.glob("*.nii.gz")):
            subj_id = img_file.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
            # Symlink image
            dst_img = imgs_out / img_file.name
            if not dst_img.exists():
                dst_img.symlink_to(img_file.resolve())
            # Symlink label
            lbl_file = src_lbls / f"{subj_id}.nii.gz"
            if lbl_file.exists():
                dst_lbl = lbls_out / lbl_file.name
                if not dst_lbl.exists():
                    dst_lbl.symlink_to(lbl_file.resolve())
            domain_map[subj_id] = domain_tag
            count += 1

    cow_labels = {v: k for k, v in cfg.TOPCOW_LABEL_SUBSET.items()}
    write_nnunet_dataset_json(
        output_dir=ds_root,
        dataset_name=ds_name,
        dataset_id=ds_id,
        channel_names={0: "angiography"},
        labels=cow_labels,
        num_training=count,
        description="TopCoW 2024 CT+MR joint — domain-conditioned normalization training",
        reference="https://arxiv.org/abs/2312.17670",
    )

    with open(ds_root / "split_domain.json", "w") as f:
        json.dump(domain_map, f, indent=2)

    logger.info("Joint TopCoW: %d total subjects.  split_domain.json written.", count)


def process_crossdomain(do_aux: bool, do_vesselness: bool, num_workers: int) -> None:
    """
    Cross-domain brain subsets — these are ZERO-SHOT TEST sets only.
    We preprocess them (resample, normalise) but do NOT include them in any
    training dataset.json.  They are placed in nnUNet_raw as standalone folders
    with an appropriate dataset.json for inference evaluation.

    CTA_LargeIA: only labels exist (no images) → generate aux maps from labels only.
    """
    subset_configs = [
        # (key,        modality, spacing_key,  ct_window,                   ds_id_offset)
        ("CTA_ISLES2024",  "ct", "brain_ct", cfg.CT_WINDOW["brain_ct"],   200),
        ("MRA_IXI_HH",     "mr", "brain_mr", None,                        201),
        ("MRA_Lausanne",   "mr", "brain_mr", None,                        202),
    ]

    for key, modality, spacing_key, ct_win, ds_id in subset_configs:
        ds_root_src = cfg.CROSS_DOMAIN_BRAIN.get(key)
        if ds_root_src is None or not ds_root_src.exists():
            logger.warning("Cross-domain %s not found – skipping.", key)
            continue

        logger.info("=== Cross-domain: %s ===", key)

        imgs_src = ds_root_src / "imagesTr"
        lbls_src = ds_root_src / "cow_seg_labelsTr"
        img_files = sorted(imgs_src.glob("*.nii.gz")) if imgs_src.exists() else []

        if not img_files:
            logger.warning("  No images in %s – skipping.", imgs_src)
            continue

        ds_name  = f"CrossDomain_{key}"
        imgs_out, lbls_out, out_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)

        job_list = []
        for img_path in img_files:
            stem    = img_path.name.replace(".nii.gz", "").replace("_0000", "")
            subj_id = stem
            lbl_path = lbls_src / f"{subj_id}.nii.gz"
            if not lbl_path.exists():
                # Try with dataset-specific naming conventions
                lbl_path = lbls_src / f"{img_path.name}"

            job_list.append({
                "img_path":    str(img_path),
                "lbl_path":    str(lbl_path),
                "out_img":     str(imgs_out / img_path.name),
                "out_lbl":     str(lbls_out / f"{subj_id}.nii.gz"),
                "modality":    modality,
                "spacing_key": spacing_key,
                "ct_window":   ct_win,
                "do_aux":      do_aux,
                "do_vesselness": do_vesselness,
                "aux_base":    str(AUX_ROOT / ds_name) if do_aux else "",
                "subject_id":  subj_id,
            })

        cow_labels = {v: k for k, v in cfg.TOPCOW_LABEL_SUBSET.items()}
        write_nnunet_dataset_json(
            output_dir=out_root,
            dataset_name=ds_name,
            dataset_id=ds_id,
            channel_names={0: "CTA" if modality == "ct" else "MRA"},
            labels=cow_labels,
            num_training=len(job_list),
            description=f"Cross-domain zero-shot test set: {key}",
        )

        logger.info("Processing %d subjects (workers=%d) ...", len(job_list), num_workers)
        results = run_parallel(_process_subject, job_list, num_workers,
                               desc=f"CrossDomain-{key}")
        _log_results(results)


def process_itktubetk(do_vesselness: bool, num_workers: int) -> None:
    """
    ITKTubeTK: unlabelled normal MRA.
    Preprocess images only (no labels, no aux maps that require labels).
    Used exclusively for tortuosity agreement testing (compare predicted
    tortuosity from model vs post-hoc pipeline tortuosity).
    """
    logger.info("=== ITKTubeTK (unlabelled MRA) ===")

    all_files = (
        sorted(cfg.ITKTUBETK_ROOT.glob("*.nii.gz")) +
        sorted(cfg.ITKTUBETK_ROOT2.glob("*.nii.gz"))
    )
    if not all_files:
        logger.warning("ITKTubeTK: no files found – skipping.")
        return

    ds_name  = "ITKTubeTK_MRA"
    ds_id    = 210
    imgs_out, _, out_root = nnunet_dirs(NNUNET_RAW_ROOT, ds_id, ds_name)

    job_list = []
    for img_path in all_files:
        subj_id = img_path.stem.replace(".nii", "")  # Normal001-MRA

        job_list.append({
            "img_path":    str(img_path),
            "lbl_path":    "",
            "out_img":     str(imgs_out / f"{subj_id}_0000.nii.gz"),
            "out_lbl":     "",
            "modality":    "mr",
            "spacing_key": "brain_mr",
            "ct_window":   None,
            "do_aux":      False,   # no labels → no aux maps
            "do_vesselness": do_vesselness,
            "aux_base":    str(AUX_ROOT / ds_name) if do_vesselness else "",
            "subject_id":  subj_id,
        })

    write_nnunet_dataset_json(
        output_dir=out_root,
        dataset_name=ds_name,
        dataset_id=ds_id,
        channel_names={0: "MRA"},
        labels={"background": 0},   # no ground truth labels
        num_training=0,             # these are inference-only subjects
        description="ITKTubeTK unlabelled normal MRA — tortuosity agreement test",
    )

    logger.info("Processing %d subjects (workers=%d) ...", len(job_list), num_workers)
    results = run_parallel(_process_subject, job_list, num_workers,
                           desc="ITKTubeTK")
    _log_results(results)


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
        description="Preprocess brain vessel datasets for nnU-Net."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "topbrain", "topcow", "crossdomain", "itktubetk"],
        help="Which dataset(s) to process.",
    )
    parser.add_argument(
        "--workers", type=int, default=cfg.NUM_WORKERS,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--skip-aux", action="store_true",
        help="Skip auxiliary map generation (vessel union, SDF, skeleton, curvature).",
    )
    parser.add_argument(
        "--skip-vesselness", action="store_true",
        help="Skip Frangi vesselness map generation (slower, optional).",
    )
    args = parser.parse_args()

    do_all       = "all" in args.tasks
    do_aux       = not args.skip_aux
    do_vess      = not args.skip_vesselness
    w            = args.workers

    logger.info("Output root: %s", cfg.OUTPUT_ROOT)
    logger.info("Aux maps: %s  |  Vesselness: %s  |  Workers: %d",
                do_aux, do_vess, w)

    if do_all or "topbrain" in args.tasks:
        process_topbrain(do_aux, do_vess, w)

    if do_all or "topcow" in args.tasks:
        process_topcow(do_aux, do_vess, w)

    if do_all or "crossdomain" in args.tasks:
        process_crossdomain(do_aux, do_vess, w)

    if do_all or "itktubetk" in args.tasks:
        process_itktubetk(do_vess, w)

    logger.info("Brain preprocessing complete.")


if __name__ == "__main__":
    main()
