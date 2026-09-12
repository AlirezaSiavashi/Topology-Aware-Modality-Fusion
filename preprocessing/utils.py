"""
utils.py
========
Reusable helpers shared across all preprocessing scripts:
  - NIfTI I/O via SimpleITK (preserves full header / orientation)
  - Resampling (image and label, correct interpolation per type)
  - Intensity normalisation (CT windowing, MRI z-score)
  - Vessel-union mask generation
  - SDF (Signed Distance Field) computation
  - nnU-Net dataset.json writer
  - Parallel worker pool wrapper
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_sitk(path: Union[str, Path]) -> sitk.Image:
    """Load a NIfTI (or any ITK-supported) image, preserving full metadata."""
    img = sitk.ReadImage(str(path))
    return img


def save_sitk(img: sitk.Image, path: Union[str, Path], compress: bool = True) -> None:
    """Save a SimpleITK image; creates parent directories automatically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), compress)


def sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    """
    Convert SimpleITK → numpy.  SimpleITK stores [x,y,z] (column-major);
    numpy is [z,y,x] (row-major).  We return [z,y,x] throughout.
    """
    return sitk.GetArrayFromImage(img)   # already [z,y,x]


def numpy_to_sitk(
    arr: np.ndarray,
    reference: sitk.Image,
    is_label: bool = False,
) -> sitk.Image:
    """
    Convert numpy [z,y,x] → SimpleITK, copying spacing/origin/direction
    from *reference*.
    """
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference)
    if is_label:
        out = sitk.Cast(out, sitk.sitkInt16)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Resampling
# ──────────────────────────────────────────────────────────────────────────────

def resample_image(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator=sitk.sitkBSpline,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample *img* to *target_spacing* (x, y, z in mm).
    Uses B-spline interpolation for images (smooth, avoids ringing).
    """
    orig_spacing = np.array(img.GetSpacing())           # (x, y, z)
    orig_size    = np.array(img.GetSize())              # (x, y, z)
    tgt_spacing  = np.array(target_spacing)

    new_size = np.round(orig_size * orig_spacing / tgt_spacing).astype(int)
    new_size = [max(1, int(s)) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tgt_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


def resample_label(
    lbl: sitk.Image,
    target_spacing: Tuple[float, float, float],
    reference_image: Optional[sitk.Image] = None,
) -> sitk.Image:
    """
    Resample a segmentation mask to *target_spacing* using nearest-neighbour
    interpolation (preserves integer label values exactly).
    If *reference_image* is given, its size/spacing is used (exact alignment).
    """
    orig_spacing = np.array(lbl.GetSpacing())
    orig_size    = np.array(lbl.GetSize())
    tgt_spacing  = np.array(target_spacing)

    if reference_image is not None:
        new_size     = list(reference_image.GetSize())
        tgt_spacing  = np.array(reference_image.GetSpacing())
    else:
        new_size = [max(1, int(round(s * o / t)))
                    for s, o, t in zip(orig_size, orig_spacing, tgt_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tgt_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(lbl.GetDirection())
    resampler.SetOutputOrigin(lbl.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(lbl)


# ──────────────────────────────────────────────────────────────────────────────
# Orientation
# ──────────────────────────────────────────────────────────────────────────────

def reorient_to_lps(img: sitk.Image) -> sitk.Image:
    """
    Reorient image to LPS+ (Left-Posterior-Superior) canonical orientation,
    consistent with TopBrain / TopCoW releases and nnU-Net convention.
    """
    return sitk.DICOMOrient(img, "LPS")


# ──────────────────────────────────────────────────────────────────────────────
# Intensity normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_ct(
    arr: np.ndarray,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    """
    CT: clip to [window_min, window_max], then scale to [0, 1].
    nnU-Net internally applies its own z-score after this, but a consistent
    windowing removes scanner-specific outliers and HU drift before that.
    """
    arr = np.clip(arr.astype(np.float32), window_min, window_max)
    arr = (arr - window_min) / (window_max - window_min + 1e-8)
    return arr


def normalize_mri(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    MRI: z-score normalisation computed inside the brain mask (foreground).
    Falls back to whole-volume z-score if no mask is provided.
    """
    arr = arr.astype(np.float32)
    if mask is not None and mask.sum() > 0:
        mu  = arr[mask > 0].mean()
        std = arr[mask > 0].std() + 1e-8
    else:
        mu  = arr.mean()
        std = arr.std() + 1e-8
    return (arr - mu) / std


def compute_brain_mask(arr: np.ndarray, percentile: float = 5.0) -> np.ndarray:
    """
    Simple intensity-threshold brain mask for MRA/MRI.
    Voxels above the p-th percentile of the nonzero region are considered brain.
    """
    nonzero = arr[arr > 0]
    if len(nonzero) == 0:
        return np.ones_like(arr, dtype=bool)
    thresh = np.percentile(nonzero, percentile)
    return arr > thresh


# ──────────────────────────────────────────────────────────────────────────────
# Vessel-union mask
# ──────────────────────────────────────────────────────────────────────────────

def make_vessel_union(
    label_arr: np.ndarray,
    foreground_classes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Collapse a multi-class label map to a binary vessel-union mask.
    Required by cbDice (topology loss operates on binary foreground).

    Parameters
    ----------
    label_arr : integer array [z,y,x]
    foreground_classes : list of class ids to include; if None, all nonzero.

    Returns
    -------
    binary uint8 array [z,y,x]
    """
    if foreground_classes is None:
        union = (label_arr > 0).astype(np.uint8)
    else:
        union = np.zeros_like(label_arr, dtype=np.uint8)
        for c in foreground_classes:
            union[label_arr == c] = 1
    return union


# ──────────────────────────────────────────────────────────────────────────────
# Signed Distance Field (SDF)
# ──────────────────────────────────────────────────────────────────────────────

def compute_sdf(
    binary_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    clip_value: float = 50.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the Signed Distance Field from a binary vessel mask:
        SDF(x) = EDT_background(x) - EDT_foreground(x)

    Positive values = background (distance to nearest vessel surface).
    Negative values = inside vessel (distance to nearest background surface).

    Parameters
    ----------
    binary_mask : uint8 array [z,y,x], foreground=1
    spacing     : voxel size in mm (z, y, x) – used for physical-space EDT
    clip_value  : mm; values clipped to [-clip, clip]
    normalize   : if True, divide by clip_value → [-1, 1]

    Returns
    -------
    float32 SDF array [z,y,x]
    """
    # scipy EDT takes spacing in (row, col, ...) = (z, y, x) order
    sp = (spacing[2], spacing[1], spacing[0])   # z,y,x

    fg = binary_mask.astype(bool)
    bg = ~fg

    edt_fg = distance_transform_edt(fg, sampling=sp).astype(np.float32)  # inside vessel
    edt_bg = distance_transform_edt(bg, sampling=sp).astype(np.float32)  # outside vessel

    sdf = edt_bg - edt_fg
    sdf = np.clip(sdf, -clip_value, clip_value)
    if normalize:
        sdf = sdf / clip_value
    return sdf


# ──────────────────────────────────────────────────────────────────────────────
# nnU-Net dataset.json writer
# ──────────────────────────────────────────────────────────────────────────────

def write_nnunet_dataset_json(
    output_dir: Union[str, Path],
    dataset_name: str,
    dataset_id: int,
    channel_names: Dict[int, str],
    labels: Dict[str, int],
    num_training: int,
    file_ending: str = ".nii.gz",
    description: str = "",
    reference: str = "",
    licence: str = "open use, attribution required",
    overwrite_image_reader_writer: str = "SimpleITKIO",
) -> None:
    """
    Write the dataset.json required by nnU-Net v2.

    Parameters
    ----------
    channel_names : {0: "CT"} or {0: "MRI_MRA", 1: "vesselness"}
    labels        : {"background": 0, "vessel": 1, ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = {
        "channel_names": {str(k): v for k, v in channel_names.items()},
        "labels":        labels,
        "numTraining":   num_training,
        "file_ending":   file_ending,
        "name":          dataset_name,
        "description":   description,
        "reference":     reference,
        "licence":       licence,
        "overwrite_image_reader_writer": overwrite_image_reader_writer,
        "dataset_id":    dataset_id,
    }

    out_path = output_dir / "dataset.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    logger.info("Wrote %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# nnU-Net directory structure helpers
# ──────────────────────────────────────────────────────────────────────────────

def nnunet_dirs(
    dataset_root: Union[str, Path],
    dataset_id: int,
    dataset_name: str,
) -> Tuple[Path, Path, Path]:
    """
    Return (images_tr, labels_tr, dataset_root) following nnU-Net v2 convention:
        Dataset{ID:03d}_{Name}/
            imagesTr/
            labelsTr/
            dataset.json
    """
    folder_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    root = Path(dataset_root) / folder_name
    images_tr = root / "imagesTr"
    labels_tr  = root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    return images_tr, labels_tr, root


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary-map output directory helpers
# ──────────────────────────────────────────────────────────────────────────────

def aux_dirs(dataset_root: Union[str, Path], dataset_name: str) -> Dict[str, Path]:
    """
    Create and return paths for auxiliary supervision maps:
        vessel_union/   – binary foreground mask
        sdf/            – signed distance field
        skeleton/       – binary centreline
        curvature/      – voxel-wise curvature field
    """
    root = Path(dataset_root) / dataset_name / "aux"
    dirs = {}
    for name in ["vessel_union", "sdf", "skeleton", "curvature"]:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        dirs[name] = p
    return dirs


# ──────────────────────────────────────────────────────────────────────────────
# Parallel processing
# ──────────────────────────────────────────────────────────────────────────────

def run_parallel(fn, items: list, num_workers: int = 8, desc: str = "") -> list:
    """
    Run *fn* over *items* using a ProcessPoolExecutor.
    Falls back to sequential execution if num_workers == 1 (easier debugging).
    Returns list of results (order preserved).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = [None] * len(items)

    if num_workers <= 1:
        for i, item in enumerate(items):
            results[i] = fn(item)
            if desc:
                logger.info("%s  [%d/%d]", desc, i + 1, len(items))
        return results

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done += 1
            if desc:
                logger.info("%s  [%d/%d]", desc, done, len(items))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Vesselness (Frangi) helper – wraps skimage's hessian_matrix_eigvals
# ──────────────────────────────────────────────────────────────────────────────

def compute_vesselness(
    image_arr: np.ndarray,
    spacing: Tuple[float, float, float],
    sigmas: Sequence[float] = (0.5, 1.0, 2.0),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
    black_ridges: bool = False,
) -> np.ndarray:
    """
    Multi-scale Frangi vesselness filter (3-D).

    Parameters
    ----------
    sigmas      : physical-space scales in mm; converted to voxel-space per axis
    black_ridges: True for dark vessels on bright background (some MRA)
    black_ridges: False for bright vessels on dark background (standard)

    Returns
    -------
    vesselness map, float32, range [0, 1]

    Notes
    -----
    skimage's frangi() uses voxel-space sigmas; we correct for anisotropic
    spacing by scaling sigma per axis.
    """
    from skimage.filters import frangi

    sp = np.array(spacing)          # (z, y, x) in mm
    out = np.zeros_like(image_arr, dtype=np.float32)

    for sigma_mm in sigmas:
        # Convert physical sigma to voxel sigma per axis
        sigma_vox = sigma_mm / sp   # (sigma_z, sigma_y, sigma_x)
        v = frangi(
            image_arr.astype(np.float32),
            sigmas=sigma_vox,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=black_ridges,
        )
        out = np.maximum(out, v)

    # Robust normalisation
    p_hi = np.percentile(out, 99.9)
    if p_hi > 0:
        out = np.clip(out / p_hi, 0, 1)
    return out.astype(np.float32)
