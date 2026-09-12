"""
skeleton_curvature.py
=====================
Ground-truth centerline extraction and voxel-wise curvature pipeline.

Steps (mirroring the proposal's tortuosity pipeline):
  A. Skeletonise the vessel-union binary mask (3-D thinning → Lee algorithm)
  B. Build branch graph (nodes = skeleton voxels; edges = 26-connectivity)
  C. Decompose into branches (endpoint-to-bifurcation segments)
  D. Prune short branches (noise artefacts)
  E. Fit a parametric spline per branch
  F. Compute arc-length curvature κ(s) per branch point
  G. Rasterise curvature back to voxel volume (sparse, skeleton voxels only)
  H. Compute tortuosity metrics per branch: DM, SOAM, RMS-curvature

All outputs are numpy arrays ready to be saved as NIfTI by the caller.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from scipy.interpolate import splprep, splev
from scipy.ndimage import label as connected_components
from skimage.morphology import skeletonize as skimage_skeletonize

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# A.  Skeletonisation
# ──────────────────────────────────────────────────────────────────────────────

def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    3-D topology-preserving thinning (Lee 1994 algorithm via scikit-image).

    Parameters
    ----------
    binary_mask : bool/uint8 array [z,y,x]

    Returns
    -------
    skeleton : bool array [z,y,x], 1-voxel-wide centrelines
    """
    mask = binary_mask.astype(bool)
    # skimage skeletonize_3d uses Lee algorithm – topology-preserving
    skel = skimage_skeletonize(mask)
    return skel.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# B.  Skeleton graph construction
# ──────────────────────────────────────────────────────────────────────────────

_NEIGHBOR_OFFSETS_26 = np.array(
    [(dz, dy, dx)
     for dz in (-1, 0, 1)
     for dy in (-1, 0, 1)
     for dx in (-1, 0, 1)
     if not (dz == 0 and dy == 0 and dx == 0)],
    dtype=np.int32,
)


def build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Convert a binary skeleton volume to a NetworkX graph.

    Nodes  : (z, y, x) tuples for every skeleton voxel
    Edges  : connect 26-connected neighbours
    """
    coords = np.argwhere(skeleton > 0)   # (N, 3) in [z, y, x]
    coord_set = {tuple(c) for c in coords}

    G = nx.Graph()
    G.add_nodes_from(map(tuple, coords))

    for vox in coords:
        z, y, x = vox
        for dz, dy, dx in _NEIGHBOR_OFFSETS_26:
            nb = (z + dz, y + dy, x + dx)
            if nb in coord_set and not G.has_edge(tuple(vox), nb):
                G.add_edge(tuple(vox), nb)

    return G


# ──────────────────────────────────────────────────────────────────────────────
# C & D.  Branch decomposition + pruning
# ──────────────────────────────────────────────────────────────────────────────

def _classify_nodes(G: nx.Graph) -> Tuple[set, set, set]:
    """
    Classify skeleton graph nodes into:
      - endpoints     : degree == 1
      - bifurcations  : degree >= 3
      - internal      : degree == 2
    """
    endpoints    = {n for n, d in G.degree() if d == 1}
    bifurcations = {n for n, d in G.degree() if d >= 3}
    internal     = set(G.nodes) - endpoints - bifurcations
    return endpoints, bifurcations, internal


def extract_branches(G: nx.Graph, min_length: int = 5) -> List[List[Tuple]]:
    """
    Decompose skeleton graph into branches (ordered lists of voxel coordinates).

    A branch runs between any two junction nodes (endpoint or bifurcation).
    Branches shorter than *min_length* voxels are discarded.

    Returns
    -------
    List of branches; each branch is an ordered list of (z, y, x) tuples.
    """
    endpoints, bifurcations, _ = _classify_nodes(G)
    junctions = endpoints | bifurcations

    branches = []
    visited_edges = set()

    def _traverse(start, prev):
        """Follow internal nodes until next junction."""
        path = [start]
        cur, prev_ = start, prev
        while True:
            neighbors = [n for n in G.neighbors(cur) if n != prev_]
            if len(neighbors) == 0:
                break
            # Continue along the single un-visited internal neighbour
            non_junction = [n for n in neighbors if n not in junctions]
            if non_junction:
                nxt = non_junction[0]
                edge = (min(cur, nxt), max(cur, nxt))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                path.append(nxt)
                prev_, cur = cur, nxt
            else:
                # Reached a junction
                path.append(neighbors[0])
                break
        return path

    for node in junctions:
        for nb in G.neighbors(node):
            edge = (min(node, nb), max(node, nb))
            if edge not in visited_edges:
                visited_edges.add(edge)
                if nb in junctions:
                    branch = [node, nb]
                else:
                    branch = [node] + _traverse(nb, node)
                if len(branch) >= min_length:
                    branches.append(branch)

    return branches


# ──────────────────────────────────────────────────────────────────────────────
# E.  Spline fitting per branch
# ──────────────────────────────────────────────────────────────────────────────

def fit_branch_spline(
    branch_coords: List[Tuple],
    spacing: Tuple[float, float, float],
    smoothing: float = 0.5,
    num_points: int = 200,
) -> Optional[np.ndarray]:
    """
    Fit a parametric B-spline to a branch and resample it uniformly.

    Parameters
    ----------
    branch_coords : ordered [(z,y,x), ...] in voxel space
    spacing       : (sz, sy, sx) voxel size in mm
    smoothing     : splprep smoothing factor (0 = exact interpolation)
    num_points    : number of points on the output curve

    Returns
    -------
    pts_mm : (num_points, 3) float64 array in physical mm space [z,y,x]
    or None if branch is too short to fit.
    """
    coords = np.array(branch_coords, dtype=np.float64)   # (N, 3) [z,y,x]
    if len(coords) < 4:
        return None

    # Convert to physical mm space
    sp = np.array(spacing)   # (sz, sy, sx)
    pts_mm = coords * sp[np.newaxis, :]

    # Remove duplicate points (can arise in dense skeletons)
    diffs = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    keep  = np.concatenate([[True], diffs > 1e-6])
    pts_mm = pts_mm[keep]

    if len(pts_mm) < 4:
        return None

    try:
        # splprep expects (3, N)
        tck, _ = splprep(pts_mm.T, s=smoothing * len(pts_mm), k=3)
        u_new  = np.linspace(0, 1, num_points)
        curve  = np.array(splev(u_new, tck)).T  # (num_points, 3)
        return curve
    except Exception as e:
        logger.debug("Spline fit failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# F.  Curvature computation along spline
# ──────────────────────────────────────────────────────────────────────────────

def compute_spline_curvature(curve_mm: np.ndarray) -> np.ndarray:
    """
    Compute curvature κ(s) at each point on the spline using the
    cross-product formula: κ = |r' × r''| / |r'|³

    Parameters
    ----------
    curve_mm : (N, 3) float64, arc-length parameterised curve in mm

    Returns
    -------
    kappa : (N,) float64 curvature in mm⁻¹
    """
    r_prime  = np.gradient(curve_mm, axis=0)   # first derivative
    r_dprime = np.gradient(r_prime,  axis=0)   # second derivative

    cross    = np.cross(r_prime, r_dprime)      # (N, 3)
    cross_norm = np.linalg.norm(cross, axis=1)  # (N,)
    speed    = np.linalg.norm(r_prime, axis=1)  # (N,)

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(speed > 1e-8, cross_norm / (speed ** 3 + 1e-12), 0.0)

    return kappa.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# G.  Rasterise curvature back to voxel space
# ──────────────────────────────────────────────────────────────────────────────

def rasterize_curvature(
    branches:      List[List[Tuple]],
    branch_curves: List[Optional[np.ndarray]],
    branch_kappas: List[Optional[np.ndarray]],
    volume_shape:  Tuple[int, int, int],
    spacing:       Tuple[float, float, float],
    clip_value:    float = 5.0,
) -> np.ndarray:
    """
    Assign voxel-wise curvature values to skeleton voxel positions.

    For each voxel in the original branch, we find the nearest point on the
    fitted spline and assign its curvature value.

    Returns
    -------
    curvature_vol : float32 [z,y,x], nonzero only at skeleton voxels
    """
    sp  = np.array(spacing)   # (sz, sy, sx)
    vol = np.zeros(volume_shape, dtype=np.float32)

    for branch, curve, kappa in zip(branches, branch_curves, branch_kappas):
        if curve is None or kappa is None:
            continue

        branch_arr = np.array(branch, dtype=np.float64)
        branch_mm  = branch_arr * sp[np.newaxis, :]  # (M, 3) physical mm

        # For each skeleton voxel, find nearest spline point
        diff = branch_mm[:, np.newaxis, :] - curve[np.newaxis, :, :]  # (M, K, 3)
        dist = np.linalg.norm(diff, axis=2)   # (M, K)
        nearest_idx = np.argmin(dist, axis=1)  # (M,)

        kappa_clipped = np.clip(kappa, 0, clip_value)

        for i, vox in enumerate(branch):
            z, y, x = int(vox[0]), int(vox[1]), int(vox[2])
            if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
                vol[z, y, x] = float(kappa_clipped[nearest_idx[i]])

    return vol


# ──────────────────────────────────────────────────────────────────────────────
# H.  Tortuosity metrics per branch
# ──────────────────────────────────────────────────────────────────────────────

def tortuosity_metrics(
    curve_mm: np.ndarray,
    kappa:    np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard vascular tortuosity metrics for one branch.

    Parameters
    ----------
    curve_mm : (N, 3) arc-length parameterised curve in physical mm
    kappa    : (N,) curvature in mm⁻¹

    Returns
    -------
    dict with keys:
        dm       – Distance Metric (arc / chord ratio), also called ICM
        soam     – Sum of Angles Metric (total turning angle normalised by arc-length)
        rms_kappa – RMS curvature
        arc_length – total arc length in mm
        chord_length – straight-line start-to-end distance in mm
    """
    # Arc length (cumulative sum of segment lengths)
    diffs = np.diff(curve_mm, axis=0)                      # (N-1, 3)
    seg_lengths = np.linalg.norm(diffs, axis=1)            # (N-1,)
    arc_length  = float(seg_lengths.sum())

    # Chord length
    chord_length = float(np.linalg.norm(curve_mm[-1] - curve_mm[0]))

    # DM (arc-over-chord ratio)
    dm = arc_length / (chord_length + 1e-8)

    # SOAM – sum of turning angles normalised by arc length
    tangents = diffs / (seg_lengths[:, np.newaxis] + 1e-12)   # unit tangents
    cos_angles = np.clip(
        (tangents[:-1] * tangents[1:]).sum(axis=1), -1, 1
    )
    angles = np.arccos(cos_angles)                            # (N-2,)
    soam   = float(angles.sum()) / (arc_length + 1e-8)

    # RMS curvature
    rms_kappa = float(np.sqrt(np.mean(kappa ** 2))) if len(kappa) > 0 else 0.0

    return {
        "dm":           dm,
        "soam":         soam,
        "rms_kappa":    rms_kappa,
        "arc_length":   arc_length,
        "chord_length": chord_length,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public high-level function
# ──────────────────────────────────────────────────────────────────────────────

def run_skeleton_curvature_pipeline(
    vessel_union:  np.ndarray,
    spacing:       Tuple[float, float, float],
    min_branch_len: int   = 5,
    smoothing:      float = 0.5,
    curvature_clip: float = 5.0,
    num_spline_pts: int   = 200,
) -> Dict:
    """
    Full pipeline: vessel-union mask → skeleton → curvature volume + tortuosity.

    Parameters
    ----------
    vessel_union    : binary uint8 [z,y,x]
    spacing         : voxel size (sz, sy, sx) in mm
    min_branch_len  : minimum branch length in voxels (prune shorter)
    smoothing       : spline smoothing factor
    curvature_clip  : clip curvature at this value (mm⁻¹)
    num_spline_pts  : points on re-sampled spline per branch

    Returns
    -------
    dict with:
        "skeleton"        : uint8 [z,y,x]
        "curvature"       : float32 [z,y,x]  (0 everywhere except skeleton voxels)
        "branch_metrics"  : list of per-branch tortuosity dicts
        "num_branches"    : int
    """
    logger.info("  Skeletonising mask (shape %s) ...", vessel_union.shape)
    skeleton = skeletonize_mask(vessel_union)

    num_skel_vox = int(skeleton.sum())
    if num_skel_vox == 0:
        logger.warning("  Empty skeleton – returning zero curvature.")
        return {
            "skeleton":       skeleton,
            "curvature":      np.zeros_like(vessel_union, dtype=np.float32),
            "branch_metrics": [],
            "num_branches":   0,
        }

    logger.info("  Skeleton voxels: %d  Building graph ...", num_skel_vox)
    G = build_skeleton_graph(skeleton)

    # Handle disconnected components separately (avoid spurious long branches)
    all_branches       = []
    all_curves         = []
    all_kappas         = []
    all_branch_metrics = []

    for cc_nodes in nx.connected_components(G):
        sub = G.subgraph(cc_nodes).copy()
        branches = extract_branches(sub, min_length=min_branch_len)

        for branch in branches:
            curve = fit_branch_spline(branch, spacing, smoothing, num_spline_pts)
            if curve is None:
                all_branches.append(branch)
                all_curves.append(None)
                all_kappas.append(None)
                continue

            kappa = compute_spline_curvature(curve)
            metrics = tortuosity_metrics(curve, kappa)

            all_branches.append(branch)
            all_curves.append(curve)
            all_kappas.append(kappa)
            all_branch_metrics.append(metrics)

    logger.info("  Branches extracted: %d  Rasterising curvature ...",
                len(all_branches))

    curvature_vol = rasterize_curvature(
        all_branches, all_curves, all_kappas,
        vessel_union.shape, spacing, curvature_clip,
    )

    return {
        "skeleton":       skeleton,
        "curvature":      curvature_vol,
        "branch_metrics": all_branch_metrics,
        "num_branches":   len(all_branch_metrics),
    }
