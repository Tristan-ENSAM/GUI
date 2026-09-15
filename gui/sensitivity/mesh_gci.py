# -*- coding: utf-8 -*-
"""Formal mesh convergence by Richardson extrapolation and the Grid Convergence
Index (GCI), per Roache / ASME V&V 20.

Unlike the Cauchy successive-difference check in ``mesh_opt.refine_until_stable``
(refine until the solution stops changing), this quantifies the discretization
error: from three systematically-refined meshes it reports the observed order of
convergence p, the Richardson-extrapolated (h->0) value, and a GCI uncertainty
band per monitored quantity, plus an asymptotic-range check.

Robust sampling (essential here): the ZOI field is reduced to a scalar per
quantity by a WINDOWED, EVF-MASKED, TIME-AVERAGED reduction sampled on a FIXED
grid via BILINEAR interpolation of the element-centroid field. Nearest-neighbour
(``mesh_opt.make_mesh_sample_fn``) is a bias when the element size changes
because the centroids move; and without the EVF mask + settled window the moving
material/void interface dominates the comparison (the noise floor seen in the
domain study). GCI assumes monotonic asymptotic convergence, so this reduction
is what gives p and the GCI a chance of being meaningful; the asymptotic-range
indicator flags when they are not.

Method (three meshes h1 < h2 < h3, h1 finest; r21 = h2/h1, r32 = h3/h2):
    e21 = f2 - f1,  e32 = f3 - f2
    p   = |ln|e32/e21| + q(p)| / ln(r21),  q(p) = ln((r21^p - s)/(r32^p - s)),
          s = sign(e32/e21)          (q = 0 when r21 = r32)
    f_ext = (r21^p f1 - f2) / (r21^p - 1)          (Richardson, h->0)
    GCI_fine = Fs |(f1 - f2)/f1| / (r21^p - 1),  Fs = 1.25 for >=3 meshes
    asymptotic range: GCI_32 / (r21^p GCI_21) ~ 1

Pure host-side Python (CPython 3.x). ``run_bundle(cfg)`` returns a
ResultsBundle-like object (see domain_convergence).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.mesh_opt import roi_grid
from gui.sensitivity.domain_opt import element_centroids_xy
from gui.sensitivity.runner_core import eulerian_instance
from gui.sensitivity.domain_convergence import window_mask, _history_window_mean

DEFAULT_QUANTITIES = ("EVF", "TEMP", "V1", "V2", "force")
_SAFETY_3PLUS = 1.25          # Roache safety factor for >= 3 meshes


# ---------------------------------------------------------------------------
# Pure GCI / Richardson math
# ---------------------------------------------------------------------------
def observed_order(f1: float, f2: float, f3: float, r21: float, r32: float,
                   tol: float = 1e-12, max_iter: int = 200) -> float:
    """Observed order of convergence p from three solutions (f1 = finest).

    Solves the Roache fixed point p = |ln|e32/e21| + q(p)| / ln(r21). For a
    constant refinement ratio (r21 == r32) this is exact (q = 0). Returns NaN
    when either difference is ~0 (already converged / undefined) or the ratio
    of differences is degenerate.
    """
    e21 = f2 - f1
    e32 = f3 - f2
    if abs(e21) < tol or abs(e32) < tol:
        return float("nan")
    ratio = e32 / e21
    if ratio <= 0.0:                       # sign change -> oscillatory, no order
        s = math.copysign(1.0, ratio)
    else:
        s = 1.0
    lnr21 = math.log(r21)
    p = abs(math.log(abs(ratio))) / lnr21          # q = 0 initial guess
    if abs(r21 - r32) < 1e-12:
        return p                                    # constant ratio: exact
    for _ in range(max_iter):
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
        except (ValueError, ZeroDivisionError):     # pragma: no cover
            return float("nan")
        p_new = abs(math.log(abs(ratio)) + q) / lnr21
        if abs(p_new - p) < tol:
            return p_new
        p = p_new
    return p


def extrapolated_value(f1: float, f2: float, r21: float, p: float) -> float:
    """Richardson-extrapolated (h->0) value from the two finest meshes."""
    denom = r21 ** p - 1.0
    if not math.isfinite(p) or abs(denom) < 1e-15:
        return float("nan")
    return (r21 ** p * f1 - f2) / denom


def gci(f_fine: float, f_coarse: float, r: float, p: float,
        safety: float = _SAFETY_3PLUS) -> float:
    """GCI on the FINE mesh of a pair: Fs |(f_fine-f_coarse)/f_fine| / (r^p-1).

    A relative discretization uncertainty (fraction; multiply by 100 for %).
    """
    denom = r ** p - 1.0
    if not math.isfinite(p) or f_fine == 0.0 or abs(denom) < 1e-15:
        return float("nan")
    e_a = abs((f_fine - f_coarse) / f_fine)
    return safety * e_a / denom


def asymptotic_ratio(e21: float, e32: float, r21: float, p: float) -> float:
    """|e32| / (r21^p * |e21|); ~1 means the meshes are in the asymptotic range.

    For an exact power-law error (f = f_exact + C h^p) this is identically 1,
    independent of any normalisation, so it is a cleaner indicator than the
    ratio of GCIs (which carries a spurious |f1|/|f2| factor)."""
    d = (r21 ** p) * abs(e21)
    if not math.isfinite(d) or d == 0.0:
        return float("nan")
    return abs(e32) / d


@dataclass
class QuantityGci:
    """GCI outcome for one monitored quantity."""
    f_fine: float                 # value on the finest mesh
    p: float                      # observed order of convergence
    f_extrapolated: float         # Richardson h->0 estimate
    gci_fine: float               # fine-grid GCI (relative uncertainty)
    gci_coarse: float             # coarse-pair GCI (for the asymptotic check)
    asymptotic_ratio: float       # ~1 in the asymptotic range
    monotonic: bool               # e21 and e32 share sign
    reliable: bool = True         # is the Richardson extrapolation trustworthy?


# Minimum r^p - 1 for the extrapolation to be numerically stable. When the
# quantity is already converged, the successive differences sit at the noise
# floor, p -> 0 and r^p - 1 -> 0, so f_extrapolated blows up (e.g. a force flat
# at -80 N/mm extrapolating to -118). Below this the extrapolation is declared
# unreliable and the finest value is used as the reference instead.
_DENOM_MIN = 0.1


def quantity_gci(f1: float, f2: float, f3: float, r21: float, r32: float,
                 safety: float = _SAFETY_3PLUS) -> QuantityGci:
    """Full GCI outcome for one quantity (f1 = finest, f3 = coarsest)."""
    p = observed_order(f1, f2, f3, r21, r32)
    g21 = gci(f1, f2, r21, p, safety)
    g32 = gci(f2, f3, r32, p, safety)
    e21, e32 = f2 - f1, f3 - f2
    monotonic = (e21 == 0.0 and e32 == 0.0) or (e21 * e32 > 0.0)
    f_ext = extrapolated_value(f1, f2, r21, p)
    denom = (r21 ** p - 1.0) if math.isfinite(p) else float("nan")
    reliable = bool(monotonic and math.isfinite(p) and math.isfinite(f_ext)
                    and f1 != 0.0 and math.isfinite(denom)
                    and denom >= _DENOM_MIN)
    return QuantityGci(
        f_fine=f1, p=p, f_extrapolated=f_ext,
        gci_fine=g21, gci_coarse=g32,
        asymptotic_ratio=asymptotic_ratio(e21, e32, r21, p),
        monotonic=monotonic, reliable=reliable)


# ---------------------------------------------------------------------------
# Robust ZOI reduction to a scalar per quantity (EVF-masked, windowed, bilinear)
# ---------------------------------------------------------------------------
def interp_weights(centroids, points):
    """Barycentric interpolation weights for `points` in the Delaunay
    triangulation of `centroids`, built ONCE. Returns (vertices, bary, inside):
    vertices (Np, 3) int node indices, bary (Np, 3) weights, inside (Np,) bool
    (False = outside the convex hull -> NaN on apply). Reuse the result across
    all frames AND all field variables of one mesh -- the triangulation of tens
    of thousands of centroids is far too costly to redo per frame."""
    from scipy.spatial import Delaunay
    centroids = np.asarray(centroids, dtype=float)
    pts = np.asarray(points, dtype=float)
    tri = Delaunay(centroids)
    simplex = tri.find_simplex(pts)
    inside = simplex >= 0
    simplex_safe = np.where(inside, simplex, 0)
    transform = tri.transform[simplex_safe]                      # (Np, 3, 2)
    delta = pts - transform[:, 2, :]                             # (Np, 2)
    bary2 = np.einsum("nij,nj->ni", transform[:, :2, :], delta)  # (Np, 2)
    bary = np.concatenate(
        [bary2, 1.0 - bary2.sum(axis=1, keepdims=True)], axis=1)  # (Np, 3)
    vertices = tri.simplices[simplex_safe]                       # (Np, 3)
    return vertices, bary, inside


def apply_weights(vals, weights) -> np.ndarray:
    """Apply precomputed `interp_weights` to a (Nt, Nc) value array -> (Nt, Np).
    Points outside the hull become NaN."""
    vertices, bary, inside = weights
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 1:
        vals = vals[None, :]
    out = np.einsum("tpk,pk->tp", vals[:, vertices], bary)       # (Nt, Np)
    out[:, ~inside] = np.nan
    return out


def bilinear_field(bundle, var, inst, points, frames=None, weights=None):
    """Resample element field `var` onto `points` (N_p, 2) by linear
    (barycentric) interpolation of the element centroids. Returns (N_t, N_p);
    points outside the centroid convex hull are NaN.

    Unlike nearest-neighbour, this does not snap to a moving centroid, so two
    element sizes are compared without the size-dependent snapping bias. The
    Delaunay triangulation is built ONCE here; pass `weights` (from
    `interp_weights`) to reuse it across variables and avoid re-triangulating.
    """
    vals = np.asarray(bundle.field(inst, var), dtype=float)
    if frames is not None:
        vals = vals[frames]
    if weights is None:
        weights = interp_weights(element_centroids_xy(bundle, inst), points)
    return apply_weights(vals, weights)


def _windowed_spatial_scalar(arr: np.ndarray, tmask: np.ndarray,
                             matmask: Optional[np.ndarray]) -> float:
    """Windowed, (optionally) EVF-masked time-mean per point, then the spatial
    mean over valid points -> one scalar. NaN if nothing is valid.

    Points outside the interpolation hull are NaN columns by construction, so
    the per-point mean is taken with the all-NaN RuntimeWarning suppressed and
    only finite points enter the spatial mean."""
    if not tmask.any():
        return float("nan")
    if matmask is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            per_pt = np.nanmean(arr[tmask], axis=0)          # (Np,)
    else:
        sel = matmask & tmask[:, None]
        cnt = sel.sum(axis=0)
        s = np.where(sel & np.isfinite(arr), arr, 0.0).sum(axis=0)
        per_pt = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    valid = np.isfinite(per_pt)
    if not valid.any():
        return float("nan")
    return float(np.mean(per_pt[valid]))


def zoi_scalars(bundle, zoi: Sequence[float], grid_step: float,
                field_vars: Sequence[str],
                window: Tuple[float, float] = (0.3, 1.0),
                evf_threshold: float = 0.9, evf_var: str = "EVF",
                force_channels: Optional[Dict[str, str]] = None,
                instance: Optional[str] = None) -> Dict[str, float]:
    """One representative scalar per monitored quantity for a single run.

    Field vars other than `evf_var` are EVF-masked (material only); `evf_var` is
    reduced unmasked (fill fraction everywhere). Forces come from the windowed
    mean of their history channel. All on a FIXED grid via bilinear sampling,
    with the Delaunay triangulation built ONCE and reused across variables.
    """
    inst = instance or eulerian_instance(bundle)
    if not inst:
        raise RuntimeError("No Eulerian instance (EVF carrier) found.")
    points = roi_grid(zoi, grid_step)
    tmask = window_mask(bundle.times, *window)
    weights = interp_weights(element_centroids_xy(bundle, inst), points)
    evf = bilinear_field(bundle, evf_var, inst, points, weights=weights)
    matmask = evf >= evf_threshold

    out: Dict[str, float] = {}
    for var in field_vars:
        if var == evf_var:
            out[evf_var] = _windowed_spatial_scalar(evf, tmask, None)
        else:
            arr = bilinear_field(bundle, var, inst, points, weights=weights)
            out[var] = _windowed_spatial_scalar(arr, tmask, matmask)
    for label, channel in dict(force_channels or {}).items():
        out[label] = _history_window_mean(bundle, channel, *window)
    return out


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------
@dataclass
class MeshGciResult:
    sizes: List[float] = field(default_factory=list)          # coarse..fine? see note
    scalars: Dict[float, Dict[str, float]] = field(default_factory=dict)
    per_quantity: Dict[str, QuantityGci] = field(default_factory=dict)
    recommended_size: Optional[float] = None      # coarsest within tolerance of f_ext
    in_asymptotic_range: bool = False
    n_runs: int = 0
    stopped_by: str = ""       # "ok"|"cancelled"|"nan"


def _mesh_sizes(finest: float, ratio: float, n: int,
                min_size: Optional[float]) -> List[float]:
    """n sizes finest, finest*ratio, ... (fine -> coarse). Refuses to go below
    min_size for the FINEST (the useful-resolution floor, e.g. ~6 um)."""
    if finest <= 0 or ratio <= 1.0 or n < 3:
        raise ValueError("need finest>0, ratio>1, n>=3 for a GCI study")
    if min_size is not None and finest < min_size:
        finest = float(min_size)
    return [finest * (ratio ** i) for i in range(n)]


def run_mesh_gci(
        run_bundle: Callable, base_cfg, zoi: Sequence[float],
        domain_dims: DomainDims, grid_step: float, finest_elem_size: float,
        ratio: float = 2.0, n_meshes: int = 3,
        tolerances: Optional[Dict[str, float]] = None,
        field_vars: Sequence[str] = ("EVF", "TEMP", "V1", "V2"),
        window: Tuple[float, float] = (0.3, 1.0), evf_threshold: float = 0.9,
        force_channels: Optional[Dict[str, str]] = None,
        min_elem_size: Optional[float] = None, safety: float = _SAFETY_3PLUS,
        should_cancel: Optional[Callable] = None,
        progress_cb: Optional[Callable] = None) -> MeshGciResult:
    """GCI/Richardson mesh convergence on a FIXED (large) domain.

    Runs `n_meshes` systematically-refined meshes (finest, finest*ratio, ...) at
    the fixed `domain_dims`, reduces each to scalar ZOI quantities, and computes
    the observed order p, the extrapolated value and the GCI per quantity from
    the three FINEST meshes. `recommended_size` is the coarsest mesh whose every
    quantity is within `tolerances` (default 1 %) of the extrapolated value --
    the cheapest mesh with a bounded discretization error.

    The domain must be large enough that the ZOI is boundary-independent (run
    the domain study, or use a conservative domain); otherwise p/GCI describe a
    boundary-contaminated field, not mesh error.
    """
    if force_channels is None:
        force_channels = {"Fc": "RF1_RP", "Ff": "RF2_RP"}
    if tolerances is None:
        tolerances = {q: 0.01 for q in DEFAULT_QUANTITIES}
    sizes = _mesh_sizes(finest_elem_size, ratio, n_meshes, min_elem_size)
    result = MeshGciResult(sizes=list(sizes))

    for h in sizes:
        if should_cancel is not None and should_cancel():
            result.stopped_by = "cancelled"
            return result
        base_cfg.elem_size = float(h)
        base_cfg.euler_geometry.h_wp = domain_dims.h_wp
        base_cfg.euler_geometry.h_void = domain_dims.h_void
        base_cfg.euler_geometry.l_wp = domain_dims.l_wp
        base_cfg.euler_geometry.l_void = domain_dims.l_void
        bundle = run_bundle(base_cfg)
        if bundle is None:
            raise RuntimeError("mesh run produced no bundle (elem_size=%.4g)" % h)
        result.scalars[h] = zoi_scalars(
            bundle, zoi, grid_step, field_vars, window=window,
            evf_threshold=evf_threshold, force_channels=force_channels)
        # The RF at the tool RP is the reaction over the model WIDTH, and the
        # slab depth = elem_size (cel_model.py:283/289/314), so the raw force
        # scales with the mesh. Divide by the width (= h) to get force per unit
        # width (N/mm): mesh-comparable and matching the experimental cutting
        # force. (Domain sizing keeps the mesh fixed, so it needs no such step.)
        if h > 0:
            for label in force_channels:
                v = result.scalars[h].get(label)
                if v is not None and math.isfinite(v):
                    result.scalars[h][label] = v / h
        result.n_runs += 1
        if progress_cb:
            progress_cb({"phase": "mesh_gci", "elem_size": h,
                         "scalars": result.scalars[h], "n_runs": result.n_runs})

    # Three finest meshes: sizes[0] (finest) .. sizes[2].
    h1, h2, h3 = sizes[0], sizes[1], sizes[2]
    r21, r32 = h2 / h1, h3 / h2
    quantities = list(field_vars) + list(force_channels.keys())
    for q in quantities:
        f1 = result.scalars[h1].get(q, float("nan"))
        f2 = result.scalars[h2].get(q, float("nan"))
        f3 = result.scalars[h3].get(q, float("nan"))
        result.per_quantity[q] = quantity_gci(f1, f2, f3, r21, r32, safety)

    # Asymptotic range: every quantity's ratio within +-10 % of 1.
    ratios = [g.asymptotic_ratio for g in result.per_quantity.values()
              if math.isfinite(g.asymptotic_ratio)]
    result.in_asymptotic_range = bool(ratios) and all(
        0.9 <= a <= 1.1 for a in ratios)

    # Recommended = coarsest size whose every quantity is within tolerance of a
    # reference: the Richardson-extrapolated (h->0) value when that extrapolation
    # is reliable, else the FINEST value (a quantity already converged below the
    # noise floor has a meaningless extrapolate -- see QuantityGci.reliable --
    # and must not block the recommendation with a garbage h->0 estimate).
    def _within_tol(h: float) -> bool:
        for q in quantities:
            g = result.per_quantity.get(q)
            tol = tolerances.get(q)
            if g is None or tol is None:
                continue
            fq = result.scalars[h].get(q, float("nan"))
            ref = g.f_extrapolated if g.reliable else g.f_fine
            if not (math.isfinite(fq) and math.isfinite(ref)) or ref == 0.0:
                return False
            if abs((fq - ref) / ref) > tol:
                return False
        return True

    for h in sorted(sizes, reverse=True):        # coarsest first
        if _within_tol(h):
            result.recommended_size = h
            break

    if any(not math.isfinite(g.p) for g in result.per_quantity.values()):
        result.stopped_by = result.stopped_by or "nan"
    else:
        result.stopped_by = result.stopped_by or "ok"
    return result
