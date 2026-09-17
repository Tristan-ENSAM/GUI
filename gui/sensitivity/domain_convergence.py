# -*- coding: utf-8 -*-
"""Eulerian domain sizing by CONVERGENCE (Option B).

Rationale
---------
Domain sizing is an *independence* problem, not a derivative one: we want the
smallest Eulerian domain whose boundaries no longer influence a fixed
measurement zone (the ZOI). The forward-difference Jacobian approach (a
``domain_jacobian`` module, since abandoned and removed) divided a field
difference by the perturbation step; when
the true boundary influence is small it is dominated by a step-independent
floor, and the linearity ratio J(h)/J(2h) saturates at 2 ("noise, not
sensitivity"). This module removes that failure mode by:

  1. Comparing the RAW change of the ZOI field between two domain sizes against
     a tolerance -- no division by the step, so no ratio to saturate.
  2. Reducing each field to a TIME-MEAN over a settled window and MASKING it by
     EVF, so points that sit in the (moving) material/void interface do not
     contaminate the comparison. This is the dominant floor for material fields
     (TEMP, V) once chip serration is excluded.
  3. DECOUPLING the ZOI from the domain: the ZOI is a fixed measurement box,
     independent of the domain dimensions, required only to stay inside the
     domain (with a margin). The domain grows OUTWARD around it.

ROI vs ZOI
----------
These are two DISTINCT zones with DIFFERENT grids, defined in different tabs:

  * ROI -- the model OUTPUT set, edited in the Geometry tab and matched to the
    real measurement fields (DIC / IRT) for the simulation-vs-experiment
    comparison. Materialised in the Abaqus model as ROI_node / ROI_elem.
  * ZOI -- the zone used HERE, defined by the user in the Optimization tab to
    size the Eulerian domain. It has its own grid (`grid_step` below) and is
    sampled host-side; it is not an Abaqus set. By default it may coincide with
    the ROI but is a separate object with its own extent and resolution.

Coordinate convention (see ``abaqus_scripts/cel_model.py:281``): the Eulerian
domain is the rectangle (-l_wp, -h_wp) -> (l_void, h_void); the cutting corner
is the fixed origin. Growing l_wp/h_wp extends the domain in -x/-y, l_void/h_void
in +x/+y. Because dimensions are floored to whole elements
(``cel_common.discretize``) and the instance is seeded by a uniform size
(``cel_model.py:399``), the ZOI node grid is invariant when the domain grows in
whole-element increments, so a fixed ROI grid is directly comparable between
sizes.

This module is pure host-side Python (CPython 3.x). ``run_bundle(cfg)`` must
return a ``ResultsBundle``-like object exposing ``times``, ``field(inst, var)``,
``history(name)``, ``history_time`` and ``instance(name).field_variables``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gui.core.domain_sizing import (
    DomainDims, DIMENSION_NAMES, diagonal, diagonal_limit,
)
from gui.sensitivity.mesh_opt import roi_grid, nearest_samples
from gui.sensitivity.runner_core import eulerian_instance

# Only these dimensions may grow OUTWARD without invading the ZOI; all four do
# here, but the driver additionally checks ZOI containment before every grow.
GROWABLE_DIMENSIONS = DIMENSION_NAMES

DEFAULT_QUANTITIES = ("EVF", "TEMP", "V1", "V2", "force")


# ---------------------------------------------------------------------------
# Small dim helpers (kept local to avoid importing private names)
# ---------------------------------------------------------------------------
def _as_dict(dims: DomainDims) -> Dict[str, float]:
    return {n: float(getattr(dims, n)) for n in DIMENSION_NAMES}


def _with(dims: DomainDims, name: str, value: float) -> DomainDims:
    d = _as_dict(dims)
    d[name] = value
    return DomainDims(**d)


def domain_bounds(dims: DomainDims) -> Tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of the Eulerian domain for `dims`.

    Mirrors the sketch rectangle in cel_model.py:281 -- origin at the cutting
    corner, so xmin=-l_wp, xmax=+l_void, ymin=-h_wp, ymax=+h_void.
    """
    return (-dims.l_wp, dims.l_void, -dims.h_wp, dims.h_void)


def zoi_inside(dims: DomainDims, zoi: Sequence[float],
               margin: float = 0.0) -> bool:
    """True when the ZOI box lies inside the domain with `margin` on every side.

    zoi = (xmin, xmax, ymin, ymax) in the model frame (same frame as the
    domain). `margin` is a distance (e.g. one element size).
    """
    zx0, zx1, zy0, zy1 = zoi
    dx0, dx1, dy0, dy1 = domain_bounds(dims)
    return (dx0 <= zx0 - margin and dx1 >= zx1 + margin and
            dy0 <= zy0 - margin and dy1 >= zy1 + margin)


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------
def window_mask(times: np.ndarray, w_start: float = 0.3,
                w_end: float = 1.0) -> np.ndarray:
    """Boolean mask of frames whose time lies in [w_start, w_end] * t_end.

    `times` are frame times (monotonically increasing). w_start/w_end are two
    fractions in [0, 1] delimiting the settled window: (0.3, 1.0) discards the
    first 30 % transient and keeps the rest. w_start <= w_end is required.
    """
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return np.zeros(0, dtype=bool)
    if not (0.0 <= w_start <= w_end <= 1.0):
        raise ValueError(
            "window must satisfy 0 <= w_start <= w_end <= 1, got (%r, %r)"
            % (w_start, w_end))
    t_end = t[-1]
    if t_end <= 0.0:
        return np.zeros(t.shape, dtype=bool)
    return (t >= w_start * t_end) & (t <= w_end * t_end)


# ---------------------------------------------------------------------------
# ZOI reduction: windowed, EVF-masked time-mean per grid point
# ---------------------------------------------------------------------------
@dataclass
class ZoiReduction:
    """A domain size reduced to comparable per-point ZOI quantities."""
    dims: DomainDims
    means: Dict[str, np.ndarray] = field(default_factory=dict)   # var -> (Np,)
    valid: Dict[str, np.ndarray] = field(default_factory=dict)   # var -> (Np,) bool
    force_mean: Optional[float] = None
    n_window_frames: int = 0


def _masked_time_mean(arr: np.ndarray, tmask: np.ndarray,
                      matmask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point mean of `arr` over frames selected by tmask AND material mask.

    arr, matmask: (Nt, Np); tmask: (Nt,). A point is valid where it is material
    (matmask) in at least one windowed frame; its mean is taken over exactly
    those frames. Invalid points -> NaN.
    """
    sel = matmask & tmask[:, None]
    cnt = sel.sum(axis=0)
    s = np.where(sel, arr, 0.0).sum(axis=0)
    valid = cnt > 0
    mean = np.where(valid, s / np.maximum(cnt, 1), np.nan)
    return mean, valid


def _time_mean(arr: np.ndarray, tmask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point mean of `arr` over the windowed frames (no material mask)."""
    if not tmask.any():
        np_ = arr.shape[1] if arr.ndim == 2 else 0
        return np.full(np_, np.nan), np.zeros(np_, dtype=bool)
    sub = arr[tmask]
    mean = np.nanmean(sub, axis=0)
    return mean, np.isfinite(mean)


def _history_window_mean(bundle, channel: str,
                         w_start: float, w_end: float) -> Optional[float]:
    """Windowed mean of a scalar history channel, or None if unavailable.

    Uses the HISTORY time base (bundle.history_time), which is sampled
    independently from the field frames.
    """
    try:
        y = np.asarray(bundle.history(channel), dtype=float)
        t = np.asarray(bundle.history_time, dtype=float)
    except Exception:
        return None
    if y.size == 0 or t.size != y.size:
        return None
    m = window_mask(t, w_start, w_end)
    if not m.any():
        return None
    return float(np.nanmean(y[m]))


def reduce_zoi(bundle, dims: DomainDims, zoi: Sequence[float], grid_step: float,
               field_vars: Sequence[str], window: Tuple[float, float] = (0.3, 1.0),
               evf_threshold: float = 0.5, evf_var: str = "EVF",
               force_channel: str = "RF1_RP",
               instance: Optional[str] = None) -> ZoiReduction:
    """Reduce one bundle to a windowed, EVF-masked ZOI mean per quantity.

    Field vars other than `evf_var` are masked by EVF >= evf_threshold (material
    only). `evf_var` itself is meaningful everywhere in the ZOI and is reduced
    unmasked. The force is the windowed mean of `force_channel`.
    """
    inst = instance or eulerian_instance(bundle)
    if not inst:
        raise RuntimeError(
            "No Eulerian instance (EVF carrier) found: cannot sample the ZOI.")
    points = roi_grid(zoi, grid_step)
    tmask = window_mask(bundle.times, *window)

    evf = np.asarray(nearest_samples(bundle, evf_var, inst, points), dtype=float)
    matmask = evf >= evf_threshold

    means: Dict[str, np.ndarray] = {}
    valid: Dict[str, np.ndarray] = {}
    for var in field_vars:
        if var == evf_var:
            continue
        arr = np.asarray(nearest_samples(bundle, var, inst, points), dtype=float)
        means[var], valid[var] = _masked_time_mean(arr, tmask, matmask)
    # EVF: compare the fill fraction everywhere in the ZOI (unmasked).
    means[evf_var], valid[evf_var] = _time_mean(evf, tmask)

    return ZoiReduction(
        dims=dims, means=means, valid=valid,
        force_mean=_history_window_mean(bundle, force_channel, *window),
        n_window_frames=int(tmask.sum()))


def _rel_change(a: np.ndarray, b: np.ndarray, joint: np.ndarray) -> float:
    """||a-b|| / ||b|| over the jointly-valid points, or NaN if undefined."""
    if not joint.any():
        return float("nan")
    ref = float(np.linalg.norm(b[joint]))
    if ref <= 0.0:
        return float("nan")
    return float(np.linalg.norm(a[joint] - b[joint]) / ref)


def zoi_discrepancy(a: ZoiReduction, b: ZoiReduction,
                    quantities: Sequence[str] = DEFAULT_QUANTITIES
                    ) -> Dict[str, float]:
    """Relative change of every quantity between two reductions (a vs reference b).

    For a field var: relative norm over points valid in BOTH reductions.
    For 'force': relative change of the windowed mean force.
    NaN propagates to "not converged" (conservative).
    """
    out: Dict[str, float] = {}
    for q in quantities:
        if q == "force":
            fa, fb = a.force_mean, b.force_mean
            if fa is None or fb is None or fb == 0.0:
                out["force"] = float("nan")
            else:
                out["force"] = abs(fa - fb) / abs(fb)
            continue
        if q not in a.means or q not in b.means:
            out[q] = float("nan")
            continue
        joint = a.valid.get(q) & b.valid.get(q)
        out[q] = _rel_change(a.means[q], b.means[q], joint)
    return out


# ---------------------------------------------------------------------------
# Convergence driver (Option B)
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceIteration:
    dims: DomainDims
    # per grown dimension -> per quantity -> relative ZOI change
    changes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    settled: Dict[str, bool] = field(default_factory=dict)   # dim -> below tol
    blocked: Dict[str, bool] = field(default_factory=dict)   # dim -> ceiling-blocked
    grown: str = ""                                          # dim grown next
    n_runs: int = 0


@dataclass
class ConvergenceResult:
    dims: DomainDims
    converged: bool
    iterations: List[ConvergenceIteration] = field(default_factory=list)
    n_runs: int = 0
    stopped_by: str = ""    # "converged"|"diagonal"|"max_iter"|"cancelled"|"zoi_outside"


def _binding_change(changes: Dict[str, float],
                    tolerances: Dict[str, float]) -> float:
    """Worst normalised change (change/tol) across thresholded quantities.

    >1 means at least one quantity still moves more than its tolerance. NaN
    (undefined comparison) is treated as +inf so it blocks convergence.
    """
    worst = 0.0
    for q, tol in tolerances.items():
        v = changes.get(q, float("nan"))
        if not math.isfinite(v) or tol <= 0:
            return float("inf")
        worst = max(worst, v / tol)
    return worst


def run_domain_convergence(
        run_bundle: Callable, cfg, zoi: Sequence[float],
        initial_dims: DomainDims, grid_step: float, elem_size: float,
        tolerances: Optional[Dict[str, float]] = None,
        field_vars: Sequence[str] = ("EVF", "TEMP", "V1", "V2"),
        window: Tuple[float, float] = (0.3, 1.0),
        evf_threshold: float = 0.5,
        grow_elems: int = 4, margin_elems: int = 1,
        max_iterations: int = 8,
        force_channel: str = "RF1_RP",
        should_cancel: Optional[Callable] = None,
        progress_cb: Optional[Callable] = None) -> ConvergenceResult:
    """Grow the domain OUTWARD until pushing each boundary no longer moves the ZOI.

    At each iteration, the current dims are the base; every growable dimension is
    grown by `grow_elems` elements and its ZOI reduction compared to the base.
    A dimension whose worst normalised change is below 1 (i.e. every quantity
    within tolerance) is 'settled'. The single most-influential unsettled
    dimension is grown, and the march repeats. The study converges to the
    SMALLEST domain where all growable dimensions are settled.

    Constraints:
      * the ZOI must stay inside the domain with `margin_elems` on every side;
        a starting domain that fails this returns stopped_by="zoi_outside";
      * `grow_elems` >= 1 (sub-element growth leaves the mesh unchanged, so it
        would produce zero variation);
      * growth stops at the reverberation ceiling diagonal_limit(elem_size).

    tolerances default to 2 % per quantity. Cost: up to 1 + len(growable) runs
    per iteration.
    """
    if grow_elems < 1:
        raise ValueError("grow_elems must be >= 1 (sub-element growth = no mesh change)")
    if tolerances is None:
        tolerances = {q: 0.02 for q in DEFAULT_QUANTITIES}
    margin = margin_elems * elem_size

    dims = initial_dims
    result = ConvergenceResult(dims=dims, converged=False)

    if not zoi_inside(dims, zoi, margin):
        result.stopped_by = "zoi_outside"
        return result
    if diagonal(dims) > diagonal_limit(elem_size):
        result.stopped_by = "diagonal"
        return result

    def _sample(d: DomainDims) -> ZoiReduction:
        cfg.euler_geometry.h_wp = d.h_wp
        cfg.euler_geometry.h_void = d.h_void
        cfg.euler_geometry.l_wp = d.l_wp
        cfg.euler_geometry.l_void = d.l_void
        bundle = run_bundle(cfg)
        if bundle is None:
            raise RuntimeError("run_bundle returned None for dims=%r" % (d,))
        return reduce_zoi(bundle, d, zoi, grid_step, field_vars,
                          window=window, evf_threshold=evf_threshold,
                          force_channel=force_channel)

    ceiling = diagonal_limit(elem_size)
    for _ in range(max_iterations):
        if should_cancel is not None and should_cancel():
            result.stopped_by = "cancelled"
            break

        base = _sample(dims)
        it = ConvergenceIteration(dims=dims, n_runs=1)

        worst_ratio, worst_dim = -1.0, ""
        for name in GROWABLE_DIMENSIONS:
            grown = _with(dims, name, getattr(dims, name) + grow_elems * elem_size)
            if diagonal(grown) > ceiling:
                # Cannot grow this direction without breaching the reverberation
                # ceiling. This is NOT independence: mark it blocked and NOT
                # settled, so a ceiling reached before convergence is reported
                # honestly rather than mistaken for a converged domain.
                it.blocked[name] = True
                it.settled[name] = False
                continue
            if should_cancel is not None and should_cancel():
                result.stopped_by = "cancelled"
                break
            sample = _sample(grown)
            it.n_runs += 1
            changes = zoi_discrepancy(sample, base, tuple(tolerances.keys()))
            it.changes[name] = changes
            ratio = _binding_change(changes, tolerances)
            it.settled[name] = ratio < 1.0
            if not it.settled[name] and (not math.isfinite(ratio)
                                         or ratio > worst_ratio):
                # +inf (undefined comparison) sorts above any finite ratio
                worst_ratio = float("inf") if not math.isfinite(ratio) else ratio
                worst_dim = name

        result.iterations.append(it)
        result.n_runs += it.n_runs
        if result.stopped_by == "cancelled":
            break

        all_settled = all(it.settled.get(n, False) for n in GROWABLE_DIMENSIONS)
        if progress_cb:
            progress_cb({"phase": "domain_convergence", "dims": _as_dict(dims),
                         "changes": it.changes, "settled": dict(it.settled),
                         "blocked": dict(it.blocked), "all_settled": all_settled})
        if all_settled:
            result.converged = True
            result.dims = dims
            result.stopped_by = "converged"
            break
        if not worst_dim:
            # No growable, unsettled dimension remains: every dimension that is
            # still above tolerance is blocked by the ceiling. We hit the
            # reverberation limit before independence -> report it.
            result.stopped_by = "diagonal"
            result.dims = dims
            break

        grown = _with(dims, worst_dim,
                      getattr(dims, worst_dim) + grow_elems * elem_size)
        it.grown = worst_dim
        if diagonal(grown) > ceiling:
            result.stopped_by = "diagonal"
            result.dims = dims
            break
        dims = grown
        result.dims = dims
    else:
        result.stopped_by = result.stopped_by or "max_iter"

    return result
