# -*- coding: utf-8 -*-
"""Eulerian-domain sizing by a sequential independence study (paper §4.1-4.4).

Purpose
-------
Find, for each Eulerian-domain dimension, the smallest value beyond which the
response measured in the fixed measurement zone (ZOI) no longer depends on the
position of that boundary. The element size and the mass-scaling factor are
held fixed (both are inputs of this study).

Algorithm (one dimension at a time, in `order`)
-----------------------------------------------
For a dimension d, the other three are held at their current value. The
dimension is grown by a CONSTANT step of `step_elems` elements::

    p_0, p_1 = p_0 + D, ..., p_n = p_0 + n*D      (D = step_elems * h)

and each new run S_n is compared with the previous one S_(n-1). The
comparison j yields, per quantity q, the error E_q,j = E_q(S_j, S_(j-1)).

Residual-influence bound ("tail bound", decision D9-b)
-------------------------------------------------------
The successive criterion E_q,j < eps_q only bounds the effect of the LAST
increment, not the residual influence of the boundary. For the mean absolute
difference of Eq. (5) the triangle inequality gives, for the candidate p_j
and n >= j+1 comparisons available::

    |S(p_j) - S(inf)| <= sum_{i=j+1..n} E_i + sum_{i>n} E_i

If the increments decay geometrically with ratio rho < 1, the unobserved part
is bounded by E_n * rho / (1 - rho), hence the bound used here::

    R_q(p_j) = sum_{i=j+1..n} E_q,i + E_q,n * rho_q / (1 - rho_q)

For j = n-1 this reduces to E_q,n / (1 - rho_q).

The geometric hypothesis is CHECKED per quantity on the last `m_ratios`
ratios rho_i = E_(i+1) / E_i (decision D10): they must all lie in [0, 1) and
be non-increasing (the decay does not slow down). rho_q is then taken as the
LARGEST of them, which is conservative as long as the decay does not slow
down. A ratio whose denominator is zero is accepted only if its numerator is
zero too (both increments vanished: zero residual).

GLOBAL fallback (decision of 2026-10-02): if the hypothesis is rejected for
ANY quantity, every quantity falls back to the successive criterion
E_max < 1 held `n_hold` consecutive times. No decision is taken before
`m_ratios + 1` comparisons exist (the hypothesis cannot be tested earlier).

Admissibility, retained value, statuses
---------------------------------------
* tail bound: the retained value is the SMALLEST p_j whose bound satisfies
  max_q R_q(p_j)/eps_q < 1 and whose run passed every safeguard
  (status "tail_bound");
* fallback: success of comparison n means E_max(n) < 1 AND both runs passed
  every safeguard; once `n_hold` consecutive successes are reached, the
  retained value is the first member of the first pair of that run of
  successes, p_(n - n_hold) (status "successive");
* neither before `n_max` comparisons, or a cap reached: the LARGEST tested
  value is retained (decision D5-a) with status "not_converged" or "cap".

Metrics (paper Eq. 5 and Eq. 7)
-------------------------------
* fields Vx (V1), Vy (V2), T (TEMP), EVF: mean of |q_a - q_b| over the ZOI
  grid points and the frames of the window T. Vx, Vy and T are masked by
  EVF >= `evf_threshold` (decision D1-b): a (point, frame) sample enters the
  mean only where BOTH runs are material. EVF itself is not masked;
* forces Fc (RF1_RP) and Ff (RF2_RP): mean of |F_a - F_b| over the history
  samples of T, divided by the element size h (the model is one element
  thick, so F/h is a force per unit width, N/mm);
* the two runs must share the same sample times inside T; otherwise the
  comparison is refused (AlignmentError) instead of silently truncated;
* E_max = max_q E_q/eps_q with q_crit its argmax (Eq. 13, 15); a NaN error
  counts as +inf (not admissible).

Safeguards
----------
Safeguards are evaluated by an injected `guard_fn(bundle) -> {name: (value,
ok)}` (energy ratios etc., lot L2). A run that produced no bundle is recorded
as a failed "job" safeguard; the study continues. The domain-diagonal
ceiling is NOT a safeguard any more: diag/h > `diagonal_coeff` only raises a
warning flag on the run (decision of 2026-10-01).

Frame convention (fact, abaqus_scripts/cel_model.py:291 and :397)
------------------------------------------------------------------
The Eulerian domain is the rectangle (-l_wp, -h_wp) -> (l_void, h_void),
translated in the assembly by euler_position (x0, y0). The ZOI is given in
the assembly frame, hence the domain bounds below include that offset.

Pure host-side module (CPython 3.x): `run_bundle(cfg)` is injected and must
return a ResultsBundle-like object or None.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gui.core.domain_sizing import DomainDims, diagonal
from gui.sensitivity.mesh_opt import roi_grid, nearest_samples
from gui.sensitivity.runner_core import eulerian_instance
from gui.sensitivity.domain_convergence import window_mask

# Paper quantity label -> bundle element-field name.
FIELD_QUANTITIES: Dict[str, str] = {"Vx": "V1", "Vy": "V2", "T": "TEMP",
                                    "EVF": "EVF"}
# Paper quantity label -> tool reference-point reaction-force history channel
# (convention Fc = RF1, Ff = RF2).
FORCE_QUANTITIES: Dict[str, str] = {"Fc": "RF1_RP", "Ff": "RF2_RP"}
# Quantities masked by the material indicator (EVF itself is not).
MASKED_QUANTITIES = ("Vx", "Vy", "T")
ALL_QUANTITIES = tuple(FIELD_QUANTITIES) + tuple(FORCE_QUANTITIES)

# Default growth order (paper Table 4): upstream length first, then depth,
# then the chip-side void extents.
DEFAULT_ORDER = ("l_wp", "h_wp", "h_void", "l_void")
DIMENSIONS = ("h_wp", "h_void", "l_wp", "l_void")

# Diagonal-over-element ratio above which a WARNING is raised. Value taken
# from gui.core.domain_sizing._DIAGONAL_OVER_ELEM_MAX (90.6); its derivation
# is not documented in the repository, it is kept as an indicator only.
DEFAULT_DIAGONAL_COEFF = 90.6

# Relative tolerance used to decide that two sample-time vectors coincide.
_TIME_RTOL = 1e-9
# Floating-point slack in the "ratios are non-increasing" test (round-off only).
_RATIO_RTOL = 1e-9


class AlignmentError(ValueError):
    """Two runs do not share the same sample times inside the window."""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def domain_bounds(dims: DomainDims, offset: Tuple[float, float] = (0.0, 0.0)
                  ) -> Tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of the Eulerian domain in the assembly frame.

    The part is the rectangle (-l_wp, -h_wp) -> (l_void, h_void)
    (cel_model.py:291), translated by euler_position (cel_model.py:397)."""
    x0, y0 = offset
    return (x0 - dims.l_wp, x0 + dims.l_void, y0 - dims.h_wp, y0 + dims.h_void)


def zoi_inside(dims: DomainDims, zoi: Sequence[float], margin: float = 0.0,
               offset: Tuple[float, float] = (0.0, 0.0)) -> bool:
    """True when the ZOI lies inside the domain with `margin` on every side."""
    zx0, zx1, zy0, zy1 = zoi
    dx0, dx1, dy0, dy1 = domain_bounds(dims, offset)
    tol = 1e-12
    return (dx0 <= zx0 - margin + tol and dx1 >= zx1 + margin - tol and
            dy0 <= zy0 - margin + tol and dy1 >= zy1 + margin - tol)


def zoi_within_crop(zoi: Sequence[float], crop: dict) -> bool:
    """True when the ZOI lies inside the extraction crop {xmin, xmax, ...}."""
    tol = 1e-12
    return (crop["xmin"] <= zoi[0] + tol and crop["xmax"] >= zoi[1] - tol and
            crop["ymin"] <= zoi[2] + tol and crop["ymax"] >= zoi[3] - tol)


def _snap_up(value: float, elem: float) -> float:
    """Smallest whole number of elements (>= 1) covering `value`."""
    n = max(1, int(math.ceil(max(0.0, float(value)) / elem - 1e-9)))
    return n * elem


def initial_dims_from_zoi(zoi: Sequence[float], elem_size: float,
                          margin_elems: int = 0,
                          offset: Tuple[float, float] = (0.0, 0.0)
                          ) -> DomainDims:
    """Initial domain = ZOI + margin, snapped up to whole elements (D2-a).

    In the domain's own frame (assembly coordinates minus `offset`):
    l_wp = -xmin, l_void = xmax, h_wp = -ymin, h_void = ymax, each increased
    by `margin_elems` elements and rounded up to a whole number of elements.
    A side of the ZOI lying on the other side of the origin contributes 0
    (then the margin, then at least one element)."""
    if elem_size <= 0:
        raise ValueError("elem_size must be > 0")
    x0, y0 = offset
    zx0, zx1, zy0, zy1 = (zoi[0] - x0, zoi[1] - x0, zoi[2] - y0, zoi[3] - y0)
    m = max(0, int(margin_elems)) * elem_size
    return DomainDims(h_wp=_snap_up(-zy0 + m, elem_size),
                      h_void=_snap_up(zy1 + m, elem_size),
                      l_wp=_snap_up(-zx0 + m, elem_size),
                      l_void=_snap_up(zx1 + m, elem_size))


def _with(dims: DomainDims, name: str, value: float) -> DomainDims:
    d = {n: float(getattr(dims, n)) for n in DIMENSIONS}
    d[name] = float(value)
    return DomainDims(**d)


def _dims_key(dims: DomainDims, elem: float) -> Tuple[int, int, int, int]:
    """Cache key: each dimension as a whole number of elements."""
    return tuple(int(round(getattr(dims, n) / elem)) for n in DIMENSIONS)


def _dims_dict(dims: DomainDims) -> Dict[str, float]:
    return {n: float(getattr(dims, n)) for n in DIMENSIONS}


# ---------------------------------------------------------------------------
# ZOI sample of one run
# ---------------------------------------------------------------------------
@dataclass
class ZoiSample:
    """Windowed ZOI data of one run, ready for Eq. (5) and Eq. (7).

    fields[q] : (Nt_w, Np) values on the fixed ZOI grid for the frames of the
        window; masked samples (non-material) are NaN.
    field_times : (Nt_w,) frame times of the window.
    forces[q] : (Nh_w,) force per unit width F/h for the history samples of
        the window, or absent when the channel is missing.
    force_times : (Nh_w,) history times of the window.
    """
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    field_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    forces: Dict[str, np.ndarray] = field(default_factory=dict)
    force_times: np.ndarray = field(default_factory=lambda: np.zeros(0))


def sample_zoi(bundle, zoi: Sequence[float], grid_step: float,
               elem_size: float, window: Tuple[float, float] = (0.3, 1.0),
               evf_threshold: float = 0.5,
               quantities: Sequence[str] = ALL_QUANTITIES,
               instance: Optional[str] = None) -> ZoiSample:
    """Reduce a results bundle to its windowed ZOI sample.

    The ZOI is sampled on a fixed regular grid of spacing `grid_step` by
    nearest element centroid. The mesh is fixed in this study (same element
    size, whole-element domain growth), so the same physical cells are read
    in every run. Raises RuntimeError when no Eulerian instance exists or the
    window contains no frame."""
    inst = instance or eulerian_instance(bundle)
    if not inst:
        raise RuntimeError("No Eulerian instance (EVF carrier) in the bundle.")
    # The extraction keeps only the elements inside the output ROI when that
    # filter is active (abaqus_scripts/cel_results.py:25-54). A ZOI point
    # outside it would silently take the value of the nearest EDGE element of
    # the crop: refuse instead.
    crop = getattr(bundle, "roi", None)
    if isinstance(crop, dict) and not zoi_within_crop(zoi, crop):
        raise RuntimeError(
            "The ZOI x[%g,%g] y[%g,%g] is not contained in the extracted ROI "
            "x[%g,%g] y[%g,%g]: widen the ROI (Geometry tab) or shrink the "
            "ZOI." % (tuple(zoi) + (crop["xmin"], crop["xmax"], crop["ymin"],
                                    crop["ymax"])))
    points = roi_grid(zoi, grid_step)
    times = np.asarray(bundle.times, dtype=float)
    tmask = window_mask(times, *window)
    if not tmask.any():
        raise RuntimeError("The time window %r contains no field frame."
                           % (tuple(window),))
    out = ZoiSample(field_times=times[tmask])

    want_fields = [q for q in quantities if q in FIELD_QUANTITIES]
    evf = None
    if want_fields:
        evf = np.asarray(nearest_samples(bundle, FIELD_QUANTITIES["EVF"],
                                         inst, points), dtype=float)[tmask]
    material = None if evf is None else (evf >= float(evf_threshold))
    for q in want_fields:
        if q == "EVF":
            out.fields[q] = evf
            continue
        arr = np.asarray(nearest_samples(bundle, FIELD_QUANTITIES[q], inst,
                                         points), dtype=float)[tmask]
        if q in MASKED_QUANTITIES:
            arr = np.where(material, arr, np.nan)
        out.fields[q] = arr

    want_forces = [q for q in quantities if q in FORCE_QUANTITIES]
    if want_forces:
        try:
            ht = np.asarray(bundle.history_time, dtype=float)
        except Exception:
            ht = np.zeros(0)
        hmask = window_mask(ht, *window) if ht.size else np.zeros(0, bool)
        out.force_times = ht[hmask] if ht.size else ht
        for q in want_forces:
            try:
                f = np.asarray(bundle.history(FORCE_QUANTITIES[q]), dtype=float)
            except Exception:
                continue                       # absent channel: no force entry
            if f.size != ht.size or not hmask.any():
                continue
            out.forces[q] = f[hmask] / float(elem_size)
    return out


# ---------------------------------------------------------------------------
# Eq. (5) / Eq. (7): mean absolute differences
# ---------------------------------------------------------------------------
def _check_times(ta: np.ndarray, tb: np.ndarray, what: str) -> None:
    if ta.shape != tb.shape or not np.allclose(
            ta, tb, rtol=_TIME_RTOL, atol=_TIME_RTOL * max(1.0, float(
                np.max(np.abs(tb))) if tb.size else 1.0)):
        raise AlignmentError(
            "%s sample times differ between the two runs (%d vs %d samples in "
            "the window): the comparison is refused." % (what, ta.size, tb.size))


def mean_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of |a - b| over the samples finite in BOTH arrays (NaN if none)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise AlignmentError("shape mismatch %r vs %r" % (a.shape, b.shape))
    d = a - b
    d = d[np.isfinite(d)]
    return float(np.mean(np.abs(d))) if d.size else float("nan")


def errors_between(sa: ZoiSample, sb: ZoiSample,
                   quantities: Sequence[str] = ALL_QUANTITIES
                   ) -> Dict[str, float]:
    """E_q(S_a, S_b) for every quantity (Eq. 5 for fields, Eq. 7 for forces).

    Raises AlignmentError when the window sample times differ. A quantity
    missing from either sample gives NaN (counted as not admissible)."""
    out: Dict[str, float] = {}
    if any(q in FIELD_QUANTITIES for q in quantities):
        _check_times(sa.field_times, sb.field_times, "Field-frame")
    if any(q in FORCE_QUANTITIES and q in sa.forces and q in sb.forces
           for q in quantities):
        _check_times(sa.force_times, sb.force_times, "History")
    for q in quantities:
        if q in FIELD_QUANTITIES:
            if q in sa.fields and q in sb.fields:
                out[q] = mean_abs_difference(sa.fields[q], sb.fields[q])
            else:
                out[q] = float("nan")
        else:
            if q in sa.forces and q in sb.forces:
                out[q] = mean_abs_difference(sa.forces[q], sb.forces[q])
            else:
                out[q] = float("nan")
    return out


def e_max(errors: Dict[str, float], thresholds: Dict[str, float]
          ) -> Tuple[float, Optional[str]]:
    """(E_max, q_crit) = max_q E_q/eps_q and its argmax (Eq. 13, 15).

    A NaN or missing error counts as +inf. Returns (nan, None) when no
    threshold is given."""
    worst, crit = float("nan"), None
    for q, eps in thresholds.items():
        if eps is None or eps <= 0:
            continue
        v = errors.get(q, float("nan"))
        r = v / eps if math.isfinite(v) else float("inf")
        if crit is None or r > worst:
            worst, crit = r, q
    return worst, crit


# ---------------------------------------------------------------------------
# Tail bound with geometric-decay check
# ---------------------------------------------------------------------------
@dataclass
class DecayCheck:
    """Outcome of the geometric-decay test for one quantity."""
    accepted: bool
    rho: float = float("nan")              # retained ratio (max of window)
    ratios: List[float] = field(default_factory=list)
    reason: str = ""


def check_geometric_decay(errors: Sequence[float], m_ratios: int) -> DecayCheck:
    """Test the geometric-decay hypothesis on the last `m_ratios` ratios.

    errors : E_1 ... E_n of one quantity (successive comparisons, constant
        step). Needs n >= m_ratios + 1.
    Accepted iff every ratio rho_i = E_(i+1)/E_i of the window lies in [0, 1)
    and the ratios are non-increasing. 0/0 counts as 0 (vanished increments);
    x/0 with x > 0 rejects. rho = the largest ratio of the window."""
    if m_ratios < 1:
        raise ValueError("m_ratios must be >= 1")
    e = [float(v) for v in errors]
    if len(e) < m_ratios + 1:
        return DecayCheck(False, reason="too_few_comparisons")
    tail = e[-(m_ratios + 1):]
    if not all(math.isfinite(v) and v >= 0.0 for v in tail):
        return DecayCheck(False, reason="undefined_error")
    ratios: List[float] = []
    for a, b in zip(tail[:-1], tail[1:]):
        if a == 0.0:
            if b == 0.0:
                ratios.append(0.0)
                continue
            return DecayCheck(False, ratios=ratios, reason="growth_from_zero")
        ratios.append(b / a)
    if any(not (0.0 <= r < 1.0) for r in ratios):
        return DecayCheck(False, ratios=ratios, reason="ratio_not_below_1")
    # Non-increasing up to floating-point rounding only (_RATIO_RTOL): this is
    # NOT a dispersion tolerance (decision D10 rejected one); it only keeps
    # mathematically equal ratios from failing on round-off.
    if any(r2 > r1 * (1.0 + _RATIO_RTOL) + _RATIO_RTOL
           for r1, r2 in zip(ratios[:-1], ratios[1:])):
        return DecayCheck(False, ratios=ratios, reason="decay_slowing")
    return DecayCheck(True, rho=max(ratios), ratios=ratios)


def tail_bound(errors: Sequence[float], j: int, rho: float) -> float:
    """R(p_j) = sum_{i=j+1..n} E_i + E_n * rho / (1 - rho).

    `errors` holds E_1..E_n (index i-1 for E_i); p_j is the domain value
    before comparison j+1, j in [0, n-1]."""
    n = len(errors)
    if not (0 <= j <= n - 1):
        raise ValueError("candidate index out of range")
    if not (0.0 <= rho < 1.0):
        return float("inf")
    observed = float(sum(errors[j:n]))           # E_(j+1) .. E_n
    return observed + float(errors[-1]) * rho / (1.0 - rho)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    """One Abaqus run of the study."""
    index: int                              # order of execution (0-based)
    dims: Dict[str, float]
    job_ok: bool
    guards: Dict[str, Tuple[Optional[float], bool]] = field(default_factory=dict)
    diagonal_ratio: float = float("nan")    # diag / h
    diagonal_warning: bool = False
    error: str = ""                         # why the run is unusable, if so
    host_wall_s: float = float("nan")       # launch -> bundle, measured here
    cost: Optional[object] = None           # CostRecord from cost_fn, if any

    @property
    def guards_ok(self) -> bool:
        return self.job_ok and all(ok for (_v, ok) in self.guards.values())


@dataclass
class Comparison:
    """Comparison j of a dimension: S(p_j) against S(p_(j-1))."""
    dimension: str
    j: int
    value_from: float
    value_to: float
    errors: Dict[str, float]
    e_max: float
    q_crit: Optional[str]
    guards_ok: bool                          # both runs passed the safeguards
    success: bool                            # E_max < 1 and guards_ok
    mode: str = ""                           # "tail_bound"|"successive"|"pending"
    decay: Dict[str, DecayCheck] = field(default_factory=dict)
    bound: Dict[str, float] = field(default_factory=dict)   # R_q at decision


@dataclass
class DimensionResult:
    name: str
    initial: float
    retained: float
    status: str          # tail_bound | successive | not_converged | cap | cancelled
    values: List[float] = field(default_factory=list)
    comparisons: List[Comparison] = field(default_factory=list)
    retained_index: int = -1
    q_crit: Optional[str] = None
    criterion: float = float("nan")          # max_q R_q/eps_q or E_max retained


@dataclass
class StudyResult:
    initial: DomainDims
    final: DomainDims
    per_dimension: Dict[str, DimensionResult] = field(default_factory=dict)
    runs: List[RunRecord] = field(default_factory=list)
    status: str = ""     # converged | partial | zoi_outside | cancelled
    warnings: List[str] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.runs)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_domain_independence(
        run_bundle: Callable, base_cfg, zoi: Sequence[float],
        initial_dims: DomainDims, grid_step: float, elem_size: float,
        thresholds: Dict[str, float],
        window: Tuple[float, float] = (0.3, 1.0),
        evf_threshold: float = 0.5,
        step_elems: int = 10, n_max: int = 8, n_hold: int = 1,
        m_ratios: int = 2,
        caps: Optional[Dict[str, float]] = None,
        order: Sequence[str] = DEFAULT_ORDER,
        margin_elems: int = 0,
        offset: Tuple[float, float] = (0.0, 0.0),
        diagonal_coeff: float = DEFAULT_DIAGONAL_COEFF,
        guard_fn: Optional[Callable] = None,
        cost_fn: Optional[Callable] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        progress_cb: Optional[Callable[[dict], None]] = None) -> StudyResult:
    """Size the four Eulerian-domain dimensions one after the other.

    run_bundle(cfg) -> ResultsBundle | None. The config passed is a deep copy
        of `base_cfg` with euler_geometry set to the candidate dimensions;
        `base_cfg` itself is never modified.
    thresholds : absolute tolerances eps_q per quantity label (Vx, Vy mm/s;
        T K or degC; EVF -; Fc, Ff N/mm). Only thresholded quantities enter
        E_max and the bounds.
    step_elems : constant growth step D, in elements (default 10).
    n_max : maximum number of comparisons per dimension (default 8); must be
        >= m_ratios + 1 so that the decay hypothesis can be tested.
    n_hold : consecutive successes required in the fallback (default 1).
    m_ratios : ratios used to test the geometric decay (default 2).
    caps : optional maximum value per dimension (mm).
    guard_fn(bundle) -> {name: (value, ok)} : run safeguards (optional).
    cost_fn(bundle_or_None, dims, host_wall_s) -> CostRecord : run cost
        (optional; see gui.sensitivity.run_record.cost_record). The host wall
        time of each run is measured here around run_bundle.

    Every run is cached by its dimensions (whole elements): a domain already
    simulated is never run again."""
    if step_elems < 1:
        raise ValueError("step_elems must be >= 1")
    if m_ratios < 1:
        raise ValueError("m_ratios must be >= 1")
    if n_max < m_ratios + 1:
        raise ValueError("n_max must be >= m_ratios + 1 (the geometric-decay "
                         "hypothesis needs m_ratios + 1 comparisons)")
    if n_hold < 1:
        raise ValueError("n_hold must be >= 1")
    if elem_size <= 0 or grid_step <= 0:
        raise ValueError("elem_size and grid_step must be > 0")
    unknown = set(order) - set(DIMENSIONS)
    if unknown:
        raise ValueError("unknown dimension(s) in order: %s" % sorted(unknown))
    thr = {q: float(v) for q, v in thresholds.items()
           if v is not None and float(v) > 0 and q in ALL_QUANTITIES}
    if not thr:
        raise ValueError("at least one positive threshold is required")
    quantities = tuple(thr.keys())
    caps = dict(caps or {})
    elem = float(elem_size)
    delta = int(step_elems) * elem

    def cancelled() -> bool:
        return bool(should_cancel is not None and should_cancel())

    def emit(ev: dict) -> None:
        if progress_cb is not None:
            progress_cb(ev)

    result = StudyResult(initial=initial_dims, final=initial_dims)
    if not zoi_inside(initial_dims, zoi, margin_elems * elem, offset):
        result.status = "zoi_outside"
        return result

    cache: Dict[Tuple[int, int, int, int], Tuple[RunRecord,
                                                 Optional[ZoiSample]]] = {}

    def simulate(dims: DomainDims) -> Tuple[RunRecord, Optional[ZoiSample]]:
        key = _dims_key(dims, elem)
        if key in cache:
            return cache[key]
        cfg = copy.deepcopy(base_cfg)
        g = cfg.euler_geometry
        g.h_wp, g.h_void = float(dims.h_wp), float(dims.h_void)
        g.l_wp, g.l_void = float(dims.l_wp), float(dims.l_void)
        ratio = diagonal(dims) / elem
        rec = RunRecord(index=len(result.runs), dims=_dims_dict(dims),
                        job_ok=False, diagonal_ratio=ratio,
                        diagonal_warning=bool(ratio > diagonal_coeff))
        if rec.diagonal_warning:
            msg = ("diagonal/h = %.1f > %.1f for %r (warning only, the study "
                   "continues)" % (ratio, diagonal_coeff, rec.dims))
            result.warnings.append(msg)
            emit({"phase": "warning", "message": msg})
        sample = None
        t0 = time.perf_counter()
        try:
            bundle = run_bundle(cfg)
        except Exception as exc:              # a failed launch is a failed job
            bundle = None
            rec.error = "%s: %s" % (type(exc).__name__, exc)
        rec.host_wall_s = time.perf_counter() - t0
        if cost_fn is not None:
            try:
                rec.cost = cost_fn(bundle, dims, rec.host_wall_s)
            except Exception as exc:          # cost is informative only
                result.warnings.append("cost of run %d: %s: %s"
                                       % (rec.index, type(exc).__name__, exc))
        if bundle is None:
            rec.error = rec.error or "no results bundle"
        else:
            try:
                sample = sample_zoi(bundle, zoi, grid_step, elem, window,
                                    evf_threshold, quantities)
                rec.job_ok = True
            except Exception as exc:
                rec.error = "%s: %s" % (type(exc).__name__, exc)
            if guard_fn is not None and rec.job_ok:
                try:
                    rec.guards = dict(guard_fn(bundle) or {})
                except Exception as exc:
                    rec.guards = {"guard_eval": (None, False)}
                    rec.error = "safeguards: %s: %s" % (type(exc).__name__, exc)
        result.runs.append(rec)
        cache[key] = (rec, sample)
        emit({"phase": "run", "record": rec, "n_runs": len(result.runs)})
        return rec, sample

    dims = initial_dims
    all_converged = True
    for name in order:
        if cancelled():
            result.status = "cancelled"
            break
        p0 = float(getattr(dims, name))
        cap = caps.get(name)
        dres = DimensionResult(name=name, initial=p0, retained=p0,
                               status="not_converged", values=[p0])
        runs: List[Tuple[RunRecord, Optional[ZoiSample]]] = [simulate(dims)]
        series: Dict[str, List[float]] = {q: [] for q in quantities}
        consecutive = 0
        decided = False
        stop_status = "not_converged"

        for n in range(1, n_max + 1):
            if cancelled():
                stop_status = "cancelled"
                break
            nxt = dres.values[-1] + delta
            if cap is not None and nxt > cap + 1e-12:
                nxt = math.floor(cap / elem + 1e-9) * elem
            if nxt <= dres.values[-1] + 1e-12:
                stop_status = "cap"
                break
            cand = _with(dims, name, nxt)
            rec_b, s_b = simulate(cand)
            rec_a, s_a = runs[-1]
            runs.append((rec_b, s_b))
            dres.values.append(nxt)

            if s_a is None or s_b is None:
                errs = {q: float("nan") for q in quantities}
            else:
                try:
                    errs = errors_between(s_b, s_a, quantities)
                except AlignmentError as exc:
                    errs = {q: float("nan") for q in quantities}
                    result.warnings.append("%s comparison %d: %s"
                                           % (name, n, exc))
            for q in quantities:
                series[q].append(errs.get(q, float("nan")))
            em, qc = e_max(errs, thr)
            pair_ok = rec_a.guards_ok and rec_b.guards_ok
            success = bool(math.isfinite(em) and em < 1.0 and pair_ok)
            consecutive = consecutive + 1 if success else 0
            comp = Comparison(dimension=name, j=n, value_from=dres.values[-2],
                              value_to=nxt, errors=errs, e_max=em, q_crit=qc,
                              guards_ok=pair_ok, success=success,
                              mode="pending")
            dres.comparisons.append(comp)

            if n >= m_ratios + 1:
                checks = {q: check_geometric_decay(series[q], m_ratios)
                          for q in quantities}
                comp.decay = checks
                if all(c.accepted for c in checks.values()):
                    comp.mode = "tail_bound"
                    for jj in range(0, n):
                        bounds = {q: tail_bound(series[q], jj, checks[q].rho)
                                  for q in quantities}
                        crit, qcrit = e_max(bounds, thr)
                        if runs[jj][0].guards_ok and crit < 1.0:
                            comp.bound = bounds
                            dres.retained = dres.values[jj]
                            dres.retained_index = jj
                            dres.q_crit, dres.criterion = qcrit, crit
                            stop_status, decided = "tail_bound", True
                            break
                else:
                    comp.mode = "successive"     # GLOBAL fallback
                    if consecutive >= n_hold:
                        jj = n - n_hold
                        dres.retained = dres.values[jj]
                        dres.retained_index = jj
                        first = dres.comparisons[jj]   # comparison jj+1
                        dres.q_crit, dres.criterion = first.q_crit, first.e_max
                        stop_status, decided = "successive", True
            emit({"phase": "comparison", "comparison": comp,
                  "n_runs": len(result.runs)})
            if decided:
                break

        if not decided:
            # D5-a: no plateau (or cap / cancel): keep the largest tested value.
            dres.retained = dres.values[-1]
            dres.retained_index = len(dres.values) - 1
            all_converged = False
        dres.status = stop_status
        result.per_dimension[name] = dres
        dims = _with(dims, name, dres.retained)
        result.final = dims
        emit({"phase": "dimension", "result": dres, "dims": _dims_dict(dims),
              "n_runs": len(result.runs)})
        if stop_status == "cancelled":
            result.status = "cancelled"
            break

    if not result.status:
        result.status = "converged" if all_converged else "partial"
    emit({"phase": "done", "result": result, "n_runs": len(result.runs)})
    return result
