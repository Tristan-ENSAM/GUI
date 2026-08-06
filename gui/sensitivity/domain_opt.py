# -*- coding: utf-8 -*-
"""
Eulerian-domain size optimiser (Optimization tab core).

Finds the smallest (h_wp, h_void, l_wp, l_void) such that the ROI fields no
longer change when the corresponding boundary is moved outward — the Cauchy
self-convergence criterion from the user's formulation:

    E_q^{k,k+1} = sqrt( 1/(Nt*Np) * sum_i sum_j [ q^{k+1}(p_j,t_i)
                                                   - q^k(p_j,t_i) ]^2 )

for each quantity q in {Vx, Vy, T, PEEQ, EVF}, evaluated at the SAME ROI points
p_j and time increments t_i. Because the Eulerian mesh is structured at a fixed
element size and anchored on x=0 / y=0, the ROI element centroids coincide
across domain sizes, so q^k and q^{k+1} are sampled at identical points (no
interpolation). A dimension is converged when every E_q < eps_q; the global
optimum is reached when all dimensions hold simultaneously.

This module is PURE: it drives an abstract `sample_fn(dims) -> {q: ndarray}`
that returns the ROI samples (shape (Nt, Np)) for a given domain. Wiring that
callback to actual Abaqus runs + extraction is done by the tab (Lot O.3); here
`sample_fn` is injected, so the search logic is unit-testable without Abaqus.

Search per dimension (agreed): doubling (d, 2d, 4d, ...) until consecutive
sizes agree within eps -> that fixes d_large; then bisection in [d0, d_large]
for the smallest d* that still matches d_large within eps. O(log) runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from gui.core.domain_sizing import DomainDims


# The five ROI quantities of the convergence criterion.
DEFAULT_QUANTITIES = ("Vx", "Vy", "T", "PEEQ", "EVF")
# The four domain dimensions, in the default growth order (upstream first,
# then depth, then the chip-side void extents).
DEFAULT_ORDER = ("l_wp", "h_wp", "h_void", "l_void")


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def roi_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean ABSOLUTE difference between two ROI sample arrays of shape (Nt, Np),
    keeping the physical unit of the quantity (so thresholds are directly
    interpretable, e.g. mm/s, K, N). NaN-safe. If the two arrays have a
    different number of frames, they are aligned on the common leading frames
    (columns/points assumed identical — same anchored/fixed ROI grid)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    nt = min(a.shape[0], b.shape[0])
    npt = min(a.shape[1], b.shape[1])
    if nt == 0 or npt == 0:
        return float("nan")
    d = a[:nt, :npt] - b[:nt, :npt]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(np.mean(np.abs(d)))


# Backwards-compatible alias (the metric is a mean absolute difference, not an
# RMS; the old name is kept so existing imports keep working).
roi_rmse = roi_error


def errors_between(sa: Dict[str, np.ndarray], sb: Dict[str, np.ndarray],
                   quantities) -> Dict[str, float]:
    """Per-quantity E_q between two domains' ROI samples."""
    out = {}
    for q in quantities:
        if q in sa and q in sb:
            out[q] = roi_rmse(sa[q], sb[q])
        else:
            out[q] = float("nan")
    return out


def all_below(errors: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    """True iff every quantity present in `thresholds` has a finite error
    strictly below its threshold. A missing/NaN error is treated as NOT
    converged (conservative)."""
    for q, eps in thresholds.items():
        e = errors.get(q, float("nan"))
        if not math.isfinite(e) or e >= eps:
            return False
    return True


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class DimResult:
    name: str
    initial: float
    final: float
    d_large: float
    history: List[tuple] = field(default_factory=list)  # (value, errors_dict)


@dataclass
class OptimizeResult:
    dims: DomainDims
    per_dim: Dict[str, DimResult]
    n_runs: int
    passes: int
    converged: bool


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
def _snap(value: float, elem: float) -> float:
    n = max(1, int(round(value / elem)))
    return n * elem


class DomainOptimizer:
    """Coordinate-descent optimiser over the four Eulerian dimensions.

    Parameters
    ----------
    sample_fn : Callable[[DomainDims], dict]
        Runs (or looks up) a simulation for the given domain and returns the
        ROI samples {q: ndarray (Nt, Np)}. Must be deterministic for a given
        DomainDims (results are cached by element-snapped dimensions).
    thresholds : dict {q: eps_q}   (physical units, per quantity)
    elem_size : float              (the dimension step; all sizes are multiples)
    caps : dict {dim: max_value}   optional hard upper bounds per dimension
    order : iterable of dim names  growth order
    quantities : iterable of q     defaults to the five criterion quantities
    max_passes : int               coordinate-descent passes (coupling)
    max_doublings : int            safety cap on the doubling phase
    """

    def __init__(self, sample_fn: Callable[[DomainDims], Dict[str, np.ndarray]],
                 thresholds: Dict[str, float], elem_size: float,
                 caps: Optional[Dict[str, float]] = None,
                 order=DEFAULT_ORDER, quantities=DEFAULT_QUANTITIES,
                 max_passes: int = 2, max_doublings: int = 20,
                 bisect_resolution: float = 0.0, grow_factor: float = 2.0):
        if elem_size <= 0:
            raise ValueError("elem_size must be > 0")
        self.sample_fn = sample_fn
        self.thresholds = dict(thresholds)
        self.elem = float(elem_size)
        self.caps = dict(caps or {})
        self.order = tuple(order)
        self.quantities = tuple(quantities)
        self.max_passes = int(max_passes)
        self.max_doublings = int(max_doublings)
        # Bracketing growth factor (>1 grows the dimension; the factor encodes
        # the direction, so no separate direction knob is needed).
        self.grow_factor = float(grow_factor) if grow_factor > 1.0 else 2.0
        # Stop the per-dimension bisection once the bracket is narrower than this
        # (a resolution on the dimension, e.g. 0.01 mm). 0 -> down to 1 element.
        self.bisect_resolution = float(bisect_resolution)
        self._cache: Dict[tuple, Dict[str, np.ndarray]] = {}
        self._n_runs = 0

    # -- sampling with cache -------------------------------------------------
    def _key(self, dims: DomainDims) -> tuple:
        return tuple(int(round(getattr(dims, n) / self.elem))
                     for n in ("h_wp", "h_void", "l_wp", "l_void"))

    def _sample(self, dims: DomainDims) -> Dict[str, np.ndarray]:
        k = self._key(dims)
        if k not in self._cache:
            self._cache[k] = self.sample_fn(dims)
            self._n_runs += 1
        return self._cache[k]

    def _with(self, dims: DomainDims, name: str, value: float) -> DomainDims:
        d = DomainDims(dims.h_wp, dims.h_void, dims.l_wp, dims.l_void)
        setattr(d, name, value)
        return d

    def _errors(self, dims_a: DomainDims, dims_b: DomainDims) -> Dict[str, float]:
        return errors_between(self._sample(dims_a), self._sample(dims_b),
                              self.quantities)

    # -- one-dimension search ------------------------------------------------
    def _search_dimension(self, dims: DomainDims, name: str,
                          progress_cb=None) -> DimResult:
        d0 = _snap(getattr(dims, name), self.elem)
        cap = self.caps.get(name)
        history: List[tuple] = []

        # Doubling phase: grow until consecutive sizes agree within eps.
        cur = d0
        d_large = None
        for _ in range(self.max_doublings):
            nxt = _snap(self.grow_factor * cur, self.elem)
            if nxt <= cur:                      # numerical floor
                nxt = _snap(cur + self.elem, self.elem)
            if cap is not None and nxt >= cap:
                nxt = _snap(cap, self.elem)
            err = self._errors(self._with(dims, name, cur),
                               self._with(dims, name, nxt))
            history.append((nxt, err))
            if progress_cb is not None:
                progress_cb({"phase": "compare", "name": name, "value": nxt,
                             "errors": err, "n_runs": self._n_runs})
            if all_below(err, self.thresholds):
                d_large = nxt
                break
            cur = nxt
            if cap is not None and cur >= cap:
                d_large = cur                   # capped, accept as reference
                break
        if d_large is None:
            d_large = cur                       # never converged -> use the max

        # Bisection: smallest d* in [d0, d_large] with E(d*, d_large) < eps.
        ref = self._with(dims, name, d_large)
        lo = int(round(d0 / self.elem))
        hi = int(round(d_large / self.elem))
        # If even d0 already matches d_large, the answer is d0.
        if all_below(self._errors(self._with(dims, name, d0), ref),
                     self.thresholds):
            best = lo
        else:
            best = hi
            while lo < hi and (hi - lo) * self.elem > self.bisect_resolution:
                mid = (lo + hi) // 2
                dmid = mid * self.elem
                conv = all_below(
                    self._errors(self._with(dims, name, dmid), ref),
                    self.thresholds)
                history.append((dmid, {}))
                if conv:
                    best = mid
                    hi = mid
                else:
                    lo = mid + 1
        final = best * self.elem
        return DimResult(name=name, initial=d0, final=final,
                         d_large=d_large, history=history)

    # -- full optimisation ---------------------------------------------------
    def optimize(self, initial: DomainDims, progress_cb=None,
                 should_cancel=None) -> OptimizeResult:
        """Run the coordinate-descent optimisation.

        progress_cb : optional callable(event: dict) invoked after each
            dimension search, with keys {"phase", "pass", "name", "result"
            (DimResult), "dims" (DomainDims), "n_runs"}. Also called once at
            the end with phase "done".
        should_cancel : optional callable() -> bool; if it returns True the
            optimisation stops early and returns the best domain so far
            (converged=False)."""
        dims = DomainDims(_snap(initial.h_wp, self.elem),
                          _snap(initial.h_void, self.elem),
                          _snap(initial.l_wp, self.elem),
                          _snap(initial.l_void, self.elem))
        per_dim: Dict[str, DimResult] = {}
        passes = 0
        changed = True
        cancelled = False
        while changed and passes < self.max_passes:
            changed = False
            passes += 1
            for name in self.order:
                if should_cancel is not None and should_cancel():
                    cancelled = True
                    break
                res = self._search_dimension(dims, name, progress_cb=progress_cb)
                per_dim[name] = res
                if abs(res.final - getattr(dims, name)) > 0.5 * self.elem:
                    changed = True
                dims = self._with(dims, name, res.final)
                if progress_cb is not None:
                    progress_cb({"phase": "dimension", "pass": passes,
                                 "name": name, "result": res, "dims": dims,
                                 "n_runs": self._n_runs})
            if cancelled:
                break

        # Global convergence check: at the final domain, growing each boundary
        # by one more element must keep all errors below threshold.
        converged = not cancelled
        if not cancelled:
            for name in self.order:
                bigger = self._with(
                    dims, name,
                    _snap(getattr(dims, name) + self.elem, self.elem))
                if not all_below(self._errors(dims, bigger), self.thresholds):
                    converged = False
                    break

        result = OptimizeResult(dims=dims, per_dim=per_dim, n_runs=self._n_runs,
                                passes=passes, converged=converged)
        if progress_cb is not None:
            progress_cb({"phase": "done", "pass": passes, "name": None,
                         "result": result, "dims": dims,
                         "n_runs": self._n_runs})
        return result


# ---------------------------------------------------------------------------
# ROI extraction from a results bundle (bridges a bundle to sample_fn)
# ---------------------------------------------------------------------------
def element_centroids_xy(bundle, inst) -> np.ndarray:
    """(n_elements, 2) initial element centroids (x, y) of an instance."""
    c = np.asarray(bundle.element_centroids_init(inst), dtype=float)
    return c[:, :2]


def grid_keys(centroids_xy: np.ndarray, elem_size: float) -> np.ndarray:
    """Integer (kx, ky) grid index of each centroid.

    The Eulerian mesh is structured at a fixed element size with node lines on
    x=0 / y=0, so a cell centroid sits at (i+0.5)*elem. round(c/elem - 0.5)
    recovers the integer cell index i, which is IDENTICAL across domain sizes
    for the same physical cell — the basis for comparing q^k and q^{k+1} at the
    same ROI points."""
    c = np.asarray(centroids_xy, dtype=float)
    return np.round(c / float(elem_size) - 0.5).astype(int)


def roi_samples(bundle, var, inst, roi, elem_size, frames=None):
    """Sample an element field at the centroids inside the ROI box.

    roi = (xmin, xmax, ymin, ymax) in the domain frame.
    Returns (keys (N_p, 2) int, values (N_t, N_p) float), ordered canonically
    by (ky, kx) so that two domains with the same element size produce
    identically-ordered arrays. `frames` optionally selects a subset of frames
    (by index)."""
    xmin, xmax, ymin, ymax = roi
    c = element_centroids_xy(bundle, inst)
    mask = ((c[:, 0] >= xmin) & (c[:, 0] <= xmax) &
            (c[:, 1] >= ymin) & (c[:, 1] <= ymax))
    idx = np.where(mask)[0]
    f = np.asarray(bundle.field(inst, var), dtype=float)
    if frames is not None:
        f = f[frames]
    if f.ndim == 1:
        f = f[None, :]
    vals = f[:, idx]                                   # (N_t, N_p)
    keys = grid_keys(c[idx], elem_size)                # (N_p, 2)
    order = np.lexsort((keys[:, 0], keys[:, 1]))       # by ky, then kx
    return keys[order], vals[:, order]


def align_to_reference(keys: np.ndarray, values: np.ndarray,
                       ref_keys: np.ndarray) -> np.ndarray:
    """Reorder/select `values` (N_t, N_p) so its columns match `ref_keys`
    (the ROI grid keys of a reference domain), filling NaN where a reference
    cell is absent. Lets every domain be compared on a fixed common ROI grid."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    lut = {(int(k[0]), int(k[1])): j for j, k in enumerate(np.asarray(keys))}
    n_t = values.shape[0]
    out = np.full((n_t, len(ref_keys)), np.nan, dtype=float)
    for j, rk in enumerate(np.asarray(ref_keys)):
        col = lut.get((int(rk[0]), int(rk[1])))
        if col is not None:
            out[:, j] = values[:, col]
    return out


# ---------------------------------------------------------------------------
# sample_fn factory (bridges the optimiser to real Abaqus runs)
# ---------------------------------------------------------------------------
def mean_ke_ie_ratio(bundle, ke_channel="ALLKE", ie_channel="ALLIE"):
    """Kinetic/internal energy ratio used as the mass-scaling guard-rail,
    computed as the ratio of the time-AGGREGATED energies:

        guard = sum_t ALLKE(t) / sum_t ALLIE(t)

    which equals mean(ALLKE)/mean(ALLIE) and, under uniform time sampling, the
    ratio of the energy time-integrals (kinetic work / internal work).

    This replaces the earlier mean of the per-sample ratio ALLKE/ALLIE: at the
    start of the step the internal energy ALLIE is ~0, so the per-sample ratio
    exploded there and dominated the average (giving spuriously huge guard
    values). Aggregating first removes that division-by-near-zero sensitivity.
    Returns None if the channels are absent or the total internal energy is not
    positive.

    NOTE: for a genuine time-integral under NON-uniform sampling, the energy
    histories would have to be weighted by the sample time steps (a time channel
    is required); the ratio of sums above is exact only for uniform sampling."""
    try:
        ke = np.asarray(bundle.history(ke_channel), dtype=float)
        ie = np.asarray(bundle.history(ie_channel), dtype=float)
    except Exception:
        return None
    n = min(ke.size, ie.size)
    if n == 0:
        return None
    ke_sum = float(np.nansum(ke[:n]))
    ie_sum = float(np.nansum(ie[:n]))
    if not (ie_sum > 0):
        return None
    return ke_sum / ie_sum


def extract_force_normalized(bundle, elem_size, channel):
    """Reaction-force history channel (e.g. RF1_RP=Fc, RF2_RP=Ff) on the tool
    reference point, normalized by the element size, as a (n_samples, 1) column
    so it plugs into the E_q metric with a single point (N_p=1).

    Dividing by the element size gives homogeneity across meshes of different
    refinement (per the user's requirement). Returns None if the channel is
    absent (older bundle, or RP history output disabled)."""
    try:
        f = np.asarray(bundle.history(channel), dtype=float)
    except Exception:
        return None
    if f.size == 0 or elem_size is None or elem_size <= 0:
        return None
    return (f / float(elem_size)).reshape(-1, 1)


def _force_rp_history(cfg):
    """Ensure the RP reaction-force history output is on for a run that needs
    Fc. Best-effort: silently ignores an unexpected config layout."""
    try:
        cfg.step.output.ho_rf_on_rp = True
    except Exception:
        pass


def make_sample_fn(base_cfg, run_bundle, roi, elem_size, quantity_field_map,
                   instance=None, force_channels=None, grid_step=None):
    """Build a sample_fn(dims) -> {quantity: (Nt, Np)} for DomainOptimizer.

    base_cfg : ModelConfig whose euler_geometry dims are overwritten per
        candidate domain (nothing else changes).
    run_bundle : callable(cfg) -> results bundle or None. THIS is the only
        Abaqus-touching part. A None return yields all-NaN samples.
    roi : (xmin, xmax, ymin, ymax) extraction box (domain frame).
    quantity_field_map : {quantity_label: bundle_field_name}.
    instance : Eulerian instance name; if None, resolved per bundle.
    force_channels : {label: history_channel} of tool-RP reaction forces to add
        as normalized scalar-per-time quantities, e.g. {"Fc": "RF1_RP",
        "Ff": "RF2_RP"} -> value = channel / elem_size. Forces the RP history
        output on; raises if a successful run lacks a requested channel.
    grid_step : if given, the field quantities are sampled on a FIXED regular
        grid over the ROI (step = grid_step) by nearest centroid, so the
        evaluation points fill the ROI at a user-chosen spacing (independent of
        the mesh). If None, the raw element centroids inside the ROI are used
        and aligned to the first run's reference keys (exact when the mesh is
        anchored)."""
    import copy
    state = {"ref_keys": None, "grid": None}
    force_channels = dict(force_channels or {})

    def _resolve_inst(bundle):
        if instance is not None:
            return instance
        try:
            from gui.sensitivity.runner_core import eulerian_instance
            return eulerian_instance(bundle)
        except Exception:
            return None

    def _grid():
        if state["grid"] is None:
            from gui.sensitivity.mesh_opt import roi_grid
            state["grid"] = roi_grid(roi, grid_step)
        return state["grid"]

    def sample_fn(dims):
        cfg = copy.deepcopy(base_cfg)
        cfg.euler_geometry.h_wp = dims.h_wp
        cfg.euler_geometry.h_void = dims.h_void
        cfg.euler_geometry.l_wp = dims.l_wp
        cfg.euler_geometry.l_void = dims.l_void
        if force_channels:
            _force_rp_history(cfg)

        bundle = run_bundle(cfg)
        out: Dict[str, np.ndarray] = {}
        if bundle is None:
            for q in quantity_field_map:
                out[q] = np.full((1, 1), np.nan)
            for label in force_channels:
                out[label] = np.full((1, 1), np.nan)
            return out
        try:
            inst = _resolve_inst(bundle)
            if grid_step is not None:
                from gui.sensitivity.mesh_opt import nearest_samples
                grid = _grid()
                for q, fname in quantity_field_map.items():
                    out[q] = nearest_samples(bundle, fname, inst, grid)
            else:
                raw = {}
                last_keys = None
                for q, fname in quantity_field_map.items():
                    keys, vals = roi_samples(bundle, fname, inst, roi,
                                             elem_size)
                    raw[q] = (keys, vals)
                    last_keys = keys
                if state["ref_keys"] is None and last_keys is not None:
                    state["ref_keys"] = last_keys
                ref = state["ref_keys"]
                for q, (keys, vals) in raw.items():
                    out[q] = (align_to_reference(keys, vals, ref)
                              if ref is not None else vals)
            for label, channel in force_channels.items():
                fval = extract_force_normalized(bundle, cfg.elem_size, channel)
                if fval is None:
                    raise RuntimeError(
                        "%s requested but the %s history is absent — enable "
                        "the tool-RP reaction-force history output."
                        % (label, channel))
                out[label] = fval
        finally:
            close = getattr(bundle, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        return out

    return sample_fn


def verify_domain(sample_fn, dims: DomainDims, thresholds: Dict[str, float],
                  elem_size: float, quantities=DEFAULT_QUANTITIES,
                  order=DEFAULT_ORDER, step=None) -> dict:
    """Perturb each Eulerian dimension by +/-1 element around an identified
    domain and check the ROI fields barely move.

    The converged direction is LARGER (a bigger domain is closer to the
    reference), so stability is judged on the grow (+1 element) side; shrinking
    by one element is expected to leave the plateau and is reported for
    information. `sample_fn(DomainDims) -> {q: (Nt, Np)}` (anchored-centroid
    grid, as for the domain optimiser). Returns
    {dim: {"plus": errors, "minus": errors}, "stable": bool}."""
    # The bracketing step is the IMPOSED resolution (defaults to one element);
    # snapped to a whole number of elements so the perturbed domain stays a
    # valid mesh.
    if step is None or step <= 0:
        step = elem_size
    n_step = max(1, int(round(step / elem_size)))
    d_step = n_step * elem_size
    base = sample_fn(dims)
    out = {}
    stable = True
    for name in order:
        v = getattr(dims, name)
        bigger = DomainDims(dims.h_wp, dims.h_void, dims.l_wp, dims.l_void)
        setattr(bigger, name, v + d_step)
        smaller = DomainDims(dims.h_wp, dims.h_void, dims.l_wp, dims.l_void)
        setattr(smaller, name, max(elem_size, v - d_step))
        e_plus = errors_between(base, sample_fn(bigger), quantities)
        e_minus = errors_between(base, sample_fn(smaller), quantities)
        out[name] = {"plus": e_plus, "minus": e_minus}
        if not all_below(e_plus, thresholds):
            stable = False
    out["stable"] = stable
    return out
