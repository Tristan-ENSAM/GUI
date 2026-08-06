# -*- coding: utf-8 -*-
"""
Element-size (mesh) convergence core for the Model tab.

Unlike the domain-size study (where the structured Eulerian mesh is anchored on
x=0 / y=0 so ROI centroids coincide exactly across domains), changing the
element size moves every centroid. Fields at two element sizes are therefore
compared on a FIXED regular ROI grid, each run's element-centroid field being
resampled onto that grid by NEAREST NEIGHBOUR (chosen by the user: no added
interpolation error, preserves the EVF material/void interface; the trade-off
is a piecewise-constant "staircase" contribution to the error).

Same Cauchy criterion as the domain study, per quantity q in {Vx, Vy, T, EVF}:
    E_q = sqrt( 1/(Nt*Np) * sum [ q_fine(p_j,t_i) - q_coarse(p_j,t_i) ]^2 )
A size is "stable" once halving it no longer changes the ROI fields (all
E_q < eps_q). The identified size is the COARSEST stable one (fewest DOF).

Pure module: the simulation is an injected `sample_fn(size) -> {q: (Nt, Np)}`,
so the search is unit-testable without Abaqus.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from gui.sensitivity.domain_opt import (
    roi_rmse, errors_between, all_below, element_centroids_xy,
    DEFAULT_QUANTITIES)


# ---------------------------------------------------------------------------
# Fixed ROI grid + nearest-neighbour resampling
# ---------------------------------------------------------------------------
def roi_grid(roi, step: float) -> np.ndarray:
    """Regular grid of evaluation points covering the ROI box.

    roi = (xmin, xmax, ymin, ymax); `step` is the spacing (mesh-independent).
    Points sit at xmin + k*step (and similarly in y), always including a point
    at or before xmax/ymax. Returns (N_p, 2)."""
    xmin, xmax, ymin, ymax = roi
    if step <= 0:
        raise ValueError("step must be > 0")
    nx = max(1, int(math.floor((xmax - xmin) / step + 1e-9)) + 1)
    ny = max(1, int(math.floor((ymax - ymin) / step + 1e-9)) + 1)
    xs = xmin + step * np.arange(nx)
    ys = ymin + step * np.arange(ny)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])


def _nearest_indices(centroids: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid for each point. Uses scipy's cKDTree when
    available, else a vectorised brute-force fallback."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(centroids).query(points)[1]
    except Exception:
        # brute force: (Np, Ne) distance^2
        diff = points[:, None, :] - centroids[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        return np.argmin(d2, axis=1)


def nearest_samples(bundle, var, inst, points, frames=None) -> np.ndarray:
    """Resample element field `var` onto `points` (N_p, 2) by nearest centroid.
    Returns (N_t, N_p). `frames` optionally selects frames by index."""
    centroids = element_centroids_xy(bundle, inst)
    idx = _nearest_indices(centroids, np.asarray(points, dtype=float))
    f = np.asarray(bundle.field(inst, var), dtype=float)
    if frames is not None:
        f = f[frames]
    if f.ndim == 1:
        f = f[None, :]
    return f[:, idx]


# ---------------------------------------------------------------------------
# Refinement search
# ---------------------------------------------------------------------------
@dataclass
class MeshConvResult:
    identified: float                 # coarsest stable element size (= final)
    history: List[tuple] = field(default_factory=list)  # (size_fine, errors)
    n_runs: int = 0
    converged: bool = False
    initial: float = 0.0              # phase-0 start size
    intermediate: float = 0.0         # coarsest stable size at the END of the
                                      # halving phase (before the optional
                                      # bisection refinement)


def refine_until_stable(sample_fn: Callable[[float], Dict[str, np.ndarray]],
                        thresholds: Dict[str, float], start_size: float,
                        quantities=DEFAULT_QUANTITIES, factor: float = 0.5,
                        min_size: Optional[float] = None, max_steps: int = 10,
                        bisection_resolution: float = 0.0,
                        max_bisect_iters: int = 60,
                        progress_cb=None, should_cancel=None) -> MeshConvResult:
    """Halve the element size until the ROI fields stop changing, then bisect.

    Halving compares consecutive sizes (size, size*factor); the first time all
    E_q < eps_q, the COARSER size of that pair is stable and the transition lies
    in [that size, that size/factor]. A bisection then returns the COARSEST
    stable size in that bracket, stopping once the bracket is narrower than
    `bisection_resolution` (a resolution on the element size, e.g. 0.001 mm;
    0 -> down to max_bisect_iters). `sample_fn(size) -> {q: (Nt, Np)}` must
    resample onto the SAME fixed ROI grid for every size. Returns
    MeshConvResult."""
    if not (0.0 < factor < 1.0):
        raise ValueError("factor must be in (0, 1)")
    size = float(start_size)
    prev = sample_fn(size)
    n_runs = 1
    history: List[tuple] = []
    identified = size
    converged = False
    stable_sample = prev
    for _ in range(max_steps):
        if should_cancel is not None and should_cancel():
            break
        nxt = size * factor
        if min_size is not None and nxt < min_size:
            break
        cur = sample_fn(nxt)
        n_runs += 1
        err = errors_between(prev, cur, quantities)
        history.append((nxt, err))
        if progress_cb is not None:
            progress_cb({"phase": "refine", "size": nxt, "errors": err,
                         "n_runs": n_runs})
        if all_below(err, thresholds):
            identified = size            # the coarser, already-stable size
            stable_sample = prev         # its (converged) reference field
            converged = True
            break
        size = nxt
        prev = cur
    else:
        identified = size                # never stabilised -> finest tried

    # End of the halving phase: this is the "intermediate" value (before the
    # optional bisection refinement below).
    intermediate = identified

    # Bisection (opt-in): only when a resolution is given. Finds the coarsest
    # Cauchy-stable size in [identified, identified/factor] using the same
    # self-convergence test E(s, s*factor) < eps. Without a resolution the
    # halving result is returned as-is (no extra runs).
    if converged and bisection_resolution > 0:
        lo = identified                  # fine, stable end
        hi = identified / factor         # coarse, unstable end (bracket bound)
        if min_size is not None:
            lo = max(lo, min_size)
        best = identified
        it = 0
        while (hi - lo) > bisection_resolution and it < max_bisect_iters:
            it += 1
            mid = 0.5 * (lo + hi)
            s_mid = sample_fn(mid)
            s_fine = sample_fn(mid * factor)
            n_runs += 2
            err = errors_between(s_mid, s_fine, quantities)
            history.append((mid, err))
            if progress_cb is not None:
                progress_cb({"phase": "bisect", "size": mid, "errors": err,
                             "n_runs": n_runs})
            if all_below(err, thresholds):
                best = mid               # stable -> try coarser
                lo = mid
            else:
                hi = mid
        identified = best
    return MeshConvResult(identified=identified, history=history,
                          n_runs=n_runs, converged=converged,
                          initial=float(start_size), intermediate=intermediate)


def verify_stability(sample_fn: Callable[[float], Dict[str, np.ndarray]],
                     size: float, thresholds: Dict[str, float],
                     quantities=DEFAULT_QUANTITIES, factor: float = 0.5,
                     step: Optional[float] = None) -> dict:
    """Perturb an identified size by one increment on each side and check the
    ROI fields barely move.

    Because the identified size is the COARSEST stable one, the meaningful
    stability test is on the FINER side (refining further must stay in the
    converged plateau, E < eps); one step COARSER is expected to leave the
    plateau, so its error is reported for information only.

    Returns {"finer": errors, "coarser": errors, "stable": bool} with
    stable == all finer-side E_q below threshold."""
    base = sample_fn(size)
    # Bracket by +/- the IMPOSED resolution when given (else factor-based).
    if step is not None and step > 0:
        finer_size = size - step if size - step > 0 else size * factor
        coarser_size = size + step
    else:
        finer_size = size * factor
        coarser_size = size / factor
    finer = errors_between(base, sample_fn(finer_size), quantities)
    coarser = errors_between(base, sample_fn(coarser_size), quantities)
    stable = all_below(finer, thresholds)
    return {"finer": finer, "coarser": coarser, "stable": stable}


# ---------------------------------------------------------------------------
# Abaqus-backed mesh sample_fn (bridges run_bundle to the fixed-grid metric)
# ---------------------------------------------------------------------------
def finest_size(start_size: float, factor: float, max_steps: int,
                min_size=None) -> float:
    """The smallest element size the halving search can reach, used as the
    fixed ROI-grid step (so the grid resolves the finest mesh)."""
    s = float(start_size) * (float(factor) ** int(max_steps))
    if min_size is not None:
        s = max(s, float(min_size))
    return s


def make_mesh_sample_fn(base_cfg, run_bundle, roi, grid_step,
                        quantity_field_map, size_attr, held_sizes=None,
                        instance=None, force_channels=None):
    """Return sample_fn(size) -> {q: (Nt, Np)} for the mesh study.

    Sets `size_attr` on a copy of the config, applies any `held_sizes`, runs one
    Abaqus job, and resamples the EULERIAN ROI fields onto a FIXED grid
    (step=grid_step) by nearest centroid.

    force_channels : {label: history_channel} of tool-RP reaction forces to add
    as normalized scalar-per-time quantities (value = channel / cfg.elem_size,
    i.e. the ACTUAL workpiece element size of that run). Forces the RP history
    output on; raises if a run lacks a requested channel.

    The grid is built once so every size is compared on identical points.
    Raises RuntimeError if a run yields no bundle."""
    import copy
    from gui.sensitivity.runner_core import eulerian_instance
    from gui.sensitivity.domain_opt import (extract_force_normalized,
                                            _force_rp_history)

    grid = roi_grid(roi, grid_step)
    force_channels = dict(force_channels or {})

    def sample_fn(size):
        cfg = copy.deepcopy(base_cfg)
        setattr(cfg, size_attr, float(size))
        if held_sizes:
            for k, v in held_sizes.items():
                setattr(cfg, k, float(v))
        if force_channels:
            _force_rp_history(cfg)
        bundle = run_bundle(cfg)
        if bundle is None:
            raise RuntimeError("mesh run produced no results bundle "
                               "(%s=%.4g)" % (size_attr, size))
        inst = instance or eulerian_instance(bundle)
        out = {q: nearest_samples(bundle, var, inst, grid)
               for q, var in quantity_field_map.items()}
        for label, channel in force_channels.items():
            fval = extract_force_normalized(bundle, cfg.elem_size, channel)
            if fval is None:
                raise RuntimeError(
                    "%s requested but the %s history is absent — enable the "
                    "tool-RP reaction-force history output." % (label, channel))
            out[label] = fval
        return out

    return sample_fn
