# -*- coding: utf-8 -*-
"""
Eulerian domain sizing by forward-difference Jacobian.

WHAT THIS ANSWERS
-----------------
"How large must the Eulerian domain be before its boundaries stop influencing
the ROI?" The domain has FOUR independent dimensions (h_wp, h_void, l_wp,
l_void), each with a distinct physical role, so a single scalar size (or the
domain diagonal) cannot answer it.

METHOD
------
At the current domain d, run one base simulation plus one perturbed simulation
per dimension (step = `step_elems` element sizes). For each quantity of
interest Q the sensitivity is reported as an ELASTICITY -- a relative change
per relative change:

    J_i = ( ||Q(d + h e_i) - Q(d)|| / ||Q(d)|| ) * ( d_i / h )

Normalising this way makes the four dimensions comparable despite their
different magnitudes: J_i is "percent change in Q per percent change in d_i".

Convergence is reached when every elasticity is below its threshold: enlarging
any dimension no longer changes the ROI. That is the domain-sizing analogue of
a mesh convergence study -- with the caveat that there is no mathematical limit
here (unlike h -> 0), only insensitivity to the boundaries.

WHY THE MESH IS HELD FIXED
--------------------------
Only the domain dimensions vary, so the mesh nodes stay on the same pitch and
the ROI sampling error is systematic rather than run-dependent -- it largely
cancels in the comparison. Varying mesh AND domain at once would mix a
sampling artefact into the sensitivity. Nearest-centroid sampling is therefore
adequate here; it would NOT be for a mesh-refinement study.

THE PIECEWISE-SMOOTH CAVEAT
---------------------------
Chip formation involves discrete events (element deletion, contact changes). A
one-element perturbation can shift WHEN such an event occurs and produce a jump
that is not a derivative. The QoI is smooth in pieces, not smooth. Hence
`linearity_check`: the same derivative is evaluated at two step sizes, and a
large discrepancy flags that a discrete event -- not the boundary -- dominates
the measurement. Widen the step in that case.

COST
----
1 + 4 runs per iteration (+4 more if the linearity check is on).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.mesh_opt import roi_grid, nearest_samples

DIMENSION_NAMES = ("h_wp", "h_void", "l_wp", "l_void")

# Reverberation constraint (see ModelConfig.mass_scaling_bounds): the mass
# scaling window is empty unless the domain diagonal stays below this multiple
# of the element size. Enlarging the domain therefore has a hard ceiling.
_DIAGONAL_OVER_ELEM_MAX = 90.6


@dataclass
class DomainSample:
    """One simulation: ROI field samples + the force history + the guard."""
    dims: DomainDims
    fields: Dict[str, np.ndarray] = field(default_factory=dict)  # var -> (Nt, Np)
    force: Optional[np.ndarray] = None                           # (Nt,)
    guard_coefficient: Optional[float] = None                    # G = ratio / ms
    n_runs: int = 1


@dataclass
class JacobianResult:
    dims: DomainDims
    elasticities: Dict[str, Dict[str, float]]   # quantity -> dimension -> J
    converged: bool
    limiting: str                               # dimension driving the next step
    guard_coefficient: Optional[float]
    linearity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_runs: int = 0
    stopped_by: str = ""                        # "converged"|"diagonal"|"max_iter"|"cancelled"


def diagonal(dims: DomainDims) -> float:
    """Domain diagonal -- the longest wave path, which sets the reverberation
    frequency c/(2L) and hence the mass-scaling ceiling."""
    return math.hypot(dims.l_wp + dims.l_void, dims.h_wp + dims.h_void)


def diagonal_limit(elem_size: float) -> float:
    """Largest diagonal that still leaves a non-empty mass-scaling window."""
    return _DIAGONAL_OVER_ELEM_MAX * elem_size


def _as_dict(dims: DomainDims) -> Dict[str, float]:
    return {n: float(getattr(dims, n)) for n in DIMENSION_NAMES}


def _with(dims: DomainDims, name: str, value: float) -> DomainDims:
    d = _as_dict(dims)
    d[name] = value
    return DomainDims(**d)


def _relative_norm(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||b||, over the whole (frames x points) array.

    Returns NaN when b is all-zero: a relative measure has no meaning then,
    and NaN propagates to "not converged" rather than silently reading as 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    denom = float(np.linalg.norm(b[mask]))
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(a[mask] - b[mask]) / denom)


def guard_coefficient(bundle, mass_scaling_factor: float,
                      settled_fraction: float = 0.3) -> Optional[float]:
    """Measure G = <ALLKE/ALLIE> / ms on the settled window.

    The ratio of TIME-AGGREGATED energies is used, not the mean of the
    instantaneous ratios: at the start ALLIE is ~0 and the instantaneous ratio
    explodes (0.10 vs 1.3e-3 later on a real run).

    IMPORTANT -- G is not a universal constant:
      * ALLIE ACCUMULATES while ALLKE does not, so the ratio decays roughly as
        1/t. G must always be measured over the SAME duration to be
        comparable between runs.
      * ALLKE scales with the moving mass, hence with the domain VOLUME, while
        ALLIE is dominated by plastic work in the cutting zone. Enlarging the
        domain therefore INCREASES G and LOWERS the mass-scaling ceiling.
    This is why G is re-measured on every run of the study instead of being
    taken as a fixed number.
    """
    try:
        ke = np.asarray(bundle.history("ALLKE"), dtype=float)
        ie = np.asarray(bundle.history("ALLIE"), dtype=float)
        t = np.asarray(bundle.history_time(), dtype=float)
    except Exception:
        return None
    if ke.size == 0 or ie.size != ke.size or t.size != ke.size:
        return None
    m = t >= settled_fraction * t[-1]
    denom = float(ie[m].sum())
    if denom <= 0 or mass_scaling_factor <= 0:
        return None
    return float(ke[m].sum() / denom) / float(mass_scaling_factor)


def sample_domain(run_bundle: Callable, cfg, dims: DomainDims, roi,
                  grid_step: float, field_vars: Sequence[str],
                  instance: str = "Euler",
                  force_channel: str = "RF1",
                  mass_scaling_factor: float = 1.0,
                  settled_fraction: float = 0.3) -> DomainSample:
    """Run one simulation at `dims` and reduce it to comparable quantities.

    The ROI grid is FIXED and independent of the mesh, so samples from
    different domains are directly comparable element-by-element.
    """
    cfg.euler_geometry.h_wp = dims.h_wp
    cfg.euler_geometry.h_void = dims.h_void
    cfg.euler_geometry.l_wp = dims.l_wp
    cfg.euler_geometry.l_void = dims.l_void
    bundle = run_bundle(cfg)
    if bundle is None:
        raise RuntimeError("run_bundle returned None for dims=%r" % (dims,))

    points = roi_grid(roi, grid_step)
    fields = {}
    for var in field_vars:
        try:
            fields[var] = np.asarray(nearest_samples(bundle, var, instance,
                                                     points), dtype=float)
        except Exception:
            fields[var] = np.full((1, len(points)), np.nan)

    force = None
    try:
        force = np.asarray(bundle.history(force_channel), dtype=float)
    except Exception:
        pass

    return DomainSample(
        dims=dims, fields=fields, force=force,
        guard_coefficient=guard_coefficient(bundle, mass_scaling_factor,
                                            settled_fraction))


def _quantity_norms(a: DomainSample, b: DomainSample,
                    field_vars: Sequence[str]) -> Dict[str, float]:
    """Relative norm of every quantity between two samples."""
    out = {v: _relative_norm(a.fields.get(v, np.empty(0)),
                             b.fields.get(v, np.empty(0)))
           for v in field_vars}
    if a.force is not None and b.force is not None:
        out["force"] = _relative_norm(a.force, b.force)
    else:
        out["force"] = float("nan")
    return out


def jacobian_at(run_bundle: Callable, cfg, dims: DomainDims, roi,
                grid_step: float, field_vars: Sequence[str],
                elem_size: float, step_elems: int = 1,
                base: Optional[DomainSample] = None,
                mass_scaling_factor: float = 1.0,
                should_cancel: Optional[Callable] = None,
                progress_cb: Optional[Callable] = None) -> Dict:
    """Forward-difference elasticities at `dims`, one per dimension.

    Perturbs only in the GROWING direction: the question is whether enlarging
    still changes the ROI, so a forward difference is the physically meaningful
    one (and shrinking could invalidate the geometry).
    """
    step = step_elems * elem_size
    if base is None:
        base = sample_domain(run_bundle, cfg, dims, roi, grid_step, field_vars,
                             mass_scaling_factor=mass_scaling_factor)
        if progress_cb:
            progress_cb({"phase": "domain_base", "dims": _as_dict(dims),
                         "guard_coefficient": base.guard_coefficient})

    elasticities: Dict[str, Dict[str, float]] = {}
    n_runs = 1
    for name in DIMENSION_NAMES:
        if should_cancel is not None and should_cancel():
            break
        d_i = getattr(dims, name)
        pert = sample_domain(run_bundle, cfg, _with(dims, name, d_i + step),
                             roi, grid_step, field_vars,
                             mass_scaling_factor=mass_scaling_factor)
        n_runs += 1
        norms = _quantity_norms(pert, base, field_vars)
        # elasticity: relative change of Q per relative change of d_i
        scale = d_i / step if step > 0 else float("nan")
        for q, val in norms.items():
            elasticities.setdefault(q, {})[name] = val * scale
        if progress_cb:
            progress_cb({"phase": "domain_jacobian", "dimension": name,
                         "elasticities": {q: elasticities[q][name]
                                          for q in elasticities},
                         "n_runs": n_runs})
    return {"base": base, "elasticities": elasticities, "n_runs": n_runs}


def run_domain_study(run_bundle: Callable, cfg, roi,
                     initial_dims: DomainDims,
                     grid_step: float,
                     thresholds: Dict[str, float],
                     elem_size: float,
                     field_vars: Sequence[str] = ("EVF", "TEMP", "V"),
                     step_elems: int = 1,
                     grow_elems: int = 4,
                     max_iterations: int = 8,
                     mass_scaling_factor: float = 1.0,
                     linearity_check: bool = True,
                     should_cancel: Optional[Callable] = None,
                     progress_cb: Optional[Callable] = None
                     ) -> List[JacobianResult]:
    """Enlarge the domain until every elasticity falls below its threshold.

    thresholds : per quantity, e.g. {"EVF": 0.02, "TEMP": 0.02, "V": 0.02,
                 "force": 0.02} -- "2% change in Q per 100% change in d_i".
    step_elems : perturbation used for the derivative (in element sizes).
    grow_elems : how much the limiting dimension grows between iterations.

    Stops early -- with stopped_by="diagonal" -- if growing would push the
    domain diagonal past `diagonal_limit(elem_size)`, i.e. past the point where
    the mass-scaling window closes. That constraint is a hard ceiling on how
    large the domain may become, and it is better to report it than to keep
    enlarging into an unusable configuration.
    """
    dims = initial_dims
    history: List[JacobianResult] = []
    base: Optional[DomainSample] = None

    # The ceiling must be checked on the STARTING domain too, not only before
    # growing: an initial domain already past the limit would otherwise be
    # studied (and reported as converged) in a configuration where no valid
    # mass-scaling factor exists.
    limit = diagonal_limit(elem_size)
    if diagonal(dims) > limit:
        return [JacobianResult(
            dims, {}, False, "", None, n_runs=0, stopped_by="diagonal")]

    for _ in range(max_iterations):
        if should_cancel is not None and should_cancel():
            history.append(JacobianResult(dims, {}, False, "", None,
                                          stopped_by="cancelled"))
            break

        j = jacobian_at(run_bundle, cfg, dims, roi, grid_step, field_vars,
                        elem_size, step_elems, base=base,
                        mass_scaling_factor=mass_scaling_factor,
                        should_cancel=should_cancel, progress_cb=progress_cb)
        base = j["base"]
        el = j["elasticities"]
        n_runs = j["n_runs"]

        linearity: Dict[str, Dict[str, float]] = {}
        if linearity_check:
            j2 = jacobian_at(run_bundle, cfg, dims, roi, grid_step, field_vars,
                             elem_size, 2 * step_elems, base=base,
                             mass_scaling_factor=mass_scaling_factor,
                             should_cancel=should_cancel,
                             progress_cb=progress_cb)
            n_runs += j2["n_runs"] - 1     # the base run is shared
            for q in el:
                for name in el[q]:
                    a = el[q][name]
                    b = j2["elasticities"].get(q, {}).get(name, float("nan"))
                    # ratio ~1 means the derivative is stable; far from 1 means
                    # a discrete event dominates and the step must be widened.
                    linearity.setdefault(q, {})[name] = (
                        (a / b) if (math.isfinite(a) and math.isfinite(b)
                                    and b != 0) else float("nan"))

        # converged when EVERY thresholded quantity is below its bound for
        # EVERY dimension; a NaN counts as not converged (conservative).
        converged = True
        worst_val, worst_dim = -1.0, ""
        for q, eps in thresholds.items():
            for name, val in el.get(q, {}).items():
                if not math.isfinite(val) or val >= eps:
                    converged = False
                if math.isfinite(val) and val > worst_val:
                    worst_val, worst_dim = val, name

        res = JacobianResult(
            dims=dims, elasticities=el, converged=converged,
            limiting=worst_dim, guard_coefficient=base.guard_coefficient,
            linearity=linearity, n_runs=n_runs,
            stopped_by="converged" if converged else "")
        history.append(res)
        if progress_cb:
            progress_cb({"phase": "domain_iteration", "dims": _as_dict(dims),
                         "elasticities": el, "converged": converged,
                         "limiting": worst_dim,
                         "guard_coefficient": base.guard_coefficient})
        if converged or not worst_dim:
            break

        # grow the most influential dimension and check the hard ceiling
        grown = _with(dims, worst_dim,
                      getattr(dims, worst_dim) + grow_elems * elem_size)
        if diagonal(grown) > diagonal_limit(elem_size):
            res.stopped_by = "diagonal"
            break
        dims = grown
        base = None            # the base run must be redone at the new dims
    else:
        if history:
            history[-1].stopped_by = history[-1].stopped_by or "max_iter"

    return history
