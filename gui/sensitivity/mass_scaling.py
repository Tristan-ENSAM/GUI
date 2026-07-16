# -*- coding: utf-8 -*-
"""
Mass-scaling coefficient identification (pipeline step 0).

Physical motivation (from the user's runs): the ROI velocity is unstable
because of spurious dynamic dilatations; adding mass scaling stabilizes it.
More mass scaling also raises the kinetic energy, so it must be bounded by an
energy guard-rail (mean ALLKE/ALLIE) to preserve the quasi-static assumption.

Method (agreed):
  - the criterion is on the ROI VELOCITY only (Vx, Vy): stable when doubling the
    factor no longer changes it (Cauchy, E_v < eps_v);
  - the factor INCREASES (doubling up from a minimum) until velocity stabilizes,
    while the guard mean(ALLKE/ALLIE) stays below its threshold (else the search
    stops — over-scaling);
  - the retained value is the LARGEST admissible factor (least costly = biggest
    time step): the guard bounds it from above, velocity stability from below;
    a dichotomy on the guard (opt-in via a resolution) refines that upper bound.

Pure module: `sample_fn(factor) -> {"Vx":(Nt,Np), "Vy":(Nt,Np), "guard": float}`
is injected (the tab wires it to real Abaqus runs; tests use an analytic one),
where "guard" is the mean ALLKE/ALLIE for that run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from gui.sensitivity.domain_opt import errors_between, all_below


@dataclass
class MassScalingResult:
    identified: float                       # identified mass-scaling factor
    guard_at_identified: Optional[float]    # mean ALLKE/ALLIE at that factor
    velocity_converged: bool                # velocity actually stabilized
    guard_ok: bool                          # guard below threshold at identified
    history: List[tuple] = field(default_factory=list)  # (factor, errs, guard)
    n_runs: int = 0


def identify_mass_scaling(
        sample_fn: Callable[[float], Dict[str, object]],
        velocity_thresholds: Dict[str, float], guard_threshold: float,
        start: float, factor: float = 2.0, max_factor: Optional[float] = None,
        bisection_resolution: float = 0.0, max_bisect_iters: int = 60,
        max_doublings: int = 12,
        progress_cb=None, should_cancel=None) -> MassScalingResult:
    """Identify the mass-scaling factor. See module docstring.

    velocity_thresholds : {"Vx": eps, "Vy": eps} (physical units, mm/s).
    guard_threshold : upper bound on mean ALLKE/ALLIE (dimensionless).
    start : minimum factor to start doubling from.
    max_factor : optional hard cap on the factor.
    bisection_resolution : stop bisecting once the factor bracket is narrower
        than this (a resolution on the parameter, e.g. 100). 0 -> bisect up to
        max_bisect_iters.
    Returns MassScalingResult. `identified` = the LARGEST admissible factor
    (largest factor with guard < threshold, where velocity is also stable). If
    the guard is exceeded before velocity stabilizes, `velocity_converged` is
    False (no fully-admissible factor — reported, not hidden)."""
    if not (factor > 1.0):
        raise ValueError("factor must be > 1 (the mass-scaling factor grows)")
    vel_q = tuple(velocity_thresholds.keys())
    cache: Dict[float, Dict[str, object]] = {}
    n = {"runs": 0}
    history: List[tuple] = []

    def sample(f):
        key = round(float(f), 12)
        if key not in cache:
            cache[key] = sample_fn(f)
            n["runs"] += 1
        return cache[key]

    def vel_only(s):
        return {q: s[q] for q in vel_q if q in s}

    def guard_of(s):
        g = s.get("guard")
        return None if g is None else float(g)

    def errs(fa, fb):
        return errors_between(vel_only(sample(fa)), vel_only(sample(fb)), vel_q)

    start = float(start)
    sample(start)

    def vel_stable(f):
        """Velocity converged at f: growing to f*factor barely changes it. If
        the factor cannot grow (capped by max_factor), stability is unverifiable
        -> reported as not converged (conservative)."""
        fn = f * factor
        if max_factor is not None and fn > max_factor:
            fn = max_factor
        if fn <= f:
            return False                     # cannot grow -> unverifiable
        return all_below(errs(f, fn), velocity_thresholds)

    # -- bracketing: double up until the energy guard is exceeded -------------
    # We keep the LARGEST admissible factor (least costly = biggest time step):
    # the guard bounds it from above, velocity stability from below.
    f = start
    f_ok = None                              # largest factor with guard OK
    f_over = None                            # first factor with guard exceeded
    velocity_ever = False
    for _ in range(max_doublings):
        if should_cancel is not None and should_cancel():
            break
        g = guard_of(sample(f))
        vs = vel_stable(f)
        if vs:
            velocity_ever = True
        fn = f * factor
        if max_factor is not None and fn > max_factor:
            fn = max_factor
        history.append((f, errs(f, fn), g))
        if progress_cb is not None:
            progress_cb({"phase": "mass_scaling", "factor": f, "guard": g,
                         "vel_stable": vs, "n_runs": n["runs"]})
        if g is not None and g >= guard_threshold:
            f_over = f                        # guard exceeded here
            break
        f_ok = f                              # guard OK here
        if fn <= f:
            break
        f = fn
    if f_ok is None:
        f_ok = start

    # -- dichotomy (opt-in): largest factor with guard < threshold -----------
    # (guard is monotone increasing with the factor). Runs only when a
    # resolution is given; otherwise the bracketing bound f_ok is kept.
    if f_over is not None and bisection_resolution > 0:
        lo, hi, best = f_ok, f_over, f_ok
        it = 0
        while (hi - lo) > bisection_resolution and it < max_bisect_iters:
            it += 1
            mid = 0.5 * (lo + hi)
            g = guard_of(sample(mid))
            if g is not None and g < guard_threshold:
                best = mid                    # guard OK -> go larger (cheaper)
                lo = mid
            else:
                hi = mid
        identified = best
    else:
        identified = f_ok

    # admissibility at the retained (largest-admissible) factor
    g_id = guard_of(sample(identified))
    guard_ok = (g_id is None) or (g_id < guard_threshold)
    velocity_converged = velocity_ever and vel_stable(identified)
    if progress_cb is not None:
        progress_cb({"phase": "mass_scaling_done", "identified": identified,
                     "guard": g_id, "n_runs": n["runs"]})
    return MassScalingResult(
        identified=identified, guard_at_identified=g_id,
        velocity_converged=velocity_converged, guard_ok=guard_ok,
        history=history, n_runs=n["runs"])


def make_mass_scaling_sample_fn(base_cfg, run_bundle, roi, grid_step,
                                velocity_field_map, instance=None):
    """Return sample_fn(factor) -> {vel_label: (Nt, Np), "guard": float} for the
    mass-scaling step.

    Sets `mass_scaling_enabled` + `mass_scaling_factor_eulerian` on a copy of the
    config, forces the energy history on, runs one Abaqus job, and returns the
    ROI velocity sampled on the fixed grid plus the guard = mean ALLKE/ALLIE.

    velocity_field_map : {"Vx": "V1", "Vy": "V2"}.
    UNTESTED without Abaqus (run_bundle is injected). Raises if a run yields no
    bundle or lacks the energy history for the guard."""
    import copy
    from gui.sensitivity.mesh_opt import roi_grid, nearest_samples
    from gui.sensitivity.runner_core import eulerian_instance
    from gui.sensitivity.domain_opt import mean_ke_ie_ratio

    grid = roi_grid(roi, grid_step)

    def sample_fn(factor):
        cfg = copy.deepcopy(base_cfg)
        cfg.step.mass_scaling_enabled = True
        cfg.step.mass_scaling_factor_eulerian = float(factor)
        try:
            # ALLKE/ALLIE come from the PRESELECT whole-model history
            cfg.step.output.ho_preselect = True
        except Exception:
            pass
        bundle = run_bundle(cfg)
        if bundle is None:
            raise RuntimeError("mass-scaling run produced no bundle "
                               "(factor=%.4g)" % factor)
        inst = instance or eulerian_instance(bundle)
        out = {q: nearest_samples(bundle, var, inst, grid)
               for q, var in velocity_field_map.items()}
        g = mean_ke_ie_ratio(bundle)
        if g is None:
            raise RuntimeError(
                "mass-scaling guard needs the ALLKE/ALLIE history — enable the "
                "PRESELECT history (step.output.ho_preselect).")
        out["guard"] = g
        return out

    return sample_fn
