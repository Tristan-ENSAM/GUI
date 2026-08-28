# -*- coding: utf-8 -*-
"""
Mass-scaling coefficient identification (pipeline step 0).

Physical reading: mass scaling acts as a LOW-PASS on the velocity fields —
raising the density slows the elastic waves, so the dynamic compression
oscillations that pollute the ROI velocity are damped. The un-scaled run
(factor = 1) is therefore NOT used as a reference; what is sought is the factor
around which the velocity fields are LEAST sensitive to the factor itself.

Method (agreed):
  - the criterion is on the ROI VELOCITY only (Vx, Vy). Each curve point is the
    normalized self-convergence sensitivity
        E(f) = max_q  MAD(V_q(f), V_q(f*factor)) / eps_q
    i.e. the difference between two CONSECUTIVE factors of the ladder, labelled
    at the SMALLER member of the pair;
  - the retained factor MINIMIZES E (argmin) subject to the energy guard-rail
    (ratio of time-aggregated ALLKE/ALLIE) staying below its threshold and the
    factor staying <= max_factor. The eps_q are WEIGHTS used to aggregate Vx and
    Vy into a single E: E < 1 is reported, not required;
  - phase 1 walks the ladder f, f*factor, ... until E RISES (the minimum is then
    bracketed) or the guard / cap stops it; phase 2 refines that bracket with a
    golden-section search on log(f) — the efficient form of a ternary search:
    one new evaluation per iteration instead of two.

Cost: an on-ladder point costs ONE Abaqus run (its partner is the next ladder
point, itself a point); an off-ladder probe costs TWO.

Pure module: `sample_fn(factor) -> {"Vx":(Nt,Np), "Vy":(Nt,Np), "guard": float}`
is injected (the tab wires it to real Abaqus runs; tests use an analytic one),
where "guard" is the ratio of time-aggregated ALLKE/ALLIE for that run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional

from gui.sensitivity.domain_opt import errors_between



@dataclass
class MassScalingResult:
    identified: float                       # retained factor (= final)
    guard_at_identified: Optional[float]    # energy guard there
    velocity_converged: bool                # E < 1 at the retained point (info)
    guard_ok: bool                          # guard below threshold at identified
    history: List[tuple] = field(default_factory=list)  # (factor, errs, guard)
    n_runs: int = 0
    initial: float = 0.0                    # phase-0 start factor
    intermediate: float = 0.0               # ladder minimum (end of phase 1)
    limited_by: str = ""                    # "minimum" | "guard" | "cap" |
                                            # "start" | "cancelled"
    sensitivity: float = float("inf")       # E at the retained point


def identify_mass_scaling(
        sample_fn: Callable[[float], Dict[str, object]],
        velocity_thresholds: Dict[str, float], guard_threshold: float,
        start: float, factor: float = 2.0, max_factor: Optional[float] = None,
        bisection_resolution: float = 0.0, max_bisect_iters: int = 60,
        max_doublings: int = 12,
        progress_cb=None, should_cancel=None) -> MassScalingResult:
    """Identify the mass-scaling factor that MINIMIZES the velocity sensitivity.

    velocity_thresholds : {"Vx": eps, "Vy": eps} (physical units, mm/s). Used as
        WEIGHTS to aggregate the channels into E = max_q E_q/eps_q, not as an
        admissibility threshold.
    guard_threshold : upper bound on the energy guard. Since the guard grows
        with the factor, it acts as a ceiling on the admissible factors.
    start : first factor of the ladder.
    max_factor : optional hard cap on the RETAINED factor (a comparison partner
        may exceed it once, see the ladder note below).
    bisection_resolution : stop the golden-section refinement once the bracket is
        narrower than this. <= 0 -> refine up to max_bisect_iters (costly: each
        off-ladder probe is two Abaqus runs).

    Phase 1 - ladder f, f*factor, ...: stop at the first RISE of E (the minimum
      is then bracketed by the two neighbours of the ladder minimum), or when the
      guard is violated, or when the next step would pass the cap.
    Phase 2 - golden-section search on log(f) inside that bracket; probes whose
      guard is violated are scored +inf so the search moves away from them.

    The retained point is the admissible evaluated factor with the lowest E
    (ties broken by the lower guard). A point designates the PAIR (f, f*factor):
    if BOTH members satisfy the guard and the larger stays within the cap, the
    LARGER one is retained (largest stable time step for the same sensitivity).

    `limited_by` states what set the result:
      "minimum"   - an interior minimum was bracketed and refined;
      "start"     - E rises from the very first ladder step (minimum at start);
      "guard"     - the search was stopped by the energy guard;
      "cap"       - E was still decreasing when the cap was reached;
      "cancelled" - cancelled before any evaluation."""
    if not (factor > 1.0):
        raise ValueError("factor must be > 1 (the mass-scaling factor grows)")
    vel_q = tuple(velocity_thresholds.keys())
    cache: Dict[float, Dict[str, object]] = {}
    n = {"runs": 0}
    history: List[tuple] = []
    evaluated: Dict[float, tuple] = {}   # factor -> (score, errors, guard, adm)

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

    def score_of(e):
        """Aggregate the per-channel errors into E = max_q E_q / eps_q."""
        vals = [e[q] / velocity_thresholds[q] for q in e
                if q in velocity_thresholds and velocity_thresholds[q] > 0]
        return max(vals) if vals else float("inf")

    def evaluate(f):
        """Sensitivity E(f) = E(f, f*factor) and guard at f, cached, emitting one
        curve point per new factor. The partner f*factor may exceed max_factor:
        the cap bounds the RETAINED factor, not the probe used to measure the
        sensitivity. Returns (score, errors, guard, admissible)."""
        key = round(float(f), 12)
        if key in evaluated:
            return evaluated[key]
        e = errs(f, f * factor)
        g = guard_of(sample(f))
        rec = (score_of(e), e, g, (g is None) or (g < guard_threshold))
        evaluated[key] = rec
        history.append((f, e, g))
        if progress_cb is not None:
            progress_cb({"phase": "mass_scaling", "factor": f, "guard": g,
                         "errors": e, "n_runs": n["runs"]})
        return rec

    def done(identified, g_id, score, limited_by, intermediate):
        guard_ok = (g_id is None) or (g_id < guard_threshold)
        if progress_cb is not None:
            progress_cb({"phase": "mass_scaling_done", "identified": identified,
                         "guard": g_id, "converged": score < 1.0,
                         "limited_by": limited_by, "n_runs": n["runs"]})
        return MassScalingResult(
            identified=identified, guard_at_identified=g_id,
            velocity_converged=score < 1.0, guard_ok=guard_ok,
            history=history, n_runs=n["runs"], initial=start,
            intermediate=float(intermediate), limited_by=limited_by,
            sensitivity=score)

    start = float(start)

    # -- Phase 1: walk the ladder until E rises (the minimum is bracketed) ----
    ladder = []                          # [(factor, score, admissible)]
    stop_reason = "cap"
    f_bad = None                         # first ladder factor violating the guard
    f = start
    for _ in range(max_doublings):
        if should_cancel is not None and should_cancel():
            break
        s, _e, _g, adm = evaluate(f)
        ladder.append((f, s, adm))
        if not adm:
            f_bad = f
            stop_reason = "guard"        # guard grows with f: nothing above is OK
            break
        if len(ladder) >= 2 and s > ladder[-2][1]:
            stop_reason = "minimum"      # E rose -> the minimum is bracketed
            break
        fn = f * factor
        if max_factor is not None and fn > max_factor:
            stop_reason = "cap"
            break
        f = fn

    if not ladder:                       # cancelled before any evaluation
        return MassScalingResult(
            identified=start, guard_at_identified=None,
            velocity_converged=False, guard_ok=False, history=history,
            n_runs=n["runs"], initial=start, intermediate=start,
            limited_by="cancelled")

    adm_ladder = [(lf, ls) for (lf, ls, la) in ladder if la]
    if not adm_ladder:                   # the guard is already violated at start
        rec = evaluated[round(start, 12)]
        return done(start, rec[2], rec[0], "guard", start)

    idx = min(range(len(adm_ladder)), key=lambda k: adm_ladder[k][1])
    f_ladder_min = adm_ladder[idx][0]
    lo = adm_ladder[idx - 1][0] if idx > 0 else f_ladder_min
    if idx + 1 < len(adm_ladder):
        hi = adm_ladder[idx + 1][0]
    elif f_bad is not None:
        # The sensitivity is still decreasing where the guard cut the ladder: the
        # CONSTRAINED optimum sits at the guard ceiling, somewhere in
        # [last admissible, first inadmissible]. Probes above the ceiling score
        # +inf, so the search converges onto the ceiling from below.
        hi = f_bad
    else:
        hi = f_ladder_min
    if 0 < idx < len(adm_ladder) - 1:
        limited_by = "minimum"
    elif idx == 0 and len(adm_ladder) > 1:
        limited_by = "start"
    else:
        limited_by = stop_reason

    # -- Phase 2: golden-section refinement of [lo, hi] on log(f) -------------
    # One new evaluation per iteration (a strict ternary search would need two
    # for the same bracket reduction). Probes violating the guard score +inf.
    if hi > lo:
        inv_phi = (5.0 ** 0.5 - 1.0) / 2.0
        a, b = math.log(lo), math.log(hi)

        def probe(x_log):
            s, _e, _g, adm = evaluate(math.exp(x_log))
            return s if adm else float("inf")

        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        sc, sd = probe(c), probe(d)
        it = 0
        while (math.exp(b) - math.exp(a)) > bisection_resolution and \
                it < max_bisect_iters:
            if should_cancel is not None and should_cancel():
                break
            it += 1
            if sc < sd:
                b, d, sd = d, c, sc
                c = b - inv_phi * (b - a)
                sc = probe(c)
            else:
                a, c, sc = c, d, sd
                d = a + inv_phi * (b - a)
                sd = probe(d)

    # -- Retain the admissible point with the lowest sensitivity -------------
    adm_pts = [(pf, rec[0], rec[2]) for pf, rec in evaluated.items() if rec[3]]
    f_best, s_best, g_best = min(
        adm_pts, key=lambda t: (t[1], t[2] if t[2] is not None else float("inf")))

    # A point is a PAIR (f, f*factor): if both members satisfy the guard and the
    # larger stays within the cap, keep the larger (bigger stable time step for
    # the same measured sensitivity). It was already run as the partner.
    f_large = f_best * factor
    within_cap = (max_factor is None) or (f_large <= max_factor)
    g_large = guard_of(sample(f_large)) if within_cap else None
    small_ok = (g_best is None) or (g_best < guard_threshold)
    large_ok = (g_large is None) or (g_large < guard_threshold)
    if within_cap and small_ok and large_ok:
        return done(f_large, g_large, s_best, limited_by, f_ladder_min)
    return done(f_best, g_best, s_best, limited_by, f_ladder_min)


def make_mass_scaling_sample_fn(base_cfg, run_bundle, roi, grid_step,
                                velocity_field_map, instance=None):
    """Return sample_fn(factor) -> {vel_label: (Nt, Np), "guard": float} for the
    mass-scaling step.

    Sets `mass_scaling_enabled` + `mass_scaling_factor` (one factor, applied
    to workpiece AND tool by the exporter) on a copy of the
    config, forces the energy history on, runs one Abaqus job, and returns the
    ROI velocity sampled on the fixed grid plus the guard = ratio of
    time-aggregated ALLKE/ALLIE.

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
        cfg.step.mass_scaling_factor = float(factor)
        # The identified factor is applied to the tool as well (ms_tool =
        # ms_eul), so each probe reflects both materials being scaled.
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
