# -*- coding: utf-8 -*-
"""A-posteriori interaction checks of the sized model (paper §5.7, Table 10).

The sizing is sequential (element size by GCI, then each domain dimension
separately), so four interactions are checked once the domain study is done
(report, Part B, T9; decisions D8-a and of 2026-10-02):

1. ``domain_combined`` - combined domain dimensions versus separately
   optimised ones: the retained domain D* is compared with D* grown by the
   study step in the FOUR directions at once, with the study's own metric
   (Eq. 5, 7) and absolute tolerances; passed iff E_max < 1 and both runs
   pass their safeguards. One run (S(D*) is reused from the study).
   The boundary influences add up (triangle inequality): each dimension can
   pass its own criterion while the combined change reaches up to the sum of
   the four. Decision D11-a: eps_q stays per dimension and a failed check
   means the domain study is redone with changed settings.
2. ``mesh_x_domain`` - element size versus final domain: the GCI study is run
   again on D* with the same plan (D8-a), and the element size h* used by the
   domain study must satisfy the GCI selection rule on D*:
   |f_q(h*) - f_q^ref| / |f_q^ref| <= eps_q for every thresholded quantity,
   f_q^ref = Richardson extrapolate when reliable, else the finest mesh value
   (same rule as mesh_gci.run_mesh_gci). h* must belong to the plan.
   Recovery rule (decision of 2026-10-07): on failure, adopt the size the
   GCI recommends ON D* (or, when none is within tolerance, extend the GCI
   plan one level finer), redo the domain study at that size, then the
   checks. One iteration: a second failure is reported as is.
3. ``ms_x_mesh`` - mass-scaling factor versus final mesh and domain: the
   fixed factor f is checked against the analytic window of
   ModelConfig.mass_scaling_bounds evaluated at (h*, D*). No run.

   Since 2026-10-07 the check no longer rejects: f below the lower bound
   (fc*dt < 1e-3) only triggers an Abaqus .sta WARNING, not a rejection,
   and the filter check MEASURES the deviation of the runtime filter on the
   run of check 4. A factor below the bound is reported as a warning and
   check 4 decides. Earlier interpretation: only the LOWER bound (the
   numerical validity of the runtime output filter) decided the check. The
   reverberation upper bound is the criterion behind the domain-diagonal
   ceiling, which the author ruled too restrictive (warning only, decision of
   2026-10-01); the energy upper bound relies on an indicative coefficient
   while R_K is MEASURED on every run. Both upper bounds are reported as
   warnings, not failures.
4. ``ms_at_point`` - mass-scaling factor measured at the final point (h*, D*)
   (decision of 2026-10-07): S(h*, D*, ms*) is compared with
   S(h*, D*, ms_lower), ms_lower being the value before ms* in the ms study
   (default ms*/2), with the study's metric and absolute tolerances; passed
   iff E_max < 1 and both runs pass their safeguards. One run (S(h*, D*,
   ms*) is the retained run of the domain study, reused). With checks 1 and
   2 it re-checks each of the three axes (ms, h, D) at the final point.
   The filter and reverberation safeguards (`ms_guard_fn`) are evaluated on
   the new run; the reused run keeps the safeguards of the domain study.

Pure host-side module; run_bundle is injected.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.domain_independence import (
    AlignmentError, DIMENSIONS, StudyResult, domain_key, e_max,
    errors_between, run_candidate,
)
from gui.sensitivity.ms_independence import align_samples, with_mass_scaling

# Action attached to a failed combined-domain check (decision D11-a).
REDO_DOMAIN_ACTION = (
    "redo the domain study (decision D11-a, eps_q kept per dimension); an "
    "identical rerun returns the same D*, so change the start domain, the "
    "step, n_hold or eps_q")

CHECK_PURPOSES = {
    "ms_x_mesh": "Mass-scaling factor against its analytic filter-ratio "
                 "bound at the final mesh and domain (informative)",
    "mesh_x_domain": "Element size h* still within the GCI tolerance on the "
                     "final domain",
    "domain_combined": "Domain dimensions sized separately remain "
                       "independent when grown together",
    "ms_at_point": "Mass-scaling factor ms* still independent at the final "
                   "mesh and domain (measured)",
}


@dataclass
class CheckResult:
    """Outcome of one interaction check (one row of Table 10)."""
    name: str
    purpose: str
    passed: Optional[bool]                 # None = not evaluable
    e_max: float = float("nan")            # E_max (or max rel/eps for GCI)
    q_crit: Optional[str] = None
    safeguards_ok: Optional[bool] = None
    conclusion: str = ""
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Combined domain
# ---------------------------------------------------------------------------
def combined_domain_dims(dims: DomainDims, delta: float,
                         caps: Optional[Dict[str, float]] = None,
                         elem_size: float = 0.0) -> DomainDims:
    """D* grown by `delta` in the four directions, each clamped to its cap
    (floored to whole elements when elem_size > 0)."""
    caps = caps or {}
    out = {}
    for n in DIMENSIONS:
        v = float(getattr(dims, n)) + float(delta)
        cap = caps.get(n)
        if cap is not None and v > cap:
            v = (math.floor(cap / elem_size + 1e-9) * elem_size
                 if elem_size > 0 else float(cap))
            v = max(v, float(getattr(dims, n)))
        out[n] = v
    return DomainDims(**out)


def combined_domain_check(run_bundle: Callable, base_cfg, study: StudyResult,
                          guard_fn: Optional[Callable] = None,
                          cost_fn: Optional[Callable] = None,
                          emit: Optional[Callable[[dict], None]] = None
                          ) -> CheckResult:
    """Check 1 (see module docstring). Adds its runs to study.runs/cache."""
    s = study.settings
    elem = float(s["elem_size"])
    res = CheckResult("domain_combined", CHECK_PURPOSES["domain_combined"],
                      passed=None)
    d_star = study.final
    d_plus = combined_domain_dims(d_star, int(s["step_elems"]) * elem,
                                  s.get("caps"), elem)
    if domain_key(d_plus, elem) == domain_key(d_star, elem):
        res.conclusion = "not evaluable: every dimension is at its cap"
        return res

    kw = dict(zoi=s["zoi"], grid_step=s["grid_step"], elem_size=elem,
              window=s["window"], evf_threshold=s["evf_threshold"],
              quantities=tuple(s["thresholds"].keys()),
              diagonal_coeff=s["diagonal_coeff"], guard_fn=guard_fn,
              cost_fn=cost_fn, warnings=study.warnings, emit=emit)

    def get(dims):
        key = domain_key(dims, elem)
        if key in study.cache and study.cache[key][1] is not None:
            return study.cache[key]
        rec, sample = run_candidate(run_bundle, base_cfg, dims,
                                    index=len(study.runs), **kw)
        study.runs.append(rec)
        study.cache[key] = (rec, sample)
        if emit is not None:
            emit({"phase": "run", "record": rec, "n_runs": len(study.runs)})
        return rec, sample

    rec_a, s_a = get(d_star)
    rec_b, s_b = get(d_plus)
    res.details = {"D_star": {n: getattr(d_star, n) for n in DIMENSIONS},
                   "D_plus": {n: getattr(d_plus, n) for n in DIMENSIONS},
                   "runs": [rec_a.index, rec_b.index]}
    res.safeguards_ok = bool(rec_a.guards_ok and rec_b.guards_ok)
    if s_a is None or s_b is None:
        res.conclusion = "not evaluable: %s" % (rec_a.error or rec_b.error)
        return res
    errs = errors_between(s_b, s_a, tuple(s["thresholds"].keys()))
    res.e_max, res.q_crit = e_max(errs, s["thresholds"])
    res.details["errors"] = errs
    res.passed = bool(math.isfinite(res.e_max) and res.e_max < 1.0
                      and res.safeguards_ok)
    res.conclusion = ("independent when grown together" if res.passed else
                      "NOT admissible: %s" % (
                          "safeguards failed" if not res.safeguards_ok
                          else "E_max >= 1 (q_crit %s)" % res.q_crit))
    if not res.passed:
        # Decision D11-a (2026-10-02): eps_q stays per dimension; a failed
        # combined check means the domain study must be redone. The study is
        # deterministic, so an identical rerun gives the same D*: something
        # must change (start domain, step, n_hold or eps_q).
        res.details["action"] = REDO_DOMAIN_ACTION
        res.warnings.append(REDO_DOMAIN_ACTION)
    return res


# ---------------------------------------------------------------------------
# 2. Mesh x domain
# ---------------------------------------------------------------------------
def mesh_domain_check(gci_result, h_star: float,
                      tolerances: Dict[str, float],
                      call_records: Optional[list] = None) -> CheckResult:
    """Check 2 from a GCI result obtained ON D* (see module docstring)."""
    res = CheckResult("mesh_x_domain", CHECK_PURPOSES["mesh_x_domain"],
                      passed=None)
    if gci_result is None:
        res.conclusion = "not evaluable: no GCI result on D*"
        return res
    sizes = list(gci_result.sizes)
    match = [h for h in sizes if math.isclose(h, h_star, rel_tol=1e-6)]
    res.details = {"h_star": h_star, "sizes": sizes,
                   "recommended_on_D_star": gci_result.recommended_size,
                   "in_asymptotic_range": gci_result.in_asymptotic_range}
    if call_records is not None:
        res.safeguards_ok = all(r.guards_ok for r in call_records)
    if not match:
        res.conclusion = ("not evaluable: h* = %.6g is not in the GCI plan %r"
                          % (h_star, sizes))
        return res
    h = match[0]
    worst, crit, rel = 0.0, None, {}
    for q, g in gci_result.per_quantity.items():
        tol = tolerances.get(q)
        if tol is None or tol <= 0:
            continue
        fq = gci_result.scalars.get(h, {}).get(q, float("nan"))
        ref = g.f_extrapolated if g.reliable else g.f_fine
        if fq is None or not (math.isfinite(fq) and math.isfinite(ref)) \
                or ref == 0.0:
            r = float("inf")
        else:
            r = abs((fq - ref) / ref) / tol
        rel[q] = r
        if crit is None or r > worst:
            worst, crit = r, q
    res.e_max, res.q_crit = worst, crit
    res.details["rel_over_tol"] = rel
    within = crit is not None and math.isfinite(worst) and worst <= 1.0
    res.passed = bool(within and res.safeguards_ok is not False)
    if not gci_result.in_asymptotic_range:
        res.warnings.append("GCI on D*: meshes not in the asymptotic range")
    res.conclusion = ("h* within the GCI tolerance on D*" if res.passed else
                      "NOT admissible: %s" % (
                          "safeguards failed" if res.safeguards_ok is False
                          else "h* outside the GCI tolerance on D* "
                               "(q_crit %s)" % crit))
    if not res.passed:
        rec = gci_result.recommended_size
        res.details["action"] = (
            ("adopt h = %.6g mm (recommended by the GCI on D*), redo the "
             "domain study at that size, then the checks" % rec)
            if rec is not None else
            "no mesh within tolerance on D*: extend the GCI plan one level "
            "finer, then redo the domain study and the checks")
        res.warnings.append(res.details["action"])
    return res


# ---------------------------------------------------------------------------
# 3. Mass scaling x mesh
# ---------------------------------------------------------------------------
def mass_scaling_window_check(cfg, h_star: float, d_star: DomainDims
                              ) -> CheckResult:
    """Check 3 (see module docstring). `cfg` is not modified."""
    res = CheckResult("ms_x_mesh", CHECK_PURPOSES["ms_x_mesh"], passed=None)
    c = copy.deepcopy(cfg)
    c.elem_size = float(h_star)
    g = c.euler_geometry
    g.h_wp, g.h_void = float(d_star.h_wp), float(d_star.h_void)
    g.l_wp, g.l_void = float(d_star.l_wp), float(d_star.l_void)
    step = c.step
    f = (float(step.mass_scaling_factor)
         if getattr(step, "mass_scaling_enabled", False) else 1.0)
    res.details["factor"] = f
    if not getattr(step, "output_filter_enabled", False):
        res.passed = True
        res.conclusion = ("output filter disabled: no lower bound on the "
                          "factor (f = %.6g)" % f)
        res.warnings.append("without the output filter the ODB velocity "
                            "fields are not anti-aliased")
        return res
    b = c.mass_scaling_bounds(
        float(step.output_filter_cutoff_hz),
        history_cutoff_hz=float(
            getattr(step, "output_filter_cutoff_history_hz", 0.0) or 0.0))
    res.details.update({k: b.get(k) for k in
                        ("ms_min", "ms_freq", "ms_guard", "ms_nyquist",
                         "dt0")})
    if b.get("ms_min") is None:
        res.conclusion = "not evaluable: window not computable"
        return res
    # Informative since 2026-10-07: below the bound Abaqus only warns in the
    # .sta; the measured filter check of ms_at_point decides.
    res.passed = True
    below = bool(f < b["ms_min"])
    res.details["below_filter_ratio_bound"] = below
    if below:
        res.warnings.append("f = %.6g below the filter-ratio bound %.6g "
                            "(fc*dt < 1e-3: Abaqus .sta warning only); the "
                            "filter check of ms_at_point decides"
                            % (f, b["ms_min"]))
    if b.get("ms_nyquist") is not None and f > b["ms_nyquist"]:
        res.warnings.append("f = %.6g above the filter Nyquist bound %.6g: "
                            "Abaqus does not filter at all above fc*dt = 0.5"
                            % (f, b["ms_nyquist"]))
    if b.get("ms_freq") is not None and f > b["ms_freq"]:
        res.warnings.append("f = %.6g above the reverberation bound %.6g "
                            "(warning only, same criterion as the diagonal "
                            "ceiling)" % (f, b["ms_freq"]))
    if b.get("ms_guard") is not None and f > b["ms_guard"]:
        res.warnings.append("f = %.6g above the indicative energy bound %.6g "
                            "(R_K is measured on every run)"
                            % (f, b["ms_guard"]))
    res.conclusion = ("f = %.6g >= filter lower bound %.6g" % (f, b["ms_min"])
                      if not below else
                      "f = %.6g below the filter lower bound %.6g: warning "
                      "only, decided by ms_at_point" % (f, b["ms_min"]))
    return res


# ---------------------------------------------------------------------------
# 4. Mass scaling measured at (h*, D*)
# ---------------------------------------------------------------------------
def _factor(cfg) -> float:
    st = cfg.step
    return (float(st.mass_scaling_factor)
            if getattr(st, "mass_scaling_enabled", False) else 1.0)


def ms_at_point_check(run_bundle: Callable, base_cfg, study: StudyResult,
                      ms_lower: Optional[float] = None,
                      guard_fn: Optional[Callable] = None,
                      cost_fn: Optional[Callable] = None,
                      emit: Optional[Callable[[dict], None]] = None
                      ) -> CheckResult:
    """Check 4 (see module docstring). Adds its run to study.runs."""
    s = study.settings
    elem = float(s["elem_size"])
    res = CheckResult("ms_at_point", CHECK_PURPOSES["ms_at_point"],
                      passed=None)
    ms_star = _factor(base_cfg)
    lower = float(ms_lower) if ms_lower is not None else ms_star / 2.0
    res.details = {"ms_star": ms_star, "ms_lower": lower}
    if ms_star <= 1.0:
        res.passed = True
        res.conclusion = "no mass scaling (ms = 1): nothing to check"
        return res
    if not (1.0 <= lower < ms_star):
        res.conclusion = ("not evaluable: ms_lower = %.6g must be in [1, ms*"
                          " = %.6g)" % (lower, ms_star))
        return res
    d_star = study.final
    key = domain_key(d_star, elem)
    rec_a, s_a = study.cache.get(key, (None, None))
    if rec_a is None or s_a is None:
        res.conclusion = "not evaluable: no usable run at D* in the study"
        return res
    quantities = tuple(s["thresholds"].keys())
    rec_b, s_b = run_candidate(
        run_bundle, with_mass_scaling(base_cfg, lower), d_star,
        index=len(study.runs), zoi=s["zoi"], grid_step=s["grid_step"],
        elem_size=elem, window=s["window"], evf_threshold=s["evf_threshold"],
        quantities=quantities, diagonal_coeff=float("inf"),
        guard_fn=guard_fn, cost_fn=cost_fn, warnings=study.warnings,
        emit=emit)
    study.runs.append(rec_b)
    if emit is not None:
        emit({"phase": "run", "record": rec_b, "n_runs": len(study.runs)})
    res.details["runs"] = [rec_a.index, rec_b.index]
    res.safeguards_ok = bool(rec_a.guards_ok and rec_b.guards_ok)
    if s_b is None:
        res.conclusion = "not evaluable: %s" % (rec_b.error or "no bundle")
        return res
    try:
        b2, a2, info = align_samples(s_b, s_a)
        errs = errors_between(a2, b2, quantities)
    except AlignmentError as exc:
        res.conclusion = "not evaluable: %s" % exc
        return res
    res.details.update(info)
    res.details["errors"] = errs
    res.e_max, res.q_crit = e_max(errs, s["thresholds"])
    res.passed = bool(math.isfinite(res.e_max) and res.e_max < 1.0
                      and res.safeguards_ok)
    res.conclusion = ("ms* = %.6g independent of ms = %.6g at (h*, D*)"
                      % (ms_star, lower) if res.passed else
                      "NOT admissible: %s" % (
                          "safeguards failed" if not res.safeguards_ok
                          else "E_max >= 1 (q_crit %s)" % res.q_crit))
    if not res.passed:
        res.details["action"] = ("redo the ms study at (h*, D*) and, if ms* "
                                 "changes, set it in the Step tab and redo "
                                 "the domain study and the checks (the GCI "
                                 "on D* revalidates h* at the new ms*)")
        res.warnings.append(res.details["action"])
    return res


# ---------------------------------------------------------------------------
# All four
# ---------------------------------------------------------------------------
@dataclass
class ChecksResult:
    checks: List[CheckResult] = field(default_factory=list)
    gci_on_d_star: Optional[object] = None      # MeshGciResult
    gci_calls: list = field(default_factory=list)   # CallRecord per GCI run
    status: str = ""      # "accepted" | "rejected" | "incomplete" | "cancelled"

    @property
    def accepted(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


def run_interaction_checks(run_bundle: Callable, base_cfg, study: StudyResult,
                           h_star: float, gci_plan: Dict[str, object],
                           gci_tolerances: Dict[str, float],
                           guard_fn: Optional[Callable] = None,
                           cost_fn: Optional[Callable] = None,
                           gci_runner_factory: Optional[Callable] = None,
                           ms_lower: Optional[float] = None,
                           ms_guard_fn: Optional[Callable] = None,
                           should_cancel: Optional[Callable[[], bool]] = None,
                           progress_cb: Optional[Callable[[dict], None]] = None
                           ) -> ChecksResult:
    """Run the four checks in cost order: mass-scaling window (no run),
    combined domain (1 run), ms at (h*, D*) (1 run), mesh x domain (one GCI
    plan on D*).

    ms_lower : ms compared with ms* in check 4 (default ms*/2).
    ms_guard_fn : safeguards of the check-4 run (default guard_fn).

    gci_plan : kwargs of mesh_gci.run_mesh_gci other than run_bundle,
        base_cfg, domain_dims, tolerances (finest_elem_size, ratio, n_meshes,
        min_elem_size, field_vars, window, evf_threshold, zoi, grid_step).
    gci_runner_factory(run_bundle) -> RecordingRunner-like wrapper used for
        the GCI runs (records cost and safeguards); identity when None."""
    from gui.sensitivity.mesh_gci import run_mesh_gci

    def cancelled():
        return bool(should_cancel is not None and should_cancel())

    def emit(ev):
        if progress_cb is not None:
            progress_cb(ev)

    out = ChecksResult()
    if study is None or not study.settings:
        out.status = "incomplete"
        return out
    d_star = study.final

    c3 = mass_scaling_window_check(base_cfg, h_star, d_star)
    out.checks.append(c3)
    emit({"phase": "check", "check": c3})
    if cancelled():
        out.status = "cancelled"
        return out

    c1 = combined_domain_check(run_bundle, base_cfg, study, guard_fn, cost_fn,
                               emit)
    out.checks.append(c1)
    emit({"phase": "check", "check": c1})
    if cancelled():
        out.status = "cancelled"
        return out

    c4 = ms_at_point_check(run_bundle, base_cfg, study, ms_lower,
                           ms_guard_fn if ms_guard_fn is not None
                           else guard_fn, cost_fn, emit)
    out.checks.append(c4)
    emit({"phase": "check", "check": c4})
    if cancelled():
        out.status = "cancelled"
        return out

    runner = (gci_runner_factory(run_bundle) if gci_runner_factory is not None
              else run_bundle)
    gci = None
    try:
        gci = run_mesh_gci(run_bundle=runner, base_cfg=copy.deepcopy(base_cfg),
                           domain_dims=d_star, tolerances=gci_tolerances,
                           should_cancel=should_cancel,
                           progress_cb=lambda e: emit(dict(e, phase="gci")),
                           **gci_plan)
    except Exception as exc:
        study.warnings.append("GCI on D*: %s: %s" % (type(exc).__name__, exc))
    out.gci_on_d_star = gci
    out.gci_calls = list(getattr(runner, "records", []))
    c2 = mesh_domain_check(gci, h_star, gci_tolerances,
                           out.gci_calls if out.gci_calls else None)
    out.checks.append(c2)
    emit({"phase": "check", "check": c2})

    if cancelled() or (gci is not None and gci.stopped_by == "cancelled"):
        out.status = "cancelled"
    elif any(c.passed is None for c in out.checks):
        out.status = "incomplete"
    else:
        out.status = "accepted" if out.accepted else "rejected"
    emit({"phase": "checks_done", "result": out})
    return out
