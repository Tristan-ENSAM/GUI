# -*- coding: utf-8 -*-
"""Mass-scaling factor by an independence study (paper step 0, §4.6, §5.3).

Purpose
-------
Find the LARGEST mass-scaling factor ms whose effect on the response measured
in the ZOI stays below the absolute tolerances eps_q. No physical bound
limits ms for this model (decision of 2026-10-07): the inertia ratios stay
far below their limit up to ms = 4000, the domain reverberation estimate
f = c / (2 L sqrt(ms)) cannot be derived (choice of L and of the wave), and
the Abaqus filter ratio fc*dt < 1e-3 is only a .sta warning. "Largest
admissible" can therefore only be MEASURED, which is what this study does.

Algorithm
---------
The mesh and the domain are held fixed (inputs of this study). The factors
ms_0 < ms_1 < ... < ms_n are run in increasing order and each run S(ms_k) is
compared, depending on `reference`, either with the first run S(ms_0)
("first": ms_0 is the reference, e.g. ms_0 = 1, the model without mass
scaling) or with the previous run S(ms_(k-1)) ("previous"), by the same
metrics as the domain study (Eq. 5 for the fields, Eq. 7 for the forces, E_max = max_q E_q/eps_q,
gui.sensitivity.domain_independence). Comparison k succeeds when E_max < 1
AND both runs pass every safeguard. The sequence stops at the first failed
comparison and the retained factor is the last ms_k reached by an unbroken
chain of successes:

Both modes use the same stop rule:

* comparison k fails, k >= 2 -> ms* = ms_(k-1)        status "converged"
* every comparison succeeds  -> ms* = ms_n            status "upper_end"
  (the largest TESTED value; a larger one was not tried)
* comparison 1 fails         -> ms* = None            status "below_range"
  (independence not shown at ms_1: the sequence must start lower, or,
  in "first" mode, ms_1 must be closer to the reference)
* cancelled                  -> ms* as reached so far status "cancelled"

Time alignment (why the domain study's exact rule is relaxed here)
-----------------------------------------------------------------
Abaqus/Explicit writes a field frame at the end of the increment that
reaches the requested output time, and the stable increment scales with
sqrt(ms). Two runs with different ms therefore have the SAME number of
frames but frame times shifted by up to about one increment, and their
history (written every increment) has different sample counts. The exact
time match of the domain study (same ms, same increment) would refuse every
comparison. Here (`align_samples`):

* field frames are paired by index; the pairing is accepted when every
  offset is at most half the median frame interval (so each frame pairs
  with its nearest neighbour), otherwise the comparison is refused;
* forces of the earlier run are linearly interpolated onto the history
  times of the later run, over the window part both runs cover.

Safeguards
----------
Injected `guard_fn(bundle) -> {name: (value, ok)}`, as in the domain study.
The Optimization tab combines the run safeguards (outputs, R_K, R_HG over T)
with the filter check and the reverberation check of gui.core.filter_check
(see `filter_guards`). Only the job and the outputs block a run; R_K, R_HG,
the filter check and the reverberation are reported only (decision of
2026-10-13, gui.sensitivity.run_record.REPORTED_GUARDS): their effect, if
any, shows in the quantities compared, and their values explain it.

Pure host-side module (CPython 3.x): `run_bundle(cfg)` is injected and must
return a ResultsBundle-like object or None.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from gui.core.domain_sizing import DomainDims
import numpy as np

from gui.sensitivity.domain_independence import (
    ALL_QUANTITIES, METRICS, AlignmentError, RunRecord, ZoiSample, e_max,
    errors_between, errors_by, mean_errors, run_candidate)

# Default sequence (decision of 2026-10-07): factor 2 between values, around
# the ms = 1000 used so far.
DEFAULT_MS_VALUES = (250.0, 500.0, 1000.0, 2000.0, 4000.0)

# What each run is compared with: the first value of the sequence (the
# reference run) or the previous value.
MS_REFERENCES = ("first", "previous")

# How the factors are chosen (2026-10-20): "list" runs the given values in
# order; "bisection" brackets ms* from the reference by multiplying by a
# factor, then bisects (geometric midpoint) until ms_hi / ms_lo <= the stop
# ratio. Default stop ratio 1.0201 = 1.01^2: with a run time t ~ ms^-1/2,
# t(ms_lo) / t(ms_hi) <= 1.01, i.e. ms_hi would save at most 1 % (Tristan,
# 2026-10-20). From a x10 bracket: ceil(log2(ln 10 / ln 1.0201)) = 7
# bisection runs.
MS_SEARCHES = ("list", "bisection")
DEFAULT_BRACKET_FACTOR = 10.0
DEFAULT_STOP_RATIO = 1.0201
DEFAULT_MS_MAX = 1.0e7
DEFAULT_MAX_RUNS = 30


def parse_ms_values(text: str) -> Tuple[float, ...]:
    """'250, 500 1000;2000' -> (250.0, 500.0, 1000.0, 2000.0).

    Raises ValueError unless there are at least two values, all >= 1 and
    strictly increasing."""
    raw = str(text).replace(";", " ").replace(",", " ").split()
    try:
        vals = tuple(float(v) for v in raw)
    except ValueError:
        raise ValueError("the ms values must be numbers separated by commas "
                         "or spaces")
    if len(vals) < 2:
        raise ValueError("at least two ms values are needed for a comparison")
    if any(not math.isfinite(v) or v < 1.0 for v in vals):
        raise ValueError("every ms value must be >= 1")
    if any(b <= a for a, b in zip(vals[:-1], vals[1:])):
        raise ValueError("the ms values must be strictly increasing")
    return vals


def filter_guards(res: Optional[dict]) -> Dict[str, Tuple[Optional[float],
                                                          bool]]:
    """Safeguards from gui.core.filter_check.check_bundle's result.

    filter : max relative deviation of the Abaqus-filtered forces from the
        offline Butterworth, over the filters checked; ok = res["passed"].
    reverb : e_rev of the force band (max over RF1, RF2); ok = its verdict.
    A check that did not run (None result, error, verdict None) is NOT
    evaluable and counts as a failure, as every safeguard of the studies."""
    if not res:
        return {"filter": (None, False), "reverb": (None, False)}
    devs = [float(r["rel_max_dev"]) for r in res.get("filters", {}).values()
            if isinstance(r, dict) and r.get("rel_max_dev") is not None]
    fval = max(devs) if devs else None
    fok = bool(res.get("passed") is True and not res.get("error"))
    rev = res.get("reverberation") or {}
    rval = rev.get("e_rev")
    rok = bool(rev.get("passed") is True and not rev.get("error"))
    return {"filter": (None if fval is None else float(fval), fok),
            "reverb": (None if rval is None else float(rval), rok)}


def align_samples(sa: ZoiSample, sb: ZoiSample
                  ) -> Tuple[ZoiSample, ZoiSample, Dict[str, float]]:
    """Bring two runs of different ms onto common sample times.

    Returns (a', b', info) where a' and b' share field_times and
    force_times, so `errors_between` applies unchanged. info gives the
    largest frame offset and its ratio to the median frame interval.
    Raises AlignmentError when the frames cannot be paired one to one."""
    ta = np.asarray(sa.field_times, dtype=float)
    tb = np.asarray(sb.field_times, dtype=float)
    info: Dict[str, float] = {}
    a2 = ZoiSample(fields=dict(sa.fields), field_times=tb.copy())
    b2 = ZoiSample(fields=dict(sb.fields), field_times=tb.copy())
    if sa.fields or sb.fields:
        if ta.shape != tb.shape:
            raise AlignmentError(
                "Field-frame counts differ in the window (%d vs %d): the "
                "comparison is refused." % (ta.size, tb.size))
        off = float(np.max(np.abs(ta - tb))) if ta.size else 0.0
        step = float(np.median(np.diff(tb))) if tb.size > 1 else float("inf")
        info = {"frame_offset_s": off,
                "frame_offset_over_interval": off / step if step > 0
                else float("inf")}
        if off > 0.5 * step:
            raise AlignmentError(
                "Field frames are offset by %.3g s, more than half the frame "
                "interval (%.3g s): the comparison is refused." % (off, step))
    fa = np.asarray(sa.force_times, dtype=float)
    fb = np.asarray(sb.force_times, dtype=float)
    common = [q for q in sa.forces if q in sb.forces]
    if common and fa.size >= 2 and fb.size:
        keep = (fb >= fa[0]) & (fb <= fa[-1])
        a2.force_times = fb[keep]
        b2.force_times = fb[keep]
        for q in common:
            a2.forces[q] = np.interp(fb[keep], fa,
                                     np.asarray(sa.forces[q], dtype=float))
            b2.forces[q] = np.asarray(sb.forces[q], dtype=float)[keep]
    return a2, b2, info


def pair_errors(s_ref: ZoiSample, s_cur: ZoiSample,
                quantities: Sequence[str], metric: str = "pointwise"
                ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(E_q of run `s_cur` against `s_ref`, alignment info). "mean" compares
    one scalar per run (no alignment: each run keeps its own sample times);
    "pointwise" aligns the two runs first (`align_samples`)."""
    if metric == "mean":
        return mean_errors(s_cur, s_ref, quantities), {}
    a2, b2, info = align_samples(s_ref, s_cur)
    return errors_by(b2, a2, quantities, metric), info


@dataclass
class MsComparison:
    """Comparison k: S(ms_k) against S(ms_0) ("first" reference) or
    S(ms_(k-1)) ("previous"); ms_from is the one compared with."""
    k: int
    ms_from: float
    ms_to: float
    errors: Dict[str, float]
    e_max: float
    q_crit: Optional[str]
    guards_ok: bool                         # both runs passed the safeguards
    success: bool                           # E_max < 1 and guards_ok
    run_from: int = -1
    run_to: int = -1
    frame_offset_over_interval: float = float("nan")


@dataclass
class MsStudyResult:
    ms_values: List[float]
    retained: Optional[float] = None
    status: str = ""     # converged | upper_end | below_range | cancelled
    runs: List[RunRecord] = field(default_factory=list)
    run_ms: List[float] = field(default_factory=list)   # ms of runs[i]
    comparisons: List[MsComparison] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    settings: Dict[str, object] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.runs)


def failure_cause(result: MsStudyResult) -> Optional[str]:
    """Why the last comparison failed: "run" (a run of the pair did not
    finish or misses outputs, so the factor was not judged), "qoi"
    (E_max >= 1), or None when it did not fail."""
    if not result.comparisons or result.comparisons[-1].success:
        return None
    return "run" if not result.comparisons[-1].guards_ok else "qoi"


def with_mass_scaling(cfg, ms: float):
    """Deep copy of `cfg` with mass scaling enabled at factor `ms`."""
    out = copy.deepcopy(cfg)
    out.step.mass_scaling_enabled = True
    out.step.mass_scaling_factor = float(ms)
    return out


def run_ms_independence(
        run_bundle: Callable, base_cfg, zoi: Sequence[float],
        domain_dims: DomainDims, grid_step: float, elem_size: float,
        thresholds: Dict[str, float],
        ms_values: Sequence[float] = DEFAULT_MS_VALUES,
        window: Tuple[float, float] = (0.3, 1.0),
        evf_threshold: float = 0.5,
        guard_fn: Optional[Callable] = None,
        cost_fn: Optional[Callable] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        progress_cb: Optional[Callable[[dict], None]] = None,
        reference: str = "previous", search: str = "list",
        factor: float = DEFAULT_BRACKET_FACTOR,
        stop_ratio: float = DEFAULT_STOP_RATIO,
        ms_max: float = DEFAULT_MS_MAX,
        max_runs: int = DEFAULT_MAX_RUNS,
        metric: str = "pointwise") -> MsStudyResult:
    """Run the ms sequence and retain the largest independent factor.

    run_bundle(cfg) -> ResultsBundle | None. Each config passed is a deep
        copy of `base_cfg` with elem_size, the domain and the mass scaling
        set; `base_cfg` itself is never modified.
    thresholds : absolute tolerances eps_q per quantity label (same as the
        domain study). Only thresholded quantities enter E_max.
    cost_fn(bundle_or_None, dims, host_wall_s) -> CostRecord (optional).
    reference : "first" compares every run with the run of ms_values[0];
        "previous" compares each run with the one before it.
    search : "list" runs `ms_values` in order (above). "bisection" runs
        ms_values[0] (the reference), then brackets ms* by multiplying by
        `factor` and narrows it by bisection (see `MS_SEARCHES`); every run
        is then compared with the reference whatever `reference` says, and
        only ms_values[0] is used.
    metric : deviation between two runs, "mean" or "pointwise"
        (domain_independence.METRICS)."""
    if search not in MS_SEARCHES:
        raise ValueError("search must be one of %s" % (MS_SEARCHES,))
    ms_values = [float(v) for v in ms_values]
    if search == "bisection":
        if not ms_values:
            raise ValueError("the reference ms is needed")
        if not (factor > 1.0 and stop_ratio > 1.0 and max_runs >= 2
                and ms_max > ms_values[0]):
            raise ValueError("bisection needs factor > 1, stop ratio > 1, "
                             "at least 2 runs and a largest ms above the "
                             "reference")
        ms_values = ms_values[:1]
        reference = "first"
    elif len(ms_values) < 2:
        raise ValueError("at least two ms values are needed")
    if any(b <= a for a, b in zip(ms_values[:-1], ms_values[1:])):
        raise ValueError("the ms values must be strictly increasing")
    if any(v < 1.0 for v in ms_values):
        raise ValueError("every ms value must be >= 1")
    if reference not in MS_REFERENCES:
        raise ValueError("reference must be one of %s" % (MS_REFERENCES,))
    if metric not in METRICS:
        raise ValueError("metric must be one of %s" % (METRICS,))
    if elem_size <= 0 or grid_step <= 0:
        raise ValueError("elem_size and grid_step must be > 0")
    thr = {q: float(v) for q, v in thresholds.items()
           if v is not None and float(v) > 0 and q in ALL_QUANTITIES}
    if not thr:
        raise ValueError("at least one positive threshold is required")
    quantities = tuple(thr.keys())
    elem = float(elem_size)

    def cancelled() -> bool:
        return bool(should_cancel is not None and should_cancel())

    def emit(ev: dict) -> None:
        if progress_cb is not None:
            progress_cb(ev)

    result = MsStudyResult(ms_values=list(ms_values))
    result.settings = {
        "zoi": tuple(float(v) for v in zoi), "grid_step": float(grid_step),
        "elem_size": elem, "thresholds": dict(thr), "window": tuple(window),
        "evf_threshold": float(evf_threshold), "ms_values": list(ms_values),
        "reference": reference, "search": search, "metric": metric,
        "domain_dims": {"h_wp": float(domain_dims.h_wp),
                        "h_void": float(domain_dims.h_void),
                        "l_wp": float(domain_dims.l_wp),
                        "l_void": float(domain_dims.l_void)}}
    cfg0 = copy.deepcopy(base_cfg)
    cfg0.elem_size = elem

    def simulate(ms: float) -> Tuple[RunRecord, Optional[ZoiSample]]:
        rec, sample = run_candidate(
            run_bundle, with_mass_scaling(cfg0, ms), domain_dims,
            index=len(result.runs), zoi=zoi, grid_step=grid_step,
            elem_size=elem, window=window, evf_threshold=evf_threshold,
            quantities=quantities, diagonal_coeff=float("inf"),
            guard_fn=guard_fn, cost_fn=cost_fn, warnings=result.warnings,
            emit=emit)
        result.runs.append(rec)
        result.run_ms.append(float(ms))
        emit({"phase": "run", "record": rec, "ms": float(ms),
              "n_runs": len(result.runs)})
        return rec, sample

    if cancelled():
        result.status = "cancelled"
        emit({"phase": "done", "result": result, "n_runs": 0})
        return result
    prev = simulate(ms_values[0])
    if search == "bisection":
        result.settings.update({"factor": float(factor),
                                "stop_ratio": float(stop_ratio),
                                "ms_max": float(ms_max),
                                "max_runs": int(max_runs)})
        return _bisection(result, prev, simulate, quantities, thr, factor,
                          stop_ratio, ms_max, max_runs, cancelled, emit)
    i_prev = 0
    status = "upper_end"
    for k in range(1, len(ms_values)):
        if cancelled():
            status = "cancelled"
            break
        cur = simulate(ms_values[k])
        if cancelled():
            # A cancel during this run: its failure says nothing about ms.
            status = "cancelled"
            break
        (rec_a, s_a), (rec_b, s_b) = prev, cur
        info: Dict[str, float] = {}
        if s_a is None or s_b is None:
            errs = {q: float("nan") for q in quantities}
        else:
            try:
                errs, info = pair_errors(s_a, s_b, quantities, metric)
            except AlignmentError as exc:
                errs = {q: float("nan") for q in quantities}
                result.warnings.append("ms comparison %d: %s" % (k, exc))
        em, qc = e_max(errs, thr)
        pair_ok = rec_a.guards_ok and rec_b.guards_ok
        success = bool(math.isfinite(em) and em < 1.0 and pair_ok)
        comp = MsComparison(k=k, ms_from=ms_values[i_prev],
                            ms_to=ms_values[k], errors=errs, e_max=em,
                            q_crit=qc, guards_ok=pair_ok, success=success,
                            run_from=rec_a.index, run_to=rec_b.index,
                            frame_offset_over_interval=float(
                                info.get("frame_offset_over_interval",
                                         float("nan"))))
        result.comparisons.append(comp)
        emit({"phase": "comparison", "comparison": comp,
              "n_runs": len(result.runs)})
        if not success:
            status = "converged" if k >= 2 else "below_range"
            break
        result.retained = ms_values[k]
        if reference == "previous":
            prev, i_prev = cur, k
    result.status = status
    emit({"phase": "done", "result": result, "n_runs": len(result.runs)})
    return result


def _compare(result: MsStudyResult, k: int, ref, cur, ms_ref: float,
             ms: float, quantities, thr) -> MsComparison:
    """Comparison k of run `cur` (factor ms) with run `ref` (factor ms_ref),
    recorded in `result` (same rules as the list search)."""
    (rec_a, s_a), (rec_b, s_b) = ref, cur
    info: Dict[str, float] = {}
    if s_a is None or s_b is None:
        errs = {q: float("nan") for q in quantities}
    else:
        try:
            errs, info = pair_errors(s_a, s_b, quantities,
                                     result.settings.get("metric",
                                                         "pointwise"))
        except AlignmentError as exc:
            errs = {q: float("nan") for q in quantities}
            result.warnings.append("ms comparison %d: %s" % (k, exc))
    em, qc = e_max(errs, thr)
    pair_ok = rec_a.guards_ok and rec_b.guards_ok
    comp = MsComparison(
        k=k, ms_from=ms_ref, ms_to=ms, errors=errs, e_max=em, q_crit=qc,
        guards_ok=pair_ok,
        success=bool(math.isfinite(em) and em < 1.0 and pair_ok),
        run_from=rec_a.index, run_to=rec_b.index,
        frame_offset_over_interval=float(info.get(
            "frame_offset_over_interval", float("nan"))))
    result.comparisons.append(comp)
    return comp


def _round_ms(v: float) -> float:
    """4 significant digits: readable factors, far finer than any stop
    ratio worth using (relative step 1e-3 at worst)."""
    return float("%.4g" % v)


def monotonicity_warnings(comparisons: Sequence[MsComparison],
                          tol: float = 0.0) -> List[str]:
    """E_max is expected to grow with ms; the search relies on it. Every
    tested factor whose E_max is LOWER, by more than `tol`, than the E_max of
    a smaller tested factor is reported (the search does not prove the
    monotonicity: it only checks it at the tested points)."""
    pts = sorted((c.ms_to, c.e_max) for c in comparisons
                 if math.isfinite(c.e_max))
    out = []
    worst_ms, worst = None, -float("inf")
    for ms, em in pts:
        if worst_ms is not None and em < worst - tol:
            out.append("E_max is not increasing with ms: %.3g at ms = %g "
                       "but %.3g at ms = %g (smaller)" % (em, ms, worst,
                                                          worst_ms))
        if em > worst:
            worst_ms, worst = ms, em
    return out


def _bisection(result, ref, simulate, quantities, thr, factor, stop_ratio,
               ms_max, max_runs, cancelled, emit) -> MsStudyResult:
    """Bracketing then bisection (search "bisection").

    1. bracket: ms = ms_ref * factor^j, j = 1, 2, ... (capped at ms_max)
       until the first comparison that fails; ms_lo = the last factor that
       passed (ms_ref itself when the first one fails), ms_hi = the failed
       one;
    2. bisection at the geometric midpoint sqrt(ms_lo * ms_hi) (the
       increment dt grows as sqrt(ms)) until ms_hi / ms_lo <= stop_ratio:
       a pass moves ms_lo up, a failure moves ms_hi down;
    3. ms* = ms_lo, the largest tested factor below which every tested
       factor passed.

    Statuses: "converged" (ms_lo > ms_ref), "below_range" (no factor above
    the reference passed down to the stop ratio), "upper_end" (ms_max
    passed: not bracketed), "run_failed" (a comparison could not be judged:
    a run did not finish or misses outputs; ms_lo is kept as reached),
    "cap" (max_runs reached), "cancelled"."""
    ms_ref = result.run_ms[0]
    lo, hi = ms_ref, None
    status = None
    k = 0

    def step(ms):
        nonlocal k
        if len(result.runs) >= max_runs:
            return "cap"
        if cancelled():
            return "cancelled"
        result.ms_values.append(ms)
        cur = simulate(ms)
        if cancelled():
            return "cancelled"
        k += 1
        comp = _compare(result, k, ref, cur, ms_ref, ms, quantities, thr)
        emit({"phase": "comparison", "comparison": comp,
              "n_runs": len(result.runs)})
        if not comp.success and not comp.guards_ok:
            return "run_failed"
        return "pass" if comp.success else "fail"

    ms = ms_ref
    while status is None and hi is None:          # 1. bracket
        ms = min(_round_ms(ms * factor), float(ms_max))
        out = step(ms)
        if out == "pass":
            lo = ms
            if ms >= ms_max:
                status = "upper_end"
        elif out == "fail":
            hi = ms
        else:
            status = out
    while status is None and hi / lo > stop_ratio:   # 2. bisection
        mid = _round_ms(math.sqrt(lo * hi))
        if not lo < mid < hi:
            break
        out = step(mid)
        if out == "pass":
            lo = mid
        elif out == "fail":
            hi = mid
        else:
            status = out
    if status is None:
        status = "converged" if lo > ms_ref else "below_range"
    result.retained = lo if lo > ms_ref else None
    result.status = status
    result.settings["ms_hi"] = hi
    result.warnings.extend(monotonicity_warnings(result.comparisons))
    emit({"phase": "done", "result": result, "n_runs": len(result.runs)})
    return result
