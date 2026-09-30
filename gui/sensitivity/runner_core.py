# -*- coding: utf-8 -*-
"""
Sensitivity runner core — the solver-agnostic pipeline.

Given a plan (Morris or Jacobian), a list of QoI, and a `solve_fn` that
turns one ModelConfig into a results bundle, this:
  1. expands the plan into one ModelConfig per run,
  2. runs each through solve_fn,
  3. reduces every bundle to the scalar QoI,
  4. analyses the QoI matrix (Morris mu*/sigma, or Jacobian sensitivities).

`solve_fn(cfg, run_index) -> bundle | None` is injected, so this module is
fully testable without Abaqus (the GUI wires in a real Abaqus subprocess +
.npz reader; tests inject a mock). NaN-safe: a failed run (solve_fn returns
None, or a QoI raises) yields NaN for that cell and is recorded in
`failures`; the analysis tolerates it.

Field-discrepancy (SSD) sensitivity for the Jacobian is provided
separately by `jacobian_field_sensitivity` in field_metrics, fed with the
kept bundles — see run_plan(keep_bundles=True).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np

from gui.sensitivity import morris_plan as mp
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import field_metrics as fm
from gui.core import units
from gui.core.logging_util import log_swallowed


def _instance_names(bundle):
    """Return the list of instance names, tolerating both the real
    ResultsBundle API (``instance_names`` is a *property* returning a
    list — see gui/results/reader.py) and any test double that exposes it
    as a *method*. Returns [] if neither works."""
    attr = getattr(bundle, "instance_names", None)
    if attr is None:
        return []
    names = attr() if callable(attr) else attr
    try:
        return list(names)
    except TypeError:
        return []


def eulerian_instance(bundle):
    """Name of the Eulerian instance (the one carrying EVF), else the first."""
    try:
        names = _instance_names(bundle)
        for name in names:
            info = bundle.instance(name)
            if "EVF" in getattr(info, "field_variables", []):
                return name
        return names[0] if names else None
    except Exception:
        log_swallowed("resolving the Eulerian instance name")
        return None


def jacobian_field_analysis(plan, bundles, field_vars, metric="ssd",
                            instance=None):
    """Per-parameter field sensitivity for a Jacobian plan, using the kept
    bundles, over the ROI. Follows the plan's FD scheme:

      forward / backward : J_i = metric(F(x0 ± delta_i), F(x0)) per delta
      central            : J_i = metric(F(x0+delta_i), F(x0-delta_i)) per
                           2*delta_i  (both perturbed runs; the base run is
                           not involved)

    J is always >= 0 (a magnitude of how much the field moves); with 'ssd'
    it is divided by the step squared, so its value depends on the
    parameter's unit. Returns
        {var: {param_path: {"sensitivity": J, "rel_pct": dF%,
                            "elasticity": e}}}
    rel_pct  : relative field change for one step, in percent, weighted over
               nodes and frames (central: from (F+ - F-)/2). Independent of
               the metric, but proportional to the step size.
    elasticity: rel_pct / (100 |delta| / |x0|) — relative field change per
               relative parameter change, dimensionless and step-independent
               to first order (x0 in kelvin for temperatures). A magnitude.
    A run the scheme needs that is missing gives NaN (no silent fallback to
    another scheme), as for the scalar QoI."""
    if not bundles or bundles[0] is None:
        return {}
    ref = bundles[0]
    inst = instance or eulerian_instance(ref)
    scheme = getattr(plan, "scheme", "central")
    nan = float("nan")
    out = {}

    def _field(idx, var):
        if idx is None or idx >= len(bundles) or bundles[idx] is None:
            return None
        return bundles[idx].field(inst, var)

    for var in field_vars:
        try:
            base_field = ref.field(inst, var)
        except Exception:
            log_swallowed("reading base field %r for field sensitivity" % var,
                          level=logging.DEBUG)
            continue
        per_param = {}
        for i, spec in enumerate(plan.specs):
            d = plan.deltas[i]
            val = rel = nan
            try:
                if scheme == "central":
                    fp = _field(plan.idx_plus.get(i), var)
                    fmn = _field(plan.idx_minus.get(i), var)
                    if fp is not None and fmn is not None:
                        val = fm.jacobian_field_sensitivity(
                            fmn, fp, delta=2.0 * d, metric=metric)
                        rel = fm.field_rel_change_pct_central(
                            base_field, fp, fmn)
                else:
                    idx = (plan.idx_plus.get(i) if scheme == "forward"
                           else plan.idx_minus.get(i))
                    pert = _field(idx, var)
                    if pert is not None:
                        val = fm.jacobian_field_sensitivity(
                            base_field, pert, delta=d, metric=metric)
                        rel = fm.field_rel_change_pct(base_field, pert)
            except Exception:
                log_swallowed("field sensitivity for %s @ %s"
                              % (var, spec.path), level=logging.DEBUG)
                val = rel = nan
            x0 = abs(jac.abs_base(plan, i)) if hasattr(plan, "base") else 0.0
            step_pct = 100.0 * abs(d) / x0 if x0 else nan
            elast = rel / step_pct if (step_pct and np.isfinite(step_pct)) \
                else nan
            per_param[spec.path] = {"sensitivity": float(val),
                                    "rel_pct": float(rel),
                                    "elasticity": float(elast)}
        out[var] = per_param
    return out


def jacobian_field_maps(plan, bundles, field_vars, instance=None):
    """Per-element, per-frame SIGNED sensitivity MAPS for a Jacobian plan.

    The map counterpart of jacobian_field_analysis: instead of reducing each
    (field var, parameter) to a scalar over the ROI, it keeps the element
    axis, returning a field dF/dparam per element and per frame. The display
    layer reduces it (per-frame slice or time aggregate) and chooses signed
    vs magnitude.

    Returns {var: {param_path: S}} where S is a (n_frames, n_elements) array
    (np.ndarray), or NaN-filled where a run is missing. Uses the plan's FD
    scheme (central uses the +delta and -delta runs, forward/backward use the
    base run). `bundles` are the kept run bundles (bundles[0] = base run)."""
    if not bundles or bundles[0] is None:
        return {}
    ref = bundles[0]
    inst = instance or eulerian_instance(ref)
    scheme = getattr(plan, "scheme", "central")
    out = {}
    for var in field_vars:
        try:
            base_field = ref.field(inst, var)
        except Exception:
            log_swallowed("reading base field %r for field maps" % var,
                          level=logging.DEBUG)
            continue
        per_param = {}
        for i, spec in enumerate(plan.specs):
            ip = plan.idx_plus.get(i)
            im = plan.idx_minus.get(i)

            def _field(idx):
                if idx is None or idx >= len(bundles) or bundles[idx] is None:
                    return None
                try:
                    return bundles[idx].field(inst, var)
                except Exception:
                    log_swallowed("reading field %r for map @ %s"
                                  % (var, spec.path), level=logging.DEBUG)
                    return None

            plus_field = _field(ip)
            minus_field = _field(im)
            try:
                S = fm.elementwise_signed_sensitivity(
                    base_field, plus_field, minus_field,
                    delta=plan.deltas[i], scheme=scheme)
            except Exception:
                log_swallowed("field map for %s @ %s" % (var, spec.path),
                              level=logging.DEBUG)
                S = np.full(np.asarray(base_field, float).shape, np.nan)
            per_param[spec.path] = S
        out[var] = per_param
    return out


@dataclass
class RunResult:
    plan_kind: str                 # "morris" | "jacobian"
    qoi_ids: List[str]
    param_paths: List[str]
    Y: np.ndarray                  # (n_runs, n_qoi), NaN where a run failed
    analyses: dict                 # {qoi_id: analysis dict (method-specific)}
    failures: List[int] = field(default_factory=list)   # failed run indices
    bundles: Optional[list] = None # kept bundles if keep_bundles=True
    # Runs actually launched (solve_fn called). Lower than Y.shape[0] only
    # when the campaign was cancelled: the rows never run are neither
    # successes nor failures.
    n_attempted: int = 0
    cancelled: bool = False

    @property
    def n_ok(self) -> int:
        """Runs launched that produced usable results."""
        return self.n_attempted - len(self.failures)

    @property
    def n_not_run(self) -> int:
        return int(self.Y.shape[0]) - self.n_attempted


def extract_qois(bundle, qoi_specs, warmup_frac: float = 0.0) -> dict:
    """Reduce one bundle to {qoi_id: float}. Any QoI that raises or is
    missing becomes NaN (sensitivity runs routinely have partial data)."""
    out = {}
    for spec in qoi_specs:
        try:
            out[spec.id] = float(spec.fn(bundle, None, warmup_frac))
        except Exception:
            log_swallowed("computing QoI %r" % spec.id, level=logging.DEBUG)
            out[spec.id] = float("nan")
    return out


def run_plan(plan, plan_kind: str, qoi_specs, solve_fn: Callable,
             base_cfg, warmup_frac: float = 0.0,
             progress: Optional[Callable[[int, int], None]] = None,
             should_cancel: Optional[Callable[[], bool]] = None,
             keep_bundles: bool = False,
             field_vars=None, field_metric: str = "ssd") -> RunResult:
    """Run every profile of `plan` and analyse the result.

    solve_fn(cfg, run_index) -> bundle | None
    progress(done, total)          optional UI callback (called before run)
    should_cancel() -> bool        optional cooperative-cancel check
    field_vars                     optional list of Eulerian field variables
                                   (e.g. ["EVF", "V", "TEMP"]) to screen as
                                   field-discrepancy QoI (Jacobian only;
                                   forces keep_bundles).
    """
    if plan_kind not in ("morris", "jacobian"):
        raise ValueError("plan_kind must be 'morris' or 'jacobian'")
    want_fields = bool(field_vars) and plan_kind == "jacobian"
    if want_fields:
        keep_bundles = True
    mod = jac if plan_kind == "jacobian" else mp
    configs = mod.plan_to_configs(base_cfg, plan)
    n = len(configs)

    qoi_ids = [s.id for s in qoi_specs]
    Y = np.full((n, len(qoi_ids)), np.nan, dtype=float)
    failures: List[int] = []
    bundles = [None] * n if keep_bundles else None
    n_attempted = 0
    cancelled = False

    for i, cfg in enumerate(configs):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        if progress is not None:
            progress(i, n)
        n_attempted += 1
        bundle = solve_fn(cfg, i)
        if bundle is None:
            failures.append(i)
            continue
        if keep_bundles:
            bundles[i] = bundle
        q = extract_qois(bundle, qoi_specs, warmup_frac)
        for j, qid in enumerate(qoi_ids):
            Y[i, j] = q[qid]
        # a run that yielded only NaN scalar QoI is still kept if we need
        # its field for field QoI; flag as failure only when nothing useful
        if len(qoi_ids) and np.all(np.isnan(Y[i, :])) and not want_fields:
            failures.append(i)
    # A cancel that lands during the LAST run leaves the loop without a break.
    if not cancelled and should_cancel is not None and should_cancel():
        cancelled = True
    if progress is not None:
        progress(n, n)

    analyses = {}
    for j, qid in enumerate(qoi_ids):
        y = Y[:, j]
        try:
            if plan_kind == "jacobian":
                analyses[qid] = jac.analyze(plan, y,
                                            q_offset=_qoi_kelvin_offset(
                                                qoi_specs[j]))
            else:
                # Only complete trajectories are analysed: a failed or
                # never-run row spoils its trajectory (see analyze_complete).
                res_dict, info = mp.analyze_complete(plan, y)
                if res_dict is None:
                    analyses[qid] = dict(
                        info, error="too few complete trajectories "
                        "(%d/%d, need 2)" % (info["n_used"],
                                             info["n_trajectories"]))
                else:
                    res_dict = dict(res_dict)
                    res_dict.update(info)
                    analyses[qid] = res_dict
        except Exception as e:                          # pragma: no cover
            analyses[qid] = {"error": str(e)}

    qoi_ids_all = list(qoi_ids)
    if want_fields and bundles is not None:
        try:
            fa = jacobian_field_analysis(plan, bundles, list(field_vars),
                                         metric=field_metric)
            for var, per in fa.items():
                key = "%s [field]" % var
                analyses[key] = per
                qoi_ids_all.append(key)
                # Parallel, intuitive column: relative field change in %
                # (weighted over nodes and frames). Stored under
                # "sensitivity" so the table/CSV render it like any column.
                rel_key = "%s \u0394%% (rel)" % var      # e.g. "V Δ% (rel)"
                analyses[rel_key] = {
                    p: {"sensitivity": d.get("rel_pct", float("nan"))}
                    for p, d in per.items()}
                qoi_ids_all.append(rel_key)
        except Exception as e:                          # pragma: no cover
            analyses["field [error]"] = {"error": str(e)}

    return RunResult(plan_kind=plan_kind, qoi_ids=qoi_ids_all,
                     param_paths=list(plan.param_paths), Y=Y,
                     analyses=analyses, failures=failures, bundles=bundles,
                     n_attempted=n_attempted, cancelled=cancelled)


def _qoi_kelvin_offset(spec) -> float:
    """273.15 for a QoI stored in °C (its elasticity must use kelvin)."""
    return units.KELVIN_OFFSET if getattr(spec, "unit", "") == "°C" else 0.0


def jacobian_ranking(result: RunResult, qoi_id: str, key: str = "sensitivity"):
    """Return [(param_path, value), ...] sorted by |value| descending, for a
    Jacobian result and one QoI. `key` picks the quantity: "sensitivity"
    (as configured per row) or "elasticity" (dimensionless); parameters
    without that key are left out."""
    a = result.analyses.get(qoi_id, {})
    rows = [(p, d[key]) for p, d in a.items()
            if isinstance(d, dict) and key in d]
    rows.sort(key=lambda t: (np.isnan(t[1]), -abs(t[1])))
    return rows


def morris_ranking(result: RunResult, qoi_id: str):
    """Return [(param_path, mu_star, sigma), ...] sorted by mu_star desc."""
    a = result.analyses.get(qoi_id, {})
    names = a.get("names", [])
    mu_star = a.get("mu_star", [])
    sigma = a.get("sigma", [])
    rows = list(zip(names, mu_star, sigma))
    rows.sort(key=lambda t: (np.isnan(t[1]), -t[1]))
    return rows
