# -*- coding: utf-8 -*-
"""Exports of the sizing studies for the paper (report, Part B, T11).

Written into the study folder created by the Optimization tab, so every
table and figure of the paper can be rebuilt from files, without copying
numbers by hand:

==================  =======================================================
file                content (paper item)
==================  =======================================================
runs.csv            every simulation with its parameters, cost, errors,
                    safeguards and decision (Appendix B)
comparisons.csv     every successive comparison of the domain study: E_q,
                    E_q/eps_q, E_max, q_crit, decay ratios, tail bounds,
                    Pareto flag (Fig. 11, Fig. 13)
dimensions.csv      initial / selected / normalised dimension (Table 9)
gci.csv             per-quantity GCI outcome (Table 8 as rewritten, P7)
gci_meshes.csv      per-mesh scalars and cost of a GCI study (Table 8)
checks.csv          interaction checks (Table 10)
summary.json        selected model, Eq. (24) normalisation, Eq. (22)-(23)
                    cost gain and speed-up, statuses
==================  =======================================================

Eq. (22)-(23): the reference cost C_initial is the cost of the STARTING
domain of the study, i.e. its first run (ZOI + margin, decision D2-a)
(decision D12 of 2026-10-02, revised the same day). summary.json gives it
under "cost_eq22_23_paper" with C = C_CPU (Eq. 11, solver time), and keeps,
for information, the same ratios by Eulerian element count and against the
most expensive domain simulated.

Consequence (fact, by construction): the study only GROWS the domain from
its start at a fixed element size, so C_opt >= C_initial up to run-to-run
timing noise; with this reference G_C <= 0 and R_C <= 1 (Eq. 22 then
measures a cost increase, not a reduction).

Pareto flag (Fig. 13): each comparison is a configuration = its candidate
run p_(j-1) (the value the comparison judges) with cost C_CPU of that run and
E_max of the comparison; it is non-dominated when no other configuration has
both a lower-or-equal cost and a lower-or-equal E_max, one strictly.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from gui.sensitivity.domain_independence import ALL_QUANTITIES, DIMENSIONS

COST_FIELDS = ("n_elem_euler", "n_elem_euler_extracted", "n_inc",
               "dt_stable_first", "dt_stable_min", "t_wall_solver_s",
               "t_wall_host_s", "n_cpu", "c_cpu_s")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _num(v):
    """CSV/JSON-friendly number: None for missing or non-finite."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def _cost(rec) -> Dict[str, object]:
    c = getattr(rec, "cost", None)
    return {k: _num(getattr(c, k, None)) if c is not None else None
            for k in COST_FIELDS}


def _guards(guards: Dict[str, tuple]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in ("outputs", "R_K", "R_HG"):
        v, ok = guards.get(name, (None, None))
        out[name] = _num(v)
        out[name + "_ok"] = ok
    return out


def write_csv(path, rows: Sequence[Dict[str, object]],
              columns: Optional[Sequence[str]] = None) -> Path:
    """Write `rows` with a fixed column order (None/NaN -> empty cell)."""
    path = Path(path)
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns),
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None or (
                isinstance(r.get(k), float) and not math.isfinite(r[k]))
                else r.get(k)) for k in columns})
    return path


def pareto_flags(points: Sequence[tuple]) -> List[Optional[bool]]:
    """Non-dominated flag for (cost, e_max) points; None when undefined."""
    flags: List[Optional[bool]] = []
    clean = [(c, e) if (c is not None and e is not None and math.isfinite(c)
                        and math.isfinite(e)) else None for c, e in points]
    for i, p in enumerate(clean):
        if p is None:
            flags.append(None)
            continue
        dominated = any(
            q is not None and j != i and q[0] <= p[0] and q[1] <= p[1]
            and (q[0] < p[0] or q[1] < p[1]) for j, q in enumerate(clean))
        flags.append(not dominated)
    return flags


# ---------------------------------------------------------------------------
# Domain study rows
# ---------------------------------------------------------------------------
def comparison_rows(study, quantities: Sequence[str] = ALL_QUANTITIES
                    ) -> List[Dict[str, object]]:
    thr = study.settings.get("thresholds", {})
    runs = {r.index: r for r in study.runs}
    rows = []
    for name, dres in study.per_dimension.items():
        for c in dres.comparisons:
            row = {"dimension": c.dimension, "j": c.j,
                   "value_from_mm": c.value_from, "value_to_mm": c.value_to,
                   "run_from": c.run_from, "run_to": c.run_to,
                   "E_max": _num(c.e_max), "q_crit": c.q_crit,
                   "guards_ok": c.guards_ok, "success": c.success,
                   "mode": c.mode}
            for q in quantities:
                row["E_" + q] = _num(c.errors.get(q))
                eps = thr.get(q)
                e = c.errors.get(q)
                row["E_%s/eps" % q] = (_num(e / eps) if eps and e is not None
                                       else None)
                d = c.decay.get(q)
                row["rho_" + q] = _num(d.rho) if d is not None else None
                row["decay_ok_" + q] = d.accepted if d is not None else None
                row["R_" + q] = _num(c.bound.get(q)) if c.bound else None
            rf = runs.get(c.run_from)
            row["C_CPU_from_s"] = _cost(rf)["c_cpu_s"] if rf else None
            row["n_elem_from"] = _cost(rf)["n_elem_euler"] if rf else None
            rows.append(row)
    flags = pareto_flags([(r["C_CPU_from_s"], r["E_max"]) for r in rows])
    for r, f in zip(rows, flags):
        r["pareto_non_dominated"] = f
    return rows


def run_rows(study, ms_factor: Optional[float] = None,
             quantities: Sequence[str] = ALL_QUANTITIES,
             case: str = "baseline", step_of: Optional[Dict[int, str]] = None
             ) -> List[Dict[str, object]]:
    """One row per simulation (Appendix B). The errors of a run are those of
    the comparison where it is the NEW run (run_to); the first run of the
    study has none."""
    to_comp = {}
    retained = {}                    # run index -> dimensions it settles
    for dres in study.per_dimension.values():
        for c in dres.comparisons:
            to_comp.setdefault(c.run_to, c)
        ri = retained_run_index(dres)
        if ri is not None:
            retained.setdefault(ri, []).append(dres.name)
    elem = study.settings.get("elem_size")
    rows = []
    for r in study.runs:
        c = to_comp.get(r.index)
        row = {"case": case,
               "step": (step_of or {}).get(r.index, "domain"),
               "run": r.index,
               "parameter": c.dimension if c else "initial",
               "value_mm": c.value_to if c else None,
               "mesh_size_mm": elem, "mass_scaling_factor": ms_factor}
        row.update({n + "_mm": r.dims.get(n) for n in DIMENSIONS})
        row.update(_cost(r))
        for q in quantities:
            row["E_" + q] = _num(c.errors.get(q)) if c else None
        row["E_max"] = _num(c.e_max) if c else None
        row["q_crit"] = c.q_crit if c else None
        row.update(_guards(r.guards))
        row["safeguards_ok"] = r.guards_ok
        row["diagonal_over_h"] = _num(r.diagonal_ratio)
        row["diagonal_warning"] = r.diagonal_warning
        row["decision"] = (("independent" if c.success else "not independent")
                           if c else "start")
        row["retained_for"] = " ".join(retained.get(r.index, []))
        row["job_ok"] = r.job_ok
        row["error"] = r.error
        rows.append(row)
    return rows


def retained_run_index(dres) -> Optional[int]:
    """RunRecord.index of the run at the retained value of a dimension.

    values[k] is judged by comparison k+1 (comparisons[k].run_from); the
    last value is only the NEW run of the last comparison."""
    k = dres.retained_index
    comps = dres.comparisons
    if k is None or k < 0 or not comps:
        return None
    if k < len(comps):
        return comps[k].run_from
    if k == len(comps):
        return comps[-1].run_to
    return None


def dimension_rows(study, t1: Optional[float]) -> List[Dict[str, object]]:
    rows = []
    for name, d in study.per_dimension.items():
        rows.append({
            "dimension": name, "initial_mm": d.initial, "selected_mm":
            d.retained, "initial_over_t1": _num(d.initial / t1) if t1 else None,
            "selected_over_t1": _num(d.retained / t1) if t1 else None,
            "status": d.status, "criterion": _num(d.criterion),
            "q_crit": d.q_crit, "n_comparisons": len(d.comparisons)})
    return rows


# ---------------------------------------------------------------------------
# GCI rows
# ---------------------------------------------------------------------------
def gci_rows(gci_result, tolerances: Optional[Dict[str, float]] = None
             ) -> List[Dict[str, object]]:
    if gci_result is None:
        return []
    sizes = list(gci_result.sizes)
    rows = []
    for q, g in gci_result.per_quantity.items():
        f = [gci_result.scalars.get(h, {}).get(q) for h in sizes[:3]]
        rows.append({"quantity": q, "f1_finest": _num(f[0]),
                     "f2": _num(f[1]) if len(f) > 1 else None,
                     "f3": _num(f[2]) if len(f) > 2 else None,
                     "p": _num(g.p), "f_extrapolated": _num(g.f_extrapolated),
                     "GCI_fine": _num(g.gci_fine),
                     "GCI_coarse": _num(g.gci_coarse),
                     "asymptotic_ratio": _num(g.asymptotic_ratio),
                     "monotonic": g.monotonic, "reliable": g.reliable,
                     "tolerance": _num((tolerances or {}).get(q))})
    return rows


def gci_mesh_rows(gci_result, calls: Optional[Sequence] = None
                  ) -> List[Dict[str, object]]:
    """Per mesh: scalars f_q, cost and safeguards (Table 8). `calls` are the
    RecordingRunner records, matched by element size."""
    if gci_result is None:
        return []
    by_h = {}
    for c in calls or []:
        by_h.setdefault(round(c.elem_size, 12), c)
    rows = []
    for h in gci_result.sizes:
        row = {"h_mm": h,
               "recommended": (gci_result.recommended_size is not None and
                               math.isclose(h, gci_result.recommended_size,
                                            rel_tol=1e-9))}
        for q, v in gci_result.scalars.get(h, {}).items():
            row["f_" + q] = _num(v)
        c = by_h.get(round(h, 12))
        row.update(_cost(c) if c is not None else
                   {k: None for k in COST_FIELDS})
        if c is not None:
            row.update(_guards(c.guards))
            row["safeguards_ok"] = c.guards_ok
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Checks rows
# ---------------------------------------------------------------------------
def check_rows(checks_result) -> List[Dict[str, object]]:
    if checks_result is None:
        return []
    return [{"check": c.name, "purpose": c.purpose, "E_max": _num(c.e_max),
             "q_crit": c.q_crit, "safeguards_ok": c.safeguards_ok,
             "passed": c.passed, "conclusion": c.conclusion,
             "warnings": " | ".join(c.warnings)}
            for c in checks_result.checks]


# ---------------------------------------------------------------------------
# Summary (Eq. 22-24)
# ---------------------------------------------------------------------------
def _gain(c_opt, c_ref) -> Dict[str, Optional[float]]:
    if c_opt is None or c_ref is None or not c_ref or not c_opt:
        return {"C_opt": _num(c_opt), "C_ref": _num(c_ref), "G_C": None,
                "R_C": None}
    return {"C_opt": _num(c_opt), "C_ref": _num(c_ref),
            "G_C": _num(1.0 - c_opt / c_ref), "R_C": _num(c_ref / c_opt)}


def summary(study, t1: Optional[float], h_star: Optional[float],
            ms_factor: Optional[float], checks_result=None) -> dict:
    from gui.sensitivity.domain_independence import domain_key
    elem = study.settings.get("elem_size")
    final = {n: getattr(study.final, n) for n in DIMENSIONS}
    initial = {n: getattr(study.initial, n) for n in DIMENSIONS}
    rec_opt = None
    key = domain_key(study.final, elem) if elem else None
    if key in study.cache:
        rec_opt = study.cache[key][0]
    rec_init = study.runs[0] if study.runs else None
    costs = [(r, _cost(r)) for r in study.runs]
    over = max(costs, key=lambda rc: (rc[1]["c_cpu_s"] or -1.0,
                                      rc[1]["n_elem_euler"] or -1))[0] \
        if costs else None

    def gains(field_name):
        def val(r):
            return _cost(r)[field_name] if r is not None else None
        return {"vs_initial_domain": _gain(val(rec_opt), val(rec_init)),
                "vs_oversized_domain": _gain(val(rec_opt), val(over))}

    norm = None
    if t1:
        norm = {"h_over_t1": _num(h_star / t1) if h_star else None}
        norm.update({n + "_over_t1": _num(v / t1) for n, v in final.items()})
    out = {
        "status": study.status,
        "n_runs": study.n_runs,
        "t1_mm": _num(t1), "h_star_mm": _num(h_star),
        "mass_scaling_factor": _num(ms_factor),
        "initial_domain_mm": initial, "selected_domain_mm": final,
        "normalised_eq24": norm,
        "dimensions": {n: {"status": d.status, "q_crit": d.q_crit,
                           "criterion": _num(d.criterion)}
                       for n, d in study.per_dimension.items()},
        "cost_eq22_23_paper": dict(
            gains("c_cpu_s")["vs_initial_domain"],
            reference="starting domain of the study, ZOI + margin "
                      "(decision D12)",
            reference_run=rec_init.index if rec_init else None,
            cost="C_CPU = N_CPU * t_wall,solver (Eq. 11)"),
        "cost_eq22_23": {"by_C_CPU": gains("c_cpu_s"),
                         "by_n_elem_euler": gains("n_elem_euler"),
                         "selected_run": rec_opt.index if rec_opt else None,
                         "initial_run": rec_init.index if rec_init else None,
                         "oversized_run": over.index if over else None},
        "settings": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in study.settings.items()},
        "warnings": list(study.warnings),
    }
    if checks_result is not None:
        out["interaction_checks"] = {
            "status": checks_result.status,
            "accepted": checks_result.accepted,
            "checks": check_rows(checks_result)}
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_domain_exports(folder, study, t1: Optional[float],
                         h_star: Optional[float],
                         ms_factor: Optional[float],
                         checks_result=None, case: str = "baseline"
                         ) -> List[Path]:
    """runs.csv, comparisons.csv, dimensions.csv, summary.json (+ checks.csv
    when a checks result is given). Returns the written paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    step_of = {}
    if checks_result is not None:
        for c in checks_result.checks:
            for i in c.details.get("runs", []) or []:
                step_of.setdefault(i, "check_" + c.name)
    paths = [
        write_csv(folder / "runs.csv",
                  run_rows(study, ms_factor, case=case, step_of=step_of)),
        write_csv(folder / "comparisons.csv", comparison_rows(study)),
        write_csv(folder / "dimensions.csv", dimension_rows(study, t1)),
    ]
    if checks_result is not None:
        paths.append(write_csv(folder / "checks.csv",
                               check_rows(checks_result)))
        if checks_result.gci_on_d_star is not None:
            paths.append(write_csv(folder / "checks_gci_on_D_star.csv",
                                   gci_rows(checks_result.gci_on_d_star)))
            paths.append(write_csv(folder / "checks_gci_meshes_on_D_star.csv",
                                   gci_mesh_rows(checks_result.gci_on_d_star,
                                                 checks_result.gci_calls)))
    p = folder / "summary.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(summary(study, t1, h_star, ms_factor, checks_result), fh,
                  indent=2, default=str)
    paths.append(p)
    return paths


def write_gci_exports(folder, gci_result, calls=None,
                      tolerances: Optional[Dict[str, float]] = None
                      ) -> List[Path]:
    """gci.csv and gci_meshes.csv for one GCI study."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return [write_csv(folder / "gci.csv", gci_rows(gci_result, tolerances)),
            write_csv(folder / "gci_meshes.csv",
                      gci_mesh_rows(gci_result, calls))]
