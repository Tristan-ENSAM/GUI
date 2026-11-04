# -*- coding: utf-8 -*-
"""Parameter sweeps of the Parameter plans tab: how the QoI evolve when the
parameters vary over a range (decision 2026-11-04).

Two designs
-----------
* ``sweep`` (one at a time): each selected parameter takes its levels in
  turn, the others stay at their reference value. The reference point is
  run once and shared (a level equal to the reference is not run twice):
  ``n_runs = 1 + sum(n_i - [ref among the levels of i])``. Shows the shape
  of QoI = f(p) per parameter; blind to interactions.
* ``grid`` (full factorial): every combination of the levels,
  ``n_runs = prod(n_i)``, plus the reference point when it is not one of
  the combinations. Shows interactions; meant for 2 or 3 parameters.

Levels (per parameter, in displayed units)
------------------------------------------
``levels(lo, hi, n, spacing)``: n values from lo to hi, evenly spaced
("linear") or in geometric progression ("geometric", lo and hi > 0, e.g. a
mass-scaling factor from 1 to 25600); or a list typed by hand
(`parse_values`).

Analysis (per QoI, `analyze`)
-----------------------------
Per parameter, on the runs where only that parameter differs from the
reference (sweep) or on the main effect (grid: mean of the QoI over the
runs at each level of the parameter): the points (x, q), the slope of the
least-squares line dq/dx, its R^2, Spearman's rank coefficient (monotony;
needs 3 points) and, when eps_q is given, max |q - q_ref| / eps_q over
those points (q_ref = the reference run).

All values are in DISPLAYED units; `plan_to_configs` converts them with
param_registry.apply_display, as the Jacobian and Morris plans do.
Pure module (no Qt).
"""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from gui.sensitivity import param_registry as pr

DESIGNS = ("sweep", "grid")
SPACINGS = ("linear", "geometric")
DEFAULT_LEVELS = 5          # a choice (example of the 2026-10-23 report)


def levels(lo: float, hi: float, n: int, spacing: str = "linear"
           ) -> np.ndarray:
    """n levels from lo to hi (both included)."""
    n = int(n)
    if n < 2:
        raise ValueError("At least 2 levels are needed.")
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
        raise ValueError("Min and Max must be two different numbers.")
    if spacing == "linear":
        return np.linspace(lo, hi, n)
    if spacing == "geometric":
        if lo <= 0 or hi <= 0:
            raise ValueError("Geometric levels need Min and Max > 0.")
        return np.geomspace(lo, hi, n)
    raise ValueError("spacing must be one of %s" % (SPACINGS,))


def parse_values(text: str) -> Optional[np.ndarray]:
    """Levels typed by hand: numbers separated by ';', spaces or commas
    (a comma is a separator, '0,5' is not 0.5: use a dot). None when the
    text is empty."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [p for p in text.replace(";", " ").replace(",", " ").split()
             if p]
    try:
        vals = np.array([float(p) for p in parts], dtype=float)
    except ValueError:
        raise ValueError("Values must be numbers separated by ';' or "
                         "spaces: %r" % text)
    if vals.size < 2:
        raise ValueError("Give at least 2 values: %r" % text)
    if not np.all(np.isfinite(vals)):
        raise ValueError("Values must be finite numbers: %r" % text)
    return vals


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


@dataclass
class SweepPlan:
    specs: list                 # list[ParamSpec]
    base: list                  # reference value per spec (displayed)
    level_values: list          # np.ndarray of levels per spec (displayed)
    design: str                 # "sweep" | "grid"
    X: np.ndarray               # (n_runs, k), displayed units
    run_kind: list              # "base", "<i>" (sweep: param i varied), "grid"
    base_row: int = 0           # row of the reference point
    temp_unit: str = "C"
    unit_system: object = None
    spacing: list = field(default_factory=list)   # per spec, for the record
    # grid only: False when the reference point is not a combination of
    # the levels (it is then run in addition, as row 0)
    base_on_grid: bool = True

    @property
    def param_paths(self):
        return [s.path for s in self.specs]

    @property
    def k(self) -> int:
        return len(self.specs)

    @property
    def n_runs(self) -> int:
        return int(self.X.shape[0])


def n_runs(level_values: Sequence[np.ndarray], base: Sequence[float],
           design: str) -> int:
    """Runs of a plan with these levels and reference values."""
    if design == "sweep":
        n = 1
        for lv, b in zip(level_values, base):
            n += sum(0 if _same(float(v), float(b)) else 1 for v in lv)
        return n
    if design == "grid":
        n = int(np.prod([len(lv) for lv in level_values])) if level_values \
            else 0
        on_grid = all(any(_same(float(v), float(b)) for v in lv)
                      for lv, b in zip(level_values, base))
        return n + (0 if on_grid else 1)
    raise ValueError("design must be one of %s" % (DESIGNS,))


def build_plan(selected, design: str = "sweep", temp_unit: str = "C",
               unit_system=None, spacing=None) -> SweepPlan:
    """selected: list of (ParamSpec, reference value, levels array), all in
    DISPLAYED units. `spacing`: per spec, how its levels were made (kept
    for the study record only)."""
    if design not in DESIGNS:
        raise ValueError("design must be one of %s" % (DESIGNS,))
    if not selected:
        raise ValueError("Select at least one parameter to vary.")
    specs, base, lvls = [], [], []
    for spec, ref, lv in selected:
        lv = np.asarray(lv, dtype=float).ravel()
        if lv.size < 2:
            raise ValueError("Parameter %r needs at least 2 levels."
                             % spec.path)
        specs.append(spec)
        base.append(float(ref))
        lvls.append(lv)
    base_arr = np.asarray(base, dtype=float)
    rows, kinds = [base_arr.copy()], ["base"]
    base_row = 0
    on_grid = True
    if design == "sweep":
        for i, lv in enumerate(lvls):
            for v in lv:
                if _same(float(v), base[i]):
                    continue                 # the shared reference run
                r = base_arr.copy()
                r[i] = float(v)
                rows.append(r)
                kinds.append(str(i))
    else:
        rows, kinds = [], []
        for combo in itertools.product(*lvls):
            r = np.asarray(combo, dtype=float)
            if all(_same(a, b) for a, b in zip(r, base_arr)):
                base_row = len(rows)
                kinds.append("base")
            else:
                kinds.append("grid")
            rows.append(r)
        on_grid = "base" in kinds
        if not on_grid:
            rows.insert(0, base_arr.copy())
            kinds.insert(0, "base")
            base_row = 0
    if unit_system is None:
        unit_system = pr.current_system(temp_unit)
    return SweepPlan(specs=specs, base=base, level_values=lvls,
                     design=design, base_on_grid=on_grid, X=np.vstack(rows), run_kind=kinds,
                     base_row=base_row, temp_unit=temp_unit,
                     unit_system=unit_system,
                     spacing=list(spacing or ["linear"] * len(specs)))


def plan_to_configs(base_cfg, plan: SweepPlan):
    """One deep-copied ModelConfig per run row, with the varied parameters
    set to their (displayed -> stored) values."""
    configs = []
    for row in plan.X:
        cfg = copy.deepcopy(base_cfg)
        for spec, value in zip(plan.specs, row):
            pr.apply_display(cfg, spec, float(value), plan.temp_unit,
                             system=plan.unit_system)
        configs.append(cfg)
    return configs


def profile_table(plan: SweepPlan):
    rows = []
    for i, row in enumerate(plan.X):
        d = {"run": i + 1, "kind": plan.run_kind[i]}
        for spec, value in zip(plan.specs, row):
            d[spec.path] = float(value)
        rows.append(d)
    return rows


def points(plan: SweepPlan, y, i: int):
    """(x, q) of parameter i: the reference run and the runs where only
    parameter i differs from it (sweep), or the main effect (grid: mean of
    q over the finite runs at each level). Sorted by x; NaN q dropped."""
    y = np.asarray(y, dtype=float)
    if plan.design == "sweep":
        rows = [plan.base_row] + [r for r, kd in enumerate(plan.run_kind)
                                  if kd == str(i)]
        x = plan.X[rows, i]
        q = y[rows]
    else:
        lv = plan.level_values[i]
        x, q = [], []
        first = 0 if plan.base_on_grid else 1     # row 0 off the grid
        grid_rows = list(range(first, plan.n_runs))
        for v in lv:
            sel = [r for r in grid_rows if _same(plan.X[r, i], float(v))]
            vals = y[sel]
            vals = vals[np.isfinite(vals)]
            x.append(float(v))
            q.append(float(vals.mean()) if vals.size else float("nan"))
        x, q = np.asarray(x), np.asarray(q)
    ok = np.isfinite(q)
    x, q = np.asarray(x, dtype=float)[ok], np.asarray(q, dtype=float)[ok]
    order = np.argsort(x, kind="stable")
    return x[order], q[order]


def _spearman(x: np.ndarray, q: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = _ranks(x)
    rq = _ranks(q)
    if np.std(rx) == 0 or np.std(rq) == 0:
        return float("nan")
    return float(np.corrcoef(rx, rq)[0, 1])


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), as Spearman needs."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(a.size, dtype=float)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def analyze(plan: SweepPlan, y, eps: Optional[float] = None) -> dict:
    """{param path: {"x", "q", "slope", "r2", "spearman", "n",
    "max_dev_eps"}} for one QoI column `y` (NaN = failed run)."""
    y = np.asarray(y, dtype=float)
    q_ref = float(y[plan.base_row]) if y.size > plan.base_row else \
        float("nan")
    out = {}
    for i, spec in enumerate(plan.specs):
        x, q = points(plan, y, i)
        d = {"x": x.tolist(), "q": q.tolist(), "n": int(x.size),
             "slope": float("nan"), "r2": float("nan"),
             "spearman": _spearman(x, q), "max_dev_eps": float("nan"),
             "q_ref": q_ref}
        if x.size >= 2 and np.ptp(x) > 0:
            a, b = np.polyfit(x, q, 1)
            d["slope"] = float(a)
            fit = a * x + b
            ss_tot = float(np.sum((q - q.mean()) ** 2))
            d["r2"] = (1.0 - float(np.sum((q - fit) ** 2)) / ss_tot
                       if ss_tot > 0 else float("nan"))
        if eps and eps > 0 and math.isfinite(q_ref) and q.size:
            d["max_dev_eps"] = float(np.max(np.abs(q - q_ref)) / eps)
        out[spec.path] = d
    return out
