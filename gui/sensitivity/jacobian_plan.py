# -*- coding: utf-8 -*-
"""
Local (Jacobian) sensitivity by finite differences.

Complements the global Morris screening with a local gradient at the
reference point: for each selected parameter, perturb it by a user-chosen
step `delta` and estimate dQoI/dparam.

Schemes
-------
  forward  :  dQ/dx ≈ (Q(x0+δ) − Q(x0)) / δ            runs: k+1
  backward :  dQ/dx ≈ (Q(x0)   − Q(x0−δ)) / δ           runs: k+1
  central  :  dQ/dx ≈ (Q(x0+δ) − Q(x0−δ)) / (2δ)        runs: 2k+1

Run layout (row 0 is always the base point x0, so Q0 is available for
normalisation regardless of scheme):
  forward  : [base, +0, +1, ..., +(k-1)]
  backward : [base, -0, -1, ..., -(k-1)]
  central  : [base, +0, -0, +1, -1, ...]

Normalisation (per parameter, optional)
---------------------------------------
If `normalize` is set, the reported sensitivity is the dimensionless
elasticity  S_i = (dQ/dx_i) · (x_i0 / Q0)  — the relative change in the
QoI per relative change in the parameter. Otherwise the raw derivative
(QoI-unit per displayed parameter-unit) is reported. The elasticity is
always computed as well (key "elasticity"), so parameters of different
units can be compared whatever the per-row choice.

A ratio x/x0 is only meaningful on a scale whose zero is physical, which
°C is not (the same state is 20 °C or 293.15 K, and the ratio would depend
on the displayed unit). Temperatures therefore get NO elasticity: for a
temperature parameter, and for every parameter when the QoI itself is a
temperature (the caller passes `q_is_temp`), "elasticity" is NaN and a
row with Norm ticked reports the raw dQ/dx instead, flagged with
"raw_fallback" = True.

Central scheme with a failed run
--------------------------------
If one of the two perturbed runs is missing, the derivative falls back to
the one-sided difference with the base run (forward if +delta survived,
backward if -delta did); "scheme_used" records which one was applied, so
the display can flag it. Less accurate (O(delta) instead of O(delta^2)).

All bounds/steps are in DISPLAYED engineering units; profiles convert to
stored units via param_registry.apply_display.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np

from gui.sensitivity import param_registry as pr

SCHEMES = ("forward", "backward", "central")


@dataclass
class JacobianPlan:
    specs: list            # list[ParamSpec]
    base: list             # base values (displayed units), per spec
    deltas: list           # FD step (displayed units), per spec
    normalize: list        # bool per spec
    scheme: str
    X: np.ndarray          # (n_runs, k), displayed units
    run_kind: list         # e.g. "base", "+0", "-0", ...
    idx_plus: dict         # spec index -> row with +delta
    idx_minus: dict        # spec index -> row with -delta
    temp_unit: str = "C"
    # UnitSystem the displayed values are expressed in (snapshot taken at
    # generation). None = the active system at conversion time (legacy).
    unit_system: object = None

    @property
    def param_paths(self):
        return [s.path for s in self.specs]

    @property
    def k(self):
        return len(self.specs)

    @property
    def n_runs(self):
        return int(self.X.shape[0])


def n_runs(k: int, scheme: str) -> int:
    return int(2 * k + 1) if scheme == "central" else int(k + 1)


def build_plan(selected, scheme="central", temp_unit="C", unit_system=None):
    """selected: list of (ParamSpec, base_value, delta, normalize_bool),
    all in DISPLAYED units. Returns a JacobianPlan. `unit_system` is the
    UnitSystem those displayed values are expressed in (default: a snapshot
    of the active one)."""
    if scheme not in SCHEMES:
        raise ValueError("scheme must be one of %s" % (SCHEMES,))
    if not selected:
        raise ValueError("Select at least one parameter to vary.")

    specs, base, deltas, norm = [], [], [], []
    for spec, x0, d, nrm in selected:
        d = float(d)
        if d == 0.0:
            raise ValueError("Parameter %r: delta must be non-zero." % spec.path)
        specs.append(spec); base.append(float(x0))
        deltas.append(d); norm.append(bool(nrm))

    k = len(specs)
    base_arr = np.asarray(base, dtype=float)
    rows = [base_arr.copy()]
    run_kind = ["base"]
    idx_plus, idx_minus = {}, {}

    for i in range(k):
        if scheme in ("forward", "central"):
            r = base_arr.copy(); r[i] += deltas[i]
            idx_plus[i] = len(rows); rows.append(r); run_kind.append("+%d" % i)
        if scheme in ("backward", "central"):
            r = base_arr.copy(); r[i] -= deltas[i]
            idx_minus[i] = len(rows); rows.append(r); run_kind.append("-%d" % i)

    X = np.vstack(rows)
    if unit_system is None:
        unit_system = pr.current_system(temp_unit)
    return JacobianPlan(specs=specs, base=base, deltas=deltas, normalize=norm,
                        scheme=scheme, X=X, run_kind=run_kind,
                        idx_plus=idx_plus, idx_minus=idx_minus,
                        temp_unit=temp_unit, unit_system=unit_system)


def plan_to_configs(base_cfg, plan: JacobianPlan):
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


def profile_table(plan: JacobianPlan):
    rows = []
    for i, row in enumerate(plan.X):
        d = {"run": i + 1, "kind": plan.run_kind[i]}
        for spec, value in zip(plan.specs, row):
            d[spec.path] = float(value)
        rows.append(d)
    return rows


def central_fallback(plan: JacobianPlan, i: int, ok):
    """Which difference to apply for parameter i given which runs are usable
    (`ok(row_index) -> bool`): the plan's own scheme when possible, else --
    central only -- the one-sided difference with the surviving perturbed
    run. Returns "central" | "forward" | "backward" | None (nothing
    computable)."""
    ip, im = plan.idx_plus.get(i), plan.idx_minus.get(i)
    have_p = ip is not None and ok(ip)
    have_m = im is not None and ok(im)
    have_0 = ok(0)
    if plan.scheme == "central":
        if have_p and have_m:
            return "central"
        if have_p and have_0:
            return "forward"
        if have_m and have_0:
            return "backward"
        return None
    if plan.scheme == "forward":
        return "forward" if (have_p and have_0) else None
    return "backward" if (have_m and have_0) else None


def analyze(plan: JacobianPlan, Y, q_is_temp: bool = False):
    """Compute the local sensitivity per parameter for one QoI vector Y
    (length == n_runs). `q_is_temp`: the QoI is a temperature, so no
    elasticity (see module docstring). Returns:
        {param_path: {"sensitivity": s, "dQdx": raw, "elasticity": e,
                      "normalized": bool, "raw_fallback": bool,
                      "scheme_used": str, "x0": x0, "Q0": Q0}}.
    NaN-safe: a missing run yields NaN, unless the central scheme can fall
    back to a one-sided difference ("scheme_used" says so)."""
    Y = np.asarray(Y, dtype=float)
    Q0 = Y[0]
    out = {}
    ok = lambda r: bool(np.isfinite(Y[r]))
    for i, spec in enumerate(plan.specs):
        d = plan.deltas[i]
        used = central_fallback(plan, i, ok)
        if used == "central":
            dQdx = (Y[plan.idx_plus[i]] - Y[plan.idx_minus[i]]) / (2.0 * d)
        elif used == "forward":
            dQdx = (Y[plan.idx_plus[i]] - Q0) / d
        elif used == "backward":
            dQdx = (Q0 - Y[plan.idx_minus[i]]) / d
        else:
            dQdx = float("nan")
        x0 = plan.base[i]
        no_elast = spec.is_temp or q_is_temp
        elast = (dQdx * (x0 / Q0) if (not no_elast and Q0 != 0.0)
                 else float("nan"))
        raw_fallback = bool(plan.normalize[i] and no_elast)
        sens = elast if (plan.normalize[i] and not no_elast) else dQdx
        out[spec.path] = {"sensitivity": float(sens), "dQdx": float(dQdx),
                          "elasticity": float(elast),
                          "normalized": bool(plan.normalize[i]
                                             and not no_elast),
                          "raw_fallback": raw_fallback,
                          "scheme_used": used or plan.scheme,
                          "x0": float(x0), "Q0": float(Q0)}
    return out
