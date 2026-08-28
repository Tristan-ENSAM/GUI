# -*- coding: utf-8 -*-
"""
Lot 2c self-test — Jacobian (finite-difference) sensitivity. No Abaqus/GUI.

    python -m tests.test_lot2c
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from gui.core.model_config import ModelConfig
from gui.sensitivity import param_registry as pr
from gui.sensitivity import jacobian_plan as jp

_failures = 0


def check(cond, msg):
    global _failures
    print("  [%s] %s" % ("ok  " if cond else "FAIL", msg))
    if not cond:
        _failures += 1


def main():
    print("=" * 60)
    print("Lot 2c self-test — Jacobian sensitivity")
    print("=" * 60)
    cfg = ModelConfig()
    sE = pr.spec_for("euler_material.E")             # GPa
    sMu = pr.spec_for("interaction.friction_coeff")   # —
    x0E = pr.get_display(cfg, sE, "C")
    x0M = pr.get_display(cfg, sMu, "C")

    paths = ["euler_material.E", "interaction.friction_coeff"]

    def Q_of(row):
        d = dict(zip(paths, row))
        return 2.0 * d["euler_material.E"] + 100.0 * d["interaction.friction_coeff"]

    for scheme, exp_runs in (("forward", 3), ("backward", 3), ("central", 5)):
        sel = [(sE, x0E, 1.0, False), (sMu, x0M, 0.01, False)]
        plan = jp.build_plan(sel, scheme=scheme)
        check(plan.n_runs == exp_runs == jp.n_runs(2, scheme),
              "%s: %d runs" % (scheme, plan.n_runs))
        Y = np.array([Q_of(r) for r in plan.X])
        res = jp.analyze(plan, Y)
        dE = res["euler_material.E"]["dQdx"]
        dM = res["interaction.friction_coeff"]["dQdx"]
        check(abs(dE - 2.0) < 1e-6 and abs(dM - 100.0) < 1e-6,
              "%s: dQ/dE=%.4f (=2), dQ/dmu=%.4f (=100)" % (scheme, dE, dM))

    # Normalised sensitivity (elasticity) = dQ/dE * E0/Q0
    sel = [(sE, x0E, 1.0, True), (sMu, x0M, 0.01, False)]
    plan = jp.build_plan(sel, scheme="central")
    Y = np.array([Q_of(r) for r in plan.X])
    res = jp.analyze(plan, Y)
    Q0 = Q_of(plan.X[0])
    exp = 2.0 * x0E / Q0
    got = res["euler_material.E"]["sensitivity"]
    check(abs(got - exp) < 1e-6,
          "normalize: elasticity S_E=%.5f (=%.5f)" % (got, exp))
    check(res["interaction.friction_coeff"]["normalized"] is False,
          "per-parameter normalize flag respected")

    # profiles round-trip displayed -> stored
    configs = jp.plan_to_configs(cfg, plan)
    check(len(configs) == plan.n_runs, "one ModelConfig per run")
    check(abs(configs[plan.idx_plus[0]].euler_material["E"]
              - (x0E + 1.0) * 1000.0) < 1e-3,
          "+delta profile: E stored = (E0+1 GPa) in MPa")

    print("-" * 60)
    if _failures:
        print("RESULT: %d FAILURE(S)" % _failures)
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
