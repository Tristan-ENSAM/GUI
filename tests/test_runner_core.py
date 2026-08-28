# -*- coding: utf-8 -*-
"""
Lot 2c-run self-test — sensitivity runner core. No Abaqus/GUI.

    python -m tests.test_runner_core
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
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import morris_plan as mp
from gui.sensitivity import runner_core as rc
from gui.results.qoi import QoISpec

_failures = 0


def check(cond, msg):
    global _failures
    print("  [%s] %s" % ("ok  " if cond else "FAIL", msg))
    if not cond:
        _failures += 1


def main():
    print("=" * 60)
    print("Lot 2c-run self-test — runner core")
    print("=" * 60)
    cfg = ModelConfig()
    sE = pr.spec_for("euler_material.E")
    sMu = pr.spec_for("interaction.friction_coeff")

    # A solver whose QoI is an exact linear function of the DISPLAYED
    # values: Q = 2*E[GPa] + 100*mu. The "bundle" is just a dict; the QoI
    # spec reads it. This validates config-building, the Y matrix and the
    # analysis without needing a real ResultsBundle.
    def solve_fn(c, i):
        E = pr.get_display(c, sE, "C")
        mu = pr.get_display(c, sMu, "C")
        return {"Q": 2.0 * E + 100.0 * mu}

    qoi = [QoISpec(id="Q", label="Q", unit="-",
                   fn=lambda b, inst, w: b["Q"])]

    # ---- Jacobian ----
    x0E = pr.get_display(cfg, sE, "C")
    x0M = pr.get_display(cfg, sMu, "C")
    plan = jac.build_plan([(sE, x0E, 1.0, False), (sMu, x0M, 0.01, False)],
                          scheme="central")
    res = rc.run_plan(plan, "jacobian", qoi, solve_fn, cfg)
    check(res.Y.shape == (plan.n_runs, 1), "Y shape = (n_runs, n_qoi)")
    check(not res.failures, "no failures")
    rank = rc.jacobian_ranking(res, "Q")
    dE = dict(res.analyses["Q"])["euler_material.E"]["dQdx"]
    dM = dict(res.analyses["Q"])["interaction.friction_coeff"]["dQdx"]
    check(abs(dE - 2.0) < 1e-6 and abs(dM - 100.0) < 1e-6,
          "Jacobian dQ/dE=%.3f (=2), dQ/dmu=%.3f (=100)" % (dE, dM))
    check(rank[0][0] == "interaction.friction_coeff",
          "ranking puts mu first (|100| > |2|)")

    # ---- Morris ----
    planm = mp.build_plan([(sE, x0E * 0.9, x0E * 1.1),
                           (sMu, x0M * 0.9, x0M * 1.1)], N=6, num_levels=4,
                          seed=1)
    resm = rc.run_plan(planm, "morris", qoi, solve_fn, cfg)
    check(resm.Y.shape == (planm.n_runs, 1), "Morris Y shape ok")
    mrank = rc.morris_ranking(resm, "Q")
    check(len(mrank) == 2 and all(np.isfinite(r[1]) for r in mrank),
          "Morris mu* finite for both params")

    # ---- failure handling ----
    def flaky(c, i):
        return None if i % 2 == 1 else solve_fn(c, i)
    resf = rc.run_plan(plan, "jacobian", qoi, flaky, cfg)
    check(len(resf.failures) > 0 and np.isnan(resf.Y[1, 0]),
          "failed runs recorded as NaN")

    # ---- cancellation ----
    seen = {"n": 0}
    def counting(c, i):
        seen["n"] += 1
        return solve_fn(c, i)
    res_cancel = rc.run_plan(plan, "jacobian", qoi, counting, cfg,
                             should_cancel=lambda: seen["n"] >= 2)
    check(seen["n"] <= 3, "cancellation stops early (ran %d/%d)"
          % (seen["n"], plan.n_runs))

    # ---- field QoI (SSD vs base run) ----
    class _Info:
        def __init__(self, fv): self.field_variables = fv

    class _Bundle:
        def __init__(self, e): self.e = e
        def instance_names(self): return ["EULER", "TOOL"]
        def instance(self, n):
            return _Info(["EVF", "V", "TEMP"] if n == "EULER" else ["TEMP"])
        def field(self, inst, var):
            # EVF field = E everywhere (6 entries); independent of mu
            return np.full((2, 3), self.e, float)

    def solve_field(c, i):
        return _Bundle(pr.get_display(c, sE, "C"))

    resf = rc.run_plan(plan, "jacobian", [], solve_field, cfg,
                       field_vars=["EVF"])
    check("EVF [field]" in resf.qoi_ids, "field QoI column added")
    je = resf.analyses["EVF [field]"]["euler_material.E"]["sensitivity"]
    jm = resf.analyses["EVF [field]"]["interaction.friction_coeff"]["sensitivity"]
    check(abs(je - 6.0) < 1e-6 and abs(jm) < 1e-9,
          "field SSD/delta^2: J_EVF(E)=%.3f (=6), J_EVF(mu)=%.3f (=0)"
          % (je, jm))

    # ---- relative field-change column (weighted over nodes & frames) ----
    rel_key = "EVF \u0394% (rel)"
    check(rel_key in resf.qoi_ids, "relative dV%% field column added")
    x0E_disp = pr.get_display(cfg, sE, "C")
    re_ = resf.analyses[rel_key]["euler_material.E"]["sensitivity"]
    rm_ = resf.analyses[rel_key]["interaction.friction_coeff"]["sensitivity"]
    # EVF = E everywhere, +delta run uses E0+1 (GPa) -> rel = 100*1/E0 ;
    # mu does not move the field -> 0. Independent of node/frame count.
    check(abs(re_ - 100.0 / x0E_disp) < 1e-6 and abs(rm_) < 1e-9,
          "relative dV%%: E=%.4f%% (=%.4f), mu=%.4f%% (=0)"
          % (re_, 100.0 / x0E_disp, rm_))

    print("-" * 60)
    if _failures:
        print("RESULT: %d FAILURE(S)" % _failures)
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
