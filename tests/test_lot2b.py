# -*- coding: utf-8 -*-
"""
Lot 2b self-test — Morris sampling engine (needs SALib, no Abaqus/GUI).

    python -m tests.test_lot2b      (from the repo root)

Validates: plan shape (N*(k+1) runs), displayed->stored conversion in the
generated ModelConfig profiles, and that Morris analysis ranks an
influential parameter above a non-influential one.
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
from gui.sensitivity import morris_plan as mp

_failures = 0


def check(cond, msg):
    global _failures
    print("  [%s] %s" % ("ok  " if cond else "FAIL", msg))
    if not cond:
        _failures += 1


def main():
    print("=" * 64)
    print("Lot 2b self-test — Morris sampling (SALib)")
    print("=" * 64)

    try:
        import importlib.metadata as _md
        print("SALib version:", _md.version("SALib"))
    except Exception as e:
        print("  [FAIL] SALib not importable:", e)
        print("  Install it in the venv:  pip install SALib")
        return 1

    cfg = ModelConfig()

    # Pick 3 well-understood parameters and bounds (displayed units).
    sE  = pr.spec_for("euler_material.E")          # GPa
    sCS = pr.spec_for("bcs.cutting_speed")          # m/min
    sMu = pr.spec_for("interaction.friction_coeff") # —
    selected = [
        (sE,  100.0, 150.0),     # 100-150 GPa
        (sCS, 30.0, 120.0),      # 30-120 m/min
        (sMu, 0.1, 0.5),         # friction 0.1-0.5
    ]

    print("== build_plan ==")
    N, k = 8, 3
    plan = mp.build_plan(selected, N=N, num_levels=4, seed=42)
    check(plan.k == k, "k == %d parameters" % k)
    check(plan.n_runs == mp.n_runs(k, N) == N * (k + 1),
          "n_runs == N*(k+1) == %d (got %d)" % (N * (k + 1), plan.n_runs))
    check(plan.X.shape == (N * (k + 1), k),
          "X shape == %s" % str((N * (k + 1), k)))
    # bounds respected
    lo = plan.X.min(axis=0); hi = plan.X.max(axis=0)
    check(lo[0] >= 100.0 - 1e-9 and hi[0] <= 150.0 + 1e-9,
          "E samples within [100,150] GPa (got [%.1f, %.1f])" % (lo[0], hi[0]))

    print("== plan_to_configs (displayed -> stored) ==")
    configs = mp.plan_to_configs(cfg, plan)
    check(len(configs) == plan.n_runs, "one ModelConfig per run")
    # Check the first profile: E stored should equal sampled_GPa * 1000 (MPa)
    e_gpa = plan.X[0, 0]
    e_stored = configs[0].euler_material["E"]
    check(abs(e_stored - e_gpa * 1000.0) < 1e-3,
          "profile 0: E %.2f GPa -> %.0f MPa stored" % (e_gpa, e_stored))
    # cutting speed: m/min -> mm/s (factor 1000/60)
    cs_mmin = plan.X[0, 1]
    cs_stored = configs[0].bcs.cutting_speed
    check(abs(cs_stored - cs_mmin * (1000.0 / 60.0)) < 1e-6,
          "profile 0: %.1f m/min -> %.2f mm/s stored" % (cs_mmin, cs_stored))
    # untouched parameter stays at base value
    check(configs[0].elem_size == cfg.elem_size,
          "non-selected parameter (elem_size) unchanged")

    print("== analyze (ranking) ==")
    # Synthetic QoI that depends strongly on friction, weakly on E, not on speed
    Xn = plan.X
    Y = 1000.0 * Xn[:, 2] + 0.001 * Xn[:, 0] + 0.0 * Xn[:, 1]
    res, n_bad = mp.analyze_safe(plan, Y)
    check(res is not None and n_bad == 0, "analyze_safe runs (no bad runs)")
    names = list(res["names"])
    mustar = {names[i]: res["mu_star"][i] for i in range(len(names))}
    # friction must dominate; cutting_speed must be ~negligible
    fr = mustar["interaction.friction_coeff"]
    cs = mustar["bcs.cutting_speed"]
    ee = mustar["euler_material.E"]
    print("      mu*: friction=%.3g  E=%.3g  speed=%.3g" % (fr, ee, cs))
    check(fr > ee and fr > cs, "friction ranked most influential (mu*)")
    check(cs <= ee + 1e-9 or cs < fr,
          "cutting_speed (absent from Y) ranks low")

    print("== analyze_safe tolerates a failed run ==")
    Y2 = Y.copy(); Y2[3] = np.nan
    res2, nb = mp.analyze_safe(plan, Y2)
    check(res2 is not None and nb == 1, "1 NaN run repaired, analysis still runs")

    print("-" * 64)
    if _failures:
        print("RESULT: %d FAILURE(S)" % _failures)
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
