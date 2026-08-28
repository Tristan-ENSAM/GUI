# -*- coding: utf-8 -*-
"""
Lot 2a self-test — runs WITHOUT Abaqus and WITHOUT SALib.

Validates:
  1. param_registry: path get/set, unit conversions (material + non-material),
     default bound suggestions.
  2. qoi: QoI extraction against a fake bundle produced by fake_builder.

Run from the repo root:
    python -m tests.test_lot2a
or:
    python tests/test_lot2a.py
Exits 0 on success, 1 on the first failure.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running as a plain script (python tests/test_lot2a.py) by putting
# the repo root on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from gui.core.model_config import ModelConfig
from gui.sensitivity import param_registry as pr
from gui.results import qoi as Q
from gui.results.reader import ResultsBundle
from gui.results.fake_builder import build_fake_results


_failures = 0


def check(cond: bool, msg: str):
    global _failures
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _failures += 1


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
def test_registry_paths():
    print("== registry: path get/set ==")
    cfg = ModelConfig()

    # Material dict leaf
    a0 = pr.get_stored(cfg, "euler_material.A")
    check(a0 == cfg.euler_material["A"], f"get_stored euler_material.A == {a0}")
    pr.set_stored(cfg, "euler_material.A", 123.0)
    check(cfg.euler_material["A"] == 123.0, "set_stored euler_material.A wrote the dict")

    # Dataclass attr leaf
    cs0 = pr.get_stored(cfg, "bcs.cutting_speed")
    check(cs0 == cfg.bcs.cutting_speed, f"get_stored bcs.cutting_speed == {cs0}")
    pr.set_stored(cfg, "bcs.cutting_speed", 2000.0)
    check(cfg.bcs.cutting_speed == 2000.0, "set_stored bcs.cutting_speed wrote the attr")

    # Top-level scalar
    pr.set_stored(cfg, "elem_size", 0.01)
    check(cfg.elem_size == 0.01, "set_stored elem_size (top-level scalar)")

    # Every registered path must resolve on a fresh cfg
    cfg2 = ModelConfig()
    all_ok = True
    for s in pr.REGISTRY:
        try:
            pr.get_stored(cfg2, s.path)
        except Exception as e:  # noqa
            all_ok = False
            print(f"      unresolved path {s.path!r}: {e}")
    check(all_ok, f"all {len(pr.REGISTRY)} registered paths resolve on a fresh cfg")


def test_registry_units():
    print("== registry: unit conversions ==")
    cfg = ModelConfig()

    # E: stored 124000 MPa  <->  displayed 124 GPa
    sE = pr.spec_for("euler_material.E")
    check(sE.display_unit == "GPa", "E display unit is GPa")
    disp_E = pr.get_display(cfg, sE)
    check(approx(disp_E, 124.0), f"E displayed = {disp_E} GPa (stored 124000 MPa)")
    pr.apply_display(cfg, sE, 200.0)            # set 200 GPa
    check(approx(cfg.euler_material["E"], 200000.0),
          f"setting 200 GPa stored {cfg.euler_material['E']} MPa")

    # rho: stored 8.96e-9 t/mm³  <->  displayed 8960 kg/m³
    sR = pr.spec_for("euler_material.rho")
    disp_rho = pr.get_display(cfg, sR)
    check(approx(disp_rho, 8960.0, 1e-3), f"rho displayed = {disp_rho} kg/m³")

    # cutting speed: stored mm/s  <->  displayed m/min
    sCS = pr.spec_for("bcs.cutting_speed")
    cfg.bcs.cutting_speed = 1000.0  # mm/s
    disp_cs = pr.get_display(cfg, sCS)
    # 1000 mm/s = 60 m/min
    check(approx(disp_cs, 60.0, 1e-6), f"1000 mm/s displayed = {disp_cs} m/min")
    pr.apply_display(cfg, sCS, 120.0)  # 120 m/min -> 2000 mm/s
    check(approx(cfg.bcs.cutting_speed, 2000.0, 1e-6),
          f"120 m/min stored {cfg.bcs.cutting_speed} mm/s")

    # temperature: stored °C, displayed K when temp_unit='K'
    sT = pr.spec_for("bcs.ambient_temperature")
    cfg.bcs.ambient_temperature = 20.0  # °C
    check(approx(pr.get_display(cfg, sT, "C"), 20.0), "ambient 20 °C displayed in C")
    check(approx(pr.get_display(cfg, sT, "K"), 293.15), "ambient 20 °C displayed in K")
    pr.apply_display(cfg, sT, 300.15, "K")     # 300.15 K -> 27 °C
    check(approx(cfg.bcs.ambient_temperature, 27.0, 1e-6),
          f"300.15 K stored {cfg.bcs.ambient_temperature} °C")


def test_default_bounds():
    print("== registry: default display bounds ==")
    cfg = ModelConfig()

    # E: 124 GPa ± 15% -> [105.4, 142.6]
    sE = pr.spec_for("euler_material.E")
    lo, hi = pr.default_display_bounds(cfg, sE)
    check(lo < 124.0 < hi and lo > 0, f"E bounds bracket 124 GPa: ({lo:.2f}, {hi:.2f})")
    check(approx(lo, 124.0 * 0.85, 1e-3) and approx(hi, 124.0 * 1.15, 1e-3),
          "E bounds are ±15% of 124 GPa")

    # rake_angle: default 0 deg, rel_range 0 -> abs_range 10 -> [-10, 10]
    sRA = pr.spec_for("tool_geometry.rake_angle")
    lo, hi = pr.default_display_bounds(cfg, sRA)
    check(approx(lo, -10.0) and approx(hi, 10.0),
          f"rake_angle bounds use abs_range: ({lo}, {hi})")

    # bounds always ordered
    all_ordered = all(
        pr.default_display_bounds(cfg, s)[0] <= pr.default_display_bounds(cfg, s)[1]
        for s in pr.REGISTRY
    )
    check(all_ordered, "all default bounds are ordered (lo <= hi)")


def test_qoi():
    print("== qoi: extraction against fake bundle ==")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fake_job.results.npz"
        # Deterministic fake run (seed fixed inside build_fake_results).
        json_path, npz_path = build_fake_results(
            out, n_frames=40, n_grid_x=30, n_grid_y=20,
            sim_time=5e-4, job_name="fake_job", seed=7,
        )
        check(json_path.exists() and npz_path.exists(), "fake bundle written")

        bundle = ResultsBundle.load(npz_path)
        try:
            vals = Q.compute_qois(bundle)
            print(f"      QoI = { {k: round(v, 4) for k, v in vals.items()} }")

            # All five QoI present and finite (the fake bundle has RF1/RF2,
            # TEMP and PEEQ — see fake_builder).
            for qid in Q.available_qoi_ids():
                check(qid in vals and np.isfinite(vals[qid]),
                      f"{qid} is present and finite")

            # Plausibility, grounded in fake_builder's analytic fields:
            #   - TEMP peaks at ~600 °C (peak constant in fake_builder)
            #   - PEEQ accumulates positive
            #   - Fx_max >= Fx_mean (max of a signal >= its mean)
            check(400.0 <= vals["T_max"] <= 700.0,
                  f"T_max plausible (~600): {vals['T_max']:.1f} °C")
            check(vals["PEEQ_max"] > 0.0, f"PEEQ_max > 0: {vals['PEEQ_max']:.4f}")
            check(vals["Fx_max"] >= vals["Fx_mean"] > 0.0,
                  f"Fx_max ({vals['Fx_max']:.1f}) >= Fx_mean ({vals['Fx_mean']:.1f}) > 0")

            # warmup_frac: skipping the entry ramp should not lower the mean
            # below zero and should change it (forces ramp up over time).
            vals_warm = Q.compute_qois(bundle, ["Fx_mean"], warmup_frac=0.5)
            check(np.isfinite(vals_warm["Fx_mean"]) and vals_warm["Fx_mean"] > 0,
                  f"Fx_mean with warmup_frac=0.5 finite & positive: "
                  f"{vals_warm['Fx_mean']:.1f}")

            # Missing-data robustness: a non-existent QoI source -> NaN, no raise.
            # (Simulate by asking for a QoI on an instance that lacks the field.)
            t_missing = Q.qoi_T_max(bundle, instance="Tool")  # no such instance
            check(np.isnan(t_missing),
                  "T_max on a non-existent instance returns NaN (no crash)")
        finally:
            bundle.close()

        # from_path convenience + graceful failure on a missing file
        vals2 = Q.compute_qois_from_path(npz_path)
        check(np.isfinite(vals2["Fx_max"]), "compute_qois_from_path works")
        vals3 = Q.compute_qois_from_path(Path(td) / "does_not_exist.results.npz")
        check(all(np.isnan(v) for v in vals3.values()),
              "compute_qois_from_path on missing file returns all-NaN (no crash)")


def main():
    print("=" * 64)
    print("Lot 2a self-test — param_registry + qoi (no Abaqus, no SALib)")
    print("=" * 64)
    test_registry_paths()
    test_registry_units()
    test_default_bounds()
    test_qoi()
    print("-" * 64)
    if _failures:
        print(f"RESULT: {_failures} FAILURE(S)")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
