# -*- coding: utf-8 -*-
"""Non-regression tests for the Sensitivity-tab audit fixes:

  1. Morris analyses complete trajectories only (no imputed runs),
  2. a cancelled campaign reports the runs it never launched,
  3. plans are pinned to the unit system they were generated under, and the
     table converts ticked rows when the display units change,
  4. closing the window stops a running campaign synchronously,
  5. config.json records the full plan (Morris seed included),
  + the per-element maps are written to the study folder.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import threading
import time

import numpy as np
import pytest

from gui.core import units, unit_system as us
from gui.core.model_config import ModelConfig
from gui.results.qoi import QoISpec
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import map_export as mx
from gui.sensitivity import morris_plan as mp
from gui.sensitivity import param_registry as pr
from gui.sensitivity import runner_core as rc

pytest.importorskip("SALib")

S_A = pr.spec_for("euler_material.A")
S_V = pr.spec_for("bcs.cutting_speed")
S_T = pr.spec_for("bcs.ambient_temperature")
S_E = pr.spec_for("euler_material.E")


class _B:
    def __init__(self, v):
        self.v = v


_Q = QoISpec("Q", "Q", "-", lambda b, inst, w: b.v)


@pytest.fixture(autouse=True)
def _restore_units():
    saved = units.active_system()
    yield
    units.set_active_system(saved)


def _only_A(cfg, i):
    """QoI that depends on A only (Q = 3 A): V has no effect at all."""
    return _B(3.0 * pr.get_display(cfg, S_A))


# ---------------------------------------------------------------------------
# 1. Morris: complete trajectories only
# ---------------------------------------------------------------------------
class TestMorrisCompleteTrajectories:
    def _plan(self):
        return mp.build_plan([(S_A, 50, 150), (S_V, 50, 150)], N=10, seed=1)

    def test_cancelled_campaign_is_not_analysed(self):
        plan = self._plan()
        n = {"k": 0}

        def solve(c, i):
            n["k"] += 1
            return _only_A(c, i)
        res = rc.run_plan(plan, "morris", [_Q], solve, ModelConfig(),
                          should_cancel=lambda: n["k"] >= 4)
        a = res.analyses["Q"]
        # 4 runs = 1 complete trajectory (k+1 = 3) -> fewer than 2: refused,
        # where the old imputation reported mu*(V) = 26 for a null effect.
        assert "error" in a
        assert a["n_used"] == 1 and a["n_trajectories"] == 10

    def test_failed_run_drops_its_trajectory_only(self):
        plan = self._plan()

        def solve(c, i):
            return None if i == 4 else _only_A(c, i)   # inside trajectory 2
        res = rc.run_plan(plan, "morris", [_Q], solve, ModelConfig())
        a = res.analyses["Q"]
        assert a["n_used"] == 9 and a["n_dropped"] == 1
        mu = dict(zip(a["names"], a["mu_star"]))
        # A null effect stays exactly null: nothing is imputed.
        assert mu["bcs.cutting_speed"] == pytest.approx(0.0, abs=1e-9)
        assert mu["euler_material.A"] > 0

    def test_seed_is_drawn_recorded_and_reproducible(self):
        p1 = mp.build_plan([(S_A, 50, 150), (S_V, 50, 150)], N=4)
        assert isinstance(p1.seed, int)
        p2 = mp.build_plan([(S_A, 50, 150), (S_V, 50, 150)], N=4,
                           seed=p1.seed)
        assert np.array_equal(p1.X, p2.X)


# ---------------------------------------------------------------------------
# 2. Cancel accounting
# ---------------------------------------------------------------------------
def test_cancel_reports_runs_not_launched(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    plan = jac.build_plan([(S_A, 100, 10, False), (S_V, 100, 10, False)])
    n = {"k": 0}

    def solve(c, i):
        n["k"] += 1
        return _B(float(i))
    res = rc.run_plan(plan, "jacobian", [_Q], solve, ModelConfig(),
                      should_cancel=lambda: n["k"] >= 2)
    assert res.cancelled and res.n_attempted == 2 and res.n_not_run == 3
    assert res.n_ok == 2
    msg, warn = SensitivityTab._run_summary(res)
    assert "cancelled" in msg and "2/5 runs launched" in msg
    assert "3 not run" in msg and warn


def test_cancel_during_last_run_is_flagged():
    plan = jac.build_plan([(S_A, 100, 10, False)], scheme="forward")
    flag = {"c": False}

    def solve(c, i):
        if i == plan.n_runs - 1:
            flag["c"] = True
            return None
        return _B(1.0)
    res = rc.run_plan(plan, "jacobian", [_Q], solve, ModelConfig(),
                      should_cancel=lambda: flag["c"])
    assert res.cancelled and res.n_attempted == plan.n_runs


def test_complete_run_summary(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    plan = jac.build_plan([(S_A, 100, 10, False)], scheme="forward")
    res = rc.run_plan(plan, "jacobian", [_Q], lambda c, i: _B(1.0),
                      ModelConfig())
    msg, warn = SensitivityTab._run_summary(res)
    assert msg.startswith("Run finished: 2/2 successful, 0 failed.")
    assert not warn


# ---------------------------------------------------------------------------
# 3. Units
# ---------------------------------------------------------------------------
def test_plan_is_pinned_to_its_unit_system():
    cfg = ModelConfig()
    x0 = pr.get_display(cfg, S_E)                     # GPa by default
    plan = jac.build_plan([(S_E, x0, 0.1 * x0, False)], scheme="forward")
    units.set_active_system(dataclasses.replace(us.UnitSystem(),
                                                modulus="MPa"))
    stored = pr.get_stored(jac.plan_to_configs(cfg, plan)[0], S_E.path)
    assert stored == pytest.approx(pr.get_stored(cfg, S_E.path))


def test_temperature_unit_label_follows_temp_unit():
    assert S_T.unit_str("K") == "K"
    assert S_T.unit_str("C") == "°C"


def _row(tab, path):
    return next(r for r, s in tab._row_spec.items() if s.path == path)


def test_ticked_row_is_converted_when_units_change(qapp):
    from PySide6.QtCore import Qt
    from gui.tabs.sensitivity_tab import SensitivityTab
    cfg = ModelConfig()
    tab = SensitivityTab(cfg)
    r = _row(tab, S_T.path)
    tab.table.item(r, 0).setCheckState(Qt.Checked)
    lo_c, hi_c = tab._cell_float(r, 3), tab._cell_float(r, 4)
    d_c = tab._cell_float(r, 5)
    cfg.ui.temp_unit = "K"
    tab.refresh_from_model()
    assert tab._cell_float(r, 3) == pytest.approx(lo_c + 273.15)
    assert tab._cell_float(r, 4) == pytest.approx(hi_c + 273.15)
    assert tab._cell_float(r, 5) == pytest.approx(d_c)    # a difference
    assert tab.table.item(r, 8).text() == "K"
    ref = tab._cell_float(r, 2)
    assert tab._cell_float(r, 6) == pytest.approx(100 * d_c / ref, rel=1e-3)


def test_ticked_material_row_is_converted_on_unit_system_change(qapp):
    from PySide6.QtCore import Qt
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    r = _row(tab, S_E.path)
    tab.table.item(r, 0).setCheckState(Qt.Checked)
    lo, d = tab._cell_float(r, 3), tab._cell_float(r, 5)
    units.set_active_system(dataclasses.replace(us.UnitSystem(),
                                                modulus="MPa"))
    tab.refresh_from_model()
    assert tab._cell_float(r, 3) == pytest.approx(lo * 1000, rel=1e-4)
    assert tab._cell_float(r, 5) == pytest.approx(d * 1000, rel=1e-4)


def test_plan_discarded_when_units_change(qapp):
    from PySide6.QtCore import Qt
    from gui.tabs.sensitivity_tab import SensitivityTab
    cfg = ModelConfig()
    tab = SensitivityTab(cfg)
    tab.table.item(_row(tab, S_A.path), 0).setCheckState(Qt.Checked)
    tab._on_generate()
    assert tab.plan is not None and tab.btn_run.isEnabled()
    cfg.ui.temp_unit = "K"
    tab.refresh_from_model()
    assert tab.plan is None and not tab.btn_run.isEnabled()
    assert "unit system changed" in tab.status.text()


# ---------------------------------------------------------------------------
# 4. Closing the window during a campaign
# ---------------------------------------------------------------------------
def test_shutdown_stops_running_campaign(qapp):
    from PySide6.QtCore import QThread
    from gui.sensitivity.run_worker import SensitivityRunWorker
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    plan = jac.build_plan([(S_A, 100, 10, False)], scheme="forward")
    started = threading.Event()

    def slow(cfg, i):                    # a "solver" that runs until cancel
        started.set()
        while not worker._cancel:
            time.sleep(0.01)
        return None
    worker = SensitivityRunWorker(plan, "jacobian", [_Q], ModelConfig(),
                                  abaqus_cmd="", abaqus_script="",
                                  workdir=".", solve_fn=slow)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    got = []
    worker.finished.connect(got.append)
    tab._worker, tab._thread = worker, thread
    thread.start()
    assert started.wait(5)
    assert tab.is_running()
    assert tab.shutdown(timeout_ms=5000)
    assert not tab.is_running() and thread.isFinished()
    qapp.processEvents()
    assert got == []                     # result dropped, not delivered


# ---------------------------------------------------------------------------
# 5. config.json: the full plan
# ---------------------------------------------------------------------------
def test_plan_record_jacobian(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    tab.plan = jac.build_plan([(S_A, 100, 10, True), (S_T, 20, 5, False)],
                              scheme="central")
    tab.plan_kind = "jacobian"
    tab.selected_qois = [_Q]
    rec = tab._plan_record(["EVF"], 4)
    json.dumps(rec)                                   # JSON-serialisable
    assert rec["qois"] == ["Q"] and rec["scheme"] == "central"
    assert rec["field_vars"] == ["EVF"] and rec["cpus"] == 4
    assert rec["unit_system"]["temp"] == "C"
    p0 = rec["varied_parameters"][0]
    assert p0 == {"path": S_A.path, "label": S_A.label, "unit": "MPa",
                  "base": 100.0, "delta": 10.0, "normalize": True}
    assert len(rec["runs"]) == 5
    assert rec["runs"][1]["job"] == "sensitivity_run001"
    assert rec["runs"][1]["kind"] == "+0"
    assert rec["runs"][1]["values"][S_A.path] == pytest.approx(110.0)


def test_plan_record_morris_has_seed_and_bounds(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    tab.plan = mp.build_plan([(S_A, 50, 150), (S_V, 50, 150)], N=3, seed=7)
    tab.plan_kind = "morris"
    tab.selected_qois = [_Q]
    rec = tab._plan_record([], 1)
    assert rec["seed"] == 7 and rec["N"] == 3
    assert rec["varied_parameters"][1]["min"] == 50.0
    assert len(rec["runs"]) == 9 and "kind" not in rec["runs"][0]


# ---------------------------------------------------------------------------
# Maps written to the study folder
# ---------------------------------------------------------------------------
def _square_mesh():
    # 2 quads side by side: nodes (0..5), elements (0,1,4,3) and (1,2,5,4)
    nodes = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    return nodes, faces


def test_write_maps_arrays_and_images(tmp_path):
    nodes, faces = _square_mesh()
    S = np.array([[1.0, -2.0], [3.0, np.nan]])        # (2 frames, 2 elem)
    files = mx.write_maps(
        tmp_path, {"TEMP": {S_A.path: S}}, nodes, faces,
        param_info={S_A.path: ("A", "MPa")}, frame_times=[0.0, 1e-3],
        scheme="central", deltas={S_A.path: 10.0})
    names = {p.name for p in files}
    stem = "map_TEMP_p01_euler_material.A"
    assert {stem + ".npz", stem + "_mean.png", stem + "_rms.png",
            "maps_index.csv", "mesh.npz"} <= names
    assert not any(n.endswith(".csv") and n != "maps_index.csv"
                   for n in names)
    with np.load(tmp_path / (stem + ".npz")) as z:    # no allow_pickle
        assert np.array_equal(z["S"], S, equal_nan=True)
        assert z["time_mean"][0] == pytest.approx(2.0)
        assert z["time_rms"][0] == pytest.approx(np.sqrt(5.0))
        assert z["time_mean"][1] == pytest.approx(-2.0)   # NaN ignored
        assert str(z["map_unit"]) == "°C / MPa"
        assert str(z["parameter"]) == S_A.path and float(z["delta"]) == 10.0
    with np.load(tmp_path / "mesh.npz") as z:
        assert z["faces"].shape == (2, 4)
        assert z["centroids_xy"][0] == pytest.approx([0.5, 0.5])
        assert z["frame_times"][1] == pytest.approx(1e-3)
    with open(tmp_path / "maps_index.csv", encoding="utf-8-sig") as f:
        idx = list(csv.DictReader(f))
    assert idx[0]["map_unit"] == "°C / MPa" and idx[0]["delta"] == "10.0"
    assert idx[0]["data"] == stem + ".npz"
    assert (tmp_path / (stem + "_mean.png")).stat().st_size > 0


def test_write_maps_skips_mismatched_map(tmp_path):
    nodes, faces = _square_mesh()
    files = mx.write_maps(tmp_path, {"EVF": {S_A.path: np.zeros((2, 3))}},
                          nodes, faces, param_info={S_A.path: ("A", "MPa")})
    assert not any(p.name.startswith("map_") for p in files)


def test_map_unit():
    assert mx.map_unit("—", "—") == "—"
    assert mx.map_unit("—", "MPa") == "1/MPa"
    assert mx.map_unit("mm/s", "—") == "mm/s"


def test_tab_writes_maps_into_study_subfolder(qapp, tmp_path):
    from gui.results.fake_builder import build_fake_results
    from gui.results.reader import ResultsBundle
    from gui.tabs.sensitivity_tab import SensitivityTab
    bundles = []
    for k in range(3):
        _, npz = build_fake_results(tmp_path / ("j%d.results.npz" % k),
                                    n_frames=3, n_grid_x=5, n_grid_y=4)
        bundles.append(ResultsBundle.load(npz))
    spec = pr.spec_for("interaction.friction_coeff")
    plan = jac.build_plan([(spec, 0.3, 0.1, False)], scheme="central")
    order = [None] * plan.n_runs
    order[0] = bundles[0]
    order[plan.idx_plus[0]] = bundles[1]
    order[plan.idx_minus[0]] = bundles[2]
    tab = SensitivityTab(ModelConfig())
    tab.plan, tab.plan_kind = plan, "jacobian"
    tab._run_field_vars = ["EVF"]
    res = rc.RunResult(plan_kind="jacobian", qoi_ids=[],
                       param_paths=list(plan.param_paths),
                       Y=np.zeros((plan.n_runs, 0)), analyses={},
                       failures=[], bundles=order,
                       n_attempted=plan.n_runs)
    tab._build_field_maps(res)
    out = tmp_path / "study" / mx.MAPS_SUBDIR
    tab._export_field_maps(out, wait=True)
    stem = "map_EVF_p01_interaction.friction_coeff"
    for name in (stem + ".npz", stem + "_mean.png", stem + "_rms.png",
                 "maps_index.csv", "mesh.npz"):
        assert (out / name).is_file(), name
    with np.load(out / (stem + ".npz")) as z:
        assert z["S"].shape[0] == 3                   # 3 frames
    with np.load(out / "mesh.npz") as z:
        assert z["frame_times"].shape == (3,)
    assert "Maps written to" in tab.status.text()
    for b in bundles:
        b.close()
