# -*- coding: utf-8 -*-
"""Non-regression tests for the second batch of Sensitivity-tab fixes:
elasticity (kelvin for temperatures, always computed, ranking), central
field SSD + field elasticity, cutting speed in the unit system, warm-up,
trust region, plan invalidation, estimate without failed runs, launcher
pipe drained during the run, Morris CSV export."""
from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from gui.core import units, unit_system as us
from gui.core.model_config import ModelConfig
from gui.results.qoi import QoISpec
from gui.sensitivity import export_results as xr
from gui.sensitivity import field_metrics as fm
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import param_registry as pr
from gui.sensitivity import runner_core as rc

S_A = pr.spec_for("euler_material.A")
S_E = pr.spec_for("euler_material.E")
S_T = pr.spec_for("bcs.ambient_temperature")
S_V = pr.spec_for("bcs.cutting_speed")


@pytest.fixture(autouse=True)
def _restore_units():
    saved = units.active_system()
    yield
    units.set_active_system(saved)


def _row(tab, path):
    return next(r for r, s in tab._row_spec.items() if s.path == path)


# ---------------------------------------------------------------------------
# Elasticity
# ---------------------------------------------------------------------------
def test_temperature_elasticity_uses_kelvin_whatever_the_display_unit():
    Y = np.array([500.0, 510.0, 490.0])
    pc = jac.build_plan([(S_T, 20.0, 5.0, True)], scheme="central",
                        temp_unit="C")
    pk = jac.build_plan([(S_T, 293.15, 5.0, True)], scheme="central",
                        temp_unit="K")
    ec = jac.analyze(pc, Y)[S_T.path]["elasticity"]
    ek = jac.analyze(pk, Y)[S_T.path]["elasticity"]
    assert ec == pytest.approx(ek)
    assert ec == pytest.approx(2.0 * 293.15 / 500.0)       # dQ/dT * T/Q


def test_temperature_qoi_elasticity_uses_kelvin():
    plan = jac.build_plan([(S_A, 100.0, 10.0, True)], scheme="central")
    Y = np.array([400.0, 420.0, 380.0])                    # T_max in °C
    e = jac.analyze(plan, Y, q_offset=273.15)[S_A.path]["elasticity"]
    assert e == pytest.approx(2.0 * 100.0 / (400.0 + 273.15))


def test_elasticity_always_reported_and_ranking_key():
    plan = jac.build_plan([(S_A, 100.0, 10.0, False),
                           (S_E, 100.0, 1.0, False)], scheme="forward")
    # Q = 2 A + 3 E ; Q0 = 500
    Y = np.array([500.0, 520.0, 503.0])
    q = QoISpec("Q", "Q", "N", lambda b, i, w: b)
    res = rc.run_plan(plan, "jacobian", [q], lambda c, i: Y[i], ModelConfig())
    a = res.analyses["Q"]
    assert a[S_A.path]["sensitivity"] == pytest.approx(2.0)   # raw
    assert a[S_A.path]["elasticity"] == pytest.approx(0.4)
    assert [p for p, _ in rc.jacobian_ranking(res, "Q")] == [S_E.path,
                                                             S_A.path]
    assert [p for p, _ in rc.jacobian_ranking(res, "Q", key="elasticity")] \
        == [S_E.path, S_A.path]
    csv_text = xr.result_to_csv(res)
    assert "elasticity" in csv_text.splitlines()[0]


def test_raw_ranking_warns_on_mixed_units(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    plan = jac.build_plan([(S_A, 100.0, 10.0, False),
                           (S_V, 60.0, 5.0, False)], scheme="forward")
    q = QoISpec("Q", "Q", "N", lambda b, i, w: b)
    res = rc.run_plan(plan, "jacobian", [q],
                      lambda c, i: float(i + 1), ModelConfig())
    tab.plan = plan
    tab._last_result = res
    tab._show_results(res)
    assert "different units" in tab.lbl_chart_note.text()
    tab.cb_chart_rank.setCurrentIndex(1)                   # elasticity
    assert tab.lbl_chart_note.text() == ""


# ---------------------------------------------------------------------------
# Field SSD
# ---------------------------------------------------------------------------
class _Info:
    field_variables = ["TEMP"]


class _FB:
    def __init__(self, arr):
        self.arr = np.asarray(arr, float)
        self.instance_names = ["Euler"]

    def instance(self, name):
        return _Info()

    def field(self, inst, var):
        return self.arr


def _field_plan(scheme):
    return jac.build_plan([(S_A, 100.0, 10.0, False)], scheme=scheme)


def test_field_central_uses_both_perturbed_runs():
    plan = _field_plan("central")
    base = np.full((2, 3), 100.0)
    plus = base + 2.0
    minus = base - 4.0                  # asymmetric: plus-only would differ
    b = [None] * plan.n_runs
    b[0] = _FB(base)
    b[plan.idx_plus[0]] = _FB(plus)
    b[plan.idx_minus[0]] = _FB(minus)
    d = rc.jacobian_field_analysis(plan, b, ["TEMP"], instance="Euler")
    r = d["TEMP"][S_A.path]
    # SSD(plus, minus) / (2 delta)^2 = 6 elements * 36 / 400
    assert r["sensitivity"] == pytest.approx(6 * 36.0 / 400.0)
    # RMS((plus-minus)/2) / RMS(base) = 3 / 100 -> 3 %
    assert r["rel_pct"] == pytest.approx(3.0)
    # elasticity = 3 % / (100 * 10 / 100) %
    assert r["elasticity"] == pytest.approx(0.3)


def test_field_central_missing_run_gives_nan():
    plan = _field_plan("central")
    b = [None] * plan.n_runs
    b[0] = _FB(np.ones((2, 3)))
    b[plan.idx_plus[0]] = _FB(np.ones((2, 3)) * 2)
    r = rc.jacobian_field_analysis(plan, b, ["TEMP"],
                                   instance="Euler")["TEMP"][S_A.path]
    assert np.isnan(r["sensitivity"]) and np.isnan(r["rel_pct"])


def test_field_forward_unchanged():
    plan = _field_plan("forward")
    base = np.full((1, 2), 10.0)
    b = [_FB(base), _FB(base + 1.0)]
    r = rc.jacobian_field_analysis(plan, b, ["TEMP"],
                                   instance="Euler")["TEMP"][S_A.path]
    assert r["sensitivity"] == pytest.approx(2 * 1.0 / 100.0)
    assert r["rel_pct"] == pytest.approx(10.0)


def test_rel_change_central_helper():
    base = np.array([[4.0, 4.0]])
    assert fm.field_rel_change_pct_central(base, base + 1, base - 1) == \
        pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Cutting speed follows the unit system
# ---------------------------------------------------------------------------
def test_cutting_speed_follows_velocity_unit():
    cfg = ModelConfig()
    stored = pr.get_stored(cfg, S_V.path)                 # mm/s
    assert pr.get_display(cfg, S_V) == pytest.approx(stored * 60 / 1000)
    assert S_V.unit_str() == "m/min"
    units.set_active_system(dataclasses.replace(us.UnitSystem(),
                                                velocity="m/s"))
    assert S_V.unit_str() == "m/s"
    assert pr.get_display(cfg, S_V) == pytest.approx(stored / 1000)
    assert S_V.to_stored(2.0) == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# Tab: trust region, plan invalidation, warm-up, estimate
# ---------------------------------------------------------------------------
def test_generate_refuses_step_outside_trust_region(qapp):
    from PySide6.QtCore import Qt
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    r = _row(tab, S_A.path)
    tab.table.item(r, 0).setCheckState(Qt.Checked)
    hi = tab._cell_float(r, 4)
    ref = tab._cell_float(r, 2)
    tab.table.item(r, 5).setText(repr(2.0 * (hi - ref)))  # Ref+Delta > Max
    tab._on_generate()
    assert tab.plan is None and "trust region" in tab.status.text()
    tab.cb_scheme.setCurrentText("backward")      # only Ref - Delta used
    tab.table.item(r, 3).setText(repr(ref - 3.0 * (hi - ref)))
    tab._on_generate()
    assert tab.plan is not None


@pytest.mark.parametrize("edit", ["qoi", "field", "scheme", "table"])
def test_plan_discarded_when_selection_changes(qapp, edit):
    from PySide6.QtCore import Qt
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    r = _row(tab, S_A.path)
    tab.table.item(r, 0).setCheckState(Qt.Checked)
    tab._on_generate()
    assert tab.plan is not None
    if edit == "qoi":
        tab._qoi_checks["PEEQ_max"].setChecked(True)
    elif edit == "field":
        tab._field_checks["EVF"].setChecked(True)
    elif edit == "scheme":
        tab.cb_scheme.setCurrentText("forward")
    else:
        tab.table.item(r, 7).setCheckState(Qt.Checked)    # Norm
    assert tab.plan is None and not tab.btn_run.isEnabled()
    assert "Plan discarded" in tab.status.text()


def test_plan_discarded_when_model_value_changes(qapp):
    from PySide6.QtCore import Qt
    from gui.tabs.sensitivity_tab import SensitivityTab
    cfg = ModelConfig()
    tab = SensitivityTab(cfg)
    tab.table.item(_row(tab, S_A.path), 0).setCheckState(Qt.Checked)
    tab._on_generate()
    tab.refresh_from_model()                               # nothing changed
    assert tab.plan is not None
    pr.set_stored(cfg, S_A.path, pr.get_stored(cfg, S_A.path) * 1.5)
    tab.refresh_from_model()
    assert tab.plan is None


def test_field_vars_are_captured_at_generate(qapp):
    from PySide6.QtCore import Qt
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    tab._field_checks["TEMP"].setChecked(True)
    tab.table.item(_row(tab, S_A.path), 0).setCheckState(Qt.Checked)
    tab._on_generate()
    assert tab.plan_field_vars == ["TEMP"]


def test_warmup_is_recorded(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    tab.plan = jac.build_plan([(S_A, 100, 10, False)], scheme="forward")
    tab.plan_kind = "jacobian"
    tab.spin_warmup.setValue(0.25)
    assert tab._plan_record([], 1)["warmup_frac"] == pytest.approx(0.25)


def test_failed_run_does_not_enter_the_time_estimate(qapp):
    import time
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    tab._run_clock0 = time.monotonic()
    tab._failed_live = []
    tab._on_progress(0, 3)                  # run 0 starts
    tab._run_t0[0] -= 100.0                 # ...and lasts 100 s
    tab._on_progress(1, 3)                  # run 0 ok, run 1 starts
    tab._run_t0[1] -= 1.0                   # run 1 dies after 1 s
    tab._failed_live.append(1)
    tab._on_progress(2, 3)
    assert tab._n_finished == 2
    assert tab._per_run_sec == pytest.approx(100.0, rel=0.01)


def test_signed_tooltip_follows_scheme():
    from gui.tabs.sensitivity_tab import SensitivityTab
    assert "(F+ - F0)" in SensitivityTab._signed_tooltip("forward")
    assert "2δ" in SensitivityTab._signed_tooltip("central")


# ---------------------------------------------------------------------------
# Launcher pipe drained while the run is in progress
# ---------------------------------------------------------------------------
def test_drain_pipe_prevents_blocking():
    from gui.sensitivity.run_worker import _drain_pipe
    import threading
    # A child that writes far more than any pipe buffer before exiting.
    code = "import sys; sys.stdout.write('x' * 2000000); sys.stdout.flush()"
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE)
    chunks = []
    t = threading.Thread(target=_drain_pipe, args=(proc.stdout, chunks),
                         daemon=True)
    t.start()
    proc.wait(timeout=30)                    # would hang without draining
    t.join(timeout=5)
    assert sum(len(c) for c in chunks) == 2000000


# ---------------------------------------------------------------------------
# Morris CSV export
# ---------------------------------------------------------------------------
def test_morris_csv_export():
    pytest.importorskip("SALib")
    from gui.sensitivity import morris_plan as mp
    plan = mp.build_plan([(S_A, 50, 150), (S_V, 50, 150)], N=4, seed=1)
    q = QoISpec("Q", "Q", "-", lambda b, i, w: b)
    res = rc.run_plan(plan, "morris", [q],
                      lambda c, i: 3 * pr.get_display(c, S_A), ModelConfig())
    lines = xr.result_to_csv(res).splitlines()
    assert lines[0].startswith("qoi,parameter,label,mu_star,sigma")
    assert lines[1].startswith("Q,euler_material.A,")          # mu* = 300
    assert ",4,4," in lines[1]                                  # traj. used
    # A QoI that could not be analysed is written with its reason.
    res_c = rc.run_plan(plan, "morris", [q], lambda c, i: None, ModelConfig())
    text = xr.result_to_csv(res_c)
    assert "not analysed" in text
