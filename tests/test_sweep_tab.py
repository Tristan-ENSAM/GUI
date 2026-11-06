# -*- coding: utf-8 -*-
"""Sweep and grid plans in the Parameter plans tab: levels from the table,
cost, record, runs table, trend chart, time curves, export, and the reuse
of finished runs by the worker."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from PySide6.QtCore import Qt

from gui.core.model_config import ModelConfig
from gui.results.qoi import QoISpec
from gui.sensitivity import param_registry as pr
from gui.sensitivity import runner_core as rc
from gui.sensitivity import export_results as xr

MU = "interaction.friction_coeff"
VC = "bcs.cutting_speed"


def _row(tab, path):
    return next(r for r, s in tab._row_spec.items() if s.path == path)


def _tab(design="sweep", paths=(MU,)):
    from tests.conftest import _connected_tab as connected_tab
    tab = connected_tab()
    tab.cb_method.setCurrentIndex({"sweep": 2, "grid": 3}[design])
    for p in paths:
        tab.table.item(_row(tab, p), 0).setCheckState(Qt.Checked)
    return tab


def test_method_controls_and_columns(qapp):
    tab = _tab("sweep")
    assert not tab.table.isColumnHidden(9)        # Values
    assert tab.table.isColumnHidden(5)            # Delta (Jacobian only)
    assert not tab.spin_nlev.isHidden() and not tab.chk_reuse.isHidden()
    tab.cb_method.setCurrentIndex(0)
    assert tab.table.isColumnHidden(9) and not tab.table.isColumnHidden(5)
    assert tab.spin_nlev.isHidden() and tab.chk_reuse.isHidden()


def test_sweep_levels_cost_and_typed_values(qapp):
    tab = _tab("sweep")
    r = _row(tab, MU)
    tab.table.item(r, 3).setText("0.1")
    tab.table.item(r, 4).setText("0.5")
    tab.spin_nlev.setValue(5)
    ref = pr.get_display(tab.cfg, pr.spec_for(MU), "C")
    sel, spacing = tab._collect_sweep()
    assert np.allclose(sel[0][2], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert spacing == ["linear"]
    shared = any(np.isclose(v, ref) for v in sel[0][2])
    assert tab._n_runs() == (5 if shared else 6)
    assert "runs (sweep)" in tab.lbl_cost.text()
    tab.table.item(r, 9).setText("0.2; 0.6")
    sel, spacing = tab._collect_sweep()
    assert np.allclose(sel[0][2], [0.2, 0.6]) and spacing == ["typed"]
    tab.table.item(r, 9).setText("abc")
    with pytest.raises(ValueError):
        tab._collect_sweep()
    assert "?" in tab.lbl_cost.text()


def test_generate_grid_and_record(qapp):
    tab = _tab("grid", (MU, VC))
    for p, vals in ((MU, "0.2 0.4"), (VC, "20; 40")):
        tab.table.item(_row(tab, p), 9).setText(vals)
    tab._on_generate()
    assert tab.plan is not None, tab.status.text()
    assert tab.plan_kind == "sweep" and tab.plan.design == "grid"
    assert tab.plan.n_runs == 5 and "in addition" in tab.status.text()
    rec = tab._plan_record([], 4)
    assert rec["plan_kind"] == "sweep" and rec["design"] == "grid"
    vp = {d["path"]: d for d in rec["varied_parameters"]}
    assert vp[MU]["levels"] == [0.2, 0.4] and vp[VC]["levels"] == [20, 40]
    assert vp[MU]["spacing"] == "typed"
    assert rec["runs"][0]["kind"] == "base"
    assert rec["runs"][1]["job"] == "grid_run001"
    assert "base" in tab.preview.toPlainText()


class _B:
    def __init__(self, mu, vc):
        self.mu, self.vc = mu, vc


def _fake_result(tab):
    """Run the plan with a fake solver: Q = 10 mu + vc, with time curves."""
    plan = tab.plan
    q = QoISpec("Q", "Q", "N", lambda b, inst, w: 10 * b.mu + b.vc)

    def solve(cfg, i):
        return _B(pr.get_display(cfg, pr.spec_for(MU)),
                  pr.get_display(cfg, pr.spec_for(VC))
                  if VC in plan.param_paths else 0.0)
    res = rc.run_plan(plan, "sweep", [q], solve, ModelConfig())
    # time curves, as run_series would give them
    res.series = [{"t": np.linspace(0, 1, 5), "Vx": np.full(5, float(i)),
                   "t_force": np.linspace(0, 1, 9),
                   "Fc": np.full(9, float(i))} for i in range(plan.n_runs)]
    tab.selected_qois = [q]
    return res


def test_results_table_chart_and_time_curves(qapp, tmp_path, monkeypatch):
    tab = _tab("grid", (MU, VC))
    for p, vals in ((MU, "0.2 0.4"), (VC, "20; 40")):
        tab.table.item(_row(tab, p), 9).setText(vals)
    tab._on_generate()
    res = _fake_result(tab)
    tab._last_result = res
    tab._show_results(res)
    t = tab.results_table
    assert t.rowCount() == 5 and t.horizontalHeaderItem(1).text() == "Varied"
    assert t.item(0, 1).text() == "base"
    # chart: Vc main effect, slope 1 N per m/min
    tab.cb_chart_param.setCurrentIndex(tab.cb_chart_param.findData(VC))
    assert "slope" in tab.lbl_chart_note.text()
    assert "Main effect" in tab.lbl_chart_note.text()
    assert not tab.cb_chart_param.isHidden()
    # time curves: Vx and Fc, one line per ticked run
    assert [tab.cb_series_qty.itemText(i)
            for i in range(tab.cb_series_qty.count())] == ["Vx", "Fc"]
    assert tab.lst_series_runs.count() == 5
    ax = tab._fig_series.axes[0]
    assert len(ax.get_lines()) == 5
    tab.lst_series_runs.item(1).setCheckState(Qt.Unchecked)
    assert len(tab._fig_series.axes[0].get_lines()) == 4
    tab.cb_series_qty.setCurrentText("Fc")
    assert tab._fig_series.axes[0].get_lines()[0].get_xdata().size == 9
    # export: trends + one row per run
    out = tmp_path / "plan.csv"
    monkeypatch.setattr(
        "gui.tabs.sensitivity_tab.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), ""))
    tab._on_export()
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert {r["parameter"] for r in rows} == {MU, VC}
    runs = list(csv.reader(open(tmp_path / "plan_runs.csv",
                                encoding="utf-8-sig")))
    assert runs[0][:3] == ["run", "job", "kind"] and len(runs) == 6
    assert runs[1][1] == "grid_run000" and runs[1][2] == "base"


def test_runs_csv_names_the_varied_parameter():
    from gui.sensitivity import sweep_plan as sp
    mu = pr.spec_for(MU)
    plan = sp.build_plan([(mu, 0.3, [0.1, 0.5])], "sweep")

    class R:
        qoi_ids = ["Q"]
        Y = np.array([[1.0], [np.nan], [3.0]])
    text = xr.runs_to_csv(R(), plan)
    lines = list(csv.reader(text.splitlines()))
    assert lines[2][2] == MU and lines[2][-1] == ""


def test_worker_reuses_a_finished_run(qapp, tmp_path):
    """A finished run with the same model parameters is read back and
    copied into the new plan folder instead of launched."""
    from gui.sensitivity.run_worker import SensitivityRunWorker
    from gui.sensitivity import sweep_plan as sp
    cfg = ModelConfig()
    mu = pr.spec_for(MU)
    ref = pr.get_display(cfg, mu)
    plan = sp.build_plan([(mu, ref, [ref, ref * 2])], "sweep")
    old = tmp_path / "old"
    old.mkdir()
    # the reference run, finished in an earlier folder
    params = cfg.to_params_dict()
    (old / "sweep_run000.meta.json").write_text(
        json.dumps({"model_config": params}), encoding="utf-8")
    (old / "sweep_run000.results.npz").write_bytes(b"npz")
    (old / "sweep_run000.sta").write_text(
        " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n", encoding="latin-1")
    new = tmp_path / "new"
    new.mkdir()
    w = SensitivityRunWorker(plan, "sweep", [], cfg, abaqus_cmd="x",
                             abaqus_script="y", workdir=str(new),
                             job_prefix="sweep", cache_folders=[str(old)])
    launched = []
    w._abaqus_solve = lambda c, i: launched.append(i) or None
    import gui.sensitivity.run_worker as rw
    loaded = []

    class _Fake:
        @staticmethod
        def load(p):
            loaded.append(p)
            return "bundle"
    orig = rw.ResultsBundle
    rw.ResultsBundle = _Fake
    try:
        cfgs = sp.plan_to_configs(cfg, plan)
        assert w._solve_with_cache(cfgs[0], 0) == "bundle"
        assert w._solve_with_cache(cfgs[1], 1) is None
    finally:
        rw.ResultsBundle = orig
    assert launched == [1] and w.n_reused == 1
    assert any((new / n).exists() or (new / "sweep_run000" / n).exists()
               for n in ("sweep_run000.results.npz",))


def test_compare_two_selected_runs(qapp, tmp_path):
    from PySide6.QtCore import QItemSelectionModel
    tab = _tab("grid", (MU, VC))
    for p, vals in ((MU, "0.2 0.4"), (VC, "20; 40")):
        tab.table.item(_row(tab, p), 9).setText(vals)
    tab._on_generate()
    res = _fake_result(tab)
    tab._last_result = res
    tab._run_workdir = tmp_path
    tab._show_results(res)
    assert not tab.btn_compare_runs.isEnabled()
    sm = tab.results_table.selectionModel()
    for r in (3, 0):                       # run 4, then the Ref run
        sm.select(tab.results_table.model().index(r, 0),
                  QItemSelectionModel.Select | QItemSelectionModel.Rows)
    assert tab.btn_compare_runs.isEnabled()
    got = []
    tab.compareRunsRequested.connect(lambda *a: got.append(a))
    tab.btn_compare_runs.click()
    assert got == [(str(tmp_path), "grid_run000", "grid_run003")]
