# -*- coding: utf-8 -*-
"""Identify tab, mode Physical parameters: tables, reference, settings,
and a whole identification with a fake solver (the worker loop, the
history file, 'Use in model')."""
from __future__ import annotations

import json

import pytest
from PySide6.QtCore import Qt

from gui.core.model_config import ModelConfig
from gui.identify import engine as ie
from gui.identify import worker as iw
from gui.sensitivity import model_qoi as mq
from gui.sensitivity import param_registry as pr

MU = "interaction.friction_coeff"
VC = "bcs.cutting_speed"


def _settings(cfg):
    b = cfg.bbox
    zoi = (float(b.xmin), 0.5 * (float(b.xmin) + float(b.xmax)),
           float(b.ymin), 0.5 * (float(b.ymin) + float(b.ymax)))
    return {"eps": {"Vx": 10.0, "Vy": 10.0, "T": 10.0, "EVF": 0.1,
                    "Fc": 10.0, "Ff": 10.0},
            "window": (0.3, 1.0), "zoi": zoi, "evf_threshold": 0.5}


def _tab(cfg=None):
    from gui.tabs.identify_tab import IdentifyTab
    cfg = cfg or ModelConfig()
    tab = IdentifyTab(cfg)
    tab.set_model_settings_getter(lambda: _settings(tab.cfg))
    return tab


def _row(tab, path):
    return next(r for r, s in tab._row_spec.items() if s.path == path)


def _fake_q(cfg):
    """Fc and Ff depend on mu and Vc; the other QoI are constant."""
    mu = cfg.interaction.friction_coeff
    vc = pr.get_display(cfg, pr.spec_for(VC))          # m/min
    return {"Fc": 400 * mu + 2.0 * vc, "Ff": 300 * mu - 0.5 * vc,
            "Vx": 0.0, "Vy": 0.0, "T": 300.0, "EVF": 1.0}


def test_tables_and_collect(qapp):
    tab = _tab()
    assert len(tab._row_spec) == 71
    r = _row(tab, MU)
    assert tab.table.item(r, 2).text() == "0.3"
    tab.table.item(r, 0).setCheckState(Qt.Checked)
    tab.table.item(r, 3).setText("0.1")
    tab.table.item(r, 4).setText("0.6")
    specs, params = tab.collect_params()
    assert [s.path for s in specs] == [MU]
    assert params[0].start == pytest.approx(0.3)       # empty Start = Ref
    tab.table.item(r, 5).setText("0.2")
    assert tab.collect_params()[1][0].start == pytest.approx(0.2)
    tab.table.item(r, 5).setText("0.9")                # outside [Min, Max]
    with pytest.raises(ValueError):
        tab.collect_params()
    # eps_q comes from Setup; a reference is needed for each ticked QoI
    assert tab.ref_table.item(mq.MODEL_QOIS.index("Fc"), 3).text() == "10"
    with pytest.raises(ValueError, match="reference"):
        tab.collect_targets()
    tab.set_reference({q: 1.0 for q in mq.MODEL_QOIS}, "typed")
    qois, targets, st = tab.collect_targets()
    # The chip thickness is not ticked by default.
    assert qois == list(mq.SIZING_QOIS) and len(targets) == 6
    assert targets[0].eps == 10.0 and st["window"] == (0.3, 1.0)


def test_controls_follow_the_strategy(qapp):
    tab = _tab()
    tab.cb_strategy.setCurrentIndex(tab.cb_strategy.findData("lm"))
    assert not tab.cb_fd.isHidden() and tab.spin_line.isHidden()
    tab.cb_strategy.setCurrentIndex(tab.cb_strategy.findData("fixed_point"))
    assert tab.cb_fd.isHidden() and not tab.spin_line.isHidden()
    s = tab.search_settings()
    assert s["strategy"] == "fixed_point" and s["tau"] == ie.DEFAULT_TAU
    assert s["line_tol"] == pytest.approx(0.01)


def test_reference_from_a_run(qapp, monkeypatch, tmp_path):
    tab = _tab()
    import gui.results.reader as rd
    monkeypatch.setattr(rd.ResultsBundle, "load", staticmethod(lambda p: "b"))
    monkeypatch.setattr(iw, "point_qoi",
                        lambda b, specs: {s.id: 5.0 for s in specs})
    assert tab.reference_from_run(str(tmp_path / "ref.results.npz"))
    assert tab.ref_table.item(0, 2).text() == "5"
    assert "ref.results.npz" in tab.lbl_ref.text()


def test_identification_with_a_fake_solver(qapp, monkeypatch, tmp_path):
    """mu and Vc are found from the Fc / Ff of a reference model, the
    history is written, and 'Use in model' writes them into the cfg."""
    cfg = ModelConfig()
    tab = _tab(cfg)
    true = ModelConfig()
    true.interaction.friction_coeff = 0.45
    pr.apply_display(true, pr.spec_for(VC), 45.0)
    ref = _fake_q(true)
    for path, lo, hi in ((MU, 0.1, 0.8), (VC, 20.0, 80.0)):
        r = _row(tab, path)
        tab.table.item(r, 0).setCheckState(Qt.Checked)
        tab.table.item(r, 3).setText(str(lo))
        tab.table.item(r, 4).setText(str(hi))
    for i, q in enumerate(mq.MODEL_QOIS):
        if q not in ("Fc", "Ff"):
            tab.ref_table.item(i, 0).setCheckState(Qt.Unchecked)
    tab.set_reference(ref, "reference model")
    tab.spin_budget.setValue(60)
    specs, params = tab.collect_params()
    qois, targets, st = tab.collect_targets()
    monkeypatch.setattr(iw, "point_qoi", lambda cfg_, specs_: _fake_q(cfg_))
    seen = []
    w = tab.make_worker(specs, params, targets, qois, st, tmp_path,
                        solve_fn=lambda c, i: seen.append(i) or c)
    its = []
    w.iteration.connect(its.append)
    tab._run_specs, tab._run_workdir = specs, tmp_path
    w.run()
    res = w.result
    assert res.status == ie.CONVERGED, res.message
    assert res.max_ratio <= 1.0 and its
    assert seen == list(range(res.n_runs))
    rec = json.loads((tmp_path / iw.RECORD_NAME).read_text(encoding="utf-8"))
    assert rec["study"] == "identify" and rec["status"] == ie.CONVERGED
    assert rec["settings"]["eps"]["Fc"] == 10.0
    tab._on_finished(res)
    assert "within" in tab.status.text() and tab.btn_apply.isEnabled()
    got = []
    tab.identified.connect(got.append)
    tab.btn_apply.click()
    assert set(got[0]) == {MU, VC}
    assert abs(_fake_q(cfg)["Fc"] - ref["Fc"]) <= 10.0
    assert abs(_fake_q(cfg)["Ff"] - ref["Ff"]) <= 10.0


def test_worker_reports_a_failed_run(qapp, monkeypatch, tmp_path):
    tab = _tab()
    r = _row(tab, MU)
    tab.table.item(r, 0).setCheckState(Qt.Checked)
    tab.set_reference({q: 1.0 for q in mq.MODEL_QOIS})
    specs, params = tab.collect_params()
    qois, targets, st = tab.collect_targets()
    w = tab.make_worker(specs, params, targets, qois, st, tmp_path,
                        solve_fn=lambda c, i: None)
    w.run()
    assert w.result.status == ie.FAILED


def test_main_window_has_identify(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    assert w.opt_tabs.widget(3) is w.identify_modes
    assert w.identify_tab._model_settings_getter is not None


def test_identify_modes_hold_sizing_and_physical(qapp):
    from gui.main import MainWindow
    from gui.tabs import identify_modes as im
    w = MainWindow(); w._dirty = False
    m = w.identify_modes
    # Model sizing first: it comes before the physical identification.
    assert m.mode() == im.MODE_SIZING
    assert m.current_page() is w.optimization_tab
    seen = []
    m.modeChanged.connect(seen.append)
    m.set_mode(im.MODE_PHYSICAL)
    assert m.current_page() is w.identify_tab
    assert seen == [im.MODE_PHYSICAL]
    m.show_page(w.optimization_tab)
    assert m.mode() == im.MODE_SIZING
    with pytest.raises(ValueError):
        m.set_mode("nope")


def test_physical_mode_refreshes_ref_column(qapp, monkeypatch):
    from gui.main import MainWindow
    from gui.tabs import identify_modes as im
    w = MainWindow(); w._dirty = False
    w.tabs.setCurrentWidget(w.opt_tabs)
    w.opt_tabs.setCurrentWidget(w.identify_modes)
    calls = []
    monkeypatch.setattr(w.identify_tab, "refresh_from_model",
                        lambda: calls.append(1))
    w.identify_modes.set_mode(im.MODE_PHYSICAL)
    assert calls == [1]
