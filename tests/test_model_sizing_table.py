# -*- coding: utf-8 -*-
"""Identify > Model sizing as one table of the numerical parameters
(Tristan 2026-11-16): bounds, method, its parameters, reference and weight
per row; the QoI compared are ticked at the top of Identify; the energy
guards are fixed at 0.05."""
from __future__ import annotations

import pytest

from gui.core.model_config import ModelConfig
from gui.tabs import optimization_tab as ot


@pytest.fixture
def tab(qapp):
    return ot.OptimizationTab(ModelConfig())


def test_one_row_per_numerical_parameter(tab):
    t = tab.sizing_table
    assert t.rowCount() == 3
    assert [t.item(r, ot.S_NAME).text() for r in range(3)] == [
        "Mass scaling ms", "Element size h", "Domain D"]
    heads = [t.horizontalHeaderItem(c).text()
             for c in range(t.columnCount())]
    assert heads[:7] == ["Parameter", "Min", "Max", "Method", "Parameters",
                         "Reference", "Weight"]
    # weights, bounds and run buttons live in the table
    for w in (*tab._eps_share.values(), tab.le_ms_max, tab.le_gci_min,
              tab.btn_ms, tab.btn_mesh, tab.btn_domain, tab.cb_ms_method,
              tab.cb_dom_method):
        assert t.isAncestorOf(w)
    # no "Sizing settings" box any more
    from PySide6.QtWidgets import QGroupBox
    titles = [g.title() for g in tab.findChildren(QGroupBox)]
    assert not any(x.startswith("Sizing settings") for x in titles)
    assert not any(x.startswith("Step 0") for x in titles)


def test_selected_row_shows_its_method_parameters(tab):
    for r in (2, 1, 0):
        tab.sizing_table.selectRow(r)
        assert tab.method_stack.currentIndex() == r
    tab.sizing_table.selectRow(1)
    assert tab.method_stack.currentWidget().isAncestorOf(tab.le_gci_finest)
    assert "Element size h" in tab.method_box.title()


def test_ms_method_column_sets_search_and_goal(tab):
    tab.cb_ms_method.setCurrentIndex(tab.cb_ms_method.findData("sweep"))
    assert (tab.ms_search(), tab.ms_goal()) == ("list", "sweep")
    tab.cb_ms_method.setCurrentIndex(tab.cb_ms_method.findData("list"))
    assert (tab.ms_search(), tab.ms_goal()) == ("list", "identify")
    assert tab.cb_ms_reference.isVisibleTo(tab)
    tab.cb_ms_method.setCurrentIndex(tab.cb_ms_method.findData("bisection"))
    assert (tab.ms_search(), tab.ms_goal()) == ("bisection", "identify")
    assert not tab.cb_ms_reference.isVisibleTo(tab)
    assert tab.lbl_ms_ref.text() == "ms = 1"
    # a study read back sets the column
    tab.set_ms_goal("sweep")
    assert tab.cb_ms_method.currentData() == "sweep"


def test_summaries_follow_the_method_parameters(tab):
    tab.sp_gci_n.setValue(3)
    tab.le_gci_finest.setText("0.001")
    tab.le_gci_ratio.setText("2")
    assert tab.lbl_h_max.text() == "0.004"
    assert tab.lbl_h_params.text() == "3 meshes, ratio 2"
    tab.le_ms_factor.setText("4")
    assert tab.lbl_ms_params.text().startswith("×4")
    tab._max["h_void"].setText("0.4")
    assert tab.lbl_d_max.text() == "1 cap(s)"


def test_domain_method_switches_the_run_button(tab):
    assert tab.btn_domain.isVisibleTo(tab)
    assert not tab.btn_dom_rich.isVisibleTo(tab)
    tab.set_dom_method("richardson")
    assert tab.btn_dom_rich.isVisibleTo(tab)
    assert tab.lbl_d_ref.text() == "Richardson"
    assert tab.cfg.optimization.dom_method == "richardson"


def test_guards_are_fixed(tab):
    g = tab.guard_settings()
    assert (g.rk_max, g.rhg_max) == (0.05, 0.05)
    assert set(tab._dom_texts) == {"window_start", "window_end"}


def test_ticked_qoi_only_are_compared(tab):
    assert tab.sizing_qois() == ["Vx", "Vy", "T", "EVF", "Fc", "Ff"]
    tab._q_use["Ff"].setChecked(False)
    tab._q_use["EVF"].setChecked(False)
    assert set(tab.thresholds()) == {"Vx", "Vy", "T", "Fc"}
    assert tab.thresholds_complete()
    # the other tabs keep every eps_q
    assert set(tab.all_thresholds()) == {"Vx", "Vy", "T", "EVF", "Fc", "Ff"}
    assert "Ff" in tab.model_settings()["eps"]
    # an unticked QoI needs no eps_q; a ticked one does
    tab._q_eps["Ff"].setText("")
    assert tab.thresholds_complete()
    tab._q_eps["Vx"].setText("")
    assert not tab.thresholds_complete()
    assert "Vx" in tab._eps_missing_text()
    for cb in tab._q_use.values():
        cb.setChecked(False)
    assert not tab.thresholds_complete()
    assert "Tick at least one QoI" in tab._eps_missing_text()


def test_ms_spec_holds_the_ticked_qoi(tab, monkeypatch):
    tab.cfg.step.output_filter_enabled = True
    tab._q_use["T"].setChecked(False)
    spec = tab._ms_spec()
    assert spec is not None
    assert set(spec["thresholds_abs"]) == {"Vx", "Vy", "EVF", "Fc", "Ff"}
    assert tab._gci_tolerances().keys() == {"EVF", "V1", "V2", "Fc", "Ff"}


def test_qoi_ticks_and_method_saved_with_the_profile(qapp):
    cfg = ModelConfig()
    t = ot.OptimizationTab(cfg)
    t._q_use["Vy"].setChecked(False)
    t.set_dom_method("richardson")
    cfg2 = ModelConfig.from_json_dict(cfg.to_json_dict())
    assert cfg2.optimization.sizing_qois == ["Vx", "T", "EVF", "Fc", "Ff"]
    t2 = ot.OptimizationTab(cfg2)
    assert t2.sizing_qois() == ["Vx", "T", "EVF", "Fc", "Ff"]
    assert t2.dom_method() == "richardson"


def test_reading_a_study_back_ticks_its_qoi(tab):
    tab._restore_settings("mesh", {
        "zoi": [float(v) for v in tab.zoi()],
        "thresholds_abs": {"Vx": 10.0, "T": 10.0}})
    assert tab.sizing_qois() == ["Vx", "T"]


def test_ticks_only_in_model_sizing_mode(tab):
    tab.set_sizing_mode(False)
    assert tab._q_use["Vx"].isHidden() and not tab._q_lbl["Vx"].isHidden()
    tab.set_sizing_mode(True)
    assert not tab._q_use["Vx"].isHidden() and tab._q_lbl["Vx"].isHidden()
