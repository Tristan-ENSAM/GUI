# -*- coding: utf-8 -*-
"""Every model value in the parameter table (decision 2026-11-06), the
Note column, the filter, and the split of Parameter plans into Explore
(sweep, grid) and Sensitivity (Jacobian, Morris)."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from gui.core.model_config import ModelConfig
from gui.sensitivity import param_registry as pr
from gui.sensitivity import sweep_plan as sp


def _numeric_leaves(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _numeric_leaves(v, f"{p}.{k}" if p else k)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield p


def test_registry_covers_every_model_value():
    cfg = ModelConfig()
    assert len(pr.REGISTRY) == 71
    assert len({s.path for s in pr.REGISTRY}) == 71
    # every registered path resolves; only the optional conductance is unset
    unset = [s.path for s in pr.REGISTRY if not pr.is_set(cfg, s)]
    assert unset == ["interaction.gap_conductance"]
    # every numeric value written for a run has a spec (the two mass-scaling
    # entries come from one factor, the master heat share from the slave)
    written = set(_numeric_leaves(cfg.to_params_dict()))
    assert len(written) == 71
    tool_mat = {s.path for s in pr.REGISTRY if s.path.startswith("tool_material.")}
    assert tool_mat == {"tool_material." + k
                        for k in ("rho", "E", "nu", "k", "Cp", "alpha")}


def test_kinds_and_remesh_marks():
    by = {s.path: s for s in pr.REGISTRY}
    assert by["tool_geometry.r_tool"].remesh
    assert by["elem_size"].remesh and by["elem_size"].kind == "numerical"
    assert by["bbox.xmin"].kind == "output" and not by["bbox.xmin"].remesh
    assert by["step.n_frames"].dtype == "int"
    assert by["interaction.friction_coeff"].kind == "physical"


def test_heat_share_keeps_the_sum_one():
    cfg = ModelConfig()
    spec = pr.spec_for("interaction.heat_fraction_to_slave")
    pr.apply_display(cfg, spec, 0.8)
    assert cfg.interaction.heat_fraction_to_slave == 0.8
    assert abs(cfg.interaction.heat_fraction_to_master - 0.2) < 1e-12


def test_geometry_value_goes_into_a_plan_config():
    cfg = ModelConfig()
    r = pr.spec_for("tool_geometry.r_tool")
    plan = sp.build_plan([(r, 0.01, [0.01, 0.02])], "sweep")
    cfgs = sp.plan_to_configs(cfg, plan)
    assert [c.tool_geometry.r_tool for c in cfgs] == [0.01, 0.02]
    assert cfg.tool_geometry.r_tool == 0.01      # base untouched


def _row(tab, path):
    return next(r for r, s in tab._row_spec.items() if s.path == path)


def test_table_lists_all_with_notes_and_unset_row(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    assert len(tab._row_spec) == 71
    assert tab.table.item(_row(tab, "tool_geometry.r_tool"), 10).text() \
        == "remesh"
    assert tab.table.item(_row(tab, "bbox.xmin"), 10).text() == "output only"
    assert tab.table.item(_row(tab, "interaction.friction_coeff"), 10).text() \
        == ""
    r = _row(tab, "interaction.gap_conductance")
    assert tab.table.item(r, 2).text() == "not set"
    assert not (tab.table.item(r, 0).flags() & Qt.ItemIsUserCheckable)
    # once set in the model, the row can be varied
    tab.cfg.interaction.gap_conductance = 0.0
    tab.refresh_from_model()
    assert tab.table.item(r, 0).flags() & Qt.ItemIsUserCheckable
    assert tab.table.item(r, 2).text() != "not set"


def test_filter_and_search(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    mu = _row(tab, "interaction.friction_coeff")
    box = _row(tab, "bbox.xmin")
    tab.cb_filter.setCurrentIndex(tab.cb_filter.findData("physical"))
    assert not tab.table.isRowHidden(mu) and tab.table.isRowHidden(box)
    tab.cb_filter.setCurrentIndex(tab.cb_filter.findData("ticked"))
    assert tab.table.isRowHidden(mu)
    tab.table.item(mu, 0).setCheckState(Qt.Checked)
    tab._apply_filter()
    assert not tab.table.isRowHidden(mu) and tab.table.isRowHidden(box)
    tab.cb_filter.setCurrentIndex(tab.cb_filter.findData("all"))
    tab.le_search.setText("frottement")
    assert not tab.table.isRowHidden(mu) and tab.table.isRowHidden(box)
    tab.le_search.setText("")
    assert not tab.table.isRowHidden(box)


def test_explore_and_sensitivity_methods(qapp):
    from gui.tabs.sensitivity_tab import (
        SensitivityTab, EXPLORE_METHODS, SENSITIVITY_METHODS)
    ex = SensitivityTab(ModelConfig(), methods=EXPLORE_METHODS)
    se = SensitivityTab(ModelConfig(), methods=SENSITIVITY_METHODS)
    assert [ex.cb_method.itemData(i) for i in range(ex.cb_method.count())] \
        == ["sweep", "grid"]
    assert [se.cb_method.itemData(i) for i in range(se.cb_method.count())] \
        == ["jacobian", "morris"]
    assert ex._method() == "sweep" and se._method() == "jacobian"
    assert not ex.table.isColumnHidden(9) and ex.table.isColumnHidden(5)
    assert se.table.isColumnHidden(9) and not se.table.isColumnHidden(5)
    assert "Sweep" in ex.lbl_intro.text() and "Jacobian" not in ex.lbl_intro.text()
    assert "Jacobian" in se.lbl_intro.text() and "Sweep" not in se.lbl_intro.text()
    # ticks are kept per tab
    ex.table.item(_row(ex, "interaction.friction_coeff"), 0).setCheckState(
        Qt.Checked)
    assert se._selected_rows() == []


def test_main_window_wires_both_tabs(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    assert w.explore_tab is not w.sensitivity_tab
    assert w.explore_tab._model_settings_getter is not None
    assert w.sensitivity_tab._known_folders_getter is not None


def test_no_element_map_when_the_eulerian_mesh_changes():
    """elem_size gives another Eulerian mesh: its field sensitivity and map
    are NaN ('mesh changes'), a physical parameter keeps its map."""
    import numpy as np
    from gui.sensitivity import jacobian_plan as jac
    from gui.sensitivity import runner_core as rc
    specs = [pr.spec_for("elem_size"), pr.spec_for("interaction.friction_coeff")]
    plan = jac.build_plan([(specs[0], 0.005, 0.001, False),
                           (specs[1], 0.3, 0.03, False)], scheme="forward")

    class B:
        def __init__(self, v):
            self.v = v

        def field(self, inst, var):
            return np.full((3, 4), self.v)
    bundles = [B(float(i)) for i in range(plan.n_runs)]
    maps = rc.jacobian_field_maps(plan, bundles, ["V1"], instance="I")
    assert np.isnan(maps["V1"]["elem_size"]).all()
    assert not np.isnan(maps["V1"]["interaction.friction_coeff"]).any()
    fa = rc.jacobian_field_analysis(plan, bundles, ["V1"], instance="I")
    assert fa["V1"]["elem_size"]["scheme_used"] == "mesh changes"


@pytest.mark.parametrize("which", ["explore", "identify"])
def test_filter_by_category(qapp, which):
    from gui.sensitivity import param_registry as pr
    from gui.widgets.param_filter import CATEGORY_PREFIX
    if which == "explore":
        from gui.tabs.sensitivity_tab import SensitivityTab, EXPLORE_METHODS
        tab = SensitivityTab(ModelConfig(), methods=EXPLORE_METHODS)
    else:
        from gui.tabs.identify_tab import IdentifyTab
        tab = IdentifyTab(ModelConfig())
    cats = list(pr.registry_by_category())
    data = [tab.cb_filter.itemData(i) for i in range(tab.cb_filter.count())]
    assert [d for d in data if d and d.startswith(CATEGORY_PREFIX)] == [
        CATEGORY_PREFIX + c for c in cats]
    mu_cat = pr.spec_for("interaction.friction_coeff").category
    tab.cb_filter.setCurrentIndex(
        tab.cb_filter.findData(CATEGORY_PREFIX + mu_cat))
    for r, spec in tab._row_spec.items():
        assert tab.table.isRowHidden(r) == (spec.category != mu_cat)
    shown_heads = [h for h in tab._header_rows
                   if not tab.table.isRowHidden(h)]
    assert len(shown_heads) == 1
