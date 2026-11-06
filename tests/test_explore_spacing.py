# -*- coding: utf-8 -*-
"""Explore spacings by step and by ratio (design 2026-11-06): Min and Max
always in the plan, a shorter last interval, per-row settings."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt

from gui.core.model_config import ModelConfig
from gui.sensitivity import sweep_plan as sw

MU = "interaction.friction_coeff"
MS = "step.mass_scaling_factor"


def test_step_keeps_min_and_max():
    assert np.allclose(sw.levels_step(0.1, 0.5, 0.15), [0.1, 0.25, 0.4, 0.5])
    assert np.allclose(sw.levels_step(0.0, 1.0, 0.25),
                       [0, 0.25, 0.5, 0.75, 1.0])       # Max not doubled
    assert np.allclose(sw.levels_step(0.0, 1.0, 5.0), [0.0, 1.0])


def test_ratio_keeps_min_and_max():
    v = sw.levels_ratio(1, 25600, 2)
    assert v[0] == 1 and v[-2] == 16384 and v[-1] == 25600 and len(v) == 16
    assert np.allclose(sw.levels_ratio(1, 10000, 10), [1, 10, 100, 1e3, 1e4])


@pytest.mark.parametrize("call", [
    lambda: sw.levels_step(0.5, 0.1, 0.1),
    lambda: sw.levels_step(0, 1, 0),
    lambda: sw.levels_step(0, 1, 1e-6),          # too many values
    lambda: sw.levels_ratio(0, 1, 2),
    lambda: sw.levels_ratio(1, 10, 1.0),
    lambda: sw.row_levels(0, 1, "linear_n", 2.5),
])
def test_bad_settings_raise(call):
    with pytest.raises(ValueError):
        call()


def test_spacing_record():
    assert sw.spacing_record("linear_step", 0.15) == "linear step=0.15"
    assert sw.spacing_record("geometric_ratio", 2) == "geometric ratio=2"
    assert sw.spacing_record("geometric_n", 5) == "geometric"


def _row(tab, path):
    return next(r for r, s in tab._row_spec.items() if s.path == path)


def _explore():
    from gui.tabs.sensitivity_tab import SensitivityTab, EXPLORE_METHODS
    return SensitivityTab(ModelConfig(), methods=EXPLORE_METHODS)


def test_row_spacing_in_the_table(qapp):
    tab = _explore()
    r = _row(tab, MU)
    tab.table.item(r, 0).setCheckState(Qt.Checked)
    tab.table.item(r, 3).setText("0.1")
    tab.table.item(r, 4).setText("0.5")
    for c in (11, 12, 13):
        assert not tab.table.isColumnHidden(c)
    cb = tab.table.cellWidget(r, 11)
    cb.setCurrentIndex(cb.findData("linear_step"))
    assert tab.table.item(r, 12).text() == "0.1"        # (Max - Min) / 4
    tab.table.item(r, 12).setText("0.15")
    sel, spacing = tab._collect_sweep()
    assert np.allclose(sel[0][2], [0.1, 0.25, 0.4, 0.5])
    assert spacing == ["linear step=0.15"]
    assert tab.table.item(r, 13).text() == "0.1; 0.25; 0.4; 0.5"
    # a typed list wins over the row's spacing
    tab.table.item(r, 9).setText("0.2 0.3")
    sel, spacing = tab._collect_sweep()
    assert np.allclose(sel[0][2], [0.2, 0.3]) and spacing == ["typed"]
    tab.table.item(r, 9).setText("")
    # a bad step shows '?' in Levels and in the cost
    tab.table.item(r, 12).setText("-1")
    assert tab.table.item(r, 13).text() == "?"
    assert "?" in tab.lbl_cost.text()


def test_ratio_row_and_default_row_in_one_plan(qapp):
    tab = _explore()
    rm, ru = _row(tab, MS), _row(tab, MU)
    for r in (rm, ru):
        tab.table.item(r, 0).setCheckState(Qt.Checked)
    tab.table.item(rm, 3).setText("1")
    tab.table.item(rm, 4).setText("25600")
    cb = tab.table.cellWidget(rm, 11)
    cb.setCurrentIndex(cb.findData("geometric_ratio"))
    assert tab.table.item(rm, 12).text() == "2"
    tab.table.item(ru, 3).setText("0.1")
    tab.table.item(ru, 4).setText("0.5")
    tab.spin_nlev.setValue(3)                          # the default row
    sel, spacing = tab._collect_sweep()
    by = {s.path: lv for s, _r, lv in sel}
    assert len(by[MS]) == 16 and by[MS][-1] == 25600
    assert np.allclose(by[MU], [0.1, 0.3, 0.5])
    assert sorted(spacing) == ["geometric ratio=2", "linear"]
    tab._on_generate()
    assert tab.plan is not None, tab.status.text()
    rec = tab._plan_record([], 4)
    vp = {d["path"]: d for d in rec["varied_parameters"]}
    assert vp[MS]["spacing"] == "geometric ratio=2"


def test_spacing_columns_hidden_in_sensitivity(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab, SENSITIVITY_METHODS
    tab = SensitivityTab(ModelConfig(), methods=SENSITIVITY_METHODS)
    for c in (9, 11, 12, 13):
        assert tab.table.isColumnHidden(c)
