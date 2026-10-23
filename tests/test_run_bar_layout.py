# -*- coding: utf-8 -*-
"""Run bar of the Optimization tabs: "Open working dir" beside the machine
choice in Model and Sensitivity, and the units of the admitted deviations
eps_q stay visible once the fields are filled."""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QLabel

from gui.core.model_config import ModelConfig
from gui.tabs.optimization_tab import OptimizationTab
from gui.tabs.sensitivity_tab import SensitivityTab
from gui.widgets import workdir_button as W


def _neighbours(w):
    """The widgets of the layout that holds `w`, in order."""
    lay = w.parentWidget().layout()

    def find(l):
        for i in range(l.count()):
            it = l.itemAt(i)
            if it.widget() is w:
                return l
            if it.layout() is not None:
                r = find(it.layout())
                if r is not None:
                    return r
        return None
    row = find(lay)
    assert row is not None
    return [row.itemAt(i).widget() for i in range(row.count())]


def test_model_tab_machine_beside_open_working_dir(qapp):
    tab = OptimizationTab(ModelConfig())
    row = _neighbours(tab.btn_open_wd)
    i = row.index(tab.btn_open_wd)
    assert isinstance(row[i + 1], QLabel) and row[i + 1].text() == "Machine:"
    assert row[i + 2] is tab.cmb_machine


def test_sensitivity_tab_has_open_working_dir_beside_machine(qapp):
    tab = SensitivityTab(ModelConfig())
    assert tab.btn_open_wd.text() == "Open working dir"
    row = _neighbours(tab.btn_open_wd)
    i = row.index(tab.btn_open_wd)
    assert row[i + 2] is tab.cmb_machine


def test_open_working_dir_opens_the_preferences_folder(qapp, tmp_path,
                                                       monkeypatch):
    opened = []
    monkeypatch.setattr(W.QDesktopServices, "openUrl",
                        lambda url: opened.append(url))
    wd = tmp_path / "wd"
    W.open_working_dir(None, SimpleNamespace(default_workdir=str(wd)))
    assert wd.is_dir()                       # created when missing
    assert opened == [QUrl.fromLocalFile(str(wd))]


def test_admitted_deviation_units_shown(qapp):
    tab = OptimizationTab(ModelConfig())
    want = {"Vx": "mm/s", "Vy": "mm/s", "T": "K", "EVF": "-",
            "Fc": "N/mm", "Ff": "N/mm"}
    for q, unit in want.items():
        le = tab._q_eps[q]
        le.setText("10")
        row = _neighbours(le)
        assert row[row.index(le) + 1].text() == unit, q
