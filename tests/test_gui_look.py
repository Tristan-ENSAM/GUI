# -*- coding: utf-8 -*-
"""Optimization sub-tabs in the order of the work, the shared theme and
the Sensitivity help toggle."""
from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from gui.core.model_config import ModelConfig
from gui.tabs.optimization_tab import OptimizationTab
from gui.tabs.sensitivity_tab import SensitivityTab
from gui.widgets import theme


def test_optimization_sub_tabs_in_work_order(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    names = [w.opt_tabs.tabText(i) for i in range(w.opt_tabs.count())]
    assert names == ["1 · Model sizing", "2 · Sensitivity",
                     "3 · Inverse identification"]
    assert w.opt_tabs.widget(0) is w.optimization_tab
    assert w.opt_tabs.widget(1) is w.sensitivity_tab


def test_style_sheet_is_complete():
    qss = theme.style_sheet()
    assert "{" in qss and "{{" not in qss       # every placeholder filled
    assert theme.ACCENT in qss


def test_run_buttons_are_primary(qapp):
    opt = OptimizationTab(ModelConfig())
    for b in (opt.btn_all, opt.btn_ms, opt.btn_mesh, opt.btn_domain,
              opt.btn_checks):
        assert b.property("primary") is True, b.text()
    sens = SensitivityTab(ModelConfig())
    assert sens.btn_run.property("primary") is True
    assert theme.mark_primary(QPushButton("x")).property("primary") is True


def test_sensitivity_intro_behind_show_help(qapp):
    tab = SensitivityTab(ModelConfig())
    tab.show()
    assert not tab.lbl_intro.isVisible()
    tab.cb_help.setChecked(True)
    assert tab.lbl_intro.isVisible()
    names = [tab.tabs_out.tabText(i) for i in range(tab.tabs_out.count())]
    assert names[:3] == ["Plan", "Log", "Table"]


def test_group_titles_bold_and_disabled_colours():
    assert "font-weight: 600; background" in theme.style_sheet()
    assert theme.DISABLED_BG in theme.style_sheet()


def test_deviation_window_fits_the_screen(qapp):
    from gui.widgets.deviation_maps_dialog import DeviationMapsDialog
    d = DeviationMapsDialog()
    g = qapp.primaryScreen().availableGeometry()
    assert d.height() == min(900, int(0.9 * g.height()))
    assert d.width() <= max(min(1250, int(0.9 * g.width())),
                            d.minimumSizeHint().width())
