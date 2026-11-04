# -*- coding: utf-8 -*-
"""Optimization sub-tabs Setup / Model sizing / Parameter plans / Compare
runs (decision 2026-11-04): the comparison settings moved to Setup without
changing what is saved, Compare runs replaces the deviation-maps window."""
from __future__ import annotations

from gui.core.model_config import ModelConfig
from gui.tabs.optimization_tab import OptimizationTab
from gui.tabs.setup_tab import SetupTab


def _inside(widget, ancestor) -> bool:
    w = widget
    while w is not None:
        if w is ancestor:
            return True
        w = w.parentWidget()
    return False


def test_comparison_settings_shown_in_setup_not_in_sizing(qapp):
    opt = OptimizationTab(ModelConfig())
    setup = SetupTab(opt)
    for w in (opt.le_zoi["xmin"], opt._q_eps["Vx"], opt.le_grid_step,
              opt._dom_texts["window_start"]):
        assert _inside(w, setup) and not _inside(w, opt)
    # weights of the steps and guards stay with the steps
    for w in (opt._eps_share["ms"], opt._dom_texts["rk_max"],
              opt.cb_reuse_runs):
        assert _inside(w, opt) and not _inside(w, setup)
    assert _inside(opt.setup_preview, setup)


def test_settings_saved_with_the_same_keys(qapp):
    """A profile written after the move reloads identically (no key
    changed): every moved field goes through cfg.optimization."""
    cfg = ModelConfig()
    opt = OptimizationTab(cfg)
    vals = {"xmin": "-0.1", "xmax": "0.05", "ymin": "-0.02", "ymax": "0.06"}
    for k, v in vals.items():
        opt.le_zoi[k].setText(v)
    opt._q_eps["Vx"].setText("12")
    opt._dom_texts["window_start"].setText("0.4")
    opt._dom_texts["rk_max"].setText("0.07")
    opt._eps_share["mesh"].setText("2")
    opt.le_grid_step.setText("0.003")
    opt._sync_opt_to_cfg()
    again = OptimizationTab(cfg)
    for k, v in vals.items():
        assert again.le_zoi[k].text() == v
    assert again._q_eps["Vx"].text() == "12"
    assert again._dom_texts["window_start"].text() == "0.4"
    assert again._dom_texts["rk_max"].text() == "0.07"
    assert again._eps_share["mesh"].text() == "2"
    assert again.le_grid_step.text() == "0.003"


def test_advanced_counts_split_between_the_two_groups(qapp):
    opt = OptimizationTab(ModelConfig())
    assert {"common", "sizing"} <= set(opt._advanced)
    opt._dom_texts["window_start"].setText("0.5")
    opt._dom_texts["rhg_max"].setText("0.2")
    common = [w for w, _d in opt._advanced["common"][1]()]
    sizing = [w for w, _d in opt._advanced["sizing"][1]()]
    assert opt._dom_texts["window_start"] in common
    assert opt._dom_texts["rhg_max"] in sizing
    assert opt._dom_texts["rhg_max"] not in common


def test_setup_help_toggles_the_settings_hints(qapp):
    opt = OptimizationTab(ModelConfig())
    setup = SetupTab(opt)
    setup.show()
    assert opt.setup_hints
    setup.cb_help.setChecked(True)
    assert all(lab.isVisible() for lab in setup._hints)
    setup.cb_help.setChecked(False)
    assert not any(lab.isVisible() for lab in setup._hints)


def test_compare_button_opens_the_compare_tab(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    assert not hasattr(w.optimization_tab, "btn_dev_maps")
    w.optimization_tab.btn_compare.click()
    assert w.tabs.currentWidget() is w.opt_tabs
    assert w.opt_tabs.currentWidget() is w.compare_tab


def test_compare_tab_uses_the_setup_settings(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    s = w.compare_tab._settings_getter
    assert s == w.optimization_tab.compare_settings
