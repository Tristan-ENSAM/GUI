# -*- coding: utf-8 -*-
"""Optimization sub-tabs (decisions 2026-11-04 and 2026-11-12): the Setup
tab is gone; its comparison settings are shown in Geometry (ZOI, chip line),
Mesh (sampling step), Step (window T) and at the top of Identify (eps_q,
step weights) without changing what is saved; Compare runs replaces the
deviation-maps window."""
from __future__ import annotations

from gui.core.model_config import ModelConfig
from gui.tabs.optimization_tab import OptimizationTab


def _inside(widget, ancestor) -> bool:
    w = widget
    while w is not None:
        if w is ancestor:
            return True
        w = w.parentWidget()
    return False


def test_comparison_settings_shown_where_they_belong(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    opt = w.optimization_tab
    where = [
        (opt.le_zoi["xmin"], w.geometry_tab),
        (opt.le_chip_y0, w.geometry_tab),
        (opt.le_grid_step, w.mesh_tab),
        (opt._dom_texts["window_start"], w.step_tab),
        (opt._q_eps["Vx"], w.identify_modes),
        (opt.le_chip_eps, w.identify_modes),
        (opt._eps_share["ms"], w.identify_modes),
    ]
    for widget, tab in where:
        assert _inside(widget, tab), widget
        assert not _inside(widget, opt)
    # eps_q sit above both modes, not inside the Model sizing page
    assert not _inside(opt._q_eps["Vx"], w.identify_modes.stack)
    # guards and run reuse stay with the steps
    for widget in (opt._dom_texts["rk_max"], opt.cb_reuse_runs):
        assert _inside(widget, opt)
    names = [w.opt_tabs.tabText(i) for i in range(w.opt_tabs.count())]
    assert "1 · Setup" not in names


def test_geometry_preview_draws_the_zoi(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    opt = w.optimization_tab
    for k, v in zip(("xmin", "xmax", "ymin", "ymax"),
                    ("-0.1", "0.05", "-0.02", "0.06")):
        opt.le_zoi[k].setText(v)
    labels = [str(p.get_label()) for p in w.geometry_tab.preview._ax.patches]
    assert "ZOI" in labels
    lines = [str(l.get_label()) for l in w.geometry_tab.preview._ax.lines]
    assert "Chip thickness line" in lines


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


def test_compare_tab_lists_the_plan_folders(qapp, tmp_path):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    for name in ("PC__p__grid__2026-11-04_10-00-00",
                 "PC__p__sweep__2026-11-04_11-00-00",
                 "PC__p__ms__2026-11-04_09-00-00"):
        (tmp_path / name).mkdir()
    w.prefs.default_workdir = str(tmp_path)
    got = [p.split("/")[-1].split("\\")[-1] for p in w._plan_folders()]
    assert got == ["PC__p__grid__2026-11-04_10-00-00",
                   "PC__p__sweep__2026-11-04_11-00-00"]
