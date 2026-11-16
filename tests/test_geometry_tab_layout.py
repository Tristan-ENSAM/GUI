# -*- coding: utf-8 -*-
"""Geometry tab (Tristan 2026-11-16): dense form with standard symbols,
parameter sketch, cutting line, a single ZOI (= extraction zone)."""
from __future__ import annotations


def test_sketch_names_every_parameter(qapp):
    from gui.widgets.geometry_sketch import GeometrySketch, LABELS
    sk = GeometrySketch()
    for key in ("h_tool", "l_tool", "r_tool", "rake", "clear", "h_wp",
                "l_wp", "h_void", "l_void", "depth", "zoi", "cut_line",
                "chip_line"):
        assert LABELS[key] in sk.texts, key


def test_geometry_tab_views_and_symbols(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    gt = w.geometry_tab
    assert [gt.views.tabText(i) for i in range(gt.views.count())] == [
        "Model (to scale)", "Parameter sketch"]
    assert gt.f_rake._lbl.text() == "γ"
    assert gt.f_clear._lbl.text() == "α"
    assert "β" in gt.f_r_tool._lbl.text()
    lines = [str(l.get_label()) for l in gt.preview._ax.lines]
    assert "Cutting line" in lines
    assert "t<sub>1</sub> = 0.05 mm" == gt.lbl_depth.text()
    assert not hasattr(gt, "f_xmin")          # one ZOI, no separate ROI


def test_zoi_fields_in_geometry_write_the_bbox(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    opt = w.optimization_tab
    opt.le_zoi["ymax"].setText("0.07")
    assert w.cfg.bbox.ymax == 0.07
    assert w._dirty
    patches = [p for p in w.geometry_tab.preview._ax.patches
               if p.get_label() == "ZOI"]
    assert len(patches) == 1
    assert patches[0].get_y() + patches[0].get_height() == \
        __import__("pytest").approx(0.07)
