# -*- coding: utf-8 -*-
"""Simulate sub-tabs (Tristan 2026-11-16): Materials symbols, BCs velocity
faces as a choice, Mesh derived quantities under the preview, Step with
the time window in the Dynamic step and advanced mass scaling / filter."""
from __future__ import annotations

from gui.core.model_config import ModelConfig


def _inside(widget, ancestor):
    w = widget
    while w is not None:
        if w is ancestor:
            return True
        w = w.parentWidget()
    return False


def test_material_labels_without_bracketed_names(qapp):
    from gui.tabs.materials_tab import MaterialsTab
    t = MaterialsTab(ModelConfig())
    lbl = t._wp_widgets["E"]._lbl
    assert "(" not in lbl.text() and lbl.text().startswith("E [")
    assert lbl.toolTip() == "E (Young)"
    assert t._wp_widgets["beta"]._lbl.text() == "β"


def test_cutting_velocity_faces_are_a_choice(qapp):
    from gui.tabs.bcs_tab import BCsTab
    cfg = ModelConfig()
    t = BCsTab(cfg)
    w = t._face_widgets
    assert w["bottom"]["velocity"].isChecked()          # default eul_bot
    w["left"]["velocity"].setChecked(True)
    assert cfg.bcs.cutting_velocity_faces == ["eul_left", "eul_bot"]
    w["bottom"]["velocity"].setChecked(False)
    assert cfg.bcs.cutting_velocity_faces == ["eul_left"]
    cfg.bcs.cutting_velocity_faces = ["eul_top"]
    t.apply_from_cfg()
    assert w["top"]["velocity"].isChecked()
    assert not w["left"]["velocity"].isChecked()
    assert not t.preview.picking_enabled
    assert t.rb_bc.text() == "Boundary conditions"
    assert t.rb_ic.text() == "Initial conditions"


def test_mesh_derived_quantities_under_the_preview(qapp):
    from gui.tabs.mesh_tab import MeshTab
    t = MeshTab(ModelConfig())
    right = t.preview.parentWidget()
    assert _inside(t.lbl_nel, right)


def test_step_time_window_and_advanced_groups(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    st, opt = w.step_tab, w.optimization_tab
    dyn = st.f_sim_time.parentWidget()
    assert dyn.title() == "Dynamic step"
    assert _inside(opt._dom_texts["window_start"], dyn)
    assert not st.advanced.is_expanded()
    assert _inside(st.f_ms_eul, st.advanced.body)
    assert _inside(st.cb_filter, st.advanced.body)
    assert st.f_ms_eul.parentWidget() is not st.cb_filter.parentWidget()
