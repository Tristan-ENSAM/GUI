# -*- coding: utf-8 -*-
"""
Regression tests for the UI features added on top of the sensitivity work:

  1. Sensitivity Ref column re-syncs to the current ModelConfig.
  2. Material user-preset save/update/delete (PresetLibrary level).
  3. Step tab stable-increment estimate (and its mass-scaling scaling).
  4. V and A are classified as nodal field outputs (not element).

Run:  QT_QPA_PLATFORM=offscreen python -m pytest tests/test_ui_features.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.core.model_config import ModelConfig
from gui.core.preferences import Preferences


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# 1. Sensitivity Ref re-sync
# ---------------------------------------------------------------------------
def _row_for(tab, path):
    for r, spec in tab._row_spec.items():
        if spec.path == path:
            return r
    raise AssertionError("row not found for %s" % path)


def test_sensitivity_ref_resync(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    cfg = ModelConfig()
    tab = SensitivityTab(cfg, prefs_getter=lambda: Preferences(),
                         cpus_getter=lambda: 1)
    r = _row_for(tab, "euler_material.E")
    ref0 = float(tab.table.item(r, 2).text())

    # Edit the model (as another tab would) and re-sync.
    cfg.euler_material["E"] = cfg.euler_material["E"] * 2.0
    tab._resync_reference_values()
    ref1 = float(tab.table.item(r, 2).text())
    assert ref1 == pytest.approx(2.0 * ref0, rel=1e-6)


def test_sensitivity_resync_preserves_checked_row(qapp):
    from gui.tabs.sensitivity_tab import SensitivityTab
    cfg = ModelConfig()
    tab = SensitivityTab(cfg, prefs_getter=lambda: Preferences(),
                         cpus_getter=lambda: 1)
    r = _row_for(tab, "euler_material.E")
    # Tick "Vary" and set a 10% relative step + a custom trust region.
    tab.table.item(r, 0).setCheckState(Qt.Checked)
    tab._set_cell(r, 3, "1.0")        # Min
    tab._set_cell(r, 4, "1000.0")     # Max
    tab._set_cell(r, 6, "10.0")       # Delta% = 10%
    tab._sync_row(r, 6)               # -> Delta column
    cfg.euler_material["E"] = 200000.0   # 200 GPa
    tab._resync_reference_values()
    new_ref = float(tab.table.item(r, 2).text())
    # Min/Max preserved; Delta rescaled to 10% of the new Ref.
    assert tab.table.item(r, 3).text() == "1.0"
    assert tab.table.item(r, 4).text() == "1000.0"
    assert float(tab.table.item(r, 6).text()) == pytest.approx(10.0, rel=1e-6)
    assert float(tab.table.item(r, 5).text()) == pytest.approx(
        0.10 * abs(new_ref), rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Material preset save / update / delete
# ---------------------------------------------------------------------------
def test_preset_save_update_delete(tmp_path, monkeypatch):
    import gui.core.presets as presets
    user_file = tmp_path / "materials_user.json"
    monkeypatch.setattr(presets, "_user_path", lambda: user_file)

    lib = presets.PresetLibrary()
    mat = {"rho": 8.96e-09, "E": 124000.0, "nu": 0.34}
    lib.save_user_preset("workpiece", "MyCopper", mat)
    assert "MyCopper" in lib.list_presets("workpiece")
    assert lib.is_user_defined("workpiece", "MyCopper")
    assert lib.get("workpiece", "MyCopper")["E"] == 124000.0

    # Update (overwrite) the same user preset.
    lib.save_user_preset("workpiece", "MyCopper", {**mat, "E": 130000.0})
    assert lib.get("workpiece", "MyCopper")["E"] == 130000.0

    # Delete it.
    assert lib.delete_user_preset("workpiece", "MyCopper") is True
    assert "MyCopper" not in lib.list_presets("workpiece")
    # Deleting a non-existent / built-in preset returns False.
    assert lib.delete_user_preset("workpiece", "MyCopper") is False

    # Persistence: a fresh library re-reads the (now empty) user file.
    lib2 = presets.PresetLibrary()
    assert "MyCopper" not in lib2.list_presets("workpiece")


# ---------------------------------------------------------------------------
# 3. Step stable-increment estimate
# ---------------------------------------------------------------------------
def test_step_stable_increment(qapp):
    from gui.tabs.step_tab import StepTab
    cfg = ModelConfig()
    st = StepTab(cfg)
    dt, n = st._stable_increment_estimate()
    assert dt is not None and dt > 0.0
    # Δt ≈ Lₑ / √(E/ρ)
    E = cfg.euler_material["E"]; rho = cfg.euler_material["rho"]
    expected = cfg.elem_size / (E / rho) ** 0.5
    assert dt == pytest.approx(expected, rel=1e-9)
    assert n == pytest.approx(cfg.step.sim_time / dt, rel=1e-9)

    # Mass scaling ×100 → c_d / 10 → Δt ×10.
    st.cb_ms_enabled.setChecked(True)
    st.f_ms_eul.set_value(100.0)
    dt2, n2 = st._stable_increment_estimate()
    assert dt2 == pytest.approx(10.0 * dt, rel=1e-6)
    assert n2 == pytest.approx(n / 10.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. V / A are nodal outputs
# ---------------------------------------------------------------------------
def test_v_and_a_are_nodal(qapp):
    from gui.tabs.step_tab import StepTab
    nodal = [a for a, _, _ in StepTab.FIELD_VARS["Nodal (always useful)"]]
    eul = [a for a, _, _ in StepTab.FIELD_VARS["Eulerian-specific (element)"]]
    assert "fo_V" in nodal and "fo_A" in nodal
    assert "fo_V" not in eul and "fo_A" not in eul
    assert "fo_EVF" in eul   # EVF stays an element quantity


# ---------------------------------------------------------------------------
# 5. Materials -> Sensitivity sync actually fires on tab change (the bug)
# ---------------------------------------------------------------------------
def test_sensitivity_resync_fires_on_tab_change(qapp):
    """The bug: switching to Sensitivity did not refresh its Ref column.
    Reproduce MainWindow's wiring (currentChanged on both nested tab
    widgets) and check the Ref updates — without the heavyweight (and, on
    close, modal-dialog-blocking) full MainWindow."""
    from PySide6.QtWidgets import QTabWidget, QWidget
    from gui.tabs.sensitivity_tab import SensitivityTab
    cfg = ModelConfig()
    st = SensitivityTab(cfg, prefs_getter=lambda: Preferences(),
                        cpus_getter=lambda: 1)
    opt = QTabWidget(); opt.addTab(st, "Sensitivity")
    top = QTabWidget()
    top.addTab(QWidget(), "Numerical Model")
    top.addTab(opt, "Optimization")

    def refresh(*_):
        if top.currentWidget() is opt and opt.currentWidget() is st:
            st.refresh_from_model()
    top.currentChanged.connect(refresh)
    opt.currentChanged.connect(refresh)

    top.show(); qapp.processEvents()

    def ref_E():
        for r, spec in st._row_spec.items():
            if spec.path == "euler_material.E":
                return float(st.table.item(r, 2).text())
        raise AssertionError("E row not found")

    before = ref_E()
    cfg.euler_material["E"] = 250000.0   # as another tab would edit it
    top.setCurrentWidget(opt)            # fires currentChanged -> refresh
    qapp.processEvents()
    assert ref_E() == pytest.approx(250.0, rel=1e-6)
    assert before != pytest.approx(250.0, rel=1e-6)


class TestMeshTabToolSeeds:
    """The Mesh tab must expose the tool nose seed and the bias sizes (they were
    only reachable through the optimization pipeline before)."""

    def _tab(self):
        from gui.tabs.mesh_tab import MeshTab
        from gui.core.model_config import ModelConfig
        return MeshTab(ModelConfig())

    def test_fields_exist_and_write_back_to_cfg(self):
        tab = self._tab()
        tab.f_tool_es.set_value(0.004)
        tab.f_inter.set_value(0.015)
        tab.f_max.set_value(0.06)
        tab._pull_from_widgets()
        assert tab.cfg.tool_elem_size == pytest.approx(0.004)
        assert tab.cfg.inter_elem_size == pytest.approx(0.015)
        assert tab.cfg.max_elem_size == pytest.approx(0.06)

    def test_apply_from_cfg_round_trip(self):
        tab = self._tab()
        tab.cfg.tool_elem_size = 0.0025
        tab.cfg.max_elem_size = 0.04
        tab.apply_from_cfg()
        assert tab.f_tool_es.value() == pytest.approx(0.0025)
        assert tab.f_max.value() == pytest.approx(0.04)

    def test_seed_preview_adds_artists(self):
        tab = self._tab()
        tab._refresh()
        labels = [ln.get_label() for ln in tab.preview._ax.get_lines()]
        assert any("seed" in (l or "").lower() for l in labels)

    def test_tool_conduction_label_is_rendered(self):
        tab = self._tab()
        tab.cfg.tool_material.update({"k": 46.0, "rho": 1.5e-8, "Cp": 2.03e8})
        tab.cfg.tool_elem_size = 0.001
        tab._refresh()
        assert "s" in tab.lbl_dt_tool.text()
        assert tab.lbl_dt_tool.text() != "-"
