# -*- coding: utf-8 -*-
"""The restructured Optimization tab: ZOI panel + the two study launchers
(mesh GCI, domain convergence). Guards must fire before any Abaqus launch.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from gui.core.model_config import ModelConfig
from gui.tabs.optimization_tab import OptimizationTab


@pytest.fixture
def tab(qapp, monkeypatch):
    t = OptimizationTab(ModelConfig())
    # Neutralise the Preferences/Abaqus path check; the input guards run first.
    monkeypatch.setattr(t, "_validate_launch", lambda: ("prefs", "wd", 1))
    return t


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: seen.append(a[2])))
    return seen


class TestRestructuredTab:
    def test_constructs_with_new_panels(self, tab):
        assert tab.btn_mesh is not None and tab.btn_domain is not None
        assert set(tab.le_zoi) == {"xmin", "xmax", "ymin", "ymax"}
        assert tab.le_gci_finest is not None
        assert tab.sp_gci_n.value() >= 3
        assert set(tab._dj_eps) == {"EVF", "TEMP", "V1", "V2", "force"}

    def test_zoi_defaults_to_roi_when_blank(self, tab):
        roi = tab.config_inputs()["roi"]
        assert tab.zoi() == pytest.approx(roi)

    def test_set_zoi_from_roi_fills_fields(self, tab):
        tab._zoi_from_roi()
        assert tab.le_zoi["xmin"].text() != ""
        assert tab.zoi() == pytest.approx(tab.config_inputs()["roi"])

    def test_domain_convergence_requires_a_tolerance(self, tab, warnings):
        for le in tab._dj_eps.values():
            le.setText("")
        tab._on_run_domain_convergence()
        assert warnings and "tolerance" in warnings[0].lower()
        assert not hasattr(tab, "_dc_worker")

    def test_tolerances_default_to_2pct(self, tab):
        assert tab._tolerances() == {q: 0.02 for q in
                                     ("EVF", "TEMP", "V1", "V2", "force")}

    def test_busy_toggles_the_study_buttons(self, tab):
        tab._busy(True, "running")
        assert not tab.btn_mesh.isEnabled()
        assert not tab.btn_domain.isEnabled()
        assert tab.btn_cancel.isEnabled()
        tab._busy(False)
        assert tab.btn_mesh.isEnabled()
        assert not tab.btn_cancel.isEnabled()


class TestRunOutputWiring:
    def test_profile_name_from_getter(self, qapp):
        tab = OptimizationTab(ModelConfig(), profile_name_getter=lambda: "myprof")
        assert tab._profile_name() == "myprof"

    def test_profile_name_defaults_untitled(self, qapp):
        tab = OptimizationTab(ModelConfig())
        assert tab._profile_name() == "Untitled"

    def test_study_run_dir_creates_folder_and_config(self, qapp, tmp_path):
        from pathlib import Path
        tab = OptimizationTab(ModelConfig(), profile_name_getter=lambda: "prof")
        d = tab._study_run_dir(tmp_path, "GCI", {"finest": 0.006})
        assert Path(d).parent == Path(tmp_path)
        assert Path(d).name.startswith("prof_GCI_")
        assert (Path(d) / "config.json").exists()
