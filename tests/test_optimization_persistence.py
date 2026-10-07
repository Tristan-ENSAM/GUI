# -*- coding: utf-8 -*-
"""The Optimization tab parameters and the full mesh config round-trip through
the .acpf profile (ModelConfig JSON), and sync between the tab widgets and
cfg.optimization."""
from __future__ import annotations

import pytest

from gui.core.model_config import ModelConfig, OptimizationCfg
from gui.tabs.optimization_tab import OptimizationTab


class TestConfigRoundTrip:
    def test_optimization_and_mesh_round_trip(self):
        cfg = ModelConfig()
        cfg.tool_elem_size = 0.0012
        cfg.inter_elem_size = 0.021
        cfg.max_elem_size = 0.055
        o = cfg.optimization
        o.zoi = {"xmin": "-0.05", "xmax": "0.05", "ymin": "-0.04", "ymax": "0.04"}
        o.criterion_rmse = {"Vx": "1.0", "T": "2"}
        o.sizing_tol = {"EVF": "0.03", "TEMP": "0.01", "V1": "0.02",
                        "V2": "0.02", "force": "0.02"}
        o.gci_finest, o.gci_ratio, o.gci_min, o.gci_n_meshes = \
            "0.005", "2", "0.006", 4
        o.caps = {"l_wp": "0.5", "h_wp": ""}
        o.margin_elems, o.centroid_step = 2, "0.01"

        cfg2 = ModelConfig.from_json_dict(cfg.to_json_dict())

        assert (cfg2.tool_elem_size, cfg2.inter_elem_size, cfg2.max_elem_size) \
            == (0.0012, 0.021, 0.055)
        o2 = cfg2.optimization
        assert o2.zoi == o.zoi
        assert o2.criterion_rmse == o.criterion_rmse
        assert o2.sizing_tol == o.sizing_tol
        assert (o2.gci_finest, o2.gci_ratio, o2.gci_min, o2.gci_n_meshes) \
            == ("0.005", "2", "0.006", 4)
        assert o2.caps == o.caps
        assert (o2.margin_elems, o2.centroid_step) == (2, "0.01")

    def test_legacy_file_without_optimization_uses_defaults(self):
        d = {"format_version": ModelConfig().FORMAT_VERSION,
             "mesh": {"elem_size": 0.004}}
        cfg = ModelConfig.from_json_dict(d)
        assert cfg.elem_size == 0.004
        assert isinstance(cfg.optimization, OptimizationCfg)
        assert cfg.optimization.gci_ratio == "2"          # default

    def test_save_load_file_round_trip(self, tmp_path):
        cfg = ModelConfig()
        cfg.optimization.gci_finest = "0.007"
        cfg.optimization.zoi["xmin"] = "-0.1"
        cfg.max_elem_size = 0.066
        p = tmp_path / "prof.acpf"
        cfg.save_to(p)
        cfg2 = ModelConfig.load_from(p)
        assert cfg2.optimization.gci_finest == "0.007"
        assert cfg2.optimization.zoi["xmin"] == "-0.1"
        assert cfg2.max_elem_size == 0.066


class TestTabPersistence:
    def test_widgets_write_into_cfg(self, qapp):
        tab = OptimizationTab(ModelConfig())
        tab.le_gci_finest.setText("0.008")
        tab.le_zoi["xmin"].setText("-0.06")
        tab._dj_eps["EVF"].setText("0.05")
        tab.sp_gci_n.setValue(5)
        tab.sp_margin.setValue(3)
        o = tab.cfg.optimization
        assert o.gci_finest == "0.008"
        assert o.zoi["xmin"] == "-0.06"
        assert o.sizing_tol["EVF"] == "0.05"
        assert o.gci_n_meshes == 5
        assert o.margin_elems == 3

    def test_cfg_restores_into_a_fresh_tab(self, qapp):
        cfg = ModelConfig()
        cfg.optimization.gci_finest = "0.009"
        cfg.optimization.zoi = {"xmin": "-0.07", "xmax": "0.07",
                                "ymin": "-0.05", "ymax": "0.05"}
        cfg.optimization.sizing_tol = {"EVF": "0.04", "TEMP": "0.02",
                                       "V1": "0.02", "V2": "0.02",
                                       "force": "0.02"}
        cfg.optimization.gci_n_meshes = 6
        cfg.optimization.margin_elems = 4
        tab = OptimizationTab(cfg)
        assert tab.le_gci_finest.text() == "0.009"
        assert tab.le_zoi["xmin"].text() == "-0.07"
        assert tab._dj_eps["EVF"].text() == "0.04"
        assert tab.sp_gci_n.value() == 6
        assert tab.sp_margin.value() == 4

    def test_changed_signal_emitted_on_edit(self, qapp):
        tab = OptimizationTab(ModelConfig())
        seen = []
        tab.changed.connect(lambda: seen.append(1))
        tab.le_gci_finest.setText("0.01")
        assert seen

    def test_loading_does_not_emit_changed(self, qapp):
        cfg = ModelConfig()
        cfg.optimization.gci_finest = "0.003"
        tab = OptimizationTab(cfg)
        seen = []
        tab.changed.connect(lambda: seen.append(1))
        tab.refresh_inputs()          # re-load from cfg must not dirty
        assert not seen



class TestDomainStudyPersistence:
    def test_round_trip_and_widgets(self, qapp):
        cfg = ModelConfig()
        o = cfg.optimization
        o.dom_step_elems, o.dom_n_max, o.dom_n_hold, o.dom_m_ratios = 7, 9, 2, 3
        o.window_start, o.window_end = "0.4", "0.95"
        o.rk_max, o.rhg_max = "0.02", "0.06"
        cfg2 = ModelConfig.from_json_dict(cfg.to_json_dict())
        o2 = cfg2.optimization
        assert (o2.dom_step_elems, o2.dom_n_max, o2.dom_n_hold,
                o2.dom_m_ratios) == (7, 9, 2, 3)
        assert (o2.window_start, o2.window_end) == ("0.4", "0.95")
        assert (o2.rk_max, o2.rhg_max) == ("0.02", "0.06")
        tab = OptimizationTab(cfg2)
        assert tab._dom_spins["dom_step_elems"].value() == 7
        assert tab._dom_texts["rhg_max"].text() == "0.06"
        tab._dom_spins["dom_n_hold"].setValue(3)
        tab._dom_texts["window_end"].setText("0.9")
        assert tab.cfg.optimization.dom_n_hold == 3
        assert tab.cfg.optimization.window_end == "0.9"

    def test_legacy_profile_gets_the_decided_defaults(self):
        cfg = ModelConfig.from_json_dict(
            {"format_version": ModelConfig().FORMAT_VERSION,
             "optimization": {"gci_ratio": "2"}})
        o = cfg.optimization
        assert (o.dom_step_elems, o.dom_n_max, o.dom_n_hold,
                o.dom_m_ratios) == (10, 8, 1, 2)
        assert (o.window_start, o.window_end) == ("0.3", "1.0")
        assert o.rhg_max == "0.05"


def test_ms_study_settings_round_trip(qapp):
    cfg = ModelConfig()
    assert cfg.optimization.ms_values == "250, 500, 1000, 2000, 4000"
    tab = OptimizationTab(cfg)
    tab.le_ms_values.setText("500, 1000")
    tab.le_ms_elem.setText("0.004")
    assert (cfg.optimization.ms_values, cfg.optimization.ms_elem_size) == \
        ("500, 1000", "0.004")
    cfg2 = ModelConfig.from_json_dict(cfg.to_json_dict())
    tab2 = OptimizationTab(cfg2)
    assert tab2.le_ms_values.text() == "500, 1000"
    assert tab2.le_ms_elem.text() == "0.004"
