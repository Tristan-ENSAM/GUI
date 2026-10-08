# -*- coding: utf-8 -*-
"""Model tab made readable for a first-time user: essential inputs visible,
the others in collapsed 'Advanced parameters' sections that count the fields
changed from their default, and sourced defaults."""
from __future__ import annotations

import pytest

from gui.core.model_config import ModelConfig, OptimizationCfg
from gui.tabs.optimization_tab import OptimizationTab


@pytest.fixture
def tab(qapp):
    return OptimizationTab(ModelConfig())


def test_default_tolerances_are_the_project_decision():
    # eps_q decision of 2026-10-07: 10 mm/s, 10 K, 0.1, 10 N/mm.
    assert OptimizationCfg().criterion_rmse == {
        "Vx": "10", "Vy": "10", "T": "10", "EVF": "0.1",
        "Fc": "10", "Ff": "10"}


def test_new_profile_can_run_without_typing_tolerances(tab):
    assert tab.thresholds_complete()
    assert tab.thresholds() == {"Vx": 10.0, "Vy": 10.0, "T": 10.0,
                                "EVF": 0.1, "Fc": 10.0, "Ff": 10.0}


def test_gci_plan_defaults_to_four_meshes(tab):
    assert tab.sp_gci_n.value() == 4
    assert tab.le_gci_ratio.text() == "2"


def test_advanced_sections_start_collapsed_and_unchanged(tab):
    assert set(tab._advanced) == {"common", "ms", "mesh", "domain"}
    for sec, _fields in tab._advanced.values():
        assert not sec.is_expanded()
        assert sec.body.isHidden()
        assert sec.n_changed() == 0
        assert "changed" not in sec.header_text()


def test_essential_inputs_stay_outside_the_advanced_sections(tab):
    bodies = [sec.body for sec, _f in tab._advanced.values()]

    def inside(w):
        return any(b.isAncestorOf(w) for b in bodies)

    for w in (*tab.le_zoi.values(), *tab._q_eps.values(), tab.le_ms_values,
              tab.le_gci_finest, tab.sp_gci_n, tab.btn_ms, tab.btn_mesh,
              tab.btn_domain, tab.btn_checks, tab.btn_init):
        assert not inside(w)
    for w in (tab.le_grid_step, tab.le_ms_elem, tab.le_gci_ratio,
              tab.le_gci_min, tab.sp_margin, *tab._max.values(),
              *tab._dom_spins.values(), *tab._dom_texts.values()):
        assert inside(w)


def test_hidden_change_is_counted_on_the_header(tab):
    sec = tab._advanced["domain"][0]
    tab.sp_margin.setValue(3)
    tab._max["l_wp"].setText("0.5")
    assert sec.n_changed() == 2
    assert "2 changed" in sec.header_text()
    tab.sp_margin.setValue(0)
    tab._max["l_wp"].setText("")
    assert sec.n_changed() == 0


def test_count_ignores_number_formatting(tab):
    sec = tab._advanced["common"][0]
    tab._dom_texts["window_start"].setText("0,30")     # = default 0.3
    assert sec.n_changed() == 0
    tab._dom_texts["rk_max"].setText("0.01")
    assert sec.n_changed() == 1


def test_loaded_profile_with_non_default_values_is_flagged(qapp):
    cfg = ModelConfig()
    cfg.optimization.rk_max = "0.01"
    cfg.optimization.gci_ratio = "1.5"
    t = OptimizationTab(cfg)
    assert t._advanced["common"][0].n_changed() == 1
    assert t._advanced["mesh"][0].n_changed() == 1


def test_expanding_shows_the_fields(tab):
    sec = tab._advanced["mesh"][0]
    sec.set_expanded(True)
    assert sec.is_expanded() and not sec.body.isHidden()
    sec.set_expanded(False)
    assert sec.body.isHidden()
