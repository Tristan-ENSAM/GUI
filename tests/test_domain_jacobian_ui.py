# -*- coding: utf-8 -*-
"""The Optimization tab's domain-sizing launcher.

The study costs 5 Abaqus runs per iteration, so its guards must fire BEFORE
anything is launched.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from gui.core.domain_sizing import DomainDims
from gui.core.model_config import ModelConfig
from gui.sensitivity.domain_jacobian import JacobianResult
from gui.tabs.optimization_tab import OptimizationTab


@pytest.fixture
def tab(qapp, monkeypatch):
    t = OptimizationTab(ModelConfig())
    # _validate_launch touches Preferences/Abaqus paths: neutralise it, the
    # geometry guards run before it matters.
    monkeypatch.setattr(t, "_validate_launch", lambda: (None, None, None))
    return t


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: seen.append(a[2])))
    return seen


def _set_domain(cfg, elem, side):
    cfg.elem_size = elem
    g = cfg.euler_geometry
    g.h_wp = g.h_void = g.l_wp = g.l_void = side


class TestLaunchGuards:
    def test_refuses_a_domain_past_the_reverberation_ceiling(self, tab, warnings):
        # Fine mesh + large domain: no mass-scaling factor could satisfy both
        # bounds, so growing the domain further is pointless.
        _set_domain(tab.cfg, 0.002, 0.15)
        tab._on_run_domain_jacobian()
        assert warnings and "exceeds the limit" in warnings[0]
        assert not hasattr(tab, "_dj_worker")

    def test_refuses_without_any_elasticity_threshold(self, tab, warnings):
        # The elasticity thresholds are SEPARATE from the physical eps_q:
        # an elasticity is dimensionless, so reusing "1 mm/s" as a bound is
        # meaningless. Clearing them must block the launch.
        _set_domain(tab.cfg, 0.0075, 0.10)
        for le in tab._dj_eps.values():
            le.setText("")
        tab._on_run_domain_jacobian()
        assert warnings and "threshold" in warnings[0].lower()
        assert not hasattr(tab, "_dj_worker")

    def test_elasticity_thresholds_are_dimensionless_and_default_to_2pct(self, tab):
        assert set(tab._dj_eps) == {"EVF", "TEMP", "V1", "V2", "force"}
        assert all(le.text() == "0.02" for le in tab._dj_eps.values())


class TestResultRendering:
    def _result(self, **kw):
        base = dict(
            dims=DomainDims(0.13, 0.10, 0.10, 0.10),
            elasticities={"EVF": {"h_wp": 0.004, "h_void": 0.011,
                                  "l_wp": 0.003, "l_void": 0.002}},
            converged=True, limiting="h_void", guard_coefficient=2.1e-6,
            linearity={}, n_runs=9, stopped_by="converged")
        base.update(kw)
        return JacobianResult(**base)

    def test_reports_convergence_and_the_guard_coefficient(self, tab):
        tab._on_domain_jacobian_done([self._result()])
        txt = tab.log.toPlainText()
        assert "converged" in txt
        assert "2.100e-06" in txt          # G must be surfaced: it grows with
                                           # the domain and moves the ms window

    def test_flags_a_failed_linearity_check(self, tab):
        # A ratio far from 1 means a discrete event dominated the difference,
        # not the boundary -- the elasticity is then meaningless.
        # ratio ~2.0 is the signature of a step-independent difference:
        # the quantity measures decorrelated noise, not boundary influence.
        tab._on_domain_jacobian_done(
            [self._result(converged=False, stopped_by="nonlinear",
                          linearity={"V1": {"h_void": 2.00}})])
        txt = tab.log.toPlainText()
        assert "LINEARITY CHECK FAILED" in txt
        assert "noise, not sensitivity" in txt

    def test_reports_the_reverberation_ceiling_as_a_stop_cause(self, tab):
        tab._on_domain_jacobian_done(
            [self._result(converged=False, stopped_by="diagonal")])
        assert "reverberation ceiling" in tab.log.toPlainText()

    def test_empty_history_is_not_a_crash(self, tab):
        tab._on_domain_jacobian_done([])
        assert "no result" in tab.lbl_status.text().lower()
