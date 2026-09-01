# -*- coding: utf-8 -*-
"""
Tests for gui.sensitivity.domain_jacobian -- Eulerian domain sizing by
forward-difference Jacobian.

The run_bundle is analytic: boundary influence on the ROI decays as
exp(-d/lambda), so the plateau is reachable and its location is known.
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.core.domain_sizing import DomainDims
import gui.sensitivity.domain_jacobian as dj
from gui.sensitivity.domain_jacobian import (
    DIMENSION_NAMES, diagonal, diagonal_limit, guard_coefficient,
    run_domain_study, jacobian_at)


class _Bundle:
    """Boundary influence decays with each dimension; ALLIE accumulates."""

    def __init__(self, dims, lam=0.05):
        self.dims, self.lam = dims, lam

    def _influence(self):
        return sum(np.exp(-getattr(self.dims, n) / self.lam)
                   for n in DIMENSION_NAMES)

    def field(self, inst, var):
        return np.linspace(0, 1, 60).reshape(3, 20) * (1 + 0.5 * self._influence())

    def history(self, var):
        if var == "ALLKE":
            return np.full(50, 1e-4)
        if var == "ALLIE":
            return np.linspace(1e-3, 6e-2, 50)
        return np.full(50, 100.0 * (1 + 0.5 * self._influence()))

    def history_time(self):
        return np.linspace(0, 3e-4, 50)


class _Cfg:
    class _G:
        h_wp = h_void = l_wp = l_void = 0.0

    def __init__(self):
        self.euler_geometry = _Cfg._G()


@pytest.fixture(autouse=True)
def _patch_sampling(monkeypatch):
    """Bypass the real ROI resampling: the analytic bundle IS the ROI sample."""
    monkeypatch.setattr(dj, "nearest_samples",
                        lambda b, var, inst, pts: b.field(inst, var))
    monkeypatch.setattr(dj, "roi_grid", lambda roi, step: np.zeros((20, 2)))


def _runner(lam=0.05):
    def run(cfg):
        g = cfg.euler_geometry
        return _Bundle(DomainDims(g.h_wp, g.h_void, g.l_wp, g.l_void), lam)
    return run


class TestGeometryConstraint:
    def test_diagonal_limit_scales_with_element_size(self):
        # The mass-scaling window closes unless diagonal < 90.6 * h.
        assert diagonal_limit(0.005) == pytest.approx(0.453)
        assert diagonal_limit(0.010) == pytest.approx(0.906)

    def test_oversized_initial_domain_runs_nothing(self):
        # Must be caught BEFORE spending a single simulation.
        hist = run_domain_study(
            _runner(), _Cfg(), (0, 1, 0, 1), DomainDims(.15, .15, .15, .15),
            0.01, {"EVF": 1e-9}, elem_size=0.002, field_vars=("EVF",),
            linearity_check=False)
        assert hist[-1].stopped_by == "diagonal"
        assert hist[-1].n_runs == 0

    def test_growth_stops_at_the_ceiling(self):
        # Unreachable threshold: it can only stop on the diagonal ceiling.
        hist = run_domain_study(
            _runner(lam=10.0), _Cfg(), (0, 1, 0, 1),
            DomainDims(.10, .10, .10, .10), 0.01, {"EVF": 1e-12},
            elem_size=0.0035, field_vars=("EVF",), max_iterations=30,
            linearity_check=False)
        assert hist[-1].stopped_by == "diagonal"
        assert diagonal(hist[-1].dims) <= diagonal_limit(0.0035)


class TestConvergence:
    def test_short_range_influence_converges(self):
        hist = run_domain_study(
            _runner(lam=0.005), _Cfg(), (0, 1, 0, 1),
            DomainDims(.10, .10, .10, .10), 0.01, {"EVF": 0.02},
            elem_size=0.0075, field_vars=("EVF",), max_iterations=6,
            linearity_check=False)
        assert hist[-1].converged is True
        assert hist[-1].stopped_by == "converged"

    def test_elasticities_decrease_as_domain_grows(self):
        hist = run_domain_study(
            _runner(lam=0.05), _Cfg(), (0, 1, 0, 1),
            DomainDims(.10, .10, .10, .10), 0.01, {"EVF": 1e-6},
            elem_size=0.0075, field_vars=("EVF",), max_iterations=5,
            linearity_check=False)
        worst = [max(r.elasticities["EVF"].values()) for r in hist
                 if r.elasticities]
        assert worst[-1] < worst[0]

    def test_nan_counts_as_not_converged(self):
        # A quantity that cannot be compared must never read as "converged".
        class _Null(_Bundle):
            def field(self, inst, var):
                return np.zeros((3, 20))       # zero norm -> NaN
        hist = run_domain_study(
            lambda cfg: _Null(DomainDims(.1, .1, .1, .1)), _Cfg(),
            (0, 1, 0, 1), DomainDims(.1, .1, .1, .1), 0.01, {"EVF": 0.5},
            elem_size=0.0075, field_vars=("EVF",), max_iterations=2,
            linearity_check=False)
        assert hist[0].converged is False


class TestJacobian:
    def test_one_elasticity_per_dimension(self):
        j = jacobian_at(_runner(), _Cfg(), DomainDims(.1, .1, .1, .1),
                        (0, 1, 0, 1), 0.01, ("EVF",), elem_size=0.0075)
        assert set(j["elasticities"]["EVF"]) == set(DIMENSION_NAMES)
        assert j["n_runs"] == 5          # base + one per dimension

    def test_forces_are_a_separate_quantity(self):
        j = jacobian_at(_runner(), _Cfg(), DomainDims(.1, .1, .1, .1),
                        (0, 1, 0, 1), 0.01, ("EVF",), elem_size=0.0075)
        assert "force" in j["elasticities"]


class TestGuardCoefficient:
    def test_uses_aggregated_energies_not_instantaneous_ratio(self):
        # ALLIE accumulates, so the instantaneous ratio explodes early on;
        # the aggregated ratio over the settled window must be far smaller.
        b = _Bundle(DomainDims(.1, .1, .1, .1))
        g = guard_coefficient(b, mass_scaling_factor=1000.0)
        ke, ie = b.history("ALLKE"), b.history("ALLIE")
        assert g == pytest.approx(1e-4 / ie[ie.size // 3 * 1:].mean() / 1000.0,
                                  rel=0.5)
        assert g < (ke[0] / ie[0]) / 1000.0      # far below the t=0 ratio

    def test_inversely_proportional_to_mass_scaling(self):
        b = _Bundle(DomainDims(.1, .1, .1, .1))
        assert (guard_coefficient(b, 1000.0)
                == pytest.approx(10 * guard_coefficient(b, 10000.0)))

    def test_returns_none_without_energy_history(self):
        class _NoHist:
            def history(self, var):
                raise KeyError(var)
            def history_time(self):
                return np.zeros(0)
        assert guard_coefficient(_NoHist(), 1000.0) is None
