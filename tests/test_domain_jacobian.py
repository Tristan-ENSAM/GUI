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

    def instance(self, name):
        # ResultsBundle exposes this; eulerian_instance() uses it to find the
        # instance carrying EVF instead of assuming a hard-coded name.
        class _Info:
            field_variables = ["EVF", "TEMP", "V1", "V2"]
        return _Info()

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
        if var == "RF1_RP":
            return np.full(50, 100.0 * (1 + 0.5 * self._influence()))
        raise KeyError(var)

    @property
    def history_time(self):
        # PROPERTY, not a method -- matches ResultsBundle. The first version of
        # this stub exposed it as a method, which hid a real bug: the module
        # called history_time() and the exception was swallowed, so every study
        # silently returned G = n/a.
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
    # The analytic stubs are a single instance; skip ODB name resolution.
    monkeypatch.setattr(dj, "eulerian_instance", lambda b: "EULER")


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
            @property
            def history_time(self):
                return np.zeros(0)
        assert guard_coefficient(_NoHist(), 1000.0) is None


class TestMatchesTheRealBundleApi:
    """Guards against the class of bug that made a whole 9-run study return
    NaN: the module calling names that ResultsBundle does not expose.

    The analytic stubs above cannot catch this on their own -- they mirror
    whatever the module happens to call. These tests check against the REAL
    ResultsBundle instead.
    """

    def test_history_time_is_a_property_not_a_method(self):
        from gui.results.reader import ResultsBundle
        attr = getattr(ResultsBundle, "history_time")
        assert isinstance(attr, property), (
            "domain_jacobian reads bundle.history_time without calling it; "
            "if it became a method the guard coefficient would silently "
            "return None")

    def test_default_field_names_exist_in_the_optimisation_mapping(self):
        # Velocity is written per component (V1/V2); asking for "V" yields
        # nothing and every elasticity comes back NaN.
        import inspect
        from gui.tabs.optimization_tab import _QUANTITIES
        written = {f for (_q, f, _u) in _QUANTITIES if f is not None}
        sig = inspect.signature(run_domain_study)
        defaults = sig.parameters["field_vars"].default
        assert set(defaults) <= written, (
            "default field_vars %r are not all produced by the generator "
            "(known: %r)" % (sorted(defaults), sorted(written)))

    def test_default_force_channel_matches_the_optimisation_mapping(self):
        import inspect
        from gui.tabs.optimization_tab import _FORCE_CHANNELS
        from gui.sensitivity.domain_jacobian import sample_domain
        default = inspect.signature(sample_domain).parameters[
            "force_channel"].default
        assert default in set(_FORCE_CHANNELS.values())

    def test_missing_quantities_raise_instead_of_returning_nan(self):
        # A 9-run study that returns NaN with no explanation is the worst
        # outcome; fail fast with the offending names instead.
        class _Empty:
            def field(self, inst, var):
                raise KeyError(var)
            def history(self, var):
                raise KeyError(var)
            @property
            def history_time(self):
                return np.zeros(5)

        with pytest.raises(RuntimeError, match="missing from the results"):
            dj.sample_domain(lambda cfg: _Empty(), _Cfg(),
                             DomainDims(.1, .1, .1, .1), (0, 1, 0, 1),
                             0.01, ("EVF",))


class TestInstanceNameResolution:
    """Abaqus upper-cases instance names in the ODB ("Euler" -> "EULER"), so
    the name must be read from the bundle, never assumed from the model.
    Hard-coding it made a real 9-run study fail with
    'Euler__element_centroids_init is not a file in the archive'.
    """

    class _Info:
        field_variables = ["EVF", "TEMP", "V1", "V2"]

    class _Upper:
        """Bundle whose Eulerian instance is spelled in upper case."""
        def __init__(self, dims):
            self.dims = dims
        def instance(self, name):
            return TestInstanceNameResolution._Info()
        def field(self, inst, var):
            assert inst == "EULER", "hard-coded instance name: %r" % inst
            return np.ones((3, 20))
        def history(self, var):
            return {"ALLKE": np.full(50, 1e-4),
                    "ALLIE": np.linspace(1e-3, 6e-2, 50),
                    "RF1_RP": np.full(50, 100.0)}[var]
        @property
        def history_time(self):
            return np.linspace(0, 3e-4, 50)

    def test_uppercase_instance_is_resolved(self, monkeypatch):
        import gui.sensitivity.runner_core as rc
        monkeypatch.setattr(rc, "_instance_names", lambda b: ["EULER", "TOOL"])
        monkeypatch.setattr(dj, "eulerian_instance", rc.eulerian_instance)
        s = dj.sample_domain(
            lambda cfg: TestInstanceNameResolution._Upper(
                DomainDims(.1, .1, .1, .1)),
            _Cfg(), DomainDims(.1, .1, .1, .1), (0, 1, 0, 1), 0.01,
            ("EVF",), mass_scaling_factor=1000.0)
        assert np.isfinite(s.fields["EVF"]).all()

    def test_no_eulerian_instance_raises(self, monkeypatch):
        monkeypatch.setattr(dj, "eulerian_instance", lambda b: None)
        with pytest.raises(RuntimeError, match="No Eulerian instance"):
            dj.sample_domain(
                lambda cfg: TestInstanceNameResolution._Upper(
                    DomainDims(.1, .1, .1, .1)),
                _Cfg(), DomainDims(.1, .1, .1, .1), (0, 1, 0, 1), 0.01,
                ("EVF",))


class TestLinearityStopsTheStudy:
    """J(h)/J(2h) ~ 2 means ||dQ|| does not depend on the step: the difference
    has saturated and measures decorrelated noise. Observed on a real campaign
    for V1/V2/force -- 2.00 on all 16 measurements -- while the study happily
    kept growing the domain. It must stop instead.
    """

    class _Saturated(_Bundle):
        """Difference independent of the perturbation: pure noise floor."""
        def field(self, inst, var):
            # value depends on the dims only through a hash -> any change
            # decorrelates it completely, whatever the step size
            seed = hash((round(self.dims.h_wp, 6), round(self.dims.h_void, 6),
                         round(self.dims.l_wp, 6), round(self.dims.l_void, 6)))
            rng = np.random.default_rng(abs(seed) % (2 ** 31))
            return rng.normal(size=(3, 20)) + 10.0

    def test_saturated_difference_stops_the_study(self, monkeypatch):
        monkeypatch.setattr(dj, "eulerian_instance", lambda b: "EULER")
        hist = run_domain_study(
            lambda cfg: TestLinearityStopsTheStudy._Saturated(
                DomainDims(cfg.euler_geometry.h_wp, cfg.euler_geometry.h_void,
                           cfg.euler_geometry.l_wp, cfg.euler_geometry.l_void)),
            _Cfg(), (0, 1, 0, 1), DomainDims(.1, .1, .1, .1), 0.01,
            {"EVF": 1e-6}, elem_size=0.005, field_vars=("EVF",),
            max_iterations=6, linearity_check=True)
        assert hist[-1].stopped_by == "nonlinear"
        # it must give up early rather than spend every iteration
        assert len(hist) < 6

    def test_tolerance_is_configurable(self, monkeypatch):
        monkeypatch.setattr(dj, "eulerian_instance", lambda b: "EULER")
        # A wide tolerance accepts the same noisy case (escape hatch, but the
        # default must be strict).
        hist = run_domain_study(
            lambda cfg: TestLinearityStopsTheStudy._Saturated(
                DomainDims(cfg.euler_geometry.h_wp, cfg.euler_geometry.h_void,
                           cfg.euler_geometry.l_wp, cfg.euler_geometry.l_void)),
            _Cfg(), (0, 1, 0, 1), DomainDims(.1, .1, .1, .1), 0.01,
            {"EVF": 1e-6}, elem_size=0.005, field_vars=("EVF",),
            max_iterations=2, linearity_check=True, linearity_tolerance=99.0)
        assert hist[-1].stopped_by != "nonlinear"
