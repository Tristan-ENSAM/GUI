# -*- coding: utf-8 -*-
"""
Unit tests for the sensitivity pipeline (gui.sensitivity.runner_core) and the
Jacobian plan (gui.sensitivity.jacobian_plan).

Fully offline: no Abaqus, no Qt. The solver is injected as `solve_fn`, so the
whole run_plan pipeline is exercised with a mock that returns lightweight fake
bundles whose QoI is an analytic function of the perturbed parameter — letting
the test check that the recovered Jacobian equals the closed-form derivative.

Includes a regression test for the historical `instance_names` bug: the helper
must tolerate `instance_names` exposed either as a property (the real
ResultsBundle) or as a method (a test double), without raising TypeError.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from gui.sensitivity import runner_core as rc
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import param_registry as pr
from gui.results.fake_builder import build_fake_results
from gui.results.reader import ResultsBundle
from gui.results.qoi import QoISpec


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeBundleValue:
    """Minimal bundle carrying one scalar; a QoISpec reads it back."""
    def __init__(self, value):
        self.value = value


class _MethodInstanceBundle:
    """Test double exposing instance_names as a METHOD (not a property) and a
    single Euler-like instance carrying EVF — the shape that triggered the
    historical TypeError when the code assumed a property."""
    def instance_names(self):                      # METHOD on purpose
        return ["Euler"]

    def instance(self, name):
        return SimpleNamespace(field_variables=["EVF", "TEMP"])


def _linear_qoi(slope=2.0, intercept=3.0):
    """A QoISpec whose value = slope * bundle.value + intercept."""
    return QoISpec("lin", "linear", "-",
                   lambda b, inst, wf: slope * b.value + intercept)


# ---------------------------------------------------------------------------
# instance_names / eulerian_instance — property vs method (regression)
# ---------------------------------------------------------------------------
class TestInstanceHelpers:

    def test_property_form_real_bundle(self, tmp_path):
        # The real ResultsBundle exposes instance_names as a PROPERTY.
        _, npz = build_fake_results(tmp_path / "j.results.npz",
                                    n_frames=3, n_grid_x=5, n_grid_y=4)
        with ResultsBundle.load(npz) as b:
            assert rc._instance_names(b) == ["Euler"]
            assert rc.eulerian_instance(b) == "Euler"   # Euler carries EVF

    def test_method_form_double(self):
        b = _MethodInstanceBundle()
        assert rc._instance_names(b) == ["Euler"]
        assert rc.eulerian_instance(b) == "Euler"

    def test_garbage_returns_empty_and_none(self):
        b = object()
        assert rc._instance_names(b) == []
        assert rc.eulerian_instance(b) is None


# ---------------------------------------------------------------------------
# extract_qois — NaN-safe
# ---------------------------------------------------------------------------
class TestExtractQois:

    def test_good_and_bad_qoi(self):
        good = QoISpec("g", "good", "-", lambda b, i, w: 1.0)

        def _raise(b, i, w):
            raise RuntimeError("boom")
        bad = QoISpec("b", "bad", "-", _raise)

        out = rc.extract_qois(_FakeBundleValue(0.0), [good, bad])
        assert out["g"] == 1.0
        assert np.isnan(out["b"])


# ---------------------------------------------------------------------------
# run_plan — end-to-end with a mock solver
# ---------------------------------------------------------------------------
def _speed_setup():
    """Base cfg + spec + plan for a single identity parameter (friction)."""
    base_cfg = SimpleNamespace(interaction=SimpleNamespace(friction_coeff=0.3))
    spec = pr.spec_for("interaction.friction_coeff")        # identity factor
    plan = jac.build_plan([(spec, 0.3, 0.1, False)], scheme="central")
    return base_cfg, spec, plan


class TestRunPlan:

    def test_jacobian_recovers_slope(self):
        base_cfg, spec, plan = _speed_setup()
        slope = 2.0

        def solve_fn(cfg, i):
            mu = pr.get_stored(cfg, "interaction.friction_coeff")
            return _FakeBundleValue(mu)

        res = rc.run_plan(plan, "jacobian", [_linear_qoi(slope)], solve_fn,
                          base_cfg)
        assert res.Y.shape == (plan.n_runs, 1)
        assert res.failures == []
        a = res.analyses["lin"]["interaction.friction_coeff"]
        # Central difference of a linear QoI returns the exact slope.
        assert a["dQdx"] == pytest.approx(slope, rel=1e-6)
        assert a["sensitivity"] == pytest.approx(slope, rel=1e-6)

    def test_failed_run_is_nan_and_recorded(self):
        base_cfg, spec, plan = _speed_setup()

        def solve_fn(cfg, i):
            if i == 1:
                return None                       # one diverged run
            return _FakeBundleValue(
                pr.get_stored(cfg, "interaction.friction_coeff"))

        res = rc.run_plan(plan, "jacobian", [_linear_qoi()], solve_fn, base_cfg)
        assert 1 in res.failures
        assert np.all(np.isnan(res.Y[1, :]))

    def test_should_cancel_stops_early(self):
        base_cfg, spec, plan = _speed_setup()
        calls = {"n": 0}

        def solve_fn(cfg, i):
            calls["n"] += 1
            return _FakeBundleValue(0.3)

        # Cancel immediately: no run should execute.
        res = rc.run_plan(plan, "jacobian", [_linear_qoi()], solve_fn, base_cfg,
                          should_cancel=lambda: True)
        assert calls["n"] == 0
        assert np.all(np.isnan(res.Y))

    def test_progress_called(self):
        base_cfg, spec, plan = _speed_setup()
        seen = []

        def solve_fn(cfg, i):
            return _FakeBundleValue(0.3)

        rc.run_plan(plan, "jacobian", [_linear_qoi()], solve_fn, base_cfg,
                    progress=lambda done, total: seen.append((done, total)))
        assert seen and seen[-1] == (plan.n_runs, plan.n_runs)

    def test_invalid_plan_kind_raises(self):
        base_cfg, spec, plan = _speed_setup()
        with pytest.raises(ValueError):
            rc.run_plan(plan, "not_a_kind", [_linear_qoi()],
                        lambda c, i: None, base_cfg)


# ---------------------------------------------------------------------------
# jacobian_ranking — pure ordering
# ---------------------------------------------------------------------------
class TestRanking:

    def test_sorted_by_abs_desc_nan_last(self):
        res = rc.RunResult(
            plan_kind="jacobian", qoi_ids=["q"], param_paths=["a", "b", "c"],
            Y=np.zeros((1, 1)),
            analyses={"q": {"a": {"sensitivity": 1.0},
                            "b": {"sensitivity": -5.0},
                            "c": {"sensitivity": float("nan")}}})
        rows = rc.jacobian_ranking(res, "q")
        paths = [p for p, _ in rows]
        assert paths[0] == "b" and paths[1] == "a" and paths[2] == "c"


# ---------------------------------------------------------------------------
# build_plan / plan_to_configs / analyze
# ---------------------------------------------------------------------------
class TestJacobianPlan:

    def test_n_runs_formula(self):
        assert jac.n_runs(3, "central") == 7
        assert jac.n_runs(3, "forward") == 4
        assert jac.n_runs(3, "backward") == 4

    def test_build_plan_errors(self):
        spec = pr.spec_for("elem_size")
        with pytest.raises(ValueError):
            jac.build_plan([(spec, 0.01, 0.001, False)], scheme="bogus")
        with pytest.raises(ValueError):
            jac.build_plan([], scheme="central")
        with pytest.raises(ValueError):
            jac.build_plan([(spec, 0.01, 0.0, False)], scheme="central")

    def test_plan_to_configs_applies_and_preserves_base(self):
        base_cfg = SimpleNamespace(
            interaction=SimpleNamespace(friction_coeff=0.3))
        spec = pr.spec_for("interaction.friction_coeff")
        plan = jac.build_plan([(spec, 0.3, 0.1, False)], scheme="central")
        configs = jac.plan_to_configs(base_cfg, plan)
        # central, 1 param: [base, +delta, -delta]
        vals = [pr.get_stored(c, "interaction.friction_coeff") for c in configs]
        assert vals[0] == pytest.approx(0.3)
        assert vals[plan.idx_plus[0]] == pytest.approx(0.4)
        assert vals[plan.idx_minus[0]] == pytest.approx(0.2)
        # The original cfg is untouched (deep-copied per run).
        assert base_cfg.interaction.friction_coeff == pytest.approx(0.3)

    def test_analyze_central(self):
        spec = pr.spec_for("elem_size")
        plan = jac.build_plan([(spec, 0.01, 0.1, False)], scheme="central")
        # Y = [Q0, Q+, Q-]
        Y = np.array([10.0, 11.0, 9.0])
        a = jac.analyze(plan, Y)["elem_size"]
        assert a["dQdx"] == pytest.approx((11.0 - 9.0) / (2 * 0.1))   # = 10

    def test_analyze_forward(self):
        spec = pr.spec_for("elem_size")
        plan = jac.build_plan([(spec, 0.01, 0.1, False)], scheme="forward")
        Y = np.array([10.0, 11.0])
        a = jac.analyze(plan, Y)["elem_size"]
        assert a["dQdx"] == pytest.approx((11.0 - 10.0) / 0.1)        # = 10

    def test_analyze_normalized(self):
        spec = pr.spec_for("interaction.friction_coeff")
        plan = jac.build_plan([(spec, 0.3, 0.1, True)], scheme="central")
        Y = np.array([10.0, 11.0, 9.0])
        a = jac.analyze(plan, Y)["interaction.friction_coeff"]
        # sensitivity = dQdx * (x0 / Q0) = 10 * (0.3/10) = 0.3
        assert a["sensitivity"] == pytest.approx(10.0 * (0.3 / 10.0))
