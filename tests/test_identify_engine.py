# -*- coding: utf-8 -*-
"""Identify engine on a fake solver with known parameters: cost, stop rule,
fixed point, Levenberg-Marquardt, stagnation, budget, resume, failed runs
and the identifiability check."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gui.identify import engine as ie


A = np.array([[2.0, 0.5, 0.0],
              [0.3, 1.5, 0.0],
              [1.0, -1.0, 0.0]])          # third parameter has no effect
B = np.array([1.0, -2.0, 0.5])
P_TRUE = np.array([0.7, 0.2, 0.4])
IDS = ("Fc", "Ff", "T")


def _q(p):
    p = np.asarray(p, float)
    p = np.r_[p, P_TRUE[len(p):]]          # parameters not identified
    v = A @ p + B
    v[0] += 3.0 * p[0] ** 2 * p[1]         # coupled, nonlinear
    return dict(zip(IDS, v))


class Solver:
    def __init__(self, fail=()):
        self.calls, self.batches, self.fail = 0, [], set(fail)

    def __call__(self, points):
        self.batches.append(len(points))
        out = []
        for p in points:
            self.calls += 1
            out.append(None if self.calls in self.fail else _q(p))
        return out


def _setup(k=2, eps=0.01, start=(0.1, 0.9, 0.5)):
    params = [ie.Param(path, 0.0, 1.0, start=s)
              for path, s in zip(("a", "b", "c")[:k], start[:k])]
    qt = _q(P_TRUE)
    targets = [ie.Target(i, qt[i], eps) for i in IDS]
    return params, targets


def test_cost_is_the_weighted_mean_in_eps_units():
    t = [ie.Target("a", 0.0, 2.0, 1.0), ie.Target("b", 0.0, 1.0, 3.0)]
    assert ie.cost({"a": 2.0, "b": 1.0}, t) == pytest.approx(1.0)
    assert ie.cost({"a": 4.0, "b": 0.0}, t) == pytest.approx(4 * 1 / 4)
    assert ie.cost(None, t) == math.inf
    assert ie.cost({"a": 1.0}, t) == math.inf
    assert ie.max_ratio({"a": 2.0, "b": 0.5}, t) == pytest.approx(1.0)
    assert ie.converged({"a": 2.0, "b": 0.5}, t)
    assert not ie.converged({"a": 2.1, "b": 0.5}, t)
    # weight 0: reported, does not decide
    t0 = [ie.Target("a", 0.0, 1.0, 0.0), ie.Target("b", 0.0, 1.0)]
    assert ie.converged({"a": 50.0, "b": 0.1}, t0)


def test_param_validation():
    with pytest.raises(ValueError):
        ie.Param("a", 1.0, 1.0)
    with pytest.raises(ValueError):
        ie.Param("a", 0.0, 1.0, start=2.0)
    p = ie.Param("a", 0.0, 4.0)
    assert p.start == 2.0 and p.scale == 4.0
    with pytest.raises(ValueError):
        ie.Target("q", 0.0, 0.0)


@pytest.mark.parametrize("scheme", ["forward", "central"])
def test_lm_finds_the_known_parameters(scheme):
    params, targets = _setup(k=2)
    solver = Solver()
    res = ie.Identification(params, targets, solver, strategy="lm",
                            budget=60, fd_scheme=scheme).run()
    assert res.status == ie.CONVERGED, res.message
    assert np.allclose(res.x, P_TRUE[:2], atol=0.01)
    assert res.max_ratio <= 1.0
    # the finite-difference runs of an iteration go out together
    assert max(solver.batches) == (2 if scheme == "forward" else 4)
    assert res.n_runs == solver.calls


def test_fixed_point_finds_the_known_parameters():
    params, targets = _setup(k=2, eps=0.05)
    res = ie.Identification(params, targets, Solver(),
                            strategy="fixed_point", budget=200,
                            line_tol=0.002).run()
    assert res.status == ie.CONVERGED, res.message
    assert np.allclose(res.x, P_TRUE[:2], atol=0.05)


def test_start_point_within_eps_costs_one_run():
    params, targets = _setup(k=2, start=tuple(P_TRUE))
    solver = Solver()
    res = ie.Identification(params, targets, solver).run()
    assert res.status == ie.CONVERGED and solver.calls == 1


def test_parameter_without_effect_is_frozen():
    """c moves no QoI: its step effect is 0 < tau, it is frozen after 2
    iterations and leaves the finite differences."""
    params, targets = _setup(k=3, eps=1e-4)
    solver = Solver()
    ident = ie.Identification(params, targets, solver, strategy="lm",
                              budget=200)
    res = ident.run()
    assert "c" in res.frozen or res.status == ie.CONVERGED
    frozen_at = next(it.iteration for it in res.iterations if "c" in it.frozen)
    assert frozen_at == 2
    later = [it for it in res.iterations if it.iteration > frozen_at]
    assert later and all("c" not in it.free for it in later)


def test_stagnation_uses_the_effect_on_the_qoi():
    S = np.array([[10.0, 0.0], [0.0, 0.001]])
    t = [ie.Target("x", 0, 1.0), ie.Target("y", 0, 1.0)]
    eff = ie.step_effect(S, np.array([0.05, 50.0]), t)
    # a small step on an influential parameter, a large one on a weak one
    assert eff[0] == pytest.approx(0.5) and eff[1] == pytest.approx(0.05)


def test_all_frozen_gets_one_last_free_pass():
    """No parameter moves the QoI and the reference is out of reach: both
    freeze, one last free pass is made, then the search stops as stalled."""
    params = [ie.Param("a", 0, 1, start=0.5), ie.Param("b", 0, 1, start=0.5)]
    targets = [ie.Target("Fc", 100.0, 1.0)]
    res = ie.Identification(params, targets, lambda pts: [{"Fc": 0.0}] * len(pts),
                            strategy="fixed_point", budget=500).run()
    assert res.status == ie.STALLED
    assert set(res.frozen) == {"a", "b"}
    # iterations 1, 2 freeze both; iteration 3 is the free pass
    assert res.iterations[-1].iteration == 3
    assert set(res.iterations[-1].free) == {"a", "b"}   # the free pass


def test_budget_stops_and_resume_reuses_the_runs():
    params, targets = _setup(k=2, eps=1e-6)
    s1 = Solver()
    r1 = ie.Identification(params, targets, s1, strategy="lm", budget=7).run()
    assert r1.status == ie.BUDGET and r1.n_runs <= 7
    rec = r1.record(params, targets)
    s2 = Solver()
    r2 = ie.Identification(params, targets, s2, strategy="lm", budget=7,
                           prior=rec["evaluations"]).run()
    # every run of the first search is read back, none launched again
    assert sum(e.reused for e in r2.evaluations) == len(r1.evaluations)
    assert s2.calls == r2.n_runs
    assert r2.evaluations[0].x == r1.evaluations[0].x
    assert rec["params"][0]["path"] == "a" and rec["targets"][0]["qid"] == "Fc"


def test_failed_start_and_failed_step():
    params, targets = _setup(k=2)
    res = ie.Identification(params, targets, Solver(fail={1})).run()
    assert res.status == ie.FAILED
    # a failed step run is rejected (J = inf) and the search goes on
    res = ie.Identification(params, targets, Solver(fail={4}),
                            strategy="lm", budget=80).run()
    assert res.status == ie.CONVERGED


def test_identifiability_flags_proportional_effects():
    params = [ie.Param("a", 0, 1), ie.Param("b", 0, 2), ie.Param("c", 0, 1)]
    t = [ie.Target("x", 0, 1.0), ie.Target("y", 0, 1.0)]
    S = np.array([[1.0, 0.5, 0.0], [2.0, 1.0, 1.0]])   # b = a in effect
    info = ie.identifiability(S, params, t)
    assert info["pairs"][0][:2] == ("a", "b")
    assert info["pairs"][0][2] == pytest.approx(1.0)
    # without c the two columns are proportional: not identifiable
    two = ie.identifiability(S[:, :2], params[:2], t)
    assert two["cond"] == math.inf or two["cond"] > 1e6


def test_golden_runs():
    assert ie.golden_runs(0.01) == 11
    assert ie.golden_runs(0.5) == 3
