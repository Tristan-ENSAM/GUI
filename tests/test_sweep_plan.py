# -*- coding: utf-8 -*-
"""Sweep (one at a time) and grid (full factorial) plans of the Parameter
plans tab: levels, run layout, configs and per-parameter trends."""
from __future__ import annotations

import numpy as np
import pytest

from gui.core.model_config import ModelConfig
from gui.sensitivity import param_registry as pr
from gui.sensitivity import sweep_plan as sp


def _spec(path):
    return pr.spec_for(path)


def test_levels_linear_geometric_and_typed():
    assert np.allclose(sp.levels(0.1, 0.5, 5), [0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(sp.levels(1, 10000, 5, "geometric"),
                       [1, 10, 100, 1000, 10000])
    with pytest.raises(ValueError):
        sp.levels(0, 10, 3, "geometric")
    with pytest.raises(ValueError):
        sp.levels(1, 1, 3)
    assert np.allclose(sp.parse_values("1; 100 1000,12800"),
                       [1, 100, 1000, 12800])
    assert sp.parse_values("  ") is None
    with pytest.raises(ValueError):
        sp.parse_values("1")


def test_sweep_shares_the_reference_run():
    mu, vc = _spec("interaction.friction_coeff"), _spec("bcs.cutting_speed")
    plan = sp.build_plan([(mu, 0.3, [0.1, 0.2, 0.3, 0.4, 0.5]),
                          (vc, 30.0, [20.0, 40.0])], "sweep")
    # 1 shared reference + 4 other mu levels + 2 Vc levels
    assert plan.n_runs == 7 == sp.n_runs(plan.level_values, plan.base,
                                         "sweep")
    assert plan.run_kind == ["base", "0", "0", "0", "0", "1", "1"]
    assert np.allclose(plan.X[0], [0.3, 30.0])
    assert all(np.isclose(plan.X[r, 1], 30.0) for r in range(1, 5))


def test_grid_adds_the_reference_when_off_grid():
    mu, vc = _spec("interaction.friction_coeff"), _spec("bcs.cutting_speed")
    plan = sp.build_plan([(mu, 0.3, [0.2, 0.4]), (vc, 30.0, [20.0, 40.0])],
                         "grid")
    assert plan.n_runs == 5 and plan.run_kind[0] == "base"
    assert not plan.base_on_grid
    on = sp.build_plan([(mu, 0.3, [0.3, 0.4]), (vc, 30.0, [30.0, 40.0])],
                       "grid")
    assert on.n_runs == 4 and on.base_on_grid
    assert on.run_kind.count("base") == 1
    assert np.allclose(on.X[on.base_row], [0.3, 30.0])


def test_configs_carry_the_values():
    mu = _spec("interaction.friction_coeff")
    plan = sp.build_plan([(mu, 0.3, [0.1, 0.5])], "sweep")
    cfgs = sp.plan_to_configs(ModelConfig(), plan)
    assert [c.interaction.friction_coeff for c in cfgs] == \
        pytest.approx([0.3, 0.1, 0.5])
    rows = sp.profile_table(plan)
    assert rows[1]["interaction.friction_coeff"] == pytest.approx(0.1)


def test_analyze_sweep_trend_and_eps():
    mu = _spec("interaction.friction_coeff")
    plan = sp.build_plan([(mu, 0.3, [0.1, 0.2, 0.3, 0.4, 0.5])], "sweep")
    x = plan.X[:, 0]
    y = 100.0 + 50.0 * x                     # linear, increasing
    y[2] = np.nan                            # one failed run
    res = sp.analyze(plan, y, eps=10.0)[mu.path]
    assert res["n"] == 4
    assert res["slope"] == pytest.approx(50.0)
    assert res["r2"] == pytest.approx(1.0)
    assert res["spearman"] == pytest.approx(1.0)
    # |q(0.5) - q(0.3)| / eps = 10 / 10
    assert res["max_dev_eps"] == pytest.approx(1.0)


def test_analyze_grid_main_effects():
    mu, vc = _spec("interaction.friction_coeff"), _spec("bcs.cutting_speed")
    plan = sp.build_plan([(mu, 0.3, [0.2, 0.4]), (vc, 30.0, [20.0, 40.0])],
                         "grid")
    y = 10 * plan.X[:, 0] + plan.X[:, 1]     # additive
    res = sp.analyze(plan, y)
    assert res[mu.path]["x"] == pytest.approx([0.2, 0.4])
    assert res[mu.path]["slope"] == pytest.approx(10.0)
    assert res[vc.path]["slope"] == pytest.approx(1.0)
    # main effect of Vc: mean over mu at each Vc level (the reference
    # row, off the grid, is not part of it): (22 + 24) / 2, (42 + 44) / 2
    assert res[vc.path]["q"] == pytest.approx([23.0, 43.0])
