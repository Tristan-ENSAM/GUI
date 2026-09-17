# -*- coding: utf-8 -*-
"""
Unit tests for the per-element sensitivity maps (Lot S.1):
  - gui.sensitivity.field_metrics.elementwise_signed_sensitivity
  - gui.sensitivity.runner_core.jacobian_field_maps

Fully offline (no Abaqus, no Qt). The maps keep the element axis: for a field
that depends linearly on the perturbed parameter per element, the recovered
signed sensitivity must equal the per-element slope.
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.sensitivity import field_metrics as fm
from gui.sensitivity import runner_core as rc
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import param_registry as pr


# ---------------------------------------------------------------------------
# elementwise_signed_sensitivity
# ---------------------------------------------------------------------------
class TestElementwiseSensitivity:

    def _setup(self, n_frames=2, n_elem=4, delta=0.1):
        base = np.tile(np.arange(n_elem, dtype=float) + 1.0, (n_frames, 1))
        slope = np.array([1.0, -2.0, 0.5, 0.0])[:n_elem]
        plus = base + slope * delta
        minus = base - slope * delta
        return base, plus, minus, slope, delta

    def test_central_recovers_slope(self):
        base, plus, minus, slope, d = self._setup()
        S = fm.elementwise_signed_sensitivity(base, plus, minus, d, "central")
        assert S.shape == base.shape
        assert np.allclose(S[0], slope)
        assert np.allclose(S[1], slope)         # same per frame here

    def test_forward_recovers_slope(self):
        base, plus, _, slope, d = self._setup()
        S = fm.elementwise_signed_sensitivity(base, plus, None, d, "forward")
        assert np.allclose(S[0], slope)

    def test_backward_recovers_slope(self):
        base, _, minus, slope, d = self._setup()
        S = fm.elementwise_signed_sensitivity(base, None, minus, d, "backward")
        assert np.allclose(S[0], slope)

    def test_sign_is_meaningful(self):
        # Element 1 has negative slope -> S < 0 there, positive where slope>0.
        base, plus, minus, slope, d = self._setup()
        S = fm.elementwise_signed_sensitivity(base, plus, minus, d, "central")
        assert S[0, 0] > 0 and S[0, 1] < 0

    def test_delta_zero_raises(self):
        base, plus, minus, _, _ = self._setup()
        with pytest.raises(ValueError):
            fm.elementwise_signed_sensitivity(base, plus, minus, 0.0, "central")

    def test_bad_scheme_raises(self):
        base, plus, minus, _, d = self._setup()
        with pytest.raises(ValueError):
            fm.elementwise_signed_sensitivity(base, plus, minus, d, "bogus")

    def test_missing_operand_all_nan(self):
        # Central needs both plus and minus; minus None -> all-NaN, shaped.
        base, plus, _, _, d = self._setup()
        S = fm.elementwise_signed_sensitivity(base, plus, None, d, "central")
        assert S.shape == base.shape
        assert np.all(np.isnan(S))

    def test_nan_input_propagates(self):
        base, plus, minus, slope, d = self._setup()
        plus = plus.copy(); plus[0, 2] = np.nan
        S = fm.elementwise_signed_sensitivity(base, plus, minus, d, "central")
        assert np.isnan(S[0, 2])
        assert not np.isnan(S[0, 0])            # other elements unaffected


# ---------------------------------------------------------------------------
# jacobian_field_maps
# ---------------------------------------------------------------------------
class _FakeFieldBundle:
    """Minimal bundle exposing .field(inst, var) -> stored array."""
    def __init__(self, by_var):
        self._by_var = by_var

    def field(self, inst, var):
        return self._by_var[var]


class TestJacobianFieldMaps:

    def test_central_map_recovers_per_element_slope(self):
        var = "S_VM"
        spec = pr.spec_for("interaction.friction_coeff")     # identity factor
        plan = jac.build_plan([(spec, 0.3, 0.1, False)], scheme="central")
        d = plan.deltas[0]
        n_frames, n_elem = 2, 4
        base = np.tile(np.arange(n_elem, dtype=float) + 1.0, (n_frames, 1))
        slope = np.array([1.0, -2.0, 0.5, 0.0])
        plus = base + slope * d
        minus = base - slope * d

        bundles = [None] * plan.n_runs
        bundles[0] = _FakeFieldBundle({var: base})           # base run
        bundles[plan.idx_plus[0]] = _FakeFieldBundle({var: plus})
        bundles[plan.idx_minus[0]] = _FakeFieldBundle({var: minus})

        maps = rc.jacobian_field_maps(plan, bundles, [var], instance="Euler")
        S = maps[var]["interaction.friction_coeff"]
        assert S.shape == (n_frames, n_elem)
        assert np.allclose(S[0], slope)

    def test_missing_perturbed_bundle_gives_nan_map(self):
        var = "TEMP"
        spec = pr.spec_for("interaction.friction_coeff")
        plan = jac.build_plan([(spec, 0.3, 0.1, False)], scheme="central")
        n_frames, n_elem = 2, 3
        base = np.ones((n_frames, n_elem))
        bundles = [None] * plan.n_runs
        bundles[0] = _FakeFieldBundle({var: base})
        # plus/minus bundles left as None -> central cannot be computed.
        maps = rc.jacobian_field_maps(plan, bundles, [var], instance="Euler")
        S = maps[var]["interaction.friction_coeff"]
        assert S.shape == base.shape
        assert np.all(np.isnan(S))

    def test_empty_bundles_returns_empty(self):
        spec = pr.spec_for("interaction.friction_coeff")
        plan = jac.build_plan([(spec, 0.3, 0.1, False)], scheme="central")
        assert rc.jacobian_field_maps(plan, [], ["S_VM"]) == {}
        assert rc.jacobian_field_maps(plan, [None], ["S_VM"]) == {}




# ---------------------------------------------------------------------------
# Morris — per-element elementary effects
# ---------------------------------------------------------------------------
def _morris_plan_2params():
    """A hand-built 2-parameter, 2-trajectory Morris plan.

    Built by instantiating MorrisPlan directly rather than through
    build_plan, so these tests do not need SALib (an optional dependency
    of the project). The trajectory layout is the one SALib produces: blocks
    of k+1 rows, one parameter moving at a time.
    """
    from gui.sensitivity.morris_plan import MorrisPlan
    specs = [pr.spec_for("interaction.friction_coeff"),
             pr.spec_for("bcs.cutting_speed")]
    X = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],      # trajectory 1 (both up)
        [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],      # trajectory 2 (both down)
    ], dtype=float)
    return MorrisPlan(specs=specs, bounds=[(0.0, 1.0), (0.0, 1.0)], N=2,
                      num_levels=4, problem={"num_vars": 2,
                                             "names": [s.path for s in specs],
                                             "bounds": [[0, 1], [0, 1]]}, X=X)


class TestElementwiseMorrisStats:

    def _linear_fields(self, plan, a, b, n_frames=2):
        """F_e = a_e*x0 + b_e*x1, one (n_frames, n_elem) array per run."""
        return [np.tile(a * row[0] + b * row[1], (n_frames, 1))
                for row in plan.X]

    def test_recovers_per_element_slopes(self):
        plan = _morris_plan_2params()
        a = np.array([1.0, -2.0, 0.0, 0.5])
        b = np.array([0.0, 3.0, -1.0, 0.25])
        fields = self._linear_fields(plan, a, b)
        st = fm.elementwise_morris_stats(fields, plan.X, plan.num_levels)
        step = fm.morris_grid_step(plan.num_levels)
        assert st["mu"].shape == (2, 2, 4)
        # Both trajectories see the same slope -> mu = slope/step, sigma = 0.
        assert np.allclose(st["mu"][0, 0], a / step)
        assert np.allclose(st["mu"][1, 0], b / step)
        assert np.allclose(st["mu_star"][0, 0], np.abs(a) / step)
        assert np.allclose(st["sigma"][0, 0], 0.0, atol=1e-9)
        assert np.all(st["n_eff"] == 2)

    def test_grid_step_is_salib_convention(self):
        # Delta = p / (2*(p-1)) -- SALib 1.5.2 _compute_delta.
        assert fm.morris_grid_step(4) == pytest.approx(4 / 6.0)
        assert fm.morris_grid_step(6) == pytest.approx(6 / 10.0)
        with pytest.raises(ValueError):
            fm.morris_grid_step(1)

    def test_matches_salib_on_a_scalar_model(self):
        """A 1x1 'field' must reproduce SALib's scalar mu*/sigma exactly:
        the map and the ranking table are then on the same scale."""
        pytest.importorskip("SALib")
        from gui.sensitivity import morris_plan as mp
        plan = _morris_plan_2params()
        rng = np.random.default_rng(0)
        Y = np.array([2.0 * r[0] - 0.5 * r[1] + 0.1 * rng.random()
                      for r in plan.X])
        fields = [np.array([[y]]) for y in Y]
        st = fm.elementwise_morris_stats(fields, plan.X, plan.num_levels)
        si = mp.analyze(plan, Y)
        for i in range(2):
            assert st["mu"][i, 0, 0] == pytest.approx(si["mu"][i])
            assert st["mu_star"][i, 0, 0] == pytest.approx(si["mu_star"][i])
            assert st["sigma"][i, 0, 0] == pytest.approx(si["sigma"][i])

    def test_missing_run_drops_only_its_effects(self):
        plan = _morris_plan_2params()
        a = np.array([1.0, -2.0, 0.0, 0.5])
        b = np.array([0.0, 3.0, -1.0, 0.25])
        fields = self._linear_fields(plan, a, b)
        fields[1] = None                     # kills both effects of traj. 1
        st = fm.elementwise_morris_stats(fields, plan.X, plan.num_levels)
        step = fm.morris_grid_step(plan.num_levels)
        assert np.all(st["n_eff"][0] == 1)   # param 0: one effect left
        assert np.allclose(st["mu"][0, 0], a / step)   # still the right slope
        assert np.all(np.isnan(st["sigma"][0]))        # ddof=1 needs 2 samples

    def test_nan_element_does_not_poison_its_neighbours(self):
        plan = _morris_plan_2params()
        a = np.array([1.0, -2.0, 0.0, 0.5])
        b = np.zeros(4)
        fields = self._linear_fields(plan, a, b)
        fields[1] = fields[1].copy()
        fields[1][0, 2] = np.nan
        st = fm.elementwise_morris_stats(fields, plan.X, plan.num_levels)
        step = fm.morris_grid_step(plan.num_levels)
        assert st["n_eff"][0, 0, 2] == 1                 # one effect lost
        assert st["n_eff"][0, 0, 0] == 2                 # neighbour intact
        assert st["mu"][0, 0, 0] == pytest.approx(a[0] / step)

    def test_bad_design_raises(self):
        plan = _morris_plan_2params()
        fields = [np.ones((1, 2)) for _ in range(5)]     # 5 rows, k+1 = 3
        with pytest.raises(ValueError):
            fm.elementwise_morris_stats(fields, plan.X[:5], plan.num_levels)
        with pytest.raises(ValueError):
            fm.elementwise_morris_stats([None] * 6, plan.X, plan.num_levels)


class TestMorrisFieldMaps:

    def test_maps_carry_the_three_indices(self):
        plan = _morris_plan_2params()
        a = np.array([1.0, -2.0, 0.0, 0.5])
        bundles = [_FakeFieldBundle({"TEMP": np.tile(a * r[0], (2, 1))})
                   for r in plan.X]
        maps = rc.morris_field_maps(plan, bundles, ["TEMP"], instance="Euler")
        per = maps["TEMP"]["interaction.friction_coeff"]
        assert set(per) == {"mu_star", "sigma", "mu"}
        assert per["mu_star"].shape == (2, 4)

    def test_build_field_maps_follows_the_method(self):
        plan = _morris_plan_2params()
        bundles = [_FakeFieldBundle({"EVF": np.ones((2, 4))}) for _ in plan.X]
        out = rc.build_field_maps(plan, "morris", bundles, ["EVF"],
                                  instance="Euler")
        assert set(out["EVF"]["interaction.friction_coeff"]) == \
            {"mu_star", "sigma", "mu"}

        jplan = jac.build_plan(
            [(pr.spec_for("interaction.friction_coeff"), 0.3, 0.1, False)],
            scheme="central")
        jb = [None] * jplan.n_runs
        jb[0] = _FakeFieldBundle({"EVF": np.ones((2, 4))})
        jb[jplan.idx_plus[0]] = _FakeFieldBundle({"EVF": np.ones((2, 4))})
        jb[jplan.idx_minus[0]] = _FakeFieldBundle({"EVF": np.zeros((2, 4))})
        out = rc.build_field_maps(jplan, "jacobian", jb, ["EVF"],
                                  instance="Euler")
        assert list(out["EVF"]["interaction.friction_coeff"]) == ["dFdtheta"]

    def test_unknown_method_returns_empty(self):
        plan = _morris_plan_2params()
        bundles = [_FakeFieldBundle({"EVF": np.ones((2, 4))}) for _ in plan.X]
        assert rc.build_field_maps(plan, "sobol", bundles, ["EVF"]) == {}
        assert rc.build_field_maps(plan, "morris", bundles, []) == {}
