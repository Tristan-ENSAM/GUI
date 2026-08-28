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
# Sensitivity tab — Maps UI wiring (headless)
# ---------------------------------------------------------------------------
class TestSensitivityMapsUI:

    def _bundles_and_plan(self, tmp_path):
        from gui.results.fake_builder import build_fake_results
        from gui.results.reader import ResultsBundle
        bundles = []
        for k in range(3):
            _, npz = build_fake_results(tmp_path / ("j%d.results.npz" % k),
                                        n_frames=3, n_grid_x=5, n_grid_y=4)
            bundles.append(ResultsBundle.load(npz))
        spec = pr.spec_for("interaction.friction_coeff")
        plan = jac.build_plan([(spec, 0.3, 0.1, False)], scheme="central")
        order = [None] * plan.n_runs
        order[0] = bundles[0]
        order[plan.idx_plus[0]] = bundles[1]
        order[plan.idx_minus[0]] = bundles[2]
        return bundles, plan, order

    def test_maps_populate_and_render(self, qapp, tmp_path):
        from gui.tabs.sensitivity_tab import SensitivityTab
        from gui.core.model_config import ModelConfig
        bundles, plan, order = self._bundles_and_plan(tmp_path)
        tab = SensitivityTab(ModelConfig())
        tab.plan = plan
        tab.plan_kind = "jacobian"
        tab._field_checks["EVF"].setChecked(True)
        res = rc.RunResult(plan_kind="jacobian", qoi_ids=[],
                           param_paths=list(plan.param_paths),
                           Y=np.zeros((plan.n_runs, 0)), analyses={},
                           failures=[], bundles=order)
        tab._build_field_maps(res)
        assert tab.cb_map_param.count() == 1
        assert tab.cb_map_field.count() == 1            # EVF only
        assert tab._map_n_frames == 3
        assert tab._map_mesh_set is True
        # signed/magnitude x aggregate combinations must not raise
        for signed in (True, False):
            for agg in (True, False):
                tab.chk_map_signed.setChecked(signed)
                tab.chk_map_aggregate.setChecked(agg)
                tab._refresh_map()
        # the frame slider is disabled while aggregating, enabled otherwise
        tab.chk_map_aggregate.setChecked(True)
        assert tab.sld_map_frame.isEnabled() is False
        tab.chk_map_aggregate.setChecked(False)
        assert tab.sld_map_frame.isEnabled() is True
        tab.sld_map_frame.setValue(0)
        tab._refresh_map()
        for b in bundles:
            b.close()

    def test_morris_clears_maps(self, qapp):
        from gui.tabs.sensitivity_tab import SensitivityTab
        from gui.core.model_config import ModelConfig
        tab = SensitivityTab(ModelConfig())
        res = rc.RunResult(plan_kind="morris", qoi_ids=[], param_paths=[],
                           Y=np.zeros((1, 0)), analyses={}, failures=[],
                           bundles=None)
        tab._build_field_maps(res)
        assert tab._field_maps == {}
        assert tab.cb_map_param.isEnabled() is False
