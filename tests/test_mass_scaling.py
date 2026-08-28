# -*- coding: utf-8 -*-
"""
Tests for the mass-scaling identification (gui.sensitivity.mass_scaling) and the
energy-ratio guard extraction. Analytic sample_fn: ROI velocity = A/factor
(stabilizes as the factor grows) and guard = C*factor (grows with the factor).
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.sensitivity.mass_scaling import identify_mass_scaling
from gui.sensitivity.domain_opt import mean_ke_ie_ratio


class _BundleH:
    def __init__(self, history):
        self._h = history

    def history(self, ch):
        return self._h[ch]


class TestGuardRatio:
    def test_ratio_of_totals(self):
        # ratio of time-aggregated energies: (5+10)/(0+100) = 0.15
        b = _BundleH({"ALLKE": np.array([5.0, 10.0]),
                      "ALLIE": np.array([0.0, 100.0])})
        assert mean_ke_ie_ratio(b) == pytest.approx(0.15)

    def test_early_near_zero_ie_does_not_blow_up(self):
        # Early samples have ALLIE ~ 0 (huge per-sample ALLKE/ALLIE); the
        # ratio-of-totals stays governed by the steady-state energies instead
        # of the early division-by-near-zero spikes.
        ke = np.array([0.01, 0.01, 9.0, 9.0])
        ie = np.array([1e-6, 1e-4, 60.0, 60.0])
        g = mean_ke_ie_ratio(_BundleH({"ALLKE": ke, "ALLIE": ie}))
        assert g == pytest.approx(18.02 / 120.0001, rel=1e-6)
        assert g < 1.0

    def test_absent_returns_none(self):
        assert mean_ke_ie_ratio(_BundleH({})) is None

    def test_zero_total_ie_returns_none(self):
        assert mean_ke_ie_ratio(
            _BundleH({"ALLKE": np.array([1.0, 2.0]),
                      "ALLIE": np.array([0.0, 0.0])})) is None


def _make_sf(A=100.0, C=0.001):
    def sf(f):
        v = np.full((2, 3), A / f)
        return {"Vx": v.copy(), "Vy": v.copy(), "guard": C * f}
    return sf


class _Models:
    """Analytic velocity models. E(f) = |v(f) - v(f*factor)| / eps."""

    @staticmethod
    def vshape(record=None):
        """v(f) = 500 + 3000/f + f/2000 -> v(f)-v(10f) = 2700/f - 0.0045 f,
        which vanishes at f = sqrt(2700/0.0045) = 774.6: a true interior minimum
        of the sensitivity. guard = 1e-4 * f."""
        def sf(f):
            if record is not None:
                record.append(f)
            a = np.full((2, 3), 500.0 + 3000.0 / f + f / 2000.0)
            return {"Vx": a.copy(), "Vy": a.copy(), "guard": 1e-4 * f}
        return sf

    @staticmethod
    def decreasing():
        """v(f) = 100/f -> sensitivity strictly decreasing: no interior minimum,
        the search is bounded by the cap."""
        def sf(f):
            a = np.full((2, 3), 100.0 / f)
            return {"Vx": a.copy(), "Vy": a.copy(), "guard": 1e-9 * f}
        return sf

    @staticmethod
    def increasing():
        """v(f) = 500 + f/2000 -> sensitivity increasing from the first step:
        the minimum sits at the start factor."""
        def sf(f):
            a = np.full((2, 3), 500.0 + f / 2000.0)
            return {"Vx": a.copy(), "Vy": a.copy(), "guard": 1e-9 * f}
        return sf


class TestArgminSearch:
    def test_refines_the_interior_minimum(self):
        # ladder 1,10,100,1e3,1e4: E = 2700, 270, 26.6, 1.8, 44.7 -> rises at 1e4
        # so the minimum is bracketed by [100, 1e4]; the golden section converges
        # to the true argmin 774.6. Both members of that pair satisfy the guard
        # (7.7e-2 and 7.7e-1 < 1.5), so the LARGER is retained.
        r = identify_mass_scaling(_Models.vshape(), {"Vx": 1.0, "Vy": 1.0},
                                  guard_threshold=1.5, start=1.0, factor=10.0,
                                  max_factor=1e6, bisection_resolution=1.0)
        assert r.limited_by == "minimum"
        assert r.intermediate == pytest.approx(1000.0)     # ladder minimum
        assert r.identified == pytest.approx(7746.0, rel=2e-3)
        assert r.sensitivity < 1e-2                        # E ~ 0 at the argmin
        assert r.guard_ok is True

    def test_guard_ceiling_keeps_the_smaller_member(self):
        # same model, guard threshold 0.5: the larger member (guard 0.77) is not
        # admissible, so the argmin itself is retained.
        r = identify_mass_scaling(_Models.vshape(), {"Vx": 1.0, "Vy": 1.0},
                                  guard_threshold=0.5, start=1.0, factor=10.0,
                                  max_factor=1e6, bisection_resolution=1.0)
        assert r.identified == pytest.approx(774.6, rel=2e-3)
        assert r.guard_at_identified == pytest.approx(0.0775, rel=2e-3)
        assert r.guard_ok is True

    def test_ladder_stops_at_the_first_rise(self):
        # phase 1 must not scan up to the cap once E has risen.
        runs = []
        identify_mass_scaling(_Models.vshape(runs), {"Vx": 1.0, "Vy": 1.0},
                              guard_threshold=1.5, start=1.0, factor=10.0,
                              max_factor=1e6, bisection_resolution=1e9)
        # ladder points 1..1e4 plus the partner of the last one (1e5)
        assert runs[:6] == pytest.approx([1.0, 10.0, 100.0, 1000.0, 10000.0,
                                          100000.0])
        assert max(runs) <= 1e5          # never walked up to the 1e6 cap

    def test_cap_bounds_a_monotone_decreasing_sensitivity(self):
        r = identify_mass_scaling(_Models.decreasing(), {"Vx": 1e-6, "Vy": 1e-6},
                                  guard_threshold=1.0, start=1.0, factor=5.0,
                                  max_factor=1e4, bisection_resolution=1e9)
        assert r.limited_by == "cap"
        assert r.identified <= 1e4

    def test_minimum_at_the_start_factor(self):
        r = identify_mass_scaling(_Models.increasing(), {"Vx": 1e-6, "Vy": 1e-6},
                                  guard_threshold=1.0, start=1.0, factor=10.0,
                                  max_factor=1e6, bisection_resolution=1e9)
        assert r.limited_by == "start"
        assert r.intermediate == pytest.approx(1.0)

    def test_guard_violated_at_start(self):
        r = identify_mass_scaling(_Models.vshape(), {"Vx": 1.0, "Vy": 1.0},
                                  guard_threshold=1e-9, start=1.0, factor=10.0,
                                  max_factor=1e6, bisection_resolution=1.0)
        assert r.limited_by == "guard"
        assert r.guard_ok is False
        assert r.identified == pytest.approx(1.0)

    def test_resolution_controls_the_cost(self):
        fine, coarse = [], []
        identify_mass_scaling(_Models.vshape(fine), {"Vx": 1.0, "Vy": 1.0},
                              guard_threshold=1.5, start=1.0, factor=10.0,
                              max_factor=1e6, bisection_resolution=1.0)
        identify_mass_scaling(_Models.vshape(coarse), {"Vx": 1.0, "Vy": 1.0},
                              guard_threshold=1.5, start=1.0, factor=10.0,
                              max_factor=1e6, bisection_resolution=2000.0)
        assert len(coarse) < len(fine)

    def test_factor_validation(self):
        with pytest.raises(ValueError):
            identify_mass_scaling(_Models.vshape(), {"Vx": 1.0}, 1.0, start=1.0,
                                  factor=0.5)

    def test_progress_events_and_cancel(self):
        events = []
        identify_mass_scaling(_Models.vshape(), {"Vx": 1.0, "Vy": 1.0},
                              guard_threshold=1.5, start=1.0, factor=10.0,
                              max_factor=1e6, bisection_resolution=1.0,
                              progress_cb=events.append)
        phases = {e["phase"] for e in events}
        assert "mass_scaling" in phases and "mass_scaling_done" in phases
        pt = [e for e in events if e["phase"] == "mass_scaling"][0]
        assert "errors" in pt and "guard" in pt and "factor" in pt

        r = identify_mass_scaling(_Models.vshape(), {"Vx": 1.0, "Vy": 1.0},
                                  guard_threshold=1.5, start=1.0, factor=10.0,
                                  should_cancel=lambda: True)
        assert r.limited_by == "cancelled"
        assert r.n_runs == 0


class TestSampleFnToolFactor:
    def test_sample_fn_sets_both_factors(self):
        # The identification probe must scale BOTH materials (ms_tool = ms_eul).
        from gui.sensitivity.mass_scaling import make_mass_scaling_sample_fn
        from gui.core.model_config import ModelConfig
        captured = {}

        def fake_run_bundle(cfg):
            captured["enabled"] = cfg.step.mass_scaling_enabled
            captured["eul"] = cfg.step.mass_scaling_factor
            captured["tool"] = cfg.step.mass_scaling_factor
            return None  # -> sample_fn raises after the cfg has been captured

        sf = make_mass_scaling_sample_fn(
            ModelConfig(), fake_run_bundle, roi=(0.0, 1.0, 0.0, 1.0),
            grid_step=0.5, velocity_field_map={"Vx": "V1", "Vy": "V2"})
        with pytest.raises(RuntimeError):
            sf(7.0)
        assert captured["enabled"] is True
        assert captured["eul"] == pytest.approx(7.0)
        assert captured["tool"] == pytest.approx(7.0)


class TestCapLadder:
    def test_single_overshoot_and_nothing_retained_past_the_cap(self):
        # Strictly decreasing sensitivity: the ladder walks up to the cap. Each
        # point is E(f, f*factor), so the last on-ladder factor needs ONE run
        # beyond the cap as its comparison partner - and that overshoot must
        # never be retained.
        runs = []

        def sf(f):
            runs.append(f)
            a = np.full((2, 3), 100.0 / f)
            return {"Vx": a.copy(), "Vy": a.copy(), "guard": 1e-9 * f}

        r = identify_mass_scaling(sf, {"Vx": 1e-6, "Vy": 1e-6},
                                  guard_threshold=1.0, start=1.0, factor=5.0,
                                  max_factor=1e4, bisection_resolution=1e9)
        ladder = [1.0, 5.0, 25.0, 125.0, 625.0, 3125.0]
        assert runs[:len(ladder) + 1] == pytest.approx(ladder + [15625.0])
        assert len([f for f in runs if f > 1e4]) == 1     # exactly one overshoot
        assert r.identified <= 1e4
