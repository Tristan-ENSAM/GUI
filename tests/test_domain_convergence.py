# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.domain_convergence (Option B).

The run_bundle is analytic: the boundary influence on the ZOI decays with each
domain dimension, so the plateau is reachable and convergence is deterministic.
Sampling is bypassed: the analytic bundle IS the ZOI sample.
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.core.domain_sizing import DomainDims
import gui.sensitivity.domain_convergence as dc
from gui.sensitivity.domain_convergence import (
    window_mask, domain_bounds, zoi_inside, reduce_zoi, zoi_discrepancy,
    run_domain_convergence, ZoiReduction, DIMENSION_NAMES,
)

_NP = 12                 # ZOI grid points; 8 material, 4 void
_NT = 10                 # field frames
_MAT = np.array([True] * 8 + [False] * 4)     # material mask per point


class _Bundle:
    """Analytic ZOI: material-field level tracks a decaying boundary influence;
    the last 4 points are permanently void (EVF = 0)."""

    def __init__(self, dims, lam=0.05):
        self.dims, self.lam = dims, lam

    # --- ResultsBundle-like surface -------------------------------------
    @property
    def times(self):
        return np.linspace(0.0, 3e-4, _NT)

    def instance(self, name):
        class _Info:
            field_variables = ["EVF", "TEMP", "V1", "V2"]
        return _Info()

    def _influence(self):
        return sum(np.exp(-getattr(self.dims, n) / self.lam)
                   for n in DIMENSION_NAMES)

    def field(self, inst, var):
        out = np.zeros((_NT, _NP), dtype=float)
        if var == "EVF":
            out[:, _MAT] = 1.0            # material
            out[:, ~_MAT] = 0.0           # void
            return out
        level = {"TEMP": 300.0, "V1": 10.0, "V2": -5.0}[var] * (
            1.0 + 0.5 * self._influence())
        out[:, _MAT] = level
        out[:, ~_MAT] = np.nan            # void: undefined, must be masked out
        return out

    def history(self, var):
        if var == "RF1_RP":
            return np.full(_NT, 100.0 * (1.0 + 0.5 * self._influence()))
        raise KeyError(var)

    @property
    def history_time(self):
        return np.linspace(0.0, 3e-4, _NT)


class _Cfg:
    class _G:
        h_wp = h_void = l_wp = l_void = 0.0

    def __init__(self):
        self.euler_geometry = _Cfg._G()


@pytest.fixture(autouse=True)
def _patch_sampling(monkeypatch):
    monkeypatch.setattr(dc, "nearest_samples",
                        lambda b, var, inst, pts: b.field(inst, var))
    monkeypatch.setattr(dc, "roi_grid", lambda roi, step: np.zeros((_NP, 2)))
    monkeypatch.setattr(dc, "eulerian_instance", lambda b: "EULER")


def _runner(lam=0.05):
    def run(cfg):
        g = cfg.euler_geometry
        return _Bundle(DomainDims(g.h_wp, g.h_void, g.l_wp, g.l_void), lam)
    return run


# --- window_mask --------------------------------------------------------------
class TestWindowMask:
    def test_default_drops_first_30pct(self):
        t = np.linspace(0.0, 1.0, 11)          # 0.0 .. 1.0
        m = window_mask(t, 0.3, 1.0)
        # keep t >= 0.3 -> indices 3..10 (8 frames: 0.3, 0.4, ..., 1.0)
        assert m.sum() == 8 and m[3] and not m[2]

    def test_custom_window(self):
        t = np.linspace(0.0, 1.0, 11)
        m = window_mask(t, 0.0, 0.5)
        assert m[0] and m[5] and not m[6]

    def test_empty_and_zero_end(self):
        assert window_mask(np.array([]), 0.3, 1.0).size == 0
        assert not window_mask(np.zeros(5), 0.3, 1.0).any()

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            window_mask(np.linspace(0, 1, 5), 0.7, 0.3)


# --- geometry helpers ---------------------------------------------------------
class TestGeometry:
    def test_domain_bounds_origin_convention(self):
        d = DomainDims(h_wp=0.1, h_void=0.2, l_wp=0.1, l_void=0.05)
        assert domain_bounds(d) == (-0.1, 0.05, -0.1, 0.2)

    def test_zoi_inside_with_margin(self):
        d = DomainDims(h_wp=0.1, h_void=0.1, l_wp=0.1, l_void=0.1)
        zoi = (-0.08, 0.08, -0.08, 0.08)
        assert zoi_inside(d, zoi, margin=0.01)
        assert not zoi_inside(d, zoi, margin=0.03)   # 0.08+0.03 > 0.10

    def test_zoi_outside_when_too_big(self):
        d = DomainDims(h_wp=0.05, h_void=0.05, l_wp=0.05, l_void=0.05)
        zoi = (-0.1, 0.1, -0.1, 0.1)
        assert not zoi_inside(d, zoi, margin=0.0)


# --- reduce_zoi ---------------------------------------------------------------
class TestReduceZoi:
    def _reduce(self, dims):
        b = _Bundle(dims)
        return reduce_zoi(b, dims, (-0.05, 0.05, -0.05, 0.05), 0.01,
                          ("EVF", "TEMP", "V1", "V2"),
                          window=(0.3, 1.0), evf_threshold=0.5)

    def test_void_points_masked_for_material_fields(self):
        r = self._reduce(DomainDims(0.1, 0.1, 0.1, 0.1))
        # material fields valid only on the 8 material points
        assert r.valid["TEMP"].sum() == 8
        assert not r.valid["TEMP"][8:].any()
        assert np.all(np.isnan(r.means["TEMP"][~_MAT]))
        assert np.all(np.isfinite(r.means["TEMP"][_MAT]))

    def test_evf_is_unmasked(self):
        r = self._reduce(DomainDims(0.1, 0.1, 0.1, 0.1))
        # EVF is meaningful everywhere in the ZOI (fill fraction)
        assert r.valid["EVF"].all()

    def test_force_is_windowed_mean(self):
        d = DomainDims(0.1, 0.1, 0.1, 0.1)
        r = self._reduce(d)
        infl = sum(np.exp(-getattr(d, n) / 0.05) for n in DIMENSION_NAMES)
        assert r.force_mean == pytest.approx(100.0 * (1 + 0.5 * infl))

    def test_window_frame_count(self):
        r = self._reduce(DomainDims(0.1, 0.1, 0.1, 0.1))
        # 10 frames, window (0.3,1) keeps t >= 0.3*t_end -> 7 frames
        assert r.n_window_frames == 7


# --- zoi_discrepancy ----------------------------------------------------------
class TestDiscrepancy:
    def test_identical_reductions_are_zero(self):
        d = DomainDims(0.1, 0.1, 0.1, 0.1)
        b = _Bundle(d)
        r = reduce_zoi(b, d, (-0.05, 0.05, -0.05, 0.05), 0.01,
                       ("EVF", "TEMP", "V1", "V2"))
        disc = zoi_discrepancy(r, r)
        for q in ("EVF", "TEMP", "V1", "V2", "force"):
            assert disc[q] == pytest.approx(0.0, abs=1e-12)

    def test_change_shrinks_as_domain_grows(self):
        def reduce(d):
            b = _Bundle(d)
            return reduce_zoi(b, d, (-0.05, 0.05, -0.05, 0.05), 0.01,
                              ("EVF", "TEMP", "V1", "V2"))
        small = reduce(DomainDims(0.05, 0.05, 0.05, 0.05))
        small_g = reduce(DomainDims(0.07, 0.05, 0.05, 0.05))
        big = reduce(DomainDims(0.30, 0.30, 0.30, 0.30))
        big_g = reduce(DomainDims(0.32, 0.30, 0.30, 0.30))
        # a fixed grow perturbs the ZOI less when the domain is already large
        assert (zoi_discrepancy(big_g, big)["TEMP"]
                < zoi_discrepancy(small_g, small)["TEMP"])

    def test_force_relative_change(self):
        a = ZoiReduction(DomainDims(0, 0, 0, 0), force_mean=110.0)
        b = ZoiReduction(DomainDims(0, 0, 0, 0), force_mean=100.0)
        assert zoi_discrepancy(a, b, ("force",))["force"] == pytest.approx(0.1)


# --- run_domain_convergence ---------------------------------------------------
class TestConvergenceDriver:
    ZOI = (-0.04, 0.04, -0.04, 0.04)

    def test_grow_elems_below_one_raises(self):
        with pytest.raises(ValueError):
            run_domain_convergence(_runner(), _Cfg(), self.ZOI,
                                   DomainDims(0.1, 0.1, 0.1, 0.1),
                                   grid_step=0.01, elem_size=0.005, grow_elems=0)

    def test_zoi_outside_start(self):
        # ZOI larger than the starting domain
        res = run_domain_convergence(
            _runner(), _Cfg(), (-0.2, 0.2, -0.2, 0.2),
            DomainDims(0.1, 0.1, 0.1, 0.1),
            grid_step=0.01, elem_size=0.005, margin_elems=1)
        assert res.stopped_by == "zoi_outside" and not res.converged

    def test_converges_for_decaying_influence(self):
        res = run_domain_convergence(
            _runner(lam=0.02), _Cfg(), self.ZOI,
            DomainDims(0.05, 0.05, 0.05, 0.05),
            grid_step=0.01, elem_size=0.005,
            tolerances={q: 0.02 for q in ("EVF", "TEMP", "V1", "V2", "force")},
            grow_elems=4, margin_elems=1, max_iterations=12)
        assert res.converged and res.stopped_by == "converged"
        # grew beyond the (tiny) starting domain
        assert res.dims.l_wp >= 0.05 and res.n_runs > 0

    def test_diagonal_ceiling_stops_growth(self):
        # A large starting domain near the ceiling with a tolerance it cannot
        # meet must stop by "diagonal" rather than run forever.
        res = run_domain_convergence(
            _runner(lam=10.0), _Cfg(), self.ZOI,          # lam huge -> never settles
            DomainDims(0.10, 0.10, 0.10, 0.10),
            grid_step=0.01, elem_size=0.005,
            tolerances={"TEMP": 1e-9}, field_vars=("EVF", "TEMP", "V1", "V2"),
            grow_elems=4, margin_elems=1, max_iterations=20)
        assert res.stopped_by in ("diagonal", "max_iter")
        assert not res.converged


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
