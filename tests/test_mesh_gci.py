# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.mesh_gci (GCI / Richardson mesh convergence).

The GCI math is checked against an EXACT power-law error f = f_exact + C h^p,
for which the observed order, the extrapolated value and the asymptotic ratio
are known in closed form. The driver is exercised with an analytic scalar model
so the recommended mesh and the per-quantity outcomes are deterministic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gui.core.domain_sizing import DomainDims
import gui.sensitivity.mesh_gci as mg
from gui.sensitivity.mesh_gci import (
    observed_order, extrapolated_value, gci, asymptotic_ratio, quantity_gci,
    bilinear_field, zoi_scalars, run_mesh_gci, MeshGciResult,
)


def _power_law(f_exact, C, p, sizes):
    """f_i = f_exact + C * h_i^p for h_i in `sizes`."""
    return [f_exact + C * (h ** p) for h in sizes]


# --- Pure GCI / Richardson math ----------------------------------------------
class TestGciMath:
    # h1<h2<h3 finest->coarsest, r=2
    SIZES = [1.0, 2.0, 4.0]

    def test_observed_order_recovers_p(self):
        for p in (1.0, 2.0, 3.0):
            f1, f2, f3 = _power_law(10.0, 1.0, p, self.SIZES)
            assert observed_order(f1, f2, f3, 2.0, 2.0) == pytest.approx(p, rel=1e-9)

    def test_richardson_recovers_exact(self):
        f1, f2, f3 = _power_law(10.0, 1.0, 2.0, self.SIZES)
        assert extrapolated_value(f1, f2, 2.0, 2.0) == pytest.approx(10.0, abs=1e-9)

    def test_gci_formula(self):
        f1, f2, f3 = _power_law(10.0, 1.0, 2.0, self.SIZES)  # 11, 14, 26
        # Fs |(f1-f2)/f1| / (r^p - 1) = 1.25 * (3/11) / 3
        assert gci(f1, f2, 2.0, 2.0) == pytest.approx(1.25 * (3.0 / 11.0) / 3.0)

    def test_asymptotic_ratio_is_one_for_power_law(self):
        f1, f2, f3 = _power_law(10.0, 1.0, 2.0, self.SIZES)
        e21, e32 = f2 - f1, f3 - f2
        assert asymptotic_ratio(e21, e32, 2.0, 2.0) == pytest.approx(1.0)

    def test_quantity_gci_bundle(self):
        f1, f2, f3 = _power_law(300.0, 5.0, 2.0, self.SIZES)
        g = quantity_gci(f1, f2, f3, 2.0, 2.0)
        assert g.p == pytest.approx(2.0)
        assert g.f_extrapolated == pytest.approx(300.0)
        assert g.asymptotic_ratio == pytest.approx(1.0)
        assert g.monotonic

    def test_converged_solution_gives_nan_order(self):
        # identical solutions -> order undefined (division by ~0)
        assert math.isnan(observed_order(5.0, 5.0, 5.0, 2.0, 2.0))

    def test_non_constant_ratio_iterates(self):
        # r21 != r32: p must still recover the power law (fixed-point solve)
        sizes = [1.0, 2.0, 3.0]                # r21=2, r32=1.5
        f1, f2, f3 = _power_law(10.0, 1.0, 2.0, sizes)
        assert observed_order(f1, f2, f3, 2.0, 1.5) == pytest.approx(2.0, rel=1e-6)


# --- Bilinear sampling --------------------------------------------------------
class TestBilinearField:
    def test_linear_field_reproduced_exactly(self, monkeypatch):
        # centroids on a coarse grid; a linear field must be reproduced exactly
        xs, ys = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        centroids = np.column_stack([xs.ravel(), ys.ravel()])
        monkeypatch.setattr(mg, "element_centroids_xy", lambda b, i: centroids)

        def lin(x, y):
            return 2.0 * x - 3.0 * y + 1.0

        vals = lin(centroids[:, 0], centroids[:, 1])[None, :]  # (1, Nc)

        class _B:
            def field(self, inst, var):
                return vals
        pts = np.array([[0.25, 0.25], [0.5, 0.75], [0.6, 0.4]])
        out = bilinear_field(_B(), "TEMP", "EULER", pts)
        expected = lin(pts[:, 0], pts[:, 1])
        np.testing.assert_allclose(out[0], expected, atol=1e-9)


# --- ZOI scalar reduction (EVF masking) --------------------------------------
class TestZoiScalars:
    def test_void_points_excluded_via_evf_mask(self, monkeypatch):
        NT, NP = 6, 10
        mat = np.array([True] * 7 + [False] * 3)

        def fake_bilinear(bundle, var, inst, points, frames=None, weights=None):
            out = np.zeros((NT, NP))
            if var == "EVF":
                out[:, mat] = 1.0
                out[:, ~mat] = 0.0
            else:
                out[:, mat] = 42.0
                out[:, ~mat] = -999.0        # garbage in void, must be excluded
            return out

        monkeypatch.setattr(mg, "bilinear_field", fake_bilinear)
        monkeypatch.setattr(mg, "element_centroids_xy",
                            lambda b, i: np.zeros((3, 2)))
        monkeypatch.setattr(mg, "interp_weights", lambda c, p: None)
        monkeypatch.setattr(mg, "roi_grid", lambda roi, step: np.zeros((NP, 2)))
        monkeypatch.setattr(mg, "eulerian_instance", lambda b: "EULER")

        class _B:
            @property
            def times(self):
                return np.linspace(0.0, 3e-4, NT)
        out = zoi_scalars(_B(), (-1, 1, -1, 1), 0.1, ("EVF", "TEMP"),
                          window=(0.0, 1.0), evf_threshold=0.5,
                          force_channels={})
        assert out["TEMP"] == pytest.approx(42.0)     # void -999 excluded
        assert out["EVF"] == pytest.approx(0.7)       # 7/10 material, unmasked


# --- Driver -------------------------------------------------------------------
class _Cfg:
    class _G:
        h_wp = h_void = l_wp = l_void = 0.0

    def __init__(self):
        self.euler_geometry = _Cfg._G()
        self.elem_size = 0.0


class _Bundle:
    def __init__(self, elem_size):
        self.elem_size = elem_size


def _runner():
    def run(cfg):
        return _Bundle(cfg.elem_size)
    return run


# analytic scalar model: f_q(h) = B_q (1 + 10 h^2)  -> p=2, f_ext=B_q
_BASE = {"EVF": 0.8, "TEMP": 300.0, "V1": 10.0, "V2": -5.0,
         "Fc": 100.0, "Ff": 50.0}


@pytest.fixture(autouse=True)
def _patch_scalars(monkeypatch):
    def fake_scalars(bundle, zoi, grid_step, field_vars, **kw):
        h = bundle.elem_size
        out = {q: _BASE[q] * (1.0 + 10.0 * h * h) for q in field_vars}
        for label in ("Fc", "Ff"):
            out[label] = _BASE[label] * h * (1.0 + 10.0 * h * h)  # raw RF ~ width
        return out
    monkeypatch.setattr(mg, "zoi_scalars", fake_scalars)


class TestDriver:
    ZOI = (-0.04, 0.04, -0.04, 0.04)
    DIMS = DomainDims(0.255, 0.055, 0.255, 0.055)

    def test_forces_are_divided_by_model_width(self):
        # fake raw RF = _BASE * h * (1 + 10 h^2) (proportional to the width h);
        # the driver must store force per width = _BASE * (1 + 10 h^2).
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            tolerances={q: 0.005 for q in _BASE})
        h1 = min(res.sizes)
        assert res.scalars[h1]["Fc"] == pytest.approx(
            _BASE["Fc"] * (1.0 + 10.0 * h1 * h1), rel=1e-9)
        # and consequently the extrapolated force recovers _BASE (power law)
        assert res.per_quantity["Fc"].f_extrapolated == pytest.approx(
            _BASE["Fc"], rel=1e-6)

    def test_recovers_order_and_extrapolation(self):
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            tolerances={q: 0.005 for q in _BASE})
        assert res.n_runs == 3
        assert res.in_asymptotic_range
        for q in ("TEMP", "V1", "Fc"):
            g = res.per_quantity[q]
            assert g.p == pytest.approx(2.0, rel=1e-6)
            assert g.f_extrapolated == pytest.approx(_BASE[q], rel=1e-6)

    def test_recommends_coarsest_within_tolerance(self):
        # dev(0.02)=10*0.02^2=0.004 < 0.005 -> coarsest 0.02 recommended
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            tolerances={q: 0.005 for q in _BASE})
        assert res.recommended_size == pytest.approx(0.02)

    def test_tighter_tolerance_recommends_finer(self):
        # dev(0.02)=0.004 > 0.002 -> excluded; dev(0.01)=0.001 < 0.002 -> 0.01
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            tolerances={q: 0.002 for q in _BASE})
        assert res.recommended_size == pytest.approx(0.01)

    def test_min_elem_size_floor(self):
        # finest below the floor is clamped up to it
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.003, ratio=2.0, n_meshes=3, min_elem_size=0.006)
        assert min(res.sizes) == pytest.approx(0.006)

    def test_cancel_stops_early(self):
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            should_cancel=lambda: True)
        assert res.stopped_by == "cancelled" and res.n_runs == 0


class TestVectorizedInterp:
    """The single-triangulation barycentric interpolation reproduces a linear
    field exactly over many frames and returns NaN outside the hull."""

    def test_linear_multiframe_exact_and_outside_hull_nan(self):
        from gui.sensitivity.mesh_gci import interp_weights, apply_weights
        xs, ys = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        centroids = np.column_stack([xs.ravel(), ys.ravel()])
        pts = np.array([[0.25, 0.25], [0.5, 0.5], [2.0, 2.0]])  # last outside
        w = interp_weights(centroids, pts)

        def lin(a, b, c):
            return a * centroids[:, 0] + b * centroids[:, 1] + c
        vals = np.stack([lin(2.0, -3.0, 1.0), lin(1.0, 1.0, 0.0)])   # (2, Nc)
        out = apply_weights(vals, w)                                 # (2, 3)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(
            out[0, :2], [2 * .25 - 3 * .25 + 1, 2 * .5 - 3 * .5 + 1], atol=1e-9)
        np.testing.assert_allclose(out[1, :2], [.5, 1.0], atol=1e-9)
        assert np.isnan(out[0, 2]) and np.isnan(out[1, 2])

    def test_bilinear_field_reuses_weights(self, monkeypatch):
        import gui.sensitivity.mesh_gci as mg
        xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
        centroids = np.column_stack([xs.ravel(), ys.ravel()])
        monkeypatch.setattr(mg, "element_centroids_xy", lambda b, i: centroids)
        vals = (2.0 * centroids[:, 0] - centroids[:, 1])[None, :]

        class _B:
            def field(self, inst, var):
                return vals
        pts = np.array([[0.3, 0.4], [0.6, 0.2]])
        w = mg.interp_weights(centroids, pts)
        a = mg.bilinear_field(_B(), "TEMP", "E", pts)             # builds weights
        b = mg.bilinear_field(_B(), "TEMP", "E", pts, weights=w)  # reuses
        np.testing.assert_allclose(a, b, atol=1e-12)


class TestReliabilityGuard:
    def test_reliable_flag_cases(self):
        # clean power law -> reliable
        assert quantity_gci(11.0, 14.0, 26.0, 2.0, 2.0).reliable
        # constant -> p NaN -> unreliable
        assert not quantity_gci(-80.0, -80.0, -80.0, 1.5, 1.5).reliable
        # near-constant, tiny non-scaling wobble (attempt-4 Fc) -> p~0 -> unreliable
        assert not quantity_gci(-82.25, -81.47, -80.67, 1.5, 1.5).reliable
        # non-monotonic (attempt-4 TEMP) -> unreliable
        assert not quantity_gci(224.8, 220.6, 220.9, 1.5, 1.5).reliable
        # slow but genuine convergence (attempt-4 V2) -> reliable
        assert quantity_gci(198.8, 203.6, 207.1, 1.5, 1.5).reliable

    def test_unreliable_quantity_does_not_block_recommendation(self, monkeypatch):
        import gui.sensitivity.mesh_gci as mg

        base = {"EVF": 0.8, "TEMP": 300.0, "V1": 10.0, "V2": -5.0}

        def fake(bundle, zoi, gs, field_vars, **kw):
            h = bundle.elem_size
            out = {q: base[q] * (1.0 + 10.0 * h * h) for q in field_vars}
            # raw RF ~ width (h); after the driver's /h it is a CONSTANT -80 /
            # -30 N/mm -> converged -> unreliable extrapolation.
            out["Fc"] = -80.0 * h
            out["Ff"] = -30.0 * h
            return out
        monkeypatch.setattr(mg, "zoi_scalars", fake)

        res = run_mesh_gci(
            (lambda cfg: _Bundle(cfg.elem_size)), _Cfg(),
            (-0.04, 0.04, -0.04, 0.04), DomainDims(0.255, 0.055, 0.255, 0.055),
            grid_step=0.01, finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            tolerances={q: 0.05 for q in ("EVF", "TEMP", "V1", "V2", "Fc", "Ff")})
        # Fc/Ff are flat -> unreliable -> reference = finest -> deviation 0 ->
        # they no longer block; the fields deviate < 5% -> a mesh is recommended.
        assert res.per_quantity["Fc"].reliable is False
        assert res.per_quantity["Ff"].reliable is False
        assert res.recommended_size is not None
