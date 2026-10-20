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
    bilinear_field, zoi_scalars, MeshGciResult,
)


def run_mesh_gci(*a, **kw):
    """The driver with the space-time mean selection (the analytic scalar
    model below); the pointwise selection is tested in TestPointwise."""
    kw.setdefault("selection", "mean")
    return mg.run_mesh_gci(*a, **kw)


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

    def test_low_order_falls_back_to_the_finest_value(self):
        # GCI of 2026-10-20 (ms 25600, 0.5/1/2 um): T has p = 0.25 and is not
        # trusted; Vx has p = 0.83 and keeps its extrapolate.
        t = quantity_gci(237.52818948, 233.96130514, 229.72458197, 2.0, 2.0)
        assert t.p == pytest.approx(0.248, abs=1e-3) and not t.reliable
        vx = quantity_gci(252.55554406, 256.52186565, 263.57172664, 2.0, 2.0)
        assert vx.p == pytest.approx(0.830, abs=1e-3) and vx.reliable

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


def _abs(rel):
    """Absolute tolerances eps_q equal to `rel` times |B_q| (units of q)."""
    return {q: rel * abs(b) for q, b in _BASE.items()}


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
            tolerances=_abs(0.005))
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
            tolerances=_abs(0.005))
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
            tolerances=_abs(0.005))
        assert res.recommended_size == pytest.approx(0.02)

    def test_selection_is_absolute_and_independent_of_the_t_origin(
            self, monkeypatch):
        # TEMP alone drifts by 10 h^2 * 300 K: 1.2 K at h = 0.02. With
        # eps_T = 1.5 K the coarsest mesh passes, whatever the temperature
        # origin (K or degC): only the difference |f(h) - f_ref| counts.
        def fake(bundle, zoi, gs, field_vars, offset=0.0, **kw):
            h = bundle.elem_size
            out = {q: _BASE[q] for q in field_vars}
            out["TEMP"] = offset + 300.0 * (1.0 + 10.0 * h * h)
            out.update(Fc=_BASE["Fc"] * h, Ff=_BASE["Ff"] * h)
            return out
        recs = []
        for offset in (0.0, -273.15):
            monkeypatch.setattr(mg, "zoi_scalars",
                                lambda *a, _o=offset, **k: fake(*a, _o, **k))
            res = run_mesh_gci(
                _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
                finest_elem_size=0.005, ratio=2.0, n_meshes=3,
                tolerances={"TEMP": 1.5})
            recs.append(res.recommended_size)
        assert recs == [pytest.approx(0.02), pytest.approx(0.02)]

    def test_lower_bound_is_kept_when_no_mesh_is_within_tolerance(self):
        # dev(0.005) = 10*0.005^2 = 2.5e-4 relative > 1e-6: no mesh passes.
        # Without h_min the study recommends nothing; with h_min = the finest
        # mesh it keeps h_min and lists the quantities outside eps_q.
        kw = dict(grid_step=0.01, finest_elem_size=0.005, ratio=2.0,
                  n_meshes=3, tolerances=_abs(1e-6))
        res = run_mesh_gci(_runner(), _Cfg(), self.ZOI, self.DIMS, **kw)
        assert res.recommended_size is None and not res.at_lower_bound
        res = run_mesh_gci(_runner(), _Cfg(), self.ZOI, self.DIMS,
                           min_elem_size=0.005, **kw)
        assert res.recommended_size == pytest.approx(0.005)
        assert res.at_lower_bound
        assert set(res.exceeding) == set(_BASE)
        assert res.exceeding["TEMP"] == pytest.approx(250.0, rel=1e-6)

    def test_lower_bound_not_used_above_it_or_when_a_mesh_passes(self):
        # finest 0.005 above h_min 0.0025: the plan can still be refined
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            tolerances=_abs(1e-6), min_elem_size=0.0025)
        assert res.recommended_size is None and not res.at_lower_bound
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            tolerances=_abs(0.005), min_elem_size=0.005)
        assert res.recommended_size == pytest.approx(0.02)
        assert not res.at_lower_bound and res.exceeding == {}

    def test_no_tolerance_recommends_nothing(self):
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3)
        assert res.recommended_size is None
        assert res.per_quantity["TEMP"].p == pytest.approx(2.0, rel=1e-6)

    def test_tighter_tolerance_recommends_finer(self):
        # dev(0.02)=0.004 > 0.002 -> excluded; dev(0.01)=0.001 < 0.002 -> 0.01
        res = run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.005, ratio=2.0, n_meshes=3,
            tolerances=_abs(0.002))
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
            tolerances=dict({q: 0.05 * abs(v) for q, v in base.items()},
                            Fc=1.0, Ff=1.0))
        # Fc/Ff are flat -> unreliable -> reference = finest -> deviation 0 ->
        # they no longer block; the fields deviate < 5% -> a mesh is recommended.
        assert res.per_quantity["Fc"].reliable is False
        assert res.per_quantity["Ff"].reliable is False
        assert res.recommended_size is not None


# --- Pointwise selection (E_q against the finest mesh) -----------------------
class TestPointwise:
    """The pointwise selection with an analytic deviation model: E_q / eps_q
    of mesh h against the finest = K_q (h - h_finest) / eps_q."""
    ZOI = (-0.04, 0.04, -0.04, 0.04)
    DIMS = DomainDims(0.255, 0.055, 0.255, 0.055)
    K = {"EVF": 0.0, "TEMP": 1000.0, "V1": 2000.0, "V2": 0.0,
         "Fc": 500.0, "Ff": 0.0}

    @pytest.fixture(autouse=True)
    def _patch_pointwise(self, monkeypatch):
        monkeypatch.setattr(mg, "_pointwise_sample",
                            lambda bundle, *a, **kw: bundle.elem_size)
        k = self.K

        def dev(h_ref, h, zoi, gs, window, evf, tol, quantities):
            return {q: k[q] * (h - h_ref) / tol[q] for q in quantities
                    if q in tol}
        monkeypatch.setattr(mg, "_pointwise_deviation", dev)

    def _run(self, tol, **kw):
        return mg.run_mesh_gci(
            _runner(), _Cfg(), self.ZOI, self.DIMS, grid_step=0.01,
            finest_elem_size=0.001, ratio=2.0, n_meshes=3,
            tolerances=tol, **kw)

    def test_default_is_pointwise(self):
        res = self._run({q: 10.0 for q in _BASE})
        assert res.selection == "pointwise"
        assert set(res.deviation) == {0.002, 0.004}

    def test_coarsest_mesh_within_eps_against_finest(self):
        # E_V1(h) = 2000 (h - 0.001) / 10: 0.2 at 0.002, 0.6 at 0.004.
        res = self._run({q: 10.0 for q in _BASE})
        assert res.recommended_size == pytest.approx(0.004)
        assert res.deviation[0.004]["V1"] == pytest.approx(0.6)

    def test_middle_mesh_when_coarsest_fails(self):
        # eps 4: E_V1 = 0.5 at 0.002, 1.5 at 0.004.
        res = self._run({q: 4.0 for q in _BASE})
        assert res.recommended_size == pytest.approx(0.002)

    def test_finest_is_never_recommended_against_itself(self):
        # eps 1: every coarser mesh fails; no lower bound -> no size.
        res = self._run({q: 1.0 for q in _BASE})
        assert res.recommended_size is None
        assert not res.at_lower_bound

    def test_lower_bound_reports_the_next_coarser_mesh(self):
        res = self._run({q: 1.0 for q in _BASE}, min_elem_size=0.001)
        assert res.recommended_size == pytest.approx(0.001)
        assert res.at_lower_bound
        # E at 0.002: V1 2.0, TEMP 1.0 (not > 1), Fc 0.5.
        assert res.exceeding == {"V1": pytest.approx(2.0)}

    def test_gci_scalars_still_reported(self):
        res = self._run({q: 10.0 for q in _BASE})
        assert res.per_quantity["V1"].p == pytest.approx(2.0, rel=1e-6)


class TestPointwiseDeviation:
    """`_pointwise_deviation` on real samples: E_q is the pointwise metric."""

    def test_constant_offset_gives_offset_over_eps(self):
        from gui.sensitivity.domain_independence import ZoiSample
        t = np.linspace(0.3, 1.0, 8)
        npts = 9 * 9            # roi_grid((-0.04, 0.04)^2, 0.01)
        a = ZoiSample(field_times=t)
        a.fields = {"EVF": np.ones((8, npts)),
                    "Vx": np.full((8, npts), 100.0)}
        b = ZoiSample(field_times=t)
        b.fields = {"EVF": np.ones((8, npts)),
                    "Vx": np.full((8, npts), 103.0)}
        a.force_times = b.force_times = np.zeros(0)
        out = mg._pointwise_deviation(
            a, b, (-0.04, 0.04, -0.04, 0.04), 0.01, (0.3, 1.0), 0.5,
            {"V1": 10.0, "TEMP": 10.0}, ["V1", "TEMP"])
        assert out["V1"] == pytest.approx(0.3)
        assert out["TEMP"] == float("inf")      # not in the samples
