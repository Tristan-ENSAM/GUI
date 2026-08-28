# -*- coding: utf-8 -*-
"""
Unit tests for gui.sensitivity.domain_opt (Eulerian-domain optimiser).

Fully offline: the simulation is replaced by an analytic `sample_fn` whose ROI
field decays with the relevant dimension, so the converged size is known in
closed form and the search can be checked without Abaqus.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.domain_opt import roi_rmse, roi_error, all_below, DomainOptimizer


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
class TestMetric:

    def test_constant_difference(self):
        a = np.full((2, 3), 1.0)
        b = np.full((2, 3), 1.5)
        assert roi_rmse(a, b) == pytest.approx(0.5)

    def test_is_mean_absolute_not_rms(self):
        # differences [0, 2] -> MAD = 1.0, whereas RMS would be sqrt(2) ~ 1.414
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, 2.0]])
        assert roi_rmse(a, b) == pytest.approx(1.0)
        assert roi_error(a, b) == pytest.approx(1.0)

    def test_zero_for_identical(self):
        a = np.array([[1.0, 2.0, 3.0]])
        assert roi_rmse(a, a) == pytest.approx(0.0)

    def test_nan_safe(self):
        a = np.array([[1.0, np.nan, 3.0]])
        b = np.array([[1.0, 0.0, 3.0]])
        assert roi_rmse(a, b) == pytest.approx(0.0)   # NaN cell ignored

    def test_frame_count_mismatch_aligns(self):
        a = np.full((3, 2), 1.0)
        b = np.full((5, 2), 1.0)                       # extra frames
        assert roi_rmse(a, b) == pytest.approx(0.0)

    def test_all_below(self):
        assert all_below({"Vx": 0.4, "T": 1.0}, {"Vx": 0.5, "T": 2.0}) is True
        assert all_below({"Vx": 0.6}, {"Vx": 0.5}) is False
        assert all_below({"Vx": float("nan")}, {"Vx": 0.5}) is False


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
def _decay_sample_fn(active, A, L, n_t=2, n_p=3, quantities=("Vx",)):
    """ROI field constant over (Nt, Np), equal to A*exp(-d/L) where d is the
    `active` dimension. The boundary effect decays as that dimension grows."""
    def sample_fn(dims):
        val = A * math.exp(-getattr(dims, active) / L)
        arr = np.full((n_t, n_p), val, dtype=float)
        return {q: arr.copy() for q in quantities}
    return sample_fn


class TestOptimizer:

    def _init(self):
        return DomainDims(h_wp=0.05, h_void=0.05, l_wp=0.05, l_void=0.05)

    def test_finds_active_dimension(self):
        # A*exp(-d/L) < eps  <=>  d > L*ln(A/eps) = 0.05*ln(20) ~ 0.1498
        # -> smallest 0.01 multiple is 0.15.
        opt = DomainOptimizer(
            _decay_sample_fn("l_wp", A=10.0, L=0.05),
            thresholds={"Vx": 0.5}, elem_size=0.01,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")},
            quantities=("Vx",))
        res = opt.optimize(self._init())
        assert res.dims.l_wp == pytest.approx(0.15)
        # inactive dimensions stay at their initial floor
        assert res.dims.h_wp == pytest.approx(0.05)
        assert res.dims.h_void == pytest.approx(0.05)
        assert res.dims.l_void == pytest.approx(0.05)
        assert res.converged is True

    def test_logarithmic_run_count(self):
        opt = DomainOptimizer(
            _decay_sample_fn("l_wp", A=10.0, L=0.05),
            thresholds={"Vx": 0.5}, elem_size=0.01,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")},
            quantities=("Vx",))
        res = opt.optimize(self._init())
        # Doubling + bisection over 4 dims x <=2 passes stays small.
        assert res.n_runs < 30

    def test_cap_is_respected(self):
        # Unsatisfiable threshold + small cap -> the active dim stops at the cap
        # and the result is reported as not converged.
        opt = DomainOptimizer(
            _decay_sample_fn("l_wp", A=10.0, L=0.05),
            thresholds={"Vx": 1e-9}, elem_size=0.01,
            caps={"l_wp": 0.08, "h_wp": 0.06, "h_void": 0.06, "l_void": 0.06},
            quantities=("Vx",), max_doublings=12)
        res = opt.optimize(self._init())
        assert res.dims.l_wp <= 0.08 + 1e-9
        assert res.converged is False

    def test_multi_quantity_waits_for_all(self):
        # Two quantities with different decay lengths; the stricter (longer L)
        # dictates the converged size.
        def sample_fn(dims):
            d = dims.l_wp
            fast = np.full((2, 3), 10.0 * math.exp(-d / 0.02))   # decays fast
            slow = np.full((2, 3), 10.0 * math.exp(-d / 0.05))   # decays slow
            return {"Vx": fast, "T": slow}
        opt = DomainOptimizer(
            sample_fn, thresholds={"Vx": 0.5, "T": 0.5}, elem_size=0.01,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")},
            quantities=("Vx", "T"))
        res = opt.optimize(self._init())
        # The slow (T) quantity needs ~0.15; the fast one only ~0.06. The
        # result must satisfy BOTH, so it is governed by the slow one.
        assert res.dims.l_wp == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# ROI extraction from a (fake) bundle
# ---------------------------------------------------------------------------
from gui.sensitivity.domain_opt import grid_keys, roi_samples, align_to_reference


class _FakeBundle:
    """Minimal bundle exposing element_centroids_init + field."""
    def __init__(self, centroids_xy, fields):
        # centroids_xy: (n, 2); store as (n, 3) like a real bundle.
        c = np.asarray(centroids_xy, dtype=float)
        self._c = np.column_stack([c, np.zeros(len(c))])
        self._fields = fields            # {var: (n_t, n)}

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        return self._fields[var]


def _grid_bundle(elem=0.01, nx=5, ny=4, n_t=2, var="Vx"):
    """A structured cell grid with centroids at (i+0.5)*elem, value encoding
    the cell index so selection/order can be checked: v = 100*kx + ky + 10*t."""
    cx = (np.arange(nx) + 0.5) * elem
    cy = (np.arange(ny) + 0.5) * elem
    XX, YY = np.meshgrid(cx, cy)                       # (ny, nx)
    cents = np.column_stack([XX.ravel(), YY.ravel()])  # (ny*nx, 2)
    kx = np.round(cents[:, 0] / elem - 0.5).astype(int)
    ky = np.round(cents[:, 1] / elem - 0.5).astype(int)
    vals = np.stack([100.0 * kx + ky + 10.0 * t for t in range(n_t)])
    return _FakeBundle(cents, {var: vals})


class TestRoiExtraction:

    def test_grid_keys(self):
        c = np.array([[0.005, 0.005], [0.015, 0.025], [-0.005, -0.015]])
        k = grid_keys(c, 0.01)
        assert k.tolist() == [[0, 0], [1, 2], [-1, -2]]

    def test_roi_selection_and_order(self):
        b = _grid_bundle(elem=0.01, nx=5, ny=4)
        # ROI selects kx in {1,2,3}, ky in {1,2} -> 6 cells.
        keys, vals = roi_samples(b, "Vx", "Euler",
                                 (0.01, 0.035, 0.01, 0.025), 0.01)
        assert keys.shape == (6, 2)
        assert vals.shape == (2, 6)
        # canonical order = sorted by (ky, kx)
        expected = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]
        assert [tuple(k) for k in keys] == expected
        # value encodes the cell: v(frame0) = 100*kx + ky
        assert vals[0].tolist() == [100 * kx + ky for (kx, ky) in expected]

    def test_alignment_across_domain_sizes(self):
        # Two domains, same element size and anchored grid, different extents:
        # their ROI keys coincide, so aligning the larger onto the smaller's
        # reference keys reproduces the same values.
        small = _grid_bundle(elem=0.01, nx=5, ny=4)
        large = _grid_bundle(elem=0.01, nx=9, ny=7)     # bigger domain
        roi = (0.01, 0.035, 0.01, 0.025)
        ref_keys, ref_vals = roi_samples(small, "Vx", "Euler", roi, 0.01)
        keys_l, vals_l = roi_samples(large, "Vx", "Euler", roi, 0.01)
        aligned = align_to_reference(keys_l, vals_l, ref_keys)
        assert np.allclose(aligned, ref_vals)           # identical on common ROI

    def test_align_fills_nan_for_missing(self):
        b = _grid_bundle(elem=0.01, nx=5, ny=4)
        keys, vals = roi_samples(b, "Vx", "Euler",
                                 (0.01, 0.025, 0.01, 0.015), 0.01)
        # reference asks for an extra key not present -> that column is NaN
        ref = np.vstack([keys, np.array([[99, 99]])])
        aligned = align_to_reference(keys, vals, ref)
        assert np.all(np.isnan(aligned[:, -1]))
        assert np.allclose(aligned[:, :-1], vals)


# ---------------------------------------------------------------------------
# Progress hook + cancellation
# ---------------------------------------------------------------------------
class TestProgressAndCancel:

    def test_progress_events_emitted(self):
        opt = DomainOptimizer(
            _decay_sample_fn("l_wp", A=10.0, L=0.05),
            thresholds={"Vx": 0.5}, elem_size=0.01,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")},
            quantities=("Vx",))
        events = []
        res = opt.optimize(DomainDims(0.05, 0.05, 0.05, 0.05),
                           progress_cb=events.append)
        assert events, "no progress events"
        assert events[-1]["phase"] == "done"
        # at least one per-dimension event with a DimResult
        dim_events = [e for e in events if e["phase"] == "dimension"]
        assert dim_events and dim_events[0]["result"].name in (
            "h_wp", "h_void", "l_wp", "l_void")

    def test_cancellation_stops_early(self):
        opt = DomainOptimizer(
            _decay_sample_fn("l_wp", A=10.0, L=0.05),
            thresholds={"Vx": 0.5}, elem_size=0.01,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")},
            quantities=("Vx",))
        res = opt.optimize(DomainDims(0.05, 0.05, 0.05, 0.05),
                           should_cancel=lambda: True)   # cancel immediately
        assert res.converged is False

    def test_live_compare_events(self):
        # each doubling comparison emits a live "compare" event with errors
        opt = DomainOptimizer(
            _decay_sample_fn("l_wp", A=10.0, L=0.05),
            thresholds={"Vx": 0.5}, elem_size=0.01,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")},
            quantities=("Vx",))
        events = []
        opt.optimize(DomainDims(0.05, 0.05, 0.05, 0.05),
                     progress_cb=events.append)
        comp = [e for e in events if e["phase"] == "compare"]
        assert comp, "no live compare events"
        assert "errors" in comp[0] and "value" in comp[0] and "name" in comp[0]


class TestDomainOptWorker:

    def test_worker_runs_and_emits_result(self, qapp):
        from gui.sensitivity.domain_opt_worker import DomainOptWorker
        w = DomainOptWorker(
            sample_fn=_decay_sample_fn("l_wp", A=10.0, L=0.05),
            initial=DomainDims(0.05, 0.05, 0.05, 0.05),
            thresholds={"Vx": 0.5}, elem_size=0.01,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")},
            quantities=("Vx",))
        got = {}
        w.finished_ok.connect(lambda r: got.update(result=r))
        events = []
        w.progress.connect(events.append)
        w.run()                                   # run synchronously in-thread
        assert "result" in got
        assert got["result"].dims.l_wp == pytest.approx(0.15)
        assert events and events[-1]["phase"] == "done"

    def test_worker_reports_failure(self, qapp):
        from gui.sensitivity.domain_opt_worker import DomainOptWorker

        def boom(dims):
            raise RuntimeError("sim failed")
        w = DomainOptWorker(
            sample_fn=boom, initial=DomainDims(0.05, 0.05, 0.05, 0.05),
            thresholds={"Vx": 0.5}, elem_size=0.01, quantities=("Vx",))
        msgs = []
        w.failed.connect(msgs.append)
        w.run()
        assert msgs and "sim failed" in msgs[0]


# ---------------------------------------------------------------------------
# sample_fn factory (with a fake Abaqus launcher)
# ---------------------------------------------------------------------------
class _MultiFieldBundle:
    """Fake bundle with several element fields on a structured cell grid."""
    def __init__(self, elem=0.01, nx=6, ny=5, n_t=2, fields=("V1", "V2", "TEMP")):
        cx = (np.arange(nx) + 0.5) * elem
        cy = (np.arange(ny) + 0.5) * elem
        XX, YY = np.meshgrid(cx, cy)
        cents = np.column_stack([XX.ravel(), YY.ravel()])
        self._c = np.column_stack([cents, np.zeros(len(cents))])
        kx = np.round(cents[:, 0] / elem - 0.5).astype(int)
        ky = np.round(cents[:, 1] / elem - 0.5).astype(int)
        self._f = {}
        for fi, f in enumerate(fields):
            self._f[f] = np.stack([100.0 * kx + ky + 10.0 * t + fi
                                   for t in range(n_t)])
        self.closed = False

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        return self._f[var]

    def close(self):
        self.closed = True


class TestMakeSampleFn:

    def test_returns_aligned_quantities(self):
        from gui.sensitivity.domain_opt import make_sample_fn
        from gui.core.model_config import ModelConfig

        made = []

        def run_bundle(cfg):
            # capture that the euler dims were set on the cfg
            made.append((cfg.euler_geometry.l_wp, cfg.euler_geometry.h_wp))
            return _MultiFieldBundle()

        qmap = {"Vx": "V1", "Vy": "V2", "T": "TEMP"}
        fn = make_sample_fn(ModelConfig(), run_bundle,
                            roi=(0.01, 0.045, 0.01, 0.035), elem_size=0.01,
                            quantity_field_map=qmap, instance="Euler")
        s = fn(DomainDims(h_wp=0.1, h_void=0.1, l_wp=0.2, l_void=0.1))
        assert set(s.keys()) == {"Vx", "Vy", "T"}
        # each is (Nt, Np); Np = cells in the ROI (kx in 1..4, ky in 1..3 = 12)
        assert s["Vx"].shape[1] == 12
        assert made[-1][0] == 0.2 and made[-1][1] == 0.1   # dims applied

    def test_failed_run_yields_nan(self):
        from gui.sensitivity.domain_opt import make_sample_fn
        from gui.core.model_config import ModelConfig
        fn = make_sample_fn(ModelConfig(), lambda cfg: None,
                            roi=(0.0, 0.05, 0.0, 0.05), elem_size=0.01,
                            quantity_field_map={"Vx": "V1"}, instance="Euler")
        s = fn(DomainDims(0.05, 0.05, 0.05, 0.05))
        assert np.all(np.isnan(s["Vx"]))

    def test_reference_keys_shared_across_calls(self):
        from gui.sensitivity.domain_opt import make_sample_fn
        from gui.core.model_config import ModelConfig
        # Two domain sizes; same anchored grid -> same ROI columns/order, so
        # the two sample arrays are directly comparable (identical here).
        fn = make_sample_fn(
            ModelConfig(),
            lambda cfg: _MultiFieldBundle(nx=int(round(cfg.euler_geometry.l_wp
                                                       / 0.01)) + 3),
            roi=(0.01, 0.045, 0.01, 0.035), elem_size=0.01,
            quantity_field_map={"Vx": "V1"}, instance="Euler")
        a = fn(DomainDims(0.05, 0.05, 0.06, 0.05))
        b = fn(DomainDims(0.05, 0.05, 0.20, 0.05))
        assert a["Vx"].shape == b["Vx"].shape
        assert np.allclose(a["Vx"], b["Vx"])


# ---------------------------------------------------------------------------
# Optimization tab (construction + config-derived logic, headless)
# ---------------------------------------------------------------------------
class TestOptimizationTab:

    def _tab(self):
        from gui.tabs.optimization_tab import OptimizationTab
        from gui.core.model_config import ModelConfig
        return OptimizationTab(ModelConfig())

    def test_config_inputs(self, qapp):
        tab = self._tab()
        inp = tab.config_inputs()
        # t1 = wp_y0 - tool_y0 = 0 - (-0.05) = 0.05 (default model)
        assert inp["t1"] == pytest.approx(0.05)
        assert inp["elem"] == pytest.approx(0.005)
        assert len(inp["roi"]) == 4

    def test_initial_domain_is_the_roi(self, qapp):
        tab = self._tab()
        c = tab.cfg
        # set a known measurement ROI (BBox) and a matching element size
        c.bbox.xmin, c.bbox.xmax = -0.20, 0.05
        c.bbox.ymin, c.bbox.ymax = -0.10, 0.15
        c.elem_size = 0.01
        tab.sp_margin.setValue(0)
        d0 = tab.compute_initial_dims()
        # domain-frame mapping: l_wp=-xmin, l_void=xmax, h_wp=-ymin, h_void=ymax
        assert d0.l_wp == pytest.approx(0.20)
        assert d0.l_void == pytest.approx(0.05)
        assert d0.h_wp == pytest.approx(0.10)
        assert d0.h_void == pytest.approx(0.15)

    def test_quantity_field_map_all_field_backed(self, qapp):
        tab = self._tab()
        m = tab.quantity_field_map()
        assert m == {"Vx": "V1", "Vy": "V2", "T": "TEMP", "EVF": "EVF"}
        # forces are always present as channels (not in the field map)
        assert tab.force_channels() == {"Fc": "RF1_RP", "Ff": "RF2_RP"}

    def test_thresholds_required_for_all_fields(self, qapp):
        tab = self._tab()
        assert tab.thresholds_complete() is False        # none set yet
        for q in ("Vx", "Vy", "T", "EVF", "Fc", "Ff"):
            tab._q_eps[q].setText("1")
        tab._q_eps["Vx"].setText("2.5")
        tab._q_eps["T"].setText("1,0")                   # comma accepted
        thr = tab.thresholds()
        assert thr["Vx"] == pytest.approx(2.5)
        assert thr["T"] == pytest.approx(1.0)
        assert tab.thresholds_complete() is True

    def test_per_variable_caps_are_minimums(self, qapp):
        tab = self._tab()
        # element caps are MINIMUMS (finest allowed), per variable
        assert tab.wp_min() is None and tab.tool_min() is None
        tab.le_wp_min.setText("0,0005")
        tab.le_tool_min.setText("0.002")
        assert tab.wp_min() == pytest.approx(0.0005)     # comma accepted
        assert tab.tool_min() == pytest.approx(0.002)

    def test_per_variable_factors(self, qapp):
        tab = self._tab()
        # each bracketing phase has its own factor + start
        assert tab._float_or(tab.le_wp_factor, 0.0) == pytest.approx(0.5)
        assert tab._float_or(tab.le_tool_factor, 0.0) == pytest.approx(0.5)
        assert tab._float_or(tab.le_domain_factor, 0.0) == pytest.approx(2.0)
        assert tab._float_or(tab.le_ms_factor, 0.0) == pytest.approx(2.0)
        assert tab._float_or(tab.le_tool_start, 0.0) == pytest.approx(0.005)

    def test_pipeline_controls_present(self, qapp):
        tab = self._tab()
        assert tab._float_or(tab.le_wp_start, 0.005) == pytest.approx(0.01)
        assert tab._float_or(tab.le_tool_start, 0.001) == pytest.approx(0.005)
        assert tab.cb_include_tool.isChecked() is True
        # comma decimal + fallback on empty
        tab.le_wp_start.setText("0,02")
        assert tab._float_or(tab.le_wp_start, 0.005) == pytest.approx(0.02)
        tab.le_wp_start.setText("")
        assert tab._float_or(tab.le_wp_start, 0.007) == pytest.approx(0.007)


# ---------------------------------------------------------------------------
# Fc (cutting force) as a normalized criterion quantity
# ---------------------------------------------------------------------------
class _FakeBundleH:
    def __init__(self, centroids_xy, fields, history):
        c = np.asarray(centroids_xy, dtype=float)
        self._c = np.column_stack([c, np.zeros(len(c))])
        self._fields = fields
        self._history = history

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        return self._fields[var]

    def history(self, channel):
        return self._history[channel]


class TestFcCriterion:
    def test_extract_fc_normalized(self):
        from gui.sensitivity.domain_opt import extract_force_normalized
        b = _FakeBundleH([[0, 0]], {}, {"RF1_RP": np.array([200.0, 200.0])})
        fc = extract_force_normalized(b, 0.01, "RF1_RP")
        assert fc.shape == (2, 1)
        assert fc[0, 0] == pytest.approx(20000.0)         # 200 / 0.01

    def test_extract_fc_absent_returns_none(self):
        from gui.sensitivity.domain_opt import extract_force_normalized
        b = _FakeBundleH([[0, 0]], {}, {})                # no RF1_RP
        assert extract_force_normalized(b, 0.01, "RF1_RP") is None

    def test_make_sample_fn_includes_fc(self):
        import gui.sensitivity.runner_core as rc
        from gui.core.model_config import ModelConfig
        from gui.sensitivity.domain_opt import make_sample_fn
        orig = rc.eulerian_instance
        rc.eulerian_instance = lambda b: "Euler"
        try:
            cents = np.array([[-0.005, -0.005], [-0.015, -0.005]])

            def run_bundle(cfg):
                return _FakeBundleH(cents, {"V1": np.full((2, 2), 1.0)},
                                    {"RF1_RP": np.array([200.0, 200.0])})
            cfg = ModelConfig(); cfg.elem_size = 0.01
            sf = make_sample_fn(cfg, run_bundle, roi=(-0.02, 0.0, -0.02, 0.0),
                                elem_size=0.01, quantity_field_map={"Vx": "V1"},
                                force_channels={"Fc": "RF1_RP"})
            out = sf(DomainDims(0.05, 0.05, 0.05, 0.05))
        finally:
            rc.eulerian_instance = orig
        assert "Fc" in out and out["Fc"][0, 0] == pytest.approx(20000.0)
