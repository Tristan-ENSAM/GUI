# -*- coding: utf-8 -*-
"""
Tests for the full mesh+domain pipeline (gui.sensitivity.mesh_pipeline).

The Abaqus run is replaced by a synthetic `run_bundle` whose Eulerian field
value depends analytically on the workpiece element size, the tool element size
and the domain length l_wp, so each of the six steps has a known behaviour and
the sequence can be checked without Abaqus. The bundle exposes anchored
centroids so both the nearest-grid metric (mesh steps) and the exact centroid
metric (domain step) work on it.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

import gui.sensitivity.runner_core as rc
from gui.core.model_config import ModelConfig
from gui.core.domain_sizing import DomainDims
from gui.sensitivity.mesh_pipeline import run_mesh_domain_pipeline


class _FakeBundle:
    def __init__(self, centroids_xy, fields, history=None):
        c = np.asarray(centroids_xy, dtype=float)
        self._c = np.column_stack([c, np.zeros(len(c))])
        self._fields = fields
        self._history = history or {}

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        return self._fields[var]

    def history(self, channel):
        return self._history[channel]


def _make_run_bundle(A_wp=100.0, A_tool=1000.0, A_dom=10.0, L=0.05, base=0.0,
                     A_ms=100.0, C_ke=1e-4):
    """Field = base + A_wp*elem + A_tool*tool + A_dom*exp(-l_wp/L) + A_ms/f,
    where f is the (effective) mass-scaling factor. Velocity thus stabilizes as
    f grows. Energy history: ALLKE = C_ke*f, ALLIE = 1 -> guard = C_ke*f."""
    def run_bundle(cfg):
        elem = cfg.elem_size
        tool = cfg.tool_elem_size
        l_wp = cfg.euler_geometry.l_wp
        f = (cfg.step.mass_scaling_factor
             if cfg.step.mass_scaling_enabled else 1.0)
        val = (base + A_wp * elem + A_tool * tool + A_dom * math.exp(-l_wp / L)
               + A_ms / f)
        n = max(2, int(round(0.2 / elem)))
        idx = np.arange(-n // 2, n // 2)
        xs = (idx + 0.5) * elem
        ys = (idx + 0.5) * elem
        XX, YY = np.meshgrid(xs, ys)
        cents = np.column_stack([XX.ravel(), YY.ravel()])
        rf1 = np.full(3, 1000.0 * elem)
        return _FakeBundle(
            cents,
            {"V1": np.full((1, len(cents)), val),
             "V2": np.full((1, len(cents)), val)},
            history={"RF1_RP": rf1, "RF2_RP": rf1,
                     "ALLKE": np.array([C_ke * f, C_ke * f]),
                     "ALLIE": np.array([1.0, 1.0])})
    return run_bundle


ROI = (-0.06, 0.0, -0.05, 0.06)
QMAP = {"Vx": "V1"}
THR = {"Vx": 1.0}


@pytest.fixture(autouse=True)
def _patch_eulerian_instance():
    orig = rc.eulerian_instance
    rc.eulerian_instance = lambda b: "Euler"
    yield
    rc.eulerian_instance = orig


class TestPipeline:
    def _run(self, **kw):
        init = DomainDims(h_wp=0.06, h_void=0.06, l_wp=0.02, l_void=0.02)
        params = dict(
            base_cfg=ModelConfig(), run_bundle=_make_run_bundle(), roi=ROI,
            quantity_field_map=QMAP, thresholds=THR, wp_start=0.02,
            tool_start=0.004, initial_domain=init, factor=0.5, max_steps=8,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")})
        params.update(kw)
        return run_mesh_domain_pipeline(**params)

    def test_identifies_wp_and_tool(self):
        res = self._run()
        # wp: E(elem,elem/2)=100*elem/2<1 <=> elem<0.02; from 0.02 halving the
        # first stable coarser size is 0.01.
        assert res.wp_elem == pytest.approx(0.01)
        # tool: E=1000*tool/2<1 <=> tool<0.002; from 0.004 -> identified 0.001.
        assert res.tool_elem == pytest.approx(0.001)

    def test_domain_grows_lwp(self):
        res = self._run()
        assert res.domain_result is not None
        # l_wp must have grown beyond its initial 0.02 (field decays with l_wp)
        assert res.domain.l_wp > 0.02

    def test_verifications_present(self):
        res = self._run()
        for v in (res.wp_verify, res.tool_verify, res.domain_verify):
            assert v is not None and "stable" in v

    def test_skip_tool(self):
        res = self._run(include_tool=False)
        # tool stays at the config default (0.001) and no tool study runs
        assert res.tool_elem == pytest.approx(ModelConfig().tool_elem_size)
        assert res.tool_conv is None

    def test_progress_and_run_count(self):
        events = []
        res = self._run(progress_cb=events.append)
        phases = {e["phase"] for e in events}
        assert {"wp_done", "tool_done", "domain_done", "done"} <= phases
        assert res.n_runs > 0

    def test_cancel_stops_early(self):
        res = self._run(should_cancel=lambda: True)
        # cancelled before any step completes a stage beyond wp; domain skipped
        assert res.domain_result is None

    def test_with_mass_scaling(self):
        # Step 0 now EXTENDS the factor upward to the largest admissible value.
        # error = A_ms/f self-convergence: 50/f <= 1 <=> f >= 50 -> onset 64;
        # guard = 1e-4*f, threshold 0.05 -> guard OK for f <= 500. The extension
        # runs from 64 up to the guard crossing (512) and bisects -> ~500.
        res = self._run(quantity_field_map={"Vx": "V1", "Vy": "V2"},
                        thresholds={"Vx": 1.0, "Vy": 1.0},
                        identify_ms=True, ms_guard_threshold=0.05,
                        ms_bisection_resolution=0.5)
        assert res.ms_factor is not None
        assert res.ms_factor == pytest.approx(500.0, abs=2.0)
        assert res.ms_result.velocity_converged is True
        assert res.ms_result.guard_ok is True
        assert res.ms_result.limited_by == "guard"
        assert res.domain_result is not None

    def test_with_fc_included(self):
        # include_fc=True adds Fc (=RF1/elem) to every step's compared
        # quantities; with RF1 ∝ elem, Fc/elem is constant (E_Fc≈0) so it does
        # not block convergence, and sizes are still identified.
        res = self._run(force_channels={"Fc": "RF1_RP"},
                        thresholds={"Vx": 1.0, "Fc": 1.0})
        assert res.wp_elem == pytest.approx(0.01)
        assert res.domain_result is not None


class TestWorker:
    def test_worker_runs(self, qapp):
        from gui.sensitivity.mesh_pipeline_worker import MeshPipelineWorker
        init = DomainDims(h_wp=0.06, h_void=0.06, l_wp=0.02, l_void=0.02)
        w = MeshPipelineWorker(
            base_cfg=ModelConfig(), run_bundle=_make_run_bundle(), roi=ROI,
            quantity_field_map=QMAP, thresholds=THR, wp_start=0.02,
            tool_start=0.004, initial_domain=init,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")})
        got = {}
        w.finished_ok.connect(lambda r: got.update(res=r))
        w.run()
        assert "res" in got
        assert got["res"].wp_elem == pytest.approx(0.01)


class TestOptionalSteps:
    def _run(self, **kw):
        init = DomainDims(h_wp=0.06, h_void=0.06, l_wp=0.02, l_void=0.02)
        params = dict(
            base_cfg=ModelConfig(), run_bundle=_make_run_bundle(), roi=ROI,
            quantity_field_map=QMAP, thresholds=THR, wp_start=0.02,
            tool_start=0.004, initial_domain=init, factor=0.5, max_steps=8,
            caps={d: 1.0 for d in ("h_wp", "h_void", "l_wp", "l_void")})
        params.update(kw)
        return run_mesh_domain_pipeline(**params)

    def test_skip_wp_uses_base_elem_size(self):
        # include_wp=False -> no wp identification; wp = base_cfg.elem_size
        base = ModelConfig()
        res = self._run(base_cfg=base, include_wp=False, do_verify=False)
        assert res.wp_conv is None
        assert res.wp_elem == pytest.approx(float(base.elem_size))

    def test_skip_domain_keeps_initial(self):
        # include_domain=False -> domain stays the initial domain
        init = DomainDims(h_wp=0.06, h_void=0.06, l_wp=0.02, l_void=0.02)
        res = self._run(initial_domain=init, include_domain=False,
                        do_verify=False)
        assert res.domain_result is None
        assert res.domain.l_wp == pytest.approx(init.l_wp)
        assert res.domain.h_void == pytest.approx(init.h_void)
