# -*- coding: utf-8 -*-
"""
Unit tests for gui.sensitivity.mesh_opt (element-size / mesh convergence).
Offline: the simulation is an analytic sample_fn whose ROI field depends on the
element size, so the converged size is known in closed form.
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.sensitivity.mesh_opt import (
    roi_grid, nearest_samples, refine_until_stable, verify_stability)


class _FakeBundle:
    def __init__(self, centroids_xy, fields):
        c = np.asarray(centroids_xy, dtype=float)
        self._c = np.column_stack([c, np.zeros(len(c))])
        self._fields = fields

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        return self._fields[var]


class TestRoiGrid:
    def test_counts_and_positions(self):
        g = roi_grid((0.0, 0.02, 0.0, 0.01), 0.01)
        assert g.shape == (6, 2)                        # nx=3, ny=2
        assert [tuple(np.round(p, 6)) for p in g[:3]] == [
            (0.0, 0.0), (0.01, 0.0), (0.02, 0.0)]

    def test_step_must_be_positive(self):
        with pytest.raises(ValueError):
            roi_grid((0, 1, 0, 1), 0.0)


class TestNearest:
    def test_nearest_selection(self):
        fb = _FakeBundle(np.array([[0.0, 0.0], [0.02, 0.0]]),
                         {"Vx": np.array([[10.0, 20.0]])})
        pts = np.array([[0.0, 0.0], [0.011, 0.0], [0.02, 0.0]])
        vals = nearest_samples(fb, "Vx", "E", pts)
        assert vals.tolist() == [[10.0, 20.0, 20.0]]

    def test_shape_multiframe(self):
        fb = _FakeBundle(np.array([[0.0, 0.0], [1.0, 0.0]]),
                         {"Vx": np.array([[1.0, 2.0], [3.0, 4.0]])})
        vals = nearest_samples(fb, "Vx", "E", np.array([[0.0, 0.0]]))
        assert vals.shape == (2, 1)
        assert vals[:, 0].tolist() == [1.0, 3.0]


def _decay_size_fn(A, quantities=("Vx",), n_t=2, n_p=3):
    """ROI field constant over (Nt, Np) = A*size. E(s, s/2) = A*s/2."""
    def sample_fn(size):
        arr = np.full((n_t, n_p), A * size, dtype=float)
        return {q: arr.copy() for q in quantities}
    return sample_fn


class TestRefine:
    def test_identifies_coarsest_stable(self):
        # no bisection resolution -> halving only; A=100, eps=1 -> stable when
        # A*s/2 < 1 <=> s < 0.02, coarsest power-of-two fraction 0.015625.
        r = refine_until_stable(_decay_size_fn(100.0), {"Vx": 1.0},
                                start_size=1.0, quantities=("Vx",),
                                max_steps=12)
        assert r.identified == pytest.approx(0.015625)
        assert r.converged is True

    def test_bisection_refines_between_halving_steps(self):
        # with a fine resolution the bisection returns the coarsest Cauchy-stable
        # size ~0.02 (coarser than the halving-only 0.015625)
        r = refine_until_stable(_decay_size_fn(100.0), {"Vx": 1.0},
                                start_size=1.0, quantities=("Vx",),
                                max_steps=12, bisection_resolution=1e-4)
        assert r.converged is True
        assert r.identified == pytest.approx(0.02, abs=1e-3)

    def test_min_size_stops_search(self):
        r = refine_until_stable(_decay_size_fn(1e6), {"Vx": 1e-9},
                                start_size=1.0, quantities=("Vx",),
                                min_size=0.1, max_steps=20)
        # cannot converge; stops at min_size floor, not converged
        assert r.converged is False
        assert r.identified >= 0.1 - 1e-9

    def test_factor_validation(self):
        with pytest.raises(ValueError):
            refine_until_stable(_decay_size_fn(1.0), {"Vx": 1.0},
                                start_size=1.0, factor=1.5)

    def test_progress_events(self):
        events = []
        refine_until_stable(_decay_size_fn(100.0), {"Vx": 1.0},
                            start_size=1.0, quantities=("Vx",), max_steps=12,
                            progress_cb=events.append)
        assert events and events[0]["phase"] == "refine"
        assert "size" in events[0] and "errors" in events[0]

    def test_initial_and_intermediate_no_bisection(self):
        # Halving only: intermediate == identified (end of halving), and
        # initial == start_size.
        r = refine_until_stable(_decay_size_fn(100.0), {"Vx": 1.0},
                                start_size=1.0, quantities=("Vx",),
                                max_steps=12)
        assert r.initial == pytest.approx(1.0)
        assert r.intermediate == pytest.approx(0.015625)
        assert r.identified == pytest.approx(r.intermediate)

    def test_intermediate_is_end_of_halving(self):
        # With bisection the final (identified) is coarser than the
        # end-of-halving intermediate; both are exposed distinctly.
        r = refine_until_stable(_decay_size_fn(100.0), {"Vx": 1.0},
                                start_size=1.0, quantities=("Vx",),
                                max_steps=12, bisection_resolution=1e-4)
        assert r.intermediate == pytest.approx(0.015625)
        assert r.identified == pytest.approx(0.02, abs=1e-3)
        assert r.identified > r.intermediate


class TestVerify:
    def test_finer_side_stable(self):
        # at the identified 0.015625: finer E = 100*0.0078125 = 0.78 < 1
        v = verify_stability(_decay_size_fn(100.0), 0.015625, {"Vx": 1.0},
                             quantities=("Vx",))
        assert v["stable"] is True
        assert v["finer"]["Vx"] == pytest.approx(0.78125, rel=1e-3)
        # coarser side leaves the plateau (informational, larger)
        assert v["coarser"]["Vx"] > v["finer"]["Vx"]


class TestToolElemSizeConfig:
    def test_default_and_serialisation(self):
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        assert c.tool_elem_size == pytest.approx(0.001)      # backward-compat
        assert c.to_params_dict()["mesh"]["tool_elem_size"] == pytest.approx(0.001)
        c.tool_elem_size = 0.0005
        assert c.to_params_dict()["mesh"]["tool_elem_size"] == pytest.approx(0.0005)


class TestMakeMeshSampleFn:
    def test_varies_size_and_samples_on_fixed_grid(self):
        from gui.sensitivity.mesh_opt import make_mesh_sample_fn
        from gui.core.model_config import ModelConfig

        seen = {}

        def run_bundle(cfg):
            # encode the varied + held sizes into the field so we can check them
            seen["elem"] = cfg.elem_size
            seen["tool"] = cfg.tool_elem_size
            cents = np.array([[0.0, 0.0], [0.02, 0.0]])
            val = cfg.elem_size
            return _FakeBundle(cents, {"V1": np.full((1, 2), val)})

        roi = (0.0, 0.02, 0.0, 0.0)
        # patch eulerian_instance to accept our fake bundle
        import gui.sensitivity.runner_core as rc
        orig = rc.eulerian_instance
        rc.eulerian_instance = lambda b: "Euler"
        try:
            sf = make_mesh_sample_fn(
                ModelConfig(), run_bundle, roi, grid_step=0.01,
                quantity_field_map={"Vx": "V1"}, size_attr="elem_size",
                held_sizes={"tool_elem_size": 0.0007})
            out = sf(0.01)
        finally:
            rc.eulerian_instance = orig
        assert seen["elem"] == pytest.approx(0.01)       # varied
        assert seen["tool"] == pytest.approx(0.0007)     # held
        # grid over x in [0,0.02] step 0.01 -> 3 points; all sampled to val
        assert out["Vx"].shape == (1, 3)
        assert np.allclose(out["Vx"], 0.01)


class TestVerifyDomain:
    def test_grow_side_governs_stability(self):
        from gui.sensitivity.domain_opt import verify_domain
        from gui.core.domain_sizing import DomainDims
        import math

        # field depends only on l_wp: A*exp(-l_wp/L), constant over ROI
        A, L = 10.0, 0.05

        def sample_fn(d):
            return {"Vx": np.full((1, 2), A * math.exp(-d.l_wp / 0.05))}

        dims = DomainDims(h_wp=0.1, h_void=0.1, l_wp=0.2, l_void=0.1)
        # +1 elem on l_wp: E = A*(exp(-0.2/L) - exp(-0.205/L)) tiny -> below 0.5
        res = verify_domain(sample_fn, dims, {"Vx": 0.5}, elem_size=0.005,
                            quantities=("Vx",))
        assert "stable" in res
        assert res["l_wp"]["plus"]["Vx"] < 0.5
        assert res["stable"] is True
