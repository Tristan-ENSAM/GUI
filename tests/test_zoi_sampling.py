# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.zoi_sampling (helpers relocated by lot L6 from
the removed mesh_opt / domain_opt / domain_convergence modules; the tests
below are the ones of those modules that covered the kept helpers)."""
from __future__ import annotations

import numpy as np
import pytest

from gui.sensitivity.zoi_sampling import (
    element_centroids_xy, history_window_mean, nearest_samples, roi_grid,
    window_mask)


class _FakeBundle:
    def __init__(self, centroids_xy, fields, history=None, history_time=None):
        c = np.asarray(centroids_xy, dtype=float)
        self._c = np.column_stack([c, np.zeros(len(c))])
        self._fields = fields
        self._h = history or {}
        self._ht = history_time

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        return self._fields[var]

    def history(self, var):
        return self._h[var]

    @property
    def history_time(self):
        return self._ht


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

    def test_centroids_xy_drop_z(self):
        fb = _FakeBundle(np.array([[1.0, 2.0]]), {})
        assert element_centroids_xy(fb, "E").tolist() == [[1.0, 2.0]]


class TestWindowMask:
    def test_default_drops_first_30pct(self):
        t = np.linspace(0.0, 1.0, 11)
        m = window_mask(t, 0.3, 1.0)
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


class TestHistoryWindowMean:
    def test_windowed_mean(self):
        t = np.linspace(0.0, 1.0, 11)
        y = np.where(t < 0.3, 100.0, 2.0)
        fb = _FakeBundle([[0, 0]], {}, {"RF1_RP": y}, t)
        assert history_window_mean(fb, "RF1_RP", 0.3, 1.0) == pytest.approx(2)

    def test_unavailable(self):
        fb = _FakeBundle([[0, 0]], {}, {"RF1_RP": np.ones(3)},
                         np.linspace(0, 1, 4))
        assert history_window_mean(fb, "RF1_RP", 0.3, 1.0) is None
        assert history_window_mean(fb, "absent", 0.3, 1.0) is None


class TestToolElemSizeConfig:
    """Moved from the removed test_mesh_opt.py (model_config coverage)."""

    def test_default_and_serialisation(self):
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        assert c.tool_elem_size == pytest.approx(0.001)
        assert c.to_params_dict()["mesh"]["tool_elem_size"] == \
            pytest.approx(0.001)
        c.tool_elem_size = 0.0005
        assert c.to_params_dict()["mesh"]["tool_elem_size"] == \
            pytest.approx(0.0005)
