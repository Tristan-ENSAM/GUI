# -*- coding: utf-8 -*-
"""Chip thickness e(t) (gui.results.chip_thickness) and its QoI."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gui.core.model_config import ModelConfig
from gui.results import chip_thickness as ct
from gui.sensitivity import model_qoi as mq

H = 0.002


class FakeBundle:
    """Eulerian grid x in [-0.1, 0.02), y in [-0.04, 0.08), EVF given by
    evf_fn(x, y, frame)."""

    def __init__(self, evf_fn, n_frames=5, roi=None, ys=0.0, tool_y=-0.04):
        xs = np.arange(-0.1 + H / 2, 0.02, H)
        yv = np.arange(-0.04 + H / 2, 0.08, H)
        X, Y = np.meshgrid(xs, yv)
        self._c = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        self.times = np.linspace(0.0, 1.0, n_frames)
        self._evf = np.array([[evf_fn(x, y, k) for x, y in self._c[:, :2]]
                              for k in range(n_frames)])
        self.roi = roi
        self.model_config = {
            "mesh": {"elem_size": H},
            "geometry": {"euler": {"workpiece_position": {"x0": 0, "y0": ys}},
                         "tool": {"position": {"x0": 0, "y0": tool_y}}}}
        self.instances = ["EULER-1"]

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        assert var == "EVF"
        return self._evf


def chip(width):
    """Chip against the tool face x = 0, from x = -width, above y = 0."""
    def f(x, y, k):
        if y < 0:
            return 1.0 if x < 0 else 0.0
        return 1.0 if (-width <= x < 0) else 0.0
    return f


def test_surface_and_t1_both_layouts():
    b = FakeBundle(chip(0.01))
    assert ct.surface_and_t1(b.model_config) == (0.0, pytest.approx(0.04))
    prof = ModelConfig().to_json_dict()
    ys, t1 = ct.surface_and_t1(prof)
    assert ys == 0.0 and t1 == pytest.approx(0.05)
    assert ct.surface_and_t1({}) == (None, None)


def test_width_on_row_counts_partial_cells_and_picks_tool_side():
    row = np.array([0, 1, 1, 0, 0, 0.3, 1, 1, 1, 0.4, 0, 0])
    assert ct.width_on_row(row, 0.1) == pytest.approx(0.1 * 3.7)
    assert math.isnan(ct.width_on_row(np.zeros(5), 0.1))
    # Run touching an end of the row: truncated by the ROI.
    assert math.isnan(ct.width_on_row(np.array([0, 0, 1, 1]), 0.1))


def test_series_and_mean_on_fake_run():
    b = FakeBundle(chip(0.03))
    e = ct.thickness_series(b, "EULER-1", H)          # line at y0 = t1
    assert np.allclose(e, 0.03)
    tmask = np.array([False, True, True, True, True])
    assert ct.mean_thickness(b, "EULER-1", H, tmask) == pytest.approx(0.03)
    # Explicit height, still inside the chip.
    assert ct.mean_thickness(b, "EULER-1", H, tmask, y0=0.01) \
        == pytest.approx(0.03)


def test_line_outside_roi_or_domain_raises():
    b = FakeBundle(chip(0.03), roi={"xmin": -0.1, "xmax": 0.02,
                                    "ymin": -0.04, "ymax": 0.02})
    with pytest.raises(ValueError, match="ROI"):
        ct.thickness_series(b, "EULER-1", H)
    b = FakeBundle(chip(0.03))
    with pytest.raises(ValueError, match="domain"):
        ct.thickness_series(b, "EULER-1", H, y0=0.5)


def test_frames_without_chip_are_left_out_of_the_mean():
    def f(x, y, k):
        if y >= 0 and k < 2:
            return 0.0                     # no chip yet on the line
        return chip(0.02)(x, y, k)
    b = FakeBundle(f)
    e = ct.thickness_series(b, "EULER-1", H)
    assert np.isnan(e[:2]).all() and np.allclose(e[2:], 0.02)
    assert ct.mean_thickness(b, "EULER-1", H, np.ones(5, bool)) \
        == pytest.approx(0.02)
    assert math.isnan(ct.mean_thickness(b, "EULER-1", H,
                                        np.array([1, 1, 0, 0, 0], bool)))


def test_model_qoi_spec(monkeypatch):
    assert mq.MODEL_QOIS[-1] == "e" and mq.SIZING_QOIS == mq.MODEL_QOIS[:6]
    monkeypatch.setattr(mq, "eulerian_instance", lambda b: "EULER-1")
    b = FakeBundle(chip(0.024))
    (spec,) = mq.make_specs((-0.1, 0, -0.04, 0), (0.0, 1.0), qois=["e"],
                            chip_y0=0.01)
    assert spec.eps_key == "e" and spec.unit == "mm"
    assert spec.fn(b) == pytest.approx(0.024)


def test_setup_settings_carry_chip_eps_and_height(qapp):
    from gui.tabs.optimization_tab import OptimizationTab
    cfg = ModelConfig()
    tab = OptimizationTab(cfg)
    assert tab.le_chip_eps.text() == "0.005"
    st = tab.model_settings()
    assert st["eps"]["e"] == pytest.approx(0.005)
    assert st["chip_y0"] is None
    tab.le_chip_y0.setText("0.03")
    tab.le_chip_eps.setText("0.004")
    assert cfg.optimization.chip_y0 == "0.03"
    assert cfg.optimization.chip_eps == "0.004"
    assert tab.model_settings()["chip_y0"] == pytest.approx(0.03)
    # The sizing studies keep their six tolerances.
    assert "e" not in tab.thresholds()
    back = ModelConfig.from_json_dict(cfg.to_json_dict())
    assert back.optimization.chip_y0 == "0.03"
