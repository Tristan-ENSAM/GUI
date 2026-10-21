# -*- coding: utf-8 -*-
"""Tests for the "mean" deviation metric (decision of 2026-10-21): one area
mean per run over the ZOI rectangle, |m(a) - m(b)| between two runs."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import gui.sensitivity.domain_independence as di
import gui.sensitivity.mesh_gci as mg
from gui.sensitivity.ms_independence import pair_errors
from gui.sensitivity.study_specs import comparison_settings, metric_of_spec
from gui.sensitivity.zoi_sampling import (
    area_mean, zoi_area_means, zoi_area_weights)

from tests.test_deviation_maps import _Bundle, _ZOI


class TestAreaWeights:
    def test_partial_elements_weighted_by_their_area_inside(self):
        # elements of 1 x 1 centred on 0.5, 1.5, 2.5 (x) and 0.5 (y);
        # ZOI x [0.25, 2.0]: overlaps 0.75, 1.0, 0 -> two elements.
        c = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                      [0.5, 1.5], [1.5, 1.5], [2.5, 1.5]])
        idx, w = zoi_area_weights(c, (0.25, 2.0, 0.0, 1.0))
        assert list(idx) == [0, 1]
        assert w == pytest.approx([0.75, 1.0])

    def test_weights_sum_to_the_zoi_area(self):
        b = _Bundle(h=0.005)
        idx, w = zoi_area_weights(b.c[:, :2], _ZOI)
        area = (_ZOI[1] - _ZOI[0]) * (_ZOI[3] - _ZOI[2])
        assert w.sum() == pytest.approx(area)


class TestAreaMean:
    def test_material_mask_per_element_and_frame(self):
        v = np.array([[1.0, 10.0], [3.0, 20.0]])
        mat = np.array([[True, False], [True, False]])
        # element 2 never material -> only element 1 (time mean 2.0)
        assert area_mean(v, [1.0, 3.0], mat) == pytest.approx(2.0)
        mat[1, 1] = True
        # element 2 mean 20 (one frame), weights 1 and 3
        assert area_mean(v, [1.0, 3.0], mat) == pytest.approx(
            (2.0 + 3 * 20.0) / 4)

    def test_nothing_valid_is_nan(self):
        assert np.isnan(area_mean(np.ones((2, 2)), [1, 1],
                                  np.zeros((2, 2), bool)))

    def test_linear_field_gives_its_value_at_the_zoi_centre(self):
        b = _Bundle(h=0.005, gx=1000.0)        # TEMP = 300 + 1000 x
        out = zoi_area_means(b, "EULER", _ZOI, ["TEMP"])
        assert out["TEMP"] == pytest.approx(300.0, abs=1e-9)

    def test_independent_of_the_grid_step(self):
        b = _Bundle(h=0.005, gx=1000.0, offset=2.0)
        kw = dict(window=(0.3, 1.0), quantities=("T", "EVF"))
        s1 = di.sample_zoi(b, _ZOI, 0.005, 0.005, **kw)
        s2 = di.sample_zoi(b, _ZOI, 0.0013, 0.005, **kw)
        assert di.run_scalar(s1, "T") == pytest.approx(
            di.run_scalar(s2, "T"))


class TestMeanErrors:
    def test_difference_of_the_means(self):
        a = di.sample_zoi(_Bundle(), _ZOI, 0.005, 0.005)
        b = di.sample_zoi(_Bundle(offset=3.0), _ZOI, 0.005, 0.005)
        e = di.mean_errors(a, b, ("T", "Vx", "EVF"))
        assert e["T"] == pytest.approx(3.0)
        assert e["Vx"] == pytest.approx(0.0)

    def test_out_of_phase_oscillation_cancels(self):
        # pointwise sees the phase shift, the means do not
        a = di.sample_zoi(_Bundle(amp=5.0), _ZOI, 0.005, 0.005)
        b = di.sample_zoi(_Bundle(amp=5.0, phase=np.pi), _ZOI, 0.005, 0.005)
        pw = di.errors_by(a, b, ("T",), "pointwise")["T"]
        mn = di.errors_by(a, b, ("T",), "mean")["T"]
        # (the window holds 3.5 periods: a residual of half a period stays)
        assert pw > 2.0 and mn < 0.2 * pw

    def test_forces_compare_their_window_means(self):
        a = di.sample_zoi(_Bundle(), _ZOI, 0.005, 0.005)
        b = di.sample_zoi(_Bundle(force_offset=0.01), _ZOI, 0.005, 0.005)
        assert di.mean_errors(a, b, ("Fc",))["Fc"] == pytest.approx(
            0.01 / 0.005)

    def test_triangle_inequality_is_exact_with_moving_material(self):
        rng = np.random.default_rng(0)
        runs = [di.sample_zoi(_Bundle(offset=o, amp=a, phase=p, void_x=v),
                              _ZOI, 0.005, 0.005)
                for o, a, p, v in zip(rng.normal(0, 5, 4),
                                      rng.uniform(0, 5, 4),
                                      rng.uniform(0, 6, 4),
                                      (None, 0.0, 0.01, -0.01))]
        for x, y, z in itertools.permutations(runs, 3):
            for q in ("T", "Vx", "EVF", "Fc"):
                d = lambda u, v: di.mean_errors(u, v, (q,))[q]
                assert d(x, z) <= d(x, y) + d(y, z) + 1e-12

    def test_pair_errors_mean_needs_no_common_times(self):
        a = di.sample_zoi(_Bundle(), _ZOI, 0.005, 0.005)
        b = di.sample_zoi(_Bundle(offset=1.0), _ZOI, 0.005, 0.005)
        b.field_times = b.field_times[:-2]
        b.fields = {q: v[:-2] for q, v in b.fields.items()}
        errs, info = pair_errors(a, b, ("T",), "mean")
        assert errs["T"] == pytest.approx(1.0) and info == {}
        with pytest.raises(di.AlignmentError):
            pair_errors(a, b, ("T",), "pointwise")

    def test_unknown_metric_refused(self):
        a = di.sample_zoi(_Bundle(), _ZOI, 0.005, 0.005)
        with pytest.raises(ValueError):
            di.errors_by(a, a, ("T",), "rms")


class TestGciScalar:
    def test_area_reduction_matches_the_studies(self):
        b = _Bundle(h=0.005, gx=1000.0, offset=2.0, amp=1.0)
        g = mg.zoi_scalars(b, _ZOI, 0.005, ["EVF", "TEMP"],
                           window=(0.3, 1.0), evf_threshold=0.5,
                           reduction="area")
        s = di.sample_zoi(b, _ZOI, 0.005, 0.005, quantities=("T", "EVF"))
        assert g["TEMP"] == pytest.approx(di.run_scalar(s, "T"))
        assert g["EVF"] == pytest.approx(di.run_scalar(s, "EVF"))


class TestKey:
    def test_metric_enters_the_key_only_when_mean(self):
        base = dict(zoi=_ZOI, eps={"T": 1.0}, window=(0.3, 1.0),
                    rk_max=0.05, rhg_max=0.05)
        assert "metric" not in comparison_settings(**base)
        assert comparison_settings(**base, metric="mean")["metric"] == "mean"

    def test_metric_of_spec(self):
        assert metric_of_spec({"metric": "mean"}) == "mean"
        assert metric_of_spec({"reduction": "area"}) == "mean"
        assert metric_of_spec({"selection": "pointwise"}) is None
