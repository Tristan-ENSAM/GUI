# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.ms_independence (paper step 0, mass scaling).

The run_bundle is analytic: every ZOI quantity drifts linearly with ms, so
the successive error between ms_(k-1) and ms_k is proportional to their
difference. ZOI sampling is bypassed as in test_domain_independence.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

import gui.sensitivity.domain_independence as di
from gui.core.domain_sizing import DomainDims
from gui.sensitivity.ms_independence import (
    DEFAULT_MS_VALUES, filter_guards, parse_ms_values, run_ms_independence)
from gui.sensitivity.study_export import write_ms_exports

_NP, _NT, _NH = 6, 11, 21
_ZOI = (-0.05, 0.05, -0.05, 0.05)
_DIMS = DomainDims(h_wp=0.2, h_void=0.2, l_wp=0.2, l_void=0.2)
_THR = {"Vx": 10.0, "Vy": 10.0, "T": 10.0, "EVF": 0.1, "Fc": 10.0,
        "Ff": 10.0}


class _Bundle:
    """Every quantity = base + slope * ms (T drives E_max)."""

    def __init__(self, ms, slope):
        self.ms, self.slope = ms, slope
        self.times = np.linspace(0.0, 1e-4, _NT)
        self.history_time = np.linspace(0.0, 1e-4, _NH)
        self.instance_names = ["EULER"]

    def instance(self, name):
        class _I:
            field_variables = ["EVF", "TEMP", "V1", "V2"]
        return _I()

    def field(self, inst, var):
        if var == "EVF":
            return np.ones((_NT, _NP))
        if var == "TEMP":
            return np.full((_NT, _NP), 300.0 + self.slope * self.ms)
        return np.full((_NT, _NP), 100.0)

    def history(self, var):
        return np.full(_NH, {"RF1_RP": 0.8, "RF2_RP": 0.3}[var])


class _Step:
    mass_scaling_enabled = False
    mass_scaling_factor = 1.0


class _G:
    h_wp = h_void = l_wp = l_void = 0.0


class _Cfg:
    def __init__(self):
        self.step = _Step()
        self.euler_geometry = _G()
        self.elem_size = 0.005


@pytest.fixture(autouse=True)
def _patch_sampling(monkeypatch):
    monkeypatch.setattr(di, "nearest_samples",
                        lambda b, var, inst, pts: b.field(inst, var))
    monkeypatch.setattr(di, "roi_grid", lambda roi, step: np.zeros((_NP, 2)))


def _runner(slope, log=None, fail_ms=None):
    def run(cfg):
        ms = cfg.step.mass_scaling_factor
        if log is not None:
            log.append(cfg)
        if fail_ms is not None and ms == fail_ms:
            return None
        return _Bundle(ms, slope)
    return run


def _study(slope, **kw):
    kw.setdefault("thresholds", _THR)
    return run_ms_independence(
        kw.pop("run_bundle", None) or _runner(slope), _Cfg(), _ZOI, _DIMS,
        grid_step=0.005, elem_size=0.004, **kw)


class TestParse:
    def test_separators(self):
        assert parse_ms_values("250, 500 1000;2000") == (250., 500., 1000.,
                                                         2000.)

    @pytest.mark.parametrize("text", ["1000", "500, 250", "0.5, 2", "a, b",
                                      "250, 250"])
    def test_rejects(self, text):
        with pytest.raises(ValueError):
            parse_ms_values(text)


class TestFilterGuards:
    def test_not_run_counts_as_failure(self):
        assert filter_guards(None) == {"filter": (None, False),
                                       "reverb": (None, False)}

    def test_values_and_verdicts(self):
        res = {"passed": True,
               "filters": {"SENSORBAND": {"rel_max_dev": 0.002},
                           "CAMERABAND": {"rel_max_dev": 0.004}},
               "reverberation": {"passed": False, "e_rev": 0.03}}
        g = filter_guards(res)
        assert g["filter"] == (pytest.approx(0.004), True)
        assert g["reverb"] == (pytest.approx(0.03), False)

    def test_unevaluated_reverberation_fails(self):
        g = filter_guards({"passed": True, "filters": {},
                           "reverberation": {"passed": None,
                                             "error": "raw missing"}})
        assert g["reverb"] == (None, False)


class TestStudy:
    def test_keeps_the_last_ms_before_the_first_failure(self):
        # E_T = slope * (ms_k - ms_(k-1)) = 0.012 * 250, 500, 1000 ...
        # -> 3 K, 6 K pass, 12 K fails: ms* = 1000, 4000 never run.
        res = _study(0.012)
        assert res.status == "converged"
        assert res.retained == 1000.0
        assert res.run_ms == [250.0, 500.0, 1000.0, 2000.0]
        assert [c.success for c in res.comparisons] == [True, True, False]
        assert res.comparisons[-1].q_crit == "T"
        assert res.comparisons[-1].e_max == pytest.approx(1.2)

    def test_every_comparison_passing_gives_the_upper_end(self):
        res = _study(0.001)
        assert res.status == "upper_end"
        assert res.retained == DEFAULT_MS_VALUES[-1]
        assert res.n_runs == len(DEFAULT_MS_VALUES)

    def test_first_comparison_failing_gives_no_factor(self):
        res = _study(1.0)
        assert res.status == "below_range"
        assert res.retained is None
        assert res.n_runs == 2

    def test_a_failed_safeguard_breaks_the_chain(self):
        def guard(bundle):
            return {"reverb": (0.02, bundle.ms < 1000.0)}
        res = _study(0.001, guard_fn=guard)
        assert res.retained == 500.0 and res.status == "converged"
        assert res.comparisons[-1].guards_ok is False
        assert res.comparisons[-1].e_max < 1.0

    def test_a_failed_run_breaks_the_chain(self):
        res = _study(0.001, run_bundle=_runner(0.001, fail_ms=2000.0))
        assert res.retained == 1000.0
        assert res.runs[-1].job_ok is False

    def test_configs_carry_ms_and_mesh_and_base_is_untouched(self):
        log = []
        base = _Cfg()
        run_ms_independence(_runner(0.001, log), base, _ZOI, _DIMS,
                            grid_step=0.005, elem_size=0.004,
                            thresholds=_THR, ms_values=(100, 200))
        assert [c.step.mass_scaling_factor for c in log] == [100.0, 200.0]
        assert all(c.step.mass_scaling_enabled for c in log)
        assert all(c.elem_size == 0.004 for c in log)
        assert all(c.euler_geometry.l_wp == 0.2 for c in log)
        assert base.step.mass_scaling_enabled is False
        assert base.elem_size == 0.005

    def test_cancel_between_runs(self):
        calls = []

        def cancel():
            calls.append(1)
            return len(calls) > 2
        res = _study(0.001, should_cancel=cancel)
        assert res.status == "cancelled"
        assert res.n_runs == 2

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            _study(0.001, ms_values=(1000, 500))
        with pytest.raises(ValueError):
            _study(0.001, thresholds={})


def test_exports(tmp_path):
    res = _study(0.012, guard_fn=lambda b: {"filter": (0.001, True),
                                            "reverb": (0.002, True)})
    paths = write_ms_exports(tmp_path, res)
    assert [p.name for p in paths] == ["ms_runs.csv", "ms_comparisons.csv",
                                       "ms_summary.json"]
    runs = list(csv.DictReader(open(tmp_path / "ms_runs.csv")))
    assert [r["mass_scaling_factor"] for r in runs] == ["250.0", "500.0",
                                                        "1000.0", "2000.0"]
    assert [r["retained"] for r in runs] == ["False", "False", "True",
                                             "False"]
    assert runs[0]["reverb"] == "0.002" and runs[0]["filter_ok"] == "True"
    comps = list(csv.DictReader(open(tmp_path / "ms_comparisons.csv")))
    assert [c["decision"] for c in comps] == ["independent", "independent",
                                              "not independent"]
    assert float(comps[-1]["E_T_over_eps"]) == pytest.approx(1.2)
    s = json.loads((tmp_path / "ms_summary.json").read_text())
    assert s["retained_ms"] == 1000.0 and s["status"] == "converged"


class _ShiftedBundle(_Bundle):
    """Frames shifted by an ms-dependent offset (one increment ~ sqrt(ms))
    and a history sampled at every increment: different count per ms."""

    def __init__(self, ms, slope, shift=1e-9):
        super().__init__(ms, slope)
        self.times = np.linspace(0.0, 1e-4, _NT) + shift * np.sqrt(ms)
        self.history_time = np.linspace(0.0, 1e-4, int(400 / np.sqrt(ms)) + 3)

    def history(self, var):
        base = {"RF1_RP": 0.8, "RF2_RP": 0.3}[var]
        return base + 1e-3 * self.history_time / 1e-4


class TestTimeAlignment:
    def _run(self, shift):
        def run(cfg):
            return _ShiftedBundle(cfg.step.mass_scaling_factor, 0.012, shift)
        return _study(0.012, run_bundle=run)

    def test_runs_of_different_ms_are_compared(self):
        res = self._run(1e-9)
        assert res.status == "converged" and res.retained == 1000.0
        assert not res.warnings
        c = res.comparisons[0]
        # identical linear forces: interpolation leaves no error
        assert c.errors["Fc"] == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < c.frame_offset_over_interval < 0.5

    def test_frames_offset_beyond_half_an_interval_are_refused(self):
        res = self._run(1e-6)
        assert res.status == "below_range"
        assert "half the frame interval" in res.warnings[0]
