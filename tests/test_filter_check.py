# -*- coding: utf-8 -*-
"""Output filters specified by acquisition rate + attenuation, the fc*dt
bounds, and the offline verification of Abaqus's runtime Butterworth."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gui.core.filter_check import (check_arrays, check_bundle,
                                   compare_filtered, format_report,
                                   offline_butterworth)
from gui.core.model_config import (ModelConfig, acquisition_from_cutoff,
                                   butterworth_cutoff_hz)


def _cfg(elem_size=0.005):
    c = ModelConfig()
    c.euler_material.update({"E": 113800.0, "nu": 0.342, "rho": 4.43e-9})
    c.elem_size = elem_size
    for k in ("l_wp", "l_void", "h_wp", "h_void"):
        setattr(c.euler_geometry, k, 0.1)
    return c


class TestCutoffFormula:
    def test_attenuation_at_nyquist(self):
        # |H(fN)|^2 = 1 / (1 + (fN/fc)^4) must equal 10^(-A/10).
        for acq, a in ((60e3, 3.0), (60e3, 20.0), (500e3, 40.0)):
            fc = butterworth_cutoff_hz(acq, a)
            h2 = 1.0 / (1.0 + (0.5 * acq / fc) ** 4)
            assert 10.0 * np.log10(h2) == pytest.approx(-a, rel=1e-9)

    def test_3db_puts_cutoff_on_nyquist(self):
        assert butterworth_cutoff_hz(60e3, 10 * np.log10(2)) == \
            pytest.approx(30e3, rel=1e-9)

    def test_more_attenuation_lowers_cutoff(self):
        assert butterworth_cutoff_hz(60e3, 20) < butterworth_cutoff_hz(60e3, 3)

    def test_inverse(self):
        fc = butterworth_cutoff_hz(123e3, 17.0)
        assert acquisition_from_cutoff(fc, 17.0) == pytest.approx(123e3)

    def test_bad_input(self):
        assert butterworth_cutoff_hz(0, 3) == 0.0
        assert butterworth_cutoff_hz(60e3, 0) == 0.0

    def test_defaults_are_consistent(self):
        s = ModelConfig().step
        fc, fh = s.output_filter_cutoff_hz, s.output_filter_cutoff_history_hz
        s.sync_filter_cutoffs()
        assert s.output_filter_cutoff_hz == pytest.approx(fc, rel=1e-5)
        assert s.output_filter_cutoff_history_hz == pytest.approx(fh, rel=1e-5)


class TestProfiles:
    def test_round_trip(self):
        c = ModelConfig()
        c.step.output_filter_camera_fps = 20000.0
        c.step.output_filter_camera_atten_db = 20.0
        c.step.sync_filter_cutoffs()
        back = ModelConfig.from_json_dict(c.to_json_dict())
        assert back.step.output_filter_camera_fps == 20000.0
        assert back.step.output_filter_cutoff_hz == pytest.approx(
            butterworth_cutoff_hz(20000.0, 20.0))

    def test_legacy_profile_keeps_its_cutoffs(self):
        d = ModelConfig().to_json_dict()
        for k in ("output_filter_camera_fps", "output_filter_camera_atten_db",
                  "output_filter_force_acq_hz", "output_filter_force_atten_db",
                  "output_filter_verify"):
            d["step"].pop(k)
        d["step"]["output_filter_cutoff_hz"] = 100000.0
        d["step"]["output_filter_cutoff_history_hz"] = 100000.0
        c = ModelConfig.from_json_dict(d)
        assert c.step.output_filter_cutoff_hz == pytest.approx(100000.0)
        assert c.step.output_filter_cutoff_history_hz == pytest.approx(1e5)
        assert c.step.output_filter_camera_fps == pytest.approx(
            acquisition_from_cutoff(100000.0, 3.0))

    def test_cutoffs_are_exported(self):
        c = ModelConfig()
        c.step.output_filter_force_acq_hz = 1e6
        c.step.sync_filter_cutoffs()
        step = c.to_params_dict()["step"]
        assert step["output_filter_cutoff_history_hz"] == pytest.approx(
            butterworth_cutoff_hz(1e6, 3.0))
        assert step["output_filter_verify"] is True


class TestRatioBounds:
    def test_ratio_scales_with_sqrt_ms(self):
        c = _cfg()
        r1 = c.filter_ratio(1e5, 1.0)
        assert c.filter_ratio(1e5, 100.0) == pytest.approx(10 * r1)
        assert r1 == pytest.approx(1e5 * c.initial_stable_dt())

    def test_lowest_cutoff_sets_lower_bound(self):
        c = _cfg()
        one = c.mass_scaling_bounds(30e3)
        both = c.mass_scaling_bounds(30e3, history_cutoff_hz=10e3)
        assert both["ms_min"] == pytest.approx(one["ms_min"] * 9.0)
        # at ms_min the lowest cutoff sits exactly on Abaqus's 1e-3
        assert c.filter_ratio(10e3, both["ms_min"]) == pytest.approx(1e-3)

    def test_highest_cutoff_sets_nyquist_bound(self):
        c = _cfg()
        b = c.mass_scaling_bounds(30e3, history_cutoff_hz=250e3)
        assert c.filter_ratio(250e3, b["ms_nyquist"]) == pytest.approx(0.5)


def _signal(n=4000, dt=1e-7, seed=0):
    """Irregular solver increments, a ramp plus a high-frequency ringing."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(dt * (1.0 + 0.02 * rng.standard_normal(n)))
    t -= t[0]
    x = 500.0 * np.minimum(t / t[-1] * 2, 1.0) \
        + 40.0 * np.sin(2 * np.pi * 400e3 * t)
    return t, x


class TestOfflineComparison:
    def test_identical_filter_passes(self):
        t, x = _signal()
        tu, y, _dt = offline_butterworth(t, x, 50e3)
        r = compare_filtered(t, x, tu, y, 50e3)
        assert r["rel_max_dev"] < 1e-12

    def test_attenuates_the_ringing(self):
        t, x = _signal()
        tu, y, _ = offline_butterworth(t, x, 20e3)
        tail = tu > 0.6 * tu[-1]
        assert np.ptp(y[tail]) < 0.05 * 80.0

    def test_above_nyquist_is_not_filtered(self):
        t, x = _signal()
        tu, y, dt = offline_butterworth(t, x, 0.6 / np.median(np.diff(t)))
        assert np.allclose(y, np.interp(tu, t, x), atol=1.0)

    def test_wrong_cutoff_fails(self):
        t, x = _signal()
        tu, y, _ = offline_butterworth(t, x, 20e3)
        arrays = {"filtercheck__RAW__time": t, "filtercheck__RAW__RF1": x,
                  "filtercheck__RAW__RF2": -x,
                  "filtercheck__SENSORBAND__time": tu,
                  "filtercheck__SENSORBAND__RF1": y,
                  "filtercheck__SENSORBAND__RF2": -y}
        ok = check_arrays(arrays, {"SENSORBAND": 20e3})
        bad = check_arrays(arrays, {"SENSORBAND": 80e3})
        assert ok["passed"] is True
        assert bad["passed"] is False
        assert "FAILED" in format_report(bad)

    def test_missing_raw_series(self):
        res = check_arrays({}, {"SENSORBAND": 1e5})
        assert res["passed"] is None and "raw" in res["error"]


def test_check_bundle_writes_meta(tmp_path):
    t, x = _signal()
    tu, y, _ = offline_butterworth(t, x, 30e3)
    npz = tmp_path / "job.results.npz"
    np.savez(npz, **{"filtercheck__RAW__time": t, "filtercheck__RAW__RF1": x,
                     "filtercheck__RAW__RF2": x,
                     "filtercheck__CAMERABAND__time": tu,
                     "filtercheck__CAMERABAND__RF1": y,
                     "filtercheck__CAMERABAND__RF2": y})
    meta = tmp_path / "job.meta.json"
    meta.write_text(json.dumps({"model_config": {"step": {
        "output_filter_enabled": True, "output_filter_verify": True,
        "output_filter_cutoff_hz": 30e3,
        "output_filter_cutoff_history_hz": 0.0}}}))
    res = check_bundle(npz)
    assert res["passed"] is True
    assert set(res["filters"]) == {"CAMERABAND"}
    assert json.loads(meta.read_text())["filter_check"]["passed"] is True


def test_check_bundle_skips_when_not_requested(tmp_path):
    npz = tmp_path / "job.results.npz"
    np.savez(npz, a=np.zeros(1))
    (tmp_path / "job.meta.json").write_text(json.dumps(
        {"model_config": {"step": {"output_filter_enabled": False}}}))
    assert check_bundle(npz) is None
    assert check_bundle(tmp_path / "job.inp") is None


def test_step_tab_filter_check(qapp):
    from gui.tabs.step_tab import StepTab
    c = _cfg(0.005)
    c.optimization.gci_finest = "0.0005"
    tab = StepTab(c)
    tab.cb_filter.setChecked(True)
    tab.f_cam_fps.set_value(20000.0)
    tab.f_cam_db.set_value(20.0)
    tab._on_change()
    assert c.step.output_filter_camera_fps == 20000.0
    assert c.step.output_filter_cutoff_hz == pytest.approx(
        butterworth_cutoff_hz(20000.0, 20.0))
    txt = tab.lbl_filter_check.text()
    assert "base mesh" in txt and "finest GCI mesh" in txt
    # no mass scaling, 0.5 um: far below Abaqus's 1e-3 -> flagged
    assert "warning" in txt
    assert "finest GCI mesh" in tab.lbl_ms_bounds.text()
    tab.cb_filter.setChecked(False)
    assert not tab.f_cam_fps.isEnabled()


# ---------------------------------------------------------------------------
# Reverberation in the force band + clean fallback
# ---------------------------------------------------------------------------
from gui.core.filter_check import reverberation_check, window_from_cfg
from gui.core.model_config import reverberation_frequency_hz


def _rev_cfg(ms=1000.0, f_force=250e3):
    return {"step": {"mass_scaling_enabled": True,
                     "mass_scaling_factor_eulerian": ms,
                     "output_filter_cutoff_history_hz": f_force},
            "geometry": {"euler": {"geometry": {
                "l_wp": 0.2, "l_void": 0.2, "h_wp": 0.2, "h_void": 0.2}}},
            "materials": {"euler": {"E": 113800.0, "nu": 0.342,
                                    "rho": 4.43e-9}}}


def _rev_arrays(f_rev, amp):
    dt = 2e-8
    t = np.arange(0.0, 6e-4, dt)
    x = 400.0 * (1.0 - np.exp(-t / 5e-5)) + 2.0 * np.sin(2 * np.pi * 8e3 * t)
    x = x + amp * np.sin(2 * np.pi * f_rev * t)
    return {"filtercheck__RAW__time": t, "filtercheck__RAW__RF1": x,
            "filtercheck__RAW__RF2": 0.5 * x}


class TestReverberation:
    def test_frequency_matches_mass_scaling_bounds(self):
        # f_rev(ms) = c_d/(2 L sqrt(ms)); at ms_freq it sits at k * fc.
        c = _cfg()
        b = c.mass_scaling_bounds(30e3)
        g = c.euler_geometry
        f = reverberation_frequency_hz(c.euler_material, g.l_wp + g.l_void,
                                       g.h_wp + g.h_void, b["ms_freq"])
        assert f == pytest.approx(3.0 * 30e3)

    def test_known_case(self):
        # 0.4 x 0.4 mm Ti6Al4V domain at ms = 1000 -> ~176 kHz
        f = reverberation_frequency_hz({"E": 113800.0, "nu": 0.342,
                                        "rho": 4.43e-9}, 0.4, 0.4, 1000.0)
        assert f == pytest.approx(176.5e3, rel=2e-3)

    def test_clean_signal_passes(self):
        f_rev, _ = 176.46e3, None
        res, clean = reverberation_check(_rev_arrays(f_rev, 0.0), _rev_cfg())
        assert res["passed"] is True
        assert res["clean_cutoff_hz"] == pytest.approx(res["f_rev_hz"] / 3)
        assert set(clean) == {"forceclean__time", "forceclean__RF1",
                              "forceclean__RF2"}

    def test_reverberation_is_detected_and_removed(self):
        res0, _ = reverberation_check(_rev_arrays(0.0, 0.0), _rev_cfg())
        f_rev = res0["f_rev_hz"]
        res, clean = reverberation_check(_rev_arrays(f_rev, 30.0), _rev_cfg())
        assert res["passed"] is False
        assert res["RF1"]["peak_over_f_rev"] == pytest.approx(1.0, abs=0.02)
        # the clean series no longer carries the reverberation
        t = clean["forceclean__time"]
        tail = t > 3e-4
        ref = 400.0 * (1.0 - np.exp(-t[tail] / 5e-5))
        assert np.max(np.abs(clean["forceclean__RF1"][tail] - ref)) < 3.0

    def test_missing_raw(self):
        res, clean = reverberation_check({}, _rev_cfg())
        assert res["passed"] is None and clean == {}

    def test_window_from_cfg(self):
        c = ModelConfig()
        c.optimization.window_start, c.optimization.window_end = "0.5", "0.9"
        assert window_from_cfg(c) == (0.5, 0.9)
        c.optimization.window_start = "x"
        assert window_from_cfg(c) == (0.3, 1.0)
        assert window_from_cfg(None) == (0.3, 1.0)


def test_check_bundle_appends_clean_forces(tmp_path):
    from gui.results.reader import ResultsBundle
    arr = _rev_arrays(0.0, 0.0)
    t = arr["filtercheck__RAW__time"]
    tu, y, _ = offline_butterworth(t, arr["filtercheck__RAW__RF1"], 250e3)
    arr.update({"filtercheck__SENSORBAND__time": tu,
                "filtercheck__SENSORBAND__RF1": y,
                "filtercheck__SENSORBAND__RF2": 0.5 * y,
                "times": np.zeros(1), "history__time": t[:10],
                "history__RF1_RP": np.zeros(10), "history__RF2_RP": np.zeros(10)})
    npz = tmp_path / "job.results.npz"
    np.savez_compressed(npz, **arr)
    mc = _rev_cfg()
    mc["step"].update({"output_filter_enabled": True,
                       "output_filter_verify": True,
                       "output_filter_cutoff_hz": 0.0})
    (tmp_path / "job.meta.json").write_text(json.dumps({
        "format_version": 1, "times": [0.0], "model_config": mc,
        "instances": {}, "history": {"n_samples": 10,
                                     "variables": ["RF1_RP", "RF2_RP"]}}))
    res = check_bundle(npz)
    assert res["reverberation"]["passed"] is True
    assert "REVERB" in format_report(res)
    meta = json.loads((tmp_path / "job.meta.json").read_text())
    assert meta["reverberation_check"]["passed"] is True
    assert meta["forceclean"]["cutoff_hz"] == pytest.approx(
        res["reverberation"]["clean_cutoff_hz"])
    with np.load(npz) as a:
        assert a["forceclean__RF1"].shape == a["forceclean__time"].shape
    check_bundle(npz)                       # idempotent, no duplicate member
    ResultsBundle.load(npz)                 # the bundle still loads
