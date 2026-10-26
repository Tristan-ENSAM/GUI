# -*- coding: utf-8 -*-
"""Preliminary checks on one run (gui.sensitivity.preliminary_checks)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from gui.sensitivity import preliminary_checks as pc

H = 0.01                         # element size [mm]
NX, NY = 10, 8                   # x in [-0.1, 0], y in [-0.05, 0.03]


class _Bundle:
    """NX x NY elements; material below y = 0; fields change with time."""

    def __init__(self, n_frames=101, t_end=1e-3, chip_drop_frame=None,
                 filter_on=True, fc=50.1e3):
        xs = -0.1 + H * (np.arange(NX) + 0.5)
        ys = -0.05 + H * (np.arange(NY) + 0.5)
        X, Y = np.meshgrid(xs, ys)
        self._c = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        self.times = np.linspace(0.0, t_end, n_frames)
        n = self._c.shape[0]
        evf = np.repeat((self._c[:, 1] < 0).astype(float)[None, :],
                        n_frames, axis=0)
        if chip_drop_frame is not None:
            # chip element landing 2 cells above the surface at x = -0.085
            k = np.argmin(np.hypot(self._c[:, 0] + 0.085,
                                   self._c[:, 1] - 0.025))
            evf[chip_drop_frame:, k] = 1.0
        tau = self.times[:, None] / (0.05 * t_end)
        # Vx settles exponentially to 1000 mm/s (time constant 0.05 t_end).
        self._f = {"EVF": evf,
                   "V1": 1000.0 * (1 - np.exp(-tau)) + 0 * evf,
                   "V2": np.full((n_frames, n), -200.0),
                   "TEMP": 20.0 + 0 * evf}
        self.history_time = np.linspace(0.0, t_end, 1001)
        self._h = {"RF1_RP": np.full(1001, 0.5),        # 50 N/mm at w = H
                   "RF2_RP": 0.2 + 0.0 * self.history_time}
        self.model_config = {"mesh": {"elem_size": H},
                             "step": {"output_filter_enabled": filter_on,
                                      "output_filter_cutoff_hz": fc}}
        self.instance_names = ["EULER"]

    def instance(self, name):
        return SimpleNamespace(field_variables=list(self._f))

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        return self._f[var]

    def history(self, var):
        return self._h[var]


ZONE = (-0.1, 0.0, -0.05, 0.0)
EPS = {"Vx": 10.0, "Vy": 10.0, "T": 10.0, "EVF": 0.1, "Fc": 10.0,
       "Ff": 10.0}


def test_moving_mean_and_settling_index():
    t = np.linspace(0, 10, 11)
    m = pc.moving_mean(t, t.copy(), 2.0)
    assert np.isnan(m[:2]).all()
    assert m[2] == pytest.approx(1.0) and m[10] == pytest.approx(9.0)
    y = np.array([5.0, 3.0, 1.2, 0.9, 1.1, 1.0])
    assert pc.settling_index(y, 0.25) == 3     # [0.9, 1.1, 1.0]
    assert pc.settling_index(y, 0.01) == 5
    assert pc.settling_index(np.array([np.nan, np.nan]), 1.0) is None


def test_stationarity_finds_vx_settling_time():
    b = _Bundle()
    L = 0.05e-3
    res = pc.stationarity(b, ZONE, L, 0.1, EPS)
    vx = res.per_quantity["Vx"]
    # The moving mean stays within 1 mm/s from t_s on, and not from the
    # frame before.
    m = vx.mean
    i = np.searchsorted(b.times, vx.t_s)
    fin = m[i:][np.isfinite(m[i:])]
    assert fin.max() - fin.min() <= 1.0 + 1e-9
    assert np.nanmax(m[i - 1:]) - np.nanmin(m[i - 1:]) > 1.0
    assert res.per_quantity["Fc"].t_s == pytest.approx(L, abs=2e-6)
    assert res.t_s == pytest.approx(vx.t_s)
    lo, hi = res.window
    assert hi == 1.0 and lo == pytest.approx(vx.t_s / 1e-3)
    assert not res.too_short


def test_frame_attenuation_and_interval():
    assert pc.butterworth_attenuation_db(50.1e3 * 10, 50.1e3) == \
        pytest.approx(40.0, abs=0.01)
    assert pc.frame_interval_for(50.1e3, 40.0) == pytest.approx(
        1.0 / (2 * 10 * 50.1e3), rel=1e-3)
    b = _Bundle(n_frames=501, t_end=600e-6)
    fc = pc.frame_check(b, ZONE, (0.3, 1.0), EPS)
    assert fc.dt == pytest.approx(1.2e-6)
    assert fc.atten_db == pytest.approx(36.8, abs=0.05)
    assert set(fc.subsampling) == {2, 4, 8}
    assert fc.subsampling[2]["Vy"] == pytest.approx(0.0)
    off = pc.frame_check(_Bundle(filter_on=False), ZONE, (0.3, 1.0), EPS)
    assert off.cutoff is None and off.atten_db is None


def test_chip_recontact():
    b = _Bundle(chip_drop_frame=40)
    # band: the 2nd and 3rd cells above the surface cell (y = 0.015, 0.025)
    r = pc.chip_recontact(b, x_tip=0.0, margin=0.03, gap=1.5, band=2.0)
    assert r.t_first == pytest.approx(b.times[40])
    assert r.n_columns == 7                 # x centres < -0.03
    assert np.all(r.band_max_evf[:40] == 0)
    clean = pc.chip_recontact(_Bundle(), x_tip=0.0, margin=0.03, band=2.0)
    assert clean.t_first is None and clean.note == ""


def test_chip_recontact_band_outside_extraction():
    r = pc.chip_recontact(_Bundle(), x_tip=0.0, margin=0.03, band=3.0)
    assert r.n_columns == 0 and r.t_first is None and r.note


def test_dialog_analyses_a_run(qapp, monkeypatch, tmp_path):
    from gui.results import reader
    from gui.widgets.preliminary_checks_dialog import PreliminaryChecksDialog
    b = _Bundle(chip_drop_frame=40)
    b.model_config["geometry"] = {
        "tool": {"position": {"x0": 0.0, "y0": -0.015}},
        "euler": {"workpiece_position": {"x0": 0.0, "y0": 0.0}}}
    b.close = lambda: None
    monkeypatch.setattr(reader.ResultsBundle, "load",
                        classmethod(lambda cls, p: b))
    run = tmp_path / "j.results.npz"
    run.write_bytes(b"")
    dlg = PreliminaryChecksDialog(
        settings_getter=lambda: {"eps": EPS, "zoi": ZONE})
    assert dlg.le_zone["xmax"].text() == "0"
    dlg.le_run.setText(str(run))
    dlg.le_band.setText("2")
    res = dlg._compute(dlg._inputs())
    assert "error" not in res, res
    assert res["margin"] == pytest.approx(0.03)          # 2 t1
    assert res["rc"].t_first == pytest.approx(b.times[40])
    dlg._on_done(res)
    # one tab of plots per check, all filled
    assert dlg.tabs.count() == 3
    assert all(f.axes for f in dlg.figs.values())
    text = dlg.txt.toPlainText()
    assert "proposed window T" in text and "A = 40 dB needs" in text
    out = tmp_path / "s.csv"
    monkeypatch.setattr(
        "gui.widgets.preliminary_checks_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), ""))
    dlg._save()
    assert out.exists() and out.with_suffix(".xlsx").exists()


def test_dialog_fits_the_screen(qapp):
    from PySide6.QtWidgets import QApplication
    from gui.widgets.preliminary_checks_dialog import PreliminaryChecksDialog
    dlg = PreliminaryChecksDialog()
    g = QApplication.primaryScreen().availableGeometry()
    assert dlg.width() <= g.width() and dlg.height() <= g.height()


def test_model_tab_opens_the_dialog(qapp):
    from gui.core.model_config import ModelConfig
    from gui.tabs.optimization_tab import OptimizationTab
    tab = OptimizationTab(ModelConfig())
    tab.btn_prelim.click()
    dlg = tab._prelim_dialog
    assert dlg.isVisible()
    assert dlg.le_zone["xmin"].text() != ""
    dlg.close()
