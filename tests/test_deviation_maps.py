# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.deviation_maps (maps of the deviation between
two runs of a sizing study)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import gui.sensitivity.domain_independence as di
from gui.sensitivity import deviation_maps as dm
from gui.sensitivity.run_cache import SUCCESS_MARK

_ZOI = (-0.02, 0.02, -0.01, 0.01)
_THR = {"Vx": 10.0, "Vy": 10.0, "T": 10.0, "EVF": 0.1, "Fc": 10.0,
        "Ff": 10.0}
_NT, _NH = 21, 81


class _Info:
    field_variables = ["EVF", "TEMP", "V1", "V2"]


class _Bundle:
    """Element centroids on a regular grid of size h over a box containing
    the ZOI. TEMP = 300 + gx*x + offset + amp*sin(w t + phase)."""

    def __init__(self, h=0.005, offset=0.0, amp=0.0, phase=0.0, void_x=None,
                 force_offset=0.0, force_amp=0.0, gx=0.0):
        xs = np.arange(-0.05 + h / 2, 0.05, h)
        ys = np.arange(-0.03 + h / 2, 0.03, h)
        X, Y = np.meshgrid(xs, ys)
        self.c = np.column_stack([X.ravel(), Y.ravel(),
                                  np.zeros(X.size)])
        self.times = np.linspace(0.0, 1e-4, _NT)
        self.history_time = np.linspace(0.0, 1e-4, _NH)
        self.instance_names = ["EULER"]
        self.offset, self.amp, self.phase = offset, amp, phase
        self.void_x, self.gx = void_x, gx
        self.force_offset, self.force_amp = force_offset, force_amp
        self.job_name = "fake"

    def instance(self, name):
        return _Info()

    def element_centroids_init(self, inst):
        return self.c

    def _wave(self, t):
        return self.amp * np.sin(2 * np.pi * t / 2e-5 + self.phase)

    def field(self, inst, var):
        x = self.c[:, 0][None, :]
        t = self.times[:, None]
        if var == "EVF":
            evf = np.ones((_NT, x.size))
            if self.void_x is not None:
                evf[:, self.c[:, 0] > self.void_x] = 0.0
            return evf
        if var == "TEMP":
            return 300.0 + self.gx * x + self.offset + self._wave(t)
        return np.full((_NT, x.size), 100.0)

    def history(self, var):
        base = {"RF1_RP": 0.8, "RF2_RP": 0.3}[var]
        return (base + self.force_offset
                + 1e-3 * self._wave(self.history_time) * self.force_amp)


def _settings(thr=None):
    return dm.CompareSettings(step="ms", zoi=_ZOI, grid_step=0.005,
                              window=(0.3, 1.0), thresholds=dict(thr or _THR))


class TestMetric:
    def test_e_inst_is_the_study_metric(self):
        a = _Bundle(amp=2.0)
        b = _Bundle(offset=3.0, amp=5.0, phase=1.0, force_offset=0.01)
        s = _settings()
        pair = dm.compare_runs(a, b, s, 0.005, 0.005)
        assert not pair.linear
        kw = dict(window=s.window, evf_threshold=0.5)
        sa = di.sample_zoi(a, _ZOI, 0.005, 0.005, **kw)
        sb = di.sample_zoi(b, _ZOI, 0.005, 0.005, **kw)
        errs = di.errors_between(sb, sa)
        for q in ("T", "Vx", "EVF", "Fc", "Ff"):
            assert pair.quantities[q].e_inst == pytest.approx(
                errs[q] / _THR[q], rel=1e-9, abs=1e-12)

    def test_offset_is_all_bias(self):
        pair = dm.compare_runs(_Bundle(), _Bundle(offset=4.0), _settings(),
                               0.005, 0.005)
        d = pair.quantities["T"]
        assert d.e_inst == pytest.approx(0.4)
        assert d.e_bias == pytest.approx(0.4)
        assert np.allclose(d.mean_map(), 0.4)
        assert np.allclose(d.frame_map(0), 0.4)

    def test_out_of_phase_oscillation_is_not_bias(self):
        # Same mean, oscillation in opposition over whole periods of T.
        a = _Bundle(amp=5.0)
        b = _Bundle(amp=5.0, phase=np.pi)
        s = dm.CompareSettings(step="ms", zoi=_ZOI, grid_step=0.005,
                               window=(0.2, 1.0), thresholds=_THR)
        d = dm.compare_runs(a, b, s, 0.005, 0.005).quantities["T"]
        assert d.e_inst > 0.2
        assert d.e_bias < 1e-9
        assert d.e_frame.shape == d.times.shape

    def test_void_of_either_run_is_not_compared(self):
        a = _Bundle()
        b = _Bundle(offset=2.0, void_x=0.0)
        pair = dm.compare_runs(a, b, _settings(), 0.005, 0.005)
        m = pair.quantities["T"].mean_map()
        x = pair.points[:, 0]
        # nearest centroid of x = 0 is at x = -0.0025 or +0.0025
        assert np.all(np.isnan(m[x > 0.003]))
        assert np.allclose(m[x < -0.003], 0.2)
        assert np.all(pair.quantities["T"].n_compared()[x > 0.003] == 0)

    def test_force_bias_and_curves(self):
        a = _Bundle(force_amp=1.0)
        b = _Bundle(force_amp=1.0, force_offset=0.02)
        d = dm.compare_runs(a, b, _settings(), 0.005, 0.005
                            ).quantities["Fc"]
        assert not d.is_field
        # 0.02 N / 0.005 mm = 4 N/mm over eps 10
        assert d.e_bias == pytest.approx(0.4)
        assert d.e_inst == pytest.approx(0.4)
        assert d.a.shape == d.b.shape == d.times.shape

    def test_different_meshes_use_linear_sampling(self):
        # A field linear in x is read exactly by the linear interpolation.
        a = _Bundle(h=0.0025, gx=100.0)
        b = _Bundle(h=0.005, gx=100.0)
        pair = dm.compare_runs(a, b, _settings(), 0.0025, 0.005)
        assert pair.linear
        assert pair.quantities["T"].e_inst == pytest.approx(0.0, abs=1e-9)

    def test_grid_shape_and_nearest_point(self):
        pair = dm.compare_runs(_Bundle(), _Bundle(), _settings(), 0.005,
                               0.005)
        ny, nx = pair.shape
        assert (ny, nx) == (5, 9)
        assert ny * nx == pair.points.shape[0]
        i = pair.nearest_point(0.019, -0.009)
        assert tuple(pair.points[i]) == pytest.approx((0.02, -0.01))
        assert pair.quantities["T"].mean_map().reshape(ny, nx).shape == (5, 9)


def _write_run(folder, job, ms, h, dims=0.2, success=True):
    params = {"step": {"mass_scaling_enabled": ms != 1.0,
                       "mass_scaling_factor_eulerian": ms},
              "mesh": {"elem_size": h},
              "geometry": {"euler": {"geometry": {
                  "h_wp": dims, "h_void": dims, "l_wp": dims,
                  "l_void": dims}}}}
    (folder / (job + ".meta.json")).write_text(
        json.dumps({"model_config": params}), encoding="utf-8")
    (folder / (job + ".results.npz")).write_bytes(b"")
    (folder / (job + ".sta")).write_text(
        SUCCESS_MARK if success else "running", encoding="utf-8")


class TestFolders:
    def test_list_runs_labels_and_reference(self, tmp_path):
        _write_run(tmp_path, "ms_run000", 1.0, 0.002)
        _write_run(tmp_path, "ms_run001", 25600.0, 0.002)
        _write_run(tmp_path, "ms_run002", 51200.0, 0.002, success=False)
        runs = dm.list_runs([tmp_path])
        assert [r.job for r in runs] == ["ms_run000", "ms_run001"]
        assert dm.default_reference(runs, "ms") == 0
        assert dm.run_label(runs[1]).startswith("ms 25600 | h 0.002 mm")

    def test_reference_mesh_and_domain(self, tmp_path):
        _write_run(tmp_path, "GCI_run000", 25600.0, 0.002)
        _write_run(tmp_path, "GCI_run001", 25600.0, 0.0005)
        _write_run(tmp_path, "GCI_run002", 25600.0, 0.001, dims=0.3)
        runs = dm.list_runs([tmp_path])
        assert dm.run_elem_size(runs[dm.default_reference(runs, "mesh")
                                     ].params) == 0.0005
        assert dm.run_domain(runs[dm.default_reference(runs, "domain")
                                  ].params)["h_wp"] == 0.3

    def test_settings_of_ms_and_gci_folders(self, tmp_path):
        ms = tmp_path / "ms"
        ms.mkdir()
        (ms / "config.json").write_text(json.dumps({
            "study": "massscaling", "parameters": {
                "zoi": {"xmin": -0.1, "xmax": 0.0, "ymin": -0.05,
                        "ymax": 0.05},
                "elem_size": 0.002, "ms_values": [1, 200],
                "grid_step": 0.005, "thresholds_abs": _THR,
                "window": [0.3, 1.0], "domain_dims": {}}}))
        s = dm.settings_of_folder(ms)
        assert s.step == "ms" and s.zoi == (-0.1, 0.0, -0.05, 0.05)
        assert s.thresholds == _THR and s.grid_step == 0.005
        gci = tmp_path / "gci"
        gci.mkdir()
        (gci / "config.json").write_text(json.dumps({
            "study": "GCI", "parameters": {
                "zoi": [-0.1, 0.0, -0.05, 0.05], "window": [0.3, 1.0],
                "finest_elem_size": 0.0005, "ratio": 2, "n_meshes": 3,
                "grid_step": 0.002, "domain_dims": {},
                "tolerances": {"TEMP": 10.0, "V1": 10.0, "V2": 10.0,
                               "EVF": 0.1, "Fc": 10.0, "Ff": 10.0}}}))
        g = dm.settings_of_folder(gci)
        assert g.step == "mesh"
        assert g.thresholds == _THR

    def test_exports(self, tmp_path):
        pair = dm.compare_runs(_Bundle(), _Bundle(offset=1.0), _settings(),
                               0.005, 0.005)
        pair.label_a, pair.label_b = "a", "b"
        paths = dm.write_pair_exports(tmp_path, pair, "dev")
        assert [p.name for p in paths] == ["dev_summary.csv", "dev_maps.csv"]
        assert (tmp_path / "dev_summary.xlsx").exists()
        text = (tmp_path / "dev_maps.csv").read_text(encoding="utf-8")
        assert text.splitlines()[0].startswith("x_mm,y_mm,Vx_mean_over_eps")


# ---------------------------------------------------------------------------
# Window (Qt)
# ---------------------------------------------------------------------------
def _study_folder(tmp_path):
    """A mass-scaling study folder with two runs written by fake_builder."""
    from gui.results.fake_builder import build_fake_results
    folder = tmp_path / "Ducobu_2017_massscaling_test"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "study": "massscaling", "parameters": {
            "zoi": {"xmin": -0.05, "xmax": 0.25, "ymin": -0.1,
                    "ymax": 0.1},
            "elem_size": 0.01, "ms_values": [1, 25600], "grid_step": 0.01,
            "thresholds_abs": _THR, "window": [0.3, 1.0],
            "domain_dims": {}}}))
    for job, ms, scale, seed in (("ms_run000", 1.0, 1.0, 1),
                                 ("ms_run001", 25600.0, 1.03, 2)):
        _write_run(folder, job, ms, 0.01)
        params = json.loads((folder / (job + ".meta.json")).read_text(
            encoding="utf-8"))["model_config"]
        js, _npz = build_fake_results(folder / (job + ".results.npz"),
                                      job_name=job, field_scale=scale,
                                      seed=seed)
        meta = json.loads(js.read_text(encoding="utf-8"))
        meta["model_config"] = params
        (folder / (job + ".meta.json")).write_text(json.dumps(meta),
                                                    encoding="utf-8")
    return folder


class _Event:
    def __init__(self, ax, x, y):
        self.inaxes, self.xdata, self.ydata = ax, x, y


def test_window_compare_views_click_export(qapp, tmp_path, monkeypatch):
    import gui.widgets.deviation_maps_dialog as dlg_mod
    monkeypatch.setattr(dlg_mod, "run_async",
                        lambda fn, done, parent: done(fn()))

    def _no_box(*a, **k):
        raise AssertionError("message box: %r" % (a[2:],))
    monkeypatch.setattr(dlg_mod.QMessageBox, "warning", _no_box)
    folder = _study_folder(tmp_path)
    dlg = dlg_mod.DeviationMapsDialog(folders=[str(folder)])
    assert dlg.cb_a.count() == 2
    assert dlg.cb_a.currentText().startswith("ms 1 |")
    assert dlg.cb_b.currentText().startswith("ms 25600 |")
    dlg._on_compare()
    assert dlg._pair is not None
    assert set(dlg._pair.quantities) == {"T", "EVF", "Fc", "Ff"}
    dlg.cb_q.setCurrentText("T")
    for i in range(dlg.cb_view.count()):
        dlg.cb_view.setCurrentIndex(i)
        dlg.sl_frame.setValue(3)
    ax = dlg.fig_map.axes[0]
    dlg._on_click(_Event(ax, 0.1, 0.05))
    assert dlg._point is not None
    assert len(dlg.fig_curves.axes[0].lines) >= 2
    # Deviation over time: ZOI mean, the point, the threshold of the step.
    dev = dlg.fig_dev.axes[0]
    assert any(ln.get_label().startswith("threshold of the step")
               for ln in dev.lines)
    assert len(dev.lines) >= 3
    dlg.cb_q.setCurrentText("Fc")
    assert not dlg.sl_frame.isEnabled()
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(tmp_path / "out"))
    monkeypatch.setattr(dlg_mod.QMessageBox, "information",
                        lambda *a, **k: None)
    dlg._on_export()
    assert (tmp_path / "out" /
            "deviation_ms_run001_vs_ms_run000_maps.xlsx").exists()


def test_window_folders_are_separate_and_runs_exclusive(qapp, tmp_path,
                                                        monkeypatch):
    import shutil
    import gui.widgets.deviation_maps_dialog as dlg_mod
    monkeypatch.setattr(dlg_mod.QMessageBox, "information",
                        lambda *a, **k: None)
    f1 = _study_folder(tmp_path)
    f2 = tmp_path / "rerun"
    f2.mkdir()
    for p in f1.iterdir():
        shutil.copy(p, f2 / p.name)
    empty = tmp_path / "empty"
    empty.mkdir()
    dlg = dlg_mod.DeviationMapsDialog(folders=[str(f1)])
    # "Add a folder" puts the folder in the Study list and selects it; the
    # lists (a), (b) then hold its runs only.
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(f2))
    dlg._on_add_folder()
    assert dlg.cb_study.currentData() == str(f2)
    assert dlg.cb_a.count() == 2
    assert all(r.folder == f2 for r in dlg._runs)
    assert dlg.add_folder(empty) == 0
    assert dlg.cb_study.findData(str(empty)) < 0
    dlg.cb_study.setCurrentIndex(dlg.cb_study.findData(str(f1)))
    assert all(r.folder == f1 for r in dlg._runs)
    # All the folders: grouped, with one disabled header per folder.
    dlg.cb_study.setCurrentIndex(dlg.cb_study.findData(dlg_mod._ALL))
    assert len(dlg._runs) == 4 and dlg.cb_a.count() == 6
    assert not dlg.cb_a.model().item(0).isEnabled()
    # The run of the other list is greyed out; picking it moves the other.
    ib = dlg.cb_b.currentData()
    k = dlg.cb_a.findData(ib)
    assert not dlg.cb_a.model().item(k).isEnabled()
    assert "already b" in dlg.cb_a.itemText(k)
    dlg.cb_a.setCurrentIndex(k)
    assert dlg.cb_a.currentData() == ib != dlg.cb_b.currentData()


def test_share_of_the_step_is_the_threshold(tmp_path):
    folder = tmp_path / "ms"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "study": "massscaling", "parameters": {
            "zoi": [-0.1, 0.0, -0.05, 0.05], "elem_size": 0.002,
            "ms_values": [1, 200], "grid_step": 0.005,
            "thresholds_abs": _THR, "window": [0.3, 1.0],
            "domain_dims": {}, "eps_shares": [0.5, 0.25, 0.25]}}))
    s = dm.settings_of_folder(folder)
    assert s.share == 0.5 and "ms" in s.source
