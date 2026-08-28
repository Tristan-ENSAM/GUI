# -*- coding: utf-8 -*-
"""
Integration tests for the global Q4 engine wired into the DIC tab
(gui.tabs.dic_tab.DICTab). These exercise the Qt widget headlessly (offscreen)
through its public-ish entry points: engine switching, the synchronous run()
path, the reference-frame mesh preview, metadata, and the save round-trip.
"""
import os
import tempfile
import numpy as np
import pytest

from gui.core.experiment_session import ExperimentSession
from gui.tabs.dic_tab import DICTab
from gui.core.exp_field_io import load_dic_field


def _speckle(H=96, W=96, n=250, seed=0):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    img = np.zeros((H, W), float)
    for _ in range(n):
        cx = rng.integers(5, W - 5)
        cy = rng.integers(5, H - 5)
        r = rng.uniform(2.0, 4.0)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _shift(img, ux, uy):
    H, W = img.shape
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    out = np.real(np.fft.ifft2(np.fft.fft2(img)
                               * np.exp(-2j * np.pi * (FX * ux + FY * uy))))
    return out.clip(0, 255).astype(np.uint8)


def _make_tab(qapp, frames, fps=1000.0):
    sess = ExperimentSession(name="t")
    tab = DICTab(sess)
    tab.set_frames(frames, fps=fps)
    return tab, sess


def _select_global(tab, elem=24, scale_mm_px=0.01, roi=(10, 10, 72, 72)):
    tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
    tab.spin_scale.setValue(scale_mm_px)
    tab.cb_scale_unit.setCurrentText("mm/px")
    tab.sp_elem.setValue(elem)
    tab.set_roi(roi)
    tab._roi_locked = True            # simulate a validated ROI for run()


@pytest.fixture
def base_frames():
    base = _speckle(seed=3)
    return [base, _shift(base, 0.6, 0.0), _shift(base, 1.2, 0.0)]


class TestEngineSwitching:

    def test_global_item_enabled(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        idx = tab.cb_engine.findData("global")
        assert idx >= 0
        assert tab.cb_engine.model().item(idx).isEnabled()

    def test_widgets_toggle(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        # Local by default: local widgets visible, global hidden.
        assert tab.sp_subset.isVisible() or not tab.isVisible()
        tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
        assert tab._is_global()
        # After switching, the global params object reflects the widgets.
        p = tab._global_params()
        assert p.engine == "global"
        assert p.variant in ("standard", "hild")


class TestGlobalRun:

    def test_run_shapes_and_conventions(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        res = tab.run()
        n_nodes = res["mesh"].n_nodes
        assert res["fields"]["Ux"].shape == (2, n_nodes)
        assert res["grid"] is None
        # +x image shift -> positive model Ux, ~0 Uy (mm_per_px=0.01, 0.6 px).
        assert abs(float(np.nanmean(res["fields"]["Ux"][0])) - 0.006) < 1e-3
        assert abs(float(np.nanmean(res["fields"]["Uy"][0]))) < 5e-4

    def test_meta_records_global(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.run()
        meta = tab._result_meta
        assert meta["engine"] == "global"
        assert "dic_global" in meta
        assert meta["dic_global"]["elem_size"] == 24

    def test_max_iter_tol_from_widgets(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.sp_maxiter.setValue(15)
        tab.sp_tol.setValue(5e-3)
        p = tab._global_params()
        assert p.max_iter == 15
        assert abs(p.tol - 5e-3) < 1e-12
        tab.run()
        assert tab._result_meta["dic_global"]["max_iter"] == 15

    def test_pyramid_from_widgets(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.sp_pyr_levels.setValue(3)
        tab.sp_pyr_sigma.setValue(1.5)
        p = tab._global_params()
        assert p.pyramid_levels == 3
        assert abs(p.pyramid_sigma - 1.5) < 1e-12
        tab.run()
        assert tab._result_meta["dic_global"]["pyramid_levels"] == 3


class TestParamsDialog:

    def test_button_and_dialog(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        # The compact panel exposes the engine combo and a Parameters button.
        assert hasattr(tab, "b_params")
        # The dialog is built immediately (hidden) so the parameter widgets are
        # parented into it and don't float as separate top-level windows.
        assert tab._params_dialog is not None
        assert not tab._params_dialog.isVisible()
        tab._open_params_dialog()
        assert tab._params_dialog is not None

    def test_param_widgets_are_parented(self, qapp, base_frames):
        """Regression: parameter widgets must have a parent (not be orphan
        top-level windows) even before the dialog is first opened."""
        tab, _ = _make_tab(qapp, base_frames)
        for name in ("sp_subset", "sp_step", "sp_search", "sp_zncc",
                     "sp_elem", "sp_coverage", "sld_int"):
            w = getattr(tab, name)
            assert w.parent() is not None

    def test_pyramid_widgets_exist(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        assert hasattr(tab, "sp_pyr_levels")
        assert hasattr(tab, "sp_pyr_sigma")
        assert tab.sp_pyr_levels.value() == 1     # single-scale by default

    def test_summary_updates_with_engine(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        assert "Local" in tab.lbl_engine_summary.text()
        tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
        assert "Global Q4" in tab.lbl_engine_summary.text()
        tab.sp_pyr_levels.setValue(4)
        assert "pyr 4" in tab.lbl_engine_summary.text()

    def test_mask_widgets_in_dialog(self, qapp, base_frames):
        """The mask controls moved into the dialog but keep their attributes,
        so masking still works through the same API."""
        tab, _ = _make_tab(qapp, base_frames)
        assert hasattr(tab, "chk_mask")
        assert hasattr(tab, "sld_int")
        tab.chk_mask.setChecked(True)
        assert "mask on" in tab.lbl_engine_summary.text()

    def test_variant_hild(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.cb_variant.setCurrentIndex(tab.cb_variant.findData("hild"))
        res = tab.run()
        assert abs(float(np.nanmean(res["fields"]["Ux"][0])) - 0.006) < 1e-3

    def test_total_pattern(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.cb_pattern.setCurrentIndex(tab.cb_pattern.findData(False))  # total
        res = tab.run()
        ux0 = float(np.nanmean(res["fields"]["Ux"][0]))
        ux1 = float(np.nanmean(res["fields"]["Ux"][1]))
        assert ux1 > 1.8 * ux0     # total grows ~linearly with frame index


class TestMeshPreview:

    def test_preview_draws_mesh_lines(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
        tab.sp_elem.setValue(24)
        tab.set_roi((10, 10, 72, 72))
        tab._on_engine_changed()
        # 3x3 elements -> 4 vertical + 4 horizontal grid lines = 8 artists.
        assert len(tab._preview_artists) == 8

    def test_preview_handles_too_small_roi(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
        tab.sp_elem.setValue(200)            # bigger than the ROI
        tab.set_roi((10, 10, 50, 50))
        tab._on_engine_changed()             # must not raise
        assert "too small" in tab.lbl_roi.text()


class TestSaveRoundTrip:

    def test_save_and_reload(self, qapp, base_frames):
        tab, sess = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.run()
        d = tempfile.mkdtemp()
        p = os.path.join(d, "g_dic.npz")
        tab.save_field(p)
        assert sess.dic_field_path == str(p)
        loaded = load_dic_field(p)
        # Velocity stored as V1/V2/Vmag; strain rates / residual carried in
        # extra. Cumulated strain is no longer produced (symmetry with local).
        for k in ("V1", "V2", "Vmag", "Exx_dot", "Eeq_dot", "residual"):
            assert k in loaded
        assert "Exx" not in loaded and "Eeq" not in loaded
        assert loaded["meta"]["engine"] == "global"


class TestWorkerLog:

    def test_worker_emits_log_with_eta(self, qapp, base_frames):
        """The global worker emits one detailed status line per frame, with the
        iteration count, residual, convergence flag, per-frame time and an ETA."""
        from PySide6.QtCore import Qt
        from gui.tabs.dic_tab import _DicGlobalWorker, _fmt_eta
        from gui.core import dic_global as dg
        from gui.core.sequence_io import ImageSequence

        assert _fmt_eta(12.0) == "12 s"
        assert _fmt_eta(185.0) == "3 min 05 s"

        seq = ImageSequence.from_array(np.asarray(base_frames), fps=1000.0)
        w = _DicGlobalWorker(seq, (10, 10, 72, 72),
                             dg.DicGlobalParams(elem_size=24),
                             1000.0, 0.01, 96, 96, 0.0, None)
        msgs, done, failed = [], [], []
        w.sig_log.connect(lambda m: msgs.append(m), Qt.QueuedConnection)
        w.sig_done.connect(lambda *_: done.append(True), Qt.QueuedConnection)
        w.sig_failed.connect(lambda m: failed.append(m), Qt.QueuedConnection)
        w.start()
        w.wait()
        for _ in range(50):
            qapp.processEvents()
        assert not failed, failed
        assert done
        # One log line per pair (3 frames -> 2 pairs).
        assert len(msgs) == len(base_frames) - 1
        assert "ETA" in msgs[0] and "iter" in msgs[0] and "residual" in msgs[0]

    def test_fmt_eta_boundaries(self, qapp):
        from gui.tabs.dic_tab import _fmt_eta
        assert _fmt_eta(-5.0) == "0 s"
        assert _fmt_eta(59.4) == "59 s"
        assert _fmt_eta(60.0) == "1 min 00 s"


class TestCoverageMaskUI:

    def test_coverage_widget_and_params(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        assert hasattr(tab, "sp_coverage")
        tab.chk_mask.setChecked(True)
        tab.sld_int.setValue(20)
        tab.sp_coverage.setValue(0.6)
        p = tab._global_params()
        assert p.mask_enabled is True
        assert p.mask_min_intensity == 20
        assert abs(p.coverage_threshold - 0.6) < 1e-12

    def test_summary_shows_coverage(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.chk_mask.setChecked(True)
        tab.sp_coverage.setValue(0.7)
        assert "cov\u22650.70" in tab.lbl_engine_summary.text()

    def test_mesh_preview_shades_excluded(self, qapp):
        # Build a sequence whose right half is dark -> some elements excluded.
        base = _speckle(seed=4)
        base[:, 48:] = 0
        frames = [base, base]
        tab, _ = _make_tab(qapp, frames)
        tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
        tab.spin_scale.setValue(0.01)
        tab.sp_elem.setValue(24)
        tab.chk_mask.setChecked(True)
        tab.sld_int.setValue(20)
        tab.sp_coverage.setValue(0.5)
        tab.set_roi((10, 10, 72, 72))
        tab._on_engine_changed()
        # Excluded elements are drawn as Polygon patches.
        n_poly = sum(1 for a in tab._preview_artists
                     if type(a).__name__ == "Polygon")
        assert n_poly > 0
        assert "kept" in tab.lbl_roi.text()

    def test_meta_records_coverage(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.chk_mask.setChecked(True)
        tab.sp_coverage.setValue(0.55)
        tab.run()
        assert tab._result_meta["dic_global"]["coverage_threshold"] == 0.55
        assert tab._result_meta["dic_global"]["mask_enabled"] is True


class TestConvectionUI:

    def test_convect_widget_and_param(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        assert hasattr(tab, "chk_convect")
        tab.chk_convect.setChecked(True)
        assert tab._global_params().convect is True

    def test_convect_in_summary(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.chk_convect.setChecked(True)
        assert "convect" in tab.lbl_engine_summary.text()

    def test_convect_in_meta(self, qapp, base_frames):
        tab, _ = _make_tab(qapp, base_frames)
        _select_global(tab)
        tab.chk_convect.setChecked(True)
        tab.run()
        assert tab._result_meta["dic_global"]["convect"] is True
