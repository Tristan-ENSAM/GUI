# -*- coding: utf-8 -*-
"""
DIC tab: ROI selection lifecycle (selector re-attached on sequence change /
scrubbing, « Select ROI » mode, « Full image »), Q4 mask preview with the tool
polygon, and run cancellation.
"""
import numpy as np
import pytest
from PySide6.QtCore import Qt

from gui.core.experiment_session import ExperimentSession
from gui.tabs.dic_tab import DICTab, _DicGlobalWorker, _DicWorker
from gui.core import dic as dl
from gui.core import dic_global as dg
from gui.core.sequence_io import ImageSequence


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


def _frames(H=96, W=96, n=3, seed=0):
    base = _speckle(H, W, seed=seed)
    return np.stack([np.roll(base, k, axis=1) for k in range(n)])


def _tab(qapp, frames=None):
    tab = DICTab(ExperimentSession(name="t"))
    tab.set_frames(_frames() if frames is None else frames, fps=1000.0)
    return tab


def _rect_visible(sel):
    return any(a.get_visible() for a in sel.artists)


class TestSelectorLifecycle:

    def test_new_sequence_releases_validation(self, qapp):
        tab = _tab(qapp)
        tab.set_roi((10, 10, 50, 50))
        tab.b_validate.setChecked(True)
        assert tab._roi_locked
        tab.set_frames(_frames(seed=1))              # "change folder"
        assert not tab._roi_locked
        assert not tab.b_validate.isChecked()
        assert tab.b_select_roi.isEnabled()

    def test_new_sequence_keeps_rectangle_visible(self, qapp):
        tab = _tab(qapp)
        tab.b_select_roi.setChecked(True)
        tab.set_roi((10, 12, 50, 40))
        old = tab._selector
        tab.set_frames(_frames(seed=1))
        sel = tab._selector
        assert sel is not old
        assert not old.active                        # old one detached
        assert sel.active                            # drawing mode kept
        assert _rect_visible(sel)
        assert np.allclose(sel.extents, (10, 60, 12, 52))

    def test_new_smaller_sequence_clips_roi(self, qapp):
        tab = _tab(qapp)
        tab.set_roi((10, 10, 80, 80))
        tab.set_frames(_frames(H=64, W=64, seed=1))
        assert tab._roi == pytest.approx((10, 10, 53, 53))

    def test_roi_outside_new_image_is_dropped(self, qapp):
        tab = _tab(qapp, _frames(H=200, W=200))
        tab.set_roi((150, 150, 40, 40))
        tab.set_frames(_frames(H=64, W=64, seed=1))
        assert tab._roi is None

    def test_scrub_keeps_rectangle(self, qapp):
        tab = _tab(qapp)
        tab.set_roi((10, 12, 50, 40))
        tab.sld_frame.setValue(2)
        assert _rect_visible(tab._selector)
        assert np.allclose(tab._selector.extents, (10, 60, 12, 52))

    def test_select_mode_toggles_selector(self, qapp):
        tab = _tab(qapp)
        assert not tab._selector.active              # off by default
        tab.b_select_roi.setChecked(True)
        assert tab._selector.active
        tab.b_select_roi.setChecked(False)
        assert not tab._selector.active

    def test_validate_leaves_select_mode(self, qapp):
        tab = _tab(qapp)
        tab.b_select_roi.setChecked(True)
        tab.set_roi((10, 10, 50, 50))
        tab.b_validate.setChecked(True)
        assert not tab.b_select_roi.isChecked()
        assert not tab.b_select_roi.isEnabled()
        assert not tab.b_full_roi.isEnabled()
        assert not tab._selector.active
        assert not _rect_visible(tab._selector)      # green outline instead

    def test_click_without_drag_keeps_roi(self, qapp):
        class _E:
            def __init__(self, x, y):
                self.xdata, self.ydata = x, y
        tab = _tab(qapp)
        tab.set_roi((10, 10, 50, 50))
        tab._on_roi_select(_E(30, 30), _E(30.5, 30.5))
        assert tab._roi == (10, 10, 50, 50)


class TestFullImageRoi:

    def test_button_sets_full_image(self, qapp):
        tab = _tab(qapp, _frames(H=80, W=96))
        tab.b_full_roi.click()
        assert tab._roi == (0.0, 0.0, 95.0, 79.0)

    def test_validate_without_roi_uses_full_image(self, qapp):
        tab = _tab(qapp, _frames(H=80, W=96))
        assert tab._roi is None
        tab.b_validate.setChecked(True)
        assert tab._roi == (0.0, 0.0, 95.0, 79.0)

    def test_full_image_run_local(self, qapp):
        pytest.importorskip("cv2")
        tab = _tab(qapp)
        tab.spin_scale.setValue(0.01)
        tab.sp_subset.setValue(15); tab.sp_step.setValue(10)
        tab.sp_search.setValue(4)
        tab.b_full_roi.click()
        res = tab.run()
        assert res["x"].size > 0

    def test_full_image_mesh_stays_inside_image(self, qapp):
        tab = _tab(qapp)
        tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
        tab.sp_elem.setValue(24)
        tab.b_full_roi.click()
        mesh = dg.build_mesh_on_roi(tab._roi, 24)
        assert mesh.nodes[:, 0].max() <= 95 and mesh.nodes[:, 1].max() <= 95


class TestQ4MaskPreview:

    def _global_tab(self, qapp, frames=None):
        tab = _tab(qapp, frames)
        tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
        tab.sp_elem.setValue(24)
        tab.set_roi((10, 10, 72, 72))
        return tab

    def test_tool_polygon_shades_elements_with_mask_off(self, qapp):
        tab = self._global_tab(qapp)
        tab.session.reference_geometry = {
            "tool_polygon_px": [[0, 0], [50, 0], [50, 95], [0, 95]]}
        assert not tab.chk_mask.isChecked()
        tab._preview_points()
        polys = [a for a in tab._preview_artists
                 if type(a).__name__ == "Polygon"]
        assert len(polys) >= 2               # excluded elements + outline
        assert "kept" in tab.lbl_roi.text()

    def test_preview_matches_solver_mask(self, qapp):
        base = _speckle(seed=4)
        base[:, 48:] = 0
        tab = self._global_tab(qapp, np.stack([base, base]))
        tab.chk_mask.setChecked(True)
        tab.sld_int.setValue(20)
        tab.sp_coverage.setValue(0.5)
        mesh = dg.build_mesh_on_roi(tab._roi, 24)
        expected = dg.element_coverage_mask(
            mesh, dg.material_mask(base, 20.0, None), 0.5)
        n_excl = sum(1 for a in tab._preview_artists
                     if type(a).__name__ == "Polygon")
        assert n_excl == int((~expected).sum()) > 0


class TestCancel:

    def _run_cancelled(self, qapp, worker):
        cancelled, done, failed = [], [], []
        worker.sig_cancelled.connect(lambda d, n: cancelled.append((d, n)),
                                     Qt.QueuedConnection)
        worker.sig_done.connect(lambda *_: done.append(True),
                                Qt.QueuedConnection)
        worker.sig_failed.connect(failed.append, Qt.QueuedConnection)
        worker.start()
        worker.requestInterruption()
        worker.wait()
        for _ in range(50):
            qapp.processEvents()
        return cancelled, done, failed

    def test_global_worker_cancels_between_pairs(self, qapp):
        seq = ImageSequence.from_array(_frames(n=5), fps=1000.0)
        w = _DicGlobalWorker(seq, (10, 10, 72, 72),
                             dg.DicGlobalParams(elem_size=24),
                             1000.0, 0.01, 96, 96, 0.0, None)
        cancelled, done, failed = self._run_cancelled(qapp, w)
        assert not failed and not done
        assert len(cancelled) == 1 and cancelled[0][1] == 4
        assert 1 <= cancelled[0][0] <= 4

    def test_local_worker_cancels_between_pairs(self, qapp):
        pytest.importorskip("cv2")
        seq = ImageSequence.from_array(_frames(n=5), fps=1000.0)
        pts = dl.make_grid((10, 10, 72, 72), 10, margin=12)
        w = _DicWorker(seq, pts, dl.DicParams(subset=15, step=10, search=4),
                       1000.0, 0.01, 96, 96, 0.0)
        cancelled, done, failed = self._run_cancelled(qapp, w)
        assert not failed and not done
        assert len(cancelled) == 1 and cancelled[0][1] == 4

    def test_cancel_button_state(self, qapp):
        tab = _tab(qapp)
        assert not tab.b_cancel.isEnabled()
        tab._set_busy(True)
        assert tab.b_cancel.isEnabled()
        assert not tab.b_choose_folder.isEnabled()
        tab._on_dic_cancelled(2, 4)
        assert not tab.b_cancel.isEnabled()
        assert tab.b_choose_folder.isEnabled()
        assert "Cancelled" in tab.lbl_status.text()
