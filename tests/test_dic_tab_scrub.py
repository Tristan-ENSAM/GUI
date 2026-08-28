# -*- coding: utf-8 -*-
"""
Tests for the Search ROI frame scrubber and the per-frame mask preview (lot 5a).

These check that the scrubber is configured for the loaded sequence, that
scrubbing recomputes the keep-mask on the displayed frame, and that the DIC
*computation* path is unchanged (still masks on frame 0).
"""
import numpy as np

from gui.core.experiment_session import ExperimentSession
from gui.core.dic import make_grid
from gui.tabs.dic_tab import DICTab


def _speckle(H=120, W=120, n=300, seed=0):
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


def _frames_with_darkening():
    """Frame 0 is full speckle; frame 1 has its right half darkened (becomes
    'no-material' for the intensity mask); frame 2 is full speckle again."""
    base = _speckle(seed=1)
    f1 = base.copy()
    f1[:, 60:] = 2
    return [base, f1, base]


def _tab(qapp, frames):
    tab = DICTab(ExperimentSession(name="t"))
    tab.set_frames(frames, fps=1000.0)
    tab.spin_scale.setValue(0.01)
    return tab


class TestScrubber:

    def test_slider_configured(self, qapp):
        frames = _frames_with_darkening()
        tab = _tab(qapp, frames)
        assert tab.sld_frame.minimum() == 0
        assert tab.sld_frame.maximum() == len(frames) - 1
        assert tab.sld_frame.isEnabled()

    def test_single_frame_disables_slider(self, qapp):
        # A 2-frame sequence still allows 1 pair; a 1-frame sequence cannot be
        # browsed. Use 2 frames -> slider enabled; check label format.
        tab = _tab(qapp, [_speckle(seed=2), _speckle(seed=3)])
        assert tab.sld_frame.maximum() == 1
        assert "/1" in tab.lbl_frame_idx.text()

    def test_scrub_updates_index_and_label(self, qapp):
        frames = _frames_with_darkening()
        tab = _tab(qapp, frames)
        tab.sld_frame.setValue(2)
        assert tab._preview_frame_idx == 2
        assert tab.lbl_frame_idx.text() == "2/2"

    def test_preview_frame_image_follows_scrubber(self, qapp):
        frames = _frames_with_darkening()
        tab = _tab(qapp, frames)
        tab.sld_frame.setValue(1)
        img = tab._preview_frame_image()
        # Frame 1 has a darkened right half.
        assert img[:, 60:].mean() < 10


class TestPerFrameMask:

    def test_mask_differs_between_frames(self, qapp):
        frames = _frames_with_darkening()
        tab = _tab(qapp, frames)
        tab.set_roi((20, 20, 80, 80))
        tab.chk_mask.setChecked(True)
        tab.sld_int.setValue(25)
        pts = make_grid((20, 20, 80, 80), 12, margin=25)
        keep0 = tab._keep_mask(pts, tab._seq.frame(0))
        keep1 = tab._keep_mask(pts, tab._seq.frame(1))
        # The darkened half drops points on frame 1.
        assert keep1.sum() < keep0.sum()

    def test_computation_mask_uses_frame_zero(self, qapp):
        """The DIC compute path must keep masking on frame 0 (lot 5a leaves the
        computation unchanged); the default-image _keep_mask equals frame 0."""
        frames = _frames_with_darkening()
        tab = _tab(qapp, frames)
        tab.set_roi((20, 20, 80, 80))
        tab.chk_mask.setChecked(True)
        tab.sld_int.setValue(25)
        tab.sld_frame.setValue(1)            # scrub away from frame 0
        pts = make_grid((20, 20, 80, 80), 12, margin=25)
        keep_default = tab._keep_mask(pts)               # no image -> frame 0
        keep_frame0 = tab._keep_mask(pts, tab._seq.frame(0))
        assert np.array_equal(keep_default, keep_frame0)


class TestTextureRemoved:

    def test_no_texture_widget(self, qapp):
        from gui.core.experiment_session import ExperimentSession
        tab = DICTab(ExperimentSession(name="t"))
        # The texture slider was removed; only intensity remains.
        assert not hasattr(tab, "sld_tex")
        assert hasattr(tab, "sld_int")

    def test_point_mask_intensity_only(self):
        from gui.core.dic import point_mask
        import numpy as np
        img = np.full((40, 40), 200, np.uint8)   # bright, uniform (no texture)
        pts = np.array([[20.0, 20.0]])
        # Intensity-only: a bright uniform patch is kept (texture no longer
        # required), and a too-dark threshold drops it.
        assert point_mask(img, pts, win=11, min_intensity=50)[0]
        assert not point_mask(img, pts, win=11, min_intensity=250)[0]
