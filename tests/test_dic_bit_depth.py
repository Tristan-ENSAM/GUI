# -*- coding: utf-8 -*-
"""Camera bit depth: stored per image stream in the session, entered in the
Experimental data tab, and used by the DIC tab to set the raw range of the
mask threshold (0..2^bits-1)."""
import numpy as np

from gui.core.experiment_session import ExperimentSession
from gui.tabs.dic_tab import DICTab
from gui.core import dic_global as dg


def _frames16(vmax=4000, H=64, W=64, n=2, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, vmax + 1, (H, W)).astype(np.uint16)
    return np.stack([img] * n)


class TestSession:

    def test_default_and_roundtrip(self, tmp_path):
        s = ExperimentSession()
        assert s.visible.bit_depth == 8 and s.ir.bit_depth == 8
        s.visible.bit_depth = 12
        p = s.save(tmp_path / "e.json")
        assert ExperimentSession.load(p).visible.bit_depth == 12

    def test_old_file_without_bit_depth_loads(self):
        s = ExperimentSession.from_json_dict({"visible": {"fps": 5000.0}})
        assert s.visible.bit_depth == 8 and s.visible.fps == 5000.0


class TestExperimentalTab:

    def test_spin_pushes_to_session(self, qapp):
        from gui.tabs.experimental_data_tab import AcquisitionTab
        s = ExperimentSession()
        tab = AcquisitionTab(s)
        tab._visible_bits.setValue(12)
        assert s.visible.bit_depth == 12
        s.ir.bit_depth = 14
        tab.apply_from_session()
        assert tab._ir_bits.value() == 14


class TestDicTab:

    def test_slider_range_follows_bit_depth(self, qapp):
        tab = DICTab(ExperimentSession())
        assert tab.sld_int.maximum() == 255
        tab.sp_bits.setValue(12)
        assert tab.sld_int.maximum() == 4095
        tab.sld_int.setValue(3000)
        tab.sp_bits.setValue(8)                  # value clipped to new range
        assert tab.sld_int.value() == 255

    def test_prefilled_from_session(self, qapp):
        s = ExperimentSession()
        s.visible.bit_depth = 12
        tab = DICTab(s)
        assert tab.sp_bits.value() == 12
        assert tab.sld_int.maximum() == 4095

    def test_warns_when_pixels_exceed_range(self, qapp):
        tab = DICTab(ExperimentSession())
        tab.set_frames(_frames16(vmax=4000))
        assert "too low" in tab.lbl_bits.text()       # 8 bits declared
        tab.sp_bits.setValue(12)
        assert "too low" not in tab.lbl_bits.text()
        assert "uint16" in tab.lbl_bits.text()

    def test_raw_threshold_masks_12bit_background(self, qapp):
        img = np.full((96, 96), 3000, np.uint16)
        img[:, 48:] = 100                              # dark background
        tab = DICTab(ExperimentSession())
        tab.sp_bits.setValue(12)
        tab.set_frames(np.stack([img, img]))
        tab.cb_engine.setCurrentIndex(tab.cb_engine.findData("global"))
        tab.sp_elem.setValue(24)
        tab.chk_mask.setChecked(True)
        tab.sld_int.setValue(1000)                     # > 255: needs 12 bits
        tab.set_roi((0, 0, 95, 95))
        assert tab._global_params().mask_min_intensity == 1000
        mesh = dg.build_mesh_on_roi(tab._roi, 24)
        n_excl = sum(1 for a in tab._preview_artists
                     if type(a).__name__ == "Polygon")
        assert 0 < n_excl < mesh.n_elements

    def test_bit_depth_locked_and_in_meta(self, qapp):
        tab = DICTab(ExperimentSession())
        tab.sp_bits.setValue(12)
        tab.set_frames(_frames16())
        tab.set_roi((5, 5, 50, 50))
        tab.b_validate.setChecked(True)
        assert not tab.sp_bits.isEnabled()
        assert tab._build_meta(64, 64)["bit_depth"] == 12
