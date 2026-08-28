# -*- coding: utf-8 -*-
"""Tests for the DIC local engine and the experimental field IO."""
import os
import tempfile

import numpy as np
import pytest


def _speckle(h=200, w=260, seed=0):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w)).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _shift(img, dx, dy):
    cv2 = pytest.importorskip("cv2")
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def test_make_grid():
    from gui.core.dic import make_grid
    g = make_grid((0, 0, 100, 50), step=10, margin=10)
    assert g.shape[1] == 2
    assert g[:, 0].min() >= 10 and g[:, 0].max() <= 90
    assert g[:, 1].min() >= 10 and g[:, 1].max() <= 40
    # empty when margins eat the ROI
    assert make_grid((0, 0, 10, 10), step=5, margin=20).shape == (0, 2)


def test_correlate_local_translation():
    pytest.importorskip("cv2")
    from gui.core.dic import make_grid, correlate_local
    base = _speckle()
    dx, dy = 3.4, -2.1
    cur = _shift(base, dx, dy)
    pts = make_grid((40, 40, 180, 120), step=20, margin=30)
    disp, valid, score = correlate_local(base, cur, pts, subset=31, search=12)
    assert valid.sum() >= 0.9 * len(pts)
    d = disp[valid]
    assert np.nanmean(d[:, 0]) == pytest.approx(dx, abs=0.1)
    assert np.nanmean(d[:, 1]) == pytest.approx(dy, abs=0.1)


def test_velocity_fields_signs_and_units():
    pytest.importorskip("cv2")
    from gui.core.dic import make_grid, velocity_fields, DicParams
    base = _speckle()
    dx, dy = 3.4, -2.1
    frames = [base, _shift(base, dx, dy), _shift(base, 2 * dx, 2 * dy)]
    pts = make_grid((40, 40, 180, 120), step=20, margin=30)
    res = velocity_fields(frames, pts, DicParams(subset=31, search=12),
                          fps=10000.0, mm_per_px=0.01, img_w=260, img_h=200)
    assert res["V1"].shape == (2, len(pts))
    ok = res["valid"][0]
    # V1 = dx*mm_per_px*fps = 340 ; V2 = -dy*mm_per_px*fps = +210 (y flip)
    assert np.nanmean(res["V1"][0][ok]) == pytest.approx(340.0, abs=5)
    assert np.nanmean(res["V2"][0][ok]) == pytest.approx(210.0, abs=5)
    # midpoint times
    assert res["t"][0] == pytest.approx(0.5e-4)
    assert res["t"][1] == pytest.approx(1.5e-4)


def test_dic_field_io_roundtrip():
    from gui.core.exp_field_io import save_dic_field, load_dic_field
    x = np.linspace(-1, 1, 5); y = np.zeros(5); t = np.array([0.0, 1e-4])
    V1 = np.random.rand(2, 5).astype(np.float32)
    V2 = np.random.rand(2, 5).astype(np.float32)
    Vmag = np.hypot(V1, V2); valid = np.ones((2, 5), bool)
    d = tempfile.mkdtemp()
    p = save_dic_field(os.path.join(d, "essai_dic"), x, y, t, V1, V2, Vmag,
                       valid, meta={"source": "computed",
                                    "dic": {"engine": "local", "subset": 31}})
    assert p.with_suffix(".json").exists()
    r = load_dic_field(p)
    assert np.allclose(r["V1"], V1)
    assert r["meta"]["source"] == "computed"
    assert r["meta"]["dic"]["engine"] == "local"
    assert r["meta"]["n_frames"] == 2 and r["meta"]["n_points"] == 5


def test_dic_tab_run_and_save(qapp, tmp_path):
    pytest.importorskip("cv2")
    import numpy as np
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.dic_tab import DICTab
    from gui.core.exp_field_io import load_dic_field
    base = _speckle()
    frames = np.stack([base, _shift(base, 3, -2), _shift(base, 6, -4)])
    s = ExperimentSession(); s.visible.fps = 10000.0
    tab = DICTab(s)
    tab.set_frames(frames, fps=10000.0)
    tab.spin_scale.setValue(0.01)
    tab.set_roi((40, 40, 180, 120))
    tab.sp_subset.setValue(31); tab.sp_step.setValue(20); tab.sp_search.setValue(12)
    res = tab.run()
    f = res["fields"]
    assert set(["Vx", "Vy", "Vmag", "Exx_dot", "Eeq_dot"]).issubset(f)
    ok = res["valid"][0]
    assert np.nanmean(f["Vx"][0][ok]) == pytest.approx(300.0, abs=8)
    assert np.nanmean(f["Vy"][0][ok]) == pytest.approx(200.0, abs=8)
    # save -> session path + reload, derived fields present
    p = tab.save_field(str(tmp_path / "essai_dic.npz"))
    assert s.dic_field_path == str(p)
    r = load_dic_field(p)
    assert np.allclose(r["V1"], f["Vx"], equal_nan=True)
    assert "Exx_dot" in r and "Eeq_dot" in r
    assert r["meta"]["dic"]["engine"] == "local"


def test_dic_preview_and_validation(qapp):
    pytest.importorskip("cv2")
    import numpy as np
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.dic_tab import DICTab
    s = ExperimentSession()
    tab = DICTab(s)
    tab.set_frames(np.stack([_speckle(), _shift(_speckle(), 2, 0)]), fps=1000.0)
    tab.set_roi((40, 40, 180, 120))
    tab.sp_step.setValue(20)
    assert tab._grid_preview() is not None and len(tab._grid_preview()) > 0
    # Run is gated on validation
    assert tab.b_run.isEnabled() is False
    # nudge moves the ROI by 1 px
    tab._nudge_roi(1, -1)
    assert tab._roi[0] == 41 and tab._roi[1] == 39
    tab.b_validate.setChecked(True)
    assert tab._roi_locked is True
    assert tab.b_run.isEnabled() is True            # enabled once validated
    assert tab._nudge_btns[0].isEnabled() is False  # frozen while validated
    tab.b_validate.setChecked(False)
    assert tab._roi_locked is False
    assert tab.b_run.isEnabled() is False


def test_dic_viewer_background(qapp):
    import numpy as np
    from gui.widgets.dic_field_viewer import DicFieldViewer
    xs = np.linspace(-1, 1, 5); ys = np.linspace(-0.5, 0.5, 4)
    gx, gy = np.meshgrid(xs, ys); x = gx.ravel(); y = gy.ravel()
    t = np.array([0.0, 1e-4]); V = np.random.rand(2, x.size)
    vw = DicFieldViewer()
    vw.set_field(x, y, t, {"Vmag": V}, units={"Vmag": "mm/s"})
    img = np.zeros((200, 260), np.uint8)
    vw.set_background(lambda i: img, mm_per_px=0.01, img_w=260, img_h=200)
    vw.chk_bg.setChecked(True)
    assert vw._bg_im is not None
    ext = vw._bg_extent()
    assert ext == pytest.approx((-1.3, 1.3, -1.0, 1.0))


def test_dic_strain_shear():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from gui.core.dic import make_grid, compute_dic_fields, DicParams
    base = _speckle()
    a = 0.02
    H, W = base.shape
    ys, xs = np.mgrid[0:H, 0:W]
    mapx = (xs - a * ys).astype(np.float32); mapy = ys.astype(np.float32)
    cur = cv2.remap(base, mapx, mapy, interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT)
    pts = make_grid((40, 40, 180, 120), step=20, margin=30)
    res = compute_dic_fields([base, cur], pts, DicParams(subset=31, search=10),
                             fps=1000.0, mm_per_px=0.01, img_w=W, img_h=H)
    ok = res["valid"][0]; f = res["fields"]
    # Cumulated strain was removed (rates only). The shear RATE equals the
    # cumulated tensor shear divided by the frame period, i.e. * fps. With a
    # single step at fps=1000 and imposed shear a: Exy_dot ~= (-a/2) * fps.
    # u_x = a*y(px) -> in model frame Exy = 0.5*dUx/dy_model = -a/2 (per step).
    assert np.nanmean(f["Exy_dot"][0][ok]) == pytest.approx(-a / 2 * 1000.0, abs=4)
    assert abs(np.nanmean(f["Exx_dot"][0][ok])) < 4


def test_dic_viewer_zoom_kept_on_scroll(qapp):
    import numpy as np
    from gui.widgets.dic_field_viewer import DicFieldViewer
    xs = np.linspace(-1, 1, 6); ys = np.linspace(-0.5, 0.5, 4)
    gx, gy = np.meshgrid(xs, ys); x = gx.ravel(); y = gy.ravel()
    t = np.array([0.0, 1e-4, 2e-4])
    V = np.random.rand(3, x.size)
    vw = DicFieldViewer()
    vw.set_field(x, y, t, {"Vmag": V}, units={"Vmag": "mm/s"})
    vw._ax.set_xlim(-0.2, 0.3); vw._ax.set_ylim(-0.1, 0.1)
    before = vw._ax.get_xlim()
    vw.sld.setValue(2)                 # scrub frame
    assert vw._ax.get_xlim() == pytest.approx(before)   # zoom preserved
    vw.reset_view()
    assert vw._ax.get_xlim()[0] < -0.5     # home resets to full extent


def test_dic_field_viewer_set_and_profile(qapp):
    import numpy as np
    from gui.widgets.dic_field_viewer import DicFieldViewer
    # regular 3x4 grid
    xs = np.linspace(-1, 1, 4); ys = np.linspace(-0.5, 0.5, 3)
    gx, gy = np.meshgrid(xs, ys)
    x = gx.ravel(); y = gy.ravel(); n = x.size
    t = np.array([0.0, 1e-4])
    V1 = np.tile(x, (2, 1)) * 100.0          # ramp in x
    V2 = np.zeros((2, n)); Vmag = np.abs(V1)
    valid = np.ones((2, n), bool)
    vw = DicFieldViewer()
    vw.set_field(x, y, t, {"V1": V1, "V2": V2, "Vmag": Vmag}, valid=valid)
    assert vw._grid is not None                    # regular grid detected
    vw._line = [(-1.0, 0.0), (1.0, 0.0)]
    d, vals = vw.sample_profile(n=20)
    assert d is not None and len(d) == 20
    assert np.isfinite(vals).all()


def test_dic_mask_invalidates_background(qapp):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.dic_tab import DICTab
    # dark top half (background), textured bright bottom half (material)
    base = np.zeros((200, 260), np.uint8)
    base[120:, :] = cv2.GaussianBlur(
        np.random.default_rng(7).integers(0, 255, (80, 260)).astype(np.uint8),
        (0, 0), 1.2)
    frames = np.stack([base, _shift(base, 2, 0)])
    s = ExperimentSession(); s.visible.fps = 1000.0
    tab = DICTab(s); tab.set_frames(frames, fps=1000.0)
    tab.spin_scale.setValue(0.01)
    tab.set_roi((20, 20, 220, 160)); tab.sp_step.setValue(20); tab.sp_search.setValue(10)
    pts = tab._grid_for_run()
    # mask off -> keep all
    assert tab._keep_mask(pts).all()
    # mask on -> dark background points dropped, regular grid preserved.
    # Masking is intensity-only now (the texture criterion was removed).
    tab.chk_mask.setChecked(True); tab.sld_int.setValue(25)
    keep = tab._keep_mask(pts)
    assert 0 < keep.sum() < len(pts)
    tab.b_validate.setChecked(True)
    res = tab.run()
    assert res["grid"] is not None                 # grid stays regular
    assert int(res["valid"][0].sum()) <= int(keep.sum())
    assert res["meta"]["mask"]["min_intensity"] == 25 if "meta" in res else True


def test_dic_viewer_no_layout_drift_on_toggle(qapp):
    import numpy as np
    from gui.widgets.dic_field_viewer import DicFieldViewer
    xs = np.linspace(-1, 1, 6); ys = np.linspace(-0.5, 0.5, 4)
    gx, gy = np.meshgrid(xs, ys); x = gx.ravel(); y = gy.ravel()
    t = np.array([0.0, 1e-4]); V = np.random.rand(2, x.size)
    vw = DicFieldViewer()
    vw.set_field(x, y, t, {"Vmag": V}, units={"Vmag": "mm/s"})
    vw.set_background(lambda i: np.zeros((200, 260), np.uint8), 0.01, 260, 200)
    p1 = vw._ax.get_position().bounds
    for _ in range(4):
        vw.chk_bg.toggle()
    p2 = vw._ax.get_position().bounds
    assert np.allclose(p1, p2, atol=2e-3)          # main axes does not drift


def test_dic_score_and_param_lock(qapp):
    pytest.importorskip("cv2")
    import numpy as np
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.dic_tab import DICTab
    base = _speckle()
    frames = np.stack([base, _shift(base, 2, 0)])
    s = ExperimentSession(); s.visible.fps = 1000.0
    tab = DICTab(s); tab.set_frames(frames, fps=1000.0)
    tab.spin_scale.setValue(0.01)
    tab.set_roi((30, 30, 200, 140)); tab.sp_step.setValue(20); tab.sp_search.setValue(10)
    tab.b_validate.setChecked(True)
    # engine + mask parameters are frozen once validated (intensity-only mask)
    for w in (tab.sp_subset, tab.sp_step, tab.sp_search, tab.sp_zncc,
              tab.cb_engine, tab.chk_mask, tab.sld_int):
        assert w.isEnabled() is False
    res = tab.run()
    assert "ZNCC" in res["fields"]
    ok = res["valid"][0]
    assert np.nanmean(res["fields"]["ZNCC"][0][ok]) > 0.9   # good match
    assert res["units"]["ZNCC"] == "-"
    tab.b_validate.setChecked(False)
    assert tab.sp_subset.isEnabled() is True


def test_dic_quality_field_bypasses_valid_mask(qapp):
    import numpy as np
    from gui.widgets.dic_field_viewer import DicFieldViewer
    xs = np.linspace(-1, 1, 5); ys = np.linspace(-0.5, 0.5, 4)
    gx, gy = np.meshgrid(xs, ys); x = gx.ravel(); y = gy.ravel(); n = x.size
    t = np.array([0.0])
    valid = np.ones((1, n), bool); valid[0, : n // 2] = False
    comps = {"Vmag": np.ones((1, n)), "ZNCC": np.linspace(0, 1, n)[None, :]}
    vw = DicFieldViewer()
    vw.set_field(x, y, t, comps, valid=valid, units={"Vmag": "mm/s", "ZNCC": "-"})
    vw.cb_comp.setCurrentText("Vmag")
    Zv, _ = vw._Z(); n_v = np.isfinite(np.ma.filled(Zv, np.nan)).sum()
    vw.cb_comp.setCurrentText("ZNCC")
    Zq, _ = vw._Z(); n_q = np.isfinite(np.ma.filled(Zq, np.nan)).sum()
    assert n_v == valid.sum()        # velocity masked to valid points
    assert n_q == n                  # quality shown everywhere


def test_wheel_step_slider_increments_by_one(qapp):
    from PySide6.QtCore import QPointF, QPoint, Qt
    from PySide6.QtGui import QWheelEvent
    from gui.widgets.num_input import WheelStepSlider
    sld = WheelStepSlider(Qt.Orientation.Horizontal)
    sld.setRange(0, 255); sld.setSingleStep(1); sld.setValue(25)
    ev = QWheelEvent(QPointF(1, 1), QPointF(1, 1), QPoint(0, 0), QPoint(0, 120),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                     Qt.ScrollPhase.NoScrollPhase, False)
    sld.wheelEvent(ev)
    assert sld.value() == 26          # +1 per notch, not +3
    ev2 = QWheelEvent(QPointF(1, 1), QPointF(1, 1), QPoint(0, 0), QPoint(0, -120),
                      Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                      Qt.ScrollPhase.NoScrollPhase, False)
    sld.wheelEvent(ev2)
    assert sld.value() == 25
