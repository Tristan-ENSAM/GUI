# -*- coding: utf-8 -*-
"""Regression tests for the Experimental Data foundation:
ExperimentSession round-trip, ImageSequence, force loader, AcquisitionTab."""
import numpy as np
import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# ExperimentSession
# ---------------------------------------------------------------------------
def test_session_roundtrip(tmp_path):
    from gui.core.experiment_session import ExperimentSession
    s = ExperimentSession(name="cu_v200", material="Cu",
                          cutting_speed_nominal=3333.0, notes="hi")
    s.visible.path = "/x/vis"; s.visible.fps = 20000.0
    s.ir.fps = 1000.0
    s.forces.fps = 50000.0; s.forces.col_t = 0; s.forces.col_fc = 1; s.forces.col_ff = 2
    s.reference_geometry = {"rake_angle": 5.0}
    p = s.save(tmp_path / "essai")
    assert p.suffix == ".json"
    s2 = ExperimentSession.load(p)
    assert s2.name == "cu_v200" and s2.material == "Cu"
    assert s2.cutting_speed_nominal == 3333.0
    assert s2.visible.fps == 20000.0 and s2.ir.fps == 1000.0
    assert s2.forces.col_t == 0 and s2.forces.col_fc == 1
    assert s2.reference_geometry["rake_angle"] == 5.0


def test_session_tolerates_partial_and_unknown_keys():
    from gui.core.experiment_session import ExperimentSession
    # Missing nested blocks -> defaults; unknown keys ignored.
    s = ExperimentSession.from_json_dict({
        "name": "x", "forces": {"fps": 12345.0}, "bogus": 1})
    assert s.name == "x"
    assert s.forces.fps == 12345.0
    assert s.visible.fps == 20000.0      # default kept
    assert not hasattr(s, "bogus")


def test_session_frame_time():
    from gui.core.experiment_session import ExperimentSession
    s = ExperimentSession(trigger_offset_s=0.1)
    s.visible.fps = 1000.0
    assert s.frame_time("visible", 50) == pytest.approx(0.1 + 0.05)


# ---------------------------------------------------------------------------
# ImageSequence
# ---------------------------------------------------------------------------
def test_image_sequence_array():
    from gui.core.sequence_io import ImageSequence
    stack = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    seq = ImageSequence.from_array(stack, fps=200.0, t0=0.01)
    assert seq.n_frames == 3
    assert seq.frame(1).shape == (4, 5)
    assert np.allclose(seq.frame(1), stack[1])
    assert seq.time(2) == pytest.approx(0.01 + 2 / 200.0)
    # clamping
    assert np.allclose(seq.frame(99), stack[2])


def test_image_sequence_folder_standard_formats(tmp_path):
    import imageio.v3 as iio
    from gui.core.sequence_io import ImageSequence
    for i in range(4):
        iio.imwrite(tmp_path / ("frame_%03d.png" % i),
                    (np.ones((8, 10)) * i * 30).astype("uint8"))
    (tmp_path / "notes.txt").write_text("ignore me")   # ignored
    seq = ImageSequence.from_path(tmp_path, fps=500.0)
    assert seq.n_frames == 4
    assert seq.frame(0).shape == (8, 10)
    assert seq.frame(3).mean() > seq.frame(0).mean()   # sorted order
    seq1 = ImageSequence.from_path(tmp_path / "frame_000.png")
    assert seq1.n_frames == 1


def test_image_sequence_folder_jpeg(tmp_path):
    import imageio.v3 as iio
    from gui.core.sequence_io import ImageSequence
    for i in range(3):
        iio.imwrite(tmp_path / ("f%02d.jpg" % i),
                    (np.ones((8, 10)) * 100).astype("uint8"))
    seq = ImageSequence.from_path(tmp_path, fps=100.0)
    assert seq.n_frames == 3


def test_image_sequence_single_frame_and_npz(tmp_path):
    from gui.core.sequence_io import ImageSequence
    seq = ImageSequence.from_array(np.zeros((6, 7)))      # 2-D -> 1 frame
    assert seq.n_frames == 1 and seq.frame(0).shape == (6, 7)
    npz = tmp_path / "stack.npz"
    np.savez(npz, frames=np.ones((4, 2, 2)))
    seq2 = ImageSequence.from_path(npz, fps=10.0)
    assert seq2.n_frames == 4 and seq2.frame(0).shape == (2, 2)


# ---------------------------------------------------------------------------
# force loader
# ---------------------------------------------------------------------------
def test_load_forces_with_time_column(tmp_path):
    from gui.core.sequence_io import load_forces
    f = tmp_path / "f.csv"
    arr = np.column_stack([np.linspace(0, 1, 10),
                           np.arange(10.0), np.arange(10.0) * -2])
    np.savetxt(f, arr, delimiter=",")
    t, fc, ff = load_forces(f, col_t=0, col_fc=1, col_ff=2)
    assert t.shape == (10,) and t[-1] == pytest.approx(1.0)
    assert fc[3] == pytest.approx(3.0) and ff[3] == pytest.approx(-6.0)


def test_load_forces_no_time_column(tmp_path):
    from gui.core.sequence_io import load_forces
    f = tmp_path / "f.txt"
    np.savetxt(f, np.column_stack([np.arange(5.0), np.arange(5.0)]))
    t, fc, ff = load_forces(f, fps=100.0, col_t=-1, col_fc=0, col_ff=1)
    assert t[1] == pytest.approx(1 / 100.0)   # derived from fps
    assert fc[4] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# AcquisitionTab (UI)
# ---------------------------------------------------------------------------
def test_acquisition_tab_pull_and_apply(qapp):
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.experimental_data_tab import AcquisitionTab
    s = ExperimentSession()
    tab = AcquisitionTab(s)
    # edit via widgets -> pull into session
    tab.fld_name.setText("essai_42")
    tab.spin_speed.setValue(1500.0)
    tab._visible_fps.setValue(25000.0)
    tab._col_t.setValue(0)
    tab._pull()
    assert s.name == "essai_42"
    assert s.cutting_speed_nominal == 1500.0
    assert s.visible.fps == 25000.0
    assert s.forces.col_t == 0
    # apply back from a freshly loaded session
    s.ir.fps = 1234.0
    tab.apply_from_session()
    assert tab._ir_fps.value() == pytest.approx(1234.0)


def test_experimental_container_has_subtabs(qapp):
    from gui.tabs.experimental_data_tab import ExperimentalDataTab
    t = ExperimentalDataTab()
    assert t.tabs.count() == 8
    assert t.tabs.tabText(0) == "Acquisition / Import"


def test_image_sequence_dir_natural_sort(tmp_path):
    """A folder of PNG frames must load in natural (capture) order, not
    lexicographic order, and support standard formats."""
    import imageio.v3 as iio
    # Non zero-padded names: lexicographic would put 10,11,12 before 2.
    for n in range(1, 13):
        img = np.full((4, 4), n, dtype=np.uint8)
        iio.imwrite(tmp_path / ("frame%d.png" % n), img)
    from gui.core.sequence_io import ImageSequence
    seq = ImageSequence.from_path(tmp_path, fps=100.0)
    assert seq.n_frames == 12
    # frame(0) is "frame1" (value 1), frame(11) is "frame12" (value 12)
    assert int(round(float(np.mean(seq.frame(0))))) == 1
    assert int(round(float(np.mean(seq.frame(11))))) == 12


# ---------------------------------------------------------------------------
# Visible calibration
# ---------------------------------------------------------------------------
def test_target_object_points():
    from gui.core.calibration import TargetSpec
    t = TargetSpec("checkerboard", cols=4, rows=3, spacing_mm=2.0)
    objp = t.object_points()
    assert objp.shape == (12, 3)
    assert objp[:, 2].max() == 0.0                 # planar target
    # spacing respected: distance between first two columns = 2 mm
    assert abs(objp[1, 0] - objp[0, 0]) == 2.0
    # asymmetric grid uses a staggered layout
    ta = TargetSpec("circles_asym", cols=4, rows=3, spacing_mm=1.0)
    pa = ta.object_points()
    assert pa.shape == (12, 3)
    assert pa[4, 0] == 1.0      # row 1 (odd) is offset by spacing in x


def test_scale_mm_per_px_pure():
    from gui.core.calibration import _scale_mm_per_px
    px = np.array([[0., 0.], [10., 0.], [20., 0.]])    # 10 px steps
    mm = np.array([[0., 0.], [2., 0.], [4., 0.]])      # 2 mm steps
    assert _scale_mm_per_px(px, mm) == pytest.approx(0.2)


def _render_checkerboard(cols, rows, sq=40, border=2):
    nx, ny = cols + 1, rows + 1
    bd = np.zeros((ny * sq, nx * sq), np.uint8)
    for r in range(ny):
        for c in range(nx):
            if (r + c) % 2 == 0:
                bd[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
    H, W = bd.shape[0] + 2 * border * sq, bd.shape[1] + 2 * border * sq
    im = np.full((H, W), 255, np.uint8)
    im[border * sq:border * sq + bd.shape[0],
       border * sq:border * sq + bd.shape[1]] = bd
    return im


def test_opencv_calibration_end_to_end():
    cv2 = pytest.importorskip("cv2")
    from gui.core.calibration import OpenCVCalibrator, TargetSpec
    base = _render_checkerboard(9, 6, sq=40)
    H, W = base.shape
    rng = np.random.default_rng(2)
    views = []
    for _ in range(8):
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = src + rng.uniform(-18, 18, (4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        views.append(cv2.warpPerspective(base, M, (W, H), borderValue=255))
    cal = OpenCVCalibrator()
    tgt = TargetSpec("checkerboard", 9, 6, spacing_mm=2.0)
    n_det = sum(cal.detect(v, tgt) is not None for v in views)
    if n_det < 3:
        pytest.skip("synthetic boards not detected in this environment")
    res = cal.calibrate(views, tgt, reference_index=0)
    # 40 px squares, 2 mm spacing -> ~0.05 mm/px
    assert res.scale_mm_per_px == pytest.approx(0.05, rel=0.1)
    assert np.isfinite(res.reproj_rms_px) and res.reproj_rms_px < 2.0
    assert np.asarray(res.camera_matrix).shape == (3, 3)
    assert len(res.dist_coeffs) >= 5


def test_calibration_visible_tab(qapp, monkeypatch):
    from gui.core.experiment_session import ExperimentSession
    from gui.core.calibration import CalibrationResult
    from gui.tabs.calibration_visible_tab import CalibrationVisibleTab
    import gui.tabs.calibration_visible_tab as mod
    s = ExperimentSession()
    tab = CalibrationVisibleTab(s)
    # target spec reflects the widgets
    tab.spin_cols.setValue(7); tab.spin_rows.setValue(5); tab.spin_spacing.setValue(1.5)
    t = tab._target()
    assert (t.cols, t.rows, t.spacing_mm) == (7, 5, 1.5)
    # save stores into the session (suppress the modal dialog)
    monkeypatch.setattr(mod.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    tab._result = CalibrationResult(
        camera_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dist_coeffs=[0, 0, 0, 0, 0], scale_mm_per_px=0.05,
        homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], reproj_rms_px=0.4,
        image_size=[640, 480], reference_index=0, n_views_used=5,
        target=t.to_dict(), images=["a", "b"])
    tab._paths = ["a", "b"]
    tab._save()
    assert s.visible_calibration["scale_mm_per_px"] == 0.05
    assert s.visible_calibration["target"]["cols"] == 7
    # a fresh tab restores it from the session without recomputing
    tab2 = CalibrationVisibleTab(s)
    assert tab2.spin_cols.value() == 7
    assert tab2._result is not None


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def test_alignment_pixel_to_model_and_angles():
    from gui.core.alignment import (pixel_to_model, line_tilt_from_vertical_deg,
                                    line_tilt_from_horizontal_deg)
    # origin at centre, x right, y up
    assert pixel_to_model(100, 50, 200, 100, 0.05) == pytest.approx((0.0, 0.0))
    assert pixel_to_model(120, 50, 200, 100, 0.05) == pytest.approx((1.0, 0.0))
    assert pixel_to_model(100, 30, 200, 100, 0.05) == pytest.approx((0.0, 1.0))
    # vertical line -> 0 deg from vertical; horizontal -> 90 deg from vertical
    assert line_tilt_from_vertical_deg((10, 10), (10, 80)) == pytest.approx(0.0)
    assert abs(line_tilt_from_vertical_deg((0, 50), (50, 50))) == pytest.approx(90.0)
    # from horizontal: horizontal -> 0; a line rising towards +x -> positive
    assert line_tilt_from_horizontal_deg((0, 50), (50, 50)) == pytest.approx(0.0)
    # points (0,100)->(100,90) in image: goes right and UP (y decreases) -> +
    assert line_tilt_from_horizontal_deg((0, 100), (100, 90)) > 0
    # a ~12 deg flank (image dy=-21 over dx=100) -> ~+11.8 deg
    assert line_tilt_from_horizontal_deg((0, 0), (100, -21)) == pytest.approx(11.86, abs=0.1)


def test_alignment_tab_compute_and_write(qapp):
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.alignment_tab import AlignmentTab
    import gui.tabs.alignment_tab as mod
    captured = {}
    s = ExperimentSession()
    tab = AlignmentTab(s, write_geometry=lambda v: captured.update(v))
    tab.set_image_array(np.zeros((100, 200, 3), np.uint8), source="t")
    tab.spin_scale.setValue(0.05)
    # Tool reference is a 4-pt polygon now. Build a quad whose rake edge is the
    # vertical segment at x=100 and whose flank edge is the horizontal segment
    # at y=10; their shared corner (the tip) is (100, 10) -> model (0, 2) mm.
    # rake edge = e0 (v0->v1), flank edge = e3 (v3->v0), shared vertex v0.
    tab._poly_verts = [(100, 10), (100, 90), (180, 90), (180, 10)]
    tab._poly_rake = 0
    tab._poly_flank = 3
    tab.set_wp_point(100, 70)                    # (50-70)*0.05 = -1 mm
    tab._recompute()
    v = tab.values()
    assert v["rake_angle"] == pytest.approx(0.0, abs=1e-6)   # vertical rake
    assert v["clear_angle"] == pytest.approx(0.0, abs=1e-6)  # horizontal flank
    assert v["tool_x0"] == pytest.approx(0.0)
    assert v["tool_y0"] == pytest.approx(2.0)
    assert v["wp_y0"] == pytest.approx(-1.0)
    assert "tool_polygon_px" in v                # polygon exported for the mask
    mod.QMessageBox.information = staticmethod(lambda *a, **k: None)
    tab._write()
    assert "tool_x0" in captured
    assert s.reference_geometry["wp_y0"] == pytest.approx(-1.0)


def test_tool_snap_line_to_edge():
    pytest.importorskip("cv2")
    from gui.core.tool_detect import snap_line_to_edge
    # vertical step edge at x=100
    img = np.zeros((200, 200), np.uint8); img[:, 100:] = 200
    q1, q2, pts = snap_line_to_edge(img, (92, 10), (96, 190), search=20, blur=1.5)
    assert len(pts) >= 2
    assert (q1[0] + q2[0]) / 2 == pytest.approx(100.0, abs=1.5)
    # horizontal step edge at y=60
    img2 = np.zeros((200, 200), np.uint8); img2[60:, :] = 200
    h1, h2, _ = snap_line_to_edge(img2, (10, 55), (190, 52), search=20, blur=1.5)
    assert (h1[1] + h2[1]) / 2 == pytest.approx(60.0, abs=1.5)


def test_alignment_fit_edges_tab(qapp):
    pytest.importorskip("cv2")
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.alignment_tab import AlignmentTab
    img = np.zeros((200, 200), np.uint8)
    img[:, 100:] = 180          # vertical edge x=100 (rake)
    img[140:, :] = 180          # horizontal edge y=140 (flank)
    tab = AlignmentTab(ExperimentSession())
    tab.set_image_array(img, source="t")
    # Rough quad, slightly off the true edges. rake = e0 (vertical, x~98),
    # flank = e3 (horizontal, y~140). The flank far end stays LEFT of the
    # L-corner so the horizontal edge is unambiguous there.
    tab._poly_verts = [(102, 138), (98, 40), (10, 40), (10, 142)]
    tab._poly_rake = 0
    tab._poly_flank = 3
    for which in ("rake", "flank"):
        tab._open_adjust_dialog(which)
        d = tab._adj[which]
        d["binary"].setChecked(True); d["binth"].setValue(90)
        d["search"].setValue(20)
    # Exclude the corner side of the flank, and the very ends of the rake.
    tab._adj["flank"]["start"].setValue(15); tab._adj["flank"]["end"].setValue(95)
    tab._adj["rake"]["start"].setValue(5); tab._adj["rake"]["end"].setValue(90)
    tab._fit_both_faces()
    verts = tab._poly_verts
    rake_mid_x = (verts[0][0] + verts[1][0]) / 2
    assert abs(rake_mid_x - 100) < 2.0          # rake snapped to x=100
    tip = tab._poly_tool().tip_point()
    assert tip is not None
    assert tip[0] == pytest.approx(100.0, abs=2.5)
    assert tip[1] == pytest.approx(140.0, abs=2.5)


def test_tool_detection_quad_and_tip():
    cv2 = pytest.importorskip("cv2")
    from gui.core.tool_detect import (detect_tool_quad, segment_intersection,
                                       nearest_edge, nearest_corner)
    img = np.full((300, 400), 240, np.uint8)
    quad_true = np.array([[120, 80], [260, 90], [250, 230], [110, 210]], np.int32)
    cv2.fillPoly(img, [quad_true], 30)
    det = detect_tool_quad(img, roi=(90, 60, 220, 200), threshold=128, invert=True)
    assert det is not None and det.shape == (4, 2)
    # every true corner is matched within a couple of pixels
    err = max(np.min(np.hypot(det[:, 0] - p[0], det[:, 1] - p[1]))
              for p in quad_true)
    assert err < 3.0
    # pure geometry helpers
    assert segment_intersection((0, 0), (10, 10), (0, 10), (10, 0)) == pytest.approx((5.0, 5.0))
    assert nearest_corner([[0, 0], [10, 0], [10, 10], [0, 10]], 9, 1) == 1
    assert nearest_edge([[0, 0], [10, 0], [10, 10], [0, 10]], 5, 0.1) == 0


def test_write_reference_geometry_updates_model(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    w._write_reference_geometry({
        "rake_angle": 7.0, "clear_angle": 3.0, "tool_x0": 1.5,
        "tool_y0": -0.5, "wp_x0": 0.2, "wp_y0": 0.0})
    assert w.cfg.tool_geometry.rake_angle == 7.0
    assert w.cfg.tool_geometry.clear_angle == 3.0
    assert w.cfg.tool_position.x0 == 1.5
    assert w.cfg.wp_position.x0 == 0.2
    assert w._dirty is True
    assert w.geometry_tab.f_rake.value() == pytest.approx(7.0)


def test_alignment_scale_unit_and_clear(qapp):
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.alignment_tab import AlignmentTab
    tab = AlignmentTab(ExperimentSession())
    tab.set_image_array(np.zeros((100, 200, 3), np.uint8), source="t")
    # px/mm unit -> mm_per_px is the reciprocal
    tab.cb_scale_unit.setCurrentText("px/mm")
    tab.spin_scale.setValue(20.0)
    assert tab._mm_per_px() == pytest.approx(0.05)
    tab.cb_scale_unit.setCurrentText("mm/px")
    tab.spin_scale.setValue(0.04)
    assert tab._mm_per_px() == pytest.approx(0.04)
    # A workpiece point selection is cleared by _clear_selections, while the
    # image and the scale are kept. (The polygon has its own Clear button.)
    tab._poly_verts = [(10, 10), (10, 80), (90, 80), (90, 10)]
    tab._poly_rake = 0
    tab._poly_flank = 3
    tab.set_wp_point(50, 50)
    assert tab._wp_pt is not None
    tab._clear_selections()
    assert tab._wp_pt is None                     # selection cleared
    assert tab._image is not None                 # image kept
    assert tab.spin_scale.value() == pytest.approx(0.04)   # scale kept
    # The dedicated polygon clear empties the polygon.
    tab._poly_clear()
    assert tab._poly_verts == [] or len(tab._poly_verts) == 0
    assert tab._poly_rake is None and tab._poly_flank is None


def test_geometry_reference_overlay(qapp):
    from gui.main import MainWindow
    import gui.tabs.alignment_tab as amod
    amod.QMessageBox.information = staticmethod(lambda *a, **k: None)
    w = MainWindow(); w._dirty = False
    al = w.experimental_tab.alignment_tab
    al.set_image_array(np.zeros((100, 120), np.uint8), source="f")
    al.spin_scale.setValue(0.02)                 # mm/px
    # Polygon: rake vertical at x=60, flank horizontal at y=80, tip (60,80).
    # rake = e0 (v0->v1): (60,80)->(60,10) vertical;
    # flank = e3 (v3->v0): (110,80)->(60,80) horizontal; shared corner v0.
    al._poly_verts = [(60, 80), (60, 10), (110, 10), (110, 80)]
    al._poly_rake = 0
    al._poly_flank = 3
    al._recompute()
    al._write()
    gp = w.geometry_tab.preview
    assert gp.has_reference_overlay()
    assert w.geometry_tab.chk_ref.isEnabled()
    assert gp._ref_mm_per_px == pytest.approx(0.02)
    # toggling visibility redraws without error
    w.geometry_tab.chk_ref.setChecked(False)
    w.geometry_tab.chk_ref.setChecked(True)
    assert gp._ref_show is True
