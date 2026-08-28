# -*- coding: utf-8 -*-
"""
Integration test for the tool-polygon flow: draw a 4-point polygon in the
Alignment tab, label tip/rake/flank, store it in the session, and check the DIC
tab's global params pick it up and feed it to the engine.
"""
import numpy as np

from gui.core.experiment_session import ExperimentSession
from gui.tabs.alignment_tab import AlignmentTab
from gui.tabs.dic_tab import DICTab


class _Evt:
    def __init__(self, ax, x, y):
        self.inaxes = ax
        self.xdata = x
        self.ydata = y


def test_alignment_polygon_to_dic(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((200, 200), np.uint8)
    atab._w = 200
    atab._h = 200

    # Draw 4 polygon points.
    atab._mode = "poly"
    for (x, y) in [(50, 150), (150, 150), (150, 50), (60, 50)]:
        atab._on_press(_Evt(atab._ax, x, y))
    assert len(atab._poly_verts) == 4

    # Label rake/flank edges; the tip is derived from their shared corner.
    atab._poly_rake = 0
    atab._poly_flank = 3
    assert atab._poly_tip_index() == 0
    vals = atab.values()
    assert "tool_polygon_px" in vals
    assert len(vals["tool_polygon_px"]) == 4

    # Store in session (as _write would).
    sess.reference_geometry = dict(vals)

    # DIC tab picks up the polygon for the global engine.
    dtab = DICTab(sess)
    dtab.cb_engine.setCurrentIndex(dtab.cb_engine.findData("global"))
    p = dtab._global_params()
    assert p.tool_polygon is not None
    assert len(p.tool_polygon) == 4


def test_dic_no_polygon_when_absent(qapp):
    sess = ExperimentSession(name="t")
    dtab = DICTab(sess)
    dtab.cb_engine.setCurrentIndex(dtab.cb_engine.findData("global"))
    p = dtab._global_params()
    assert p.tool_polygon is None


def test_poly_clear(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((100, 100), np.uint8)
    atab._w = atab._h = 100
    atab._mode = "poly"
    for (x, y) in [(10, 10), (50, 10), (50, 50), (10, 50)]:
        atab._on_press(_Evt(atab._ax, x, y))
    atab._poly_rake = 0
    atab._poly_flank = 3
    atab._poly_clear()
    assert atab._poly_verts == []
    assert atab._poly_rake is None and atab._poly_flank is None
    assert "tool_polygon_px" not in atab.values()


def test_tip_derived_from_edges(qapp):
    """Tip is the shared corner of adjacent rake/flank edges (no manual Set
    tip). Opposite edges share no corner -> no derived tip."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((200, 200), np.uint8)
    atab._w = atab._h = 200
    atab._mode = "poly"
    for (x, y) in [(50, 150), (150, 150), (150, 50), (60, 50)]:
        atab._on_press(_Evt(atab._ax, x, y))
    atab._poly_rake = 0          # edge 0 = v0-v1
    atab._poly_flank = 3         # edge 3 = v3-v0  -> shared vertex 0
    assert atab._poly_tip_index() == 0
    atab._poly_flank = 2         # edge 2 = v2-v3  -> opposite to edge 0
    assert atab._poly_tip_index() is None


def test_draw_mode_activates_via_button(qapp):
    """Regression: clicking 'Draw 4-pt polygon' must set poly mode without an
    AttributeError on removed rough-draw buttons, and clicks must add vertices.
    """
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((200, 200), np.uint8)
    atab._w = atab._h = 200
    atab.b_poly.setChecked(True)              # used to raise AttributeError
    assert atab._mode == "poly"
    for (x, y) in [(50, 150), (150, 150), (150, 50), (60, 50)]:
        atab._on_press(_Evt(atab._ax, x, y))
    assert len(atab._poly_verts) == 4


def test_zoom_panel_is_integrated(qapp):
    """The magnifier lives in an integrated panel (its own canvas/axes in a
    dedicated right-hand column), not a floating dialog."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((200, 200), np.uint8)
    atab._w = atab._h = 200
    # Integrated zoom axes/canvas exist and are distinct from the main canvas.
    assert atab._zoom_ax is not None
    assert atab._zoom_canvas is not atab._canvas
    # No floating zoom dialog any more.
    assert not hasattr(atab, "_zoom_dialog") or atab.__dict__.get("_zoom_dialog") is None
    # Cursor motion updates the zoom without error.
    atab._on_motion(_Evt(atab._ax, 80, 80))
    assert atab._zoom_bubble == (80, 80)


def test_buttons_deactivate_after_action(qapp):
    """One-shot mode buttons auto-untoggle once their action completes."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((200, 200), np.uint8)
    atab._w = atab._h = 200
    atab.b_poly.setChecked(True)
    for (x, y) in [(110, 40), (160, 40), (160, 160), (105, 160)]:
        atab._on_press(_Evt(atab._ax, x, y))
    assert not atab.b_poly.isChecked()          # untoggled after 4th point
    assert atab._mode == "idle"
    atab.b_poly_rake.setChecked(True)
    atab._on_press(_Evt(atab._ax, 135, 40))
    assert not atab.b_poly_rake.isChecked()     # untoggled after labelling
    atab.b_poly_flank.setChecked(True)
    atab._on_press(_Evt(atab._ax, 110, 100))
    assert not atab.b_poly_flank.isChecked()


def test_extend_moves_fourth_vertex_to_corner(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((496, 592), np.uint8)
    atab._w, atab._h = 592, 496
    atab._poly_verts = [(300, 400), (400, 380), (420, 300), (280, 320)]
    atab._poly_rake = 0       # v0-v1, tip v0
    atab._poly_flank = 3      # v3-v0, tip v0  (shared vertex 0)
    atab._poly_extend_borders()
    corners = {(0.0, 0.0), (591.0, 0.0), (0.0, 495.0), (591.0, 495.0)}
    # v2 is the 4th vertex (not tip, not the two face far-ends) -> a corner.
    assert tuple(atab._poly_verts[2]) in corners


def test_binary_fit_locks_on_edge():
    """snap_line_to_edge with a binary threshold locks onto the tool/background
    boundary."""
    from gui.core.tool_detect import snap_line_to_edge
    img = np.full((200, 200), 10.0)
    img[:, :100] = 200.0
    q1, q2, pts = snap_line_to_edge(img, (95, 20), (95, 180), search=20,
                                    blur=1.0, binary_threshold=100)
    assert len(pts) > 0
    assert abs(pts[:, 0].mean() - 100.0) < 2.0


def test_fit_subrange_and_clip(qapp):
    """Fit uses only the [start,end] sub-range of each face and clips captured
    points to the image window."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    img = np.full((200, 200), 10.0)
    img[:, :100] = 200.0                  # sharp vertical edge at x=100
    atab._image = img.astype(np.uint8)
    atab._w = atab._h = 200
    atab._open_adjust_dialog("rake")
    atab._poly_verts = [(98, 30), (98, 170), (160, 170), (160, 30)]
    atab._poly_rake = 0
    atab._poly_flank = 3
    r = atab._adj["rake"]
    r["binary"].setChecked(True); r["binth"].setValue(100)
    r["search"].setValue(15)
    r["start"].setValue(20); r["end"].setValue(80)
    atab._fit_both_faces()
    pts = atab._rake_pts
    assert pts is not None and len(pts) > 0
    assert (pts[:, 0] >= 0).all() and (pts[:, 0] < 200).all()
    assert (pts[:, 1] >= 0).all() and (pts[:, 1] < 200).all()


def test_param_sliders_scroll_step_one(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._open_adjust_dialog("rake")
    atab._open_adjust_dialog("flank")
    for which in ("rake", "flank"):
        d = atab._adj[which]
        for key in ("search", "blur", "binth", "start", "end"):
            assert d[key].singleStep() == 1


def test_fit_lands_on_edge_not_band_border(qapp):
    """The fit must snap to the actual edge (centre of the gradient band), not
    to the border of the search window, even when the initial face is offset."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    img = np.full((200, 200), 10.0)
    img[:, :100] = 200.0                  # sharp vertical edge at x=100
    atab._image = img.astype(np.uint8)
    atab._w = atab._h = 200
    atab._open_adjust_dialog("rake")
    # Face deliberately offset to x=90; the true edge is at x=100.
    atab._poly_verts = [(90, 30), (90, 170), (160, 170), (160, 30)]
    atab._poly_rake = 0
    atab._poly_flank = 3
    r = atab._adj["rake"]
    r["binary"].setChecked(True); r["binth"].setValue(100)
    r["search"].setValue(20)
    atab._fit_both_faces()
    pts = atab._rake_pts
    assert pts is not None and len(pts) > 0
    assert abs(pts[:, 0].mean() - 100.0) < 2.0      # on the edge, not at x=70/110


def test_per_face_ranges_independent(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._open_adjust_dialog("rake")
    atab._open_adjust_dialog("flank")
    atab._adj["rake"]["start"].setValue(10); atab._adj["rake"]["end"].setValue(50)
    atab._adj["flank"]["start"].setValue(60); atab._adj["flank"]["end"].setValue(90)
    assert atab._fit_settings_range("rake") == (0.10, 0.50)
    assert atab._fit_settings_range("flank") == (0.60, 0.90)


def test_per_face_settings_independent(qapp):
    """Each Adjust dialog has its own search/smoothing/binary settings."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._open_adjust_dialog("rake")
    atab._open_adjust_dialog("flank")
    atab._adj["rake"]["search"].setValue(10)
    atab._adj["flank"]["search"].setValue(40)
    assert atab._face_settings("rake")[0] == 10
    assert atab._face_settings("flank")[0] == 40


def test_fit_recales_segment_onto_edge(qapp):
    """The fitted face must move ONTO the detected edge (both endpoints), not
    stay anchored above it parallel to the edge."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    img = np.full((200, 200), 10.0)
    img[:, :100] = 200.0                  # vertical edge at x=100
    atab._image = img.astype(np.uint8)
    atab._w = atab._h = 200
    atab._open_adjust_dialog("rake")
    # Rake face offset to x=85; true edge at x=100.
    atab._poly_verts = [(85, 30), (85, 170), (160, 170), (160, 30)]
    atab._poly_rake = 0
    atab._poly_flank = 3
    r = atab._adj["rake"]
    r["binary"].setChecked(True); r["binth"].setValue(100)
    r["search"].setValue(25)
    atab._fit_both_faces()
    # The rake far endpoint (v1) and the tip (v0) now sit on the edge (~100).
    assert abs(atab._poly_verts[0][0] - 100.0) < 2.0
    assert abs(atab._poly_verts[1][0] - 100.0) < 2.0


def test_axes_locked_to_image(qapp):
    """A wide search band reaching past the image borders must not rescale the
    main view."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((200, 200), np.uint8)
    atab._w = atab._h = 200
    atab._open_adjust_dialog("rake")
    atab._poly_verts = [(10, 10), (10, 190), (190, 190), (190, 10)]
    atab._poly_rake = 0
    atab._poly_flank = 3
    atab._adj["rake"]["show_fit"].setChecked(True)
    atab._adj["rake"]["search"].setValue(80)     # band extends past the borders
    atab._redraw()
    xlim = atab._ax.get_xlim()
    assert abs(xlim[0] - (-0.5)) < 1e-6 and abs(xlim[1] - 199.5) < 1e-6


def test_smoothing_max_increased(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._open_adjust_dialog("rake")
    assert atab._adj["rake"]["blur"].maximum() == 200    # 20.0 px after /10


def test_extend_disabled_until_faces_set(qapp):
    """'Extend faces to border' is disabled until both rake and flank are set."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((200, 200), np.uint8)
    atab._w = atab._h = 200
    atab._mode = "poly"
    for (x, y) in [(50, 150), (150, 150), (150, 50), (60, 50)]:
        atab._on_press(_Evt(atab._ax, x, y))
    assert not atab.b_poly_extend.isEnabled()       # no faces labelled yet
    atab._poly_rake = 0; atab._update_poly_label()
    assert not atab.b_poly_extend.isEnabled()       # only one face
    atab._poly_flank = 3; atab._update_poly_label()
    assert atab.b_poly_extend.isEnabled()           # both set -> enabled


def test_two_adjust_dialogs(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._open_adjust_dialog("rake")
    atab._open_adjust_dialog("flank")
    assert "rake" in atab._adjust_dialogs and "flank" in atab._adjust_dialogs
    assert atab._adjust_dialogs["rake"] is not atab._adjust_dialogs["flank"]


def test_single_adjust_open_at_a_time(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._open_adjust_dialog("rake")
    assert atab._adjust_dialogs["rake"].isVisible()
    atab._open_adjust_dialog("flank")
    # Opening flank hides rake.
    assert not atab._adjust_dialogs["rake"].isVisible()
    assert atab._adjust_dialogs["flank"].isVisible()


def test_overlay_only_when_dialog_open(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._image = np.zeros((200, 200), np.uint8)
    atab._w = atab._h = 200
    atab._open_adjust_dialog("rake")
    atab._adj["rake"]["show_fit"].setChecked(True)
    assert "rake" in atab._active_overlay_faces()       # open + checked
    atab._adjust_dialogs["rake"].hide()
    assert "rake" not in atab._active_overlay_faces()    # checked but closed


def test_adjust_has_no_fit_or_move_button(qapp):
    """Per-face Adjust dialogs only hold parameters now (no Fit / Move)."""
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    atab._open_adjust_dialog("rake")
    d = atab._adj["rake"]
    assert "fit" not in d and "move" not in d
    # The global fit button and the move (Adjust on row 1) live on the panel.
    assert hasattr(atab, "b_poly_fit") and hasattr(atab, "b_poly_move")


def test_fit_both_faces_global(qapp):
    sess = ExperimentSession(name="t")
    atab = AlignmentTab(sess)
    img = np.full((200, 200), 10.0)
    img[:, :100] = 200.0
    atab._image = img.astype(np.uint8)
    atab._w = atab._h = 200
    atab._open_adjust_dialog("rake")
    atab._poly_verts = [(98, 30), (98, 170), (160, 170), (160, 30)]
    atab._poly_rake = 0
    atab._poly_flank = 3
    atab._adj["rake"]["binary"].setChecked(True)
    atab._adj["rake"]["binth"].setValue(100)
    atab._adj["rake"]["search"].setValue(15)
    atab._fit_both_faces()
    assert atab._rake_pts is not None and atab._flank_pts is not None
