# -*- coding: utf-8 -*-
"""
Alignment — reference geometry from an image, written into the Numerical Model.

Tool geometry is SEMI-AUTOMATIC (robust on low-contrast / textured images):
  1. draw a rough line over the RAKE face (2 clicks) and over the FLANK face
     (2 clicks) — only approximate placement is needed,
  2. "Fit edges" snaps each rough line onto the strongest local image edge
     (perpendicular gradient search + robust line fit; serration averages out),
  3. the intersection of the two fitted lines is the TOOL TIP.
Endpoints can be dragged (with a magnifier) and re-fitted; the workpiece
reference is a manual point.

rake_angle / clear_angle = tilt of the two fitted faces from the vertical;
tool (x0,y0) = the tip. Frame: origin at image centre, x right (cutting
direction, tip towards +x), y up. Scale mm/px from the visible calibration or
a manual override (unit mm/px or px/mm). "Write to Numerical Model" pushes the
values into cfg.Geometry.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QDoubleSpinBox, QSlider, QSplitter,
    QFileDialog, QMessageBox, QAbstractSpinBox, QCheckBox, QDialog,
    QDialogButtonBox,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from gui.core.experiment_session import ExperimentSession
from gui.core.sequence_io import _read_image
from gui.core.alignment import (pixel_to_model, line_tilt_from_vertical_deg,
                                 line_tilt_from_horizontal_deg)
from gui.core import tool_detect as td
from gui.core.logging_util import log_swallowed


def _line_intersection(p1, u1, p2, u2):
    """Intersection of two lines given as point + unit direction.
    Returns the (x, y) point, or None if the lines are (near-)parallel."""
    p1 = np.asarray(p1, float); u1 = np.asarray(u1, float)
    p2 = np.asarray(p2, float); u2 = np.asarray(u2, float)
    # Solve p1 + t*u1 = p2 + s*u2  ->  [u1, -u2] [t, s]^T = p2 - p1.
    A = np.array([[u1[0], -u2[0]], [u1[1], -u2[1]]])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    rhs = p2 - p1
    t = (rhs[0] * A[1, 1] - rhs[1] * A[0, 1]) / det
    return p1 + t * u1


class AlignmentTab(QWidget):
    def __init__(self, session: ExperimentSession, write_geometry=None,
                 parent=None):
        super().__init__(parent)
        self.session = session
        self._write_geometry = write_geometry
        self._image = None
        self._w = 0
        self._h = 0
        self._rake_rough: list = []      # [p1, p2] rough rake line
        self._flank_rough: list = []     # [p1, p2] rough flank line
        self._rake_fit = None            # (q1, q2) fitted
        self._flank_fit = None
        self._rake_pts = None            # snapped edge points (for display)
        self._flank_pts = None
        self._wp_pt = None
        self._mode = "idle"
        self._drag = None                # ("rake"/"flank", index 0/1)
        self._magnifier = None
        # Tool polygon (DIC mask): 4 clicked vertices + semantic labels.
        self._poly_verts: list = []      # up to 4 (x, y) pixel vertices
        self._poly_tip = None            # (unused: tip is derived from edges)
        self._poly_rake = None           # edge index labelled as rake face
        self._poly_flank = None          # edge index labelled as flank face
        self._rake_pts = None            # fitted edge sample points (Adjust)
        self._flank_pts = None
        self._show_fit_overlay = False   # visualise edge-search / smoothing
        self._zoom_bubble = None         # (x, y) cursor position for the bubble
        self._zoom_radius = 30           # half-size (px) of the zoom view
        self._zoom_ax = None             # set by _build_zoom_panel
        # Per-face Adjust settings (independent rake / flank), filled by
        # _build_adjust_dialog; one dialog window per face.
        self._adj = {"rake": {}, "flank": {}}
        self._adjust_dialogs = {}

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.addWidget(self._build_image_group())
        ll.addWidget(self._build_scale_group())
        ll.addWidget(self._build_tool_polygon_group())
        ll.addWidget(self._build_wp_group())
        ll.addWidget(self._build_result_group())
        ll.addStretch()
        left.setMinimumWidth(380)
        splitter.addWidget(left)

        self._fig = Figure(tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_axis_off()
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        splitter.addWidget(self._canvas)

        # Dedicated zoom column on the right (constant slot, magnifier on top).
        splitter.addWidget(self._build_zoom_panel())

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([420, 760, 260])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)
        self._prefill_scale_from_calibration()

    # =====================================================================
    # Builders
    # =====================================================================
    def _build_image_group(self):
        g = QGroupBox("Reference image")
        v = QVBoxLayout(g)
        b = QPushButton("Choose image…")
        b.clicked.connect(self._choose_image)
        v.addWidget(b)
        self.lbl_image = QLabel("(no image)")
        self.lbl_image.setStyleSheet("color:#666;")
        self.lbl_image.setWordWrap(True)
        v.addWidget(self.lbl_image)
        return g

    def _build_zoom_panel(self):
        """Constant zoom column on the right: a small magnifier canvas at the
        top that mirrors the area under the cursor (radius set by scroll)."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 8, 8, 8)
        v.addWidget(QLabel("Zoom"))
        self._zoom_fig = Figure(figsize=(2.4, 2.4))
        self._zoom_canvas = FigureCanvas(self._zoom_fig)
        self._zoom_ax = self._zoom_fig.add_axes([0, 0, 1, 1])
        self._zoom_ax.set_xticks([]); self._zoom_ax.set_yticks([])
        self._zoom_canvas.setMinimumSize(240, 240)
        self._zoom_canvas.setMaximumHeight(260)
        v.addWidget(self._zoom_canvas)
        self.lbl_zoom = QLabel("move cursor over the image \u00b7 scroll to zoom")
        self.lbl_zoom.setStyleSheet("color:#666;")
        self.lbl_zoom.setWordWrap(True)
        v.addWidget(self.lbl_zoom)
        v.addStretch(1)
        w.setMinimumWidth(240)
        return w

    def _build_scale_group(self):
        g = QGroupBox("Scale")
        form = QFormLayout(g)
        row = QHBoxLayout()
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(1e-9, 1e9); self.spin_scale.setDecimals(6)
        self.spin_scale.setValue(1.0)
        self.spin_scale.valueChanged.connect(self._recompute)
        self.cb_scale_unit = QComboBox()
        self.cb_scale_unit.addItems(["mm/px", "px/mm"])
        self.cb_scale_unit.currentIndexChanged.connect(self._recompute)
        row.addWidget(self.spin_scale, 1)
        row.addWidget(self.cb_scale_unit)
        w = QWidget(); w.setLayout(row)
        form.addRow("Resolution:", w)
        b = QPushButton("From visible calibration")
        b.clicked.connect(self._prefill_scale_from_calibration)
        form.addRow(b)
        return g

    def _mm_per_px(self) -> float:
        v = float(self.spin_scale.value())
        if self.cb_scale_unit.currentText() == "px/mm":
            return 1.0 / v if v != 0 else float("nan")
        return v


    def _build_tool_polygon_group(self):
        g = QGroupBox("Tool reference")
        v = QVBoxLayout(g)
        # Row 1: draw + adjust-vertices (move endpoint) + extend.
        row1 = QHBoxLayout()
        self.b_poly = QPushButton("Draw Tool"); self.b_poly.setCheckable(True)
        self.b_poly.toggled.connect(
            lambda on: self._set_mode("poly", on, self.b_poly))
        self.b_poly_move = QPushButton("Adjust\u2026"); self.b_poly_move.setCheckable(True)
        self.b_poly_move.setToolTip("Move polygon vertices.")
        self.b_poly_move.toggled.connect(
            lambda on: self._set_mode("poly_move", on, self.b_poly_move))
        self.b_poly_extend = QPushButton("Extend faces to border")
        self.b_poly_extend.clicked.connect(self._poly_extend_borders)
        self.b_poly_extend.setEnabled(False)
        self.b_poly_extend.setToolTip("Set the rake and flank edges first.")
        row1.addWidget(self.b_poly, 1); row1.addWidget(self.b_poly_move, 1)
        row1.addWidget(self.b_poly_extend, 1)
        v.addLayout(row1)
        # Row 2: label rake + adjust rake | label flank + adjust flank.
        row2 = QHBoxLayout()
        self.b_poly_rake = QPushButton("Set rake edge"); self.b_poly_rake.setCheckable(True)
        self.b_poly_rake.toggled.connect(
            lambda on: self._set_mode("poly_rake", on, self.b_poly_rake))
        self.b_adjust_rake = QPushButton("Adjust\u2026")
        self.b_adjust_rake.clicked.connect(lambda: self._open_adjust_dialog("rake"))
        self.b_poly_flank = QPushButton("Set flank edge"); self.b_poly_flank.setCheckable(True)
        self.b_poly_flank.toggled.connect(
            lambda on: self._set_mode("poly_flank", on, self.b_poly_flank))
        self.b_adjust_flank = QPushButton("Adjust\u2026")
        self.b_adjust_flank.clicked.connect(lambda: self._open_adjust_dialog("flank"))
        row2.addWidget(self.b_poly_rake, 1); row2.addWidget(self.b_adjust_rake, 1)
        row2.addWidget(self.b_poly_flank, 1); row2.addWidget(self.b_adjust_flank, 1)
        v.addLayout(row2)
        # Row 3: global fit (both faces, using each face's Adjust settings).
        self.b_poly_fit = QPushButton("Fit edges")
        self.b_poly_fit.clicked.connect(self._fit_both_faces)
        v.addWidget(self.b_poly_fit)
        # Row 4: clear.
        self.b_poly_clear = QPushButton("Clear polygon")
        self.b_poly_clear.clicked.connect(self._poly_clear)
        v.addWidget(self.b_poly_clear)
        self.lbl_poly = QLabel("Draw 4 points around the tool, then label the "
                               "rake and flank edges (the tip is their shared "
                               "corner).")
        self.lbl_poly.setStyleSheet("color:#666;"); self.lbl_poly.setWordWrap(True)
        v.addWidget(self.lbl_poly)
        return g

    def _update_extend_enabled(self):
        """Enable 'Extend faces to border' only when both faces are labelled."""
        if hasattr(self, "b_poly_extend"):
            self.b_poly_extend.setEnabled(
                self._poly_rake is not None and self._poly_flank is not None
                and len(self._poly_verts) == 4)

    def _poly_clear(self):
        self._poly_verts = []
        self._poly_tip = self._poly_rake = self._poly_flank = None
        self._redraw(); self._update_poly_label()

    def _poly_extend_borders(self):
        from gui.core.tool_polygon import ToolPolygon
        if len(self._poly_verts) != 4 or self._poly_rake is None \
                or self._poly_flank is None:
            self.lbl_poly.setText("Need 4 points and rake+flank edges labelled "
                                  "before extending.")
            return
        tp = ToolPolygon(vertices=list(self._poly_verts),
                         rake_edge=self._poly_rake, flank_edge=self._poly_flank)
        if tp.derived_tip_index() is None:
            self.lbl_poly.setText("Rake and flank edges must be ADJACENT "
                                  "(share the tip corner). Re-label them.")
            return
        tp.extend_faces_to_border(self._w, self._h)
        self._poly_verts = list(tp.vertices)
        self._redraw(); self._recompute(); self._update_poly_label()

    def _open_adjust_dialog(self, which="rake"):
        """Open the per-face Adjust dialog ('rake' or 'flank'). Only one Adjust
        dialog may be open at a time, so any other one is hidden first."""
        for w, dlg in self._adjust_dialogs.items():
            if w != which and dlg is not None:
                dlg.hide()
        dlg = self._adjust_dialogs.get(which)
        if dlg is None:
            dlg = self._build_adjust_dialog(which)
            self._adjust_dialogs[which] = dlg
        dlg.show(); dlg.raise_(); dlg.activateWindow()
        self._redraw()                    # overlay visibility depends on this

    def _build_adjust_dialog(self, which):
        """Build the Adjust dialog for a single face ('rake' or 'flank'). All
        settings (edge search, smoothing, binary threshold, fit range) are
        independent per face and stored in self._adj[which]. The Fit button
        fits only this face."""
        title = "Adjust %s face" % which
        col = "orange" if which == "rake" else "magenta"
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        v = QVBoxLayout(dlg)
        d = self._adj[which]

        srch = QHBoxLayout()
        d["search"] = QSlider(Qt.Orientation.Horizontal)
        d["search"].setRange(3, 80); d["search"].setValue(25)
        d["search"].setSingleStep(1); d["search"].setPageStep(1)
        d["lbl_search"] = QLabel("25 px")
        d["search"].valueChanged.connect(
            lambda val, dd=d: (dd["lbl_search"].setText("%d px" % val),
                               self._redraw()))
        srch.addWidget(QLabel("Edge search:")); srch.addWidget(d["search"], 1)
        srch.addWidget(d["lbl_search"])
        v.addLayout(srch)

        smo = QHBoxLayout()
        d["blur"] = QSlider(Qt.Orientation.Horizontal)
        d["blur"].setRange(0, 200); d["blur"].setValue(20)
        d["blur"].setSingleStep(1); d["blur"].setPageStep(1)
        d["lbl_blur"] = QLabel("2.0 px")
        d["blur"].valueChanged.connect(
            lambda val, dd=d: (dd["lbl_blur"].setText("%.1f px" % (val / 10.0)),
                               self._redraw()))
        smo.addWidget(QLabel("Smoothing:")); smo.addWidget(d["blur"], 1)
        smo.addWidget(d["lbl_blur"])
        v.addLayout(smo)

        binr = QHBoxLayout()
        d["binary"] = QCheckBox("Binary")
        d["binary"].setToolTip("Binarise the image (smoothed gray >= threshold) "
                               "before fitting, to lock onto the tool/background "
                               "boundary rather than speckle.")
        d["binary"].toggled.connect(lambda *_: self._redraw())
        d["binth"] = QSlider(Qt.Orientation.Horizontal)
        d["binth"].setRange(0, 255); d["binth"].setValue(60)
        d["binth"].setSingleStep(1); d["binth"].setPageStep(1)
        d["lbl_binth"] = QLabel("60")
        d["binth"].valueChanged.connect(
            lambda val, dd=d: (dd["lbl_binth"].setText("%d" % val), self._redraw()))
        binr.addWidget(d["binary"])
        binr.addWidget(QLabel("Threshold:")); binr.addWidget(d["binth"], 1)
        binr.addWidget(d["lbl_binth"])
        v.addLayout(binr)

        d["show_fit"] = QCheckBox("Visualise edge search / smoothing")
        d["show_fit"].toggled.connect(self._on_show_fit_toggled)
        v.addWidget(d["show_fit"])

        rng = QHBoxLayout()
        d["start"] = QSlider(Qt.Orientation.Horizontal)
        d["start"].setRange(0, 100); d["start"].setValue(0)
        d["start"].setSingleStep(1); d["start"].setPageStep(1)
        d["end"] = QSlider(Qt.Orientation.Horizontal)
        d["end"].setRange(0, 100); d["end"].setValue(100)
        d["end"].setSingleStep(1); d["end"].setPageStep(1)
        d["lbl_range"] = QLabel("0\u2013100 %")
        d["start"].valueChanged.connect(self._on_fit_range_changed)
        d["end"].valueChanged.connect(self._on_fit_range_changed)
        rng.addWidget(QLabel("Fit range:"))
        rng.addWidget(d["start"], 1); rng.addWidget(d["end"], 1)
        rng.addWidget(d["lbl_range"])
        v.addLayout(rng)

        d["lbl_info"] = QLabel("Set the search/smoothing/binary/range for the "
                               "%s face. Use 'Fit edges' to fit both faces with "
                               "these settings." % which)
        d["lbl_info"].setStyleSheet("color:#666;"); d["lbl_info"].setWordWrap(True)
        v.addWidget(d["lbl_info"])
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        # Hiding the dialog turns its overlay off (overlay is active only while
        # the dialog is open), so refresh the view on close.
        bb.rejected.connect(lambda: (dlg.hide(), self._redraw()))
        bb.accepted.connect(lambda: (dlg.hide(), self._redraw()))
        v.addWidget(bb)
        return dlg

    def _on_fit_range_changed(self, *_):
        for which in ("rake", "flank"):
            d = self._adj[which]
            if "start" not in d:
                continue
            a = min(d["start"].value(), d["end"].value())
            b = max(d["start"].value(), d["end"].value())
            d["lbl_range"].setText("%d\u2013%d %%" % (a, b))
        self._redraw()

    def _on_show_fit_toggled(self, on):
        # Overlay visibility is resolved by _active_overlay_faces (dialog open
        # AND box checked); just refresh.
        self._redraw()

    def _face_settings(self, which):
        """(search px, blur px, binary threshold or None) for one face, from its
        own Adjust dialog. Safe defaults if that dialog is not built yet."""
        d = self._adj.get(which, {})
        if "search" not in d:
            return 25, 2.0, None
        search = int(d["search"].value())
        blur = d["blur"].value() / 10.0
        binth = float(d["binth"].value()) if d["binary"].isChecked() else None
        return search, blur, binth

    def _fit_settings_range(self, which="rake"):
        """Fit sub-range (start, end) as fractions in [0,1] along a given face
        ('rake' or 'flank'), ordered. Full face if that dialog is not built."""
        d = self._adj.get(which, {})
        if "start" not in d:
            return 0.0, 1.0
        a = min(d["start"].value(), d["end"].value()) / 100.0
        b = max(d["start"].value(), d["end"].value()) / 100.0
        if b - a < 0.05:
            b = min(1.0, a + 0.05)
        return a, b

    def _fit_face_line(self, which):
        """Fit one labelled face onto the strongest image edge using that face's
        own Adjust settings and its [start, end] sub-range. Returns
        (point_on_line, unit_dir, far_idx, far_v, captured_pts) or None. Does not
        modify the polygon (the caller decides how to place the vertices)."""
        import gui.core.tool_detect as td
        edge = self._poly_rake if which == "rake" else self._poly_flank
        if self._image is None or edge is None or len(self._poly_verts) != 4:
            return None
        tip_idx = self._poly_tip_index()
        if tip_idx is None:
            return None
        search, blur, binth = self._face_settings(which)
        fa, fb = self._fit_settings_range(which)
        verts = [np.array(v, float) for v in self._poly_verts]
        a, b = edge, (edge + 1) % 4
        if a == tip_idx:
            tip_v, far_v, far_idx = verts[a], verts[b], b
        else:
            tip_v, far_v, far_idx = verts[b], verts[a], a
        d = far_v - tip_v
        s1 = tip_v + fa * d
        s2 = tip_v + fb * d
        q1, q2, pts = td.snap_line_to_edge(self._image, tuple(s1), tuple(s2),
                                           search=search, blur=blur,
                                           binary_threshold=binth)
        p0 = np.array(q1, float)
        fq = np.array(q2, float) - p0
        nfq = np.hypot(*fq)
        u = fq / nfq if nfq > 1e-6 else (d / max(np.hypot(*d), 1e-9))
        H, W = self._image.shape[:2]
        if pts is not None and len(pts):
            pts = np.asarray(pts, float)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        return p0, u, far_idx, far_v, pts

    def _fit_both_faces(self):
        """Fit BOTH faces at once using each face's own Adjust settings. The new
        tip is the intersection of the two fitted lines (the true tool corner),
        and each far endpoint is the projection of the old far point onto its
        fitted line."""
        if self._poly_rake is None or self._poly_flank is None \
                or len(self._poly_verts) != 4:
            self.lbl_poly.setText("Label both rake and flank edges first.")
            return
        tip_idx = self._poly_tip_index()
        if tip_idx is None:
            self.lbl_poly.setText("Rake/flank must be adjacent (shared tip).")
            return
        fr = self._fit_face_line("rake")
        ff = self._fit_face_line("flank")
        if fr is None or ff is None:
            return
        verts = [list(map(float, v)) for v in self._poly_verts]
        pr, ur, far_r, oldfar_r, pts_r = fr
        pf, uf, far_f, oldfar_f, pts_f = ff
        self._rake_pts, self._flank_pts = pts_r, pts_f
        new_tip = _line_intersection(pr, ur, pf, uf)
        if new_tip is None:                       # near-parallel: keep old tip
            new_tip = np.array(verts[tip_idx], float)
        verts[tip_idx] = [float(new_tip[0]), float(new_tip[1])]
        for (p0, u, far_idx, old_far) in ((pr, ur, far_r, oldfar_r),
                                          (pf, uf, far_f, oldfar_f)):
            proj = p0 + u * float(np.dot(old_far - p0, u))
            verts[far_idx] = [float(proj[0]), float(proj[1])]
        self._poly_verts = [tuple(v) for v in verts]
        self._redraw(); self._recompute(); self._update_poly_label()

    def _nearest_poly_vertex(self, x, y):
        if not self._poly_verts:
            return None
        d = [np.hypot(vx - x, vy - y) for (vx, vy) in self._poly_verts]
        return int(np.argmin(d))

    def _nearest_poly_edge(self, x, y):
        if len(self._poly_verts) != 4:
            return None
        best, bd = None, 1e18
        for e in range(4):
            a = np.asarray(self._poly_verts[e], float)
            b = np.asarray(self._poly_verts[(e + 1) % 4], float)
            mid = 0.5 * (a + b)
            dd = np.hypot(mid[0] - x, mid[1] - y)
            if dd < bd:
                bd, best = dd, e
        return best

    def _poly_tip_index(self):
        """Tip = shared corner of the labelled rake and flank edges (adjacent
        edges); None if not both labelled or not adjacent."""
        from gui.core.tool_polygon import ToolPolygon
        if self._poly_rake is None or self._poly_flank is None:
            return None
        return ToolPolygon.shared_vertex(self._poly_rake, self._poly_flank, 4)

    def _update_poly_label(self):
        n = len(self._poly_verts)
        parts = ["%d/4 pts" % n]
        parts.append("rake=%s" % ("set" if self._poly_rake is not None else "-"))
        parts.append("flank=%s" % ("set" if self._poly_flank is not None else "-"))
        tip = self._poly_tip_index()
        parts.append("tip=%s" % ("auto" if tip is not None else "-"))
        self.lbl_poly.setText(" \u00b7 ".join(parts))
        self._update_extend_enabled()

    def _build_wp_group(self):
        g = QGroupBox("Workpiece reference")
        v = QVBoxLayout(g)
        self.b_wp = QPushButton("Pick workpiece point"); self.b_wp.setCheckable(True)
        self.b_wp.toggled.connect(lambda on: self._set_mode("wp", on, self.b_wp))
        v.addWidget(self.b_wp)
        return g

    def _build_result_group(self):
        g = QGroupBox("Reference geometry")
        grid = QGridLayout(g)
        self.f_rake = self._deg(); self.f_clear = self._deg()
        self.f_tool_x = self._mm(); self.f_tool_y = self._mm()
        self.f_wp_x = self._mm(); self.f_wp_y = self._mm()
        cells = [("Rake:", self.f_rake), ("Clear:", self.f_clear),
                 ("Tool x0:", self.f_tool_x), ("Tool y0:", self.f_tool_y),
                 ("WP x0:", self.f_wp_x), ("WP y0:", self.f_wp_y)]
        for i, (label, field) in enumerate(cells):
            r, c = divmod(i, 2)
            grid.addWidget(QLabel(label), r, 2 * c)
            grid.addWidget(field, r, 2 * c + 1)
        self.b_write = QPushButton("Write to Numerical Model")
        self.b_write.clicked.connect(self._write)
        grid.addWidget(self.b_write, 3, 0, 1, 4)
        return g

    def _readonly(self, s):
        s.setReadOnly(True)
        s.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return s

    def _deg(self):
        s = QDoubleSpinBox(); s.setRange(-180.0, 180.0); s.setDecimals(3)
        s.setSuffix(" °"); return self._readonly(s)

    def _mm(self):
        s = QDoubleSpinBox(); s.setRange(-1e6, 1e6); s.setDecimals(4)
        s.setSuffix(" mm"); return self._readonly(s)

    # =====================================================================
    # Image / scale
    # =====================================================================
    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an alignment image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)")
        if not path:
            return
        try:
            img = _read_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Image", "Could not read image:\n%s" % e)
            return
        self.set_image_array(img, source=path)

    def set_image_array(self, img, source="<array>"):
        self._image = np.asarray(img)
        self._h, self._w = self._image.shape[0], self._image.shape[1]
        self.lbl_image.setText("%s  (%d x %d px)" % (source, self._w, self._h))
        self._clear_selections()

    def _prefill_scale_from_calibration(self):
        s = (self.session.visible_calibration or {}).get("scale_mm_per_px")
        if s:
            self.cb_scale_unit.setCurrentText("mm/px")
            self.spin_scale.setValue(float(s))

    # =====================================================================
    # Modes
    # =====================================================================
    def _set_mode(self, mode, on, btn):
        if on:
            # Uncheck the other exclusive (checkable) mode buttons that exist.
            others = [getattr(self, n, None) for n in (
                "b_poly", "b_poly_rake", "b_poly_flank", "b_poly_move", "b_wp")]
            for b in others:
                if b is not None and b is not btn and b.isCheckable():
                    b.blockSignals(True); b.setChecked(False); b.blockSignals(False)
            self._mode = mode
        elif self._mode == mode:
            self._mode = "idle"

    def _finish_mode(self, btn):
        """Called when a one-shot action (draw done, edge labelled, point
        picked) completes: untoggle its button and return to idle so the user
        doesn't have to deselect it manually."""
        self._mode = "idle"
        if btn is not None and btn.isCheckable():
            btn.blockSignals(True); btn.setChecked(False); btn.blockSignals(False)

    # =====================================================================
    # Testable API
    # =====================================================================
    def set_wp_point(self, px, py):
        self._wp_pt = (float(px), float(py)); self._redraw(); self._recompute()

    # =====================================================================
    # Mouse handling
    # =====================================================================
    def _on_press(self, event):
        if self._image is None or event.inaxes is not self._ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self._mode == "wp":
            self.set_wp_point(x, y)
            self._finish_mode(self.b_wp)
        elif self._mode == "poly":
            if len(self._poly_verts) >= 4:
                self._poly_verts = []
                self._poly_rake = self._poly_flank = None
            self._poly_verts.append((x, y))
            self._redraw(); self._update_poly_label()
            if len(self._poly_verts) == 4:
                self._finish_mode(self.b_poly)        # done drawing 4 points
        elif self._mode == "poly_rake":
            e = self._nearest_poly_edge(x, y)
            if e is not None:
                self._poly_rake = e
                self._redraw(); self._recompute(); self._update_poly_label()
                self._finish_mode(self.b_poly_rake)
        elif self._mode == "poly_flank":
            e = self._nearest_poly_edge(x, y)
            if e is not None:
                self._poly_flank = e
                self._redraw(); self._recompute(); self._update_poly_label()
                self._finish_mode(self.b_poly_flank)
        elif self._mode == "poly_move":
            i = self._nearest_poly_vertex(x, y)
            if i is not None:
                self._drag = ("poly", i)

    def _on_motion(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            if self._zoom_bubble is not None:
                self._zoom_bubble = None
                self._update_zoom_window()        # blank the magnifier
            return
        x, y = float(event.xdata), float(event.ydata)
        # Dragging a polygon vertex (Adjust 'move endpoint' mode).
        if self._mode == "poly_move" and self._drag is not None:
            _, i = self._drag
            self._poly_verts[i] = (x, y)
            self._zoom_bubble = (x, y)
            self._redraw(); self._recompute()
            return
        # Magnifier follows the cursor inside the figure.
        self._zoom_bubble = (x, y)
        self._update_zoom_window()

    def _on_release(self, event):
        if self._mode == "poly_move" and self._drag is not None:
            self._drag = None
            self._redraw(); self._recompute(); self._update_poly_label()

    def _on_scroll(self, event):
        """Scroll wheel changes the magnifier zoom radius."""
        if event.inaxes is not self._ax:
            return
        step = -4 if event.button == "up" else 4
        self._zoom_radius = int(np.clip(self._zoom_radius + step, 5, 200))
        if event.xdata is not None:
            self._zoom_bubble = (float(event.xdata), float(event.ydata))
        self._update_zoom_window()

    # =====================================================================
    # Geometry
    # =====================================================================
    def _poly_tool(self):
        """Build a ToolPolygon from the current 4 vertices + rake/flank labels,
        or None if incomplete."""
        from gui.core.tool_polygon import ToolPolygon
        if len(self._poly_verts) != 4 or self._poly_rake is None \
                or self._poly_flank is None:
            return None
        tp = ToolPolygon(vertices=list(self._poly_verts),
                         rake_edge=self._poly_rake, flank_edge=self._poly_flank)
        return tp if tp.is_complete() else None

    def _recompute(self, *_):
        if self._image is None:
            return
        s = self._mm_per_px()
        # Tool rake/clear angles and tip come from the polygon faces.
        tp = self._poly_tool()
        if tp is not None:
            ar = tp.rake_angle_deg()
            af = tp.flank_angle_deg()
            if np.isfinite(ar):
                self.f_rake.setValue(ar)
            if np.isfinite(af):
                self.f_clear.setValue(af)
            tipx, tipy = tp.tip_point()
            x, y = pixel_to_model(tipx, tipy, self._w, self._h, s)
            self.f_tool_x.setValue(x); self.f_tool_y.setValue(y)
        if self._wp_pt is not None:
            x, y = pixel_to_model(self._wp_pt[0], self._wp_pt[1], self._w, self._h, s)
            self.f_wp_x.setValue(x); self.f_wp_y.setValue(y)

    def _active_overlay_faces(self):
        """Faces whose Adjust dialog is currently OPEN and whose 'Visualise' box
        is checked. The overlay state is only honoured while the dialog is
        visible (and only one Adjust dialog is open at a time)."""
        out = []
        for which in ("rake", "flank"):
            d = self._adj.get(which, {})
            dlg = self._adjust_dialogs.get(which)
            if (dlg is not None and dlg.isVisible()
                    and d.get("show_fit") is not None
                    and d["show_fit"].isChecked()):
                out.append(which)
        return out

    def _draw_fit_overlay(self, ax=None):
        """Per-face background + search-band + captured-points overlay, shown for
        each face whose 'Visualise edge search / smoothing' box is checked. The
        background (smoothing/binary) uses the first active face's settings."""
        if ax is None:
            ax = self._ax
        import scipy.ndimage as ndi
        from gui.core.calibration import _to_gray
        active = self._active_overlay_faces()
        if not active:
            return
        # Background from the first active face's smoothing/binary settings.
        _, blur0, binth0 = self._face_settings(active[0])
        g = _to_gray(self._image).astype(float)
        gs = ndi.gaussian_filter(g, blur0) if blur0 > 0 else g   # smooth THEN
        bg = (gs >= binth0).astype(float) if binth0 is not None else gs
        ax.imshow(bg, cmap="gray", alpha=0.65)
        tip_idx = self._poly_tip_index()
        edges = {"rake": (self._poly_rake, "orange"),
                 "flank": (self._poly_flank, "magenta")}
        for which in active:
            edge, col = edges[which]
            if edge is None or len(self._poly_verts) != 4:
                continue
            search, _, _ = self._face_settings(which)
            fa, fb = self._fit_settings_range(which)
            a_i, b_i = edge, (edge + 1) % 4
            va = np.asarray(self._poly_verts[a_i], float)
            vb = np.asarray(self._poly_verts[b_i], float)
            if tip_idx == b_i:
                va, vb = vb, va
            seg = vb - va
            a = va + fa * seg
            b = va + fb * seg
            d = b - a
            L = float(np.hypot(*d))
            if L < 1e-6:
                continue
            d /= L
            nrm = np.array([-d[1], d[0]]) * search
            band = np.array([a + nrm, b + nrm, b - nrm, a - nrm, a + nrm])
            ax.plot(band[:, 0], band[:, 1], "--", color=col, lw=1.0, alpha=0.9)
            pts = self._rake_pts if which == "rake" else self._flank_pts
            if pts is not None and len(pts):
                ax.plot(pts[:, 0], pts[:, 1], ".", color="lime", ms=3)

    # =====================================================================
    # Drawing
    # =====================================================================
    def _render_scene(self, ax, center_marker=True, lock_limits=False):
        """Draw the shared scene (image, optional fit overlay, tool polygon,
        labelled faces, derived tip) onto ``ax``. Used by both the main figure
        and the zoom panel so they stay in sync. When ``lock_limits`` is True
        the axes are clamped to the image extent (main view)."""
        if self._image is not None:
            is_rgb = (self._image.ndim == 3 and self._image.shape[-1] in (3, 4))
            ax.imshow(self._image, cmap=None if is_rgb else "gray")
            if center_marker:
                ax.plot(self._w / 2.0, self._h / 2.0, "+", color="cyan",
                        ms=12, mew=1.5)
        if self._image is not None and self._active_overlay_faces():
            self._draw_fit_overlay(ax)
        if self._wp_pt is not None:
            ax.plot(self._wp_pt[0], self._wp_pt[1], "o", color="lime", ms=7)
        if self._poly_verts:
            vx = [p[0] for p in self._poly_verts]
            vy = [p[1] for p in self._poly_verts]
            if len(self._poly_verts) == 4:
                ax.plot(vx + [vx[0]], vy + [vy[0]], "-", color="yellow",
                        lw=1.5, alpha=0.9)
            else:
                ax.plot(vx, vy, "o-", color="yellow", lw=1.0, ms=5)
            ax.plot(vx, vy, "s", color="yellow", ms=5)
            if len(self._poly_verts) == 4:
                for e, col in ((self._poly_rake, "orange"),
                               (self._poly_flank, "magenta")):
                    if e is not None:
                        a = self._poly_verts[e]; b = self._poly_verts[(e + 1) % 4]
                        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=col, lw=3.0)
            tip_idx = self._poly_tip_index()
            if tip_idx is not None and len(self._poly_verts) == 4:
                tp = self._poly_verts[tip_idx]
                ax.plot(tp[0], tp[1], "o", color="red", ms=9)
        # Lock the axes to the image extent so off-image overlays (e.g. a wide
        # search band reaching past the borders) never rescale the view.
        if lock_limits and self._image is not None:
            ax.set_xlim(-0.5, self._w - 0.5)
            ax.set_ylim(self._h - 0.5, -0.5)     # image y downwards

    def _redraw(self):
        self._ax.clear(); self._ax.set_axis_off()
        self._render_scene(self._ax, lock_limits=True)
        self._canvas.draw_idle()
        # Keep the integrated zoom panel in sync.
        self._update_zoom_window()

    def _update_zoom_window(self):
        """Refresh the integrated zoom panel: same scene as the main figure,
        centred on the cursor, with a discrete crosshair mire. Blanked when the
        cursor is outside the figure or no image is loaded."""
        if getattr(self, "_zoom_ax", None) is None:
            return
        ax = self._zoom_ax
        ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        if self._image is None or self._zoom_bubble is None:
            self._zoom_canvas.draw_idle()
            return
        x, y = self._zoom_bubble
        z = max(5, int(self._zoom_radius))
        # Same scene as the main view (image + overlay + polygon), no centre +.
        self._render_scene(ax, center_marker=False)
        ax.set_xlim(x - z, x + z); ax.set_ylim(y + z, y - z)
        # Discrete crosshair mire (thin lines) through the cursor.
        ax.axhline(y, color="red", lw=0.5, alpha=0.7)
        ax.axvline(x, color="red", lw=0.5, alpha=0.7)
        ax.plot(x, y, "+", color="red", ms=10, mew=1.0)
        self._zoom_canvas.draw_idle()
        self.lbl_zoom.setText("zoom \u00b1%d px  \u00b7 scroll to change" % z)

    def _clear_selections(self):
        self._rake_pts = self._flank_pts = None
        self._wp_pt = None; self._drag = None
        self._mode = "idle"
        self._redraw()

    # =====================================================================
    # Write
    # =====================================================================
    def values(self):
        vals = {"rake_angle": float(self.f_rake.value()),
                "clear_angle": float(self.f_clear.value()),
                "tool_x0": float(self.f_tool_x.value()),
                "tool_y0": float(self.f_tool_y.value()),
                "wp_x0": float(self.f_wp_x.value()),
                "wp_y0": float(self.f_wp_y.value())}
        # Tool polygon for the DIC mask (pixel vertices), included when drawn.
        if len(self._poly_verts) == 4:
            vals["tool_polygon_px"] = [list(map(float, v))
                                       for v in self._poly_verts]
        return vals

    def _write(self):
        vals = self.values()
        self.session.reference_geometry = dict(vals)
        if self._write_geometry is None:
            QMessageBox.information(self, "Write",
                "Reference geometry stored in the session.\n"
                "(No Numerical Model bound in this context.)")
            return
        try:
            self._write_geometry(vals)
        except Exception as e:
            log_swallowed("writing geometry to numerical model")
            QMessageBox.warning(self, "Write", "Could not write to the model:\n%s" % e)
            return
        QMessageBox.information(self, "Write",
            "Reference geometry written to the Numerical Model (Geometry tab).")
