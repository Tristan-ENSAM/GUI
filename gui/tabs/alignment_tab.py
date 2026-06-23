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
    QFileDialog, QMessageBox, QAbstractSpinBox,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from gui.core.experiment_session import ExperimentSession
from gui.core.sequence_io import _read_image
from gui.core.alignment import (pixel_to_model, line_tilt_from_vertical_deg,
                                 line_tilt_from_horizontal_deg)
from gui.core import tool_detect as td
from gui.core.logging_util import log_swallowed


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

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.addWidget(self._build_image_group())
        ll.addWidget(self._build_scale_group())
        ll.addWidget(self._build_tool_group())
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
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 900])

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

    def _build_tool_group(self):
        g = QGroupBox("Tool — rough draw + auto fit")
        v = QVBoxLayout(g)

        draw = QHBoxLayout()
        self.b_rake = QPushButton("Draw rake line"); self.b_rake.setCheckable(True)
        self.b_flank = QPushButton("Draw flank line"); self.b_flank.setCheckable(True)
        self.b_move = QPushButton("Move endpoint"); self.b_move.setCheckable(True)
        for b, m in ((self.b_rake, "rake"), (self.b_flank, "flank"),
                     (self.b_move, "move")):
            b.toggled.connect(lambda on, mm=m, bb=b: self._set_mode(mm, on, bb))
            draw.addWidget(b)
        v.addLayout(draw)

        srch = QHBoxLayout()
        srch.addWidget(QLabel("Edge search:"))
        self.sld_search = QSlider(Qt.Orientation.Horizontal)
        self.sld_search.setRange(3, 80); self.sld_search.setValue(25)
        self.lbl_search = QLabel("25 px")
        self.sld_search.valueChanged.connect(
            lambda x: self.lbl_search.setText("%d px" % x))
        srch.addWidget(self.sld_search, 1); srch.addWidget(self.lbl_search)
        v.addLayout(srch)

        smo = QHBoxLayout()
        smo.addWidget(QLabel("Smoothing:"))
        self.sld_blur = QSlider(Qt.Orientation.Horizontal)
        self.sld_blur.setRange(0, 100); self.sld_blur.setValue(20)   # /10 -> sigma
        self.lbl_blur = QLabel("2.0 px")
        self.sld_blur.valueChanged.connect(
            lambda x: self.lbl_blur.setText("%.1f px" % (x / 10.0)))
        smo.addWidget(self.sld_blur, 1); smo.addWidget(self.lbl_blur)
        v.addLayout(smo)

        self.b_fit = QPushButton("Fit edges")
        self.b_fit.clicked.connect(self._fit_edges)
        v.addWidget(self.b_fit)

        zoom = QHBoxLayout()
        zoom.addWidget(QLabel("Magnifier zoom:"))
        self.sld_zoom = QSlider(Qt.Orientation.Horizontal)
        self.sld_zoom.setRange(5, 200); self.sld_zoom.setValue(30)
        self.lbl_zoom = QLabel("30 px")
        self.sld_zoom.valueChanged.connect(
            lambda x: self.lbl_zoom.setText("%d px" % x))
        zoom.addWidget(self.sld_zoom, 1); zoom.addWidget(self.lbl_zoom)
        v.addLayout(zoom)

        self.b_clear = QPushButton("Clear selections")
        self.b_clear.clicked.connect(self._clear_selections)
        v.addWidget(self.b_clear)

        self.lbl_hint = QLabel("Choose an image, roughly draw the rake and "
                               "flank lines, then 'Fit edges'.")
        self.lbl_hint.setStyleSheet("color:#666;"); self.lbl_hint.setWordWrap(True)
        v.addWidget(self.lbl_hint)
        return g

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
            for b in (self.b_rake, self.b_flank, self.b_move, self.b_wp):
                if b is not btn:
                    b.blockSignals(True); b.setChecked(False); b.blockSignals(False)
            self._mode = mode
        elif self._mode == mode:
            self._mode = "idle"

    # =====================================================================
    # Testable API
    # =====================================================================
    def set_rake_line(self, p1, p2):
        self._rake_rough = [tuple(map(float, p1)), tuple(map(float, p2))]
        self._rake_fit = self._rake_pts = None
        self._redraw(); self._recompute()

    def set_flank_line(self, p1, p2):
        self._flank_rough = [tuple(map(float, p1)), tuple(map(float, p2))]
        self._flank_fit = self._flank_pts = None
        self._redraw(); self._recompute()

    def set_wp_point(self, px, py):
        self._wp_pt = (float(px), float(py)); self._redraw(); self._recompute()

    def fit_edges(self):
        """Snap the rough rake/flank lines to the strongest local edges."""
        if self._image is None:
            return
        search = int(self.sld_search.value())
        blur = self.sld_blur.value() / 10.0
        if len(self._rake_rough) == 2:
            q1, q2, pts = td.snap_line_to_edge(self._image, *self._rake_rough,
                                               search=search, blur=blur)
            self._rake_fit = (q1, q2); self._rake_pts = pts
        if len(self._flank_rough) == 2:
            q1, q2, pts = td.snap_line_to_edge(self._image, *self._flank_rough,
                                               search=search, blur=blur)
            self._flank_fit = (q1, q2); self._flank_pts = pts
        self._redraw(); self._recompute()

    def _fit_edges(self):
        if len(self._rake_rough) < 2 or len(self._flank_rough) < 2:
            QMessageBox.information(self, "Fit edges", "Draw both the rake and "
                                    "the flank line first.")
            return
        self.fit_edges()
        self.lbl_hint.setText("Edges fitted. Drag endpoints (Move endpoint) "
                              "and re-fit if needed.")

    # =====================================================================
    # Mouse handling
    # =====================================================================
    def _endpoints(self):
        eps = []
        for key, rough in (("rake", self._rake_rough), ("flank", self._flank_rough)):
            for i, p in enumerate(rough):
                eps.append((key, i, p))
        return eps

    def _on_press(self, event):
        if self._image is None or event.inaxes is not self._ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self._mode in ("rake", "flank"):
            rough = self._rake_rough if self._mode == "rake" else self._flank_rough
            if len(rough) >= 2:
                rough.clear()
            rough.append((x, y))
            if self._mode == "rake":
                self._rake_fit = self._rake_pts = None
            else:
                self._flank_fit = self._flank_pts = None
            self._redraw(); self._recompute()
        elif self._mode == "move":
            eps = self._endpoints()
            if eps:
                key, i, _ = min(eps, key=lambda e: np.hypot(e[2][0] - x, e[2][1] - y))
                self._drag = (key, i)
                self._magnifier = (x, y)
        elif self._mode == "wp":
            self.set_wp_point(x, y)

    def _on_motion(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self._mode == "move" and self._drag is not None:
            key, i = self._drag
            rough = self._rake_rough if key == "rake" else self._flank_rough
            rough[i] = (x, y)
            if key == "rake":
                self._rake_fit = self._rake_pts = None
            else:
                self._flank_fit = self._flank_pts = None
            self._magnifier = (x, y)
            self._redraw()

    def _on_release(self, event):
        if self._mode == "move" and self._drag is not None:
            self._drag = None; self._magnifier = None
            self._redraw(); self._recompute()

    # =====================================================================
    # Geometry
    # =====================================================================
    def _rake_line(self):
        if self._rake_fit is not None:
            return self._rake_fit
        return tuple(self._rake_rough) if len(self._rake_rough) == 2 else None

    def _flank_line(self):
        if self._flank_fit is not None:
            return self._flank_fit
        return tuple(self._flank_rough) if len(self._flank_rough) == 2 else None

    def _tip(self):
        rk, fl = self._rake_line(), self._flank_line()
        if rk is None or fl is None:
            return None
        return td.segment_intersection(rk[0], rk[1], fl[0], fl[1])

    def _recompute(self, *_):
        if self._image is None:
            return
        s = self._mm_per_px()
        rk, fl = self._rake_line(), self._flank_line()
        if rk is not None:
            a = line_tilt_from_vertical_deg(rk[0], rk[1])     # rake from vertical
            if np.isfinite(a):
                self.f_rake.setValue(a)
        if fl is not None:
            a = line_tilt_from_horizontal_deg(fl[0], fl[1])   # clear from horizontal
            if np.isfinite(a):
                self.f_clear.setValue(a)
        tip = self._tip()
        if tip is not None:
            x, y = pixel_to_model(tip[0], tip[1], self._w, self._h, s)
            self.f_tool_x.setValue(x); self.f_tool_y.setValue(y)
        if self._wp_pt is not None:
            x, y = pixel_to_model(self._wp_pt[0], self._wp_pt[1], self._w, self._h, s)
            self.f_wp_x.setValue(x); self.f_wp_y.setValue(y)

    # =====================================================================
    # Drawing
    # =====================================================================
    def _redraw(self):
        self._ax.clear(); self._ax.set_axis_off()
        if self._image is not None:
            is_rgb = (self._image.ndim == 3 and self._image.shape[-1] in (3, 4))
            self._ax.imshow(self._image, cmap=None if is_rgb else "gray")
            self._ax.plot(self._w / 2.0, self._h / 2.0, "+", color="cyan", ms=12, mew=1.5)
        # rough lines (thin), endpoints
        for rough, col in ((self._rake_rough, "orange"), (self._flank_rough, "magenta")):
            if rough:
                xs = [p[0] for p in rough]; ys = [p[1] for p in rough]
                self._ax.plot(xs, ys, ":", color=col, lw=1.0)
                self._ax.plot(xs, ys, "o", color=col, ms=5)
        # snapped points + fitted lines
        for pts, col in ((self._rake_pts, "lime"), (self._flank_pts, "lime")):
            if pts is not None and len(pts):
                self._ax.plot(pts[:, 0], pts[:, 1], ".", color=col, ms=2)
        for fit, col in ((self._rake_fit, "orange"), (self._flank_fit, "magenta")):
            if fit is not None:
                self._ax.plot([fit[0][0], fit[1][0]], [fit[0][1], fit[1][1]],
                              "-", color=col, lw=2.0)
        tip = self._tip()
        if tip is not None:
            self._ax.plot(tip[0], tip[1], "*", color="red", ms=15)
        if self._wp_pt is not None:
            self._ax.plot(self._wp_pt[0], self._wp_pt[1], "o", color="lime", ms=7)
        if self._magnifier is not None:
            self._draw_magnifier(*self._magnifier)
        self._canvas.draw_idle()

    def _draw_magnifier(self, x, y):
        if self._image is None:
            return
        z = int(self.sld_zoom.value()) if hasattr(self, "sld_zoom") else 30
        ins = self._ax.inset_axes([0.72, 0.72, 0.27, 0.27])
        ins.set_xticks([]); ins.set_yticks([])
        is_rgb = (self._image.ndim == 3 and self._image.shape[-1] in (3, 4))
        ins.imshow(self._image, cmap=None if is_rgb else "gray")
        ins.set_xlim(x - z, x + z); ins.set_ylim(y + z, y - z)
        ins.plot(x, y, "+", color="red", ms=10, mew=1.5)
        for sp in ins.spines.values():
            sp.set_edgecolor("red")

    def _clear_selections(self):
        self._rake_rough = []; self._flank_rough = []
        self._rake_fit = self._flank_fit = None
        self._rake_pts = self._flank_pts = None
        self._wp_pt = None; self._drag = None; self._magnifier = None
        for b in (self.b_rake, self.b_flank, self.b_move, self.b_wp):
            b.blockSignals(True); b.setChecked(False); b.blockSignals(False)
        self._mode = "idle"
        self._redraw()

    # =====================================================================
    # Write
    # =====================================================================
    def values(self):
        return {"rake_angle": float(self.f_rake.value()),
                "clear_angle": float(self.f_clear.value()),
                "tool_x0": float(self.f_tool_x.value()),
                "tool_y0": float(self.f_tool_y.value()),
                "wp_x0": float(self.f_wp_x.value()),
                "wp_y0": float(self.f_wp_y.value())}

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
