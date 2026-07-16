# -*- coding: utf-8 -*-
"""
DIC tab — compute (or import) velocity fields from the visible image sequence.

Workflow:
  1. Load the visible sequence (from the session, or pick a folder).
  2. Set the scale mm/px (from the visible calibration, or manual).
  3. Draw a search ROI on the reference frame (material region).
  4. Choose the engine (local subset; global q4dic later) and its parameters.
  5. Run -> velocity fields (mm/s, model frame) shown as a heatmap with a time
     slider and a profile-extraction tool.
  6. Save -> <stem>_dic.npz + .json, recorded in session.dic_field_path.
     Import -> load an external field file into the viewer.

Velocity is instantaneous (frame i -> i+1) on a fixed Eulerian grid, matching
the CEL Eulerian velocity field used downstream for inverse identification.
"""
from __future__ import annotations
import os
import logging
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, QSplitter,
    QFileDialog, QMessageBox, QProgressBar, QGridLayout, QSlider,
    QDialog, QDialogButtonBox,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector

from gui.core.experiment_session import ExperimentSession
from gui.core.sequence_io import ImageSequence
from gui.core import dic as dic_engine
from gui.core import dic_global as dic_global_engine
from gui.core.exp_field_io import save_dic_field, load_dic_field
from gui.widgets.dic_field_viewer import DicFieldViewer
from gui.widgets.num_input import DecimalSpinBox, WheelStepSlider
from gui.core.logging_util import log_swallowed


class _DicWorker(QThread):
    """Runs compute_dic_fields off the UI thread, reporting per-pair progress
    and a detailed per-frame log (valid-point count, mean ZNCC, timing, ETA)."""
    sig_progress = Signal(int, int)
    sig_log = Signal(str)
    sig_done = Signal(object)
    sig_failed = Signal(str)

    def __init__(self, seq, pts, params, fps, mmpp, w, h, trig, keep=None,
                 mask_per_frame=False, mask_params=None):
        super().__init__()
        self._args = (seq, pts, params, fps, mmpp, w, h, trig, keep)
        self._mask_per_frame = mask_per_frame
        self._mask_params = mask_params

    def _on_frame(self, info):
        """One-line status with an ETA from the running mean frame time, for
        the local engine (reports valid-point count and mean ZNCC)."""
        i = info["index"] + 1
        n = info["n_pairs"]
        done_frac = i / n if n else 1.0
        eta = info["elapsed_s"] / done_frac - info["elapsed_s"] if done_frac else 0.0
        z = info["mean_zncc"]
        z_txt = ("%.3f" % z) if z is not None else "n/a"
        self.sig_log.emit(
            "Frame %d/%d | %d/%d valid | mean ZNCC %s | %.2f s/frame | ETA %s"
            % (i, n, info["n_valid"], info["n_total"], z_txt,
               info["frame_s"], _fmt_eta(eta)))

    def run(self):
        seq, pts, params, fps, mmpp, w, h, trig, keep = self._args
        try:
            res = dic_engine.compute_dic_fields(
                seq, pts, params, fps=fps, mm_per_px=mmpp, img_w=w, img_h=h,
                trigger_offset_s=trig, point_keep=keep,
                mask_per_frame=self._mask_per_frame,
                mask_params=self._mask_params,
                progress=lambda i, n: self.sig_progress.emit(i, n),
                on_frame=self._on_frame)
            self.sig_done.emit(res)
        except Exception as e:                       # pragma: no cover
            self.sig_failed.emit(str(e))


class _DicGlobalWorker(QThread):
    """Runs compute_dic_global_fields off the UI thread, reporting per-pair
    progress and a detailed per-frame log (iterations, residual, timing, ETA).
    Mirrors _DicWorker but for the global Q4 engine, which takes the ROI (it
    builds its own mesh) instead of a precomputed point grid."""
    sig_progress = Signal(int, int)
    sig_log = Signal(str)
    sig_done = Signal(object)
    sig_failed = Signal(str)

    def __init__(self, seq, roi, params, fps, mmpp, w, h, trig, sigma_f=None):
        super().__init__()
        self._args = (seq, roi, params, fps, mmpp, w, h, trig, sigma_f)

    def _on_frame(self, info):
        """Build a one-line status string with an ETA from the running mean
        frame time, and emit it to the UI."""
        i = info["index"] + 1
        n = info["n_pairs"]
        done_frac = i / n if n else 1.0
        eta = info["elapsed_s"] / done_frac - info["elapsed_s"] if done_frac else 0.0
        conv = "ok" if info["converged"] else "NOT CONVERGED"
        res = info["residual"]
        res_txt = ("%.3e" % res) if res is not None else "n/a"
        self.sig_log.emit(
            "Frame %d/%d | %d iter | residual %s | %s | %.2f s/frame | ETA %s"
            % (i, n, info["n_iter"], res_txt, conv, info["frame_s"],
               _fmt_eta(eta)))

    def run(self):
        seq, roi, params, fps, mmpp, w, h, trig, sigma_f = self._args
        try:
            res = dic_global_engine.compute_dic_global_fields(
                seq, roi, params, fps=fps, mm_per_px=mmpp, img_w=w, img_h=h,
                trigger_offset_s=trig, sigma_f=sigma_f,
                progress=lambda i, n: self.sig_progress.emit(i, n),
                on_frame=self._on_frame)
            self.sig_done.emit(res)
        except Exception as e:                       # pragma: no cover
            self.sig_failed.emit(str(e))


def _fmt_eta(seconds: float) -> str:
    """Human-readable ETA (e.g. '12 s', '3 min 05 s')."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "%.0f s" % seconds
    m = int(seconds // 60)
    s = int(seconds % 60)
    return "%d min %02d s" % (m, s)


class DICTab(QWidget):
    def __init__(self, session: ExperimentSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._seq = None
        self._roi = None              # (x, y, w, h) px
        self._result = None           # dict from velocity_fields

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.addWidget(self._build_sequence_group())
        ll.addWidget(self._build_scale_group())
        ll.addWidget(self._build_roi_group(), stretch=1)   # ROI gets the room
        left.setMinimumWidth(360)
        splitter.addWidget(left)

        # Right side: the viewer on top, engine parameters + run/save BELOW the
        # plots (so the ROI selection on the left can be large).
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.viewer = DicFieldViewer()
        rl.addWidget(self.viewer, stretch=1)
        bottom = QHBoxLayout()
        bottom.addWidget(self._build_params_group(), stretch=1)
        bottom.addWidget(self._build_run_group(), stretch=2)
        rl.addLayout(bottom)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1100])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)
        self._prefill_scale()

    # =====================================================================
    # Builders
    # =====================================================================
    def _build_sequence_group(self):
        g = QGroupBox("Visible sequence")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        b1 = QPushButton("Load from session")
        b1.clicked.connect(self._load_from_session)
        b2 = QPushButton("Choose folder…")
        b2.clicked.connect(self._choose_folder)
        row.addWidget(b1); row.addWidget(b2)
        v.addLayout(row)
        self.lbl_seq = QLabel("(no sequence)")
        self.lbl_seq.setStyleSheet("color:#666;"); self.lbl_seq.setWordWrap(True)
        v.addWidget(self.lbl_seq)
        return g

    def _build_scale_group(self):
        g = QGroupBox("Scale & timing")
        form = QFormLayout(g)
        row = QHBoxLayout()
        self.spin_scale = DecimalSpinBox()
        self.spin_scale.setRange(1e-9, 1e9); self.spin_scale.setDecimals(6)
        self.spin_scale.setValue(1.0)
        self.cb_scale_unit = QComboBox()
        self.cb_scale_unit.addItems(["mm/px", "px/mm"])
        row.addWidget(self.spin_scale, 1)
        row.addWidget(self.cb_scale_unit)
        w = QWidget(); w.setLayout(row)
        form.addRow("Resolution:", w)
        b = QPushButton("From visible calibration")
        b.clicked.connect(self._prefill_scale)
        form.addRow(b)
        self.spin_fps = DecimalSpinBox()
        self.spin_fps.setRange(1.0, 1e9); self.spin_fps.setDecimals(1)
        self.spin_fps.setSuffix(" fps"); self.spin_fps.setValue(self.session.visible.fps)
        form.addRow("Frame rate:", self.spin_fps)
        return g

    def _mm_per_px(self) -> float:
        """Canonical mm/px from the value + selected unit (mm/px or px/mm)."""
        v = float(self.spin_scale.value())
        if self.cb_scale_unit.currentText() == "px/mm":
            return 1.0 / v if v != 0 else float("nan")
        return v

    def _build_roi_group(self):
        g = QGroupBox("Search ROI (reference frame)")
        v = QVBoxLayout(g)
        self._roi_fig = Figure(figsize=(4.2, 3.4), tight_layout=True)
        self._roi_canvas = FigureCanvas(self._roi_fig)
        self._roi_ax = self._roi_fig.add_subplot(111)
        self._roi_ax.set_axis_off()
        v.addWidget(self._roi_canvas, 1)

        # Frame scrubber: preview the mask/mesh on any frame of the sequence
        # (visualisation only; the DIC computation is unaffected here).
        frow = QHBoxLayout()
        self.lbl_frame = QLabel("Frame:")
        self.sld_frame = WheelStepSlider(Qt.Orientation.Horizontal)
        self.sld_frame.setRange(0, 0)
        self.sld_frame.setSingleStep(1); self.sld_frame.setPageStep(1)
        self.sld_frame.setEnabled(False)
        self.sld_frame.valueChanged.connect(self._on_preview_frame_changed)
        self.lbl_frame_idx = QLabel("0/0")
        self.lbl_frame_idx.setFixedWidth(56)
        frow.addWidget(self.lbl_frame)
        frow.addWidget(self.sld_frame, 1)
        frow.addWidget(self.lbl_frame_idx)
        v.addLayout(frow)
        self._preview_frame_idx = 0

        row = QHBoxLayout()
        self.lbl_roi = QLabel("Load a sequence, then drag a rectangle.")
        self.lbl_roi.setStyleSheet("color:#666;"); self.lbl_roi.setWordWrap(True)
        row.addWidget(self.lbl_roi, 1)
        # nudge arrows (move the ROI by 1 px)
        self._nudge_btns = []
        for txt, dx, dy in (("←", -1, 0), ("↑", 0, -1), ("↓", 0, 1), ("→", 1, 0)):
            b = QPushButton(txt); b.setFixedWidth(28)
            b.setToolTip("Move the ROI 1 px")
            b.clicked.connect(lambda _=False, ax=dx, ay=dy: self._nudge_roi(ax, ay))
            row.addWidget(b); self._nudge_btns.append(b)
        self.b_validate = QPushButton("Validate ROI")
        self.b_validate.setCheckable(True)
        self.b_validate.setToolTip("Freeze the ROI and the measurement points "
                                   "(they turn green). Required before running.")
        self.b_validate.toggled.connect(self._on_validate)
        row.addWidget(self.b_validate)
        v.addLayout(row)
        self._selector = None
        self._roi_locked = False
        self._preview_artists = []
        return g

    def _build_params_group(self):
        """Compact panel kept in the Search-ROI view: just the engine selector
        and a button opening the parameters dialog (which holds everything
        else). Keeping this minimal avoids crowding the main view."""
        self._build_param_widgets()           # create all parameter widgets
        g = QGroupBox("Engine")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel("Engine:"))
        row.addWidget(self.cb_engine, 1)
        v.addLayout(row)
        self.b_params = QPushButton("Parameters\u2026")
        self.b_params.setToolTip("Open the DIC parameters (engine settings, "
                                 "mask, multi-scale pyramid).")
        self.b_params.clicked.connect(self._open_params_dialog)
        v.addWidget(self.b_params)
        self.lbl_engine_summary = QLabel("")
        self.lbl_engine_summary.setStyleSheet("color:#666;")
        self.lbl_engine_summary.setWordWrap(True)
        v.addWidget(self.lbl_engine_summary)
        v.addStretch(1)
        # Build the parameters dialog NOW (hidden) so that every parameter
        # widget gets parented into it immediately. Otherwise the widgets,
        # created without a parent in _build_param_widgets, would float as
        # separate top-level windows until the dialog is first opened.
        self._params_dialog = self._build_params_dialog()
        self._params_dialog.hide()
        self._update_engine_summary()
        return g

    def _build_param_widgets(self):
        """Create every parameter widget once (kept as attributes on self so
        _params/_global_params/_keep_mask/_lock_params are unchanged). They are
        laid out later inside the parameters dialog."""
        self.cb_engine = QComboBox()
        self.cb_engine.addItem("Local (subset ZNCC)", "local")
        self.cb_engine.addItem("Global Q4 (q4dic)", "global")
        self.cb_engine.currentIndexChanged.connect(self._on_engine_changed)

        # --- Local engine widgets (subset ZNCC) ---
        self._local_widgets = []
        self.sp_subset = QSpinBox(); self.sp_subset.setRange(5, 301)
        self.sp_subset.setSingleStep(2); self.sp_subset.setValue(31)
        self.sp_subset.setSuffix(" px")
        self.sp_step = QSpinBox(); self.sp_step.setRange(1, 200)
        self.sp_step.setValue(16); self.sp_step.setSuffix(" px")
        self.lbl_subset = QLabel("Subset:"); self.lbl_step = QLabel("Step:")
        self.sp_search = QSpinBox(); self.sp_search.setRange(1, 200)
        self.sp_search.setValue(16); self.sp_search.setSuffix(" px")
        self.sp_zncc = DecimalSpinBox(); self.sp_zncc.setRange(0.0, 1.0)
        self.sp_zncc.setSingleStep(0.05); self.sp_zncc.setValue(0.5)
        self.lbl_search = QLabel("Search:"); self.lbl_zncc = QLabel("ZNCC min:")
        self._local_widgets += [self.lbl_subset, self.sp_subset,
                                self.lbl_step, self.sp_step,
                                self.lbl_search, self.sp_search,
                                self.lbl_zncc, self.sp_zncc]
        for sp in (self.sp_subset, self.sp_step, self.sp_search):
            sp.valueChanged.connect(self._preview_points)

        # --- Global engine widgets (Q4) ---
        self._global_widgets = []
        self.sp_elem = QSpinBox(); self.sp_elem.setRange(4, 400)
        self.sp_elem.setValue(24); self.sp_elem.setSuffix(" px")
        self.sp_elem.setToolTip("Q4 element side in pixels. The ROI is trimmed "
                                "to an integer number of elements.")
        self.cb_variant = QComboBox()
        self.cb_variant.addItem("Standard (full Hessian)", "standard")
        self.cb_variant.addItem("Hild (fixed Hessian)", "hild")
        self.lbl_elem = QLabel("Element:"); self.lbl_variant = QLabel("Variant:")
        self.cb_pattern = QComboBox()
        self.cb_pattern.addItem("Incremental (i \u2192 i+1)", True)
        self.cb_pattern.addItem("Total (0 \u2192 i)", False)
        self.chk_uinit = QCheckBox("Init from previous frame")
        self.chk_uinit.setToolTip("Use the previous pair's solution as the "
                                  "initial guess (faster when motion varies "
                                  "slowly).")
        self.lbl_pattern = QLabel("Pattern:")
        self.sp_maxiter = QSpinBox(); self.sp_maxiter.setRange(1, 1000)
        self.sp_maxiter.setValue(30)
        self.sp_maxiter.setToolTip("Maximum Newton-Raphson iterations per image "
                                   "pair.")
        self.sp_tol = DecimalSpinBox(); self.sp_tol.setRange(1e-6, 1.0)
        self.sp_tol.setDecimals(6); self.sp_tol.setSingleStep(1e-4)
        self.sp_tol.setValue(1e-4); self.sp_tol.setSuffix(" px")
        self.sp_tol.setToolTip("Convergence threshold on the nodal correction "
                               "norm ||dU|| (pixels).")
        self.lbl_maxiter = QLabel("Max iter:"); self.lbl_tol = QLabel("Residual:")
        # Multi-scale pyramid
        self.sp_pyr_levels = QSpinBox(); self.sp_pyr_levels.setRange(1, 8)
        self.sp_pyr_levels.setValue(1)
        self.sp_pyr_levels.setToolTip("Gaussian-pyramid levels. 1 = single "
                                      "scale. More levels widen the "
                                      "displacement-capture range for large "
                                      "inter-frame motion.")
        self.sp_pyr_sigma = DecimalSpinBox(); self.sp_pyr_sigma.setRange(0.1, 5.0)
        self.sp_pyr_sigma.setDecimals(2); self.sp_pyr_sigma.setSingleStep(0.1)
        self.sp_pyr_sigma.setValue(1.0)
        self.sp_pyr_sigma.setToolTip("Anti-aliasing Gaussian sigma applied "
                                     "before each 2x downsampling.")
        self.lbl_pyr_levels = QLabel("Pyramid levels:")
        self.lbl_pyr_sigma = QLabel("Pyramid sigma:")
        self.sp_coverage = DecimalSpinBox(); self.sp_coverage.setRange(0.0, 1.0)
        self.sp_coverage.setDecimals(2); self.sp_coverage.setSingleStep(0.05)
        self.sp_coverage.setValue(0.5)
        self.sp_coverage.setToolTip("Material-coverage threshold: a Q4 element "
                                    "is kept when at least this fraction of its "
                                    "pixels are material (mask must be enabled).")
        self.lbl_coverage = QLabel("Coverage min:")
        self.chk_convect = QCheckBox("Convect mesh (Lagrangian)")
        self.chk_convect.setToolTip("Move the mesh with the material between "
                                    "frames; folded-over elements (det(J)<=0) "
                                    "are excluded. Fields are reported at the "
                                    "reference node positions.")
        self.chk_uncert = QCheckBox("Compute uncertainty")
        self.chk_uncert.setEnabled(False)
        self.chk_uncert.setToolTip("Analytic displacement uncertainty "
                                   "(sigma_Ux/sigma_Uy) from Cov = 2 sigma_f^2 "
                                   "[H]^-1. Needs the image noise sigma_f from "
                                   "the Noise tab (not implemented yet).")
        self._global_widgets += [self.lbl_elem, self.sp_elem,
                                 self.lbl_variant, self.cb_variant,
                                 self.lbl_pattern, self.cb_pattern,
                                 self.chk_uinit,
                                 self.lbl_maxiter, self.sp_maxiter,
                                 self.lbl_tol, self.sp_tol,
                                 self.lbl_pyr_levels, self.sp_pyr_levels,
                                 self.lbl_pyr_sigma, self.sp_pyr_sigma,
                                 self.lbl_coverage, self.sp_coverage,
                                 self.chk_convect,
                                 self.chk_uncert]
        self.sp_elem.valueChanged.connect(self._preview_points)
        self.sp_coverage.valueChanged.connect(self._preview_points)

        # --- Mask widgets (background / tool), intensity-only ---
        self.chk_mask = QCheckBox("Enable mask")
        self.chk_mask.toggled.connect(self._preview_points)
        self.sld_int = WheelStepSlider(Qt.Orientation.Horizontal)
        self.sld_int.setRange(0, 255); self.sld_int.setValue(25)
        self.sld_int.setSingleStep(1); self.sld_int.setPageStep(1)
        self.sld_int.valueChanged.connect(self._preview_points)
        self.lbl_int = QLabel("25")
        self.sld_int.valueChanged.connect(lambda v: self.lbl_int.setText(str(v)))
        self.lbl_mask = QLabel("")
        self.lbl_mask.setStyleSheet("color:#666;")

        self._params_dialog = None
        # Keep the compact summary under the Parameters button in sync.
        for wdg in (self.sp_subset, self.sp_step, self.sp_search, self.sp_zncc,
                    self.sp_elem, self.sp_maxiter, self.sp_tol,
                    self.sp_pyr_levels, self.sp_pyr_sigma, self.sp_coverage):
            wdg.valueChanged.connect(self._update_engine_summary)
        for cb in (self.cb_variant, self.cb_pattern):
            cb.currentIndexChanged.connect(self._update_engine_summary)
        self.chk_mask.toggled.connect(self._update_engine_summary)
        self.chk_uinit.toggled.connect(self._update_engine_summary)
        self.chk_convect.toggled.connect(self._update_engine_summary)
        self._set_global_visible(False)        # local by default

    def _build_params_dialog(self):
        """Build (once) the modeless dialog holding all parameter widgets."""
        dlg = QDialog(self)
        dlg.setWindowTitle("DIC parameters")
        outer = QVBoxLayout(dlg)

        # Engine-specific group (local + global widgets share grid rows; only
        # one set is visible, toggled by _set_global_visible).
        eng = QGroupBox("Engine parameters")
        grid = QGridLayout(eng)
        grid.addWidget(self.lbl_subset, 0, 0); grid.addWidget(self.sp_subset, 0, 1)
        grid.addWidget(self.lbl_step, 0, 2); grid.addWidget(self.sp_step, 0, 3)
        grid.addWidget(self.lbl_search, 1, 0); grid.addWidget(self.sp_search, 1, 1)
        grid.addWidget(self.lbl_zncc, 1, 2); grid.addWidget(self.sp_zncc, 1, 3)
        grid.addWidget(self.lbl_elem, 0, 0); grid.addWidget(self.sp_elem, 0, 1)
        grid.addWidget(self.lbl_variant, 0, 2); grid.addWidget(self.cb_variant, 0, 3)
        grid.addWidget(self.lbl_pattern, 1, 0); grid.addWidget(self.cb_pattern, 1, 1)
        grid.addWidget(self.chk_uinit, 1, 2, 1, 2)
        grid.addWidget(self.lbl_maxiter, 2, 0); grid.addWidget(self.sp_maxiter, 2, 1)
        grid.addWidget(self.lbl_tol, 2, 2); grid.addWidget(self.sp_tol, 2, 3)
        grid.addWidget(self.lbl_pyr_levels, 3, 0); grid.addWidget(self.sp_pyr_levels, 3, 1)
        grid.addWidget(self.lbl_pyr_sigma, 3, 2); grid.addWidget(self.sp_pyr_sigma, 3, 3)
        grid.addWidget(self.lbl_coverage, 4, 0); grid.addWidget(self.sp_coverage, 4, 1)
        grid.addWidget(self.chk_convect, 5, 0, 1, 4)
        grid.addWidget(self.chk_uncert, 6, 0, 1, 4)
        outer.addWidget(eng)

        # Mask group.
        mg = QGroupBox("Mask (background / tool)")
        form = QFormLayout(mg)
        form.addRow(self.chk_mask)
        ri = QHBoxLayout(); ri.addWidget(self.sld_int, 1); ri.addWidget(self.lbl_int)
        wi = QWidget(); wi.setLayout(ri)
        form.addRow("Min intensity:", wi)
        form.addRow(self.lbl_mask)
        outer.addWidget(mg)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.hide)
        bb.accepted.connect(dlg.hide)
        outer.addWidget(bb)
        return dlg

    def _open_params_dialog(self):
        if self._params_dialog is None:        # safety: build on demand
            self._params_dialog = self._build_params_dialog()
        self._set_global_visible(self._is_global())
        self._params_dialog.show()
        self._params_dialog.raise_()
        self._params_dialog.activateWindow()

    def _update_engine_summary(self):
        """Short text under the Parameters button summarising the key settings,
        so the main view still conveys the current configuration at a glance."""
        if not hasattr(self, "lbl_engine_summary"):
            return                       # called before the panel is built
        if self._is_global():
            txt = ("Global Q4 \u00b7 elem %d px \u00b7 %s \u00b7 %s \u00b7 pyr %d"
                   % (int(self.sp_elem.value()),
                      self.cb_variant.currentData(),
                      "incr" if self.cb_pattern.currentData() else "total",
                      int(self.sp_pyr_levels.value())))
            if self.chk_convect.isChecked():
                txt += " \u00b7 convect"
        else:
            txt = ("Local \u00b7 subset %d \u00b7 step %d \u00b7 search %d \u00b7 "
                   "ZNCC\u2265%.2f" % (int(self.sp_subset.value()),
                                       int(self.sp_step.value()),
                                       int(self.sp_search.value()),
                                       float(self.sp_zncc.value())))
        if self.chk_mask.isChecked():
            txt += " \u00b7 mask on"
            if self._is_global():
                txt += " (cov\u2265%.2f)" % float(self.sp_coverage.value())
        self.lbl_engine_summary.setText(txt)

    def _set_global_visible(self, on):
        """Show the global-engine widgets and hide the local ones (or vice
        versa). Only the parameter widgets toggle; the engine combo stays."""
        for w in self._local_widgets:
            w.setVisible(not on)
        for w in self._global_widgets:
            w.setVisible(on)

    def _is_global(self) -> bool:
        return self.cb_engine.currentData() == "global"

    def _on_engine_changed(self, *_):
        """Switch the visible parameter set and refresh the reference-frame
        overlay (point grid for local, Q4 mesh for global)."""
        self._set_global_visible(self._is_global())
        self._update_engine_summary()
        self._preview_points()

    def _preview_frame_image(self):
        """The image currently shown in the Search ROI panel (the scrubber
        frame). Falls back to frame 0 when no sequence is loaded."""
        if self._seq is None or self._seq.n_frames == 0:
            return None
        idx = min(max(0, self._preview_frame_idx), self._seq.n_frames - 1)
        return self._seq.frame(idx)

    def _keep_mask(self, pts, image=None):
        """Boolean keep-mask for the given grid points (all True if masking is
        off or no sequence). ``image`` defaults to frame 0 so the DIC
        computation keeps its reference-frame mask; the preview passes the
        currently displayed frame to show the mask per frame."""
        if pts is None or len(pts) == 0 or not self.chk_mask.isChecked() \
                or self._seq is None or self._seq.n_frames == 0:
            return np.ones(len(pts) if pts is not None else 0, bool)
        if image is None:
            image = self._seq.frame(0)
        win = max(5, int(self.sp_subset.value()) // 2)
        return dic_engine.point_mask(
            image, pts, win=win,
            min_intensity=float(self.sld_int.value()))

    def _build_run_group(self):
        g = QGroupBox("Run / save")
        v = QVBoxLayout(g)
        self.b_run = QPushButton("Run DIC")
        self.b_run.setEnabled(False)
        self.b_run.setToolTip("Validate the ROI first.")
        self.b_run.clicked.connect(self._run_clicked)
        v.addWidget(self.b_run)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        row = QHBoxLayout()
        self.b_save = QPushButton("Save field…"); self.b_save.setEnabled(False)
        self.b_save.clicked.connect(self._save_clicked)
        self.b_import = QPushButton("Import field…")
        self.b_import.clicked.connect(self._import_clicked)
        row.addWidget(self.b_save); row.addWidget(self.b_import)
        v.addLayout(row)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:#666;"); self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)
        return g

    # =====================================================================
    # Sequence + scale
    # =====================================================================
    def _params(self) -> dic_engine.DicParams:
        return dic_engine.DicParams(
            engine=self.cb_engine.currentData(),
            subset=int(self.sp_subset.value()), step=int(self.sp_step.value()),
            search=int(self.sp_search.value()), zncc_min=float(self.sp_zncc.value()),
            subpixel=True)

    def _global_params(self) -> "dic_global_engine.DicGlobalParams":
        tool_poly = None
        ref_geo = getattr(self.session, "reference_geometry", None) or {}
        if isinstance(ref_geo, dict):
            tp = ref_geo.get("tool_polygon_px")
            if tp and len(tp) >= 3:
                tool_poly = [list(map(float, v)) for v in tp]
        return dic_global_engine.DicGlobalParams(
            elem_size=int(self.sp_elem.value()),
            variant=str(self.cb_variant.currentData()),
            incremental=bool(self.cb_pattern.currentData()),
            u_init_previous=bool(self.chk_uinit.isChecked()),
            max_iter=int(self.sp_maxiter.value()),
            tol=float(self.sp_tol.value()),
            pyramid_levels=int(self.sp_pyr_levels.value()),
            pyramid_sigma=float(self.sp_pyr_sigma.value()),
            mask_enabled=bool(self.chk_mask.isChecked()),
            mask_min_intensity=float(self.sld_int.value()),
            coverage_threshold=float(self.sp_coverage.value()),
            convect=bool(self.chk_convect.isChecked()),
            tool_polygon=tool_poly)

    def _load_from_session(self):
        path = self.session.visible.path
        if not path:
            QMessageBox.information(self, "Sequence", "No visible sequence in "
                                    "the session (set it in Acquisition).")
            return
        self._load_path(path, self.session.visible.fps)

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if d:
            self._load_path(d, self.spin_fps.value())

    def _load_path(self, path, fps):
        try:
            seq = ImageSequence.from_path(path, fps=fps)
        except Exception as e:
            QMessageBox.warning(self, "Sequence", "Could not load:\n%s" % e)
            return
        self.set_sequence(seq, source=str(path))

    def set_sequence(self, seq, source="<seq>"):
        self._seq = seq
        n = seq.n_frames
        self.lbl_seq.setText("%s  (%d frames)" % (source, n))
        self.spin_fps.setValue(float(seq.fps))
        # Configure the preview scrubber for this sequence.
        self._preview_frame_idx = 0
        self.sld_frame.blockSignals(True)
        self.sld_frame.setRange(0, max(0, n - 1))
        self.sld_frame.setValue(0)
        self.sld_frame.setEnabled(n > 1)
        self.sld_frame.blockSignals(False)
        self.lbl_frame_idx.setText("0/%d" % (max(0, n - 1)))
        self._show_reference_frame()

    def _on_preview_frame_changed(self, idx):
        """Scrub to another frame: redraw the image and recompute the mask /
        mesh overlay on it. Disabled once the ROI is validated (the geometry is
        frozen), but the image can still be browsed."""
        self._preview_frame_idx = int(idx)
        n = self._seq.n_frames if self._seq is not None else 0
        self.lbl_frame_idx.setText("%d/%d" % (int(idx), max(0, n - 1)))
        self._show_reference_frame()

    def _show_reference_frame(self):
        self._roi_ax.clear(); self._roi_ax.set_axis_off()
        self._preview_artists = []
        if self._seq is not None and self._seq.n_frames:
            img = self._preview_frame_image()
            is_rgb = (img.ndim == 3 and img.shape[-1] in (3, 4))
            self._roi_ax.imshow(img, cmap=None if is_rgb else "gray")
            # The rectangle selector is only meaningful while the ROI is not
            # locked; rebuild it on each draw so it stays attached to the axes.
            if not self._roi_locked:
                self._selector = RectangleSelector(
                    self._roi_ax, self._on_roi_select, useblit=False,
                    interactive=True, button=[1])
        self._preview_points()

    def _on_roi_select(self, eclick, erelease):
        if self._roi_locked:
            return
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        self.set_roi((min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))

    def set_roi(self, roi):
        self._roi = tuple(float(v) for v in roi)
        self.lbl_roi.setText("ROI: x=%.0f y=%.0f w=%.0f h=%.0f px" % self._roi)
        self._preview_points()

    def _grid_preview(self):
        if self._roi is None:
            return None
        p = self._params()
        margin = p.subset // 2 + p.search
        return dic_engine.make_grid(self._roi, p.step, margin=margin)

    def _preview_mesh(self):
        """Draw the Q4 element lines for the current ROI and element size
        (global engine). Trimming follows build_mesh_on_roi, so the drawn mesh
        is exactly what the solver will use. When the mask is enabled, elements
        excluded by material coverage are shaded (red) on the current frame."""
        if self._roi is None:
            return
        try:
            mesh = dic_global_engine.build_mesh_on_roi(
                self._roi, int(self.sp_elem.value()))
        except ValueError:
            self.lbl_roi.setText(
                "ROI: x=%.0f y=%.0f w=%.0f h=%.0f px  (too small for element "
                "size)" % self._roi)
            return
        color = "lime" if self._roi_locked else "cyan"
        nx = mesh.n_elem_x + 1
        ny = mesh.n_elem_y + 1
        nodes = mesh.nodes.reshape(ny, nx, 2)

        # Element coverage shading (mask on): compute validity on the displayed
        # frame with the same intensity threshold as the solver.
        valid_elements = None
        n_valid = mesh.n_elements
        if self.chk_mask.isChecked():
            img = self._preview_frame_image()
            if img is not None:
                mat = dic_global_engine.material_mask_intensity(
                    img, float(self.sld_int.value()))
                valid_elements = dic_global_engine.element_coverage_mask(
                    mesh, mat, float(self.sp_coverage.value()))
                n_valid = int(valid_elements.sum())
                from matplotlib.patches import Polygon
                for e in range(mesh.n_elements):
                    if valid_elements[e]:
                        continue
                    coords = mesh.nodes[mesh.connectivity[e]]
                    poly = Polygon(coords, closed=True, facecolor="red",
                                   edgecolor="none", alpha=0.25)
                    self._roi_ax.add_patch(poly)
                    self._preview_artists.append(poly)

        for j in range(nx):
            ln, = self._roi_ax.plot(nodes[:, j, 0], nodes[:, j, 1], "-",
                                    color=color, lw=0.6, alpha=0.8)
            self._preview_artists.append(ln)
        for i in range(ny):
            ln, = self._roi_ax.plot(nodes[i, :, 0], nodes[i, :, 1], "-",
                                    color=color, lw=0.6, alpha=0.8)
            self._preview_artists.append(ln)

        if valid_elements is not None:
            self.lbl_roi.setText(
                "ROI: x=%.0f y=%.0f w=%.0f h=%.0f px  (%dx%d Q4, %d/%d elements "
                "kept)" % (self._roi + (mesh.n_elem_x, mesh.n_elem_y,
                                        n_valid, mesh.n_elements)))
        else:
            self.lbl_roi.setText(
                "ROI: x=%.0f y=%.0f w=%.0f h=%.0f px  (%dx%d Q4 elements, "
                "%d nodes)" % (self._roi + (mesh.n_elem_x, mesh.n_elem_y,
                                            mesh.n_nodes)))

    def _preview_points(self, *_):
        """Overlay the measurement geometry on the reference frame. Local
        engine: the subset point grid (kept cyan/green, excluded red). Global
        engine: the Q4 mesh element lines."""
        if not hasattr(self, "_roi_ax"):
            return
        for a in self._preview_artists:
            try:
                a.remove()
            except Exception:
                log_swallowed("removing a DIC preview artist",
                              level=logging.DEBUG)
        self._preview_artists = []
        if self._is_global():
            self._preview_mesh()
            self._roi_canvas.draw_idle()
            return
        pts = self._grid_preview()
        if pts is not None and len(pts):
            keep = self._keep_mask(pts, self._preview_frame_image())
            kept = pts[keep]; excl = pts[~keep]
            color = "lime" if self._roi_locked else "cyan"
            if len(kept):
                sc = self._roi_ax.scatter(kept[:, 0], kept[:, 1], s=4,
                                          c=color, marker=".")
                self._preview_artists.append(sc)
            if len(excl):
                sx = self._roi_ax.scatter(excl[:, 0], excl[:, 1], s=4,
                                          c="red", marker="x")
                self._preview_artists.append(sx)
            self.lbl_roi.setText("ROI: x=%.0f y=%.0f w=%.0f h=%.0f px  (%d/%d pts)"
                                 % (self._roi + (int(keep.sum()), len(pts))))
            if hasattr(self, "lbl_mask"):
                self.lbl_mask.setText("%d points kept" % int(keep.sum()))
        if self._roi_locked and self._roi is not None:
            x, y, w, h = self._roi
            ln, = self._roi_ax.plot([x, x + w, x + w, x, x],
                                    [y, y, y + h, y + h, y], "-",
                                    color="lime", lw=1.5)
            self._preview_artists.append(ln)
        self._roi_canvas.draw_idle()

    def _image_size(self):
        if self._seq is None or self._seq.n_frames == 0:
            return None
        f0 = self._seq.frame(0)
        return int(f0.shape[1]), int(f0.shape[0])      # (w, h)

    def _nudge_roi(self, dx, dy):
        if self._roi is None or self._roi_locked:
            return
        x, y, w, h = self._roi
        nx, ny = x + dx, y + dy
        wh = self._image_size()
        if wh is not None:                              # keep ROI inside image
            iw, ih = wh
            nx = min(max(0.0, nx), max(0.0, iw - w))
            ny = min(max(0.0, ny), max(0.0, ih - h))
        self.set_roi((nx, ny, w, h))
        if self._selector is not None:
            try:
                self._selector.extents = (nx, nx + w, ny, ny + h)
            except Exception:
                log_swallowed("nudge ROI selector")

    def _lock_params(self, lock):
        """Freeze engine + mask parameters (they define the frozen grid/mask)."""
        widgets = [getattr(self, n, None) for n in (
            "cb_engine", "sp_subset", "sp_step", "sp_search", "sp_zncc",
            "chk_mask", "sld_int",
            "sp_elem", "cb_variant", "cb_pattern", "chk_uinit", "chk_uncert",
            "sp_maxiter", "sp_tol", "sp_pyr_levels", "sp_pyr_sigma",
            "sp_coverage", "chk_convect")]
        for w in widgets:
            if w is not None:
                w.setEnabled(not lock)

    def _on_validate(self, on):
        self._roi_locked = bool(on)
        self.b_validate.setText("ROI validated \u2713" if on else "Validate ROI")
        if self._selector is not None:
            self._selector.set_active(not on)
        for b in self._nudge_btns:
            b.setEnabled(not on)
        self._lock_params(on)
        # Run requires a validated ROI
        self.b_run.setEnabled(on)
        self._preview_points()

    def _prefill_scale(self):
        s = (self.session.visible_calibration or {}).get("scale_mm_per_px")
        if s:
            self.cb_scale_unit.setCurrentText("mm/px")
            self.spin_scale.setValue(float(s))

    # =====================================================================
    # Run / save / import
    # =====================================================================
    def set_frames(self, frames, fps=None):
        """Test/headless helper: inject an in-memory sequence."""
        arr = np.asarray(frames)
        seq = ImageSequence.from_array(
            arr, fps=float(fps) if fps else self.spin_fps.value())
        self.set_sequence(seq, source="<array>")

    def _grid_for_run(self):
        if self._seq is None or self._seq.n_frames < 2:
            raise ValueError("need a sequence of at least 2 frames")
        if self._roi is None:
            f0 = self._seq.frame(0)
            self._roi = (0.0, 0.0, float(f0.shape[1]), float(f0.shape[0]))
        p = self._params()
        margin = p.subset // 2 + p.search
        pts = dic_engine.make_grid(self._roi, p.step, margin=margin)
        if len(pts) == 0:
            raise ValueError("ROI too small for the subset/search/step")
        return pts

    def _roi_for_run(self):
        """ROI as (x, y, w, h) in px for the global engine (which builds its
        own mesh). Falls back to the whole image if no ROI was drawn."""
        if self._seq is None or self._seq.n_frames < 2:
            raise ValueError("need a sequence of at least 2 frames")
        if self._roi is None:
            f0 = self._seq.frame(0)
            self._roi = (0.0, 0.0, float(f0.shape[1]), float(f0.shape[0]))
        # Validate the ROI can hold at least one element (clear error early).
        dic_global_engine.build_mesh_on_roi(self._roi, int(self.sp_elem.value()))
        return self._roi

    def _build_meta(self, w, h):
        meta = {
            "source": "computed",
            "fps": float(self.spin_fps.value()),
            "trigger_offset_s": float(self.session.trigger_offset_s),
            "mm_per_px": self._mm_per_px(),
            "image_size": [int(w), int(h)],
            "roi_px": list(self._roi),
            "engine": self.cb_engine.currentData(),
        }
        if self._is_global():
            meta["dic_global"] = self._global_params().to_json_dict()
        else:
            meta["dic"] = self._params().to_json_dict()
            meta["mask"] = ({"min_intensity": float(self.sld_int.value())}
                            if self.chk_mask.isChecked() else None)
        return meta

    def run(self):
        """Synchronous compute (used by tests). The GUI uses the threaded path."""
        f0 = self._seq.frame(0)
        h, w = f0.shape[0], f0.shape[1]
        if self._is_global():
            roi = self._roi_for_run()
            res = dic_global_engine.compute_dic_global_fields(
                self._seq, roi, self._global_params(),
                fps=float(self.spin_fps.value()), mm_per_px=self._mm_per_px(),
                img_w=w, img_h=h,
                trigger_offset_s=float(self.session.trigger_offset_s))
        else:
            pts = self._grid_for_run()
            keep = self._keep_mask(pts)
            res = dic_engine.compute_dic_fields(
                self._seq, pts, self._params(), fps=float(self.spin_fps.value()),
                mm_per_px=self._mm_per_px(), img_w=w, img_h=h,
                trigger_offset_s=float(self.session.trigger_offset_s),
                point_keep=keep, mask_per_frame=self.chk_mask.isChecked(),
                mask_params=self._mask_params())
        self._result = res
        self._result_meta = self._build_meta(w, h)
        return res

    def _mask_params(self):
        """Mask parameters (intensity/window) for per-frame masking, or None
        when masking is off. Intensity-only (texture criterion removed)."""
        if not self.chk_mask.isChecked():
            return None
        return {"min_intensity": float(self.sld_int.value()),
                "win": max(5, int(self.sp_subset.value()) // 2)}

    def _set_busy(self, busy):
        self.b_run.setEnabled(not busy and self._roi_locked)
        self.b_validate.setEnabled(not busy)   # cannot un-validate while running
        self.b_import.setEnabled(not busy)
        self.progress.setVisible(busy)

    def _run_clicked(self):
        if not self._roi_locked:
            QMessageBox.information(self, "Run DIC", "Validate the ROI first.")
            return
        f0 = self._seq.frame(0)
        h, w = f0.shape[0], f0.shape[1]
        self.progress.setRange(0, self._seq.n_frames - 1)
        self.progress.setValue(0)

        if self._is_global():
            try:
                roi = self._roi_for_run()
            except Exception as e:
                QMessageBox.warning(self, "Run DIC", str(e))
                return
            self._pending_meta = self._build_meta(w, h)
            self._set_busy(True)
            self.lbl_status.setText("Computing (global Q4)…")
            self._worker = _DicGlobalWorker(
                self._seq, roi, self._global_params(),
                float(self.spin_fps.value()), self._mm_per_px(), w, h,
                float(self.session.trigger_offset_s), None)
        else:
            try:
                pts = self._grid_for_run()
            except Exception as e:
                QMessageBox.warning(self, "Run DIC", str(e))
                return
            keep = self._keep_mask(pts)
            self._pending_meta = self._build_meta(w, h)
            self._set_busy(True)
            self.lbl_status.setText("Computing…")
            self._worker = _DicWorker(
                self._seq, pts, self._params(), float(self.spin_fps.value()),
                self._mm_per_px(), w, h, float(self.session.trigger_offset_s),
                keep, mask_per_frame=self.chk_mask.isChecked(),
                mask_params=self._mask_params())
        self._worker.sig_progress.connect(lambda i, n: self.progress.setValue(i))
        if hasattr(self._worker, "sig_log"):
            self._worker.sig_log.connect(self._on_dic_log)
        self._worker.sig_done.connect(self._on_dic_done)
        self._worker.sig_failed.connect(self._on_dic_failed)
        self._worker.start()

    def _on_dic_log(self, msg):
        """Show the latest per-frame status line from the global worker."""
        self.lbl_status.setText(msg)

    def _on_dic_done(self, res):
        self._result = res
        self._result_meta = self._pending_meta
        self.viewer.set_field(res["x"], res["y"], res["t"], res["fields"],
                              valid=res["valid"], units=res["units"])
        f0 = self._seq.frame(0)
        self.viewer.set_background(self._seq.frame, self._mm_per_px(),
                                   f0.shape[1], f0.shape[0])
        self.b_save.setEnabled(True)
        self._set_busy(False)
        self.lbl_status.setText("Computed %d frames x %d points."
                                % (res["t"].size, res["x"].size))

    def _on_dic_failed(self, msg):
        self._set_busy(False)
        self.lbl_status.setText("")
        QMessageBox.warning(self, "Run DIC", msg)

    def _save_clicked(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save DIC field", "%s_dic.npz" % self.session.name,
            "NumPy npz (*.npz)")
        if not path:
            return
        self.save_field(path)
        QMessageBox.information(self, "Save", "DIC field saved.")

    def save_field(self, path):
        r = self._result
        f = r["fields"]
        extra = {k: v for k, v in f.items()
                 if k not in ("Vx", "Vy", "Vmag")}
        p = save_dic_field(path, r["x"], r["y"], r["t"],
                           f["Vx"], f["Vy"], f["Vmag"], r["valid"],
                           meta=self._result_meta, extra=extra, units=r["units"])
        self.session.dic_field_path = str(p)
        return p

    def _import_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import DIC field", "", "NumPy npz (*.npz)")
        if not path:
            return
        try:
            d = load_dic_field(path)
        except Exception as e:
            QMessageBox.warning(self, "Import", "Could not load:\n%s" % e)
            return
        skip = {"x", "y", "t", "valid", "meta"}
        comps = {k: d[k] for k in d if k not in skip}
        units = d.get("meta", {}).get("units", {})
        self.viewer.set_field(d["x"], d["y"], d["t"], comps,
                              valid=d.get("valid"), units=units)
        self.session.dic_field_path = str(path)
        self.lbl_status.setText("Imported %s" % os.path.basename(path))
