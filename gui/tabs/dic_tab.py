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
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, QSplitter,
    QFileDialog, QMessageBox, QProgressBar, QGridLayout, QSlider,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector

from gui.core.experiment_session import ExperimentSession
from gui.core.sequence_io import ImageSequence
from gui.core import dic as dic_engine
from gui.core.exp_field_io import save_dic_field, load_dic_field
from gui.widgets.dic_field_viewer import DicFieldViewer
from gui.widgets.num_input import DecimalSpinBox, WheelStepSlider
from gui.core.logging_util import log_swallowed


class _DicWorker(QThread):
    """Runs compute_dic_fields off the UI thread, reporting per-pair progress."""
    sig_progress = Signal(int, int)
    sig_done = Signal(object)
    sig_failed = Signal(str)

    def __init__(self, seq, pts, params, fps, mmpp, w, h, trig, keep=None):
        super().__init__()
        self._args = (seq, pts, params, fps, mmpp, w, h, trig, keep)

    def run(self):
        seq, pts, params, fps, mmpp, w, h, trig, keep = self._args
        try:
            res = dic_engine.compute_dic_fields(
                seq, pts, params, fps=fps, mm_per_px=mmpp, img_w=w, img_h=h,
                trigger_offset_s=trig, point_keep=keep,
                progress=lambda i, n: self.sig_progress.emit(i, n))
            self.sig_done.emit(res)
        except Exception as e:                       # pragma: no cover
            self.sig_failed.emit(str(e))


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
        bottom.addWidget(self._build_params_group(), stretch=2)
        bottom.addWidget(self._build_mask_group(), stretch=1)
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
        g = QGroupBox("Engine & parameters")
        grid = QGridLayout(g)
        self.cb_engine = QComboBox()
        self.cb_engine.addItem("Local (subset ZNCC)", "local")
        self.cb_engine.addItem("Global Q4 (q4dic) — later", "global")
        self.cb_engine.model().item(1).setEnabled(False)
        grid.addWidget(QLabel("Engine:"), 0, 0)
        grid.addWidget(self.cb_engine, 0, 1, 1, 3)

        self.sp_subset = QSpinBox(); self.sp_subset.setRange(5, 301)
        self.sp_subset.setSingleStep(2); self.sp_subset.setValue(31)
        self.sp_subset.setSuffix(" px")
        self.sp_step = QSpinBox(); self.sp_step.setRange(1, 200)
        self.sp_step.setValue(16); self.sp_step.setSuffix(" px")
        grid.addWidget(QLabel("Subset:"), 1, 0); grid.addWidget(self.sp_subset, 1, 1)
        grid.addWidget(QLabel("Step:"), 1, 2); grid.addWidget(self.sp_step, 1, 3)

        self.sp_search = QSpinBox(); self.sp_search.setRange(1, 200)
        self.sp_search.setValue(16); self.sp_search.setSuffix(" px")
        self.sp_zncc = DecimalSpinBox(); self.sp_zncc.setRange(0.0, 1.0)
        self.sp_zncc.setSingleStep(0.05); self.sp_zncc.setValue(0.5)
        grid.addWidget(QLabel("Search:"), 2, 0); grid.addWidget(self.sp_search, 2, 1)
        grid.addWidget(QLabel("ZNCC min:"), 2, 2); grid.addWidget(self.sp_zncc, 2, 3)

        for sp in (self.sp_subset, self.sp_step, self.sp_search):
            sp.valueChanged.connect(self._preview_points)
        return g

    def _build_mask_group(self):
        g = QGroupBox("Mask (background / tool)")
        form = QFormLayout(g)
        self.chk_mask = QCheckBox("Enable mask")
        self.chk_mask.toggled.connect(self._preview_points)
        form.addRow(self.chk_mask)
        self.sld_int = WheelStepSlider(Qt.Orientation.Horizontal)
        self.sld_int.setRange(0, 255); self.sld_int.setValue(25)
        self.sld_int.setSingleStep(1); self.sld_int.setPageStep(1)
        self.sld_int.valueChanged.connect(self._preview_points)
        self.lbl_int = QLabel("25")
        self.sld_int.valueChanged.connect(lambda v: self.lbl_int.setText(str(v)))
        ri = QHBoxLayout(); ri.addWidget(self.sld_int, 1); ri.addWidget(self.lbl_int)
        wi = QWidget(); wi.setLayout(ri)
        form.addRow("Min intensity:", wi)
        self.sld_tex = WheelStepSlider(Qt.Orientation.Horizontal)
        self.sld_tex.setRange(0, 80); self.sld_tex.setValue(8)
        self.sld_tex.setSingleStep(1); self.sld_tex.setPageStep(1)
        self.sld_tex.valueChanged.connect(self._preview_points)
        self.lbl_tex = QLabel("8")
        self.sld_tex.valueChanged.connect(lambda v: self.lbl_tex.setText(str(v)))
        rt = QHBoxLayout(); rt.addWidget(self.sld_tex, 1); rt.addWidget(self.lbl_tex)
        wt = QWidget(); wt.setLayout(rt)
        form.addRow("Min texture:", wt)
        self.lbl_mask = QLabel("")
        self.lbl_mask.setStyleSheet("color:#666;")
        form.addRow(self.lbl_mask)
        return g

    def _keep_mask(self, pts):
        """Boolean keep-mask for the given grid points (all True if masking is
        off or no sequence)."""
        if pts is None or len(pts) == 0 or not self.chk_mask.isChecked() \
                or self._seq is None or self._seq.n_frames == 0:
            return np.ones(len(pts) if pts is not None else 0, bool)
        win = max(5, int(self.sp_subset.value()) // 2)
        return dic_engine.point_mask(
            self._seq.frame(0), pts, win=win,
            min_intensity=float(self.sld_int.value()),
            min_texture=float(self.sld_tex.value()))

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
        self._show_reference_frame()

    def _show_reference_frame(self):
        self._roi_ax.clear(); self._roi_ax.set_axis_off()
        self._preview_artists = []
        if self._seq is not None and self._seq.n_frames:
            img = self._seq.frame(0)
            is_rgb = (img.ndim == 3 and img.shape[-1] in (3, 4))
            self._roi_ax.imshow(img, cmap=None if is_rgb else "gray")
            self._selector = RectangleSelector(
                self._roi_ax, self._on_roi_select, useblit=False,
                interactive=True, button=[1])
        self._roi_canvas.draw_idle()

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

    def _preview_points(self, *_):
        """Overlay the measurement grid on the reference frame: kept points
        (cyan, green once validated) and, when masking, excluded points (red)."""
        if not hasattr(self, "_roi_ax"):
            return
        for a in self._preview_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._preview_artists = []
        pts = self._grid_preview()
        if pts is not None and len(pts):
            keep = self._keep_mask(pts)
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
            "chk_mask", "sld_int", "sld_tex")]
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
        self._seq = ImageSequence.from_array(
            arr, fps=float(fps) if fps else self.spin_fps.value())
        self.lbl_seq.setText("<array>  (%d frames)" % self._seq.n_frames)

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

    def _build_meta(self, w, h):
        return {
            "source": "computed",
            "fps": float(self.spin_fps.value()),
            "trigger_offset_s": float(self.session.trigger_offset_s),
            "mm_per_px": self._mm_per_px(),
            "image_size": [int(w), int(h)],
            "roi_px": list(self._roi),
            "dic": self._params().to_json_dict(),
            "mask": ({"min_intensity": float(self.sld_int.value()),
                      "min_texture": float(self.sld_tex.value())}
                     if self.chk_mask.isChecked() else None),
        }

    def run(self):
        """Synchronous compute (used by tests). The GUI uses the threaded path."""
        pts = self._grid_for_run()
        keep = self._keep_mask(pts)
        f0 = self._seq.frame(0)
        h, w = f0.shape[0], f0.shape[1]
        res = dic_engine.compute_dic_fields(
            self._seq, pts, self._params(), fps=float(self.spin_fps.value()),
            mm_per_px=self._mm_per_px(), img_w=w, img_h=h,
            trigger_offset_s=float(self.session.trigger_offset_s),
            point_keep=keep)
        self._result = res
        self._result_meta = self._build_meta(w, h)
        return res

    def _set_busy(self, busy):
        self.b_run.setEnabled(not busy and self._roi_locked)
        self.b_validate.setEnabled(not busy)   # cannot un-validate while running
        self.b_import.setEnabled(not busy)
        self.progress.setVisible(busy)

    def _run_clicked(self):
        if not self._roi_locked:
            QMessageBox.information(self, "Run DIC", "Validate the ROI first.")
            return
        try:
            pts = self._grid_for_run()
        except Exception as e:
            QMessageBox.warning(self, "Run DIC", str(e))
            return
        keep = self._keep_mask(pts)
        f0 = self._seq.frame(0)
        h, w = f0.shape[0], f0.shape[1]
        self._pending_meta = self._build_meta(w, h)
        self.progress.setRange(0, self._seq.n_frames - 1)
        self.progress.setValue(0)
        self._set_busy(True)
        self.lbl_status.setText("Computing…")
        self._worker = _DicWorker(
            self._seq, pts, self._params(), float(self.spin_fps.value()),
            self._mm_per_px(), w, h, float(self.session.trigger_offset_s), keep)
        self._worker.sig_progress.connect(lambda i, n: self.progress.setValue(i))
        self._worker.sig_done.connect(self._on_dic_done)
        self._worker.sig_failed.connect(self._on_dic_failed)
        self._worker.start()

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
