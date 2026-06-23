# -*- coding: utf-8 -*-
"""
Calibration — visible camera (full Zhang).

Import several target images (different poses), pick the calibration target
(user-selectable: checkerboard or circle grid + dimensions + spacing),
choose which image lies in the cutting plane (reference), run the
calibration, review the results (camera matrix, distortion, mm/px, RMS) and
save them into the session's `visible_calibration`.

The calibration engine sits behind `VisibleCalibrationBackend`; here we use
the OpenCV implementation. A q4dic-based engine can be plugged in later.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QListWidget, QSplitter,
    QFileDialog, QMessageBox, QPlainTextEdit,
)

from gui.core.experiment_session import ExperimentSession
from gui.core.sequence_io import ImageSequence, _read_image
from gui.core.calibration import TargetSpec, OpenCVCalibrator, CalibrationResult
from gui.core.logging_util import log_swallowed
from gui.widgets.image_sequence_viewer import ImageSequenceViewer


_PATTERNS = [("Checkerboard (inner corners)", "checkerboard"),
             ("Circle grid — symmetric", "circles"),
             ("Circle grid — asymmetric", "circles_asym")]


class CalibrationVisibleTab(QWidget):
    calibrationChanged = Signal()

    def __init__(self, session: ExperimentSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._paths: list[str] = []
        self._result: CalibrationResult | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.addWidget(self._build_target_group())
        ll.addWidget(self._build_images_group())
        ll.addWidget(self._build_actions_group())
        ll.addStretch()
        left.setMinimumWidth(360)
        splitter.addWidget(left)

        self.viewer = ImageSequenceViewer("target", cmap="gray")
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 900])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        self.apply_from_session()

    # =====================================================================
    # Builders
    # =====================================================================
    def _build_target_group(self) -> QGroupBox:
        g = QGroupBox("Calibration target")
        form = QFormLayout(g)
        self.cb_pattern = QComboBox()
        for label, value in _PATTERNS:
            self.cb_pattern.addItem(label, value)
        self.spin_cols = QSpinBox(); self.spin_cols.setRange(2, 100); self.spin_cols.setValue(9)
        self.spin_rows = QSpinBox(); self.spin_rows.setRange(2, 100); self.spin_rows.setValue(6)
        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setRange(1e-4, 1e4); self.spin_spacing.setDecimals(4)
        self.spin_spacing.setValue(1.0); self.spin_spacing.setSuffix(" mm")
        self.spin_cols.setToolTip("Checkerboard: number of INNER corners across.\n"
                                  "Circle grid: number of circles across.")
        form.addRow("Pattern:", self.cb_pattern)
        form.addRow("Cols:", self.spin_cols)
        form.addRow("Rows:", self.spin_rows)
        form.addRow("Spacing:", self.spin_spacing)
        return g

    def _build_images_group(self) -> QGroupBox:
        g = QGroupBox("Target images (several poses)")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        b_add = QPushButton("Add images…")
        b_add.clicked.connect(self._add_images)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear_images)
        row.addWidget(b_add); row.addWidget(b_clear); row.addStretch(1)
        v.addLayout(row)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)
        v.addWidget(self.list)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference plane (cutting plane):"))
        self.cb_reference = QComboBox()
        self.cb_reference.setToolTip(
            "Image where the target lies in the cutting plane. mm/px and the "
            "pixel->mm homography are computed on this view.")
        ref_row.addWidget(self.cb_reference, 1)
        v.addLayout(ref_row)
        return g

    def _build_actions_group(self) -> QGroupBox:
        g = QGroupBox("Calibration")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.b_calibrate = QPushButton("Calibrate")
        self.b_calibrate.clicked.connect(self._calibrate)
        self.b_save = QPushButton("Save calibration")
        self.b_save.clicked.connect(self._save)
        self.b_save.setEnabled(False)
        row.addWidget(self.b_calibrate); row.addWidget(self.b_save); row.addStretch(1)
        v.addLayout(row)
        self.txt_result = QPlainTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setMaximumBlockCount(2000)
        self.txt_result.setPlaceholderText("Calibration results will appear here.")
        v.addWidget(self.txt_result)
        return g

    # =====================================================================
    # Target spec
    # =====================================================================
    def _target(self) -> TargetSpec:
        return TargetSpec(pattern=self.cb_pattern.currentData(),
                          cols=int(self.spin_cols.value()),
                          rows=int(self.spin_rows.value()),
                          spacing_mm=float(self.spin_spacing.value()))

    # =====================================================================
    # Images
    # =====================================================================
    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add target images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)")
        if not paths:
            return
        self._paths.extend(paths)
        self._refresh_list()

    def _clear_images(self):
        self._paths = []
        self._refresh_list()
        self.viewer.set_sequence(None)

    def _refresh_list(self):
        self.list.blockSignals(True)
        self.cb_reference.blockSignals(True)
        self.list.clear(); self.cb_reference.clear()
        for p in self._paths:
            self.list.addItem(Path(p).name)
            self.cb_reference.addItem(Path(p).name)
        self.list.blockSignals(False)
        self.cb_reference.blockSignals(False)
        if self._paths:
            self.list.setCurrentRow(0)

    def _on_select(self, row: int):
        if row < 0 or row >= len(self._paths):
            self.viewer.set_sequence(None)
            return
        try:
            img = _read_image(self._paths[row])
            self.viewer.set_sequence(ImageSequence.from_array(img))
        except Exception:
            # Auto-refresh must not pop a modal dialog (e.g. a restored
            # session whose image paths no longer exist).
            log_swallowed("previewing target image")
            self.viewer.set_sequence(None)

    # =====================================================================
    # Calibrate / save
    # =====================================================================
    def _calibrate(self):
        if len(self._paths) < 3:
            QMessageBox.warning(self, "Calibrate",
                                "Full Zhang calibration needs at least 3 target "
                                "images (poses).")
            return
        try:
            images = [_read_image(p) for p in self._paths]
        except Exception as e:
            QMessageBox.critical(self, "Calibrate", "Could not read images:\n%s" % e)
            return
        try:
            backend = OpenCVCalibrator()
            res = backend.calibrate(images, self._target(),
                                    reference_index=self.cb_reference.currentIndex())
        except Exception as e:
            QMessageBox.critical(self, "Calibrate", "Calibration failed:\n%s" % e)
            return
        res.images = list(self._paths)
        self._result = res
        self.b_save.setEnabled(True)
        self._show_result(res)

    def _show_result(self, res: CalibrationResult):
        K = np.asarray(res.camera_matrix)
        lines = [
            "Views used: %d / %d" % (res.n_views_used, len(self._paths)),
            "Reprojection RMS: %.4f px" % res.reproj_rms_px,
            "Scale: %.6g mm/px" % res.scale_mm_per_px,
            "Image size: %d x %d px" % (res.image_size[0], res.image_size[1]),
            "",
            "Camera matrix K:",
            "  fx=%.2f  fy=%.2f" % (K[0, 0], K[1, 1]),
            "  cx=%.2f  cy=%.2f" % (K[0, 2], K[1, 2]),
            "Distortion [k1 k2 p1 p2 k3]:",
            "  " + "  ".join("%.5g" % c for c in res.dist_coeffs[:5]),
            "Reference plane: image #%d (%s)" % (
                res.reference_index,
                Path(self._paths[res.reference_index]).name
                if 0 <= res.reference_index < len(self._paths) else "?"),
        ]
        if res.reproj_rms_px > 1.0:
            lines.append("")
            lines.append("Note: RMS > 1 px — add more / sharper poses for a "
                         "better fit.")
        self.txt_result.setPlainText("\n".join(lines))

    def _save(self):
        if self._result is None:
            return
        self.session.visible_calibration = self._result.to_dict()
        self.calibrationChanged.emit()
        QMessageBox.information(
            self, "Save calibration",
            "Calibration stored in the session (Save session to persist it).")

    # =====================================================================
    # External hooks
    # =====================================================================
    def apply_from_session(self):
        """Reflect a calibration already stored in the session (after loading
        a session file), without re-running anything."""
        cal = self.session.visible_calibration or {}
        tgt = TargetSpec.from_dict(cal.get("target"))
        idx = self.cb_pattern.findData(tgt.pattern)
        if idx >= 0:
            self.cb_pattern.setCurrentIndex(idx)
        self.spin_cols.setValue(tgt.cols)
        self.spin_rows.setValue(tgt.rows)
        self.spin_spacing.setValue(tgt.spacing_mm)
        self._paths = list(cal.get("images", []))
        self._refresh_list()
        if cal.get("camera_matrix"):
            try:
                self._result = CalibrationResult(
                    camera_matrix=cal["camera_matrix"],
                    dist_coeffs=cal["dist_coeffs"],
                    scale_mm_per_px=cal["scale_mm_per_px"],
                    homography=cal["homography"],
                    reproj_rms_px=cal["reproj_rms_px"],
                    image_size=cal["image_size"],
                    reference_index=cal.get("reference_index", 0),
                    n_views_used=cal.get("n_views_used", 0),
                    target=tgt.to_dict(), images=self._paths)
                self.b_save.setEnabled(True)
                self._show_result(self._result)
            except Exception:
                log_swallowed("restoring stored calibration", )
