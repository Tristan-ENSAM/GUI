# -*- coding: utf-8 -*-
"""
Experimental Data — tab container + the first sub-tab (Acquisition / Import).

ExperimentalDataTab owns one ExperimentSession and hosts the experimental
workflow as an inner QTabWidget. Only Acquisition / Import is implemented for
now; the other stages are placeholders.

AcquisitionTab is the entry point: it loads/saves the session file, edits the
session metadata, points at the raw streams (visible, IR, forces) and their
no-load counterparts, and previews each stream. Temporal sync is assumed done
upstream by the hardware trigger, so a frame's time is just
trigger_offset + index / fps.
"""
from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QDoubleSpinBox, QSpinBox, QSplitter, QTabWidget,
    QFileDialog, QMessageBox, QScrollArea, QFrame,
)

from gui.core.experiment_session import ExperimentSession
from gui.core.sequence_io import ImageSequence, load_forces
from gui.core.logging_util import log_swallowed
from gui.widgets.image_sequence_viewer import ImageSequenceViewer
from gui.widgets.force_viewer import ForceViewer
from gui.tabs.calibration_visible_tab import CalibrationVisibleTab
from gui.tabs.alignment_tab import AlignmentTab
from gui.tabs.dic_tab import DICTab


def _placeholder(text: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #888; font-style: italic;")
    lay.addStretch()
    lay.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignHCenter)
    lay.addStretch()
    return w


class ExperimentalDataTab(QWidget):
    """Top 'Experimental Data' tab: owns the session, hosts the sub-tabs."""

    def __init__(self, write_geometry=None, parent=None):
        super().__init__(parent)
        self.session = ExperimentSession()

        self.acquisition_tab = AcquisitionTab(self.session)
        self.calib_visible_tab = CalibrationVisibleTab(self.session)
        self.alignment_tab = AlignmentTab(self.session, write_geometry=write_geometry)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.acquisition_tab, "Acquisition / Import")
        self.tabs.addTab(self.calib_visible_tab, "Calib. visible")
        self.tabs.addTab(_placeholder("Calibration — thermal camera (later)"),
                         "Calib. thermal")
        self.tabs.addTab(self.alignment_tab, "Alignment")
        self.dic_tab = DICTab(self.session)
        self.tabs.addTab(self.dic_tab, "DIC")
        self.tabs.addTab(_placeholder("IRT temperature fields (later)"), "IRT")
        self.tabs.addTab(_placeholder("Forces (later)"), "Forces")
        self.tabs.addTab(_placeholder("Noise / baseline (a vide) (later)"),
                         "Noise")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs)


class AcquisitionTab(QWidget):
    """Load/save the session and import + preview the raw streams."""

    sessionChanged = Signal()

    def __init__(self, session: ExperimentSession, parent=None):
        super().__init__(parent)
        self.session = session

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        # ---- left: session bar + import forms --------------------------
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.addWidget(self._build_session_bar())
        left_lay.addWidget(self._build_import_group())
        left_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(left)
        scroll.setMinimumWidth(380)
        splitter.addWidget(scroll)

        # ---- right: previews -------------------------------------------
        self.view_visible = ImageSequenceViewer("visible", cmap="gray")
        self.view_ir      = ImageSequenceViewer("IR", cmap="inferno")
        self.view_forces  = ForceViewer()
        self.preview_tabs = QTabWidget()
        self.preview_tabs.addTab(self.view_visible, "Visible")
        self.preview_tabs.addTab(self.view_ir, "IR")
        self.preview_tabs.addTab(self.view_forces, "Forces")
        splitter.addWidget(self.preview_tabs)

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
    def _build_session_bar(self) -> QGroupBox:
        g = QGroupBox("Test / session")
        form = QFormLayout(g)

        self.fld_name = QLineEdit(self.session.name)
        self.fld_material = QLineEdit(self.session.material)
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.0, 1e6)
        self.spin_speed.setDecimals(3)
        self.spin_speed.setSuffix(" mm/s")
        self.spin_speed.setValue(self.session.cutting_speed_nominal)
        self.spin_trigger = QDoubleSpinBox()
        self.spin_trigger.setRange(-1e6, 1e6)
        self.spin_trigger.setDecimals(6)
        self.spin_trigger.setSuffix(" s")
        self.spin_trigger.setToolTip(
            "Common t = 0 set by the hardware trigger; all streams share it.")
        self.spin_trigger.setValue(self.session.trigger_offset_s)
        self.fld_notes = QLineEdit(self.session.notes)

        for w in (self.fld_name, self.fld_material, self.fld_notes):
            w.editingFinished.connect(self._pull)
        self.spin_speed.valueChanged.connect(self._pull)
        self.spin_trigger.valueChanged.connect(self._pull)

        form.addRow("Name:", self.fld_name)
        form.addRow("Material:", self.fld_material)
        form.addRow("Nominal cutting speed:", self.spin_speed)
        form.addRow("Trigger offset (t0):", self.spin_trigger)
        form.addRow("Notes:", self.fld_notes)

        btns = QHBoxLayout()
        b_load = QPushButton("Load session…")
        b_load.clicked.connect(self._load_session)
        b_save = QPushButton("Save session…")
        b_save.clicked.connect(self._save_session)
        btns.addWidget(b_load)
        btns.addWidget(b_save)
        btns.addStretch(1)
        form.addRow(btns)
        return g

    def _build_import_group(self) -> QGroupBox:
        g = QGroupBox("Raw streams")
        lay = QVBoxLayout(g)
        lay.addWidget(self._build_image_row("visible", "Visible (Photron)",
                                            self.session.visible))
        lay.addWidget(self._build_image_row("ir", "IR (Telops)",
                                            self.session.ir))
        lay.addWidget(self._build_force_row())
        return g

    def _build_image_row(self, key: str, title: str, cfg) -> QGroupBox:
        g = QGroupBox(title)
        form = QFormLayout(g)

        fld_path = QLineEdit(cfg.path)
        fld_path.setReadOnly(True)
        b_folder = QPushButton("Folder…")
        b_folder.setToolTip("Load the folder containing the image sequence "
                            "(png, jpg, tiff, … one file per frame).")
        b_folder.clicked.connect(
            lambda: self._browse_image(key, noload=False))
        r1 = QHBoxLayout(); r1.addWidget(fld_path, 1)
        r1.addWidget(b_folder)
        w1 = QWidget(); w1.setLayout(r1)
        form.addRow("Cutting:", w1)

        fld_noload = QLineEdit(cfg.noload_path)
        fld_noload.setReadOnly(True)
        b_folder_nl = QPushButton("Folder…")
        b_folder_nl.clicked.connect(
            lambda: self._browse_image(key, noload=True))
        r2 = QHBoxLayout(); r2.addWidget(fld_noload, 1)
        r2.addWidget(b_folder_nl)
        w2 = QWidget(); w2.setLayout(r2)
        form.addRow("No-load (a vide):", w2)

        spin_fps = QDoubleSpinBox()
        spin_fps.setRange(1e-6, 1e9)
        spin_fps.setDecimals(2)
        spin_fps.setSuffix(" fps")
        spin_fps.setValue(cfg.fps)
        spin_fps.valueChanged.connect(self._pull)
        form.addRow("Frame rate:", spin_fps)

        # stash widgets for apply/pull
        setattr(self, "_%s_path" % key, fld_path)
        setattr(self, "_%s_noload" % key, fld_noload)
        setattr(self, "_%s_fps" % key, spin_fps)
        return g

    def _build_force_row(self) -> QGroupBox:
        g = QGroupBox("Forces (DAQ)")
        form = QFormLayout(g)
        cfg = self.session.forces

        self._forces_path = QLineEdit(cfg.path)
        self._forces_path.setReadOnly(True)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_forces(noload=False))
        r1 = QHBoxLayout(); r1.addWidget(self._forces_path, 1); r1.addWidget(b)
        w1 = QWidget(); w1.setLayout(r1)
        form.addRow("Cutting:", w1)

        self._forces_noload = QLineEdit(cfg.noload_path)
        self._forces_noload.setReadOnly(True)
        b2 = QPushButton("Browse…")
        b2.clicked.connect(lambda: self._browse_forces(noload=True))
        r2 = QHBoxLayout(); r2.addWidget(self._forces_noload, 1); r2.addWidget(b2)
        w2 = QWidget(); w2.setLayout(r2)
        form.addRow("No-load (a vide):", w2)

        self._forces_fps = QDoubleSpinBox()
        self._forces_fps.setRange(1e-6, 1e12)
        self._forces_fps.setDecimals(2)
        self._forces_fps.setSuffix(" Hz")
        self._forces_fps.setValue(cfg.fps)
        self._forces_fps.valueChanged.connect(self._pull)
        form.addRow("Sampling rate:", self._forces_fps)

        # column mapping
        self._col_t  = QSpinBox(); self._col_t.setRange(-1, 64); self._col_t.setValue(cfg.col_t)
        self._col_fc = QSpinBox(); self._col_fc.setRange(0, 64); self._col_fc.setValue(cfg.col_fc)
        self._col_ff = QSpinBox(); self._col_ff.setRange(0, 64); self._col_ff.setValue(cfg.col_ff)
        for s in (self._col_t, self._col_fc, self._col_ff):
            s.valueChanged.connect(self._pull)
        self._col_t.setToolTip("Column index of the time vector (-1 if none, "
                               "time derived from sampling rate).")
        cols = QHBoxLayout()
        cols.addWidget(QLabel("t:")); cols.addWidget(self._col_t)
        cols.addWidget(QLabel("Fc:")); cols.addWidget(self._col_fc)
        cols.addWidget(QLabel("Ff:")); cols.addWidget(self._col_ff)
        cols.addStretch(1)
        wc = QWidget(); wc.setLayout(cols)
        form.addRow("Columns:", wc)
        return g

    # =====================================================================
    # Session <-> widgets
    # =====================================================================
    def _pull(self, *_):
        s = self.session
        s.name = self.fld_name.text().strip() or "experiment"
        s.material = self.fld_material.text().strip()
        s.cutting_speed_nominal = float(self.spin_speed.value())
        s.trigger_offset_s = float(self.spin_trigger.value())
        s.notes = self.fld_notes.text().strip()
        s.visible.fps = float(self._visible_fps.value())
        s.ir.fps = float(self._ir_fps.value())
        s.forces.fps = float(self._forces_fps.value())
        s.forces.col_t = int(self._col_t.value())
        s.forces.col_fc = int(self._col_fc.value())
        s.forces.col_ff = int(self._col_ff.value())
        self.sessionChanged.emit()

    def apply_from_session(self):
        """Refresh all widgets from the current session, then (re)build the
        previews from whatever paths are set."""
        s = self.session
        widgets = [self.fld_name, self.fld_material, self.spin_speed,
                   self.spin_trigger, self.fld_notes,
                   self._visible_path, self._visible_noload, self._visible_fps,
                   self._ir_path, self._ir_noload, self._ir_fps,
                   self._forces_path, self._forces_noload, self._forces_fps,
                   self._col_t, self._col_fc, self._col_ff]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.fld_name.setText(s.name)
            self.fld_material.setText(s.material)
            self.spin_speed.setValue(s.cutting_speed_nominal)
            self.spin_trigger.setValue(s.trigger_offset_s)
            self.fld_notes.setText(s.notes)
            self._visible_path.setText(s.visible.path)
            self._visible_noload.setText(s.visible.noload_path)
            self._visible_fps.setValue(s.visible.fps)
            self._ir_path.setText(s.ir.path)
            self._ir_noload.setText(s.ir.noload_path)
            self._ir_fps.setValue(s.ir.fps)
            self._forces_path.setText(s.forces.path)
            self._forces_noload.setText(s.forces.noload_path)
            self._forces_fps.setValue(s.forces.fps)
            self._col_t.setValue(s.forces.col_t)
            self._col_fc.setValue(s.forces.col_fc)
            self._col_ff.setValue(s.forces.col_ff)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._refresh_visible()
        self._refresh_ir()
        self._refresh_forces()

    # =====================================================================
    # Browse + preview
    # =====================================================================
    def _browse_image(self, key: str, noload: bool):
        path = QFileDialog.getExistingDirectory(
            self, "Pick the folder containing the image sequence", "")
        if not path:
            return
        cfg = getattr(self.session, key)
        if noload:
            cfg.noload_path = path
            getattr(self, "_%s_noload" % key).setText(path)
        else:
            cfg.path = path
            getattr(self, "_%s_path" % key).setText(path)
        self.sessionChanged.emit()
        if not noload:
            (self._refresh_visible if key == "visible" else self._refresh_ir)()

    def _browse_forces(self, noload: bool):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a force file", "",
            "Text/CSV (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        if noload:
            self.session.forces.noload_path = path
            self._forces_noload.setText(path)
        else:
            self.session.forces.path = path
            self._forces_path.setText(path)
        self.sessionChanged.emit()
        if not noload:
            self._refresh_forces()

    def _refresh_visible(self):
        self._preview_image("visible", self.view_visible)

    def _refresh_ir(self):
        self._preview_image("ir", self.view_ir)

    def _preview_image(self, key: str, viewer):
        cfg = getattr(self.session, key)
        if not cfg.path:
            viewer.set_sequence(None)
            return
        try:
            seq = ImageSequence.from_path(
                cfg.path, fps=cfg.fps, t0=self.session.trigger_offset_s)
            viewer.set_sequence(seq)
        except Exception:
            # Auto-refresh (incl. on session load) must not block on a modal.
            log_swallowed("loading %s stream" % key)
            viewer.set_sequence(None)

    def _refresh_forces(self):
        cfg = self.session.forces
        if not cfg.path:
            self.view_forces.clear()
            return
        try:
            t, fc, ff = load_forces(cfg.path, fps=cfg.fps, col_t=cfg.col_t,
                                    col_fc=cfg.col_fc, col_ff=cfg.col_ff)
            self.view_forces.set_signal(t + self.session.trigger_offset_s,
                                        fc, ff)
        except Exception:
            log_swallowed("loading force signal")
            self.view_forces.clear()

    # =====================================================================
    # Load / save the session file
    # =====================================================================
    def _load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load experiment session", "",
            "Experiment files (*.json);;All files (*)")
        if not path:
            return
        try:
            loaded = ExperimentSession.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Load session",
                                 "Could not load session:\n%s" % e)
            return
        # copy fields into the shared session object (kept by reference)
        for f_name in vars(loaded):
            setattr(self.session, f_name, getattr(loaded, f_name))
        self.apply_from_session()
        self.sessionChanged.emit()

    def _save_session(self):
        self._pull()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save experiment session",
            "%s.json" % self.session.name,
            "Experiment files (*.json);;All files (*)")
        if not path:
            return
        try:
            written = self.session.save(path)
        except Exception as e:
            QMessageBox.critical(self, "Save session",
                                 "Could not save session:\n%s" % e)
            return
        QMessageBox.information(self, "Save session", "Saved to:\n%s" % written)
