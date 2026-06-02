# -*- coding: utf-8 -*-
"""
Preferences dialog: user-level settings (Abaqus paths, default workdir,
default temperature unit).

Designed to be lightweight and self-contained:
  - open with `dlg = PreferencesDialog(parent, prefs)`
  - if dlg.exec() returns Accepted, read dlg.result_prefs() and save_preferences(...)
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QDialogButtonBox, QFileDialog, QCheckBox, QGroupBox,
)

from gui.core.preferences import Preferences


class _PathField(QHBoxLayout):
    """A line-edit + 'Browse...' button on a single row. The dialog wraps
    each path field with this helper so the user can pick from a file
    chooser instead of typing the full path."""

    def __init__(self, initial: str, browse_kind: str = "file",
                 file_filter: str = "All files (*)"):
        super().__init__()
        self.edit = QLineEdit(initial)
        self.edit.setMinimumWidth(380)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse(browse_kind, file_filter))
        self.addWidget(self.edit, stretch=1)
        self.addWidget(btn)

    def _browse(self, kind: str, file_filter: str):
        if kind == "dir":
            path = QFileDialog.getExistingDirectory(
                None, "Pick a directory", self.edit.text() or str(Path.home()),
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, "Pick a file", self.edit.text() or str(Path.home()),
                file_filter,
            )
        if path:
            self.edit.setText(path)

    def value(self) -> str:
        return self.edit.text().strip()


class PreferencesDialog(QDialog):
    """Modal editor for user preferences. Doesn't write to disk by itself —
    callers should save the result after Accept."""

    def __init__(self, parent, prefs: Preferences):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(620)
        self._initial = prefs

        outer = QVBoxLayout(self)

        # ---- Abaqus paths group ----
        g_paths = QGroupBox("Abaqus executable and scripts")
        f_paths = QFormLayout(g_paths)

        self.fld_cmd     = _PathField(prefs.abaqus_cmd,
                                       "file", "Batch files (*.bat);;All files (*)")
        self.fld_script  = _PathField(prefs.abaqus_script,
                                       "file", "Python scripts (*.py);;All files (*)")
        self.fld_extract = _PathField(prefs.abaqus_extract_script,
                                       "file", "Python scripts (*.py);;All files (*)")

        f_paths.addRow("Abaqus command (.bat):",      self.fld_cmd)
        f_paths.addRow("Model-generator script:",     self.fld_script)
        f_paths.addRow("Extraction script (.odb):",   self.fld_extract)
        outer.addWidget(g_paths)

        note = QLabel(
            "Paths are stored in your user profile, not in the project file.\n"
            "Future versions will support remote execution on an HPC cluster\n"
            "where the local Abaqus paths aren't applicable."
        )
        note.setStyleSheet("color: #888; font-style: italic;")
        note.setWordWrap(True)
        outer.addWidget(note)

        # ---- Workdir group ----
        g_wd = QGroupBox("Default working directory")
        f_wd = QFormLayout(g_wd)
        self.fld_workdir = _PathField(prefs.default_workdir, "dir")
        f_wd.addRow("Default workdir:", self.fld_workdir)
        outer.addWidget(g_wd)

        # ---- Display group ----
        g_disp = QGroupBox("Display")
        f_disp = QVBoxLayout(g_disp)
        self.cb_kelvin_default = QCheckBox(
            "New profiles open with temperatures in Kelvin (default: °C)"
        )
        self.cb_kelvin_default.setChecked(prefs.temp_unit_default == "K")
        self.cb_kelvin_default.setToolTip(
            "Only affects the initial unit of FRESH profiles. Loaded files\n"
            "keep whatever unit they were saved with."
        )
        f_disp.addWidget(self.cb_kelvin_default)
        outer.addWidget(g_disp)

        # ---- OK / Cancel ----
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def result_prefs(self) -> Preferences:
        """Build a new Preferences instance from the dialog field values."""
        return replace(
            self._initial,
            abaqus_cmd            = self.fld_cmd.value(),
            abaqus_script         = self.fld_script.value(),
            abaqus_extract_script = self.fld_extract.value(),
            default_workdir       = self.fld_workdir.value(),
            temp_unit_default     = "K" if self.cb_kelvin_default.isChecked() else "C",
        )
