# -*- coding: utf-8 -*-
"""
Preferences dialog: user-level settings (Abaqus paths, default workdir,
default temperature unit).

Designed to be lightweight and self-contained:
  - open with `dlg = PreferencesDialog(parent, prefs)`
  - if dlg.exec() returns Accepted, read dlg.result_prefs() and save_preferences(...)
"""
from __future__ import annotations
import os
from dataclasses import replace
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QDialogButtonBox, QFileDialog, QCheckBox, QGroupBox,
    QComboBox, QHeaderView, QMessageBox, QTableWidget, QTableWidgetItem,
)

from gui.core.machines import (
    LOCAL_NAME, QUEUE, Machine, default_name, machines)
from gui.core.preferences import Preferences


def _native_path(p: str) -> str:
    """Display/store paths with the OS-native separator. Qt's file chooser
    returns forward slashes on Windows; normalise so the dialog shows (and
    saves) consistent backslashes there."""
    p = (p or "").strip()
    return os.path.normpath(p) if p else p


class _PathField(QHBoxLayout):
    """A line-edit + 'Browse...' button on a single row. The dialog wraps
    each path field with this helper so the user can pick from a file
    chooser instead of typing the full path."""

    def __init__(self, initial: str, browse_kind: str = "file",
                 file_filter: str = "All files (*)"):
        super().__init__()
        self.edit = QLineEdit(_native_path(initial))
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
            self.edit.setText(_native_path(path))

    def value(self) -> str:
        return _native_path(self.edit.text())


class PreferencesDialog(QDialog):
    """Modal editor for user preferences. Doesn't write to disk by itself —
    callers should save the result after Accept."""

    def __init__(self, parent, prefs: Preferences):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(720)
        self._initial = prefs

        outer = QVBoxLayout(self)

        # ---- Abaqus paths group ----
        g_paths = QGroupBox("Abaqus executable and scripts")
        f_paths = QFormLayout(g_paths)

        self.fld_cmd     = _PathField(prefs.abaqus_cmd,
                                       "file", "Batch files (*.bat);;All files (*)")
        self.fld_script  = _PathField(prefs.abaqus_script,
                                       "file", "Python scripts (*.py);;All files (*)")

        f_paths.addRow("Abaqus command (.bat):",      self.fld_cmd)
        f_paths.addRow("Model-generator script:",     self.fld_script)
        outer.addWidget(g_paths)

        note = QLabel(
            "Paths are stored in your user profile, not in the project file.\n"
            "On the other machines the model-generator script above is still\n"
            "the one used: it is copied to their queue folder."
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

        # ---- Simulation database group ----
        g_db = QGroupBox("Simulation database")
        f_db = QFormLayout(g_db)
        self.fld_run_db = _PathField(getattr(prefs, "run_db_dir", ""), "dir")
        self.fld_run_db.edit.setToolTip(
            "Folder (e.g. on an external drive) where finished runs are kept\n"
            "and looked up by every study, whatever its folder. Empty = no\n"
            "database. When the drive is not plugged in, studies run without\n"
            "it.")
        f_db.addRow("Database folder:", self.fld_run_db)
        self.cb_db_archive = QCheckBox(
            "Copy every finished run into the database")
        self.cb_db_archive.setChecked(bool(getattr(prefs, "run_db_archive",
                                                   True)))
        self.cb_db_keep_odb = QCheckBox(
            "Copy the .odb too (about 1 GB per run)")
        self.cb_db_keep_odb.setChecked(bool(getattr(prefs, "run_db_keep_odb",
                                                    False)))
        self.cb_db_any_release = QCheckBox(
            "Reuse runs made with another Abaqus release")
        self.cb_db_any_release.setChecked(bool(getattr(
            prefs, "run_db_any_release", False)))
        self.cb_db_any_release.setToolTip(
            "Off: a run is reused only if it was made with the same Abaqus\n"
            "release as the machine of the study (or, when a release is not\n"
            "known, on that same PC). Two releases do not give the same\n"
            "results.")
        for cb in (self.cb_db_archive, self.cb_db_keep_odb,
                   self.cb_db_any_release):
            f_db.addRow(cb)
        outer.addWidget(g_db)

        # ---- Machines group ----
        g_exec = QGroupBox("Machines")
        v_exec = QVBoxLayout(g_exec)
        lbl = QLabel(
            "\"%s\" runs Abaqus here with the command above. The other "
            "machines are PCs reached through a queue folder on a shared "
            "drive, where their agent (start_agent.bat, written in that "
            "folder when you click OK) runs the queued jobs. Each study "
            "chooses its machine; the default one is used otherwise."
            % LOCAL_NAME)
        lbl.setWordWrap(True)
        v_exec.addWidget(lbl)
        self.tbl_machines = QTableWidget(0, 3)
        self.tbl_machines.setHorizontalHeaderLabels(
            ["Name", "Queue folder (shared drive)",
             "Abaqus command on that PC"])
        hdr = self.tbl_machines.horizontalHeader()
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tbl_machines.setColumnWidth(0, 120)
        self.tbl_machines.setColumnWidth(2, 250)
        self.tbl_machines.verticalHeader().setVisible(False)
        self.tbl_machines.setMinimumHeight(110)
        for m in machines(prefs)[1:]:
            self._add_machine_row(m.name, m.queue_dir, m.abaqus_cmd)
        v_exec.addWidget(self.tbl_machines)
        row = QHBoxLayout()
        btn_add = QPushButton("Add a machine")
        btn_add.clicked.connect(self._on_add_machine)
        btn_folder = QPushButton("Queue folder\u2026")
        btn_folder.setToolTip("Pick the queue folder of the selected machine.")
        btn_folder.clicked.connect(self._on_pick_queue)
        btn_rm = QPushButton("Remove")
        btn_rm.clicked.connect(self._on_remove_machine)
        row.addWidget(btn_add)
        row.addWidget(btn_folder)
        row.addWidget(btn_rm)
        row.addStretch(1)
        v_exec.addLayout(row)
        f_def = QFormLayout()
        self.cmb_default_machine = QComboBox()
        self.cmb_default_machine.setToolTip(
            "Machine used by a study that has not chosen one.")
        self._default_wanted = default_name(prefs)
        self._refresh_default_combo()
        self.tbl_machines.itemChanged.connect(
            lambda _i: self._refresh_default_combo())
        f_def.addRow("Default machine:", self.cmb_default_machine)
        v_exec.addLayout(f_def)
        outer.addWidget(g_exec)

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

    # ---- machines ------------------------------------------------------
    def _add_machine_row(self, name="", queue_dir="", abaqus_cmd=""):
        r = self.tbl_machines.rowCount()
        self.tbl_machines.insertRow(r)
        for c, text in enumerate((name, _native_path(queue_dir), abaqus_cmd)):
            self.tbl_machines.setItem(r, c, QTableWidgetItem(text))
        return r

    def _on_add_machine(self):
        taken = set(self._machine_names()) | {LOCAL_NAME}
        n = 1
        while "Machine %d" % n in taken:
            n += 1
        r = self._add_machine_row("Machine %d" % n, "",
                                  r"C:\SIMULIA\Commands\abaqus.bat")
        self.tbl_machines.setCurrentCell(r, 0)
        self._refresh_default_combo()

    def _on_remove_machine(self):
        r = self.tbl_machines.currentRow()
        if r >= 0:
            self.tbl_machines.removeRow(r)
            self._refresh_default_combo()

    def _on_pick_queue(self):
        r = self.tbl_machines.currentRow()
        if r < 0:
            return
        item = self.tbl_machines.item(r, 1)
        path = QFileDialog.getExistingDirectory(
            self, "Queue folder", (item.text() if item else "")
            or str(Path.home()))
        if path:
            self.tbl_machines.setItem(r, 1, QTableWidgetItem(_native_path(path)))

    def _cell(self, r, c) -> str:
        item = self.tbl_machines.item(r, c)
        return item.text().strip() if item else ""

    def _machine_names(self):
        return [self._cell(r, 0) for r in range(self.tbl_machines.rowCount())]

    def _machines(self):
        return [Machine(self._cell(r, 0), QUEUE, _native_path(self._cell(r, 1)),
                        self._cell(r, 2))
                for r in range(self.tbl_machines.rowCount())]

    def _refresh_default_combo(self):
        cur = self.cmb_default_machine.currentText() or self._default_wanted
        self.cmb_default_machine.blockSignals(True)
        self.cmb_default_machine.clear()
        self.cmb_default_machine.addItems(
            [LOCAL_NAME] + [n for n in self._machine_names() if n])
        i = self.cmb_default_machine.findText(cur)
        self.cmb_default_machine.setCurrentIndex(max(i, 0))
        self.cmb_default_machine.blockSignals(False)

    def machine_problems(self) -> list:
        """What keeps the machine list from being saved."""
        out, seen = [], {LOCAL_NAME}
        for m in self._machines():
            if not m.name:
                out.append("A machine has no name.")
            elif m.name in seen:
                out.append("Two machines are named \"%s\"." % m.name)
            seen.add(m.name)
            if m.name and not m.queue_dir:
                out.append("Machine \"%s\" has no queue folder." % m.name)
        return out

    def accept(self):
        problems = self.machine_problems()
        if problems:
            QMessageBox.warning(self, "Machines",
                                "\u2022 " + "\n\u2022 ".join(problems))
            return
        super().accept()

    def result_prefs(self) -> Preferences:
        """Build a new Preferences instance from the dialog field values.
        The execution fields get the default machine's values."""
        ms = self._machines()
        default = self.cmb_default_machine.currentText() or LOCAL_NAME
        dm = next((m for m in ms if m.name == default), None)
        return replace(
            self._initial,
            abaqus_cmd            = self.fld_cmd.value(),
            abaqus_script         = self.fld_script.value(),
            default_workdir       = self.fld_workdir.value(),
            run_db_dir            = self.fld_run_db.value(),
            run_db_archive        = self.cb_db_archive.isChecked(),
            run_db_keep_odb       = self.cb_db_keep_odb.isChecked(),
            run_db_any_release    = self.cb_db_any_release.isChecked(),
            temp_unit_default     = "K" if self.cb_kelvin_default.isChecked() else "C",
            machines              = [m.to_dict() for m in ms],
            default_machine       = default,
            execution_mode        = "remote" if dm is not None else "local",
            remote_queue_dir      = (dm.queue_dir if dm is not None
                                     else self._initial.remote_queue_dir),
            remote_abaqus_cmd     = (dm.abaqus_cmd if dm is not None
                                     else self._initial.remote_abaqus_cmd),
        )
