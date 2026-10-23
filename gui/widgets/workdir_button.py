# -*- coding: utf-8 -*-
"""
"Open working dir" button shared by the tabs that launch Abaqus runs
(Sensitivity, Optimization > Model): opens the working directory of the
Preferences in the file explorer, creating it when missing.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox, QPushButton


def open_working_dir(parent, prefs) -> None:
    """Open `prefs.default_workdir` in the file explorer."""
    wd = getattr(prefs, "default_workdir", None) if prefs else None
    if not wd:
        QMessageBox.information(parent, "Working directory",
                                "No working directory set in Preferences.")
        return
    p = Path(wd)
    if not p.exists():
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(parent, "Working directory",
                                "Cannot open '%s': %s" % (wd, e))
            return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))


def workdir_button(parent, prefs_getter) -> QPushButton:
    """A push button opening the Preferences working directory."""
    btn = QPushButton("Open working dir", parent)
    btn.setToolTip("Open the Preferences working directory.")
    btn.clicked.connect(lambda: open_working_dir(
        parent, prefs_getter() if prefs_getter else None))
    return btn
