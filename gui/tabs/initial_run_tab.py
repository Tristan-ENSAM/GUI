# -*- coding: utf-8 -*-
"""Optimization > Initial run: the preliminary checks on the initial run.

Decision 2026-11-12: the Setup tab is gone. Its comparison settings are
entered where they belong (ZOI and chip-thickness line in Numerical Model >
Geometry, ZOI sampling step in Mesh, window T in Step, eps_q and the step
weights at the top of Optimization > Identify), and this tab holds the
preliminary checks. For now they read one finished run, as in Setup
before; launching the run and checking it while it runs comes next.

The checks read their defaults from the Model sizing tab
(OptimizationTab.prelim_settings), which still owns every setting.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget)

from gui.widgets.preliminary_checks_panel import PreliminaryChecksPanel


class InitialRunTab(QWidget):
    """`sizing_tab` is the Model sizing tab (OptimizationTab): its
    `setup_preview`, `prelim_settings` and `_draw_preview` are used.
    `start_dir` is a path or a callable returning it (start folder of the
    file dialogs)."""

    def __init__(self, sizing_tab, start_dir="", parent=None):
        super().__init__(parent)
        self._sizing = sizing_tab
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        intro = QLabel(
            "Checks on the initial run, before any study: steady regime "
            "(start of the window T), chip touching the workpiece again, "
            "and output-frame subsampling. Open a finished run below. The "
            "ZOI is set in Numerical Model > Geometry, the window T in "
            "Step, the ZOI sampling step in Mesh and ε_q in Identify.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#6b7280;")
        root.addWidget(intro)

        gpre = QGroupBox("Preliminary checks on one run")
        gl = QVBoxLayout(gpre)
        gl.setContentsMargins(4, 4, 4, 4)
        self.prelim = PreliminaryChecksPanel(
            settings_getter=sizing_tab.prelim_settings, start_dir=start_dir,
            parent=gpre)
        gl.addWidget(self.prelim)

        gprev = QGroupBox("Model with the ZOI")
        pv = QVBoxLayout(gprev)
        pv.addWidget(sizing_tab.setup_preview, 1)
        btn = QPushButton("Refresh preview")
        btn.clicked.connect(sizing_tab._draw_preview)
        pv.addWidget(btn)
        sizing_tab.setup_preview.setMinimumHeight(260)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(gpre)
        split.addWidget(gprev)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)
