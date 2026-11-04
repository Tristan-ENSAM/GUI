# -*- coding: utf-8 -*-
"""Optimization > Setup: what every other Optimization tab compares with.

Top: the comparison settings (ZOI, admitted deviations eps_q, time window T,
ZOI sampling step) next to a preview of the model with the ZOI. Bottom: the
preliminary checks on one finished run, which help to fix T and the output
frames before the sizing studies.

The comparison-settings widgets are built and owned by the Model sizing tab
(gui/tabs/optimization_tab.py), which reads, validates and saves them with
the profile as before; this tab only shows them. So moving them here changes
no setting, no key of a saved profile and no result.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSplitter, QVBoxLayout, QWidget)

from gui.widgets.preliminary_checks_panel import PreliminaryChecksPanel


class SetupTab(QWidget):
    """`sizing_tab` is the Model sizing tab (OptimizationTab): its
    `settings_group`, `setup_hints`, `setup_preview`, `prelim_settings` and
    `_draw_preview` are used. `start_dir` is a path or a callable returning
    it (start folder of the file dialogs)."""

    def __init__(self, sizing_tab, start_dir="", parent=None):
        super().__init__(parent)
        self._sizing = sizing_tab
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        intro = QLabel(
            "Set here what the runs are compared on, before any study: the "
            "comparison zone ZOI, the admitted deviation ε_q of each "
            "quantity and the time window T. The Model sizing, Parameter "
            "plans and Compare runs tabs all use these values. The "
            "preliminary checks below read one finished run to choose T "
            "and the number of output frames.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#6b7280;")
        intro.setVisible(False)
        self._hints = [intro] + list(getattr(sizing_tab, "setup_hints", []))
        top = QHBoxLayout()
        top.addWidget(intro, 1)
        self.cb_help = QCheckBox("Show help")
        self.cb_help.setToolTip("Show the grey explanations of the "
                                "comparison settings.")
        self.cb_help.toggled.connect(self._show_help)
        top.addWidget(self.cb_help, 0, Qt.AlignRight | Qt.AlignTop)
        root.addLayout(top)

        # ---- comparison settings | preview -----------------------------
        inputs = QWidget()
        il = QVBoxLayout(inputs)
        il.setContentsMargins(0, 0, 4, 0)
        il.addWidget(sizing_tab.settings_group)
        il.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inputs)
        scroll.setMinimumWidth(inputs.sizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width())

        gprev = QGroupBox("Preview")
        pv = QVBoxLayout(gprev)
        pv.addWidget(sizing_tab.setup_preview, 1)
        btn = QPushButton("Refresh preview")
        btn.clicked.connect(sizing_tab._draw_preview)
        pv.addWidget(btn)

        upper = QSplitter(Qt.Horizontal)
        upper.addWidget(scroll)
        upper.addWidget(gprev)
        upper.setStretchFactor(0, 3)
        upper.setStretchFactor(1, 2)
        upper.setChildrenCollapsible(False)

        # ---- preliminary checks ----------------------------------------
        gpre = QGroupBox("Preliminary checks on one run")
        gl = QVBoxLayout(gpre)
        gl.setContentsMargins(4, 4, 4, 4)
        self.prelim = PreliminaryChecksPanel(
            settings_getter=sizing_tab.prelim_settings, start_dir=start_dir,
            parent=gpre)
        gl.addWidget(self.prelim)

        split = QSplitter(Qt.Vertical)
        split.addWidget(upper)
        split.addWidget(gpre)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        # Room for the preview (it draws the ZOI the fields above set) and
        # for the plots of the checks; the splitter handle moves both.
        sizing_tab.setup_preview.setMinimumHeight(300)
        split.setSizes([440, 520])
        root.addWidget(split, 1)

    def _show_help(self, on):
        for lab in self._hints:
            lab.setVisible(bool(on))
