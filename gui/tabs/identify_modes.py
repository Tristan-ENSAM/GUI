# -*- coding: utf-8 -*-
"""Optimization > Identify: the two identification modes in one tab.

- Model sizing: the numerical parameters (mass scaling, element size,
  domain) identified against a numerical reference (ms = 1 run,
  Richardson extrapolation), the protocol of the OptimizationTab, unchanged.
- Physical parameters: the IdentifyTab (bounds, start point, fixed point or
  Levenberg-Marquardt against a typed or run reference).

The two pages are the existing tab widgets, so their state, signals and
running studies are those of the widgets themselves.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel,
                               QStackedWidget, QVBoxLayout, QWidget)

MODE_SIZING = "sizing"
MODE_PHYSICAL = "physical"
MODES = ((MODE_SIZING, "Model sizing (numerical parameters)"),
         (MODE_PHYSICAL, "Physical parameters"))


class IdentifyModes(QWidget):
    """Mode selector over the Model sizing and Physical parameters pages."""
    modeChanged = Signal(str)

    def __init__(self, sizing_tab: QWidget, physical_tab: QWidget,
                 parent=None):
        super().__init__(parent)
        self.sizing_tab = sizing_tab
        self.physical_tab = physical_tab
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setContentsMargins(8, 6, 8, 0)
        row.addWidget(QLabel("Mode:"))
        self.cb_mode = QComboBox()
        for key, text in MODES:
            self.cb_mode.addItem(text, key)
        self.cb_mode.setToolTip(
            "Model sizing: mass scaling, element size and domain against a\n"
            "numerical reference (ms = 1 run, Richardson extrapolation).\n"
            "Physical parameters: any ticked parameter against a typed\n"
            "reference or the QoI of a finished run.")
        row.addWidget(self.cb_mode)
        row.addStretch(1)
        root.addLayout(row)
        self._root = root
        self.stack = QStackedWidget()
        self.stack.addWidget(sizing_tab)
        self.stack.addWidget(physical_tab)
        root.addWidget(self.stack, 1)
        self.cb_mode.currentIndexChanged.connect(self._on_mode)

    def set_header(self, widget: QWidget) -> None:
        """Show `widget` under the mode selector, above both modes (the
        admitted deviations eps_q, common to the two modes)."""
        self._root.insertWidget(1, widget)

    def mode(self) -> str:
        return self.cb_mode.currentData()

    def set_mode(self, mode: str) -> None:
        i = self.cb_mode.findData(mode)
        if i < 0:
            raise ValueError("unknown Identify mode %r" % (mode,))
        self.cb_mode.setCurrentIndex(i)

    def current_page(self) -> QWidget:
        return self.stack.currentWidget()

    def show_page(self, page: QWidget) -> None:
        """Select the mode whose page is `page`."""
        self.set_mode(MODE_SIZING if page is self.sizing_tab
                      else MODE_PHYSICAL)

    def _on_mode(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        self.modeChanged.emit(self.mode())
