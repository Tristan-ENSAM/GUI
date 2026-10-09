# -*- coding: utf-8 -*-
"""
Collapsible "Advanced parameters" section.

A flat arrow button that shows or hides a body widget. The section starts
collapsed so a first-time user only sees the essential inputs of a panel.
Because a hidden field still drives the study, the header says how many of
its fields differ from their default value ("2 changed"), so a change can
never go unnoticed while the section is closed.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    toggled = Signal(bool)

    def __init__(self, title: str = "Advanced parameters", parent=None):
        super().__init__(parent)
        self._title = title
        self._n_changed = 0
        self._btn = QToolButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(False)
        self._btn.setArrowType(Qt.RightArrow)
        self._btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn.setAutoRaise(True)
        self._btn.toggled.connect(self.set_expanded)
        self.body = QWidget()
        self.body.setVisible(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self._btn, 0, Qt.AlignLeft)
        lay.addWidget(self.body)
        self._refresh_header()

    def is_expanded(self) -> bool:
        return self._btn.isChecked()

    def set_expanded(self, on: bool):
        on = bool(on)
        if self._btn.isChecked() != on:
            self._btn.setChecked(on)     # re-enters through toggled
            return
        self._btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)
        self.toggled.emit(on)

    def n_changed(self) -> int:
        return self._n_changed

    def set_changed_count(self, n: int):
        """Number of fields of the body that differ from their default."""
        self._n_changed = int(n)
        self._refresh_header()

    def header_text(self) -> str:
        return self._btn.text()

    def _refresh_header(self):
        if self._n_changed:
            self._btn.setText("%s — %d changed from default"
                              % (self._title, self._n_changed))
            self._btn.setStyleSheet("QToolButton { color: #b45309; }")
        else:
            self._btn.setText(self._title)
            self._btn.setStyleSheet("")
