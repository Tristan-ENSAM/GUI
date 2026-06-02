# -*- coding: utf-8 -*-
"""
Reusable labeled-numeric-field widget.

Emits `valueChanged(float)` whenever the user types a valid number,
which the GeometryTab uses to refresh the preview live.
"""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt, QLocale
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QCheckBox


# Force C locale on numeric validators so the dot ('.') is accepted as the
# decimal separator regardless of the system locale (e.g. French Windows
# would otherwise expect a comma and reject any keystroke containing '.').
# This also means the validator will not flag a partially-typed number
# like "0." or "1.2e" as invalid in mid-edit, which is required to let
# the user modify a value in place without first clearing the field.
_C_LOCALE = QLocale(QLocale.Language.C)
_C_LOCALE.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)


class NumField(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label: str, value: float, unit: str = "",
                 decimals: int = 6, minimum: float = -1e9, maximum: float = 1e9,
                 compact: bool = False, parent=None):
        """Labeled numeric input.

        `compact=True` is used inside a PairRow: the label is shorter, no
        stretch is added at the end, and width budgets are tighter so two
        fields fit comfortably on one line.
        """
        super().__init__(parent)
        self._lbl = QLabel(label)
        if compact:
            self._lbl.setMinimumWidth(56)
        else:
            self._lbl.setMinimumWidth(110)
        self._edit = QLineEdit(f"{value:g}")
        v = QDoubleValidator(minimum, maximum, decimals, self)
        v.setNotation(QDoubleValidator.ScientificNotation)
        v.setLocale(_C_LOCALE)
        self._edit.setValidator(v)
        self._edit.setLocale(_C_LOCALE)
        self._edit.setAlignment(Qt.AlignRight)
        self._edit.setMaximumWidth(90 if compact else 110)
        self._unit = QLabel(unit)
        self._unit.setMinimumWidth(22 if compact else 30)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self._lbl)
        lay.addWidget(self._edit)
        lay.addWidget(self._unit)
        if not compact:
            lay.addStretch()

        self._edit.textChanged.connect(self._on_text)

    def _on_text(self, txt: str):
        try:
            # accept comma as decimal separator too (FR locales)
            val = float(txt.replace(",", "."))
        except ValueError:
            return
        self.valueChanged.emit(val)

    def value(self) -> float:
        try:
            return float(self._edit.text().replace(",", "."))
        except ValueError:
            return 0.0

    def set_value(self, val: float):
        self._edit.blockSignals(True)
        self._edit.setText(f"{val:g}")
        self._edit.blockSignals(False)


class IntField(QWidget):
    valueChanged = Signal(int)

    def __init__(self, label: str, value: int, unit: str = "",
                 minimum: int = 0, maximum: int = 10_000_000, parent=None):
        super().__init__(parent)
        self._lbl = QLabel(label)
        self._lbl.setMinimumWidth(110)
        self._edit = QLineEdit(str(value))
        self._edit.setValidator(QIntValidator(minimum, maximum, self))
        self._edit.setAlignment(Qt.AlignRight)
        self._edit.setMaximumWidth(110)
        self._unit = QLabel(unit)
        self._unit.setMinimumWidth(30)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self._lbl)
        lay.addWidget(self._edit)
        lay.addWidget(self._unit)
        lay.addStretch()

        self._edit.textChanged.connect(self._on_text)

    def _on_text(self, txt: str):
        try:
            self.valueChanged.emit(int(txt))
        except ValueError:
            return

    def value(self) -> int:
        try:
            return int(self._edit.text())
        except ValueError:
            return 0

    def set_value(self, val: int):
        self._edit.blockSignals(True)
        self._edit.setText(str(val))
        self._edit.blockSignals(False)


class BoolField(QWidget):
    valueChanged = Signal(bool)

    def __init__(self, label: str, value: bool, parent=None):
        super().__init__(parent)
        self._cb = QCheckBox(label)
        self._cb.setChecked(value)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self._cb)
        lay.addStretch()
        self._cb.toggled.connect(self.valueChanged.emit)

    def value(self) -> bool:
        return self._cb.isChecked()

    def set_value(self, val: bool):
        self._cb.blockSignals(True)
        self._cb.setChecked(val)
        self._cb.blockSignals(False)


class PairRow(QWidget):
    """A row containing two NumFields side-by-side, used to compact pairs
    of related parameters (e.g. x0/y0, xmin/xmax) onto a single line.

    The two NumFields are accessible as `.left` and `.right` so callers
    keep their existing per-field signal wiring."""

    def __init__(self, left: NumField, right: NumField, parent=None):
        super().__init__(parent)
        self.left = left
        self.right = right
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(left)
        lay.addWidget(right)
        lay.addStretch()

