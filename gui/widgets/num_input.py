# -*- coding: utf-8 -*-
"""
DecimalSpinBox — a QDoubleSpinBox that accepts both '.' and ',' as the decimal
separator on manual entry (convenient for FR/EN keyboards), and always
displays with '.' (locale C). Parsing tolerates the suffix/prefix.
"""
from __future__ import annotations

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QDoubleSpinBox, QSlider


class WheelStepSlider(QSlider):
    """QSlider whose mouse-wheel moves by exactly one singleStep per notch
    (the default multiplies by the system wheel-scroll-lines, e.g. 3)."""
    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if d == 0:
            e.ignore(); return
        step = self.singleStep() or 1
        self.setValue(self.value() + (step if d > 0 else -step))
        e.accept()


class DecimalSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLocale(QLocale(QLocale.Language.C))   # '.' as decimal point
        self.setKeyboardTracking(False)

    def validate(self, text, pos):
        return super().validate(text.replace(",", "."), pos)

    def valueFromText(self, text):
        t = text.replace(",", ".")
        suf, pre = self.suffix(), self.prefix()
        if suf and t.endswith(suf):
            t = t[:-len(suf)]
        if pre and t.startswith(pre):
            t = t[len(pre):]
        t = t.strip()
        try:
            return float(t)
        except ValueError:
            return self.value()
