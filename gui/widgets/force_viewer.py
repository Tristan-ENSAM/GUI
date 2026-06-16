# -*- coding: utf-8 -*-
"""ForceViewer — plot the cutting (Fc) and feed (Ff) force signals vs time."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class ForceViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(4, 2.5), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self._canvas)
        self.clear()

    def clear(self):
        self._ax.clear()
        self._ax.set_xlabel("t [s]")
        self._ax.set_ylabel("force [N]")
        self._ax.text(0.5, 0.5, "no force signal", ha="center", va="center",
                      color="#999", transform=self._ax.transAxes)
        self._canvas.draw_idle()

    def set_signal(self, t, fc, ff):
        self._ax.clear()
        t = np.asarray(t, float)
        self._ax.plot(t, np.asarray(fc, float), label="Fc (RF1)", lw=0.8)
        self._ax.plot(t, np.asarray(ff, float), label="Ff (RF2)", lw=0.8)
        self._ax.set_xlabel("t [s]")
        self._ax.set_ylabel("force [N]")
        self._ax.legend(fontsize=8)
        self._ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()
