# -*- coding: utf-8 -*-
"""
TimeSeriesViewer — matplotlib widget for plotting one or more time
series (e.g. RF1/RF2 reaction forces on the tool RP).

Includes:
  - Multiple curves on a shared time axis.
  - A vertical "current time" line that follows the field-viewer slider.
  - Toggleable visibility per-curve.
"""
from __future__ import annotations
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class TimeSeriesViewer(QWidget):
    """Multi-curve time-series plot with a movable vertical cursor.

    Curves are added via `add_series(...)`. The current-time cursor
    moves with `set_current_time(t)`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._fig    = Figure(figsize=(5, 4), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._ax     = self._fig.add_subplot(111)
        self._ax.set_xlabel("time [s]")
        self._ax.set_ylabel("value")
        self._ax.grid(True, alpha=0.25, linestyle=":")

        self._toolbar = NavigationToolbar(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # State
        self._lines: dict[str, "matplotlib.lines.Line2D"] = {}
        # The vertical "current time" line. Created lazily on first
        # set_current_time so it's drawn on top of the curves.
        self._cursor = None

    def add_series(self, name: str, t: np.ndarray, y: np.ndarray,
                    color: str = None, linestyle: str = "-"):
        """Add a curve. If a curve with this name already exists, it's
        replaced (useful for refreshing on bundle reload)."""
        if name in self._lines:
            self._lines[name].remove()
            del self._lines[name]
        (line,) = self._ax.plot(t, y, label=name, color=color,
                                 linestyle=linestyle, linewidth=1.2)
        self._lines[name] = line
        self._ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        self._canvas.draw_idle()

    def clear(self):
        """Remove every curve and the cursor."""
        for line in self._lines.values():
            line.remove()
        self._lines.clear()
        if self._cursor is not None:
            self._cursor.remove()
            self._cursor = None
        # Clear the legend
        leg = self._ax.get_legend()
        if leg is not None:
            leg.remove()
        self._canvas.draw_idle()

    def set_current_time(self, t: float):
        """Position the vertical cursor at time `t`."""
        if self._cursor is None:
            self._cursor = self._ax.axvline(
                t, color="#d33", linewidth=1.0, linestyle="--", alpha=0.8,
            )
        else:
            self._cursor.set_xdata([t, t])
        self._canvas.draw_idle()
