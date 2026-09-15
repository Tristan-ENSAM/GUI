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


# Curves are routed to one of two stacked axes by name. Forces (N) and
# energies (mJ) differ by orders of magnitude and are physically unrelated:
# drawing them on a single axis makes one of the two unreadable and invites
# meaningless visual comparisons.
_ENERGY_PREFIXES = ("ALL",)          # ALLKE, ALLIE, ALLVD, ALLWK, ALLPD...
_ENERGY_NAMES = frozenset(("ETOTAL",))


def _is_energy(name: str) -> bool:
    n = (name or "").upper()
    return n in _ENERGY_NAMES or n.startswith(_ENERGY_PREFIXES)


class TimeSeriesViewer(QWidget):
    """Time-series plot with a movable vertical cursor, on TWO stacked axes.

    Top axis: forces and everything else. Bottom axis: energies (ALL*).
    They share the time axis, so the cursor and any zoom stay aligned.

    Curves are added via `add_series(...)`; routing is automatic from the
    variable name. The current-time cursor moves with `set_current_time(t)`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._fig    = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        # sharex: one time axis for both, so zoom/pan and the cursor stay
        # aligned between the force and energy panels.
        self._ax, self._ax_energy = self._fig.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
        self._ax.set_ylabel("force")
        self._ax_energy.set_xlabel("time [s]")
        self._ax_energy.set_ylabel("energy")
        # Energy quantities (ALLKE/ALLIE/ALLVD...) span orders of magnitude and
        # are >= 0, so a log y-axis reads far better. Non-positive samples (a 0
        # at t=0) are simply not drawn by matplotlib.
        self._ax_energy.set_yscale("log")
        for ax in (self._ax, self._ax_energy):
            ax.grid(True, alpha=0.25, linestyle=":")

        self._toolbar = NavigationToolbar(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # State
        self._lines: dict[str, "matplotlib.lines.Line2D"] = {}
        # One cursor per axis (a Line2D belongs to a single axes).
        self._cursors = []

    def add_series(self, name: str, t: np.ndarray, y: np.ndarray,
                    color: str = None, linestyle: str = "-",
                    energy: bool = None):
        """Add a curve. If a curve with this name already exists, it's
        replaced (useful for refreshing on bundle reload).

        `energy` forces the panel (energy vs force); when None it is inferred
        from the name. Callers that prefix the label (e.g. 'run·ALLKE' for
        overlays) MUST pass `energy` explicitly, since the name no longer
        starts with 'ALL'."""
        if name in self._lines:
            self._lines[name].remove()
            del self._lines[name]
        is_energy = _is_energy(name) if energy is None else bool(energy)
        ax = self._ax_energy if is_energy else self._ax
        (line,) = ax.plot(t, y, label=name, color=color,
                          linestyle=linestyle, linewidth=1.2)
        self._lines[name] = line
        # Legend only on the axes that actually carry curves.
        if ax.get_lines():
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        self._canvas.draw_idle()

    def clear(self):
        """Remove every curve and the cursor."""
        for line in self._lines.values():
            line.remove()
        self._lines.clear()
        for c in self._cursors:
            c.remove()
        self._cursors = []
        for ax in (self._ax, self._ax_energy):
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        self._canvas.draw_idle()

    def set_current_time(self, t: float):
        """Position the vertical cursor at time `t`."""
        if not self._cursors:
            self._cursors = [
                ax.axvline(t, color="#d33", linewidth=1.0, linestyle="--",
                           alpha=0.8)
                for ax in (self._ax, self._ax_energy)
            ]
        else:
            for c in self._cursors:
                c.set_xdata([t, t])
        self._canvas.draw_idle()
