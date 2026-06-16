# -*- coding: utf-8 -*-
"""
ImageSequenceViewer — preview one frame of an ImageSequence with a slider.

Reusable for the visible and the IR streams. A colormap is applied for
single-channel (thermal / grayscale) frames; RGB frames are shown as-is.
The label shows 'frame i/N   t = ... s' using the sequence fps / t0.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class ImageSequenceViewer(QWidget):
    def __init__(self, title: str = "", cmap: str = "gray", parent=None):
        super().__init__(parent)
        self._seq = None
        self._cmap = cmap
        self._im = None
        self._cbar = None

        self._fig = Figure(figsize=(4, 3), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_axis_off()

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slide)

        self._label = QLabel(title or "no sequence")
        self._label.setStyleSheet("color: #555;")

        bottom = QHBoxLayout()
        bottom.addWidget(self._slider, stretch=1)
        bottom.addWidget(self._label)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self._canvas, stretch=1)
        lay.addLayout(bottom)

    def set_sequence(self, seq, cmap: str = None):
        """Attach an ImageSequence (or None to clear) and show frame 0."""
        self._seq = seq
        if cmap is not None:
            self._cmap = cmap
        self._ax.clear()
        self._ax.set_axis_off()
        self._im = None
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        n = seq.n_frames if seq is not None else 0
        self._slider.blockSignals(True)
        self._slider.setEnabled(n > 1)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        if n == 0:
            self._label.setText("no sequence")
            self._canvas.draw_idle()
            return
        self._show(0)

    def _on_slide(self, i):
        self._show(int(i))

    def _show(self, i: int):
        if self._seq is None:
            return
        frame = np.asarray(self._seq.frame(i))
        is_rgb = (frame.ndim == 3 and frame.shape[-1] in (3, 4))
        if self._im is None:
            if is_rgb:
                self._im = self._ax.imshow(frame)
            else:
                self._im = self._ax.imshow(frame, cmap=self._cmap)
                self._cbar = self._fig.colorbar(self._im, ax=self._ax,
                                                fraction=0.046, pad=0.04)
        else:
            self._im.set_data(frame)
            if not is_rgb:
                self._im.set_clim(np.nanmin(frame), np.nanmax(frame))
        t = self._seq.time(i)
        self._label.setText("frame %d/%d   t = %.4g s"
                            % (i + 1, self._seq.n_frames, t))
        self._canvas.draw_idle()
