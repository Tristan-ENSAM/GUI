# -*- coding: utf-8 -*-
"""
Results tab — load and explore Abaqus simulation results.

Layout:
  - Top bar: "Load results", run picker (combobox for future multi-run),
             field-variable picker, frame slider, time label, play/pause.
  - Splitter: field viewer on the left, time-series viewer on the right.

The field viewer shows the selected field on the Eulerian mesh,
clipped to the bundle's ROI. The slider drives both the field viewer
(frame index) and the time-series viewer (vertical cursor at the
matching time).

Multi-run is not exposed in the UI in this first iteration, but the
internal `_runs` dict already keys bundles by run name, so adding a
combobox + comparison view is a future-proof one-liner.
"""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox,
    QLabel, QSplitter, QFileDialog, QMessageBox, QFrame,
)

import numpy as np

from gui.results.reader import ResultsBundle, ResultsLoadError
from gui.widgets.field_viewer       import FieldViewer
from gui.widgets.time_series_viewer import TimeSeriesViewer


class ResultsTab(QWidget):
    """Top-level Results tab."""

    # Emitted when a bundle is loaded or unloaded. Other tabs that may
    # want to react (e.g. the future Optimization tab) can listen.
    runsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # State -------------------------------------------------------
        # Multi-run ready: a dict keyed by run name. For now the UI
        # only addresses the most recently loaded run.
        self._runs: dict[str, ResultsBundle] = {}
        self._active_run: str | None = None
        self._frame_idx: int = 0
        # Cached per-frame (vmin, vmax) for the active field — computed
        # over all frames so the colormap stays stable while scrubbing.
        self._field_vmin: float = 0.0
        self._field_vmax: float = 1.0

        # Animation timer ----------------------------------------------
        # Drives the play/pause: at each tick, advances the slider by one
        # frame. The interval is roughly 50 ms (~20 fps) — adjust if the
        # field viewer struggles on big meshes.
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_timer_tick)

        # Build the UI
        self._build_ui()

    # =====================================================================
    # UI construction
    # =====================================================================
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ----- Top bar: actions + selectors -----
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self.btn_load = QPushButton("Load results…")
        self.btn_load.setToolTip(
            "Pick a <name>.results.json or <name>.results.npz file.\n"
            "The companion file is auto-located."
        )
        self.btn_load.clicked.connect(self._on_load_clicked)
        bar.addWidget(self.btn_load)

        # Run picker (currently single-run, but the combo is here for
        # future multi-run comparison).
        bar.addWidget(QLabel("Run:"))
        self.cb_run = QComboBox()
        self.cb_run.setMinimumWidth(180)
        self.cb_run.currentTextChanged.connect(self._on_run_changed)
        bar.addWidget(self.cb_run)

        bar.addWidget(self._vline())

        # Field-variable picker (populated after load)
        bar.addWidget(QLabel("Field:"))
        self.cb_field = QComboBox()
        self.cb_field.setMinimumWidth(110)
        self.cb_field.currentTextChanged.connect(self._on_field_changed)
        bar.addWidget(self.cb_field)

        # Colormap picker — small list of useful presets
        bar.addWidget(QLabel("Cmap:"))
        self.cb_cmap = QComboBox()
        for cm in ("viridis", "inferno", "plasma", "RdBu_r", "coolwarm", "Greys"):
            self.cb_cmap.addItem(cm)
        self.cb_cmap.currentTextChanged.connect(self._on_field_changed)
        bar.addWidget(self.cb_cmap)

        bar.addStretch()
        outer.addLayout(bar)

        # ----- Second bar: slider + play/pause + time label -----
        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(36)
        self.btn_play.setToolTip("Play / pause the animation.")
        self.btn_play.clicked.connect(self._on_play_clicked)
        slider_row.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.slider, stretch=1)

        self.lbl_time = QLabel("frame —/— · t = — s")
        self.lbl_time.setMinimumWidth(220)
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider_row.addWidget(self.lbl_time)
        outer.addLayout(slider_row)

        # ----- Splitter: field viewer | time-series viewer -----
        self.field_viewer = FieldViewer()
        self.ts_viewer    = TimeSeriesViewer()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.field_viewer)
        split.addWidget(self.ts_viewer)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, stretch=1)

    @staticmethod
    def _vline() -> QFrame:
        """Vertical separator for the top bar."""
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFrameShadow(QFrame.Sunken)
        return f

    # =====================================================================
    # Load + run management
    # =====================================================================
    def _on_load_clicked(self):
        """Open a file picker, then try to load the chosen bundle."""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load results bundle",
            "",  # start in cwd; could remember the last folder later
            "Results files (*.npz *.json);;NumPy archive (*.npz);;"
            "JSON metadata (*.json);;All files (*)"
        )
        if not path_str:
            return
        self.load_bundle(path_str)

    def load_bundle(self, path: str | Path):
        """Programmatically load a bundle (also used by tests). Wraps
        ResultsBundle.load with friendly error reporting."""
        try:
            bundle = ResultsBundle.load(path)
        except ResultsLoadError as e:
            QMessageBox.critical(self, "Cannot load results", str(e))
            return

        # Name the run by job_name; if a run with that name is already
        # loaded, suffix with a counter to keep both around.
        base = bundle.job_name or Path(path).stem
        name = base
        i = 2
        while name in self._runs:
            name = f"{base} ({i})"
            i += 1
        self._runs[name] = bundle

        # Refresh the run combobox (block signals while populating to
        # avoid spurious _on_run_changed firings).
        self.cb_run.blockSignals(True)
        self.cb_run.clear()
        for k in self._runs:
            self.cb_run.addItem(k)
        self.cb_run.setCurrentText(name)
        self.cb_run.blockSignals(False)
        # Manually trigger the change handler now that everything is set.
        self._on_run_changed(name)

        self.runsChanged.emit()

    def _on_run_changed(self, name: str):
        """User picked a different run in the combobox (or load just
        added a new one)."""
        if not name or name not in self._runs:
            return
        self._active_run = name
        bundle = self._runs[name]

        # ----- Populate the field combobox -----
        # We take the first instance's fields; multi-instance handling
        # comes in phase 2.
        instances = bundle.instance_names
        if not instances:
            return
        info = bundle.instance(instances[0])
        self.cb_field.blockSignals(True)
        self.cb_field.clear()
        for v in info.field_variables:
            self.cb_field.addItem(v)
        self.cb_field.blockSignals(False)

        # ----- Build the static mesh in the field viewer -----
        self._init_field_mesh()

        # ----- Populate the time-series viewer with history -----
        self.ts_viewer.clear()
        hist_t = bundle.history_time
        for var in bundle.history_info.variables:
            try:
                y = bundle.history(var)
                self.ts_viewer.add_series(var, hist_t, y)
            except KeyError:
                pass

        # ----- Reset the slider for the new bundle -----
        nf = bundle.n_frames
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, nf - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(nf > 1)
        self.slider.blockSignals(False)
        self._frame_idx = 0

        # Draw the first frame
        self._on_field_changed()  # uses current cb_field value

    # =====================================================================
    # Field rendering
    # =====================================================================
    def _init_field_mesh(self):
        """Push the active bundle's mesh into the FieldViewer. Called
        once per loaded bundle."""
        if self._active_run is None:
            return
        bundle = self._runs[self._active_run]
        info = bundle.instance(bundle.instance_names[0])

        nodes = bundle.nodes_init(info.name)         # (n_nodes, 3)
        elems = bundle.elements(info.name)            # (n_elements, 8)
        # 2D projection: take the (x, y) of the first 4 nodes of each
        # element (the z=zmin face for standard C3D8 ordering).
        # The mesh viewer doesn't care about z.
        nodes_xy = nodes[:, :2]
        face_idx = elems[:, :4]
        self.field_viewer.set_mesh(nodes_xy, face_idx)

    def _on_field_changed(self):
        """A new field variable or colormap was picked. Recompute the
        global vmin/vmax (over all frames) for a stable colormap, and
        re-render the current frame."""
        if self._active_run is None:
            return
        bundle = self._runs[self._active_run]
        var = self.cb_field.currentText()
        if not var:
            return
        info = bundle.instance(bundle.instance_names[0])
        try:
            field = bundle.field(info.name, var)
        except KeyError:
            return

        # Global colormap range over all frames — stable while scrubbing.
        # For long runs this is a one-shot O(n_frames * n_elements) read,
        # which is fine for the sizes we deal with (~500 frames, ~10k
        # elements ⇒ ~20 MB).
        self._field_vmin = float(np.nanmin(field))
        self._field_vmax = float(np.nanmax(field))
        # Guard against a constant field (everything zero)
        if self._field_vmin >= self._field_vmax:
            self._field_vmax = self._field_vmin + 1e-12

        self._render_current_frame()

    def _render_current_frame(self):
        """Push the active frame's values into the field viewer, and
        update the time cursor in the time-series viewer."""
        if self._active_run is None:
            return
        bundle = self._runs[self._active_run]
        var = self.cb_field.currentText()
        if not var:
            return
        info = bundle.instance(bundle.instance_names[0])
        try:
            field = bundle.field(info.name, var)
        except KeyError:
            return

        idx = self._frame_idx
        idx = max(0, min(idx, field.shape[0] - 1))
        values = np.asarray(field[idx])
        t = float(bundle.times[idx])

        cmap = self.cb_cmap.currentText() or "viridis"
        title = f"{var}   frame {idx}/{field.shape[0]-1}   t = {t:.3e} s"
        self.field_viewer.set_values(
            values, vmin=self._field_vmin, vmax=self._field_vmax,
            cmap=cmap, title=title,
        )

        # Time cursor on history plot
        self.ts_viewer.set_current_time(t)

        # Time label in the slider row
        self.lbl_time.setText(
            f"frame {idx}/{field.shape[0]-1} · t = {t:.4e} s"
        )

    # =====================================================================
    # Slider + animation
    # =====================================================================
    def _on_slider_changed(self, value: int):
        self._frame_idx = int(value)
        self._render_current_frame()

    def _on_play_clicked(self):
        """Toggle the playback timer."""
        if self._timer.isActive():
            self._timer.stop()
            self.btn_play.setText("▶")
        else:
            # If we're already at the end, restart from the beginning.
            if self._frame_idx >= self.slider.maximum():
                self.slider.setValue(0)
            self._timer.start()
            self.btn_play.setText("⏸")

    def _on_timer_tick(self):
        """Advance one frame. Stop the timer at the last frame."""
        next_idx = self._frame_idx + 1
        if next_idx > self.slider.maximum():
            # Reached the end — stop playback (don't loop by default)
            self._timer.stop()
            self.btn_play.setText("▶")
            return
        # setValue triggers _on_slider_changed -> _render_current_frame
        self.slider.setValue(next_idx)
