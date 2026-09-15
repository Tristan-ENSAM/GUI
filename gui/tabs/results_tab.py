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
import logging

from gui.results.reader import ResultsBundle, ResultsLoadError
from gui.results.export_txt import export_bundle
from gui.widgets.field_viewer       import FieldViewer
from gui.widgets.time_series_viewer import TimeSeriesViewer, _is_energy
from gui.core.logging_util import log_swallowed


# Eulerian cells with volume fraction below this are treated as empty and
# hidden when rendering EULER fields (only show where material is present).
_EVF_MIN = 1e-3


class ResultsTab(QWidget):
    """Top-level Results tab."""

    # Emitted when a bundle is loaded or unloaded. Other tabs that may
    # want to react (e.g. the future Optimization tab) can listen.
    runsChanged = Signal()

    # Stable colour per run + linestyle per quantity for overlays.
    _PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf")
    _LINESTYLES = ("-", "--", ":", "-.")

    def __init__(self, parent=None):
        super().__init__(parent)

        # State -------------------------------------------------------
        # Multi-run ready: a dict keyed by run name. For now the UI
        # only addresses the most recently loaded run.
        self._runs: dict[str, ResultsBundle] = {}
        self._active_run: str | None = None
        self._run_colors: dict[str, str] = {}
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

        self.btn_export = QPushButton("Export…")
        self.btn_export.setToolTip(
            "Export every field to .txt (one file per quantity):\n"
            "rows = elements (index in header), columns = time steps (s)."
        )
        self.btn_export.clicked.connect(self._on_export_clicked)
        bar.addWidget(self.btn_export)

        # Run picker (currently single-run, but the combo is here for
        # future multi-run comparison).
        bar.addWidget(QLabel("Run:"))
        self.cb_run = QComboBox()
        self.cb_run.setMinimumWidth(180)
        self.cb_run.currentTextChanged.connect(self._on_run_changed)
        bar.addWidget(self.cb_run)

        self.btn_close = QPushButton("Close run")
        self.btn_close.setToolTip(
            "Remove the active run from the comparison.")
        self.btn_close.clicked.connect(self._on_close_clicked)
        bar.addWidget(self.btn_close)

        bar.addWidget(self._vline())

        # Instance picker (Eulerian workpiece / Lagrangian tool). Defaults
        # to the Eulerian instance, which carries the cutting fields.
        bar.addWidget(QLabel("Instance:"))
        self.cb_inst = QComboBox()
        self.cb_inst.setMinimumWidth(110)
        self.cb_inst.currentTextChanged.connect(self._on_instance_changed)
        bar.addWidget(self.cb_inst)

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
        """Open a file picker (multiple selection), then load each bundle."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load results bundle(s)",
            "",
            "Results files (*.npz *.json);;NumPy archive (*.npz);;"
            "JSON metadata (*.json);;All files (*)"
        )
        for path_str in paths:
            self.load_bundle(path_str)

    def _on_export_clicked(self):
        """Export every field of the loaded run to .txt (one per quantity)."""
        if self._active_run is None:
            QMessageBox.information(self, "Export",
                                    "Load a results bundle first.")
            return
        outdir = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the exported .txt files", "")
        if not outdir:
            return
        bundle = self._runs[self._active_run]
        try:
            written = export_bundle(bundle, outdir)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(
            self, "Export complete",
            "Wrote %d file(s) to:\n%s\n\n"
            "Layout: one row per element (index in the header column), "
            "one column per time step (time in seconds in the header row)."
            % (len(written), outdir))

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

    def _run_color(self, name: str) -> str:
        """A stable colour for a run (assigned once, by load order)."""
        if name not in self._run_colors:
            self._run_colors[name] = self._PALETTE[
                len(self._run_colors) % len(self._PALETTE)]
        return self._run_colors[name]

    def _refresh_time_series(self):
        """Overlay the history of EVERY loaded run: one stable colour per run,
        a linestyle per quantity, and a run-prefixed legend when more than one
        run is loaded."""
        self.ts_viewer.clear()
        multi = len(self._runs) > 1
        for name, bundle in self._runs.items():
            color = self._run_color(name)
            try:
                hist_t = bundle.history_time
                variables = list(bundle.history_info.variables)
            except Exception:
                continue
            for k, var in enumerate(variables):
                try:
                    y = bundle.history(var)
                except KeyError:
                    continue
                label = ("%s\u00b7%s" % (name, var)) if multi else var
                self.ts_viewer.add_series(
                    label, hist_t, y, color=color,
                    linestyle=self._LINESTYLES[k % len(self._LINESTYLES)],
                    energy=_is_energy(var))

    def _on_close_clicked(self):
        """Remove the active run from the comparison."""
        if self._active_run is None:
            return
        self._runs.pop(self._active_run, None)
        self._run_colors.pop(self._active_run, None)
        self.cb_run.blockSignals(True)
        self.cb_run.clear()
        for k in self._runs:
            self.cb_run.addItem(k)
        self.cb_run.blockSignals(False)
        if self._runs:
            new = self.cb_run.currentText() or next(iter(self._runs))
            self.cb_run.setCurrentText(new)
            self._on_run_changed(new)
        else:
            self._active_run = None
            self.ts_viewer.clear()
            self.field_viewer.clear()
            self.cb_inst.clear()
            self.cb_field.clear()
        self.runsChanged.emit()

    def _on_run_changed(self, name: str):
        """User picked a different run (or a load added one). The time-series
        panel overlays ALL loaded runs; the field viewer follows this active
        run. Instance / field / colormap and the frame index are preserved
        across the switch when the new run supports them."""
        if not name or name not in self._runs:
            return
        prev_inst = self.cb_inst.currentText()
        prev_field = self.cb_field.currentText()
        prev_frame = self._frame_idx
        self._active_run = name
        bundle = self._runs[name]

        # Time-series overlay (independent of the active run).
        self._refresh_time_series()

        instances = bundle.instance_names
        if not instances:
            # No field data: the overlaid history is still shown. Clear the
            # field pickers rather than popping a modal on every switch.
            self.cb_inst.blockSignals(True); self.cb_inst.clear()
            self.cb_inst.blockSignals(False)
            self.cb_field.blockSignals(True); self.cb_field.clear()
            self.cb_field.blockSignals(False)
            self.field_viewer.clear()
            return

        # Instance picker: keep the previous choice if the new run has it.
        self.cb_inst.blockSignals(True)
        self.cb_inst.clear()
        for nm in instances:
            self.cb_inst.addItem(nm)
        self.cb_inst.setCurrentText(
            prev_inst if prev_inst in instances
            else self._default_instance(bundle))
        self.cb_inst.blockSignals(False)

        # Field combo + mesh, keeping the previous field if available.
        self._load_instance_fields(prefer=prev_field)

        # Frame: keep the same index, clamped to this run's range.
        nf = bundle.n_frames
        fi = max(0, min(prev_frame, nf - 1))
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, nf - 1))
        self.slider.setValue(fi)
        self.slider.setEnabled(nf > 1)
        self.slider.blockSignals(False)
        self._frame_idx = fi

        # Draw the frame (cb_cmap is untouched -> colormap preserved).
        self._on_field_changed()
    # =====================================================================
    # Instance selection
    # =====================================================================
    def _default_instance(self, bundle) -> str:
        """Pick the instance to show by default: the Eulerian one (the
        cutting workpiece, which carries PEEQ/TEMP/MISES/EVF); else the
        instance with the most field variables; else the first."""
        names = bundle.instance_names
        euler = [n for n in names
                 if getattr(bundle.instance(n), "kind", "") == "eulerian"
                 and bundle.instance(n).field_variables]
        if euler:
            return max(euler, key=lambda n: len(bundle.instance(n).field_variables))
        with_fields = [n for n in names if bundle.instance(n).field_variables]
        if with_fields:
            return max(with_fields,
                       key=lambda n: len(bundle.instance(n).field_variables))
        return names[0]

    def _current_instance(self):
        """The instance currently selected in the picker (fallback: first)."""
        if self._active_run is None:
            return None
        names = self._runs[self._active_run].instance_names
        if not names:
            return None
        sel = self.cb_inst.currentText()
        return sel if sel in names else names[0]

    def _load_instance_fields(self, prefer: str = None):
        """Populate the field combo and mesh for the selected instance."""
        if self._active_run is None:
            return
        bundle = self._runs[self._active_run]
        inst = self._current_instance()
        if inst is None:
            return
        info = bundle.instance(inst)
        self.cb_field.blockSignals(True)
        self.cb_field.clear()
        for v in info.field_variables:
            self.cb_field.addItem(v)
        # Default to a field that actually shows structure at frame 0
        # (PEEQ is zero everywhere initially and looks blank). Prefer EVF
        # (material vs void), then a temperature field.
        prefs = ([prefer] if prefer else []) + ["EVF", "TEMP", "NT11", "V"]
        for pref in prefs:
            i = self.cb_field.findText(pref)
            if pref and i >= 0:
                self.cb_field.setCurrentIndex(i)
                break
        self.cb_field.blockSignals(False)
        self._init_field_mesh()

    def _on_instance_changed(self, name: str):
        """User picked a different instance: rebuild field combo + mesh,
        then redraw the current frame."""
        if self._active_run is None or not name:
            return
        self._load_instance_fields()
        self._on_field_changed()

    # =====================================================================
    # Field rendering
    # =====================================================================
    def _init_field_mesh(self):
        """Push the active bundle's mesh into the FieldViewer. Called
        once per loaded bundle."""
        if self._active_run is None:
            return
        bundle = self._runs[self._active_run]
        inst = self._current_instance()
        if inst is None:
            return
        info = bundle.instance(inst)

        nodes = bundle.nodes_init(info.name)         # (n_nodes, 3)
        elems = bundle.elements(info.name)            # (n_elements, 8)
        # 2D projection: take the (x, y) of the first 4 nodes of each
        # element (the z=zmin face for standard C3D8 ordering).
        # The mesh viewer doesn't care about z.
        nodes_xy = nodes[:, :2]
        conn = np.asarray(elems)                      # (n_elem, 8) local idx
        # Robust 2D face: order each element's projected nodes CCW around
        # their centroid. Taking elems[:, :4] blindly can pick a face that
        # spans the thin (z) direction, collapsing to a zero-area quad in
        # x-y (invisible). Angle-ordering all 8 projected nodes traces the
        # correct footprint regardless of C3D8 ordering / thin axis.
        P = nodes_xy[conn]                            # (n_elem, 8, 2)
        c = P.mean(axis=1, keepdims=True)
        ang = np.arctan2(P[:, :, 1] - c[:, :, 1], P[:, :, 0] - c[:, :, 0])
        order = np.argsort(ang, axis=1)
        face_idx = np.take_along_axis(conn, order, axis=1)   # (n_elem, 8)
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
        inst = self._current_instance()
        if inst is None:
            return
        info = bundle.instance(inst)
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
        inst = self._current_instance()
        if inst is None:
            return
        info = bundle.instance(inst)
        try:
            field = bundle.field(info.name, var)
        except KeyError:
            return

        idx = self._frame_idx
        idx = max(0, min(idx, field.shape[0] - 1))
        values = np.asarray(field[idx], dtype=float)
        t = float(bundle.times[idx])

        # Eulerian instance: hide cells with no material (EVF ~ 0). Those
        # carry meaningless TEMP/V and clutter the plot. NaN cells render
        # transparent in the field viewer.
        if "EVF" in info.field_variables:
            try:
                evf = np.asarray(bundle.field(info.name, "EVF")[idx],
                                 dtype=float)
                if evf.shape == values.shape:
                    values = values.copy()
                    values[evf <= _EVF_MIN] = np.nan
            except (KeyError, IndexError):
                log_swallowed("masking field values by EVF",
                              level=logging.DEBUG)

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
