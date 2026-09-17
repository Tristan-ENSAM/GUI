# -*- coding: utf-8 -*-
"""
SensitivityMapPanel — the viewer for per-element sensitivity maps.

One matplotlib canvas (the shared FieldViewer, toolbar included, so the
picture is exported with its "Save the figure" button) plus the selectors
the study needs:

  * Output   — which evaluated field the map is about (EVF, V, TEMP...).
  * Parameter— which perturbed parameter it is differentiated against
               (A, B, n, C, m, friction...).
  * Quantity — what is drawn, which depends on the METHOD that produced
               the maps and is never mixed between methods:
                 Jacobian -> dF/dtheta, signed or magnitude,
                 Morris   -> mu*, sigma, mu.
  * Frame / aggregate — a single frame, or all frames reduced at once.

The panel also owns the study's persistence: "Save maps (.npz)" writes
every map plus the mesh to one file, "Load maps (.npz)" reads one back —
so a campaign can be re-examined later without re-running Abaqus, and
without the results bundles it came from.

"Open in a window" pops the same panel in a separate resizable window
(the embedded copy in the Sensitivity tab stays where it is).

The panel is a pure consumer of `map_io.SensitivityMapSet`: it neither
runs nor analyses anything, which keeps the maths testable headlessly.
"""
from __future__ import annotations

import logging

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QSlider, QPushButton, QFileDialog,
)

from gui.sensitivity import map_io as mio
from gui.widgets.field_viewer import FieldViewer
from gui.core.logging_util import log_swallowed

_NO_MAPS_HINT = (
    "No sensitivity map yet. Run a plan with at least one ROI field ticked, "
    "or load a study with “Load maps (.npz)”."
)


class SensitivityMapPanel(QWidget):
    """Selectors + FieldViewer for one SensitivityMapSet."""

    def __init__(self, parent=None, allow_detach: bool = True):
        super().__init__(parent)
        self._mapset = None
        self._entries = []          # [(quantity, mode, label, diverging)]
        self._windows = []          # detached windows, kept alive
        self._mesh_key = None       # id of the mesh currently pushed

        root = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Output:"))
        self.cb_field = QComboBox()
        self.cb_field.setToolTip("Evaluated output field the map is about.")
        row1.addWidget(self.cb_field, 1)
        row1.addWidget(QLabel("Parameter:"))
        self.cb_param = QComboBox()
        self.cb_param.setToolTip("Perturbed parameter the map differentiates "
                                 "against.")
        row1.addWidget(self.cb_param, 1)
        row1.addWidget(QLabel("Quantity:"))
        self.cb_quantity = QComboBox()
        self.cb_quantity.setToolTip(
            "What is drawn. Jacobian: dF/dθ per element (signed or "
            "magnitude). Morris: μ*, σ or μ per element.")
        row1.addWidget(self.cb_quantity, 1)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        self.chk_aggregate = QCheckBox("Aggregate over time")
        self.chk_aggregate.setToolTip(
            "Reduce all frames to one map: time-mean for a signed quantity, "
            "time-RMS for a magnitude.")
        row2.addWidget(self.chk_aggregate)
        row2.addWidget(QLabel("Frame:"))
        self.sld_frame = QSlider(Qt.Horizontal)
        self.sld_frame.setMinimum(0)
        self.sld_frame.setMaximum(0)
        self.sld_frame.setEnabled(False)
        row2.addWidget(self.sld_frame, 1)
        self.lbl_frame = QLabel("-")
        self.lbl_frame.setMinimumWidth(110)
        row2.addWidget(self.lbl_frame)
        self.btn_save = QPushButton("Save maps (.npz)…")
        self.btn_save.setToolTip(
            "Write every map of this study (all outputs × parameters), "
            "the mesh and the frame times to a single .npz.")
        self.btn_save.setEnabled(False)
        row2.addWidget(self.btn_save)
        self.btn_load = QPushButton("Load maps (.npz)…")
        self.btn_load.setToolTip(
            "Re-open a study saved earlier — no Abaqus run needed.")
        row2.addWidget(self.btn_load)
        self.btn_detach = QPushButton("Open in a window")
        self.btn_detach.setToolTip(
            "Show the same map in a separate, resizable window.")
        self.btn_detach.setEnabled(False)
        self.btn_detach.setVisible(bool(allow_detach))
        row2.addWidget(self.btn_detach)
        root.addLayout(row2)

        self.viewer = FieldViewer()
        root.addWidget(self.viewer, 1)

        self.lbl_hint = QLabel(_NO_MAPS_HINT)
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("color: #6b7280;")
        root.addWidget(self.lbl_hint)

        self.cb_field.currentIndexChanged.connect(self._on_field_changed)
        self.cb_param.currentIndexChanged.connect(self.refresh)
        self.cb_quantity.currentIndexChanged.connect(self.refresh)
        self.chk_aggregate.toggled.connect(self._on_aggregate_toggled)
        self.sld_frame.valueChanged.connect(self.refresh)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_detach.clicked.connect(self._on_detach)

    # =====================================================================
    # Data in
    # =====================================================================
    @property
    def map_set(self):
        return self._mapset

    def set_map_set(self, mapset, hint: str = None):
        """Show `mapset` (a map_io.SensitivityMapSet), or clear the panel
        when it is None. `hint` overrides the status line — use it to say
        WHY there is nothing to show."""
        self._mapset = mapset if (mapset is not None
                                  and mapset.is_drawable()) else None
        if self._mapset is None:
            self._clear()
            self.lbl_hint.setText(hint or _NO_MAPS_HINT)
            return
        self._populate()
        self.lbl_hint.setText(hint or mio.describe(self._mapset))

    # =====================================================================
    # Internals
    # =====================================================================
    def _clear(self):
        for cb in (self.cb_field, self.cb_param, self.cb_quantity):
            cb.blockSignals(True)
            cb.clear()
            cb.setEnabled(False)
            cb.blockSignals(False)
        self._entries = []
        self._mesh_key = None
        self.chk_aggregate.setEnabled(False)
        self.sld_frame.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_detach.setEnabled(False)
        self.lbl_frame.setText("-")
        try:
            self.viewer.clear()
        except Exception:
            log_swallowed("clearing the sensitivity-map viewer",
                          level=logging.DEBUG)

    def _populate(self):
        ms = self._mapset
        for cb in (self.cb_field, self.cb_param, self.cb_quantity):
            cb.blockSignals(True)
            cb.clear()
        for var in ms.field_vars:
            self.cb_field.addItem(ms.field_label(var), var)
        for p in ms.param_paths:
            unit = ms.param_units.get(p)
            label = ms.param_label(p)
            self.cb_param.addItem(
                "%s [%s]" % (label, unit) if unit else label, p)
        self._entries = mio.display_entries(ms.quantities)
        for i, (_q, _mode, label, _div) in enumerate(self._entries):
            self.cb_quantity.addItem(label, i)
        for cb in (self.cb_field, self.cb_param, self.cb_quantity):
            cb.setEnabled(cb.count() > 0)
            cb.blockSignals(False)

        n_frames = ms.n_frames
        self.sld_frame.blockSignals(True)
        self.sld_frame.setMinimum(0)
        self.sld_frame.setMaximum(max(0, n_frames - 1))
        self.sld_frame.setValue(max(0, n_frames - 1))
        self.sld_frame.blockSignals(False)
        self.chk_aggregate.setEnabled(True)
        self.sld_frame.setEnabled(n_frames > 1
                                  and not self.chk_aggregate.isChecked())
        self.btn_save.setEnabled(True)
        self.btn_detach.setEnabled(True)

        self._push_mesh()
        self.refresh()

    def _push_mesh(self):
        ms = self._mapset
        if ms is None:
            return False
        key = (id(ms), int(np.asarray(ms.faces).shape[0]))
        if key == self._mesh_key:
            return True
        try:
            self.viewer.set_mesh(np.asarray(ms.nodes_xy, dtype=float),
                                 np.asarray(ms.faces, dtype=int))
        except Exception:
            log_swallowed("pushing the sensitivity-map mesh",
                          level=logging.WARNING)
            self._mesh_key = None
            return False
        self._mesh_key = key
        return True

    def _current_entry(self):
        i = self.cb_quantity.currentData()
        if i is None or not (0 <= int(i) < len(self._entries)):
            return None
        return self._entries[int(i)]

    def _on_field_changed(self, *_):
        self.refresh()

    def _on_aggregate_toggled(self, *_):
        ms = self._mapset
        self.sld_frame.setEnabled(
            ms is not None and ms.n_frames > 1
            and not self.chk_aggregate.isChecked())
        self.refresh()

    def _frame_text(self, frame_idx):
        """'12/30' plus the frame time when the study carries one."""
        ms = self._mapset
        txt = "%d" % frame_idx
        try:
            times = np.asarray(ms.times, dtype=float)
            if times.size > frame_idx:
                txt += "  (t=%.4g s)" % float(times[frame_idx])
        except Exception:
            log_swallowed("labelling the map frame", level=logging.DEBUG)
        return txt

    def refresh(self, *_):
        """Redraw the selected (output x parameter x quantity) map."""
        ms = self._mapset
        if ms is None:
            return
        entry = self._current_entry()
        var = self.cb_field.currentData()
        path = self.cb_param.currentData()
        if entry is None or var is None or path is None:
            return
        quantity, mode, qlabel, diverging = entry
        S = ms.get(var, path, quantity)
        if S is None or S.ndim != 2 or S.size == 0:
            self.lbl_hint.setText(
                "No %s map for %s / %s in this study."
                % (qlabel, ms.field_label(var), ms.param_label(path)))
            return
        if not self._push_mesh():
            return
        aggregate = self.chk_aggregate.isChecked()
        frame = int(self.sld_frame.value())
        try:
            values, frame_txt = mio.reduce_map(S, mode=mode, frame=frame,
                                               aggregate=aggregate)
        except ValueError:
            log_swallowed("reducing the sensitivity map", level=logging.DEBUG)
            return
        self.lbl_frame.setText("agg" if aggregate else self._frame_text(frame))
        vmin, vmax, cmap = mio.color_range(values, diverging)
        title = "%s · %s — %s (%s)" % (
            qlabel, ms.field_label(var), ms.param_label(path), frame_txt)
        self.viewer.set_values(values, vmin=vmin, vmax=vmax, cmap=cmap,
                               title=title)

    # =====================================================================
    # Persistence / detach
    # =====================================================================
    def _on_save(self):
        if self._mapset is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sensitivity maps", "sensitivity_maps.npz",
            "NumPy archives (*.npz);;All files (*)")
        if not path:
            return
        try:
            written = mio.save_npz(self._mapset, path)
        except Exception as e:
            self.lbl_hint.setStyleSheet("color: #b91c1c;")
            self.lbl_hint.setText("Could not save the maps: %s" % e)
            return
        self.lbl_hint.setStyleSheet("color: #15803d;")
        self.lbl_hint.setText("Maps saved to %s" % written)

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load sensitivity maps", "",
            "NumPy archives (*.npz);;All files (*)")
        if not path:
            return
        self.load_file(path)

    def load_file(self, path):
        """Load a .npz study into the panel. Returns True on success; on
        failure the panel keeps what it had and the hint says why."""
        try:
            mapset = mio.load_npz(path)
        except mio.MapLoadError as e:
            self.lbl_hint.setStyleSheet("color: #b91c1c;")
            self.lbl_hint.setText(str(e))
            return False
        except Exception as e:                        # pragma: no cover
            self.lbl_hint.setStyleSheet("color: #b91c1c;")
            self.lbl_hint.setText("Could not load %s: %s" % (path, e))
            return False
        self.lbl_hint.setStyleSheet("color: #6b7280;")
        self.set_map_set(mapset, hint="Loaded %s — %s"
                         % (path, mio.describe(mapset)))
        return True

    def _on_detach(self):
        if self._mapset is None:
            return
        win = SensitivityMapWindow(self._mapset, parent=self)
        self._windows.append(win)
        win.show()
        win.raise_()
        return win


class SensitivityMapWindow(QWidget):
    """A standalone window holding one SensitivityMapPanel.

    Qt.Window so it gets its own frame; parented to the tab so it is
    destroyed with it rather than leaking when the app closes."""

    def __init__(self, mapset, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Sensitivity map")
        self.resize(900, 720)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.panel = SensitivityMapPanel(self, allow_detach=False)
        lay.addWidget(self.panel)
        self.panel.set_map_set(mapset)
