# -*- coding: utf-8 -*-
"""
Sensitivity tab — local Jacobian (finite differences) or Morris screening.

Lets the user:
  * tick which model parameters to vary (Ref = current model value, in
    displayed units), set the FD step (Jacobian) or Min/Max (Morris),
  * choose the scalar QoI and, for the Jacobian, the ROI fields,
  * generate the plan, run it through Abaqus in a background thread, then
    read the results table, the ranking chart and the per-element maps.

Each campaign gets its own study folder (config.json records the full plan);
the per-element maps (.npz arrays + PNG images) are written to its
`sensitivity_maps/` sub-folder.
"""
from __future__ import annotations

import copy
import threading
import warnings
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QCheckBox, QPlainTextEdit, QAbstractItemView, QSplitter, QComboBox,
    QProgressBar, QTabWidget, QFileDialog, QSlider, QDoubleSpinBox,
)
from PySide6.QtCore import QThread, QTimer

from gui.sensitivity import param_registry as pr
from gui.sensitivity import morris_plan as mp
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import runner_core as rc
from gui.sensitivity import export_results as xr
from gui.sensitivity import map_export as mx
from gui.sensitivity import zoi_proposal as zp
from gui.sensitivity.run_worker import SensitivityRunWorker
from gui.core.remote_exec import is_remote, launch_problems
from gui.widgets.field_viewer import FieldViewer
from gui.core.sta_parser import parse_sta
from gui.results import qoi as qoi_mod
from gui.core.logging_util import log_swallowed
import logging

# QoI ticked by default — the ones we can also measure on the planing rig
# (cutting/feed forces and peak temperature).
_DEFAULT_QOIS = ("Fx_mean", "Fy_mean", "T_max")

# Geometric / mesh parameters are excluded from sensitivity: changing them
# re-runs `discretize` (re-meshing), which can stall the identification.
# Tool and Eulerian-domain dimensions get a dedicated dimension-optimisation
# tab later. The element size (the discretize step) is excluded for the
# same reason.
_EXCLUDED_CATEGORIES = {"Géométrie outil", "Géométrie pièce"}
_EXCLUDED_PATHS = {"elem_size"}


class SensitivityTab(QWidget):
    # Emitted from the map-export thread: (output folder, files written,
    # error message or "").
    _mapsExported = Signal(str, int, str)
    # ZOI proposed from the maps, (xmin, xmax, ymin, ymax) [mm]: the main
    # window copies it into the Model tab.
    zoiProposed = Signal(tuple)

    def __init__(self, cfg, prefs_getter=None, cpus_getter=None,
                 profile_name_getter=None):
        super().__init__()
        self.cfg = cfg
        self._prefs_getter = prefs_getter
        self._profile_name_getter = profile_name_getter
        self._cpus_getter = cpus_getter
        self.plan = None                 # last generated JacobianPlan
        self.plan_kind = "jacobian"
        self.selected_qois = []          # list[QoISpec]
        self.plan_field_vars = []        # ROI fields chosen with the plan
        self._thread = None              # QThread for the run worker
        self._worker = None              # SensitivityRunWorker
        self._last_result = None         # rc.RunResult
        # Per-element sensitivity maps: {var: {param_path: S (n_frames,n_elem)}}
        self._field_maps = {}
        self._map_param_paths = []       # ordered, matches cb_map_param
        self._map_field_vars = []        # ordered, matches cb_map_field
        self._map_n_frames = 0
        self._map_mesh_set = False
        self._per_run_sec = None         # measured/estimated wall-clock per run
        self._per_frame_sec = None       # measured wall-clock between two frames
        self._cur_frame = None           # (current, total) for the running run
        self._run_t0 = {}                # run index -> monotonic start
        self._run_durations = []         # durations of SUCCESSFUL finished runs
        self._n_finished = 0             # finished runs, successful or not
        self._running_index = None       # index of the run in progress
        self._run_workdir = None         # workdir of the active run batch
        self._run_total = 0
        self._sta_timer = None           # live .sta poller during a run
        self._row_spec = {}              # table row -> ParamSpec
        self._table_units = None         # UnitSystem the table is shown in
        self._run_field_vars = None      # ROI fields of the active/last run
        self._run_plan = None            # plan of the active/last run
        self._run_full_domain = False    # whole-domain extraction of that run
        self._map_mesh = None            # (nodes_xy, face_idx) of the maps
        self._map_extra = {}             # centroids / frame times for export
        self._map_schemes = {}           # {var: {path: FD scheme used}}
        self._maps_thread = None         # background map export
        # Base-run EVF / frame times / extracted zone of the maps, for the
        # ZOI proposal (material mask, window T, edge check).
        self._map_base_evf = None
        self._map_times = None
        self._map_extent = None
        self._map_full_domain = False
        self._zoi_proposal = None        # last zoi_proposal.ZoiProposal
        # Callable -> {"eps": {Vx, Vy, T, EVF, ...}, "window": (a, b)} from
        # the Model tab (set by the main window); None in standalone use.
        self._model_settings_getter = None
        self._mapsExported.connect(self._on_maps_exported)

        root = QVBoxLayout(self)

        intro = QLabel(
            "Local sensitivity (Jacobian by finite differences): tick the "
            "parameters to vary, set the step (Delta or Delta%), pick the "
            "QoI, then generate and run. The Ref column is the base point; "
            "Min/Max define a trust region. Cost = k+1 runs (forward/"
            "backward) or 2k+1 (central)."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        split = QSplitter(Qt.Vertical)
        root.addWidget(split, 1)

        # ---- Parameters table ------------------------------------------
        param_box = QGroupBox("Parameters to vary")
        pv = QVBoxLayout(param_box)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Vary", "Parameter", "Ref", "Min", "Max", "Delta", "Delta%",
             "Norm", "Unit"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (0, 2, 3, 4, 5, 6, 7, 8):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        pv.addWidget(self.table)
        # (param_box is placed in a horizontal splitter together with the
        #  controls at the end of __init__, so the output panel below can
        #  span the full width.)

        # ---- QoI + controls + preview ----------------------------------
        bottom = QWidget()
        bl = QVBoxLayout(bottom)

        # ---- Method selector -------------------------------------------
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.cb_method = QComboBox()
        self.cb_method.addItems(["Jacobian (finite differences)",
                                 "Morris (global screening)"])
        self.cb_method.currentIndexChanged.connect(self._on_method_changed)
        self.cb_method.currentIndexChanged.connect(self._mark_plan_stale)
        method_row.addWidget(self.cb_method)
        method_row.addSpacing(16)

        # Jacobian-only controls
        self.lbl_scheme = QLabel("FD scheme:")
        method_row.addWidget(self.lbl_scheme)
        self.cb_scheme = QComboBox()
        self.cb_scheme.addItems(["central", "forward", "backward"])
        self.cb_scheme.currentIndexChanged.connect(self._update_cost)
        self.cb_scheme.currentIndexChanged.connect(self._mark_plan_stale)
        method_row.addWidget(self.cb_scheme)

        # Morris-only controls
        self.lbl_traj = QLabel("Trajectories N:")
        method_row.addWidget(self.lbl_traj)
        self.spin_traj = QSpinBox()
        self.spin_traj.setRange(2, 1000)
        self.spin_traj.setValue(10)
        self.spin_traj.setToolTip(
            "Number of Morris trajectories. Total runs = N × (k+1) for k\n"
            "parameters. 10–20 is typical for screening.")
        self.spin_traj.valueChanged.connect(self._update_cost)
        self.spin_traj.valueChanged.connect(self._mark_plan_stale)
        method_row.addWidget(self.spin_traj)
        self.lbl_levels = QLabel("Grid levels:")
        method_row.addWidget(self.lbl_levels)
        self.spin_levels = QSpinBox()
        self.spin_levels.setRange(2, 20)
        self.spin_levels.setValue(4)
        self.spin_levels.setToolTip("Morris grid levels p (4 is the common default).")
        self.spin_levels.valueChanged.connect(self._mark_plan_stale)
        method_row.addWidget(self.spin_levels)

        self.lbl_hint = QLabel("step = Delta per parameter")
        self.lbl_hint.setStyleSheet("color: #6b7280;")
        method_row.addWidget(self.lbl_hint)
        method_row.addStretch(1)
        bl.addLayout(method_row)
        self._on_method_changed()   # set initial visibility

        qoi_box = QGroupBox("Quantities of interest (QoI) to screen")
        qg = QGridLayout(qoi_box)
        self._qoi_checks = {}
        for i, q in enumerate(qoi_mod.REGISTRY):
            cb = QCheckBox("%s  [%s]" % (q.label, q.unit))
            cb.setChecked(q.id in _DEFAULT_QOIS)
            self._qoi_checks[q.id] = cb
            cb.toggled.connect(self._mark_plan_stale)
            qg.addWidget(cb, i // 2, i % 2)
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Warm-up (force history):"))
        self.spin_warmup = QDoubleSpinBox()
        self.spin_warmup.setRange(0.0, 0.9)
        self.spin_warmup.setSingleStep(0.05)
        self.spin_warmup.setDecimals(2)
        self.spin_warmup.setValue(0.0)
        self.spin_warmup.setToolTip(
            "Fraction of the RF1/RF2 history ignored at the start (tool\n"
            "entering the material) before the force QoI are computed\n"
            "(mean AND max). 0 = use the whole signal. Field QoI (T_max,\n"
            "PEEQ_max) are not affected. Applied when the run is analysed.")
        wrow.addWidget(self.spin_warmup)
        wrow.addStretch(1)
        qg.addLayout(wrow, (len(qoi_mod.REGISTRY) + 1) // 2, 0, 1, 2)
        bl.addWidget(qoi_box)

        # Field QoI: screen how much each parameter moves whole Eulerian
        # fields in the ROI (SSD vs the base run). Jacobian only. Vx, Vy,
        # TEMP and EVF are the fields the Model tab sizes the model on; V is
        # the velocity magnitude.
        field_box = QGroupBox("Field QoI (SSD, same FD scheme)")
        fg = QHBoxLayout(field_box)
        self._field_checks = {}
        for var, label in (("V1", "Vx"), ("V2", "Vy"), ("TEMP", "TEMP"),
                           ("EVF", "EVF (chip)"), ("V", "|V|")):
            cb = QCheckBox(label)
            self._field_checks[var] = cb
            cb.toggled.connect(self._mark_plan_stale)
            fg.addWidget(cb)
        fg.addSpacing(16)
        self.chk_full_domain = QCheckBox("Whole Eulerian domain")
        self.chk_full_domain.setToolTip(
            "Extract the fields on the whole Eulerian instance instead of the "
            "ROI box (Geometry tab). The ROI of the model is not changed; the "
            "result files are larger.")
        self.chk_full_domain.toggled.connect(self._mark_plan_stale)
        fg.addWidget(self.chk_full_domain)
        fg.addStretch(1)
        bl.addWidget(field_box)

        ctrl = QHBoxLayout()
        ctrl.addStretch(1)
        self.lbl_cost = QLabel("—")
        f = QFont(); f.setBold(True); self.lbl_cost.setFont(f)
        ctrl.addWidget(self.lbl_cost)
        self.btn_gen = QPushButton("Generate plan")
        self.btn_gen.clicked.connect(self._on_generate)
        ctrl.addWidget(self.btn_gen)
        ctrl.addWidget(QLabel("CPUs:"))
        self.lbl_cpus = QLabel("—")
        self.lbl_cpus.setToolTip("CPU cores used per run — synchronised with "
                                 "the Job tab (set it there).")
        ctrl.addWidget(self.lbl_cpus)
        self.btn_run = QPushButton("Run plan")
        self.btn_run.setToolTip("Launch every profile through Abaqus, "
                                "sequentially, then analyse.")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        ctrl.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        ctrl.addWidget(self.btn_cancel)
        self.btn_export = QPushButton("Save results…")
        self.btn_export.setToolTip("Export the sensitivity table and the "
                                   "field-SSD ranking to CSV.")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        ctrl.addWidget(self.btn_export)
        bl.addLayout(ctrl)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        bl.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        bl.addWidget(self.progress)

        # Bottom sub-panel: Plan preview | live run log | results ranking.
        self.tabs_out = QTabWidget()
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QFont("Consolas, monospace"))
        self.preview.setPlaceholderText(
            "The generated plan (one row per run) appears here.")
        self.tabs_out.addTab(self.preview, "Plan")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas, monospace"))
        self.log.setPlaceholderText("Abaqus output streams here during a run.")
        self.tabs_out.addTab(self.log, "Run log")

        self.results_table = QTableWidget(0, 0)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabs_out.addTab(self.results_table, "Results")

        chart_w = QWidget(); cv = QVBoxLayout(chart_w)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("QoI:"))
        self.cb_chart_qoi = QComboBox()
        self.cb_chart_qoi.currentIndexChanged.connect(self._draw_chart)
        crow.addWidget(self.cb_chart_qoi)
        self.lbl_chart_rank = QLabel("Rank by:")
        crow.addWidget(self.lbl_chart_rank)
        self.cb_chart_rank = QComboBox()
        self.cb_chart_rank.addItem("Sensitivity (as set per row)",
                                   "sensitivity")
        self.cb_chart_rank.addItem("Elasticity (dimensionless)", "elasticity")
        self.cb_chart_rank.setToolTip(
            "Sensitivity: dQ/dx in QoI unit per parameter unit, or the "
            "elasticity where Norm is ticked.\nElasticity: (dQ/Q)/(dx/x), "
            "comparable across parameters of different units (temperatures "
            "in kelvin).")
        self.cb_chart_rank.currentIndexChanged.connect(self._draw_chart)
        crow.addWidget(self.cb_chart_rank)
        crow.addStretch(1)
        cv.addLayout(crow)
        self.lbl_chart_note = QLabel("")
        self.lbl_chart_note.setWordWrap(True)
        self.lbl_chart_note.setStyleSheet("color: #b45309;")
        cv.addWidget(self.lbl_chart_note)
        self._fig = Figure(figsize=(5, 3))
        self._canvas = FigureCanvas(self._fig)
        cv.addWidget(self._canvas, 1)
        self.tabs_out.addTab(chart_w, "Chart")

        # ---- Sensitivity maps tab (per-element dF/dparam) ---------------
        # Reuses the Results FieldViewer. Two dropdowns pick the (parameter x
        # field) map; a toggle switches signed vs magnitude; a frame slider
        # scrubs time, with an aggregate option (mean if signed, RMS if
        # magnitude) over all frames.
        maps_w = QWidget(); mvl = QVBoxLayout(maps_w)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Parameter:"))
        self.cb_map_param = QComboBox()
        mrow.addWidget(self.cb_map_param)
        mrow.addWidget(QLabel("Field:"))
        self.cb_map_field = QComboBox()
        mrow.addWidget(self.cb_map_field)
        self.chk_map_signed = QCheckBox("Signed")
        self.chk_map_signed.setChecked(True)
        self.chk_map_signed.setToolTip(
            self._signed_tooltip("central"))
        mrow.addWidget(self.chk_map_signed)
        self.chk_map_aggregate = QCheckBox("Aggregate over time")
        self.chk_map_aggregate.setToolTip(
            "Reduce all frames to one map: time-mean if Signed, time-RMS if "
            "magnitude.")
        mrow.addWidget(self.chk_map_aggregate)
        mrow.addWidget(QLabel("Frame:"))
        self.sld_map_frame = QSlider(Qt.Horizontal)
        self.sld_map_frame.setMinimum(0); self.sld_map_frame.setMaximum(0)
        self.sld_map_frame.setEnabled(False)
        mrow.addWidget(self.sld_map_frame, 1)
        self.lbl_map_frame = QLabel("-")
        mrow.addWidget(self.lbl_map_frame)
        mvl.addLayout(mrow)
        self.fv_map = FieldViewer()
        mvl.addWidget(self.fv_map, 1)
        self.lbl_map_hint = QLabel(
            "Run a Jacobian plan with at least one ROI field ticked to get "
            "per-element sensitivity maps.")
        self.lbl_map_hint.setStyleSheet("color: #6b7280;")
        mvl.addWidget(self.lbl_map_hint)
        # ZOI proposed from the maps: S*(e) = max over (field, parameter) of
        # mean_T |dq/dp * delta| / eps_q (eps_q and T from the Model tab);
        # ZOI = smallest rectangle containing every element with S* >= 1.
        zrow = QHBoxLayout()
        self.btn_zoi_propose = QPushButton("Propose ZOI (S* \u2265 1)")
        self.btn_zoi_propose.setToolTip(
            "S* = max over (field, parameter) of mean over T of "
            "|dq/dp \u00b7 \u03b4| / \u03b5_q, with \u03b5_q and the window T "
            "of the Model tab; Vx, Vy, T masked by EVF \u2265 0.5 of the base "
            "run. ZOI = smallest rectangle containing every element with "
            "S* \u2265 1.")
        self.btn_zoi_propose.setEnabled(False)
        self.btn_zoi_propose.clicked.connect(self._on_propose_zoi)
        zrow.addWidget(self.btn_zoi_propose)
        self.btn_zoi_apply = QPushButton("Copy ZOI to the Model tab")
        self.btn_zoi_apply.setEnabled(False)
        self.btn_zoi_apply.clicked.connect(self._on_apply_zoi)
        zrow.addWidget(self.btn_zoi_apply)
        self.lbl_zoi = QLabel("")
        self.lbl_zoi.setWordWrap(True)
        zrow.addWidget(self.lbl_zoi, 1)
        mvl.addLayout(zrow)
        self.tabs_out.addTab(maps_w, "Maps")
        # Wire map controls (no-op until maps are computed).
        self.cb_map_param.currentIndexChanged.connect(self._refresh_map)
        self.cb_map_field.currentIndexChanged.connect(self._refresh_map)
        self.chk_map_signed.toggled.connect(self._refresh_map)
        self.chk_map_aggregate.toggled.connect(self._on_map_aggregate_toggled)
        self.sld_map_frame.valueChanged.connect(self._refresh_map)

        bl.addStretch(1)          # keep the controls top-aligned in their column

        # ---- Final assembly --------------------------------------------
        # Top row: parameters table (left) and controls (right) side by side.
        # Bottom: the Plan/Run log/Results/Chart panel, full width, larger.
        top_split = QSplitter(Qt.Horizontal)
        top_split.addWidget(param_box)
        top_split.addWidget(bottom)
        top_split.setStretchFactor(0, 3)
        top_split.setStretchFactor(1, 2)
        top_split.setSizes([560, 440])

        split.addWidget(top_split)
        split.addWidget(self.tabs_out)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 480])

        self._populate_table()
        self._update_cost()

    # ------------------------------------------------------------------
    # Build the parameter table from the registry
    # ------------------------------------------------------------------
    def _temp_unit(self) -> str:
        return getattr(self.cfg.ui, "temp_unit", "C")

    def _populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        tu = self._temp_unit()
        self._table_units = pr.current_system(tu)
        for category, specs in pr.registry_by_category().items():
            if category in _EXCLUDED_CATEGORIES:
                continue
            specs = [s for s in specs if s.path not in _EXCLUDED_PATHS]
            if not specs:
                continue
            # category header row
            r = self.table.rowCount()
            self.table.insertRow(r)
            head = QTableWidgetItem(category)
            fnt = head.font(); fnt.setBold(True); head.setFont(fnt)
            head.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(r, 0, head)
            self.table.setSpan(r, 0, 1, 9)

            for spec in specs:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self._row_spec[r] = spec
                lo, hi = pr.default_display_bounds(self.cfg, spec, tu)
                ref = pr.get_display(self.cfg, spec, tu)
                delta = (hi - lo) / 2.0
                # col 0: "vary" checkbox
                chk = QTableWidgetItem()
                chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                chk.setCheckState(Qt.Unchecked)
                self.table.setItem(r, 0, chk)
                # col 1: label (read-only)
                name = QTableWidgetItem(spec.label)
                name.setFlags(Qt.ItemIsEnabled)
                name.setToolTip(spec.path)
                self.table.setItem(r, 1, name)
                # col 2: reference value (read-only) — base point, FIRST
                it_ref = QTableWidgetItem(_fmt(ref))
                it_ref.setFlags(Qt.ItemIsEnabled)
                it_ref.setToolTip("Reference (default) value — the Jacobian "
                                  "base point.")
                self.table.setItem(r, 2, it_ref)
                # cols 3/4: trust region min/max (editable)
                self.table.setItem(r, 3, QTableWidgetItem(_fmt(lo)))
                self.table.setItem(r, 4, QTableWidgetItem(_fmt(hi)))
                # col 5: FD step Delta (absolute, editable)
                self.table.setItem(r, 5, QTableWidgetItem(_fmt(delta)))
                # col 6: Delta% = Delta / |Ref| * 100 (editable, synced)
                pct = (100.0 * delta / abs(ref)) if ref else 0.0
                self.table.setItem(r, 6, QTableWidgetItem(_fmt(pct)))
                # col 7: Normalize checkbox (report elasticity)
                nrm = QTableWidgetItem()
                nrm.setCheckState(Qt.Unchecked)
                if spec.is_temp:
                    # °C has no physical zero: (dx/x) would depend on the
                    # displayed unit. Temperatures keep the raw dQ/dT.
                    nrm.setFlags(Qt.ItemIsEnabled)
                    nrm.setToolTip("No elasticity for a temperature: the "
                                   "ratio dx/x would depend on the unit "
                                   "(°C or K). The raw dQ/dT is reported.")
                else:
                    nrm.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    nrm.setToolTip("Report the dimensionless elasticity "
                                   "(dQ/Q)/(dx/x) instead of the raw dQ/dx. "
                                   "The real Min/Max/Delta below stay "
                                   "unchanged. Not available when the QoI "
                                   "is a temperature (raw value shown, "
                                   "marked 'raw').")
                self.table.setItem(r, 7, nrm)
                # col 8: unit (read-only)
                unit = QTableWidgetItem(spec.unit_str(tu))
                unit.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(r, 8, unit)
        self.table.blockSignals(False)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _selected_rows(self):
        out = []
        for r, spec in self._row_spec.items():
            it = self.table.item(r, 0)
            if it is not None and it.checkState() == Qt.Checked:
                out.append((r, spec))
        return out

    def _on_item_changed(self, item):
        col = item.column()
        if col in (0, 3, 4, 5, 6, 7):      # anything the plan is built from
            self._mark_plan_stale()
        if col == 0:                       # "vary" checkbox toggled
            self._update_cost()
            return
        if col in (5, 6, 3, 4, 7):         # delta / delta% / min / max / norm
            self._sync_row(item.row(), col)

    # Columns: 0 Vary | 1 Parameter | 2 Ref | 3 Min | 4 Max | 5 Delta |
    #          6 Delta% | 7 Norm | 8 Unit
    def _sync_row(self, row, col):
        spec = self._row_spec.get(row)
        if spec is None:
            return
        ref = self._cell_float(row, 2)
        if ref is None:
            return
        self.table.blockSignals(True)
        try:
            if col == 5:                   # Delta edited -> recompute Delta%
                d = self._cell_float(row, 5)
                if d is not None and ref:
                    self._set_cell(row, 6, _fmt(100.0 * d / abs(ref)))
            elif col == 6:                 # Delta% edited -> recompute Delta
                p = self._cell_float(row, 6)
                if p is not None:
                    self._set_cell(row, 5, _fmt(p / 100.0 * abs(ref)))
            self._flag_trust_region(row, ref)
        finally:
            self.table.blockSignals(False)

    def _flag_trust_region(self, row, ref):
        """Détrompeur: colour Delta red if Ref ± Delta leaves [Min, Max]."""
        d = self._cell_float(row, 5)
        lo = self._cell_float(row, 3)
        hi = self._cell_float(row, 4)
        item = self.table.item(row, 5)
        if item is None:
            return
        from PySide6.QtGui import QColor
        bad = (d is not None and lo is not None and hi is not None
               and (ref - d < lo - 1e-12 or ref + d > hi + 1e-12))
        item.setForeground(QColor("#b91c1c") if bad else QColor("#111111"))
        item.setToolTip("Ref ± Delta leaves the [Min, Max] trust region."
                        if bad else "")

    def _cell_float(self, row, col):
        it = self.table.item(row, col)
        if it is None:
            return None
        try:
            return float(it.text())
        except (ValueError, AttributeError):
            return None

    def _set_cell(self, row, col, text):
        it = self.table.item(row, col)
        if it is None:
            self.table.setItem(row, col, QTableWidgetItem(text))
        else:
            it.setText(text)

    # ------------------------------------------------------------------
    # Method handling (Jacobian only)
    # ------------------------------------------------------------------
    def _scheme(self) -> str:
        return self.cb_scheme.currentText()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_from_model()

    def refresh_from_model(self):
        """Public hook: mirror the current Numerical Model into this tab.
        Called by MainWindow when Sensitivity becomes the visible page
        (showEvent is unreliable for a doubly-nested tab page)."""
        self._resync_reference_values()   # mirror the current Numerical Model
        if self._plan_base_changed():
            self._mark_plan_stale(reason="a varied parameter's model value "
                                  "changed")
        self._update_cost()          # refresh CPU mirror + cost estimate

    def _resync_reference_values(self):
        """Refresh the Ref column (col 2) from the *current* ModelConfig, so
        edits made in the Numerical Model tabs are reflected here. For rows
        the user is actively configuring (Vary ticked), Min/Max and Delta%
        are preserved and only the absolute Delta is recomputed from the new
        Ref; untouched rows get their default trust region recomputed.

        If the display unit system changed (Settings, or the °C/K switch),
        the ticked rows' Min/Max/Delta are CONVERTED to the new units (same
        physical values) and Delta% is recomputed from them, instead of being
        kept as numbers that now mean something else. A plan generated under
        the old units is invalidated."""
        tu = self._temp_unit()
        new_units = pr.current_system(tu)
        old_units = self._table_units
        units_changed = old_units is not None and old_units != new_units
        self.table.blockSignals(True)
        try:
            for r, spec in self._row_spec.items():
                try:
                    new_ref = pr.get_display(self.cfg, spec, tu)
                except Exception:
                    log_swallowed("resyncing Ref for %s" % spec.path,
                                  level=logging.DEBUG)
                    continue
                self._set_cell(r, 2, _fmt(new_ref))
                # keep the unit column in sync too (it can depend on tu)
                self._set_cell(r, 8, spec.unit_str(tu))
                chk = self.table.item(r, 0)
                is_checked = (chk is not None
                              and chk.checkState() == Qt.Checked)
                if is_checked and units_changed:
                    self._convert_row_units(r, spec, old_units, new_units,
                                            new_ref)
                    self._flag_trust_region(r, new_ref)
                elif is_checked:
                    # preserve the user's trust region + relative step;
                    # rescale only the absolute Delta to the new Ref.
                    pct = self._cell_float(r, 6)
                    if pct is not None and new_ref:
                        self._set_cell(r, 5, _fmt(pct / 100.0 * abs(new_ref)))
                    self._flag_trust_region(r, new_ref)
                else:
                    lo, hi = pr.default_display_bounds(self.cfg, spec, tu)
                    delta = (hi - lo) / 2.0
                    self._set_cell(r, 3, _fmt(lo))
                    self._set_cell(r, 4, _fmt(hi))
                    self._set_cell(r, 5, _fmt(delta))
                    pct = (100.0 * delta / abs(new_ref)) if new_ref else 0.0
                    self._set_cell(r, 6, _fmt(pct))
                    self._flag_trust_region(r, new_ref)
        finally:
            self.table.blockSignals(False)
        self._table_units = new_units
        self._invalidate_plan_if_units_changed()

    def _convert_row_units(self, row, spec, old_units, new_units, new_ref):
        """Re-express a ticked row's Min, Max and Delta in `new_units`.
        Min/Max are absolute values (a temperature gets the °C/K offset);
        Delta is a difference, so it takes the scale but not the offset
        (conversions are affine: conv(d) - conv(0))."""
        def conv(v):
            return spec.to_display(spec.to_stored(v, system=old_units),
                                   system=new_units)
        for col in (3, 4):
            v = self._cell_float(row, col)
            if v is not None:
                self._set_cell(row, col, _fmt(conv(v)))
        d = self._cell_float(row, 5)
        if d is not None:
            d_new = conv(d) - conv(0.0)
            self._set_cell(row, 5, _fmt(d_new))
            self._set_cell(row, 6, _fmt(100.0 * d_new / abs(new_ref))
                           if new_ref else _fmt(0.0))

    def _plan_units_stale(self) -> bool:
        """True if the current plan was generated under other display units
        than the ones the table now shows."""
        plan_units = getattr(self.plan, "unit_system", None)
        return (plan_units is not None
                and plan_units != pr.current_system(self._temp_unit()))

    def _mark_plan_stale(self, *_, reason="the settings changed"):
        """Discard the generated plan when anything it was built from is
        edited (table, QoI, fields, method settings, model values): running
        it would silently use the old selection. Never during a run."""
        if self.plan is None or self._thread is not None:
            return
        self.plan = None
        self.btn_run.setEnabled(False)
        self.preview.clear()
        self._warn("Plan discarded because %s since it was generated: "
                   "generate it again." % reason)

    def _plan_base_changed(self) -> bool:
        """True if a varied parameter's model value (the Jacobian base point)
        is no longer the one the plan was generated from."""
        plan = self.plan
        if plan is None or self.plan_kind != "jacobian":
            return False
        for spec, x0 in zip(plan.specs, plan.base):
            try:
                now = pr.get_display(self.cfg, spec, plan.temp_unit,
                                     system=plan.unit_system)
            except Exception:
                return True
            if abs(now - x0) > 1e-9 * max(1.0, abs(x0)):
                return True
        return False

    def _invalidate_plan_if_units_changed(self):
        # Never while a campaign runs: the worker and the maps still use it.
        if self._thread is not None or not self._plan_units_stale():
            return
        self.plan = None
        self.btn_run.setEnabled(False)
        self.preview.clear()
        self._warn("The unit system changed since the plan was generated: "
                   "the plan was discarded, generate it again.")

    def _current_cpus(self) -> int:
        if self._cpus_getter:
            try:
                return int(self._cpus_getter())
            except Exception:
                log_swallowed("reading CPU count from getter",
                              level=logging.DEBUG)
        return 1

    def _n_runs(self) -> int:
        k = len(self._selected_rows())
        if not k:
            return 0
        if self._method() == "morris":
            return mp.n_runs(k, self.spin_traj.value())
        return jac.n_runs(k, self._scheme())

    def _method(self) -> str:
        return "morris" if self.cb_method.currentIndex() == 1 else "jacobian"

    def _on_method_changed(self, *_):
        morris = self._method() == "morris"
        for w in (self.lbl_scheme, self.cb_scheme):
            w.setVisible(not morris)
        for w in (self.lbl_traj, self.spin_traj, self.lbl_levels, self.spin_levels):
            w.setVisible(morris)
        self.lbl_hint.setText(
            "screens Min..Max globally (mu*, sigma)" if morris
            else "step = Delta per parameter")
        self._update_cost()

    def _update_cost(self):
        if not hasattr(self, "lbl_cpus"):
            return   # called during construction before the cost labels exist
        self.lbl_cpus.setText(str(self._current_cpus()))
        k = len(self._selected_rows())
        if k == 0:
            self.lbl_cost.setText("0 parameters selected")
            return
        if self._method() == "morris":
            runs = mp.n_runs(k, self.spin_traj.value())
            base = "k=%d  →  %d runs (Morris N=%d × (k+1))" % (
                k, runs, self.spin_traj.value())
        else:
            runs = jac.n_runs(k, self._scheme())
            base = "k=%d  →  %d runs (%s FD)" % (k, runs, self._scheme())
        # Total wall-clock is estimated live from the running job's .sta
        # (see the run section); before any run we can only give the count.
        if self._per_run_sec:
            base += "   ~%s total" % _fmt_duration(runs * self._per_run_sec)
        self.lbl_cost.setText(base)

    # ------------------------------------------------------------------
    # Generate the plan
    # ------------------------------------------------------------------
    def _collect_morris(self):
        selected = []
        for r, spec in self._selected_rows():
            try:
                lo = float(self.table.item(r, 3).text())   # Min
                hi = float(self.table.item(r, 4).text())   # Max
            except (ValueError, AttributeError):
                raise ValueError("%s: min/max must be numbers." % spec.label)
            if hi <= lo:
                raise ValueError("%s: Max must be greater than Min." % spec.label)
            selected.append((spec, lo, hi))
        return selected

    def _collect_jacobian(self):
        selected = []
        tu = self._temp_unit()
        for r, spec in self._selected_rows():
            try:
                delta = float(self.table.item(r, 5).text())
            except (ValueError, AttributeError):
                raise ValueError("%s: delta must be a number." % spec.label)
            norm = self.table.item(r, 7).checkState() == Qt.Checked
            x0 = pr.get_display(self.cfg, spec, tu, system=self._table_units)
            selected.append((spec, x0, delta, norm))
        return selected

    def _outside_trust_region(self, selected):
        """Labels of the parameters whose FD points leave [Min, Max]. Only
        the points the scheme evaluates count (forward: Ref+Delta,
        backward: Ref-Delta, central: both)."""
        scheme = self._scheme()
        rows = {spec.path: r for r, spec in self._selected_rows()}
        bad = []
        for spec, x0, d, _norm in selected:
            r = rows.get(spec.path)
            lo, hi = self._cell_float(r, 3), self._cell_float(r, 4)
            if lo is None or hi is None:
                continue
            pts = []
            if scheme in ("forward", "central"):
                pts.append(x0 + d)
            if scheme in ("backward", "central"):
                pts.append(x0 - d)
            tol = 1e-9 * max(1.0, abs(lo), abs(hi))
            if any(p < lo - tol or p > hi + tol for p in pts):
                bad.append(spec.label)
        return bad

    def _selected_qoi_specs(self):
        return [q for q in qoi_mod.REGISTRY
                if self._qoi_checks[q.id].isChecked()]

    def _selected_field_vars(self):
        return [v for v, cb in self._field_checks.items() if cb.isChecked()]

    def _on_generate(self):
        method = self._method()
        outside = []
        try:
            qois = self._selected_qoi_specs()
            field_vars = self._selected_field_vars()
            if method == "morris":
                # Morris screens scalar QoI globally (mu*, sigma). The field
                # (ROI) sensitivity is a Jacobian-only construction.
                if field_vars:
                    self._warn("Field (ROI) screening is only available with "
                               "the Jacobian method; ignoring the field "
                               "selection for Morris.")
                    field_vars = []
                if not qois:
                    self._warn("Tick at least one scalar QoI for Morris.")
                    return
                selected = self._collect_morris()
                if not selected:
                    self._warn("Tick at least one parameter to vary.")
                    return
                plan = mp.build_plan(selected, N=self.spin_traj.value(),
                                     num_levels=self.spin_levels.value(),
                                     temp_unit=self._temp_unit(),
                                     unit_system=self._table_units)
            else:
                if not qois and not field_vars:
                    self._warn("Tick at least one QoI (a scalar QoI, or a ROI "
                               "field).")
                    return
                selected = self._collect_jacobian()
                if not selected:
                    self._warn("Tick at least one parameter to vary.")
                    return
                # Not blocking: the plan is built, the status line warns.
                outside = self._outside_trust_region(selected)
                plan = jac.build_plan(selected, scheme=self._scheme(),
                                      temp_unit=self._temp_unit(),
                                      unit_system=self._table_units)
        except ValueError as e:
            self._warn(str(e))
            return
        except ImportError as e:
            self._warn("Morris needs the SALib package: %s\n"
                       "Install it (pip install SALib) and try again." % e)
            return
        except Exception as e:                       # pragma: no cover
            self._warn("Could not build the plan: %s" % e)
            return

        self.plan = plan
        self.plan_kind = method
        self.selected_qois = qois
        self.plan_field_vars = list(field_vars)
        qoi_names = [q.id for q in qois] + ["%s[field]" % v for v in field_vars]
        self.status.setStyleSheet("color: #15803d;")
        if method == "morris":
            self.status.setText(
                "Morris plan ready: %d parameters, N=%d trajectories, "
                "%d runs, %d QoI (%s), seed %d. Ready for the run step." % (
                    plan.k, plan.N, plan.n_runs, len(qoi_names),
                    ", ".join(qoi_names) if qoi_names else "—", plan.seed))
        else:
            self.status.setText(
                "Jacobian (%s FD) plan ready: %d parameters, %d runs, %d QoI "
                "(%s). Ready for the run step." % (
                    self._scheme(), plan.k, plan.n_runs, len(qoi_names),
                    ", ".join(qoi_names) if qoi_names else "—"))
            pcts = [self._cell_float(r, 6) for r, _s in self._selected_rows()]
            pcts = [abs(p) for p in pcts if p is not None]
            if field_vars and len(pcts) > 1 and \
                    max(pcts) - min(pcts) > 1e-6 * max(pcts):
                self.status.setText(
                    self.status.text() + "  Note: the \u0394% (rel) field "
                    "columns are comparable between parameters only with "
                    "the same Delta% on every row.")
            if outside:
                self.status.setStyleSheet("color: #b45309;")
                self.status.setText(
                    self.status.text() + "  Warning: Ref ± Delta leaves the "
                    "[Min, Max] trust region for %s." % ", ".join(outside))
        self._show_preview(plan)
        self.btn_run.setEnabled(True)
        self.tabs_out.setCurrentWidget(self.preview)

    # ------------------------------------------------------------------
    # Run the plan through Abaqus (background worker)
    # ------------------------------------------------------------------
    def _on_run(self):
        if self.plan is None:
            self._warn("Generate a plan first.")
            return
        if self._thread is not None:
            self._warn("A run is already in progress.")
            return
        if self._plan_units_stale():
            self._invalidate_plan_if_units_changed()
            return
        prefs = self._prefs_getter() if self._prefs_getter else None
        if prefs is None:
            self._warn("No preferences available (Abaqus command/script).")
            return
        from pathlib import Path
        problems = launch_problems(prefs, prefs.default_workdir)
        wd = Path(prefs.default_workdir)
        try:
            wd.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            problems.append("Cannot create workdir '%s': %s" % (wd, e))
        if problems:
            self._warn("Cannot launch:\n• " + "\n• ".join(problems)
                       + "\nFix the paths in Preferences.")
            return

        self.log.clear()
        self.tabs_out.setCurrentWidget(self.log)
        self.progress.setVisible(True)
        self.progress.setRange(0, self.plan.n_runs)
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_gen.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.status.setStyleSheet("color: #1d4ed8;")
        self.status.setText("Running %d simulations…" % self.plan.n_runs)

        # Timing state for the live wall-clock estimate (from the running
        # job's .sta, plus measured durations of finished runs).
        # Group this sensitivity campaign in its own timestamped folder
        # ({profile}_sensitivity_{stamp}) with a config.json; its run files
        # are prefixed with "sensitivity".
        from gui.core.run_output import create_study_dir
        try:
            _pname = (self._profile_name_getter()
                      if self._profile_name_getter else None)
        except Exception:
            _pname = None
        # Morris ignores the ROI fields (Jacobian-only construction): do not
        # record or pass them, so config.json says what was really run.
        field_vars = list(getattr(self, "plan_field_vars", []))
        cpus = self._current_cpus()
        _study_cfg = self._plan_record(field_vars, cpus)
        try:
            wd = create_study_dir(wd, _pname, "sensitivity", _study_cfg)
        except OSError:
            pass   # fall back to the flat working directory

        import time
        self._run_workdir = wd
        self._run_total = self.plan.n_runs
        self._run_t0 = {}
        self._run_durations = []
        self._n_finished = 0
        self._running_index = None
        self._per_run_sec = None
        self._per_frame_sec = None
        self._cur_frame = None
        self._failed_live = []           # run indices reported failed live
        self._run_clock0 = time.monotonic()
        self._sta_timer = QTimer(self)
        self._sta_timer.setInterval(3000)
        self._sta_timer.timeout.connect(self._poll_sta)
        self._sta_timer.start()

        self._run_field_vars = list(field_vars)
        self._run_plan = self.plan
        self._run_full_domain = (bool(field_vars)
                                 and self.chk_full_domain.isChecked())
        # The worker gets its own copy of the model: it expands the plan in
        # its thread, while the user may keep editing the live cfg here.
        self._worker = SensitivityRunWorker(
            self.plan, self.plan_kind, self.selected_qois,
            copy.deepcopy(self.cfg),
            abaqus_cmd=prefs.abaqus_cmd, abaqus_script=prefs.abaqus_script,
            workdir=str(wd), cpus=cpus,
            warmup_frac=float(self.spin_warmup.value()),
            job_prefix="sensitivity", field_vars=field_vars,
            remote_prefs=prefs if is_remote(prefs) else None,
            extract_full_domain=self._run_full_domain)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.runDone.connect(self._on_run_done)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._thread.start()

    def _plan_record(self, field_vars, cpus) -> dict:
        """Everything needed to understand -- and regenerate -- this campaign,
        for the study folder's config.json: the varied parameters with their
        base/step or bounds (in the plan's units, which are recorded), the
        method settings (Morris seed included) and the value of every
        parameter in every run, keyed to the job name on disk."""
        plan = self.plan
        tu = getattr(plan, "temp_unit", "C")
        system = getattr(plan, "unit_system", None)
        params = []
        for i, spec in enumerate(plan.specs):
            d = {"path": spec.path, "label": spec.label,
                 "unit": self._plan_unit(plan, spec)}
            if self.plan_kind == "jacobian":
                d.update(base=float(plan.base[i]), delta=float(plan.deltas[i]),
                         normalize=bool(plan.normalize[i]))
            else:
                lo, hi = plan.bounds[i]
                d.update(min=float(lo), max=float(hi))
            params.append(d)
        rec = {"plan_kind": self.plan_kind,
               "n_runs": int(plan.n_runs),
               "qois": [q.id for q in self.selected_qois],
               "field_vars": list(field_vars),
               "extract_full_domain": bool(field_vars)
                                      and self.chk_full_domain.isChecked(),
               "cpus": int(cpus),
               "warmup_frac": float(self.spin_warmup.value()),
               "temp_unit": tu,
               "unit_system": (system.to_dict()
                               if hasattr(system, "to_dict") else None),
               "varied_parameters": params}
        if self.plan_kind == "jacobian":
            rec["scheme"] = plan.scheme
            rows = jac.profile_table(plan)
        else:
            rec.update(N=int(plan.N), num_levels=int(plan.num_levels),
                       seed=plan.seed)
            rows = mp.profile_table(plan)
        runs = []
        for d in rows:
            run = {"run": d["run"], "job": "sensitivity_run%03d" % (d["run"] - 1)}
            if "kind" in d:
                run["kind"] = d["kind"]
            run["values"] = {p: float(d[p]) for p in plan.param_paths}
            runs.append(run)
        rec["runs"] = runs
        return rec

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status.setStyleSheet("color: #b45309;")
            self.status.setText("Cancelling: stopping the current Abaqus "
                                "job, no further run will start…")
            self.btn_cancel.setEnabled(False)

    def _on_run_done(self, index, ok):
        """Per-run completion, reported live by the worker. A failed run is
        flagged immediately in the log (and folded into the estimate line)
        instead of only surfacing in the final tally."""
        if not ok:
            if index not in self._failed_live:
                self._failed_live.append(index)
            self.log.appendPlainText(
                "[run %d] FAILED — no usable results (see output above)."
                % (index + 1))

    def _on_progress(self, done, total):
        import time
        now = time.monotonic()
        self._run_total = total
        # The run that was in progress just finished -> count it, and keep
        # its duration for the estimate only if it succeeded: a run that dies
        # at start-up would drag the per-run time down. (runDone, which fills
        # _failed_live, is emitted before this progress call.)
        if self._running_index is not None and self._running_index in self._run_t0:
            self._n_finished += 1
            dur = now - self._run_t0[self._running_index]
            failed = self._running_index in getattr(self, "_failed_live", [])
            if dur > 0 and not failed:
                self._run_durations.append(dur)
                self._per_run_sec = sum(self._run_durations) / len(self._run_durations)
        # Next run (if any) starts now.
        if done < total:
            self._running_index = done
            self._run_t0[done] = now
            self._cur_frame = None        # fresh .sta for the new run
        else:
            self._running_index = None
            self._cur_frame = None
        # Smooth bar on a 0..1000 scale (sub-run progress added by _poll_sta).
        self.progress.setRange(0, 1000)
        if total > 0:
            self.progress.setValue(int(round(self._n_finished
                                              / total * 1000)))
        self._update_estimate(self._n_finished, total)

    def _poll_sta(self):
        """Real-time progress of the running run from its .sta file.

        The total-time estimate is built explicitly from the three
        quantities the user reasons about:
            per-run  ≈ (wall-clock per frame) × (frames per run)
            total    ≈ per-run × (number of runs)
        The per-frame time is measured live as wall_time / frames_done, so
        an estimate appears after the very first output frame. Finished-run
        durations, when available, take over as the more reliable per-run
        baseline. The progress bar shows overall progress including the
        fraction of the current run already done."""
        if self._running_index is None or self._run_workdir is None:
            return
        sta = self._run_workdir / ("sensitivity_run%03d.sta"
                                   % self._running_index)
        cur_frac = 0.0
        try:
            snap = parse_sta(sta)
            if snap.is_ready():
                wall = _hms_to_sec(snap.wall_time)
                fcur = snap.frame_current
                ftot = snap.frame_total or int(
                    getattr(self.cfg.step, "n_frames", 0) or 0)
                if fcur and ftot:
                    self._cur_frame = (fcur, ftot)
                    cur_frac = max(0.0, min(1.0, fcur / float(ftot)))
                    if wall and fcur > 0:
                        self._per_frame_sec = wall / float(fcur)
                        # per-run from the explicit time-per-frame × n_frames
                        if not self._run_durations:
                            self._per_run_sec = self._per_frame_sec * float(ftot)
                elif snap.step_time is not None:
                    st = float(getattr(self.cfg.step, "sim_time", 0.0) or 0.0)
                    if st > 0:
                        cur_frac = max(0.0, min(1.0, snap.step_time / st))
        except Exception:
            log_swallowed("reading .sta for live progress", level=logging.DEBUG)
        # Finished-run durations are the most reliable per-run baseline.
        if self._run_durations:
            self._per_run_sec = sum(self._run_durations) / len(self._run_durations)
        # Smooth overall progress: finished runs + fraction of the current one.
        n_done = self._n_finished
        if self._run_total > 0:
            overall = (n_done + cur_frac) / self._run_total
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(round(overall * 1000)))
        self._update_estimate(n_done, self._run_total)

    def _update_estimate(self, done, total):
        import time
        elapsed = time.monotonic() - getattr(self, "_run_clock0", time.monotonic())
        n_done = self._n_finished
        cur = min(n_done + (1 if self._running_index is not None else 0), total)
        msg = "Run %d/%d" % (cur, total)
        if self._cur_frame is not None and self._running_index is not None:
            msg += "   ·   frame %d/%d" % self._cur_frame
        if self._per_frame_sec:
            msg += "   ·   ~%s/frame" % _fmt_duration(self._per_frame_sec)
        if self._per_run_sec:
            remaining = max(0, total - n_done) * self._per_run_sec
            # subtract the part of the current run already elapsed
            if self._running_index is not None and self._cur_frame:
                frac = self._cur_frame[0] / float(self._cur_frame[1])
                remaining = max(0.0, remaining - frac * self._per_run_sec)
            msg += ("   ·   ~%s/run   ·   ~%s remaining   ·   est. total ~%s"
                    % (_fmt_duration(self._per_run_sec),
                       _fmt_duration(remaining),
                       _fmt_duration(total * self._per_run_sec)))
        msg += "   ·   elapsed %s" % _fmt_duration(elapsed)
        n_failed = len(getattr(self, "_failed_live", []))
        if n_failed:
            msg += "   ·   %d failed so far" % n_failed
            self.status.setStyleSheet("color: #b45309;")
        else:
            self.status.setStyleSheet("color: #1d4ed8;")
        self.status.setText(msg)

    def _on_log(self, text):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def _on_run_finished(self, result):
        self._last_result = result
        self._teardown_thread()
        msg, warn = self._run_summary(result, scheme=self._result_scheme())
        self.status.setStyleSheet("color: #b45309;" if warn
                                  else "color: #15803d;")
        self.status.setText(msg)
        self._show_results(result)
        self._build_field_maps(result)
        self.btn_export.setEnabled(result.Y.shape[0] > 0)
        self.tabs_out.setCurrentWidget(self.results_table)
        if self._field_maps and self._run_workdir is not None:
            self._export_field_maps(Path(self._run_workdir) / mx.MAPS_SUBDIR)

    @staticmethod
    def _run_summary(result, scheme=None):
        """Status line for a finished (or cancelled) campaign, and whether it
        deserves a warning colour. Runs never launched after a Cancel are
        reported as such, not counted as successes."""
        total = int(result.Y.shape[0])
        n_att = getattr(result, "n_attempted", total)
        n_fail = len(result.failures)
        n_ok = n_att - n_fail
        if getattr(result, "cancelled", False):
            msg = ("Run cancelled: %d/%d runs launched — %d successful, "
                   "%d failed or interrupted, %d not run."
                   % (n_att, total, n_ok, n_fail, total - n_att))
        else:
            msg = "Run finished: %d/%d successful, %d failed." % (
                n_ok, total, n_fail)
        warn = bool(n_fail) or n_att < total
        if result.plan_kind == "morris":
            notes = []
            for qid in result.qoi_ids:
                a = result.analyses.get(qid, {})
                n_tr = a.get("n_trajectories")
                if n_tr is None:
                    continue
                if "error" in a:
                    notes.append("%s: not analysed (%d/%d complete "
                                 "trajectories, 2 needed)"
                                 % (qid, a.get("n_used", 0), n_tr))
                elif a.get("n_dropped"):
                    notes.append("%s: %d/%d trajectories" % (
                        qid, a["n_used"], n_tr))
            if notes:
                warn = True
                msg += (" Morris uses only complete trajectories (a failed "
                        "or missing run drops its trajectory) — "
                        + "; ".join(notes) + ".")
        if result.plan_kind == "jacobian" and scheme == "central":
            n_fb = sum(1 for a in result.analyses.values()
                       if isinstance(a, dict)
                       for d in a.values()
                       if isinstance(d, dict)
                       and d.get("scheme_used") in ("forward", "backward"))
            if n_fb:
                warn = True
                msg += (" %d value(s) fell back to a one-sided difference "
                        "because a perturbed run failed (marked fwd/bwd, "
                        "less accurate)." % n_fb)
        return msg + " See Results.", warn

    # =====================================================================
    # Per-element sensitivity maps
    # =====================================================================
    def _clear_map(self):
        self._map_mesh_set = False
        self._map_mesh = None
        self._map_extra = {}
        try:
            self.fv_map.clear()
        except Exception:
            log_swallowed("clearing the sensitivity-map viewer",
                          level=logging.DEBUG)

    def _set_map_controls_enabled(self, on):
        for w in (self.cb_map_param, self.cb_map_field, self.chk_map_signed,
                  self.chk_map_aggregate):
            w.setEnabled(on)
        self.sld_map_frame.setEnabled(
            on and not self.chk_map_aggregate.isChecked())

    def _build_field_maps(self, result):
        """Compute per-element sensitivity maps from the kept run bundles and
        populate the Maps tab. No-op (and clears) unless this was a Jacobian
        run with ROI field(s) and bundles were kept."""
        self._field_maps = {}
        self._map_param_paths = []
        self._map_field_vars = []
        self._map_n_frames = 0
        self._map_base_evf = None
        self._map_times = None
        self._map_extent = None
        self._zoi_proposal = None
        self.btn_zoi_propose.setEnabled(False)
        self.btn_zoi_apply.setEnabled(False)
        self.lbl_zoi.setText("")
        # The fields of THIS run (captured at launch), not the checkboxes as
        # they may have been edited since; tests without a launch fall back.
        field_vars = (list(self._run_field_vars)
                      if self._run_field_vars is not None
                      else self._selected_field_vars())
        bundles = getattr(result, "bundles", None)
        plan = self._run_plan or self.plan    # tests set .plan directly
        if (result.plan_kind != "jacobian" or not field_vars
                or not bundles or plan is None):
            self._clear_map()
            self._set_map_controls_enabled(False)
            self.lbl_map_hint.setText(
                "Run a Jacobian plan with at least one ROI field ticked to "
                "get per-element sensitivity maps.")
            return
        ref = bundles[0] if bundles else None
        inst = rc.eulerian_instance(ref) if ref is not None else None
        try:
            self._map_schemes = {}
            maps = rc.jacobian_field_maps(plan, bundles, field_vars,
                                          instance=inst,
                                          schemes_out=self._map_schemes)
        except Exception as e:
            log_swallowed("computing sensitivity maps", level=logging.WARNING)
            self._clear_map(); self._set_map_controls_enabled(False)
            self.lbl_map_hint.setText("Could not compute maps: %s" % e)
            return
        if not maps or not self._set_map_mesh(ref, inst):
            self._clear_map(); self._set_map_controls_enabled(False)
            self.lbl_map_hint.setText(
                "No field maps were produced (check the mesh / fields).")
            return
        self._field_maps = maps
        self._map_field_vars = list(field_vars)
        self._map_param_paths = list(plan.param_paths)
        self._store_zoi_inputs(ref, inst)
        for per in maps.values():
            for S in per.values():
                arr = np.asarray(S)
                if arr.ndim == 2 and arr.size:
                    self._map_n_frames = arr.shape[0]
                    break
            if self._map_n_frames:
                break
        # Populate dropdowns (block signals to avoid premature redraws).
        self.cb_map_param.blockSignals(True)
        self.cb_map_field.blockSignals(True)
        self.cb_map_param.clear()
        for p in self._map_param_paths:
            try:
                self.cb_map_param.addItem(pr.spec_for(p).label, p)
            except Exception:
                self.cb_map_param.addItem(p, p)
        self.cb_map_field.clear()
        self.cb_map_field.addItems(self._map_field_vars)
        self.cb_map_param.blockSignals(False)
        self.cb_map_field.blockSignals(False)
        self.sld_map_frame.blockSignals(True)
        self.sld_map_frame.setMinimum(0)
        self.sld_map_frame.setMaximum(max(0, self._map_n_frames - 1))
        self.sld_map_frame.setValue(max(0, self._map_n_frames - 1))
        self.sld_map_frame.blockSignals(False)
        self._set_map_controls_enabled(True)
        self.chk_map_signed.setToolTip(self._signed_tooltip(
            getattr(plan, "scheme", "central")))
        self.lbl_map_hint.setText(
            "%d parameter(s) \u00d7 %d field(s), %d frame(s). "
            "Signed = %s difference; magnitude = |dF/d\u03b8|."
            % (len(self._map_param_paths), len(self._map_field_vars),
               self._map_n_frames, getattr(plan, "scheme", "central")))
        self._refresh_map()

    def _store_zoi_inputs(self, ref, inst):
        """Keep what the ZOI proposal needs from the base run: EVF (material
        mask), frame times (window T) and the extracted zone (edge check)."""
        try:
            self._map_base_evf = np.asarray(ref.field(inst, "EVF"), float)
        except Exception:
            log_swallowed("reading the base-run EVF for the ZOI proposal",
                          level=logging.DEBUG)
            self._map_base_evf = None
        try:
            self._map_times = np.asarray(ref.times, float)
        except Exception:
            self._map_times = None
        crop = getattr(ref, "roi", None)
        self._map_extent = ((crop["xmin"], crop["xmax"], crop["ymin"],
                             crop["ymax"]) if isinstance(crop, dict) else None)
        self._map_full_domain = crop is None
        has_eps_field = any(v in zp.FIELD_TO_EPS for v in self._field_maps)
        self.btn_zoi_propose.setEnabled(
            has_eps_field and self._map_times is not None)

    def set_model_settings_getter(self, fn):
        """fn() -> {"eps": {Vx, Vy, T, EVF: eps_q}, "window": (a, b)}, read
        from the Model tab when a ZOI is proposed."""
        self._model_settings_getter = fn

    def _model_settings(self):
        if self._model_settings_getter is None:
            raise ValueError("the Model tab settings are not available")
        st = self._model_settings_getter()
        return dict(st.get("eps") or {}), tuple(st["window"])

    def _on_propose_zoi(self):
        if not self._field_maps or self._map_mesh is None:
            return
        try:
            eps, window = self._model_settings()
        except Exception as e:
            self.lbl_zoi.setText("Set \u03b5_q and the window T in the Model "
                                 "tab first (%s)." % e)
            return
        need = sorted({zp.FIELD_TO_EPS[v] for v in self._field_maps
                       if v in zp.FIELD_TO_EPS} - {k for k, v in eps.items()
                                                    if v and v > 0})
        if need:
            self.lbl_zoi.setText("Missing \u03b5_q in the Model tab: %s."
                                 % ", ".join(need))
            return
        plan = self._run_plan or self.plan
        deltas = {sp.path: float(d) for sp, d in zip(plan.specs, plan.deltas)}
        nodes_xy, face_idx = self._map_mesh
        verts = np.asarray(nodes_xy, float)[np.asarray(face_idx)]
        try:
            prop = zp.propose_zoi(self._field_maps, deltas, eps, verts,
                                  self._map_times, window=window,
                                  evf_base=self._map_base_evf,
                                  extent=self._map_extent)
        except Exception as e:
            log_swallowed("proposing a ZOI", level=logging.WARNING)
            self.lbl_zoi.setText("Could not propose a ZOI: %s" % e)
            return
        self._zoi_proposal = prop
        self._show_s_star(prop)
        self.lbl_zoi.setText(self._zoi_summary(prop))
        self.btn_zoi_apply.setEnabled(prop.bbox is not None)
        if self._run_workdir is not None:
            self._export_zoi(Path(self._run_workdir) / mx.MAPS_SUBDIR, prop,
                             eps, window, deltas, verts)

    def _zoi_summary(self, prop) -> str:
        if prop.bbox is None:
            return ("No element with S* \u2265 1: no parameter variation of the "
                    "plan changes a field by more than \u03b5_q.")
        txt = ("ZOI x [%.4g, %.4g] y [%.4g, %.4g] mm, %d element(s) with "
               "S* \u2265 1." % (prop.bbox + (prop.n_selected,)))
        if prop.sides_at_extent:
            zone = ("whole Eulerian domain" if self._map_full_domain
                    else "ROI box")
            txt += (" Reaches the edge of the extracted zone (%s) on %s: "
                    "the extraction, not the sensitivity, bounds it there%s."
                    % (zone, ", ".join(prop.sides_at_extent),
                       "" if self._map_full_domain else
                       "; rerun with \u2018Whole Eulerian domain\u2019"))
        if prop.skipped_fields:
            txt += " Not used (no \u03b5_q): %s." % ", ".join(
                prop.skipped_fields)
        return txt

    def _show_s_star(self, prop):
        vals = np.asarray(prop.s_star, float)
        finite = vals[np.isfinite(vals)]
        vmax = max(1.0, float(finite.max())) if finite.size else 1.0
        self.fv_map.set_values(
            vals, vmin=0.0, vmax=vmax, cmap="inferno",
            title="S* = max mean_T |dq/dp \u00b7 \u03b4| / \u03b5_q "
                  "(ZOI: S* \u2265 1)")

    def _export_zoi(self, out_dir, prop, eps, window, deltas, verts):
        try:
            files = zp.write_proposal(
                out_dir, prop, verts, eps=eps, window=window, deltas=deltas,
                extent_kind=("whole Eulerian domain" if self._map_full_domain
                             else "ROI box"))
            self.log.appendPlainText("[zoi] %d file(s) written to %s"
                                     % (len(files), out_dir))
        except Exception as e:
            log_swallowed("writing the ZOI proposal", level=logging.WARNING)
            self.log.appendPlainText("[zoi] export FAILED: %s" % e)

    def _on_apply_zoi(self):
        if self._zoi_proposal is not None and self._zoi_proposal.bbox:
            self.zoiProposed.emit(tuple(self._zoi_proposal.bbox))

    @staticmethod
    def _signed_tooltip(scheme) -> str:
        formula = {"central": "(F+ - F-)/(2\u03b4)",
                   "forward": "(F+ - F0)/\u03b4",
                   "backward": "(F0 - F-)/\u03b4"}.get(scheme, "dF/d\u03b8")
        return ("Signed: %s difference %s per element, shows direction "
                "(diverging colour). Off: magnitude |dF/d\u03b8| "
                "(sequential)." % (scheme, formula))

    def _set_map_mesh(self, bundle, inst):
        """Push the base-run mesh into fv_map. Mirrors ResultsTab: angle-order
        each element's projected nodes into a 2D footprint."""
        if bundle is None or inst is None:
            return False
        try:
            info = bundle.instance(inst)
            nodes_xy, face_idx = mx.element_faces(
                bundle.nodes_init(info.name), bundle.elements(info.name))
            self.fv_map.set_mesh(nodes_xy, face_idx)
            self._map_mesh_set = True
            self._map_mesh = (nodes_xy, face_idx)
            # Extras for the on-disk export; optional, never fatal here.
            self._map_extra = {}
            try:
                self._map_extra["centroids_xy"] = np.asarray(
                    bundle.element_centroids_init(info.name))[:, :2]
            except Exception:
                log_swallowed("reading element centroids for the map export",
                              level=logging.DEBUG)
            try:
                self._map_extra["frame_times"] = np.asarray(bundle.times)
            except Exception:
                log_swallowed("reading frame times for the map export",
                              level=logging.DEBUG)
            return True
        except Exception:
            log_swallowed("building the sensitivity-map mesh",
                          level=logging.WARNING)
            return False

    def _on_map_aggregate_toggled(self, *_):
        self.sld_map_frame.setEnabled(
            bool(self._field_maps) and not self.chk_map_aggregate.isChecked())
        self._refresh_map()

    def _refresh_map(self, *_):
        """Render the selected (parameter x field) map into fv_map, applying
        the signed/magnitude toggle and the frame/aggregate choice."""
        if not self._field_maps or not self._map_mesh_set:
            return
        var = self.cb_map_field.currentText()
        p = self.cb_map_param.currentData()
        if not var or p is None:
            return
        S = self._field_maps.get(var, {}).get(p)
        if S is None:
            return
        S = np.asarray(S, dtype=float)
        if S.ndim != 2 or S.size == 0:
            return
        signed = self.chk_map_signed.isChecked()
        aggregate = self.chk_map_aggregate.isChecked()
        if aggregate:
            with np.errstate(invalid="ignore"):
                if signed:
                    values = np.nanmean(S, axis=0)             # keeps sign
                    frame_txt = "mean over %d frames" % S.shape[0]
                else:
                    values = np.sqrt(np.nanmean(S * S, axis=0))   # >= 0
                    frame_txt = "RMS over %d frames" % S.shape[0]
            self.lbl_map_frame.setText("agg")
        else:
            f = int(np.clip(self.sld_map_frame.value(), 0, S.shape[0] - 1))
            row = S[f]
            values = row if signed else np.abs(row)
            frame_txt = "frame %d/%d" % (f, S.shape[0] - 1)
            self.lbl_map_frame.setText("%d" % f)
        finite = values[np.isfinite(values)]
        if signed:
            m = float(np.max(np.abs(finite))) if finite.size else 1.0
            vmin, vmax, cmap = -m, m, "RdBu_r"
        else:
            vmax = float(np.max(finite)) if finite.size else 1.0
            vmin, cmap = 0.0, "inferno"
        try:
            plabel = pr.spec_for(p).label
        except Exception:
            plabel = p
        used = self._map_schemes.get(var, {}).get(p)
        scheme = self._result_scheme()
        fb = (" \u2014 %s fallback (a run failed)" % used
              if scheme == "central" and used in ("forward", "backward")
              else "")
        title = "dF/d(%s) \u00b7 %s \u2014 %s (%s)%s" % (
            plabel, var, "signed" if signed else "|.|", frame_txt, fb)
        self.fv_map.set_values(values, vmin=vmin, vmax=vmax, cmap=cmap,
                               title=title)

    # ---- on-disk export of the maps (study folder) -----------------------
    def _map_export_job(self, out_dir):
        """Snapshot everything the export needs, on the GUI thread, and
        return a no-argument callable that writes the files (safe to run in
        another thread: it only touches these arrays and the disk)."""
        plan = self._run_plan or self.plan
        nodes_xy, face_idx = self._map_mesh
        param_info = {}
        for p in self._map_param_paths:
            try:
                spec = pr.spec_for(p)
                param_info[p] = (spec.label, self._plan_unit(plan, spec))
            except Exception:
                param_info[p] = (p, "—")
        deltas = {s.path: float(d) for s, d in zip(plan.specs, plan.deltas)}
        maps = {v: dict(per) for v, per in self._field_maps.items()}
        # Per-map scheme actually used (a central map may have fallen back).
        base_scheme = getattr(plan, "scheme", "")
        schemes = {v: {p: self._map_schemes.get(v, {}).get(p, base_scheme)
                       for p in per} for v, per in maps.items()}
        extra = dict(self._map_extra)

        def job():
            return mx.write_maps(out_dir, maps, nodes_xy, face_idx,
                                 param_info=param_info,
                                 centroids_xy=extra.get("centroids_xy"),
                                 frame_times=extra.get("frame_times"),
                                 scheme=schemes,
                                 deltas=deltas)
        return job

    def _export_field_maps(self, out_dir, wait=False):
        """Write the maps (.npz arrays + PNG images) to `out_dir`, a sub-folder
        of the study folder. Runs in a background thread so rendering a few
        dozen images does not freeze the window; `wait=True` (tests) runs it
        inline. The outcome reaches the status line via _mapsExported."""
        if (not self._field_maps or self._map_mesh is None
                or (self._run_plan or self.plan) is None):
            return
        try:
            job = self._map_export_job(out_dir)
        except Exception as e:
            log_swallowed("preparing the map export", level=logging.WARNING)
            self._on_maps_exported(str(out_dir), 0, str(e))
            return

        def run():
            try:
                files = job()
                self._mapsExported.emit(str(out_dir), len(files), "")
            except Exception as e:
                log_swallowed("writing the sensitivity maps",
                              level=logging.WARNING)
                self._mapsExported.emit(str(out_dir), 0, str(e) or repr(e))

        self.log.appendPlainText("[maps] writing to %s …" % out_dir)
        if wait:
            run()
            return
        # Not a daemon: a window closed mid-export still finishes the files.
        self._maps_thread = threading.Thread(target=run, name="map-export")
        self._maps_thread.start()

    def _on_maps_exported(self, out_dir, n_files, error):
        if error:
            self.log.appendPlainText("[maps] export FAILED: %s" % error)
            self.status.setStyleSheet("color: #b45309;")
            self.status.setText(self.status.text()
                                + "  Maps export failed: %s" % error)
            return
        self.log.appendPlainText("[maps] %d files written to %s"
                                 % (n_files, out_dir))
        self.status.setText(self.status.text()
                            + "  Maps written to %s." % out_dir)

    # ---- window closing ---------------------------------------------------
    def is_running(self) -> bool:
        """True while a campaign is in progress."""
        return self._thread is not None

    def shutdown(self, timeout_ms: int = 60000) -> bool:
        """Stop a running campaign synchronously -- the window is closing.

        Terminates the Abaqus job in flight (``abaqus terminate``, then the
        process tree), stops the campaign loop and waits for the worker
        thread. The worker's result is dropped: its signals are disconnected
        first so nothing lands on a closing window. Returns True if the
        thread ended within `timeout_ms`."""
        worker, thread = self._worker, self._thread
        if worker is None or thread is None:
            return True
        with warnings.catch_warnings():
            # PySide warns (instead of raising) on a signal with no slot.
            warnings.simplefilter("ignore", RuntimeWarning)
            for sig in (worker.progress, worker.log, worker.runDone,
                        worker.finished, worker.failed):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass          # nothing connected
        if self._sta_timer is not None:
            self._sta_timer.stop()
            self._sta_timer = None
        worker.stop_blocking()
        thread.quit()
        ended = bool(thread.wait(int(timeout_ms)))
        if not ended:
            logging.getLogger(__name__).warning(
                "sensitivity worker thread still running after %d ms",
                timeout_ms)
        self._thread = None
        self._worker = None
        return ended

    def _on_export(self):
        if self._last_result is None:
            self._warn("Nothing to export yet — run a plan first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sensitivity results",
            "sensitivity_results.csv", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            label_for = lambda p: pr.spec_for(p).label
            xr.write_csv(self._last_result, path, label_for=label_for)
        except Exception as e:
            self._warn("Export failed: %s" % e)
            return
        self.status.setStyleSheet("color: #15803d;")
        self.status.setText("Results exported to %s" % path)

    def _on_run_failed(self, msg):
        self._teardown_thread()
        self._warn("Run failed: %s" % msg)

    def _teardown_thread(self):
        if getattr(self, "_sta_timer", None) is not None:
            self._sta_timer.stop()
            self._sta_timer = None
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_gen.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _show_results(self, result):
        """Fill the results table: one row per parameter, columns per QoI.
        Morris shows mu* (sigma); Jacobian shows the sensitivity value."""
        paths = result.param_paths
        qoi_ids = result.qoi_ids
        labels = {p: pr.spec_for(p).label for p in paths}
        tbl = self.results_table
        tbl.clear()
        tbl.setRowCount(len(paths))
        tbl.setColumnCount(1 + len(qoi_ids))
        header = ["Parameter"] + list(qoi_ids)
        tbl.setHorizontalHeaderLabels(header)
        for i, p in enumerate(paths):
            tbl.setItem(i, 0, QTableWidgetItem(labels.get(p, p)))
            for j, qid in enumerate(qoi_ids):
                a = result.analyses.get(qid, {})
                cell = self._result_cell(result.plan_kind, a, p,
                                         scheme=self._result_scheme())
                tbl.setItem(i, j + 1, QTableWidgetItem(cell))
        tbl.resizeColumnsToContents()
        # populate the chart QoI selector and draw
        self.cb_chart_qoi.blockSignals(True)
        self.cb_chart_qoi.clear()
        self.cb_chart_qoi.addItems(list(result.qoi_ids))
        self.cb_chart_qoi.blockSignals(False)
        self._draw_chart()

    def _draw_chart(self):
        self._fig.clear()
        result = self._last_result
        if result is None or not result.qoi_ids:
            self._canvas.draw_idle()
            return
        qid = self.cb_chart_qoi.currentText() or result.qoi_ids[0]
        jacobian = result.plan_kind == "jacobian"
        self.lbl_chart_rank.setVisible(jacobian)
        self.cb_chart_rank.setVisible(jacobian)
        self.lbl_chart_note.setText("")
        if jacobian:
            key = self.cb_chart_rank.currentData() or "sensitivity"
            rows = rc.jacobian_ranking(result, qid, key=key)
            vals = [abs(s) for _, s in rows]
            xlabel = "|%s|" % key
            if key == "sensitivity":
                self.lbl_chart_note.setText(self._raw_ranking_note(result, qid))
            else:
                self.lbl_chart_note.setText(self._elasticity_note(result, qid))
        else:
            rows = [(p, ms) for p, ms, _ in rc.morris_ranking(result, qid)]
            vals = [ms for _, ms in rows]
            xlabel = "mu*"
        labels = [pr.spec_for(p).label for p, _ in rows]
        ax = self._fig.add_subplot(111)
        y = range(len(rows))
        ax.barh(list(y), vals, color="#2563eb")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()                                  # most influential on top
        ax.set_xlabel("%s — %s" % (xlabel, qid), fontsize=9)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _result_scheme(self):
        plan = self._run_plan or self.plan
        return getattr(plan, "scheme", None)

    def _elasticity_note(self, result, qid) -> str:
        """Why some bars are missing from an elasticity ranking."""
        a = result.analyses.get(qid, {})
        rows = {p: d for p, d in a.items() if isinstance(d, dict)}
        if rows and not any("elasticity" in d for d in rows.values()):
            return ("Field QoI have no elasticity: compare the \u0394% (rel) "
                    "columns, with the same Delta% on every row.")
        missing = [p for p, d in rows.items()
                   if not np.isfinite(d.get("elasticity", np.nan))]
        if not missing:
            return ""
        if len(missing) == len(rows):
            return ("No elasticity for a temperature QoI (°C has no "
                    "physical zero): rank by sensitivity instead.")
        labels = []
        for p in missing:
            try:
                labels.append(pr.spec_for(p).label)
            except Exception:
                labels.append(p)
        return ("Not ranked (no elasticity for a temperature, or no "
                "value): %s." % ", ".join(labels))

    def _raw_ranking_note(self, result, qid) -> str:
        """Warning shown above a raw-sensitivity ranking whose bars are not
        comparable: parameters in different units, or a mix of raw and
        normalised rows. Empty when the ranking is homogeneous."""
        a = result.analyses.get(qid, {})
        rows = {p: d for p, d in a.items()
                if isinstance(d, dict) and "sensitivity" in d}
        if len(rows) < 2:
            return ""
        norm = {bool(d.get("normalized")) for d in rows.values()}
        if len(norm) > 1:
            return ("Mixed ranking: some rows are elasticities (Norm ticked), "
                    "others raw derivatives. Rank by elasticity to compare "
                    "them.")
        plan = self._run_plan or self.plan
        if norm == {True} or plan is None:
            return ""
        unit_of = {s.path: self._plan_unit(plan, s)
                   for s in getattr(plan, "specs", [])}
        if len({unit_of.get(p, "?") for p in rows}) > 1:
            return ("Parameters have different units: this raw ranking "
                    "depends on the units chosen. Rank by elasticity to "
                    "compare them.")
        return ""

    @staticmethod
    def _result_cell(plan_kind, analysis, path, scheme=None):
        if "error" in analysis:
            return "err"
        if plan_kind == "jacobian":
            d = analysis.get(path)
            if not d:
                return "—"
            return _fmt(d["sensitivity"]) + _cell_marks(d, scheme)
        # morris: analysis has names / mu_star / sigma arrays
        names = list(analysis.get("names", []))
        if path not in names:
            return "—"
        k = names.index(path)
        mu_star = analysis.get("mu_star", [])
        sigma = analysis.get("sigma", [])
        return "%s (%s)" % (_fmt(mu_star[k]), _fmt(sigma[k]))

    def _show_preview(self, plan):
        rows = (jac.profile_table(plan) if self.plan_kind == "jacobian"
                else mp.profile_table(plan))
        paths = plan.param_paths
        has_kind = "kind" in rows[0] if rows else False
        head_cells = ["run"]
        if has_kind:
            head_cells.append("kind")
        head_cells += ["%s [%s]" % (s.label, self._plan_unit(plan, s))
                       for s in plan.specs]
        header = " | ".join(head_cells)
        lines = [header, "-" * len(header)]
        for d in rows:
            cells = ["%4d" % d["run"]]
            if has_kind:
                cells.append("%-5s" % d["kind"])
            cells += [_fmt(d[p]) for p in paths]
            lines.append(" | ".join(cells))
        self.preview.setPlainText("\n".join(lines))

    @staticmethod
    def _plan_unit(plan, spec) -> str:
        """Unit of the plan's values for `spec` (the plan's own snapshot)."""
        return spec.unit_str(getattr(plan, "temp_unit", "C"),
                             system=getattr(plan, "unit_system", None))

    def _warn(self, msg):
        self.status.setStyleSheet("color: #b91c1c;")
        self.status.setText(msg)


def _fmt(x: float) -> str:
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return "%.4g" % x
    return "%.4f" % x


def _hms_to_sec(hms):
    """Parse an Abaqus .sta wall-clock 'HH:MM:SS' string to seconds."""
    if not hms:
        return None
    try:
        parts = [int(p) for p in str(hms).split(":")]
        s = 0
        for p in parts:
            s = s * 60 + p
        return float(s)
    except Exception:
        log_swallowed("parsing .sta wall-clock %r" % hms, level=logging.DEBUG)
        return None


def _fmt_duration(seconds: float) -> str:
    """Human-friendly wall-clock estimate."""
    s = int(round(seconds))
    if s < 90:
        return "%d s" % s
    m = s / 60.0
    if m < 90:
        return "%.0f min" % m
    h = m / 60.0
    if h < 48:
        return "%.1f h" % h
    return "%.1f days" % (h / 24.0)


def _cell_marks(d, scheme=None) -> str:
    """Suffixes flagging how a Jacobian value was obtained: 'raw' when Norm
    was ticked but no elasticity exists (temperature), 'fwd'/'bwd' when a
    central difference fell back to one side because a run failed."""
    marks = []
    if d.get("raw_fallback"):
        marks.append("raw")
    used = d.get("scheme_used")
    if scheme == "central" and used in ("forward", "backward"):
        marks.append("fwd" if used == "forward" else "bwd")
    return (" (%s)" % ", ".join(marks)) if marks else ""
