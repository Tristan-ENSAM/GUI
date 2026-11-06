# -*- coding: utf-8 -*-
"""
Explore and Sensitivity tabs (attributes `explore_tab` and `sensitivity_tab` of the main window) —
sweep (one parameter at a time), grid (all combinations), local Jacobian
(finite differences) or Morris screening.

Lets the user:
  * tick which model parameters to vary (Ref = current model value, in
    displayed units), set the FD step (Jacobian), Min/Max (Morris), or the
    levels (sweep / grid: n values from Min to Max, or typed in Values),
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
    QListWidget, QListWidgetItem, QLineEdit,
)
from PySide6.QtCore import QThread, QTimer

from gui.sensitivity import param_registry as pr
from gui.widgets.param_filter import fill_filter_combo, keep as filter_keep
from gui.sensitivity import morris_plan as mp
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import sweep_plan as sw
from gui.sensitivity import runner_core as rc
from gui.sensitivity import export_results as xr
from gui.sensitivity import map_export as mx
from gui.sensitivity import zoi_proposal as zp
from gui.sensitivity import model_qoi as mq
from gui.sensitivity.run_worker import SensitivityRunWorker, script_log_path
from gui.core.remote_exec import is_remote, launch_problems
from gui.widgets.run_progress import RunProgressBars
from gui.widgets.field_viewer import FieldViewer
from gui.widgets.machine_combo import MachineCombo
from gui.widgets.workdir_button import workdir_button
from gui.widgets.theme import mark_primary
from gui.core.sta_parser import parse_sta
from gui.results import qoi as qoi_mod
from gui.core.logging_util import log_swallowed
import logging


# Every registered parameter is offered (decision 2026-11-06): geometry,
# mesh and outputs too. Those that rebuild the mesh or only change the
# outputs carry a warning in the Note column, never a block.
_NOTE_TEXT = {"output": "output only", "numerical": "numerical",
              **pr.MESH_NOTE_LABEL}
_NOTE_TIP = {
    "output": "Changes only what is written or extracted (ROI box, frames, "
              "filters), not the modelled physics.",
    "numerical": "A setting of the model (domain, mesh, mass scaling, "
                 "duration), not a physical property.",
    **pr.MESH_NOTE_TIP,
}
_MESH_FIELD_TIP = ("\nPer-element field sensitivities and maps are computed "
                   "only when the runs have the same Eulerian mesh (checked "
                   "on their element centroids); the scalar QoI are always "
                   "computed.")
# Methods of the tab, in the order of the method combo.
ALL_METHODS = ("jacobian", "morris", "sweep", "grid")
EXPLORE_METHODS = ("sweep", "grid")
SENSITIVITY_METHODS = ("jacobian", "morris")
_METHOD_LABELS = {
    "jacobian": "Jacobian (finite differences)",
    "morris": "Morris (global screening)",
    "sweep": "Sweep (one parameter at a time)",
    "grid": "Grid (all combinations)",
}


def _note_of(spec, cfg=None) -> str:
    """output | a mesh note (pr.mesh_note) | numerical | "" (the Note column
    of a row), for the model `cfg`."""
    if spec.kind == "output":
        return "output"
    mesh = pr.mesh_note(spec, cfg)
    if mesh:
        return mesh
    if spec.kind == "numerical":
        return "numerical"
    return ""


class SensitivityTab(QWidget):
    # Emitted from the map-export thread: (output folder, files written,
    # error message or "").
    _mapsExported = Signal(str, int, str)
    # ZOI proposed from the maps, (xmin, xmax, ymin, ymax) [mm]: the main
    # window copies it into the Model tab.
    zoiProposed = Signal(tuple)
    # Two runs of a sweep / grid plan to compare: (plan folder, job of the
    # reference run a, job of the compared run b).
    compareRunsRequested = Signal(str, str, str)

    def __init__(self, cfg, prefs_getter=None, cpus_getter=None,
                 profile_name_getter=None, methods=ALL_METHODS):
        super().__init__()
        self.cfg = cfg
        # Explore holds sweep / grid, Sensitivity Jacobian / Morris; the
        # same class serves both so the plans and runs stay identical.
        self._methods = tuple(m for m in ALL_METHODS if m in methods) \
            or ALL_METHODS
        self._prefs_getter = prefs_getter
        self._profile_name_getter = profile_name_getter
        self._cpus_getter = cpus_getter
        self.plan = None                 # last generated JacobianPlan
        self.plan_kind = "jacobian"
        self.selected_qois = []          # list[QoISpec]
        self.plan_field_vars = []        # ROI fields chosen with the plan
        # Model-tab settings (ZOI, window T, EVF threshold, eps_q) frozen
        # with the plan when a Model-tab QoI is ticked, else None.
        self._plan_model_qoi = None
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
        self._bars_job = None            # job shown on the run_bars
        self._bars_remote = False
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
        # Callable -> study folders of Model sizing (runs that a sweep / grid
        # plan may reuse); set by the main window.
        self._known_folders_getter = None
        self._run_job_prefix = "sensitivity"   # job names of the active run
        self._series_zone_used = None    # ZOI of the time curves of the run
        self._mapsExported.connect(self._on_maps_exported)

        root = QVBoxLayout(self)

        help_lines = [
            "Tick the parameters to vary, pick the QoI, then generate and "
            "run the plan. The Ref column is the current model value. Every "
            "parameter of the model is listed; the Note column flags those "
            "that rebuild the mesh or only change the outputs."]
        if "sweep" in self._methods:
            help_lines.append(
                "Sweep: each parameter takes its values in turn, the others "
                "stay at Ref; the Ref run is shared (cost 1 + the values "
                "different from Ref). Shows QoI = f(parameter), blind to "
                "interactions.")
        if "grid" in self._methods:
            help_lines.append(
                "Grid: every combination of the values (cost = product of "
                "the numbers of values, + 1 when Ref is not one of them). "
                "Shows interactions; meant for 2 or 3 parameters.")
        if "sweep" in self._methods or "grid" in self._methods:
            help_lines.append(
                "Values of a row, from Min to Max (both always included): "
                "n values (Levels, Spacing above, or the row's Spacing), a "
                "step (Min, Min + step, ... then Max) or a ratio (Min, "
                "Min x ratio, ... then Max), or the Values cell typed by "
                "hand (e.g. 1; 100; 1000). The Levels column shows them.")
        if "jacobian" in self._methods:
            help_lines.append(
                "Jacobian: finite differences around Ref with the step "
                "Delta, Min/Max a trust region; cost k+1 (forward/backward) "
                "or 2k+1 (central).")
        if "morris" in self._methods:
            help_lines.append("Morris: global screening over Min..Max.")
        intro = QLabel("\n".join(help_lines))
        intro.setWordWrap(True)
        # Grey help, hidden until "Show help" (as in the Model sizing tab).
        intro.setStyleSheet("color:#6b7280;")
        intro.setVisible(False)
        self.lbl_intro = intro
        top = QHBoxLayout()
        top.addStretch(1)
        self.cb_help = QCheckBox("Show help")
        self.cb_help.setToolTip("Show the grey explanation of this tab.")
        self.cb_help.toggled.connect(intro.setVisible)
        top.addWidget(self.cb_help)
        root.addLayout(top)
        root.addWidget(intro)

        split = QSplitter(Qt.Vertical)
        root.addWidget(split, 1)

        # ---- Parameters table ------------------------------------------
        param_box = QGroupBox("Parameters to vary")
        pv = QVBoxLayout(param_box)
        # Filter: every parameter, the physical ones, or the ticked ones,
        # and a search on the name.
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Show:"))
        self.cb_filter = QComboBox()
        fill_filter_combo(self.cb_filter)
        self.cb_filter.currentIndexChanged.connect(self._apply_filter)
        frow.addWidget(self.cb_filter)
        self.le_search = QLineEdit()
        self.le_search.setPlaceholderText("Search a parameter…")
        self.le_search.setClearButtonEnabled(True)
        self.le_search.textChanged.connect(self._apply_filter)
        frow.addWidget(self.le_search, 1)
        pv.addLayout(frow)
        self.table = QTableWidget(0, 14)
        self.table.setHorizontalHeaderLabels(
            ["Vary", "Parameter", "Ref", "Min", "Max", "Delta", "Delta%",
             "Norm", "Unit", "Values", "Note", "Spacing", "n / step / ratio",
             "Levels"])
        self.table.horizontalHeaderItem(11).setToolTip(
            "How the values of the row are made from Min to Max. default: "
            "Levels and Spacing above. linear, step: Min, Min + step, ... "
            "then Max. geometric, ratio: Min, Min x ratio, ... then Max. Min "
            "and Max are always in the plan; when the last step overshoots "
            "Max, the last interval is shorter. A Values cell typed by hand "
            "wins.")
        self.table.horizontalHeaderItem(12).setToolTip(
            "The n, step or ratio of the row's Spacing (step in the unit of "
            "the row).")
        self.table.horizontalHeaderItem(13).setToolTip(
            "The values the plan will run for the ticked rows.")
        self.table.horizontalHeaderItem(10).setToolTip(
            "What a new value changes, for the current model. remesh: "
            "another Eulerian element size. may remesh: an Eulerian side "
            "with Discretize off. \u00b1 elements: an Eulerian side with "
            "Discretize on (same element size). grid shift: the Eulerian "
            "grid moves. tool mesh: only the tool is remeshed. output only: "
            "changes what is written or extracted, not the physics. "
            "numerical: a setting of the model. A warning, not a block.")
        self.table.horizontalHeaderItem(9).setToolTip(
            "Sweep / grid: values typed by hand, separated by ';' or spaces "
            "(use a dot for decimals). Empty: Levels values from Min to Max.")
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12):
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
        for m in self._methods:
            self.cb_method.addItem(_METHOD_LABELS[m], m)
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

        # Sweep / grid controls: n levels from Min to Max, linear or
        # geometric (a "Values" cell typed by hand wins for its row).
        self.lbl_nlev = QLabel("Levels:")
        method_row.addWidget(self.lbl_nlev)
        self.spin_nlev = QSpinBox()
        self.spin_nlev.setRange(2, 50)
        self.spin_nlev.setValue(sw.DEFAULT_LEVELS)
        self.spin_nlev.setToolTip(
            "Number of values of each parameter, from Min to Max (both\n"
            "included). A row whose Values cell is filled uses those values.")
        self.spin_nlev.valueChanged.connect(self._update_cost)
        self.spin_nlev.valueChanged.connect(self._mark_plan_stale)
        method_row.addWidget(self.spin_nlev)
        self.lbl_spacing = QLabel("Spacing:")
        method_row.addWidget(self.lbl_spacing)
        self.cb_spacing = QComboBox()
        self.cb_spacing.addItem("linear", "linear")
        self.cb_spacing.addItem("geometric", "geometric")
        self.cb_spacing.setToolTip(
            "linear: evenly spaced values. geometric: constant ratio between\n"
            "two values (Min and Max > 0), e.g. a mass-scaling factor.")
        self.cb_spacing.currentIndexChanged.connect(self._update_cost)
        self.cb_spacing.currentIndexChanged.connect(self._mark_plan_stale)
        method_row.addWidget(self.cb_spacing)

        self.lbl_hint = QLabel("step = Delta per parameter")
        self.lbl_hint.setStyleSheet("color: #6b7280;")
        method_row.addWidget(self.lbl_hint)
        method_row.addStretch(1)
        bl.addLayout(method_row)

        # QoI of the Model tab (same reduction, same eps_q) + other QoI.
        qoi_row = QHBoxLayout()
        model_box = QGroupBox("QoI of the sizing studies (judged against \u03b5_q)")
        mg = QGridLayout(model_box)
        self._model_qoi_checks = {}
        for i, q in enumerate(mq.MODEL_QOIS):
            cb = QCheckBox("%s  [%s]" % (mq.LABELS[q], mq.UNITS[q]))
            cb.setChecked(False)
            cb.setEnabled(False)          # until the Model tab is connected
            cb.setToolTip(
                "Computed exactly as the ms / mesh / domain studies of "
                "Model sizing, with the ZOI, window T and EVF threshold of "
                "the Setup tab. Fields: area mean over the ZOI of the time mean of each "
                "element over its material frames in T. Forces: mean of F/w "
                "over T. A \"/\u03b5_q\" column is added to the results.")
            cb.toggled.connect(self._mark_plan_stale)
            self._model_qoi_checks[q] = cb
            mg.addWidget(cb, i // 2, i % 2)
        self.lbl_model_qoi = QLabel(
            "Settings read from the Setup tab when the plan is generated.")
        self.lbl_model_qoi.setWordWrap(True)
        self.lbl_model_qoi.setStyleSheet("color: #6b7280;")
        mg.addWidget(self.lbl_model_qoi, (len(mq.MODEL_QOIS) + 1) // 2, 0,
                     1, 2)
        qoi_row.addWidget(model_box, 1)

        # The "Other QoI" (Fx/Fy mean and max, T_max, PEEQ_max) and their
        # warm-up fraction were removed (QoI sort of 2026-11-06): the plans
        # use the QoI of the sizing studies, judged against eps_q. Old plan
        # folders that hold them are still read.
        self._qoi_checks = {}
        bl.addLayout(qoi_row)

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
        # Same pair as in the Model tab: where the run folders are, and the
        # machine the runs go to.
        self.btn_open_wd = workdir_button(self, self._prefs_getter)
        ctrl.addWidget(self.btn_open_wd)
        ctrl.addWidget(QLabel("Machine:"))
        self.cmb_machine = MachineCombo(self._prefs_getter)
        ctrl.addWidget(self.cmb_machine)
        self.chk_reuse = QCheckBox("Reuse finished runs")
        self.chk_reuse.setChecked(True)
        self.chk_reuse.setToolTip(
            "Sweep / grid: a run whose model parameters are EXACTLY those of\n"
            "a finished run (earlier sweep / grid plans in the working\n"
            "directory, studies of Model sizing) is not launched again: its\n"
            "results are read back and copied into the new plan folder.\n"
            "The number of CPUs, the machine and the Abaqus version are not\n"
            "compared.")
        ctrl.addWidget(self.chk_reuse)
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
        mark_primary(self.btn_run)
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
        self.btn_compare_runs = QPushButton("Compare 2 runs…")
        self.btn_compare_runs.setToolTip(
            "Sweep / grid: select two rows of the Table, then open them in\n"
            "Compare runs (where and when they differ, in \u03b5_q). The Ref\n"
            "run, or else the lower run number, is the reference (a).")
        self.btn_compare_runs.setEnabled(False)
        self.btn_compare_runs.clicked.connect(self._on_compare_runs)
        ctrl.addWidget(self.btn_compare_runs)
        bl.addLayout(ctrl)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        bl.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        bl.addWidget(self.progress)
        # Stages of the run in progress (solver, extraction, transfer).
        self.run_bars = RunProgressBars(horizontal=True)
        self.run_bars.setVisible(False)
        bl.addWidget(self.run_bars)

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
        self.tabs_out.addTab(self.log, "Log")

        self.results_table = QTableWidget(0, 0)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.itemSelectionChanged.connect(
            self._sync_compare_button)
        self.tabs_out.addTab(self.results_table, "Table")

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
        # Sweep / grid: QoI against one parameter.
        self.lbl_chart_param = QLabel("Parameter:")
        crow.addWidget(self.lbl_chart_param)
        self.cb_chart_param = QComboBox()
        self.cb_chart_param.currentIndexChanged.connect(self._draw_chart)
        crow.addWidget(self.cb_chart_param)
        self.lbl_chart_param.setVisible(False)
        self.cb_chart_param.setVisible(False)
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

        # ---- Time curves (sweep / grid): one QoI over time, per run -----
        tc_w = QWidget(); tcl = QHBoxLayout(tc_w)
        tc_left = QVBoxLayout()
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Quantity:"))
        self.cb_series_qty = QComboBox()
        self.cb_series_qty.currentIndexChanged.connect(self._draw_series)
        trow.addWidget(self.cb_series_qty)
        trow.addStretch(1)
        tc_left.addLayout(trow)
        self.lbl_series_note = QLabel(
            "Run a sweep or grid plan to compare the runs over time.")
        self.lbl_series_note.setWordWrap(True)
        self.lbl_series_note.setStyleSheet("color: #6b7280;")
        tc_left.addWidget(self.lbl_series_note)
        self._fig_series = Figure(figsize=(5, 3))
        self._canvas_series = FigureCanvas(self._fig_series)
        tc_left.addWidget(self._canvas_series, 1)
        tcl.addLayout(tc_left, 1)
        self.lst_series_runs = QListWidget()
        self.lst_series_runs.setToolTip("Runs drawn (tick / untick).")
        self.lst_series_runs.setMaximumWidth(260)
        self.lst_series_runs.itemChanged.connect(self._draw_series)
        tcl.addWidget(self.lst_series_runs)
        self.tabs_out.addTab(tc_w, "Time curves")
        self._series_tab = tc_w

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
            "of the Setup tab; Vx, Vy, T masked by EVF \u2265 0.5 of the base "
            "run. ZOI = smallest rectangle containing every element with "
            "S* \u2265 1.")
        self.btn_zoi_propose.setEnabled(False)
        self.btn_zoi_propose.clicked.connect(self._on_propose_zoi)
        zrow.addWidget(self.btn_zoi_propose)
        self.btn_zoi_apply = QPushButton("Copy ZOI to the Setup tab")
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
        self._on_method_changed()   # visibility of the method controls

    # ------------------------------------------------------------------
    # Build the parameter table from the registry
    # ------------------------------------------------------------------
    def _temp_unit(self) -> str:
        return getattr(self.cfg.ui, "temp_unit", "C")

    def _set_note(self, item, spec):
        """Note column of a row: what a new value changes, for the current
        model (the mesh notes follow the Discretize option)."""
        from PySide6.QtGui import QColor
        note = _note_of(spec, self.cfg)
        item.setText(_NOTE_TEXT.get(note, ""))
        tip = _NOTE_TIP.get(note, "")
        if note in pr.MESH_NOTE_LABEL and note != "tool mesh":
            tip += _MESH_FIELD_TIP
        item.setToolTip(tip)
        item.setForeground(QColor("#b45309" if note in pr.MESH_NOTE_LABEL
                                  else "#6b7280"))

    def _populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        tu = self._temp_unit()
        self._table_units = pr.current_system(tu)
        self._header_rows = {}
        for category, specs in pr.registry_by_category().items():
            # category header row
            r = self.table.rowCount()
            self.table.insertRow(r)
            head = QTableWidgetItem(category)
            fnt = head.font(); fnt.setBold(True); head.setFont(fnt)
            head.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(r, 0, head)
            self.table.setSpan(r, 0, 1, self.table.columnCount())
            head_row = r
            self._header_rows[head_row] = []

            for spec in specs:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self._row_spec[r] = spec
                self._header_rows[head_row].append(r)
                lo, hi = pr.default_display_bounds(self.cfg, spec, tu)
                ref = (pr.get_display(self.cfg, spec, tu)
                       if pr.is_set(self.cfg, spec) else 0.0)
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
                # col 9: values typed by hand (sweep / grid), empty
                self.table.setItem(r, 9, QTableWidgetItem(""))
                # col 10: what a new value changes (warning only)
                it_note = QTableWidgetItem("")
                it_note.setFlags(Qt.ItemIsEnabled)
                self._set_note(it_note, spec)
                self.table.setItem(r, 10, it_note)
                # cols 11-13: per-row spacing, its n / step / ratio, and the
                # levels the plan will run (sweep / grid)
                cb = QComboBox()
                cb.addItem("default", "default")
                for key, label, _what in sw.ROW_SPACINGS:
                    cb.addItem(label, key)
                cb.currentIndexChanged.connect(
                    lambda _i, row=r: self._on_row_spacing_changed(row))
                self.table.setCellWidget(r, 11, cb)
                self.table.setItem(r, 12, QTableWidgetItem(""))
                it_lv = QTableWidgetItem("")
                it_lv.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(r, 13, it_lv)
                self._set_row_available(r, spec)
        self.table.blockSignals(False)
        self._apply_filter()

    def _row_spacing(self, row) -> str:
        cb = self.table.cellWidget(row, 11)
        return (cb.currentData() if cb is not None else None) or "default"

    def _on_row_spacing_changed(self, row):
        """Pre-fill the n / step / ratio cell when it does not fit the new
        spacing: n = Levels, step = (Max - Min) / 4, ratio = 2."""
        how = self._row_spacing(row)
        cur = self._cell_float(row, 12)
        lo, hi = self._cell_float(row, 3), self._cell_float(row, 4)
        text = None
        if how in ("linear_n", "geometric_n"):
            if cur is None or not float(cur).is_integer() or cur < 2:
                text = str(self.spin_nlev.value())
        elif how == "linear_step":
            if (cur is None or cur <= 0) and lo is not None and hi is not None \
                    and hi > lo:
                text = "%.6g" % ((hi - lo) / 4.0)
        elif how == "geometric_ratio":
            if cur is None or cur <= 1:
                text = "2"
        if text is not None:
            self._set_cell(row, 12, text)   # itemChanged updates the cost
        self._mark_plan_stale()
        self._update_cost()

    def _row_levels(self, row, spec):
        """(levels, how they were made) of a row, displayed units.
        ValueError when Min/Max/Values/step are not usable."""
        it = self.table.item(row, 9)
        lv = sw.parse_values(it.text() if it is not None else "")
        if lv is not None:
            return lv, "typed"
        lo, hi = self._cell_float(row, 3), self._cell_float(row, 4)
        if lo is None or hi is None:
            raise ValueError("Min and Max must be numbers.")
        how = self._row_spacing(row)
        if how == "default":
            g = self.cb_spacing.currentData() or "linear"
            return sw.levels(lo, hi, self.spin_nlev.value(), g), g
        val = self._cell_float(row, 12)
        if val is None:
            what = dict((k, w) for k, _l, w in sw.ROW_SPACINGS)[how]
            raise ValueError("give the %s of the row." % what)
        return sw.row_levels(lo, hi, how, val), sw.spacing_record(how, val)

    def _refresh_level_cells(self):
        """Show in the Levels column the values each ticked row will run
        (or why it cannot), empty for the others."""
        if self._method() not in sw.DESIGNS:
            return
        ticked = {r for r, _s in self._selected_rows()}
        self.table.blockSignals(True)
        try:
            for r, spec in self._row_spec.items():
                it = self.table.item(r, 13)
                if it is None:
                    continue
                if r not in ticked:
                    it.setText(""); it.setToolTip("")
                    continue
                try:
                    lv, _how = self._row_levels(r, spec)
                    txt = "; ".join("%.6g" % float(v) for v in lv)
                    it.setText(txt)
                    it.setToolTip("%d values: %s" % (len(lv), txt))
                except ValueError as e:
                    it.setText("?")
                    it.setToolTip(str(e))
        finally:
            self.table.blockSignals(False)

    def _set_row_available(self, row, spec):
        """A value the model does not hold yet (e.g. the contact
        conductance before it is set) cannot be varied: its row is shown
        greyed out with "not set" as Ref."""
        ok = pr.is_set(self.cfg, spec)
        chk = self.table.item(row, 0)
        if chk is None:
            return
        if ok:
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setToolTip("")
        else:
            chk.setCheckState(Qt.Unchecked)
            chk.setFlags(Qt.NoItemFlags)
            chk.setToolTip("Not set in the model: give it a value in the "
                           "Numerical Model tabs to vary it.")
            ref = self.table.item(row, 2)
            if ref is not None:
                ref.setText("not set")

    def _apply_filter(self, *_):
        """Show the rows the filter and the search keep; a category header
        is shown when one of its rows is."""
        if not hasattr(self, "_header_rows"):
            return
        mode = self.cb_filter.currentData() or "all"
        words = self.le_search.text().strip().lower()
        for head, rows in self._header_rows.items():
            any_shown = False
            for r in rows:
                spec = self._row_spec[r]
                it = self.table.item(r, 0)
                keep = filter_keep(spec, mode, it is not None
                                   and it.checkState() == Qt.Checked)
                if keep and words:
                    keep = (words in spec.label.lower()
                            or words in spec.path.lower()
                            or words in spec.category.lower())
                self.table.setRowHidden(r, not keep)
                any_shown = any_shown or keep
            self.table.setRowHidden(head, not any_shown)

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
        if col in (0, 3, 4, 5, 6, 7, 9, 12):  # anything the plan is built from
            self._mark_plan_stale()
        if col in (0, 3, 4, 9, 12):        # changes the number of runs
            self._update_cost()
        if col in (0, 9, 12):
            return
        if col in (5, 6, 3, 4, 7):         # delta / delta% / min / max / norm
            self._sync_row(item.row(), col)

    # Columns: 0 Vary | 1 Parameter | 2 Ref | 3 Min | 4 Max | 5 Delta |
    #          6 Delta% | 7 Norm | 8 Unit | 9 Values (sweep / grid)
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
        self._resync_notes()
        if self._plan_base_changed():
            self._mark_plan_stale(reason="a varied parameter's model value "
                                  "changed")
        self._update_cost()          # refresh CPU mirror + cost estimate

    def _resync_notes(self):
        """The mesh notes depend on the model (Discretize): refresh them."""
        for r, spec in self._row_spec.items():
            it = self.table.item(r, 10)
            if it is not None:
                self._set_note(it, spec)

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
                self._set_row_available(r, spec)
                if not pr.is_set(self.cfg, spec):
                    continue
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
        if plan is None or self.plan_kind not in ("jacobian", "sweep"):
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
        method = self._method()
        if method == "morris":
            return mp.n_runs(k, self.spin_traj.value())
        if method in sw.DESIGNS:
            sel, _sp = self._collect_sweep()
            return sw.n_runs([lv for _s, _r, lv in sel],
                             [r for _s, r, _lv in sel], method)
        return jac.n_runs(k, self._scheme())

    def _method(self) -> str:
        """jacobian | morris | sweep | grid (the combo entry)."""
        m = self.cb_method.currentData()
        return m if m in ALL_METHODS else self._methods[0]

    @staticmethod
    def _plan_kind_of(method: str) -> str:
        """The plan module of a method: sweep and grid share sweep_plan."""
        return "sweep" if method in sw.DESIGNS else method

    def _on_method_changed(self, *_):
        method = self._method()
        morris = method == "morris"
        sweep = method in sw.DESIGNS
        for w in (self.lbl_scheme, self.cb_scheme):
            w.setVisible(method == "jacobian")
        for w in (self.lbl_traj, self.spin_traj, self.lbl_levels, self.spin_levels):
            w.setVisible(morris)
        for w in (self.lbl_nlev, self.spin_nlev, self.lbl_spacing,
                  self.cb_spacing):
            w.setVisible(sweep)
        # Delta / Delta% / Norm are the Jacobian step; Values the levels.
        for c in (5, 6, 7):
            self.table.setColumnHidden(c, method != "jacobian")
        for c in (9, 11, 12, 13):
            self.table.setColumnHidden(c, not sweep)
        self.chk_reuse.setVisible(sweep)
        self.lbl_hint.setText({
            "morris": "screens Min..Max globally (mu*, sigma)",
            "sweep": "each parameter in turn, the others at Ref",
            "grid": "every combination of the values",
        }.get(method, "step = Delta per parameter"))
        self._update_cost()

    def _update_cost(self):
        if not hasattr(self, "lbl_cpus"):
            return   # called during construction before the cost labels exist
        self.lbl_cpus.setText(str(self._current_cpus()))
        self._refresh_level_cells()
        k = len(self._selected_rows())
        if k == 0:
            self.lbl_cost.setText("0 parameters selected")
            return
        method = self._method()
        if method == "morris":
            runs = mp.n_runs(k, self.spin_traj.value())
            base = "k=%d  →  %d runs (Morris N=%d × (k+1))" % (
                k, runs, self.spin_traj.value())
        elif method in sw.DESIGNS:
            try:
                runs = self._n_runs()
            except ValueError:
                self.lbl_cost.setText("k=%d  →  ? runs (check Min/Max/Values)"
                                      % k)
                return
            base = "k=%d  →  %d runs (%s)" % (k, runs, method)
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

    def _collect_sweep(self):
        """[(spec, Ref, levels)] of the ticked rows, in displayed units, and
        how each row's levels were made ("linear", "geometric", "typed").
        ValueError naming the row when Min/Max/Values are not usable."""
        selected, spacing = [], []
        tu = self._temp_unit()
        for r, spec in self._selected_rows():
            x0 = pr.get_display(self.cfg, spec, tu, system=self._table_units)
            try:
                lv, how = self._row_levels(r, spec)
            except ValueError as e:
                raise ValueError("%s: %s" % (spec.label, e))
            spacing.append(how)
            selected.append((spec, x0, lv))
        return selected, spacing

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
        """Ticked QoI: the Model-tab ones (built with the Model tab settings,
        frozen in self._plan_model_qoi), then the other ones. Raises
        ValueError when a Model-tab QoI is ticked and those settings are
        missing."""
        specs = []
        self._plan_model_qoi = None
        model = [q for q in mq.MODEL_QOIS
                 if self._model_qoi_checks[q].isChecked()]
        if model:
            st = self._model_qoi_settings(model)
            specs += mq.make_specs(st["zoi"], st["window"],
                                   st["evf_threshold"], qois=model,
                                   chip_y0=st.get("chip_y0"))
            self._plan_model_qoi = st
        specs += [q for q in qoi_mod.REGISTRY
                  if q.id in self._qoi_checks
                  and self._qoi_checks[q.id].isChecked()]
        return specs

    def _model_qoi_settings(self, qois) -> dict:
        """ZOI, window T, EVF threshold and eps_q of the Model tab for the
        Model-tab QoI `qois`; ValueError when one is missing or the ZOI
        leaves the zone the runs extract."""
        if self._model_settings_getter is None:
            raise ValueError("the Model sizing QoI need the settings of the "
                             "Optimization > Setup tab")
        try:
            st = self._model_settings_getter()
            window = tuple(float(v) for v in st["window"])
            zoi = tuple(float(v) for v in st["zoi"])
        except Exception as e:
            raise ValueError("Set the ZOI and the window T in the Setup tab "
                             "first (%s)." % e)
        eps = {k: float(v) for k, v in (st.get("eps") or {}).items()}
        missing = [q for q in qois if not (eps.get(q) or 0) > 0]
        if missing:
            raise ValueError("Set \u03b5_q of %s in the Setup tab first."
                             % ", ".join(missing))
        if not (zoi[0] < zoi[1] and zoi[2] < zoi[3]):
            raise ValueError("The ZOI of the Setup tab is empty.")
        if any(q in mq.FIELD_QUANTITIES for q in qois) \
                and not self.chk_full_domain.isChecked():
            b = self.cfg.bbox
            crop = {"xmin": float(b.xmin), "xmax": float(b.xmax),
                    "ymin": float(b.ymin), "ymax": float(b.ymax)}
            if not mq.zoi_within_crop(zoi, crop):
                raise ValueError(
                    "The ZOI of the Setup tab x[%g, %g] y[%g, %g] is "
                    "not inside the ROI the runs extract x[%g, %g] y[%g, %g]: "
                    "widen the ROI (Geometry tab) or tick 'Whole Eulerian "
                    "domain'." % (zoi + (crop["xmin"], crop["xmax"],
                                         crop["ymin"], crop["ymax"])))
        return {"zoi": zoi, "window": window,
                "evf_threshold": float(st.get("evf_threshold", 0.5)),
                "chip_y0": st.get("chip_y0"),
                "eps": {q: eps[q] for q in qois}}

    def _selected_field_vars(self):
        return [v for v, cb in self._field_checks.items() if cb.isChecked()]

    def _on_generate(self):
        method = self._method()
        outside = []
        try:
            qois = self._selected_qoi_specs()
            field_vars = self._selected_field_vars()
            if method in sw.DESIGNS:
                if field_vars:
                    self._warn("Field (ROI) screening is only available with "
                               "the Jacobian method; ignoring the field "
                               "selection for the %s." % method)
                    field_vars = []
                if not qois:
                    self._warn("Tick at least one scalar QoI for the %s."
                               % method)
                    return
                selected, spacing = self._collect_sweep()
                if not selected:
                    self._warn("Tick at least one parameter to vary.")
                    return
                plan = sw.build_plan(selected, method,
                                     temp_unit=self._temp_unit(),
                                     unit_system=self._table_units,
                                     spacing=spacing)
            elif method == "morris":
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
        self.plan_kind = self._plan_kind_of(method)
        self.selected_qois = qois
        self.plan_field_vars = list(field_vars)
        qoi_names = [q.id for q in qois] + ["%s[field]" % v for v in field_vars]
        self.status.setStyleSheet("color: #15803d;")
        if method in sw.DESIGNS:
            msg = ("%s plan ready: %d parameters, %d runs, %d QoI (%s)."
                   % ("Sweep" if method == "sweep" else "Grid", plan.k,
                      plan.n_runs, len(qoi_names),
                      ", ".join(qoi_names) if qoi_names else "—"))
            if method == "grid" and not plan.base_on_grid:
                msg += (" The Ref point is not one of the combinations: it "
                        "is run in addition (run 1).")
            if method == "grid" and plan.k > 3:
                self.status.setStyleSheet("color: #b45309;")
                msg += (" A grid of %d parameters grows fast: a sweep, or "
                        "Morris, screens them for fewer runs." % plan.k)
            self.status.setText(msg + " Ready for the run step.")
        elif method == "morris":
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
        if (self._prefs_getter() if self._prefs_getter else None) is None:
            self._warn("No preferences available (Abaqus command/script).")
            return
        prefs = self.cmb_machine.run_prefs()
        if prefs is None:
            self._warn("Machine \"%s\" is not in Preferences - Settings - "
                       "Machines any more: choose another one."
                       % self.cmb_machine.machine())
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
        # ({machine}__{profile}__sensitivity__{stamp}) with a config.json; its run files
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
        # Sweep / grid campaigns: folder and job names "sweep" / "grid".
        sweep = self.plan_kind == "sweep"
        self._run_job_prefix = self.plan.design if sweep else "sensitivity"
        root_wd = wd
        _study_cfg = self._plan_record(field_vars, cpus)
        try:
            wd = create_study_dir(wd, _pname, self._run_job_prefix,
                                  _study_cfg,
                                  machine=self.cmb_machine.machine_host())
        except OSError:
            pass   # fall back to the flat working directory
        cache_folders = (self._reuse_folders(root_wd, exclude=wd)
                         if sweep and self.chk_reuse.isChecked() else None)
        series_zone, evf_thr = self._series_zone() if sweep else (None, 0.5)
        self._series_zone_used = series_zone

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
        self._bars_job = None
        self._bars_remote = is_remote(prefs)
        self.run_bars.start(remote=self._bars_remote)
        self.run_bars.setVisible(True)
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
            warmup_frac=0.0,
            job_prefix=self._run_job_prefix, field_vars=field_vars,
            eps=(self._plan_model_qoi or {}).get("eps"),
            remote_prefs=prefs if is_remote(prefs) else None,
            extract_full_domain=self._run_full_domain,
            series_zone=series_zone, evf_threshold=evf_thr,
            cache_folders=cache_folders)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.runDone.connect(self._on_run_done)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._thread.start()

    def set_known_folders_getter(self, fn):
        """fn() -> study folders of Model sizing, whose finished runs a
        sweep / grid plan may reuse."""
        self._known_folders_getter = fn

    def _reuse_folders(self, root, exclude=None) -> list:
        """Folders whose finished runs a sweep / grid plan may reuse: the
        earlier sweep / grid plan folders of the working directory and the
        study folders of Model sizing."""
        out = []
        try:
            for d in sorted(Path(root).iterdir()):
                if d.is_dir() and ("__sweep__" in d.name
                                   or "__grid__" in d.name) \
                        and (exclude is None or d != Path(exclude)):
                    out.append(str(d))
        except OSError:
            log_swallowed("listing earlier plan folders", level=logging.DEBUG)
        if self._known_folders_getter is not None:
            try:
                for f in self._known_folders_getter() or []:
                    if f and str(f) not in out:
                        out.append(str(f))
            except Exception:
                log_swallowed("reading the Model sizing study folders",
                              level=logging.DEBUG)
        return out

    def _series_zone(self):
        """(ZOI, EVF threshold) of the time curves: those frozen with the
        plan's Model sizing QoI, else the current Setup settings, else no
        zone (forces only)."""
        st = self._plan_model_qoi
        if not st and self._model_settings_getter is not None:
            try:
                st = self._model_settings_getter()
            except Exception:
                st = None
        try:
            zoi = tuple(float(v) for v in st["zoi"])
            if zoi[0] < zoi[1] and zoi[2] < zoi[3]:
                return zoi, float(st.get("evf_threshold", 0.5))
        except Exception:
            pass
        return None, 0.5

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
            elif self.plan_kind == "sweep":
                d.update(base=float(plan.base[i]),
                         levels=[float(v) for v in plan.level_values[i]],
                         spacing=(plan.spacing[i] if i < len(plan.spacing)
                                  else "linear"))
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
               "machine": self.cmb_machine.machine_label(),
               "warmup_frac": 0.0,
               "temp_unit": tu,
               "unit_system": (system.to_dict()
                               if hasattr(system, "to_dict") else None),
               "varied_parameters": params}
        if self.plan_kind == "jacobian":
            rec["scheme"] = plan.scheme
            rows = jac.profile_table(plan)
        elif self.plan_kind == "sweep":
            zone, thr = self._series_zone()
            rec.update(design=plan.design, base_on_grid=bool(
                plan.base_on_grid), reuse_runs=bool(self.chk_reuse.isChecked()),
                time_curves_zoi=list(zone) if zone else None,
                time_curves_evf_threshold=thr)
            rows = sw.profile_table(plan)
        else:
            rec.update(N=int(plan.N), num_levels=int(plan.num_levels),
                       seed=plan.seed)
            rows = mp.profile_table(plan)
        runs = []
        for d in rows:
            run = {"run": d["run"],
                   "job": "%s_run%03d" % (self._job_prefix_of_plan(),
                                          d["run"] - 1)}
            if "kind" in d:
                k_ = d["kind"]
                if self.plan_kind == "sweep" and k_ not in ("base", "grid"):
                    k_ = plan.param_paths[int(k_)]     # the varied parameter
                run["kind"] = k_
            run["values"] = {p: float(d[p]) for p in plan.param_paths}
            runs.append(run)
        rec["runs"] = runs
        st = getattr(self, "_plan_model_qoi", None)
        if st:
            # ZOI [mm], window T (fractions of the simulated time), EVF
            # threshold and eps_q of the Model tab, as used for this plan.
            rec["model_qoi"] = {"zoi": list(st["zoi"]),
                                "window": list(st["window"]),
                                "evf_threshold": st["evf_threshold"],
                                "eps": dict(st["eps"])}
        return rec

    def _job_prefix_of_plan(self) -> str:
        if self.plan_kind == "sweep" and self.plan is not None:
            return self.plan.design
        return "sensitivity"

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
        job = "%s_run%03d" % (self._run_job_prefix, self._running_index)
        from gui.sensitivity.run_cache import job_dir
        run_dir = job_dir(self._run_workdir, job)     # <campaign>/<job>/
        sta = run_dir / ("%s.sta" % job)
        if job != self._bars_job:
            self._bars_job = job
            self.run_bars.start(remote=self._bars_remote)
        self.run_bars.poll(sta_path=sta,
                           log_path=script_log_path(run_dir, job),
                           run_dir=run_dir, job=job,
                           sim_time=float(getattr(self.cfg.step, "sim_time",
                                                  0.0) or 0.0) or None)
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
        self._sync_compare_button()
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

    def refresh_machines(self):
        """The machine list of the Preferences changed."""
        self.cmb_machine.refresh()

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
        """fn() -> {"eps": {Vx, Vy, T, EVF, Fc, Ff: eps_q}, "window": (a, b),
        "zoi": (xmin, xmax, ymin, ymax), "evf_threshold": t}, read from the
        Model tab when a ZOI is proposed or a plan is generated. Connecting
        it makes the Model-tab QoI the default ones."""
        self._model_settings_getter = fn
        on = fn is not None
        # Wiring, not a user edit: the defaults change without discarding a
        # plan already generated.
        # The chip thickness is not ticked by default (the six QoI of the
        # sizing studies are).
        for q, cb in self._model_qoi_checks.items():
            cb.blockSignals(True)
            cb.setEnabled(on)
            cb.setChecked(on and q != mq.CHIP_QOI)
            cb.blockSignals(False)

    def _model_settings(self):
        if self._model_settings_getter is None:
            raise ValueError("the Setup tab settings are not available")
        st = self._model_settings_getter()
        return dict(st.get("eps") or {}), tuple(st["window"])

    def _on_propose_zoi(self):
        if not self._field_maps or self._map_mesh is None:
            return
        try:
            eps, window = self._model_settings()
        except Exception as e:
            self.lbl_zoi.setText("Set \u03b5_q and the window T in the Setup "
                                 "tab first (%s)." % e)
            return
        need = sorted({zp.FIELD_TO_EPS[v] for v in self._field_maps
                       if v in zp.FIELD_TO_EPS} - {k for k, v in eps.items()
                                                    if v and v > 0})
        if need:
            self.lbl_zoi.setText("Missing \u03b5_q in the Setup tab: %s."
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
        written = [path]
        try:
            label_for = lambda p: pr.spec_for(p).label
            xr.write_csv(self._last_result, path, label_for=label_for)
            plan = self._sweep_plan()
            if self._last_result.plan_kind == "sweep" and plan is not None:
                # one row per run next to the trends
                p_runs = str(Path(path).with_name(
                    Path(path).stem + "_runs" + (Path(path).suffix or ".csv")))
                jobs = ["%s_run%03d" % (plan.design, i)
                        for i in range(plan.n_runs)]
                xr.write_runs_csv(self._last_result, plan, p_runs, jobs=jobs,
                                  label_for=label_for)
                written.append(p_runs)
        except Exception as e:
            self._warn("Export failed: %s" % e)
            return
        self.status.setStyleSheet("color: #15803d;")
        self.status.setText("Results exported to %s" % " and ".join(written))

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
        self.run_bars.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_gen.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _show_results(self, result):
        """Fill the results table: one row per parameter, columns per QoI.
        Morris shows mu* (sigma); Jacobian shows the sensitivity value.
        Sweep / grid: one row per run (parameter values and QoI)."""
        sweep = result.plan_kind == "sweep"
        self.lbl_chart_param.setVisible(sweep)
        self.cb_chart_param.setVisible(sweep)
        if sweep:
            self._show_sweep_results(result)
            return
        self._clear_series()
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

    # ------------------------------------------------------------------
    # Sweep / grid results: runs table, trend chart, time curves
    # ------------------------------------------------------------------
    def _sweep_plan(self):
        plan = self._run_plan or self.plan
        return plan if isinstance(plan, sw.SweepPlan) else None

    def _show_sweep_results(self, result):
        plan = self._sweep_plan()
        tbl = self.results_table
        tbl.clear()
        if plan is None:
            tbl.setRowCount(0)
            tbl.setColumnCount(0)
            return
        n_q = int(result.Y.shape[1])
        qoi_ids = list(result.qoi_ids[:n_q])
        header = (["Run", "Varied"]
                  + ["%s [%s]" % (sp_.label, self._plan_unit(plan, sp_))
                     for sp_ in plan.specs] + qoi_ids)
        tbl.setColumnCount(len(header))
        tbl.setHorizontalHeaderLabels(header)
        tbl.setRowCount(plan.n_runs)
        failed = set(result.failures)
        n_att = getattr(result, "n_attempted", plan.n_runs)
        for r in range(plan.n_runs):
            tbl.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            tbl.setItem(r, 1, QTableWidgetItem(
                self._run_kind_label(plan, plan.run_kind[r])))
            for j, v in enumerate(plan.X[r]):
                tbl.setItem(r, 2 + j, QTableWidgetItem(_fmt(float(v))))
            for j in range(n_q):
                v = float(result.Y[r, j]) if r < result.Y.shape[0] else \
                    float("nan")
                if np.isfinite(v):
                    txt = _fmt(v)
                elif r in failed:
                    txt = "failed"
                elif r >= n_att:
                    txt = "not run"
                else:
                    txt = "—"
                tbl.setItem(r, 2 + plan.k + j, QTableWidgetItem(txt))
        tbl.resizeColumnsToContents()
        self.cb_chart_qoi.blockSignals(True)
        self.cb_chart_qoi.clear()
        self.cb_chart_qoi.addItems(qoi_ids)
        self.cb_chart_qoi.blockSignals(False)
        self.cb_chart_param.blockSignals(True)
        self.cb_chart_param.clear()
        for sp_ in plan.specs:
            self.cb_chart_param.addItem(sp_.label, sp_.path)
        self.cb_chart_param.blockSignals(False)
        self._draw_chart()
        self._fill_series(result, plan)

    def _selected_run_rows(self):
        return sorted({i.row() for i in self.results_table.selectedIndexes()})

    def _sync_compare_button(self):
        res = self._last_result
        ok = (res is not None and res.plan_kind == "sweep"
              and self._sweep_plan() is not None
              and self._run_workdir is not None
              and len(self._selected_run_rows()) == 2)
        self.btn_compare_runs.setEnabled(ok)

    def _on_compare_runs(self):
        plan = self._sweep_plan()
        rows = self._selected_run_rows()
        if plan is None or len(rows) != 2 or self._run_workdir is None:
            self._warn("Select two runs (two rows) of the Table.")
            return
        failed = set(getattr(self._last_result, "failures", []) or [])
        bad = [r + 1 for r in rows if r in failed]
        if bad:
            self._warn("Run %s failed: it has no results to compare."
                       % ", ".join(map(str, bad)))
            return
        a, b = rows
        if b == plan.base_row:            # the Ref run is the reference
            a, b = b, a
        job = lambda r: "%s_run%03d" % (plan.design, r)
        self.compareRunsRequested.emit(str(self._run_workdir), job(a), job(b))

    def _qoi_eps(self, qid):
        """eps_q of a QoI of the run (Model sizing QoI only), else None."""
        eps = (self._plan_model_qoi or {}).get("eps") or {}
        for q in self.selected_qois:
            if q.id == qid and getattr(q, "eps_key", "") in eps:
                return float(eps[q.eps_key])
        return None

    def _qoi_unit(self, qid) -> str:
        for q in self.selected_qois:
            if q.id == qid:
                return getattr(q, "unit", "") or ""
        return ""

    def _draw_sweep_chart(self, result, qid):
        plan = self._sweep_plan()
        idx = self.cb_chart_param.currentIndex()
        if plan is None or idx < 0 or idx >= plan.k:
            self._canvas.draw_idle()
            return
        spec = plan.specs[idx]
        a = (result.analyses.get(qid) or {}).get(spec.path)
        ax = self._fig.add_subplot(111)
        unit_q = self._qoi_unit(qid)
        if not isinstance(a, dict):
            self.lbl_chart_note.setText("No result for %s." % qid)
            self._canvas.draw_idle()
            return
        j = list(result.qoi_ids).index(qid)
        y = np.asarray(result.Y[:, j], dtype=float)
        eps = self._qoi_eps(qid)
        q_ref = a.get("q_ref", float("nan"))
        x_ref = float(plan.base[idx])
        if eps and np.isfinite(q_ref):
            ax.axhspan(q_ref - eps, q_ref + eps, color="#16a34a", alpha=0.12,
                       label="Ref ± ε_q")
        if plan.design == "grid" and plan.k > 1:
            # One thin line per combination of the other parameters: lines
            # that are not parallel show an interaction.
            first = 0 if plan.base_on_grid else 1
            others = [c for c in range(plan.k) if c != idx]
            groups = {}
            for r in range(first, plan.n_runs):
                key = tuple(round(float(plan.X[r, c]), 12) for c in others)
                groups.setdefault(key, []).append(r)
            for key, rows in sorted(groups.items()):
                rows = [r for r in rows if np.isfinite(y[r])]
                if not rows:
                    continue
                rows.sort(key=lambda r: plan.X[r, idx])
                lab = ", ".join("%s=%s" % (plan.specs[c].label, _fmt(v))
                                for c, v in zip(others, key))
                ax.plot(plan.X[rows, idx], y[rows], "o-", linewidth=0.9,
                        markersize=3, alpha=0.7, label=lab)
            ax.plot(a["x"], a["q"], "s-", color="#111827", linewidth=2,
                    label="main effect (mean)")
        else:
            ax.plot(a["x"], a["q"], "o-", color="#2563eb", label=qid)
        if np.isfinite(q_ref):
            ax.plot([x_ref], [q_ref], "*", color="#b91c1c", markersize=12,
                    label="Ref run")
        sp_i = plan.spacing[idx] if idx < len(plan.spacing) else "linear"
        if str(sp_i).startswith("geometric") or (
                sp_i == "typed" and np.all(np.asarray(a["x"]) > 0)
                and len(a["x"]) > 2
                and np.max(a["x"]) / np.min(a["x"]) >= 100):
            ax.set_xscale("log")
        ax.set_xlabel("%s [%s]" % (spec.label, self._plan_unit(plan, spec)),
                      fontsize=9)
        ax.set_ylabel("%s%s" % (qid, " [%s]" % unit_q if unit_q else ""),
                      fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        parts = ["%d points" % a.get("n", 0)]
        if np.isfinite(a.get("slope", np.nan)):
            parts.append("slope of the least-squares line %s %s/%s" % (
                _fmt(a["slope"]), unit_q or "QoI unit",
                self._plan_unit(plan, spec) or "unit"))
        if np.isfinite(a.get("r2", np.nan)):
            parts.append("R² %.3f" % a["r2"])
        if np.isfinite(a.get("spearman", np.nan)):
            parts.append("Spearman ρ %.2f (±1 = monotone)"
                         % a["spearman"])
        if np.isfinite(a.get("max_dev_eps", np.nan)):
            parts.append("max |q − q_ref| / ε_q = %.2f"
                         % a["max_dev_eps"])
        note = "; ".join(parts) + "."
        if plan.design == "grid":
            note += (" Main effect: mean of the QoI over the other "
                     "parameters at each value.")
        self.lbl_chart_note.setStyleSheet("color: #374151;")
        self.lbl_chart_note.setText(note)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _clear_series(self):
        self.lst_series_runs.blockSignals(True)
        self.lst_series_runs.clear()
        self.lst_series_runs.blockSignals(False)
        self.cb_series_qty.blockSignals(True)
        self.cb_series_qty.clear()
        self.cb_series_qty.blockSignals(False)
        self._fig_series.clear()
        self._canvas_series.draw_idle()
        self.lbl_series_note.setText(
            "Run a sweep or grid plan to compare the runs over time.")

    def _fill_series(self, result, plan):
        series = getattr(result, "series", None) or []
        qtys = []
        for d in series:
            for q in (d or {}):
                if q not in ("t", "t_force") and q not in qtys:
                    qtys.append(q)
        order = list(mq.MODEL_QOIS)
        qtys.sort(key=lambda q: order.index(q) if q in order else 99)
        keep = self.cb_series_qty.currentText()
        self.cb_series_qty.blockSignals(True)
        self.cb_series_qty.clear()
        self.cb_series_qty.addItems(qtys)
        if keep in qtys:
            self.cb_series_qty.setCurrentText(keep)
        self.cb_series_qty.blockSignals(False)
        self.lst_series_runs.blockSignals(True)
        self.lst_series_runs.clear()
        for r in range(plan.n_runs):
            vals = ", ".join("%s=%s" % (sp_.label, _fmt(float(v)))
                             for sp_, v in zip(plan.specs, plan.X[r]))
            it = QListWidgetItem("%d · %s" % (r + 1, vals))
            it.setToolTip("Run %d (%s): %s" % (
                r + 1, self._run_kind_label(plan, plan.run_kind[r]), vals))
            it.setData(Qt.UserRole, r)
            ok = r < len(series) and series[r]
            it.setFlags((Qt.ItemIsUserCheckable | Qt.ItemIsEnabled) if ok
                        else Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if ok else Qt.Unchecked)
            self.lst_series_runs.addItem(it)
        self.lst_series_runs.blockSignals(False)
        self._draw_series()

    def _draw_series(self, *_):
        self._fig_series.clear()
        result = self._last_result
        plan = self._sweep_plan()
        q = self.cb_series_qty.currentText()
        series = getattr(result, "series", None) if result else None
        if not series or plan is None or not q:
            self._canvas_series.draw_idle()
            return
        ax = self._fig_series.add_subplot(111)
        tkey = "t_force" if q in ("Fc", "Ff") else "t"
        t_end = None
        drawn = 0
        for k in range(self.lst_series_runs.count()):
            it = self.lst_series_runs.item(k)
            if it.checkState() != Qt.Checked:
                continue
            r = int(it.data(Qt.UserRole))
            d = series[r] if r < len(series) else None
            if not d or q not in d or tkey not in d:
                continue
            t = np.asarray(d[tkey], dtype=float)
            v = np.asarray(d[q], dtype=float)
            if t.size != v.size or not t.size:
                continue
            t_end = float(t[-1]) if t_end is None else max(t_end, float(t[-1]))
            base = r == plan.base_row
            ax.plot(t, v, color="#111827" if base else None,
                    linewidth=1.8 if base else 0.9,
                    label=("%d (Ref)" % (r + 1)) if base else str(r + 1))
            drawn += 1
        window = (self._plan_model_qoi or {}).get("window")
        if window and t_end:
            ax.axvspan(float(window[0]) * t_end, float(window[1]) * t_end,
                       color="#2563eb", alpha=0.07, label="window T")
        ax.set_xlabel("time [s]", fontsize=9)
        unit = mq.UNITS.get(q, "")
        ax.set_ylabel("%s [%s]" % (q, unit) if unit else q, fontsize=9)
        ax.grid(True, alpha=0.3)
        if drawn and drawn <= 12:
            ax.legend(fontsize=7, loc="best", title="run", title_fontsize=7)
        note = ("%s over time: area mean over the ZOI at each frame (Vx, Vy, "
                "T over the material elements, EVF ≥ threshold)"
                % q if tkey == "t" else
                "%s over time: reaction force / width at each history sample"
                % q)
        note += ". The Table tab gives each run's parameter values."
        if not any(d and "Vx" in d for d in series):
            note += (" No ZOI was set when the plan ran: the fields have no "
                     "time curves.")
        self.lbl_series_note.setText(note)
        self._fig_series.tight_layout()
        self._canvas_series.draw_idle()

    def _draw_chart(self):
        self._fig.clear()
        result = self._last_result
        if result is None or not result.qoi_ids:
            self._canvas.draw_idle()
            return
        if result.plan_kind == "sweep":
            self.lbl_chart_rank.setVisible(False)
            self.cb_chart_rank.setVisible(False)
            self.lbl_chart_note.setText("")
            qid = self.cb_chart_qoi.currentText() or result.qoi_ids[0]
            self._draw_sweep_chart(result, qid)
            return
        self.lbl_chart_note.setStyleSheet("color: #b45309;")
        qid = self.cb_chart_qoi.currentText() or result.qoi_ids[0]
        jacobian = result.plan_kind == "jacobian"
        self.lbl_chart_rank.setVisible(jacobian)
        self.cb_chart_rank.setVisible(jacobian)
        self.lbl_chart_note.setText("")
        if jacobian:
            key = self.cb_chart_rank.currentData() or "sensitivity"
            if qid.endswith(rc.EPS_SUFFIX):
                key = "sensitivity"       # already dimensionless
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
        if qid.endswith(rc.EPS_SUFFIX):
            # Dimensionless: 1 = the admitted deviation eps_q.
            ax.axvline(1.0, color="#b91c1c", linestyle="--", linewidth=1)
            self.lbl_chart_note.setText(
                "Red line: \u03b5_q. Jacobian: |dQ/dx\u00b7\u03b4| / "
                "\u03b5_q, the change over one step \u03b4. Morris: "
                "\u03bc* / \u03b5_q, the mean change over the whole "
                "Min..Max range.")
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
        mod = {"jacobian": jac, "sweep": sw}.get(self.plan_kind, mp)
        rows = mod.profile_table(plan)
        if self.plan_kind == "sweep":
            # "base", "grid", or the label of the parameter varied
            rows = [dict(d, kind=self._run_kind_label(plan, d["kind"]))
                    for d in rows]
        paths = plan.param_paths
        has_kind = "kind" in rows[0] if rows else False
        kind_w = max([5] + [len(str(d.get("kind", ""))) for d in rows])
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
                cells.append("%-*s" % (kind_w, d["kind"]))
            cells += [_fmt(d[p]) for p in paths]
            lines.append(" | ".join(cells))
        self.preview.setPlainText("\n".join(lines))

    @staticmethod
    def _run_kind_label(plan, kind) -> str:
        """Sweep run kind for display: "base", "grid", or the label of the
        parameter that run varies."""
        if kind in ("base", "grid"):
            return kind
        try:
            return plan.specs[int(kind)].label
        except (ValueError, IndexError):
            return str(kind)

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
