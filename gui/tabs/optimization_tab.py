# -*- coding: utf-8 -*-
"""
Optimization tab: size the CEL model for the paper's methodology.

Three studies, all measured in the ZOI (the Optimization measurement zone,
distinct from the output ROI of the Geometry tab):

  * the mass-scaling factor by an independence study on a fixed mesh and
    domain (gui.sensitivity.ms_independence): increasing ms compared
    successively against the same ABSOLUTE tolerances eps_q as the domain
    study, with the filter and reverberation checks as extra safeguards;
  * mesh convergence by Richardson extrapolation / GCI on a fixed domain
    (gui.sensitivity.mesh_gci), with RELATIVE tolerances per quantity;
  * Eulerian-domain sizing by a sequential independence study
    (gui.sensitivity.domain_independence): each dimension grown by a
    constant step from the initial domain = ZOI + margin, successive runs
    compared by the mean absolute difference (paper Eq. 5, 7) against
    ABSOLUTE tolerances eps_q, residual influence bounded by a geometric tail
    (fallback: successive criterion), run safeguards R_K, R_HG and outputs.

All studies share the time window T. Each candidate is one Abaqus run
(run_simul) launched here, replicating the Sensitivity tab's run mechanism;
the studies receive a deep copy of the current config, which is never
modified. The study cores are unit-tested elsewhere.
"""
from __future__ import annotations

import copy
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QGroupBox,
    QCheckBox, QLineEdit, QPlainTextEdit, QTabWidget, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QProgressBar, QSplitter,
    QScrollArea, QFrame
)
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui.core.async_call import run_async
from gui.core.domain_sizing import DomainDims
from gui.core.model_config import OptimizationCfg as _OptimizationCfg
from gui.core.logging_util import log_swallowed
from gui.core.sta_parser import parse_sta
from gui.sensitivity.mesh_gci_worker import MeshGciWorker
from gui.sensitivity.domain_independence import initial_dims_from_zoi
from gui.sensitivity.domain_independence_worker import DomainIndependenceWorker
from gui.sensitivity.ms_independence import filter_guards, parse_ms_values
from gui.sensitivity.ms_independence_worker import MsIndependenceWorker
from gui.sensitivity.run_record import (
    GuardSettings, cost_record, guard_reasons, make_guard_fn)
from gui.sensitivity.run_worker import (
    abaqus_terminate_job, build_abaqus_args, kill_process_tree_by_pid,
    script_log_path)
from gui.results.reader import ResultsBundle
from gui.widgets.geometry_preview import GeometryPreview


# Quantity -> bundle element-field name (velocity components V1/V2 are written
# per element by run_simul; T=TEMP; PEEQ/EVF as-is). Units are informative.
_QUANTITIES = [
    ("Vx", "V1", "mm/s"),
    ("Vy", "V2", "mm/s"),
    ("T",  "TEMP", "K or °C"),
    ("EVF", "EVF", "-"),
    ("Fc", None, "N/mm"),
    ("Ff", None, "N/mm"),
]
# Convergence table (report, Part B, T7): one row per comparison of the
# domain study, per GCI quantity and per interaction check.
_TABLE_COLUMNS = ("study", "item", "from", "to", "E_max / ratio", "q_crit",
                  "safeguards", "decision", "mode", "C_CPU [s]")
# Default values of the persisted Optimization settings (one instance, read
# only, so the widget defaults cannot drift from the dataclass).
OptimizationCfgDefaults = _OptimizationCfg()
# Domain-study settings: (attribute of cfg.optimization, label, min, max).
_DOM_SPINS = [
    ("dom_step_elems", "step \u0394 (elems)", 1, 1000),
    ("dom_n_max", "n_max", 2, 50),
    ("dom_n_hold", "n_hold", 1, 10),
    ("dom_m_ratios", "m (ratios)", 1, 10),
]
# Domain-study text settings: (attribute, label, unit/tooltip).
_DOM_TEXTS = [
    ("window_start", "T start", "fraction of the simulated time"),
    ("window_end", "T end", "fraction of the simulated time"),
    ("rk_max", "G_K,max", "max of R_K = \u03a3ALLKE/\u03a3ALLIE over T"),
    ("rhg_max", "G_HG,max", "max of R_HG = \u03a3ALLAE/\u03a3ALLIE over T"),
]
# Force quantities -> the tool-RP reaction-force history channel.
_FORCE_CHANNELS = {"Fc": "RF1_RP", "Ff": "RF2_RP"}
_DIM_ORDER = ("l_wp", "h_wp", "h_void", "l_void")


class OptimizationTab(QWidget):
    # Emitted when a persisted optimization parameter changes, so the
    # main window can mark the profile dirty.
    changed = Signal()
    # Carries log text to the GUI thread. run_bundle runs in the study
    # workers' threads; touching the QPlainTextEdit from there is a data race
    # with the GUI thread's painting that can end in an access violation.
    _log_requested = Signal(str)

    def __init__(self, cfg, prefs_getter=None, cpus_getter=None,
                 profile_name_getter=None):
        super().__init__()
        self.cfg = cfg
        self._prefs_getter = prefs_getter
        self._profile_name_getter = profile_name_getter
        self._loading = False   # guard: True while populating from cfg
        self._cpus_getter = cpus_getter
        self._initial = None            # DomainDims = ZOI + margin (D2-a)
        self._cancel_evt = threading.Event()
        # Published by run_bundle so _on_cancel can name the job to
        # `abaqus terminate` and reach the solver behind the launcher.
        self._current_job = None
        self._current_proc = None
        self._current_abaqus_cmd = None
        self._current_run_dir = None
        self._last_domain_result = None  # StudyResult of the domain study
        self._last_domain_dir = None     # its study folder (exports)
        self._last_gci = None            # (MeshGciResult, calls, tol, dir)
        self._last_checks = None         # ChecksResult
        self._last_ms = None             # (MsStudyResult, folder)
        self._current_sta = None        # current job's .sta path (for progress)
        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(500)
        self._sim_timer.timeout.connect(self._poll_sta)

        # ================================================================
        # Layout (top to bottom)
        #   inputs | preview      horizontal splitter; the inputs scroll
        #   status bar            working dir, cancel, progress, status
        #   Log | Convergence | Table
        # A vertical splitter separates inputs and outputs. The inputs are
        # grouped in the order of the workflow (common settings, 1 mesh,
        # 2 domain, 3 checks), each step with its own run button.
        # ================================================================
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        _NUM_W = 78                       # width of a numeric field (px)

        def num_edit(text="", placeholder="", tip=""):
            le = QLineEdit(text)
            le.setPlaceholderText(placeholder)
            le.setFixedWidth(_NUM_W)
            if tip:
                le.setToolTip(tip)
            return le

        def spin_box(lo, hi, value):
            # No fixed width: the width comes from the style's size hint,
            # which accounts for the arrow buttons (side by side in the
            # Windows 11 style, where a fixed 64 px hid the value) and for
            # the widest value of the range. _NUM_W is only a floor.
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(value)
            sp.setMinimumWidth(_NUM_W)
            return sp

        def hint(text):
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#6b7280;")
            return lab

        def grid(group):
            g = QGridLayout(group)
            g.setHorizontalSpacing(8)
            g.setVerticalSpacing(4)
            return g

        # ---- Common settings: ZOI, sampling, window T, safeguards ------
        # The ZOI is DISTINCT from the ROI (Geometry tab, model output set
        # for DIC/IRT); empty fields default to the ROI. T and the
        # safeguards are shared by every study.
        gcom = QGroupBox("Common settings \u2014 ZOI, window T, safeguards")
        cg0 = grid(gcom)
        self.le_zoi = {}
        for c, (lbl, key) in enumerate([("x min", "xmin"), ("x max", "xmax"),
                                        ("y min", "ymin"), ("y max", "ymax")]):
            cg0.addWidget(QLabel(lbl), 0, 2 * c)
            le = num_edit(placeholder="= ROI",
                          tip="ZOI bound [mm]; empty = ROI bound")
            self.le_zoi[key] = le
            cg0.addWidget(le, 0, 2 * c + 1)
            le.textChanged.connect(self._draw_preview)
        self.btn_zoi_from_roi = QPushButton("ZOI = ROI")
        self.btn_zoi_from_roi.clicked.connect(self._zoi_from_roi)
        cg0.addWidget(self.btn_zoi_from_roi, 0, 8)
        cg0.addWidget(QLabel("grid step"), 1, 0)
        self.le_grid_step = num_edit(placeholder="= elem",
                                     tip="ZOI sampling step [mm]; empty = "
                                         "element size")
        cg0.addWidget(self.le_grid_step, 1, 1)
        self._dom_texts = {}
        for i, (attr, label, tip) in enumerate(_DOM_TEXTS):
            lab = QLabel(label); lab.setToolTip(tip)
            cg0.addWidget(lab, 1, 2 + 2 * i)
            le = num_edit(str(getattr(OptimizationCfgDefaults, attr)), tip=tip)
            self._dom_texts[attr] = le
            cg0.addWidget(le, 1, 3 + 2 * i)
        cg0.setColumnStretch(10, 1)

        # ---- Step 0 · mass-scaling factor by an independence study -------
        gms = QGroupBox("0 \u00b7 Mass scaling \u2014 independence study")
        sg = grid(gms)
        sg.addWidget(QLabel("ms values"), 0, 0)
        self.le_ms_values = QLineEdit(OptimizationCfgDefaults.ms_values)
        self.le_ms_values.setToolTip(
            "Mass-scaling factors to test, strictly increasing, separated by\n"
            "commas or spaces.")
        sg.addWidget(self.le_ms_values, 0, 1, 1, 4)
        sg.addWidget(QLabel("mesh [mm]"), 0, 5)
        self.le_ms_elem = num_edit(placeholder="= elem",
                                   tip="Element size of the ms runs [mm]; "
                                       "empty = Mesh tab element size")
        sg.addWidget(self.le_ms_elem, 0, 6)
        self.btn_ms = QPushButton("Run mass-scaling study")
        self.btn_ms.setToolTip(
            "Runs the ms values in increasing order on the current domain and\n"
            "compares each run with the previous one (E_max with the absolute\n"
            "eps_q of step 2). Safeguards: outputs, R_K, R_HG, filter check and\n"
            "reverberation check. Keeps the largest ms reached by an unbroken\n"
            "chain of successes; stops at the first failure.")
        self.btn_ms.clicked.connect(self._on_run_ms_independence)
        sg.addWidget(self.btn_ms, 1, 0, 1, 2)
        sg.addWidget(hint("Uses the \u03b5_q of step 2 and the current "
                          "domain. Needs the output filter with verification "
                          "(Step tab)."), 1, 2, 1, 6)
        sg.setColumnStretch(7, 1)

        # ---- Step 1 · mesh size by Richardson / GCI ---------------------
        gmesh = QGroupBox("1 \u00b7 Mesh size \u2014 Richardson / GCI on a "
                          "fixed domain")
        mg = grid(gmesh)
        mg.addWidget(QLabel("finest [mm]"), 0, 0)
        self.le_gci_finest = num_edit(placeholder="= elem")
        mg.addWidget(self.le_gci_finest, 0, 1)
        mg.addWidget(QLabel("ratio"), 0, 2)
        self.le_gci_ratio = num_edit("2")
        mg.addWidget(self.le_gci_ratio, 0, 3)
        mg.addWidget(QLabel("floor [mm]"), 0, 4)
        self.le_gci_min = num_edit(placeholder="none")
        mg.addWidget(self.le_gci_min, 0, 5)
        mg.addWidget(QLabel("n meshes"), 0, 6)
        self.sp_gci_n = spin_box(3, 6, 3)
        mg.addWidget(self.sp_gci_n, 0, 7)
        # Relative tolerances of the GCI study ONLY ("force" applies to Fc
        # and Ff). The persisted key (sizing_tol) is kept for old profiles.
        mg.addWidget(QLabel("tolerances (rel.)"), 1, 0)
        self._dj_eps = {}
        tol_row = QHBoxLayout()
        tol_row.setSpacing(4)
        for q in ("EVF", "TEMP", "V1", "V2", "force"):
            tol_row.addWidget(QLabel(q))
            le = num_edit("0.02", tip="Relative tolerance on %s (0.02 = 2%%). "
                                      "Empty = excluded." % q)
            le.setFixedWidth(56)
            self._dj_eps[q] = le
            tol_row.addWidget(le)
            tol_row.addSpacing(12)
        tol_row.addStretch(1)
        mg.addLayout(tol_row, 1, 1, 1, 8)
        self.btn_mesh = QPushButton("Run mesh convergence (GCI)")
        self.btn_mesh.setToolTip(
            "GCI/Richardson mesh convergence on the current (fixed) domain:\n"
            "n systematically-refined meshes, observed order p, extrapolated\n"
            "value and GCI per quantity. Recommends the coarsest mesh within\n"
            "tolerance of the reference.")
        self.btn_mesh.clicked.connect(self._on_run_mesh_gci)
        mg.addWidget(self.btn_mesh, 2, 0, 1, 4)
        mg.addWidget(hint("Uses the current domain: keep it conservative "
                          "(paper P9)."), 2, 4, 1, 5)
        mg.setColumnStretch(8, 1)

        # ---- Step 2 · Eulerian domain -----------------------------------
        gdom = QGroupBox("2 \u00b7 Eulerian domain \u2014 sequential "
                         "independence study")
        dg = grid(gdom)
        # absolute tolerances eps_q (E_max)
        dg.addWidget(QLabel("\u03b5_q (absolute)"), 0, 0)
        self._q_eps = {}
        _unit = {q: u for (q, _f, u) in _QUANTITIES}
        eps_row = QHBoxLayout()
        eps_row.setSpacing(4)
        for q in ("Vx", "Vy", "T", "EVF", "Fc", "Ff"):
            lbl = QLabel(q)
            tip = "%s tolerance [%s]" % (q, _unit[q])
            if q in ("Fc", "Ff"):
                tip += (" \u2014 %s on the tool RP divided by the element "
                        "size" % ("RF1" if q == "Fc" else "RF2"))
            lbl.setToolTip(tip)
            eps_row.addWidget(lbl)
            le = num_edit(placeholder=_unit[q], tip=tip)
            le.setFixedWidth(64)
            self._q_eps[q] = le
            eps_row.addWidget(le)
            eps_row.addSpacing(12)
        eps_row.addStretch(1)
        dg.addLayout(eps_row, 0, 1, 1, 9)
        # initial domain + caps, one row per dimension pair
        self._max = {}
        self._init_lbl = {}
        dg.addWidget(QLabel("dimension"), 1, 0)
        dg.addWidget(QLabel("initial [mm]"), 1, 1)
        dg.addWidget(QLabel("max cap [mm]"), 1, 2)
        dg.addWidget(QLabel("dimension"), 1, 4)
        dg.addWidget(QLabel("initial [mm]"), 1, 5)
        dg.addWidget(QLabel("max cap [mm]"), 1, 6)
        for r, (d_left, d_right) in enumerate(
                [("l_wp", "h_void"), ("h_wp", "l_void")], start=2):
            for d, c0 in ((d_left, 0), (d_right, 4)):
                dg.addWidget(QLabel(d), r, c0)
                il = QLabel("\u2014"); il.setStyleSheet("color:#374151;")
                self._init_lbl[d] = il
                dg.addWidget(il, r, c0 + 1)
                mx = num_edit(placeholder="no cap")
                self._max[d] = mx
                dg.addWidget(mx, r, c0 + 2)
        # study settings
        set_row = QHBoxLayout()
        set_row.setSpacing(4)
        set_row.addWidget(QLabel("margin (elems)"))
        self.sp_margin = spin_box(0, 50, 0)
        set_row.addWidget(self.sp_margin)
        self._dom_spins = {}
        for attr, label, lo, hi in _DOM_SPINS:
            set_row.addSpacing(10)
            set_row.addWidget(QLabel(label))
            sp = spin_box(lo, hi,
                          int(getattr(OptimizationCfgDefaults, attr)))
            self._dom_spins[attr] = sp
            set_row.addWidget(sp)
        set_row.addStretch(1)
        dg.addLayout(set_row, 4, 0, 1, 10)
        self.btn_init = QPushButton("Compute initial domain")
        self.btn_init.clicked.connect(self.compute_initial)
        dg.addWidget(self.btn_init, 5, 0, 1, 2)
        self.lbl_init = QLabel("\u2014")
        self.lbl_init.setStyleSheet("font-weight: bold;")
        dg.addWidget(self.lbl_init, 5, 2, 1, 8)
        self.btn_domain = QPushButton("Run domain sizing (independence)")
        self.btn_domain.setToolTip(
            "Sequential independence study: each dimension grown by a constant\n"
            "step from ZOI + margin (mesh and mass scaling held fixed), runs\n"
            "compared by the mean absolute difference in the ZOI against the\n"
            "absolute eps_q, residual influence bounded by a geometric tail.\n"
            "The domain diagonal only raises a warning.")
        self.btn_domain.clicked.connect(self._on_run_domain_independence)
        dg.addWidget(self.btn_domain, 6, 0, 1, 2)
        dg.addWidget(hint("Initial domain = ZOI + margin. Tail bound on the "
                          "last m ratios, fallback on the successive rule."),
                     6, 2, 1, 8)
        dg.setColumnStretch(9, 1)

        # ---- Step 3 · interaction checks --------------------------------
        gchk = QGroupBox("3 \u00b7 Interaction checks (paper \u00a75.7)")
        kg = grid(gchk)
        self.btn_checks = QPushButton("Run interaction checks")
        self.btn_checks.setToolTip(
            "A-posteriori checks of the sized model (paper \u00a75.7):\n"
            "mass-scaling factor inside its window at (h*, D*), the four\n"
            "dimensions grown together (1 run), ms* against the previous ms\n"
            "at (h*, D*) (1 run), and the GCI plan of step 1 run again on D*\n"
            "(h* must stay within tolerance). Available once a domain study\n"
            "has finished.")
        self.btn_checks.setEnabled(False)
        self.btn_checks.clicked.connect(self._on_run_interaction_checks)
        kg.addWidget(self.btn_checks, 0, 0)
        kg.addWidget(hint("f in its window at (h*, D*); D* grown in the four "
                          "directions (1 run); ms* vs the previous ms at "
                          "(h*, D*) (1 run); GCI of step 1 re-run on D*."),
                     0, 1)
        kg.setColumnStretch(1, 1)

        # ---- Inputs column (scrolls instead of squeezing) ---------------
        inputs = QWidget()
        il_ = QVBoxLayout(inputs)
        il_.setContentsMargins(0, 0, 4, 0)
        for gbox in (gcom, gms, gmesh, gdom, gchk):
            il_.addWidget(gbox)
        il_.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(inputs)
        # Never narrower than its content: the preview shrinks instead, and
        # only a vertical scrollbar can appear.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(inputs.sizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width())

        # ---- Preview (reuses the Geometry tab's preview widget) --------
        gprev = QGroupBox("Preview")
        pv = QVBoxLayout(gprev)
        self.preview = GeometryPreview()
        pv.addWidget(self.preview, 1)
        btn_prev = QPushButton("Refresh preview")
        btn_prev.clicked.connect(self._draw_preview)
        pv.addWidget(btn_prev)

        self._hsplit = QSplitter(Qt.Horizontal)
        self._hsplit.addWidget(scroll)
        self._hsplit.addWidget(gprev)
        self._hsplit.setStretchFactor(0, 3)
        self._hsplit.setStretchFactor(1, 2)
        self._hsplit.setChildrenCollapsible(False)

        # ---- Status bar (always visible) --------------------------------
        bar = QHBoxLayout()
        self.btn_open_wd = QPushButton("Open working dir")
        self.btn_open_wd.setToolTip("Open the Preferences working directory.")
        self.btn_open_wd.clicked.connect(self._open_working_dir)
        bar.addWidget(self.btn_open_wd)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        bar.addWidget(self.btn_cancel)
        # Live per-simulation progress bar (fed by parsing the current job's
        # .sta file, step_time/sim_time).
        self.sim_progress = QProgressBar()
        self.sim_progress.setRange(0, 100)
        self.sim_progress.setFormat("current simulation: %p%")
        self.sim_progress.setFixedWidth(220)
        self.sim_progress.setVisible(False)
        bar.addWidget(self.sim_progress)
        self.lbl_status = QLabel("")
        bar.addWidget(self.lbl_status, 1)

        # ---- Output tabs -----------------------------------------------
        self.tabs = QTabWidget()
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self._log_requested.connect(self._append_log)
        self.tabs.addTab(self.log, "Log")
        conv = QWidget(); cv = QVBoxLayout(conv)
        cv.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 3.2))
        self.canvas = FigureCanvas(self.fig)
        # Matplotlib navigation toolbar: interactive zoom / pan / home / save.
        self._nav = NavigationToolbar2QT(self.canvas, conv)
        cv.addWidget(self._nav)
        # Three axes (report, Part B, T8): the domain study (E_max per
        # comparison versus the tested value, one series per dimension), the
        # GCI study (f_q(h) relative to its reference, per quantity) and the
        # cost-E_max map (paper Fig. 13).
        # A fourth axis shows the mass-scaling study (E_max per comparison
        # versus the larger ms of the pair, paper Fig. 8).
        (self._ax_ms, self._ax_domain, self._ax_gci,
         self._ax_cost) = self.fig.subplots(1, 4)
        cv.addWidget(self.canvas, 1)
        self.table = QTableWidget(0, len(_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_TABLE_COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tabs.addTab(conv, "Convergence")
        self.tabs.addTab(self.table, "Table")

        _top = QWidget()
        tl = QVBoxLayout(_top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self._hsplit, 1)
        tl.addLayout(bar)
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.addWidget(_top)
        self._splitter.addWidget(self.tabs)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setChildrenCollapsible(False)
        outer.addWidget(self._splitter)

        # Auto-refresh the preview when the inputs that affect it change.
        for _d in _DIM_ORDER:
            self._max[_d].textChanged.connect(self._draw_preview)
        self.le_grid_step.textChanged.connect(self._draw_preview)
        self.sp_margin.valueChanged.connect(self._draw_preview)

        self._wire_opt_persistence()
        self.refresh_inputs()
        self._draw_preview()

    # =====================================================================
    # Config-derived inputs (pure; unit-testable)
    # =====================================================================
    def config_inputs(self) -> dict:
        c = self.cfg
        t1 = float(c.wp_position.y0 - c.tool_position.y0)
        rake = float(c.tool_geometry.rake_angle)
        mu = float(c.interaction.friction_coeff)
        roi = (float(c.bbox.xmin), float(c.bbox.xmax),
               float(c.bbox.ymin), float(c.bbox.ymax))
        elem = float(c.elem_size)
        tip = (float(c.tool_position.x0), float(c.tool_position.y0))
        return {"t1": t1, "rake": rake, "mu": mu, "roi": roi, "elem": elem,
                "tip": tip}

    def grid_step(self) -> float:
        """Spacing of the fixed ROI comparison grid (the evaluation points).
        Defaults to the element size when left blank."""
        txt = self.le_grid_step.text().strip().replace(",", ".")
        try:
            v = float(txt)
            if v > 0:
                return v
        except (ValueError, TypeError):
            pass
        return float(self.cfg.elem_size)


    def euler_offset(self):
        """(x0, y0) translation of the Eulerian instance in the assembly
        (cel_model.py:397); the ZOI is given in the assembly frame."""
        return (float(self.cfg.euler_position.x0),
                float(self.cfg.euler_position.y0))

    def compute_initial_dims(self) -> DomainDims:
        """Initial Eulerian domain = the ZOI + margin (decision D2-a), mapped
        to the domain dimensions (part rectangle (-l_wp, -h_wp) ->
        (l_void, h_void), cel_model.py:291, translated by euler_position) and
        snapped up to whole elements:
            l_wp = -xmin, l_void = xmax, h_wp = -ymin, h_void = ymax
        in the domain's own frame, each + margin. A blank ZOI is the ROI, so
        this reproduces the previous ROI-based value when no ZOI is set."""
        return initial_dims_from_zoi(
            self.zoi(), float(self.cfg.elem_size),
            margin_elems=int(self.sp_margin.value()),
            offset=self.euler_offset())

    # =====================================================================
    # UI actions
    # =====================================================================
    # ===================================================================
    # Persistence of the optimization parameters (saved in the .acpf profile)
    # ===================================================================
    def _opt_line_edits(self):
        les = [self.le_grid_step, self.le_gci_finest, self.le_gci_ratio,
               self.le_gci_min, self.le_ms_values, self.le_ms_elem]
        les += list(self.le_zoi.values())
        les += list(self._q_eps.values())
        les += list(self._dj_eps.values())
        les += list(self._max.values())
        les += list(self._dom_texts.values())
        return les

    def _wire_opt_persistence(self):
        for le in self._opt_line_edits():
            le.textChanged.connect(self._sync_opt_to_cfg)
        self.sp_margin.valueChanged.connect(self._sync_opt_to_cfg)
        self.sp_gci_n.valueChanged.connect(self._sync_opt_to_cfg)
        for sp in self._dom_spins.values():
            sp.valueChanged.connect(self._sync_opt_to_cfg)

    def _sync_opt_to_cfg(self, *_):
        """Write the current widget values into cfg.optimization. No-op while
        loading (so populating widgets from a file does not re-dirty it)."""
        if self._loading:
            return
        o = self.cfg.optimization
        o.zoi = {k: self.le_zoi[k].text()
                 for k in ("xmin", "xmax", "ymin", "ymax")}
        o.criterion_rmse = {q: le.text() for q, le in self._q_eps.items()}
        o.sizing_tol = {q: le.text() for q, le in self._dj_eps.items()}
        o.gci_finest = self.le_gci_finest.text()
        o.gci_ratio = self.le_gci_ratio.text()
        o.gci_min = self.le_gci_min.text()
        o.gci_n_meshes = int(self.sp_gci_n.value())
        o.caps = {d: le.text() for d, le in self._max.items()}
        o.margin_elems = int(self.sp_margin.value())
        o.centroid_step = self.le_grid_step.text()
        for attr, sp in self._dom_spins.items():
            setattr(o, attr, int(sp.value()))
        for attr, le in self._dom_texts.items():
            setattr(o, attr, le.text())
        o.ms_values = self.le_ms_values.text()
        o.ms_elem_size = self.le_ms_elem.text()
        self.changed.emit()

    def _load_opt_from_cfg(self):
        """Populate the widgets from cfg.optimization (called on construction
        and after a profile is opened via _rebind_cfg -> refresh_inputs)."""
        o = getattr(self.cfg, "optimization", None)
        if o is None:
            return
        self._loading = True
        try:
            for k in ("xmin", "xmax", "ymin", "ymax"):
                self.le_zoi[k].setText(str(o.zoi.get(k, "")))
            for q, le in self._q_eps.items():
                le.setText(str(o.criterion_rmse.get(q, "")))
            for q, le in self._dj_eps.items():
                le.setText(str(o.sizing_tol.get(q, "0.02")))
            self.le_gci_finest.setText(str(o.gci_finest))
            self.le_gci_ratio.setText(str(o.gci_ratio or "2"))
            self.le_gci_min.setText(str(o.gci_min))
            self.sp_gci_n.setValue(int(o.gci_n_meshes or 3))
            for d, le in self._max.items():
                le.setText(str(o.caps.get(d, "")))
            self.sp_margin.setValue(int(o.margin_elems or 0))
            self.le_grid_step.setText(str(o.centroid_step))
            for attr, sp in self._dom_spins.items():
                sp.setValue(int(getattr(o, attr, getattr(
                    OptimizationCfgDefaults, attr))))
            for attr, le in self._dom_texts.items():
                le.setText(str(getattr(o, attr, getattr(
                    OptimizationCfgDefaults, attr))))
            self.le_ms_values.setText(str(getattr(
                o, "ms_values", OptimizationCfgDefaults.ms_values)))
            self.le_ms_elem.setText(str(getattr(o, "ms_elem_size", "")))
        finally:
            self._loading = False

    def refresh_inputs(self):
        # The Inputs-from-model panel was removed; refreshing now just redraws
        # the preview from the current config (kept for _rebind_cfg callers).
        self._load_opt_from_cfg()
        self._draw_preview()

    def compute_initial(self):
        self.refresh_inputs()
        try:
            self._initial = self.compute_initial_dims()
        except Exception as e:
            QMessageBox.warning(self, "Initial domain",
                                "Cannot compute: %s" % e)
            return
        d = self._initial
        self.lbl_init.setText(
            "h_wp=%.4g  h_void=%.4g  l_wp=%.4g  l_void=%.4g"
            % (d.h_wp, d.h_void, d.l_wp, d.l_void))
        self._draw_preview()

    def _draw_preview(self, *_):
        """Reuse the Geometry tab's preview (tool + workpiece), then overlay the
        optimization elements: the measurement ROI (= initial Eulerian domain)
        with its evaluation points, and the max (cap) domain."""
        from matplotlib.patches import Rectangle
        try:
            self.preview.update_from_config(self.cfg)
        except Exception:
            log_swallowed("geometry preview update", level=logging.DEBUG)
            return
        ax = self.preview._ax
        try:
            inp = self.config_inputs()
        except Exception:
            self.preview._canvas.draw_idle()
            return
        ex0 = float(self.cfg.euler_position.x0)
        ey0 = float(self.cfg.euler_position.y0)
        tip_x, tip_y = inp["tip"]
        # refresh the read-only initial-dimension display
        try:
            di = self.compute_initial_dims()
            for d in _DIM_ORDER:
                self._init_lbl[d].setText("%.4g" % getattr(di, d))
        except Exception:
            pass

        def rect(x0, x1, y0, y1, **kw):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **kw))

        # max (cap) domain, in the Eulerian frame (drawn if all caps are set)
        cp = self.caps()
        if set(cp.keys()) >= set(_DIM_ORDER):
            rect(-cp["l_wp"] + ex0, cp["l_void"] + ex0, -cp["h_wp"] + ey0,
                 cp["h_void"] + ey0, fill=False, edgecolor="#7c3aed", lw=1.6,
                 ls="--", zorder=5)                          # max cap (purple)
        # ROI (Geometry tab, model output set for DIC/IRT) -- dotted green,
        # no points; the studies do NOT sample it.
        xmin, xmax, ymin, ymax = inp["roi"]
        rect(xmin, xmax, ymin, ymax, fill=False, edgecolor="#15803d",
             lw=1.3, ls=":", zorder=6)
        # ZOI (Optimization measurement zone) + measurement points at the
        # centroid step -- distinct colour; this is what the studies sample.
        zx0, zx1, zy0, zy1 = self.zoi()
        rect(zx0, zx1, zy0, zy1, fill=False, edgecolor="#c2410c",
             lw=1.6, zorder=7)
        step = self.grid_step()
        if step > 0:
            gx = np.arange(zx0, zx1 + 1e-9, step)
            gy = np.arange(zy0, zy1 + 1e-9, step)
            if gx.size and gy.size:
                XX, YY = np.meshgrid(gx, gy)
                ax.scatter(XX.ravel(), YY.ravel(), s=4, c="#c2410c",
                           alpha=0.6, zorder=7)
        ax.plot([tip_x], [tip_y], marker="v", color="k", markersize=7,
                zorder=8)
        ax.set_title("ROI (green dotted) \u2014 ZOI + points (orange) \u2014 "
                     "max cap (purple dashed)", fontsize=7)
        self.preview._canvas.draw_idle()

    def thresholds(self) -> dict:
        out = {}
        for q, le in self._q_eps.items():
            txt = le.text().strip().replace(",", ".")
            if txt:
                try:
                    out[q] = float(txt)
                except ValueError:
                    pass
        return out

    def thresholds_complete(self) -> bool:
        """True iff every field/force has a threshold (all are required)."""
        return set(self.thresholds().keys()) >= {q for (q, _f, _u) in _QUANTITIES}

    def quantity_field_map(self) -> dict:
        # all field-backed quantities are always used (Fc/Ff have no element
        # field -> excluded here, handled as forces)
        return {q: f for (q, f, _u) in _QUANTITIES if f is not None}

    def force_channels(self) -> dict:
        # Fc and Ff are always part of the criterion now
        return dict(_FORCE_CHANNELS)

    def caps(self) -> dict:
        out = {}
        for d, ce in self._max.items():
            txt = ce.text().strip().replace(",", ".")
            if txt:
                try:
                    out[d] = float(txt)
                except ValueError:
                    pass
        return out

    def _profile_name(self):
        try:
            n = self._profile_name_getter() if self._profile_name_getter else None
        except Exception:
            n = None
        return n or "Untitled"

    def _study_run_dir(self, workdir, prefix, study_config):
        """Timestamped per-study folder {profile}_{PREFIX}_{stamp} + config.json
        (via gui.core.run_output). Falls back to the flat working dir if the
        folder cannot be created."""
        from gui.core.run_output import create_study_dir
        try:
            return create_study_dir(workdir, self._profile_name(), prefix,
                                    study_config)
        except OSError:
            log_swallowed("creating study dir", level=logging.DEBUG)
            return Path(workdir)

    # -- Abaqus launcher (replicates the Sensitivity run mechanism) --------
    def _emit_log_tail(self, log_path, offset: int) -> int:
        """Emit whatever run_simul.py appended since `offset`; return the new
        offset. Same contract as SensitivityRunWorker._emit_log_tail: read from
        a byte position rather than re-reading, and latin-1 so a decode error
        cannot silently stop the live log."""
        try:
            with open(log_path, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
        except OSError:
            return offset          # not created yet, or already gone
        if chunk:
            self._log_ui(chunk.decode("latin-1", errors="replace"))
        return offset

    def _make_run_bundle(self, prefs, run_dir, cpus, prefix):
        import subprocess

        counter = {"i": 0}
        # Path of the LAST launched job's .sta and its name, readable by the
        # study's cost hook right after run_bundle returns (same thread).
        state = {"sta": None, "job": None, "filter_check": None}

        def run_bundle(cfg):
            i = counter["i"]; counter["i"] += 1
            job = "%s_run%03d" % (prefix, i)
            out_path = Path(run_dir) / ("%s.results.npz" % job)
            self._current_sta = Path(run_dir) / ("%s.sta" % job)
            state["sta"], state["job"] = self._current_sta, job
            state["filter_check"] = None
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                log_swallowed("removing stale bundle", level=logging.DEBUG)
            args = build_abaqus_args(
                prefs.abaqus_cmd, prefs.abaqus_script,
                cfg.to_params_dict(), {"cpus": cpus, "job_name": job})
            _ms = (float(getattr(cfg.step, "mass_scaling_factor", 1.0))
                   if getattr(cfg.step, "mass_scaling_enabled", False) else 1.0)
            self._log_ui("\n%s\n[%s] ms=%.4g wp=%.4g tool=%.4g | "
                         "h_wp=%.4g h_void=%.4g l_wp=%.4g l_void=%.4g\n%s\n"
                         % ("-" * 60, job, _ms,
                            float(cfg.elem_size), float(cfg.tool_elem_size),
                            cfg.euler_geometry.h_wp, cfg.euler_geometry.h_void,
                            cfg.euler_geometry.l_wp, cfg.euler_geometry.l_void,
                            "-" * 60))
            # Published BEFORE Popen so a Cancel landing during start-up can
            # still name the job to `abaqus terminate`.
            self._current_job = job
            self._current_abaqus_cmd = prefs.abaqus_cmd
            self._current_run_dir = run_dir
            try:
                proc = subprocess.Popen(
                    args, cwd=str(run_dir),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except Exception as e:
                self._log_ui("failed to start Abaqus: %s\n" % e)
                self._current_job = None
                return None
            self._current_proc = proc

            # Follow the run by TAILING THE SCRIPT'S LOG, not its stdout.
            # `abaqus cae noGUI=` runs run_simul.py in a separate kernel
            # process whose stdout reaches nobody (see M7 in _review/REVIEW.md),
            # so the old `iter(proc.stdout.readline, b"")` blocked until the
            # process exited. That is what made Cancel look like it did
            # nothing: the cancel test sat inside a loop no line ever woke.
            log_path = script_log_path(run_dir, job)
            offset = 0
            while proc.poll() is None:
                if self._cancel_evt.is_set():
                    break
                offset = self._emit_log_tail(log_path, offset)
                time.sleep(0.4)
            # The lines written since the last tick explain how the run ended.
            self._emit_log_tail(log_path, offset)
            # Whatever the launcher itself put on stdout (licence banner, a
            # fatal error before the script starts). Read once, after exit.
            try:
                rest = proc.stdout.read()
                if rest:
                    self._log_ui(rest.decode("cp1252", errors="replace"))
            except Exception:
                log_swallowed("reading Abaqus launcher output",
                              level=logging.DEBUG)
            proc.wait()
            self._current_proc = None
            self._current_job = None
            if self._cancel_evt.is_set() or proc.returncode != 0 \
                    or not out_path.exists():
                self._log_ui("[%s] no bundle (rc=%s)\n" % (job, proc.returncode))
                return None
            # Same post-run check as the Sensitivity runs (run_worker): a
            # no-op unless the Step tab requests the filter verification.
            try:
                from gui.core.filter_check import (
                    check_bundle, format_report, window_from_cfg)
                fc = check_bundle(out_path, window=window_from_cfg(cfg))
                state["filter_check"] = fc
                report = format_report(fc)
                if report:
                    self._log_ui(report)
            except Exception as e:
                self._log_ui("[%s] filter check failed: %s\n" % (job, e))
            try:
                return ResultsBundle.load(out_path)
            except Exception as e:
                self._log_ui("[%s] load failed: %s\n" % (job, e))
                return None

        run_bundle.state = state
        return run_bundle

    # -- worker callbacks --------------------------------------------------
    def _poll_sta(self):
        """Update the per-simulation progress bar from the current job's .sta."""
        p = self._current_sta
        if p is None or not Path(p).exists():
            return
        try:
            prog = parse_sta(p)
        except Exception:
            return
        frac = prog.fraction() if prog.is_ready() else None
        if frac is not None:
            self.sim_progress.setVisible(True)
            self.sim_progress.setValue(int(max(0.0, min(1.0, frac)) * 100))

    def _start_progress(self):
        self._current_sta = None
        self.sim_progress.setValue(0)
        self.sim_progress.setVisible(True)
        self._sim_timer.start()

    def _stop_progress(self):
        self._sim_timer.stop()
        self.sim_progress.setVisible(False)
        self._current_sta = None

    def _log_ui(self, text):
        """Append to the log from any thread.

        The emit is queued when it comes from a worker thread, so the widget
        is only ever touched by the GUI thread.
        """
        self._log_requested.emit(text.rstrip("\n"))

    def _append_log(self, text):
        self.log.appendPlainText(text)

    # ===================================================================
    # ZOI (measurement zone) — distinct from the ROI
    # ===================================================================
    def zoi(self):
        """The measurement ZOI bbox (xmin, xmax, ymin, ymax).

        DISTINCT from the ROI (Geometry tab, model output set). Empty panel
        fields fall back to the ROI, so a blank ZOI reproduces the previous
        ROI-as-ZOI behaviour. The studies sample THIS zone, not the ROI.
        """
        roi = self.config_inputs()["roi"]
        out = []
        for k, d in zip(("xmin", "xmax", "ymin", "ymax"), roi):
            out.append(self._float_or(self.le_zoi[k], d))
        return tuple(out)

    def _zoi_from_roi(self):
        for k, v in zip(("xmin", "xmax", "ymin", "ymax"),
                        self.config_inputs()["roi"]):
            self.le_zoi[k].setText("%.6g" % v)
        self._draw_preview()

    def _tolerances(self):
        """Relative per-quantity tolerances (empty field = quantity excluded)."""
        out = {}
        for q, le in self._dj_eps.items():
            txt = le.text().strip().replace(",", ".")
            if txt:
                try:
                    out[q] = float(txt)
                except ValueError:
                    pass
        return out

    def _gci_tolerances(self):
        """Relative GCI tolerances keyed by the GCI quantity names (the panel
        "force" entry applies to Fc and Ff)."""
        tol = self._tolerances()
        out = {q: tol[q] for q in ("EVF", "TEMP", "V1", "V2") if q in tol}
        if "force" in tol:
            out["Fc"] = tol["force"]
            out["Ff"] = tol["force"]
        return out

    def _dims_from_cfg(self):
        g = self.cfg.euler_geometry
        return DomainDims(h_wp=float(g.h_wp), h_void=float(g.h_void),
                          l_wp=float(g.l_wp), l_void=float(g.l_void))

    def _busy(self, on, msg="", color="#1d4ed8"):
        self.btn_ms.setEnabled(not on)
        self.btn_mesh.setEnabled(not on)
        self.btn_domain.setEnabled(not on)
        self.btn_checks.setEnabled((not on) and self._checks_available())
        self.btn_cancel.setEnabled(on)
        if msg:
            self.lbl_status.setStyleSheet("color: %s;" % color)
            self.lbl_status.setText(msg)

    # ===================================================================
    # Shared launch helpers (preserved verbatim)
    # ===================================================================
    def _open_working_dir(self):
        """Open the Preferences working directory in the file explorer."""
        prefs = self._prefs_getter() if self._prefs_getter else None
        wd = getattr(prefs, "default_workdir", None) if prefs else None
        if not wd:
            QMessageBox.information(self, "Working directory",
                                    "No working directory set in Preferences.")
            return
        p = Path(wd)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "Working directory",
                                    "Cannot open '%s': %s" % (wd, e))
                return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _validate_launch(self):
        """Shared pre-flight for a run: returns (prefs, workdir, cpus) or None
        (after showing a warning)."""
        prefs = self._prefs_getter() if self._prefs_getter else None
        if prefs is None:
            QMessageBox.warning(self, "Preferences",
                                "No preferences (Abaqus command/script).")
            return None
        problems = []
        if not Path(prefs.abaqus_cmd).exists():
            problems.append("Abaqus command not found: %s" % prefs.abaqus_cmd)
        if not Path(prefs.abaqus_script).exists():
            problems.append("Script not found: %s" % prefs.abaqus_script)
        wd = Path(prefs.default_workdir)
        try:
            wd.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            problems.append("Cannot create workdir '%s': %s" % (wd, e))
        if problems:
            QMessageBox.warning(self, "Cannot launch",
                                "\u2022 " + "\n\u2022 ".join(problems))
            return None
        cpus = int(self._cpus_getter()) if self._cpus_getter else 1
        return prefs, wd, cpus

    def _float_or(self, line_edit, default):
        txt = line_edit.text().strip().replace(",", ".")
        try:
            return float(txt)
        except (ValueError, TypeError):
            return default

    # ===================================================================
    # Shared settings: time window T, safeguards, domain-study integers
    # ===================================================================
    def window(self):
        """Time window T as (start, end) fractions of the simulated time.

        Raises ValueError unless 0 <= start < end <= 1."""
        a = self._float_or(self._dom_texts["window_start"], None)
        b = self._float_or(self._dom_texts["window_end"], None)
        if a is None or b is None or not (0.0 <= a < b <= 1.0):
            raise ValueError("the time window must satisfy 0 <= T start < "
                             "T end <= 1 (fractions of the simulated time)")
        return (a, b)

    def guard_settings(self) -> GuardSettings:
        """Run safeguards: G_K,max, G_HG,max (> 0) and the window T."""
        rk = self._float_or(self._dom_texts["rk_max"], None)
        rhg = self._float_or(self._dom_texts["rhg_max"], None)
        if rk is None or rhg is None or rk <= 0 or rhg <= 0:
            raise ValueError("G_K,max and G_HG,max must be positive numbers")
        return GuardSettings(rk_max=rk, rhg_max=rhg, window=self.window())

    def domain_settings(self) -> dict:
        """Integer settings of the domain study (step, n_max, n_hold, m)."""
        d = {attr: int(sp.value()) for attr, sp in self._dom_spins.items()}
        if d["dom_n_max"] < d["dom_m_ratios"] + 1:
            raise ValueError("n_max must be >= m + 1: the geometric-decay "
                             "test needs m + 1 comparisons")
        return d

    def _study_cfg_copy(self):
        """Deep copy of the current config for a study, with the history
        outputs the studies read forced on (PRESELECT carries ALLKE, ALLIE
        and, if hypothesis H2 holds, ALLAE; RF on the tool RP gives Fc, Ff).
        The tab's own config is never modified by a study."""
        cfg = copy.deepcopy(self.cfg)
        try:
            cfg.step.output.ho_preselect = True
            cfg.step.output.ho_rf_on_rp = True
        except AttributeError:
            log_swallowed("forcing the history outputs", level=logging.DEBUG)
        return cfg

    # ===================================================================
    # 0 - Mass-scaling factor: independence study (paper step 0, §5.3)
    # ===================================================================
    def ms_settings(self):
        """(ms values, element size) of step 0; raises ValueError."""
        values = parse_ms_values(self.le_ms_values.text())
        elem = self._float_or(self.le_ms_elem, float(self.cfg.elem_size))
        if elem is None or elem <= 0:
            raise ValueError("the element size of the ms study must be > 0")
        return values, float(elem)

    def _on_run_ms_independence(self):
        val = self._validate_launch()
        if val is None:
            return
        prefs, wd, cpus = val
        thr = self.thresholds()
        if not self.thresholds_complete():
            QMessageBox.warning(
                self, "Mass-scaling criterion",
                "Set the six absolute tolerances eps_q of step 2 "
                "(Vx, Vy, T, EVF, Fc, Ff): the ms study uses the same E_max.")
            return
        try:
            window = self.window()
            guards = self.guard_settings()
            ms_values, elem = self.ms_settings()
        except ValueError as e:
            QMessageBox.warning(self, "Mass-scaling study settings", str(e))
            return
        if not getattr(self.cfg.step, "output_filter_enabled", False):
            QMessageBox.warning(
                self, "Mass-scaling study",
                "Enable the output filter (Step tab): the filter and "
                "reverberation checks are safeguards of the ms study.")
            return
        base_cfg = self._study_cfg_copy()
        base_cfg.step.output_filter_verify = True
        zoi = self.zoi()
        dims = self._dims_from_cfg()
        study_cfg = {
            "zoi": {"xmin": zoi[0], "xmax": zoi[1],
                    "ymin": zoi[2], "ymax": zoi[3]},
            "elem_size": elem, "ms_values": list(ms_values),
            "grid_step": self.grid_step(), "thresholds_abs": thr,
            "window": list(window), "evf_threshold": 0.5,
            "rk_max": guards.rk_max, "rhg_max": guards.rhg_max,
            "domain_dims": {"h_wp": dims.h_wp, "h_void": dims.h_void,
                            "l_wp": dims.l_wp, "l_void": dims.l_void}}
        run_dir = self._study_run_dir(wd, "massscaling", study_cfg)
        run_bundle = self._make_run_bundle(prefs, run_dir, cpus, "ms")
        self._pending_ms_dir = run_dir

        def cost_fn(bundle, d, host_wall_s):
            return cost_record(bundle, run_bundle.state.get("sta"),
                               host_wall_s=host_wall_s, n_cpu=cpus,
                               dims=d, elem_size=elem)

        guard_fn_core = make_guard_fn(guards)

        def guard_fn(bundle):
            out = dict(guard_fn_core(bundle))
            why = guard_reasons(bundle, guards)
            fc = run_bundle.state.get("filter_check")
            if fc is None:
                why["filter"] = "filter check not run (no verification data)"
            out.update(filter_guards(fc))
            if why:
                self._log_ui("    safeguards not evaluable: %s"
                             % "; ".join("%s: %s" % kv for kv in why.items()))
            return out

        self._last_ms = None
        self._cancel_evt.clear()
        self.log.clear()
        self.tabs.setCurrentIndex(0)
        self._busy(True, "Mass-scaling study (independence)\u2026")
        self._log_ui("=" * 68)
        self._log_ui("MASS-SCALING FACTOR BY INDEPENDENCE (mesh, domain fixed)")
        self._log_ui("  ms: %s | mesh %.4g mm | domain h_wp=%.4g h_void=%.4g "
                     "l_wp=%.4g l_void=%.4g | T [%.3g, %.3g]"
                     % (", ".join("%g" % v for v in ms_values), elem,
                        dims.h_wp, dims.h_void, dims.l_wp, dims.l_void,
                        window[0], window[1]))
        self._log_ui("  eps_q: " + "  ".join("%s=%.4g" % kv
                                              for kv in sorted(thr.items())))
        self._log_ui("  safeguards: R_K < %.4g, R_HG < %.4g, outputs, filter "
                     "check, reverberation check"
                     % (guards.rk_max, guards.rhg_max))
        self._log_ui("=" * 68)
        self._ms_worker = MsIndependenceWorker(
            run_bundle=run_bundle, base_cfg=base_cfg, zoi=zoi,
            domain_dims=dims, grid_step=self.grid_step(), elem_size=elem,
            thresholds=thr, ms_values=ms_values, window=window,
            evf_threshold=0.5, guard_fn=guard_fn, cost_fn=cost_fn)
        self._ms_worker.progress.connect(self._on_ms_progress)
        self._ms_worker.finished_ok.connect(self._on_ms_done)
        self._ms_worker.failed.connect(self._on_fail)
        self._start_progress()
        self._ms_worker.start()

    def _on_ms_progress(self, ev):
        phase = ev.get("phase")
        if phase == "run":
            r = ev["record"]
            g = "  ".join("%s=%s%s" % (k, self._fmt(v), "" if ok else " FAIL")
                          for k, (v, ok) in sorted(r.guards.items()))
            cpu = self._fmt(getattr(r.cost, "c_cpu_s", None), "%.0f s") \
                if r.cost is not None else "n/a"
            self._log_ui("[run %d] ms=%g | %s | %s | C_CPU=%s"
                         % (r.index, ev.get("ms", float("nan")),
                            "job ok" if r.job_ok
                            else "JOB FAILED: %s" % r.error,
                            g or "no safeguard", cpu))
        elif phase == "comparison":
            c = ev["comparison"]
            errs = "  ".join("%s=%s" % (q, self._fmt(v))
                             for q, v in c.errors.items())
            off = getattr(c, "frame_offset_over_interval", float("nan"))
            self._log_ui("  ms %g->%g | %s | E_max=%s (%s) | %s%s%s"
                         % (c.ms_from, c.ms_to, errs,
                            self._fmt(c.e_max, "%.3g"), c.q_crit or "-",
                            "success" if c.success else "not independent",
                            "" if c.guards_ok else " (safeguards)",
                            "" if not math.isfinite(off) else
                            " | frame offset %.2g %% of the interval"
                            % (100.0 * off)))
        elif phase == "warning":
            self._log_ui("  [WARNING] %s" % ev.get("message", ""))

    def _on_ms_done(self, res):
        self._stop_progress()
        folder = getattr(self, "_pending_ms_dir", None)
        self._last_ms = (res, folder)
        self._busy(False)
        why = {
            "converged": "largest independent ms = %s (next value failed)",
            "upper_end": "every comparison passed: ms = %s is the LARGEST "
                         "TESTED value",
            "below_range": "the first comparison failed (%s): start the "
                           "sequence lower",
            "cancelled": "cancelled (ms reached: %s)",
        }.get(res.status, res.status + " (%s)")
        why = why % ("n/a" if res.retained is None else "%g" % res.retained)
        self._log_ui("=" * 68)
        self._log_ui("MASS-SCALING RESULT: %s | %d runs" % (why, res.n_runs))
        for w in res.warnings:
            self._log_ui("  [WARNING] %s" % w)
        if folder is not None:
            from gui.sensitivity.study_export import write_ms_exports
            try:
                paths = write_ms_exports(folder, res)
                self._log_ui("[EXPORT] %s -> %s"
                             % (", ".join(p.name for p in paths), folder))
            except Exception as e:
                self._log_ui("[EXPORT] failed: %s: %s"
                             % (type(e).__name__, e))
        if res.retained is not None:
            self._log_ui("  next: set ms = %g in the Step tab" % res.retained)
        self._refresh_convergence_view()
        ok = res.status == "converged"
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#15803d" if ok else "#b45309"))
        self.lbl_status.setText("Mass-scaling study \u2014 %s" % why)

    # ===================================================================
    # 4 - Eulerian domain sizing: sequential independence study (paper §4)
    # ===================================================================
    def _on_run_domain_independence(self):
        val = self._validate_launch()
        if val is None:
            return
        prefs, wd, cpus = val
        thr = self.thresholds()
        if not self.thresholds_complete():
            QMessageBox.warning(
                self, "Domain criterion",
                "Set the six absolute tolerances eps_q of step 2 "
                "(Vx, Vy, T, EVF, Fc, Ff): they define E_max for the domain "
                "study.")
            return
        try:
            window = self.window()
            guards = self.guard_settings()
            ds = self.domain_settings()
        except ValueError as e:
            QMessageBox.warning(self, "Domain study settings", str(e))
            return
        elem = float(self.cfg.elem_size)
        zoi = self.zoi()
        offset = self.euler_offset()
        margin = int(self.sp_margin.value())
        dims0 = self.compute_initial_dims()
        caps = self.caps()
        study_cfg = {
            "zoi": {"xmin": zoi[0], "xmax": zoi[1],
                    "ymin": zoi[2], "ymax": zoi[3]},
            "elem_size": elem, "margin_elems": margin,
            "euler_offset": list(offset),
            "grid_step": self.grid_step(), "thresholds_abs": thr,
            "window": list(window), "evf_threshold": 0.5,
            "step_elems": ds["dom_step_elems"], "n_max": ds["dom_n_max"],
            "n_hold": ds["dom_n_hold"], "m_ratios": ds["dom_m_ratios"],
            "rk_max": guards.rk_max, "rhg_max": guards.rhg_max,
            "caps": caps,
            "initial_dims": {"h_wp": dims0.h_wp, "h_void": dims0.h_void,
                             "l_wp": dims0.l_wp, "l_void": dims0.l_void}}
        run_dir = self._study_run_dir(wd, "domainsizing", study_cfg)
        run_bundle = self._make_run_bundle(prefs, run_dir, cpus, "domainsizing")
        self._pending_domain_dir = run_dir

        def cost_fn(bundle, dims, host_wall_s):
            return cost_record(bundle, run_bundle.state.get("sta"),
                               host_wall_s=host_wall_s, n_cpu=cpus,
                               dims=dims, elem_size=elem)

        guard_fn_core = make_guard_fn(guards)

        def guard_fn(bundle):
            out = guard_fn_core(bundle)
            why = guard_reasons(bundle, guards)
            if why:
                self._log_ui("    safeguards not evaluable: %s"
                             % "; ".join("%s: %s" % kv for kv in why.items()))
            return out

        self._last_domain_result = None
        self._last_checks = None
        self._cancel_evt.clear()
        self.log.clear()
        self.tabs.setCurrentIndex(0)
        self._busy(True, "Domain sizing (independence)\u2026")
        self._log_ui("=" * 68)
        self._log_ui("DOMAIN SIZING BY SEQUENTIAL INDEPENDENCE (ZOI fixed)")
        self._log_ui("  ZOI  x[%.4g,%.4g] y[%.4g,%.4g]" % zoi)
        self._log_ui("  initial = ZOI + %d elem: h_wp=%.4g h_void=%.4g "
                     "l_wp=%.4g l_void=%.4g"
                     % (margin, dims0.h_wp, dims0.h_void, dims0.l_wp,
                        dims0.l_void))
        self._log_ui("  mesh %.4g mm (held) | step %d elem | n_max %d | "
                     "n_hold %d | m %d | T [%.3g, %.3g]"
                     % (elem, ds["dom_step_elems"], ds["dom_n_max"],
                        ds["dom_n_hold"], ds["dom_m_ratios"], window[0],
                        window[1]))
        self._log_ui("  eps_q: " + "  ".join("%s=%.4g" % kv
                                              for kv in sorted(thr.items())))
        self._log_ui("  safeguards: R_K < %.4g, R_HG < %.4g, outputs present"
                     % (guards.rk_max, guards.rhg_max))
        self._log_ui("=" * 68)
        self._di_worker = DomainIndependenceWorker(
            run_bundle=run_bundle, base_cfg=self._study_cfg_copy(), zoi=zoi,
            initial_dims=dims0, grid_step=self.grid_step(), elem_size=elem,
            thresholds=thr, window=window, evf_threshold=0.5,
            step_elems=ds["dom_step_elems"], n_max=ds["dom_n_max"],
            n_hold=ds["dom_n_hold"], m_ratios=ds["dom_m_ratios"],
            caps=caps, margin_elems=margin, offset=offset,
            guard_fn=guard_fn, cost_fn=cost_fn)
        self._di_worker.progress.connect(self._on_di_progress)
        self._di_worker.finished_ok.connect(self._on_di_done)
        self._di_worker.failed.connect(self._on_fail)
        self._start_progress()
        self._di_worker.start()

    @staticmethod
    def _fmt(v, fmt="%.4g"):
        return "n/a" if v is None or (isinstance(v, float) and
                                      not math.isfinite(v)) else fmt % v

    def _on_di_progress(self, ev):
        phase = ev.get("phase")
        if phase == "run":
            r = ev["record"]
            d = r.dims
            g = "  ".join("%s=%s%s" % (k, self._fmt(v), "" if ok else " FAIL")
                          for k, (v, ok) in sorted(r.guards.items()))
            c = r.cost
            cpu = self._fmt(getattr(c, "c_cpu_s", None), "%.0f s") \
                if c is not None else "n/a"
            self._log_ui(
                "[run %d] h_wp=%.4g h_void=%.4g l_wp=%.4g l_void=%.4g | %s | "
                "%s | C_CPU=%s%s"
                % (r.index, d["h_wp"], d["h_void"], d["l_wp"], d["l_void"],
                   "job ok" if r.job_ok else "JOB FAILED: %s" % r.error,
                   g or "no safeguard", cpu,
                   "  [diag/h=%.1f: warning]" % r.diagonal_ratio
                   if r.diagonal_warning else ""))
        elif phase == "comparison":
            c = ev["comparison"]
            errs = "  ".join("%s=%s" % (q, self._fmt(v))
                             for q, v in c.errors.items())
            self._log_ui("  %s %.4g->%.4g | %s | E_max=%s (%s) | %s%s | %s"
                         % (c.dimension, c.value_from, c.value_to, errs,
                            self._fmt(c.e_max, "%.3g"), c.q_crit or "-",
                            "success" if c.success else "not independent",
                            "" if c.guards_ok else " (safeguards)",
                            c.mode))
        elif phase == "dimension":
            r = ev["result"]
            self._log_ui("  => %s retained %.4g mm (%s%s)"
                         % (r.name, r.retained, r.status,
                            ", q_crit %s, criterion %s"
                            % (r.q_crit, self._fmt(r.criterion, "%.3g"))
                            if r.q_crit else ""))
        elif phase == "warning":
            self._log_ui("  [WARNING] %s" % ev.get("message", ""))

    def _on_di_done(self, res):
        self._stop_progress()
        self._last_domain_result = res
        self._last_domain_dir = getattr(self, "_pending_domain_dir", None)
        self._busy(False)
        why = {
            "converged": "every dimension independent",
            "partial": "at least one dimension NOT converged (largest tested "
                       "value kept)",
            "zoi_outside": "the ZOI is not inside the initial domain with the "
                           "margin",
            "cancelled": "cancelled",
        }.get(res.status, res.status or "stopped")
        d = res.final
        self._log_ui("=" * 68)
        self._log_ui("RESULT: %s" % why)
        for name, r in res.per_dimension.items():
            self._log_ui("  %-6s %.4g -> %.4g mm  [%s]"
                         % (name, r.initial, r.retained, r.status))
        self._log_ui("  final: h_wp=%.4g h_void=%.4g l_wp=%.4g l_void=%.4g | "
                     "%d runs" % (d.h_wp, d.h_void, d.l_wp, d.l_void,
                                  res.n_runs))
        for w in res.warnings:
            self._log_ui("  [WARNING] %s" % w)
        self._write_domain_exports()
        self._refresh_convergence_view()
        if self._checks_available():
            self._log_ui("  next: 'Run interaction checks' (paper \u00a75.7)")
        ok = res.status == "converged"
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#15803d" if ok else "#b45309"))
        self.lbl_status.setText("Domain sizing \u2014 %s" % why)

    # ===================================================================
    # Convergence view: table, plots, exports (report, Part B, T7, T8, T11)
    # ===================================================================
    def _t1(self):
        """Uncut chip thickness t1 = wp y0 - tool y0 (paper Eq. 24)."""
        try:
            return float(self.config_inputs()["t1"]) or None
        except Exception:
            return None

    def _ms_factor(self):
        st = self.cfg.step
        return (float(st.mass_scaling_factor)
                if getattr(st, "mass_scaling_enabled", False) else 1.0)

    def _add_table_row(self, values):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate(values):
            self.table.setItem(r, c, QTableWidgetItem(
                "" if v is None else str(v)))

    def _refresh_convergence_view(self):
        """Rebuild the table and the three plots from the last results."""
        self.table.setRowCount(0)
        if self._last_ms is not None:
            mres = self._last_ms[0]
            mruns = {r.index: r for r in mres.runs}
            for c in mres.comparisons:
                rt = mruns.get(c.run_to)
                cost = getattr(getattr(rt, "cost", None), "c_cpu_s", None)
                self._add_table_row([
                    "ms", "ms", "%g" % c.ms_from, "%g" % c.ms_to,
                    self._fmt(c.e_max, "%.3g"), c.q_crit,
                    "ok" if c.guards_ok else "FAIL",
                    "independent" if c.success else "not independent",
                    "", self._fmt(cost, "%.0f")])
            self._add_table_row([
                "ms", "ms", "", "n/a" if mres.retained is None
                else "%g" % mres.retained, "", "", "", "RETAINED",
                mres.status, ""])
        res = self._last_domain_result
        runs = {r.index: r for r in res.runs} if res is not None else {}
        if res is not None:
            for name, d in res.per_dimension.items():
                for c in d.comparisons:
                    rf = runs.get(c.run_from)
                    cost = getattr(getattr(rf, "cost", None), "c_cpu_s", None)
                    self._add_table_row([
                        "domain", name, "%.4g" % c.value_from,
                        "%.4g" % c.value_to, self._fmt(c.e_max, "%.3g"),
                        c.q_crit, "ok" if c.guards_ok else "FAIL",
                        "independent" if c.success else "not independent",
                        c.mode, self._fmt(cost, "%.0f")])
                self._add_table_row([
                    "domain", name, "%.4g" % d.initial, "%.4g" % d.retained,
                    self._fmt(d.criterion, "%.3g"), d.q_crit, "",
                    "RETAINED", d.status, ""])
        if self._last_gci is not None:
            gres = self._last_gci[0]
            for q, g in gres.per_quantity.items():
                self._add_table_row([
                    "GCI", q, "", "", self._fmt(g.gci_fine, "%.3g"), "",
                    "", "reliable" if g.reliable else "unreliable",
                    "p=%s" % self._fmt(g.p, "%.3g"), ""])
        if self._last_checks is not None:
            for c in self._last_checks.checks:
                self._add_table_row([
                    "check", c.name, "", "", self._fmt(c.e_max, "%.3g"),
                    c.q_crit, {True: "ok", False: "FAIL",
                               None: ""}[c.safeguards_ok],
                    {True: "passed", False: "FAILED",
                     None: "not evaluable"}[c.passed], "", ""])
        self._plot_convergence()

    def _plot_convergence(self):
        axm = self._ax_ms
        axd, axg, axc = self._ax_domain, self._ax_gci, self._ax_cost
        for ax in (axm, axd, axg, axc):
            ax.clear()
        if self._last_ms is not None:
            mres = self._last_ms[0]
            xs = [c.ms_to for c in mres.comparisons]
            ys = [c.e_max if math.isfinite(c.e_max) else float("nan")
                  for c in mres.comparisons]
            if xs:
                axm.plot(xs, ys, marker="o", lw=1.2, color="#1d4ed8")
                for c in mres.comparisons:
                    if not c.success:
                        axm.plot([c.ms_to], [c.e_max if math.isfinite(
                            c.e_max) else float("nan")], marker="x", ms=8,
                            color="#b91c1c", ls="none")
                if mres.retained is not None:
                    axm.axvline(mres.retained, ls=":", lw=1.0,
                                color="#15803d")
            axm.axhline(1.0, ls="--", lw=1.0, color="#b91c1c")
            axm.set_xscale("log")
            axm.set_yscale("log")
            axm.set_xlabel("ms_k", fontsize=7)
            axm.set_ylabel("E_max(ms_k, ms_k-1)", fontsize=7)
        res = self._last_domain_result
        if res is not None:
            runs = {r.index: r for r in res.runs}
            pts = []
            for name, d in res.per_dimension.items():
                xs = [c.value_from for c in d.comparisons]
                ys = [c.e_max if math.isfinite(c.e_max) else float("nan")
                      for c in d.comparisons]
                if xs:
                    axd.plot(xs, ys, marker="o", lw=1.2, label=name)
                    axd.plot([d.retained], [d.criterion if math.isfinite(
                        d.criterion) else float("nan")], marker="*",
                        ms=10, color=axd.lines[-1].get_color())
                for c in d.comparisons:
                    cr = getattr(runs.get(c.run_from), "cost", None)
                    if math.isfinite(c.e_max) and cr is not None:
                        pts.append((getattr(cr, "c_cpu_s", None),
                                    getattr(cr, "n_elem_euler", None),
                                    c.e_max))
            axd.axhline(1.0, ls="--", lw=1.0, color="#b91c1c")
            axd.set_yscale("log")
            axd.set_xlabel("dimension value p_j [mm]", fontsize=7)
            axd.set_ylabel("E_max(p_j, p_j+1)", fontsize=7)
            if axd.get_legend_handles_labels()[0]:
                axd.legend(fontsize=6)
            # Cost axis: C_CPU (Eq. 11) when every point has it, else the
            # Eulerian element count as a proxy (stated on the axis).
            use_cpu = bool(pts) and all(p[0] is not None for p in pts)
            pts2 = [((p[0] if use_cpu else p[1]), p[2]) for p in pts
                    if (p[0] if use_cpu else p[1]) is not None]
            if pts2:
                from gui.sensitivity.study_export import pareto_flags
                flags = pareto_flags(pts2)
                for (cx, ey), f in zip(pts2, flags):
                    axc.plot([cx], [ey], marker="o" if f else "x",
                             color="#1d4ed8" if f else "#9ca3af",
                             ls="none")
                axc.axhline(1.0, ls="--", lw=1.0, color="#b91c1c")
                axc.set_yscale("log")
                axc.set_xlabel("C_CPU of the candidate [s]" if use_cpu else
                               "N_elem of the candidate (C_CPU unavailable)",
                               fontsize=7)
                axc.set_ylabel("E_max", fontsize=7)
        if self._last_gci is not None:
            gres = self._last_gci[0]
            for q, g in gres.per_quantity.items():
                ref = g.f_extrapolated if g.reliable else g.f_fine
                hs = [h for h in gres.sizes if q in gres.scalars.get(h, {})]
                ys = []
                for h in hs:
                    v = gres.scalars[h][q]
                    ys.append(abs(v / ref - 1.0) if (
                        v is not None and ref and math.isfinite(ref))
                        else float("nan"))
                if hs:
                    axg.plot(hs, ys, marker="s", lw=1.0, label=q)
            axg.set_xscale("log")
            axg.set_xlabel("h [mm]", fontsize=7)
            axg.set_ylabel("|f_q(h)/f_ref - 1|", fontsize=7)
            if axg.get_legend_handles_labels()[0]:
                axg.legend(fontsize=6)
        for ax, title in ((axm, "Mass scaling"), (axd, "Domain study"),
                          (axg, "Mesh GCI"),
                          (axc, "Cost \u2013 E_max")):
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6)
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def _write_domain_exports(self):
        """Write the domain-study files (and the checks, if any) into the
        domain study folder; logs the written names."""
        res, folder = self._last_domain_result, self._last_domain_dir
        if res is None or folder is None:
            return []
        from gui.sensitivity.study_export import write_domain_exports
        try:
            paths = write_domain_exports(
                folder, res, self._t1(), res.settings.get("elem_size"),
                self._ms_factor(), self._last_checks)
        except Exception as e:
            self._log_ui("[EXPORT] failed: %s: %s" % (type(e).__name__, e))
            return []
        self._log_ui("[EXPORT] %s -> %s" % (", ".join(p.name for p in paths),
                                            folder))
        return paths

    def ms_lower_for_checks(self):
        """The ms value before the current ms* in the last ms study, or None
        (the check then uses ms*/2)."""
        if self._last_ms is None:
            return None
        values = list(self._last_ms[0].ms_values)
        ms = self._ms_factor()
        for a, b in zip(values[:-1], values[1:]):
            if math.isclose(b, ms, rel_tol=1e-9):
                return float(a)
        return None

    def _checks_available(self) -> bool:
        res = self._last_domain_result
        return bool(res is not None and res.runs and
                    res.status in ("converged", "partial"))

    # ===================================================================
    # 6 - Interaction checks (paper §5.7, report T9)
    # ===================================================================
    def _on_run_interaction_checks(self):
        study = self._last_domain_result
        if not self._checks_available():
            QMessageBox.warning(self, "Interaction checks",
                                "Run a domain study first.")
            return
        val = self._validate_launch()
        if val is None:
            return
        prefs, wd, cpus = val
        try:
            guards = self.guard_settings()
            window = tuple(study.settings["window"])
        except ValueError as e:
            QMessageBox.warning(self, "Interaction checks", str(e))
            return
        h_star = float(study.settings["elem_size"])
        finest = self._float_or(self.le_gci_finest, h_star)
        gci_plan = {
            "zoi": tuple(study.settings["zoi"]),
            "grid_step": float(study.settings["grid_step"]),
            "finest_elem_size": finest,
            "ratio": self._float_or(self.le_gci_ratio, 2.0),
            "n_meshes": int(self.sp_gci_n.value()),
            "min_elem_size": self._float_or(self.le_gci_min, None),
            "field_vars": ("EVF", "TEMP", "V1", "V2"), "window": window,
            "evf_threshold": 0.5}
        gci_tol = self._gci_tolerances()
        folder = self._last_domain_dir or wd
        run_bundle = self._make_run_bundle(prefs, folder, cpus, "checks")

        def cost_fn(bundle, dims, host_wall_s):
            return cost_record(bundle, run_bundle.state.get("sta"),
                               host_wall_s=host_wall_s, n_cpu=cpus,
                               dims=dims, elem_size=h_star)

        from gui.sensitivity.interaction_checks_worker import (
            InteractionChecksWorker)
        from gui.sensitivity.run_record import RecordingRunner
        self._cancel_evt.clear()
        self.tabs.setCurrentIndex(0)
        self._busy(True, "Interaction checks\u2026")
        self._log_ui("=" * 68)
        self._log_ui("INTERACTION CHECKS on h*=%.4g mm, D*: h_wp=%.4g "
                     "h_void=%.4g l_wp=%.4g l_void=%.4g"
                     % (h_star, study.final.h_wp, study.final.h_void,
                        study.final.l_wp, study.final.l_void))
        self._log_ui("  GCI plan on D*: finest %.4g | ratio %.3g | n %d"
                     % (finest, gci_plan["ratio"], gci_plan["n_meshes"]))
        self._log_ui("=" * 68)
        # Check ms_at_point: ms* against the value before it in the last ms
        # study (ms*/2 without one), with the ms study's safeguards (filter
        # and reverberation checks) when the output filter is on.
        base_cfg = self._study_cfg_copy()
        ms_lower = self.ms_lower_for_checks()
        guard_core = make_guard_fn(guards)
        ms_guard_fn = None
        if getattr(base_cfg.step, "output_filter_enabled", False):
            base_cfg.step.output_filter_verify = True

            def ms_guard_fn(bundle):
                out = dict(guard_core(bundle))
                out.update(filter_guards(run_bundle.state.get("filter_check")))
                return out
        else:
            self._log_ui("  ms_at_point: output filter off, the filter and "
                         "reverberation safeguards are not evaluated")
        self._log_ui("  ms_at_point: ms* = %g against ms = %s"
                     % (self._ms_factor(), "%g" % ms_lower
                        if ms_lower is not None else "ms*/2"))
        self._checks_worker = InteractionChecksWorker(
            run_bundle=run_bundle, base_cfg=base_cfg,
            study=study, h_star=h_star, gci_plan=gci_plan,
            gci_tolerances=gci_tol, guard_fn=guard_core,
            cost_fn=cost_fn, ms_lower=ms_lower, ms_guard_fn=ms_guard_fn,
            gci_runner_factory=lambda rb: RecordingRunner(
                rb, n_cpu=cpus, guard_settings=guards))
        self._checks_worker.progress.connect(self._on_checks_progress)
        self._checks_worker.finished_ok.connect(self._on_checks_done)
        self._checks_worker.failed.connect(self._on_fail)
        self._start_progress()
        self._checks_worker.start()

    def _on_checks_progress(self, ev):
        phase = ev.get("phase")
        if phase in ("run", "warning"):
            self._on_di_progress(ev)
        elif phase == "gci":
            self._on_mesh_progress(dict(ev, phase="mesh_gci"))
        elif phase == "check":
            c = ev["check"]
            self._log_ui("  [%s] %s \u2014 %s%s"
                         % (c.name, {True: "PASSED", False: "FAILED",
                                     None: "NOT EVALUABLE"}[c.passed],
                            c.conclusion,
                            "".join("\n      warning: %s" % w
                                    for w in c.warnings)))

    def _on_checks_done(self, res):
        self._stop_progress()
        self._last_checks = res
        self._busy(False)
        why = {"accepted": "model ACCEPTED (all checks passed)",
               "rejected": "model REJECTED (at least one check failed)",
               "incomplete": "INCOMPLETE (a check could not be evaluated)",
               "cancelled": "cancelled"}.get(res.status, res.status)
        self._log_ui("=" * 68)
        self._log_ui("INTERACTION CHECKS: %s" % why)
        for c in res.checks:
            if c.details.get("action"):
                self._log_ui("  ACTION (%s): %s" % (c.name, c.details["action"]))
        self._write_domain_exports()
        self._refresh_convergence_view()
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#15803d" if res.status == "accepted"
                            else "#b45309"))
        self.lbl_status.setText("Interaction checks \u2014 %s" % why)

    # ===================================================================
    # 3 - Mesh convergence by GCI / Richardson (fixed domain)
    # ===================================================================
    def _on_run_mesh_gci(self):
        val = self._validate_launch()
        if val is None:
            return
        prefs, wd, cpus = val
        finest = self._float_or(self.le_gci_finest, float(self.cfg.elem_size))
        ratio = self._float_or(self.le_gci_ratio, 2.0)
        nmesh = int(self.sp_gci_n.value())
        minh = self._float_or(self.le_gci_min, None)
        gci_tol = self._gci_tolerances()
        dims = self._dims_from_cfg()
        zoi = self.zoi()
        try:
            window = self.window()
        except ValueError as e:
            QMessageBox.warning(self, "Time window", str(e))
            return
        study_cfg = {
            "zoi": {"xmin": zoi[0], "xmax": zoi[1],
                    "ymin": zoi[2], "ymax": zoi[3]},
            "window": list(window),
            "finest_elem_size": finest, "ratio": ratio, "n_meshes": nmesh,
            "min_elem_size": minh, "grid_step": self.grid_step(),
            "tolerances": gci_tol, "field_vars": ["EVF", "TEMP", "V1", "V2"],
            "evf_threshold": 0.5,
            "domain_dims": {"h_wp": dims.h_wp, "h_void": dims.h_void,
                            "l_wp": dims.l_wp, "l_void": dims.l_void}}
        run_dir = self._study_run_dir(wd, "GCI", study_cfg)
        run_bundle = self._make_run_bundle(prefs, run_dir, cpus, "GCI")
        # T10: every GCI run records its cost and safeguards (mesh_gci has no
        # hook of its own); the records feed gci_meshes.csv (paper Table 8).
        from gui.sensitivity.run_record import RecordingRunner
        try:
            guards = self.guard_settings()
        except ValueError as e:
            QMessageBox.warning(self, "Safeguards", str(e))
            return
        recorder = RecordingRunner(run_bundle, n_cpu=cpus,
                                   guard_settings=guards)
        self._pending_gci = (recorder, gci_tol, run_dir)
        self._cancel_evt.clear()
        self.log.clear()
        self.tabs.setCurrentIndex(0)
        self._busy(True, "Mesh convergence (GCI)\u2026")
        self._log_ui("=" * 68)
        self._log_ui("MESH CONVERGENCE (GCI / Richardson) on a fixed domain")
        self._log_ui("  finest %.4g mm | ratio %.3g | n %d | floor %s | "
                     "T [%.3g, %.3g]"
                     % (finest, ratio, nmesh,
                        "n/a" if minh is None else "%.4g" % minh,
                        window[0], window[1]))
        self._log_ui("=" * 68)
        # A deep copy: run_mesh_gci sets elem_size and the domain on the cfg
        # it receives (mesh_gci.py:337-341); given self.cfg it used to leave
        # the user's model at the coarsest element size after the study.
        self._mesh_worker = MeshGciWorker(
            run_bundle=recorder, base_cfg=self._study_cfg_copy(), zoi=zoi,
            domain_dims=dims,
            grid_step=self.grid_step(), finest_elem_size=finest, ratio=ratio,
            n_meshes=nmesh, tolerances=(gci_tol or None),
            field_vars=("EVF", "TEMP", "V1", "V2"), window=window,
            evf_threshold=0.5, min_elem_size=minh)
        self._mesh_worker.progress.connect(self._on_mesh_progress)
        self._mesh_worker.finished_ok.connect(self._on_mesh_done)
        self._mesh_worker.failed.connect(self._on_fail)
        self._start_progress()
        self._mesh_worker.start()

    def _on_mesh_progress(self, ev):
        if ev.get("phase") != "mesh_gci":
            return
        sc = ev.get("scalars", {})
        self._log_ui("  h=%.4g -> %s" % (
            ev.get("elem_size", 0),
            "  ".join("%s=%.4g" % (q, v) for q, v in sorted(sc.items())
                      if v is not None)))

    def _on_mesh_done(self, res):
        self._stop_progress()
        self._busy(False)
        self._log_ui("=" * 68)
        self._log_ui("MESH GCI RESULT (%s)" % (res.stopped_by or ""))
        for q, g in sorted(res.per_quantity.items()):
            self._log_ui(
                "  %-5s p=%.3g  f_ext=%.4g  GCI=%.3g%%  asym=%.3g%s%s"
                % (q, g.p, g.f_extrapolated, 100.0 * g.gci_fine,
                   g.asymptotic_ratio,
                   "" if g.monotonic else "  (non-monotonic)",
                   "" if g.reliable else "  [extrapolation unreliable "
                   "\u2014 converged/noisy, ref = finest]"))
        self._log_ui("  in asymptotic range: %s" % res.in_asymptotic_range)
        rec = res.recommended_size
        self._log_ui("  recommended element size: %s"
                     % ("none within tolerance" if rec is None
                        else "%.4g mm" % rec))
        recorder, tol, folder = getattr(self, "_pending_gci",
                                        (None, {}, None))
        calls = list(getattr(recorder, "records", []) or [])
        for c in calls:
            g = "  ".join("%s=%s%s" % (k, self._fmt(v), "" if ok else " FAIL")
                          for k, (v, ok) in sorted(c.guards.items()))
            self._log_ui("  h=%.4g | C_CPU=%s | N_elem=%s | %s"
                         % (c.elem_size, self._fmt(c.cost.c_cpu_s, "%.0f s"),
                            c.cost.n_elem_euler, g or "no safeguard"))
        self._last_gci = (res, calls, tol, folder)
        if folder is not None:
            from gui.sensitivity.study_export import write_gci_exports
            try:
                paths = write_gci_exports(folder, res, calls, tol)
                self._log_ui("[EXPORT] %s -> %s"
                             % (", ".join(p.name for p in paths), folder))
            except Exception as e:
                self._log_ui("[EXPORT] failed: %s: %s"
                             % (type(e).__name__, e))
        self._refresh_convergence_view()
        ok = rec is not None and res.in_asymptotic_range
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#15803d" if ok else "#b45309"))
        self.lbl_status.setText(
            "Mesh GCI \u2014 recommended %s"
            % ("n/a" if rec is None else "%.4g mm" % rec))

    # ===================================================================
    # Cancel / failure
    # ===================================================================
    def _on_cancel(self):
        """Stop the study AND the run currently in flight, without freezing.

        Same two stages as SensitivityRunWorker.cancel, in the same order:
          1. ``abaqus terminate job=<name>`` -- the clean route: it stops the
             solver AND releases the licence tokens.
          2. kill the process tree -- the fallback, for when Abaqus does not
             answer (no .cid yet, job already finishing, hung solver). This
             leaves the tokens checked out, hence the ordering.

        Both stages run OFF the GUI thread (see gui/core/async_call), and the
        pause between them is a QTimer, not a wait(). Doing it inline cost up
        to 30 s of frozen window -- finding M3.

        Before any of this, cancelling only called ``proc.terminate()`` on the
        ``abaqus cae`` launcher: the solver behind it survived as an orphan
        (M1), the licence stayed checked out, and the test itself sat in a loop
        reading a stdout that never produces a line (M7).

        The run interrupted here is reported as failed -- run_bundle returns
        None on a set cancel flag, which the studies already treat as "no
        usable result" -- so a half-written bundle is never read as data.
        """
        for attr in ("_di_worker", "_mesh_worker", "_checks_worker",
                     "_ms_worker"):
            w = getattr(self, attr, None)
            if w is not None and w.isRunning():
                w.cancel()
        self._cancel_evt.set()
        self.lbl_status.setText("Cancelling the current run\u2026")

        job, proc = self._current_job, self._current_proc
        cmd, run_dir = self._current_abaqus_cmd, self._current_run_dir
        if not (job and cmd):
            return              # nothing started yet: the flag is enough
        self._log_ui("[CANCEL] asking Abaqus to terminate job %s" % job)
        run_async(lambda: abaqus_terminate_job(cmd, job, run_dir),
                  lambda ok: self._after_terminate(bool(ok), proc), self)

    def _after_terminate(self, accepted: bool, proc):
        """Back on the GUI thread once Abaqus has answered (or not)."""
        if accepted:
            # Let the solver unwind. A timer, so the window stays alive.
            QTimer.singleShot(10000, lambda: self._kill_if_alive(proc))
        else:
            self._kill_if_alive(proc)

    def _kill_if_alive(self, proc):
        """Force-kill the tree, unless the run has already exited.

        `returncode` is read rather than poll() called: the study thread is
        polling the same Popen, and two threads reaping one child race for its
        exit status. The attribute is set by whichever poll() saw it exit.
        """
        if proc is None or proc.returncode is not None:
            return
        self._log_ui("[CANCEL] Abaqus did not answer; killing the process "
                     "tree (licence tokens stay checked out)")
        run_async(lambda: kill_process_tree_by_pid(proc.pid),
                  lambda _ok: None, self)

    def _on_fail(self, msg):
        self._stop_progress()
        self._busy(False)
        self.lbl_status.setStyleSheet("color: #b91c1c;")
        self.lbl_status.setText("Study failed: %s" % msg)
        self._log_ui("ERROR: %s" % msg)
