# -*- coding: utf-8 -*-
"""
Optimization > Model tab: size the CEL model in four steps.

Three studies, all measured in the ZOI (the Optimization measurement zone,
distinct from the output ROI of the Geometry tab), then the final checks:

  * the mass-scaling factor by an independence study on a fixed mesh and
    domain (gui.sensitivity.ms_independence): increasing ms compared
    successively against the same ABSOLUTE tolerances eps_q as the domain
    study, with the filter and reverberation checks as extra safeguards;
  * mesh convergence by Richardson extrapolation / GCI on a fixed domain
    (gui.sensitivity.mesh_gci): the recommended size is the coarsest
    within the same ABSOLUTE tolerances eps_q of the reference (the relative
    GCI is reported, not used to select);
  * Eulerian-domain sizing by a sequential independence study
    (gui.sensitivity.domain_independence): each dimension grown by a
    constant step from the initial domain = ZOI + margin, successive runs
    compared by the mean absolute difference (paper Eq. 5, 7) against
    ABSOLUTE tolerances eps_q, residual influence bounded by a geometric tail
    (fallback: successive criterion), run safeguards R_K, R_HG and outputs.

All studies share the time window T. Each candidate is one Abaqus run
(run_simul) launched here, replicating the Sensitivity tab's run mechanism;
the studies receive a deep copy of the current config, which they never
modify. Finished runs are reused by parameter content (resume, load), each
study leaves a record of its result for the current model, and the whole
chain can run in one go: see gui/tabs/model_steps.py. The study cores are
unit-tested elsewhere.
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
from gui.core.remote_exec import (
    RemoteProcess, is_remote, launch_problems, submit_remote)
from gui.results.reader import ResultsBundle
from gui.sensitivity.study_specs import (
    CHECKS_CONFIG, DIM_KEYS, ZOI_KEYS, grid_set_of_spec, ms_of,
    write_checks_config, zoi_tuple)
from gui.sensitivity.study_state import STEPS, in_model
from gui.tabs.model_steps import ModelStepsMixin
from gui.widgets.collapsible import CollapsibleSection
from gui.widgets.geometry_preview import GeometryPreview


# Quantity -> bundle element-field name (velocity components V1/V2 are written
# per element by run_simul; T=TEMP; PEEQ/EVF as-is). Units are informative.
_QUANTITIES = [
    ("Vx", "V1", "mm/s"),
    ("Vy", "V2", "mm/s"),
    ("T",  "TEMP", "K"),
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
# Domain-study settings: (attribute of cfg.optimization, label, min, max,
# tooltip). Defaults: OptimizationCfg (decided by the author on 2026-10-02).
_DOM_SPINS = [
    ("dom_step_elems", "growth step \u0394 [elements]", 1, 1000,
     "Each side of the domain grows by this many elements between two\n"
     "runs. Default 10. A larger step needs fewer runs but gives a\n"
     "coarser final size."),
    ("dom_n_max", "max comparisons per side", 2, 50,
     "The growth of one side stops after this many comparisons, converged\n"
     "or not. Default 8 (with the default step: up to 80 elements of\n"
     "growth per side)."),
    ("dom_n_hold", "passes in a row (fallback rule)", 1, 10,
     "When the decay test cannot conclude, a side is converged after this\n"
     "many successive passes (E_max < 1). Default 1."),
    ("dom_m_ratios", "ratios in the decay test (m)", 1, 10,
     "Number of successive E_max ratios used to bound the residual\n"
     "influence by a geometric tail. Must be at most the max comparisons\n"
     "minus 1. Default 2."),
]
# Shared text settings: (attribute, label, tooltip).
_DOM_TEXTS = [
    ("window_start", "start", "Start of the time window T, as a fraction "
                              "of the simulated time (default 0.3)"),
    ("window_end", "end", "End of the time window T, as a fraction of the "
                          "simulated time (default 1.0)"),
    ("rk_max", "max kinetic / internal energy G_K",
     "A run is rejected if R_K = \u03a3ALLKE/\u03a3ALLIE over T exceeds this\n"
     "value (too much kinetic energy, typically from mass scaling).\n"
     "Default 0.05 (5 %), the value used on the reference case."),
    ("rhg_max", "max artificial / internal energy G_HG",
     "A run is rejected if R_HG = \u03a3ALLAE/\u03a3ALLIE over T exceeds this\n"
     "value (too much hourglass energy). Default 0.05 (5 %)."),
]
# Domain dimensions: short label and definition (Eulerian part rectangle
# (-l_wp, -h_wp) -> (l_void, h_void), cel_model.py create_parts).
_DIM_LABELS = {"l_wp": "l_wp (\u2212x)", "l_void": "l_void (+x)",
               "h_wp": "h_wp (\u2212y)", "h_void": "h_void (+y)"}
_DIM_TIPS = {
    "l_wp": "Extent of the Eulerian domain towards \u2212x from its origin "
            "[mm] (Geometry tab, workpiece l_wp)",
    "l_void": "Extent of the Eulerian domain towards +x from its origin "
              "[mm] (Geometry tab, void l_void)",
    "h_wp": "Extent of the Eulerian domain towards \u2212y from its origin "
            "[mm] (Geometry tab, workpiece h_wp)",
    "h_void": "Extent of the Eulerian domain towards +y from its origin "
              "[mm] (Geometry tab, void h_void)",
}
# Source of each default eps_q (OptimizationCfg.criterion_rmse).
_EPS_HELP = {
    "Vx": "Default 10 mm/s",
    "Vy": "Default 10 mm/s",
    "T": "Default 10 K",
    "EVF": "Default 0.1 (volume fraction, no unit)",
    "Fc": "Default 10 N/mm",
    "Ff": "Default 10 N/mm",
}
for _q in _EPS_HELP:
    _EPS_HELP[_q] += (" (value used on the reference case, Ti6Al4V "
                      "orthogonal cutting).")
# Force quantities -> the tool-RP reaction-force history channel.
_FORCE_CHANNELS = {"Fc": "RF1_RP", "Ff": "RF2_RP"}
# GCI quantity name -> common tolerance label (the GCI selects with eps_q).
_GCI_NAMES = {"EVF": "EVF", "TEMP": "T", "V1": "Vx", "V2": "Vy",
              "Fc": "Fc", "Ff": "Ff"}
_DIM_ORDER = ("l_wp", "h_wp", "h_void", "l_void")


# Preview colours: the Geometry tab draws the domain (blue), the workpiece
# (green), the tool (orange) and the ROI (red dashed); the overlays of this
# tab take colours none of those use.
_C_ZOI = "#7c3aed"          # ZOI and its sampling points (purple)
_C_START = "#0f766e"        # step-2 starting domain (teal, dashed)
_C_CAP = "#374151"          # largest size allowed (dark grey, dash-dot)
_C_DSTAR = "#1e3a8a"        # D* found by step 2, not in the model (navy)
_MAX_PREVIEW_POINTS = 2000  # sampling points drawn (subsampled beyond)
_MAX_GRID_POINTS = 5000000  # beyond this the points are not even counted


class OptimizationTab(ModelStepsMixin, QWidget):
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
        # Steps, resume/load, pipeline (gui/tabs/model_steps.py)
        self._active = None             # context of the running study
        self._pipeline = False          # True while "Run all steps" runs
        self._is_busy = False
        self._model_refresher = None    # main window: reload model tabs
        self._study_cache = None        # RunCache of the running study
        self._study_offline = False     # loading: never launch Abaqus
        self._step_status = {}          # step -> status QLabel
        self._btn_apply = {}            # step -> "Use in model" button
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

        # Grey help texts: hidden by default, shown by the "Show help"
        # check box at the top of the inputs (the tooltips are always there).
        self._hints = []

        def hint(text):
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#6b7280;")
            lab.setVisible(False)
            self._hints.append(lab)
            return lab

        def grid(group):
            g = QGridLayout(group)
            g.setHorizontalSpacing(8)
            g.setVerticalSpacing(4)
            return g

        # Each panel shows its essential inputs; the rest sits in a
        # collapsed "Advanced parameters" section whose header counts the
        # fields that differ from their default (_refresh_advanced_counts).
        # Every default states its source in its tooltip.
        self._advanced = {}     # panel key -> (CollapsibleSection, fields)

        def advanced(key, fields_fn):
            sec = CollapsibleSection("Advanced parameters")
            self._advanced[key] = (sec, fields_fn)
            return sec

        def sub_grid(widget):
            g = QGridLayout(widget)
            g.setContentsMargins(14, 0, 0, 4)
            g.setHorizontalSpacing(8)
            g.setVerticalSpacing(4)
            return g

        _APPLY_TIPS = {
            "ms": "Write ms* in the Step tab (mass scaling on, factor ms*).",
            "mesh": "Write h* as the element size of the Mesh tab.",
            "domain": "Write D* as the Eulerian domain of the Geometry tab.",
        }

        def status_row(step):
            """Status of the step for the current model (+ 'Use in model')."""
            row = QHBoxLayout()
            row.setSpacing(6)
            lab = QLabel("")
            lab.setWordWrap(True)
            self._step_status[step] = lab
            row.addWidget(lab, 1)
            if step in _APPLY_TIPS:
                b = QPushButton("Use in model")
                b.setToolTip(_APPLY_TIPS[step])
                b.setVisible(False)
                b.clicked.connect(
                    lambda _=False, s=step: self._on_apply_step(s))
                self._btn_apply[step] = b
                row.addWidget(b, 0, Qt.AlignTop)
            return row

        # ---- How to use (grey help, shown with "Show help") --------------
        intro = hint(
            "Makes the simulation results independent of three numerical "
            "choices (mass scaling, element size, Eulerian domain). Check "
            "the comparison settings, then click 'Run all steps', or run "
            "steps 0 to 3 one by one and click 'Use in model' after each. "
            "Each step says whether it is done for the current model. "
            "Hover over a field for its definition and the source of its "
            "default.")

        # ---- Comparison settings: ZOI, eps_q (+ advanced: grid, T, guards)
        # The ZOI is DISTINCT from the ROI (Geometry tab, model output set
        # for DIC/IRT); empty fields default to the ROI. T, the
        # safeguards and the tolerances eps_q are shared by every study.
        gcom = QGroupBox("Comparison settings (used by every step)")
        cv0 = QVBoxLayout(gcom)
        cv0.setSpacing(4)
        cg0 = QGridLayout()
        cg0.setHorizontalSpacing(8)
        cg0.setVerticalSpacing(4)
        cv0.addLayout(cg0)
        lab = QLabel("Comparison zone ZOI [mm]")
        lab.setToolTip(
            "Zone where two runs are compared (measurement zone of this tab).\n"
            "It is NOT the ROI: the ROI is the output zone of the Geometry\n"
            "tab, matched to the DIC/IRT fields. An empty bound takes the\n"
            "ROI bound. The Sensitivity tab can propose a ZOI from its maps\n"
            "('Copy ZOI to the Model tab').")
        cg0.addWidget(lab, 0, 0)
        self.le_zoi = {}
        zoi_row = QHBoxLayout()
        zoi_row.setSpacing(4)
        for lbl, key in [("x min", "xmin"), ("x max", "xmax"),
                         ("y min", "ymin"), ("y max", "ymax")]:
            zoi_row.addWidget(QLabel(lbl))
            le = num_edit(placeholder="= ROI",
                          tip="ZOI bound [mm]; empty = ROI bound")
            self.le_zoi[key] = le
            zoi_row.addWidget(le)
            zoi_row.addSpacing(6)
            le.textChanged.connect(self._schedule_preview)
        self.btn_zoi_from_roi = QPushButton("ZOI = ROI")
        self.btn_zoi_from_roi.setToolTip("Copy the ROI of the Geometry tab "
                                         "into the ZOI fields.")
        self.btn_zoi_from_roi.clicked.connect(self._zoi_from_roi)
        zoi_row.addWidget(self.btn_zoi_from_roi)
        zoi_row.addStretch(1)
        cg0.addLayout(zoi_row, 0, 1)
        # Absolute tolerances eps_q: ONE set for every study (decision of
        # 2026-10-07): the ms and domain E_max and the GCI mesh selection.
        lab = QLabel("Admitted deviation ε_q")
        lab.setToolTip(
            "Largest difference between two runs that you treat as\n"
            "negligible, one value per quantity, in its own unit (absolute,\n"
            "not a percentage). A run pair passes when every quantity\n"
            "differs by less than its ε_q in the ZOI (E_max < 1).\n"
            "Defaults: values used on the reference case (Ti6Al4V\n"
            "orthogonal cutting).")
        cg0.addWidget(lab, 1, 0)
        self._q_eps = {}
        _unit = {q: u for (q, _f, u) in _QUANTITIES}
        eps_row = QHBoxLayout()
        eps_row.setSpacing(4)
        for q in ("Vx", "Vy", "T", "EVF", "Fc", "Ff"):
            lbl = QLabel(q)
            tip = "%s: admitted deviation [%s]. %s" % (
                q, _unit[q], _EPS_HELP[q])
            if q in ("Fc", "Ff"):
                tip += ("\n%s on the tool RP divided by the element size."
                        % ("RF1" if q == "Fc" else "RF2"))
            lbl.setToolTip(tip)
            eps_row.addWidget(lbl)
            le = num_edit(placeholder=_unit[q], tip=tip)
            le.setFixedWidth(64)
            self._q_eps[q] = le
            eps_row.addWidget(le)
            eps_row.addSpacing(10)
        eps_row.addStretch(1)
        cg0.addLayout(eps_row, 1, 1)
        cg0.setColumnStretch(1, 1)
        cv0.addWidget(hint(
            "ε_q: defaults are those of the reference case (Ti6Al4V). "
            "For another "
            "material or cutting condition, set each one to the smallest "
            "difference that matters for your comparison with the "
            "experiment, e.g. not larger than the measurement uncertainty "
            "of that quantity."))
        sec = advanced("common", lambda: [
            (self.le_grid_step, ""),
            *[(self._dom_texts[a], str(getattr(OptimizationCfgDefaults, a)))
              for a, _l, _t in _DOM_TEXTS]])
        ag = sub_grid(sec.body)
        ag.addWidget(QLabel("ZOI sampling step [mm]"), 0, 0)
        self.le_grid_step = num_edit(
            placeholder="= elem",
            tip="Spacing of the points where the ZOI is sampled [mm].\n"
                "Default (empty): the element size of the Mesh tab (the\n"
                "coarser meshes of steps 0 and 1 then have several points\n"
                "per element).")
        ag.addWidget(self.le_grid_step, 0, 1)
        lab = QLabel("Time window T (fraction of the simulated time)")
        lab.setToolTip(
            "Part of the simulated time over which the runs are compared\n"
            "(0 = start, 1 = end). It should cover the steady cutting regime\n"
            "only: check the cutting force of a run in the Results tab and\n"
            "start T after the force has stabilised.\n"
            "Default 0.3 to 1.0: the value hard-coded in the studies before\n"
            "it became a setting. Also used by the Sensitivity tab.")
        ag.addWidget(lab, 1, 0)
        self._dom_texts = {}
        win_row = QHBoxLayout()
        win_row.setSpacing(4)
        for i, (attr, label, tip) in enumerate(_DOM_TEXTS):
            lab = QLabel(label)
            lab.setToolTip(tip)
            le = num_edit(str(getattr(OptimizationCfgDefaults, attr)),
                          tip=tip)
            self._dom_texts[attr] = le
            if attr.startswith("window_"):      # start, end on one row
                win_row.addWidget(lab)
                win_row.addWidget(le)
                win_row.addSpacing(6)
            else:                               # one safeguard per row
                ag.addWidget(lab, i, 0)
                ag.addWidget(le, i, 1)
        win_row.addStretch(1)
        ag.addLayout(win_row, 1, 1, 1, 2)
        ag.setColumnStretch(3, 1)
        cv0.addWidget(sec)

        # ---- Step 0 · mass-scaling factor by an independence study -------
        gms = QGroupBox("Step 0 · Mass scaling factor ms")
        sv = QVBoxLayout(gms)
        sv.setSpacing(4)
        sg = QGridLayout()
        sg.setHorizontalSpacing(8)
        sv.addLayout(sg)
        sg.addWidget(QLabel("ms values to test"), 0, 0)
        self.le_ms_values = QLineEdit(OptimizationCfgDefaults.ms_values)
        self.le_ms_values.setToolTip(
            "Mass-scaling factors to test, strictly increasing, separated by\n"
            "commas or spaces. Default 250 to 4000, factor 2 between values.\n"
            "If every comparison passes, the study adds values by doubling\n"
            "the last one (up to 5 more) until a comparison fails.")
        sg.addWidget(self.le_ms_values, 0, 1)
        self.btn_ms = QPushButton("Run mass-scaling study")
        self.btn_ms.setToolTip(
            "Runs the ms values in increasing order on the current domain and\n"
            "compares each run with the previous one (E_max with the absolute\n"
            "eps_q of the common settings). Safeguards: outputs, R_K, R_HG, filter check and\n"
            "reverberation check. Keeps the largest ms reached by an unbroken\n"
            "chain of successes; stops at the first failure. Finished runs of\n"
            "an interrupted study are reused when it is resumed.")
        self.btn_ms.clicked.connect(self._on_run_ms_independence)
        sg.addWidget(self.btn_ms, 0, 2)
        sg.setColumnStretch(1, 1)
        sv.addLayout(status_row("ms"))
        sv.addWidget(hint(
            "Finds ms*, the largest factor that does not change the results "
            "beyond ε_q (larger ms = faster runs). Runs on the coarsest mesh "
            "of step 1 and the Eulerian domain of the Geometry tab. Before: "
            "enable the output filter (Step tab). Result: ms*, for the Step "
            "tab."))
        sec = advanced("ms", lambda: [(self.le_ms_elem, "")])
        ag = sub_grid(sec.body)
        ag.addWidget(QLabel("element size of these runs [mm]"), 0, 0)
        self.le_ms_elem = num_edit(
            placeholder="= coarsest",
            tip="Element size of the ms runs [mm]. Default (empty): the\n"
                "coarsest mesh of the step-1 plan (finest × ratio^(n−1)),\n"
                "the cheapest runs of the plan.")
        ag.addWidget(self.le_ms_elem, 0, 1)
        ag.setColumnStretch(2, 1)
        sv.addWidget(sec)

        # ---- Step 1 · mesh size by Richardson / GCI ---------------------
        gmesh = QGroupBox("Step 1 · Element size h (mesh convergence)")
        mv = QVBoxLayout(gmesh)
        mv.setSpacing(4)
        mg = QGridLayout()
        mg.setHorizontalSpacing(8)
        mv.addLayout(mg)
        mg.addWidget(QLabel("finest element size [mm]"), 0, 0)
        self.le_gci_finest = num_edit(
            placeholder="= elem",
            tip="Size of the finest mesh of the plan [mm]. Default (empty):\n"
                "the element size of the Mesh tab. The other meshes are\n"
                "finest × ratio, finest × ratio², ...")
        mg.addWidget(self.le_gci_finest, 0, 1)
        mg.addWidget(QLabel("number of meshes"), 0, 2)
        self.sp_gci_n = spin_box(3, 6, OptimizationCfgDefaults.gci_n_meshes)
        self.sp_gci_n.setToolTip(
            "Meshes in the plan (at least 3 for Richardson / GCI). Default 4\n"
            "(reference case: 0.5 / 1 / 2 / 4 µm).")
        mg.addWidget(self.sp_gci_n, 0, 3)
        # One tolerance set for the three axes (decision of 2026-10-07): the
        # mesh is selected with the common absolute eps_q. The old relative
        # GCI tolerances (persisted key sizing_tol) are no longer read.
        self.btn_mesh = QPushButton("Run mesh convergence (GCI)")
        self.btn_mesh.setToolTip(
            "GCI/Richardson mesh convergence on the current (fixed) domain:\n"
            "n systematically-refined meshes, observed order p, extrapolated\n"
            "value and GCI per quantity. Recommends the coarsest mesh within\n"
            "the absolute tolerances eps_q (common settings) of the reference.")
        self.btn_mesh.clicked.connect(self._on_run_mesh_gci)
        mg.addWidget(self.btn_mesh, 0, 4)
        mg.setColumnStretch(5, 1)
        mv.addLayout(status_row("mesh"))
        mv.addWidget(hint(
            "Finds h*, the coarsest mesh whose results stay within ε_q "
            "of the reference value (the GCI in % is only reported). Runs "
            "at ms* on the Eulerian domain of the Geometry tab: keep it "
            "generous. Result: h*, for the Mesh tab."))
        sec = advanced("mesh", lambda: [
            (self.le_gci_ratio, str(OptimizationCfgDefaults.gci_ratio)),
            (self.le_gci_min, "")])
        ag = sub_grid(sec.body)
        ag.addWidget(QLabel("refinement ratio"), 0, 0)
        self.le_gci_ratio = num_edit(
            OptimizationCfgDefaults.gci_ratio,
            tip="Size ratio between two successive meshes. Default 2.\n"
                "Celik et al. (2008, J. Fluids Eng. 130, 078001) recommend a\n"
                "ratio above 1.3 for the GCI.")
        ag.addWidget(self.le_gci_ratio, 0, 1)
        ag.addWidget(QLabel("smallest allowed finest size [mm]"), 1, 0)
        self.le_gci_min = num_edit(
            placeholder="none",
            tip="If the finest size is below this floor, the plan starts at\n"
                "the floor instead [mm]. Default (empty): no floor.")
        ag.addWidget(self.le_gci_min, 1, 1)
        ag.setColumnStretch(2, 1)
        mv.addWidget(sec)

        # ---- Step 2 · Eulerian domain -----------------------------------
        gdom = QGroupBox("Step 2 · Eulerian domain size")
        dv = QVBoxLayout(gdom)
        dv.setSpacing(4)
        dg = QGridLayout()
        dg.setHorizontalSpacing(8)
        dg.setVerticalSpacing(4)
        dv.addLayout(dg)
        # initial domain (read-only), one row per dimension pair
        self._init_lbl = {}
        dg.addWidget(QLabel("starting domain [mm]"), 0, 0)
        for r, pair in enumerate([("l_wp", "l_void"), ("h_wp", "h_void")]):
            for c, d in enumerate(pair):
                lab = QLabel(_DIM_LABELS[d])
                lab.setToolTip(_DIM_TIPS[d])
                dg.addWidget(lab, r, 1 + 2 * c)
                il = QLabel("—"); il.setStyleSheet("color:#374151;")
                self._init_lbl[d] = il
                dg.addWidget(il, r, 2 + 2 * c)
        dg.setColumnStretch(5, 1)
        btn_row = QHBoxLayout()
        self.btn_domain = QPushButton("Run domain sizing (independence)")
        self.btn_domain.setToolTip(
            "Sequential independence study: each dimension grown by a constant\n"
            "step from ZOI + margin (mesh and mass scaling held fixed), runs\n"
            "compared by the mean absolute difference in the ZOI against the\n"
            "absolute eps_q, residual influence bounded by a geometric tail.\n"
            "The domain diagonal only raises a warning.")
        self.btn_domain.clicked.connect(self._on_run_domain_independence)
        btn_row.addWidget(self.btn_domain)
        btn_row.addStretch(1)
        dv.addLayout(btn_row)
        dv.addLayout(status_row("domain"))
        dv.addWidget(hint(
            "Starts from the ZOI plus the margin (teal box of the preview) "
            "and grows each side until the results in the ZOI stop changing "
            "beyond ε_q. Runs at ms* and h* (the element size of the Mesh "
            "tab). Result: D*, for the Geometry tab (Eulerian part)."))
        sec = advanced("domain", lambda: [
            (self.sp_margin, 0),
            *[(le, "") for le in self._max.values()],
            *[(sp, int(getattr(OptimizationCfgDefaults, a)))
              for a, sp in self._dom_spins.items()]])
        ag = sub_grid(sec.body)
        lab = QLabel("margin around the ZOI [elements]")
        lab.setToolTip("Elements added on each side of the ZOI to build the "
                       "starting domain. Default 0.")
        ag.addWidget(lab, 0, 0)
        self.sp_margin = spin_box(0, 50, 0)
        self.sp_margin.setToolTip(lab.toolTip())
        ag.addWidget(self.sp_margin, 0, 1)
        self._dom_spins = {}
        for r, (attr, label, lo, hi, tip) in enumerate(_DOM_SPINS, start=1):
            lab = QLabel(label)
            lab.setToolTip(tip)
            ag.addWidget(lab, r, 0)
            sp = spin_box(lo, hi,
                          int(getattr(OptimizationCfgDefaults, attr)))
            sp.setToolTip(tip)
            self._dom_spins[attr] = sp
            ag.addWidget(sp, r, 1)
        lab = QLabel("largest size allowed [mm]")
        lab.setToolTip("Upper bound of each dimension during the study\n"
                       "(grey dash-dot lines in the preview). Empty = no cap.")
        ag.addWidget(lab, 0, 2, 1, 2)
        self._max = {}
        for r, d in enumerate(_DIM_ORDER, start=1):
            lab = QLabel(_DIM_LABELS[d])
            lab.setToolTip(_DIM_TIPS[d])
            ag.addWidget(lab, r, 2)
            mx = num_edit(placeholder="no cap",
                          tip="Cap of %s [mm]; empty = no cap" % d)
            self._max[d] = mx
            ag.addWidget(mx, r, 3)
        ag.setColumnStretch(4, 1)
        dv.addWidget(sec)

        # ---- Step 3 · interaction checks --------------------------------
        gchk = QGroupBox("Step 3 · Final checks at (ms*, h*, D*)")
        kv = QVBoxLayout(gchk)
        self.btn_checks = QPushButton("Run interaction checks")
        self.btn_checks.setToolTip(
            "A-posteriori checks of the sized model:\n"
            "mass-scaling factor inside its window at (h*, D*), the four\n"
            "dimensions grown together (1 run), ms* against the previous ms\n"
            "at (h*, D*) (1 run), and the GCI plan of step 1 run again on D*\n"
            "(h* must stay within tolerance). Available once a domain study\n"
            "has finished; it is read back from its folder when needed.")
        self.btn_checks.setEnabled(False)
        self.btn_checks.clicked.connect(self._on_run_interaction_checks)
        kv.addWidget(self.btn_checks, 0, Qt.AlignLeft)
        kv.addLayout(status_row("checks"))
        kv.addWidget(hint(
            "Checks that the three values still hold together: D* grown on "
            "all sides (1 run), ms* against the previous ms (1 run), and the "
            "mesh plan of step 1 run again on D*. Available after step 2."))

        # ---- Top row: the whole pipeline, reading a study back, help -----
        top = QHBoxLayout()
        top.setSpacing(6)
        self.btn_all = QPushButton("Run all steps")
        self.btn_all.setToolTip(
            "Runs steps 0 to 3 in order. A step already done for this model\n"
            "is skipped, an interrupted one is resumed. Each result is\n"
            "written into the model (Step, Mesh and Geometry tabs) before\n"
            "the next step starts. Stops at the first step that does not\n"
            "succeed.")
        self.btn_all.clicked.connect(self._on_run_all)
        top.addWidget(self.btn_all)
        self.btn_open = QPushButton("Open a study\u2026")
        self.btn_open.setToolTip(
            "Reads back a study from its folder, without any Abaqus run:\n"
            "its settings are put back in this tab, its result is shown and\n"
            "recorded for this model. A step-2 folder also brings back its\n"
            "final checks. An unfinished study can then be resumed.")
        self.btn_open.clicked.connect(self._on_open_study)
        top.addWidget(self.btn_open)
        top.addStretch(1)
        self.cb_help = QCheckBox("Show help")
        self.cb_help.setToolTip("Show the grey explanations under each step.")
        self.cb_help.toggled.connect(self._show_help)
        top.addWidget(self.cb_help)

        # ---- Inputs column (scrolls instead of squeezing) ---------------
        inputs = QWidget()
        il_ = QVBoxLayout(inputs)
        il_.setContentsMargins(0, 0, 4, 0)
        il_.addLayout(top)
        il_.addWidget(intro)
        for gbox in (gcom, gms, gmesh, gdom, gchk):
            il_.addWidget(gbox)
        il_.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(inputs)
        # Never narrower than its content: the preview shrinks instead, and
        # only a vertical scrollbar can appear. Measured with the advanced
        # sections open, so opening one never needs a horizontal scrollbar.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for sec, _f in self._advanced.values():
            sec.body.setVisible(True)
        scroll.setMinimumWidth(inputs.sizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width())
        for sec, _f in self._advanced.values():
            sec.body.setVisible(False)

        # ---- Preview (reuses the Geometry tab's preview widget) --------
        gprev = QGroupBox("Preview")
        pv = QVBoxLayout(gprev)
        self.preview = GeometryPreview()
        # Home re-fits the view to the model AND this tab's overlays.
        self.preview.fit_override = self._draw_preview
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
        # GCI study (|f_q(h) - f_ref| / eps_q, per quantity) and the
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

        # Auto-refresh the preview when the inputs that affect it change
        # (debounced: a "ZOI = ROI" click changes four fields at once).
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(60)
        self._preview_timer.timeout.connect(self._draw_preview)
        for _d in _DIM_ORDER:
            self._max[_d].textChanged.connect(self._schedule_preview)
        self.le_grid_step.textChanged.connect(self._schedule_preview)
        self.sp_margin.valueChanged.connect(self._schedule_preview)

        self._wire_opt_persistence()
        self.refresh_inputs()

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
        v = self._grid_step_set()
        return float(self.cfg.elem_size) if v is None else v

    def _grid_step_set(self):
        """The sampling step typed in the tab, or None (blank or not a
        positive number: the element size is used)."""
        v = self._float_or(self.le_grid_step, None)
        return v if v is not None and v > 0 else None


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
        for _sec, fields in self._advanced.values():
            for w, _default in fields():
                sig = (w.textChanged if isinstance(w, QLineEdit)
                       else w.valueChanged)
                sig.connect(self._refresh_advanced_counts)
        self._refresh_advanced_counts()

    def _refresh_advanced_counts(self, *_):
        """Show, on each collapsed 'Advanced parameters' header, how many of
        its fields differ from their default, so a hidden change is seen."""
        for sec, fields in self._advanced.values():
            n = 0
            for w, default in fields():
                if isinstance(w, QLineEdit):
                    cur = w.text().strip().replace(",", ".")
                    ref = str(default).strip()
                    try:
                        same = float(cur) == float(ref)
                    except ValueError:
                        same = cur == ref
                else:
                    same = int(w.value()) == int(default)
                n += 0 if same else 1
            sec.set_changed_count(n)

    def _sync_opt_to_cfg(self, *_):
        """Write the current widget values into cfg.optimization. No-op while
        loading (so populating widgets from a file does not re-dirty it)."""
        if self._loading:
            return
        o = self.cfg.optimization
        o.zoi = {k: self.le_zoi[k].text()
                 for k in ("xmin", "xmax", "ymin", "ymax")}
        o.criterion_rmse = {q: le.text() for q, le in self._q_eps.items()}
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
        self._refresh_step_status()

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
            self.le_gci_finest.setText(str(o.gci_finest))
            self.le_gci_ratio.setText(str(o.gci_ratio or "2"))
            self.le_gci_min.setText(str(o.gci_min))
            self.sp_gci_n.setValue(int(
                o.gci_n_meshes or OptimizationCfgDefaults.gci_n_meshes))
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
        """Reload the panel from cfg.optimization and redraw (called after
        a profile is opened, via MainWindow._rebind_cfg)."""
        self._load_opt_from_cfg()
        self._draw_preview()
        self._refresh_step_status()

    def forget_results(self):
        """Drop the study results held in memory (another profile was
        opened: they belong to the previous one). No-op while a study
        runs."""
        if self._is_busy:
            return
        self._last_ms = self._last_gci = self._last_checks = None
        self._last_domain_result = self._last_domain_dir = None
        self._last_domain_ms = self._last_domain_spec = None
        self._refresh_convergence_view()
        self._refresh_step_status()

    def showEvent(self, event):
        # Other tabs may have changed the model meanwhile.
        super().showEvent(event)
        self._draw_preview()
        self._refresh_step_status()

    def _show_help(self, on):
        for lab in self._hints:
            lab.setVisible(bool(on))

    def _schedule_preview(self, *_):
        self._preview_timer.start()

    def _sync_run_buttons(self):
        """The run and open buttons are off while a study runs and for the
        whole of 'Run all steps' (also between two steps: a study started
        there would replace the one the pipeline starts next)."""
        running = bool(self._is_busy or self._pipeline)
        for b in (self.btn_ms, self.btn_mesh, self.btn_domain, self.btn_all,
                  self.btn_open):
            b.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        # The checks button and the "Use in model" buttons.
        self._refresh_step_status()

    def _retire_worker(self, attr):
        """Before a new study replaces the worker in `attr`: wait for the
        previous one to return from run() (its result is already handled
        when no study is busy). A QThread destroyed while it runs aborts
        the program."""
        w = getattr(self, attr, None)
        if w is not None and w.isRunning():
            w.wait(10000)

    def _cannot_start(self, what) -> bool:
        """True (with a log line) when a study is busy: nothing new starts
        until it ends."""
        if not self._is_busy:
            return False
        self._log_ui("%s not started: another study is running." % what)
        return True

    def is_running(self) -> bool:
        """True while a study or 'Run all steps' runs (also between two
        steps of the pipeline): its results belong to this profile."""
        return bool(self._is_busy or self._pipeline or self._active)

    def shutdown(self, timeout_ms: int = 60000) -> bool:
        """Stop a running study synchronously: the window is closing.

        The study's result is dropped (the workers' signals are
        disconnected first, so nothing lands on a closing window, and the
        step keeps the record it had). The Abaqus job in flight is stopped
        like a Cancel does (``abaqus terminate``, then the process tree; the
        remote agent is asked to stop it). The finished runs stay in the
        study folder, where 'Open a study' finds them. Returns True if every
        study thread ended within `timeout_ms`."""
        import warnings
        self._pipeline = False
        self._active = None
        self._cancel_evt.set()
        workers = []
        for attr in ("_di_worker", "_mesh_worker", "_checks_worker",
                     "_ms_worker"):
            w = getattr(self, attr, None)
            if w is None or not w.isRunning():
                continue
            with warnings.catch_warnings():
                # PySide warns (instead of raising) on a signal with no slot.
                warnings.simplefilter("ignore", RuntimeWarning)
                for sig in (w.progress, w.finished_ok, w.failed):
                    try:
                        sig.disconnect()
                    except (RuntimeError, TypeError):
                        pass
            w.cancel()
            workers.append(w)
        job, proc = self._current_job, self._current_proc
        cmd, run_dir = self._current_abaqus_cmd, self._current_run_dir
        if isinstance(proc, RemoteProcess):
            try:
                proc.cancel()
            except Exception:
                log_swallowed("stopping the remote job on close")
        elif proc is not None:
            asked = False
            if job and cmd:
                try:
                    asked = bool(abaqus_terminate_job(cmd, job, run_dir))
                except Exception:
                    log_swallowed("abaqus terminate on close")
            if asked:
                # The study thread returns once the solver has unwound.
                for w in workers:
                    w.wait(10000)
            # `returncode`, not poll(): the study thread reaps this Popen.
            if proc.returncode is None:
                try:
                    if not kill_process_tree_by_pid(proc.pid):
                        proc.kill()
                except Exception:
                    log_swallowed("killing the Abaqus job on close")
        ended = all(bool(w.wait(int(timeout_ms))) for w in workers)
        if not ended:
            logging.getLogger(__name__).warning(
                "a Model tab study thread is still running after %d ms",
                timeout_ms)
        self._sim_timer.stop()
        self._is_busy = False
        return ended

    def _draw_preview(self, *_):
        """The model as the Geometry tab draws it (Eulerian domain,
        workpiece, tool, ROI), plus what this tab compares and grows: the
        ZOI and its sampling points, the step-2 starting domain, the largest
        size allowed, and D* when step 2 found one the model does not use
        yet. Every element has a legend entry and the view fits them all.
        Never raises: a drawing problem must not stop a study's
        bookkeeping (it is called when a study ends)."""
        if hasattr(self, "_preview_timer"):
            self._preview_timer.stop()
        try:
            self.preview.update_from_config(self.cfg)
        except Exception:
            log_swallowed("geometry preview update", level=logging.DEBUG)
            return
        try:
            self._draw_overlays(self.preview._ax)
        except Exception:
            log_swallowed("drawing the Model tab overlays",
                          level=logging.DEBUG)
        self.preview._canvas.draw_idle()

    def _draw_overlays(self, ax):
        from matplotlib.patches import Rectangle
        # The base drawing, named for this tab; the tool reference point
        # (where the BCs apply) is not needed here.
        names = {"Eulerian domain": "Eulerian domain (Geometry tab)",
                 "Workpiece (reference)": "Workpiece",
                 "ROI / bbox": "ROI (results written here)"}
        for art in list(ax.patches) + list(ax.lines):
            lab = str(art.get_label())
            if lab in names:
                art.set_label(names[lab])
            elif lab.startswith("Tool RP"):
                art.remove()
        inp = self.config_inputs()
        ex0, ey0 = self.euler_offset()
        boxes = [inp["roi"]]          # (x0, x1, y0, y1) the view must hold
        xs_extra, ys_extra = [], []

        def rect(b, **kw):
            ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2],
                                   fill=False, **kw))
            boxes.append(tuple(b))

        def euler_box(h_wp, h_void, l_wp, l_void):
            return (-l_wp + ex0, l_void + ex0, -h_wp + ey0, h_void + ey0)

        def same_box(a, b):
            return all(math.isclose(u, v, rel_tol=1e-9, abs_tol=1e-12)
                       for u, v in zip(a, b))

        # Step-2 starting domain = ZOI + margin (also shown as numbers).
        di = None
        try:
            di = self.compute_initial_dims()
            for d in _DIM_ORDER:
                self._init_lbl[d].setText("%.4g" % getattr(di, d))
        except Exception:
            for d in _DIM_ORDER:
                self._init_lbl[d].setText("\u2014")
        zoi = self.zoi()
        zoi_ok = zoi[1] > zoi[0] and zoi[3] > zoi[2]
        if zoi_ok:
            rect(zoi, edgecolor=_C_ZOI, lw=1.8, zorder=7,
                 label="ZOI (comparison zone)")
            self._draw_sampling_points(ax, zoi)
        if di is not None:
            start = euler_box(di.h_wp, di.h_void, di.l_wp, di.l_void)
            # With no margin it lies on the ZOI: drawn on top, named so.
            on_zoi = zoi_ok and same_box(start, zoi)
            rect(start, edgecolor=_C_START, lw=1.3, ls="--",
                 zorder=8 if on_zoi else 6,
                 label="Step 2 starting domain (= ZOI)" if on_zoi
                 else "Step 2 starting domain (ZOI + margin)")
        # Largest size allowed: a box when the four caps are set, else one
        # line per cap that is set.
        cp = self.caps()
        if set(cp) >= set(_DIM_ORDER):
            rect(euler_box(cp["h_wp"], cp["h_void"], cp["l_wp"],
                           cp["l_void"]),
                 edgecolor=_C_CAP, lw=1.3, ls="-.", zorder=5,
                 label="Largest size allowed (step 2)")
        elif cp:
            first = True
            for d, v in cp.items():
                kw = dict(color=_C_CAP, lw=1.2, ls="-.", zorder=5,
                          label="Largest size allowed (step 2)"
                          if first else "_nolegend_")
                if d in ("l_wp", "l_void"):
                    x = -v + ex0 if d == "l_wp" else v + ex0
                    ax.axvline(x, **kw)
                    xs_extra.append(x)
                else:
                    y = -v + ey0 if d == "h_wp" else v + ey0
                    ax.axhline(y, **kw)
                    ys_extra.append(y)
                first = False
        # D* found by step 2 but not (yet) in the model.
        st, rec = self._step_state("domain")
        if st == "done" and not in_model(
                "domain", self._model_values()["dims"], rec.get("value")):
            v = rec["value"]
            rect(euler_box(v[0], v[1], v[2], v[3]), edgecolor=_C_DSTAR,
                 lw=1.8, ls=":", zorder=6,
                 label="D* found by step 2 (not in the model yet)")
        # What would make a study fail or sample nothing.
        warn = []
        roi = inp["roi"]
        h_wp, h_void, l_wp, l_void = self.cfg.effective_euler_dims()
        eul = euler_box(h_wp, h_void, l_wp, l_void)

        def inside(a, b):
            tol = 1e-9 * max(1.0, *[abs(v) for v in b])
            return (a[0] >= b[0] - tol and a[1] <= b[1] + tol
                    and a[2] >= b[2] - tol and a[3] <= b[3] + tol)
        if not zoi_ok:
            warn.append("The ZOI is empty (min \u2265 max).")
        else:
            if not inside(zoi, roi):
                warn.append("The ZOI goes beyond the ROI: no results are "
                            "written there.")
            if not inside(zoi, eul):
                warn.append("The ZOI goes beyond the Eulerian domain of the "
                            "Geometry tab (steps 0 and 1 run on it).")
        if warn:
            ax.text(0.02, 0.98, "\n".join("\u26a0 " + w for w in warn),
                    transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    color="#8a1f11", zorder=20,
                    bbox=dict(boxstyle="round", facecolor="#fff3cd",
                              edgecolor="#8a1f11", alpha=0.95))
        self._fit_view(ax, boxes, xs_extra, ys_extra)
        ax.set_title("")
        ax.legend(loc="best", fontsize=7, framealpha=0.9)

    def _fit_view(self, ax, boxes, xs_extra, ys_extra):
        """Limits that hold the drawing and every overlay, centred on them
        and already shaped like the axes box: the equal-aspect adjustment
        of matplotlib then has nothing to shrink (it shrinks one direction
        around the view centre and cut off what lay at its edge)."""
        bx = list(xs_extra) + [v for b in boxes for v in b[:2]]
        by = list(ys_extra) + [v for b in boxes for v in b[2:]]
        try:
            ax.relim(visible_only=True)       # the model's own shapes
            dl = ax.dataLim
            if all(math.isfinite(v) for v in (dl.x0, dl.x1)):
                bx += [dl.x0, dl.x1]
            if all(math.isfinite(v) for v in (dl.y0, dl.y1)):
                by += [dl.y0, dl.y1]
        except Exception:
            (x0, x1), (y0, y1) = self.preview._compute_fit_limits(self.cfg)
            bx += [x0, x1]
            by += [y0, y1]
        bx = [v for v in bx if math.isfinite(v)]
        by = [v for v in by if math.isfinite(v)]
        if not bx or not by:
            return
        w = max(max(bx) - min(bx), 1e-6)
        h = max(max(by) - min(by), 1e-6)
        cx, cy = 0.5 * (max(bx) + min(bx)), 0.5 * (max(by) + min(by))
        pad = 0.06 * max(w, h)
        w, h = w + 2.0 * pad, h + 2.0 * pad
        try:
            pos = ax.get_position()
            fw, fh = ax.figure.get_size_inches()
            ratio = (pos.height * fh) / (pos.width * fw)
        except Exception:
            ratio = None
        if ratio and math.isfinite(ratio) and ratio > 0:
            if h / w < ratio:
                h = w * ratio
            else:
                w = h / ratio
        ax.set_xlim(cx - 0.5 * w, cx + 0.5 * w)
        ax.set_ylim(cy - 0.5 * h, cy + 0.5 * h)

    def _draw_sampling_points(self, ax, zoi):
        """The points where the studies sample the ZOI (the grid of
        gui.sensitivity.zoi_sampling.roi_grid), thinned out for drawing."""
        step = self.grid_step()
        if not step or step <= 0:
            return
        fx = (zoi[1] - zoi[0]) / step
        fy = (zoi[3] - zoi[2]) / step
        if not (math.isfinite(fx) and math.isfinite(fy)):
            return
        n_est = (fx + 1.0) * (fy + 1.0)
        if n_est > _MAX_GRID_POINTS:
            ax.plot([], [], ls="none", marker=".", color=_C_ZOI,
                    label="ZOI sampling points: %.3g (not drawn)" % n_est)
            return
        nx = int(math.floor(fx + 1e-9)) + 1
        ny = int(math.floor(fy + 1e-9)) + 1
        n = nx * ny
        k = max(1, int(math.ceil(math.sqrt(n / float(_MAX_PREVIEW_POINTS)))))
        xs = zoi[0] + step * np.arange(0, nx, k)
        ys = zoi[2] + step * np.arange(0, ny, k)
        XX, YY = np.meshgrid(xs, ys)
        label = ("ZOI sampling points (%d)" % n if k == 1 else
                 "ZOI sampling points (%d, 1 in %d per direction shown)"
                 % (n, k))
        ax.scatter(XX.ravel(), YY.ravel(), s=3, c=_C_ZOI, alpha=0.5,
                   linewidths=0, zorder=7, label=label)

    def thresholds(self) -> dict:
        out = {}
        for q, le in self._q_eps.items():
            txt = le.text().strip().replace(",", ".")
            if txt:
                try:
                    v = float(txt)
                except ValueError:
                    continue
                if math.isfinite(v):
                    out[q] = v
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
                    v = float(txt)
                except ValueError:
                    continue
                if math.isfinite(v):
                    out[d] = v
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
        """The run launcher of one study: run_bundle(cfg) -> bundle | None.

        Finished runs are reused: when the study cache (self._study_cache,
        a RunCache set by the launcher) holds a run with exactly the
        parameters of `cfg`, its saved bundle is returned and Abaqus is not
        launched; a run taken from another study folder is copied into
        `run_dir`, so every folder holds the runs its replay needs. A run of
        `run_dir` whose analysis did not complete gives None again (same
        replay). With self._study_offline set (loading a study), a missing
        run is not launched either: the miss is recorded in
        state["miss"] and the study is stopped. A run that cannot be made
        (Abaqus not started, results not back) also stops the study, with
        state["launch_error"]: it is not a result of the model. New jobs are
        numbered after the runs already in `run_dir`, so a resumed study
        never overwrites one of its earlier runs."""
        import subprocess
        from gui.sensitivity.run_cache import (
            copy_run, next_job_index, remove_job_files, same_folder,
            sta_outcome, write_failed_marker)

        cache = getattr(self, "_study_cache", None)
        offline = bool(getattr(self, "_study_offline", False))
        counter = {"i": next_job_index(run_dir, prefix)}
        # Path of the LAST launched job's .sta and its name, readable by the
        # study's cost hook right after run_bundle returns (same thread).
        state = {"sta": None, "job": None, "filter_check": None,
                 "miss": None, "launch_error": None, "analysis_failed": [],
                 "n_reused": 0, "n_launched": 0}

        def stop_study():
            self._cancel_evt.set()
            for attr in ("_di_worker", "_mesh_worker", "_checks_worker",
                         "_ms_worker"):
                w = getattr(self, attr, None)
                if w is not None and hasattr(w, "cancel"):
                    w.cancel()

        def cannot_run(why):
            """A run could not be made or its results did not come back:
            stop the study (it stays resumable) instead of counting the run
            as a failed one."""
            if state["launch_error"] is None:
                state["launch_error"] = why
            self._log_ui("[STOP] %s" % why)
            stop_study()
            return None

        def reuse(hit, cfg):
            """Return the saved bundle of a finished run (cache hit)."""
            self._current_sta = hit.sta
            state["sta"], state["job"] = hit.sta, hit.job
            state["filter_check"] = None
            try:
                from gui.core.filter_check import check_bundle, window_from_cfg
                state["filter_check"] = check_bundle(
                    hit.npz, write_meta=False, window=window_from_cfg(cfg))
            except Exception as e:
                self._log_ui("[%s] filter check failed: %s\n" % (hit.job, e))
            try:
                bundle = ResultsBundle.load(hit.npz)
            except Exception as e:
                self._log_ui("[%s] saved run unreadable (%s): launching it "
                             "again\n" % (hit.job, e))
                return None
            state["n_reused"] += 1
            if same_folder(hit.folder, run_dir):
                self._log_ui("[%s] already computed: reused" % hit.job)
                return bundle
            copied = None
            if not offline:
                # A copy in this study's folder: the folder stays complete
                # when the other one is moved or deleted.
                i = counter["i"]; counter["i"] += 1
                job = "%s_run%03d" % (prefix, i)
                if copy_run(hit.folder, hit.job, run_dir, job):
                    copied = (cache.add_run(run_dir, job)
                              if cache is not None else None)
            if copied is not None:
                self._current_sta = copied.sta
                state["sta"], state["job"] = copied.sta, copied.job
                self._log_ui("[%s] already computed (%s in %s): reused, "
                             "copied into this study" % (
                                 copied.job, hit.job, Path(hit.folder).name))
            else:
                self._log_ui("[%s] already computed (from %s): reused"
                             % (hit.job, Path(hit.folder).name))
            return bundle

        def run_bundle(cfg):
            params = cfg.to_params_dict()
            hit = (cache.lookup(params, prefer=run_dir) if cache is not None
                   else None)
            if hit is not None:
                bundle = reuse(hit, cfg)
                if bundle is not None:
                    return bundle
            failed = (cache.lookup_failed(params, run_dir)
                      if cache is not None else None)
            if failed is not None:
                # Its analysis did not complete when it ran: the replay
                # meets the same failure (Abaqus is deterministic).
                self._current_sta = failed.sta
                state["sta"], state["job"] = failed.sta, failed.job
                state["filter_check"] = None
                state["analysis_failed"].append(failed.job)
                self._log_ui("[%s] its analysis did not complete when it "
                             "ran: counted as a failed run again (not "
                             "relaunched)" % failed.job)
                return None
            if offline:
                # Loading: never launch. Record the first missing run and
                # stop the study; the tab reports why the folder is short.
                if state["miss"] is None:
                    state["miss"] = params
                stop_study()
                return None
            if self._cancel_evt.is_set():
                return None
            i = counter["i"]; counter["i"] += 1
            job = "%s_run%03d" % (prefix, i)
            out_path = Path(run_dir) / ("%s.results.npz" % job)
            self._current_sta = Path(run_dir) / ("%s.sta" % job)
            state["sta"], state["job"] = self._current_sta, job
            state["filter_check"] = None
            # Files left by an interrupted run of the same name (a stale
            # .meta.json next to a new .npz would pass for a finished run).
            for p in remove_job_files(run_dir, job):
                log_swallowed("removing stale file %s" % p,
                              level=logging.DEBUG)
            state["n_launched"] += 1
            args = build_abaqus_args(
                prefs.abaqus_cmd, prefs.abaqus_script,
                params, {"cpus": cpus, "job_name": job})
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
                if is_remote(prefs):
                    # Same contract as the Popen: poll/stdout/returncode. The
                    # agent mirrors .sta/.gui.log/.results.npz into run_dir.
                    self._log_ui("[%s] submitted to the remote agent\n" % job)
                    proc = submit_remote(prefs, run_dir, params,
                                         {"cpus": cpus, "job_name": job})
                else:
                    proc = subprocess.Popen(
                        args, cwd=str(run_dir),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except Exception as e:
                self._log_ui("failed to start Abaqus: %s\n" % e)
                self._current_job = None
                return cannot_run("%s could not be started: %s" % (job, e))
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
            if self._cancel_evt.is_set() and isinstance(proc, RemoteProcess):
                # A Cancel that landed while the run was being submitted saw
                # no process to stop; stop it now (no-op if already done).
                proc.cancel()
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
            if self._cancel_evt.is_set():
                self._log_ui("[%s] no bundle (cancelled)\n" % job)
                return None
            if proc.returncode != 0 or not out_path.exists():
                how = sta_outcome(run_dir, job)
                if how == "not_completed":
                    # Abaqus stopped the analysis: a result of the model
                    # (e.g. too large a mass scaling), kept for the replay.
                    write_failed_marker(run_dir, job, params,
                                        "analysis not completed (rc=%s)"
                                        % proc.returncode)
                    if cache is not None:
                        cache.add_failed(run_dir, job)
                    state["analysis_failed"].append(job)
                    self._log_ui("[%s] the analysis did not complete (see "
                                 "%s.msg in the study folder): counted as a "
                                 "failed run" % (job, job))
                    return None
                return cannot_run(
                    "%s ended without results (rc=%s): %s" % (
                        job, proc.returncode,
                        "the analysis completed but its results were not "
                        "written or did not come back" if how == "success"
                        else "Abaqus did not start or was stopped (its .sta "
                        "has no final line)"))
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
                # Writing its result into the run's files failed (file
                # held, share read-only...): the verdict itself, as a
                # replay of this run computes it.
                try:
                    from gui.core.filter_check import (
                        check_bundle, window_from_cfg)
                    state["filter_check"] = check_bundle(
                        out_path, write_meta=False,
                        window=window_from_cfg(cfg))
                except Exception:
                    log_swallowed("filter check without writing",
                                  level=logging.DEBUG)
            try:
                bundle = ResultsBundle.load(out_path)
            except Exception as e:
                self._log_ui("[%s] load failed: %s\n" % (job, e))
                return cannot_run("the results of %s cannot be read: %s"
                                  % (job, e))
            if cache is not None:
                cache.add_run(run_dir, job)
            return bundle

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

    def set_zoi(self, bbox):
        """Fill the ZOI fields with (xmin, xmax, ymin, ymax) [mm], e.g. the
        ZOI proposed from the sensitivity maps."""
        for k, v in zip(("xmin", "xmax", "ymin", "ymax"), bbox):
            self.le_zoi[k].setText("%.6g" % float(v))
        self._draw_preview()

    def model_settings(self) -> dict:
        """eps_q and the window T, for the ZOI proposal of the Sensitivity
        tab (same values as the studies of this tab)."""
        return {"eps": self.thresholds(), "window": self.window()}

    def _zoi_from_roi(self):
        for k, v in zip(("xmin", "xmax", "ymin", "ymax"),
                        self.config_inputs()["roi"]):
            self.le_zoi[k].setText("%.6g" % v)
        self._draw_preview()

    def _gci_tolerances(self):
        """The absolute common tolerances eps_q keyed by the GCI quantity
        names (TEMP = T, V1 = Vx, V2 = Vy; same units)."""
        thr = self.thresholds()
        return {g: thr[q] for g, q in _GCI_NAMES.items() if q in thr}

    def _dims_from_cfg(self):
        g = self.cfg.euler_geometry
        return DomainDims(h_wp=float(g.h_wp), h_void=float(g.h_void),
                          l_wp=float(g.l_wp), l_void=float(g.l_void))

    def _busy(self, on, msg="", color="#1d4ed8"):
        self._is_busy = bool(on)
        self._sync_run_buttons()
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

    def _validate_launch(self, folder=None):
        """Shared pre-flight for a run: returns (prefs, workdir, cpus) or None
        (after showing a warning). `folder`: the study folder the runs go
        to, when it is not a new one in the working directory (a resumed
        study, the final checks): in remote mode it must be reachable by
        the compute PC too."""
        prefs = self._prefs_getter() if self._prefs_getter else None
        if prefs is None:
            QMessageBox.warning(self, "Preferences",
                                "No preferences (Abaqus command/script).")
            return None
        problems = launch_problems(prefs, folder or prefs.default_workdir)
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
        """The number in a field, or `default` (blank, not a number, or not
        finite)."""
        txt = line_edit.text().strip().replace(",", ".")
        try:
            v = float(txt)
        except (ValueError, TypeError):
            return default
        return v if math.isfinite(v) else default

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
    def _gci_plan_sizes(self):
        """Element sizes of the step-1 plan, finest first, from the panel
        (blank finest = the Mesh tab's element size); raises ValueError."""
        from gui.sensitivity.mesh_gci import _mesh_sizes
        return _mesh_sizes(
            self._float_or(self.le_gci_finest, float(self.cfg.elem_size)),
            self._float_or(self.le_gci_ratio, 2.0),
            int(self.sp_gci_n.value()),
            self._float_or(self.le_gci_min, None))

    def _freeze_plan_start(self):
        """A blank 'finest element size' means the Mesh tab's element size.
        The first study that uses it writes it into the field: h* is later
        written into the Mesh tab, and the plan must not follow it (each
        new run of steps 0 and 1 would test coarser meshes)."""
        if self._float_or(self.le_gci_finest, None) is not None:
            return
        self.le_gci_finest.setText("%g" % float(self.cfg.elem_size))

    def ms_settings(self):
        """(ms values, element size) of step 0; raises ValueError. A blank
        element size is the coarsest mesh of the step-1 plan."""
        values = parse_ms_values(self.le_ms_values.text())
        elem = self._float_or(self.le_ms_elem, None)
        if elem is None:
            elem = self._gci_plan_sizes()[-1]
        if elem is None or elem <= 0:
            raise ValueError("the element size of the ms study must be > 0")
        return values, float(elem)

    def _study_base_cfg(self, spec, elem=None):
        """Config of a study's runs: the current model with what the study
        was started with (mass scaling, filter verification, window T of
        the filter check, element size), so a resumed or loaded study asks
        for exactly the runs it made."""
        cfg = self._study_cfg_copy()
        w = spec.get("window")
        if w:
            cfg.optimization.window_start = "%.12g" % float(w[0])
            cfg.optimization.window_end = "%.12g" % float(w[1])
        bm = spec.get("base_ms")
        if bm is not None:
            cfg.step.mass_scaling_enabled = bool(bm[0])
            cfg.step.mass_scaling_factor = float(bm[1])
        fv = spec.get("filter_verify")
        if fv is not None:
            cfg.step.output_filter_verify = bool(fv)
        if elem is not None:
            cfg.elem_size = float(elem)
        return cfg

    @staticmethod
    def _zoi_dict(zoi):
        return {k: float(v) for k, v in zip(ZOI_KEYS, zoi)}

    @staticmethod
    def _dims_dict(d):
        return {k: float(getattr(d, k)) for k in DIM_KEYS}

    def _ms_spec(self):
        """Settings of a new mass-scaling study from the panel, or None
        after a warning."""
        if not self.thresholds_complete():
            QMessageBox.warning(
                self, "Mass-scaling criterion",
                "Set the six absolute tolerances eps_q of the common settings "
                "(Vx, Vy, T, EVF, Fc, Ff): the ms study uses the same E_max.")
            return None
        try:
            window = self.window()
            guards = self.guard_settings()
            ms_values, elem = self.ms_settings()
        except ValueError as e:
            QMessageBox.warning(self, "Mass-scaling study settings", str(e))
            return None
        if not getattr(self.cfg.step, "output_filter_enabled", False):
            QMessageBox.warning(
                self, "Mass-scaling study",
                "Enable the output filter (Step tab): the filter and "
                "reverberation checks are safeguards of the ms study.")
            return None
        return {
            "zoi": self._zoi_dict(self.zoi()),
            "elem_size": elem, "ms_values": list(ms_values),
            "grid_step": self.grid_step(),
            "grid_step_set": self._grid_step_set(),
            "thresholds_abs": self.thresholds(),
            "window": list(window), "evf_threshold": 0.5,
            "rk_max": guards.rk_max, "rhg_max": guards.rhg_max,
            "domain_dims": self._dims_dict(self._dims_from_cfg()),
            "filter_verify": True, "base_ms": self._base_ms()}

    def _on_run_ms_independence(self):
        if self.is_running():
            return
        if self._offer_resume("ms"):
            return
        self._launch_new("ms")

    def _start_ms(self, spec, run_dir, prefs, cpus, mode="new", then=None,
                  **extra):
        if self._cannot_start("The mass-scaling study"):
            return False
        zoi = zoi_tuple(spec)
        dims = DomainDims(**{k: float(spec["domain_dims"][k])
                             for k in DIM_KEYS})
        elem = float(spec["elem_size"])
        ms_values = tuple(float(v) for v in spec["ms_values"])
        thr = {k: float(v) for k, v in spec["thresholds_abs"].items()}
        window = tuple(float(v) for v in spec["window"])
        grid = float(spec["grid_step"])
        guards = GuardSettings(rk_max=float(spec["rk_max"]),
                               rhg_max=float(spec["rhg_max"]), window=window)
        base_cfg = self._study_base_cfg(spec)
        base_cfg.step.output_filter_verify = True
        run_bundle = self._begin_study("ms", spec, run_dir, prefs, cpus, mode,
                                       then, **extra)
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
        self._busy(True, "Mass-scaling study (independence)\u2026"
                   if mode != "load" else "Reading back a mass-scaling "
                   "study\u2026")
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
        self._retire_worker("_ms_worker")
        self._ms_worker = MsIndependenceWorker(
            run_bundle=run_bundle, base_cfg=base_cfg, zoi=zoi,
            domain_dims=dims, grid_step=grid, elem_size=elem,
            thresholds=thr, ms_values=ms_values, window=window,
            evf_threshold=float(spec.get("evf_threshold", 0.5)),
            guard_fn=guard_fn, cost_fn=cost_fn)
        self._ms_worker.progress.connect(self._on_ms_progress)
        self._ms_worker.finished_ok.connect(self._on_ms_done)
        self._ms_worker.failed.connect(self._on_fail)
        self._start_progress()
        self._ms_worker.start()
        return True

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
        if folder is not None and self._exports_allowed("ms", res):
            from gui.sensitivity.study_export import write_ms_exports
            try:
                paths = write_ms_exports(folder, res)
                self._log_ui("[EXPORT] %s -> %s"
                             % (", ".join(p.name for p in paths), folder))
            except Exception as e:
                self._log_ui("[EXPORT] failed: %s: %s"
                             % (type(e).__name__, e))
        self._refresh_convergence_view()
        ok = res.status == "converged"
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#15803d" if ok else "#b45309"))
        self.lbl_status.setText("Mass-scaling study \u2014 %s" % why)
        self._after_study("ms", res)

    # ===================================================================
    # 4 - Eulerian domain sizing: sequential independence study (paper §4)
    # ===================================================================
    def _domain_spec(self):
        """Settings of a new domain study from the panel, or None after a
        warning."""
        if not self.thresholds_complete():
            QMessageBox.warning(
                self, "Domain criterion",
                "Set the six absolute tolerances eps_q of the common settings "
                "(Vx, Vy, T, EVF, Fc, Ff): they define E_max for the domain "
                "study.")
            return None
        try:
            window = self.window()
            guards = self.guard_settings()
            ds = self.domain_settings()
        except ValueError as e:
            QMessageBox.warning(self, "Domain study settings", str(e))
            return None
        dims0 = self.compute_initial_dims()
        return {
            "zoi": self._zoi_dict(self.zoi()),
            "elem_size": float(self.cfg.elem_size),
            "margin_elems": int(self.sp_margin.value()),
            "euler_offset": list(self.euler_offset()),
            "grid_step": self.grid_step(),
            "grid_step_set": self._grid_step_set(),
            "thresholds_abs": self.thresholds(),
            "window": list(window), "evf_threshold": 0.5,
            "step_elems": ds["dom_step_elems"], "n_max": ds["dom_n_max"],
            "n_hold": ds["dom_n_hold"], "m_ratios": ds["dom_m_ratios"],
            "rk_max": guards.rk_max, "rhg_max": guards.rhg_max,
            "caps": self.caps(),
            "initial_dims": self._dims_dict(dims0),
            "base_ms": self._base_ms(),
            "filter_verify": bool(getattr(self.cfg.step,
                                          "output_filter_verify", True))}

    def _on_run_domain_independence(self):
        if self.is_running():
            return
        if self._offer_resume("domain"):
            return
        self._launch_new("domain")

    def _start_domain(self, spec, run_dir, prefs, cpus, mode="new",
                      then=None, **extra):
        if self._cannot_start("The domain study"):
            return False
        zoi = zoi_tuple(spec)
        elem = float(spec["elem_size"])
        offset = tuple(float(v) for v in spec["euler_offset"])
        margin = int(spec["margin_elems"])
        dims0 = DomainDims(**{k: float(spec["initial_dims"][k])
                              for k in DIM_KEYS})
        caps = {k: float(v) for k, v in (spec.get("caps") or {}).items()}
        thr = {k: float(v) for k, v in spec["thresholds_abs"].items()}
        window = tuple(float(v) for v in spec["window"])
        grid = float(spec["grid_step"])
        guards = GuardSettings(rk_max=float(spec["rk_max"]),
                               rhg_max=float(spec["rhg_max"]), window=window)
        ds = {"dom_step_elems": int(spec["step_elems"]),
              "dom_n_max": int(spec["n_max"]),
              "dom_n_hold": int(spec["n_hold"]),
              "dom_m_ratios": int(spec["m_ratios"])}
        # The runs take the element size from the cfg (the core only uses
        # its elem_size argument to sample), so both are the study's.
        base_cfg = self._study_base_cfg(spec, elem=elem)
        run_bundle = self._begin_study("domain", spec, run_dir, prefs, cpus,
                                       mode, then, **extra)
        self._pending_domain_dir = run_dir
        self._pending_domain_spec = spec
        bm = spec.get("base_ms")
        self._pending_domain_ms = None if bm is None else ms_of(bm)

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
        self._busy(True, "Domain sizing (independence)\u2026"
                   if mode != "load" else "Reading back a domain study\u2026")
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
        self._retire_worker("_di_worker")
        self._di_worker = DomainIndependenceWorker(
            run_bundle=run_bundle, base_cfg=base_cfg, zoi=zoi,
            initial_dims=dims0, grid_step=grid, elem_size=elem,
            thresholds=thr, window=window,
            evf_threshold=float(spec.get("evf_threshold", 0.5)),
            step_elems=ds["dom_step_elems"], n_max=ds["dom_n_max"],
            n_hold=ds["dom_n_hold"], m_ratios=ds["dom_m_ratios"],
            caps=caps, margin_elems=margin, offset=offset,
            guard_fn=guard_fn, cost_fn=cost_fn)
        self._di_worker.progress.connect(self._on_di_progress)
        self._di_worker.finished_ok.connect(self._on_di_done)
        self._di_worker.failed.connect(self._on_fail)
        self._start_progress()
        self._di_worker.start()
        return True

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
        self._last_domain_ms = getattr(self, "_pending_domain_ms", None)
        self._last_domain_spec = getattr(self, "_pending_domain_spec", None)
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
        if self._exports_allowed("domain", res):
            self._write_domain_exports()
        self._refresh_convergence_view()
        ok = res.status == "converged"
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#15803d" if ok else "#b45309"))
        self.lbl_status.setText("Domain sizing \u2014 %s" % why)
        self._after_study("domain", res)

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
            # Log axes only around finite points: with every E_max NaN
            # (nothing sampled) the empty log axis fails to draw.
            if any(math.isfinite(y) and y > 0 for y in ys):
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
            gres, gtol = self._last_gci[0], self._last_gci[2] or {}
            gci_pts = False
            for q, g in gres.per_quantity.items():
                ref = g.f_extrapolated if g.reliable else g.f_fine
                eps = gtol.get(q)
                hs = [h for h in gres.sizes if q in gres.scalars.get(h, {})]
                ys = []
                for h in hs:
                    v = gres.scalars[h][q]
                    ys.append(abs(v - ref) / eps if (
                        v is not None and eps and math.isfinite(ref))
                        else float("nan"))
                if hs and eps:
                    axg.plot(hs, ys, marker="s", lw=1.0, label=q)
                    gci_pts = gci_pts or any(math.isfinite(y) for y in ys)
            axg.axhline(1.0, ls="--", lw=1.0, color="#b91c1c")
            if gci_pts:
                axg.set_xscale("log")
            axg.set_xlabel("h [mm]", fontsize=7)
            axg.set_ylabel("|f_q(h) - f_ref| / eps_q", fontsize=7)
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
        # The ms the runs were made with (a resumed study keeps its own).
        ms = getattr(self, "_last_domain_ms", None)
        try:
            paths = write_domain_exports(
                folder, res, self._t1(), res.settings.get("elem_size"),
                self._ms_factor() if ms is None else ms, self._last_checks)
        except Exception as e:
            self._log_ui("[EXPORT] failed: %s: %s" % (type(e).__name__, e))
            return []
        self._log_ui("[EXPORT] %s -> %s" % (", ".join(p.name for p in paths),
                                            folder))
        return paths

    def ms_lower_for_checks(self):
        """The ms value before the current ms* in the ms study (the one of
        record when step 0 is done for this model, else the last one run
        here), or None (the check then uses ms*/2)."""
        values = None
        st, rec = self._step_state("ms")
        if st == "done":
            values = rec.get("values")
        if not values and self._last_ms is not None:
            values = list(self._last_ms[0].ms_values)
        if not values:
            return None
        ms = self._ms_factor()
        for a, b in zip(values[:-1], values[1:]):
            if math.isclose(float(b), ms, rel_tol=1e-5):
                return float(a)
        return None

    # ===================================================================
    # 6 - Interaction checks (paper §5.7, report T9)
    # ===================================================================
    def _checks_spec(self, study):
        """Settings of the final checks on `study` (StudyResult of step 2),
        or None after a warning."""
        try:
            guards = self.guard_settings()
            window = [float(v) for v in study.settings["window"]]
        except ValueError as e:
            QMessageBox.warning(self, "Interaction checks", str(e))
            return None
        if not self.thresholds_complete():
            QMessageBox.warning(
                self, "Interaction checks",
                "Set the six absolute tolerances eps_q of the common settings "
                "(Vx, Vy, T, EVF, Fc, Ff): the GCI selects the mesh with them.")
            return None
        h_star = float(study.settings["elem_size"])
        zoi = [float(v) for v in study.settings["zoi"]]
        plan = self._gci_plan_for_checks(h_star)
        gci_plan = dict(plan, zoi=zoi,
                        grid_step=float(study.settings["grid_step"]),
                        field_vars=["EVF", "TEMP", "V1", "V2"],
                        window=window, evf_threshold=0.5)
        d = study.final
        dspec = getattr(self, "_last_domain_spec", None)
        filter_on = bool(getattr(self.cfg.step, "output_filter_enabled",
                                 False))
        return {
            "h_star": h_star,
            "d_star": [float(d.h_wp), float(d.h_void), float(d.l_wp),
                       float(d.l_void)],
            "gci_plan": gci_plan, "gci_tolerances": self._gci_tolerances(),
            "zoi": zoi,
            "thresholds_abs": {k: float(v) for k, v in
                               study.settings["thresholds"].items()},
            "window": window, "rk_max": guards.rk_max,
            "rhg_max": guards.rhg_max, "ms_lower": self.ms_lower_for_checks(),
            # As the step-2 study had it: the checks sample like it did.
            "grid_step_set": (self._grid_step_set() if dspec is None
                              else grid_set_of_spec(dspec)),
            "base_ms": self._base_ms(),
            # The ms_at_point check reads the filter and reverberation
            # checks, which need the verification output.
            "filter_verify": True if filter_on else bool(getattr(
                self.cfg.step, "output_filter_verify", True))}

    def _on_run_interaction_checks(self):
        if self.is_running():
            return
        if self._offer_resume("checks"):
            return
        if not self._checks_available():
            QMessageBox.warning(self, "Interaction checks",
                                "Run a domain study first.")
            return
        if not self._confirm_prerequisites("checks"):
            return
        self._run_checks("new")

    def _start_checks(self, spec, study, folder, prefs, cpus, mode="new",
                      then=None, **extra):
        if self._cannot_start("The final checks"):
            return False
        h_star = float(spec["h_star"])
        p = dict(spec["gci_plan"])
        window = tuple(float(v) for v in spec["window"])
        gci_plan = {
            "zoi": tuple(float(v) for v in p["zoi"]),
            "grid_step": float(p["grid_step"]),
            "finest_elem_size": float(p["finest_elem_size"]),
            "ratio": float(p["ratio"]), "n_meshes": int(p["n_meshes"]),
            "min_elem_size": (None if p.get("min_elem_size") is None
                              else float(p["min_elem_size"])),
            "field_vars": tuple(p.get("field_vars")
                                or ("EVF", "TEMP", "V1", "V2")),
            "window": tuple(float(v) for v in p.get("window") or window),
            "evf_threshold": float(p.get("evf_threshold", 0.5))}
        gci_tol = {k: float(v) for k, v in spec["gci_tolerances"].items()}
        guards = GuardSettings(rk_max=float(spec["rk_max"]),
                               rhg_max=float(spec["rhg_max"]), window=window)
        ms_lower = spec.get("ms_lower")
        ms_lower = None if ms_lower is None else float(ms_lower)
        # c1 and c4 take the element size and the mass scaling from this
        # cfg: those of the checked point (h*, ms*), whatever the panel.
        base_cfg = self._study_base_cfg(spec, elem=h_star)
        if mode != "load":
            # The folder may hold the settings of a valid result of the
            # checks: kept, to be put back if this run gives no result.
            try:
                extra["checks_config_before"] = (
                    Path(folder) / CHECKS_CONFIG).read_text(encoding="utf-8")
            except OSError:
                pass
            write_checks_config(folder, spec)
        run_bundle = self._begin_study("checks", spec, folder, prefs, cpus,
                                       mode, then, **extra)

        def cost_fn(bundle, dims, host_wall_s):
            return cost_record(bundle, run_bundle.state.get("sta"),
                               host_wall_s=host_wall_s, n_cpu=cpus,
                               dims=dims, elem_size=h_star)

        from gui.sensitivity.interaction_checks_worker import (
            InteractionChecksWorker)
        from gui.sensitivity.run_record import RecordingRunner
        self._busy(True, "Interaction checks\u2026" if mode != "load"
                   else "Reading back the final checks\u2026")
        self._log_ui("=" * 68)
        self._log_ui("INTERACTION CHECKS on h*=%.4g mm, D*: h_wp=%.4g "
                     "h_void=%.4g l_wp=%.4g l_void=%.4g"
                     % (h_star, study.final.h_wp, study.final.h_void,
                        study.final.l_wp, study.final.l_void))
        self._log_ui("  GCI plan on D*: finest %.4g | ratio %.3g | n %d"
                     % (gci_plan["finest_elem_size"], gci_plan["ratio"],
                        gci_plan["n_meshes"]))
        self._log_ui("=" * 68)
        # Check ms_at_point: ms* against the value before it in the ms
        # study (ms*/2 without one), with the ms study's safeguards (filter
        # and reverberation checks) when the output filter is on.
        guard_core = make_guard_fn(guards)
        ms_guard_fn = None
        if getattr(base_cfg.step, "output_filter_enabled", False):

            def ms_guard_fn(bundle):
                out = dict(guard_core(bundle))
                out.update(filter_guards(run_bundle.state.get("filter_check")))
                return out
        else:
            self._log_ui("  ms_at_point: output filter off, the filter and "
                         "reverberation safeguards are not evaluated")
        ms_star = (float(base_cfg.step.mass_scaling_factor)
                   if base_cfg.step.mass_scaling_enabled else 1.0)
        self._log_ui("  ms_at_point: ms* = %g against ms = %s"
                     % (ms_star, "%g" % ms_lower
                        if ms_lower is not None else "ms*/2"))
        self._retire_worker("_checks_worker")
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
        return True

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
        if self._exports_allowed("checks", res):
            self._write_domain_exports()
        self._refresh_convergence_view()
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#15803d" if res.status == "accepted"
                            else "#b45309"))
        self.lbl_status.setText("Interaction checks \u2014 %s" % why)
        self._after_study("checks", res)

    # ===================================================================
    # 3 - Mesh convergence by GCI / Richardson (fixed domain)
    # ===================================================================
    def _gci_spec(self):
        """Settings of a new mesh-convergence study from the panel, or None
        after a warning."""
        if not self.thresholds_complete():
            QMessageBox.warning(
                self, "Mesh convergence criterion",
                "Set the six absolute tolerances eps_q of the common settings "
                "(Vx, Vy, T, EVF, Fc, Ff): the GCI selects the mesh with them.")
            return None
        try:
            window = self.window()
        except ValueError as e:
            QMessageBox.warning(self, "Time window", str(e))
            return None
        try:
            guards = self.guard_settings()
        except ValueError as e:
            QMessageBox.warning(self, "Safeguards", str(e))
            return None
        return {
            "zoi": self._zoi_dict(self.zoi()),
            "window": list(window),
            "finest_elem_size": self._float_or(self.le_gci_finest,
                                               float(self.cfg.elem_size)),
            "ratio": self._float_or(self.le_gci_ratio, 2.0),
            "n_meshes": int(self.sp_gci_n.value()),
            "min_elem_size": self._float_or(self.le_gci_min, None),
            "grid_step": self.grid_step(),
            "grid_step_set": self._grid_step_set(),
            "tolerances": self._gci_tolerances(),
            "field_vars": ["EVF", "TEMP", "V1", "V2"],
            "evf_threshold": 0.5,
            "rk_max": guards.rk_max, "rhg_max": guards.rhg_max,
            "domain_dims": self._dims_dict(self._dims_from_cfg()),
            "base_ms": self._base_ms(),
            "filter_verify": bool(getattr(self.cfg.step,
                                          "output_filter_verify", True))}

    def _on_run_mesh_gci(self):
        if self.is_running():
            return
        if self._offer_resume("mesh"):
            return
        self._launch_new("mesh")

    def _start_gci(self, spec, run_dir, prefs, cpus, mode="new", then=None,
                   **extra):
        if self._cannot_start("The mesh study"):
            return False
        zoi = zoi_tuple(spec)
        window = tuple(float(v) for v in spec["window"])
        finest = float(spec["finest_elem_size"])
        ratio = float(spec["ratio"])
        nmesh = int(spec["n_meshes"])
        minh = spec.get("min_elem_size")
        minh = None if minh is None else float(minh)
        gci_tol = {k: float(v) for k, v in spec["tolerances"].items()}
        dims = DomainDims(**{k: float(spec["domain_dims"][k])
                             for k in DIM_KEYS})
        grid = float(spec["grid_step"])
        guards = GuardSettings(rk_max=float(spec["rk_max"]),
                               rhg_max=float(spec["rhg_max"]), window=window)
        # A deep copy: run_mesh_gci sets elem_size and the domain on the cfg
        # it receives (mesh_gci.py:337-341); given self.cfg it used to leave
        # the user's model at the coarsest element size after the study.
        base_cfg = self._study_base_cfg(spec)
        run_bundle = self._begin_study("mesh", spec, run_dir, prefs, cpus,
                                       mode, then,
                                       plan={"finest_elem_size": finest,
                                             "ratio": ratio,
                                             "n_meshes": nmesh,
                                             "min_elem_size": minh},
                                       **extra)
        # T10: every GCI run records its cost and safeguards (mesh_gci has no
        # hook of its own); the records feed gci_meshes.csv (paper Table 8).
        from gui.sensitivity.run_record import RecordingRunner
        recorder = RecordingRunner(run_bundle, n_cpu=cpus,
                                   guard_settings=guards)
        self._pending_gci = (recorder, gci_tol, run_dir)
        self._busy(True, "Mesh convergence (GCI)\u2026" if mode != "load"
                   else "Reading back a mesh convergence study\u2026")
        self._log_ui("=" * 68)
        self._log_ui("MESH CONVERGENCE (GCI / Richardson) on a fixed domain")
        self._log_ui("  finest %.4g mm | ratio %.3g | n %d | floor %s | "
                     "T [%.3g, %.3g]"
                     % (finest, ratio, nmesh,
                        "n/a" if minh is None else "%.4g" % minh,
                        window[0], window[1]))
        self._log_ui("=" * 68)
        self._retire_worker("_mesh_worker")
        self._mesh_worker = MeshGciWorker(
            run_bundle=recorder, base_cfg=base_cfg, zoi=zoi,
            domain_dims=dims,
            grid_step=grid, finest_elem_size=finest, ratio=ratio,
            n_meshes=nmesh, tolerances=(gci_tol or None),
            field_vars=tuple(spec.get("field_vars")
                             or ("EVF", "TEMP", "V1", "V2")),
            window=window,
            evf_threshold=float(spec.get("evf_threshold", 0.5)),
            min_elem_size=minh)
        self._mesh_worker.progress.connect(self._on_mesh_progress)
        self._mesh_worker.finished_ok.connect(self._on_mesh_done)
        self._mesh_worker.failed.connect(self._on_fail)
        self._start_progress()
        self._mesh_worker.start()
        return True

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
        if folder is not None and self._exports_allowed("mesh", res):
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
        self._after_study("mesh", res)

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
        if self._pipeline:
            self._pipeline_stop("cancelled")
        self.lbl_status.setText("Cancelling the current run\u2026")

        job, proc = self._current_job, self._current_proc
        cmd, run_dir = self._current_abaqus_cmd, self._current_run_dir
        if not (job and cmd):
            return              # nothing started yet: the flag is enough
        if isinstance(proc, RemoteProcess):
            # The agent runs `abaqus terminate` (then the tree kill) itself.
            self._log_ui("[CANCEL] asking the remote agent to stop job %s" % job)
            run_async(proc.cancel, lambda _r: None, self)
            return
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
        act = getattr(self, "_active", None)
        state = getattr((act or {}).get("run_bundle"), "state", None) or {}
        if state.get("miss") is None:
            # (A study read back that stopped on a missing run is no
            # error: _after_study says what the folder lacks.)
            self.lbl_status.setStyleSheet("color: #b91c1c;")
            self.lbl_status.setText("Study failed: %s" % msg)
            self._log_ui("ERROR: %s" % msg)
        if act is not None:
            self._after_study(act["step"], None, error=msg)
