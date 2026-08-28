# -*- coding: utf-8 -*-
"""
Optimization tab: minimise the Eulerian domain (h_wp, h_void, l_wp, l_void)
while keeping the ROI fields close to a self-converged large domain.

Pipeline:
  1. initial domain from Merchant (gui.core.domain_sizing) using the current
     model config (t1 = wp_y0 - tool_y0, rake, friction, ROI = bbox);
  2. DomainOptimizer (gui.sensitivity.domain_opt) grows each dimension by
     doubling+bisection until every ROI-field error E_q < eps_q;
  3. each candidate domain is one Abaqus run (run_simul) via a background
     DomainOptWorker; ROI samples are extracted at the (anchored) element
     centroids and compared at identical points.

Only the four EulerGeometry dimensions change between candidate runs; every
other model setting is taken from the current config. The Abaqus launcher is
built here (replicating the Sensitivity tab's run mechanism); the optimiser
core and the extraction are fully unit-tested elsewhere.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QLineEdit, QPlainTextEdit, QTabWidget, QSpinBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar,
)
from PySide6.QtGui import QDoubleValidator, QDesktopServices
from PySide6.QtCore import QUrl

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui.core.domain_sizing import DomainDims, initial_domain_dimensions, \
    merchant_shear_angle, chip_thickness, shear_band_bracket
from gui.core.logging_util import log_swallowed
from gui.core.sta_parser import parse_sta
from gui.sensitivity.mesh_pipeline_worker import MeshPipelineWorker
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
# Force quantities -> the tool-RP reaction-force history channel.
_FORCE_CHANNELS = {"Fc": "RF1_RP", "Ff": "RF2_RP"}
_DIM_ORDER = ("l_wp", "h_wp", "h_void", "l_void")


class OptimizationTab(QWidget):

    def __init__(self, cfg, prefs_getter=None, cpus_getter=None):
        super().__init__()
        self.cfg = cfg
        self._prefs_getter = prefs_getter
        self._cpus_getter = cpus_getter
        self._initial = None            # DomainDims from Merchant
        self._cancel_evt = threading.Event()
        self._hist = {}                 # param key -> list of (value, {q: E_q})
        self._current_sta = None        # current job's .sta path (for progress)
        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(500)
        self._sim_timer.timeout.connect(self._poll_sta)

        root = QVBoxLayout(self)

        # (Inputs-from-model panel removed as requested — the values are read
        # directly from the config when needed.)

        # ---- Initial domain = the measurement ROI ----------------------
        gdom = QGroupBox("Initial eulerian domain")
        dg = QGridLayout(gdom)
        self._max = {}
        self._init_lbl = {}
        # dim | initial (read-only) | max cap | mm, on two column groups.
        dg.addWidget(QLabel("dim"), 0, 0)
        dg.addWidget(QLabel("initial"), 0, 1)
        dg.addWidget(QLabel("max cap"), 0, 2)
        dg.addWidget(QLabel("dim"), 0, 5)
        dg.addWidget(QLabel("initial"), 0, 6)
        dg.addWidget(QLabel("max cap"), 0, 7)
        _pairs = [("l_wp", "h_void"), ("h_wp", "l_void")]
        for r, (d_left, d_right) in enumerate(_pairs, start=1):
            for d, c0 in ((d_left, 0), (d_right, 5)):
                dg.addWidget(QLabel(d), r, c0)
                il = QLabel("—"); il.setStyleSheet("color:#374151;")
                self._init_lbl[d] = il
                dg.addWidget(il, r, c0 + 1)
                mx = QLineEdit(); mx.setPlaceholderText("no cap")
                self._max[d] = mx
                dg.addWidget(mx, r, c0 + 2)
                dg.addWidget(QLabel("mm"), r, c0 + 3)
        r0 = 3
        dg.addWidget(QLabel("Margin (elems):"), r0, 0)
        self.sp_margin = QSpinBox(); self.sp_margin.setRange(0, 50)
        self.sp_margin.setValue(0)
        dg.addWidget(self.sp_margin, r0, 1)
        dg.addWidget(QLabel("Centroid step:"), r0, 5)
        self.le_grid_step = QLineEdit(); self.le_grid_step.setPlaceholderText(
            "= element size")
        dg.addWidget(self.le_grid_step, r0, 6)
        dg.addWidget(QLabel("mm"), r0, 7)
        self.btn_init = QPushButton("Compute initial domain")
        self.btn_init.clicked.connect(self.compute_initial)
        dg.addWidget(self.btn_init, r0 + 1, 5, 1, 4)
        self.lbl_init = QLabel("—")
        self.lbl_init.setStyleSheet("font-weight: bold;")
        dg.addWidget(self.lbl_init, r0 + 1, 0, 1, 4)
        # (added to the left column below)

        # ---- Convergence criterion -------------------------------------
        gcrit = QGroupBox("Convergence criterion")
        cg = QGridLayout(gcrit)
        cg.addWidget(QLabel("quantity"), 0, 0)
        cg.addWidget(QLabel("ε_q"), 0, 1)
        cg.addWidget(QLabel("unit"), 0, 2)
        cg.addWidget(QLabel("quantity"), 0, 4)
        cg.addWidget(QLabel("ε_q"), 0, 5)
        cg.addWidget(QLabel("unit"), 0, 6)
        self._q_eps = {}
        _unit = {q: u for (q, _f, u) in _QUANTITIES}
        _cols = [["Vx", "Vy", "T"], ["EVF", "Fc", "Ff"]]
        for r in range(3):
            for col, c0 in ((_cols[0], 0), (_cols[1], 4)):
                q = col[r]
                lbl = QLabel(q)
                if q in ("Fc", "Ff"):
                    rf = "RF1" if q == "Fc" else "RF2"
                    lbl.setToolTip(
                        "%s = %s on the tool RP, divided by the element size "
                        "(homogeneity across meshes), scalar per time (N_p=1)."
                        % (q, rf))
                cg.addWidget(lbl, r + 1, c0)
                le = QLineEdit()
                le.setPlaceholderText("threshold")
                self._q_eps[q] = le
                cg.addWidget(le, r + 1, c0 + 1)
                cg.addWidget(QLabel(_unit[q]), r + 1, c0 + 2)

        # ---- Optimisation pipeline (per-variable start / factor / cap) --
        gmesh = QGroupBox("Optimisation pipeline")
        mg = QGridLayout(gmesh)
        for c, h in enumerate(["variable", "start", "factor", "cap",
                               "bisection res"]):
            mg.addWidget(QLabel(h), 0, c)

        # Step 0 - mass scaling (cap = MAX; growth factor > 1)
        self.cb_identify_ms = QCheckBox("mass scaling")
        self.cb_identify_ms.setToolTip(
            "Step 0 \u2014 criterion on Vx/Vy; keeps the LARGEST factor "
            "admissible under the energy guard (least costly).")
        self.cb_identify_ms.setChecked(False)
        self.cb_identify_ms.toggled.connect(self._enforce_pipeline_selection)
        mg.addWidget(self.cb_identify_ms, 1, 0)
        self.le_ms_start = QLineEdit("1"); mg.addWidget(self.le_ms_start, 1, 1)
        self.le_ms_factor = QLineEdit("2")
        mg.addWidget(self.le_ms_factor, 1, 2)
        self.le_ms_max = QLineEdit(); self.le_ms_max.setPlaceholderText("max")
        mg.addWidget(self.le_ms_max, 1, 3)
        self.le_ms_bisect = QLineEdit()
        self.le_ms_bisect.setPlaceholderText("e.g. 100")
        mg.addWidget(self.le_ms_bisect, 1, 4)
        mg.addWidget(QLabel("energy guard \u27e8ALLKE/ALLIE\u27e9 <"), 2, 0, 1, 2)
        self.le_ms_guard = QLineEdit()
        self.le_ms_guard.setPlaceholderText("e.g. 0.05")
        mg.addWidget(self.le_ms_guard, 2, 2, 1, 3)

        # wp elem (cap = MIN, finest; shrink factor < 1) — optional
        self.cb_include_wp = QCheckBox("wp elem")
        self.cb_include_wp.setChecked(True)
        self.cb_include_wp.toggled.connect(self._enforce_pipeline_selection)
        mg.addWidget(self.cb_include_wp, 3, 0)
        self.le_wp_start = QLineEdit("0.01")
        mg.addWidget(self.le_wp_start, 3, 1)
        self.le_wp_factor = QLineEdit("0.5")
        mg.addWidget(self.le_wp_factor, 3, 2)
        self.le_wp_min = QLineEdit(); self.le_wp_min.setPlaceholderText("min")
        mg.addWidget(self.le_wp_min, 3, 3)
        self.le_wp_bisect = QLineEdit()
        self.le_wp_bisect.setPlaceholderText("e.g. 0.001")
        mg.addWidget(self.le_wp_bisect, 3, 4)

        # tool elem / nose (cap = MIN, finest; shrink factor < 1) — optional
        self.cb_include_tool = QCheckBox("tool elem")
        self.cb_include_tool.setChecked(True)
        self.cb_include_tool.toggled.connect(self._enforce_pipeline_selection)
        mg.addWidget(self.cb_include_tool, 4, 0)
        self.le_tool_start = QLineEdit("0.005")
        mg.addWidget(self.le_tool_start, 4, 1)
        self.le_tool_factor = QLineEdit("0.5")
        mg.addWidget(self.le_tool_factor, 4, 2)
        self.le_tool_min = QLineEdit()
        self.le_tool_min.setPlaceholderText("min")
        mg.addWidget(self.le_tool_min, 4, 3)
        self.le_tool_bisect = QLineEdit()
        self.le_tool_bisect.setPlaceholderText("e.g. 0.001")
        mg.addWidget(self.le_tool_bisect, 4, 4)

        # Eulerian domain (start = ROI; cap = the max caps above; factor > 1)
        self.cb_include_domain = QCheckBox("euler dim")
        self.cb_include_domain.setChecked(True)
        self.cb_include_domain.toggled.connect(self._enforce_pipeline_selection)
        mg.addWidget(self.cb_include_domain, 5, 0)
        self.le_domain_factor = QLineEdit("2")
        mg.addWidget(self.le_domain_factor, 5, 2)
        _mc = QLabel("= max caps \u2191"); _mc.setStyleSheet("color:#6b7280;")
        mg.addWidget(_mc, 5, 3)
        self.le_domain_bisect = QLineEdit()
        self.le_domain_bisect.setPlaceholderText("e.g. 0.01")
        mg.addWidget(self.le_domain_bisect, 5, 4)

        self.cb_do_verify = QCheckBox(
            "verification passes (bracket each identified value by "
            "\u00b1resolution)")
        self.cb_do_verify.setChecked(True)
        mg.addWidget(self.cb_do_verify, 6, 0, 1, 5)
        # NOTE: the single launcher is the "Run optimization" button in the
        # bottom run-controls bar (wired to the checkbox-driven pipeline). The
        # former in-group "Run full pipeline" button was removed to avoid two
        # launchers with different semantics.

        # ---- Preview (reuses the Geometry tab's preview widget) --------
        gprev = QGroupBox("Preview")
        pv = QVBoxLayout(gprev)
        self.preview = GeometryPreview()
        pv.addWidget(self.preview, 1)
        btn_prev = QPushButton("Refresh preview")
        btn_prev.clicked.connect(self._draw_preview)
        pv.addWidget(btn_prev)

        # ---- Two-column assembly ---------------------------------------
        # Left: inputs, initial domain, then convergence criterion (stacked).
        # Right: the whole preview.
        cols = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(gdom)
        left.addWidget(gcrit)
        left.addWidget(gmesh)
        left.addStretch(1)
        cols.addLayout(left, 1)
        cols.addWidget(gprev, 1)
        root.addLayout(cols)

        # ---- Run controls ----------------------------------------------
        rc = QHBoxLayout()
        self.btn_run = QPushButton("Run optimization")
        self.btn_run.setToolTip(
            "Run the pipeline steps selected by the checkboxes above "
            "(mass scaling, wp/tool element size, Eulerian domain, "
            "verification). Only the checked steps are executed.")
        # Single launcher: drives the checkbox-driven pipeline worker.
        self.btn_run.clicked.connect(self._on_run_pipeline)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_pipeline)
        rc.addWidget(self.btn_run)
        self.btn_open_wd = QPushButton("Open working dir")
        self.btn_open_wd.setToolTip("Open the Preferences working directory in "
                                    "the file explorer.")
        self.btn_open_wd.clicked.connect(self._open_working_dir)
        rc.addWidget(self.btn_open_wd)
        rc.addWidget(self.btn_cancel)
        self.lbl_status = QLabel("")
        rc.addWidget(self.lbl_status, 1)
        root.addLayout(rc)

        # Live per-simulation progress bar (fed by parsing the current job's
        # .sta file, step_time/sim_time).
        self.sim_progress = QProgressBar()
        self.sim_progress.setRange(0, 100)
        self.sim_progress.setFormat("current simulation: %p%")
        self.sim_progress.setVisible(False)
        root.addWidget(self.sim_progress)

        # ---- Output tabs -----------------------------------------------
        self.tabs = QTabWidget()
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "Log")
        conv = QWidget(); cv = QVBoxLayout(conv)
        self.fig = Figure(figsize=(6, 4.5))
        self.canvas = FigureCanvas(self.fig)
        # Matplotlib navigation toolbar: interactive zoom / pan / home / save.
        self._nav = NavigationToolbar2QT(self.canvas, conv)
        cv.addWidget(self._nav)
        # One axis per identified parameter: mass scaling, wp element size and
        # tool element size get their own axis (incompatible value scales); the
        # four Eulerian domain dimensions share a single axis (same mm scale).
        (self._ax_ms, self._ax_wp), (self._ax_tool, self._ax_domain) = \
            self.fig.subplots(2, 2)
        # Route a single-parameter history key to its dedicated axis.
        self._param_axes = {
            "mass_scaling": self._ax_ms,
            "wp_elem": self._ax_wp,
            "tool_elem": self._ax_tool,
        }
        self._ax_titles = {
            id(self._ax_ms): "mass scaling",
            id(self._ax_wp): "wp element size",
            id(self._ax_tool): "tool element size",
            id(self._ax_domain): "Eulerian domain",
        }
        cv.addWidget(self.canvas, 1)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["parameter", "initial", "intermediate", "final"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        cv.addWidget(self.table)
        self.tabs.addTab(conv, "Convergence")
        root.addWidget(self.tabs, 1)

        # Auto-refresh the preview when the inputs that affect it change.
        for _d in _DIM_ORDER:
            self._max[_d].textChanged.connect(self._draw_preview)
        self.le_grid_step.textChanged.connect(self._draw_preview)
        self.sp_margin.valueChanged.connect(self._draw_preview)

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

    def wp_min(self):
        """Cap of the workpiece element size = MINIMUM (finest allowed)."""
        return self._float_or(self.le_wp_min, None)

    def tool_min(self):
        """Cap of the tool-nose element size = MINIMUM (finest allowed)."""
        return self._float_or(self.le_tool_min, None)

    def compute_initial_dims(self) -> DomainDims:
        """Initial Eulerian domain = the user measurement ROI (BBox), mapped to
        the domain dimensions with the run_simul convention (material
        x in [-l_wp, 0], y in [-h_wp, 0]; void x in [0, l_void], y in [0,
        h_void]), snapped up to whole elements (+ optional margin):
            l_wp = -xmin, l_void = xmax, h_wp = -ymin, h_void = ymax.
        Negative sides (ROI not straddling the axes) are floored at 0."""
        inp = self.config_inputs()
        xmin, xmax, ymin, ymax = inp["roi"]
        elem = inp["elem"]
        m = int(self.sp_margin.value()) * elem

        def snap_up(v):
            import math
            v = max(0.0, float(v))
            n = max(1, int(math.ceil((v + m) / elem - 1e-9)))
            return n * elem
        return DomainDims(
            h_wp=snap_up(-ymin), h_void=snap_up(ymax),
            l_wp=snap_up(-xmin), l_void=snap_up(xmax))

    # =====================================================================
    # UI actions
    # =====================================================================
    def refresh_inputs(self):
        # The Inputs-from-model panel was removed; refreshing now just redraws
        # the preview from the current config (kept for _rebind_cfg callers).
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
        # measurement ROI (= initial Eulerian domain) + evaluation points at the
        # centroid step, filling the whole ROI
        xmin, xmax, ymin, ymax = inp["roi"]
        rect(xmin, xmax, ymin, ymax, fill=False, edgecolor="#15803d",
             lw=1.4, zorder=7)
        step = self.grid_step()
        if step > 0:
            gx = np.arange(xmin, xmax + 1e-9, step)
            gy = np.arange(ymin, ymax + 1e-9, step)
            if gx.size and gy.size:
                XX, YY = np.meshgrid(gx, gy)
                ax.scatter(XX.ravel(), YY.ravel(), s=4, c="#15803d",
                           alpha=0.5, zorder=7)
        ax.plot([tip_x], [tip_y], marker="v", color="k", markersize=7,
                zorder=8)
        ax.set_title("measurement ROI = initial domain (green)  "
                     "max cap (purple)", fontsize=7)
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

    # -- Abaqus launcher (replicates the Sensitivity run mechanism) --------
    def _make_run_bundle(self, prefs, workdir, cpus):
        import subprocess

        counter = {"i": 0}

        def run_bundle(cfg):
            self._cancel_evt.clear() if False else None
            i = counter["i"]; counter["i"] += 1
            job = "opt_run%03d" % i
            out_path = Path(workdir) / ("%s.results.npz" % job)
            self._current_sta = Path(workdir) / ("%s.sta" % job)
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                log_swallowed("removing stale bundle", level=logging.DEBUG)
            args = [prefs.abaqus_cmd, "cae",
                    "noGUI=%s" % prefs.abaqus_script, "--",
                    "--model_cfg", repr(cfg.to_params_dict()),
                    "--run_cfg", repr({"cpus": cpus, "job_name": job})]
            _ms = (float(getattr(cfg.step, "mass_scaling_factor", 1.0))
                   if getattr(cfg.step, "mass_scaling_enabled", False) else 1.0)
            self._log_ui("\n%s\n[%s] ms=%.4g wp=%.4g tool=%.4g | "
                         "h_wp=%.4g h_void=%.4g l_wp=%.4g l_void=%.4g\n%s\n"
                         % ("-" * 60, job, _ms,
                            float(cfg.elem_size), float(cfg.tool_elem_size),
                            cfg.euler_geometry.h_wp, cfg.euler_geometry.h_void,
                            cfg.euler_geometry.l_wp, cfg.euler_geometry.l_void,
                            "-" * 60))
            try:
                proc = subprocess.Popen(
                    args, cwd=str(workdir),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except Exception as e:
                self._log_ui("failed to start Abaqus: %s\n" % e)
                return None
            try:
                for raw in iter(proc.stdout.readline, b""):
                    if self._cancel_evt.is_set():
                        try:
                            proc.terminate()
                        except Exception:
                            log_swallowed("terminating Abaqus",
                                          level=logging.DEBUG)
                        break
                    self._log_ui(raw.decode("cp1252", errors="replace"))
            except Exception:
                log_swallowed("streaming Abaqus stdout", level=logging.DEBUG)
            proc.wait()
            if self._cancel_evt.is_set() or proc.returncode != 0 \
                    or not out_path.exists():
                self._log_ui("[%s] no bundle (rc=%s)\n" % (job, proc.returncode))
                return None
            try:
                return ResultsBundle.load(out_path)
            except Exception as e:
                self._log_ui("[%s] load failed: %s\n" % (job, e))
                return None

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
        # Safe to call from the worker thread for QPlainTextEdit append via
        # signals would be cleaner; appendPlainText is used read-mostly here.
        self.log.appendPlainText(text.rstrip("\n"))

    def _e_max(self, errors) -> float:
        """E_max = max over quantities of the normalized error E_q / eps_q (the
        admissibility criterion is E_max < 1). NaN components are skipped."""
        thr = self.thresholds()
        vals = [errors[q] / thr[q] for q in thr
                if q in errors and errors[q] == errors[q] and thr[q] > 0]
        return max(vals) if vals else float("nan")

    def _plot(self):
        """Redraw the four per-parameter convergence axes from `self._hist`.

        Each axis shows, versus the parameter value, the per-quantity
        normalized errors E_q/eps_q (thin) and E_max (bold), with the
        admissibility line E_max = 1. Mass scaling, wp and tool element sizes
        each get their own axis (incompatible value scales); the four Eulerian
        domain dimensions share the fourth axis (same mm scale)."""
        thr = self.thresholds()
        axes = [self._ax_ms, self._ax_wp, self._ax_tool, self._ax_domain]
        drawn = set()
        for ax in axes:
            ax.clear()

        def _series(ax, pts, prefix=""):
            """Plot one (value, errors) history on `ax`. Returns True if drawn."""
            if not pts:
                return False
            xs = [v for (v, _e) in pts]
            for q in pts[0][1].keys():
                if q not in thr or thr[q] <= 0:
                    continue
                ys = [e.get(q, float("nan")) / thr[q] for (_v, e) in pts]
                ax.plot(xs, ys, marker="o", lw=0.8, alpha=0.5,
                        label="%s%s" % (prefix, q))
            ymax = [self._e_max(e) for (_v, e) in pts]
            ax.plot(xs, ymax, marker="s", lw=1.8,
                    label="%sE_max" % prefix)
            return True

        # single-parameter axes
        for key, ax in self._param_axes.items():
            if _series(ax, self._hist.get(key, [])):
                drawn.add(id(ax))
        # domain axis: the four dimensions share one axis
        for name in _DIM_ORDER:
            if _series(self._ax_domain, self._hist.get(name, []),
                       prefix="%s·" % name):
                drawn.add(id(self._ax_domain))

        for ax in axes:
            ax.axhline(1.0, ls="--", lw=1.0, color="#b91c1c", alpha=0.85)
            ax.set_title(self._ax_titles[id(ax)], fontsize=8)
            ax.set_yscale("log")
            ax.tick_params(labelsize=6)
            if id(ax) in drawn:
                ax.legend(fontsize=5, ncol=2)
        # Mass scaling, wp and tool span decades on their value axis -> log x.
        for ax in (self._ax_ms, self._ax_wp, self._ax_tool):
            ax.set_xscale("log")
        for ax in (self._ax_tool, self._ax_domain):
            ax.set_xlabel("parameter value", fontsize=7)
        for ax in (self._ax_ms, self._ax_tool):
            ax.set_ylabel("E_q/ε_q", fontsize=7)
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    # ===================================================================
    # Mesh + domain full pipeline
    # ===================================================================
    def _step_checkboxes(self):
        return [self.cb_identify_ms, self.cb_include_wp, self.cb_include_tool,
                self.cb_include_domain]

    def _enforce_pipeline_selection(self, *_):
        """At least one pipeline step must stay checked; if the user unchecks the
        last one, re-check it."""
        boxes = self._step_checkboxes()
        if not any(b.isChecked() for b in boxes):
            sender = self.sender()
            if sender in boxes:
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
                QMessageBox.information(
                    self, "Pipeline",
                    "At least one pipeline step must be selected.")

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

    def _on_run_pipeline(self):
        d0 = self.compute_initial_dims()
        if min(d0.h_wp, d0.h_void, d0.l_wp, d0.l_void) <= 0:
            QMessageBox.warning(self, "Initial domain",
                                "The measurement ROI (BBox) is degenerate.")
            return
        if self._initial is None:
            self.compute_initial()
        if self._initial is None:
            return
        thr = self.thresholds()
        if not self.thresholds_complete():
            QMessageBox.warning(self, "Thresholds",
                                "A threshold \u03b5_q is required for every "
                                "field (Vx, Vy, T, EVF, Fc, Ff).")
            return
        val = self._validate_launch()
        if val is None:
            return
        if self.cb_identify_ms.isChecked():
            problems = []
            if not ("Vx" in thr and "Vy" in thr):
                problems.append("Set Vx and Vy thresholds (the mass-scaling "
                                "criterion is on the velocity).")
            if self._float_or(self.le_ms_guard, None) is None:
                problems.append("Set the energy guard threshold "
                                "\u27e8ALLKE/ALLIE\u27e9.")
            if self._float_or(self.le_ms_bisect, None) is None:
                problems.append("Set the mass-scaling factor bisection "
                                "resolution (the dichotomy stop step).")
            if problems:
                QMessageBox.warning(self, "Mass scaling", "\n".join(problems))
                return
        prefs, wd, cpus = val
        inp = self.config_inputs()
        opt_roi = inp["roi"]
        self._cancel_evt.clear()
        run_bundle = self._make_run_bundle(prefs, wd, cpus)

        self.log.clear()
        self._hist = {}
        self.table.setRowCount(0)
        self._plot()
        self.tabs.setCurrentIndex(0)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setStyleSheet("color: #1d4ed8;")
        self.lbl_status.setText("Running pipeline… (many Abaqus runs)")

        self._pipe_worker = MeshPipelineWorker(
            base_cfg=self.cfg, run_bundle=run_bundle, roi=opt_roi,
            quantity_field_map=self.quantity_field_map(), thresholds=thr,
            wp_start=self._float_or(self.le_wp_start, inp["elem"]),
            tool_start=self._float_or(self.le_tool_start,
                                      float(self.cfg.tool_elem_size)),
            initial_domain=self._initial,
            wp_factor=self._float_or(self.le_wp_factor, 0.5),
            tool_factor=self._float_or(self.le_tool_factor, 0.5),
            domain_grow_factor=self._float_or(self.le_domain_factor, 2.0),
            wp_min=self.wp_min(), tool_min=self.tool_min(),
            caps=self.caps() or None, order=_DIM_ORDER,
            include_wp=self.cb_include_wp.isChecked(),
            include_tool=self.cb_include_tool.isChecked(),
            include_domain=self.cb_include_domain.isChecked(),
            do_verify=self.cb_do_verify.isChecked(),
            force_channels=self.force_channels(),
            grid_step=self.grid_step(),
            identify_ms=self.cb_identify_ms.isChecked(),
            ms_start=self._float_or(self.le_ms_start, 1.0),
            ms_factor_growth=self._float_or(self.le_ms_factor, 2.0),
            ms_max_factor=self._float_or(self.le_ms_max, None),
            ms_guard_threshold=self._float_or(self.le_ms_guard, None),
            ms_bisection_resolution=self._float_or(self.le_ms_bisect, 0.0),
            wp_bisection_resolution=self._float_or(self.le_wp_bisect, 0.0),
            tool_bisection_resolution=self._float_or(self.le_tool_bisect, 0.0),
            domain_bisection_resolution=self._float_or(self.le_domain_bisect,
                                                       0.0))
        self._pipe_worker.progress.connect(self._on_pipeline_progress)
        self._pipe_worker.finished_ok.connect(self._on_pipeline_finished)
        self._pipe_worker.failed.connect(self._on_pipeline_failed)
        self._start_progress()
        self._pipe_worker.start()

    def _on_cancel_pipeline(self):
        self._cancel_evt.set()
        if getattr(self, "_pipe_worker", None) is not None:
            self._pipe_worker.cancel()
        self.lbl_status.setText("Cancelling pipeline…")

    def _on_pipeline_progress(self, ev):
        phase = ev.get("phase", "")
        sub = ev.get("sub") or {}
        # ---- live convergence points: append to the matching axis history ----
        # Each sub-optimizer emits a (value, errors) pair per Abaqus comparison;
        # we route it to its parameter history key and redraw the subplots.
        if phase == "ms" and "errors" in sub and "factor" in sub:
            self._hist.setdefault("mass_scaling", []).append(
                (sub["factor"], sub["errors"]))
            g = sub.get("guard")
            self._log_ui("    factor=%.4g → ⟨ALLKE/ALLIE⟩=%s | E_max=%.3g"
                         % (sub["factor"],
                            ("%.4g" % g) if g is not None else "n/a",
                            self._e_max(sub["errors"])))
            self._live_status("mass scaling", sub["factor"], sub["errors"],
                              sub.get("n_runs"))
            self._plot()
            return
        if phase == "wp" and "errors" in sub and "size" in sub:
            self._hist.setdefault("wp_elem", []).append(
                (sub["size"], sub["errors"]))
            self._live_status("wp element", sub["size"], sub["errors"],
                              sub.get("n_runs"))
            self._plot()
            return
        if phase == "tool" and "errors" in sub and "size" in sub:
            self._hist.setdefault("tool_elem", []).append(
                (sub["size"], sub["errors"]))
            self._live_status("tool element", sub["size"], sub["errors"],
                              sub.get("n_runs"))
            self._plot()
            return
        if phase == "domain" and sub.get("phase") == "compare":
            self._hist.setdefault(sub["name"], []).append(
                (sub["value"], sub["errors"]))
            self._live_status(sub["name"], sub["value"], sub["errors"],
                              sub.get("n_runs"))
            self._plot()
            return
        # ---- textual step milestones ----
        if phase == "ms_done":
            self._log_ui("  → mass-scaling factor = %.4g "
                         "(⟨ALLKE/ALLIE⟩=%.4g) — %s"
                         % (ev.get("identified", float("nan")),
                            ev.get("guard", float("nan")),
                            self._ms_verdict(ev.get("limited_by", ""),
                                             ev.get("converged"))))
        elif phase in ("wp_done", "tool_done"):
            self._log_ui("  → identified %s = %.4g (runs so far: %d)"
                         % (phase.split("_")[0], ev.get("identified", float("nan")),
                            ev.get("n_runs", 0)))
        elif phase == "domain_done":
            d = ev["dims"]
            self._log_ui("  → domain h_wp=%.4g h_void=%.4g l_wp=%.4g "
                         "l_void=%.4g" % (d.h_wp, d.h_void, d.l_wp, d.l_void))
        elif phase.endswith("_start"):
            self._log_ui("[%s]" % phase.replace("_start", ""))
        elif phase.startswith("verify_") and phase.endswith("_done"):
            self._log_ui("  → %s stable=%s" % (phase, ev.get("stable")))

    @staticmethod
    def _ms_verdict(limited_by, converged):
        """Human-readable mass-scaling verdict from the `limited_by` reason.

        The search minimizes the velocity sensitivity E(f) = E(f, f*factor)
        under the energy guard-rail; `converged` reports whether E < 1 at the
        retained point (informational, not an admissibility condition)."""
        txt = {
            "minimum": "sensitivity minimum bracketed and refined",
            "start": "minimum at the start factor — E already rises at the "
                     "first ladder step (lower the start factor to explore "
                     "below it)",
            "guard": "guard-limited — the energy guard-rail stopped the search "
                     "before the sensitivity minimum",
            "cap": "cap reached — the sensitivity was still decreasing at the "
                   "max factor (raise the cap to keep exploring)",
            "cancelled": "cancelled before any evaluation",
        }.get(limited_by, "converged" if converged else "NOT converged")
        return txt + (" | E < 1" if converged else " | E >= 1")

    def _live_status(self, name, value, errors, n_runs):
        """Update the status label with the current parameter value and E_max."""
        emax = self._e_max(errors)
        try:
            vtxt = "%.4g" % float(value)
        except (TypeError, ValueError):
            vtxt = str(value)
        self.lbl_status.setStyleSheet("color: #1d4ed8;")
        self.lbl_status.setText(
            "Optimizing %s… value=%s | E_max=%.3g | runs=%s"
            % (name, vtxt, emax, "?" if n_runs is None else n_runs))

    def _on_pipeline_finished(self, result):
        self._stop_progress()
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        d = result.domain

        def _vtxt(v):
            return "n/a" if v is None else ("stable" if v["stable"]
                                            else "NOT stable")
        self.lbl_status.setStyleSheet("color: #15803d;")
        self.lbl_status.setText(
            "Pipeline done — wp*=%.4g tool*=%.4g | domain l_wp=%.4g h_wp=%.4g "
            "| %d runs" % (result.wp_elem, result.tool_elem, d.l_wp, d.h_wp,
                           result.n_runs))
        ms_line = ""
        if result.ms_factor is not None:
            r = result.ms_result
            ms_line = ("mass-scaling factor: %.4g  (⟨ALLKE/ALLIE⟩=%.4g) — %s\n"
                       % (result.ms_factor,
                          (r.guard_at_identified if r else float("nan")),
                          self._ms_verdict(getattr(r, "limited_by", ""),
                                           r.velocity_converged if r else None)))
        self._log_ui(
            "\n=== PIPELINE RESULT ===\n"
            + ms_line +
            "wp element size  : %.4g  (verify: %s)\n"
            "tool element size: %.4g  (verify: %s)\n"
            "domain           : h_wp=%.4g h_void=%.4g l_wp=%.4g l_void=%.4g "
            "(verify: %s)\n"
            "grid step (fixed ROI grid): %.4g | total Abaqus runs: %d"
            % (result.wp_elem, _vtxt(result.wp_verify),
               result.tool_elem, _vtxt(result.tool_verify),
               d.h_wp, d.h_void, d.l_wp, d.l_void, _vtxt(result.domain_verify),
               result.grid_step, result.n_runs))
        self._fill_pipeline_table(result)

    def _fill_pipeline_table(self, result):
        """Populate the results table with one row per identified parameter,
        in pipeline order (mass scaling, wp element, tool element, then the four
        Eulerian domain dimensions). Columns: parameter | initial |
        intermediate (end of bracketing) | final (end of dichotomy). Steps that
        were skipped (unchecked) contribute no row."""
        def _fmt(v):
            try:
                return "%.4g" % float(v)
            except (TypeError, ValueError):
                return ""

        rows = []
        if getattr(result, "ms_result", None) is not None:
            r = result.ms_result
            rows.append(("mass_scaling", r.initial, r.intermediate,
                         r.identified))
        if getattr(result, "wp_conv", None) is not None:
            r = result.wp_conv
            rows.append(("wp_elem", r.initial, r.intermediate, r.identified))
        if getattr(result, "tool_conv", None) is not None:
            r = result.tool_conv
            rows.append(("tool_elem", r.initial, r.intermediate, r.identified))
        if getattr(result, "domain_result", None) is not None:
            for name in _DIM_ORDER:
                dr = result.domain_result.per_dim.get(name)
                if dr is not None:
                    # DimResult: d_large is the end-of-bracketing (intermediate)
                    # reference; final is the end-of-dichotomy retained value.
                    rows.append((name, dr.initial, dr.d_large, dr.final))

        self.table.setRowCount(0)
        for (name, ini, inter, fin) in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(name)))
            self.table.setItem(r, 1, QTableWidgetItem(_fmt(ini)))
            self.table.setItem(r, 2, QTableWidgetItem(_fmt(inter)))
            self.table.setItem(r, 3, QTableWidgetItem(_fmt(fin)))

    def _on_pipeline_failed(self, msg):
        self._stop_progress()
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setStyleSheet("color: #b91c1c;")
        self.lbl_status.setText("Pipeline failed: %s" % msg)
        self._log_ui("PIPELINE ERROR: %s" % msg)
