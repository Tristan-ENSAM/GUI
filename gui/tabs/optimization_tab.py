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
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QGroupBox,
    QCheckBox, QLineEdit, QPlainTextEdit, QTabWidget, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QProgressBar
)
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui.core.domain_sizing import DomainDims
from gui.core.logging_util import log_swallowed
from gui.core.sta_parser import parse_sta
from gui.sensitivity.mesh_gci_worker import MeshGciWorker
from gui.sensitivity.domain_convergence_worker import DomainConvergenceWorker
from gui.sensitivity.run_worker import (
    abaqus_terminate_job, build_abaqus_args, kill_process_tree_by_pid,
    script_log_path)
from gui.core.domain_sizing import (
    DIMENSION_NAMES, diagonal, diagonal_limit)
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
    # Emitted when a persisted optimization parameter changes, so the
    # main window can mark the profile dirty.
    changed = Signal()


    def __init__(self, cfg, prefs_getter=None, cpus_getter=None,
                 profile_name_getter=None):
        super().__init__()
        self.cfg = cfg
        self._prefs_getter = prefs_getter
        self._profile_name_getter = profile_name_getter
        self._loading = False   # guard: True while populating from cfg
        self._cpus_getter = cpus_getter
        self._initial = None            # DomainDims from Merchant
        self._cancel_evt = threading.Event()
        # Published by run_bundle so _on_cancel can name the job to
        # `abaqus terminate` and reach the solver behind the launcher.
        self._current_job = None
        self._current_proc = None
        self._current_abaqus_cmd = None
        self._current_run_dir = None
        self._hist = {}                 # param key -> list of (value, {q: E_q})
        self._current_sta = None        # current job's .sta path (for progress)
        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(500)
        self._sim_timer.timeout.connect(self._poll_sta)

        root = QVBoxLayout(self)

        # (Inputs-from-model panel removed as requested — the values are read
        # directly from the config when needed.)

        # ---- Initial domain = the measurement ROI ----------------------
        gdom = QGroupBox("4 \u00b7 Eulerian domain (initial + caps)")
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
        gcrit = QGroupBox("2 \u00b7 Convergence criterion (RMSE thresholds)")
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

        # ---- 1 · ZOI (measurement zone for the optimization) ----------
        # DISTINCT from the ROI (Geometry tab, model output set for DIC/IRT):
        # this is the user zone the domain/mesh studies sample. Own bbox; the
        # sampling grid reuses the "Centroid step" above. Must stay inside the
        # domain (margin above). Empty fields default to the ROI.
        gzoi = QGroupBox("1 \u00b7 ZOI (measurement zone)")
        zg = QGridLayout(gzoi)
        self.le_zoi = {}
        for c, (lbl, key) in enumerate([("x min", "xmin"), ("x max", "xmax")]):
            zg.addWidget(QLabel(lbl), 0, 2 * c); le = QLineEdit()
            le.setPlaceholderText("= ROI"); self.le_zoi[key] = le
            zg.addWidget(le, 0, 2 * c + 1)
        for c, (lbl, key) in enumerate([("y min", "ymin"), ("y max", "ymax")]):
            zg.addWidget(QLabel(lbl), 1, 2 * c); le = QLineEdit()
            le.setPlaceholderText("= ROI"); self.le_zoi[key] = le
            zg.addWidget(le, 1, 2 * c + 1)
        self.btn_zoi_from_roi = QPushButton("Set ZOI = ROI")
        self.btn_zoi_from_roi.clicked.connect(self._zoi_from_roi)
        zg.addWidget(self.btn_zoi_from_roi, 2, 0, 1, 2)
        _zt = QLabel("mm \u2014 sampled at the Centroid step; kept inside the "
                     "domain")
        _zt.setStyleSheet("color:#6b7280;")
        zg.addWidget(_zt, 2, 2, 1, 2)
        for _le in self.le_zoi.values():
            _le.textChanged.connect(self._draw_preview)

        # ---- 3 · Mesh convergence (GCI / Richardson) ------------------
        gmeshgci = QGroupBox("3 \u00b7 Mesh convergence (GCI)")
        mgl = QGridLayout(gmeshgci)
        mgl.addWidget(QLabel("finest elem"), 0, 0)
        self.le_gci_finest = QLineEdit()
        self.le_gci_finest.setPlaceholderText("= element size")
        mgl.addWidget(self.le_gci_finest, 0, 1)
        mgl.addWidget(QLabel("mm"), 0, 2)
        mgl.addWidget(QLabel("ratio"), 0, 3)
        self.le_gci_ratio = QLineEdit("2")
        mgl.addWidget(self.le_gci_ratio, 0, 4)
        mgl.addWidget(QLabel("min elem"), 1, 0)
        self.le_gci_min = QLineEdit()
        self.le_gci_min.setPlaceholderText("floor e.g. 0.006")
        mgl.addWidget(self.le_gci_min, 1, 1)
        mgl.addWidget(QLabel("mm"), 1, 2)
        mgl.addWidget(QLabel("n meshes"), 1, 3)
        self.sp_gci_n = QSpinBox(); self.sp_gci_n.setRange(3, 6)
        self.sp_gci_n.setValue(3)
        mgl.addWidget(self.sp_gci_n, 1, 4)
        _gt = QLabel("On a FIXED (large) domain; refine finest\u2192coarse, "
                     "GCI per quantity.")
        _gt.setStyleSheet("color:#6b7280;")
        mgl.addWidget(_gt, 2, 0, 1, 5)


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
        left.addWidget(gzoi)
        # Panels 2 (criteria) and 3 (mesh GCI) share a row to save vertical
        # space (the left column is cramped at 16:9).
        row23 = QHBoxLayout()
        row23.addWidget(gcrit, 1)
        row23.addWidget(gmeshgci, 1)
        left.addLayout(row23)
        left.addWidget(gdom)
        left.addStretch(1)
        cols.addLayout(left, 1)
        cols.addWidget(gprev, 1)
        root.addLayout(cols)

        # ---- Run controls ----------------------------------------------
        rc = QHBoxLayout()
        # Sizing tolerances (relative, dimensionless): the stop criterion for
        # BOTH studies. For the mesh GCI, "force" applies to Fc and Ff.
        eg = QGroupBox("Sizing tolerances (relative)")
        egl = QGridLayout(eg)
        egl.setContentsMargins(8, 6, 8, 6)
        self._dj_eps = {}
        for i, q in enumerate(("EVF", "TEMP", "V1", "V2", "force")):
            egl.addWidget(QLabel(q), 0, 2 * i)
            le = QLineEdit("0.02"); le.setFixedWidth(58)
            le.setToolTip("Relative tolerance on %s (0.02 = 2%%). Empty = "
                          "excluded from the criterion." % q)
            self._dj_eps[q] = le
            egl.addWidget(le, 0, 2 * i + 1)
        rc.addWidget(eg)
        self.btn_mesh = QPushButton("Run mesh convergence (GCI)")
        self.btn_mesh.setToolTip(
            "GCI/Richardson mesh convergence on a FIXED (large) domain: three\n"
            "systematically-refined meshes, observed order p, extrapolated\n"
            "value and GCI uncertainty per quantity. Recommends the coarsest\n"
            "mesh within tolerance of the extrapolated value.")
        self.btn_mesh.clicked.connect(self._on_run_mesh_gci)
        rc.addWidget(self.btn_mesh)
        self.btn_domain = QPushButton("Run domain sizing (convergence)")
        self.btn_domain.setToolTip(
            "Grow the domain outward around the fixed ZOI until pushing each\n"
            "boundary no longer changes the windowed, EVF-masked ZOI field\n"
            "beyond tolerance (mesh held fixed). Smallest adequate domain,\n"
            "bounded by the reverberation ceiling.")
        self.btn_domain.clicked.connect(self._on_run_domain_convergence)
        rc.addWidget(self.btn_domain)
        self.btn_open_wd = QPushButton("Open working dir")
        self.btn_open_wd.setToolTip("Open the Preferences working directory.")
        self.btn_open_wd.clicked.connect(self._open_working_dir)
        rc.addWidget(self.btn_open_wd)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
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
    # ===================================================================
    # Persistence of the optimization parameters (saved in the .acpf profile)
    # ===================================================================
    def _opt_line_edits(self):
        les = [self.le_grid_step, self.le_gci_finest, self.le_gci_ratio,
               self.le_gci_min]
        les += list(self.le_zoi.values())
        les += list(self._q_eps.values())
        les += list(self._dj_eps.values())
        les += list(self._max.values())
        return les

    def _wire_opt_persistence(self):
        for le in self._opt_line_edits():
            le.textChanged.connect(self._sync_opt_to_cfg)
        self.sp_margin.valueChanged.connect(self._sync_opt_to_cfg)
        self.sp_gci_n.valueChanged.connect(self._sync_opt_to_cfg)

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

        def run_bundle(cfg):
            i = counter["i"]; counter["i"] += 1
            job = "%s_run%03d" % (prefix, i)
            out_path = Path(run_dir) / ("%s.results.npz" % job)
            self._current_sta = Path(run_dir) / ("%s.sta" % job)
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

    def _dims_from_cfg(self):
        g = self.cfg.euler_geometry
        return DomainDims(h_wp=float(g.h_wp), h_void=float(g.h_void),
                          l_wp=float(g.l_wp), l_void=float(g.l_void))

    def _busy(self, on, msg="", color="#1d4ed8"):
        self.btn_mesh.setEnabled(not on)
        self.btn_domain.setEnabled(not on)
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
    # 4 - Eulerian domain sizing by convergence (grow outward, ZOI fixed)
    # ===================================================================
    def _on_run_domain_convergence(self):
        val = self._validate_launch()
        if val is None:
            return
        prefs, wd, cpus = val
        elem = float(self.cfg.elem_size)
        dims = self._dims_from_cfg()
        zoi = self.zoi()
        tol = self._tolerances()
        if not tol:
            QMessageBox.warning(self, "Tolerances",
                                "Set at least one relative tolerance: it is the "
                                "stopping criterion of the study.")
            return
        study_cfg = {
            "zoi": {"xmin": zoi[0], "xmax": zoi[1],
                    "ymin": zoi[2], "ymax": zoi[3]},
            "elem_size": elem, "margin_elems": int(self.sp_margin.value()),
            "grid_step": self.grid_step(), "tolerances": tol,
            "field_vars": ["EVF", "TEMP", "V1", "V2"], "evf_threshold": 0.5,
            "grow_elems": 4, "max_iterations": 8,
            "initial_dims": {"h_wp": dims.h_wp, "h_void": dims.h_void,
                             "l_wp": dims.l_wp, "l_void": dims.l_void}}
        run_dir = self._study_run_dir(wd, "domainsizing", study_cfg)
        run_bundle = self._make_run_bundle(prefs, run_dir, cpus, "domainsizing")
        self._cancel_evt.clear()
        self.log.clear()
        self.tabs.setCurrentIndex(0)
        self._busy(True, "Domain sizing (convergence)\u2026")
        self._log_ui("=" * 68)
        self._log_ui("DOMAIN SIZING BY CONVERGENCE (grow outward, ZOI fixed)")
        self._log_ui("  ZOI  x[%.4g,%.4g] y[%.4g,%.4g]" % zoi)
        self._log_ui("  mesh %.4g mm (held fixed) | margin %d elem"
                     % (elem, int(self.sp_margin.value())))
        self._log_ui("=" * 68)
        self._dc_worker = DomainConvergenceWorker(
            run_bundle=run_bundle, base_cfg=self.cfg, zoi=zoi, initial_dims=dims,
            grid_step=self.grid_step(), elem_size=elem, tolerances=tol,
            field_vars=("EVF", "TEMP", "V1", "V2"), evf_threshold=0.5,
            grow_elems=4, margin_elems=int(self.sp_margin.value()),
            max_iterations=8)
        self._dc_worker.progress.connect(self._on_dc_progress)
        self._dc_worker.finished_ok.connect(self._on_dc_done)
        self._dc_worker.failed.connect(self._on_fail)
        self._start_progress()
        self._dc_worker.start()

    def _on_dc_progress(self, ev):
        if ev.get("phase") != "domain_convergence":
            return
        d = ev.get("dims", {})
        self._log_ui("  dims h_wp=%.4g h_void=%.4g l_wp=%.4g l_void=%.4g "
                     "| settled=%s"
                     % (d.get("h_wp", 0), d.get("h_void", 0), d.get("l_wp", 0),
                        d.get("l_void", 0), ev.get("settled")))

    def _on_dc_done(self, res):
        self._stop_progress()
        self._busy(False)
        d = res.dims
        why = {
            "converged": "converged \u2014 smallest domain no longer perturbed "
                         "by the boundaries",
            "diagonal": "STOPPED at the reverberation ceiling before "
                        "independence could be reached",
            "zoi_outside": "the ZOI is not inside the initial domain (enlarge "
                           "the domain or reduce the margin)",
            "max_iter": "stopped at the iteration cap, NOT converged",
            "cancelled": "cancelled",
        }.get(res.stopped_by, res.stopped_by or "stopped")
        self._log_ui("=" * 68)
        self._log_ui("RESULT: %s" % why)
        self._log_ui("  h_wp=%.4g h_void=%.4g l_wp=%.4g l_void=%.4g | %d runs"
                     % (d.h_wp, d.h_void, d.l_wp, d.l_void, res.n_runs))
        ok = res.converged
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#15803d" if ok else "#b45309"))
        self.lbl_status.setText("Domain sizing \u2014 %s" % why)

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
        tol = self._tolerances()
        gci_tol = {q: tol[q] for q in ("EVF", "TEMP", "V1", "V2") if q in tol}
        if "force" in tol:
            gci_tol["Fc"] = tol["force"]
            gci_tol["Ff"] = tol["force"]
        dims = self._dims_from_cfg()
        zoi = self.zoi()
        study_cfg = {
            "zoi": {"xmin": zoi[0], "xmax": zoi[1],
                    "ymin": zoi[2], "ymax": zoi[3]},
            "finest_elem_size": finest, "ratio": ratio, "n_meshes": nmesh,
            "min_elem_size": minh, "grid_step": self.grid_step(),
            "tolerances": gci_tol, "field_vars": ["EVF", "TEMP", "V1", "V2"],
            "evf_threshold": 0.5,
            "domain_dims": {"h_wp": dims.h_wp, "h_void": dims.h_void,
                            "l_wp": dims.l_wp, "l_void": dims.l_void}}
        run_dir = self._study_run_dir(wd, "GCI", study_cfg)
        run_bundle = self._make_run_bundle(prefs, run_dir, cpus, "GCI")
        self._cancel_evt.clear()
        self.log.clear()
        self.tabs.setCurrentIndex(0)
        self._busy(True, "Mesh convergence (GCI)\u2026")
        self._log_ui("=" * 68)
        self._log_ui("MESH CONVERGENCE (GCI / Richardson) on a fixed domain")
        self._log_ui("  finest %.4g mm | ratio %.3g | n %d | floor %s"
                     % (finest, ratio, nmesh,
                        "n/a" if minh is None else "%.4g" % minh))
        self._log_ui("=" * 68)
        self._mesh_worker = MeshGciWorker(
            run_bundle=run_bundle, base_cfg=self.cfg, zoi=zoi, domain_dims=dims,
            grid_step=self.grid_step(), finest_elem_size=finest, ratio=ratio,
            n_meshes=nmesh, tolerances=(gci_tol or None),
            field_vars=("EVF", "TEMP", "V1", "V2"), evf_threshold=0.5,
            min_elem_size=minh)
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
        """Stop the study AND the run currently in flight.

        Same two stages as SensitivityRunWorker.cancel, in the same order:
          1. ``abaqus terminate job=<name>`` -- the clean route: it stops the
             solver AND releases the licence tokens.
          2. kill the process tree -- the fallback, for when Abaqus does not
             answer (no .cid yet, job already finishing, hung solver). This
             leaves the tokens checked out, hence the ordering.

        Before this, cancelling only called ``proc.terminate()`` on the
        ``abaqus cae`` launcher: the solver behind it survived as an orphan
        (M1), the licence stayed checked out, and the test itself sat in a
        loop reading a stdout that never produces a line (M7).

        The run interrupted here is reported as failed -- run_bundle returns
        None on a set cancel flag, which the studies already treat as "no
        usable result" -- so a half-written bundle is never read as data.
        """
        for attr in ("_dc_worker", "_mesh_worker"):
            w = getattr(self, attr, None)
            if w is not None and w.isRunning():
                w.cancel()
        self._cancel_evt.set()
        self.lbl_status.setText("Cancelling the current run\u2026")

        job, proc = self._current_job, self._current_proc
        cmd, run_dir = self._current_abaqus_cmd, self._current_run_dir
        if job and cmd:
            self._log_ui("[CANCEL] asking Abaqus to terminate job %s" % job)
            if abaqus_terminate_job(cmd, job, run_dir):
                if proc is not None:
                    try:
                        # Give the solver a moment to unwind before force-killing.
                        proc.wait(timeout=10.0)
                        return
                    except Exception:
                        log_swallowed("waiting for the terminated job to exit",
                                      level=logging.DEBUG)
        if proc is not None and proc.poll() is None:
            self._log_ui("[CANCEL] Abaqus did not answer; killing the process "
                         "tree (licence tokens stay checked out)")
            kill_process_tree_by_pid(proc.pid)

    def _on_fail(self, msg):
        self._stop_progress()
        self._busy(False)
        self.lbl_status.setStyleSheet("color: #b91c1c;")
        self.lbl_status.setText("Study failed: %s" % msg)
        self._log_ui("ERROR: %s" % msg)
