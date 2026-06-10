# -*- coding: utf-8 -*-
"""
Sensitivity tab — Morris screening (Lot 2b, UI).

Lets the user:
  * tick which model parameters to vary and edit their min/max
    (pre-filled from the current model value, in displayed units),
  * choose which QoI to screen,
  * set N (trajectories) and the Morris grid levels,
  * generate the sampling plan (cost = N*(k+1) runs) and preview it.

Running the plan and plotting mu*/sigma come next (Lots 2c / 2d). The
generated plan is kept on the tab (`self.plan`) for the runner to pick up.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QCheckBox, QPlainTextEdit, QAbstractItemView, QSplitter, QComboBox,
    QProgressBar, QTabWidget, QFileDialog,
)
from PySide6.QtCore import QThread, QTimer

from gui.sensitivity import param_registry as pr
from gui.sensitivity import morris_plan as mp
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import runner_core as rc
from gui.sensitivity import export_results as xr
from gui.sensitivity.run_worker import SensitivityRunWorker
from gui.core.sta_parser import parse_sta
from gui.results import qoi as qoi_mod

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
    def __init__(self, cfg, prefs_getter=None, cpus_getter=None):
        super().__init__()
        self.cfg = cfg
        self._prefs_getter = prefs_getter
        self._cpus_getter = cpus_getter
        self.plan = None                 # last generated JacobianPlan
        self.plan_kind = "jacobian"
        self.selected_qois = []          # list[QoISpec]
        self._thread = None              # QThread for the run worker
        self._worker = None              # SensitivityRunWorker
        self._last_result = None         # rc.RunResult
        self._per_run_sec = None         # measured/estimated wall-clock per run
        self._run_t0 = {}                # run index -> monotonic start
        self._run_durations = []         # measured durations of finished runs
        self._running_index = None       # index of the run in progress
        self._run_workdir = None         # workdir of the active run batch
        self._run_total = 0
        self._sta_timer = None           # live .sta poller during a run
        self._row_spec = {}              # table row -> ParamSpec

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
        method_row.addWidget(QLabel("Method: Jacobian (finite differences)"))
        method_row.addSpacing(16)
        method_row.addWidget(QLabel("FD scheme:"))
        self.cb_scheme = QComboBox()
        self.cb_scheme.addItems(["central", "forward", "backward"])
        self.cb_scheme.currentIndexChanged.connect(self._update_cost)
        method_row.addWidget(self.cb_scheme)
        self.lbl_hint = QLabel("step = Delta per parameter")
        self.lbl_hint.setStyleSheet("color: #6b7280;")
        method_row.addWidget(self.lbl_hint)
        method_row.addStretch(1)
        bl.addLayout(method_row)

        qoi_box = QGroupBox("Quantities of interest (QoI) to screen")
        qg = QGridLayout(qoi_box)
        self._qoi_checks = {}
        for i, q in enumerate(qoi_mod.REGISTRY):
            cb = QCheckBox("%s  [%s]" % (q.label, q.unit))
            cb.setChecked(q.id in _DEFAULT_QOIS)
            self._qoi_checks[q.id] = cb
            qg.addWidget(cb, i // 2, i % 2)
        bl.addWidget(qoi_box)

        # Field QoI: screen how much each parameter moves whole Eulerian
        # fields in the ZOI (SSD vs the base run). Jacobian only.
        field_box = QGroupBox("Field QoI in the ZOI (SSD vs base run)")
        fg = QHBoxLayout(field_box)
        self._field_checks = {}
        for var, label in (("EVF", "EVF (chip)"), ("V", "V (flow)"),
                           ("TEMP", "TEMP")):
            cb = QCheckBox(label)
            self._field_checks[var] = cb
            fg.addWidget(cb)
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
        crow.addWidget(self.cb_chart_qoi); crow.addStretch(1)
        cv.addLayout(crow)
        self._fig = Figure(figsize=(5, 3))
        self._canvas = FigureCanvas(self._fig)
        cv.addWidget(self._canvas, 1)
        self.tabs_out.addTab(chart_w, "Chart")
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
                nrm.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                nrm.setCheckState(Qt.Unchecked)
                nrm.setToolTip("Report the dimensionless elasticity "
                               "(dQ/Q)/(dx/x) instead of the raw dQ/dx. "
                               "The real Min/Max/Delta below stay unchanged.")
                self.table.setItem(r, 7, nrm)
                # col 8: unit (read-only)
                unit = QTableWidgetItem(spec.display_unit)
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
    def _is_jacobian(self) -> bool:
        return True

    def _scheme(self) -> str:
        return self.cb_scheme.currentText()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_cost()          # refresh CPU mirror + cost estimate

    def _current_cpus(self) -> int:
        if self._cpus_getter:
            try:
                return int(self._cpus_getter())
            except Exception:
                pass
        return 1

    def _n_runs(self) -> int:
        k = len(self._selected_rows())
        return jac.n_runs(k, self._scheme()) if k else 0

    def _update_cost(self):
        self.lbl_cpus.setText(str(self._current_cpus()))
        k = len(self._selected_rows())
        if k == 0:
            self.lbl_cost.setText("0 parameters selected")
            return
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
                lo = float(self.table.item(r, 2).text())
                hi = float(self.table.item(r, 4).text())
            except (ValueError, AttributeError):
                raise ValueError("%s: min/max must be numbers." % spec.label)
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
            x0 = pr.get_display(self.cfg, spec, tu)
            selected.append((spec, x0, delta, norm))
        return selected

    def _selected_qoi_specs(self):
        return [q for q in qoi_mod.REGISTRY
                if self._qoi_checks[q.id].isChecked()]

    def _selected_field_vars(self):
        return [v for v, cb in self._field_checks.items() if cb.isChecked()]

    def _on_generate(self):
        try:
            qois = self._selected_qoi_specs()
            field_vars = self._selected_field_vars()
            if not qois and not field_vars:
                self._warn("Tick at least one QoI (a scalar QoI, or a ZOI "
                           "field).")
                return
            selected = self._collect_jacobian()
            if not selected:
                self._warn("Tick at least one parameter to vary.")
                return
            plan = jac.build_plan(selected, scheme=self._scheme(),
                                  temp_unit=self._temp_unit())
        except ValueError as e:
            self._warn(str(e))
            return
        except Exception as e:                       # pragma: no cover
            self._warn("Could not build the plan: %s" % e)
            return

        self.plan = plan
        self.plan_kind = "jacobian"
        self.selected_qois = qois
        qoi_names = [q.id for q in qois] + ["%s[field]" % v for v in field_vars]
        self.status.setStyleSheet("color: #15803d;")
        self.status.setText(
            "Jacobian (%s FD) plan ready: %d parameters, %d runs, %d QoI "
            "(%s). Ready for the run step." % (
                self._scheme(), plan.k, plan.n_runs, len(qoi_names),
                ", ".join(qoi_names) if qoi_names else "—"))
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
        prefs = self._prefs_getter() if self._prefs_getter else None
        if prefs is None:
            self._warn("No preferences available (Abaqus command/script).")
            return
        from pathlib import Path
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
        import time
        self._run_workdir = wd
        self._run_total = self.plan.n_runs
        self._run_t0 = {}
        self._run_durations = []
        self._running_index = None
        self._failed_live = []           # run indices reported failed live
        self._run_clock0 = time.monotonic()
        self._sta_timer = QTimer(self)
        self._sta_timer.setInterval(3000)
        self._sta_timer.timeout.connect(self._poll_sta)
        self._sta_timer.start()

        field_vars = self._selected_field_vars()
        self._worker = SensitivityRunWorker(
            self.plan, self.plan_kind, self.selected_qois, self.cfg,
            abaqus_cmd=prefs.abaqus_cmd, abaqus_script=prefs.abaqus_script,
            workdir=str(wd), cpus=self._current_cpus(),
            job_prefix="sens", field_vars=field_vars)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.runDone.connect(self._on_run_done)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._thread.start()

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status.setStyleSheet("color: #b45309;")
            self.status.setText("Cancelling after the current run…")
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
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        # The run that was in progress just finished -> record its duration.
        if self._running_index is not None and self._running_index in self._run_t0:
            dur = now - self._run_t0[self._running_index]
            if dur > 0:
                self._run_durations.append(dur)
                self._per_run_sec = sum(self._run_durations) / len(self._run_durations)
        # Next run (if any) starts now.
        if done < total:
            self._running_index = done
            self._run_t0[done] = now
        else:
            self._running_index = None
        self._update_estimate(done, total)

    def _poll_sta(self):
        """Live estimate of the running run's total wall-clock from its .sta
        (wall time so far / fraction done), refined by finished-run durations.
        Reuses the same parser as the Job tab."""
        if self._running_index is None or self._run_workdir is None:
            return
        import time
        sta = self._run_workdir / ("sens_run%03d.sta" % self._running_index)
        run_est = None
        try:
            snap = parse_sta(sta)
            if snap.is_ready():
                frac = snap.fraction()
                if frac is None and snap.step_time is not None:
                    st = float(getattr(self.cfg.step, "sim_time", 0.0) or 0.0)
                    frac = (snap.step_time / st) if st > 0 else None
                wall = _hms_to_sec(snap.wall_time)
                if frac and frac > 0.02 and wall:
                    run_est = wall / frac
        except Exception:
            run_est = None
        # Per-run baseline: measured runs if any, else the live .sta estimate.
        if self._run_durations:
            per = sum(self._run_durations) / len(self._run_durations)
        else:
            per = run_est
        if per:
            self._per_run_sec = per
        self._update_estimate(self.progress.value(), self._run_total)

    def _update_estimate(self, done, total):
        import time
        elapsed = time.monotonic() - getattr(self, "_run_clock0", time.monotonic())
        msg = "Run %d/%d" % (min(done + (1 if self._running_index is not None
                                          else 0), total), total)
        if self._per_run_sec:
            remaining = max(0, total - len(self._run_durations)) * self._per_run_sec
            msg += "   ·   ~%s per run   ·   ~%s remaining   ·   est. total ~%s" % (
                _fmt_duration(self._per_run_sec),
                _fmt_duration(remaining),
                _fmt_duration(total * self._per_run_sec))
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
        n_ok = result.Y.shape[0] - len(result.failures)
        self.status.setStyleSheet("color: #15803d;")
        self.status.setText(
            "Run finished: %d/%d successful, %d failed. See Results."
            % (n_ok, result.Y.shape[0], len(result.failures)))
        self._show_results(result)
        self.btn_export.setEnabled(result.Y.shape[0] > 0)
        self.tabs_out.setCurrentWidget(self.results_table)

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
                cell = self._result_cell(result.plan_kind, a, p)
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
        if result.plan_kind == "jacobian":
            rows = rc.jacobian_ranking(result, qid)        # [(path, sens)]
            vals = [abs(s) for _, s in rows]
            xlabel = "|sensitivity|"
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

    @staticmethod
    def _result_cell(plan_kind, analysis, path):
        if "error" in analysis:
            return "err"
        if plan_kind == "jacobian":
            d = analysis.get(path)
            if not d:
                return "—"
            return _fmt(d["sensitivity"])
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
        head_cells += ["%s [%s]" % (s.label, s.display_unit) for s in plan.specs]
        header = " | ".join(head_cells)
        lines = [header, "-" * len(header)]
        for d in rows:
            cells = ["%4d" % d["run"]]
            if has_kind:
                cells.append("%-5s" % d["kind"])
            cells += [_fmt(d[p]) for p in paths]
            lines.append(" | ".join(cells))
        self.preview.setPlainText("\n".join(lines))

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
