# -*- coding: utf-8 -*-
"""Identify tab, mode Physical parameters (design validated 2026-11-06).

Find the values of the ticked parameters whose QoI match a reference:
  * parameters: every model value (the table of Explore / Sensitivity),
    with Min / Max (search bounds), Start (empty = the model value) and
    the Note column;
  * reference: the six QoI of the sizing studies (Vx, Vy, T, EVF, Fc, Ff,
    with the ZOI, window T and eps_q of the Setup tab), typed or read from
    a finished run (numerical reference); a weight per QoI;
  * search: fixed point (one parameter at a time) or Levenberg-Marquardt,
    run budget, stagnation tau and n (gui.identify.engine).
Every run goes to <workdir>/<...>__identify__<stamp>/ and finished runs
with the same model parameters are reused (so a search started again
resumes). The history is written to identify.json after each iteration.
"""
from __future__ import annotations

import copy
import logging
import math
import warnings
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QCheckBox, QPlainTextEdit, QComboBox, QLineEdit, QFileDialog,
    QSplitter, QTabWidget, QProgressBar,
)

from gui.core.logging_util import log_swallowed
from gui.core.remote_exec import is_remote, launch_problems
from gui.identify import engine as ie
from gui.sensitivity import model_qoi as mq
from gui.sensitivity import param_registry as pr
from gui.widgets.machine_combo import MachineCombo
from gui.widgets.theme import mark_primary

_NOTE = {"output": "output only", "remesh": "remesh",
         "numerical": "numerical", "": ""}

# Columns of the parameter table
C_USE, C_NAME, C_REF, C_MIN, C_MAX, C_START, C_UNIT, C_NOTE = range(8)
# Columns of the reference table
R_USE, R_QOI, R_REF, R_EPS, R_W, R_UNIT = range(6)


def _note(spec) -> str:
    if spec.kind == "output":
        return "output"
    if spec.remesh:
        return "remesh"
    return "numerical" if spec.kind == "numerical" else ""


def _fmt(v) -> str:
    try:
        return "%.6g" % float(v)
    except (TypeError, ValueError):
        return ""


def setup_qoi_settings(getter, cfg, qois) -> dict:
    """ZOI, window T, EVF threshold and eps_q of the Setup tab for the QoI
    `qois`; ValueError naming what is missing."""
    if getter is None:
        raise ValueError("Identify needs the settings of the Setup tab.")
    try:
        st = getter()
        window = tuple(float(v) for v in st["window"])
        zoi = tuple(float(v) for v in st["zoi"])
    except Exception as e:
        raise ValueError("Set the ZOI and the window T in the Setup tab "
                         "first (%s)." % e)
    eps = {k: float(v) for k, v in (st.get("eps") or {}).items()}
    missing = [q for q in qois if not (eps.get(q) or 0) > 0]
    if missing:
        raise ValueError("Set ε_q of %s in the Setup tab first."
                         % ", ".join(missing))
    if not (zoi[0] < zoi[1] and zoi[2] < zoi[3]):
        raise ValueError("The ZOI of the Setup tab is empty.")
    if any(q in mq.FIELD_QUANTITIES for q in qois):
        b = cfg.bbox
        crop = {"xmin": float(b.xmin), "xmax": float(b.xmax),
                "ymin": float(b.ymin), "ymax": float(b.ymax)}
        if not mq.zoi_within_crop(zoi, crop):
            raise ValueError("The ZOI of the Setup tab is not inside the ROI "
                             "the runs extract: widen the ROI (Geometry "
                             "tab).")
    return {"zoi": zoi, "window": window,
            "evf_threshold": float(st.get("evf_threshold", 0.5)),
            "chip_y0": st.get("chip_y0"),
            "eps": {q: eps[q] for q in qois}}


class IdentifyTab(QWidget):
    """Identify, mode Physical parameters."""
    identified = Signal(dict)       # {path: value} applied to the model

    def __init__(self, cfg, prefs_getter=None, cpus_getter=None,
                 profile_name_getter=None):
        super().__init__()
        self.cfg = cfg
        self._prefs_getter = prefs_getter
        self._cpus_getter = cpus_getter
        self._profile_name_getter = profile_name_getter
        self._model_settings_getter = None
        self._known_folders_getter = None
        self._row_spec = {}
        self._header_rows = {}
        self._thread = None
        self._worker = None
        self._run_specs = []
        self._run_workdir = None
        self.last_result = None

        root = QVBoxLayout(self)
        help_lbl = QLabel(
            "Find the values of the ticked parameters whose QoI match a "
            "reference. Min and Max bound the search; Start empty = the "
            "model value. The reference is the QoI of the sizing studies "
            "(ZOI, window T and ε_q of the Setup tab), typed or read "
            "from a finished run. The search stops when every QoI is within "
            "ε_q, or when the run budget is spent. A parameter whose "
            "step moves no QoI by more than τ ε_q, n iterations in "
            "a row, is frozen; when all are frozen, one last iteration frees "
            "them all.\nFixed point: one parameter at a time (golden-section "
            "search over Min..Max). Levenberg-Marquardt: all together, "
            "finite-difference Jacobian (k runs forward, 2k central).")
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color:#6b7280;")
        help_lbl.setVisible(False)
        self.lbl_intro = help_lbl
        top = QHBoxLayout()
        top.addStretch(1)
        self.cb_help = QCheckBox("Show help")
        self.cb_help.toggled.connect(help_lbl.setVisible)
        top.addWidget(self.cb_help)
        root.addLayout(top)
        root.addWidget(help_lbl)

        split = QSplitter(Qt.Vertical)
        root.addWidget(split, 1)
        upper = QSplitter(Qt.Horizontal)
        split.addWidget(upper)

        # ---- parameters -------------------------------------------------
        pbox = QGroupBox("Parameters to identify")
        pv = QVBoxLayout(pbox)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Show:"))
        self.cb_filter = QComboBox()
        for label, key in (("All", "all"), ("Physical", "physical"),
                           ("Ticked", "ticked")):
            self.cb_filter.addItem(label, key)
        self.cb_filter.currentIndexChanged.connect(self._apply_filter)
        frow.addWidget(self.cb_filter)
        self.le_search = QLineEdit()
        self.le_search.setPlaceholderText("Search a parameter…")
        self.le_search.setClearButtonEnabled(True)
        self.le_search.textChanged.connect(self._apply_filter)
        frow.addWidget(self.le_search, 1)
        pv.addLayout(frow)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Parameter", "Ref", "Min", "Max", "Start", "Unit",
             "Note"])
        self.table.horizontalHeaderItem(C_START).setToolTip(
            "Start point of the search; empty = Ref (the model value).")
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(C_NAME, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        pv.addWidget(self.table)
        upper.addWidget(pbox)

        # ---- reference + search -----------------------------------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rbox = QGroupBox("Reference (QoI of the sizing studies)")
        rl = QVBoxLayout(rbox)
        self.ref_table = QTableWidget(len(mq.MODEL_QOIS), 6)
        self.ref_table.setHorizontalHeaderLabels(
            ["Use", "QoI", "Reference", "ε_q", "Weight", "Unit"])
        self.ref_table.verticalHeader().setVisible(False)
        for i, q in enumerate(mq.MODEL_QOIS):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            # The chip thickness is off by default: the six QoI of the
            # sizing studies are the default reference.
            chk.setCheckState(Qt.Unchecked if q == mq.CHIP_QOI
                              else Qt.Checked)
            self.ref_table.setItem(i, R_USE, chk)
            for c, text, editable in ((R_QOI, mq.LABELS[q], False),
                                      (R_REF, "", True), (R_EPS, "", False),
                                      (R_W, "1", True),
                                      (R_UNIT, mq.UNITS[q], False)):
                it = QTableWidgetItem(text)
                if not editable:
                    it.setFlags(Qt.ItemIsEnabled)
                self.ref_table.setItem(i, c, it)
        self.ref_table.item(0, R_EPS).setToolTip("From the Setup tab.")
        self.ref_table.resizeColumnsToContents()
        rl.addWidget(self.ref_table)
        brow = QHBoxLayout()
        self.btn_ref_run = QPushButton("Reference from a run…")
        self.btn_ref_run.setToolTip(
            "Numerical reference: read the QoI of a finished run "
            "(<job>.results.npz) with the ZOI and window T of the Setup tab.")
        self.btn_ref_run.clicked.connect(self._on_ref_from_run)
        brow.addWidget(self.btn_ref_run)
        brow.addStretch(1)
        rl.addLayout(brow)
        self.lbl_ref = QLabel("")
        self.lbl_ref.setWordWrap(True)
        self.lbl_ref.setStyleSheet("color:#6b7280;")
        rl.addWidget(self.lbl_ref)
        rv.addWidget(rbox)

        sbox = QGroupBox("Search")
        sg = QGridLayout(sbox)
        sg.addWidget(QLabel("Strategy:"), 0, 0)
        self.cb_strategy = QComboBox()
        self.cb_strategy.addItem("Levenberg-Marquardt (all together)", "lm")
        self.cb_strategy.addItem("Fixed point (one parameter at a time)",
                                 "fixed_point")
        self.cb_strategy.currentIndexChanged.connect(self._sync_controls)
        sg.addWidget(self.cb_strategy, 0, 1, 1, 3)
        sg.addWidget(QLabel("Run budget:"), 1, 0)
        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(1, 10000)
        self.spin_budget.setValue(50)
        self.spin_budget.setToolTip("The search stops after this many runs "
                                    "(runs read back from earlier folders "
                                    "do not count).")
        sg.addWidget(self.spin_budget, 1, 1)
        sg.addWidget(QLabel("τ:"), 1, 2)
        self.spin_tau = QDoubleSpinBox()
        self.spin_tau.setRange(0.0, 1.0)
        self.spin_tau.setDecimals(3)
        self.spin_tau.setSingleStep(0.05)
        self.spin_tau.setValue(ie.DEFAULT_TAU)
        self.spin_tau.setToolTip(
            "A parameter stagnates when its step moves every QoI by less "
            "than τ ε_q (max_q |dq/dp · Δp| / ε_q "
            "< τ).")
        sg.addWidget(self.spin_tau, 1, 3)
        sg.addWidget(QLabel("Stagnant iterations n:"), 2, 0)
        self.spin_nstag = QSpinBox()
        self.spin_nstag.setRange(1, 20)
        self.spin_nstag.setValue(ie.DEFAULT_N_STAG)
        self.spin_nstag.setToolTip("Iterations in a row below τ before "
                                   "the parameter is frozen.")
        sg.addWidget(self.spin_nstag, 2, 1)
        self.lbl_fd = QLabel("Finite differences:")
        sg.addWidget(self.lbl_fd, 3, 0)
        self.cb_fd = QComboBox()
        self.cb_fd.addItem("forward (k runs)", "forward")
        self.cb_fd.addItem("central (2k runs)", "central")
        sg.addWidget(self.cb_fd, 3, 1)
        self.lbl_fd_step = QLabel("step [% of Max-Min]:")
        sg.addWidget(self.lbl_fd_step, 3, 2)
        self.spin_fd_step = QDoubleSpinBox()
        self.spin_fd_step.setRange(0.01, 49.0)
        self.spin_fd_step.setValue(1.0)
        sg.addWidget(self.spin_fd_step, 3, 3)
        self.lbl_line = QLabel("1-D search interval [% of Max-Min]:")
        sg.addWidget(self.lbl_line, 4, 0, 1, 2)
        self.spin_line = QDoubleSpinBox()
        self.spin_line.setRange(0.01, 50.0)
        self.spin_line.setValue(1.0)
        self.spin_line.setToolTip(
            "The golden-section search of one parameter stops when its "
            "interval is this share of Max - Min (1 %: about 11 runs).")
        sg.addWidget(self.spin_line, 4, 2, 1, 2)
        rv.addWidget(sbox)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Machine:"))
        self.cmb_machine = MachineCombo(self._prefs_getter)
        run_row.addWidget(self.cmb_machine)
        self.btn_run = mark_primary(QPushButton("Run identification"))
        self.btn_run.clicked.connect(self._on_run)
        run_row.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        run_row.addWidget(self.btn_cancel)
        self.btn_apply = QPushButton("Use in model")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip("Write the best values found into the "
                                  "model.")
        self.btn_apply.clicked.connect(self._on_apply)
        run_row.addWidget(self.btn_apply)
        run_row.addStretch(1)
        rv.addLayout(run_row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        rv.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        rv.addWidget(self.status)
        rv.addStretch(1)
        upper.addWidget(right)

        # ---- outputs ------------------------------------------------------
        self.tabs_out = QTabWidget()
        self.iter_table = QTableWidget(0, 6)
        self.iter_table.setHorizontalHeaderLabels(
            ["Iteration", "Runs", "J", "max |q-q_ref|/ε_q", "Free",
             "Frozen"])
        self.iter_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.tabs_out.addTab(self.iter_table, "Iterations")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs_out.addTab(self.log, "Log")
        split.addWidget(self.tabs_out)

        self._populate_table()
        self._sync_controls()

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------
    def set_model_settings_getter(self, fn):
        self._model_settings_getter = fn
        self.refresh_eps()

    def set_known_folders_getter(self, fn):
        self._known_folders_getter = fn

    def refresh_machines(self):
        self.cmb_machine.refresh()

    def _temp_unit(self) -> str:
        return getattr(self.cfg.ui, "temp_unit", "C")

    # ------------------------------------------------------------------
    # parameter table
    # ------------------------------------------------------------------
    def _populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self._row_spec, self._header_rows = {}, {}
        tu = self._temp_unit()
        self._table_units = pr.current_system(tu)
        for category, specs in pr.registry_by_category().items():
            h = self.table.rowCount()
            self.table.insertRow(h)
            head = QTableWidgetItem(category)
            f = head.font(); f.setBold(True); head.setFont(f)
            head.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(h, 0, head)
            self.table.setSpan(h, 0, 1, self.table.columnCount())
            self._header_rows[h] = []
            for spec in specs:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self._row_spec[r] = spec
                self._header_rows[h].append(r)
                chk = QTableWidgetItem()
                chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                chk.setCheckState(Qt.Unchecked)
                self.table.setItem(r, C_USE, chk)
                name = QTableWidgetItem(spec.label)
                name.setFlags(Qt.ItemIsEnabled)
                name.setToolTip(spec.path)
                self.table.setItem(r, C_NAME, name)
                for c in (C_REF, C_UNIT, C_NOTE):
                    it = QTableWidgetItem("")
                    it.setFlags(Qt.ItemIsEnabled)
                    self.table.setItem(r, c, it)
                for c in (C_MIN, C_MAX, C_START):
                    self.table.setItem(r, c, QTableWidgetItem(""))
                self.table.item(r, C_NOTE).setText(_NOTE[_note(spec)])
                self._fill_row(r, spec, bounds=True)
        self.table.blockSignals(False)
        self._apply_filter()

    def _fill_row(self, r, spec, bounds=False):
        tu = self._temp_unit()
        ok = pr.is_set(self.cfg, spec)
        chk = self.table.item(r, C_USE)
        if not ok:
            chk.setCheckState(Qt.Unchecked)
            chk.setFlags(Qt.NoItemFlags)
            chk.setToolTip("Not set in the model.")
            self.table.item(r, C_REF).setText("not set")
        else:
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setToolTip("")
            self.table.item(r, C_REF).setText(
                _fmt(pr.get_display(self.cfg, spec, tu)))
            if bounds:
                lo, hi = pr.default_display_bounds(self.cfg, spec, tu)
                self.table.item(r, C_MIN).setText(_fmt(lo))
                self.table.item(r, C_MAX).setText(_fmt(hi))
        self.table.item(r, C_UNIT).setText(spec.unit_str(tu))

    def refresh_from_model(self):
        """Mirror the current model: Ref column; untouched rows also get
        their default bounds."""
        self.table.blockSignals(True)
        try:
            for r, spec in self._row_spec.items():
                ticked = self.table.item(r, C_USE).checkState() == Qt.Checked
                self._fill_row(r, spec, bounds=not ticked)
        finally:
            self.table.blockSignals(False)
        self.refresh_eps()

    def _apply_filter(self, *_):
        mode = self.cb_filter.currentData() or "all"
        words = self.le_search.text().strip().lower()
        for head, rows in self._header_rows.items():
            shown = False
            for r in rows:
                spec = self._row_spec[r]
                keep = True
                if mode == "physical":
                    keep = spec.kind == "physical"
                elif mode == "ticked":
                    keep = self.table.item(r, C_USE).checkState() == Qt.Checked
                if keep and words:
                    keep = (words in spec.label.lower()
                            or words in spec.path.lower()
                            or words in spec.category.lower())
                self.table.setRowHidden(r, not keep)
                shown = shown or keep
            self.table.setRowHidden(head, not shown)

    def _on_item_changed(self, item):
        if item.column() == C_USE:
            self._sync_controls()

    def _selected_rows(self):
        return [(r, s) for r, s in self._row_spec.items()
                if self.table.item(r, C_USE).checkState() == Qt.Checked]

    def _cell(self, table, r, c):
        it = table.item(r, c)
        txt = (it.text() if it is not None else "").strip().replace(",", ".")
        if not txt:
            return None
        return float(txt)

    def collect_params(self):
        """(specs, engine Params) of the ticked rows, displayed units.
        ValueError naming the row."""
        specs, params = [], []
        tu = self._temp_unit()
        for r, spec in self._selected_rows():
            try:
                lo, hi = self._cell(self.table, r, C_MIN), \
                    self._cell(self.table, r, C_MAX)
                if lo is None or hi is None:
                    raise ValueError("Min and Max must be numbers.")
                start = self._cell(self.table, r, C_START)
                if start is None:
                    start = pr.get_display(self.cfg, spec, tu,
                                           system=self._table_units)
                params.append(ie.Param(spec.path, lo, hi, start=start))
            except ValueError as e:
                raise ValueError("%s: %s" % (spec.label, e))
            specs.append(spec)
        if not params:
            raise ValueError("Tick at least one parameter to identify.")
        return specs, params

    # ------------------------------------------------------------------
    # reference
    # ------------------------------------------------------------------
    def _used_qois(self):
        return [q for i, q in enumerate(mq.MODEL_QOIS)
                if self.ref_table.item(i, R_USE).checkState() == Qt.Checked]

    def refresh_eps(self):
        """eps_q column from the Setup tab."""
        eps = {}
        if self._model_settings_getter is not None:
            try:
                eps = self._model_settings_getter().get("eps") or {}
            except Exception:
                log_swallowed("reading eps_q of the Setup tab",
                              level=logging.DEBUG)
        for i, q in enumerate(mq.MODEL_QOIS):
            v = eps.get(q)
            self.ref_table.item(i, R_EPS).setText(_fmt(v) if v else "")

    def collect_targets(self):
        """(qois, engine Targets, Setup settings). ValueError when a
        reference, a weight or a Setup setting is missing."""
        qois = self._used_qois()
        if not qois:
            raise ValueError("Tick at least one QoI of the reference.")
        st = setup_qoi_settings(self._model_settings_getter, self.cfg, qois)
        targets = []
        for i, q in enumerate(mq.MODEL_QOIS):
            if q not in qois:
                continue
            try:
                ref = self._cell(self.ref_table, i, R_REF)
                w = self._cell(self.ref_table, i, R_W)
            except ValueError:
                raise ValueError("%s: the reference and the weight must be "
                                 "numbers." % mq.LABELS[q])
            if ref is None:
                raise ValueError("%s: give its reference value (or read it "
                                 "from a run)." % mq.LABELS[q])
            targets.append(ie.Target(q, ref, st["eps"][q],
                                     1.0 if w is None else w))
        return qois, targets, st

    def set_reference(self, values: dict, source: str = ""):
        for i, q in enumerate(mq.MODEL_QOIS):
            if q in values and values[q] is not None \
                    and math.isfinite(float(values[q])):
                self.ref_table.item(i, R_REF).setText(_fmt(values[q]))
        self.lbl_ref.setText(("Reference: %s" % source) if source else "")

    def _on_ref_from_run(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference run", "",
            "Run results (*.results.npz);;All files (*)")
        if path:
            self.reference_from_run(path)

    def reference_from_run(self, path):
        """Fill the references with the QoI of a finished run, computed
        with the ZOI, window T and EVF threshold of the Setup tab."""
        from gui.results.reader import ResultsBundle
        from gui.identify.worker import point_qoi
        try:
            qois = self._used_qois() or list(mq.SIZING_QOIS)
            st = setup_qoi_settings(self._model_settings_getter, self.cfg,
                                    qois)
            specs = mq.make_specs(st["zoi"], st["window"],
                                  st["evf_threshold"], qois=qois,
                                  chip_y0=st.get("chip_y0"))
            q = point_qoi(ResultsBundle.load(path), specs)
        except Exception as e:
            self._warn("Reference not read: %s" % e)
            return False
        self.set_reference(q, "run %s" % Path(path).name)
        return True

    # ------------------------------------------------------------------
    # search settings
    # ------------------------------------------------------------------
    def _sync_controls(self, *_):
        lm = self.cb_strategy.currentData() == "lm"
        for w in (self.lbl_fd, self.cb_fd, self.lbl_fd_step,
                  self.spin_fd_step):
            w.setVisible(lm)
        for w in (self.lbl_line, self.spin_line):
            w.setVisible(not lm)

    def search_settings(self) -> dict:
        return {"strategy": self.cb_strategy.currentData(),
                "budget": int(self.spin_budget.value()),
                "tau": float(self.spin_tau.value()),
                "n_stag": int(self.spin_nstag.value()),
                "fd_scheme": self.cb_fd.currentData(),
                "fd_step": float(self.spin_fd_step.value()) / 100.0,
                "line_tol": float(self.spin_line.value()) / 100.0}

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def _warn(self, msg):
        self.status.setStyleSheet("color:#b91c1c;")
        self.status.setText(msg)

    def _current_cpus(self) -> int:
        try:
            return int(self._cpus_getter()) if self._cpus_getter else 1
        except Exception:
            return 1

    def is_running(self) -> bool:
        return self._thread is not None

    def make_worker(self, specs, params, targets, qois, st, workdir,
                    prefs=None, solve_fn=None):
        from gui.identify.worker import IdentifyWorker
        known = []
        if self._known_folders_getter is not None:
            try:
                known = list(self._known_folders_getter())
            except Exception:
                log_swallowed("listing known folders", level=logging.DEBUG)
        kw = dict(abaqus_cmd=getattr(prefs, "abaqus_cmd", ""),
                  abaqus_script=getattr(prefs, "abaqus_script", ""),
                  workdir=str(workdir), cpus=self._current_cpus(),
                  remote_prefs=prefs if prefs is not None
                  and is_remote(prefs) else None,
                  cache_folders=known, solve_fn=solve_fn)
        return IdentifyWorker(
            specs, params, targets,
            mq.make_specs(st["zoi"], st["window"], st["evf_threshold"],
                          qois=qois, chip_y0=st.get("chip_y0")),
            copy.deepcopy(self.cfg), settings={
                "zoi": list(st["zoi"]), "window": list(st["window"]),
                "evf_threshold": st["evf_threshold"], "eps": st["eps"]},
            temp_unit=self._temp_unit(), unit_system=self._table_units,
            search=self.search_settings(), **kw)

    def _on_run(self):
        if self._thread is not None:
            self._warn("An identification is already running.")
            return
        try:
            specs, params = self.collect_params()
            qois, targets, st = self.collect_targets()
        except ValueError as e:
            self._warn(str(e))
            return
        prefs = self.cmb_machine.run_prefs() if self._prefs_getter else None
        if prefs is None:
            self._warn("No preferences available (Abaqus command/script, "
                       "machine).")
            return
        problems = launch_problems(prefs, prefs.default_workdir)
        wd = Path(prefs.default_workdir)
        try:
            wd.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            problems.append("Cannot create workdir '%s': %s" % (wd, e))
        if problems:
            self._warn("Cannot launch:\n• " + "\n• ".join(problems))
            return
        from gui.core.run_output import create_study_dir
        try:
            pname = (self._profile_name_getter()
                     if self._profile_name_getter else None)
        except Exception:
            pname = None
        rec = {"study": "identify", "search": self.search_settings(),
               "params": [{"path": p.path, "lo": p.lo, "hi": p.hi,
                           "start": p.start} for p in params],
               "targets": [{"qid": t.qid, "ref": t.ref, "eps": t.eps,
                            "weight": t.weight} for t in targets]}
        try:
            wd = create_study_dir(wd, pname, "identify", rec,
                                  machine=self.cmb_machine.machine_host())
        except OSError:
            pass
        self.start(self.make_worker(specs, params, targets, qois, st, wd,
                                    prefs=prefs), specs, wd)

    def start(self, worker, specs, workdir):
        """Run `worker` in a thread (tests pass a worker with a fake
        solver)."""
        self._run_specs = list(specs)
        self._run_workdir = workdir
        self.iter_table.setRowCount(0)
        self.log.clear()
        self.progress.setRange(0, int(worker._search.get("budget", 1)))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_apply.setEnabled(False)
        self.status.setStyleSheet("color:#1d4ed8;")
        self.status.setText("Identification running…")
        self._worker = worker
        self._thread = QThread(self)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(lambda d, t: self.progress.setValue(d))
        worker.log.connect(self._on_log)
        worker.iteration.connect(self._on_iteration)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_log(self, text):
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)

    def _on_iteration(self, it):
        r = self.iter_table.rowCount()
        self.iter_table.insertRow(r)
        for c, v in enumerate((str(it.iteration), str(it.runs), _fmt(it.J),
                               _fmt(it.max_ratio), ", ".join(it.free),
                               ", ".join(it.frozen))):
            self.iter_table.setItem(r, c, QTableWidgetItem(v))

    _STATUS = {
        ie.CONVERGED: "Every QoI is within ε_q.",
        ie.BUDGET: "Stopped: run budget spent (or cancelled).",
        ie.STALLED: "Stopped: every parameter stagnates.",
        ie.NO_PROGRESS: "Stopped: no step lowers the cost.",
        ie.FAILED: "Stopped: the start point could not be run.",
    }

    def _on_finished(self, res):
        self.last_result = res
        self._teardown()
        ok = res.status == ie.CONVERGED
        self.status.setStyleSheet("color:#15803d;" if ok else
                                  "color:#b45309;")
        vals = ", ".join("%s = %s" % (s.label, _fmt(v))
                         for s, v in zip(self._run_specs, res.x))
        self.status.setText("%s %s\nBest point: %s (J = %s, max |q-q_ref|/"
                            "ε_q = %s, %d runs).%s"
                            % (self._STATUS.get(res.status, res.status),
                               res.message, vals, _fmt(res.J),
                               _fmt(res.max_ratio), res.n_runs,
                               ("\nFolder: %s" % self._run_workdir)
                               if self._run_workdir else ""))
        self.btn_apply.setEnabled(res.q is not None)

    def _on_failed(self, msg):
        self._teardown()
        self._warn("Identification failed: %s" % msg)

    def _teardown(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Cancelling after the current run…")

    def shutdown(self, timeout_ms: int = 60000) -> bool:
        worker, thread = self._worker, self._thread
        if worker is None or thread is None:
            return True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for sig in (worker.progress, worker.log, worker.iteration,
                        worker.finished, worker.failed):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
        worker.stop_blocking()
        thread.quit()
        ended = bool(thread.wait(int(timeout_ms)))
        self._thread = None
        self._worker = None
        return ended

    def _on_apply(self):
        """Write the best point into the model."""
        res = self.last_result
        if res is None:
            return
        tu = self._temp_unit()
        applied = {}
        for spec, v in zip(self._run_specs, res.x):
            pr.apply_display(self.cfg, spec, float(v), tu,
                             system=self._table_units)
            applied[spec.path] = float(v)
        self.refresh_from_model()
        self.identified.emit(applied)
        self.status.setText(self.status.text()
                            + "\nWritten into the model.")
