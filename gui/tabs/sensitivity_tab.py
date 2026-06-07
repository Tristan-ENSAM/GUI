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
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QCheckBox, QPlainTextEdit, QAbstractItemView, QSplitter, QComboBox,
)

from gui.sensitivity import param_registry as pr
from gui.sensitivity import morris_plan as mp
from gui.sensitivity import jacobian_plan as jac
from gui.results import qoi as qoi_mod

# QoI ticked by default — the ones we can also measure on the planing rig
# (cutting/feed forces and peak temperature).
_DEFAULT_QOIS = ("Fx_mean", "Fy_mean", "T_max")


class SensitivityTab(QWidget):
    def __init__(self, cfg, prefs_getter=None):
        super().__init__()
        self.cfg = cfg
        self._prefs_getter = prefs_getter
        self.plan = None                 # last generated plan (Morris or Jacobian)
        self.plan_kind = "morris"        # "morris" | "jacobian"
        self.selected_qois = []          # list[QoISpec]
        self._row_spec = {}              # table row -> ParamSpec

        root = QVBoxLayout(self)

        intro = QLabel(
            "Morris screening: tick the parameters to vary, set their "
            "min/max, pick the QoI, then generate the plan. Cost = "
            "N×(k+1) runs (k = number of ticked parameters)."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        split = QSplitter(Qt.Vertical)
        root.addWidget(split, 1)

        # ---- Parameters table ------------------------------------------
        param_box = QGroupBox("Parameters to vary")
        pv = QVBoxLayout(param_box)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Vary", "Parameter", "Min", "Max", "Delta", "Norm", "Unit"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (0, 2, 3, 4, 5, 6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        pv.addWidget(self.table)
        split.addWidget(param_box)

        # ---- QoI + controls + preview ----------------------------------
        bottom = QWidget()
        bl = QVBoxLayout(bottom)

        # ---- Method selector -------------------------------------------
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.cb_method = QComboBox()
        self.cb_method.addItems(["Morris (screening)", "Jacobian (finite diff.)"])
        self.cb_method.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self.cb_method)
        method_row.addSpacing(16)
        method_row.addWidget(QLabel("FD scheme:"))
        self.cb_scheme = QComboBox()
        self.cb_scheme.addItems(["central", "forward", "backward"])
        self.cb_scheme.currentIndexChanged.connect(self._update_cost)
        method_row.addWidget(self.cb_scheme)
        self.lbl_hint = QLabel("")
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

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Trajectories N:"))
        self.sp_N = QSpinBox(); self.sp_N.setRange(2, 500); self.sp_N.setValue(10)
        self.sp_N.valueChanged.connect(self._update_cost)
        ctrl.addWidget(self.sp_N)
        ctrl.addWidget(QLabel("Levels:"))
        self.sp_levels = QSpinBox(); self.sp_levels.setRange(2, 12)
        self.sp_levels.setValue(4)
        ctrl.addWidget(self.sp_levels)
        ctrl.addWidget(QLabel("Seed:"))
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 999999)
        self.sp_seed.setValue(0)
        ctrl.addWidget(self.sp_seed)
        ctrl.addStretch(1)
        self.lbl_cost = QLabel("—")
        f = QFont(); f.setBold(True); self.lbl_cost.setFont(f)
        ctrl.addWidget(self.lbl_cost)
        self.btn_gen = QPushButton("Generate plan")
        self.btn_gen.clicked.connect(self._on_generate)
        ctrl.addWidget(self.btn_gen)
        bl.addLayout(ctrl)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        bl.addWidget(self.status)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(QFont("Consolas, monospace"))
        self.preview.setPlaceholderText(
            "The generated plan (one row per run) appears here.")
        bl.addWidget(self.preview, 1)

        split.addWidget(bottom)
        split.setSizes([320, 320])

        self._populate_table()
        self._update_cost()
        self._on_method_changed()

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
            # category header row
            r = self.table.rowCount()
            self.table.insertRow(r)
            head = QTableWidgetItem(category)
            fnt = head.font(); fnt.setBold(True); head.setFont(fnt)
            head.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(r, 0, head)
            self.table.setSpan(r, 0, 1, 7)

            for spec in specs:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self._row_spec[r] = spec
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
                # cols 2/3: Morris min/max (editable), pre-filled
                lo, hi = pr.default_display_bounds(self.cfg, spec, tu)
                self.table.setItem(r, 2, QTableWidgetItem(_fmt(lo)))
                self.table.setItem(r, 3, QTableWidgetItem(_fmt(hi)))
                # col 4: Jacobian FD step (editable), default = half-band
                self.table.setItem(r, 4, QTableWidgetItem(_fmt((hi - lo) / 2.0)))
                # col 5: Normalize checkbox (Jacobian)
                nrm = QTableWidgetItem()
                nrm.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                nrm.setCheckState(Qt.Unchecked)
                self.table.setItem(r, 5, nrm)
                # col 6: unit (read-only)
                unit = QTableWidgetItem(spec.display_unit)
                unit.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(r, 6, unit)
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
        if item.column() == 0:       # "vary" checkbox toggled
            self._update_cost()

    # ------------------------------------------------------------------
    # Method handling
    # ------------------------------------------------------------------
    def _is_jacobian(self) -> bool:
        return self.cb_method.currentIndex() == 1

    def _scheme(self) -> str:
        return self.cb_scheme.currentText()

    def _on_method_changed(self, *args):
        jac_on = self._is_jacobian()
        self.cb_scheme.setEnabled(jac_on)
        self.sp_N.setEnabled(not jac_on)
        self.sp_levels.setEnabled(not jac_on)
        self.sp_seed.setEnabled(not jac_on)
        if jac_on:
            self.lbl_hint.setText("uses Delta + Norm columns")
        else:
            self.lbl_hint.setText("uses Min/Max columns")
        self._update_cost()

    def _update_cost(self):
        k = len(self._selected_rows())
        if k == 0:
            self.lbl_cost.setText("0 parameters selected")
        elif self._is_jacobian():
            self.lbl_cost.setText("k=%d  →  %d runs (%s FD)" % (
                k, jac.n_runs(k, self._scheme()), self._scheme()))
        else:
            self.lbl_cost.setText(
                "k=%d  →  %d runs" % (k, mp.n_runs(k, self.sp_N.value())))

    # ------------------------------------------------------------------
    # Generate the plan
    # ------------------------------------------------------------------
    def _collect_morris(self):
        selected = []
        for r, spec in self._selected_rows():
            try:
                lo = float(self.table.item(r, 2).text())
                hi = float(self.table.item(r, 3).text())
            except (ValueError, AttributeError):
                raise ValueError("%s: min/max must be numbers." % spec.label)
            selected.append((spec, lo, hi))
        return selected

    def _collect_jacobian(self):
        selected = []
        tu = self._temp_unit()
        for r, spec in self._selected_rows():
            try:
                delta = float(self.table.item(r, 4).text())
            except (ValueError, AttributeError):
                raise ValueError("%s: delta must be a number." % spec.label)
            norm = self.table.item(r, 5).checkState() == Qt.Checked
            x0 = pr.get_display(self.cfg, spec, tu)
            selected.append((spec, x0, delta, norm))
        return selected

    def _selected_qoi_specs(self):
        return [q for q in qoi_mod.REGISTRY
                if self._qoi_checks[q.id].isChecked()]

    def _on_generate(self):
        jacobian = self._is_jacobian()
        try:
            qois = self._selected_qoi_specs()
            if not qois:
                self._warn("Tick at least one QoI.")
                return
            if jacobian:
                selected = self._collect_jacobian()
                if not selected:
                    self._warn("Tick at least one parameter to vary.")
                    return
                plan = jac.build_plan(selected, scheme=self._scheme(),
                                      temp_unit=self._temp_unit())
            else:
                selected = self._collect_morris()
                if not selected:
                    self._warn("Tick at least one parameter to vary.")
                    return
                seed = self.sp_seed.value() or None
                plan = mp.build_plan(
                    selected, N=self.sp_N.value(),
                    num_levels=self.sp_levels.value(), seed=seed,
                    temp_unit=self._temp_unit())
        except ImportError:
            self._warn("SALib is not installed in this Python environment. "
                       "Install it:  pip install SALib")
            return
        except ValueError as e:
            self._warn(str(e))
            return
        except Exception as e:                       # pragma: no cover
            self._warn("Could not build the plan: %s" % e)
            return

        self.plan = plan
        self.plan_kind = "jacobian" if jacobian else "morris"
        self.selected_qois = qois
        method = ("Jacobian (%s FD)" % self._scheme()) if jacobian else "Morris"
        self.status.setStyleSheet("color: #15803d;")
        self.status.setText(
            "%s plan ready: %d parameters, %d runs, %d QoI (%s). "
            "Ready for the run step." % (
                method, plan.k, plan.n_runs, len(qois),
                ", ".join(q.id for q in qois)))
        self._show_preview(plan)

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
