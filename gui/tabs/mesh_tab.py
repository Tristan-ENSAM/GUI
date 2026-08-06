# -*- coding: utf-8 -*-
"""
Mesh tab (simplified).

Owns only the global mesh seed: element size + the "discretize" option.
All per-element controls (integration order, hourglass control, distortion
control, kinematic split, etc.) were removed on purpose: those are advanced
choices that an expert should set directly in the Abaqus generator source,
not through the GUI. The element-type configuration stored in ModelConfig
keeps its defaults and is still emitted to the solver.

Shows a dedicated geometry preview with the mesh overlay always on, plus a
read-only "Derived quantities" panel (effective dims, element count, stable
time-increment estimate).
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSplitter,
    QGridLayout, QFrame, QScrollArea,
)

from gui.core.model_config import ModelConfig
from gui.widgets.param_field import NumField, BoolField
from gui.widgets.geometry_preview import GeometryPreview


def _section_header(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "background-color: #e8eef5; color: #1f4060; "
        "font-weight: bold; padding: 3px 6px; "
        "border-left: 3px solid #1f6fb2;"
    )
    return lbl


class MeshTab(QWidget):
    """Editor for the global mesh seed (element size + discretize).

    Emits `meshChanged()` whenever a parameter changes, so MainWindow can
    refresh the Geometry preview and flip the dirty bit."""

    meshChanged = Signal()

    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        form = QWidget()
        form_lay = QVBoxLayout(form)
        form_lay.setContentsMargins(8, 8, 8, 8)
        form_lay.setSpacing(8)
        form_lay.addWidget(self._build_seeds_group())
        form_lay.addWidget(self._build_info_panel())
        form_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(form)
        scroll.setMinimumWidth(360)
        splitter.addWidget(scroll)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.preview = GeometryPreview()
        right_lay.addWidget(self.preview)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 900])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        self._refresh()

    # =====================================================================
    # Sub-groups
    # =====================================================================
    def _build_seeds_group(self) -> QGroupBox:
        g = QGroupBox("Global mesh seeds")
        lay = QVBoxLayout(g)

        lay.addWidget(_section_header("Element size"))
        eg = self.cfg.euler_geometry

        self.f_es = NumField("elem_size", self.cfg.elem_size, "mm",
                             minimum=1e-9)
        self.f_es.setToolTip(
            "Target element edge length. Used as the global seed size.\n"
            "Smaller = finer mesh, higher element count, smaller stable dt.\n"
            "Typical for orthogonal cutting: 0.5 to 5 % of the chip thickness."
        )
        lay.addWidget(self.f_es)

        self.f_disc = BoolField(
            "Discretize dims to elem_size (floor)", eg.discretize,
        )
        self.f_disc.setToolTip(
            "If checked: workpiece/void dimensions are floored to a multiple\n"
            "of elem_size before meshing, so element sizes match elem_size\n"
            "exactly in both directions.\n"
            "If unchecked: Abaqus rounds the seed count per edge; actual\n"
            "element size may differ slightly (see Derived quantities)."
        )
        lay.addWidget(self.f_disc)

        note = QLabel(
            "Element type and advanced controls (integration order, hourglass,\n"
            "distortion control...) are set by an expert in the generator source."
        )
        note.setStyleSheet("color: #888; font-style: italic;")
        lay.addWidget(note)

        # ---- Tool mesh (nose seed + bias seeding) --------------------------
        lay.addWidget(_section_header("Tool mesh"))

        self.f_tool_es = NumField("min_elem_size (nose)",
                                  self.cfg.tool_elem_size, "mm", minimum=1e-9)
        self.f_tool_es.setToolTip(
            "Finest tool seed: the nose, and the fine end of the rake/clearance\n"
            "bias. The tool is a rigid body, so it does NOT constrain the\n"
            "mechanical stable increment — but its temperature dofs stay active,\n"
            "so this size drives the tool's CONDUCTION limit, which goes as L^2\n"
            "(halving it divides the tool's stable dt by 4)."
        )
        lay.addWidget(self.f_tool_es)

        self.f_inter = NumField("inter_elem_size (face end)",
                                self.cfg.inter_elem_size, "mm", minimum=1e-9)
        self.f_inter.setToolTip(
            "Coarse end of the rake + clearance face bias, AND fine end of the\n"
            "border bias (shared junction -> continuous size there).")
        lay.addWidget(self.f_inter)

        self.f_max = NumField("max_elem_size (border end)",
                              self.cfg.max_elem_size, "mm", minimum=1e-9)
        self.f_max.setToolTip(
            "Coarsest tool seed: the outer end of the two border edges.")
        lay.addWidget(self.f_max)

        for w in (self.f_es, self.f_disc, self.f_tool_es,
                  self.f_inter, self.f_max):
            w.valueChanged.connect(self._on_change)
        return g

    def _build_info_panel(self) -> QGroupBox:
        g = QGroupBox("Derived quantities")
        grid = QGridLayout(g)

        self.lbl_eff_dims = QLabel("-")
        self.lbl_eff_es   = QLabel("-")
        self.lbl_nel      = QLabel("-")
        self.lbl_dt       = QLabel("-")
        self.lbl_dt_tool  = QLabel("-")
        for lbl in (self.lbl_eff_dims, self.lbl_eff_es, self.lbl_nel,
                    self.lbl_dt, self.lbl_dt_tool):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        grid.addWidget(QLabel("Effective (h_wp, h_void, l_wp, l_void):"), 0, 0)
        grid.addWidget(self.lbl_eff_dims,              0, 1)
        grid.addWidget(QLabel("Effective elem_size:"), 1, 0)
        grid.addWidget(self.lbl_eff_es,                1, 1)
        grid.addWidget(QLabel("Eulerian element count (~):"), 2, 0)
        grid.addWidget(self.lbl_nel,                   2, 1)
        grid.addWidget(QLabel("Stable dt estimate (Eulerian, mechanical):"), 3, 0)
        grid.addWidget(self.lbl_dt,                    3, 1)
        grid.addWidget(QLabel("Tool conduction dt estimate:"), 4, 0)
        grid.addWidget(self.lbl_dt_tool,               4, 1)
        grid.setColumnStretch(1, 1)
        return g

    # =====================================================================
    # Sync widgets <-> cfg, then refresh preview & info panel
    # =====================================================================
    def _on_change(self, *_):
        self._pull_from_widgets()
        self._refresh()
        self.meshChanged.emit()

    def _pull_from_widgets(self):
        self.cfg.elem_size                 = self.f_es.value()
        self.cfg.euler_geometry.discretize = self.f_disc.value()
        self.cfg.tool_elem_size            = self.f_tool_es.value()
        self.cfg.inter_elem_size           = self.f_inter.value()
        self.cfg.max_elem_size             = self.f_max.value()

    def _refresh(self):
        # Preview always shows the mesh on this tab.
        self.preview.update_from_config(self.cfg, show_mesh=True)

        h_wp, h_void, l_wp, l_void = self.cfg.effective_euler_dims()
        self.lbl_eff_dims.setText(
            "%.6g, %.6g, %.6g, %.6g mm" % (h_wp, h_void, l_wp, l_void))

        es_x, es_y = self.cfg.effective_elem_sizes()
        if self.cfg.euler_geometry.discretize:
            self.lbl_eff_es.setText("%.6g mm  (= elem_size, dims floored)" % es_x)
        elif abs(es_x - es_y) < 1e-12:
            self.lbl_eff_es.setText("%.6g mm  (elem_size = %g)"
                                    % (es_x, self.cfg.elem_size))
        else:
            self.lbl_eff_es.setText("x: %.6g mm,  y: %.6g mm  (elem_size = %g)"
                                    % (es_x, es_y, self.cfg.elem_size))

        n = self.cfg.n_elements_estimate()
        self.lbl_nel.setText(("%d" % n))
        dt = self.cfg.stable_dt_estimate()
        if dt > 0:
            self.lbl_dt.setText("%.3e s" % dt)
        else:
            self.lbl_dt.setText("-  (need E, rho, elem_size > 0)")

        # Tool conduction limit: the tool is rigid (no mechanical contribution)
        # but its temperature dofs stay active, so this can govern the global
        # increment. It is NOT relaxed by mass scaling (rho*Cp invariant).
        dt_t = self.cfg.tool_thermal_dt_estimate()
        if dt_t > 0:
            txt = "%.3e s  (from nose seed; real mesh is usually finer)" % dt_t
            if dt > 0 and dt_t < dt:
                txt += "  — GOVERNS"
            self.lbl_dt_tool.setText(txt)
        else:
            self.lbl_dt_tool.setText("-  (need tool k, rho, Cp, seed > 0)")

    # =====================================================================
    # External hooks
    # =====================================================================
    def on_external_change(self):
        """Called by MainWindow when another tab edits something affecting
        the mesh display (geometry dims, materials...). The cfg is the source
        of truth, so we just re-render."""
        self._refresh()

    def apply_from_cfg(self):
        """Push cfg values into the widgets (used after Open / New)."""
        self.f_es.set_value(self.cfg.elem_size)
        self.f_disc.set_value(self.cfg.euler_geometry.discretize)
        self.f_tool_es.set_value(self.cfg.tool_elem_size)
        self.f_inter.set_value(self.cfg.inter_elem_size)
        self.f_max.set_value(self.cfg.max_elem_size)
        self._refresh()
