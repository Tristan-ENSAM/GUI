# -*- coding: utf-8 -*-
"""
Analysis tab: select formulation (CEL vs Lagrangian) and configure
formulation-specific options.

Emits `analysisChanged()` whenever any setting changes, so the MainWindow
can broadcast the update to other tabs (notably the geometry preview).
"""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QButtonGroup,
    QLabel, QCheckBox, QComboBox, QScrollArea, QFrame,
)

from gui.core.model_config import ModelConfig


class AnalysisTab(QWidget):
    analysisChanged = Signal()

    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        # Inner widget holding the actual controls
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(12, 12, 12, 12)
        inner_lay.addWidget(self._build_formulation_group())
        inner_lay.addWidget(self._lagrangian_group)
        inner_lay.addStretch()

        # Scroll wrapper
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._refresh_visibility()

    # =====================================================================
    # UI construction
    # =====================================================================
    def _build_formulation_group(self) -> QGroupBox:
        g = QGroupBox("Formulation")
        lay = QVBoxLayout(g)

        info = QLabel(
            "Select the solver formulation. This determines which Abaqus\n"
            "generator script is launched and how the workpiece is modeled."
        )
        info.setStyleSheet("color: #555;")
        lay.addWidget(info)

        self.rb_cel = QRadioButton(
            "CEL — Coupled Eulerian-Lagrangian (current default)"
        )
        self.rb_cel.setToolTip(
            "Eulerian workpiece (material flows through a fixed mesh).\n"
            "Best for large deformations and chip formation without mesh\n"
            "distortion. Uses ExplicitDynamicsStep + VolFraction predefined\n"
            "field. Generator: abq_odb_generator.py."
        )
        self.rb_lag = QRadioButton(
            "Lagrangian — Continuum with element deletion"
        )
        self.rb_lag.setToolTip(
            "Deformable workpiece mesh that follows the material.\n"
            "Chip forms by material separation: fully-damaged elements\n"
            "are deleted (Johnson-Cook damage initiation + evolution).\n"
            "Uses TempDisplacementDynamicsStep (explicit, thermal-coupled).\n"
            "Generator: abq_lagrangian_generator.py."
        )
        # mutually exclusive
        self._bg_form = QButtonGroup(self)
        self._bg_form.addButton(self.rb_cel)
        self._bg_form.addButton(self.rb_lag)

        self.rb_cel.setChecked(self.cfg.analysis.formulation == "CEL")
        self.rb_lag.setChecked(self.cfg.analysis.formulation == "Lagrangian")

        lay.addWidget(self.rb_cel)
        lay.addWidget(self.rb_lag)

        self.rb_cel.toggled.connect(self._on_formulation_changed)
        self.rb_lag.toggled.connect(self._on_formulation_changed)

        # The Lagrangian group is constructed unconditionally and shown/hidden
        # by _refresh_visibility().
        self._lagrangian_group = self._build_lagrangian_group()
        return g

    def _build_lagrangian_group(self) -> QGroupBox:
        g = QGroupBox("Lagrangian options")
        lay = QVBoxLayout(g)

        # --- Kinematics ---
        kin_lay = QHBoxLayout()
        kin_lay.addWidget(QLabel("Kinematics:"))
        self.cb_motion = QComboBox()
        self.cb_motion.addItem("Tool moves (workpiece fixed)", "tool_moves")
        self.cb_motion.addItem("Workpiece moves (tool fixed) — like CEL",
                               "workpiece_moves")
        idx = self.cb_motion.findData(self.cfg.analysis.tool_motion)
        if idx >= 0:
            self.cb_motion.setCurrentIndex(idx)
        self.cb_motion.setToolTip(
            "Which body carries the cutting-speed velocity BC.\n"
            "Numerically equivalent in steady-state, but the choice may\n"
            "affect spurious oscillations at the transient onset."
        )
        kin_lay.addWidget(self.cb_motion)
        kin_lay.addStretch()
        lay.addLayout(kin_lay)

        # --- Tool body ---
        tool_lay = QHBoxLayout()
        tool_lay.addWidget(QLabel("Tool body:"))
        self.cb_tool_type = QComboBox()
        self.cb_tool_type.addItem("Rigid (RigidBody)", True)
        self.cb_tool_type.addItem("Deformable elastic", False)
        idx = self.cb_tool_type.findData(self.cfg.analysis.tool_rigid)
        if idx >= 0:
            self.cb_tool_type.setCurrentIndex(idx)
        self.cb_tool_type.setToolTip(
            "Rigid: fastest, no tool heating, same as the current CEL setup.\n"
            "Deformable: lets the tool heat up; ~2-3x slower runs."
        )
        tool_lay.addWidget(self.cb_tool_type)
        tool_lay.addStretch()
        lay.addLayout(tool_lay)

        # --- RP location ---
        rp_lay = QHBoxLayout()
        rp_lay.addWidget(QLabel("Tool Reference Point:"))
        self.cb_rp = QComboBox()
        self.cb_rp.addItem("TR — top-right corner (recommended)", "TR")
        self.cb_rp.addItem("BR — bottom-right corner",            "BR")
        self.cb_rp.addItem("Centroid",                            "centroid")
        idx = self.cb_rp.findData(self.cfg.analysis.rp_location)
        if idx >= 0:
            self.cb_rp.setCurrentIndex(idx)
        self.cb_rp.setToolTip(
            "Which point on the tool carries the kinematic BC.\n"
            "TR (top-right) is far from the plastically active cutting\n"
            "edge — numerically cleanest. BR is also acceptable.\n"
            "Centroid is rarely used (asymmetric inertia balance)."
        )
        rp_lay.addWidget(self.cb_rp)
        rp_lay.addStretch()
        lay.addLayout(rp_lay)

        # --- Element deletion ---
        self.cb_deletion = QCheckBox(
            "Enable element deletion (fully-damaged elements are removed)"
        )
        self.cb_deletion.setChecked(self.cfg.analysis.element_deletion)
        self.cb_deletion.setToolTip(
            "Required for chip formation by material separation in continuum\n"
            "Lagrangian cutting. Disable only for debugging."
        )
        lay.addWidget(self.cb_deletion)

        # --- wire change handlers ---
        self.cb_motion.currentIndexChanged.connect(self._on_setting_changed)
        self.cb_tool_type.currentIndexChanged.connect(self._on_setting_changed)
        self.cb_rp.currentIndexChanged.connect(self._on_setting_changed)
        self.cb_deletion.toggled.connect(self._on_setting_changed)

        return g

    # =====================================================================
    # Event handlers
    # =====================================================================
    def _on_formulation_changed(self, _checked: bool):
        # We only act on the "checked" transition to avoid double-firing.
        if not _checked:
            return
        self.cfg.analysis.formulation = "CEL" if self.rb_cel.isChecked() else "Lagrangian"
        self._refresh_visibility()
        self.analysisChanged.emit()

    def _on_setting_changed(self, *_):
        a = self.cfg.analysis
        a.tool_motion      = self.cb_motion.currentData()
        a.tool_rigid       = self.cb_tool_type.currentData()
        a.rp_location      = self.cb_rp.currentData()
        a.element_deletion = self.cb_deletion.isChecked()
        self.analysisChanged.emit()

    def apply_from_cfg(self):
        """Push cfg.analysis values into the widgets without firing the
        usual change signals. Used when loading a profile from disk."""
        a = self.cfg.analysis

        # Block signals while we mutate widgets, so a single explicit
        # `analysisChanged.emit()` at the end is the only notification.
        widgets = (self.rb_cel, self.rb_lag, self.cb_motion,
                   self.cb_tool_type, self.cb_rp, self.cb_deletion)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.rb_cel.setChecked(a.formulation == "CEL")
            self.rb_lag.setChecked(a.formulation == "Lagrangian")

            idx = self.cb_motion.findData(a.tool_motion)
            if idx >= 0:
                self.cb_motion.setCurrentIndex(idx)
            idx = self.cb_tool_type.findData(a.tool_rigid)
            if idx >= 0:
                self.cb_tool_type.setCurrentIndex(idx)
            idx = self.cb_rp.findData(a.rp_location)
            if idx >= 0:
                self.cb_rp.setCurrentIndex(idx)
            self.cb_deletion.setChecked(a.element_deletion)
        finally:
            for w in widgets:
                w.blockSignals(False)

        self._refresh_visibility()
        self.analysisChanged.emit()

    def _refresh_visibility(self):
        self._lagrangian_group.setVisible(
            self.cfg.analysis.formulation == "Lagrangian"
        )
