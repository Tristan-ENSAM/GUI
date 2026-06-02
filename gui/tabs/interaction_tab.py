# -*- coding: utf-8 -*-
"""
Interaction tab: tool-workpiece contact properties.

Mirrors the ContactProperty block in abq_odb_generator.py:
    IntProp.TangentialBehavior(formulation, table=((mu,),), fraction=slip_frac)
    IntProp.NormalBehavior(pressureOverclosure=HARD)
    # optional: IntProp.HeatGeneration(...)

Three sub-groups:
  - Tangential behaviour: penalty / rough / frictionless + μ + slip tol
  - Normal behaviour:     hard / soft (exponential) / soft (linear)
  - Heat generation:      enable + slave/master fractions
"""
from __future__ import annotations
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QScrollArea,
    QFrame,
)

from gui.core.model_config import ModelConfig
from gui.widgets.param_field import NumField, BoolField, PairRow


def _section_header(title: str):
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "background-color: #e8eef5; color: #1f4060; "
        "font-weight: bold; padding: 3px 6px; "
        "border-left: 3px solid #1f6fb2;"
    )
    return lbl


class InteractionTab(QWidget):
    """Editor for `cfg.interaction`. Emits `interactionChanged()` on edits."""

    interactionChanged = Signal()

    # Maps GUI string -> Abaqus symbolic constant name (used by the generator
    # at script-write time; we keep human-friendly strings in cfg).
    TANG_FORMULATIONS = [
        ("Penalty (Coulomb-like with µ)", "penalty"),
        ("Rough (no slip)",               "rough"),
        ("Frictionless",                  "frictionless"),
    ]
    NORMAL_FORMS = [
        ("Hard (no penetration)",        "hard"),
        ("Soft — exponential",           "exponential"),
        ("Soft — linear",                "linear"),
    ]

    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        lay.addWidget(self._build_tangential_group())
        lay.addWidget(self._build_normal_group())
        lay.addWidget(self._build_heat_group())
        lay.addStretch()

    # =====================================================================
    # Sub-groups
    # =====================================================================
    def _build_tangential_group(self) -> QGroupBox:
        g = QGroupBox("Tangential behaviour")
        v = QVBoxLayout(g)

        v.addWidget(_section_header("Formulation"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Formulation:"))
        self.cb_tang = QComboBox()
        for label, value in self.TANG_FORMULATIONS:
            self.cb_tang.addItem(label, value)
        idx = self.cb_tang.findData(self.cfg.interaction.tangential_formulation)
        if idx >= 0:
            self.cb_tang.setCurrentIndex(idx)
        self.cb_tang.setToolTip(
            "Penalty: Coulomb-style with a single friction coefficient μ. "
            "Default for chip-formation simulations.\n"
            "Rough: enforces no relative slip (full sticking).\n"
            "Frictionless: tangential stress = 0 (rarely realistic for cutting)."
        )
        row.addWidget(self.cb_tang, stretch=1)
        v.addLayout(row)

        v.addWidget(_section_header("Parameters"))
        self.f_mu   = NumField("μ (friction coefficient)", self.cfg.interaction.friction_coeff,
                               "—", minimum=0.0, maximum=5.0, compact=True)
        self.f_slip = NumField("Slip tolerance fraction", self.cfg.interaction.slip_tolerance,
                               "—", minimum=1e-6, maximum=1.0, compact=True)
        self.f_mu.setToolTip(
            "Coulomb friction coefficient. Typical for metal cutting:\n"
            "  - dry steel-steel:        0.3 – 0.6\n"
            "  - carbide on most metals: 0.2 – 0.4\n"
            "  - lubricated:             0.05 – 0.2"
        )
        self.f_slip.setToolTip(
            "Maximum elastic slip allowed before slip starts (PENALTY only).\n"
            "Default 0.005. Smaller = stiffer contact but smaller stable Δt."
        )
        v.addWidget(PairRow(self.f_mu, self.f_slip))

        self.cb_tang.currentIndexChanged.connect(self._on_change)
        self.f_mu.valueChanged.connect(self._on_change)
        self.f_slip.valueChanged.connect(self._on_change)

        # Initial visibility state
        self._refresh_tangential_visibility()
        return g

    def _build_normal_group(self) -> QGroupBox:
        g = QGroupBox("Normal behaviour")
        v = QVBoxLayout(g)

        v.addWidget(_section_header("Pressure-overclosure"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Type:"))
        self.cb_normal = QComboBox()
        for label, value in self.NORMAL_FORMS:
            self.cb_normal.addItem(label, value)
        idx = self.cb_normal.findData(self.cfg.interaction.pressure_overclosure)
        if idx >= 0:
            self.cb_normal.setCurrentIndex(idx)
        self.cb_normal.setToolTip(
            "Hard: penalty-stiffness enforces no penetration (default).\n"
            "Soft (exponential / linear): allows a small overclosure-pressure\n"
            "relationship — useful for surface roughness or compliant contact\n"
            "but parameters must be calibrated."
        )
        row.addWidget(self.cb_normal, stretch=1)
        v.addLayout(row)

        # When soft is selected we'd expose more parameters; left as a future
        # extension to keep the UI focused on what the CEL generator currently
        # uses.
        note = QLabel(
            "Note: Soft-contact parameters (clearance, max pressure...) are not\n"
            "exposed yet. Stay on Hard unless you have a specific need."
        )
        note.setStyleSheet("color: #888; font-style: italic;")
        v.addWidget(note)

        self.cb_normal.currentIndexChanged.connect(self._on_change)
        return g

    def _build_heat_group(self) -> QGroupBox:
        g = QGroupBox("Heat generation (friction → thermal)")
        v = QVBoxLayout(g)

        self.cb_heat = BoolField(
            "Enable friction-induced heating (HeatGeneration on the contact)",
            self.cfg.interaction.heat_generation,
        )
        self.cb_heat.setToolTip(
            "Adds *Gap Heat Generation to the ContactProperty.\n"
            "All frictional work converts to heat (η = 1 by default).\n"
            "The fractions below set how heat splits between the two surfaces."
        )
        v.addWidget(self.cb_heat)

        v.addWidget(_section_header("Heat partitioning"))
        self.f_heat_slave = NumField(
            "Fraction to slave  (workpiece)",
            self.cfg.interaction.heat_fraction_to_slave,
            "—", minimum=0.0, maximum=1.0, compact=True,
        )
        self.f_heat_master = NumField(
            "Fraction to master (tool)",
            self.cfg.interaction.heat_fraction_to_master,
            "—", minimum=0.0, maximum=1.0, compact=True,
        )
        self.f_heat_slave.setToolTip(
            "Fraction of frictional heat absorbed by the workpiece side.\n"
            "Equal split (0.5 / 0.5) is the Abaqus default."
        )
        self.f_heat_master.setToolTip(
            "Fraction of frictional heat absorbed by the tool side.\n"
            "Should typically sum with 'to slave' to 1.0."
        )
        v.addWidget(PairRow(self.f_heat_slave, self.f_heat_master))

        # Wire and set initial visibility
        self.cb_heat.valueChanged.connect(self._on_change)
        self.f_heat_slave.valueChanged.connect(self._on_change)
        self.f_heat_master.valueChanged.connect(self._on_change)
        self._refresh_heat_visibility()
        return g

    # =====================================================================
    # Visibility helpers
    # =====================================================================
    def _refresh_tangential_visibility(self):
        """μ and slip-tolerance only meaningful for PENALTY."""
        tang = self.cb_tang.currentData()
        is_penalty = (tang == "penalty")
        self.f_mu.setEnabled(is_penalty)
        self.f_slip.setEnabled(is_penalty)
        if not is_penalty:
            self.f_mu.setToolTip(
                f"Disabled: not applicable for {tang!r} formulation."
            )
            self.f_slip.setToolTip(
                f"Disabled: not applicable for {tang!r} formulation."
            )

    def _refresh_heat_visibility(self):
        enabled = self.cb_heat.value()
        self.f_heat_slave.setEnabled(enabled)
        self.f_heat_master.setEnabled(enabled)

    # =====================================================================
    # Sync widgets -> cfg
    # =====================================================================
    def _on_change(self, *_):
        self._pull_from_widgets()
        self._refresh_tangential_visibility()
        self._refresh_heat_visibility()
        self.interactionChanged.emit()

    def _pull_from_widgets(self):
        i = self.cfg.interaction
        i.tangential_formulation = self.cb_tang.currentData()
        i.friction_coeff         = self.f_mu.value()
        i.slip_tolerance         = self.f_slip.value()
        i.pressure_overclosure   = self.cb_normal.currentData()
        i.heat_generation         = self.cb_heat.value()
        i.heat_fraction_to_slave  = self.f_heat_slave.value()
        i.heat_fraction_to_master = self.f_heat_master.value()

    def apply_from_cfg(self):
        """Push cfg.interaction values into the widgets without firing
        the change signals (used when loading a profile)."""
        widgets = (self.cb_tang, self.f_mu, self.f_slip, self.cb_normal,
                   self.cb_heat, self.f_heat_slave, self.f_heat_master)
        for w in widgets:
            w.blockSignals(True)
        try:
            i = self.cfg.interaction
            idx = self.cb_tang.findData(i.tangential_formulation)
            if idx >= 0:
                self.cb_tang.setCurrentIndex(idx)
            self.f_mu.set_value(i.friction_coeff)
            self.f_slip.set_value(i.slip_tolerance)
            idx = self.cb_normal.findData(i.pressure_overclosure)
            if idx >= 0:
                self.cb_normal.setCurrentIndex(idx)
            self.cb_heat.set_value(i.heat_generation)
            self.f_heat_slave.set_value(i.heat_fraction_to_slave)
            self.f_heat_master.set_value(i.heat_fraction_to_master)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._refresh_tangential_visibility()
        self._refresh_heat_visibility()
