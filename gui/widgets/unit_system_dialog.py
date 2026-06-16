# -*- coding: utf-8 -*-
"""Modal dialog to choose the display unit system: four base units (mass,
length, time, temperature) plus a few named overrides (modulus, strength,
conductivity, specific heat, fracture energy, velocity), with presets."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QComboBox, QFormLayout, QVBoxLayout, QHBoxLayout, QLabel,
    QDialogButtonBox, QGroupBox, QPushButton,
)

from gui.core import unit_system as us


class UnitSystemDialog(QDialog):
    def __init__(self, parent, system: us.UnitSystem):
        super().__init__(parent)
        self.setWindowTitle("Unit system")
        self.setModal(True)
        self._start = us.UnitSystem.from_dict(system.to_dict())

        root = QVBoxLayout(self)

        intro = QLabel(
            "Choose the four base units; most quantities (density, velocity, "
            "strain rate, expansion) follow from them. A few quantities keep "
            "a conventional named unit you can override below.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        # --- preset row ---
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.cb_preset = QComboBox()
        self.cb_preset.addItem("— custom —", None)
        for name in us.PRESETS:
            self.cb_preset.addItem(name, name)
        self.cb_preset.activated.connect(self._on_preset)
        preset_row.addWidget(self.cb_preset, 1)
        root.addLayout(preset_row)

        # --- base units ---
        base_box = QGroupBox("Base units")
        base_form = QFormLayout(base_box)
        self.cb_mass = self._combo(us.MASS_CHOICES, system.mass)
        self.cb_length = self._combo(us.LENGTH_CHOICES, system.length)
        self.cb_time = self._combo(us.TIME_CHOICES, system.time)
        self.cb_temp = self._combo(us.TEMP_CHOICES, system.temp,
                                   labels={"C": "°C", "K": "K"})
        base_form.addRow("Mass", self.cb_mass)
        base_form.addRow("Length", self.cb_length)
        base_form.addRow("Time", self.cb_time)
        base_form.addRow("Temperature", self.cb_temp)
        root.addWidget(base_box)

        # --- named overrides ---
        ovr_box = QGroupBox("Named units (overrides)")
        ovr_form = QFormLayout(ovr_box)
        self.cb_modulus = self._combo(system.options_for("modulus"),
                                      system.modulus)
        self.cb_strength = self._combo(system.options_for("strength"),
                                       system.strength)
        self.cb_cond = self._combo(system.options_for("conductivity"),
                                   system.conductivity)
        self.cb_cp = self._combo(system.options_for("specific_heat"),
                                 system.specific_heat)
        self.cb_gf = self._combo(system.options_for("fracture_energy"),
                                 system.fracture_energy)
        self.cb_vel = self._combo(system.options_for("velocity"),
                                  system.velocity)
        ovr_form.addRow("Modulus (E)", self.cb_modulus)
        ovr_form.addRow("Strength (A, B, σy)", self.cb_strength)
        ovr_form.addRow("Conductivity", self.cb_cond)
        ovr_form.addRow("Specific heat", self.cb_cp)
        ovr_form.addRow("Fracture energy", self.cb_gf)
        ovr_form.addRow("Velocity", self.cb_vel)
        root.addWidget(ovr_box)

        note = QLabel(
            "Note: stored model values never change — only how they are "
            "displayed and entered. Saved with the profile.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-style: italic;")
        root.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _combo(self, choices, current, labels=None) -> QComboBox:
        cb = QComboBox()
        for c in choices:
            cb.addItem((labels or {}).get(c, c), c)
        idx = cb.findData(current)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        return cb

    def _on_preset(self, _idx):
        name = self.cb_preset.currentData()
        if not name:
            return
        s = us.PRESETS[name]
        for cb, val in (
            (self.cb_mass, s.mass), (self.cb_length, s.length),
            (self.cb_time, s.time), (self.cb_temp, s.temp),
            (self.cb_modulus, s.modulus), (self.cb_strength, s.strength),
            (self.cb_cond, s.conductivity), (self.cb_cp, s.specific_heat),
            (self.cb_gf, s.fracture_energy), (self.cb_vel, s.velocity),
        ):
            i = cb.findData(val)
            if i >= 0:
                cb.setCurrentIndex(i)

    def result_system(self) -> us.UnitSystem:
        return us.UnitSystem(
            mass=self.cb_mass.currentData(),
            length=self.cb_length.currentData(),
            time=self.cb_time.currentData(),
            temp=self.cb_temp.currentData(),
            modulus=self.cb_modulus.currentData(),
            strength=self.cb_strength.currentData(),
            conductivity=self.cb_cond.currentData(),
            specific_heat=self.cb_cp.currentData(),
            fracture_energy=self.cb_gf.currentData(),
            velocity=self.cb_vel.currentData(),
        )
