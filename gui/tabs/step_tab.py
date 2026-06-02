# -*- coding: utf-8 -*-
"""
Step tab.

Owns the dynamic step parameters:

  - Step duration (`sim_time`, in seconds)
  - Field-output sampling (`n_frames`, number of intervals over the step)
  - Field-output variables (individual checkboxes per Abaqus identifier)
  - History-output settings (PRESELECT, RP forces, sampling interval)

These all map onto `cfg.step.*` (`StepCfg`), serialised into the
`step` block of both the JSON profile and the params dict passed to
the Abaqus generator.
"""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QGridLayout,
    QScrollArea, QFrame, QCheckBox,
)

from gui.core.model_config import ModelConfig
from gui.widgets.param_field import NumField, IntField


def _section_header(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "background-color: #e8eef5; color: #1f4060; "
        "font-weight: bold; padding: 3px 6px; "
        "border-left: 3px solid #1f6fb2;"
    )
    return lbl


class StepTab(QWidget):
    """Editor for `cfg.step` (StepCfg). Emits `stepChanged` on edits."""

    stepChanged = Signal()

    # Field-output variables, grouped by category. Each entry:
    #   (cfg attribute name, Abaqus identifier, short human description)
    # The Abaqus identifier is what ends up in the .inp file (and what
    # abq_odb_generator.py joins together for the *Output card).
    FIELD_VARS = {
        "Mechanical (element)": [
            ("fo_S",      "S",      "Stress tensor"),
            ("fo_PEEQ",   "PEEQ",   "Equivalent plastic strain"),
            ("fo_VP",     "VP",     "Viscoplastic strain"),
            ("fo_P",      "P",      "Hydrostatic pressure"),
            ("fo_ERV",    "ERV",    "von Mises equivalent strain rate"),
        ],
        "Thermal (element)": [
            ("fo_TEMP",   "TEMP",   "Element-averaged temperature"),
            ("fo_HFL",    "HFL",    "Heat flux vector"),
            ("fo_HP",     "HP",     "Heat power per unit volume"),
        ],
        "Eulerian-specific (element)": [
            ("fo_EVF",    "EVF",    "Element volume fraction"),
            ("fo_MFL",    "MFL",    "Mass flux"),
            ("fo_A",      "A",      "Nodal acceleration"),
            ("fo_V",      "V",      "Nodal velocity"),
        ],
        "Damage / failure (element)": [
            ("fo_DMICRT", "DMICRT", "Damage initiation criterion"),
            ("fo_SDEG",   "SDEG",   "Stiffness degradation"),
            ("fo_STATUS", "STATUS", "Element status (1 active / 0 deleted)"),
            ("fo_SDV",    "SDV",    "Solution-dependent state variables"),
        ],
        "Contact": [
            ("fo_CSTRESS", "CSTRESS", "Contact stresses (CPRESS + CSHEAR)"),
        ],
        "Nodal (always useful)": [
            ("fo_U",      "U",      "Nodal displacement"),
            ("fo_RF",     "RF",     "Nodal reaction force"),
            ("fo_NT",     "NT",     "Nodal temperature"),
        ],
    }

    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        # Inner widget holding all controls
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(10, 10, 10, 10)
        inner_lay.setSpacing(10)
        inner_lay.addWidget(self._build_duration_group())
        inner_lay.addWidget(self._build_mass_scaling_group())
        inner_lay.addWidget(self._build_field_output_group())
        inner_lay.addWidget(self._build_history_output_group())
        inner_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # =====================================================================
    # Sub-groups
    # =====================================================================
    def _build_duration_group(self) -> QGroupBox:
        g = QGroupBox("Step duration & sampling")
        v = QVBoxLayout(g)
        v.setSpacing(4)
        v.addWidget(_section_header("Dynamic step"))

        self.f_sim_time = NumField(
            "Total simulation time [s]", self.cfg.step.sim_time, "",
            minimum=1e-9, maximum=1e3, decimals=6,
        )
        self.f_sim_time.setToolTip(
            "Duration of the *Dynamic, Explicit step. Typical orthogonal-cutting\n"
            "runs are between 1e-4 and 1e-3 s, depending on cutting speed and\n"
            "Eulerian domain length."
        )
        v.addWidget(self.f_sim_time)

        self.f_n_frames = IntField(
            "Number of field-output frames", self.cfg.step.n_frames,
            minimum=1, maximum=100000,
        )
        self.f_n_frames.setToolTip(
            "Number of equally-spaced frames written to the .odb across the step.\n"
            "More frames = larger .odb but smoother time series.\n"
            "Sampling interval (s) = sim_time / n_frames."
        )
        v.addWidget(self.f_n_frames)

        # Live indicator of the sampling interval
        self.lbl_dt = QLabel()
        self.lbl_dt.setStyleSheet(
            "QLabel { color: #555; font-style: italic; padding-left: 4px; }"
        )
        v.addWidget(self.lbl_dt)

        self.f_sim_time.valueChanged.connect(self._on_change)
        self.f_n_frames.valueChanged.connect(self._on_change)
        self._refresh_dt_label()
        return g

    def _build_mass_scaling_group(self) -> QGroupBox:
        """Mass scaling controls.

        When enabled, the Eulerian (workpiece) material's density is
        multiplied by `mass_scaling_factor_eulerian` AND its specific
        heat Cp is divided by the same factor — preserving the thermal
        diffusivity k/(ρ·Cp). Same logic for the tool. Stable time-step
        scales as sqrt(factor), so a factor of 100 → ~10× speedup.

        Use with caution: only valid in regimes where inertia is not
        dominant (e.g. quasi-static cutting at low cutting speed). The
        tool factor often stays at 1.0 because the tool is rigid in CEL.
        """
        g = QGroupBox("Mass scaling (CEL only)")
        v = QVBoxLayout(g)
        v.setSpacing(4)

        # Master toggle
        self.cb_ms_enabled = QCheckBox("Enable mass scaling")
        self.cb_ms_enabled.setChecked(self.cfg.step.mass_scaling_enabled)
        self.cb_ms_enabled.setToolTip(
            "When enabled, the GUI multiplies each material's density by\n"
            "its factor at .inp write time, and divides the matching Cp\n"
            "by the same factor. This preserves k/(rho*Cp); only mechanical\n"
            "inertia is artificially scaled."
        )
        self.cb_ms_enabled.toggled.connect(self._on_change)
        v.addWidget(self.cb_ms_enabled)

        explainer = QLabel(
            "Effective: ρ_eff = factor × ρ  ;  Cp_eff = Cp / factor. "
            "Speed-up scales as √factor."
        )
        explainer.setStyleSheet("color: #666; font-style: italic;")
        explainer.setWordWrap(True)
        v.addWidget(explainer)

        # Factor: eulerian
        self.f_ms_eul = NumField(
            "Factor (Eulerian / workpiece)",
            self.cfg.step.mass_scaling_factor_eulerian, "",
            minimum=1.0, maximum=1e8, decimals=3,
        )
        self.f_ms_eul.setToolTip(
            "Mass-scaling factor applied to the Eulerian (workpiece)\n"
            "material. Typical values: 1 (no scaling) to ~1000."
        )
        v.addWidget(self.f_ms_eul)
        self.f_ms_eul.valueChanged.connect(self._on_change)

        # Factor: tool
        self.f_ms_tool = NumField(
            "Factor (Tool)",
            self.cfg.step.mass_scaling_factor_tool, "",
            minimum=1.0, maximum=1e8, decimals=3,
        )
        self.f_ms_tool.setToolTip(
            "Mass-scaling factor applied to the tool material. Often kept\n"
            "at 1.0 since the tool is rigid in CEL and mass scaling has\n"
            "no effect on rigid bodies."
        )
        v.addWidget(self.f_ms_tool)
        self.f_ms_tool.valueChanged.connect(self._on_change)

        # Live indicator: stable-dt speedup estimate (sqrt of factor)
        self.lbl_ms_speedup = QLabel()
        self.lbl_ms_speedup.setStyleSheet(
            "QLabel { color: #555; font-style: italic; padding-left: 4px; }"
        )
        v.addWidget(self.lbl_ms_speedup)

        self._refresh_ms_visibility()
        return g

    def _refresh_ms_visibility(self):
        """Grey out the factor fields when mass scaling is disabled, and
        refresh the speed-up estimate label."""
        enabled = self.cb_ms_enabled.isChecked()
        self.f_ms_eul.setEnabled(enabled)
        self.f_ms_tool.setEnabled(enabled)
        if enabled:
            f_eul = self.f_ms_eul.value()
            self.lbl_ms_speedup.setText(
                f"≈ stable-dt speed-up: √{f_eul:.0f} ≈ {f_eul ** 0.5:.2f}×"
            )
        else:
            self.lbl_ms_speedup.setText("(disabled — materials unchanged)")

    def _build_field_output_group(self) -> QGroupBox:
        g = QGroupBox("Field output — variables")
        v = QVBoxLayout(g)
        v.setSpacing(4)
        explainer = QLabel(
            "Tick the variables you want written to the .odb at each frame. "
            "Unticking a variable shrinks the .odb file size."
        )
        explainer.setStyleSheet("color: #666; font-style: italic;")
        explainer.setWordWrap(True)
        v.addWidget(explainer)

        # Master select-all / clear-all row
        all_row = QHBoxLayout()
        from PySide6.QtWidgets import QPushButton
        btn_all  = QPushButton("Select all")
        btn_none = QPushButton("Clear all")
        btn_def  = QPushButton("Defaults")
        btn_all.clicked.connect(lambda: self._set_all_field_outputs(True))
        btn_none.clicked.connect(lambda: self._set_all_field_outputs(False))
        btn_def.clicked.connect(self._set_field_outputs_default)
        all_row.addWidget(btn_all); all_row.addWidget(btn_none)
        all_row.addWidget(btn_def); all_row.addStretch()
        all_w = QWidget(); all_w.setLayout(all_row)
        v.addWidget(all_w)

        # Per-category groups of checkboxes
        self._fo_checkboxes: dict[str, QCheckBox] = {}
        for category, items in self.FIELD_VARS.items():
            v.addWidget(_section_header(category))
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8); grid.setVerticalSpacing(2)
            for i, (attr, ident, desc) in enumerate(items):
                cb = QCheckBox(f"{ident}")
                cb.setToolTip(desc)
                cb.setChecked(getattr(self.cfg.step.output, attr))
                cb.toggled.connect(self._on_change)
                self._fo_checkboxes[attr] = cb
                # 3 columns
                grid.addWidget(cb, i // 3, i % 3)
            v.addLayout(grid)

        return g

    def _build_history_output_group(self) -> QGroupBox:
        g = QGroupBox("History output")
        v = QVBoxLayout(g)
        v.setSpacing(4)

        self.cb_ho_preselect = QCheckBox("PRESELECT (default Abaqus history variables)")
        self.cb_ho_preselect.setChecked(self.cfg.step.output.ho_preselect)
        self.cb_ho_preselect.setToolTip(
            "Abaqus's PRESELECT history variable list. Lightweight."
        )
        self.cb_ho_preselect.toggled.connect(self._on_change)
        v.addWidget(self.cb_ho_preselect)

        self.cb_ho_rf_on_rp = QCheckBox("Reaction forces on Tool RP (RF1, RF2)")
        self.cb_ho_rf_on_rp.setChecked(self.cfg.step.output.ho_rf_on_rp)
        self.cb_ho_rf_on_rp.setToolTip(
            "History output of RF1 and RF2 at the tool's reference point —\n"
            "directly readable as cutting forces (Fx, Fy) over time."
        )
        self.cb_ho_rf_on_rp.toggled.connect(self._on_change)
        v.addWidget(self.cb_ho_rf_on_rp)

        self.f_ho_dt = NumField(
            "History sampling interval [s]",
            self.cfg.step.output.ho_time_interval, "",
            minimum=1e-12, maximum=1.0, decimals=10,
        )
        self.f_ho_dt.setToolTip(
            "Time interval between two history-output samples. Default 1e-6 s\n"
            "gives 100 samples for a 1e-4 s step — plenty for force/temperature\n"
            "plots without bloating the .odb."
        )
        v.addWidget(self.f_ho_dt)
        self.f_ho_dt.valueChanged.connect(self._on_change)

        return g

    # =====================================================================
    # Helpers
    # =====================================================================
    def _set_all_field_outputs(self, on: bool):
        for cb in self._fo_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)
        self._on_change()

    def _set_field_outputs_default(self):
        """Restore the dataclass defaults (everything ON, matches what the
        current abq_odb_generator.py hardcodes)."""
        from gui.core.model_config import OutputCfg
        defaults = OutputCfg()
        for attr, cb in self._fo_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(getattr(defaults, attr))
            cb.blockSignals(False)
        self._on_change()

    def _refresh_dt_label(self):
        st = self.f_sim_time.value()
        n  = max(1, self.f_n_frames.value())
        dt = st / n
        self.lbl_dt.setText(f"≈ 1 frame every {dt:.3e} s")

    # =====================================================================
    # Sync widgets ↔ cfg
    # =====================================================================
    def _on_change(self, *_):
        self._pull_from_widgets()
        self._refresh_dt_label()
        self._refresh_ms_visibility()
        self.stepChanged.emit()

    def _pull_from_widgets(self):
        s = self.cfg.step
        s.sim_time = self.f_sim_time.value()
        s.n_frames = self.f_n_frames.value()
        s.mass_scaling_enabled         = self.cb_ms_enabled.isChecked()
        s.mass_scaling_factor_eulerian = self.f_ms_eul.value()
        s.mass_scaling_factor_tool     = self.f_ms_tool.value()
        for attr, cb in self._fo_checkboxes.items():
            setattr(s.output, attr, cb.isChecked())
        s.output.ho_preselect      = self.cb_ho_preselect.isChecked()
        s.output.ho_rf_on_rp       = self.cb_ho_rf_on_rp.isChecked()
        s.output.ho_time_interval  = self.f_ho_dt.value()

    def apply_from_cfg(self):
        """Push cfg values into widgets (used after Open / New)."""
        s = self.cfg.step
        widgets = [self.f_sim_time, self.f_n_frames, self.cb_ho_preselect,
                   self.cb_ho_rf_on_rp, self.f_ho_dt,
                   self.cb_ms_enabled, self.f_ms_eul, self.f_ms_tool] \
                  + list(self._fo_checkboxes.values())
        for w in widgets:
            w.blockSignals(True)
        try:
            self.f_sim_time.set_value(s.sim_time)
            self.f_n_frames.set_value(s.n_frames)
            self.cb_ms_enabled.setChecked(s.mass_scaling_enabled)
            self.f_ms_eul.set_value(s.mass_scaling_factor_eulerian)
            self.f_ms_tool.set_value(s.mass_scaling_factor_tool)
            for attr, cb in self._fo_checkboxes.items():
                cb.setChecked(getattr(s.output, attr))
            self.cb_ho_preselect.setChecked(s.output.ho_preselect)
            self.cb_ho_rf_on_rp.setChecked(s.output.ho_rf_on_rp)
            self.f_ho_dt.set_value(s.output.ho_time_interval)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._refresh_dt_label()
        self._refresh_ms_visibility()
