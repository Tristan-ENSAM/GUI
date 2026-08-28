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
            ("fo_V",      "V",      "Nodal velocity"),
            ("fo_A",      "A",      "Nodal acceleration"),
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
        # Field/history output selection and time scaling were removed: what
        # the solver computes is fixed in the generator source by an expert,
        # and extraction defaults are fixed (nodal V + NT11, element EVF,
        # history RF1/RF2 synced to the field frames).
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

        # Apply cfg → widgets once everything is built. This puts the
        # mass-scaling and history-sync fields into the correct
        # enabled/disabled state from the start (matching the cfg
        # defaults), rather than waiting for the first user edit.
        self.apply_from_cfg()

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

        # Live estimate of the explicit stable time increment and the
        # resulting number of increments (driven by the Eulerian material
        # E/ρ, the element size and any mass/time scaling).
        self.lbl_stable_dt = QLabel()
        self.lbl_stable_dt.setStyleSheet(
            "QLabel { color: #1f4060; padding-left: 4px; }"
        )
        self.lbl_stable_dt.setWordWrap(True)
        v.addWidget(self.lbl_stable_dt)

        self.f_sim_time.valueChanged.connect(self._on_change)
        self.f_n_frames.valueChanged.connect(self._on_change)
        self._refresh_dt_label()
        return g

    def _build_mass_scaling_group(self) -> QGroupBox:
        """Mass scaling controls.

        When enabled, the Eulerian (workpiece) material's density is
        multiplied by `mass_scaling_factor` AND its specific
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

        # ONE factor for BOTH bodies (the tool factor was removed: two knobs
        # with no reason to differ, and leaving the tool at 1 silently halved
        # the intended scaling of the model's inertia).
        self.f_ms_eul = NumField(
            "Factor (workpiece AND tool)",
            self.cfg.step.mass_scaling_factor, "",
            minimum=1.0, maximum=1e8, decimals=3,
        )
        self.f_ms_eul.setToolTip(
            "Mass-scaling factor, applied identically to the Eulerian\n"
            "workpiece and to the tool.\n"
            "See the admissible window computed below: it is bounded from\n"
            "BELOW by the output filter's numerical validity and from ABOVE\n"
            "by the domain reverberation dropping into the filtered band."
        )
        v.addWidget(self.f_ms_eul)
        self.f_ms_eul.valueChanged.connect(self._on_change)

        # ---- Output filter (drives the lower bound of the window) ----------
        self.cb_filter = QCheckBox("Filter field output (Butterworth, runtime)")
        self.cb_filter.setChecked(self.cfg.step.output_filter_enabled)
        self.cb_filter.setToolTip(
            "Abaqus filters BEFORE writing to the ODB, at the solver\n"
            "increment. This is the only way to prevent aliasing: once\n"
            "aliased data is written, no post-processing can recover it."
        )
        v.addWidget(self.cb_filter)
        self.cb_filter.toggled.connect(self._on_change)

        self.f_fc = NumField(
            "Field cutoff (DIC chain)", self.cfg.step.output_filter_cutoff_hz,
            "Hz", minimum=1.0, maximum=1e9, decimals=1,
        )
        self.f_fc.setToolTip(
            "Bandwidth of the DIC velocity measurement.\n"
            "Correlating two images separated by dt and dividing by dt is\n"
            "ALREADY a moving average over that interval, and it dominates the\n"
            "exposure blur. The two cascade:\n"
            "    H(f) = sinc(pi f dt_frames) x sinc(pi f t_exposure)\n"
            "e.g. 60 kfps + 5 us exposure -> 25.6 kHz (the exposure alone would\n"
            "give 88.6 kHz -- using it would filter 3.5x too high).\n"
            "This cutoff drives the LOWER bound of the mass-scaling window."
        )
        v.addWidget(self.f_fc)
        self.f_fc.valueChanged.connect(self._on_change)

        self.f_fc_hist = NumField(
            "History cutoff (anti-aliasing)",
            self.cfg.step.output_filter_cutoff_history_hz,
            "Hz", minimum=1.0, maximum=1e9, decimals=1,
        )
        self.f_fc_hist.setToolTip(
            "Applies to the reaction forces at the tool RP.\n"
            "Forces are acquired far faster than images, so this filter is NOT\n"
            "meant to match the sensor: it only prevents ALIASING at the output\n"
            "rate. Apply the real dynamometer bandwidth afterwards, in\n"
            "post-processing (history output is 1-D, so that is cheap).\n"
            "KEEP THIS HIGH: an IIR filter needs cutoff/(1/dt) > 1e-3, so a low\n"
            "cutoff here would demand an unreachable mass-scaling factor.\n"
            "A sensible value is the output rate / 6."
        )
        v.addWidget(self.f_fc_hist)
        self.f_fc_hist.valueChanged.connect(self._on_change)

        # ---- Admissible mass-scaling window (computed, no run needed) ------
        self.lbl_ms_bounds = QLabel()
        self.lbl_ms_bounds.setWordWrap(True)
        self.lbl_ms_bounds.setStyleSheet(
            "QLabel { padding: 6px; border: 1px solid #ccc; "
            "background: #fafafa; }")
        self.lbl_ms_bounds.setToolTip(
            "Bounds derived analytically from the mesh, the materials and the\n"
            "domain size - no reference simulation needed.\n\n"
            "LOWER: Abaqus rejects an output filter whose cutoff/sampling\n"
            "ratio is below 1e-3; mass scaling raises the solver increment as\n"
            "sqrt(ms), which raises that ratio.\n\n"
            "UPPER: mass scaling lowers the domain reverberation as 1/sqrt(ms).\n"
            "Scaled too far it falls INTO the filtered band and the filter can\n"
            "no longer remove it. A margin k = 3 is imposed so the 2nd-order\n"
            "Butterworth still attenuates it strongly (k = 1 would leave the\n"
            "artefact sitting at the -3 dB point).\n\n"
            "The energy-guard bound uses an INDICATIVE coefficient measured on\n"
            "one 5 um run; it depends on the mesh through ALLIE and should be\n"
            "re-measured per configuration."
        )
        v.addWidget(self.lbl_ms_bounds)

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
        if enabled:
            f_eul = self.f_ms_eul.value()
            self.lbl_ms_speedup.setText(
                f"≈ stable-dt speed-up: √{f_eul:.0f} ≈ {f_eul ** 0.5:.2f}×"
            )
        else:
            self.lbl_ms_speedup.setText("(disabled — materials unchanged)")
        self.f_fc.setEnabled(self.cb_filter.isChecked())
        self.f_fc_hist.setEnabled(self.cb_filter.isChecked())
        self._refresh_ms_bounds()

    def _refresh_ms_bounds(self):
        """Show the admissible mass-scaling window, computed analytically."""
        self._pull_from_widgets()
        fc = self.cfg.step.output_filter_cutoff_hz
        if not self.cfg.step.output_filter_enabled:
            self.lbl_ms_bounds.setText(
                "Output filter disabled — no lower bound on the factor.\n"
                "Without it the ODB stores ALIASED velocity fields, which no "
                "post-processing can undo.")
            return
        b = self.cfg.mass_scaling_bounds(fc)
        if b["ms_min"] is None or b["ms_max"] is None:
            self.lbl_ms_bounds.setText(
                "Admissible window: not computable (check E, ν, ρ, elem_size "
                "and the domain dimensions).")
            return
        cur = self.cfg.step.mass_scaling_factor
        if b["empty"]:
            self.lbl_ms_bounds.setText(
                f"⚠ EMPTY window: lower bound {b['ms_min']:.0f} exceeds upper "
                f"bound {b['ms_max']:.0f} ({b['limiting']}).\n"
                "The mesh is too fine relative to the domain for any factor to "
                "satisfy both. Coarsen the mesh, shrink the domain, or raise "
                "the filter cutoff.")
            return
        inside = b["ms_min"] <= cur <= b["ms_max"]
        mark = "✓ inside" if inside else "⚠ OUTSIDE"
        self.lbl_ms_bounds.setText(
            f"Admissible factor: {b['ms_min']:.0f} … {b['ms_max']:.0f}   "
            f"(current {cur:.0f} — {mark})\n"
            f"lower = filter validity · upper = {b['limiting']} · "
            f"dt₀ = {b['dt0']:.3e} s"
        )

    def _refresh_dt_label(self):
        st = self.f_sim_time.value()
        n  = max(1, self.f_n_frames.value())
        dt = st / n
        self.lbl_dt.setText(f"≈ 1 frame every {dt:.3e} s")

    def _stable_increment_estimate(self):
        """Rough explicit stable time increment Δt ≈ Lₑ / c_d, with the
        dilatational wave speed c_d ≈ √(E/ρ) of the Eulerian (workpiece)
        material, in the Abaqus t-mm-s system (E in MPa = N/mm², ρ in
        t/mm³ → c_d in mm/s). Mass scaling lowers c_d by √κ_m (ρ_eff =
        κ_m·ρ). Returns (Δt_seconds, n_increments) or (None, None).

        This is an *estimate*: the real solver increment is recomputed on
        the smallest deformed element with stability/​bulk-viscosity
        corrections, so treat it as an order of magnitude."""
        m = getattr(self.cfg, "euler_material", {}) or {}
        try:
            E = float(m.get("E", 0.0))        # MPa internal
            rho = float(m.get("rho", 0.0))    # t/mm³ internal
            Le = float(getattr(self.cfg, "elem_size", 0.0))  # mm
        except (TypeError, ValueError):
            return None, None
        if E <= 0.0 or rho <= 0.0 or Le <= 0.0:
            return None, None
        rho_eff = rho
        ms_on = getattr(self, "cb_ms_enabled", None)
        if ms_on is not None and ms_on.isChecked():
            rho_eff *= max(1.0, self.f_ms_eul.value())
        c_d = (E / rho_eff) ** 0.5            # mm/s
        if c_d <= 0.0:
            return None, None
        dt = Le / c_d                          # s
        sim = self.f_sim_time.value()
        n = (sim / dt) if dt > 0 else None
        return dt, n

    def _refresh_stable_dt_label(self):
        dt, n = self._stable_increment_estimate()
        if dt is None:
            self.lbl_stable_dt.setText(
                "Stable increment: need E, ρ (Materials) and element size "
                "(Mesh) to estimate.")
            return
        txt = (f"≈ stable increment ~{dt:.3e} s  ·  "
               f"~{n:,.0f} increments over the step (estimate)")
        ms_on = getattr(self, "cb_ms_enabled", None)
        if ms_on is not None and ms_on.isChecked():
            txt += f"  ·  mass scaling ×{self.f_ms_eul.value():.0f} applied"
        self.lbl_stable_dt.setText(txt)

    def showEvent(self, event):
        # The Eulerian material (E/ρ) and element size are edited in other
        # tabs; refresh the estimate every time the Step tab is shown.
        super().showEvent(event)
        self._refresh_stable_dt_label()

    # =====================================================================
    # Sync widgets ↔ cfg
    # =====================================================================
    def _on_change(self, *_):
        self._pull_from_widgets()
        self._refresh_dt_label()
        self._refresh_stable_dt_label()
        self._refresh_ms_visibility()
        self.stepChanged.emit()

    def _pull_from_widgets(self):
        s = self.cfg.step
        s.sim_time = self.f_sim_time.value()
        s.n_frames = self.f_n_frames.value()
        s.mass_scaling_enabled         = self.cb_ms_enabled.isChecked()
        s.mass_scaling_factor          = self.f_ms_eul.value()
        s.output_filter_enabled        = self.cb_filter.isChecked()
        s.output_filter_cutoff_hz      = self.f_fc.value()
        s.output_filter_cutoff_history_hz = self.f_fc_hist.value()
        # History sampling is always synced to the field-output frame count;
        # RF1/RF2 and PRESELECT are always written (fixed extraction).
        s.output.ho_n_intervals = s.n_frames
        s.output.ho_preselect   = True
        s.output.ho_rf_on_rp    = True

    def apply_from_cfg(self):
        """Push cfg values into widgets (used after Open / New)."""
        s = self.cfg.step
        widgets = [self.f_sim_time, self.f_n_frames,
                   self.cb_ms_enabled, self.f_ms_eul,
                   self.cb_filter, self.f_fc, self.f_fc_hist]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.f_sim_time.set_value(s.sim_time)
            self.f_n_frames.set_value(s.n_frames)
            self.cb_ms_enabled.setChecked(s.mass_scaling_enabled)
            self.f_ms_eul.set_value(s.mass_scaling_factor)
            self.cb_filter.setChecked(s.output_filter_enabled)
            self.f_fc.set_value(s.output_filter_cutoff_hz)
            self.f_fc_hist.set_value(s.output_filter_cutoff_history_hz)
        finally:
            for w in widgets:
                w.blockSignals(False)
        # Keep the fixed-output invariants in the config.
        s.output.ho_n_intervals = s.n_frames
        s.output.ho_preselect   = True
        s.output.ho_rf_on_rp    = True
        self._refresh_dt_label()
        self._refresh_stable_dt_label()
        self._refresh_ms_visibility()
