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
import copy

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QScrollArea, QFrame, QCheckBox
)

from gui.core.filter_check import DEFAULT_TOLERANCE as FILTER_CHECK_TOL
from gui.core.model_config import ModelConfig, OUTPUT_FILTER_ORDER
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
    # cel_model.py joins together for the *Output card).
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
        self.cb_filter = QCheckBox("Filter output (Butterworth, runtime)")
        self.cb_filter.setChecked(self.cfg.step.output_filter_enabled)
        self.cb_filter.setToolTip(
            "Abaqus filters BEFORE writing to the ODB, at the solver\n"
            "increment. This is the only way to prevent aliasing: once\n"
            "aliased data is written, no post-processing can recover it.\n"
            "Ticking it also checks, live, both filters against the solver\n"
            "increment (base mesh and finest GCI mesh) - see below.")
        v.addWidget(self.cb_filter)
        self.cb_filter.toggled.connect(self._on_change)

        atten_tip = (
            "Attenuation of the Butterworth (order %d) at the Nyquist\n"
            "frequency of the acquisition (rate / 2). The cutoff follows:\n"
            "    fc = (rate/2) / (10^(A/10) - 1)^(1/(2N))\n"
            "3 dB puts the -3 dB point on the Nyquist frequency; a larger\n"
            "attenuation lowers fc (stronger anti-aliasing, and a higher\n"
            "lower bound on the mass-scaling factor)." % OUTPUT_FILTER_ORDER)
        self.f_cam_fps = NumField(
            "Camera acquisition rate", self.cfg.step.output_filter_camera_fps,
            "fps", minimum=1.0, maximum=1e9, decimals=1,
        )
        self.f_cam_fps.setToolTip(
            "Frame rate of the camera whose DIC velocity fields the FIELD\n"
            "output is compared with. Filters V/ERV field output.")
        self.f_cam_db = NumField(
            "Camera attenuation at Nyquist",
            self.cfg.step.output_filter_camera_atten_db, "dB",
            minimum=0.01, maximum=200.0, decimals=2,
        )
        self.f_cam_db.setToolTip(atten_tip)
        self.lbl_fc = QLabel()
        self.lbl_fc.setStyleSheet(
            "QLabel { color: #555; font-style: italic; padding-left: 4px; }")
        for w in (self.f_cam_fps, self.f_cam_db):
            v.addWidget(w)
            w.valueChanged.connect(self._on_change)
        v.addWidget(self.lbl_fc)

        self.f_force_acq = NumField(
            "Force acquisition rate",
            self.cfg.step.output_filter_force_acq_hz,
            "Hz", minimum=1.0, maximum=1e10, decimals=1,
        )
        self.f_force_acq.setToolTip(
            "Sampling rate of the force measurement (dynamometer). Filters\n"
            "the reaction forces at the tool RP (history output).")
        self.f_force_db = NumField(
            "Force attenuation at Nyquist",
            self.cfg.step.output_filter_force_atten_db, "dB",
            minimum=0.01, maximum=200.0, decimals=2,
        )
        self.f_force_db.setToolTip(atten_tip)
        self.lbl_fc_hist = QLabel()
        self.lbl_fc_hist.setStyleSheet(
            "QLabel { color: #555; font-style: italic; padding-left: 4px; }")
        for w in (self.f_force_acq, self.f_force_db):
            v.addWidget(w)
            w.valueChanged.connect(self._on_change)
        v.addWidget(self.lbl_fc_hist)

        self.cb_filter_verify = QCheckBox(
            "Verify the filters after each run (offline Butterworth)")
        self.cb_filter_verify.setChecked(self.cfg.step.output_filter_verify)
        self.cb_filter_verify.setToolTip(
            "Writes the forces through BOTH filters, next to the raw forces\n"
            "(every increment), and after the run compares each Abaqus-\n"
            "filtered series with the same Butterworth applied offline to the\n"
            "raw one. Result in the job output and in <job>.meta.json\n"
            "(\"filter_check\"). Tolerance: %.3g %% of the peak force (a\n"
            "choice, not an Abaqus figure)." % (100.0 * FILTER_CHECK_TOL))
        self.cb_filter_verify.toggled.connect(self._on_change)
        v.addWidget(self.cb_filter_verify)

        # Live check of fc*dt against Abaqus's limits, per filter and mesh.
        self.lbl_filter_check = QLabel()
        self.lbl_filter_check.setWordWrap(True)
        self.lbl_filter_check.setTextFormat(Qt.RichText)
        self.lbl_filter_check.setStyleSheet(
            "QLabel { padding: 6px; border: 1px solid #ccc; "
            "background: #fafafa; }")
        self.lbl_filter_check.setToolTip(
            "fc * dt, with dt the initial solver increment (analytical,\n"
            "times sqrt of the mass-scaling factor). Abaqus/Explicit checks it\n"
            "once, at the start of the step:\n"
            "  < 1e-3 : .sta WARNING 'The cutoff frequency used with the\n"
            "           filter ... is too low' (possible filter instability;\n"
            "           the run goes on and the filter is applied);\n"
            "  > 0.5  : NO filtering at all, silently (Analysis Guide).\n"
            "The finest GCI mesh (Optimization > Model) has the smallest\n"
            "increment, hence the binding lower bound.")
        v.addWidget(self.lbl_filter_check)

        # ---- Admissible mass-scaling window (computed, no run needed) ------
        self.lbl_ms_bounds = QLabel()
        self.lbl_ms_bounds.setWordWrap(True)
        self.lbl_ms_bounds.setStyleSheet(
            "QLabel { padding: 6px; border: 1px solid #ccc; "
            "background: #fafafa; }")
        self.lbl_ms_bounds.setToolTip(
            "Bounds derived analytically from the mesh, the materials and the\n"
            "domain size - no reference simulation needed.\n\n"
            "LOWER: Abaqus warns in the .sta when an output filter's\n"
            "cutoff/sampling ratio is below 1e-3 (possible instability);\n"
            "mass scaling raises the solver increment as sqrt(ms), which\n"
            "raises that ratio. The lowest of the two cutoffs decides.\n\n"
            "UPPER 0: above a ratio of 0.5 Abaqus does not filter at all.\n\n"
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
        on = self.cb_filter.isChecked()
        for w in (self.f_cam_fps, self.f_cam_db, self.f_force_acq,
                  self.f_force_db, self.cb_filter_verify):
            w.setEnabled(on)
        self._refresh_ms_bounds()

    def _mesh_cases(self):
        """[(label, cfg)] for the base mesh and, when finer, the finest mesh
        of the GCI study (Optimization > Model, 'finest'; empty = base)."""
        cases = [("base mesh h = %.4g mm" % float(self.cfg.elem_size),
                  self.cfg)]
        opt = getattr(self.cfg, "optimization", None)
        txt = str(getattr(opt, "gci_finest", "") or "").strip()
        try:
            h_fine = float(txt.replace(",", "."))
        except ValueError:
            h_fine = 0.0
        if 0.0 < h_fine < float(self.cfg.elem_size):
            c = copy.deepcopy(self.cfg)
            c.elem_size = h_fine
            cases.append(("finest GCI mesh h = %.4g mm" % h_fine, c))
        return cases

    def _refresh_ms_bounds(self):
        """Show the admissible mass-scaling window, computed analytically,
        and the fc*dt check of both filters."""
        self._pull_from_widgets()
        s = self.cfg.step
        self.lbl_fc.setText(f"→ camera filter cutoff fc = "
                            f"{s.output_filter_cutoff_hz:,.0f} Hz")
        self.lbl_fc_hist.setText(f"→ force filter cutoff fc = "
                                 f"{s.output_filter_cutoff_history_hz:,.0f} Hz")
        self._refresh_filter_check()
        fc = s.output_filter_cutoff_hz
        if not s.output_filter_enabled:
            self.lbl_ms_bounds.setText(
                "Output filter disabled — no lower bound on the factor.\n"
                "Without it the ODB stores ALIASED velocity fields, which no "
                "post-processing can undo.")
            return
        cur = s.mass_scaling_factor
        lines = []
        for label, c in self._mesh_cases():
            b = c.mass_scaling_bounds(
                fc, history_cutoff_hz=s.output_filter_cutoff_history_hz)
            if b["ms_min"] is None or b["ms_max"] is None:
                lines.append(f"{label}: window not computable (check E, ν, ρ, "
                             "elem_size and the domain dimensions).")
                continue
            if b["empty"]:
                lines.append(
                    f"{label}: ⚠ EMPTY window — lower bound {b['ms_min']:.0f} "
                    f"exceeds upper bound {b['ms_max']:.0f} ({b['limiting']}). "
                    "Coarsen the mesh or shrink the domain.")
                continue
            inside = b["ms_min"] <= cur <= b["ms_max"]
            mark = "✓ inside" if inside else "⚠ OUTSIDE"
            lines.append(
                f"{label}: admissible factor {b['ms_min']:.0f} … "
                f"{b['ms_max']:.0f} (current {cur:.0f} — {mark}); "
                f"upper = {b['limiting']} · dt₀ = {b['dt0']:.3e} s")
        lines.append("lower = filter validity (fc·dt ≥ 1e-3, lowest cutoff)")
        self.lbl_ms_bounds.setText("\n".join(lines))

    def _refresh_filter_check(self):
        """fc*dt of each filter vs Abaqus's limits (1e-3 warning, 0.5 no
        filtering), at the current factor, for each mesh case."""
        s = self.cfg.step
        if not s.output_filter_enabled:
            self.lbl_filter_check.setText(
                "<i>Filter check off (output filter disabled).</i>")
            return
        ms = s.mass_scaling_factor if s.mass_scaling_enabled else 1.0
        lo, hi = ModelConfig._FILTER_MIN_RATIO, ModelConfig._FILTER_MAX_RATIO
        rows = [f"<b>Filter check</b> (fc·dt₀·√ms, ms = {ms:g}; "
                f"valid range {lo:g} … {hi:g})"]
        for label, c in self._mesh_cases():
            parts = []
            for name, fc in (("camera", s.output_filter_cutoff_hz),
                             ("force", s.output_filter_cutoff_history_hz)):
                r = c.filter_ratio(fc, ms)
                if r <= 0:
                    parts.append(f"{name}: not computable")
                elif r < lo:
                    need = (lo / r) ** 2 * ms
                    parts.append(
                        f"<span style='color:#b00'>{name} {r:.3g} ⚠ &lt; "
                        f"{lo:g} — Abaqus .sta warning (ms ≥ {need:.0f} "
                        f"clears it)</span>")
                elif r > hi:
                    parts.append(
                        f"<span style='color:#b00'>{name} {r:.3g} ⚠ &gt; "
                        f"{hi:g} — NOT filtered by Abaqus</span>")
                else:
                    parts.append(f"{name} {r:.3g} ✓")
            rows.append(f"{label}: " + " · ".join(parts))
        self.lbl_filter_check.setText("<br>".join(rows))

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
        # Same for the mass-scaling window and the filter check (element
        # size, materials, domain, finest GCI mesh live in other tabs).
        self._refresh_ms_bounds()

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
        s.output_filter_camera_fps     = self.f_cam_fps.value()
        s.output_filter_camera_atten_db = self.f_cam_db.value()
        s.output_filter_force_acq_hz   = self.f_force_acq.value()
        s.output_filter_force_atten_db = self.f_force_db.value()
        s.output_filter_verify         = self.cb_filter_verify.isChecked()
        s.sync_filter_cutoffs()
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
                   self.cb_filter, self.f_cam_fps, self.f_cam_db,
                   self.f_force_acq, self.f_force_db, self.cb_filter_verify]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.f_sim_time.set_value(s.sim_time)
            self.f_n_frames.set_value(s.n_frames)
            self.cb_ms_enabled.setChecked(s.mass_scaling_enabled)
            self.f_ms_eul.set_value(s.mass_scaling_factor)
            self.cb_filter.setChecked(s.output_filter_enabled)
            self.f_cam_fps.set_value(s.output_filter_camera_fps)
            self.f_cam_db.set_value(s.output_filter_camera_atten_db)
            self.f_force_acq.set_value(s.output_filter_force_acq_hz)
            self.f_force_db.set_value(s.output_filter_force_atten_db)
            self.cb_filter_verify.setChecked(s.output_filter_verify)
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
