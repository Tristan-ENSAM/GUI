# -*- coding: utf-8 -*-
"""
Materials tab: edit the tool and workpiece material properties.

All values are entered and displayed in human-friendly SI / engineering
units (kg/m³, GPa, MPa, W/(m·K), J/(kg·K), 1/K, °C or K). The conversion
to the Abaqus internal consistent unit system (t-mm-s-MPa-°C) is performed
in this tab via `gui.core.units`, so:
  - `cfg.tool_material` / `cfg.euler_material` always hold Abaqus-internal
    values, ready to be passed verbatim to abq_odb_generator.py;
  - the user never sees the awkward 8.96e-9 t/mm³ — they see 8960 kg/m³.

Temperature display unit (°C or K) follows `cfg.ui.temp_unit`. The internal
storage is always °C (matching the Abaqus side).
"""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QScrollArea,
    QFrame, QComboBox, QPushButton, QInputDialog, QMessageBox,
)

from gui.core.model_config import ModelConfig
from gui.core import units
from gui.core.presets import PresetLibrary
from gui.widgets.param_field import NumField, PairRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _section_header(title: str) -> QLabel:
    """A small colored label that visually marks the start of a sub-section
    inside a larger group box, mimicking Abaqus 'Behaviour' headers."""
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "background-color: #e8eef5; color: #1f4060; "
        "font-weight: bold; padding: 3px 6px; "
        "border-left: 3px solid #1f6fb2;"
    )
    return lbl


def _make_field(key: str, label_override: str | None,
                gui_value: float, temp_unit: str,
                compact: bool = False, **extra) -> NumField:
    """Create a NumField pre-configured for a given material key.

    The label combines the display name and the SI unit (e.g. "ρ [kg/m³]").
    `extra` is forwarded to NumField (min, max, decimals, ...).
    """
    unit_str = units.display_unit(key, temp_unit)
    label = label_override if label_override is not None else units.display_label(key)
    if unit_str and unit_str != "—":
        full_label = f"{label} [{unit_str}]"
    else:
        full_label = label
    return NumField(full_label, gui_value, "", compact=compact, **extra)


# ---------------------------------------------------------------------------
# MaterialsTab
# ---------------------------------------------------------------------------
class MaterialsTab(QWidget):
    """Two side-by-side material editors: Tool and Workpiece.

    Emits `materialsChanged()` whenever any field is edited. The cfg dicts
    always store Abaqus-internal values; this widget converts on the fly."""

    materialsChanged = Signal()

    # Field tables: list of (cfg_dict_key, optional_label_override).
    # Order matters — it dictates the on-screen layout.
    TOOL_FIELDS = [
        # General
        ("rho", None),
        # Elastic
        ("E", None), ("nu", None),
        # Thermal
        ("k", None), ("Cp", None),
        # Expansion
        ("alpha", None),
    ]
    WP_FIELDS = TOOL_FIELDS + [
        # Inelastic heat fraction
        ("beta", None),
        # Johnson-Cook plasticity
        ("A", None), ("B", None), ("n", None), ("m", None),
        ("Tm", None), ("Tr", None),
        # Rate dependent
        ("C", None), ("eps_dot0", None),
        # Damage initiation
        ("D1", None), ("D2", None), ("D3", None), ("D4", None), ("D5", None),
        ("eps0", None),
        # Damage evolution
        ("Gf", None),
    ]

    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.lib = PresetLibrary()

        # Maps cfg-dict-key -> NumField widget, one per side.
        # Built lazily in the column builders.
        self._tool_widgets: dict[str, NumField] = {}
        self._wp_widgets:   dict[str, NumField] = {}

        # Preset combo boxes (used by _refresh_combo and the apply/save/copy
        # button handlers). Stored on self so we can refresh them after
        # the user saves a new preset.
        self._tool_combo: QComboBox | None = None
        self._wp_combo:   QComboBox | None = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        cols = QHBoxLayout(inner)
        cols.setContentsMargins(8, 8, 8, 8)
        cols.setSpacing(12)

        cols.addWidget(self._build_tool_column(),      stretch=1)
        cols.addWidget(self._build_workpiece_column(), stretch=1)

    # =====================================================================
    # Preset bar (one per column)
    # =====================================================================
    def _build_preset_bar(self, kind: str) -> QWidget:
        """Build a row with: [Library combo] [Apply] [Save as...] [Copy from other].

        The Apply button is preferred over auto-applying on combo change,
        so the user can scan presets without losing the current values."""
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 4)
        h.setSpacing(6)

        h.addWidget(QLabel("Library:"))
        combo = QComboBox()
        combo.setMinimumWidth(180)
        self._refresh_combo(combo, kind)
        h.addWidget(combo, stretch=1)

        btn_apply = QPushButton("Apply")
        btn_apply.setToolTip("Replace the current material with the selected preset.")
        btn_apply.clicked.connect(lambda: self._apply_preset(kind, combo))
        h.addWidget(btn_apply)

        btn_save = QPushButton("Save as…")
        btn_save.setToolTip("Save the current material as a new user preset.")
        btn_save.clicked.connect(lambda: self._save_as_preset(kind))
        h.addWidget(btn_save)

        # Cross-copy button: copy the OTHER side's current material into this
        # side. Useful when you want to use the same alloy on both — saves
        # re-applying the same preset twice. Doesn't touch the library.
        other_kind = "workpiece" if kind == "tool" else "tool"
        btn_copy = QPushButton(f"← Copy {other_kind}")
        btn_copy.setToolTip(
            f"Copy the current {other_kind} material into the {kind} side."
        )
        btn_copy.clicked.connect(lambda: self._copy_from_other(kind))
        h.addWidget(btn_copy)

        if kind == "tool":
            self._tool_combo = combo
        else:
            self._wp_combo = combo
        return bar

    def _refresh_combo(self, combo: QComboBox, kind: str):
        """Populate a combo with the current preset names for `kind`.
        User-defined presets get a small marker so they're distinguishable
        from bundled ones at a glance."""
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("— select a preset —", None)
        for name in self.lib.list_presets(kind):
            label = name
            if self.lib.is_user_defined(kind, name):
                label = f"{name}  (user)"
            combo.addItem(label, name)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    # ----- handlers -----
    def _apply_preset(self, kind: str, combo: QComboBox):
        name = combo.currentData()
        if not name:
            QMessageBox.information(
                self, "Apply preset", "Select a preset from the list first."
            )
            return
        material = self.lib.get(kind, name)
        if material is None:
            QMessageBox.warning(self, "Apply preset",
                                f"Preset {name!r} not found.")
            return
        target = (self.cfg.tool_material if kind == "tool"
                  else self.cfg.euler_material)
        target.clear()
        target.update(material)
        self.apply_from_cfg()
        self.materialsChanged.emit()

    def _save_as_preset(self, kind: str):
        # Make sure cfg reflects the latest typed-in values
        self._pull_from_widgets()
        name, ok = QInputDialog.getText(
            self, "Save as preset",
            f"Name for the new {kind} preset:",
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        if self.lib.is_user_defined(kind, name):
            reply = QMessageBox.question(
                self, "Overwrite",
                f"A user preset named {name!r} already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        material = (self.cfg.tool_material if kind == "tool"
                    else self.cfg.euler_material)
        try:
            self.lib.save_user_preset(kind, name, material)
        except Exception as e:
            QMessageBox.critical(
                self, "Save preset",
                f"Failed to save preset:\n{type(e).__name__}: {e}",
            )
            return

        combo = self._tool_combo if kind == "tool" else self._wp_combo
        if combo is not None:
            self._refresh_combo(combo, kind)
            idx = combo.findData(name)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _copy_from_other(self, kind: str):
        """Copy the OTHER side's current material into `kind`."""
        if kind == "tool":
            source = self.cfg.euler_material
            target = self.cfg.tool_material
        else:
            source = self.cfg.tool_material
            target = self.cfg.euler_material

        # Only the elastic-thermal block is shared between tool and workpiece;
        # JC params on the workpiece are left untouched when copying tool ->
        # workpiece, and there's no JC on the tool side to copy in the other
        # direction.
        common_keys = ("rho", "E", "nu", "k", "Cp", "alpha")
        for k in common_keys:
            if k in source:
                target[k] = source[k]
        self.apply_from_cfg()
        self.materialsChanged.emit()

    # =====================================================================
    # Column builders
    # =====================================================================
    def _temp_unit(self) -> str:
        return getattr(self.cfg.ui, "temp_unit", "C")

    def _new_field(self, key: str, store: dict[str, NumField],
                   abq_value: float, **extra) -> NumField:
        gui_value = units.abaqus_to_gui(key, abq_value, self._temp_unit())
        f = _make_field(key, None, gui_value, self._temp_unit(), **extra)
        store[key] = f
        f.valueChanged.connect(self._on_change)
        return f

    def _build_tool_column(self) -> QGroupBox:
        g = QGroupBox("Tool material")
        lay = QVBoxLayout(g)

        # Preset bar at the top
        lay.addWidget(self._build_preset_bar("tool"))

        m = self.cfg.tool_material
        S = self._tool_widgets

        # --- General ---
        lay.addWidget(_section_header("General — Density"))
        lay.addWidget(self._new_field("rho", S, m["rho"], minimum=1e-15))

        # --- Mechanical / Elastic ---
        lay.addWidget(_section_header("Mechanical — Elastic"))
        lay.addWidget(PairRow(
            self._new_field("E",  S, m["E"],  minimum=0.0, compact=True),
            self._new_field("nu", S, m["nu"], minimum=0.0, maximum=0.5,
                            compact=True),
        ))

        # --- Thermal ---
        lay.addWidget(_section_header("Thermal — Conductivity, Specific heat"))
        lay.addWidget(PairRow(
            self._new_field("k",  S, m["k"],  compact=True),
            self._new_field("Cp", S, m["Cp"], compact=True),
        ))

        # --- Mechanical / Expansion ---
        lay.addWidget(_section_header("Mechanical — Thermal expansion"))
        lay.addWidget(self._new_field("alpha", S, m["alpha"], decimals=10))

        lay.addStretch()
        return g

    def _build_workpiece_column(self) -> QGroupBox:
        g = QGroupBox("Workpiece material")
        lay = QVBoxLayout(g)

        # Preset bar at the top
        lay.addWidget(self._build_preset_bar("workpiece"))

        m = self.cfg.euler_material   # used in both CEL and Lagrangian
        S = self._wp_widgets

        # --- General ---
        lay.addWidget(_section_header("General — Density"))
        lay.addWidget(self._new_field("rho", S, m["rho"], minimum=1e-15))

        # --- Mechanical / Elastic ---
        lay.addWidget(_section_header("Mechanical — Elastic"))
        lay.addWidget(PairRow(
            self._new_field("E",  S, m["E"],  minimum=0.0, compact=True),
            self._new_field("nu", S, m["nu"], minimum=0.0, maximum=0.5,
                            compact=True),
        ))

        # --- Thermal ---
        lay.addWidget(_section_header("Thermal — Conductivity, Specific heat"))
        lay.addWidget(PairRow(
            self._new_field("k",  S, m["k"],  compact=True),
            self._new_field("Cp", S, m["Cp"], compact=True),
        ))

        # --- Mechanical / Expansion ---
        lay.addWidget(_section_header("Mechanical — Thermal expansion"))
        lay.addWidget(self._new_field("alpha", S, m["alpha"], decimals=10))

        # --- Inelastic heat fraction ---
        lay.addWidget(_section_header("Mechanical — Inelastic heat fraction"))
        f_beta = self._new_field("beta", S, m["beta"], minimum=0.0, maximum=1.0)
        f_beta.setToolTip(
            "Fraction of plastic work converted to heat (Taylor-Quinney).\n"
            "Typical: 0.9 for metals."
        )
        lay.addWidget(f_beta)

        # --- Plasticity / Johnson-Cook ---
        lay.addWidget(_section_header("Plastic — Johnson-Cook"))
        lay.addWidget(PairRow(
            self._new_field("A", S, m["A"], minimum=0.0, compact=True),
            self._new_field("B", S, m["B"], minimum=0.0, compact=True),
        ))
        lay.addWidget(PairRow(
            self._new_field("n", S, m["n"], minimum=0.0, compact=True),
            self._new_field("m", S, m["m"], minimum=0.0, compact=True),
        ))
        lay.addWidget(PairRow(
            self._new_field("Tm", S, m["Tm"], compact=True),
            self._new_field("Tr", S, m["Tr"], compact=True),
        ))

        # --- Rate dependence ---
        lay.addWidget(_section_header("Plastic — Rate dependent (Johnson-Cook)"))
        lay.addWidget(PairRow(
            self._new_field("C",        S, m["C"],        compact=True),
            self._new_field("eps_dot0", S, m["eps_dot0"], compact=True),
        ))

        # --- Damage initiation ---
        lay.addWidget(_section_header("Damage — Johnson-Cook initiation"))
        lay.addWidget(PairRow(
            self._new_field("D1", S, m["D1"], compact=True),
            self._new_field("D2", S, m["D2"], compact=True),
        ))
        lay.addWidget(PairRow(
            self._new_field("D3", S, m["D3"], compact=True),
            self._new_field("D4", S, m["D4"], compact=True),
        ))
        lay.addWidget(PairRow(
            self._new_field("D5",   S, m["D5"],   compact=True),
            self._new_field("eps0", S, m["eps0"], compact=True),
        ))

        # --- Damage evolution ---
        lay.addWidget(_section_header("Damage — Evolution (ENERGY, EXPONENTIAL)"))
        f_gf = self._new_field("Gf", S, m["Gf"], minimum=0.0)
        f_gf.setToolTip(
            "Fracture energy per unit area for ductile failure.\n"
            "Equivalent to mJ/mm² in the Abaqus internal system."
        )
        lay.addWidget(f_gf)

        lay.addStretch()
        return g

    # =====================================================================
    # cfg ↔ widgets
    # =====================================================================
    def _on_change(self, *_):
        self._pull_from_widgets()
        self.materialsChanged.emit()

    def _pull_from_widgets(self):
        """Read every field, convert from GUI display unit to Abaqus internal,
        and write back to the cfg dicts."""
        tu = self._temp_unit()
        for key, w in self._tool_widgets.items():
            self.cfg.tool_material[key] = units.gui_to_abaqus(key, w.value(), tu)
        for key, w in self._wp_widgets.items():
            self.cfg.euler_material[key] = units.gui_to_abaqus(key, w.value(), tu)

    def apply_from_cfg(self):
        """Push the Abaqus-internal values from cfg back into the widgets,
        converting to the GUI display unit on the way."""
        tu = self._temp_unit()
        for key, w in self._tool_widgets.items():
            abq = self.cfg.tool_material.get(key, 0.0)
            w.set_value(units.abaqus_to_gui(key, abq, tu))
        for key, w in self._wp_widgets.items():
            abq = self.cfg.euler_material.get(key, 0.0)
            w.set_value(units.abaqus_to_gui(key, abq, tu))

    def refresh_temp_unit(self):
        """Called when the user toggles °C ↔ K. We rebuild the labels of
        Tm/Tr/... and recompute their displayed values from the (unchanged)
        Abaqus-internal storage."""
        tu = self._temp_unit()
        for store, mat in (
            (self._tool_widgets, self.cfg.tool_material),
            (self._wp_widgets,   self.cfg.euler_material),
        ):
            for key, w in store.items():
                spec = units.MATERIAL_FIELDS.get(key)
                if spec is None:
                    continue
                _, unit_marker, _ = spec
                if unit_marker != "TEMP":
                    continue
                # Update the displayed value (Kelvin <-> Celsius)
                abq = mat.get(key, 0.0)
                w.set_value(units.abaqus_to_gui(key, abq, tu))
                # Update the label
                full_label = f"{units.display_label(key)} [{units.display_unit(key, tu)}]"
                w._lbl.setText(full_label)
