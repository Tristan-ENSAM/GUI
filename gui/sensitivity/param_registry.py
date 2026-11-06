# -*- coding: utf-8 -*-
"""
Parameter registry for the sensitivity study (Morris).

This module answers one question: *which scalar parameters of a
`ModelConfig` can be varied in a sensitivity study, and how do we read /
write them?*

Design
------
A `ParamSpec` describes one variable parameter:

  - `path`         : dotted path into a ModelConfig, e.g. "euler_material.A"
                     or "bcs.cutting_speed". `get_stored` / `set_stored`
                     walk this path (handling both dataclass attributes and
                     the raw material dicts at the leaf).
  - `label`        : human label for the UI.
  - `category`     : group header for the UI tree.
  - `display_unit` : unit shown to the user (e.g. "GPa", "mm", "m/min").
  - `dtype`        : "float" | "int".
  - the conversion between the **stored** value (the Abaqus-internal value
    that `ModelConfig.to_params_dict()` serialises) and the **displayed**
    value (engineering units the user reads in the Materials / BCs tabs).

Why two unit systems?
---------------------
`gui.core.units` already documents that materials are STORED in the Abaqus
consistent system (E=124000 MPa, rho=8.96e-9 t/mm³) but DISPLAYED in
engineering units (E=124 GPa, rho=8960 kg/m³). A sensitivity study must
let the user set min/max in the units they actually read, then write the
perturbed value back into the cfg in the stored units the solver expects.

To avoid re-inventing (and contradicting) those factors, the material
ParamSpecs are generated programmatically from `units.MATERIAL_FIELDS`,
the single source of truth used by the Materials tab. Non-material
parameters (geometry, kinematics, mesh) are hand-authored below; most are
identity conversions (the cfg already stores them in the unit the user
reads, e.g. mm, deg), with two documented exceptions:
  - cutting speed       : stored mm/s, displayed in the unit system's
                          velocity unit (m/min by default, unit_kind)
  - ambient temperature : stored °C, displayed °C or K (cfg.ui.temp_unit)

The registry does NOT decide bounds — the UI (Lot 2b) lets the user enter
min/max. The registry only provides a *default* symmetric range around the
current value (`default_display_bounds`) to pre-fill those fields.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from gui.core import units
from gui.core import unit_system as _us


def current_system(temp_unit: str = "C"):
    """Snapshot of the active display UnitSystem with its temperature base
    set to `temp_unit`. A copy: later Settings changes do not alter it."""
    return dataclasses.replace(units.active_system(), temp=temp_unit)


# ---------------------------------------------------------------------------
# Path walking — read/write a dotted path on a ModelConfig
# ---------------------------------------------------------------------------
def _step(obj, key: str):
    """Descend one level: dict key if `obj` is a dict, else attribute."""
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def get_stored(cfg, path: str):
    """Read the STORED (Abaqus-internal) value at `path` on a ModelConfig.

    Handles both dataclass attributes ("bcs.cutting_speed") and the raw
    material dicts at the leaf ("euler_material.A").
    """
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = _step(obj, p)
    return _step(obj, parts[-1])


def set_stored(cfg, path: str, value) -> None:
    """Write the STORED value at `path` on a ModelConfig (in place)."""
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = _step(obj, p)
    leaf = parts[-1]
    if isinstance(obj, dict):
        obj[leaf] = value
    else:
        setattr(obj, leaf, value)


# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    """One variable parameter of the model.

    Conversion model (when `is_temp` is False):
        displayed = stored / factor
        stored    = displayed * factor
    so `factor` is "displayed-units to stored-units". For material fields
    this matches `units.gui_to_abaqus` (gui_value * factor = abaqus_value).

    When `is_temp` is True the conversion goes through
    `units.temp_to_abaqus` / `temp_from_abaqus` (stored is always °C).
    """
    path: str
    label: str
    category: str
    display_unit: str = "—"
    dtype: str = "float"          # "float" | "int"
    factor: float = 1.0           # displayed * factor = stored
    is_temp: bool = False         # stored °C; displayed per temp_unit
    mat_key: str = ""             # material key -> convert via the ACTIVE
                                  # unit system (units.*), not the static
                                  # factor, so it follows Settings.
    unit_kind: str = ""           # non-material quantity kind of the unit
                                  # system (e.g. "velocity_named"): converts
                                  # and labels through it like mat_key.
    # Default ± half-range used to pre-fill the UI min/max, expressed on the
    # DISPLAYED value. If the displayed default is ~0 (e.g. rake_angle=0),
    # `abs_range` is used as an absolute half-width instead of `rel_range`.
    rel_range: float = 0.20
    abs_range: float = 0.0
    # What changing the value does, shown as a warning in the tables (never
    # a block): "physical" changes the modelled physics, "numerical" a
    # setting of the model (mesh, domain, mass scaling, duration...),
    # "output" only what is written or extracted (ROI box, frames, filters).
    kind: str = "physical"
    # True when a new value rebuilds the geometry or the mesh.
    remesh: bool = False

    # -- conversions --
    # `system` pins the conversion to one UnitSystem instead of the app-wide
    # active one. A plan passes the system it was generated under, so the
    # values it holds keep their meaning even if Settings change afterwards.
    def to_display(self, stored: float, temp_unit: str = "C",
                   system=None) -> float:
        system = system if system is not None else current_system(temp_unit)
        if self.is_temp:
            return system.from_internal("temperature", stored)
        if self.mat_key:
            return system.from_internal(_us.field_kind(self.mat_key), stored)
        if self.unit_kind:
            return system.from_internal(self.unit_kind, stored)
        return stored / self.factor if self.factor else stored

    def to_stored(self, displayed: float, temp_unit: str = "C",
                  system=None) -> float:
        system = system if system is not None else current_system(temp_unit)
        if self.is_temp:
            return system.to_internal("temperature", displayed)
        if self.mat_key:
            val = system.to_internal(_us.field_kind(self.mat_key), displayed)
            return int(round(val)) if self.dtype == "int" else val
        if self.unit_kind:
            val = system.to_internal(self.unit_kind, displayed)
            return int(round(val)) if self.dtype == "int" else val
        val = displayed * self.factor
        return int(round(val)) if self.dtype == "int" else val

    def unit_str(self, temp_unit: str = "C", system=None) -> str:
        """Live display unit: material keys resolve through the unit system,
        temperatures follow its temperature base; other params keep their
        authored `display_unit`."""
        system = system if system is not None else current_system(temp_unit)
        if self.mat_key:
            return system.unit_label(_us.field_kind(self.mat_key))
        if self.is_temp:
            return system.unit_label("temperature")
        if self.unit_kind:
            return system.unit_label(self.unit_kind)
        return self.display_unit


# ---------------------------------------------------------------------------
# Material specs — generated from units.MATERIAL_FIELDS (single source of
# truth). We only expose the keys that actually live in `euler_material`.
# ---------------------------------------------------------------------------
# Grouping of material keys into UI categories. Keys not listed here are
# still exposed under "Matériau pièce — autres".
_MAT_GROUPS = [
    ("Matériau — JC plasticité",   ["A", "B", "n", "C", "m"]),
    ("Matériau — JC thermique",    ["Tm", "Tr", "eps_dot0"]),
    ("Matériau — propriétés",      ["rho", "E", "nu", "k", "Cp", "alpha", "beta"]),
    ("Matériau — JC endommagement", ["D1", "D2", "D3", "D4", "D5", "eps0", "Gf"]),
]

# Sensible default ± ranges (fraction of the displayed value) per material
# key. These are only UI pre-fills; the user overrides them. Conservative
# 15–25 % for most, with absolute fallbacks for angles/zero-ish defaults.
_MAT_REL_RANGE = {
    "A": 0.20, "B": 0.20, "n": 0.20, "C": 0.30, "m": 0.20,
    "Tm": 0.05, "Tr": 0.10, "eps_dot0": 0.0,
    "rho": 0.05, "E": 0.15, "nu": 0.10, "k": 0.20, "Cp": 0.15,
    "alpha": 0.20, "beta": 0.10,
    "D1": 0.30, "D2": 0.30, "D3": 0.30, "D4": 0.30, "D5": 0.30,
    "eps0": 0.0, "Gf": 0.30,
}


def _material_specs(prefix: str = "euler_material",
                    only=None, category: str = "") -> list[ParamSpec]:
    """Specs of the material dict at `prefix` (the workpiece by default);
    `only` limits them to some keys, `category` puts them all in one
    group."""
    specs: list[ParamSpec] = []
    grouped_keys: dict[str, str] = {}
    for cat, keys in _MAT_GROUPS:
        for k in keys:
            grouped_keys[k] = cat

    # Iterate in the order units.MATERIAL_FIELDS declares them so the UI
    # ordering is stable and matches the Materials tab.
    for key, (label, unit, factor) in units.MATERIAL_FIELDS.items():
        if only is not None and key not in only:
            continue
        cat = category or grouped_keys.get(key, "Matériau — autres")
        is_temp = (unit == "TEMP")
        # For TEMP keys the displayed unit resolves later via temp_unit;
        # store a neutral placeholder, the UI calls display_unit() anyway.
        disp_unit = "°C" if is_temp else unit
        # factor is None for TEMP fields in MATERIAL_FIELDS
        f = 1.0 if (factor is None) else float(factor)
        rel = _MAT_REL_RANGE.get(key, 0.20)
        specs.append(ParamSpec(
            path=f"{prefix}.{key}",
            label=f"{label}",
            category=cat,
            display_unit=disp_unit,
            dtype="float",
            factor=f,
            is_temp=is_temp,
            mat_key=key,
            rel_range=rel,
            # absolute fallback when displayed default is ~0
            abs_range=0.0,
        ))
    return specs


# ---------------------------------------------------------------------------
# Non-material specs — hand-authored. Conversions are identity unless noted.
# ---------------------------------------------------------------------------
def _non_material_specs() -> list[ParamSpec]:
    return [
        # --- Tool geometry (stored in mm / deg, displayed as-is) ---
        ParamSpec("tool_geometry.rake_angle",  "Angle de coupe (rake)",  "Géométrie outil", "deg", rel_range=0.0, abs_range=10.0, remesh=True),
        ParamSpec("tool_geometry.clear_angle", "Angle de dépouille",     "Géométrie outil", "deg", rel_range=0.40, abs_range=3.0, remesh=True),
        ParamSpec("tool_geometry.r_tool",      "Rayon d'arête r",        "Géométrie outil", "mm",  rel_range=0.50, remesh=True),
        ParamSpec("tool_geometry.h_tool",      "Hauteur outil h",        "Géométrie outil", "mm",  rel_range=0.20, remesh=True),
        ParamSpec("tool_geometry.l_tool",      "Longueur outil l",       "Géométrie outil", "mm",  rel_range=0.20, remesh=True),
        ParamSpec("tool_position.x0",          "Position outil x0",      "Géométrie outil", "mm",  rel_range=0.20, abs_range=0.01, remesh=True),
        ParamSpec("tool_position.y0",          "Position outil y0",      "Géométrie outil", "mm",  rel_range=0.20, abs_range=0.01, remesh=True),

        # --- Workpiece / Eulerian geometry (the domain D of Model sizing) ---
        ParamSpec("euler_geometry.h_wp",       "Hauteur pièce h_wp",     "Géométrie pièce", "mm",  rel_range=0.20, kind="numerical", remesh=True),
        ParamSpec("euler_geometry.l_wp",       "Longueur pièce l_wp",    "Géométrie pièce", "mm",  rel_range=0.20, kind="numerical", remesh=True),
        ParamSpec("euler_geometry.h_void",     "Hauteur du vide h_void", "Géométrie pièce", "mm",  rel_range=0.20, kind="numerical", remesh=True),
        ParamSpec("euler_geometry.l_void",     "Longueur du vide l_void", "Géométrie pièce", "mm", rel_range=0.20, kind="numerical", remesh=True),
        ParamSpec("wp_position.x0",            "Position pièce x0",      "Géométrie pièce", "mm",  rel_range=0.20, abs_range=0.01, remesh=True),
        ParamSpec("wp_position.y0",            "Position pièce y0",      "Géométrie pièce", "mm",  rel_range=0.20, abs_range=0.01, remesh=True),
        ParamSpec("euler_position.x0",         "Position domaine eulérien x0", "Géométrie pièce", "mm", rel_range=0.20, abs_range=0.01, kind="numerical", remesh=True),
        ParamSpec("euler_position.y0",         "Position domaine eulérien y0", "Géométrie pièce", "mm", rel_range=0.20, abs_range=0.01, kind="numerical", remesh=True),

        # --- Process / boundary conditions ---
        # cutting_speed: stored mm/s, displayed in the system's velocity unit.
        ParamSpec("bcs.cutting_speed",         "Vitesse de coupe",       "Procédé / CL", "m/min", unit_kind="velocity_named", rel_range=0.30),
        ParamSpec("bcs.initial_velocity",      "Vitesse initiale",       "Procédé / CL", "m/min", unit_kind="velocity_named", rel_range=0.30),
        # ambient_temperature: stored °C, displayed per temp_unit.
        ParamSpec("bcs.ambient_temperature",   "Température ambiante",   "Procédé / CL", "°C", is_temp=True, rel_range=0.0, abs_range=50.0),

        # --- Contact ---
        ParamSpec("interaction.friction_coeff", "Coefficient de frottement µ", "Contact", "—", rel_range=0.50),
        ParamSpec("interaction.slip_tolerance", "Tolérance de glissement", "Contact", "—", rel_range=0.50, kind="numerical"),
        ParamSpec("interaction.heat_conversion_fraction", "Fraction convertie en chaleur η", "Contact", "—", rel_range=0.20),
        # The master share follows as 1 - slave (as in the Interaction tab).
        ParamSpec("interaction.heat_fraction_to_slave", "Part de chaleur vers la pièce", "Contact", "—", rel_range=0.20),
        # None (= not written, Abaqus default) until set in the Interaction tab.
        ParamSpec("interaction.gap_conductance", "Conductance thermique de contact", "Contact", "mW/(mm²·K)", rel_range=0.50, abs_range=100.0),

        # --- Numerical ---
        ParamSpec("elem_size",                 "Taille d'élément",       "Numérique", "mm", rel_range=0.40, kind="numerical", remesh=True),
        ParamSpec("tool_elem_size",            "Maille outil, pointe",   "Numérique", "mm", rel_range=0.40, kind="numerical", remesh=True),
        ParamSpec("inter_elem_size",           "Maille outil, bout des faces", "Numérique", "mm", rel_range=0.40, kind="numerical", remesh=True),
        ParamSpec("max_elem_size",             "Maille outil, bords",    "Numérique", "mm", rel_range=0.40, kind="numerical", remesh=True),
        ParamSpec("step.mass_scaling_factor", "Mass scaling (pièce + outil)", "Numérique", "—", rel_range=0.0, abs_range=0.0, kind="numerical"),
        ParamSpec("step.sim_time",             "Durée simulée",          "Numérique", "s",  rel_range=0.20, kind="numerical"),

        # --- Outputs: what is written or extracted, not the physics ---
        ParamSpec("bbox.xmin", "ROI d'extraction xmin", "Sorties", "mm", rel_range=0.20, abs_range=0.01, kind="output"),
        ParamSpec("bbox.xmax", "ROI d'extraction xmax", "Sorties", "mm", rel_range=0.20, abs_range=0.01, kind="output"),
        ParamSpec("bbox.ymin", "ROI d'extraction ymin", "Sorties", "mm", rel_range=0.20, abs_range=0.01, kind="output"),
        ParamSpec("bbox.ymax", "ROI d'extraction ymax", "Sorties", "mm", rel_range=0.20, abs_range=0.01, kind="output"),
        ParamSpec("bbox.zmin", "ROI d'extraction zmin", "Sorties", "mm", rel_range=0.20, abs_range=0.001, kind="output"),
        ParamSpec("bbox.zmax", "ROI d'extraction zmax", "Sorties", "mm", rel_range=0.20, abs_range=0.001, kind="output"),
        ParamSpec("step.n_frames", "Frames de champ", "Sorties", "—", dtype="int", rel_range=0.20, kind="output"),
        ParamSpec("step.output.ho_n_intervals", "Points des historiques", "Sorties", "—", dtype="int", rel_range=0.20, kind="output"),
        ParamSpec("step.output_filter_camera_fps", "Filtre champs : cadence caméra", "Sorties", "Hz", rel_range=0.20, kind="output"),
        ParamSpec("step.output_filter_camera_atten_db", "Filtre champs : atténuation", "Sorties", "dB", rel_range=0.20, kind="output"),
        ParamSpec("step.output_filter_force_acq_hz", "Filtre efforts : acquisition", "Sorties", "Hz", rel_range=0.20, kind="output"),
        ParamSpec("step.output_filter_force_atten_db", "Filtre efforts : atténuation", "Sorties", "dB", rel_range=0.20, kind="output"),
        ParamSpec("step.output_filter_cutoff_hz", "Filtre champs : coupure", "Sorties", "Hz", rel_range=0.20, kind="output"),
        ParamSpec("step.output_filter_cutoff_history_hz", "Filtre efforts : coupure", "Sorties", "Hz", rel_range=0.20, kind="output"),
    ]


# Values whose change rebuilds the Eulerian mesh (element count or element
# positions): the element fields of two runs are then not comparable
# element by element, so no per-element field difference is computed.
EULER_MESH_PATHS = frozenset({
    "elem_size", "euler_geometry.h_wp", "euler_geometry.l_wp",
    "euler_geometry.h_void", "euler_geometry.l_void",
    "euler_position.x0", "euler_position.y0",
})


# Tool material: the keys the model writes for the tool (no Johnson-Cook).
_TOOL_MAT_KEYS = ("rho", "E", "nu", "k", "Cp", "alpha")


# Values that follow another one when it is written (same rule as the tab
# that edits them): path -> (dependent path, function of the new value).
_LINKED = {
    "interaction.heat_fraction_to_slave":
        ("interaction.heat_fraction_to_master", lambda v: max(0.0, 1.0 - v)),
}


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------
REGISTRY: list[ParamSpec] = (
    _material_specs()
    + _material_specs("tool_material", _TOOL_MAT_KEYS, "Matériau outil")
    + _non_material_specs())

# Fast lookup by path
_BY_PATH: dict[str, ParamSpec] = {s.path: s for s in REGISTRY}


def spec_for(path: str) -> ParamSpec:
    """Return the ParamSpec for a dotted path. Raises KeyError if unknown."""
    try:
        return _BY_PATH[path]
    except KeyError:
        raise KeyError(f"No registered parameter at path {path!r}.")


def registry_by_category() -> dict[str, list[ParamSpec]]:
    """Group the registry by `category`, preserving insertion order."""
    out: dict[str, list[ParamSpec]] = {}
    for s in REGISTRY:
        out.setdefault(s.category, []).append(s)
    return out


def default_display_bounds(cfg, spec: ParamSpec,
                           temp_unit: str = "C") -> tuple[float, float]:
    """Suggest a (min, max) range in DISPLAYED units to pre-fill the UI.

    Uses a symmetric band around the current displayed value:
      - if |displayed| is large enough: value ± rel_range * value
      - otherwise (value ~ 0): value ± abs_range

    The returned tuple is always (lo, hi) with lo <= hi.
    """
    if not is_set(cfg, spec):
        return (0.0, 0.0)
    stored = float(get_stored(cfg, spec.path))
    disp = spec.to_display(stored, temp_unit)
    if abs(disp) > 1e-12 and spec.rel_range > 0.0:
        half = abs(disp) * spec.rel_range
    else:
        half = spec.abs_range
    lo, hi = disp - half, disp + half
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def is_set(cfg, spec: ParamSpec) -> bool:
    """True when the model holds a number at `spec.path` (an optional
    value such as the contact conductance is None until it is set)."""
    try:
        v = get_stored(cfg, spec.path)
    except (AttributeError, KeyError):
        return False
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def get_display(cfg, spec: ParamSpec, temp_unit: str = "C",
                system=None) -> float:
    """Current value of `spec` in DISPLAYED units."""
    return spec.to_display(float(get_stored(cfg, spec.path)), temp_unit,
                           system=system)


def apply_display(cfg, spec: ParamSpec, displayed: float,
                  temp_unit: str = "C", system=None) -> None:
    """Write a DISPLAYED value back into the cfg (converting to stored)."""
    stored = spec.to_stored(displayed, temp_unit, system=system)
    set_stored(cfg, spec.path, stored)
    link = _LINKED.get(spec.path)
    if link is not None:
        set_stored(cfg, link[0], link[1](stored))
