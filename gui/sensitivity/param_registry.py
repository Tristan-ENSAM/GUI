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
  - cutting speed       : stored mm/s, displayed m/min (units.SPEED_MMIN_TO_MMS)
  - ambient temperature : stored °C, displayed °C or K (cfg.ui.temp_unit)

The registry does NOT decide bounds — the UI (Lot 2b) lets the user enter
min/max. The registry only provides a *default* symmetric range around the
current value (`default_display_bounds`) to pre-fill those fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gui.core import units


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
    # Default ± half-range used to pre-fill the UI min/max, expressed on the
    # DISPLAYED value. If the displayed default is ~0 (e.g. rake_angle=0),
    # `abs_range` is used as an absolute half-width instead of `rel_range`.
    rel_range: float = 0.20
    abs_range: float = 0.0

    # -- conversions --
    def to_display(self, stored: float, temp_unit: str = "C") -> float:
        if self.is_temp:
            return units.temp_from_abaqus(stored, temp_unit)
        if self.mat_key:
            return units.abaqus_to_gui(self.mat_key, stored, temp_unit)
        return stored / self.factor if self.factor else stored

    def to_stored(self, displayed: float, temp_unit: str = "C") -> float:
        if self.is_temp:
            return units.temp_to_abaqus(displayed, temp_unit)
        if self.mat_key:
            val = units.gui_to_abaqus(self.mat_key, displayed, temp_unit)
            return int(round(val)) if self.dtype == "int" else val
        val = displayed * self.factor
        return int(round(val)) if self.dtype == "int" else val

    def unit_str(self, temp_unit: str = "C") -> str:
        """Live display unit: material keys resolve through the active unit
        system; other params keep their authored `display_unit`."""
        if self.mat_key:
            return units.display_unit(self.mat_key, temp_unit)
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


def _material_specs() -> list[ParamSpec]:
    specs: list[ParamSpec] = []
    grouped_keys: dict[str, str] = {}
    for cat, keys in _MAT_GROUPS:
        for k in keys:
            grouped_keys[k] = cat

    # Iterate in the order units.MATERIAL_FIELDS declares them so the UI
    # ordering is stable and matches the Materials tab.
    for key, (label, unit, factor) in units.MATERIAL_FIELDS.items():
        cat = grouped_keys.get(key, "Matériau — autres")
        is_temp = (unit == "TEMP")
        # For TEMP keys the displayed unit resolves later via temp_unit;
        # store a neutral placeholder, the UI calls display_unit() anyway.
        disp_unit = "°C" if is_temp else unit
        # factor is None for TEMP fields in MATERIAL_FIELDS
        f = 1.0 if (factor is None) else float(factor)
        rel = _MAT_REL_RANGE.get(key, 0.20)
        specs.append(ParamSpec(
            path=f"euler_material.{key}",
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
    SPEED = units.SPEED_MMIN_TO_MMS   # m/min * SPEED = mm/s
    return [
        # --- Tool geometry (stored in mm / deg, displayed as-is) ---
        ParamSpec("tool_geometry.rake_angle",  "Angle de coupe (rake)",  "Géométrie outil", "deg", rel_range=0.0, abs_range=10.0),
        ParamSpec("tool_geometry.clear_angle", "Angle de dépouille",     "Géométrie outil", "deg", rel_range=0.40, abs_range=3.0),
        ParamSpec("tool_geometry.r_tool",      "Rayon d'arête r",        "Géométrie outil", "mm",  rel_range=0.50),
        ParamSpec("tool_geometry.h_tool",      "Hauteur outil h",        "Géométrie outil", "mm",  rel_range=0.20),
        ParamSpec("tool_geometry.l_tool",      "Longueur outil l",       "Géométrie outil", "mm",  rel_range=0.20),

        # --- Workpiece / Eulerian geometry ---
        ParamSpec("euler_geometry.h_wp",       "Hauteur pièce h_wp",     "Géométrie pièce", "mm",  rel_range=0.20),
        ParamSpec("euler_geometry.l_wp",       "Longueur pièce l_wp",    "Géométrie pièce", "mm",  rel_range=0.20),

        # --- Process / boundary conditions ---
        # cutting_speed: stored mm/s, displayed m/min.
        ParamSpec("bcs.cutting_speed",         "Vitesse de coupe",       "Procédé / CL", "m/min", factor=SPEED, rel_range=0.30),
        # ambient_temperature: stored °C, displayed per temp_unit.
        ParamSpec("bcs.ambient_temperature",   "Température ambiante",   "Procédé / CL", "°C", is_temp=True, rel_range=0.0, abs_range=50.0),

        # --- Contact ---
        ParamSpec("interaction.friction_coeff", "Coefficient de frottement µ", "Contact", "—", rel_range=0.50),

        # --- Numerical ---
        ParamSpec("elem_size",                 "Taille d'élément",       "Numérique", "mm", rel_range=0.40),
        ParamSpec("step.mass_scaling_factor_eulerian", "Mass scaling (pièce)", "Numérique", "—", rel_range=0.0, abs_range=0.0),
    ]


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------
REGISTRY: list[ParamSpec] = _material_specs() + _non_material_specs()

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


def get_display(cfg, spec: ParamSpec, temp_unit: str = "C") -> float:
    """Current value of `spec` in DISPLAYED units."""
    return spec.to_display(float(get_stored(cfg, spec.path)), temp_unit)


def apply_display(cfg, spec: ParamSpec, displayed: float,
                  temp_unit: str = "C") -> None:
    """Write a DISPLAYED value back into the cfg (converting to stored)."""
    set_stored(cfg, spec.path, spec.to_stored(displayed, temp_unit))
