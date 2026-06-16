# -*- coding: utf-8 -*-
"""
Unit conversions between the GUI (human-friendly SI / engineering units)
and the Abaqus internal consistent system used by abq_odb_generator.py.

The Abaqus model uses the consistent unit system:
    length   : mm
    time     : s
    mass     : t  (tonne, 1 t = 1000 kg)
    force    : N
    stress   : MPa
    density  : t/mm³
    energy   : mJ
    spec.    : mJ/(t·°C)
    cond.    : mW/(mm·°C)
    temp.    : °C
This is the system you see in your existing `params` dicts (rho=8.96e-9 for
copper, E=124000 for Cu in MPa, Cp=383e6, etc.).

The GUI exposes more natural units; this module centralises the conversion
so the rest of the code only ever sees Abaqus-internal numbers in
ModelConfig and only ever shows SI numbers to the user.

Quick reference for common materials (so the user can sanity-check values):
    Copper    : ρ=8960 kg/m³, E=124 GPa, Cp=383 J/(kg·K), k=386 W/(m·K)
    Ti-6Al-4V : ρ=4430 kg/m³, E=109 GPa
    Steel     : ρ=7850 kg/m³, E=210 GPa
"""
from __future__ import annotations

import dataclasses

from gui.core import unit_system as _us

# ----------------------------------------------------------------------------
# Active display unit system. The whole app converts through THIS object; the
# default reproduces the historical engineering display exactly (see the
# constants below and tests/test_unit_system.py). MainWindow swaps it when the
# user edits the unit system in Settings.
# ----------------------------------------------------------------------------
_ACTIVE = _us.UnitSystem()


def set_active_system(system: "_us.UnitSystem") -> None:
    global _ACTIVE
    _ACTIVE = system


def active_system() -> "_us.UnitSystem":
    return _ACTIVE


def _sys_for(temp_unit):
    """The active system, with its temperature base overridden by an explicit
    `temp_unit` argument when one is given (backwards-compat: callers still
    pass 'C'/'K')."""
    if temp_unit is None or temp_unit == _ACTIVE.temp:
        return _ACTIVE
    return dataclasses.replace(_ACTIVE, temp=temp_unit)

# ----------------------------------------------------------------------------
# Conversion factors (GUI value × FACTOR = Abaqus internal value)
# ----------------------------------------------------------------------------

# Density: kg/m³  ->  t/mm³
#   1 kg/m³ = 1e-3 t/m³ = 1e-3 / 1e9 t/mm³ = 1e-12 t/mm³
RHO_KGM3_TO_TMM3 = 1.0e-12

# Stress / modulus
GPA_TO_MPA = 1000.0     # 1 GPa = 1000 MPa
MPA_TO_MPA = 1.0        # passthrough (for A, B which are already MPa)

# Conductivity: W/(m·K)  ->  mW/(mm·°C)
#   1 W/(m·K) = 1000 mW / 1000 mm / K = 1 mW/(mm·K)
#   Kelvin and Celsius increments are identical, so 1 W/(m·K) = 1 mW/(mm·°C)
K_WMK_TO_MWMMK = 1.0    # purely numerical coincidence, but very convenient

# Specific heat: J/(kg·K)  ->  mJ/(t·°C)
#   1 J/(kg·K) = 1000 mJ / (1e-3 t) / K = 1e6 mJ/(t·K) = 1e6 mJ/(t·°C)
CP_JKGK_TO_MJTK = 1.0e6

# Thermal expansion: 1/K -> 1/°C  (identical, increments are the same)
ALPHA_INVK_TO_INVC = 1.0

# Fracture energy: N/mm  ->  mJ/mm²
#   1 N/mm × 1 mm = 1 N·mm = 1 mJ ; divided by mm² of crack area gives mJ/mm².
#   In practice: Gf is normally cited in J/m²; the GUI unit N/mm is just
#   J/m² / 1000 = 1e-3 J/m². The conversion N/mm -> mJ/mm² is unity because
#   the Abaqus internal energy/area unit is mJ/mm² = N/mm by definition.
GF_NMM_TO_MJMM2 = 1.0


# Cutting speed: m/min  ->  mm/s
#   1 m/min = 1000 mm / 60 s = 1000/60 mm/s ≈ 16.6667 mm/s
#   Used by the BCs tab so the user can type cutting speeds in the unit
#   conventional in machining (m/min) while cfg stores Abaqus-internal mm/s.
SPEED_MMIN_TO_MMS = 1000.0 / 60.0


# ----------------------------------------------------------------------------
# Temperature: special-cased because GUI may show K or °C
# ----------------------------------------------------------------------------
KELVIN_OFFSET = 273.15


def temp_to_abaqus(value: float, display_unit: str) -> float:
    """Convert a temperature displayed in `display_unit` to Abaqus internal
    units (°C). `display_unit` is 'C' or 'K'."""
    if display_unit == "K":
        return value - KELVIN_OFFSET
    return value  # already in °C


def temp_from_abaqus(value_c: float, display_unit: str) -> float:
    """Inverse: convert Abaqus-internal °C to whatever the GUI is showing."""
    if display_unit == "K":
        return value_c + KELVIN_OFFSET
    return value_c


# ----------------------------------------------------------------------------
# Per-field declarative table for the Materials tab.
# Each entry: (display_label, display_unit_str, factor_gui_to_abaqus)
# Temperature fields are listed with factor=None — they're handled specially
# because of the K/°C toggle.
# ----------------------------------------------------------------------------
MATERIAL_FIELDS = {
    "rho":      ("ρ (density)",       "kg/m³",     RHO_KGM3_TO_TMM3),
    "E":        ("E (Young)",         "GPa",       GPA_TO_MPA),
    "nu":       ("ν (Poisson)",       "—",         1.0),
    "k":        ("k (conductivity)",  "W/(m·K)",   K_WMK_TO_MWMMK),
    "Cp":       ("Cp (spec. heat)",   "J/(kg·K)",  CP_JKGK_TO_MJTK),
    "alpha":    ("α (expansion)",     "1/K",       ALPHA_INVK_TO_INVC),
    "beta":     ("β (Taylor-Quinney)", "—",        1.0),

    # Johnson-Cook plasticity
    "A":        ("A",                 "MPa",       MPA_TO_MPA),
    "B":        ("B",                 "MPa",       MPA_TO_MPA),
    "n":        ("n",                 "—",         1.0),
    "m":        ("m",                 "—",         1.0),
    "Tm":       ("Tm (melt)",         "TEMP",      None),   # special
    "Tr":       ("Tr (room)",         "TEMP",      None),   # special
    "C":        ("C",                 "—",         1.0),
    "eps_dot0": ("ε̇₀",                "1/s",       1.0),

    # JC damage initiation
    "D1":       ("D1", "—", 1.0),
    "D2":       ("D2", "—", 1.0),
    "D3":       ("D3", "—", 1.0),
    "D4":       ("D4", "—", 1.0),
    "D5":       ("D5", "—", 1.0),
    "eps0":     ("ε̇ ref", "1/s", 1.0),

    # Damage evolution
    "Gf":       ("Gf (fracture energy)", "N/mm", GF_NMM_TO_MJMM2),
}


def gui_to_abaqus(key: str, gui_value: float, temp_unit: str = "C") -> float:
    """Convert a single material field from GUI display value to Abaqus
    internal value, using the active unit system. `temp_unit` overrides the
    active system's temperature base (backwards-compat)."""
    kind = _us.field_kind(key)
    return _sys_for(temp_unit).to_internal(kind, gui_value)


def abaqus_to_gui(key: str, abq_value: float, temp_unit: str = "C") -> float:
    """Inverse of `gui_to_abaqus`."""
    kind = _us.field_kind(key)
    return _sys_for(temp_unit).from_internal(kind, abq_value)


def display_unit(key: str, temp_unit: str = "C") -> str:
    """Return the unit string for a key under the active unit system."""
    kind = _us.field_kind(key)
    return _sys_for(temp_unit).unit_label(kind)


def speed_factor() -> float:
    """m/min↔mm/s style factor for the active system's velocity unit:
    display_value * speed_factor() = internal mm/s."""
    return _ACTIVE.factor("velocity_named")


def speed_unit() -> str:
    return _ACTIVE.unit_label("velocity_named")


def display_label(key: str) -> str:
    """Return the label for a key (without the unit)."""
    spec = MATERIAL_FIELDS.get(key)
    return spec[0] if spec else key
