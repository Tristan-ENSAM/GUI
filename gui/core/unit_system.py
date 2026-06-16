# -*- coding: utf-8 -*-
"""
Configurable unit system.

The Abaqus model is always stored in ONE internal consistent system:
    mass = t, length = mm, time = s, temperature = °C
(hence rho = 8.96e-9 for copper, E = 124000 MPa, Cp = 383e6, etc.).

The user picks how values are *displayed/entered* by choosing four base
units — mass, length, time, temperature — from which most quantities are
derived dimensionally (density, velocity, strain rate, thermal expansion).
A few quantities that have no readable base-composed form keep a
conventional *named* unit chosen from a short list ("a few overrides"):
modulus, strength, conductivity, specific heat, fracture energy, velocity.

Everything funnels through ONE direction:
    internal_value = display_value * factor(kind, system)
so `factor` of the default engineering preset reproduces the historical
hard-coded constants in units.py exactly (see tests/test_unit_system.py).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields

# --- base unit -> internal reference (t, mm, s) ----------------------------
MASS_TO_T = {"kg": 1.0e-3, "g": 1.0e-6, "t": 1.0}
LENGTH_TO_MM = {"m": 1.0e3, "mm": 1.0, "µm": 1.0e-3}
TIME_TO_S = {"ms": 1.0e-3, "s": 1.0}
TEMP_CHOICES = ("C", "K")
KELVIN_OFFSET = 273.15

MASS_CHOICES = tuple(MASS_TO_T.keys())
LENGTH_CHOICES = tuple(LENGTH_TO_MM.keys())
TIME_CHOICES = tuple(TIME_TO_S.keys())

# --- derived kinds: (M, L, T) exponents + label template ------------------
# Temperature enters these only as Θ⁻¹ (expansion), where Kelvin and Celsius
# increments are identical, so the temperature contributes a factor of 1.
_DERIVED = {
    "density":     ((1, -3, 0), lambda M, L, T, TH: f"{M}/{L}³"),
    "velocity":    ((0, 1, -1), lambda M, L, T, TH: f"{L}/{T}"),
    "strain_rate": ((0, 0, -1), lambda M, L, T, TH: f"1/{T}"),
    "expansion":   ((0, 0, 0),  lambda M, L, T, TH: f"1/{TH}"),
    "length":      ((0, 1, 0),  lambda M, L, T, TH: f"{L}"),
    "time":        ((0, 0, 1),  lambda M, L, T, TH: f"{T}"),
}

# --- named/override kinds: {label: factor_to_internal} --------------------
# factor_to_internal converts a value in that named unit to the Abaqus
# internal unit (MPa, mW/(mm·°C), mJ/(t·°C), mJ/mm², mm/s).
_NAMED = {
    "modulus":         {"GPa": 1.0e3, "MPa": 1.0, "kPa": 1.0e-3, "Pa": 1.0e-6},
    "strength":        {"GPa": 1.0e3, "MPa": 1.0, "kPa": 1.0e-3, "Pa": 1.0e-6},
    "conductivity":    {"W/(m·K)": 1.0, "mW/(mm·K)": 1.0},
    "specific_heat":   {"J/(kg·K)": 1.0e6, "kJ/(kg·K)": 1.0e9, "J/(g·K)": 1.0e9},
    "fracture_energy": {"N/mm": 1.0, "J/m²": 1.0e-3},
    "velocity_named":  {"m/min": 1000.0 / 60.0, "m/s": 1.0e3, "mm/s": 1.0},
}

# Which attribute of UnitSystem holds the chosen label for each named kind.
_NAMED_ATTR = {
    "modulus": "modulus",
    "strength": "strength",
    "conductivity": "conductivity",
    "specific_heat": "specific_heat",
    "fracture_energy": "fracture_energy",
    "velocity_named": "velocity",
}

# --- material field -> kind ------------------------------------------------
# (the *labels* still come from units.MATERIAL_FIELDS; here we only map the
#  physical kind so the factor/unit follow the active system.)
FIELD_KIND = {
    "rho": "density",
    "E": "modulus",
    "nu": "dimensionless",
    "k": "conductivity",
    "Cp": "specific_heat",
    "alpha": "expansion",
    "beta": "dimensionless",
    "A": "strength", "B": "strength",
    "n": "dimensionless", "m": "dimensionless",
    "Tm": "temperature", "Tr": "temperature",
    "C": "dimensionless",
    "eps_dot0": "strain_rate",
    "D1": "dimensionless", "D2": "dimensionless", "D3": "dimensionless",
    "D4": "dimensionless", "D5": "dimensionless",
    "eps0": "strain_rate",
    "Gf": "fracture_energy",
}


@dataclass
class UnitSystem:
    """A display unit system. Defaults reproduce the historical engineering
    display (kg/m³, GPa, MPa, W/(m·K), J/(kg·K), m/min, °C)."""
    mass: str = "kg"
    length: str = "m"
    time: str = "s"
    temp: str = "C"                       # "C" | "K"
    modulus: str = "GPa"
    strength: str = "MPa"
    conductivity: str = "W/(m·K)"
    specific_heat: str = "J/(kg·K)"
    fracture_energy: str = "N/mm"
    velocity: str = "m/min"

    # --- helpers ---------------------------------------------------------
    def _temp_symbol(self) -> str:
        return "K" if self.temp == "K" else "°C"

    def factor(self, kind: str) -> float:
        """display_value * factor = internal_value."""
        if kind in (None, "dimensionless"):
            return 1.0
        if kind == "temperature":
            return 1.0          # absolute temp handled by to/from_internal
        if kind in _DERIVED:
            (a, b, c), _ = _DERIVED[kind]
            return (MASS_TO_T[self.mass] ** a
                    * LENGTH_TO_MM[self.length] ** b
                    * TIME_TO_S[self.time] ** c)
        if kind in _NAMED:
            label = getattr(self, _NAMED_ATTR[kind])
            return _NAMED[kind][label]
        return 1.0

    def unit_label(self, kind: str) -> str:
        if kind in (None, "dimensionless"):
            return "—"
        if kind == "temperature":
            return self._temp_symbol()
        if kind in _DERIVED:
            _, fmt = _DERIVED[kind]
            return fmt(self.mass, self.length, self.time, self._temp_symbol())
        if kind in _NAMED:
            return getattr(self, _NAMED_ATTR[kind])
        return ""

    # --- value conversion ------------------------------------------------
    def to_internal(self, kind: str, value: float) -> float:
        if kind == "temperature":
            return value - KELVIN_OFFSET if self.temp == "K" else value
        return value * self.factor(kind)

    def from_internal(self, kind: str, value: float) -> float:
        if kind == "temperature":
            return value + KELVIN_OFFSET if self.temp == "K" else value
        f = self.factor(kind)
        return value / f if f else value

    # --- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UnitSystem":
        if not isinstance(d, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def options_for(self, attr: str):
        """Allowed labels for a named-override attribute (for the UI)."""
        for kind, a in _NAMED_ATTR.items():
            if a == attr:
                return tuple(_NAMED[kind].keys())
        return ()


# --- bundled presets -------------------------------------------------------
PRESETS = {
    "Engineering (SI)": UnitSystem(),
    "SI strict (kg·m·s·K)": UnitSystem(
        mass="kg", length="m", time="s", temp="K",
        modulus="Pa", strength="Pa", conductivity="W/(m·K)",
        specific_heat="J/(kg·K)", fracture_energy="J/m²", velocity="m/s"),
    "Millimetre (g·mm·s)": UnitSystem(
        mass="g", length="mm", time="s", temp="C",
        modulus="MPa", strength="MPa", conductivity="W/(m·K)",
        specific_heat="J/(kg·K)", fracture_energy="N/mm", velocity="mm/s"),
}


def field_kind(key: str) -> str:
    return FIELD_KIND.get(key, "dimensionless")
