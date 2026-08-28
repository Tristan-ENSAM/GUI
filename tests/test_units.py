# -*- coding: utf-8 -*-
"""
Unit tests for gui.core.units.

Pure logic — no Qt, no Abaqus. Covers temperature conversion (C/K),
the GUI<->Abaqus material conversions (round-trip invariants plus the
documented engineering sanity-check values), the MATERIAL_FIELDS table,
display unit/label lookups, and the active-unit-system getter/setter.

The documented sanity values (module docstring): copper rho=8960 kg/m³ ->
8.96e-9 t/mm³, E=124 GPa -> 124000 MPa, Cp=383 J/(kg·K) -> 383e6 mJ/(t·°C).
These are the expected values, not invented constants.
"""
from __future__ import annotations

import pytest

from gui.core import units
from gui.core import unit_system as us


# ---------------------------------------------------------------------------
# Keep the global active unit system pristine across tests.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_active_system():
    prev = units.active_system()
    try:
        yield
    finally:
        units.set_active_system(prev)


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
class TestTemperature:

    def test_celsius_passthrough(self):
        assert units.temp_to_abaqus(150.0, "C") == 150.0
        assert units.temp_from_abaqus(150.0, "C") == 150.0

    def test_kelvin_offset(self):
        assert units.temp_to_abaqus(300.0, "K") == pytest.approx(300.0 - 273.15)
        assert units.temp_from_abaqus(0.0, "K") == pytest.approx(273.15)

    def test_kelvin_roundtrip(self):
        x = 537.0
        assert units.temp_from_abaqus(
            units.temp_to_abaqus(x, "K"), "K") == pytest.approx(x)


# ---------------------------------------------------------------------------
# Material conversions
# ---------------------------------------------------------------------------
class TestMaterialConversions:

    def test_documented_copper_values(self):
        # Module docstring sanity values for the default unit system.
        assert units.gui_to_abaqus("rho", 8960.0) == pytest.approx(8.96e-9, rel=1e-9)
        assert units.gui_to_abaqus("E", 124.0) == pytest.approx(124000.0, rel=1e-9)
        assert units.gui_to_abaqus("Cp", 383.0) == pytest.approx(383e6, rel=1e-9)

    def test_roundtrip_invariant(self):
        # Invertible for every kind of field, whatever the factor.
        for key, val in (("rho", 8960.0), ("E", 124.0), ("Cp", 383.0),
                         ("k", 386.0), ("A", 90.0), ("n", 0.31), ("nu", 0.34)):
            internal = units.gui_to_abaqus(key, val)
            assert units.abaqus_to_gui(key, internal) == pytest.approx(val, rel=1e-9)

    def test_dimensionless_passthrough(self):
        # nu / n have factor 1.0: gui == abaqus.
        assert units.gui_to_abaqus("nu", 0.33) == pytest.approx(0.33)
        assert units.gui_to_abaqus("n", 0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# MATERIAL_FIELDS table
# ---------------------------------------------------------------------------
class TestMaterialFields:

    def test_temp_fields_have_no_factor(self):
        # Tm / Tr are special-cased (K/°C toggle) -> factor None.
        assert units.MATERIAL_FIELDS["Tm"][2] is None
        assert units.MATERIAL_FIELDS["Tr"][2] is None

    def test_known_units(self):
        assert units.MATERIAL_FIELDS["rho"][1] == "kg/m³"
        assert units.MATERIAL_FIELDS["E"][1] == "GPa"
        assert units.MATERIAL_FIELDS["Cp"][1] == "J/(kg·K)"


# ---------------------------------------------------------------------------
# Display unit / label
# ---------------------------------------------------------------------------
class TestDisplay:

    def test_display_unit_default_system(self):
        assert units.display_unit("E") == "GPa"
        assert units.display_unit("rho") == "kg/m³"

    def test_display_label(self):
        assert units.display_label("E") == "E (Young)"

    def test_display_label_unknown_key(self):
        # Unknown key falls back to the key itself.
        assert units.display_label("not_a_field") == "not_a_field"


# ---------------------------------------------------------------------------
# Active system getter/setter
# ---------------------------------------------------------------------------
class TestActiveSystem:

    def test_set_and_get(self):
        custom = us.UnitSystem(temp="K")
        units.set_active_system(custom)
        assert units.active_system() is custom
        # Temperature display now follows the active system base.
        # (restored to the previous system by the autouse fixture)
