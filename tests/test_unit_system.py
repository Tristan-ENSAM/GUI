# -*- coding: utf-8 -*-
"""Unit system engine: regression against the historical hard-coded factors,
plus a few base-change checks.

Run:  python -m pytest tests/test_unit_system.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gui.core import unit_system as us
from gui.core import units


def test_default_preset_matches_historical_factors():
    s = us.UnitSystem()                 # engineering default
    # density kg/m³ -> t/mm³
    assert s.to_internal("density", 1.0) == pytest.approx(1.0e-12)
    # modulus GPa -> MPa ; strength MPa -> MPa
    assert s.to_internal("modulus", 1.0) == pytest.approx(1.0e3)
    assert s.to_internal("strength", 1.0) == pytest.approx(1.0)
    # conductivity W/(m·K) -> mW/(mm·°C)
    assert s.to_internal("conductivity", 1.0) == pytest.approx(1.0)
    # specific heat J/(kg·K) -> mJ/(t·°C)
    assert s.to_internal("specific_heat", 1.0) == pytest.approx(1.0e6)
    # expansion, strain rate
    assert s.to_internal("expansion", 1.0) == pytest.approx(1.0)
    assert s.to_internal("strain_rate", 1.0) == pytest.approx(1.0)
    # fracture energy N/mm -> mJ/mm²
    assert s.to_internal("fracture_energy", 1.0) == pytest.approx(1.0)
    # velocity m/min -> mm/s
    assert s.to_internal("velocity_named", 1.0) == pytest.approx(1000.0 / 60.0)


def test_default_labels():
    s = us.UnitSystem()
    assert s.unit_label("density") == "kg/m³"
    assert s.unit_label("modulus") == "GPa"
    assert s.unit_label("strength") == "MPa"
    assert s.unit_label("strain_rate") == "1/s"
    assert s.unit_label("velocity_named") == "m/min"
    assert s.unit_label("temperature") == "°C"


def test_temperature_offset():
    s = us.UnitSystem(temp="K")
    assert s.to_internal("temperature", 293.15) == pytest.approx(20.0)
    assert s.from_internal("temperature", 20.0) == pytest.approx(293.15)
    assert s.unit_label("temperature") == "K"


def test_base_change_recomputes():
    # length m -> mm changes density and velocity factors.
    s = us.UnitSystem(mass="kg", length="mm", time="s")
    # density kg/mm³ -> t/mm³ : only mass converts (kg->t = 1e-3)
    assert s.to_internal("density", 1.0) == pytest.approx(1.0e-3)
    assert s.unit_label("density") == "kg/mm³"
    # mass g, length µm
    s2 = us.UnitSystem(mass="g", length="µm", time="ms")
    # density g/µm³ -> t/mm³ : (g->t=1e-6) * (µm->mm=1e-3)^-3 = 1e-6 * 1e9 = 1e3
    assert s2.to_internal("density", 1.0) == pytest.approx(1.0e3)
    # strain rate 1/ms -> 1/s : (ms->s=1e-3)^-1 = 1e3
    assert s2.to_internal("strain_rate", 1.0) == pytest.approx(1.0e3)
    assert s2.unit_label("strain_rate") == "1/ms"


def test_roundtrip_serialisation():
    s = us.UnitSystem(mass="g", length="mm", time="ms", temp="K",
                      modulus="MPa", velocity="mm/s")
    d = s.to_dict()
    s2 = us.UnitSystem.from_dict(d)
    assert s2 == s
    # unknown keys ignored
    s3 = us.UnitSystem.from_dict({**d, "bogus": 1})
    assert s3 == s


def test_units_module_delegates_to_active_system():
    # Default active system reproduces the historical units.* behaviour.
    units.set_active_system(us.UnitSystem())
    assert units.gui_to_abaqus("E", 124.0) == pytest.approx(124000.0)   # GPa->MPa
    assert units.abaqus_to_gui("rho", 8.96e-9) == pytest.approx(8960.0)  # ->kg/m³
    assert units.display_unit("Cp") == "J/(kg·K)"
    assert units.display_unit("Tm", "K") == "K"

    # Switch the active system: a g·mm·s system changes density display.
    units.set_active_system(us.UnitSystem(mass="g", length="mm", time="s",
                                          modulus="MPa"))
    # 8.96e-9 t/mm³ -> g/mm³ : factor g/mm³->t/mm³ = 1e-6, so /1e-6
    assert units.abaqus_to_gui("rho", 8.96e-9) == pytest.approx(8.96e-3)
    assert units.display_unit("rho") == "g/mm³"
    assert units.gui_to_abaqus("E", 124000.0) == pytest.approx(124000.0)  # MPa
    assert units.display_unit("E") == "MPa"
    # restore default for other tests
    units.set_active_system(us.UnitSystem())


def test_materials_tab_follows_unit_system():
    from PySide6.QtWidgets import QApplication
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from gui.core.model_config import ModelConfig
    from gui.tabs.materials_tab import MaterialsTab
    app = QApplication.instance() or QApplication([])
    cfg = ModelConfig()
    units.set_active_system(cfg.units)
    mt = MaterialsTab(cfg)
    w = mt._wp_widgets["rho"]
    assert "kg/m³" in w._lbl.text()
    v0 = w.value()

    # Switch to g·mm·s and refresh.
    new = us.UnitSystem(mass="g", length="mm", time="s", modulus="MPa")
    cfg.units = new
    units.set_active_system(new)
    mt.refresh_units()
    w = mt._wp_widgets["rho"]
    assert "g/mm³" in w._lbl.text()
    assert w.value() == pytest.approx(v0 * 1.0e-6)   # kg/m³ -> g/mm³
    # internal storage unchanged
    assert cfg.euler_material["rho"] == pytest.approx(8.96e-9, rel=1e-6)
    units.set_active_system(us.UnitSystem())


def test_unit_system_persisted_in_profile():
    from gui.core.model_config import ModelConfig
    cfg = ModelConfig()
    cfg.units = us.UnitSystem(mass="g", length="mm", time="ms", temp="K",
                              modulus="MPa", velocity="mm/s")
    cfg.ui.temp_unit = "K"
    d = cfg.to_json_dict()
    cfg2 = ModelConfig.from_json_dict(d)
    assert cfg2.units == cfg.units
    assert cfg2.ui.temp_unit == "K"
    # Legacy profile (no "units" block) seeds the system from ui.temp_unit.
    d.pop("units")
    d["ui"]["temp_unit"] = "K"
    cfg3 = ModelConfig.from_json_dict(d)
    assert cfg3.units.temp == "K"


def test_job_params_persisted_in_profile():
    from gui.core.model_config import ModelConfig
    cfg = ModelConfig()
    cfg.job.job_name = "ortho_cu_v200"
    cfg.job.cpus = 8
    d = cfg.to_json_dict()
    cfg2 = ModelConfig.from_json_dict(d)
    assert cfg2.job.job_name == "ortho_cu_v200"
    assert cfg2.job.cpus == 8
    # Legacy profile without a "job" block keeps the defaults.
    d.pop("job")
    cfg3 = ModelConfig.from_json_dict(d)
    assert cfg3.job.job_name == "Cutting_job"
    assert cfg3.job.cpus == 4
