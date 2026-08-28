# -*- coding: utf-8 -*-
"""
Unit tests for gui.core.model_config.

Pure logic — no Qt, no Abaqus. Covers the `discretize` helper, the structure
of `to_params_dict`, and the JSON save/load round-trip (`to_json_dict` /
`from_json_dict`), including the documented behaviour that the `units` block
is authoritative for the temperature base (ui.temp_unit follows it), the
tolerance to missing keys, and the future-version guard.
"""
from __future__ import annotations

import pytest

from gui.core.model_config import ModelConfig, discretize


# ---------------------------------------------------------------------------
# discretize
# ---------------------------------------------------------------------------
class TestDiscretize:

    def test_floors_to_multiple(self):
        assert discretize(1.0, 0.3) == pytest.approx(0.9)   # 3 * 0.3
        assert discretize(1.0, 0.25) == pytest.approx(1.0)  # 4 * 0.25 exactly

    def test_returns_zero_when_smaller_than_step(self):
        assert discretize(0.1, 0.3) == 0.0

    def test_nonpositive_step_raises(self):
        with pytest.raises(ValueError):
            discretize(1.0, 0.0)
        with pytest.raises(ValueError):
            discretize(1.0, -0.1)


# ---------------------------------------------------------------------------
# to_params_dict — structure only (Abaqus correctness is validated elsewhere)
# ---------------------------------------------------------------------------
class TestToParamsDict:

    def test_top_level_keys(self):
        # "analysis" was dropped: the build is CEL-only and run_simul.py reads
        # none of its keys (the fields stay on ModelConfig for the GUI's own
        # use of formulation / rp_location / tool_motion).
        d = ModelConfig().to_params_dict()
        assert set(d.keys()) == {"geometry", "materials",
                                 "mesh", "interaction", "bcs", "step"}

    def test_dead_element_configs_are_gone(self):
        # Element types are frozen in run_simul.py, so the per-body element
        # configs are neither exported nor present on ModelConfig any more.
        c = ModelConfig()
        assert not hasattr(c, "euler_element")
        assert not hasattr(c, "tool_element")
        assert not hasattr(c, "wp_element")
        mesh = c.to_params_dict()["mesh"]
        assert "euler_element" not in mesh
        assert "tool_element" not in mesh

    def test_time_scaling_is_gone(self):
        # kt was removed from run_simul.py, so the knob no longer exists.
        c = ModelConfig()
        assert not hasattr(c.step, "time_scaling_enabled")
        assert not hasattr(c.step, "time_scaling_factor")
        assert not any("time_scaling" in k
                       for k in c.to_params_dict()["step"])

    def test_geometry_substructure(self):
        d = ModelConfig().to_params_dict()
        assert set(d["geometry"].keys()) == {"tool", "euler", "bbox"}
        assert set(d["geometry"]["tool"].keys()) == {"position", "geometry"}
        assert set(d["materials"].keys()) == {"euler", "tool"}

    def test_materials_are_plain_dicts(self):
        d = ModelConfig().to_params_dict()
        assert isinstance(d["materials"]["euler"], dict)
        assert "rho" in d["materials"]["euler"]


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------
class TestJsonRoundtrip:

    def test_scalar_fields_roundtrip(self):
        c = ModelConfig()
        c.elem_size = 0.0123
        c.tool_geometry.rake_angle = 7.5
        c.bcs.cutting_speed = 99.0
        mat_key = next(iter(c.euler_material))
        c.euler_material[mat_key] = c.euler_material[mat_key] + 1.0

        c2 = ModelConfig.from_json_dict(c.to_json_dict())
        assert c2.elem_size == pytest.approx(0.0123)
        assert c2.tool_geometry.rake_angle == pytest.approx(7.5)
        assert c2.bcs.cutting_speed == pytest.approx(99.0)
        assert c2.euler_material[mat_key] == pytest.approx(c.euler_material[mat_key])

    def test_units_block_is_authoritative_for_temp(self):
        # Documented design: the `units` block drives the temperature base and
        # ui.temp_unit follows it on load (model_config.from_json_dict).
        c = ModelConfig()
        c.units.temp = "K"
        c2 = ModelConfig.from_json_dict(c.to_json_dict())
        assert c2.units.temp == "K"
        assert c2.ui.temp_unit == "K"

    def test_tolerates_missing_keys(self):
        # An empty dict yields a default ModelConfig, not an error.
        c = ModelConfig.from_json_dict({})
        assert c.elem_size == ModelConfig().elem_size

    def test_future_version_rejected(self):
        with pytest.raises(ValueError):
            ModelConfig.from_json_dict(
                {"format_version": ModelConfig.FORMAT_VERSION + 1})


class TestToolMeshSeeding:
    """Tool nose seed + bias seeding, exposed in the Mesh tab and consumed by
    run_simul.py (previously hard-coded there)."""

    def test_defaults_are_the_three_level_sizes(self):
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        assert c.tool_elem_size == pytest.approx(0.001)   # min (nose)
        assert c.inter_elem_size == pytest.approx(0.02)   # face end / junction
        assert c.max_elem_size == pytest.approx(0.05)     # border end

    def test_exported_under_mesh_key(self):
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        c.tool_elem_size = 0.004
        c.inter_elem_size = 0.015
        c.max_elem_size = 0.06
        mesh = c.to_params_dict()["mesh"]
        assert mesh["tool_elem_size"] == pytest.approx(0.004)
        assert mesh["inter_elem_size"] == pytest.approx(0.015)
        assert mesh["max_elem_size"] == pytest.approx(0.06)

    def test_tool_conduction_dt_scales_as_L_squared(self):
        # dt = L^2 / (2 alpha): doubling the seed multiplies dt by 4.
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        c.tool_material.update({"k": 46.0, "rho": 1.5e-8, "Cp": 2.03e8})
        c.tool_elem_size = 0.001
        d1 = c.tool_conduction_dt = c.tool_thermal_dt_estimate()
        c.tool_elem_size = 0.002
        d2 = c.tool_thermal_dt_estimate()
        assert d2 / d1 == pytest.approx(4.0)

    def test_matches_the_measured_sta_increment(self):
        # opt_run005.sta reported 1.101244e-08 s with the tool as critical
        # instance; that corresponds to a smallest tool element of ~0.577 um
        # for k=46 W/(m.K), rho=15000 kg/m3, Cp=203 J/(kg.K).
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        c.tool_material.update({"k": 46.0, "rho": 1.5e-8, "Cp": 2.03e8})
        c.tool_elem_size = 0.000577
        assert c.tool_thermal_dt_estimate() == pytest.approx(1.1012e-8, rel=1e-3)

    def test_invariant_under_mass_scaling(self):
        # rho * f and Cp / f leave rho*Cp -> alpha -> the limit unchanged.
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        c.tool_material.update({"k": 46.0, "rho": 1.5e-8, "Cp": 2.03e8})
        c.tool_elem_size = 0.001
        base = c.tool_thermal_dt_estimate()
        f = 1000.0
        c.tool_material.update({"rho": 1.5e-8 * f, "Cp": 2.03e8 / f})
        assert c.tool_thermal_dt_estimate() == pytest.approx(base)

    def test_zero_when_material_missing(self):
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        c.tool_material.update({"k": 0.0})
        assert c.tool_thermal_dt_estimate() == 0.0


class TestExportFloatCleanup:
    """Unit conversions leave representation noise (4.430000000000001e-09,
    24.850000000000023); the export rounds to 12 significant digits."""

    def test_conversion_noise_is_removed(self):
        c = ModelConfig()
        c.euler_material["rho"] = 4.430000000000001e-09
        c.euler_material["Tr"] = 298 - 273.15          # -> 24.850000000000023
        mats = c.to_params_dict()["materials"]["euler"]
        assert repr(mats["rho"]) == "4.43e-09"
        assert repr(mats["Tr"]) == "24.85"

    def test_repeating_decimal_is_shortened_not_broken(self):
        # 40 m/min = 2000/3 mm/s: a genuine repeating decimal, shortened to 12
        # significant digits. The relative change must stay negligible.
        c = ModelConfig()
        c.bcs.cutting_speed = 40 / 60 * 1000
        v = c.to_params_dict()["bcs"]["cutting_speed"]
        assert v == pytest.approx(2000 / 3, rel=1e-11)

    def test_bools_and_ints_are_untouched(self):
        # bool is a subclass of int, not float: it must pass through as-is.
        c = ModelConfig()
        d = c.to_params_dict()
        assert d["mesh"]["discretize"] is True or d["mesh"]["discretize"] is False
        assert isinstance(d["step"]["n_frames"], int)

    def test_nested_structures_are_cleaned(self):
        c = ModelConfig()
        c.tool_geometry.r_tool = 0.010000000000000002
        assert repr(c.to_params_dict()["geometry"]["tool"]
                    ["geometry"]["r_tool"]) == "0.01"
