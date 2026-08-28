# -*- coding: utf-8 -*-
"""
Unit tests for gui.sensitivity.param_registry.

Pure logic — no Qt, no Abaqus. Covers:
  - the dotted-path reader/writer (get_stored / set_stored), including the
    mixed dataclass-attribute + dict-leaf walk used for material params,
  - ParamSpec stored<->displayed conversions (identity, non-identity factor,
    int rounding, temperature) via round-trip invariants (no hard-coded
    factors beyond the documented speed constant),
  - the public lookups (spec_for, registry_by_category),
  - default_display_bounds (relative band, absolute fallback, lo <= hi).

A tiny duck-typed config (SimpleNamespace + a dict leaf) stands in for a real
ModelConfig so these tests stay decoupled from the heavy config dataclass.
"""
from __future__ import annotations

from types import SimpleNamespace
import pytest

from gui.sensitivity import param_registry as pr
from gui.core import units


def _duck_cfg():
    """Minimal object exposing the dotted paths the tests touch."""
    return SimpleNamespace(
        tool_geometry=SimpleNamespace(r_tool=0.02, rake_angle=0.0),
        interaction=SimpleNamespace(friction_coeff=0.3),
        elem_size=0.01,
        euler_material={"A": 124.0, "B": 200.0},   # dict leaf, like the real cfg
    )


# ---------------------------------------------------------------------------
# Path walking
# ---------------------------------------------------------------------------
class TestPathWalk:

    def test_get_attr_path(self):
        cfg = _duck_cfg()
        assert pr.get_stored(cfg, "tool_geometry.r_tool") == 0.02
        assert pr.get_stored(cfg, "elem_size") == 0.01

    def test_get_dict_leaf(self):
        cfg = _duck_cfg()
        assert pr.get_stored(cfg, "euler_material.A") == 124.0

    def test_set_attr_path(self):
        cfg = _duck_cfg()
        pr.set_stored(cfg, "tool_geometry.r_tool", 0.05)
        assert cfg.tool_geometry.r_tool == 0.05

    def test_set_dict_leaf(self):
        cfg = _duck_cfg()
        pr.set_stored(cfg, "euler_material.A", 130.0)
        assert cfg.euler_material["A"] == 130.0

    def test_roundtrip(self):
        cfg = _duck_cfg()
        for path, val in (("elem_size", 0.004),
                          ("interaction.friction_coeff", 0.42),
                          ("euler_material.B", 250.0)):
            pr.set_stored(cfg, path, val)
            assert pr.get_stored(cfg, path) == val


# ---------------------------------------------------------------------------
# ParamSpec conversions
# ---------------------------------------------------------------------------
class TestParamSpecConversions:

    def test_identity_factor(self):
        s = pr.spec_for("tool_geometry.r_tool")        # factor 1.0, no mat_key
        assert s.to_stored(0.5) == pytest.approx(0.5)
        assert s.to_display(0.5) == pytest.approx(0.5)

    def test_speed_factor(self):
        s = pr.spec_for("bcs.cutting_speed")           # factor = SPEED_MMIN_TO_MMS
        disp = 60.0                                    # m/min
        stored = s.to_stored(disp)
        assert stored == pytest.approx(disp * units.SPEED_MMIN_TO_MMS)
        assert s.to_display(stored) == pytest.approx(disp)

    def test_int_dtype_rounds(self):
        s = pr.ParamSpec(path="x", label="x", category="c",
                         dtype="int", factor=1.0)
        v = s.to_stored(2.6)
        assert isinstance(v, int) and v == 3

    def test_temperature_roundtrip(self):
        s = pr.spec_for("bcs.ambient_temperature")     # is_temp
        x = 25.0
        # Invertible regardless of the chosen temp unit's internal factors.
        assert s.to_stored(s.to_display(x, "C"), "C") == pytest.approx(x)

    def test_material_roundtrip(self):
        # A material spec converts through the active unit system; only assert
        # invertibility, not a specific factor.
        s = pr.spec_for("euler_material.E")
        x = s.to_display(124000.0)
        assert s.to_stored(x) == pytest.approx(124000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
class TestLookups:

    def test_spec_for_known(self):
        s = pr.spec_for("elem_size")
        assert s.path == "elem_size"

    def test_spec_for_unknown_raises(self):
        with pytest.raises(KeyError):
            pr.spec_for("not.a.real.path")

    def test_registry_by_category(self):
        groups = pr.registry_by_category()
        assert isinstance(groups, dict) and groups
        # Every spec appears exactly once across the groups.
        total = sum(len(v) for v in groups.values())
        assert total == len(pr.REGISTRY)
        # Insertion order: the first group is the first spec's category.
        assert next(iter(groups)) == pr.REGISTRY[0].category


# ---------------------------------------------------------------------------
# Default display bounds
# ---------------------------------------------------------------------------
class TestDefaultBounds:

    def test_relative_band(self):
        cfg = _duck_cfg()                              # r_tool = 0.02
        s = pr.spec_for("tool_geometry.r_tool")        # rel_range 0.50
        lo, hi = pr.default_display_bounds(cfg, s)
        assert lo <= hi
        assert lo == pytest.approx(0.01) and hi == pytest.approx(0.03)

    def test_absolute_fallback_for_zero(self):
        cfg = _duck_cfg()                              # rake_angle = 0.0
        s = pr.spec_for("tool_geometry.rake_angle")    # rel_range 0, abs_range 10
        lo, hi = pr.default_display_bounds(cfg, s)
        assert lo <= hi
        assert lo == pytest.approx(-10.0) and hi == pytest.approx(10.0)
