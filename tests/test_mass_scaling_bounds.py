# -*- coding: utf-8 -*-
"""
Analytical mass-scaling window (ModelConfig.initial_stable_dt /
mass_scaling_bounds).

The point of these formulas is that the admissible window is COMPUTABLE from
the mesh, the materials and the domain size -- no reference simulation. The
reference values below come from two real Abaqus runs of the same model.
"""
from __future__ import annotations

import pytest

from gui.core.model_config import ModelConfig


def _cfg(elem_size=0.005, l=0.1, h=0.1):
    c = ModelConfig()
    c.euler_material.update({"E": 113800.0, "nu": 0.342, "rho": 4.43e-9})
    c.elem_size = elem_size
    c.euler_geometry.l_wp = l
    c.euler_geometry.l_void = l
    c.euler_geometry.h_wp = h
    c.euler_geometry.h_void = h
    return c


class TestInitialStableDt:
    """dt0 = (h/sqrt(3))/c_d * (sqrt(1+b1^2) - b1), validated on real runs."""

    @pytest.mark.parametrize("elem_size,measured", [
        (0.005, 4.30630e-10),   # .sta of the 5 um run
        (0.002, 1.72300e-10),   # .sta of the 2 um run
    ])
    def test_matches_abaqus(self, elem_size, measured):
        got = _cfg(elem_size).initial_stable_dt()
        assert got == pytest.approx(measured, rel=1e-3)

    def test_naive_formula_would_be_wrong(self):
        # h/c_d overestimates by ~1.84x: the characteristic length of a hex is
        # h/sqrt(3), and bulk viscosity shaves a further ~6%.
        c = _cfg(0.005)
        naive = 0.005 / 6313.4e3      # mm / (mm/s)
        assert naive / c.initial_stable_dt() == pytest.approx(1.84, rel=0.02)

    def test_zero_on_bad_input(self):
        c = _cfg()
        c.euler_material["E"] = 0.0
        assert c.initial_stable_dt() == 0.0


class TestBounds:
    def test_lower_bound_is_filter_validity(self):
        b = _cfg().mass_scaling_bounds(88.6e3)
        # ms > (1e-3 / (fc*dt0))^2
        assert b["ms_min"] == pytest.approx(687, rel=0.02)

    def test_upper_bound_scales_with_domain(self):
        # The reverberation bound goes as 1/L^2: doubling the domain divides
        # the admissible factor by four.
        small = _cfg(l=0.1, h=0.1).mass_scaling_bounds(88.6e3)["ms_freq"]
        big = _cfg(l=0.2, h=0.2).mass_scaling_bounds(88.6e3)["ms_freq"]
        assert small / big == pytest.approx(4.0, rel=1e-6)

    def test_refining_the_mesh_closes_the_window(self):
        # Lower bound goes as 1/dt0^2 (so as 1/h^2) while the upper bound does
        # not move with h at all -> refining eventually empties the window.
        coarse = _cfg(0.005).mass_scaling_bounds(88.6e3)
        fine = _cfg(0.002).mass_scaling_bounds(88.6e3)
        assert coarse["empty"] is False
        assert fine["ms_min"] > coarse["ms_min"]
        assert fine["ms_freq"] == pytest.approx(coarse["ms_freq"])

    def test_empty_window_is_flagged(self):
        b = _cfg(0.002, l=0.2, h=0.2).mass_scaling_bounds(88.6e3)
        assert b["empty"] is True

    def test_raising_the_cutoff_widens_the_window(self):
        low = _cfg().mass_scaling_bounds(30e3)
        high = _cfg().mass_scaling_bounds(88.6e3)
        assert high["ms_min"] < low["ms_min"]

    def test_unusable_input_returns_empty(self):
        c = _cfg()
        c.euler_material["rho"] = 0.0
        b = c.mass_scaling_bounds(88.6e3)
        assert b["ms_min"] is None and b["empty"] is True


class TestSingleFactor:
    """One GUI field, applied to BOTH bodies by the exporter."""

    def test_tool_field_is_gone(self):
        s = ModelConfig().step
        assert not hasattr(s, "mass_scaling_factor_tool")
        assert not hasattr(s, "mass_scaling_factor_eulerian")
        assert hasattr(s, "mass_scaling_factor")

    def test_export_expands_to_both_keys(self):
        c = ModelConfig()
        c.step.mass_scaling_factor = 2500.0
        step = c.to_params_dict()["step"]
        assert step["mass_scaling_factor_eulerian"] == pytest.approx(2500.0)
        assert step["mass_scaling_factor_tool"] == pytest.approx(2500.0)
        assert "mass_scaling_factor" not in step

    def test_filter_is_exported(self):
        c = ModelConfig()
        c.step.output_filter_enabled = True
        c.step.output_filter_cutoff_hz = 88600.0
        step = c.to_params_dict()["step"]
        assert step["output_filter_enabled"] is True
        assert step["output_filter_cutoff_hz"] == pytest.approx(88600.0)
