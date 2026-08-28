# -*- coding: utf-8 -*-
"""
Unit tests for gui.core.domain_sizing (Merchant-based initial Eulerian domain
sizing for the Optimization tab). Pure maths — no Qt, no Abaqus.
"""
from __future__ import annotations

import math
import pytest

from gui.core.domain_sizing import (
    merchant_shear_angle, chip_thickness, shear_band_bracket,
    initial_domain_dimensions
)


# ---------------------------------------------------------------------------
# Merchant shear angle
# ---------------------------------------------------------------------------
class TestMerchantShearAngle:

    def test_frictionless_zero_rake(self):
        assert merchant_shear_angle(0.0, 0.0) == pytest.approx(45.0)

    def test_known_value(self):
        # phi = 45 - atan(0.3)/2 (deg)
        expected = 45.0 - 0.5 * math.degrees(math.atan(0.3))
        assert merchant_shear_angle(0.0, 0.3) == pytest.approx(expected)

    def test_rake_shifts_up(self):
        # +10 deg rake adds 5 deg to phi.
        base = merchant_shear_angle(0.0, 0.3)
        assert merchant_shear_angle(10.0, 0.3) == pytest.approx(base + 5.0)

    def test_clamped_for_extreme_friction(self):
        # atan(1e9) ~ 90 deg -> phi -> 0 -> clamped to 1 deg.
        assert merchant_shear_angle(0.0, 1e9) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Chip thickness
# ---------------------------------------------------------------------------
class TestChipThickness:

    def test_phi45_zero_rake_equals_t1(self):
        assert chip_thickness(0.05, 0.0, 0.0, phi_deg=45.0) == pytest.approx(0.05)

    def test_matches_formula(self):
        t1, rake, mu = 0.05, 5.0, 0.3
        phi = merchant_shear_angle(rake, mu)
        exp = t1 * math.cos(math.radians(phi) - math.radians(rake)) \
            / math.sin(math.radians(phi))
        assert chip_thickness(t1, rake, mu) == pytest.approx(exp)


# ---------------------------------------------------------------------------
# Shear-band bracket
# ---------------------------------------------------------------------------
class TestShearBandBracket:

    def test_floors(self):
        t1, rake, mu = 0.05, 0.0, 0.3
        b = shear_band_bracket(t1, rake, mu)
        phi = math.radians(merchant_shear_angle(rake, mu))
        assert b.h_wp == pytest.approx(t1)
        assert b.l_wp == pytest.approx(t1 / math.tan(phi))
        assert b.h_void == pytest.approx(chip_thickness(t1, rake, mu))
        assert b.l_void == pytest.approx(chip_thickness(t1, rake, mu))  # factor 1

    def test_l_void_factor(self):
        b = shear_band_bracket(0.05, 0.0, 0.3, l_void_factor=2.0)
        assert b.l_void == pytest.approx(2.0 * chip_thickness(0.05, 0.0, 0.3))

    def test_nonpositive_t1_raises(self):
        with pytest.raises(ValueError):
            shear_band_bracket(0.0, 0.0, 0.3)


# ---------------------------------------------------------------------------
# Initial domain dimensions
# ---------------------------------------------------------------------------
def _is_multiple(v, e):
    return abs(v / e - round(v / e)) < 1e-9


class TestInitialDomain:

    def test_bracket_only_snaps_up(self):
        d = initial_domain_dimensions(0.05, 0.0, 0.3, elem_size=0.01,
                                      roi=None, margin_elems=0)
        # h_wp = t1 = 0.05 (already a multiple); the others ceil to 0.07.
        assert d.h_wp == pytest.approx(0.05)
        assert d.l_wp == pytest.approx(0.07)
        assert d.h_void == pytest.approx(0.07)
        assert d.l_void == pytest.approx(0.07)

    def test_roi_is_contained(self):
        roi = (-0.2, 0.02, -0.15, 0.03)
        d = initial_domain_dimensions(0.05, 0.0, 0.3, elem_size=0.01,
                                      roi=roi, margin_elems=0)
        # Domain must contain the ROI in every direction.
        assert d.l_wp >= 0.20 - 1e-9          # -xmin
        assert d.h_wp >= 0.15 - 1e-9          # -ymin
        assert d.l_void >= 0.02 - 1e-9        # xmax
        assert d.h_void >= 0.03 - 1e-9        # ymax
        # The bracket still dominates the small downstream/top extents.
        assert d.l_void == pytest.approx(0.07)

    def test_margin_adds_elements(self):
        roi = (-0.2, 0.02, -0.15, 0.03)
        d0 = initial_domain_dimensions(0.05, 0.0, 0.3, 0.01, roi=roi,
                                       margin_elems=0)
        d2 = initial_domain_dimensions(0.05, 0.0, 0.3, 0.01, roi=roi,
                                       margin_elems=2)
        assert d2.l_wp == pytest.approx(d0.l_wp + 0.02)
        assert d2.h_wp == pytest.approx(d0.h_wp + 0.02)

    def test_all_multiples_of_element(self):
        d = initial_domain_dimensions(0.05, 0.0, 0.3, 0.01,
                                      roi=(-0.2, 0.02, -0.15, 0.03),
                                      margin_elems=2)
        for v in (d.h_wp, d.h_void, d.l_wp, d.l_void):
            assert _is_multiple(v, 0.01)

    def test_tool_bbox_contained(self):
        # A tool footprint extending downstream/up forces l_void/h_void to grow.
        d = initial_domain_dimensions(0.05, 0.0, 0.3, 0.01, roi=None,
                                      tool_bbox=(0.0, 0.30, 0.0, 0.25),
                                      margin_elems=0)
        assert d.l_void >= 0.30 - 1e-9
        assert d.h_void >= 0.25 - 1e-9
