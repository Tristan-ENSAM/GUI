# -*- coding: utf-8 -*-
"""
Unit tests for gui.core.tool_geometry_calc (pure tool-geometry maths,
extracted from the GeometryPreview widget). No Qt, no matplotlib.

The closure system solved by solve_tool_dimensions satisfies
    h - tan(clear)*l = h_tool   and   l - tan(rake)*h = l_tool,
which is asserted directly as an invariant (independent of how it is solved),
rather than against hand-computed magic numbers.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from gui.core.tool_geometry_calc import (
    ToolGeometryError, solve_tool_dimensions, tool_polygon,
    point_segment_distance, resolve_tool_translation)


# ---------------------------------------------------------------------------
# solve_tool_dimensions
# ---------------------------------------------------------------------------
class TestSolveToolDimensions:

    def test_zero_angles_identity(self):
        h, l = solve_tool_dimensions(10.0, 20.0, 0.0, 0.0)
        assert h == pytest.approx(10.0) and l == pytest.approx(20.0)

    def test_closure_invariant(self):
        # For a valid positive case the solved (h, l) must satisfy the closure
        # relations exactly.
        rake_deg, clear_deg = 20.0, 10.0
        h, l = solve_tool_dimensions(10.0, 20.0, rake_deg, clear_deg)
        tr = math.tan(math.radians(rake_deg))
        tc = math.tan(math.radians(clear_deg))
        assert h - tc * l == pytest.approx(10.0, abs=1e-9)
        assert l - tr * h == pytest.approx(20.0, abs=1e-9)
        # Positive angles enlarge the bounding box.
        assert h > 10.0 and l > 20.0

    def test_singular_raises(self):
        # tan(45)*tan(45) = 1 -> det = 0.
        with pytest.raises(ToolGeometryError):
            solve_tool_dimensions(10.0, 20.0, 45.0, 45.0)

    def test_beyond_singular_raises(self):
        # tan(60)*tan(60) = 3 > 1 -> det < 0.
        with pytest.raises(ToolGeometryError):
            solve_tool_dimensions(10.0, 20.0, 60.0, 60.0)


# ---------------------------------------------------------------------------
# tool_polygon
# ---------------------------------------------------------------------------
class TestToolPolygon:

    def test_sharp_corner_rectangle(self):
        # rake = clear = 0, r = 0 -> rectangle [0, l] x [0, h] with a sharp
        # cutting edge at the origin.
        p = tool_polygon(10.0, 20.0, 0.0, 0.0, 0.0)
        assert p.ndim == 2 and p.shape[1] == 2
        assert p[:, 0].min() == pytest.approx(0.0)
        assert p[:, 0].max() == pytest.approx(20.0)
        assert p[:, 1].min() == pytest.approx(0.0)
        assert p[:, 1].max() == pytest.approx(10.0)
        # the sharp tip (0,0) is a vertex
        assert np.any((np.abs(p[:, 0]) < 1e-9) & (np.abs(p[:, 1]) < 1e-9))

    def test_fillet_replaces_corner(self):
        # r > 0 adds the fillet arc (n_fillet points) and removes the sharp
        # corner: no vertex sits exactly at the origin, and the closest
        # approach is strictly positive.
        p = tool_polygon(10.0, 20.0, 1.0, 0.0, 0.0, n_fillet=24)
        assert p.shape[0] > 6                       # more points than the sharp case
        dmin = float(np.min(np.hypot(p[:, 0], p[:, 1])))
        assert dmin > 0.0

    def test_singular_propagates(self):
        with pytest.raises(ToolGeometryError):
            tool_polygon(10.0, 20.0, 0.5, 45.0, 45.0)


# ---------------------------------------------------------------------------
# point_segment_distance
# ---------------------------------------------------------------------------
class TestPointSegmentDistance:

    def test_on_segment(self):
        assert point_segment_distance(0, 0, -1, 0, 1, 0) == pytest.approx(0.0)

    def test_perpendicular(self):
        assert point_segment_distance(0, 1, -1, 0, 1, 0) == pytest.approx(1.0)

    def test_beyond_endpoint(self):
        # Past the (1,0) end -> distance to that endpoint.
        assert point_segment_distance(2, 0, -1, 0, 1, 0) == pytest.approx(1.0)

    def test_degenerate_segment(self):
        # Zero-length segment -> distance to the point.
        assert point_segment_distance(3, 4, 0, 0, 0, 0) == pytest.approx(5.0)


class TestResolveToolTranslation:
    """The tool's LOCAL frame (BL = the theoretical zero-radius corner) must
    be translated by (dx, dy) so that (1) the fillet is tangent to y0_tool
    and (2) the tool's outer boundary passes through (x0_tool, y0_wp)."""

    PARAMS = dict(h_tool=0.3, l_tool=0.3, r_tool=0.01,
                  rake_deg=-30.0, clear_deg=30.0)

    def _check_constraints(self, dx, dy, params, x0_tool, y0_tool, y0_wp):
        # Constraint 1: the fillet's lowest point (tangent to horizontal) is
        # at world y = y0_tool.
        rake = math.radians(params["rake_deg"]); clear = math.radians(params["clear_deg"])
        det = 1.0 - math.tan(rake) * math.tan(clear)
        h = (params["h_tool"] + math.tan(clear) * params["l_tool"]) / det
        l = (math.tan(rake) * params["h_tool"] + params["l_tool"]) / det
        TL = (h * math.tan(rake), h)
        BR = (l, l * math.tan(clear))
        ur = (TL[0] / math.hypot(*TL), TL[1] / math.hypot(*TL))
        ub = (BR[0] / math.hypot(*BR), BR[1] / math.hypot(*BR))
        alpha = math.acos(max(-1.0, min(1.0, ur[0]*ub[0]+ur[1]*ub[1])))
        r = params["r_tool"]
        bis = (ur[0]+ub[0], ur[1]+ub[1]); bn = math.hypot(*bis)
        C = ((r/math.sin(alpha/2))*bis[0]/bn, (r/math.sin(alpha/2))*bis[1]/bn)
        bottom_world_y = (C[1] - r) + dy
        assert bottom_world_y == pytest.approx(y0_tool, abs=1e-9)

        # Constraint 2: the boundary passes through (x0_tool, y0_wp).
        y_local = y0_wp - dy
        if y_local <= C[1] - r + 1e-6:
            return  # non-engaging case, checked separately
        Pry = None
        t = r / math.tan(alpha/2)
        Pry = t * ur[1]
        if (C[1]-r-1e-9) <= y_local <= Pry+1e-9:
            dyc = y_local - C[1]
            x_local = C[0] - math.sqrt(max(0.0, r*r - dyc*dyc))
        else:
            x_local = y_local * math.tan(rake)
        assert x_local + dx == pytest.approx(x0_tool, abs=1e-9)

    def test_contact_on_rake_face(self):
        dx, dy, engages, source = resolve_tool_translation(
            **self.PARAMS, x0_tool=0.0, y0_tool=-0.05, y0_wp=0.0)
        assert engages is True
        assert source == "rake face"
        self._check_constraints(dx, dy, self.PARAMS, 0.0, -0.05, 0.0)

    def test_contact_on_fillet(self):
        # r_tool (0.05) exceeds the uncut chip thickness (0.03) here, so the
        # tangent-x shortcut is not reachable and this genuinely exercises
        # the coincidence-on-arc branch (not "fillet-tangent-x").
        params = dict(h_tool=0.3, l_tool=0.3, r_tool=0.05,
                      rake_deg=5.0, clear_deg=5.0)
        dx, dy, engages, source = resolve_tool_translation(
            **params, x0_tool=0.0, y0_tool=-0.03, y0_wp=0.0)
        assert engages is True
        assert source == "fillet"
        self._check_constraints(dx, dy, params, 0.0, -0.03, 0.0)

    def test_non_engaging_tool_is_flagged(self):
        # y0_tool ABOVE y0_wp: the tool's deepest point never reaches the
        # workpiece surface -> no cutting engagement, must be flagged.
        dx, dy, engages, reason = resolve_tool_translation(
            **self.PARAMS, x0_tool=0.0, y0_tool=0.05, y0_wp=0.0)
        assert engages is False
        assert dx is None
        assert "no engagement" in reason or "no cutting" in reason

    def test_zero_radius_keeps_dy_but_shifts_dx_when_depth_nonzero(self):
        # At r_tool -> 0 the depth reference (dy) matches the old convention
        # exactly, but dx generally does NOT equal x0_tool unless the depth
        # of cut is zero (y0_tool == y0_wp) — x0_tool's meaning changed from
        # "BL's x" to "x where the boundary crosses y0_wp".
        dx, dy, engages, source = resolve_tool_translation(
            h_tool=0.3, l_tool=0.3, r_tool=0.0, rake_deg=-30.0, clear_deg=30.0,
            x0_tool=0.0, y0_tool=-0.05, y0_wp=0.0)
        assert engages is True
        assert dy == pytest.approx(-0.05)
        assert dx != pytest.approx(0.0)   # NOT the old (dx=x0_tool) behaviour

    def test_zero_radius_zero_depth_matches_old_convention(self):
        # Special case where both conventions DO coincide: zero depth of cut.
        dx, dy, engages, source = resolve_tool_translation(
            h_tool=0.3, l_tool=0.3, r_tool=0.0, rake_deg=-30.0, clear_deg=30.0,
            x0_tool=0.0, y0_tool=0.0, y0_wp=0.0)
        assert dx == pytest.approx(0.0, abs=1e-9)
        assert dy == pytest.approx(0.0, abs=1e-9)

    def test_leading_edge_tangent_x_when_visible_and_reachable(self):
        # rake=30, clear=20, y0_tool=-0.04: the fillet's vertical-tangent
        # (leftmost) point is exposed by this corner AND reachable
        # (r_tool=0.01 <= uncut chip thickness=0.04) -> must be used for x.
        params = dict(h_tool=0.3, l_tool=0.3, r_tool=0.01,
                      rake_deg=30.0, clear_deg=20.0)
        dx, dy, engages, source = resolve_tool_translation(
            **params, x0_tool=0.0, y0_tool=-0.04, y0_wp=0.0)
        assert engages is True
        assert source == "fillet-tangent-x"

        # Both tangency constraints verified independently and numerically.
        rake = math.radians(30.0); clear = math.radians(20.0)
        h, l = solve_tool_dimensions(params["h_tool"], params["l_tool"],
                                    params["rake_deg"], params["clear_deg"])
        TL = (h * math.tan(rake), h); BR = (l, l * math.tan(clear))
        ur = (TL[0] / math.hypot(*TL), TL[1] / math.hypot(*TL))
        ub = (BR[0] / math.hypot(*BR), BR[1] / math.hypot(*BR))
        alpha = math.acos(max(-1.0, min(1.0, ur[0]*ub[0] + ur[1]*ub[1])))
        r = params["r_tool"]
        bis = (ur[0]+ub[0], ur[1]+ub[1]); bn = math.hypot(*bis)
        C = ((r/math.sin(alpha/2))*bis[0]/bn, (r/math.sin(alpha/2))*bis[1]/bn)
        bottom_world = (C[0] + dx, C[1] - r + dy)
        left_world = (C[0] - r + dx, C[1] + dy)
        assert bottom_world[1] == pytest.approx(-0.04, abs=1e-9)   # depth (Y)
        assert left_world[0] == pytest.approx(0.0, abs=1e-9)       # leading edge (X)

    def test_falls_back_when_visible_but_not_reachable(self):
        # Same corner shape (visible), but r_tool now EXCEEDS the uncut chip
        # thickness (0.03 > 0.02) -> the tangent point would float above the
        # workpiece surface, so constraint 2 must fall back to a crossing.
        dx, dy, engages, source = resolve_tool_translation(
            h_tool=0.3, l_tool=0.3, r_tool=0.03, rake_deg=30.0, clear_deg=20.0,
            x0_tool=0.0, y0_tool=-0.02, y0_wp=0.0)
        assert engages is True
        assert source != "fillet-tangent-x"

    def test_negative_rake_still_uses_crossing_not_tangent(self):
        # Regression: the original (rake=-30) example must still resolve on
        # the rake face, unaffected by the new leading-edge tangent logic
        # (that direction of tangency is not exposed by this corner).
        dx, dy, engages, source = resolve_tool_translation(
            h_tool=0.3, l_tool=0.3, r_tool=0.01, rake_deg=-30.0, clear_deg=30.0,
            x0_tool=0.0, y0_tool=-0.05, y0_wp=0.0)
        assert engages is True
        assert source == "rake face"
        assert dx == pytest.approx(0.03098076211353316, rel=1e-6)
        assert dy == pytest.approx(-0.05366025403784439, rel=1e-6)

    def test_down_tangent_not_visible_falls_back_to_arc_endpoint(self):
        # rake=20, clear=-45 (negative clearance -- unusual but a valid
        # closure system) exposes the horizontal-tangent-not-visible case:
        # the depth reference falls back to the lower of the two arc
        # endpoints (P_on_bot here) instead of the circle's true bottom.
        params = dict(h_tool=0.3, l_tool=0.3, r_tool=0.01,
                      rake_deg=20.0, clear_deg=-45.0)
        dx, dy, engages, source = resolve_tool_translation(
            **params, x0_tool=0.0, y0_tool=-0.05, y0_wp=0.0)
        assert engages is True   # must not raise/crash

        rake = math.radians(params["rake_deg"])
        clear = math.radians(params["clear_deg"])
        h, l = solve_tool_dimensions(params["h_tool"], params["l_tool"],
                                    params["rake_deg"], params["clear_deg"])
        TL = (h * math.tan(rake), h); BR = (l, l * math.tan(clear))
        ur = (TL[0] / math.hypot(*TL), TL[1] / math.hypot(*TL))
        ub = (BR[0] / math.hypot(*BR), BR[1] / math.hypot(*BR))
        alpha = math.acos(max(-1.0, min(1.0, ur[0]*ub[0] + ur[1]*ub[1])))
        r = params["r_tool"]
        t = r / math.tan(alpha / 2)
        Pr = (t * ur[0], t * ur[1]); Pb = (t * ub[0], t * ub[1])
        expected_bottom_local = min(Pb[1], Pr[1])
        assert expected_bottom_local + dy == pytest.approx(-0.05, abs=1e-9)
