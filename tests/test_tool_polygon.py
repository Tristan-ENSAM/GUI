# -*- coding: utf-8 -*-
"""
Tests for the tool-polygon geometry helpers (gui.core.tool_polygon).
"""
import numpy as np
import pytest

from gui.core.tool_polygon import (ToolPolygon, snap_edge_to_border,
                                    _edge_vertices)


def test_edge_vertices():
    assert _edge_vertices(4, 0) == (0, 1)
    assert _edge_vertices(4, 3) == (3, 0)


def test_snap_edge_to_border():
    p1, p2 = snap_edge_to_border((2, 50), (198, 50), 200, 200, tol=8)
    assert p1 == (0.0, 50.0)
    assert p2 == (199.0, 50.0)


def test_snap_no_effect_when_far():
    p1, p2 = snap_edge_to_border((50, 50), (100, 100), 200, 200, tol=8)
    assert p1 == (50.0, 50.0) and p2 == (100.0, 100.0)


class TestToolPolygon:

    def _tp(self):
        return ToolPolygon(vertices=[(50, 100), (150, 100), (150, 20), (60, 20)],
                           tip_index=0, rake_edge=0, flank_edge=3)

    def test_complete(self):
        assert self._tp().is_complete()
        incomplete = ToolPolygon(vertices=[(0, 0), (1, 0), (1, 1), (0, 1)])
        assert not incomplete.is_complete()

    def test_complete_requires_distinct_edges(self):
        tp = ToolPolygon(vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
                         tip_index=0, rake_edge=1, flank_edge=1)
        assert not tp.is_complete()

    def test_tip_point(self):
        assert self._tp().tip_point() == (50.0, 100.0)

    def test_face_lengths(self):
        lr, lf = self._tp().face_lengths_px()
        assert abs(lr - 100.0) < 1e-9        # edge 0: (50,100)->(150,100)
        assert lf > 0

    def test_angles_finite(self):
        tp = self._tp()
        assert np.isfinite(tp.rake_angle_deg())
        assert np.isfinite(tp.flank_angle_deg())

    def test_to_model_dict(self):
        d = self._tp().to_model_dict(200, 200, 0.01)
        for k in ("tool_x0", "tool_y0", "rake_angle", "clear_angle",
                  "rake_len_mm", "flank_len_mm", "tool_polygon_px"):
            assert k in d
        assert len(d["tool_polygon_px"]) == 4

    def test_derived_tip_from_adjacent_edges(self):
        # rake=edge0 (v0-v1), flank=edge3 (v3-v0) share vertex 0 -> tip=0.
        tp = ToolPolygon(vertices=[(100, 100), (150, 90), (160, 30), (60, 40)],
                         rake_edge=0, flank_edge=3)
        assert tp.derived_tip_index() == 0
        assert tp.is_complete()
        assert tp.tip_point() == (100.0, 100.0)

    def test_opposite_edges_no_tip(self):
        # rake=edge0, flank=edge2 are opposite -> no shared vertex.
        tp = ToolPolygon(vertices=[(0, 0), (10, 0), (10, 10), (0, 10)],
                         rake_edge=0, flank_edge=2)
        assert tp.derived_tip_index() is None
        assert not tp.is_complete()

    def test_extend_faces_keeps_four_vertices(self):
        tp = ToolPolygon(vertices=[(100, 100), (150, 90), (160, 30), (60, 40)],
                         rake_edge=0, flank_edge=3)
        tp.extend_faces_to_border(200, 200)
        assert len(tp.vertices) == 4
        # The tip stays in place; the far endpoints reach a border.
        assert tp.tip_point() == (100.0, 100.0)
        on_border = [
            (abs(x) < 1e-6 or abs(x - 199) < 1e-6
             or abs(y) < 1e-6 or abs(y - 199) < 1e-6)
            for (x, y) in tp.vertices]
        assert sum(on_border) >= 2

    def test_extend_requires_adjacent_edges(self):
        tp = ToolPolygon(vertices=[(0, 0), (10, 0), (10, 10), (0, 10)],
                         rake_edge=0, flank_edge=2)
        with pytest.raises(ValueError):
            tp.extend_faces_to_border(200, 200)

    def test_polygon_px(self):
        tp = self._tp()
        poly = tp.polygon_px()
        assert len(poly) == 4
        assert all(isinstance(v, tuple) for v in poly)


def test_flank_angle_uses_horizontal_convention():
    """Flank/clear angle is measured from the HORIZONTAL (Geometry tab
    convention), not the vertical: a horizontal flank face gives ~0 deg."""
    import numpy as np
    # tip=v0; rake edge0 (v0->v1) vertical; flank edge3 (v3->v0) horizontal.
    tp = ToolPolygon(vertices=[(100, 400), (100, 100), (400, 100), (400, 400)],
                     rake_edge=0, flank_edge=3)
    assert abs(tp.flank_angle_deg()) < 1.0          # horizontal -> ~0
    assert abs(tp.rake_angle_deg()) < 1.0           # vertical -> ~0
    # Flank rising 5 deg above horizontal (image y up = model +).
    dy = -300 * np.tan(np.radians(5))
    tp2 = ToolPolygon(vertices=[(100, 400), (100, 100), (400, 100 + dy), (400, 400)],
                      rake_edge=0, flank_edge=3)
    # edge3 = v3->v0 = (400,400)->(100,400): still horizontal here; check edge2
    # variant instead via a clearly inclined flank.
    tp3 = ToolPolygon(vertices=[(100, 400), (100, 100),
                                (400, 400 + dy), (400, 400)],
                      rake_edge=0, flank_edge=3)
    assert np.isfinite(tp3.flank_angle_deg())
