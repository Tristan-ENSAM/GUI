# -*- coding: utf-8 -*-
"""
Tool-polygon geometry for the Alignment tab (pure, no Qt).

The user draws a 4-sided polygon over the tool, then labels which vertex is the
TOOL TIP and which two edges are the RAKE face and the FLANK face. The two
remaining edges may be "snapped" onto the image borders so the tool is cut
cleanly at the frame edge (no missing tool pixels). From the tip and the two
labelled faces this module also derives the rake/flank angles and a coarse tool
size, reusing the same frame conventions as ``gui.core.alignment``.

Conventions (identical to alignment.py):
  - pixel coordinates: x = column (right), y = row (down),
  - model frame: origin at image centre, x right (cutting direction, tip
    towards +x), y up; via ``pixel_to_model``.

This module only handles geometry; the interactive drawing/labelling lives in
the Alignment tab, and the rasterisation of the polygon to a DIC mask lives in
``gui.core.dic_global.polygon_mask``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from gui.core.alignment import (pixel_to_model, line_tilt_from_vertical_deg,
                                line_tilt_from_horizontal_deg)


Point = Tuple[float, float]


def _edge_vertices(n_vertices: int, edge_index: int) -> Tuple[int, int]:
    """Vertex indices (a, b) of edge ``edge_index`` of a closed polygon with
    ``n_vertices`` vertices: edge k joins vertex k to vertex (k+1) % n."""
    return edge_index % n_vertices, (edge_index + 1) % n_vertices


def snap_edge_to_border(p1: Point, p2: Point, width: int, height: int,
                        tol: float) -> Tuple[Point, Point]:
    """Snap an edge's endpoints onto the nearest image border when they lie
    within ``tol`` pixels of it, so an edge meant to follow the frame edge is
    exactly on it (no tool pixels missed). Each endpoint is snapped
    independently on x and y."""
    def snap(pt):
        x, y = float(pt[0]), float(pt[1])
        if x <= tol:
            x = 0.0
        elif x >= width - 1 - tol:
            x = float(width - 1)
        if y <= tol:
            y = 0.0
        elif y >= height - 1 - tol:
            y = float(height - 1)
        return (x, y)
    return snap(p1), snap(p2)


def _ray_to_border(origin, direction, width: int, height: int):
    """Smallest positive t such that origin + t*direction lands on an image
    border (x in {0, W-1} or y in {0, H-1}). Returns None if the ray never
    reaches a border in the positive direction."""
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    ts = []
    if abs(dx) > 1e-12:
        for bx in (0.0, width - 1.0):
            t = (bx - ox) / dx
            if t > 1e-9:
                y = oy + t * dy
                if -0.5 <= y <= height - 0.5:
                    ts.append(t)
    if abs(dy) > 1e-12:
        for by in (0.0, height - 1.0):
            t = (by - oy) / dy
            if t > 1e-9:
                x = ox + t * dx
                if -0.5 <= x <= width - 0.5:
                    ts.append(t)
    return min(ts) if ts else None


@dataclass
class ToolPolygon:
    """A 4-sided tool polygon with semantic labels.

    Attributes
    ----------
    vertices : list of (x, y)
        The 4 polygon vertices in pixel coordinates, in order (the polygon is
        closed by joining the last vertex back to the first).
    tip_index : int or None
        Index (0..3) of the vertex that is the tool tip.
    rake_edge : int or None
        Index (0..3) of the edge that is the rake face (edge k joins vertex k
        to vertex k+1).
    flank_edge : int or None
        Index of the edge that is the flank face.
    """
    vertices: List[Point] = field(default_factory=list)
    tip_index: Optional[int] = None
    rake_edge: Optional[int] = None
    flank_edge: Optional[int] = None

    # ----- validation -------------------------------------------------------
    def is_complete(self) -> bool:
        """True when the polygon has 4 vertices and rake/flank edges are set to
        two DISTINCT edges (the tip is then derived from them)."""
        return (len(self.vertices) == 4
                and self.rake_edge is not None and self.flank_edge is not None
                and self.rake_edge != self.flank_edge
                and self.shared_vertex(self.rake_edge, self.flank_edge) is not None)

    @staticmethod
    def shared_vertex(edge_a: int, edge_b: int, n: int = 4) -> Optional[int]:
        """Index of the vertex shared by two edges of an n-gon, or None if the
        edges are not adjacent (e.g. opposite edges of a quad share no vertex).
        Edge k joins vertex k to vertex (k+1) % n."""
        va = set(_edge_vertices(n, edge_a))
        vb = set(_edge_vertices(n, edge_b))
        common = va & vb
        return next(iter(common)) if len(common) == 1 else None

    def derived_tip_index(self) -> Optional[int]:
        """Tip vertex deduced as the vertex shared by the rake and flank edges
        (their common corner). None if the two labelled edges are not adjacent.
        This supersedes the manual ``tip_index`` when both faces are labelled.
        """
        if self.rake_edge is None or self.flank_edge is None:
            return None
        return self.shared_vertex(self.rake_edge, self.flank_edge,
                                  len(self.vertices))

    def polygon_px(self) -> List[Point]:
        """The closed polygon vertices (pixel), for rasterisation/masking."""
        return [tuple(map(float, v)) for v in self.vertices]

    # ----- border snapping --------------------------------------------------
    def extend_faces_to_border(self, width: int, height: int) -> None:
        """Extend the rake and flank faces from the tip until they hit an image
        border (in place), keeping exactly 4 vertices.

        The tip stays fixed; the far endpoint of each labelled face is moved
        along the face direction to the first image border it reaches. This
        cleanly cuts the tool at the frame edge without adding vertices and
        without dragging vertices into the corners.
        """
        if len(self.vertices) != 4:
            raise ValueError("need exactly 4 vertices")
        if self.rake_edge is None or self.flank_edge is None:
            raise ValueError("label rake and flank edges first")
        tip_idx = self.derived_tip_index()
        if tip_idx is None:
            raise ValueError("rake and flank edges must be adjacent (shared tip)")
        verts = [list(map(float, v)) for v in self.vertices]
        tip = np.asarray(verts[tip_idx], float)
        far_indices = []
        for edge in (self.rake_edge, self.flank_edge):
            a, b = _edge_vertices(4, edge)
            far_idx = b if a == tip_idx else a
            far_indices.append(far_idx)
            far = np.asarray(verts[far_idx], float)
            d = far - tip
            n = np.hypot(*d)
            if n < 1e-9:
                continue
            d = d / n
            tmax = _ray_to_border(tip, d, width, height)
            if tmax is not None and tmax > 0:
                pt = tip + d * tmax
                verts[far_idx] = [float(pt[0]), float(pt[1])]
        # The remaining 4th vertex (neither the tip nor the two face far-ends)
        # closes the polygon between the two free edges: place it at the image
        # corner nearest to its current position, so the tool is cut cleanly.
        used = {tip_idx, far_indices[0], far_indices[1]}
        rest = [i for i in range(4) if i not in used]
        if rest:
            ri = rest[0]
            cx, cy = verts[ri]
            corner_x = 0.0 if cx < width / 2.0 else float(width - 1)
            corner_y = 0.0 if cy < height / 2.0 else float(height - 1)
            verts[ri] = [corner_x, corner_y]
        self.vertices = [tuple(float(c) for c in v) for v in verts]

    # ----- derived tool geometry -------------------------------------------
    def tip_point(self) -> Point:
        """Pixel coordinates of the tip vertex (derived from the rake/flank
        shared corner when both faces are labelled, else the manual tip)."""
        idx = self.derived_tip_index()
        if idx is None:
            idx = self.tip_index
        if idx is None:
            raise ValueError("tip not defined (label rake and flank edges)")
        return tuple(map(float, self.vertices[idx]))

    def _tip_idx(self) -> Optional[int]:
        idx = self.derived_tip_index()
        return idx if idx is not None else self.tip_index

    def _edge_dir_from_tip(self, edge_index: int) -> Tuple[Point, Point]:
        """Return the labelled edge as (tip, far) so its direction points away
        from the tip. If neither endpoint is the tip, the edge is returned as-is
        (tip not on that edge)."""
        a, b = _edge_vertices(4, edge_index)
        va = tuple(map(float, self.vertices[a]))
        vb = tuple(map(float, self.vertices[b]))
        tip = self._tip_idx()
        if tip == a:
            return va, vb
        if tip == b:
            return vb, va
        return va, vb

    def rake_angle_deg(self) -> float:
        """Rake-face tilt from the vertical (deg), same convention as
        alignment.line_tilt_from_vertical_deg."""
        if self.rake_edge is None:
            raise ValueError("rake edge not labelled")
        p1, p2 = self._edge_dir_from_tip(self.rake_edge)
        return line_tilt_from_vertical_deg(p1, p2)

    def flank_angle_deg(self) -> float:
        """Flank-face tilt from the HORIZONTAL (deg), matching the Geometry tab's
        clear_angle convention (clearance face inclines by `clear` from the
        horizontal, geometry_preview.py: BR = BL + l*(1, tan(clear))). Uses
        alignment.line_tilt_from_horizontal_deg."""
        if self.flank_edge is None:
            raise ValueError("flank edge not labelled")
        p1, p2 = self._edge_dir_from_tip(self.flank_edge)
        return line_tilt_from_horizontal_deg(p1, p2)

    def face_lengths_px(self) -> Tuple[float, float]:
        """Pixel lengths of the rake and flank faces (rough tool size)."""
        def elen(e):
            a, b = _edge_vertices(4, e)
            va = np.asarray(self.vertices[a], float)
            vb = np.asarray(self.vertices[b], float)
            return float(np.hypot(*(vb - va)))
        return elen(self.rake_edge), elen(self.flank_edge)

    def to_model_dict(self, width: int, height: int, mm_per_px: float) -> dict:
        """Tool geometry in the model frame for the numerical model:
        tip (mm), rake/flank angles (deg), face lengths (mm), and the raw pixel
        polygon for the DIC mask."""
        tipx, tipy = self.tip_point()
        tx_mm, ty_mm = pixel_to_model(tipx, tipy, width, height, mm_per_px)
        lr_px, lf_px = self.face_lengths_px()
        return {
            "tool_x0": tx_mm, "tool_y0": ty_mm,
            "rake_angle": self.rake_angle_deg(),
            "clear_angle": self.flank_angle_deg(),
            "rake_len_mm": lr_px * mm_per_px,
            "flank_len_mm": lf_px * mm_per_px,
            "tool_polygon_px": [list(map(float, v)) for v in self.vertices],
        }
