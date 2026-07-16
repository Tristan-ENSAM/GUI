# -*- coding: utf-8 -*-
"""
Pure tool-geometry calculations, decoupled from the (Qt + matplotlib)
GeometryPreview widget so they can be unit-tested without a GUI.

These functions were factored out of gui/widgets/geometry_preview.py verbatim;
that module re-imports them for backward compatibility. Only `math` and
`numpy` are used here.

Convention (matches the user's hand-derived equations):

    BL = (0, 0)                              cutting-edge anchor
    TL = BL + h * (tan(rake), 1)             rake face leans by `rake`
    BR = BL + l * (1, tan(clear))            clearance face inclines by `clear`
    TR = (l, h)                              top-right corner

where (h, l) come from solving the closure system in solve_tool_dimensions.
rake is measured from the vertical, clear from the horizontal.
"""
from __future__ import annotations

import math
import numpy as np


class ToolGeometryError(ValueError):
    """Raised when the tool inputs cannot form a valid closed outline."""
    pass


def solve_tool_dimensions(h_tool: float, l_tool: float,
                          rake_deg: float, clear_deg: float
                          ) -> tuple[float, float]:
    """Compute the overall (h, l) bounding-box dimensions of the tool from
    the four GUI inputs.

    Returns (h, l) where:
        h = total height between the bottom (BR) and the top of the
            cutting edge (TL.y = h_tool exact in the matrix derivation),
            measured at x=0,
        l = total length between the cutting tip (BL at x=0) and the
            right edge of the tool (x=l).

    The closure system is (I - B) (h, l)^T = (h_tool, l_tool)^T with
        B = [[0,         tan(clear)],
             [tan(rake), 0         ]],
    i.e. it satisfies h - tan(clear)*l = h_tool and l - tan(rake)*h = l_tool.

    Raises ToolGeometryError if rake + clear is too close to 90° (or
    beyond), which makes the trapezoidal closure impossible.
    """
    rake  = math.radians(rake_deg)
    clear = math.radians(clear_deg)

    # det(I - B) = 1 - tan(rake) * tan(clear).
    # We refuse a singular or near-singular system. The 1e-6 threshold is
    # arbitrary but corresponds to rake + clear within ≈1e-3 rad of 90°,
    # which is already a deeply pathological tool.
    det = 1.0 - math.tan(rake) * math.tan(clear)
    if det <= 1e-6:
        raise ToolGeometryError(
            f"Invalid tool geometry: rake={rake_deg:.2f}° and "
            f"clear={clear_deg:.2f}° give tan(rake)·tan(clear) ≥ 1 — the "
            f"rake and clearance faces cannot close the tool outline. "
            f"Reduce rake or clear (typical: rake ≤ 30°, clear ≤ 15°)."
        )

    # (I - B)⁻¹ · (h_tool, l_tool):
    #   (I - B)⁻¹ = (1/det) * [[1,         tan(clear)],
    #                          [tan(rake), 1         ]]
    h = (h_tool + math.tan(clear) * l_tool) / det
    l = (math.tan(rake)  * h_tool + l_tool) / det

    if h <= 0.0 or l <= 0.0:
        raise ToolGeometryError(
            f"Invalid tool geometry: solved (h={h:.4g}, l={l:.4g}) is "
            f"non-positive. Check that h_tool and l_tool are positive and "
            f"that rake/clear are within sensible bounds."
        )
    return h, l


def tool_polygon(h_tool: float, l_tool: float, r_tool: float,
                 rake_deg: float, clear_deg: float,
                 n_fillet: int = 24) -> np.ndarray:
    """Build the tool outline in the tool's local frame.

    Faces (counterclockwise outline):
        l1 = rake face       BL -> TL    (tilted by rake from vertical)
        l2 = top face        TL -> TR    (horizontal, at y = h)
        l3 = right face      TR -> BR    (vertical, at x = l)
        l4 = bottom face     BR -> BL    (inclined by clear from horizontal)

    A circular fillet of radius r_tool replaces the BL corner — the actual
    cutting edge, tangent to l1 and l4.

    Sign conventions:
        rake > 0  : top of l1 leans toward +x (positive rake)
        clear > 0 : BR moves upward (positive clearance)

    Raises ToolGeometryError if the closure system is singular or yields
    a non-positive (h, l).
    """
    # 1. Solve the closure system to get the actual (h, l) of the outline
    h, l = solve_tool_dimensions(h_tool, l_tool, rake_deg, clear_deg)

    rake  = math.radians(rake_deg)
    clear = math.radians(clear_deg)

    BL = np.array([0.0, 0.0])
    TL = BL + h * np.array([math.tan(rake), 1.0])
    TR = np.array([l, h])
    BR = BL + l * np.array([1.0, math.tan(clear)])

    # Outgoing unit vectors at BL along l1 (rake) and l4 (bottom).
    # The fillet is the convex arc tangent to both, replacing BL.
    u_rake = (TL - BL); u_rake /= np.linalg.norm(u_rake)
    u_bot  = (BR - BL); u_bot  /= np.linalg.norm(u_bot)

    if r_tool > 0:
        # Half-angle of the interior corner at BL
        cos_a = float(np.clip(np.dot(u_rake, u_bot), -1.0, 1.0))
        alpha = math.acos(cos_a)            # interior corner angle
        t = r_tool / math.tan(alpha / 2.0)  # tangent-point distance from BL

        P_on_rake = BL + t * u_rake         # tangent point on l1
        P_on_bot  = BL + t * u_bot          # tangent point on l4

        # Arc center: on the interior bisector at distance r/sin(α/2)
        bisector = (u_rake + u_bot)
        bisector /= np.linalg.norm(bisector)
        C = BL + (r_tool / math.sin(alpha / 2.0)) * bisector

        # Short arc between the tangent points (the one that does NOT contain BL)
        a_start = math.atan2(P_on_bot[1]  - C[1], P_on_bot[0]  - C[0])
        a_end   = math.atan2(P_on_rake[1] - C[1], P_on_rake[0] - C[0])
        delta = a_end - a_start
        while delta <= -math.pi: delta += 2 * math.pi
        while delta >   math.pi: delta -= 2 * math.pi
        angs = np.linspace(a_start, a_start + delta, n_fillet)
        arc  = np.column_stack([C[0] + r_tool * np.cos(angs),
                                C[1] + r_tool * np.sin(angs)])
    else:
        P_on_rake = BL.copy()
        P_on_bot  = BL.copy()
        arc = np.array([BL])

    # Counterclockwise outline:
    #   start on l4 at P_on_bot, go to BR, up l3 to TR, across l2 to TL,
    #   down l1 to P_on_rake, then arc back to P_on_bot.
    poly = np.vstack([
        P_on_bot,
        BR, TR, TL,
        P_on_rake,
        arc[::-1],   # reverse so we end at P_on_bot
    ])
    return poly


def point_segment_distance(px: float, py: float,
                           x0: float, y0: float,
                           x1: float, y1: float) -> float:
    """Shortest distance from (px, py) to the segment (x0,y0)-(x1,y1)."""
    vx, vy = x1 - x0, y1 - y0
    wx, wy = px - x0, py - y0
    seg_len2 = vx * vx + vy * vy
    if seg_len2 < 1e-30:
        return ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
    # Parametric position of the projection clamped to [0, 1]
    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    nx = x0 + t * vx
    ny = y0 + t * vy
    return ((px - nx) ** 2 + (py - ny) ** 2) ** 0.5
