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


def resolve_tool_translation(h_tool: float, l_tool: float, r_tool: float,
                             rake_deg: float, clear_deg: float,
                             x0_tool: float, y0_tool: float, y0_wp: float
                             ) -> tuple:
    """Compute the world-space (dx, dy) translation of the tool's LOCAL frame
    (BL = (0,0), the theoretical zero-radius corner) so that two physical
    constraints hold exactly:

      1. The fillet is TANGENT to the horizontal line y = y0_tool — i.e. the
         arc's lowest point (where its tangent is horizontal) sits exactly at
         y0_tool. This is what makes y0_tool the TRUE depth-of-cut reference;
         translating BL itself to y0_tool (the previous behaviour) places the
         wrong point there, since the fillet recedes from BL by construction.
      2. The tool's outer boundary — the fillet arc OR the straight rake face,
         whichever actually reaches that height — passes through the point
         (x0_tool, y0_wp). This is the initial-contact condition: at world
         y = y0_wp (the workpiece top surface), the tool boundary must sit
         exactly at x = x0_tool.

    Returns (dx, dy, engages, reason):
      - if engages is True: reason describes which point/face constraint 2
        was solved on: "fillet-tangent-x" (leading-edge tangent point, the
        physically correct choice when it exists and is reached — see
        below), "fillet" (boundary/arc crossing at y0_wp), or "rake face";
      - if engages is False: dx is None and reason explains why (the
        workpiece surface y0_wp is not reached by the tool at this y0_tool —
        the tool would not actually cut; the CALLER should warn the user
        rather than silently building/running an invalid configuration).

    Constraint 1 (depth) uses the fillet's HORIZONTAL-tangent point (lowest
    point of the circle) when that point actually lies on the VISIBLE arc
    (between P_on_bot and P_on_rake, i.e. the corner is rounded enough for
    that direction of tangency to physically exist on the tool). Otherwise
    the deepest point on the visible arc is one of its two endpoints.

    Constraint 2 (leading edge / initial contact) is more subtle: the point
    of the tool that material first touches, sliding in from -x, is the
    fillet's VERTICAL-tangent point (leftmost point of the circle) ONLY WHEN
    BOTH:
      (a) that direction of tangency is visible on the arc (same check as
          above, rotated 90°) — a shallow/acute corner may not expose a
          literal leftmost bulge at all;
      (b) that point, once positioned for the requested depth of cut, is
          not "floating" above the workpiece surface: its world height is
          y0_tool + r_tool, so this requires r_tool <= y0_wp - y0_tool
          (r_tool no larger than the uncut chip thickness).
    If either fails, constraint 2 falls back to the boundary/arc crossing at
    y0_wp (fillet or rake face, whichever the crossing height actually
    falls on) — this is the ONLY formula used prior to this refinement, and
    remains correct e.g. for the typical negative-rake case where the
    leftmost-point direction is not exposed by the corner at all.

    NOTE — this changes what x0_tool MEANS, even at r_tool -> 0: previously
    (dx, dy) = (x0_tool, y0_tool) placed BL itself at (x0_tool, y0_tool),
    regardless of y0_wp. Here x0_tool is instead the x-position of whichever
    of the two points above actually applies — a different quantity whenever
    y0_tool != y0_wp, even for a sharp tool (r_tool = 0, where the tangent
    checks are moot and the crossing formula is always used). Existing
    profiles will therefore see the tool's x-position shift once this is
    applied — an intended consequence of the constraints as specified, not a
    residual bug, but worth re-checking any saved profile's resulting
    geometry after this change."""
    rake = math.radians(rake_deg)
    clear = math.radians(clear_deg)
    # Reuse solve_tool_dimensions so this raises the same ToolGeometryError
    # as tool_polygon() on a singular/pathological closure system, instead of
    # duplicating the check (or silently dividing by ~0).
    h, l = solve_tool_dimensions(h_tool, l_tool, rake_deg, clear_deg)

    BLx, BLy = 0.0, 0.0
    TLx, TLy = BLx + h * math.tan(rake), BLy + h
    BRx, BRy = BLx + l, BLy + l * math.tan(clear)

    ur_x, ur_y = TLx - BLx, TLy - BLy
    ur_n = math.hypot(ur_x, ur_y); ur_x /= ur_n; ur_y /= ur_n
    ub_x, ub_y = BRx - BLx, BRy - BLy
    ub_n = math.hypot(ub_x, ub_y); ub_x /= ub_n; ub_y /= ub_n

    cos_a = max(-1.0, min(1.0, ur_x * ub_x + ur_y * ub_y))
    alpha = math.acos(cos_a)
    t = r_tool / math.tan(alpha / 2.0) if r_tool > 0 else 0.0
    Prx, Pry = BLx + t * ur_x, BLy + t * ur_y   # P_on_rake (local)
    Pbx, Pby = BLx + t * ub_x, BLy + t * ub_y   # P_on_bot  (local)

    def _in_arc(angle, a_lo, a_hi):
        while angle < a_lo - 1e-9: angle += 2.0 * math.pi
        while angle > a_hi + 1e-9: angle -= 2.0 * math.pi
        return (a_lo - 1e-9) <= angle <= (a_hi + 1e-9)

    if r_tool > 0:
        bis_x, bis_y = ur_x + ub_x, ur_y + ub_y
        bis_n = math.hypot(bis_x, bis_y); bis_x /= bis_n; bis_y /= bis_n
        Cx = BLx + (r_tool / math.sin(alpha / 2.0)) * bis_x
        Cy = BLy + (r_tool / math.sin(alpha / 2.0)) * bis_y

        # Visible arc span (matches tool_polygon's short-arc convention).
        a_bot  = math.atan2(Pby - Cy, Pbx - Cx)
        a_rake = math.atan2(Pry - Cy, Prx - Cx)
        delta = a_rake - a_bot
        while delta <= -math.pi: delta += 2.0 * math.pi
        while delta > math.pi: delta -= 2.0 * math.pi
        a_lo, a_hi = sorted((a_bot, a_bot + delta))

        down_visible = _in_arc(-math.pi / 2.0, a_lo, a_hi)
        left_visible = _in_arc(math.pi, a_lo, a_hi)

        if down_visible:
            bottom_local_y = Cy - r_tool
        else:
            # The horizontal-tangent direction isn't exposed by this corner:
            # the deepest point on the actual visible arc is one of its ends.
            bottom_local_y = min(Pby, Pry)
    else:
        # r_tool -> 0: everything degenerates to BL.
        Cx = Cy = 0.0
        left_visible = down_visible = False
        bottom_local_y = BLy

    dy = y0_tool - bottom_local_y
    y_local = y0_wp - dy

    if y_local <= bottom_local_y - 1e-12:
        return None, dy, False, ("surface above the tool's deepest point: "
                                 "the tool does not reach the workpiece — "
                                 "no cutting engagement")

    reachable = r_tool <= (y0_wp - y0_tool) + 1e-12   # r_tool <= uncut chip thickness
    if r_tool > 0 and left_visible and reachable:
        x_local = Cx - r_tool
        source = "fillet-tangent-x"
    elif r_tool > 0 and (bottom_local_y - 1e-9) <= y_local <= (Pry + 1e-9):
        dyc = y_local - Cy
        disc = r_tool * r_tool - dyc * dyc
        if disc < 0:
            return None, dy, False, "y0(wp) not reached by the fillet arc"
        x_local = Cx - math.sqrt(disc)
        source = "fillet"
    else:
        if y_local > TLy + 1e-9:
            return None, dy, False, ("y0(wp) is above the tool's rake face "
                                     "(top TL) — geometry out of range")
        x_local = y_local * math.tan(rake)
        source = "rake face"

    dx = x0_tool - x_local
    return dx, dy, True, source


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
