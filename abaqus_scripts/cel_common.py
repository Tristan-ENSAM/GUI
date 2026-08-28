# -*- coding: utf-8 -*-
"""
Pure-Python helpers SHARED between the Abaqus generator script
(``run_simul.py``, executed by Abaqus Python 2.7) and the GUI package
(Python 3.x).

WHY THIS MODULE EXISTS
----------------------
``resolve_tool_translation`` used to be duplicated: one copy in
``run_simul.py`` and one in ``gui/core/tool_geometry_calc.py``, kept in sync
by hand. Any divergence between the two would silently place the tool
differently in the preview and in the actual Abaqus model — exactly the class
of bug that is hardest to notice. There is now a single implementation, here.

HARD CONSTRAINT — PYTHON 2.7 COMPATIBILITY
------------------------------------------
This file is imported by Abaqus Python 2.7. It must therefore avoid:
  * f-strings,
  * type annotations (PEP 484/526),
  * ``from __future__ import annotations``,
  * pathlib, dataclasses, and any 3.x-only stdlib.
Only ``math`` is used. Do NOT add numpy here: keeping this module
dependency-free is what makes it safe to import from both interpreters.
"""
import math


class ToolGeometryError(ValueError):
    """Raised when the tool inputs cannot form a valid closed outline."""
    pass


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------
def cfg_get(cfg, path, default=None):
    """Read a nested key from a config dict, e.g. "geometry.tool.position.x0".

    Returns `default` if any level is missing or is not a dict."""
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_mapping(mapping, value, parameter_name):
    """Translate a GUI string value into its Abaqus constant, STRICTLY.

    A typo in a hand-edited profile (e.g. "penality" instead of "penalty")
    must NOT be silently replaced by a default and change the physics: it
    raises here, before the model is built, with the list of valid values."""
    if value not in mapping:
        raise ValueError(
            "%s=%r is not a valid value. Expected one of: %s"
            % (parameter_name, value, sorted(mapping.keys())))
    return mapping[value]


# ---------------------------------------------------------------------------
# Tool geometry
# ---------------------------------------------------------------------------
def solve_tool_dimensions(h_tool, l_tool, rake_deg, clear_deg):
    """Compute the overall (h, l) bounding-box dimensions of the tool from
    the four GUI inputs.

    The closure system is (I - B) (h, l)^T = (h_tool, l_tool)^T with
        B = [[0,         tan(clear)],
             [tan(rake), 0         ]],
    i.e. it satisfies h - tan(clear)*l = h_tool and l - tan(rake)*h = l_tool.

    Raises ToolGeometryError if rake + clear is too close to 90 deg (or
    beyond), which makes the trapezoidal closure impossible."""
    rake = math.radians(rake_deg)
    clear = math.radians(clear_deg)

    # det(I - B) = 1 - tan(rake) * tan(clear). Refuse a singular or
    # near-singular system; 1e-6 corresponds to rake + clear within ~1e-3 rad
    # of 90 deg, already a deeply pathological tool.
    det = 1.0 - math.tan(rake) * math.tan(clear)
    if det <= 1e-6:
        raise ToolGeometryError(
            "Invalid tool geometry: rake=%.2f deg and clear=%.2f deg give "
            "tan(rake)*tan(clear) >= 1 - the rake and clearance faces cannot "
            "close the tool outline. Reduce rake or clear (typical: "
            "rake <= 30 deg, clear <= 15 deg)." % (rake_deg, clear_deg))

    h = (h_tool + math.tan(clear) * l_tool) / det
    l = (math.tan(rake) * h_tool + l_tool) / det

    if h <= 0.0 or l <= 0.0:
        raise ToolGeometryError(
            "Invalid tool geometry: solved (h=%.4g, l=%.4g) is non-positive. "
            "Check that h_tool and l_tool are positive and that rake/clear "
            "are within sensible bounds." % (h, l))
    return h, l


def resolve_tool_translation(h_tool, l_tool, r_tool, rake_deg, clear_deg,
                             x0_tool, y0_tool, y0_wp):
    """World-space (dx, dy) translation of the tool's LOCAL frame (BL, the
    theoretical zero-radius corner) so that two physical constraints hold:

      1. DEPTH: the fillet's horizontal-tangent point (lowest point of the
         circle) sits at y = y0_tool, when that direction of tangency is
         exposed by the corner; otherwise the deepest point of the visible
         arc is one of its endpoints.
      2. LEADING EDGE: the point material first touches, sliding in from -x,
         sits at x = x0_tool. That is the fillet's vertical-tangent
         (leftmost) point ONLY WHEN both (a) that direction of tangency is
         visible on the arc and (b) it is not floating above the workpiece
         surface, which requires r_tool <= y0_wp - y0_tool (radius no larger
         than the uncut chip thickness). Otherwise it falls back to where the
         tool boundary (fillet arc or rake face) crosses y = y0_wp.

    Returns (dx, dy, engages, reason):
      - engages True: reason is "fillet-tangent-x", "fillet" or "rake face";
      - engages False: dx is None and reason explains why the tool does not
        reach the workpiece. Callers MUST surface this rather than silently
        building a non-cutting configuration.

    NOTE - x0_tool's MEANING changed when this replaced the old
    (dx, dy) = (x0_tool, y0_tool) convention: it is now the x of whichever
    reference point above applies, not the x of the BL corner. The two only
    coincide at zero depth of cut."""
    rake = math.radians(rake_deg)
    clear = math.radians(clear_deg)
    # Reuse solve_tool_dimensions so this raises the same ToolGeometryError
    # on a singular/pathological closure system.
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
        while angle < a_lo - 1e-9:
            angle += 2.0 * math.pi
        while angle > a_hi + 1e-9:
            angle -= 2.0 * math.pi
        return (a_lo - 1e-9) <= angle <= (a_hi + 1e-9)

    if r_tool > 0:
        bis_x, bis_y = ur_x + ub_x, ur_y + ub_y
        bis_n = math.hypot(bis_x, bis_y); bis_x /= bis_n; bis_y /= bis_n
        Cx = BLx + (r_tool / math.sin(alpha / 2.0)) * bis_x
        Cy = BLy + (r_tool / math.sin(alpha / 2.0)) * bis_y

        # Visible arc span (matches tool_polygon's short-arc convention).
        a_bot = math.atan2(Pby - Cy, Pbx - Cx)
        a_rake = math.atan2(Pry - Cy, Prx - Cx)
        delta = a_rake - a_bot
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        while delta > math.pi:
            delta -= 2.0 * math.pi
        a_lo, a_hi = (a_bot, a_bot + delta) if a_bot <= a_bot + delta \
            else (a_bot + delta, a_bot)

        down_visible = _in_arc(-math.pi / 2.0, a_lo, a_hi)
        left_visible = _in_arc(math.pi, a_lo, a_hi)

        if down_visible:
            bottom_local_y = Cy - r_tool
        else:
            bottom_local_y = min(Pby, Pry)
    else:
        Cx = Cy = 0.0
        left_visible = down_visible = False
        bottom_local_y = BLy

    dy = y0_tool - bottom_local_y
    y_local = y0_wp - dy

    if y_local <= bottom_local_y - 1e-12:
        return None, dy, False, ("the tool's deepest point does not reach the "
                                 "workpiece surface: no cutting engagement")

    reachable = r_tool <= (y0_wp - y0_tool) + 1e-12
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
                                     "(top TL) - geometry out of range")
        x_local = y_local * math.tan(rake)
        source = "rake face"

    dx = x0_tool - x_local
    return dx, dy, True, source


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------
def discretize(dim, element_size):
    """Round `dim` DOWN to the nearest multiple of `element_size` (floor).

    Uses Decimal to avoid IEEE-754 artefacts (e.g. 0.3 // 0.1 == 2.0 in
    binary floating point)."""
    from decimal import Decimal
    d = Decimal(str(dim))
    es = Decimal(str(element_size))
    if es <= 0:
        raise ValueError(
            "element_size must be > 0, got: {0}".format(element_size))
    n = d // es
    if n <= 0:
        raise ValueError(
            "discretize({0}, {1}) -> {2} elements: dim too small relative to "
            "element_size".format(dim, element_size, int(n)))
    return float(n * es)
