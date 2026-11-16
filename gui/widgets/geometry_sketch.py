# -*- coding: utf-8 -*-
"""Parameter sketch of the Geometry tab: one schematic, not to scale, that
shows where each geometry parameter is measured (tool, workpiece, Eulerian
domain, ZOI, cutting and chip lines).

The shapes are drawn with fixed, exaggerated proportions (rake angle 12°,
clearance angle 8°, large edge radius) so every dimension stays readable;
the to-scale model is the other view of the tab (GeometryPreview).
"""
from __future__ import annotations

import math

from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QVBoxLayout, QWidget

# Colours of GeometryPreview, so both views read the same.
_C_WP, _C_WP_E = "#a8d8a8", "#2e7d32"
_C_EUL, _C_EUL_E = "#cce5ff", "#1f6fb2"
_C_TOOL, _C_TOOL_E = "#f4b860", "#a8631c"
_C_ZOI = "#d33"
_C_DIM = "#333"

# Symbols shown on the sketch, also used as the labels of the form.
LABELS = {
    "h_tool": r"$h_\mathrm{tool}$", "l_tool": r"$l_\mathrm{tool}$",
    "r_tool": r"$r_\beta$", "rake": r"$\gamma$", "clear": r"$\alpha$",
    "tool_xy": r"$(x_0, y_0)$ tool",
    "h_wp": r"$h_\mathrm{wp}$", "l_wp": r"$l_\mathrm{wp}$",
    "wp_xy": r"$(x_0, y_0)$ workpiece",
    "h_void": r"$h_\mathrm{void}$", "l_void": r"$l_\mathrm{void}$",
    "eul_xy": r"$(x_0, y_0)$ Eulerian domain",
    "depth": r"$t_1$", "zoi": "ZOI",
    "cut_line": r"cutting line $y = y_0$ (tool)",
    "chip_line": r"chip line, $y_e$ above the workpiece",
}


def _dim(ax, p0, p1, text, offset=(0.0, 0.0), ha="center", va="center",
         color=_C_DIM):
    """Dimension arrow from p0 to p1 with its label at the middle (+offset)."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=color, lw=0.9,
                                shrinkA=0, shrinkB=0))
    mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
    ax.text(mx + offset[0], my + offset[1], text, ha=ha, va=va,
            fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                      alpha=0.85))


def draw_parameter_sketch(ax) -> list:
    """Draw the sketch on `ax`; return the texts written (for tests)."""
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    # Sketch frame: workpiece x in [-6, 0], y in [-3, 0]; void up to 2.4.
    l_wp, h_wp, h_void, l_void = 6.0, 3.0, 2.4, 4.2
    depth = 0.8                        # depth of cut t1
    gam, alp = math.radians(12.0), math.radians(8.0)
    H, L = 3.6, 2.4                    # tool height / length on the sketch
    ty = -depth                        # tool tip at the cutting line
    # Eulerian domain and workpiece
    ax.add_patch(Rectangle((-l_wp, -h_wp), l_wp + l_void, h_wp + h_void,
                           fc=_C_EUL, ec=_C_EUL_E, lw=1.0, alpha=0.45))
    ax.add_patch(Rectangle((-l_wp, -h_wp), l_wp, h_wp, fc=_C_WP, ec=_C_WP_E,
                           lw=1.2, alpha=0.8))
    # Tool: rake face from the tip at angle gamma to the vertical, clearance
    # face at alpha to the horizontal.
    tip = (0.0, ty)
    rake_top = (H * math.tan(gam), ty + H)
    clr_end = (L, ty + L * math.tan(alp))
    ax.add_patch(Polygon([tip, rake_top, (L, ty + H), clr_end], closed=True,
                         fc=_C_TOOL, ec=_C_TOOL_E, lw=1.3))
    r = 0.28
    ax.add_patch(Circle((r * 0.9, ty + r), r, fill=False, ec=_C_TOOL_E,
                        lw=0.8, ls=":"))
    texts = []

    def t(x, y, s, **kw):
        kw.setdefault("fontsize", 9)
        ax.text(x, y, s, **kw)
        texts.append(s)

    # Tool dimensions
    _dim(ax, (L + 0.35, ty + L * math.tan(alp)), (L + 0.35, ty + H),
         LABELS["h_tool"], offset=(0.55, 0.0)); texts.append(LABELS["h_tool"])
    _dim(ax, (H * math.tan(gam), ty + H + 0.3), (L, ty + H + 0.3),
         LABELS["l_tool"], offset=(0.0, 0.3)); texts.append(LABELS["l_tool"])
    t(-0.15, ty - 0.55, LABELS["r_tool"], color=_C_TOOL_E, ha="right",
      fontsize=11)
    ax.annotate("", xy=(0.1, ty + 0.12), xytext=(-0.2, ty - 0.4),
                arrowprops=dict(arrowstyle="->", color=_C_TOOL_E, lw=0.8))
    # Rake angle gamma: between the vertical and the rake face
    ax.plot([0, 0], [ty, ty + 2.2], color=_C_DIM, lw=0.6, ls="--")
    ax.add_patch(Arc(tip, 3.6, 3.6, theta1=90 - math.degrees(gam),
                     theta2=90, color=_C_DIM, lw=0.9))
    t(0.08, ty + 1.95, LABELS["rake"], ha="left", fontsize=11)
    # Clearance angle alpha: between the horizontal and the clearance face
    ax.plot([0, L], [ty, ty], color=_C_DIM, lw=0.6, ls="--")
    ax.add_patch(Arc(tip, 3.2, 3.2, theta1=0, theta2=math.degrees(alp),
                     color=_C_DIM, lw=0.9))
    t(1.65, ty + 0.05, LABELS["clear"], ha="left", va="bottom", fontsize=11)
    # Tool position: the tip
    ax.plot(*tip, marker="o", ms=5, mfc="white", mec=_C_TOOL_E, zorder=8)
    t(0.25, ty - 0.3, LABELS["tool_xy"], color=_C_TOOL_E, ha="left",
      va="top", fontsize=8)
    # Workpiece dimensions and position (top-right corner)
    _dim(ax, (-l_wp - 0.35, -h_wp), (-l_wp - 0.35, 0.0), LABELS["h_wp"],
         offset=(-0.55, 0.0)); texts.append(LABELS["h_wp"])
    _dim(ax, (-l_wp, -h_wp - 0.35), (0.0, -h_wp - 0.35), LABELS["l_wp"],
         offset=(0.0, -0.3)); texts.append(LABELS["l_wp"])
    ax.plot(0, 0, marker="s", ms=5, mfc="white", mec=_C_WP_E, zorder=8)
    t(-0.15, 0.12, LABELS["wp_xy"] + "\n" + LABELS["eul_xy"], color=_C_WP_E,
      ha="right", va="bottom", fontsize=8)
    # Void (Eulerian domain without material)
    _dim(ax, (-l_wp - 0.35, 0.0), (-l_wp - 0.35, h_void), LABELS["h_void"],
         offset=(-0.6, 0.0), color=_C_EUL_E); texts.append(LABELS["h_void"])
    _dim(ax, (0.0, -h_wp - 0.35), (l_void, -h_wp - 0.35),
         LABELS["l_void"], offset=(0.0, -0.3), color=_C_EUL_E)
    texts.append(LABELS["l_void"])
    # Depth of cut t1, cutting line and chip line
    _dim(ax, (-0.55, ty), (-0.55, 0.0), LABELS["depth"], offset=(-0.3, 0.0))
    texts.append(LABELS["depth"])
    ax.plot([-l_wp, l_void], [ty, ty], color=_C_TOOL_E, lw=0.9, ls="-.")
    t(-l_wp + 0.1, ty - 0.08, LABELS["cut_line"], color=_C_TOOL_E,
      va="top", fontsize=8)
    ax.plot([-l_wp, 0.0], [depth, depth], color=_C_ZOI, lw=1.0, ls=":")
    t(-l_wp + 0.1, depth + 0.08, LABELS["chip_line"], color=_C_ZOI,
      va="bottom", fontsize=8)
    _dim(ax, (-2.0, 0.0), (-2.0, depth), r"$y_e$", offset=(0.3, 0.0),
         color=_C_ZOI)
    # ZOI
    ax.add_patch(Rectangle((-3.2, -1.8), 4.4, 3.6, fill=False, ec=_C_ZOI,
                           lw=1.2, ls="--"))
    t(-3.1, 1.65, LABELS["zoi"], color=_C_ZOI, va="top")
    ax.set_xlim(-l_wp - 1.6, l_void + 1.2)
    ax.set_ylim(-h_wp - 1.0, h_void + 1.0)
    ax.set_title("Parameters (not to scale)", fontsize=10)
    return texts


class GeometrySketch(QWidget):
    """The parameter sketch in a Qt widget (drawn once)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.texts = draw_parameter_sketch(self.ax)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
