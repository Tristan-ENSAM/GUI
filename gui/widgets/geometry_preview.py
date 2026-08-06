# -*- coding: utf-8 -*-
"""
Matplotlib-based 2D preview of the cutting model geometry.

Reproduces what abq_odb_generator.py builds, viewed in the XY plane:
  - Tool: rectangle (h_tool x l_tool) with a fillet of radius r_tool between
    the rake face and the clearance face, rotated by rake/clear angles.
    NOTE: in your script the angles are encoded via the AngularDimensions
    (90 + clear_angle on l3-l4, 90 + rake_angle on l1-l2). The result is a
    quadrilateral with a filleted corner at the cutting edge — we
    approximate it the same way here.
  - Workpiece: rectangle (-l_wp, -h_wp) -> (0, 0), then translated so its
    right face touches the tool's left face (mimicking `translateTo`).
  - Eulerian domain: rectangle (-l_wp, -h_wp) -> (l_void, h_void).
  - ROI / bbox: dashed rectangle.
"""
from __future__ import annotations
import math
import logging
import numpy as np

from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal

from gui.core.model_config import ModelConfig
from gui.core.logging_util import log_swallowed


# ---------------------------------------------------------------------------
# Custom legend handler: draws a triangular arrow next to the label
# ---------------------------------------------------------------------------
class _ArrowLegendHandler(HandlerBase):
    """Draw a horizontal arrow in the legend handle box.

    The legend usually shows a short line for `Line2D` artists. For
    velocity / Eulerian BC entries we want a TRIANGULAR ARROW instead,
    because the data in the plot is drawn with arrows and the legend
    should match.

    Used through `handler_map` like:
        ax.legend(handler_map={proxy_artist: _ArrowLegendHandler()})

    The proxy artist is just an invisible `Line2D` placeholder created
    via `ax.plot([], [], ..., label=...)`; its color and linestyle are
    forwarded to the arrow so the legend matches the plot.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        # The colour and linestyle of the placeholder Line2D drive the
        # arrow's appearance.
        color = orig_handle.get_color()
        ls    = orig_handle.get_linestyle()
        lw    = orig_handle.get_linewidth()
        # Arrow goes from left to right in the legend box, centred vertically.
        # `xdescent` is generally 0; we still subtract it for safety.
        y = height / 2.0 - ydescent
        x0 = -xdescent
        x1 = -xdescent + width
        arrow = FancyArrowPatch(
            (x0, y), (x1, y),
            arrowstyle="-|>", mutation_scale=12,
            color=color, linewidth=lw, linestyle=ls,
            shrinkA=0, shrinkB=0,
            transform=trans,
        )
        return [arrow]


# A sentinel attribute name. Any Line2D with this attribute set to True
# will get the arrow handler when the legend is rebuilt.
_ARROW_FLAG = "_gui_legend_arrow"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
# These pure geometry calculations live in gui.core.tool_geometry_calc so
# they can be unit-tested without Qt/matplotlib. Re-imported here under the
# previous private names for backward compatibility.
from gui.core.tool_geometry_calc import (
    ToolGeometryError,
    solve_tool_dimensions as _solve_tool_dimensions,
    tool_polygon as _tool_polygon,
    point_segment_distance as _point_segment_distance_calc,
    resolve_tool_translation as _resolve_tool_translation,
)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class GeometryPreview(QWidget):
    """Embeds a Matplotlib FigureCanvas that draws the model.

    What is displayed is driven entirely by `cfg.analysis.formulation`:
      - CEL:        Eulerian domain (blue) + workpiece reference (green)
                    + tool (orange) + ROI (dashed)
      - Lagrangian: workpiece (green, meshed body) + tool (orange) +
                    Tool RP marker + ROI (dashed)

    The optional mesh overlay (toggled from the Mesh seeds group of the
    GeometryTab via update_from_config's `show_mesh` arg) is drawn over
    the active deformable domain: the Eulerian box in CEL mode, the
    workpiece rectangle in Lagrangian mode.

    Interactive picking
    -------------------
    When `picking_enabled` is True, clicking near a "pickable" face emits
    `facePicked(face_id)` with one of:
      - "work_bot", "work_top", "work_left", "work_right" : workpiece edges
      - "eul_left", "eul_right"                          : Eulerian side faces
    The list of currently-pickable segments is rebuilt on every redraw
    via `update_from_config(... pickable=True)`. We use a click-distance
    threshold in DATA coordinates so the threshold scales correctly with
    zoom level.
    """

    # Signal emitted on a click within `_PICK_TOLERANCE_FRACTION` of the
    # axes' current x-span from a pickable face. The argument is the face
    # identifier string (see class docstring).
    facePicked = Signal(str)

    # Tolerance as a fraction of the current x-span: 2 % gives a generous
    # click target without making nearby faces ambiguous.
    _PICK_TOLERANCE_FRACTION = 0.02

    # Mesh-overlay safety cap: above this many lines, we skip the mesh
    # to keep the preview snappy.
    MAX_MESH_LINES = 5000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg = None  # last config received, kept for external redraws
        self.picking_enabled: bool = False
        # Reference-image watermark (set from the Alignment tab). Drawn behind
        # the sketch, centred on the model origin at the alignment scale, so
        # the experimental tool/workpiece can be compared with the model.
        self._ref_image = None
        self._ref_mm_per_px = None
        self._ref_show = False
        self._ref_alpha = 0.45
        # Toggle for the BCs/ICs tab preview: which class of overlay to show.
        # "BC": cutting velocity + Eulerian inflow/outflow + encastrement
        # "IC": initial Eulerian velocity field + initial temperature label
        # Ignored unless `show_bcs=True` on update_from_config.
        self.bc_view_mode: str = "BC"
        # List of (face_id, (x0,y0,x1,y1)) tuples populated by _draw_bcs.
        # Used by _on_click to resolve clicks to face IDs.
        self._pickable_segments: list[tuple[str, tuple[float, float, float, float]]] = []

        # --- matplotlib canvas ---
        self._fig = Figure(figsize=(6, 5), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        self._mpl_toolbar = NavigationToolbar(self._canvas, self)

        # Override the toolbar's Home action so the house icon always
        # re-fits the view to the current model (instead of restoring the
        # axis limits captured the very first time the figure was drawn,
        # which become stale as soon as we redraw on parameter changes).
        self._install_home_override()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._mpl_toolbar)
        lay.addWidget(self._canvas)

        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.grid(True, linestyle=":", alpha=0.4)
        self._ax.set_xlabel("x [mm]")
        self._ax.set_ylabel("y [mm]")

        # Wheel-to-zoom: scrolling the mouse wheel over the axes zooms in/out
        # around the cursor position.
        self._canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        # Click-to-pick: only the BCs tab enables this, via picking_enabled
        self._canvas.mpl_connect("button_release_event", self._on_click_pick)

    def _install_home_override(self):
        """Re-route the toolbar Home button to our own 'fit to model' logic.

        The toolbar Home action normally pops the view stack back to the
        initial state. Our preview is regenerated on every parameter
        change (`_ax.clear()` + `set_xlim/ylim`), so the stack accumulates
        intermediate states that aren't useful. We replace the action with
        a recompute-and-refit each time."""
        # Find the home QAction. In PySide6/matplotlib it's the first item
        # of toolbar.actions() that triggers `self.home()`.
        for action in self._mpl_toolbar.actions():
            # Disconnect every slot bound to this action's `triggered`
            # signal — typically just `NavigationToolbar.home` — and rebind
            # to our handler. Identification by text is locale-fragile, so
            # we use the action's icon text / object name when possible.
            txt = (action.text() or "").lower()
            tip = (action.toolTip() or "").lower()
            if "home" in txt or "reset" in tip or "original" in tip:
                try:
                    action.triggered.disconnect()
                except (TypeError, RuntimeError):
                    pass  # nothing was connected (unlikely but harmless)
                action.triggered.connect(self.fit_view)
                return

    # ----- public API -----
    def fit_view(self):
        """Recompute optimal axis limits for the current cfg and apply them.
        Called by the Home toolbar button and at the end of update_from_config."""
        if self._cfg is None:
            return
        xlim, ylim = self._compute_fit_limits(self._cfg)
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)
        self._canvas.draw_idle()

    def _compute_fit_limits(self, cfg: ModelConfig):
        """Return (xlim, ylim) tuples that fit all visible shapes with a
        10% margin. Same logic that update_from_config uses on each redraw."""
        is_lagrangian = (cfg.analysis.formulation == "Lagrangian")
        show_euler = not is_lagrangian

        h_wp_eff, h_void_eff, l_wp_eff, l_void_eff = cfg.effective_euler_dims()
        eul_x = -l_wp_eff + cfg.euler_position.x0
        eul_y = -h_wp_eff + cfg.euler_position.y0
        eul_w = l_wp_eff + l_void_eff
        eul_h = h_wp_eff + h_void_eff

        wp_x = -l_wp_eff + cfg.wp_position.x0
        wp_y = -h_wp_eff + cfg.wp_position.y0

        g = cfg.tool_geometry
        # The tool geometry may be invalid (e.g. rake + clear ≥ 90°). In
        # that case we just compute the fit from the other shapes — the
        # main update_from_config path will already have refused to draw
        # the tool and displayed a message in its place.
        try:
            tool_local = _tool_polygon(
                h_tool=g.h_tool, l_tool=g.l_tool, r_tool=g.r_tool,
                rake_deg=g.rake_angle, clear_deg=g.clear_angle,
            )
            dx, dy, _engages, _reason = self._tool_offset(cfg)
            tool_world = tool_local + np.array([dx, dy])
            xs = [wp_x, wp_x + l_wp_eff,
                  float(tool_world[:, 0].min()), float(tool_world[:, 0].max())]
            ys = [wp_y, wp_y + h_wp_eff,
                  float(tool_world[:, 1].min()), float(tool_world[:, 1].max())]
        except ToolGeometryError:
            # No tool contribution to the fit; fall back to the other shapes
            xs = [wp_x, wp_x + l_wp_eff]
            ys = [wp_y, wp_y + h_wp_eff]
        if show_euler:
            xs += [eul_x, eul_x + eul_w]
            ys += [eul_y, eul_y + eul_h]
        pad = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)

    def _on_scroll_zoom(self, event):
        """Zoom the matplotlib axes around the cursor position when the user
        scrolls the wheel inside the axes. One notch zooms by 20%."""
        if event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Scaling factor per scroll notch (negative `step` = zoom out)
        if event.button == "up":
            scale = 1.0 / 1.20
        elif event.button == "down":
            scale = 1.20
        else:
            return

        cur_xlim = self._ax.get_xlim()
        cur_ylim = self._ax.get_ylim()
        cx, cy = event.xdata, event.ydata

        # New limits: keep the point under the cursor stationary, scale the
        # span around it.
        new_xspan = (cur_xlim[1] - cur_xlim[0]) * scale
        new_yspan = (cur_ylim[1] - cur_ylim[0]) * scale
        # Position of cursor as a fraction of the current span:
        fx = (cx - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        fy = (cy - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        new_xlim = (cx - fx * new_xspan, cx + (1 - fx) * new_xspan)
        new_ylim = (cy - fy * new_yspan, cy + (1 - fy) * new_yspan)
        self._ax.set_xlim(new_xlim)
        self._ax.set_ylim(new_ylim)
        self._canvas.draw_idle()

    def _on_click_pick(self, event):
        """When picking is enabled, resolve a click to the nearest pickable
        face (if within tolerance) and emit `facePicked(face_id)`.

        We only react to left-button clicks (button == 1) in the axes,
        and only when no Matplotlib toolbar mode is active (pan / zoom
        would otherwise also fire button_release_event).
        """
        if not self.picking_enabled:
            return
        if event.inaxes is not self._ax:
            return
        if event.button != 1:
            return
        # Skip if a toolbar mode is active (pan/zoom)
        if self._mpl_toolbar.mode != "":
            return
        if event.xdata is None or event.ydata is None:
            return

        cur_xlim = self._ax.get_xlim()
        tol = self._PICK_TOLERANCE_FRACTION * (cur_xlim[1] - cur_xlim[0])

        best_face = None
        best_dist = tol  # only accept distances <= tol

        for face_id, (x0, y0, x1, y1) in self._pickable_segments:
            d = self._point_segment_distance(
                event.xdata, event.ydata, x0, y0, x1, y1,
            )
            if d < best_dist:
                best_dist = d
                best_face = face_id

        if best_face is not None:
            self.facePicked.emit(best_face)

    @staticmethod
    def _point_segment_distance(px, py, x0, y0, x1, y1) -> float:
        """Shortest distance from (px, py) to the segment (x0,y0)-(x1,y1).
        Delegates to gui.core.tool_geometry_calc.point_segment_distance."""
        return _point_segment_distance_calc(px, py, x0, y0, x1, y1)

    # ----- public API -----
    def set_reference_overlay(self, image, mm_per_px, show=True):
        """Set the reference image (numpy array) and its scale (mm/px) to be
        drawn as a watermark, centred on the model origin. Redraws if a config
        is already loaded."""
        self._ref_image = None if image is None else np.asarray(image)
        self._ref_mm_per_px = float(mm_per_px) if mm_per_px else None
        self._ref_show = bool(show)
        if self._cfg is not None:
            self.update_from_config(self._cfg)

    def set_reference_visible(self, on: bool):
        self._ref_show = bool(on)
        if self._cfg is not None:
            self.update_from_config(self._cfg)

    def has_reference_overlay(self) -> bool:
        return self._ref_image is not None and bool(self._ref_mm_per_px)

    def _draw_reference_overlay(self):
        """imshow the reference image behind the sketch. The image is centred
        on the model origin (matching the Alignment convention: origin at
        image centre, x right, y up) and scaled by mm/px, so the tool tip in
        the image lands at the model tool position."""
        if not self._ref_show or self._ref_image is None or not self._ref_mm_per_px:
            return
        img = self._ref_image
        h, w = img.shape[0], img.shape[1]
        s = self._ref_mm_per_px
        if not np.isfinite(s) or s <= 0:
            return
        extent = (-w / 2.0 * s, w / 2.0 * s, -h / 2.0 * s, h / 2.0 * s)
        is_rgb = (img.ndim == 3 and img.shape[-1] in (3, 4))
        try:
            self._ax.imshow(img, extent=extent, origin="upper",
                            cmap=None if is_rgb else "gray",
                            alpha=self._ref_alpha, zorder=-10)
            # imshow would otherwise switch the axes to its own aspect; keep
            # the equal x/y scaling so the geometry is not distorted.
            self._ax.set_aspect("equal", adjustable="datalim")
        except Exception:
            log_swallowed("drawing the reference image overlay",
                          level=logging.DEBUG)

    def update_from_config(self, cfg: ModelConfig,
                            show_mesh: bool = False,
                            show_bcs: bool = False) -> None:
        self._cfg = cfg
        # Reset pickable segments on every redraw — they're rebuilt below.
        self._pickable_segments = []
        self._ax.clear()
        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.grid(True, linestyle=":", alpha=0.4)
        self._ax.set_xlabel("x [mm]")
        self._ax.set_ylabel("y [mm]")

        # Reference-image watermark first, so it sits behind everything.
        self._draw_reference_overlay()

        is_lagrangian = (cfg.analysis.formulation == "Lagrangian")
        # In CEL mode we draw the Eulerian background domain; in Lagrangian
        # mode we don't (the workpiece IS the deformable body).
        show_euler = not is_lagrangian

        # --- Eulerian domain (CEL only) ---
        h_wp_eff, h_void_eff, l_wp_eff, l_void_eff = cfg.effective_euler_dims()
        eul_x = -l_wp_eff + cfg.euler_position.x0
        eul_y = -h_wp_eff + cfg.euler_position.y0
        eul_w = l_wp_eff + l_void_eff
        eul_h = h_wp_eff + h_void_eff

        if show_euler:
            eul = Rectangle((eul_x, eul_y), eul_w, eul_h,
                            facecolor="#cce5ff", edgecolor="#1f6fb2",
                            linewidth=1.2, alpha=0.55, label="Eulerian domain")
            self._ax.add_patch(eul)
            # In CEL the mesh covers the full Eulerian domain (workpiece + void)
            if show_mesh and cfg.elem_size > 0:
                self._draw_mesh(eul_x, eul_y, eul_w, eul_h, cfg.elem_size,
                                color="#1f6fb2")

        # --- Workpiece ---
        wp_x = -l_wp_eff + cfg.wp_position.x0
        wp_y = -h_wp_eff + cfg.wp_position.y0
        wp_label = "Workpiece" if is_lagrangian else "Workpiece (reference)"
        wp = Rectangle((wp_x, wp_y), l_wp_eff, h_wp_eff,
                       facecolor="#a8d8a8", edgecolor="#2e7d32",
                       linewidth=1.2, alpha=0.75, label=wp_label)
        self._ax.add_patch(wp)

        # In Lagrangian mode the workpiece is the deformable body, so the
        # mesh overlay goes here.
        if is_lagrangian and show_mesh and cfg.elem_size > 0:
            self._draw_mesh(wp_x, wp_y, l_wp_eff, h_wp_eff, cfg.elem_size,
                            color="#2e7d32")

        # --- Tool ---
        # The tool outline depends on a closure system that can be ill-
        # conditioned for extreme rake/clear values. We catch the error
        # so the rest of the scene (Eulerian, workpiece, BCs) still
        # renders, and we display a friendly message in place of the tool.
        g = cfg.tool_geometry
        tool_world = None        # remains None if the geometry is invalid
        try:
            tool_local = _tool_polygon(
                h_tool=g.h_tool, l_tool=g.l_tool, r_tool=g.r_tool,
                rake_deg=g.rake_angle, clear_deg=g.clear_angle,
            )
            dx, dy, engages, reason = self._tool_offset(cfg)
            tool_world = tool_local + np.array([dx, dy])
            tool = Polygon(tool_world, closed=True,
                           facecolor="#f4b860", edgecolor="#a8631c",
                           linewidth=1.4, alpha=0.9, label="Tool")
            self._ax.add_patch(tool)
            # Tool SEED preview (edge seeds coloured by size). Only the boundary
            # seeding is shown — the real 2D elements are generated by Abaqus
            # (SWEEP on the fillet) and cannot be reproduced here.
            if show_mesh:
                self._draw_tool_seeds(tool_local, g, cfg)
            if not engages:
                # The tool's geometry is valid but it does not reach the
                # workpiece surface: no cutting engagement. run_simul.py
                # refuses to build this model — warn here too rather than
                # silently drawing a non-cutting configuration.
                # Anchored to the AXES corner (transAxes, not data
                # coordinates) so it stays visible regardless of where the
                # tool/workpiece sit or how far the user pans/zooms.
                self._ax.text(
                    0.02, 0.02,
                    "⚠ Tool does not engage the workpiece\n(%s)" % reason,
                    transform=self._ax.transAxes, ha="left", va="bottom",
                    fontsize=8, color="#8a1f11",
                    bbox=dict(boxstyle="round", facecolor="#fff3cd",
                             edgecolor="#8a1f11", alpha=0.95),
                    zorder=20)
        except ToolGeometryError as e:
            # Big in-axes message so the user knows why the tool is missing.
            # The error message can be long; matplotlib's text wrapping
            # keeps it readable within the axes regardless of zoom.
            txt = self._ax.text(
                0.5, 0.5,
                "Invalid tool geometry\n" + str(e),
                transform=self._ax.transAxes,
                ha="center", va="center",
                fontsize=10, color="#a8631c",
                wrap=True,
                bbox=dict(facecolor="#fff4e6", edgecolor="#a8631c",
                          boxstyle="round,pad=0.6"),
            )
            # Force the message to wrap to a sensible width relative to
            # the axes (set _get_wrap_line_width to a fixed fraction of
            # the axis width — matplotlib uses this internally when
            # wrap=True is set).
            try:
                txt._get_wrap_line_width = lambda: 380.0
            except Exception:
                log_swallowed("setting the annotation wrap width",
                              level=logging.DEBUG)

        # --- Tool RP marker ---
        # Always shown, in both formulations: the Reference Point is where
        # the velocity / encastrement BC will be applied in the Abaqus model.
        #   - CEL:        RP is hard-coded to TR (see tool_instance.vertices[4]
        #                 in abq_odb_generator.py).
        #   - Lagrangian: RP is the corner the user picked in the Analysis tab.
        rp_loc = "TR" if not is_lagrangian else cfg.analysis.rp_location
        rp_world = None    # remains None when the tool geometry is invalid
        # Only place the RP if we have a valid tool (otherwise no anchor
        # point makes sense). _tool_rp_world_position returns None when
        # the closure system is invalid — we double-belt that here.
        if tool_world is not None:
            rp_world = self._tool_rp_world_position(cfg, tool_world, rp_loc)
            if rp_world is not None:
                self._ax.plot(rp_world[0], rp_world[1], marker="o",
                              markerfacecolor="white",
                              markeredgecolor="#a8631c",
                              markersize=8, markeredgewidth=1.5,
                              label=f"Tool RP ({rp_loc})")

        # --- BBox / ROI (dashed) ---
        bb = cfg.bbox
        if (bb.xmax > bb.xmin) and (bb.ymax > bb.ymin):
            roi = Rectangle((bb.xmin, bb.ymin),
                            bb.xmax - bb.xmin, bb.ymax - bb.ymin,
                            facecolor="none", edgecolor="#d33",
                            linewidth=1.0, linestyle="--", label="ROI / bbox")
            self._ax.add_patch(roi)

        # --- Boundary conditions overlay (optional) ---
        # BC overlays that depend on the tool outline (temperature hatching
        # on the tool, encastrement marker, etc.) are skipped when the
        # tool geometry is invalid. The Eulerian-domain BCs (inflow/outflow
        # arrows, velocity on faces) still render.
        if show_bcs:
            self._draw_bcs(cfg, tool_world, rp_world,
                           eul_x, eul_y, eul_w, eul_h,
                           wp_x, wp_y, l_wp_eff, h_wp_eff,
                           is_lagrangian)

        # --- Origin marker ---
        self._ax.plot(0, 0, marker="+", color="black", markersize=10)

        # --- Axis limits: fit all visible shapes with a 10% margin ---
        xlim, ylim = self._compute_fit_limits(cfg)
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

        # Build handler_map so any Line2D placeholder that was flagged as
        # an "arrow" entry (velocities, Eulerian BC types) is rendered as
        # a triangular arrow in the legend box, matching the plot.
        handler_map = {}
        for child in self._ax.get_lines():
            if getattr(child, _ARROW_FLAG, False):
                handler_map[child] = _ArrowLegendHandler()
        self._ax.legend(loc="upper right", fontsize=8, framealpha=0.85,
                         handler_map=handler_map)
        self._canvas.draw_idle()

    # ----- helpers -----
    @staticmethod
    def _tool_offset(cfg: ModelConfig):
        """World-space (dx, dy) translation for the tool's LOCAL polygon
        (BL-anchored, as built by _tool_polygon), matching exactly what
        run_simul.py applies via _resolve_tool_translation — so the preview
        and the Abaqus model always agree.

        Returns (dx, dy, engages, reason):
          - engages=True  -> dx/dy satisfy both the fillet-tangency (depth of
            cut) and boundary-coincidence (initial contact) constraints.
          - engages=False -> the tool does not reach the workpiece surface at
            all; dx falls back to the raw cfg.tool_position.x0 (there is no
            well-defined "correct" x in this case) purely so something
            sensible can still be drawn, and the caller must warn the user.
            `dy` is always the depth-correct value (independent of engagement).

        Raises ToolGeometryError if the underlying tool geometry itself is
        invalid (singular closure system) — same condition _tool_polygon
        raises on, left to the caller's existing try/except."""
        g = cfg.tool_geometry
        dx, dy, engages, reason = _resolve_tool_translation(
            g.h_tool, g.l_tool, g.r_tool, g.rake_angle, g.clear_angle,
            cfg.tool_position.x0, cfg.tool_position.y0, cfg.wp_position.y0)
        if not engages:
            dx = cfg.tool_position.x0
        return dx, dy, engages, reason

    @staticmethod
    def _tool_rp_world_position(cfg: ModelConfig, tool_world: np.ndarray,
                                 rp_location: str | None = None):
        """Return the world-frame (x, y) of the tool Reference Point.

        `tool_world` is the polygon vertices already translated to
        cfg.tool_position.
        `rp_location` overrides cfg.analysis.rp_location when supplied —
        used by the preview to force TR in CEL mode regardless of the
        Lagrangian-only setting in the Analysis tab.

        Returns None if the location is unknown or if the tool geometry
        is invalid (in which case `tool_world` will not have been built
        either and the caller already handles the error)."""
        g = cfg.tool_geometry
        tx, ty, _engages, _reason = GeometryPreview._tool_offset(cfg)
        loc = rp_location if rp_location is not None else cfg.analysis.rp_location
        # Resolve the true (h, l) of the tool outline — these are NOT the
        # raw GUI inputs h_tool / l_tool, they are solved from the closure
        # system (see _solve_tool_dimensions). The TR / BR corners are at
        # (l, h) / (l, l*tan(clear)) in the local frame.
        try:
            h, l = _solve_tool_dimensions(
                g.h_tool, g.l_tool, g.rake_angle, g.clear_angle
            )
        except ToolGeometryError:
            return None
        import math as _m
        if loc == "TR":
            return np.array([l + tx, h + ty])
        elif loc == "BR":
            return np.array([l + tx,
                             l * _m.tan(_m.radians(g.clear_angle)) + ty])
        elif loc == "centroid":
            return np.array([tool_world[:, 0].mean(), tool_world[:, 1].mean()])
        return None

    def _draw_tool_seeds(self, tool_local, g, cfg) -> None:
        """Overlay the tool EDGE seeding on the tool outline, coloured by size.

        Mirrors run_simul's bias seeding exactly, in terms of the three GUI
        sizes (nose/min, inter, max):
          - rake + clearance faces + fillet: bias from `tool_elem_size` (nose)
            at the cutting edge to `inter_elem_size` at the far end of each face;
          - top + right border faces: bias from `inter_elem_size` (junction) to
            `max_elem_size` at the outer corner.
        Only the boundary SEEDS are drawn (small dots); the real 2D elements
        (SWEEP on the fillet) are produced by Abaqus and are NOT reproduced.

        `tool_local` is the outline from tool_polygon(): [P_on_bot, BR, TR, TL,
        P_on_rake, arc_reversed...]. The arc is the nose; segments TL->P_on_rake
        and P_on_bot->BR are the rake/clearance faces; BR->TR and TR->TL are the
        borders."""
        try:
            n_fillet = 24                       # must match tool_polygon default
            arc = tool_local[4 + 1:]            # after P_on_rake -> the fillet
            P_on_bot  = tool_local[0]
            BR, TR, TL = tool_local[1], tool_local[2], tool_local[3]
            P_on_rake = tool_local[4]
        except Exception:
            return
        # Same corrected offset as the main draw; engages/reason are ignored
        # here since the main draw path already shows the warning if needed.
        off = np.array(self._tool_offset(cfg)[:2])

        nose  = float(getattr(cfg, "tool_elem_size", 0.001))
        inter = float(getattr(cfg, "inter_elem_size", 0.02))
        cmax  = float(getattr(cfg, "max_elem_size", 0.05))
        if not (nose > 0 and inter > 0 and cmax > 0):
            return

        def geometric_seeds(p0, p1, s_start, s_end):
            """Seed points along p0->p1 whose spacing grades from s_start (at p0)
            to s_end (at p1), like seedEdgeByBias(biasMethod=SINGLE). Works in
            both directions: s_start may be larger OR smaller than s_end."""
            p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
            L = float(np.hypot(*(p1 - p0)))
            s_start = max(float(s_start), 1e-9)
            s_end = max(float(s_end), 1e-9)
            if L <= 0:
                return np.empty((0, 2))
            # number of elements from the average size (bias keeps both endpoints)
            n = max(1, int(round(2.0 * L / (s_start + s_end))))
            # geometric spacing: first cell ~ s_start, last cell ~ s_end
            r = (s_end / s_start) ** (1.0 / max(n - 1, 1)) if n > 1 else 1.0
            cum = np.cumsum([s_start * r ** k for k in range(n)])
            ts = np.concatenate([[0.0], cum / cum[-1]])
            return p0 + np.outer(ts, (p1 - p0))

        SEED = "#333333"                      # single colour for all seeds

        def arc_seeds(arc_pts, s):
            """Resample the fillet arc at a target seed spacing `s` (mm),
            following its true arc length. Reacts to min_elem_size, unlike the
            fixed-resolution outline points."""
            if arc_pts.shape[0] < 2 or s <= 0:
                return arc_pts
            seg = np.hypot(*np.diff(arc_pts, axis=0).T)
            Lc = np.concatenate([[0.0], np.cumsum(seg)])
            total = float(Lc[-1])
            if total <= 0:
                return arc_pts
            n = max(1, int(round(total / s)))
            targets = np.linspace(0.0, total, n + 1)
            x = np.interp(targets, Lc, arc_pts[:, 0])
            y = np.interp(targets, Lc, arc_pts[:, 1])
            return np.column_stack([x, y])

        segments = [
            geometric_seeds(TL, P_on_rake, inter, nose),   # rake face inter->min
            geometric_seeds(P_on_bot, BR, nose, inter),    # clear face min->inter
            geometric_seeds(TL, TR, inter, cmax),          # top border inter->max
            geometric_seeds(BR, TR, inter, cmax),          # right border inter->max
            arc_seeds(arc, nose),                           # nose fillet at min
        ]
        first = True
        for pts in segments:
            if pts.shape[0] == 0:
                continue
            w = pts + off
            self._ax.plot(w[:, 0], w[:, 1], linestyle="none", marker="o",
                          markersize=3.0, markerfacecolor=SEED,
                          markeredgecolor="white", markeredgewidth=0.4,
                          zorder=6, label=("Tool seeds" if first else None))
            first = False

    def _draw_mesh(self, x0: float, y0: float, w: float, h: float,
                   es: float, color: str = "#1f6fb2") -> None:
        """Overlay a structured grid on a rectangular region.

        The grid lines are computed so they fall EXACTLY on the rectangle
        edges, matching what Abaqus will produce: the number of divisions
        per direction is `round(L / es)`, and the actual cell size per
        direction is `L / round(L / es)`. The two directions can therefore
        differ when L_x and L_y don't have the same remainder against `es`.

        This is the visual reflection of the same calculation done in
        `ModelConfig.effective_elem_sizes()`.

        If the number of grid lines would exceed MAX_MESH_LINES, display a
        text notice instead — drawing tens of thousands of lines blocks the
        UI for visible time and adds nothing readable.
        """
        if w <= 0 or h <= 0 or es <= 0:
            return
        nx = max(1, int(round(w / es)))
        ny = max(1, int(round(h / es)))
        total_lines = (nx + 1) + (ny + 1)
        if total_lines > self.MAX_MESH_LINES:
            self._ax.text(
                x0 + w / 2, y0 + h / 2,
                f"(mesh hidden: {total_lines} lines > {self.MAX_MESH_LINES} cap)",
                ha="center", va="center", fontsize=8,
                color="#666", style="italic",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.7, edgecolor="#aaa"),
            )
            return

        # Use parametric coordinates so the last line is guaranteed to land
        # exactly on the right/top edge, regardless of float rounding.
        segs = []
        for i in range(nx + 1):
            x = x0 + (i / nx) * w
            segs.append([(x, y0), (x, y0 + h)])
        for j in range(ny + 1):
            y = y0 + (j / ny) * h
            segs.append([(x0, y), (x0 + w, y)])
        lc = LineCollection(segs, colors=color, linewidths=0.3, alpha=0.4)
        self._ax.add_collection(lc)

    # =====================================================================
    # BCs overlay (called from update_from_config when show_bcs=True)
    # =====================================================================

    # Color codes for each Eulerian BC type. FREE is the "transparent"
    # default — we still draw a thin grey line so the user sees the face
    # has been considered, but it doesn't compete visually with the more
    # restrictive types.
    _INFLOW_COLORS = {
        "FREE":   ("#bbbbbb", "-",  1.0),   # color, linestyle, linewidth
        "NONE":   ("#2e7d32", "-",  3.0),   # green = wall
        "VOID":   ("#1f6fb2", ":",  2.5),   # blue dotted = void
    }
    _OUTFLOW_COLORS = {
        "FREE":          ("#bbbbbb", "-",  1.0),
        "NONREFLECTING": ("#d97c00", "--", 2.5),  # orange dashed
        "EQUILIBRIUM":   ("#d4b800", "-",  2.5),  # yellow
        "ZERO_PRESSURE": ("#c0392b", "-",  3.0),  # red
    }

    def _draw_bcs(self, cfg, tool_world, rp_world,
                  eul_x, eul_y, eul_w, eul_h,
                  wp_x, wp_y, l_wp_eff, h_wp_eff,
                  is_lagrangian: bool):
        """Overlay BC or IC decorations on the preview, based on
        `self.bc_view_mode`:
          - "BC": cutting velocity arrows + per-face inflow/outflow arrows
                  (only on faces with face_enabled_{side}=True) + tool
                  encastrement.
          - "IC": initial Eulerian velocity arrow field + Tini cross-hatch
                  pattern on tool and Eulerian domain.

        No floating text labels are drawn over the figure — all numeric
        values appear in the LEGEND (e.g. "Cutting velocity (60 m/min)").
        """
        b = cfg.bcs
        mode = getattr(self, "bc_view_mode", "BC")

        # Pickable segments: 4 Eulerian faces (CEL only)
        if not is_lagrangian:
            self._pickable_segments.append(
                ("eul_left",   (eul_x, eul_y,
                                eul_x, eul_y + eul_h)))
            self._pickable_segments.append(
                ("eul_right",  (eul_x + eul_w, eul_y,
                                eul_x + eul_w, eul_y + eul_h)))
            self._pickable_segments.append(
                ("eul_bot",    (eul_x,         eul_y,
                                eul_x + eul_w, eul_y)))
            self._pickable_segments.append(
                ("eul_top",    (eul_x,         eul_y + eul_h,
                                eul_x + eul_w, eul_y + eul_h)))

        # =================================================================
        # IC view: initial velocity field + Tini cross-hatch on bodies
        # =================================================================
        if mode == "IC":
            if not is_lagrangian and abs(b.initial_velocity) > 0:
                v_sign = 1.0 if b.initial_velocity > 0 else -1.0
                v_mmin = b.initial_velocity * 60.0 / 1000.0
                self._draw_velocity_field(
                    eul_x, eul_y, eul_w, eul_h, v_sign,
                    color="#1f6fb2",
                )
                proxy = self._ax.plot([], [], color="#1f6fb2", linewidth=1.5,
                              label=f"Initial Eulerian velocity ({v_mmin:.3g} m/min)")[0]
                setattr(proxy, _ARROW_FLAG, True)

            # Tini: thin dotted-red hatch pattern on the tool polygon AND
            # inside the Eulerian domain (CEL only).
            T_label = self._format_temperature(b.ambient_temperature, cfg)
            if tool_world is not None:
                self._draw_temperature_hatch(tool_world,
                                             color="#e74c3c", legend=False)
            if not is_lagrangian:
                self._draw_eul_temperature_hatch(eul_x, eul_y, eul_w, eul_h,
                                                  color="#e74c3c")
            self._ax.plot([], [], color="#e74c3c", linestyle=":",
                          linewidth=1.4,
                          label=f"Initial temperature ({T_label})")
            return

        # =================================================================
        # BC view
        # =================================================================
        # Inflow/outflow arrows on the enabled faces
        if not is_lagrangian:
            self._decorate_euler_faces(eul_x, eul_y, eul_w, eul_h, b)

        # Cutting velocity arrows
        v_cut = b.cutting_speed
        if abs(v_cut) > 0 and not is_lagrangian:
            v_mmin = v_cut * 60.0 / 1000.0
            faces = list(b.cutting_velocity_faces or [])
            for face_id in faces:
                seg = self._eulerian_face_segment(
                    face_id, eul_x, eul_y, eul_w, eul_h)
                if seg is None:
                    continue
                self._draw_face_velocity_arrows(
                    seg, v_cut, color="#1f6fb2",
                )
            if faces:
                proxy = self._ax.plot([], [], color="#1f6fb2", linewidth=2.0,
                              label=f"Cutting velocity ({v_mmin:.3g} m/min)")[0]
                setattr(proxy, _ARROW_FLAG, True)

        # Tool encastrement on l_2 + l_3 (CEL = always; Lagrangian tool_moves: skip)
        tool_is_fixed = not (
            cfg.analysis.formulation == "Lagrangian"
            and cfg.analysis.tool_motion == "tool_moves"
        )
        if tool_is_fixed and tool_world is not None:
            self._draw_tool_encastrement(tool_world, cfg)

    # ---------------------------------------------------------------
    # Helpers used by _draw_bcs
    # ---------------------------------------------------------------
    @staticmethod
    def _eulerian_face_segment(face_id: str, eul_x, eul_y, eul_w, eul_h):
        """Resolve a face id ("eul_left" / "eul_right" / "eul_top" / "eul_bot")
        to a segment ((x0,y0), (x1,y1))."""
        if face_id == "eul_left":
            return ((eul_x, eul_y),
                    (eul_x, eul_y + eul_h))
        if face_id == "eul_right":
            return ((eul_x + eul_w, eul_y),
                    (eul_x + eul_w, eul_y + eul_h))
        if face_id == "eul_bot":
            return ((eul_x,           eul_y),
                    (eul_x + eul_w,   eul_y))
        if face_id == "eul_top":
            return ((eul_x,           eul_y + eul_h),
                    (eul_x + eul_w,   eul_y + eul_h))
        return None

    @staticmethod
    def _format_temperature(value_c: float, cfg) -> str:
        """Format a temperature stored internally in °C using the user's
        currently-selected temperature unit. Returns e.g. '20 °C' or '293.15 K'."""
        tu = getattr(cfg.ui, "temp_unit", "C")
        if tu == "K":
            return f"{value_c + 273.15:.4g} K"
        return f"{value_c:.4g} °C"

    def _draw_velocity_field(self, eul_x, eul_y, eul_w, eul_h,
                              v_sign: float, color: str):
        """Draw a small grid of uniform velocity arrows inside the Eulerian
        domain to represent the initial Eulerian velocity field (CEL only).
        No floating label — value is provided through the legend."""
        nx, ny = 5, 3
        margin_x = 0.08 * eul_w
        margin_y = 0.10 * eul_h
        cur_xlim = self._ax.get_xlim()
        # Slightly longer arrows than the across-face inflow/outflow ones
        # so the initial-velocity field reads as a clear, dominant arrow
        # pattern when in IC view.
        L = min(0.13 * eul_w, 0.08 * (cur_xlim[1] - cur_xlim[0]))
        L = max(L, 1e-6)

        for i in range(nx):
            for j in range(ny):
                cx = eul_x + margin_x + (eul_w - 2 * margin_x) * i / (nx - 1)
                cy = eul_y + margin_y + (eul_h - 2 * margin_y) * j / (ny - 1)
                self._ax.annotate(
                    "", xy=(cx + v_sign * L / 2.0, cy),
                    xytext=(cx - v_sign * L / 2.0, cy),
                    arrowprops=dict(
                        arrowstyle="-|>", color=color, linewidth=1.6,
                        mutation_scale=16, shrinkA=0, shrinkB=0,
                    ),
                )

    def _draw_face_velocity_arrows(self, segment, v_cut: float,
                                    color: str):
        """Distribute several v_cut arrows along `segment`. Arrows are
        always horizontal (v1 in Abaqus); their sign and magnitude reflect
        v_cut.

        `segment` = ((x0,y0), (x1,y1))
        """
        (x0, y0), (x1, y1) = segment
        n = 6
        seg_len = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
        cur_xlim = self._ax.get_xlim()
        L_max = 0.08 * (cur_xlim[1] - cur_xlim[0])
        L = min(0.15 * seg_len, L_max) if seg_len > 0 else L_max
        L = max(L, 1e-6)
        v_sign = 1.0 if v_cut > 0 else -1.0

        # No background highlight on the face — the user requested arrows
        # alone. The face is the rectangle edge already drawn by the
        # Eulerian-domain rectangle.

        for i in range(1, n + 1):
            t = i / (n + 1)
            cx = x0 + t * (x1 - x0)
            cy = y0 + t * (y1 - y0)
            self._ax.annotate(
                "", xy=(cx + v_sign * L / 2.0, cy),
                xytext=(cx - v_sign * L / 2.0, cy),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, linewidth=1.8,
                    mutation_scale=16, shrinkA=0, shrinkB=0,
                ),
            )
        # No floating text annotation — value goes in the legend.

    def _draw_temperature_hatch(self, polygon_world, color: str = "#e74c3c",
                                  legend: bool = False):
        """Draw a thin, dotted-red hatch pattern over the polygon to
        represent the initial temperature field.

        Implementation: dashed/dotted diagonal lines clipped to the polygon.
        Compared to matplotlib's built-in `hatch="///"`, this gives us:
          - a finer, dotted look (linestyle=":")
          - a brighter red colour (we control it ourselves)
          - a controllable spacing
        """
        from matplotlib.patches import Polygon as MplPolygon
        # Add an invisible polygon to serve as the clip path for the
        # hatch lines.
        clip_patch = MplPolygon(
            polygon_world, closed=True,
            facecolor="none", edgecolor="none",
        )
        self._ax.add_patch(clip_patch)

        # Bounding box of the polygon
        xs = [p[0] for p in polygon_world]
        ys = [p[1] for p in polygon_world]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        self._draw_diagonal_hatch(xmin, xmax, ymin, ymax,
                                  color=color, clip_patch=clip_patch)

        if legend:
            self._ax.plot([], [], color=color, linestyle=":",
                          linewidth=1.0, label="Initial temperature")

    def _draw_eul_temperature_hatch(self, eul_x, eul_y, eul_w, eul_h,
                                      color: str = "#e74c3c"):
        """Draw the same hatch pattern inside the Eulerian rectangle.

        Implementation note: we use a Polygon (explicit 4 corners) rather
        than a Rectangle patch as the clip mask, because Rectangle's
        transform stack handles zoom/pan less predictably for clip-path
        purposes — diagonal hatch lines were leaking outside the visible
        Eulerian box on zoom-out.
        """
        from matplotlib.patches import Polygon as MplPolygon
        corners = [
            (eul_x,           eul_y          ),
            (eul_x + eul_w,   eul_y          ),
            (eul_x + eul_w,   eul_y + eul_h  ),
            (eul_x,           eul_y + eul_h  ),
        ]
        clip_patch = MplPolygon(
            corners, closed=True,
            facecolor="none", edgecolor="none",
        )
        self._ax.add_patch(clip_patch)
        self._draw_diagonal_hatch(eul_x, eul_x + eul_w,
                                  eul_y, eul_y + eul_h,
                                  color=color, clip_patch=clip_patch)

    def _draw_diagonal_hatch(self, xmin, xmax, ymin, ymax,
                              color: str, clip_patch):
        """Draw fine dotted diagonal lines covering [xmin..xmax]×[ymin..ymax],
        clipped to `clip_patch`. Lines run at 45° (y = x + offset).
        """
        cur_xlim = self._ax.get_xlim()
        ax_span = cur_xlim[1] - cur_xlim[0]
        # Spacing of the diagonal lines, measured along the y intercept axis.
        spacing = 0.025 * ax_span
        # Sweep offsets so the diagonals fully cover the bounding box
        c_min = ymin - xmax
        c_max = ymax - xmin
        n = int((c_max - c_min) / spacing) + 1

        for k in range(n + 1):
            c = c_min + k * spacing
            # Diagonal: y = x + c. We extend slightly past the bbox so the
            # clip masks the edges cleanly.
            x_start = xmin - spacing
            x_end   = xmax + spacing
            y_start = x_start + c
            y_end   = x_end   + c
            line = self._ax.plot(
                [x_start, x_end], [y_start, y_end],
                color=color, linestyle=":", linewidth=0.8,
                alpha=0.95,
            )[0]
            line.set_clip_path(clip_patch)

    def _draw_tool_encastrement(self, tool_world, cfg):
        """Draw a hatched 'encastrement' decoration along the tool's top
        edge (l_2, between TL and TR) and right edge (l_3, between TR
        and BR). These are the faces that anchor the rigid tool in the
        CEL setup — visually showing where the tool is held in space.

        Coordinate convention (matches `_tool_polygon`, see
        `_solve_tool_dimensions` for the derivation):
            BL = (0, 0)
            TL = BL + h * (tan(rake), 1)
            TR = (l, h)
            BR = BL + l * (1, tan(clear))
        where (h, l) are NOT the GUI inputs (h_tool, l_tool) but the
        solved bounding-box dimensions. Previously this method used
        (l_tool, h_tool) directly, which placed TL/TR/BR at the wrong
        spots whenever rake or clear was non-zero — so the hatching
        either overshot or fell short of the actual top/right faces.
        We catch ToolGeometryError silently here: if the tool failed to
        build, `_draw_bcs` already skipped this call entirely via its
        `tool_world is not None` guard, so this is mostly belt-and-braces.
        """
        import math as _m
        g = cfg.tool_geometry
        tx, ty, _engages, _reason = self._tool_offset(cfg)
        try:
            h, l = _solve_tool_dimensions(
                g.h_tool, g.l_tool, g.rake_angle, g.clear_angle
            )
        except ToolGeometryError:
            return
        rake  = _m.radians(g.rake_angle)
        clear = _m.radians(g.clear_angle)
        TL = (tx + h * _m.tan(rake),  ty + h)
        TR = (tx + l,                  ty + h)
        BR = (tx + l,                  ty + l * _m.tan(clear))

        # Draw the top edge (l_2) with hatching tick marks slanting the
        # OTHER way (slant_sign=-1) so the two edges (l_2 and l_3) form
        # a visually coherent corner pattern.
        self._draw_hatched_edge(TL, TR, normal_dir=(0.0, 1.0),
                                color="#444", slant_sign=-1)
        # Draw the right edge (l_3) with hatching tick marks pointing RIGHT
        self._draw_hatched_edge(TR, BR, normal_dir=(1.0, 0.0),
                                color="#444", slant_sign=+1)

        # No legend entry — the hatched marks are a classical mechanical
        # encastrement symbol; adding a separate legend line that doesn't
        # match would be more confusing than helpful.

    def _draw_hatched_edge(self, p0, p1, normal_dir, color="#444",
                            slant_sign: int = +1):
        """Draw the edge from p0 to p1 plus hatch lines along it.

        Hatches use a FIXED tangential spacing (proportional to the axis
        span) so two edges of different lengths look consistent.
        Each hatch slants 45° relative to the edge.
        `slant_sign` controls which side of the normal the hatch leans
        toward (+1 = default, -1 = flipped — used to reverse the hatching
        direction on one of the edges so a corner reads consistently).
        """
        cur_xlim = self._ax.get_xlim()
        ax_span = cur_xlim[1] - cur_xlim[0]
        hatch_len     = 0.018 * ax_span
        hatch_spacing = 0.018 * ax_span

        # The edge itself, in solid dark
        self._ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                      color=color, linewidth=1.4)

        ex, ey = (p1[0] - p0[0]), (p1[1] - p0[1])
        elen = (ex * ex + ey * ey) ** 0.5
        if elen < 1e-12:
            return
        ux, uy = ex / elen, ey / elen
        nx, ny = normal_dir

        n_h = max(2, int((elen - hatch_spacing) / hatch_spacing))
        step = elen / (n_h + 1)

        # Slant: hatch direction = normal +/- tangent (normalised), where
        # the sign is controlled by `slant_sign`. The default (+1) makes
        # the hatch lean toward p0; -1 makes it lean toward p1.
        slant_x = nx - slant_sign * ux
        slant_y = ny - slant_sign * uy
        slant_len = (slant_x ** 2 + slant_y ** 2) ** 0.5
        if slant_len > 1e-12:
            slant_x /= slant_len
            slant_y /= slant_len

        for k in range(1, n_h + 1):
            s_along = k * step
            sx = p0[0] + s_along * ux
            sy = p0[1] + s_along * uy
            ex_end = sx + hatch_len * slant_x
            ey_end = sy + hatch_len * slant_y
            self._ax.plot([sx, ex_end], [sy, ey_end],
                          color=color, linewidth=0.9)

    def _decorate_euler_faces(self, eul_x, eul_y, eul_w, eul_h, b):
        """Decorate the FOUR side faces of the Eulerian domain with small
        in/out arrows representing the inflow / outflow BC types.

        Convention:
          - Inflow arrows point INTO the domain (across the face from
            outside).
          - Outflow arrows point OUT of the domain (across the face from
            inside).
          - In 'both' mode both sets are drawn, with a slight tangential
            offset so they don't perfectly stack.

        Colour and style of each arrow encode the Abaqus BC type
        (FREE / NONE / VOID for inflow ; FREE / NONREFLECTING /
        EQUILIBRIUM / ZERO_PRESSURE for outflow).
        """
        # Each face is described by:
        #   p0, p1  : two endpoints in axes coordinates
        #   n_out   : outward unit normal (away from the domain)
        # The tangent direction (p1 - p0)/|p1-p0| is implicit.
        faces = {
            "left":   {"p0": (eul_x,         eul_y         ),
                       "p1": (eul_x,         eul_y + eul_h ),
                       "n_out": (-1.0, 0.0)},
            "right":  {"p0": (eul_x + eul_w, eul_y         ),
                       "p1": (eul_x + eul_w, eul_y + eul_h ),
                       "n_out": ( 1.0, 0.0)},
            "bottom": {"p0": (eul_x,         eul_y         ),
                       "p1": (eul_x + eul_w, eul_y         ),
                       "n_out": ( 0.0, -1.0)},
            "top":    {"p0": (eul_x,         eul_y + eul_h ),
                       "p1": (eul_x + eul_w, eul_y + eul_h ),
                       "n_out": ( 0.0,  1.0)},
        }

        cur_xlim = self._ax.get_xlim()
        ax_span = cur_xlim[1] - cur_xlim[0]
        # Arrow length: ~3% of the axis span. The arrows are short, drawn
        # ACROSS the face (from outside to inside for inflow, from inside
        # to outside for outflow).
        L_arrow = 0.030 * ax_span
        # Number of arrows distributed along each face
        n_arrows = 5

        for face_key, geom in faces.items():
            # Skip face entirely if not enabled
            if not getattr(b, f"face_enabled_{face_key}", False):
                continue
            mode = getattr(b, f"eulerian_bc_mode_{face_key}", "both")
            inflow_key  = getattr(b, f"eulerian_inflow_{face_key}",  "FREE")
            outflow_key = getattr(b, f"eulerian_outflow_{face_key}", "FREE")

            draw_inflow  = mode in ("inflow",  "both")
            draw_outflow = mode in ("outflow", "both")

            p0, p1 = geom["p0"], geom["p1"]
            nx, ny = geom["n_out"]
            # Tangent (unit) along the face
            tx, ty = p1[0] - p0[0], p1[1] - p0[1]
            seg_len = (tx * tx + ty * ty) ** 0.5
            if seg_len < 1e-12:
                continue
            tx, ty = tx / seg_len, ty / seg_len

            # In 'both' mode we shift the inflow vs outflow arrows along
            # the tangent so the two arrowheads at the same position
            # don't visually merge. Offset = quarter of an arrow length.
            shift = L_arrow * 0.40

            for k in range(1, n_arrows + 1):
                t = k / (n_arrows + 1)
                base_x = p0[0] + t * (p1[0] - p0[0])
                base_y = p0[1] + t * (p1[1] - p0[1])

                if draw_inflow:
                    self._draw_arrow_across_face(
                        base_x + (-shift if draw_outflow else 0.0) * tx,
                        base_y + (-shift if draw_outflow else 0.0) * ty,
                        nx, ny, L_arrow,
                        direction="in", bc_key=inflow_key,
                    )
                if draw_outflow:
                    self._draw_arrow_across_face(
                        base_x + (+shift if draw_inflow else 0.0) * tx,
                        base_y + (+shift if draw_inflow else 0.0) * ty,
                        nx, ny, L_arrow,
                        direction="out", bc_key=outflow_key,
                    )

        # Build the legend entries for the BC arrow styles actually used.
        # We collect (label, color, linestyle) once per (direction, key)
        # so the legend stays compact. Only count enabled faces.
        legend_seen: set[tuple[str, str]] = set()
        for face_key in faces:
            if not getattr(b, f"face_enabled_{face_key}", False):
                continue
            mode = getattr(b, f"eulerian_bc_mode_{face_key}", "both")
            if mode in ("inflow", "both"):
                ikey = getattr(b, f"eulerian_inflow_{face_key}", "FREE")
                sig = ("in", ikey)
                if sig not in legend_seen:
                    legend_seen.add(sig)
                    color, ls, _ = self._INFLOW_COLORS.get(
                        ikey, ("#888", "-", 1.0))
                    proxy = self._ax.plot([], [], color=color, linestyle=ls,
                                  linewidth=2.0,
                                  label=f"Inflow: {ikey}")[0]
                    setattr(proxy, _ARROW_FLAG, True)
            if mode in ("outflow", "both"):
                okey = getattr(b, f"eulerian_outflow_{face_key}", "FREE")
                sig = ("out", okey)
                if sig not in legend_seen:
                    legend_seen.add(sig)
                    color, ls, _ = self._OUTFLOW_COLORS.get(
                        okey, ("#888", "-", 1.0))
                    proxy = self._ax.plot([], [], color=color, linestyle=ls,
                                  linewidth=2.0,
                                  label=f"Outflow: {okey}")[0]
                    setattr(proxy, _ARROW_FLAG, True)

    def _draw_arrow_across_face(self, base_x, base_y, nx, ny,
                                 L_arrow, direction: str, bc_key: str):
        """Draw a single arrow crossing a face.

        - `base_x, base_y`: point on the face
        - `nx, ny`: outward unit normal of the face
        - `direction` = "in" : arrow points along -n (into the domain).
                                Starts outside the face, tip on it.
        - `direction` = "out": arrow points along +n (out of the domain).
                                Starts on the face, tip outside.

        Uses FancyArrowPatch via Matplotlib's annotate, with a large
        mutation_scale so the arrowhead is visibly a triangle, not a
        thin tip.
        """
        if direction == "in":
            color, ls, _ = self._INFLOW_COLORS.get(bc_key, ("#888", "-", 1.0))
            start = (base_x + nx * L_arrow, base_y + ny * L_arrow)
            end   = (base_x,                base_y               )
        else:
            color, ls, _ = self._OUTFLOW_COLORS.get(bc_key, ("#888", "-", 1.0))
            start = (base_x,                base_y               )
            end   = (base_x + nx * L_arrow, base_y + ny * L_arrow)

        # mutation_scale governs the head size in display units. ~12-14
        # gives a clearly-visible triangular head on the typical preview.
        self._ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="-|>", color=color, linewidth=1.5,
                linestyle=ls, mutation_scale=14,
                shrinkA=0, shrinkB=0,
            ),
        )

    def _draw_velocity_arrow(self, x: float, y: float, direction: float,
                             text: str, color: str = "#1f6fb2",
                             label: str = "", thin: bool = False):
        """Draw a horizontal velocity arrow rooted at (x, y), pointing in
        +x if direction > 0 (else -x). The length is set by the current
        axis x-span so the arrow is visible regardless of zoom level.

        `thin=True` is used for the initial-velocity arrow to visually
        distinguish it from the main cutting-velocity arrow.
        """
        # Length: ~12 % of the current x-span. We compute this from the
        # data limits we have so far (the axes are about to be re-fitted).
        cur_xlim = self._ax.get_xlim()
        L = 0.12 * (cur_xlim[1] - cur_xlim[0])
        dx = direction * L

        head_w = abs(L) * 0.08 if not thin else abs(L) * 0.05
        head_l = abs(L) * 0.10 if not thin else abs(L) * 0.07
        lw = 2.0 if not thin else 1.2

        # We don't want the arrow to register a label with the auto-legend
        # if we already have a separate legend entry — Matplotlib's annotate
        # would duplicate it. Use a plot() with the proper label and an
        # invisible head for the legend, then overlay the actual annotate
        # for the arrowhead.
        self._ax.annotate(
            "", xy=(x + dx, y), xytext=(x, y),
            arrowprops=dict(
                arrowstyle=f"-|>,head_width={head_w/2:.4f},"
                           f"head_length={head_l:.4f}",
                color=color, linewidth=lw,
            ),
        )
        # Legend entry — single plot point, invisible marker but with label
        self._ax.plot([], [], color=color, linewidth=lw, label=label)

        # Text label slightly above the arrow
        self._ax.text(
            x + dx / 2.0, y + (head_l * 1.5 if not thin else head_l),
            text, fontsize=8, color=color, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.85, linewidth=0.5),
        )

    def _draw_encastrement(self, x: float, y: float,
                           orientation: str = "down", label: str = ""):
        """Draw a 'ground' / encastrement symbol at (x, y).

        orientation:
          - 'down': hatched ground BELOW the point (point is on top of soil)
                    — used for a workpiece fixed at its bottom edge
          - 'up':   hatched ground ABOVE the point (used for the tool RP
                    when the tool is fixed in space)

        We use a small line + diagonal tick marks (classical mechanical
        symbol) rather than a full hatched patch, to keep it lightweight
        and not visually heavy.
        """
        cur_xlim = self._ax.get_xlim()
        s = 0.025 * (cur_xlim[1] - cur_xlim[0])    # symbol half-width

        # Horizontal base line
        sgn_y = -1.0 if orientation == "down" else +1.0
        y_base = y + sgn_y * s * 0.4
        self._ax.plot([x - s, x + s], [y_base, y_base],
                      color="#444", linewidth=1.4)

        # Diagonal tick marks (5 of them, slanting away from the body)
        for i in range(5):
            x0 = x - s + i * (s / 2.0)
            x1 = x0 - s * 0.25  # slant
            y1 = y_base + sgn_y * s * 0.4
            self._ax.plot([x0, x1], [y_base, y1],
                          color="#444", linewidth=1.0)

        # Small connecting line from anchor point to base line
        self._ax.plot([x, x], [y, y_base], color="#444",
                      linewidth=1.0, linestyle=":")

        # Legend entry
        self._ax.plot([], [], color="#444", linewidth=1.4,
                      marker="$\u22a5$", markersize=10, label=label)
