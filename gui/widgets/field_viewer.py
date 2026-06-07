# -*- coding: utf-8 -*-
"""
FieldViewer — matplotlib widget that renders a scalar field on an
unstructured 2D mesh, with a colorbar.

Why PolyCollection rather than pcolormesh:
  - Works for both structured (Eulerian regular grid) and unstructured
    (Lagrangian deformed mesh) meshes without branching.
  - Lets us update only the face colours when changing frame, without
    rebuilding the geometry — fast enough for slider scrubbing.

Element-to-2D-polygon projection:
  - The underlying meshes are 3D HEX with one element in z. The 2D
    polygon for visualisation is the z=zmin (or z=zmax — they're
    parallel) face of each hex, i.e. nodes [0, 1, 2, 3] in standard
    Abaqus C3D8 ordering.
"""
from __future__ import annotations
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import PolyCollection


class FieldViewer(QWidget):
    """A matplotlib canvas displaying one scalar field on one 2D mesh.

    The mesh is set via `set_mesh(...)` (once, when a new bundle is
    loaded) and the field values are updated via `set_values(...)`
    (every time the user moves the slider or picks a new variable).
    Only the colours are recomputed on each `set_values`, so scrubbing
    the slider stays smooth even for ~10k elements.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._fig    = Figure(figsize=(5, 4))
        self._canvas = FigureCanvas(self._fig)
        self._ax     = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlabel("x [mm]")
        self._ax.set_ylabel("y [mm]")
        self._ax.grid(True, alpha=0.25, linestyle=":")
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # State
        self._polys: PolyCollection | None = None
        self._cbar = None
        self._title = ""

    # =====================================================================
    # Mesh / values API
    # =====================================================================
    def set_mesh(self, nodes_xy: np.ndarray, element_face_indices: np.ndarray):
        """Set the 2D mesh.

        Parameters
        ----------
        nodes_xy:
            (n_nodes, 2) array of nodal (x, y) coordinates.
        element_face_indices:
            (n_elements, 4) array of node indices defining the visible
            face of each element. For 3D HEX with one element in z,
            this is the (0, 1, 2, 3) bottom face.
        """
        # Clear any previous patch collection (keep the colorbar — removing
        # and recreating it crashes matplotlib in some versions, and also
        # resets the axes aspect/adjustable, which pushes the mesh off-view).
        if self._polys is not None:
            try:
                self._polys.remove()
            except Exception:
                pass
            self._polys = None

        # Build (n_elements, 4, 2) vertices array for PolyCollection
        verts = nodes_xy[element_face_indices]   # (n_elements, 4, 2)

        # Initial dummy values (all zeros) — set_values() will overwrite
        n_elements = verts.shape[0]
        self._polys = PolyCollection(
            verts,
            array=np.zeros(n_elements, dtype=np.float32),
            edgecolors="none",
            cmap="viridis",
        )
        self._ax.add_collection(self._polys)

        # Colorbar: create ONCE, then just point it at the new collection.
        if self._cbar is None:
            self._cbar = self._fig.colorbar(self._polys, ax=self._ax,
                                            fraction=0.05, pad=0.04)
        else:
            try:
                self._cbar.update_normal(self._polys)
            except Exception:
                pass

        # Fit the axes to the mesh extent (with a small margin). Do this
        # LAST and re-assert adjustable='box' so the colorbar layout can't
        # silently switch us to 'datalim' and ignore these limits.
        xs = nodes_xy[:, 0]; ys = nodes_xy[:, 1]
        pad_x = 0.02 * (xs.max() - xs.min() + 1e-12)
        pad_y = 0.02 * (ys.max() - ys.min() + 1e-12)
        self._ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        self._ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
        self._ax.set_aspect("equal", adjustable="box")
        self._canvas.draw_idle()

    def set_values(self, values: np.ndarray, vmin: float = None,
                    vmax: float = None, cmap: str = None,
                    title: str = None):
        """Update the field values shown on the mesh.

        Parameters
        ----------
        values:
            (n_elements,) array of scalar values, one per element.
        vmin, vmax:
            Optional color-range clipping. If None, computed from `values`.
            Pass fixed values across all frames to keep the colormap
            stable while scrubbing the slider.
        cmap:
            Optional colormap name (e.g. "viridis", "inferno", "RdBu_r").
        title:
            Optional title for the plot.
        """
        if self._polys is None:
            return    # set_mesh hasn't been called yet — nothing to draw

        values = np.asarray(values, dtype=np.float32).ravel()
        # Guard against a field/mesh size mismatch: PolyCollection silently
        # fails to colour if the array length != number of polygons. Print
        # a clear diagnostic (visible in run_gui_debug.bat) and bail.
        n_polys = len(self._polys.get_paths())
        if values.shape[0] != n_polys:
            import sys
            sys.stderr.write(
                "[FieldViewer] size mismatch: %d field values vs %d mesh "
                "elements -> field not drawn.\n" % (values.shape[0], n_polys))
            return

        self._polys.set_array(values)
        if cmap is not None:
            self._polys.set_cmap(cmap)
        finite = values[np.isfinite(values)]
        if vmin is None:
            vmin = float(finite.min()) if finite.size else 0.0
        if vmax is None:
            vmax = float(finite.max()) if finite.size else 1.0
        if vmin >= vmax:
            vmax = vmin + 1e-12
        self._polys.set_clim(vmin, vmax)
        if self._cbar is not None:
            # Newer matplotlib: update_normal is deprecated, just redraw.
            self._cbar.update_normal(self._polys)

        if title is not None:
            self._ax.set_title(title, fontsize=10)
            self._title = title

        self._canvas.draw_idle()

    def clear(self):
        """Remove the field & mesh; leave an empty axes."""
        if self._polys is not None:
            self._polys.remove()
            self._polys = None
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        self._ax.set_title("")
        self._canvas.draw_idle()
