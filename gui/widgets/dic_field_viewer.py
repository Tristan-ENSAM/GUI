# -*- coding: utf-8 -*-
"""
DicFieldViewer — heatmap of a DIC field (displacement / velocity / strain rate
/ strain) with a time slider, a component selector, an interpolation selector
and a profile-extraction tool (draw a line, get the field along it vs arc
length — the experimental counterpart of a numerical path extraction).

Performance / zoom: the image (and colorbar) are built ONCE in `set_field`;
moving the slider only calls AxesImage.set_data, so the user's zoom is kept and
scrubbing is smooth (clearing the axes every frame used to reset the view).
The toolbar Home button resets to the full field extent.
"""
from __future__ import annotations
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
    QCheckBox, QDialog,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

_INTERP = ["nearest", "bilinear", "bicubic", "spline16", "spline36"]
_QUALITY_FIELDS = {"ZNCC"}


def _bilinear_grid(ux, uy, Z, qx, qy):
    """Bilinear interpolation of Z (ny,nx) on increasing axes ux,uy at (qx,qy).
    NaN outside the grid or where a needed corner is NaN. Pure numpy."""
    qx = np.asarray(qx, float); qy = np.asarray(qy, float)
    out = np.full(qx.shape, np.nan)
    nx, ny = ux.size, uy.size
    if nx < 2 or ny < 2:
        return out
    inside = (qx >= ux[0]) & (qx <= ux[-1]) & (qy >= uy[0]) & (qy <= uy[-1])
    ix = np.clip(np.searchsorted(ux, qx) - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(uy, qy) - 1, 0, ny - 2)
    for k in np.nonzero(inside)[0]:
        i, j = ix[k], iy[k]
        x0, x1 = ux[i], ux[i + 1]; y0, y1 = uy[j], uy[j + 1]
        tx = 0.0 if x1 == x0 else (qx[k] - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (qy[k] - y0) / (y1 - y0)
        z00, z10, z01, z11 = Z[j, i], Z[j, i + 1], Z[j + 1, i], Z[j + 1, i + 1]
        if not np.all(np.isfinite([z00, z10, z01, z11])):
            continue
        out[k] = ((z00 * (1 - tx) + z10 * tx) * (1 - ty)
                  + (z01 * (1 - tx) + z11 * tx) * ty)
    return out


class _ProfileWindow(QDialog):
    """Standalone resizable window showing the extracted profile."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DIC profile")
        self.resize(640, 360)
        self._fig = Figure(figsize=(6, 3.2), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._ax = self._fig.add_subplot(111)
        lay = QVBoxLayout(self)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._canvas, 1)

    def set_curve(self, dist, vals, ylabel):
        self._ax.clear()
        self._ax.set_xlabel("distance along line [mm]")
        self._ax.set_ylabel(ylabel)
        self._ax.grid(True, alpha=0.3, linestyle=":")
        if dist is not None:
            self._ax.plot(dist, vals, "-", color="#1f6fb2")
        self._canvas.draw_idle()


class DicFieldViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = self._y = self._t = None
        self._comps: dict = {}
        self._units: dict = {}
        self._valid = None
        self._grid = None            # (ux, uy, ix, iy)
        self._vlim: dict = {}
        self._frame = 0
        self._profile_mode = False
        self._line = []
        self._im = None              # AxesImage (grid) or PathCollection (scatter)
        self._is_image = False
        self._cbar = None
        self._extent = None
        # background visible image (spatial + temporal correspondence)
        self._bg_provider = None     # callable(frame_index) -> image array
        self._bg_mmpp = None
        self._bg_wh = None
        self._bg_im = None

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Component:"))
        self.cb_comp = QComboBox()
        self.cb_comp.currentIndexChanged.connect(self._on_component)
        ctrl.addWidget(self.cb_comp)
        ctrl.addWidget(QLabel("Interp.:"))
        self.cb_interp = QComboBox(); self.cb_interp.addItems(_INTERP)
        self.cb_interp.currentTextChanged.connect(self._on_interp)
        ctrl.addWidget(self.cb_interp)
        self.chk_bg = QCheckBox("Show image")
        self.chk_bg.toggled.connect(lambda *_: self._build())
        ctrl.addWidget(self.chk_bg)
        self.b_profile = QPushButton("Pick profile line"); self.b_profile.setCheckable(True)
        self.b_profile.toggled.connect(lambda on: setattr(self, "_profile_mode", on))
        ctrl.addWidget(self.b_profile)
        self.b_clear = QPushButton("Clear profile")
        self.b_clear.clicked.connect(self._clear_profile)
        ctrl.addWidget(self.b_clear)
        self.b_profile_win = QPushButton("Profile window")
        self.b_profile_win.setToolTip("Open the extracted profile in a separate "
                                      "resizable window.")
        self.b_profile_win.clicked.connect(self._open_profile_window)
        ctrl.addWidget(self.b_profile_win)
        ctrl.addStretch(1)

        srow = QHBoxLayout()
        srow.addWidget(QLabel("Frame:"))
        self.sld = QSlider(Qt.Orientation.Horizontal)
        self.sld.setRange(0, 0)
        self.sld.valueChanged.connect(self._on_frame)
        self.lbl_t = QLabel("—")
        srow.addWidget(self.sld, 1); srow.addWidget(self.lbl_t)

        self._fig = Figure(figsize=(6, 6), constrained_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        for _act in self._toolbar.actions():
            if _act.text() == "Home":
                _act.triggered.connect(lambda *_: self.reset_view())
        # Heatmap + dedicated colorbar axis. The profile is shown in a separate
        # window (button above) so it doesn't eat the heatmap's space.
        gs = self._fig.add_gridspec(1, 2, width_ratios=[30, 1])
        self._ax = self._fig.add_subplot(gs[0, 0])
        self._cax = self._fig.add_subplot(gs[0, 1])
        self._ax.set_aspect("equal", adjustable="box")
        self._line_artist = None
        self._profile_win = None
        self._canvas.mpl_connect("button_press_event", self._on_click)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(ctrl)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._canvas, 1)
        lay.addLayout(srow)

    # ------------------------------------------------------------------
    def set_field(self, x, y, t, comps: dict, valid=None, units=None):
        self._x = np.asarray(x, float); self._y = np.asarray(y, float)
        self._t = np.asarray(t, float)
        self._comps = {k: np.asarray(v, float) for k, v in comps.items()}
        self._units = dict(units or {})
        self._valid = None if valid is None else np.asarray(valid, bool)
        self._grid = self._detect_grid()
        # stable per-component colour limits (robust percentiles)
        self._vlim = {}
        for k, v in self._comps.items():
            vv = v.copy()
            if self._valid is not None:
                vv = np.where(self._valid, vv, np.nan)
            finite = vv[np.isfinite(vv)]
            self._vlim[k] = ((float(np.percentile(finite, 2)),
                              float(np.percentile(finite, 98)))
                             if finite.size else (0.0, 1.0))
        self.cb_comp.blockSignals(True)
        self.cb_comp.clear(); self.cb_comp.addItems(list(self._comps.keys()))
        self.cb_comp.blockSignals(False)
        self.sld.setRange(0, max(0, self._t.size - 1))
        self._frame = 0; self.sld.setValue(0)
        self._build()

    def _detect_grid(self):
        if self._x is None or self._x.size == 0:
            return None
        ux = np.unique(np.round(self._x, 6)); uy = np.unique(np.round(self._y, 6))
        if ux.size * uy.size != self._x.size:
            return None
        ix = np.searchsorted(ux, np.round(self._x, 6))
        iy = np.searchsorted(uy, np.round(self._y, 6))
        return (ux, uy, ix, iy)

    # ------------------------------------------------------------------
    def _Z(self):
        """Current component as a 2D masked grid (or None if scattered)."""
        comp = self.cb_comp.currentText()
        if comp not in self._comps:
            return None, comp
        v = self._comps[comp][self._frame].astype(float).copy()
        # Quality fields (ZNCC) are shown everywhere they were computed, so the
        # low-quality / rejected zones remain visible; other fields are masked
        # to the valid points.
        if self._valid is not None and comp not in _QUALITY_FIELDS:
            v[~self._valid[self._frame]] = np.nan
        if self._grid is None:
            return v, comp
        ux, uy, ix, iy = self._grid
        Z = np.full((uy.size, ux.size), np.nan)
        Z[iy, ix] = v
        return np.ma.masked_invalid(Z), comp

    def _clabel(self, comp):
        u = self._units.get(comp, "")
        return "%s [%s]" % (comp, u) if u else comp

    def _build(self):
        """(Re)build the image/scatter, colorbar and limits once."""
        self._ax.clear()
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlabel("x [mm]"); self._ax.set_ylabel("y [mm]")
        self._im = None; self._line_artist = None; self._bg_im = None
        self._cax.clear()
        if self._x is None or self._x.size == 0:
            self._canvas.draw_idle(); return
        self._draw_background()
        Z, comp = self._Z()
        vmin, vmax = self._vlim.get(comp, (None, None))
        cmap = plt_cmap()
        alpha = self._field_alpha()
        if self._grid is not None:
            ux, uy, _, _ = self._grid
            dx = (ux[1] - ux[0]) if ux.size > 1 else 1.0
            dy = (uy[1] - uy[0]) if uy.size > 1 else 1.0
            self._extent = (ux[0] - dx / 2, ux[-1] + dx / 2,
                            uy[0] - dy / 2, uy[-1] + dy / 2)
            self._im = self._ax.imshow(
                Z, extent=self._extent, origin="lower",
                interpolation=self.cb_interp.currentText(), cmap=cmap,
                vmin=vmin, vmax=vmax, aspect="equal", alpha=alpha, zorder=1)
            self._is_image = True
        else:
            self._im = self._ax.scatter(self._x, self._y, c=Z, s=14, cmap=cmap,
                                        vmin=vmin, vmax=vmax, alpha=alpha, zorder=1)
            self._is_image = False
            self._extent = None
        self._cax.clear()
        self._cbar = self._fig.colorbar(self._im, cax=self._cax,
                                        label=self._clabel(comp))
        if self._extent is not None:
            self._ax.set_xlim(self._extent[0], self._extent[1])
            self._ax.set_ylim(self._extent[2], self._extent[3])
        self._draw_line_artist()
        self._update_time_label()
        self._draw_profile()
        self._canvas.draw_idle()
        # make the current full view the toolbar 'Home'
        try:
            self._toolbar.update()
        except Exception:
            pass

    def set_background(self, provider, mm_per_px, img_w, img_h):
        """Provide the visible image behind the field. `provider(frame_index)`
        returns the image for the displayed field frame (spatial + temporal
        correspondence). Centred on the model origin at mm_per_px, like the
        Alignment convention."""
        self._bg_provider = provider
        self._bg_mmpp = float(mm_per_px) if mm_per_px else None
        self._bg_wh = (int(img_w), int(img_h))
        if self._x is not None:
            self._build()

    def _bg_extent(self):
        if self._bg_wh is None or not self._bg_mmpp:
            return None
        w, h = self._bg_wh; s = self._bg_mmpp
        return (-w / 2.0 * s, w / 2.0 * s, -h / 2.0 * s, h / 2.0 * s)

    def _draw_background(self):
        ext = self._bg_extent()
        if not self.chk_bg.isChecked() or self._bg_provider is None or ext is None:
            return None
        try:
            img = np.asarray(self._bg_provider(self._frame))
        except Exception:
            return None
        is_rgb = (img.ndim == 3 and img.shape[-1] in (3, 4))
        self._bg_im = self._ax.imshow(
            img, extent=ext, origin="upper", cmap=None if is_rgb else "gray",
            zorder=-10, aspect="equal")
        return self._bg_im

    def _field_alpha(self):
        return 0.65 if (self.chk_bg.isChecked() and self._bg_provider) else 1.0

    def reset_view(self):
        if self._extent is not None:
            self._ax.set_xlim(self._extent[0], self._extent[1])
            self._ax.set_ylim(self._extent[2], self._extent[3])
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_frame(self, i):
        self._frame = int(i)
        self._update_time_label()
        self._update_values()       # data only -> keeps zoom

    def _update_time_label(self):
        if self._t is not None and self._t.size:
            self.lbl_t.setText("t = %.6g s" % self._t[self._frame])

    def _update_values(self):
        if self._im is None:
            return
        Z, comp = self._Z()
        if self._is_image:
            self._im.set_data(Z)
        else:
            self._im.set_array(np.asarray(Z))
        if self._bg_im is not None and self._bg_provider is not None:
            try:
                self._bg_im.set_data(np.asarray(self._bg_provider(self._frame)))
            except Exception:
                pass
        self._draw_profile()
        self._canvas.draw_idle()

    def _on_component(self, *_):
        # component change -> new clim + colorbar label; keep current zoom
        if self._im is None:
            return
        Z, comp = self._Z()
        vmin, vmax = self._vlim.get(comp, (None, None))
        if self._is_image:
            self._im.set_data(Z)
        else:
            self._im.set_array(np.asarray(Z))
        self._im.set_clim(vmin, vmax)
        if self._cbar is not None:
            self._cbar.set_label(self._clabel(comp))
        self._draw_profile()
        self._canvas.draw_idle()

    def _on_interp(self, mode):
        if self._im is not None and self._is_image:
            self._im.set_interpolation(mode)
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    def _toggle_profile(self, on):
        self._profile_mode = bool(on)

    def _clear_profile(self):
        self._line = []
        self._draw_line_artist()
        self._draw_profile()
        self._canvas.draw_idle()

    def _on_click(self, event):
        if not self._profile_mode or event.inaxes is not self._ax or event.xdata is None:
            return
        if len(self._line) >= 2:
            self._line = []
        self._line.append((float(event.xdata), float(event.ydata)))
        self._draw_line_artist()
        self._draw_profile()
        self._canvas.draw_idle()

    def _draw_line_artist(self):
        if self._line_artist is not None:
            try:
                self._line_artist.remove()
            except Exception:
                pass
            self._line_artist = None
        if len(self._line) >= 1:
            xs = [p[0] for p in self._line]; ys = [p[1] for p in self._line]
            self._line_artist, = self._ax.plot(xs, ys, "-o", color="red",
                                                lw=1.5, ms=4)

    def sample_profile(self, n=200):
        if len(self._line) != 2 or self._x is None:
            return None, None
        comp = self.cb_comp.currentText()
        v = self._comps[comp][self._frame].astype(float).copy()
        if self._valid is not None and comp not in _QUALITY_FIELDS:
            v[~self._valid[self._frame]] = np.nan
        (x0, y0), (x1, y1) = self._line
        tt = np.linspace(0, 1, n)
        lx = x0 + tt * (x1 - x0); ly = y0 + tt * (y1 - y0)
        dist = np.hypot(lx - x0, ly - y0)
        if self._grid is not None:
            ux, uy, ix, iy = self._grid
            Z = np.full((uy.size, ux.size), np.nan); Z[iy, ix] = v
            return dist, _bilinear_grid(ux, uy, Z, lx, ly)
        try:
            from scipy.interpolate import griddata
        except Exception:
            return dist, np.full(n, np.nan)
        ok = np.isfinite(v)
        if ok.sum() < 3:
            return None, None
        vals = griddata(np.column_stack([self._x, self._y])[ok], v[ok],
                        np.column_stack([lx, ly]), method="linear")
        return dist, vals

    def _open_profile_window(self):
        if self._profile_win is None:
            self._profile_win = _ProfileWindow(self)
        self._profile_win.show()
        self._profile_win.raise_()
        self._draw_profile()

    def _draw_profile(self):
        """Update the profile window (if open) with the current line/component."""
        if self._profile_win is None or not self._profile_win.isVisible():
            return
        comp = self.cb_comp.currentText()
        dist, vals = self.sample_profile()
        self._profile_win.set_curve(dist, vals, self._clabel(comp))


def plt_cmap():
    import matplotlib
    return matplotlib.colormaps.get_cmap("viridis").copy()
