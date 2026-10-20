# -*- coding: utf-8 -*-
"""Deviation maps between two runs of the sizing studies (Model tab).

Shows, for a reference run and a compared run of the mass-scaling, mesh or
domain studies, where in the ZOI and when in the time window T the two runs
differ (gui.sensitivity.deviation_maps):

* a map of |q_b - q_a| / eps_q on the ZOI grid: at one frame, averaged over
  T, or as the difference of the time means (bias);
* a click on a ZOI point plots q_a(t) and q_b(t) at that point; without a
  point, the spatial mean of |q_b - q_a| / eps_q per frame;
* for the forces, the two force curves.

Nothing is run: the saved results of the study folders are read.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui.core.async_call import run_async
from gui.sensitivity import deviation_maps as dm

_VIEWS = (("mean", "Mean over T of |b − a| / ε"),
          ("frame", "One frame: |b − a| / ε"),
          ("bias", "Difference of the time means |b̄ − ā| / ε"))
_UNITS = {"Vx": "mm/s", "Vy": "mm/s", "T": "K", "EVF": "-", "Fc": "N/mm",
          "Ff": "N/mm"}


def _fmt(v, f="%.3f"):
    return f % v if v is not None and math.isfinite(v) else "–"


class DeviationMapsDialog(QDialog):
    """Non-modal window; `folders` are the study folders listed at start."""

    def __init__(self, folders=(), start_dir="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deviation maps between two runs")
        self.resize(1100, 850)
        self._folders = []
        self._runs = []
        self._pair = None
        self._pair_jobs = ("a", "b")
        self._point = None
        self._busy = False
        self._start_dir = str(start_dir or "")

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Where and when two runs of a study differ. Pick the reference "
            "run (a) and the compared run (b), then Compare. The value shown "
            "is |b − a| / ε_q: 1 means a deviation equal to the tolerance. "
            "Click a point of the map to see the two time curves there.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Study"))
        self.cb_study = QComboBox()
        self.cb_study.setToolTip(
            "The study folders recorded for this model. The runs of the\n"
            "chosen one are listed below; 'Add a folder' adds the runs of\n"
            "another folder to them (e.g. a run made again elsewhere).")
        self.cb_study.addItem("(choose a study folder)", None)
        for f in folders or ():
            if Path(f).is_dir():
                self.cb_study.addItem(Path(f).name, str(f))
        self.cb_study.currentIndexChanged.connect(self._on_study)
        row.addWidget(self.cb_study, 1)
        b = QPushButton("Add a folder…")
        b.clicked.connect(self._on_add_folder)
        row.addWidget(b)
        lay.addLayout(row)
        self.lbl_folders = QLabel("No study folder.")
        self.lbl_folders.setWordWrap(True)
        lay.addWidget(self.lbl_folders)

        row = QHBoxLayout()
        row.addWidget(QLabel("Reference (a)"))
        self.cb_a = QComboBox()
        self.cb_a.setMinimumContentsLength(30)
        row.addWidget(self.cb_a, 1)
        row.addWidget(QLabel("Compared (b)"))
        self.cb_b = QComboBox()
        self.cb_b.setMinimumContentsLength(30)
        row.addWidget(self.cb_b, 1)
        self.btn_compare = QPushButton("Compare")
        self.btn_compare.clicked.connect(self._on_compare)
        row.addWidget(self.btn_compare)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output"))
        self.cb_q = QComboBox()
        self.cb_q.currentIndexChanged.connect(self._on_quantity)
        row.addWidget(self.cb_q)
        row.addWidget(QLabel("Map"))
        self.cb_view = QComboBox()
        for key, text in _VIEWS:
            self.cb_view.addItem(text, key)
        self.cb_view.currentIndexChanged.connect(self._redraw)
        row.addWidget(self.cb_view)
        self.sl_frame = QSlider(Qt.Horizontal)
        self.sl_frame.setEnabled(False)
        self.sl_frame.valueChanged.connect(self._redraw)
        row.addWidget(self.sl_frame, 1)
        self.lbl_frame = QLabel("")
        row.addWidget(self.lbl_frame)
        self.btn_export = QPushButton("Export (CSV + xlsx)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export)
        row.addWidget(self.btn_export)
        lay.addLayout(row)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)

        split = QSplitter(Qt.Vertical)
        self.fig_map = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas_map = FigureCanvas(self.fig_map)
        self.canvas_map.mpl_connect("button_press_event", self._on_click)
        self.fig_curves = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas_curves = FigureCanvas(self.fig_curves)
        for canvas in (self.canvas_map, self.canvas_curves):
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            tb = NavigationToolbar2QT(canvas, w)
            v.addWidget(tb)
            v.addWidget(canvas, 1)
            if canvas is self.canvas_map:
                self._map_toolbar = tb
            split.addWidget(w)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)

        if self.cb_study.count() > 1:
            self.cb_study.setCurrentIndex(self.cb_study.count() - 1)
        self._redraw()

    def _on_study(self, *_):
        folder = self.cb_study.currentData()
        self._folders, self._runs = [], []
        if folder:
            self.add_folder(folder)
        else:
            self._refill_runs()

    # ---- runs ----------------------------------------------------------
    def add_folder(self, folder, quiet=False) -> int:
        folder = Path(folder)
        if not folder.is_dir() or any(
                folder.resolve() == f.resolve() for f in self._folders):
            return 0
        self._folders.append(folder)
        before = len(self._runs)
        self._runs = dm.list_runs(self._folders)
        n = len(self._runs) - before
        if n == 0 and not quiet:
            QMessageBox.information(
                self, "Add a study folder",
                "%s holds no finished run (a run needs its .results.npz, "
                ".meta.json and a .sta that reports a completed analysis)."
                % folder.name)
        self._refill_runs()
        return n

    def _refill_runs(self):
        names = ", ".join(f.name for f in self._folders)
        self.lbl_folders.setText(
            "Study folders: %s (%d finished runs)" % (names, len(self._runs))
            if self._folders else "No study folder.")
        for cb in (self.cb_a, self.cb_b):
            cb.blockSignals(True)
            cb.clear()
            for r in self._runs:
                cb.addItem(dm.run_label(r))
            cb.blockSignals(False)
        if not self._runs:
            return
        step = None
        for f in self._folders:
            try:
                step = dm.settings_of_folder(f).step
                break
            except Exception:
                continue
        ia = dm.default_reference(self._runs, step)
        self.cb_a.setCurrentIndex(max(ia, 0))
        self.cb_b.setCurrentIndex(1 if ia == 0 and len(self._runs) > 1 else 0)

    def _on_add_folder(self):
        start = str(self._folders[-1].parent) if self._folders \
            else self._start_dir
        folder = QFileDialog.getExistingDirectory(
            self, "Add a study folder", start)
        if folder:
            self.add_folder(folder)

    def _settings_for(self, run):
        """Settings of the reference run's folder, else of the first study
        folder that has them."""
        errors = []
        for f in [run.folder] + self._folders:
            try:
                return dm.settings_of_folder(f), None
            except Exception as e:
                errors.append(str(e))
        return None, "; ".join(errors)

    # ---- comparison ----------------------------------------------------
    def _on_compare(self):
        if self._busy or not self._runs:
            return
        ra = self._runs[self.cb_a.currentIndex()]
        rb = self._runs[self.cb_b.currentIndex()]
        if ra is rb:
            QMessageBox.information(self, "Compare",
                                    "Pick two different runs.")
            return
        settings, err = self._settings_for(ra)
        if settings is None:
            QMessageBox.warning(
                self, "Compare", "No comparison settings (ZOI, sampling step, "
                "window, ε_q) found in the study folders: %s" % err)
            return
        self._busy = True
        self.btn_compare.setEnabled(False)
        self.lbl_summary.setText("Reading the two runs…")

        def work():
            from gui.results.reader import ResultsBundle
            try:
                with ResultsBundle.load(ra.npz) as ba, \
                        ResultsBundle.load(rb.npz) as bb:
                    pair = dm.compare_runs(
                        ba, bb, settings, dm.run_elem_size(ra.params),
                        dm.run_elem_size(rb.params))
                pair.label_a, pair.label_b = dm.run_label(ra), dm.run_label(rb)
                return ("ok", pair, (ra.job, rb.job))
            except Exception as e:
                return ("error", "%s: %s" % (type(e).__name__, e))

        run_async(work, self._on_compared, self)

    def _on_compared(self, res):
        self._busy = False
        self.btn_compare.setEnabled(True)
        if not res or res[0] != "ok":
            self.lbl_summary.setText("")
            QMessageBox.warning(self, "Compare", "The comparison failed: %s"
                                % (res[1] if res else "unknown error"))
            return
        self._pair, self._pair_jobs = res[1], res[2]
        self._point = None
        self.btn_export.setEnabled(True)
        q_prev = self.cb_q.currentText()
        self.cb_q.blockSignals(True)
        self.cb_q.clear()
        for q in self._pair.quantities:
            self.cb_q.addItem(q)
        i = self.cb_q.findText(q_prev)
        self.cb_q.setCurrentIndex(i if i >= 0 else 0)
        self.cb_q.blockSignals(False)
        self._write_summary()
        self._on_quantity()

    def _write_summary(self):
        p = self._pair
        cells = ["%s %s / %s" % (q, _fmt(d.e_inst), _fmt(d.e_bias))
                 for q, d in p.quantities.items()]
        s = p.settings
        self.lbl_summary.setText(
            "<b>E/ε (all samples) / bias E/ε (time means):</b> %s<br>"
            "a = %s<br>b = %s<br>ZOI x[%g, %g] y[%g, %g] mm, step %g mm, "
            "T = %g–%g of the simulated time, sampling %s, largest frame "
            "offset %s of a frame interval."
            % ("&nbsp;&nbsp; ".join(cells), p.label_a, p.label_b,
               *s.zoi, s.grid_step, s.window[0], s.window[1],
               "linear (different meshes)" if p.linear else
               "nearest element (as the studies)",
               _fmt(p.frame_offset_over_interval, "%.3g")))

    def _current(self):
        if self._pair is None:
            return None
        return self._pair.quantities.get(self.cb_q.currentText())

    def _on_quantity(self, *_):
        d = self._current()
        self.sl_frame.blockSignals(True)
        if d is not None and d.is_field:
            self.sl_frame.setRange(0, max(0, d.times.size - 1))
            self.sl_frame.setEnabled(True)
        else:
            self.sl_frame.setEnabled(False)
        self.sl_frame.blockSignals(False)
        self._redraw()

    # ---- drawing -------------------------------------------------------
    def _map_values(self, d):
        view = self.cb_view.currentData()
        if view == "frame":
            return d.frame_map(self.sl_frame.value())
        if view == "bias":
            return d.bias_map()
        return d.mean_map()

    def _redraw(self, *_):
        self._draw_map()
        self._draw_curves()

    def _draw_map(self):
        fig = self.fig_map
        fig.clear()
        ax = fig.add_subplot(111)
        d = self._current()
        self.lbl_frame.setText("")
        if d is None:
            ax.text(0.5, 0.5, "Add a study folder, pick two runs, Compare.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas_map.draw_idle()
            return
        if not d.is_field:
            ax.text(0.5, 0.5, "%s is a force: no map, see the curves below."
                    % d.quantity, ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas_map.draw_idle()
            return
        p = self._pair
        ny, nx = p.shape
        vals = self._map_values(d).reshape(ny, nx)
        g = p.settings.grid_step
        xs = p.points[:nx, 0]
        ys = p.points[::nx, 1]
        extent = (xs[0] - g / 2, xs[-1] + g / 2, ys[0] - g / 2, ys[-1] + g / 2)
        finite = vals[np.isfinite(vals)]
        vmax = max(1.0, float(np.percentile(finite, 99))) if finite.size \
            else 1.0
        import matplotlib
        cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#d0d0d0")
        im = ax.imshow(vals, origin="lower", extent=extent, cmap=cmap,
                       vmin=0.0, vmax=vmax, interpolation="nearest",
                       aspect="equal")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("/ ε_%s (ε = %g %s)" % (d.quantity, d.eps,
                                             _UNITS.get(d.quantity, "")))
        if finite.size and finite.max() > 1.0 and ny > 1 and nx > 1:
            ax.contour(xs, ys, np.where(np.isfinite(vals), vals, 0.0),
                       levels=[1.0], colors="red", linewidths=1.0)
        if self._point is not None:
            x, y = p.points[self._point]
            ax.plot([x], [y], marker="s", mfc="none", mec="white", mew=2, ms=10)
        view = self.cb_view.currentData()
        title = dict(_VIEWS)[view].replace("ε", "ε_" + d.quantity)
        if view == "frame":
            k = self.sl_frame.value()
            self.lbl_frame.setText("t = %.4g s" % d.times[k])
            title += "  (t = %.4g s)" % d.times[k]
        ax.set_title(title + "\nmean over the map: %s   (grey: not compared,"
                     " red line: 1)" % _fmt(float(np.mean(finite))
                                           if finite.size else None),
                     fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        self.canvas_map.draw_idle()

    def _draw_curves(self):
        fig = self.fig_curves
        fig.clear()
        ax = fig.add_subplot(111)
        d = self._current()
        if d is None:
            ax.set_axis_off()
            self.canvas_curves.draw_idle()
            return
        unit = _UNITS.get(d.quantity, "")
        if not d.is_field:
            ax.plot(d.times, d.a, lw=0.8, label="a (reference)")
            ax.plot(d.times, d.b, lw=0.8, label="b")
            ax.set_ylabel("%s [%s]" % (d.quantity, unit))
            ax.set_title("%s over T: E/ε %s, bias E/ε %s"
                         % (d.quantity, _fmt(d.e_inst), _fmt(d.e_bias)),
                         fontsize=9)
            ax.legend(fontsize=8)
        elif self._point is None:
            ax.plot(d.times, d.e_frame, lw=1.0)
            ax.axhline(1.0, color="red", lw=0.8)
            ax.set_ylabel("mean over the ZOI of |b − a| / ε")
            ax.set_title("%s: deviation per frame (click a map point for "
                         "its curves)" % d.quantity, fontsize=9)
        else:
            i = self._point
            x, y = self._pair.points[i]
            ax.plot(d.times, d.a[:, i], lw=1.0, marker=".", ms=3,
                    label="a (reference)")
            ax.plot(d.times, d.b[:, i], lw=1.0, marker=".", ms=3, label="b")
            ax.set_ylabel("%s [%s]" % (d.quantity, unit))
            ax.set_title("%s at x = %.4g, y = %.4g mm: mean |b − a| / ε %s, "
                         "|b̄ − ā| / ε %s, %d frames compared (gaps: void)"
                         % (d.quantity, x, y, _fmt(d.mean_map()[i]),
                            _fmt(d.bias_map()[i]), d.n_compared()[i]),
                         fontsize=9)
            ax.legend(fontsize=8)
        if d.is_field and self.cb_view.currentData() == "frame":
            ax.axvline(d.times[self.sl_frame.value()], color="grey", lw=0.8,
                       ls="--")
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.25, linestyle=":")
        self.canvas_curves.draw_idle()

    def _on_click(self, event):
        d = self._current()
        if (d is None or not d.is_field or event.inaxes is None
                or event.xdata is None
                or getattr(self._map_toolbar, "mode", "")):
            return
        self._point = self._pair.nearest_point(event.xdata, event.ydata)
        self._redraw()

    # ---- export --------------------------------------------------------
    def _on_export(self):
        if self._pair is None:
            return
        start = str(self._folders[0]) if self._folders else self._start_dir
        folder = QFileDialog.getExistingDirectory(
            self, "Export the comparison into", start)
        if not folder:
            return
        a, b = self._pair_jobs
        try:
            paths = dm.write_pair_exports(folder, self._pair,
                                          "deviation_%s_vs_%s" % (b, a))
        except OSError as e:
            QMessageBox.warning(self, "Export", str(e))
            return
        QMessageBox.information(self, "Export", "Written:\n" + "\n".join(
            str(p) for p in paths))
