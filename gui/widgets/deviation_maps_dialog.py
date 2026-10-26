# -*- coding: utf-8 -*-
"""Deviation maps between two runs of the sizing studies (Model tab).

Shows, for a reference run and a compared run of the mass-scaling, mesh or
domain studies, where in the ZOI and when in the time window T the two runs
differ (gui.sensitivity.deviation_maps):

* a map of |q_b - q_a| / eps_q on the ZOI grid: at one frame, averaged over
  T, or as the difference of the time means (bias);
* a click on a ZOI point plots, right of the map, q_a(t) and q_b(t) at that
  point and, below them, |q_b - q_a| / eps_q at that point over time, with
  the spatial mean over the ZOI per frame and the threshold of the step;
* for the forces, the two force curves and |F_b - F_a| / eps over time.

Each study folder is one entry of the Study list and the run lists (a), (b)
only hold the runs of the chosen entry; "All the folders" lists the runs of
every folder, grouped by folder, to compare runs made in different folders.
The run chosen in one list is greyed out in the other one.

Nothing is run: the saved results of the study folders are read.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
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
_ALL = "__all__"                    # Study entry listing every folder
_FILES = "__files__"                # Study entry of the result files added
_BASE_TEXT = Qt.UserRole + 1        # run label without the "taken" note
_C_THR = "#c0392b"                  # threshold of the step (red)


def _fmt(v, f="%.3f"):
    return f % v if v is not None and math.isfinite(v) else "–"


def _plot_panel(canvas):
    """A widget holding the matplotlib toolbar over `canvas`."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    tb = NavigationToolbar2QT(canvas, w)
    v.addWidget(tb)
    v.addWidget(canvas, 1)
    return w, tb


class DeviationMapsDialog(QDialog):
    """Non-modal window; `folders` are the study folders listed at start.
    `settings_getter` returns the Model tab's current comparison settings
    (dm.CompareSettings), used when the folder of the reference run is not
    a study of the Model tab (e.g. a Sensitivity-tab folder)."""

    def __init__(self, folders=(), start_dir="", parent=None,
                 settings_getter=None):
        super().__init__(parent)
        self.setWindowTitle("Deviation maps between two runs")
        self._fit_to_screen(parent)
        self._folders = []               # study folders, in the Study list
        self._folder_runs = {}           # str(folder) -> its finished runs
        self._runs = []                  # runs of the chosen Study entry
        self._pair = None
        self._pair_jobs = ("a", "b")
        self._point = None
        self._busy = False
        self._start_dir = str(start_dir or "")
        self._settings_getter = settings_getter

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Where and when two runs of a study differ. Pick the study "
            "folder (or add result files), the reference run (a) and the "
            "compared run (b), then "
            "Compare. The value shown is |b − a| / ε_q: 1 means a deviation "
            "equal to the admitted deviation ε_q. Click a point of the map "
            "to see the two curves there and their deviation over time.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Study folder"))
        self.cb_study = QComboBox()
        self.cb_study.setToolTip(
            "One entry per study folder: the lists (a) and (b) hold the\n"
            "finished runs of the chosen folder only. 'All the folders'\n"
            "lists the runs of every folder, grouped by folder (e.g. to\n"
            "compare a run with the same run made again elsewhere).")
        self.cb_study.currentIndexChanged.connect(self._on_study)
        row.addWidget(self.cb_study, 1)
        b = QPushButton("Add a folder…")
        b.setToolTip("Add a study folder to the list and select it.")
        b.clicked.connect(self._on_add_folder)
        row.addWidget(b)
        b = QPushButton("Add result files…")
        b.setToolTip(
            "Add single runs by their result file (<job>.results.npz or\n"
            "<job>.meta.json; both must be in the same folder), e.g. two\n"
            "runs made with the same parameters to show their\n"
            "repeatability. They are listed under 'Result files'. Outside a\n"
            "study folder, the comparison uses the Model sizing tab's\n"
            "settings (ZOI, sampling step, T, ε_q).")
        b.clicked.connect(self._on_add_files)
        row.addWidget(b)
        lay.addLayout(row)
        self.lbl_folders = QLabel("No study folder.")
        self.lbl_folders.setWordWrap(True)
        self.lbl_folders.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.lbl_folders)

        row = QHBoxLayout()
        row.addWidget(QLabel("Reference (a)"))
        self.cb_a = QComboBox()
        self.cb_a.setMinimumContentsLength(30)
        self.cb_a.currentIndexChanged.connect(lambda *_: self._on_pick("a"))
        row.addWidget(self.cb_a, 1)
        row.addWidget(QLabel("Compared (b)"))
        self.cb_b = QComboBox()
        self.cb_b.setMinimumContentsLength(30)
        self.cb_b.currentIndexChanged.connect(lambda *_: self._on_pick("b"))
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

        # Map on the left over the whole height; on the right, the two
        # curves at the clicked point above their deviation over time with
        # the threshold. The "compressed" layout keeps the colorbar next to
        # an equal-aspect map instead of leaving the map against the right
        # edge.
        self.fig_map = Figure(figsize=(5, 5), layout="compressed")
        self.canvas_map = FigureCanvas(self.fig_map)
        self.canvas_map.mpl_connect("button_press_event", self._on_click)
        self.fig_curves = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas_curves = FigureCanvas(self.fig_curves)
        self.fig_dev = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas_dev = FigureCanvas(self.fig_dev)
        w_map, self._map_toolbar = _plot_panel(self.canvas_map)
        w_curves, _ = _plot_panel(self.canvas_curves)
        w_dev, _ = _plot_panel(self.canvas_dev)
        right = QSplitter(Qt.Vertical)
        right.addWidget(w_curves)
        right.addWidget(w_dev)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(w_map)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([600, 650])
        lay.addWidget(split, 1)

        self.cb_study.blockSignals(True)
        self.cb_study.addItem("(add a study folder)", None)
        self.cb_study.blockSignals(False)
        for f in folders or ():
            self.add_folder(f, quiet=True, select=False)
        if self._folders:
            self.cb_study.setCurrentIndex(self._study_index(self._folders[-1]))
        self._redraw()

    def _fit_to_screen(self, parent):
        """Open at 1250 x 900, or at 90 % of the available screen when that
        is smaller (a 1920 x 1080 screen at 125-150 % scaling has only
        about 700-830 px of height), centred on the screen."""
        screen = (parent.screen() if parent is not None else None) \
            or QApplication.primaryScreen()
        if screen is None:
            self.resize(1250, 900)
            return
        g = screen.availableGeometry()
        w = min(1250, int(0.9 * g.width()))
        h = min(900, int(0.9 * g.height()))
        self.resize(w, h)
        self.move(g.x() + (g.width() - w) // 2, g.y() + (g.height() - h) // 2)

    # ---- study folders -------------------------------------------------
    def _study_index(self, folder) -> int:
        return self.cb_study.findData(str(folder))

    def _rebuild_study_list(self, select=None):
        """One entry per folder, plus 'All the folders' when there are
        several; `select` (a folder or _ALL) is made current."""
        keep = self.cb_study.currentData() if select is None else str(select)
        self.cb_study.blockSignals(True)
        self.cb_study.clear()
        files = self._folder_runs.get(_FILES, [])
        if not self._folders and not files:
            self.cb_study.addItem("(add a study folder or result files)", None)
        for f in self._folders:
            n = len(self._folder_runs.get(str(f), ()))
            self.cb_study.addItem("%s  (%d finished run%s)"
                                  % (f.name, n, "" if n == 1 else "s"),
                                  str(f))
            self.cb_study.setItemData(self.cb_study.count() - 1, str(f),
                                      Qt.ToolTipRole)
        if files:
            self.cb_study.addItem("Result files  (%d)" % len(files), _FILES)
        n_groups = len(self._folders) + (1 if files else 0)
        if n_groups > 1:
            self.cb_study.addItem("All the folders (%d)" % n_groups, _ALL)
        i = self.cb_study.findData(keep) if keep is not None else -1
        self.cb_study.setCurrentIndex(max(i, 0))
        self.cb_study.blockSignals(False)
        self._on_study()

    def add_folder(self, folder, quiet=False, select=True) -> int:
        """Add a study folder to the Study list (and select it); returns
        its number of finished runs. A folder already listed is only
        selected; a folder without a finished run is not added."""
        folder = Path(folder)
        if not folder.is_dir():
            return 0
        for f in self._folders:
            if f.resolve() == folder.resolve():
                if select:
                    self.cb_study.setCurrentIndex(self._study_index(f))
                return len(self._folder_runs.get(str(f), ()))
        runs = dm.list_runs([folder])
        if not runs:
            if not quiet:
                QMessageBox.information(
                    self, "Add a study folder",
                    "%s holds no finished run (a run needs its .results.npz,"
                    " .meta.json and a .sta that reports a completed "
                    "analysis). The folder is not added." % folder.name)
            return 0
        self._folders.append(folder)
        self._folder_runs[str(folder)] = runs
        self._rebuild_study_list(select=folder if select else None)
        return len(runs)

    def _on_add_folder(self):
        cur = self.cb_study.currentData()
        start = (str(Path(cur).parent) if cur and cur != _ALL
                 else str(self._folders[-1].parent) if self._folders
                 else self._start_dir)
        folder = QFileDialog.getExistingDirectory(
            self, "Add a study folder", start)
        if folder:
            self.add_folder(folder)

    def add_files(self, paths, quiet=False) -> int:
        """Add single result files to the 'Result files' entry (and select
        it); returns the number of files added. A file already listed or
        not a <job>.results.npz is skipped."""
        files = self._folder_runs.setdefault(_FILES, [])
        known = {Path(r.npz).resolve() for r in files}
        added, refused = 0, []
        for p in paths or ():
            try:
                run = dm.run_of_file(p)
            except ValueError as e:
                refused.append(str(e))
                continue
            if Path(run.npz).resolve() in known:
                continue
            known.add(Path(run.npz).resolve())
            files.append(run)
            added += 1
        if refused and not quiet:
            QMessageBox.information(self, "Add result files",
                                    "\n".join(refused))
        if added:
            self._rebuild_study_list(select=_FILES)
        return added

    def _on_add_files(self):
        files = self._folder_runs.get(_FILES, [])
        start = (str(files[-1].folder) if files
                 else str(self._folders[-1]) if self._folders
                 else self._start_dir)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add result files", start,
            "Result files (*.results.npz *.meta.json);;All files (*)")
        if paths:
            self.add_files(paths)

    def _on_study(self, *_):
        key = self.cb_study.currentData()
        if key == _ALL:
            folders = list(self._folders)
            if self._folder_runs.get(_FILES):
                folders.append(_FILES)
        elif key:
            folders = [key if key == _FILES else Path(key)]
        else:
            folders = []
        self._runs = [r for f in folders
                      for r in self._folder_runs.get(str(f), ())]
        if not folders:
            self.lbl_folders.setText("No study folder.")
        elif key == _FILES:
            self.lbl_folders.setText(
                "Result files: %s" % ", ".join(str(r.npz) for r in self._runs))
        elif key == _ALL:
            self.lbl_folders.setText(
                "Runs of every folder, grouped by folder (%d finished runs)."
                % len(self._runs))
        else:
            self.lbl_folders.setText("Folder: %s (%d finished runs)"
                                     % (folders[0], len(self._runs)))
        self._refill_runs(grouped=key == _ALL)

    # ---- runs ----------------------------------------------------------
    def _refill_runs(self, grouped=False):
        bold = QFont()
        bold.setBold(True)
        for cb in (self.cb_a, self.cb_b):
            cb.blockSignals(True)
            cb.clear()
            group = None
            for i, r in enumerate(self._runs):
                g = self._group_of(r)
                if grouped and g != group:
                    group = g
                    cb.addItem("— %s —" % g, None)
                    item = cb.model().item(cb.count() - 1)
                    item.setEnabled(False)
                    item.setFont(bold)
                text = self._label(r, folder=False)
                cb.addItem(text, i)
                cb.setItemData(cb.count() - 1, text, _BASE_TEXT)
            cb.blockSignals(False)
        if not self._runs:
            return
        step = None
        for f in dict.fromkeys(r.folder for r in self._runs):
            try:
                step = dm.settings_of_folder(f).step
                break
            except Exception:
                continue
        ia = max(dm.default_reference(self._runs, step), 0)
        ib = 1 if ia == 0 and len(self._runs) > 1 else 0
        for cb, i in ((self.cb_a, ia), (self.cb_b, ib)):
            cb.blockSignals(True)
            cb.setCurrentIndex(cb.findData(i))
            cb.blockSignals(False)
        self._mark_taken()

    def _is_file(self, run) -> bool:
        return any(r is run for r in self._folder_runs.get(_FILES, ()))

    def _group_of(self, run) -> str:
        return "Result files" if self._is_file(run) else run.folder.name

    def _label(self, run, folder=True) -> str:
        """dm.run_label, or the file name when the run's parameters are
        unknown (a result file without its .meta.json)."""
        if run.params:
            text = dm.run_label(run, folder=folder)
        else:
            text = "%s | parameters unknown (none in its .meta.json)" % run.job
            if folder:
                text += " (%s)" % run.folder.name
        return text

    def _run_index(self, cb):
        i = cb.currentData()
        return i if isinstance(i, int) else None

    def _on_pick(self, which):
        """Keep (a) and (b) on two different runs: picking in one list the
        run of the other moves the other one to the next run."""
        cb, other = ((self.cb_a, self.cb_b) if which == "a"
                     else (self.cb_b, self.cb_a))
        i = self._run_index(cb)
        if i is not None and i == self._run_index(other) \
                and len(self._runs) > 1:
            j = (i + 1) % len(self._runs)
            other.blockSignals(True)
            other.setCurrentIndex(other.findData(j))
            other.blockSignals(False)
        self._mark_taken()

    def _mark_taken(self):
        """Grey out, in each list, the run chosen in the other one."""
        for cb, other, note in ((self.cb_a, self.cb_b, "already b"),
                                (self.cb_b, self.cb_a, "already a")):
            taken = self._run_index(other)
            model = cb.model()
            for k in range(cb.count()):
                i = cb.itemData(k)
                if not isinstance(i, int):
                    continue
                base = cb.itemData(k, _BASE_TEXT)
                item = model.item(k)
                if i == taken:
                    item.setText("%s   [%s]" % (base, note))
                    item.setEnabled(False)
                else:
                    item.setText(base)
                    item.setEnabled(True)

    def _settings_for(self, run):
        """Settings of the reference run's folder, else the Model tab's
        current ones; (settings, None) or (None, reason)."""
        try:
            return dm.settings_of_folder(run.folder), None
        except Exception as e:
            err = str(e)
        if self._settings_getter is not None:
            try:
                s = self._settings_getter()
                if s is not None:
                    return s, None
            except Exception as e:
                err += "; Model sizing tab: %s" % e
        return None, err

    # ---- comparison ----------------------------------------------------
    def _on_compare(self):
        if self._busy or not self._runs:
            return
        ia, ib = self._run_index(self.cb_a), self._run_index(self.cb_b)
        if ia is None or ib is None or ia == ib:
            QMessageBox.information(self, "Compare",
                                    "Pick two different runs.")
            return
        ra, rb = self._runs[ia], self._runs[ib]
        settings, err = self._settings_for(ra)
        if settings is None:
            QMessageBox.warning(
                self, "Compare", "No comparison settings (ZOI, sampling step, "
                "window, ε_q) found: %s" % err)
            return
        self._busy = True
        self.btn_compare.setEnabled(False)
        self.lbl_summary.setText("Reading the two runs…")

        def work():
            from gui.results.reader import ResultsBundle
            try:
                measured = []
                with ResultsBundle.load(ra.npz) as ba, \
                        ResultsBundle.load(rb.npz) as bb:
                    sizes = []
                    for run, bundle in ((ra, ba), (rb, bb)):
                        h = dm.run_elem_size(run.params)
                        if not math.isfinite(h):
                            h = dm.bundle_elem_size(bundle)
                            measured.append("%s: %g mm" % (run.job, h))
                        sizes.append(h)
                    pair = dm.compare_runs(ba, bb, settings, *sizes)
                pair.label_a, pair.label_b = self._label(ra), self._label(rb)
                pair.measured_sizes = measured
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

    def _threshold(self) -> float:
        """Threshold of the step in units of ε_q (its share of ε_q)."""
        return float(self._pair.settings.share) if self._pair else 1.0

    def _write_summary(self):
        p = self._pair
        cells = ["%s %s / %s / %s" % (q, _fmt(d.e_mean), _fmt(d.e_inst),
                                      _fmt(d.e_bias))
                 for q, d in p.quantities.items()]
        s = p.settings
        self.lbl_summary.setText(
            "<b>E/ε of the means (decides the studies) / E/ε (all samples)"
            " / bias E/ε (time means per point):</b> %s<br>"
            "Threshold of the step: %s ε_q (its share of ε_q).<br>"
            "a = %s<br>b = %s<br>ZOI x[%g, %g] y[%g, %g] mm, step %g mm, "
            "T = %g–%g of the simulated time, sampling %s, largest frame "
            "offset %s of a frame interval. Settings: %s."
            % ("&nbsp;&nbsp; ".join(cells), _fmt(s.share, "%.3g"),
               p.label_a, p.label_b,
               *s.zoi, s.grid_step, s.window[0], s.window[1],
               "linear (different meshes)" if p.linear else
               "nearest element (as the studies)",
               _fmt(p.frame_offset_over_interval, "%.3g"),
               s.source or "–")
            + ("<br>Element size measured on the mesh (not in the "
               ".meta.json): %s." % ", ".join(p.measured_sizes)
               if p.measured_sizes else "")
            + ("<br><b>The two runs are identical: |b − a| = 0 on every "
               "compared sample of every output.</b>" if self._identical()
               else ""))

    def _identical(self) -> bool:
        """True when b equals a on every compared sample of every output
        (e.g. a run and its exact repeat)."""
        qs = list(self._pair.quantities.values()) if self._pair else []
        return bool(qs) and all(
            np.array_equal(np.isfinite(d.a), np.isfinite(d.b))
            and np.all(np.nan_to_num(d.a) == np.nan_to_num(d.b))
            for d in qs)

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
        self._draw_deviation()

    @staticmethod
    def _message(fig, canvas, text):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
                transform=ax.transAxes)
        ax.set_axis_off()
        canvas.draw_idle()

    def _draw_map(self):
        fig = self.fig_map
        fig.clear()
        d = self._current()
        self.lbl_frame.setText("")
        if d is None:
            self._message(fig, self.canvas_map,
                          "Pick the study folder and two runs, then Compare.")
            return
        if not d.is_field:
            self._message(fig, self.canvas_map,
                          "%s is a force: no map, see the curves." % d.quantity)
            return
        ax = fig.add_subplot(111)
        p = self._pair
        ny, nx = p.shape
        vals = self._map_values(d).reshape(ny, nx)
        g = p.settings.grid_step
        xs = p.points[:nx, 0]
        ys = p.points[::nx, 1]
        extent = (xs[0] - g / 2, xs[-1] + g / 2, ys[0] - g / 2, ys[-1] + g / 2)
        finite = vals[np.isfinite(vals)]
        thr = self._threshold()
        vmax = max(1.0, float(np.percentile(finite, 99))) if finite.size \
            else 1.0
        import matplotlib
        cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#d0d0d0")
        im = ax.imshow(vals, origin="lower", extent=extent, cmap=cmap,
                       vmin=0.0, vmax=vmax, interpolation="nearest",
                       aspect="equal")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("|b − a| / ε_%s   (ε_%s = %g %s)"
                     % (d.quantity, d.quantity, d.eps,
                        _UNITS.get(d.quantity, "")))
        if finite.size and finite.max() > thr and ny > 1 and nx > 1:
            ax.contour(xs, ys, np.where(np.isfinite(vals), vals, 0.0),
                       levels=[thr], colors=_C_THR, linewidths=1.0)
        if self._point is not None:
            x, y = p.points[self._point]
            ax.plot([x], [y], marker="s", mfc="none", mec="white", mew=2, ms=10)
        view = self.cb_view.currentData()
        title = dict(_VIEWS)[view].replace("ε", "ε_" + d.quantity)
        if view == "frame":
            k = self.sl_frame.value()
            self.lbl_frame.setText("t = %.4g s" % d.times[k])
            title += "  (t = %.4g s)" % d.times[k]
        ax.set_title("%s\nmap mean %s · grey: not compared · red line: "
                     "threshold %s" % (title, _fmt(float(np.mean(finite))
                                                   if finite.size else None),
                                       _fmt(thr, "%.3g")),
                     fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        self.canvas_map.draw_idle()

    def _draw_curves(self):
        """The two runs: force histories, or the field at the clicked
        point."""
        fig = self.fig_curves
        fig.clear()
        d = self._current()
        if d is None:
            self.canvas_curves.draw_idle()
            return
        if d.is_field and self._point is None:
            self._message(fig, self.canvas_curves,
                          "Click a point of the map to plot %s of the two "
                          "runs there." % d.quantity)
            return
        ax = fig.add_subplot(111)
        unit = _UNITS.get(d.quantity, "")
        if not d.is_field:
            ax.plot(d.times, d.a, lw=0.8, label="a (reference)")
            ax.plot(d.times, d.b, lw=0.8, label="b")
            ax.set_title("%s of the two runs over T" % d.quantity, fontsize=9)
        else:
            i = self._point
            x, y = self._pair.points[i]
            ax.plot(d.times, d.a[:, i], lw=1.0, marker=".", ms=3,
                    label="a (reference)")
            ax.plot(d.times, d.b[:, i], lw=1.0, marker=".", ms=3, label="b")
            ax.set_title("%s at x = %.4g, y = %.4g mm (gaps: void)"
                         % (d.quantity, x, y), fontsize=9)
            if self.cb_view.currentData() == "frame":
                ax.axvline(d.times[self.sl_frame.value()], color="grey",
                           lw=0.8, ls="--")
        ax.set_ylabel("%s [%s]" % (d.quantity, unit))
        ax.set_xlabel("time [s]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, linestyle=":")
        self.canvas_curves.draw_idle()

    def _draw_deviation(self):
        """|b − a| / ε over time (at the clicked point and its mean over
        the ZOI, or the force), with the threshold of the step."""
        fig = self.fig_dev
        fig.clear()
        d = self._current()
        if d is None:
            self.canvas_dev.draw_idle()
            return
        ax = fig.add_subplot(111)
        thr = self._threshold()
        if not d.is_field:
            ax.plot(d.times, d.e_frame, lw=0.8, color="C2",
                    label="|%s_b − %s_a| / ε" % (d.quantity, d.quantity))
            title = "%s: deviation over T" % d.quantity
        else:
            ax.plot(d.times, d.e_frame, lw=1.0, color="0.55",
                    label="mean over the ZOI of |b − a| / ε")
            title = "%s: deviation per frame" % d.quantity
            if self._point is not None:
                i = self._point
                x, y = self._pair.points[i]
                ax.plot(d.times, d.point_deviation(i), lw=1.0, marker=".",
                        ms=3, color="C2",
                        label="|b − a| / ε at x = %.4g, y = %.4g mm" % (x, y))
                title += (" at the point: mean %s, |b̄ − ā| / ε %s, %d "
                          "frames" % (_fmt(d.mean_map()[i]),
                                         _fmt(d.bias_map()[i]),
                                         d.n_compared()[i]))
            if self.cb_view.currentData() == "frame":
                ax.axvline(d.times[self.sl_frame.value()], color="grey",
                           lw=0.8, ls="--")
        ax.axhline(thr, color=_C_THR, lw=1.0,
                   label="threshold of the step = %s ε" % _fmt(thr, "%.3g"))
        if not math.isclose(thr, 1.0):
            ax.axhline(1.0, color=_C_THR, lw=0.8, ls=":", label="ε_q")
        ax.set_title("%s\nvalue that decides the study: %s"
                     % (title, _fmt(d.e_mean)), fontsize=9)
        ax.set_ylabel("|b − a| / ε_%s [-]" % d.quantity)
        ax.set_xlabel("time [s]")
        # Room above the threshold for the legend.
        ax.set_ylim(0.0, max(1.6 * thr, 1.15, 1.15 * ax.get_ylim()[1]))
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
        ax.grid(True, alpha=0.25, linestyle=":")
        self.canvas_dev.draw_idle()

    def _on_click(self, event):
        d = self._current()
        if (d is None or not d.is_field or event.inaxes is None
                or event.xdata is None
                or event.inaxes is not self.fig_map.axes[0]
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
