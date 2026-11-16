# -*- coding: utf-8 -*-
"""Preliminary checks on one finished run (Initial run tab).

Before the sizing studies, reads ONE saved run and shows
(gui.sensitivity.preliminary_checks):

* P0 - when Fc, Ff and the zone means of Vx, Vy, T, EVF settle: trailing
  moving mean over L, band tau * eps_q, settling time t_s per quantity and
  the window T it proposes;
* P1 - the attenuation of the field filter (Butterworth, order 2) at the
  Nyquist frequency of the frames, the frame interval a wanted attenuation
  needs, and the change of the window means with one frame in 2, 4, 8;
* chip re-contact - the first frame where material comes back into a band
  of cells above the uncut surface, upstream of the tool;
* the last frame read, as a map (EVF, T, Vx or Vy), with the checks drawn
  on it: the zone, the watched re-contact cells, the chip-thickness line,
  the tool tip and the tool.

Editing a parameter of a check recomputes the checks on the same file (no
new run).

Nothing is run.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui.core.async_call import run_async
from gui.sensitivity import preliminary_checks as pc

_UNITS = {"Vx": "mm/s", "Vy": "mm/s", "T": "°C", "EVF": "–",
          "Fc": "N/mm", "Ff": "N/mm"}
_C_MEAN = "#1d4ed8"
_C_THR = "#c0392b"
_C_TS = "#15803d"
_C_ZONE = "#d97706"
_C_BAND = "#db2777"
_MAP_CMAPS = {"EVF": "Greys", "T": "inferno", "Vx": "coolwarm",
              "Vy": "coolwarm"}


def _fmt(v, f="%.4g"):
    return f % v if v is not None and math.isfinite(v) else "–"


def _num(le, default=None):
    try:
        return float(le.text().strip().replace(",", "."))
    except ValueError:
        return default


class PreliminaryChecksPanel(QWidget):
    """Part of the Initial run tab. `settings_getter()` returns the comparison settings
    {"eps", "window", "zoi", ...} (eps_q and the default zone).

    Left: the inputs, each with a short explanation, the buttons and the
    text summary. Right: one tab of plots per check, each with a caption
    that says what is drawn."""

    # the result dict of each finished analysis (keys "inp", "st", "fc",
    # "rc", ...), for the stop rules of the Initial run tab
    analysed = Signal(object)

    def __init__(self, settings_getter=None, start_dir="", parent=None):
        super().__init__(parent)
        self._settings_getter = settings_getter
        # A path, or a callable returning it (read when a file dialog opens).
        self._start_dir_src = start_dir
        self._result = None
        self._pending = False         # parameters edited during an analysis
        v = QVBoxLayout(self)

        intro = QLabel(
            "Reads one finished run (.results.npz) to fix, before the "
            "sizing studies: the simulated time and the window T (P0), the "
            "number of output frames (P1) and whether the chip comes back "
            "onto the uncut surface. No Abaqus run. Hover a field for more.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6b7280;")
        self._help_labels = [intro]
        top = QHBoxLayout()
        top.addWidget(intro, 1)
        self.cb_help = QCheckBox("Show help")
        self.cb_help.setToolTip("Show or hide the grey explanations of the "
                                "inputs and plots.")
        self.cb_help.setChecked(True)
        self.cb_help.toggled.connect(self._show_help)
        top.addWidget(self.cb_help, 0, Qt.AlignRight | Qt.AlignTop)
        v.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_plots())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 820])
        v.addWidget(split, 1)
        self._fill_zone_from_model()
        for le in (list(self.le_zone.values())
                   + [self.le_len, self.le_tau, self.le_att, self.le_margin,
                      self.le_gap, self.le_band]):
            le.editingFinished.connect(self._on_param_edited)

    # ---- layout --------------------------------------------------------
    def _help(self, text):
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color: #6b7280; font-size: 8.5pt;")
        self._help_labels.append(lab)
        return lab

    def _show_help(self, on):
        for lab in self._help_labels:
            lab.setVisible(on)

    @staticmethod
    def _field(form, label, le, unit, tip):
        le.setToolTip(tip)
        le.setMaximumWidth(90)
        row = QHBoxLayout()
        row.addWidget(le)
        row.addWidget(QLabel(unit))
        row.addStretch(1)
        form.addRow(label, row)

    def _build_left(self):
        inputs = QWidget()
        lv = QVBoxLayout(inputs)
        lv.setContentsMargins(0, 0, 4, 0)

        box = QGroupBox("Run and zone")
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.le_run = QLineEdit()
        self.le_run.setPlaceholderText("<job>.results.npz")
        row.addWidget(self.le_run, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row.addWidget(b)
        f.addRow("Run file", row)
        self.le_zone = {}
        zg = QGridLayout()
        for i, k in enumerate(("xmin", "xmax", "ymin", "ymax")):
            le = QLineEdit()
            le.setPlaceholderText(k)
            le.setToolTip("%s of the zone [mm]" % k)
            self.le_zone[k] = le
            zg.addWidget(QLabel(k), i // 2, 2 * (i % 2))
            zg.addWidget(le, i // 2, 2 * (i % 2) + 1)
        b = QPushButton("Zone = ZOI")
        b.setToolTip("Copy the ZOI of the comparison settings into the zone.")
        b.clicked.connect(self._zone_from_zoi)
        zg.addWidget(b, 0, 4)
        f.addRow("Zone [mm]", zg)
        f.addRow(self._help(
            "The run file gives everything else (element size, filter, tool "
            "and workpiece position). The zone is the rectangle over which "
            "Vx, Vy, T and EVF are averaged at each frame; it starts as the "
            "ZOI (Geometry tab). Before the ZOI is fixed, take a wide zone "
            "around the tool. It must lie inside the extracted region."))
        lv.addWidget(box)

        box = QGroupBox("P0 · stationarity → simulated time and window T")
        f = QFormLayout(box)
        self.le_len = QLineEdit("50")
        self._field(f, "Moving-mean length L", self.le_len, "µs",
                    "Length of the trailing moving mean.")
        self.le_tau = QLineEdit("0.1")
        self._field(f, "Settling band τ", self.le_tau, "× ε_q",
                    "Fraction of the admitted deviation ε_q.")
        f.addRow(self._help(
            "At each instant the mean of the last L of the series is taken "
            "(moving mean). A quantity is settled at t_s when, from t_s to "
            "the end of the run, this moving mean varies by less than τ·ε_q "
            "(e.g. τ = 0.1, ε_q = 10 mm/s: 1 mm/s). Take L at least one "
            "chip-segmentation period. The proposed window T starts at the "
            "largest t_s of the six quantities. Defaults are choices."))
        lv.addWidget(box)

        box = QGroupBox("P1 · output frames → number of frames")
        f = QFormLayout(box)
        self.le_att = QLineEdit("40")
        self._field(f, "Wanted attenuation A", self.le_att, "dB",
                    "Attenuation of the field filter wanted at the Nyquist "
                    "frequency of the frames.")
        f.addRow(self._help(
            "The frames sample the filtered fields every Δt; content above "
            "the Nyquist frequency f_N = 1/(2Δt) folds back (aliasing). The "
            "check gives the attenuation of the field filter (Butterworth, "
            "order 2) at f_N and the largest Δt that reaches A (40 dB = "
            "amplitude / 100). It also recomputes the window means keeping "
            "one frame in 2, 4 and 8. Default A is a choice."))
        lv.addWidget(box)

        box = QGroupBox("Chip re-contact → largest h_void, end of T")
        f = QFormLayout(box)
        self.le_margin = QLineEdit()
        self.le_margin.setPlaceholderText("2 × t1")
        self._field(f, "Margin upstream of the tool", self.le_margin, "mm",
                    "Empty: twice the uncut chip thickness t1 read in the "
                    "run.")
        self.le_gap = QLineEdit("1.5")
        self._field(f, "Gap above the surface", self.le_gap, "elements",
                    "Watched cells start gap element sizes above the "
                    "surface cell.")
        self.le_band = QLineEdit("3")
        self._field(f, "Band height", self.le_band, "elements",
                    "Watched cells: gap ≤ k < gap + band element sizes "
                    "above the surface cell.")
        f.addRow(self._help(
            "Upstream of the tool (x ≤ tool tip − margin), the uncut surface "
            "of each column is its highest full cell at the first frame. "
            "The watched cells lie k element sizes above it, with gap ≤ k < "
            "gap + band (defaults: the 2nd, 3rd and 4th cells). The check "
            "gives the first instant where one of them holds material "
            "(EVF ≥ 0.5): the chip has fallen back onto the surface. The "
            "band must be inside the extracted region."))
        lv.addWidget(box)
        lv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(inputs)

        bottom = QWidget()
        bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        brow = QHBoxLayout()
        self.btn_run = QPushButton("Analyse")
        self.btn_run.clicked.connect(self._analyse)
        brow.addWidget(self.btn_run)
        self.btn_save = QPushButton("Save CSV…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        brow.addWidget(self.btn_save)
        brow.addStretch(1)
        bv.addLayout(brow)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        bv.addWidget(self.lbl_status)
        bv.addWidget(QLabel("Results"))
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QFont("Consolas, monospace"))
        self.txt.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.txt.setPlaceholderText("Choose a run, then Analyse.")
        bv.addWidget(self.txt, 1)

        left = QSplitter(Qt.Vertical)
        left.addWidget(scroll)
        left.addWidget(bottom)
        left.setSizes([480, 320])
        return left

    def _build_plots(self):
        self.tabs = QTabWidget()
        self.figs = {}
        self.canvases = {}
        captions = {
            "p0": ("P0 · stationarity",
                   "One plot per quantity, against time. Grey: the raw "
                   "series (zone mean at each frame for Vx, Vy, T, EVF; "
                   "Fc, Ff = tool reaction force / width at each "
                   "increment). Blue: moving mean over L. Red dashed: final "
                   "moving mean ± τ·ε_q/2 (visual guide; the criterion is "
                   "max − min of the blue curve after t_s ≤ τ·ε_q). Green "
                   "line: t_s of that quantity. Green shading: proposed "
                   "window T, from the largest t_s to the end."),
            "p1": ("P1 · frames",
                   "Left: attenuation of the field filter against frequency "
                   "(Butterworth, order 2). Dotted: cutoff f_c. Blue: "
                   "Nyquist frequency f_N of the frames of this run, with "
                   "the attenuation it gets. Red dashed: wanted A; red "
                   "square: the f_N (and largest frame interval Δt) that "
                   "reaches it. The frames are fine when the blue point is "
                   "on or above the red line. Right: change of the window "
                   "mean when keeping one frame in k, divided by ε_q, per "
                   "quantity; grey dashed line: τ. Bars well under τ mean "
                   "fewer frames would do."),
            "rc": ("Chip re-contact",
                   "Largest EVF over the watched cells (above the uncut "
                   "surface, upstream of the tool) against time. 0: no "
                   "material there. Red dashed: threshold 0.5. Red line: "
                   "first re-contact. Green line: t_s from P0. A re-contact "
                   "after t_s bounds the end of the window T; before t_s, "
                   "h_void must be lowered."),
        }
        captions["map"] = (
            "Last frame",
            "The last frame read (while the run goes on: the latest one), "
            "coloured by the chosen field; Vx, Vy and T only where the "
            "element holds material (EVF ≥ 0.5). Orange: the zone of P0 "
            "and P1. Pink: the cells watched for chip re-contact. Dotted: "
            "the chip-thickness line (y0 of Identify, default t1). Black "
            "line: the tool at that frame; cross: the tool tip of the "
            "model. Editing a check parameter redraws it.")
        for key, (title, cap) in captions.items():
            w = QWidget()
            wl = QVBoxLayout(w)
            wl.setContentsMargins(4, 4, 4, 4)
            lab = self._help(cap)
            wl.addWidget(lab)
            if key == "map":
                row = QHBoxLayout()
                row.addWidget(QLabel("Field"))
                self.cb_field = QComboBox()
                self.cb_field.addItems(["EVF", "T", "Vx", "Vy"])
                self.cb_field.setToolTip("Field drawn on the last frame.")
                self.cb_field.currentTextChanged.connect(self._redraw_map)
                row.addWidget(self.cb_field)
                row.addStretch(1)
                wl.addLayout(row)
            fig = Figure(layout="constrained")
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(250)
            wl.addWidget(NavigationToolbar2QT(canvas, w))
            wl.addWidget(canvas, 1)
            self.figs[key] = fig
            self.canvases[key] = canvas
            self.tabs.addTab(w, title)
        return self.tabs

    @property
    def _start_dir(self) -> str:
        src = self._start_dir_src
        try:
            src = src() if callable(src) else src
        except Exception:
            src = ""
        return str(src or "")

    # ------------------------------------------------------------------
    def _model(self) -> dict:
        if self._settings_getter is None:
            return {}
        try:
            return dict(self._settings_getter() or {})
        except Exception:
            return {}

    def _fill_zone_from_model(self):
        z = self._model().get("zoi")
        if z:
            for k, val in zip(("xmin", "xmax", "ymin", "ymax"), z):
                self.le_zone[k].setText("%.6g" % float(val))

    def _zone_from_zoi(self):
        self._fill_zone_from_model()
        self._on_param_edited()

    def _on_param_edited(self):
        """A check parameter changed: recompute on the file already read
        (nothing is run); after the current analysis if one is running."""
        if self._result is None:
            return
        if not self.btn_run.isEnabled():
            self._pending = True
            return
        self._analyse()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Run results", self._start_dir,
            "Run results (*.results.npz);;All files (*)")
        if path:
            self.le_run.setText(path)

    def _inputs(self):
        path = self.le_run.text().strip()
        if not path or not Path(path).is_file():
            raise ValueError("Choose a .results.npz file.")
        zone = tuple(_num(self.le_zone[k]) for k in
                     ("xmin", "xmax", "ymin", "ymax"))
        if None in zone or not (zone[0] < zone[1] and zone[2] < zone[3]):
            raise ValueError("The zone needs xmin < xmax and ymin < ymax.")
        L = _num(self.le_len)
        tau = _num(self.le_tau)
        att = _num(self.le_att)
        gap, band = _num(self.le_gap), _num(self.le_band)
        if not (L and L > 0 and tau and tau > 0 and att and att > 0
                and gap is not None and gap >= 0 and band and band >= 1):
            raise ValueError("L, τ and A must be > 0, gap ≥ 0 and "
                             "band ≥ 1.")
        eps = {k: float(v) for k, v in
               (self._model().get("eps") or {}).items()}
        if not eps:
            raise ValueError("Set ε_q in Optimization > Identify first.")
        return dict(path=path, zone=zone, L=L * 1e-6, tau=tau, att=att,
                    margin=_num(self.le_margin), gap=gap, band=band, eps=eps)

    def analyse_file(self, path) -> None:
        """Analyse `path` (e.g. the live bundle of a running initial run)
        with the parameters shown; does nothing while an analysis runs."""
        if not self.btn_run.isEnabled():
            return
        self.le_run.setText(str(path))
        self._analyse()

    def _analyse(self):
        try:
            inp = self._inputs()
        except ValueError as e:
            self.lbl_status.setStyleSheet("color: #b91c1c;")
            self.lbl_status.setText(str(e))
            return
        self.btn_run.setEnabled(False)
        self.lbl_status.setStyleSheet("color: #1d4ed8;")
        self.lbl_status.setText("Reading the run…")
        run_async(lambda: self._compute(inp), self._on_done, self)

    @staticmethod
    def _compute(inp):
        from gui.results.reader import ResultsBundle
        out = {"inp": inp}
        try:
            b = ResultsBundle.load(inp["path"])
        except Exception as e:
            return {"error": "cannot read the run: %s" % e}
        try:
            st = pc.stationarity(b, inp["zone"], inp["L"], inp["tau"],
                                 inp["eps"])
            out["st"] = st
            window = st.window or (0.3, 1.0)
            out["window_used"] = window
            out["fc"] = pc.frame_check(b, inp["zone"], window, inp["eps"])
            geo = b.model_config.get("geometry", {})
            tool = (geo.get("tool") or {}).get("position") or {}
            wp = (geo.get("euler") or {}).get("workpiece_position") or {}
            x_tip = float(tool.get("x0", 0.0))
            margin = inp["margin"]
            if margin is None:
                t1 = float(wp.get("y0", 0.0)) - float(tool.get("y0", 0.0))
                margin = 2.0 * abs(t1) if t1 else 0.1
            out["margin"] = margin
            try:
                out["rc"] = pc.chip_recontact(b, x_tip, margin, inp["gap"],
                                              inp["band"])
            except Exception as e:
                out["rc_error"] = str(e)
            out["tip"] = (x_tip, float(tool.get("y0", float("nan"))))
            try:
                out["map"] = pc.frame_map(b)
                out["tool"] = pc.tool_outline(b)
            except Exception as e:
                out["map_error"] = str(e)
            from gui.results.chip_thickness import surface_and_t1
            out["surface"] = surface_and_t1(b.model_config)
        except Exception as e:
            return {"error": str(e)}
        finally:
            try:
                b.close()
            except Exception:
                pass
        return out

    def _on_done(self, res):
        self.btn_run.setEnabled(True)
        if self._pending:
            self._pending = False
            self._analyse()
            if not self.btn_run.isEnabled():
                return          # a newer analysis runs; keep its result
        if not res or "error" in res:
            self.lbl_status.setStyleSheet("color: #b91c1c;")
            self.lbl_status.setText((res or {}).get("error", "failed"))
            return
        self._result = res
        self.btn_save.setEnabled(True)
        self.lbl_status.setStyleSheet("color: #15803d;")
        self.lbl_status.setText("Done.")
        self.txt.setPlainText("\n".join(self._summary(res)))
        self._draw(res)
        self.analysed.emit(res)

    # ------------------------------------------------------------------
    def _summary(self, res):
        st, fc, inp = res["st"], res["fc"], res["inp"]
        lines = ["P0  stationarity  (L = %g µs, band τ·ε_q,"
                 " τ = %g)" % (inp["L"] * 1e6, inp["tau"])]
        for q in pc.QUANTITIES:
            s = st.per_quantity.get(q)
            if s is None:
                lines.append("  %-4s  not available (no data or no ε_q)"
                             % q)
                continue
            ts = ("%.1f µs" % (s.t_s * 1e6) if s.t_s is not None
                  else "not settled")
            lines.append("  %-4s  t_s = %s   final moving mean = %s %s"
                         % (q, ts, _fmt(s.final), _UNITS.get(q, "")))
        if st.t_s is None:
            lines.append("  -> the run is NOT settled: lengthen the simulated "
                         "time or change the model.")
        else:
            lo, hi = st.window
            lines.append("  -> t_s = %.1f µs of t_end = %.1f µs: "
                         "proposed window T = [%.3f, %.3f]%s"
                         % (st.t_s * 1e6, st.t_end * 1e6, lo, hi,
                            "  (SHORTER than 2 L: lengthen the run)"
                            if st.too_short else ""))
        lines.append("")
        lines.append("P1  frames  (dt = %s µs, Nyquist %s kHz)"
                     % (_fmt(fc.dt * 1e6, "%.3g"),
                        _fmt(fc.f_nyquist / 1e3, "%.4g")))
        if fc.cutoff is None:
            lines.append("  field filter OFF in this run: nothing attenuates "
                         "the content above the Nyquist frequency.")
        else:
            need = pc.frame_interval_for(fc.cutoff, inp["att"])
            lines.append("  field filter %s kHz (Butterworth, order %d): "
                         "A = %s dB at the Nyquist of the frames"
                         % (_fmt(fc.cutoff / 1e3), pc.FILTER_ORDER,
                            _fmt(fc.atten_db, "%.1f")))
            lines.append("  A = %g dB needs dt ≤ %s µs  (%s)"
                         % (inp["att"], _fmt(need * 1e6, "%.3g"),
                            "met" if fc.dt <= need * (1 + 1e-9)
                            else "NOT met"))
        w = res["window_used"]
        lines.append("  subsampling over T = [%.3f, %.3f]: |mean(1 frame in "
                     "k) − mean(all)| / ε_q" % w)
        for k, row in sorted(fc.subsampling.items()):
            lines.append("    k = %d   " % k + "   ".join(
                "%s %s" % (q, _fmt(v, "%.3f")) for q, v in row.items()))
        lines.append("")
        if "rc" in res:
            rc = res["rc"]
            lines.append("Chip re-contact  (margin %g mm upstream of the tool"
                         ", %d columns watched, x in [%s, %s] mm)"
                         % (res["margin"], rc.n_columns,
                            _fmt(rc.x_range[0]), _fmt(rc.x_range[1])))
            if rc.note:
                lines.append("  note: " + rc.note)
            lines.append("  first material in the band: %s"
                         % ("none in the run" if rc.t_first is None
                            else "%.1f µs" % (rc.t_first * 1e6)))
            if rc.t_first is not None:
                if st.t_s is not None and rc.t_first > st.t_s:
                    lines.append(
                        "  -> the outputs after %.1f µs are suspect: end the "
                        "window T there, T = [%.3f, %.3f], or lower h_void."
                        % (rc.t_first * 1e6, st.t_s / st.t_end,
                           rc.t_first / st.t_end))
                else:
                    lines.append("  -> the chip comes back before the run "
                                 "settles: lower h_void (or protect the "
                                 "uncut surface).")
        else:
            lines.append("Chip re-contact: %s" % res.get("rc_error", "–"))
        return lines

    def _draw(self, res):
        self._draw_p0(res)
        self._draw_p1(res)
        self._draw_rc(res)
        self._draw_map(res)
        for c in self.canvases.values():
            c.draw_idle()

    def _redraw_map(self, *_):
        if self._result is not None:
            self._draw_map(self._result)
            self.canvases["map"].draw_idle()

    def _draw_map(self, res):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Rectangle
        fig = self.figs["map"]
        fig.clear()
        ax = fig.subplots()
        fm = res.get("map")
        q = self.cb_field.currentText()
        if fm is None or q not in fm.values:
            ax.text(0.5, 0.5, res.get("map_error") or "%s not in the run" % q,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8)
            ax.set_axis_off()
            return
        data = np.ma.masked_invalid(fm.values[q])
        kw = {"cmap": _MAP_CMAPS.get(q, "viridis")}
        if q == "EVF":
            kw.update(vmin=0.0, vmax=1.0)
        elif q in ("Vx", "Vy") and data.count():
            m = float(np.max(np.abs(data)))
            kw.update(vmin=-m, vmax=m)
        mesh = ax.pcolormesh(fm.x_edges, fm.y_edges, data, shading="flat",
                             **kw)
        fig.colorbar(mesh, ax=ax, label="%s [%s]" % (q, _UNITS.get(q, "")))
        inp = res["inp"]
        z = inp["zone"]
        ax.add_patch(Rectangle((z[0], z[2]), z[1] - z[0], z[3] - z[2],
                               fill=False, edgecolor=_C_ZONE, lw=1.6,
                               label="zone (P0, P1)"))
        rc = res.get("rc")
        if rc is not None and rc.cells is not None and len(rc.cells):
            band = np.isin(fm.index, np.asarray(rc.cells)) & (fm.index >= 0)
            ax.pcolormesh(fm.x_edges, fm.y_edges,
                          np.ma.masked_where(~band, band.astype(float)),
                          shading="flat", cmap=ListedColormap([_C_BAND]),
                          vmin=0, vmax=1, alpha=0.5)
            ax.plot([], [], "s", color=_C_BAND, alpha=0.6,
                    label="watched re-contact cells")
        ys, t1 = res.get("surface") or (None, None)
        y0 = self._model().get("chip_y0")
        y0 = t1 if y0 is None else y0
        if ys is not None and y0 is not None:
            ax.axhline(ys + float(y0), color=_C_ZONE, ls=":", lw=1.0,
                       label="chip-thickness line")
        segs = [np.column_stack(p) for p in res.get("tool") or []]
        if segs:
            ax.add_collection(LineCollection(segs, colors="black",
                                             linewidths=0.9, label="tool"))
        tip = res.get("tip")
        if tip and all(math.isfinite(v) for v in tip):
            ax.plot([tip[0]], [tip[1]], "x", color="black", ms=7, mew=1.5,
                    label="tool tip (model)")
        ax.set_xlim(fm.x_edges[0], fm.x_edges[-1])
        ax.set_ylim(fm.y_edges[0], fm.y_edges[-1])
        ax.set_aspect("equal")
        ax.set_title("%s at frame %d, t = %.1f µs"
                     % (q, fm.frame, fm.time * 1e6), fontsize=9)
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("y [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
                   ncol=3, fontsize=7.5, frameon=False)

    def _draw_p0(self, res):
        fig = self.figs["p0"]
        fig.clear()
        st = res["st"]
        axes = fig.subplots(3, 2, sharex=True)
        for ax, q in zip(axes.ravel(), pc.QUANTITIES):
            s = st.per_quantity.get(q)
            unit = _UNITS.get(q, "")
            ax.set_title("%s [%s]" % (q, unit), fontsize=9, pad=3)
            ax.tick_params(labelsize=7)
            if s is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            t = s.times * 1e6
            if st.t_s is not None:
                ax.axvspan(st.t_s * 1e6, st.t_end * 1e6, color=_C_TS,
                           alpha=0.08, lw=0, label="proposed window T")
            ax.plot(t, s.values, color="#9ca3af", lw=0.6, label="raw series")
            ax.plot(t, s.mean, color=_C_MEAN, lw=1.2,
                    label="moving mean over L")
            if math.isfinite(s.final):
                for i, sg in enumerate((-0.5, 0.5)):
                    ax.axhline(s.final + sg * s.tol, color=_C_THR, ls="--",
                               lw=0.8, label=None if i else
                               "final ± τ·ε_q/2")
            if s.t_s is not None:
                ax.axvline(s.t_s * 1e6, color=_C_TS, lw=1,
                           label="t_s (settled from here)")
        for a in axes[-1]:
            a.set_xlabel("t [µs]", fontsize=8)
        h, lab = [], []
        for a in axes.ravel():
            for hh, ll in zip(*a.get_legend_handles_labels()):
                if ll not in lab:
                    h.append(hh)
                    lab.append(ll)
        if h:
            fig.legend(h, lab, loc="outside upper center", ncol=3,
                       fontsize=7.5, frameon=False)

    def _draw_p1(self, res):
        fig = self.figs["p1"]
        fig.clear()
        fc, inp = res["fc"], res["inp"]
        ax, bx = fig.subplots(1, 2)
        if fc.cutoff is None:
            ax.text(0.5, 0.5, "field filter OFF in this run:\nnothing "
                    "attenuates the content above f_N", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
        else:
            f_a = 1.0 / (2.0 * pc.frame_interval_for(fc.cutoff,
                                                     inp["att"]))
            hi = max(fc.f_nyquist, fc.cutoff, f_a) * 2
            f = np.logspace(math.log10(fc.cutoff / 10), math.log10(hi), 300)
            ax.semilogx(f / 1e3, [pc.butterworth_attenuation_db(x, fc.cutoff)
                                  for x in f],
                        color="#374151", lw=1.2, label="filter attenuation")
            ax.axvline(fc.cutoff / 1e3, color="#6b7280", ls=":", lw=1,
                       label="cutoff f_c = %s kHz" % _fmt(fc.cutoff / 1e3))
            ax.axhline(inp["att"], color=_C_THR, ls="--", lw=0.9,
                       label="wanted A = %g dB" % inp["att"])
            ax.plot([f_a / 1e3], [inp["att"]], "s", color=_C_THR, ms=5,
                    label="f_N needed ≥ %s kHz (Δt ≤ %s µs)"
                    % (_fmt(f_a / 1e3), _fmt(5e5 / f_a, "%.3g")))
            ax.plot([fc.f_nyquist / 1e3], [fc.atten_db], "o", color=_C_MEAN,
                    label="frames: f_N = %s kHz, %s dB"
                    % (_fmt(fc.f_nyquist / 1e3), _fmt(fc.atten_db, "%.1f")))
            ax.axvline(fc.f_nyquist / 1e3, color=_C_MEAN, lw=0.8)
            ax.set_xlabel("frequency [kHz]", fontsize=8)
            ax.set_ylabel("attenuation [dB]", fontsize=8)
            ax.set_title("Field filter at the frame Nyquist frequency",
                         fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
            ax.tick_params(labelsize=7)
        ks = sorted(fc.subsampling)
        qs = [q for q in pc.QUANTITIES
              if any(q in fc.subsampling[k] for k in ks)]
        if ks and qs:
            width = 0.8 / len(ks)
            x = np.arange(len(qs))
            for i, k in enumerate(ks):
                y = [fc.subsampling[k].get(q, np.nan) for q in qs]
                bx.bar(x + (i - (len(ks) - 1) / 2) * width, y, width,
                       label="1 frame in %d" % k)
            bx.axhline(inp["tau"], color="#6b7280", ls="--", lw=0.9,
                       label="τ = %g" % inp["tau"])
            bx.set_xticks(x, qs)
            bx.legend(fontsize=7)
        bx.set_ylabel("|mean(1 in k) − mean(all)| / ε_q", fontsize=8)
        bx.set_title("Subsampling over T = [%.3f, %.3f]"
                     % res["window_used"], fontsize=9)
        bx.tick_params(labelsize=7)

    def _draw_rc(self, res):
        fig = self.figs["rc"]
        fig.clear()
        ax = fig.subplots()
        rc = res.get("rc")
        st = res["st"]
        ax.set_title("Largest EVF in the watched cells", fontsize=9)
        if rc is not None and np.isfinite(rc.band_max_evf).any():
            ax.plot(rc.times * 1e6, rc.band_max_evf, color=_C_MEAN,
                    label="largest EVF in the band")
            ax.axhline(0.5, color=_C_THR, ls="--", lw=0.8,
                       label="threshold 0.5")
            if rc.t_first is not None:
                ax.axvline(rc.t_first * 1e6, color=_C_THR, lw=1,
                           label="first re-contact %.1f µs"
                           % (rc.t_first * 1e6))
            if st.t_s is not None:
                ax.axvline(st.t_s * 1e6, color=_C_TS, lw=1,
                           label="t_s (P0) %.1f µs" % (st.t_s * 1e6))
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7.5)
        else:
            msg = (rc.note if rc is not None and rc.note
                   else res.get("rc_error", "no watched cell"))
            ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True,
                    transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("t [µs]", fontsize=8)
        ax.set_ylabel("EVF [–]", fontsize=8)
        ax.tick_params(labelsize=7)

    def _save(self):
        if self._result is None:
            return
        res = self._result
        default = str(Path(res["inp"]["path"]).with_suffix("")) + \
            "_preliminary.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default,
                                              "CSV (*.csv)")
        if not path:
            return
        from gui.sensitivity.study_export import write_csv
        rows = []
        for q, s in res["st"].per_quantity.items():
            rows.append({"check": "P0", "quantity": q,
                         "t_s_us": None if s.t_s is None else s.t_s * 1e6,
                         "band": s.tol, "final_moving_mean": s.final})
        fc = res["fc"]
        rows.append({"check": "P1", "quantity": "frames",
                     "dt_us": fc.dt * 1e6, "atten_db": fc.atten_db,
                     "cutoff_hz": fc.cutoff})
        for k, row in fc.subsampling.items():
            for q, val in row.items():
                rows.append({"check": "P1 subsampling", "quantity": q,
                             "k": k, "dev_over_eps": val})
        rc = res.get("rc")
        if rc is not None:
            rows.append({"check": "chip re-contact", "quantity": "EVF band",
                         "t_first_us": None if rc.t_first is None
                         else rc.t_first * 1e6,
                         "columns": rc.n_columns, "note": rc.note})
        try:
            write_csv(path, rows)
        except OSError as e:
            QMessageBox.warning(self, "Save CSV", str(e))
            return
        self.lbl_status.setText("Saved %s (+ .xlsx)." % path)
