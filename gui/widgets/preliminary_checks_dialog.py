# -*- coding: utf-8 -*-
"""Preliminary checks on one finished run (Model tab).

Before the sizing studies, reads ONE saved run and shows
(gui.sensitivity.preliminary_checks):

* P0 - when Fc, Ff and the zone means of Vx, Vy, T, EVF settle: trailing
  moving mean over L, band tau * eps_q, settling time t_s per quantity and
  the window T it proposes;
* P1 - the attenuation of the field filter (Butterworth, order 2) at the
  Nyquist frequency of the frames, the frame interval a wanted attenuation
  needs, and the change of the window means with one frame in 2, 4, 8;
* chip re-contact - the first frame where material comes back into a band
  of cells above the uncut surface, upstream of the tool.

Nothing is run.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
    QVBoxLayout, QWidget)

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


def _fmt(v, f="%.4g"):
    return f % v if v is not None and math.isfinite(v) else "–"


def _num(le, default=None):
    try:
        return float(le.text().strip().replace(",", "."))
    except ValueError:
        return default


class PreliminaryChecksDialog(QDialog):
    """Non-modal window. `settings_getter()` returns the Model tab settings
    {"eps", "window", "zoi", ...} (eps_q and the default zone)."""

    def __init__(self, settings_getter=None, start_dir="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preliminary checks on one run")
        self.resize(1200, 900)
        self._settings_getter = settings_getter
        self._start_dir = start_dir
        self._result = None
        v = QVBoxLayout(self)

        intro = QLabel(
            "Reads one finished run (.results.npz) to fix, before the "
            "studies: the simulated time and the window T (P0), the number "
            "of output frames (P1) and whether the chip comes back onto the "
            "uncut surface. No Abaqus run.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6b7280;")
        v.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Run:"))
        self.le_run = QLineEdit()
        self.le_run.setPlaceholderText("<job>.results.npz")
        row.addWidget(self.le_run, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row.addWidget(b)
        v.addLayout(row)

        box = QGroupBox("Settings")
        g = QGridLayout(box)
        g.addWidget(QLabel("Zone [mm]"), 0, 0)
        self.le_zone = {}
        for i, k in enumerate(("xmin", "xmax", "ymin", "ymax")):
            le = QLineEdit()
            le.setPlaceholderText(k)
            le.setToolTip("Zone of the field means (default: the ZOI of the "
                          "Model tab). Before the ZOI is fixed, take a wide "
                          "zone around the tool.")
            self.le_zone[k] = le
            g.addWidget(le, 0, 1 + i)
        g.addWidget(QLabel("P0: moving-mean length L [µs]"), 1, 0)
        self.le_len = QLineEdit("50")
        self.le_len.setToolTip("Length of the trailing moving mean. Choice: "
                               "a few chip-formation periods if the chip is "
                               "serrated.")
        g.addWidget(self.le_len, 1, 1)
        g.addWidget(QLabel("band τ [fraction of ε_q]"), 1, 2)
        self.le_tau = QLineEdit("0.1")
        self.le_tau.setToolTip("A quantity is settled once its moving mean "
                               "stays within a band τ·ε_q wide "
                               "until the end of the run.")
        g.addWidget(self.le_tau, 1, 3)
        g.addWidget(QLabel("P1: wanted attenuation A [dB]"), 2, 0)
        self.le_att = QLineEdit("40")
        self.le_att.setToolTip("Attenuation of the field filter wanted at "
                               "the Nyquist frequency of the frames "
                               "(anti-aliasing). Gives the largest frame "
                               "interval.")
        g.addWidget(self.le_att, 2, 1)
        g.addWidget(QLabel("Chip: margin upstream of the tool [mm]"), 3, 0)
        self.le_margin = QLineEdit()
        self.le_margin.setPlaceholderText("2 × t1")
        self.le_margin.setToolTip("Columns closer to the tool tip than this "
                                  "are not watched (the chip forms there). "
                                  "Empty: twice the uncut chip thickness t1.")
        g.addWidget(self.le_margin, 3, 1)
        g.addWidget(QLabel("gap / band [elements]"), 3, 2)
        self.le_gap = QLineEdit("1.5")
        self.le_gap.setToolTip("Watched cells: k element sizes above the "
                               "surface cell, gap ≤ k < gap + band.")
        g.addWidget(self.le_gap, 3, 3)
        self.le_band = QLineEdit("3")
        g.addWidget(self.le_band, 3, 4)
        v.addWidget(box)

        brow = QHBoxLayout()
        self.btn_run = QPushButton("Analyse")
        self.btn_run.clicked.connect(self._analyse)
        brow.addWidget(self.btn_run)
        self.btn_save = QPushButton("Save CSV…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        brow.addWidget(self.btn_save)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        brow.addWidget(self.lbl_status, 1)
        v.addLayout(brow)

        split = QSplitter(Qt.Vertical)
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QFont("Consolas, monospace"))
        split.addWidget(self.txt)
        self.fig = Figure(figsize=(9, 8), layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        w = QWidget()
        wl = QVBoxLayout(w)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(NavigationToolbar2QT(self.canvas, w))
        wl.addWidget(self.canvas, 1)
        split.addWidget(w)
        split.setSizes([220, 680])
        v.addWidget(split, 1)
        self._fill_zone_from_model()

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
            raise ValueError("Set ε_q in Optimization > Model first.")
        return dict(path=path, zone=zone, L=L * 1e-6, tau=tau, att=att,
                    margin=_num(self.le_margin), gap=gap, band=band, eps=eps)

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
        self.fig.clear()
        st = res["st"]
        axes = self.fig.subplots(4, 2, sharex=True)
        for ax, q in zip(axes.ravel(), pc.QUANTITIES):
            s = st.per_quantity.get(q)
            ax.set_title(q, fontsize=9, pad=3)
            if s is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            t = s.times * 1e6
            ax.plot(t, s.values, color="#9ca3af", lw=0.6)
            ax.plot(t, s.mean, color=_C_MEAN, lw=1.2)
            if math.isfinite(s.final):
                for sg in (-0.5, 0.5):
                    ax.axhline(s.final + sg * s.tol, color=_C_THR, ls="--",
                               lw=0.8)
            if s.t_s is not None:
                ax.axvline(s.t_s * 1e6, color=_C_TS, lw=1)
            ax.set_ylabel(_UNITS.get(q, ""), fontsize=8)
            ax.tick_params(labelsize=7)
        ax = axes.ravel()[6]
        rc = res.get("rc")
        ax.set_title("Chip re-contact: max EVF in the band", fontsize=9,
                     pad=3)
        if rc is not None and np.isfinite(rc.band_max_evf).any():
            ax.plot(rc.times * 1e6, rc.band_max_evf, color=_C_MEAN)
            ax.axhline(0.5, color=_C_THR, ls="--", lw=0.8)
            if rc.t_first is not None:
                ax.axvline(rc.t_first * 1e6, color=_C_THR, lw=1)
        ax.tick_params(labelsize=7)
        axes.ravel()[7].axis("off")
        axes[2, 1].xaxis.set_tick_params(labelbottom=True)
        for a in (axes[3, 0], axes[2, 1]):
            a.set_xlabel("t [µs]", fontsize=8)
        self.canvas.draw_idle()

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
