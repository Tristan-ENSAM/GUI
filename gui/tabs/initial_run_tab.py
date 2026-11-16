# -*- coding: utf-8 -*-
"""Optimization > Initial run: the preliminary checks on the initial run.

Decision 2026-11-12: the Setup tab is gone (its settings are in Geometry,
Mesh, Step and Identify) and this tab checks the initial run. It can launch
the current model and re-read it while it runs (gui.core.live_run): every
interval the results so far are extracted and the checks (stationarity P0,
output frames P1, chip re-contact) are run again on them. A finished run
can still be opened by hand.

The last frame read is drawn with the checks on it (zone, watched
re-contact cells, chip-thickness line, tool); editing a check parameter
recomputes on the same file.

Automatic stop (decision 2026-11-12): after each read, two rules, each with
its check box, can stop the run (preliminary_checks.stop_proposal): every
quantity settled for at least 2 L, or the first chip re-contact. On such a
stop, Ttot = the stop time and T = [t_s / Ttot, 1] are written to the Step
tab, with the number of frames that keeps the frame interval of the run
(the one P1 checked). "Apply Ttot and T to Step" does the same by hand from
the checks shown. The stopped run keeps its last read as its final results
(gui.core.live_run).

Remote mode (Preferences > Execution remote): the run goes to the agent of
the compute PC, which makes the reads there (gui.core.live_run).

The checks read their defaults from the Model sizing tab
(OptimizationTab.prelim_settings), which still owns every setting.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSplitter, QVBoxLayout, QWidget)

from gui.core.live_run import DEFAULT_INTERVAL_S, LiveRunWorker
from gui.widgets.preliminary_checks_panel import PreliminaryChecksPanel
from gui.widgets.theme import mark_primary


def _last_time(meta_path) -> float:
    try:
        times = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        times = times.get("times") or []
        return float(times[-1]) if times else float("nan")
    except Exception:
        return float("nan")


class InitialRunTab(QWidget):
    """`sizing_tab` is the Model sizing tab (OptimizationTab): its `cfg`,
    `setup_preview`, `prelim_settings` and `_draw_preview` are used.
    `start_dir` is a path or a callable returning it (start folder of the
    file dialogs). `prefs_getter`, `cpus_getter` and `profile_name_getter`
    give the Preferences, the CPU count of the Job tab and the profile
    name, for launching the run."""

    def __init__(self, sizing_tab, start_dir="", prefs_getter=None,
                 cpus_getter=None, profile_name_getter=None, step_tab=None,
                 parent=None):
        super().__init__(parent)
        self._sizing = sizing_tab
        self._step_tab = step_tab
        self._proposal = None         # StopProposal of the last analysis
        self._frame_dt = None         # frame interval of the last analysis
        self._stop_reason = None      # proposal that stopped the run
        self._prefs_getter = prefs_getter
        self._cpus_getter = cpus_getter
        self._profile_name_getter = profile_name_getter
        self._worker = None
        self._thread = None
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        intro = QLabel(
            "Checks on the initial run, before any study: steady regime "
            "(start of the window T), output-frame subsampling and chip "
            "touching the workpiece again. Launch the current model to see "
            "them while it runs, or open a finished run below. The ZOI is "
            "set in Numerical Model > Geometry, the window T in Step, the "
            "ZOI sampling step in Mesh and ε_q in Identify.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#6b7280;")
        root.addWidget(intro)

        # ---- launch / stop -------------------------------------------
        glive = QGroupBox("Initial run of the current model (this PC, or the "
                          "compute PC in remote mode)")
        lv = QHBoxLayout(glive)
        self.btn_launch = mark_primary(QPushButton("Launch the initial run"))
        self.btn_launch.setToolTip(
            "Run the current model (on this PC, or on the compute PC when\n"
            "Preferences are in remote mode) and re-read its results every\n"
            "interval while it runs: the checks below are run again on the\n"
            "results so far.")
        self.btn_launch.clicked.connect(self._on_launch)
        lv.addWidget(self.btn_launch)
        self.btn_stop = QPushButton("Stop the run")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("abaqus terminate, as Cancel; the results "
                                 "read so far stay shown.")
        self.btn_stop.clicked.connect(self._on_stop)
        lv.addWidget(self.btn_stop)
        lv.addSpacing(12)
        lv.addWidget(QLabel("re-read every"))
        self.le_interval = QLineEdit("%g" % DEFAULT_INTERVAL_S)
        self.le_interval.setFixedWidth(50)
        self.le_interval.setToolTip(
            "Seconds between two reads of the running .odb. Default 30 s (a\n"
            "choice). One read took about 1 s on a small model on CL-WTS-1;\n"
            "on a large model it is longer (not measured).")
        lv.addWidget(self.le_interval)
        lv.addWidget(QLabel("s"))
        lv.addSpacing(12)
        self.lbl_live = QLabel("")
        self.lbl_live.setWordWrap(True)
        lv.addWidget(self.lbl_live, 1)
        root.addWidget(glive)

        gstop = QGroupBox("Automatic stop of the initial run")
        sl = QHBoxLayout(gstop)
        self.cb_stop_stable = QCheckBox("when every quantity is settled for "
                                        "2 L")
        self.cb_stop_stable.setChecked(True)
        self.cb_stop_stable.setToolTip(
            "Stop when the six quantities of P0 are settled (moving mean\n"
            "within τ·ε_q) and the settled part is at least 2 L long.\n"
            "Ttot = the last time read; T starts at the largest t_s.")
        sl.addWidget(self.cb_stop_stable)
        self.cb_stop_rc = QCheckBox("at the first chip re-contact")
        self.cb_stop_rc.setChecked(True)
        self.cb_stop_rc.setToolTip(
            "Stop at the first frame where material comes back into the\n"
            "watched band above the uncut surface: the later outputs are\n"
            "suspect. Ttot = that time. Before the run settles, nothing is\n"
            "written to Step (lower h_void instead).")
        sl.addWidget(self.cb_stop_rc)
        sl.addSpacing(12)
        self.btn_apply = QPushButton("Apply Ttot and T to Step")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip(
            "Write the Ttot and window T proposed by the checks shown to\n"
            "the Step tab (done automatically when a rule stops the run).")
        self.btn_apply.clicked.connect(self._on_apply)
        sl.addWidget(self.btn_apply)
        self.lbl_stop = QLabel("")
        self.lbl_stop.setWordWrap(True)
        sl.addWidget(self.lbl_stop, 1)
        root.addWidget(gstop)

        gpre = QGroupBox("Preliminary checks on one run")
        gl = QVBoxLayout(gpre)
        gl.setContentsMargins(4, 4, 4, 4)
        self.prelim = PreliminaryChecksPanel(
            settings_getter=sizing_tab.prelim_settings, start_dir=start_dir,
            parent=gpre)
        self.prelim.analysed.connect(self._on_analysed)
        gl.addWidget(self.prelim)

        gprev = QGroupBox("Model with the ZOI")
        pv = QVBoxLayout(gprev)
        pv.addWidget(sizing_tab.setup_preview, 1)
        btn = QPushButton("Refresh preview")
        btn.clicked.connect(sizing_tab._draw_preview)
        pv.addWidget(btn)
        sizing_tab.setup_preview.setMinimumHeight(260)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(gpre)
        split.addWidget(gprev)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

    # ------------------------------------------------------------------
    def _status(self, text, color="#1d4ed8"):
        self.lbl_live.setStyleSheet("color:%s;" % color)
        self.lbl_live.setText(text)

    def _launch_problem(self, prefs):
        if prefs is None:
            return "No preferences available (Abaqus command and script)."
        from gui.core.remote_exec import is_remote, launch_problems
        if is_remote(prefs):
            lines = launch_problems(prefs, getattr(prefs, "default_workdir",
                                                   "") or "")
            return "\n".join(lines) if lines else None
        if not getattr(prefs, "abaqus_cmd", ""):
            return "Set the Abaqus command in Preferences."
        if not Path(getattr(prefs, "abaqus_script", "") or "").is_file():
            return "run_simul.py not found: check the script in Preferences."
        return None

    def _on_launch(self):
        prefs = self._prefs_getter() if self._prefs_getter else None
        problem = self._launch_problem(prefs)
        if problem:
            QMessageBox.warning(self, "Initial run", problem)
            return
        try:
            interval = float(self.le_interval.text().replace(",", "."))
            if not interval > 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Initial run",
                                "The re-read interval must be > 0 s.")
            return
        from gui.core.run_output import create_study_dir, job_name
        cfg = copy.deepcopy(self._sizing.cfg)
        try:
            pname = (self._profile_name_getter()
                     if self._profile_name_getter else None)
        except Exception:
            pname = None
        try:
            cpus = int(self._cpus_getter()) if self._cpus_getter else 1
        except Exception:
            cpus = 1
        try:
            folder = create_study_dir(prefs.default_workdir, pname,
                                      "initial", {"cpus": cpus,
                                                  "interval_s": interval})
        except OSError as e:
            QMessageBox.warning(self, "Initial run",
                                "Cannot create the run folder: %s" % e)
            return
        job = job_name("initial", 0)
        self._stop_reason = None
        from gui.core.remote_exec import is_remote, submit_remote
        submit = None
        if is_remote(prefs):
            def submit(run_dir, model_params, run_params, live_interval):
                return submit_remote(prefs, run_dir, model_params,
                                     run_params, live_interval=live_interval)
        self.start_worker(LiveRunWorker(
            cfg, abaqus_cmd=prefs.abaqus_cmd,
            abaqus_script=prefs.abaqus_script, run_dir=folder / job,
            job_name=job, cpus=cpus, interval=interval, submit=submit))

    def start_worker(self, worker):
        """Run `worker` (a LiveRunWorker) in a thread (tests inject one)."""
        self._worker = worker
        self._t0 = time.monotonic()
        self._thread = QThread(self)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.snapshot.connect(self._on_snapshot)
        worker.finished.connect(self._on_finished)
        self._thread.start()
        self.btn_launch.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._status("Running %s in %s; first read in %g s."
                     % (worker.job_name, worker.run_dir, worker.interval))

    def _on_snapshot(self, path, seconds):
        t = _last_time(path.replace(".results.npz", ".meta.json"))
        self._status("Read %d at %s: results up to t = %.4g s (read in "
                     "%.1f s). Checks updated below."
                     % (self._worker.n_snapshots if self._worker else 0,
                        time.strftime("%H:%M:%S"), t, seconds))
        self.prelim.analyse_file(path)

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
            self.btn_stop.setEnabled(False)
            self._status("Stopping the run…", "#b45309")

    def _on_finished(self, final):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        self._thread = None
        self.btn_launch.setEnabled(True)
        self.btn_stop.setEnabled(False)
        stopped = bool(getattr(self._worker, "stopped", False))
        if final and stopped:
            self._status("Run stopped: its last read is kept as its "
                         "results (%s)." % final, "#15803d")
            self.prelim.analyse_file(final)
        elif final:
            self._status("Run finished: checks on its full results.",
                         "#15803d")
            self.prelim.analyse_file(final)
        else:
            err = getattr(self._worker, "error", "")
            self._status("Run stopped or failed%s: the checks show the last "
                         "results read." % ((" (%s)" % err) if err else ""),
                         "#b45309")
        self._worker = None

    def is_running(self) -> bool:
        return self._worker is not None

    def shutdown(self):
        """Window closing: stop a running initial run."""
        if self._worker is not None:
            self._worker._stop = True
            self._worker._stop_blocking(self._worker._proc)

    # ---- automatic stop and transfer to Step -----------------------------
    def _on_analysed(self, res):
        from gui.sensitivity.preliminary_checks import stop_proposal
        st = res.get("st")
        fc = res.get("fc")
        self._frame_dt = getattr(fc, "dt", None)
        self._proposal = None
        if st is None:
            return
        rc = res.get("rc")
        p = self._proposal = stop_proposal(st, rc)
        self.btn_apply.setEnabled(p is not None and p.window is not None)
        if p is None:
            self.lbl_stop.setText("No stop rule met yet."
                                  if self.is_running() else "")
            return
        if self._stop_reason is not None:
            return                    # keep the rule that stopped the run
        self.lbl_stop.setText(self._describe(p))
        if not self.is_running():
            return
        rule = stop_proposal(st, rc,
                             on_stable=self.cb_stop_stable.isChecked(),
                             on_recontact=self.cb_stop_rc.isChecked())
        if rule is None:
            return
        self._stop_reason = rule
        self._on_stop()
        self._status("Stopping the run: %s." % self._describe(rule),
                     "#b45309")
        if rule.window is not None:
            self.apply_to_step(rule)

    @staticmethod
    def _describe(p):
        why = ("every quantity settled for 2 L" if p.reason == "stable"
               else "chip re-contact")
        text = "%s at t = %.1f µs" % (why, p.t_stop * 1e6)
        if p.window is not None:
            text += ": Ttot = %.4g s, T = [%.3f, %.3f]" % (
                (p.t_stop,) + tuple(p.window))
        if p.note:
            text += " (%s)" % p.note
        return text

    def _on_apply(self):
        if self._proposal is not None and self._proposal.window is not None:
            self.apply_to_step(self._proposal)

    def apply_to_step(self, p) -> bool:
        """Write Ttot = p.t_stop, the number of frames that keeps the frame
        interval of the run, and T = p.window. False when nothing could be
        written (no Step tab, no window)."""
        if p is None or p.window is None:
            return False
        done = []
        st = self._step_tab
        if st is not None:
            st.f_sim_time.set_value(float(p.t_stop))
            dt = self._frame_dt
            if dt and dt > 0:
                st.f_n_frames.set_value(max(1, int(round(p.t_stop / dt))))
            st._on_change()
            done.append("Ttot")
        texts = getattr(self._sizing, "_dom_texts", {})
        if "window_start" in texts and "window_end" in texts:
            texts["window_start"].setText("%.4g" % p.window[0])
            texts["window_end"].setText("%.4g" % p.window[1])
            done.append("T")
        if done:
            self.lbl_stop.setText("Written to Step: %s. %s"
                                  % (" and ".join(done), self._describe(p)))
        return bool(done)
