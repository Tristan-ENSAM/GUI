# -*- coding: utf-8 -*-
"""
One bar per stage of a run: Simulation (the solver, from the .sta),
Extraction (reading the .odb into the results bundle, from the script log)
and, in remote mode, Transfer (the files coming back to the run folder).

Before, a single bar followed the .sta and simply waited at its last value
while the results were extracted; each stage now starts from 0 on its own
bar. The caller owns the polling timer and calls `poll(...)` with the paths
of the current job.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QProgressBar,
                               QWidget)

from gui.core.run_progress import (StageProgress, extraction_progress,
                                   transfer_progress)
from gui.core.sta_parser import parse_sta

STAGES = ("simulation", "extraction", "transfer")
_TITLES = {"simulation": "Simulation", "extraction": "Extraction",
           "transfer": "Transfer"}


def simulation_progress(sta_path, sim_time: Optional[float] = None
                        ) -> StageProgress:
    """Solver stage from the .sta: frame ratio, else step time / sim_time;
    the text gives the remaining solver time when it can be estimated."""
    from gui.sensitivity.cost_model import fmt_duration, remaining_from_progress
    try:
        snap = parse_sta(sta_path)
    except Exception:
        return StageProgress()
    if not snap.is_ready():
        return StageProgress()
    frac = snap.fraction()
    if frac is None and snap.step_time is not None and sim_time:
        frac = snap.step_time / float(sim_time)
    if frac is None:
        return StageProgress()
    frac = max(0.0, min(1.0, frac))
    if snap.wallclock_total_s is not None:
        return StageProgress(1.0, "done", done=True)
    left = remaining_from_progress(snap.wall_time_seconds(), frac)
    return StageProgress(frac, "" if left is None
                         else "~%s left" % fmt_duration(left))


class RunProgressBars(QWidget):
    """Simulation / Extraction / Transfer bars of the current run.

    horizontal=True puts the three bars side by side, each titled in its own
    text (for a status bar); otherwise one row per stage with a label."""

    def __init__(self, parent=None, horizontal: bool = False,
                 bar_width: Optional[int] = None):
        super().__init__(parent)
        self.bars = {}
        self._remote = False
        self._horizontal = horizontal
        if horizontal:
            lay = QHBoxLayout(self)
        else:
            lay = QGridLayout(self)
            lay.setColumnStretch(1, 1)
        lay.setContentsMargins(0, 0, 0, 0)
        self._labels = {}
        for row, stage in enumerate(STAGES):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            if bar_width:
                bar.setFixedWidth(bar_width)
            self.bars[stage] = bar
            if horizontal:
                lay.addWidget(bar)
            else:
                lbl = QLabel(_TITLES[stage] + ":")
                self._labels[stage] = lbl
                lay.addWidget(lbl, row, 0)
                lay.addWidget(bar, row, 1)
            self._set(stage, StageProgress())
        self._show_transfer(False)

    # -- state ------------------------------------------------------------
    def _show_transfer(self, on: bool):
        self.bars["transfer"].setVisible(on)
        if "transfer" in self._labels:
            self._labels["transfer"].setVisible(on)

    def _set(self, stage: str, prog: StageProgress, waiting: str = "waiting"):
        bar = self.bars[stage]
        title = _TITLES[stage] if self._horizontal else ""
        if prog.fraction is None:
            bar.setValue(0)
            text = waiting
        else:
            bar.setValue(int(round(max(0.0, min(1.0, prog.fraction)) * 100)))
            # A literal '%' would be read by QProgressBar as a placeholder.
            text = "%p%" + ((" · " + prog.text.replace("%", ""))
                            if prog.text and prog.text != "done" else "")
            if prog.done:
                text = "done"
        bar.setFormat(("%s: %s" % (title, text)) if title else text)
        bar.setToolTip("%s: %s" % (_TITLES[stage], prog.text or text))

    def start(self, remote: bool = False):
        """A new run: every bar back to 0; Transfer only in remote mode."""
        self._remote = bool(remote)
        for stage in STAGES:
            self._set(stage, StageProgress())
        self._show_transfer(self._remote)

    def set_stage(self, stage: str, prog: StageProgress, waiting="waiting"):
        self._set(stage, prog, waiting)

    def poll(self, sta_path=None, log_path=None, run_dir=None, job=None,
             sim_time: Optional[float] = None):
        """Read the files of the current job and update the bars."""
        ext = extraction_progress(log_path) if log_path else StageProgress()
        sim = simulation_progress(sta_path, sim_time) if sta_path \
            else StageProgress()
        if ext.fraction is not None and not sim.done:
            # The extraction starts once the solver has finished.
            sim = StageProgress(1.0, "done", done=True)
        self._set("simulation", sim)
        self._set("extraction", ext)
        if self._remote and run_dir is not None and job:
            tr = transfer_progress(run_dir, job)
            if tr.fraction is not None:
                self._set("transfer", tr)

    def finish(self, success: bool):
        """The run is over: a success fills every bar; a failure leaves them
        where they were, so it shows which stage stopped."""
        if success:
            for stage in STAGES:
                self._set(stage, StageProgress(1.0, "done", done=True))
