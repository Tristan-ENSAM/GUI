# -*- coding: utf-8 -*-
"""Background worker for the mass-scaling independence study (paper step 0).

Runs `ms_independence.run_ms_independence` off the GUI thread and forwards
its progress events as Qt signals (same contract as DomainIndependenceWorker).
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.sensitivity.ms_independence import run_ms_independence


class MsIndependenceWorker(QThread):
    progress = Signal(object)      # event dict from the study
    finished_ok = Signal(object)   # MsStudyResult
    failed = Signal(str)

    def __init__(self, parent=None, **study_kwargs):
        """`study_kwargs` are passed unchanged to run_ms_independence;
        `should_cancel` and `progress_cb` are supplied by the worker."""
        super().__init__(parent)
        self._kw = dict(study_kwargs)
        self._cancel = False

    def cancel(self):
        """Request a stop, checked BETWEEN runs (the job in flight is
        OptimizationTab._on_cancel's job)."""
        self._cancel = True

    def run(self):
        try:
            result = run_ms_independence(
                progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel, **self._kw)
        except Exception as e:                       # pragma: no cover
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(result)
