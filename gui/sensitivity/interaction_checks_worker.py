# -*- coding: utf-8 -*-
"""Background worker for the a-posteriori interaction checks (paper §5.7).

Runs `interaction_checks.run_interaction_checks` off the GUI thread and
forwards its progress events as Qt signals (same pattern as the study
workers)."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.sensitivity.interaction_checks import run_interaction_checks


class InteractionChecksWorker(QThread):
    progress = Signal(object)      # event dict
    finished_ok = Signal(object)   # ChecksResult
    failed = Signal(str)

    def __init__(self, parent=None, **kwargs):
        """`kwargs` go to run_interaction_checks (run_bundle, base_cfg, study,
        h_star, gci_plan, gci_tolerances, guard_fn, cost_fn,
        gci_runner_factory, ms_lower, ms_guard_fn); `should_cancel` and
        `progress_cb` are supplied here."""
        super().__init__(parent)
        self._kw = dict(kwargs)
        self._cancel = False

    def cancel(self):
        """Request a stop, checked between runs; the job in flight is
        interrupted by OptimizationTab._on_cancel."""
        self._cancel = True

    def run(self):
        try:
            result = run_interaction_checks(
                progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel, **self._kw)
        except Exception as e:                       # pragma: no cover
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(result)
