# -*- coding: utf-8 -*-
"""Background worker for the Eulerian-domain independence study (paper §4).

Runs `domain_independence.run_domain_independence` off the GUI thread and
forwards its progress events as Qt signals. Replaces DomainConvergenceWorker
as the domain study wired to the Optimization tab.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.sensitivity.domain_independence import run_domain_independence


class DomainIndependenceWorker(QThread):
    progress = Signal(object)      # event dict from the study
    finished_ok = Signal(object)   # StudyResult
    failed = Signal(str)

    def __init__(self, parent=None, **study_kwargs):
        """`study_kwargs` are passed unchanged to run_domain_independence
        (run_bundle, base_cfg, zoi, initial_dims, grid_step, elem_size,
        thresholds, window, step_elems, n_max, n_hold, m_ratios, caps,
        margin_elems, offset, guard_fn, cost_fn, ...). `should_cancel` and
        `progress_cb` are supplied by the worker itself."""
        super().__init__(parent)
        self._kw = dict(study_kwargs)
        self._cancel = False

    def cancel(self):
        """Request a stop, checked BETWEEN runs.

        The flag alone does not touch the job in flight; interrupting it is
        OptimizationTab._on_cancel's job, which holds the process handle and
        the job name (same split as MeshGciWorker)."""
        self._cancel = True

    def run(self):
        try:
            result = run_domain_independence(
                progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel, **self._kw)
        except Exception as e:                       # pragma: no cover
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(result)
