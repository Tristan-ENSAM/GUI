# -*- coding: utf-8 -*-
"""Background worker for the convergence-based Eulerian domain sizing (Option B).

Runs `domain_convergence.run_domain_convergence` off the GUI thread, forwarding
its progress events as Qt signals. This is the only domain-sizing study wired
to the Optimization tab; the earlier Jacobian-based one was abandoned.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.core.domain_sizing import REVERB_MARGIN_K
from gui.sensitivity.domain_convergence import run_domain_convergence


class DomainConvergenceWorker(QThread):
    progress = Signal(object)      # event dict from the study
    finished_ok = Signal(object)   # ConvergenceResult
    failed = Signal(str)

    def __init__(self, run_bundle, base_cfg, zoi, initial_dims, grid_step,
                 elem_size, tolerances=None, field_vars=("EVF", "TEMP", "V1", "V2"),
                 window=(0.3, 1.0), evf_threshold=0.5, grow_elems=4,
                 margin_elems=1, max_iterations=8, force_channel="RF1_RP",
                 diagonal_ceiling=None, reverb_margin_k=REVERB_MARGIN_K,
                 parent=None):
        super().__init__(parent)
        self._kw = dict(
            run_bundle=run_bundle, cfg=base_cfg, zoi=tuple(zoi),
            initial_dims=initial_dims, grid_step=float(grid_step),
            elem_size=float(elem_size), tolerances=tolerances,
            field_vars=tuple(field_vars), window=tuple(window),
            evf_threshold=float(evf_threshold), grow_elems=int(grow_elems),
            margin_elems=int(margin_elems), max_iterations=int(max_iterations),
            force_channel=str(force_channel),
            diagonal_ceiling=(None if diagonal_ceiling is None
                              else float(diagonal_ceiling)),
            reverb_margin_k=float(reverb_margin_k))
        self._cancel = False

    def cancel(self):
        """Request a stop, checked BETWEEN runs.

        This flag alone does not touch the job in flight. Interrupting it is
        OptimizationTab._on_cancel's job, because the process handle and the
        job name live there (`abaqus terminate`, then the process tree). That
        split is why this docstring used to claim the in-flight run was never
        interrupted: true of this flag, false of the Cancel button, which has
        terminated the launcher since the tab was written.

        The half-written bundle the old wording worried about cannot be read
        as data: run_bundle returns None once the cancel event is set, and
        run_domain_convergence treats a None bundle as a failed run.
        """
        self._cancel = True

    def run(self):
        try:
            result = run_domain_convergence(
                progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel, **self._kw)
        except Exception as e:                       # pragma: no cover
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(result)
