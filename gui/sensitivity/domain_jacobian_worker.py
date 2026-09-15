# -*- coding: utf-8 -*-
"""Background worker for the Jacobian-based Eulerian domain sizing study.

Runs `domain_jacobian.run_domain_study` off the GUI thread, forwarding its
progress events as Qt signals.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.sensitivity.domain_jacobian import run_domain_study


class DomainJacobianWorker(QThread):
    progress = Signal(object)      # event dict from the study
    finished_ok = Signal(object)   # list[JacobianResult]
    failed = Signal(str)

    def __init__(self, run_bundle, base_cfg, roi, initial_dims, grid_step,
                 thresholds, elem_size, field_vars=("EVF", "TEMP", "V"),
                 step_elems=1, grow_elems=4, max_iterations=8,
                 mass_scaling_factor=1.0, linearity_check=True, parent=None):
        super().__init__(parent)
        self._kw = dict(
            run_bundle=run_bundle, cfg=base_cfg, roi=roi,
            initial_dims=initial_dims, grid_step=grid_step,
            thresholds=thresholds, elem_size=elem_size,
            field_vars=tuple(field_vars), step_elems=int(step_elems),
            grow_elems=int(grow_elems), max_iterations=int(max_iterations),
            mass_scaling_factor=float(mass_scaling_factor),
            linearity_check=bool(linearity_check))
        self._cancel = False

    def cancel(self):
        """Request a stop. Checked BETWEEN runs: the Abaqus job currently in
        flight is not interrupted (the study would otherwise be left with a
        half-written bundle)."""
        self._cancel = True

    def run(self):
        try:
            history = run_domain_study(
                progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel, **self._kw)
        except Exception as e:                       # pragma: no cover
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(history)
