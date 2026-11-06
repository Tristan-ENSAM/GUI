# -*- coding: utf-8 -*-
"""Background worker for the Richardson test on the domain
(gui.sensitivity.domain_richardson). Mirrors MeshGciWorker."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.sensitivity.domain_richardson import run_domain_richardson


class DomainRichardsonWorker(QThread):
    progress = Signal(object)      # event dict
    finished_ok = Signal(object)   # DomainRichardsonResult
    failed = Signal(str)

    def __init__(self, run_bundle, base_cfg, zoi, base_dims, elem_size,
                 parent=None, **kw):
        super().__init__(parent)
        self._args = (run_bundle, base_cfg, tuple(zoi), base_dims,
                      float(elem_size))
        self._kw = dict(kw)
        self._cancel = False

    def cancel(self):
        """Request a stop, checked between runs (the job in flight is
        stopped by OptimizationTab._on_cancel)."""
        self._cancel = True

    def run(self):
        try:
            res = run_domain_richardson(
                *self._args, progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel, **self._kw)
        except Exception as e:                       # pragma: no cover
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(res)
