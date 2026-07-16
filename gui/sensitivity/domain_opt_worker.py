# -*- coding: utf-8 -*-
"""
Background worker for the Eulerian-domain optimiser (Optimization tab).

DomainOptimizer.optimize() is synchronous and each sample_fn call is a full
Abaqus run, so the optimisation must run off the UI thread. This QThread wraps
it, forwarding the optimiser's progress events as Qt signals and supporting
cancellation.

The `sample_fn` is INJECTED, so the worker is agnostic to how a candidate
domain is actually simulated: the tab supplies a real Abaqus-backed callback
(one run_simul job per domain, then ROI extraction), while tests supply a
lightweight analytic callback. Nothing about the Abaqus launch mechanism is
hard-coded here.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from PySide6.QtCore import QThread, Signal

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.domain_opt import DomainOptimizer, OptimizeResult


class DomainOptWorker(QThread):
    """Runs a DomainOptimizer in the background.

    Signals
    -------
    progress(object) : the optimiser's event dict after each dimension search
        (and once with phase "done").
    finished_ok(object) : the final OptimizeResult.
    failed(str) : an error message if the optimisation raised.
    """
    progress = Signal(object)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, sample_fn: Callable[[DomainDims], Dict[str, object]],
                 initial: DomainDims, thresholds: Dict[str, float],
                 elem_size: float, caps: Optional[Dict[str, float]] = None,
                 order=None, quantities=None, max_passes: int = 2,
                 max_doublings: int = 20, bisect_resolution: float = 0.0,
                 grow_factor: float = 2.0, parent=None):
        super().__init__(parent)
        # Build the optimiser here so all knobs live in one place; sample_fn is
        # the only injection point that touches Abaqus.
        kwargs = dict(thresholds=thresholds, elem_size=elem_size, caps=caps,
                      max_passes=max_passes, max_doublings=max_doublings,
                      bisect_resolution=bisect_resolution,
                      grow_factor=grow_factor)
        if order is not None:
            kwargs["order"] = order
        if quantities is not None:
            kwargs["quantities"] = quantities
        self._optimizer = DomainOptimizer(sample_fn, **kwargs)
        self._initial = initial
        self._cancel = False

    def cancel(self):
        """Request an early, graceful stop (checked between dimensions)."""
        self._cancel = True

    def run(self):
        try:
            result = self._optimizer.optimize(
                self._initial,
                progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel)
        except Exception as e:                      # surface, don't crash the UI
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(result)
