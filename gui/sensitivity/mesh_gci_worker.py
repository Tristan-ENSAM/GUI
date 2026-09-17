# -*- coding: utf-8 -*-
"""Background worker for the GCI/Richardson mesh convergence study.

Runs `mesh_gci.run_mesh_gci` off the GUI thread, forwarding its progress events
as Qt signals. Mirrors DomainConvergenceWorker.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.sensitivity.mesh_gci import run_mesh_gci


class MeshGciWorker(QThread):
    progress = Signal(object)      # event dict from the study
    finished_ok = Signal(object)   # MeshGciResult
    failed = Signal(str)

    def __init__(self, run_bundle, base_cfg, zoi, domain_dims, grid_step,
                 finest_elem_size, ratio=2.0, n_meshes=3, tolerances=None,
                 field_vars=("EVF", "TEMP", "V1", "V2"), window=(0.3, 1.0),
                 evf_threshold=0.5, force_channels=None, min_elem_size=None,
                 parent=None):
        super().__init__(parent)
        self._kw = dict(
            run_bundle=run_bundle, base_cfg=base_cfg, zoi=tuple(zoi),
            domain_dims=domain_dims, grid_step=float(grid_step),
            finest_elem_size=float(finest_elem_size), ratio=float(ratio),
            n_meshes=int(n_meshes), tolerances=tolerances,
            field_vars=tuple(field_vars), window=tuple(window),
            evf_threshold=float(evf_threshold), force_channels=force_channels,
            min_elem_size=(None if min_elem_size is None else float(min_elem_size)))
        self._cancel = False

    def cancel(self):
        """Request a stop, checked BETWEEN runs.

        This flag alone does not touch the job in flight; interrupting it is
        OptimizationTab._on_cancel's job, which holds the process handle and
        the job name. See DomainConvergenceWorker.cancel for why the previous
        wording ("the in-flight Abaqus job is not interrupted") was true of
        the flag but false of the Cancel button.
        """
        self._cancel = True

    def run(self):
        try:
            result = run_mesh_gci(
                progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel, **self._kw)
        except Exception as e:                       # pragma: no cover
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(result)
