# -*- coding: utf-8 -*-
"""Background worker running the full mesh+domain convergence pipeline."""
from __future__ import annotations

from typing import Callable, Dict, Optional

from PySide6.QtCore import QThread, Signal

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.mesh_pipeline import run_mesh_domain_pipeline


class MeshPipelineWorker(QThread):
    progress = Signal(object)      # event dict from the pipeline
    finished_ok = Signal(object)   # PipelineResult
    failed = Signal(str)

    def __init__(self, base_cfg, run_bundle, roi, quantity_field_map,
                 thresholds, wp_start, tool_start, initial_domain,
                 factor=0.5, max_steps=8, wp_min=None, tool_min=None,
                 caps=None, order=None, include_tool=True, do_verify=True,
                 max_doublings=8, force_channels=None, grid_step=None,
                 identify_ms=False, ms_start=1.0, ms_factor_growth=2.0,
                 ms_max_factor=None, ms_guard_threshold=None,
                 ms_bisection_resolution=0.0, wp_bisection_resolution=0.0,
                 tool_bisection_resolution=0.0,
                 domain_bisection_resolution=0.0, wp_factor=0.5,
                 tool_factor=0.5, domain_grow_factor=2.0,
                 include_wp=True, include_domain=True, parent=None):
        super().__init__(parent)
        self._kw = dict(
            base_cfg=base_cfg, run_bundle=run_bundle, roi=roi,
            quantity_field_map=quantity_field_map, thresholds=thresholds,
            wp_start=wp_start, tool_start=tool_start,
            initial_domain=initial_domain, factor=factor, max_steps=max_steps,
            wp_min=wp_min, tool_min=tool_min, caps=caps,
            include_tool=include_tool, do_verify=do_verify,
            max_doublings=max_doublings, force_channels=force_channels,
            grid_step=grid_step, identify_ms=identify_ms, ms_start=ms_start,
            ms_factor_growth=ms_factor_growth, ms_max_factor=ms_max_factor,
            ms_guard_threshold=ms_guard_threshold,
            ms_bisection_resolution=ms_bisection_resolution,
            wp_bisection_resolution=wp_bisection_resolution,
            tool_bisection_resolution=tool_bisection_resolution,
            domain_bisection_resolution=domain_bisection_resolution,
            wp_factor=wp_factor, tool_factor=tool_factor,
            domain_grow_factor=domain_grow_factor,
            include_wp=include_wp, include_domain=include_domain)
        if order is not None:
            self._kw["order"] = order
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = run_mesh_domain_pipeline(
                progress_cb=lambda ev: self.progress.emit(ev),
                should_cancel=lambda: self._cancel, **self._kw)
        except Exception as e:
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        self.finished_ok.emit(result)
