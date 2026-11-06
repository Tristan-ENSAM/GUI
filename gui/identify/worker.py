# -*- coding: utf-8 -*-
"""Run an identification (gui.identify.engine) with Abaqus, in a QThread.

The engine asks for the QoI at a list of parameter points; each point is
one run of the model with those values (written into a copy of the model,
in displayed units, as the parameter plans do), solved by the same code as
the Explore / Sensitivity plans: a finished run with the same model
parameters (this folder first, then the cache folders) is read back
instead of launched, so an interrupted identification resumes for the
cost of the extractions only. The history is written to identify.json in
the folder after every iteration.
"""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path

from PySide6.QtCore import Signal

from gui.core.logging_util import log_swallowed
from gui.identify import engine as ie
from gui.sensitivity import param_registry as pr
from gui.sensitivity import runner_core as rc
from gui.sensitivity.run_worker import SensitivityRunWorker

RECORD_NAME = "identify.json"


def apply_point(base_cfg, specs, values, temp_unit="C", system=None):
    """Copy of the model with the parameters `specs` at `values`
    (displayed units)."""
    cfg = copy.deepcopy(base_cfg)
    for spec, v in zip(specs, values):
        pr.apply_display(cfg, spec, float(v), temp_unit, system=system)
    return cfg


def point_qoi(bundle, qoi_specs) -> dict:
    """{qid: value} of a run, without the NaN values (a missing QoI makes
    the cost infinite: the engine rejects the point)."""
    q = rc.extract_qois(bundle, qoi_specs, 0.0)
    return {k: float(v) for k, v in q.items()
            if v is not None and math.isfinite(float(v))}


class IdentifyWorker(SensitivityRunWorker):
    """SensitivityRunWorker whose loop is the identification instead of a
    plan: same Abaqus launch, cancel, log and run reuse."""
    iteration = Signal(object)      # engine.Iteration

    def __init__(self, specs, params, targets, qoi_specs, base_cfg, *,
                 settings: dict, temp_unit: str = "C", unit_system=None,
                 search: dict, record_extra: dict = None, **kw):
        kw.setdefault("job_prefix", "identify")
        kw.setdefault("cache_folders", [])
        super().__init__(None, "identify", qoi_specs, base_cfg, **kw)
        self._specs = list(specs)
        self._params = list(params)
        self._targets = list(targets)
        self._settings = dict(settings)
        self._temp_unit = temp_unit
        self._system = unit_system
        self._search = dict(search)
        self._record_extra = dict(record_extra or {})
        self._n = 0
        self._ident = None
        self.result = None

    def _evaluate(self, points):
        out = []
        for p in points:
            i = self._n
            self._n += 1
            self.progress.emit(i, int(self._search.get("budget", 0)))
            cfg = apply_point(self._base_cfg, self._specs, p,
                              self._temp_unit, self._system)
            solve = self._solve_fn or self._solve_with_cache
            bundle = solve(cfg, i)
            if bundle is None:
                out.append(None)
                continue
            try:
                out.append(point_qoi(bundle, self._qoi_specs))
            except Exception as e:
                self.log.emit("[run %d] QoI not computed: %s\n" % (i + 1, e))
                out.append(None)
        return out

    def _on_iteration(self, it):
        self.iteration.emit(it)
        self.save_record(self._ident.snapshot())

    def record(self, res) -> dict:
        rec = res.record(self._params, self._targets)
        rec.update({"study": "identify", "settings": self._settings,
                    "search": self._search,
                    "units": [s.unit_str(self._temp_unit, self._system)
                              for s in self._specs]})
        rec.update(self._record_extra)
        return rec

    def save_record(self, res) -> None:
        try:
            (Path(self._workdir) / RECORD_NAME).write_text(
                json.dumps(self.record(res), indent=1, default=float),
                encoding="utf-8")
        except Exception:
            log_swallowed("writing identify.json", level=logging.DEBUG)

    def run(self):
        try:
            s = self._search
            self._ident = ie.Identification(
                self._params, self._targets, self._evaluate,
                strategy=s.get("strategy", "lm"),
                budget=int(s.get("budget", 50)),
                tau=float(s.get("tau", ie.DEFAULT_TAU)),
                n_stag=int(s.get("n_stag", ie.DEFAULT_N_STAG)),
                fd_scheme=s.get("fd_scheme", "forward"),
                fd_step=float(s.get("fd_step", 0.01)),
                line_tol=float(s.get("line_tol", 0.01)),
                progress=self._on_iteration,
                cancelled=lambda: self._cancel)
            res = self._ident.run()
            self.result = res
            self.save_record(res)
            self.finished.emit(res)
        except Exception as e:                          # pragma: no cover
            self.failed.emit("%s" % e)
