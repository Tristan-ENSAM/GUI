# -*- coding: utf-8 -*-
"""Tests for DomainConvergenceWorker.

The worker is a thin QThread wrapper. We call run() SYNCHRONOUSLY (not via
start()) so the study runs in the test thread with no event loop, and capture
the emitted signals. Sampling is bypassed exactly as in test_domain_convergence.
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.core.domain_sizing import DomainDims
import gui.sensitivity.domain_convergence as dc
from gui.sensitivity.domain_convergence import DIMENSION_NAMES, ConvergenceResult
from gui.sensitivity.domain_convergence_worker import DomainConvergenceWorker

_NP, _NT = 10, 8


class _Bundle:
    def __init__(self, dims, lam=0.02):
        self.dims, self.lam = dims, lam

    @property
    def times(self):
        return np.linspace(0.0, 3e-4, _NT)

    def instance(self, name):
        class _Info:
            field_variables = ["EVF", "TEMP", "V1", "V2"]
        return _Info()

    def _influence(self):
        return sum(np.exp(-getattr(self.dims, n) / self.lam)
                   for n in DIMENSION_NAMES)

    def field(self, inst, var):
        out = np.zeros((_NT, _NP), dtype=float)
        if var == "EVF":
            out[:] = 1.0
            return out
        out[:] = {"TEMP": 300.0, "V1": 10.0, "V2": -5.0}[var] * (
            1.0 + 0.5 * self._influence())
        return out

    def history(self, var):
        if var == "RF1_RP":
            return np.full(_NT, 100.0)
        raise KeyError(var)

    @property
    def history_time(self):
        return np.linspace(0.0, 3e-4, _NT)


class _Cfg:
    class _G:
        h_wp = h_void = l_wp = l_void = 0.0

    def __init__(self):
        self.euler_geometry = _Cfg._G()


@pytest.fixture(autouse=True)
def _patch_sampling(monkeypatch):
    monkeypatch.setattr(dc, "nearest_samples",
                        lambda b, var, inst, pts: b.field(inst, var))
    monkeypatch.setattr(dc, "roi_grid", lambda roi, step: np.zeros((_NP, 2)))
    monkeypatch.setattr(dc, "eulerian_instance", lambda b: "EULER")


def _runner():
    def run(cfg):
        g = cfg.euler_geometry
        return _Bundle(DomainDims(g.h_wp, g.h_void, g.l_wp, g.l_void))
    return run


def test_worker_emits_result(qapp):
    got = {}
    w = DomainConvergenceWorker(
        _runner(), _Cfg(), (-0.03, 0.03, -0.03, 0.03),
        DomainDims(0.05, 0.05, 0.05, 0.05), grid_step=0.01, elem_size=0.005,
        tolerances={q: 0.02 for q in ("EVF", "TEMP", "V1", "V2", "force")},
        grow_elems=4, margin_elems=1, max_iterations=12)
    w.finished_ok.connect(lambda r: got.setdefault("res", r))
    w.failed.connect(lambda msg: got.setdefault("err", msg))
    w.run()                          # synchronous
    assert "err" not in got
    assert isinstance(got.get("res"), ConvergenceResult)
    assert got["res"].converged and got["res"].n_runs > 0


def test_worker_reports_zoi_outside(qapp):
    got = {}
    w = DomainConvergenceWorker(
        _runner(), _Cfg(), (-0.2, 0.2, -0.2, 0.2),
        DomainDims(0.05, 0.05, 0.05, 0.05), grid_step=0.01, elem_size=0.005,
        margin_elems=1)
    w.finished_ok.connect(lambda r: got.setdefault("res", r))
    w.run()
    assert got["res"].stopped_by == "zoi_outside" and not got["res"].converged
