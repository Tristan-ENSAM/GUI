# -*- coding: utf-8 -*-
"""Tests for MeshGciWorker (thin QThread wrapper). Runs synchronously."""
from __future__ import annotations

import pytest

from gui.core.domain_sizing import DomainDims
import gui.sensitivity.mesh_gci as mg
from gui.sensitivity.mesh_gci import MeshGciResult
from gui.sensitivity.mesh_gci_worker import MeshGciWorker

_BASE = {"EVF": 0.8, "TEMP": 300.0, "V1": 10.0, "V2": -5.0, "Fc": 100.0, "Ff": 50.0}


class _Cfg:
    class _G:
        h_wp = h_void = l_wp = l_void = 0.0

    def __init__(self):
        self.euler_geometry = _Cfg._G()
        self.elem_size = 0.0


class _Bundle:
    def __init__(self, elem_size):
        self.elem_size = elem_size


@pytest.fixture(autouse=True)
def _patch_scalars(monkeypatch):
    def fake_scalars(bundle, zoi, grid_step, field_vars, **kw):
        h = bundle.elem_size
        out = {q: _BASE[q] * (1.0 + 10.0 * h * h) for q in field_vars}
        out["Fc"] = _BASE["Fc"] * h * (1.0 + 10.0 * h * h)  # raw RF ~ width
        out["Ff"] = _BASE["Ff"] * h * (1.0 + 10.0 * h * h)
        return out
    monkeypatch.setattr(mg, "zoi_scalars", fake_scalars)


def _runner():
    def run(cfg):
        return _Bundle(cfg.elem_size)
    return run


def test_worker_emits_result(qapp):
    got = {}
    w = MeshGciWorker(
        _runner(), _Cfg(), (-0.04, 0.04, -0.04, 0.04),
        DomainDims(0.255, 0.055, 0.255, 0.055), grid_step=0.01,
        finest_elem_size=0.005, ratio=2.0, n_meshes=3,
        tolerances={q: 0.005 for q in _BASE})
    w.finished_ok.connect(lambda r: got.setdefault("res", r))
    w.failed.connect(lambda m: got.setdefault("err", m))
    w.run()
    assert "err" not in got
    assert isinstance(got.get("res"), MeshGciResult)
    assert got["res"].n_runs == 3
    assert got["res"].per_quantity["TEMP"].p == pytest.approx(2.0, rel=1e-6)
