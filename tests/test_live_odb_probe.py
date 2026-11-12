# -*- coding: utf-8 -*-
"""The live .odb read probe (abaqus_scripts/live_odb_probe.py), with a fake
odbAccess: it reports frames, history and the last field, and a failed
open is reported, not raised."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "abaqus_scripts" / "live_odb_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("live_odb_probe", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Keyed(dict):
    pass


def _fake_odb():
    blk = _Obj(data=[[1.0]] * 7)
    frame = _Obj(frameValue=2e-4,
                 fieldOutputs={"EVF": _Obj(bulkDataBlocks=[blk])})
    hist = {"RF1 at RP": _Obj(data=[(0.0, 1.0), (1e-4, 2.0), (2.1e-4, 3.0)]),
            "RF2 at RP": _Obj(data=[(0.0, 1.0)])}
    step = _Obj(frames=[frame, frame],
                historyRegions={"Node RP": _Obj(historyOutputs=hist)})
    closed = []
    odb = _Obj(steps={"Cut": step}, close=lambda: closed.append(True))
    return odb, closed


def test_probe_reads_frames_history_and_field():
    mod = _load()
    odb, closed = _fake_odb()
    mod.openOdb = lambda path, readOnly=True: odb
    res = mod.probe_once("x.odb", "Cut")
    assert res["frames"] == 2 and res["t_last_frame"] == 2e-4
    assert res["history_points"] == 3 and res["t_last_history"] == 2.1e-4
    assert res["field"] == "EVF" and res["field_values"] == 7
    assert closed and "error" not in res
    assert "frames 2" in mod._fmt(res)


def test_probe_reports_a_failed_open():
    mod = _load()

    def boom(path, readOnly=True):
        raise RuntimeError("odb is locked")
    mod.openOdb = boom
    res = mod.probe_once("x.odb", "Cut")
    assert "odb is locked" in res["error"]
    assert "ERROR" in mod._fmt(res)


def test_options():
    mod = _load()
    path, opts = mod._parse(["a.odb", "--every", "5", "--count", "3"])
    assert path == "a.odb" and opts["every"] == 5.0 and opts["count"] == 3
    with pytest.raises(SystemExit):
        mod._parse([])
