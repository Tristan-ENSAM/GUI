# -*- coding: utf-8 -*-
"""
Field extraction in ``abaqus_scripts/cel_results.py`` through
``bulkDataBlocks`` must give exactly what the per-value loop gave.

The per-value loop (``for v in fo.values``) builds one Python object per value
of the instance, ROI-dropped values included; the bulk path reads labels and
data as numpy arrays. These fakes expose both interfaces on the same data, so
each test runs the extraction twice (bulk on / bulk off) and compares.
Nothing here touches a real ODB.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "abaqus_scripts"


@pytest.fixture(scope="module")
def cel_results():
    saved = {n: sys.modules[n] for n in
             ("odbAccess", "abaqusConstants", "cel_results", "cel_common")
             if n in sys.modules}
    odb_stub = types.ModuleType("odbAccess")
    odb_stub.openOdb = lambda *a, **k: None
    const_stub = types.ModuleType("abaqusConstants")
    const_stub.CENTROID = "CENTROID"
    const_stub.NODAL = "NODAL"
    sys.modules["odbAccess"] = odb_stub
    sys.modules["abaqusConstants"] = const_stub
    added = str(_SCRIPTS) not in sys.path
    if added:
        sys.path.insert(0, str(_SCRIPTS))
    try:
        sys.modules.pop("cel_results", None)
        yield importlib.import_module("cel_results")
    finally:
        if added:
            sys.path.remove(str(_SCRIPTS))
        for n in ("odbAccess", "abaqusConstants", "cel_results", "cel_common"):
            sys.modules.pop(n, None)
        sys.modules.update(saved)


class _Value:
    def __init__(self, elem, node, data):
        self.elementLabel = elem
        self.nodeLabel = node
        self.data = data


class _Block:
    def __init__(self, elem, node, data):
        self.elementLabels = elem
        self.nodeLabels = node
        self.data = data


class _Field:
    """One frame of one field: labels + data, offered both ways."""

    def __init__(self, labels, data, nodal, bulk_on, n_blocks=2):
        self._labels = np.asarray(labels)
        self._data = np.asarray(data, dtype=np.float32)
        self._nodal = nodal
        self._bulk_on = bulk_on
        self._n_blocks = n_blocks
        self.componentLabels = ("V1", "V2") if self._data.ndim == 2 else ()

    def getSubset(self, **kw):
        return self

    @property
    def values(self):
        out = []
        for lbl, d in zip(self._labels, self._data):
            d = tuple(float(x) for x in d) if self._data.ndim == 2 else float(d)
            out.append(_Value(None if self._nodal else int(lbl),
                              int(lbl) if self._nodal else None, d))
        return out

    @property
    def bulkDataBlocks(self):
        if not self._bulk_on:
            raise AttributeError("bulkDataBlocks")
        blocks = []
        for idx in np.array_split(np.arange(len(self._labels)), self._n_blocks):
            lab = self._labels[idx]
            d = self._data[idx]
            d = d.reshape(len(idx), -1)
            blocks.append(_Block(None if self._nodal else lab,
                                 lab if self._nodal else None, d))
        return blocks


class _Frame:
    def __init__(self, fields):
        self.fieldOutputs = fields


class _Step:
    def __init__(self, frames):
        self.frames = frames


class _Assembly:
    def __init__(self, name):
        self.instances = {name: object()}


INST = "EULER-1"


def _make_step(key, n_frames, labels, per_frame, nodal, bulk_on):
    return _Step([_Frame({key: _Field(labels, per_frame(fi), nodal, bulk_on)})
                  for fi in range(n_frames)])


def test_scalar_field_bulk_matches_values(cel_results):
    rng = np.random.default_rng(0)
    labels = rng.permutation(np.arange(1, 401))          # 400 elements
    kept = list(range(50, 151))                           # ROI keeps 101
    data = rng.normal(size=(5, 400)).astype(np.float32)
    key = "TEMP_ASSEMBLY_EULER_EULER-1"
    res = []
    for bulk in (True, False):
        step = _make_step(key, 5, labels, lambda fi: data[fi], False, bulk)
        res.append(cel_results._extract_field(step, "TEMP", INST, kept,
                                              _Assembly(INST)))
    np.testing.assert_array_equal(res[0], res[1])
    assert not np.isnan(res[0]).any()


def test_nodal_vector_bulk_matches_values(cel_results):
    rng = np.random.default_rng(1)
    labels = rng.permutation(np.arange(1, 301))          # 300 nodes
    kept_nodes = list(range(10, 60))                      # 50 kept
    elements = rng.integers(0, 50, size=(30, 4))          # local node indices
    data = rng.normal(size=(4, 300, 2)).astype(np.float32)
    key = "V_CAMERABAND"
    res = []
    for bulk in (True, False):
        step = _make_step(key, 4, labels, lambda fi: data[fi], True, bulk)
        res.append(cel_results._extract_nodal_vector_to_elem(
            step, "V", INST, kept_nodes, elements, _Assembly(INST),
            filter_suffix="CAMERABAND"))
    for k in ("V1", "V2", "V"):
        np.testing.assert_array_equal(res[0][k], res[1][k])
    assert not np.isnan(res[0]["V"]).any()


def test_kept_label_absent_from_odb_stays_nan(cel_results):
    labels = np.array([1, 2, 3])
    step = _make_step("EVF_ASSEMBLY_EULER_EULER-1", 2, labels,
                      lambda fi: np.array([1.0, 2.0, 3.0]), False, True)
    out = cel_results._extract_field(step, "EVF", INST, [2, 99],
                                     _Assembly(INST))
    assert out[:, 0].tolist() == [2.0, 2.0]
    assert np.isnan(out[:, 1]).all()
