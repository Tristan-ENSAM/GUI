# -*- coding: utf-8 -*-
"""
History-output extraction in ``abaqus_scripts/cel_results.py``.

Why this file exists
--------------------
A history request carrying a Butterworth filter is written to the ODB under a
SUFFIXED name: RF1 + the 'SensorBand' filter (cel_model.py:596-599) becomes
``RF1_SENSORBAND``, exactly like the filtered field outputs (V ->
V_CAMERABAND, cel_model.py:554-555). ``_extract_history_rf`` used to look up
the bare "RF1", found nothing, and returned (None, None, None); extraction
then wrote a bundle WITHOUT ``history__RF1_RP``, and the failure only surfaced
much later — and far from its cause — as
``KeyError: "No history variable 'RF1_RP'. Available: ['ALLKE','ALLIE']"``
raised by ``gui/results/reader.py:338-342``.

``cel_results`` imports ``odbAccess`` / ``abaqusConstants`` at module level, so
it is not importable outside Abaqus. The fixture below installs minimal stubs
for those two modules; nothing in this file touches a real ODB.
"""
from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "abaqus_scripts"
_CEL_RESULTS = _SCRIPTS / "cel_results.py"


# ---------------------------------------------------------------------------
# Import harness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cel_results():
    """Import abaqus_scripts/cel_results.py with the Abaqus modules stubbed."""
    saved_modules = {}
    for name in ("odbAccess", "abaqusConstants", "cel_results", "cel_common"):
        if name in sys.modules:
            saved_modules[name] = sys.modules[name]

    odb_stub = types.ModuleType("odbAccess")
    odb_stub.openOdb = lambda *a, **k: None          # never called here
    const_stub = types.ModuleType("abaqusConstants")
    const_stub.CENTROID = "CENTROID"
    const_stub.NODAL = "NODAL"
    sys.modules["odbAccess"] = odb_stub
    sys.modules["abaqusConstants"] = const_stub

    path_added = str(_SCRIPTS) not in sys.path
    if path_added:
        sys.path.insert(0, str(_SCRIPTS))
    try:
        sys.modules.pop("cel_results", None)
        module = importlib.import_module("cel_results")
        yield module
    finally:
        if path_added:
            sys.path.remove(str(_SCRIPTS))
        for name in ("odbAccess", "abaqusConstants", "cel_results", "cel_common"):
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


# ---------------------------------------------------------------------------
# Minimal ODB doubles
# ---------------------------------------------------------------------------
class _FakeOutput:
    """Stand-in for a HistoryOutput: only ``.data`` (a list of (t, v) pairs)."""

    def __init__(self, pairs):
        self.data = list(pairs)


class _FakeRegion:
    """Stand-in for a HistoryRegion. ``historyOutputs`` is a plain dict, which
    offers the same three operations the production code uses on the Abaqus
    repository: ``in``, ``keys()`` and ``[]``."""

    def __init__(self, outputs):
        self.historyOutputs = outputs


class _FakeStep:
    def __init__(self, regions):
        self.historyRegions = regions


def _pairs(values, dt=1.0e-6):
    return [(i * dt, float(v)) for i, v in enumerate(values)]


def _rp_region(rf1_key, rf2_key, rf1=(10.0, 20.0, 30.0), rf2=(1.0, 2.0, 3.0)):
    return _FakeRegion({rf1_key: _FakeOutput(_pairs(rf1)),
                        rf2_key: _FakeOutput(_pairs(rf2))})


def _energy_region(ke_key="ALLKE", ie_key="ALLIE"):
    return _FakeRegion({ke_key: _FakeOutput(_pairs((5.0, 6.0, 7.0))),
                        ie_key: _FakeOutput(_pairs((50.0, 60.0, 70.0)))})


# ---------------------------------------------------------------------------
# _find_history_key
# ---------------------------------------------------------------------------
class TestFindHistoryKey:
    def test_exact_name_wins_when_no_filter_was_requested(self, cel_results):
        """Both series now coexist in every filtered run (create_step always
        emits the unfiltered request too). With no filter asked for, the bare
        name is the one to read."""
        outputs = {"RF1": _FakeOutput([]), "RF1_SENSORBAND": _FakeOutput([])}
        assert cel_results._find_history_key(outputs, "RF1") == "RF1"

    def test_filtered_series_wins_when_a_filter_was_requested(self, cel_results):
        """THE REGRESSION THIS GUARDS: once the unfiltered request is always
        emitted, 'RF1' and 'RF1_SENSORBAND' sit side by side. Preferring the
        exact name would silently extract forces that were never
        band-limited -- the opposite of what enabling the filter asks for."""
        outputs = {"RF1": _FakeOutput([]), "RF1_SENSORBAND": _FakeOutput([])}
        assert (cel_results._find_history_key(outputs, "RF1", "SENSORBAND")
                == "RF1_SENSORBAND")

    def test_filter_requested_but_absent_falls_back_to_the_bare_name(
            self, cel_results):
        """A filter configured for the FIELD outputs only leaves the history
        unsuffixed; extraction must still find it rather than return None."""
        outputs = {"RF1": _FakeOutput([])}
        assert (cel_results._find_history_key(outputs, "RF1", "SENSORBAND")
                == "RF1")

    def test_a_foreign_suffix_is_not_mistaken_for_the_filtered_series(
            self, cel_results):
        """Only a suffix carrying the filter's name counts."""
        outputs = {"RF1": _FakeOutput([]), "RF1_SOMETHINGELSE": _FakeOutput([])}
        assert (cel_results._find_history_key(outputs, "RF1", "SENSORBAND")
                == "RF1")

    def test_suffixed_name_is_resolved(self, cel_results):
        outputs = {"RF1_SENSORBAND": _FakeOutput([])}
        assert (cel_results._find_history_key(outputs, "RF1")
                == "RF1_SENSORBAND")

    def test_absent_returns_none(self, cel_results):
        outputs = {"ALLKE": _FakeOutput([]), "ALLIE": _FakeOutput([])}
        assert cel_results._find_history_key(outputs, "RF1") is None

    def test_no_prefix_collision(self, cel_results):
        """'RF1' must not be satisfied by 'RF12' or by a key merely
        CONTAINING the base name: the match is on '<base>_'."""
        outputs = {"RF12": _FakeOutput([]), "XRF1_SENSORBAND": _FakeOutput([])}
        assert cel_results._find_history_key(outputs, "RF1") is None

    def test_ambiguity_is_deterministic_and_reported(self, cel_results, capsys):
        outputs = {"RF1_B": _FakeOutput([]), "RF1_A": _FakeOutput([])}
        assert cel_results._find_history_key(outputs, "RF1") == "RF1_A"
        assert "[WARNING]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _extract_history_rf
# ---------------------------------------------------------------------------
class TestExtractHistoryRF:
    def test_unfiltered_run(self, cel_results):
        step = _FakeStep({"Node RP": _rp_region("RF1", "RF2")})
        t, rf1, rf2 = cel_results._extract_history_rf(step)
        assert t is not None
        np.testing.assert_allclose(rf1, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(rf2, [1.0, 2.0, 3.0])

    def test_filtered_run_is_the_regression_case(self, cel_results):
        """The exact shape of the failed campaign: SensorBand on, energies
        present under their bare names in another region."""
        step = _FakeStep({
            "Assembly ASSEMBLY": _energy_region(),
            "Node RP": _rp_region("RF1_SENSORBAND", "RF2_SENSORBAND"),
        })
        t, rf1, rf2 = cel_results._extract_history_rf(step)
        assert t is not None, "filtered RF1/RF2 must still be found"
        np.testing.assert_allclose(rf1, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(rf2, [1.0, 2.0, 3.0])

    def test_energy_only_region_is_skipped_not_matched(self, cel_results):
        step = _FakeStep({"Assembly ASSEMBLY": _energy_region()})
        assert cel_results._extract_history_rf(step) == (None, None, None)

    def test_time_comes_from_the_rf1_channel(self, cel_results):
        step = _FakeStep({"Node RP": _rp_region("RF1_SENSORBAND",
                                                "RF2_SENSORBAND")})
        t, _, _ = cel_results._extract_history_rf(step)
        np.testing.assert_allclose(t, [0.0, 1.0e-6, 2.0e-6])
        assert t.dtype == np.float64

    def test_half_filtered_region_is_not_matched(self, cel_results):
        """RF1 filtered but RF2 missing: no partial extraction."""
        step = _FakeStep({"Node RP": _FakeRegion(
            {"RF1_SENSORBAND": _FakeOutput(_pairs((1.0, 2.0)))})})
        assert cel_results._extract_history_rf(step) == (None, None, None)


# ---------------------------------------------------------------------------
# _extract_history_energy
# ---------------------------------------------------------------------------
class TestExtractHistoryEnergy:
    def test_bare_names(self, cel_results):
        """H-Output-2 (PRESELECT) is deliberately unfiltered
        (cel_model.py:601-606): the bare names are the normal case."""
        step = _FakeStep({"Assembly ASSEMBLY": _energy_region()})
        t, ke, ie = cel_results._extract_history_energy(step)
        assert t is not None
        np.testing.assert_allclose(ke, [5.0, 6.0, 7.0])
        np.testing.assert_allclose(ie, [50.0, 60.0, 70.0])

    def test_suffixed_names_would_also_be_found(self, cel_results):
        """Defensive path: guards the day that request gets a filter."""
        step = _FakeStep({"Assembly ASSEMBLY": _energy_region(
            "ALLKE_SENSORBAND", "ALLIE_SENSORBAND")})
        t, ke, ie = cel_results._extract_history_energy(step)
        assert t is not None
        np.testing.assert_allclose(ke, [5.0, 6.0, 7.0])

    def test_rp_region_alone_yields_nothing(self, cel_results):
        step = _FakeStep({"Node RP": _rp_region("RF1_SENSORBAND",
                                                "RF2_SENSORBAND")})
        assert cel_results._extract_history_energy(step) == (None, None, None)


# ---------------------------------------------------------------------------
# Python 2.7 compatibility — cel_results.py runs under Abaqus Python
# ---------------------------------------------------------------------------
class TestPython27Compatibility:
    def test_no_fstrings_or_annotations(self):
        tree = ast.parse(_CEL_RESULTS.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not isinstance(node, ast.JoinedStr), \
                "f-strings are not valid in Python 2.7"
            assert not isinstance(node, ast.AnnAssign), \
                "variable annotations are not valid in Python 2.7"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.returns is None, \
                    "return annotations are not valid in Python 2.7"
                for a in list(node.args.args) + list(node.args.kwonlyargs):
                    assert a.annotation is None, \
                        "argument annotations are not valid in Python 2.7"


# ---------------------------------------------------------------------------
# The bare-name lookup must not come back
# ---------------------------------------------------------------------------
class TestNoHardCodedBareNames:
    def test_extractors_go_through_the_resolver(self):
        """Locks the fix: the two extractors must not index historyOutputs
        with a literal channel name again."""
        src = _CEL_RESULTS.read_text(encoding="utf-8")
        for literal in ('outputs["RF1"]', 'outputs["RF2"]',
                        'outputs["ALLKE"]', 'outputs["ALLIE"]'):
            assert literal not in src, (
                "%s bypasses _find_history_key and breaks as soon as a "
                "filter suffixes the name" % literal)
