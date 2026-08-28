# -*- coding: utf-8 -*-
"""
abaqus_scripts/cel_common.py holds the pure-Python helpers shared by the GUI
(Python 3.x) and run_simul.py (Abaqus Python 2.7).

Two invariants matter and are locked here:
  1. there is exactly ONE implementation (resolve_tool_translation used to be
     duplicated and the two copies had already drifted apart in their error
     messages);
  2. the module stays importable by Python 2.7, which forbids f-strings, type
     annotations and 3.x-only stdlib.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SHARED = _REPO / "abaqus_scripts" / "cel_common.py"
_RUN_SIMUL = _REPO / "abaqus_scripts" / "run_simul.py"
# Since the split, the model-building code (and its cel_common import) lives
# in cel_model.py; run_simul.py is a thin orchestrator.
_CEL_MODEL = _REPO / "abaqus_scripts" / "cel_model.py"
_GUI_CALC = _REPO / "gui" / "core" / "tool_geometry_calc.py"


class TestSingleImplementation:
    def test_helpers_defined_only_in_shared_module(self):
        for name in ("def resolve_tool_translation", "def solve_tool_dimensions",
                     "def cfg_get", "def discretize"):
            assert name in _SHARED.read_text(encoding="utf-8")
            assert name not in _CEL_MODEL.read_text(encoding="utf-8"), (
                "%s must not be re-defined in cel_model.py" % name)
            assert name not in _GUI_CALC.read_text(encoding="utf-8"), (
                "%s must not be re-defined in tool_geometry_calc.py" % name)

    def test_gui_reexports_shared_symbols(self):
        # Existing callers import these from the GUI module; they must keep
        # working after the move.
        from gui.core.tool_geometry_calc import (
            ToolGeometryError, solve_tool_dimensions, resolve_tool_translation)
        assert resolve_tool_translation(
            0.3, 0.3, 0.01, 30.0, 20.0, 0.0, -0.04, 0.0)[3] == "fillet-tangent-x"
        with pytest.raises(ToolGeometryError):
            solve_tool_dimensions(0.3, 0.3, 60.0, 40.0)


class TestPython27Compatibility:
    """The shared module is imported by Abaqus Python 2.7."""

    def test_no_fstrings_or_annotations(self):
        tree = ast.parse(_SHARED.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not isinstance(node, ast.JoinedStr), \
                "f-strings are not valid in Python 2.7"
            assert not isinstance(node, ast.AnnAssign), \
                "variable annotations are not valid in Python 2.7"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.returns is None, \
                    "return annotations are not valid in Python 2.7"
                args = node.args
                for a in list(args.args) + list(args.kwonlyargs):
                    assert a.annotation is None, \
                        "argument annotations are not valid in Python 2.7"

    def test_no_future_annotations_import(self):
        # Checked on the AST, not the raw text: the module docstring legitimately
        # MENTIONS this import as something to avoid.
        tree = ast.parse(_SHARED.read_text(encoding="utf-8"))
        futures = [a.name for n in tree.body
                   if isinstance(n, ast.ImportFrom) and n.module == "__future__"
                   for a in n.names]
        assert "annotations" not in futures

    def test_only_stdlib_math_at_module_level(self):
        # numpy must NOT creep in: dependency-freeness is what makes this
        # module safe to import from both interpreters.
        tree = ast.parse(_SHARED.read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = []
        for n in top:
            if isinstance(n, ast.Import):
                names += [a.name for a in n.names]
            else:
                names.append(n.module)
        assert names == ["math"], "unexpected module-level imports: %s" % names


class TestRunSimulImportsShared:
    def test_run_simul_imports_from_cel_common(self):
        src = _CEL_MODEL.read_text(encoding="utf-8")
        assert "from cel_common import" in src

    def test_plain_import_is_enough(self):
        # No sys.path juggling is needed: Python puts the executed script's
        # own directory first on sys.path, and cel_common.py sits next to
        # cel_model.py. Confirmed by real Abaqus runs under `abaqus cae
        # noGUI=`. This test just pins the files being colocated.
        assert _SHARED.parent == _CEL_MODEL.parent
