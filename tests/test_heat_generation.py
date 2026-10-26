# -*- coding: utf-8 -*-
"""HeatGeneration keyword fallback in cel_model.py. Abaqus 2022 GA refuses
`slaveFraction` (TypeError "keyword error on slaveFraction", 2026-10-26), so
the helper tries each name of the slave fraction and, at the Abaqus default
0.5, finally leaves it out. cel_model.py imports Abaqus, so the helper is
extracted from the source and run against fake contact properties.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_MODEL = Path(__file__).resolve().parent.parent / "abaqus_scripts" / "cel_model.py"
_NAMES = ("_HEAT_FRACTION_KEYWORDS", "_ABAQUS_DEFAULT_SLAVE_FRACTION",
          "_set_heat_generation")


def _helper():
    tree = ast.parse(_MODEL.read_text(encoding="utf-8"))
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in _NAMES)
            or (isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) in _NAMES for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(_MODEL), "exec"), ns)
    return ns["_set_heat_generation"]


class _Prop:
    """Fake ContactProperty whose HeatGeneration accepts only `accepted`."""
    def __init__(self, accepted):
        self.accepted = set(accepted)
        self.calls = []

    def HeatGeneration(self, **kw):
        bad = set(kw) - self.accepted
        if bad:
            raise TypeError("keyword error on %s" % sorted(bad)[0])
        self.calls.append(kw)


def test_secondary_fraction_keyword():
    prop = _Prop({"conversionFraction", "secondaryFraction"})
    _helper()(prop, 0.5)
    assert prop.calls == [{"conversionFraction": 1.0, "secondaryFraction": 0.5}]


def test_slave_fraction_keyword_on_older_release():
    prop = _Prop({"conversionFraction", "slaveFraction"})
    _helper()(prop, 0.3)
    assert prop.calls == [{"conversionFraction": 1.0, "slaveFraction": 0.3}]


def test_default_fraction_left_out_when_no_keyword_works():
    prop = _Prop({"conversionFraction"})
    _helper()(prop, 0.5)
    assert prop.calls == [{"conversionFraction": 1.0}]


def test_non_default_fraction_without_keyword_raises():
    prop = _Prop({"conversionFraction"})
    with pytest.raises(TypeError):
        _helper()(prop, 0.3)
    assert prop.calls == []
