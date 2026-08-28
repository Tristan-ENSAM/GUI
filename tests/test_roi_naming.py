# -*- coding: utf-8 -*-
"""
There is a SINGLE region of interest in the pipeline: the ROI (the field
extraction zone). It is materialised in the Abaqus model as two declinations
of the same zone -- ROI_node (nodes) and ROI_elem (elements).

The older name "ZOI" (zone of interest) was a synonym that coexisted with
"ROI" and made the code ambiguous; run_simul.py even created the same node
set twice ('ROI' and 'ZOI_nodes'). These tests lock the unified convention in
so the duplication cannot silently come back.
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
# The ROI sets are created by the model builder since the split.
_RUN_SIMUL = _REPO / "abaqus_scripts" / "cel_model.py"


def _python_sources():
    for base in ("gui", "abaqus_scripts", "tests"):
        for path in (_REPO / base).rglob("*.py"):
            if "__pycache__" in path.parts or path.name == Path(__file__).name:
                continue
            yield path


class TestRoiNaming:
    def test_no_zoi_anywhere(self):
        offenders = [
            str(p.relative_to(_REPO))
            for p in _python_sources()
            if "ZOI" in p.read_text(encoding="utf-8")
        ]
        assert offenders == [], (
            "ZOI is the retired synonym of ROI; found in: %s" % offenders)

    def test_run_simul_creates_both_roi_declinations(self):
        src = _RUN_SIMUL.read_text(encoding="utf-8")
        assert "Set(name='ROI_node', nodes=roi_nodes)" in src
        assert "Set(name='ROI_elem', elements=roi_elems)" in src

    def test_no_duplicate_plain_roi_node_set(self):
        # 'ROI' used to be created as a node set IDENTICAL to 'ZOI_nodes'
        # (same `roi_nodes`), i.e. the same zone under two names.
        src = _RUN_SIMUL.read_text(encoding="utf-8")
        assert "Set(name='ROI', nodes=" not in src
