# -*- coding: utf-8 -*-
"""ROI and ZOI are two DISTINCT zones. Only the ROI is materialised in the
Abaqus model (ROI_node / ROI_elem, the extraction output set). The ZOI is a
host-side sampling concept; mesh nodes in its bbox are NOT the effective
measurement points, and an orphan-node display cloud does not survive (Abaqus
purges nodes not connected to elements), so the model stays ROI-only.
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_MODEL = _REPO / "abaqus_scripts" / "cel_model.py"


def _model_src():
    return _MODEL.read_text(encoding="utf-8")


class TestRoiNaming:
    def test_creates_both_roi_declinations(self):
        src = _model_src()
        assert "Set(name='ROI_node', nodes=roi_nodes)" in src
        assert "Set(name='ROI_elem', elements=roi_elems)" in src

    def test_model_creates_no_zoi_set(self):
        assert "name='ZOI" not in _model_src()

    def test_no_duplicate_plain_roi_node_set(self):
        assert "Set(name='ROI', nodes=" not in _model_src()
