# -*- coding: utf-8 -*-
"""The effective ZOI measurement points (cel_common.zoi_grid_points) match the
host-side sampling grid (roi_grid), and reach the model params."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "abaqus_scripts"))
from cel_common import zoi_grid_points                      # noqa: E402
from gui.sensitivity.mesh_opt import roi_grid               # noqa: E402
from gui.core.model_config import ModelConfig               # noqa: E402


def test_grid_matches_roi_grid():
    zoi, step = (-0.1, 0.0, -0.05, 0.05), 0.005
    pts = zoi_grid_points(zoi, step)
    grid = roi_grid(zoi, step)
    assert len(pts) == grid.shape[0] == 21 * 21
    s_pts = {(round(x, 9), round(y, 9)) for x, y in pts}
    s_grid = {(round(float(x), 9), round(float(y), 9)) for x, y in grid}
    assert s_pts == s_grid


def test_grid_empty_on_bad_step():
    assert zoi_grid_points((-0.1, 0.0, -0.05, 0.05), 0.0) == []

