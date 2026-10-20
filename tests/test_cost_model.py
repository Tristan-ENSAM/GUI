# -*- coding: utf-8 -*-
"""Predicted solver time t = a + b * N_inc (gui.sensitivity.cost_model)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from gui.sensitivity.cost_model import (
    fit_cost, fmt_duration, n_inc_at_ms, remaining_from_progress)
from gui.sensitivity.study_export import ms_run_rows

# ms study of 2026-10-19 on CL-WTS-1 (2 um, 20 CPU): ms, N_inc, solver [s].
MS_2UM = [(1, 3683198, 37266), (200, 260217, 3858), (400, 183989, 3092),
          (800, 130108, 2748), (1600, 91997, 2145), (3200, 65058, 1435),
          (6400, 45995, 1330), (12800, 32523, 1080), (25600, 23003, 2927),
          (51200, 16253, 5873)]


def test_fit_leaves_out_the_runs_slowed_by_another_process():
    fit = fit_cost([n for _m, n, _t in MS_2UM], [t for _m, _n, t in MS_2UM])
    assert fit.a == pytest.approx(820, rel=0.02)
    assert fit.b == pytest.approx(0.0112, rel=0.02)
    assert fit.kept == [True] * 8 + [False, False]
    assert fit.predict(23003) == pytest.approx(1079, rel=0.02)


def test_fit_needs_two_distinct_runs():
    assert fit_cost([100], [10]) is None
    assert fit_cost([100, 100], [10, 11]) is None
    f = fit_cost([100, 200], [20, 30])
    assert (f.a, f.b) == (pytest.approx(10), pytest.approx(0.1))


def test_increments_scale_as_one_over_sqrt_ms():
    n = n_inc_at_ms(51200, [12800, 25600], [32523, 23003])
    assert n == pytest.approx(23003 / math.sqrt(2))      # nearest: 25600
    assert n == pytest.approx(16253, rel=0.001)           # measured


def test_remaining_time_of_a_running_job():
    assert remaining_from_progress(600, 0.25) == pytest.approx(1800)
    assert remaining_from_progress(600, 0.0) is None
    assert remaining_from_progress(None, 0.5) is None
    assert fmt_duration(45) == "45 s"
    assert fmt_duration(1080) == "18 min"
    assert fmt_duration(37266) == "10 h 21"
    assert fmt_duration(2.38e6) == "27 d 13 h"


def test_ms_export_carries_the_predicted_time():
    runs = [SimpleNamespace(
        index=i, dims={}, guards={}, guards_ok=True, job_ok=True, error="",
        cost=SimpleNamespace(n_inc=n, t_wall_solver_s=t))
        for i, (_m, n, t) in enumerate(MS_2UM)]
    res = SimpleNamespace(runs=runs, run_ms=[m for m, _n, _t in MS_2UM],
                          settings={"elem_size": 0.002}, retained=25600)
    rows = ms_run_rows(res)
    assert rows[8]["t_wall_solver_pred_s"] == pytest.approx(1079, rel=0.02)
    assert rows[8]["cost_in_fit"] is False and rows[0]["cost_in_fit"]
