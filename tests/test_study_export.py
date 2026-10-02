# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.study_export (lot L4, paper tables/appendix)."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

import gui.sensitivity.domain_independence as di
from gui.sensitivity.interaction_checks import ChecksResult, CheckResult
from gui.sensitivity.mesh_gci import MeshGciResult, QuantityGci
from gui.sensitivity.run_record import CallRecord, CostRecord
from gui.sensitivity.study_export import (
    comparison_rows, dimension_rows, gci_mesh_rows, gci_rows, pareto_flags,
    retained_run_index, run_rows, summary, write_domain_exports,
    write_gci_exports)

from tests.test_domain_independence import (
    _Cfg, _D0, _ELEM, _ZOI, _runner)


@pytest.fixture(autouse=True)
def _patch_sampling(monkeypatch):
    monkeypatch.setattr(di, "nearest_samples",
                        lambda b, var, inst, pts: b.field(inst, var))
    monkeypatch.setattr(di, "roi_grid", lambda roi, step: np.zeros((10, 2)))


def _cost_fn(bundle, dims, host):
    n = int(round((dims.l_wp + dims.l_void) / _ELEM) *
            round((dims.h_wp + dims.h_void) / _ELEM))
    return CostRecord(n_elem_euler=n, n_cpu=2, t_wall_solver_s=n * 0.01,
                      c_cpu_s=2 * n * 0.01, t_wall_host_s=host)


@pytest.fixture
def study():
    return di.run_domain_independence(
        _runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM, thresholds={"Vx": 50.0},
        step_elems=5, n_max=8, m_ratios=2, cost_fn=_cost_fn)


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestRows:
    def test_comparisons_have_errors_bounds_and_pareto(self, study):
        rows = comparison_rows(study)
        n = sum(len(d.comparisons) for d in study.per_dimension.values())
        assert len(rows) == n
        r = rows[0]
        assert r["E_Vx"] is not None and r["E_Vx/eps"] == pytest.approx(
            r["E_Vx"] / 50.0)
        assert r["E_Fc"] is None                         # not thresholded
        assert any(x["R_Vx"] is not None for x in rows)  # decision rows
        assert all(x["pareto_non_dominated"] in (True, False) for x in rows)

    def test_runs_one_row_per_simulation(self, study):
        rows = run_rows(study, ms_factor=8.0)
        assert len(rows) == study.n_runs
        assert rows[0]["parameter"] == "initial"
        assert rows[0]["decision"] == "start"
        assert rows[1]["E_Vx"] is not None
        assert {r["mass_scaling_factor"] for r in rows} == {8.0}
        assert rows[0]["n_elem_euler"] > 0
        retained = [r for r in rows if r["retained_for"]]
        names = " ".join(r["retained_for"] for r in retained).split()
        assert sorted(names) == sorted(study.per_dimension)

    def test_retained_run_index(self, study):
        for d in study.per_dimension.values():
            i = retained_run_index(d)
            run = study.runs[i]
            assert run.dims[d.name] == pytest.approx(d.retained)

    def test_dimension_rows_normalised_by_t1(self, study):
        rows = dimension_rows(study, t1=0.05)
        r = rows[0]
        assert r["selected_over_t1"] == pytest.approx(r["selected_mm"] / 0.05)
        assert dimension_rows(study, None)[0]["selected_over_t1"] is None

    def test_pareto_flags(self):
        f = pareto_flags([(1, 5), (2, 1), (3, 3), (None, 1), (1, 5)])
        assert f == [True, True, False, None, True]


class TestSummary:
    def test_eq22_23_both_references(self, study):
        s = summary(study, t1=0.05, h_star=_ELEM, ms_factor=1.0)
        c = s["cost_eq22_23"]["by_C_CPU"]
        ini = c["vs_initial_domain"]
        assert ini["R_C"] == pytest.approx(ini["C_ref"] / ini["C_opt"])
        assert ini["G_C"] == pytest.approx(1 - ini["C_opt"] / ini["C_ref"])
        over = c["vs_oversized_domain"]
        assert over["C_ref"] >= over["C_opt"]
        assert s["normalised_eq24"]["h_over_t1"] == pytest.approx(0.2)
        assert s["selected_domain_mm"]["l_wp"] == pytest.approx(
            study.final.l_wp)

    def test_missing_cost_gives_none(self):
        res = di.run_domain_independence(
            _runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 50.0}, step_elems=5, order=("l_wp",))
        s = summary(res, None, None, None)
        assert s["cost_eq22_23"]["by_C_CPU"]["vs_initial_domain"]["G_C"] \
            is None
        assert s["normalised_eq24"] is None


def _gci():
    return MeshGciResult(
        sizes=[0.005, 0.01, 0.02],
        scalars={0.005: {"TEMP": 100.0}, 0.01: {"TEMP": 101.0},
                 0.02: {"TEMP": 104.0}},
        per_quantity={"TEMP": QuantityGci(100.0, 1.58, 99.5, 0.006, 0.02,
                                          1.0, True, True)},
        recommended_size=0.01, in_asymptotic_range=True)


class TestGciAndFiles:
    def test_gci_rows_and_meshes(self):
        rows = gci_rows(_gci(), {"TEMP": 0.02})
        assert rows[0]["f1_finest"] == 100.0 and rows[0]["f3"] == 104.0
        calls = [CallRecord(i, h, {}, True,
                            CostRecord(n_elem_euler=10 * (i + 1), c_cpu_s=1.0),
                            {"R_K": (0.001, True)})
                 for i, h in enumerate([0.005, 0.01, 0.02])]
        m = gci_mesh_rows(_gci(), calls)
        assert [r["h_mm"] for r in m] == [0.005, 0.01, 0.02]
        assert [r["recommended"] for r in m] == [False, True, False]
        assert m[1]["n_elem_euler"] == 20 and m[1]["R_K_ok"] is True
        assert gci_rows(None) == [] and gci_mesh_rows(None) == []

    def test_written_files(self, study, tmp_path):
        chk = ChecksResult(checks=[CheckResult("ms_x_mesh", "p", True,
                                               conclusion="ok")],
                           gci_on_d_star=_gci(), status="accepted")
        paths = write_domain_exports(tmp_path, study, 0.05, _ELEM, 1.0, chk)
        names = {p.name for p in paths}
        assert {"runs.csv", "comparisons.csv", "dimensions.csv",
                "checks.csv", "summary.json",
                "checks_gci_on_D_star.csv"} <= names
        runs = _read(tmp_path / "runs.csv")
        assert len(runs) == study.n_runs
        s = json.loads((tmp_path / "summary.json").read_text())
        assert s["interaction_checks"]["status"] == "accepted"
        g = write_gci_exports(tmp_path / "gci", _gci(), None, {"TEMP": 0.02})
        assert _read(g[0])[0]["quantity"] == "TEMP"
        # NaN / None are written as empty cells, never "nan"
        assert "nan" not in (tmp_path / "comparisons.csv").read_text()


def test_paper_cost_reference_is_the_oversized_domain(study):
    s = summary(study, t1=0.05, h_star=_ELEM, ms_factor=1.0)
    paper = s["cost_eq22_23_paper"]
    over = s["cost_eq22_23"]["by_C_CPU"]["vs_oversized_domain"]
    assert paper["G_C"] == over["G_C"] and paper["R_C"] == over["R_C"]
    assert "D12" in paper["reference"]
    costs = [r.cost.c_cpu_s for r in study.runs]
    assert paper["C_ref"] == pytest.approx(max(costs))
