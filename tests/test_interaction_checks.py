# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.interaction_checks (lot L4, paper §5.7)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from gui.core.domain_sizing import DomainDims
from gui.core.model_config import ModelConfig
import gui.sensitivity.domain_independence as di
import gui.sensitivity.interaction_checks as ic
from gui.sensitivity.interaction_checks import (
    combined_domain_check, combined_domain_dims, mass_scaling_window_check,
    mesh_domain_check, run_interaction_checks)
from gui.sensitivity.mesh_gci import MeshGciResult, QuantityGci
from gui.sensitivity.run_record import RecordingRunner

from tests.test_domain_independence import (   # analytic bundle + helpers
    _Bundle, _Cfg, _D0, _ELEM, _ZOI, _runner, _dims_of)


@pytest.fixture(autouse=True)
def _patch_sampling(monkeypatch):
    monkeypatch.setattr(di, "nearest_samples",
                        lambda b, var, inst, pts: b.field(inst, var))
    monkeypatch.setattr(di, "roi_grid",
                        lambda roi, step: np.zeros((10, 2)))


def _study(thr=None, n_max=8, **kw):
    log = []
    res = di.run_domain_independence(
        _runner(log), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
        thresholds=thr or {"Vx": 50.0}, step_elems=5, n_max=n_max,
        m_ratios=2, **kw)
    return res, log


class TestCombinedDomain:
    def test_dims_grown_and_clamped(self):
        d = combined_domain_dims(_D0, 0.05, {"h_void": 0.12}, 0.01)
        assert d.l_wp == pytest.approx(0.15)
        assert d.h_void == pytest.approx(0.12)

    def test_reuses_d_star_and_runs_one_domain(self):
        study, log = _study()
        n0 = len(log)
        run = _runner(log)
        # Check judged with a looser tolerance than the per-dimension one,
        # to exercise the passing branch (see the additivity test below).
        study.settings["thresholds"] = {"Vx": 200.0}
        c = combined_domain_check(run, _Cfg(), study)
        assert len(log) == n0 + 1                     # only D+ was run
        assert c.passed is True and c.safeguards_ok is True
        assert c.e_max < 1.0
        assert set(c.details["D_plus"]) == {"h_wp", "h_void", "l_wp",
                                            "l_void"}

    def test_boundary_influences_add_up(self):
        """Each dimension passes its own bound, yet growing the four at once
        moves the ZOI by the SUM of the four increments (the analytic
        influences are additive): the combined check detects it."""
        study, _ = _study()
        assert study.status == "converged"
        c = combined_domain_check(_runner(), _Cfg(), study)
        d = study.final
        delta = 5 * _ELEM
        expect = 1000.0 * sum(
            math.exp(-getattr(d, n) / 0.1)
            - math.exp(-(getattr(d, n) + delta) / 0.1)
            for n in ("h_wp", "h_void", "l_wp", "l_void"))
        assert c.details["errors"]["Vx"] == pytest.approx(expect)
        assert c.passed is False

    def test_fails_when_tolerances_too_tight(self):
        study, _ = _study()
        study.settings["thresholds"] = {"Vx": 1e-9}
        c = combined_domain_check(_runner(), _Cfg(), study)
        assert c.passed is False and "E_max" in c.conclusion

    def test_all_at_cap_is_not_evaluable(self):
        study, _ = _study()
        study.settings["caps"] = {n: getattr(study.final, n)
                                  for n in ("h_wp", "h_void", "l_wp",
                                            "l_void")}
        c = combined_domain_check(_runner(), _Cfg(), study)
        assert c.passed is None and "cap" in c.conclusion


def _gci(sizes, scalars, per_q, rec=None):
    return MeshGciResult(sizes=list(sizes), scalars=scalars,
                         per_quantity=per_q, recommended_size=rec,
                         in_asymptotic_range=True)


class TestMeshDomain:
    def test_h_star_within_tolerance(self):
        g = _gci([0.005, 0.01, 0.02],
                 {0.005: {"TEMP": 100.0}, 0.01: {"TEMP": 101.0},
                  0.02: {"TEMP": 104.0}},
                 {"TEMP": QuantityGci(100.0, 2.0, 99.67, 0.004, 0.01, 1.0,
                                      True, True)}, rec=0.01)
        c = mesh_domain_check(g, 0.01, {"TEMP": 0.02})
        assert c.passed is True
        assert c.e_max == pytest.approx(abs(101 - 99.67) / 99.67 / 0.02)

    def test_unreliable_uses_finest(self):
        g = _gci([0.005, 0.01, 0.02],
                 {0.005: {"Fc": 80.0}, 0.01: {"Fc": 79.0},
                  0.02: {"Fc": 81.0}},
                 {"Fc": QuantityGci(80.0, float("nan"), float("nan"), 0, 0,
                                    float("nan"), False, False)})
        c = mesh_domain_check(g, 0.02, {"Fc": 0.02})
        assert c.passed is True
        assert c.e_max == pytest.approx(1.0 / 80.0 / 0.02)

    def test_h_star_not_in_plan(self):
        g = _gci([0.005, 0.01, 0.02], {}, {})
        c = mesh_domain_check(g, 0.0075, {"TEMP": 0.02})
        assert c.passed is None and "not in the GCI plan" in c.conclusion

    def test_outside_tolerance_and_safeguards(self):
        g = _gci([0.005, 0.01, 0.02],
                 {0.005: {"TEMP": 100.0}, 0.01: {"TEMP": 110.0},
                  0.02: {"TEMP": 130.0}},
                 {"TEMP": QuantityGci(100.0, 1.0, 90.0, 0.1, 0.2, 1.0,
                                      True, True)})
        assert mesh_domain_check(g, 0.01, {"TEMP": 0.02}).passed is False
        bad = [SimpleNamespace(guards_ok=False)]
        c = mesh_domain_check(g, 0.005, {"TEMP": 0.5}, bad)
        assert c.passed is False and c.safeguards_ok is False

    def test_no_result(self):
        assert mesh_domain_check(None, 0.01, {}).passed is None


class TestMassScalingWindow:
    def _cfg(self, f, filt=True):
        cfg = ModelConfig()
        cfg.step.mass_scaling_enabled = True
        cfg.step.mass_scaling_factor = f
        cfg.step.output_filter_enabled = filt
        return cfg

    def test_lower_bound_decides(self):
        cfg = self._cfg(1.0)
        d = DomainDims(0.3, 0.2, 0.5, 0.2)
        b = cfg.mass_scaling_bounds(cfg.step.output_filter_cutoff_hz)
        ok = mass_scaling_window_check(self._cfg(b["ms_min"] * 1.01),
                                       cfg.elem_size, d)
        ko = mass_scaling_window_check(self._cfg(b["ms_min"] * 0.5),
                                       cfg.elem_size, d)
        assert ok.passed is True and ko.passed is False
        assert ko.details["ms_min"] == pytest.approx(b["ms_min"])

    def test_upper_bounds_are_warnings_only(self):
        d = DomainDims(0.3, 0.2, 0.5, 0.2)
        cfg = self._cfg(1e12)
        c = mass_scaling_window_check(cfg, cfg.elem_size, d)
        assert c.passed is True
        assert any("reverberation" in w for w in c.warnings)

    def test_filter_disabled(self):
        cfg = self._cfg(1.0, filt=False)
        c = mass_scaling_window_check(cfg, cfg.elem_size, _D0)
        assert c.passed is True and "disabled" in c.conclusion

    def test_cfg_not_modified(self):
        cfg = self._cfg(10.0)
        before = (cfg.elem_size, cfg.euler_geometry.l_wp)
        mass_scaling_window_check(cfg, 0.123, DomainDims(1, 1, 1, 1))
        assert (cfg.elem_size, cfg.euler_geometry.l_wp) == before


class TestRunAll:
    def test_three_checks_and_gci_on_d_star(self, monkeypatch):
        study, log = _study()
        seen = {}

        def fake_gci(run_bundle, base_cfg, domain_dims, tolerances,
                     should_cancel=None, progress_cb=None, **plan):
            seen["dims"] = domain_dims
            seen["plan"] = plan
            for h in (0.01, 0.02, 0.04):
                c = _Cfg()
                c.elem_size = h
                c.euler_geometry.h_wp = domain_dims.h_wp
                run_bundle(c)
            return _gci([0.01, 0.02, 0.04],
                        {0.01: {"TEMP": 100.0}, 0.02: {"TEMP": 100.5},
                         0.04: {"TEMP": 102.0}},
                        {"TEMP": QuantityGci(100.0, 2.0, 99.8, 0.003, 0.01,
                                             1.0, True, True)}, rec=0.02)
        monkeypatch.setattr("gui.sensitivity.mesh_gci.run_mesh_gci",
                            fake_gci)
        cfg = ModelConfig()
        cfg.step.output_filter_enabled = False
        study.settings["thresholds"] = {"Vx": 200.0}   # see additivity test
        events = []
        out = run_interaction_checks(
            _runner(log), cfg, study, h_star=0.01,
            gci_plan={"zoi": _ZOI, "grid_step": 0.01,
                      "finest_elem_size": 0.01, "ratio": 2.0, "n_meshes": 3},
            gci_tolerances={"TEMP": 0.02},
            gci_runner_factory=lambda rb: RecordingRunner(rb, n_cpu=2),
            progress_cb=events.append)
        assert seen["dims"] == study.final
        assert [c.name for c in out.checks] == ["ms_x_mesh",
                                                "domain_combined",
                                                "mesh_x_domain"]
        assert len(out.gci_calls) == 3
        assert out.status == "accepted" and out.accepted
        assert events[-1]["phase"] == "checks_done"

    def test_gci_failure_is_not_evaluable(self, monkeypatch):
        study, _ = _study()

        def boom(**kw):
            raise RuntimeError("mesh run produced no bundle")
        monkeypatch.setattr("gui.sensitivity.mesh_gci.run_mesh_gci",
                            lambda *a, **k: boom())
        cfg = ModelConfig()
        cfg.step.output_filter_enabled = False
        out = run_interaction_checks(
            _runner(), cfg, study, h_star=0.01, gci_plan={},
            gci_tolerances={"TEMP": 0.02})
        assert out.status == "incomplete"
        assert out.checks[-1].passed is None
        assert any("GCI on D*" in w for w in study.warnings)
