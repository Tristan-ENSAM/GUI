# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.domain_independence (paper §4.1-4.4 domain study).

The run_bundle is analytic: the boundary influence on the ZOI decays
exponentially with each domain dimension, so the successive errors decay
geometrically with a constant step (ratio exp(-D/lam)). ZOI sampling is
bypassed: the analytic bundle IS the ZOI sample (one column per grid point).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from gui.core.domain_sizing import DomainDims
import gui.sensitivity.domain_independence as di
from gui.sensitivity.domain_independence import (
    AlignmentError, ZoiSample, check_geometric_decay, e_max, errors_between,
    initial_dims_from_zoi, mean_abs_difference, run_domain_independence,
    sample_zoi, tail_bound, zoi_inside, domain_bounds,
)

_NP = 10                         # ZOI grid points: 7 material, 3 void
_NT = 11                         # field frames
_NH = 21                         # history samples
_MAT = np.array([True] * 7 + [False] * 3)
_ELEM = 0.01                     # mm
_LAM = 0.1                       # influence decay length (mm)
_DIMS = ("h_wp", "h_void", "l_wp", "l_void")


class _Bundle:
    """Analytic bundle: every quantity = base * (1 + influence(dims))."""

    def __init__(self, dims, lam=_LAM, wobble=None, times=None):
        self.dims, self.lam, self.wobble = dims, lam, wobble
        self._times = (np.linspace(0.0, 1e-4, _NT) if times is None
                       else np.asarray(times))

    @property
    def times(self):
        return self._times

    @property
    def instance_names(self):
        return ["EULER"]

    def instance(self, name):
        class _Info:
            field_variables = ["EVF", "TEMP", "V1", "V2"]
        return _Info()

    def influence(self):
        return sum(np.exp(-getattr(self.dims, n) / self.lam) for n in _DIMS)

    def field(self, inst, var):
        out = np.zeros((len(self._times), _NP), dtype=float)
        if var == "EVF":
            out[:, _MAT] = 1.0
            return out
        base = {"TEMP": 300.0, "V1": 1000.0, "V2": -500.0}[var]
        out[:, _MAT] = base * (1.0 + self.influence())
        out[:, ~_MAT] = 12345.0          # garbage in the void: must be masked
        return out

    @property
    def history_time(self):
        return np.linspace(0.0, 1e-4, _NH)

    def history(self, var):
        base = {"RF1_RP": 0.8, "RF2_RP": 0.3}[var]
        if self.wobble is not None and var == "RF2_RP":
            # Ff carries ONLY a growing alternating term: its increments grow,
            # so the geometric-decay hypothesis is rejected at every step.
            return np.full(_NH, base + self.wobble(self.dims))
        return np.full(_NH, base * (1.0 + self.influence()))


class _G:
    def __init__(self):
        self.h_wp = self.h_void = self.l_wp = self.l_void = 0.0


class _Cfg:
    def __init__(self):
        self.euler_geometry = _G()


@pytest.fixture(autouse=True)
def _patch_sampling(monkeypatch):
    monkeypatch.setattr(di, "nearest_samples",
                        lambda b, var, inst, pts: b.field(inst, var))
    monkeypatch.setattr(di, "roi_grid", lambda roi, step: np.zeros((_NP, 2)))


def _dims_of(cfg):
    g = cfg.euler_geometry
    return DomainDims(h_wp=g.h_wp, h_void=g.h_void, l_wp=g.l_wp,
                      l_void=g.l_void)


def _runner(log=None, fail=None, **kw):
    def run(cfg):
        d = _dims_of(cfg)
        if log is not None:
            log.append(d)
        if fail is not None and fail(d):
            return None
        return _Bundle(d, **kw)
    return run


_ZOI = (-0.05, 0.05, -0.05, 0.05)
_D0 = DomainDims(h_wp=0.1, h_void=0.1, l_wp=0.1, l_void=0.1)
_THR = {"Vx": 1.0, "Vy": 0.5, "T": 0.3, "EVF": 0.01, "Fc": 0.01, "Ff": 0.01}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_initial_dims_zoi_plus_margin_snapped(self):
        d = initial_dims_from_zoi((-0.123, 0.051, -0.08, 0.02), 0.01,
                                  margin_elems=2)
        assert d.l_wp == pytest.approx(0.15)     # 0.123+0.02 -> 15 elems
        assert d.l_void == pytest.approx(0.08)   # 0.051+0.02 -> 8 elems
        assert d.h_wp == pytest.approx(0.10)
        assert d.h_void == pytest.approx(0.04)

    def test_initial_dims_respect_euler_offset(self):
        d0 = initial_dims_from_zoi((-0.1, 0.1, -0.1, 0.1), 0.01)
        d1 = initial_dims_from_zoi((0.9, 1.1, -0.1, 0.1), 0.01,
                                   offset=(1.0, 0.0))
        assert d1 == d0

    def test_zoi_inside_with_offset_and_margin(self):
        dims = DomainDims(h_wp=0.1, h_void=0.1, l_wp=0.1, l_void=0.1)
        assert domain_bounds(dims, (1.0, 0.0)) == pytest.approx(
            (0.9, 1.1, -0.1, 0.1))
        assert zoi_inside(dims, (0.95, 1.05, -0.05, 0.05), 0.05, (1.0, 0.0))
        assert not zoi_inside(dims, (0.95, 1.05, -0.05, 0.05), 0.06,
                              (1.0, 0.0))
        assert not zoi_inside(dims, (-0.05, 0.05, -0.05, 0.05), 0.0,
                              (1.0, 0.0))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_mad_ignores_nan_in_either(self):
        a = np.array([[1.0, np.nan, 3.0]])
        b = np.array([[2.0, 5.0, np.nan]])
        assert mean_abs_difference(a, b) == pytest.approx(1.0)
        assert math.isnan(mean_abs_difference(a[:, 1:2], b[:, 1:2]))

    def test_sample_masks_void_and_normalises_forces(self):
        s = sample_zoi(_Bundle(_D0), _ZOI, 0.01, _ELEM, window=(0.3, 1.0))
        assert np.all(np.isnan(s.fields["Vx"][:, ~_MAT]))
        assert np.all(np.isfinite(s.fields["Vx"][:, _MAT]))
        assert np.all(s.fields["EVF"][:, ~_MAT] == 0.0)   # EVF not masked
        expect = 0.8 * (1.0 + _Bundle(_D0).influence()) / _ELEM
        assert s.forces["Fc"] == pytest.approx(np.full(s.forces["Fc"].size,
                                                       expect))
        assert s.field_times.min() >= 0.3e-4 - 1e-15

    def test_errors_physical_units(self):
        sa = sample_zoi(_Bundle(_D0), _ZOI, 0.01, _ELEM)
        d1 = DomainDims(h_wp=0.2, h_void=0.1, l_wp=0.1, l_void=0.1)
        sb = sample_zoi(_Bundle(d1), _ZOI, 0.01, _ELEM)
        e = errors_between(sb, sa)
        dinf = (np.exp(-0.1 / _LAM) - np.exp(-0.2 / _LAM))
        assert e["Vx"] == pytest.approx(1000.0 * dinf)
        assert e["T"] == pytest.approx(300.0 * dinf)
        assert e["EVF"] == pytest.approx(0.0)
        assert e["Fc"] == pytest.approx(0.8 * dinf / _ELEM)

    def test_misaligned_times_refused(self):
        sa = sample_zoi(_Bundle(_D0), _ZOI, 0.01, _ELEM)
        sb = sample_zoi(_Bundle(_D0, times=np.linspace(0, 1e-4, _NT + 3)),
                        _ZOI, 0.01, _ELEM)
        with pytest.raises(AlignmentError):
            errors_between(sa, sb)

    def test_e_max_nan_is_inf_and_q_crit(self):
        em, qc = e_max({"Vx": 0.5, "T": 0.9}, {"Vx": 1.0, "T": 1.0})
        assert (em, qc) == (pytest.approx(0.9), "T")
        em, qc = e_max({"Vx": float("nan"), "T": 0.1}, {"Vx": 1.0, "T": 1.0})
        assert em == float("inf") and qc == "Vx"


# ---------------------------------------------------------------------------
# Decay test and tail bound
# ---------------------------------------------------------------------------
class TestDecay:
    def test_geometric_accepted_rho_is_max(self):
        c = check_geometric_decay([8.0, 4.0, 2.0, 0.9], 2)
        assert c.accepted and c.rho == pytest.approx(0.5)

    def test_equal_ratios_accepted_despite_round_off(self):
        e = [math.exp(-k * 0.37) for k in range(1, 8)]
        assert check_geometric_decay(e, 3).accepted

    def test_slowing_decay_rejected(self):
        c = check_geometric_decay([8.0, 2.0, 1.0], 2)        # 0.25 then 0.5
        assert not c.accepted and c.reason == "decay_slowing"

    def test_ratio_not_below_one_rejected(self):
        assert not check_geometric_decay([1.0, 2.0, 1.0], 2).accepted

    def test_too_few_comparisons(self):
        c = check_geometric_decay([1.0, 0.5], 2)
        assert not c.accepted and c.reason == "too_few_comparisons"

    def test_vanished_increments_accepted(self):
        c = check_geometric_decay([1.0, 0.0, 0.0], 2)
        assert c.accepted and c.rho == 0.0
        assert not check_geometric_decay([0.0, 0.1, 0.0], 2).accepted

    def test_nan_rejected(self):
        assert not check_geometric_decay([1.0, float("nan"), 0.2], 2).accepted

    def test_tail_bound_formula(self):
        e = [4.0, 2.0, 1.0]
        assert tail_bound(e, 2, 0.5) == pytest.approx(1.0 + 1.0)
        assert tail_bound(e, 0, 0.5) == pytest.approx(7.0 + 1.0)
        assert tail_bound(e, 1, 1.0) == float("inf")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _expected_tail_retained(p0, delta, n_comp, eps, base, rho):
    """Smallest p_j satisfying the bound for an exact exponential influence."""
    E = [base * (math.exp(-(p0 + (i - 1) * delta) / _LAM)
                 - math.exp(-(p0 + i * delta) / _LAM))
         for i in range(1, n_comp + 1)]
    for j in range(n_comp):
        if tail_bound(E, j, rho) < eps:
            return p0 + j * delta
    return None


def _growing_wobble(d):
    k = int(round(d.l_wp / _ELEM))
    return 1e-7 * k * k * (1 if k % 2 else -1)


class TestDriver:
    def test_tail_bound_path_and_cache_and_no_mutation(self):
        cfg = _Cfg()
        before = copy.deepcopy(cfg.euler_geometry.__dict__)
        log = []
        res = run_domain_independence(
            _runner(log), cfg, _ZOI, _D0, grid_step=0.01, elem_size=_ELEM,
            thresholds={"Vx": 50.0}, step_elems=5, n_max=8, m_ratios=2)
        assert cfg.euler_geometry.__dict__ == before
        keys = [tuple(round(getattr(d, n) / _ELEM) for n in _DIMS)
                for d in log]
        assert len(keys) == len(set(keys))              # cache: no rerun
        assert res.n_runs == len(log) <= 1 + 4 * 8
        dr = res.per_dimension["l_wp"]
        assert dr.status == "tail_bound"
        # Vx error for growing l_wp alone; other dims fixed -> same ratio
        rho = math.exp(-0.05 / _LAM)
        n_comp = len(dr.comparisons)
        assert n_comp >= 3
        exp = _expected_tail_retained(0.1, 0.05, n_comp, 50.0, 1000.0, rho)
        assert dr.retained == pytest.approx(exp)
        assert res.status == "converged"
        assert res.final.l_wp == pytest.approx(dr.retained)

    def test_no_decision_before_m_plus_one_comparisons(self):
        res = run_domain_independence(
            _runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 1e9}, step_elems=5, n_max=8, m_ratios=2)
        for dr in res.per_dimension.values():
            assert len(dr.comparisons) == 3              # first possible
            assert dr.status == "tail_bound"
            assert dr.retained == pytest.approx(dr.initial)

    def test_global_fallback_when_one_quantity_not_geometric(self):
        # Ff gets an oscillating term: its increments do not decay
        # geometrically -> EVERY quantity falls back to the successive rule.
        wob = _growing_wobble
        res = run_domain_independence(
            _runner(wobble=wob), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 50.0, "Ff": 1.0}, step_elems=5, n_max=8,
            m_ratios=2, order=("l_wp",))
        dr = res.per_dimension["l_wp"]
        assert dr.status == "successive"
        assert dr.comparisons[-1].mode == "successive"
        # n_hold = 1: retained = first member of the first successful pair
        first_ok = next(c for c in dr.comparisons if c.success)
        assert dr.retained == pytest.approx(first_ok.value_from)

    def test_n_hold_two(self):
        wob = _growing_wobble
        res = run_domain_independence(
            _runner(wobble=wob), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 50.0, "Ff": 1.0}, step_elems=5, n_max=8,
            n_hold=2, m_ratios=2, order=("l_wp",))
        dr = res.per_dimension["l_wp"]
        succ = [c.success for c in dr.comparisons]
        k = next(i for i in range(1, len(succ)) if succ[i] and succ[i - 1])
        assert dr.retained == pytest.approx(dr.comparisons[k - 1].value_from)

    def test_not_converged_keeps_largest(self):
        res = run_domain_independence(
            _runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 1e-12}, step_elems=5, n_max=4, m_ratios=2,
            order=("h_void",))
        dr = res.per_dimension["h_void"]
        assert dr.status == "not_converged"
        assert dr.retained == pytest.approx(0.1 + 4 * 0.05)
        assert res.status == "partial"

    def test_cap_stops_and_keeps_cap(self):
        res = run_domain_independence(
            _runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 1e-12}, step_elems=5, n_max=8, m_ratios=2,
            caps={"h_void": 0.23}, order=("h_void",))
        dr = res.per_dimension["h_void"]
        assert dr.status == "cap"
        assert dr.values == pytest.approx([0.1, 0.15, 0.2, 0.23])
        assert dr.retained == pytest.approx(0.23)

    def test_failed_job_does_not_stop_the_study(self):
        bad = lambda d: abs(d.l_wp - 0.15) < 1e-9
        res = run_domain_independence(
            _runner(fail=bad), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 5.0}, step_elems=5, n_max=8, m_ratios=2,
            order=("l_wp",))
        failed = [r for r in res.runs if not r.job_ok]
        assert len(failed) == 1 and failed[0].error
        dr = res.per_dimension["l_wp"]
        assert len(dr.comparisons) > 3
        assert dr.retained > 0.15 + 1e-9        # nothing before the failure

    def test_diagonal_only_warns(self):
        big = DomainDims(h_wp=0.5, h_void=0.5, l_wp=0.5, l_void=0.5)
        res = run_domain_independence(
            _runner(), _Cfg(), _ZOI, big, 0.01, _ELEM,
            thresholds={"Vx": 5.0}, step_elems=5, n_max=4, m_ratios=2,
            order=("l_wp",))
        assert all(r.diagonal_warning for r in res.runs)
        assert res.warnings
        assert len(res.per_dimension["l_wp"].comparisons) >= 3

    def test_guard_failure_blocks_candidate(self):
        bad = lambda b: {"R_K": (0.5, abs(b.dims.l_wp - 0.1) > 1e-9)}
        res = run_domain_independence(
            _runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 1e9}, step_elems=5, n_max=8, m_ratios=2,
            order=("l_wp",), guard_fn=bad)
        dr = res.per_dimension["l_wp"]
        assert dr.retained == pytest.approx(0.15)   # p_0 fails its guard
        assert not res.runs[0].guards_ok

    def test_zoi_outside(self):
        res = run_domain_independence(
            _runner(), _Cfg(), (-0.5, 0.05, -0.05, 0.05), _D0, 0.01, _ELEM,
            thresholds={"Vx": 5.0})
        assert res.status == "zoi_outside" and res.n_runs == 0

    def test_cancel(self):
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 2
        res = run_domain_independence(
            _runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 5.0}, should_cancel=cancel)
        assert res.status == "cancelled"

    def test_invalid_settings(self):
        with pytest.raises(ValueError):
            run_domain_independence(_runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
                                    thresholds={"Vx": 1.0}, n_max=2,
                                    m_ratios=2)
        with pytest.raises(ValueError):
            run_domain_independence(_runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
                                    thresholds={})
        with pytest.raises(ValueError):
            run_domain_independence(_runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
                                    thresholds={"Vx": 1.0}, step_elems=0)

    def test_progress_events(self):
        ev = []
        run_domain_independence(
            _runner(), _Cfg(), _ZOI, _D0, 0.01, _ELEM,
            thresholds={"Vx": 5.0}, step_elems=5, order=("l_wp",),
            progress_cb=ev.append)
        phases = [e["phase"] for e in ev]
        assert phases[0] == "run" and phases[-1] == "done"
        assert "comparison" in phases and "dimension" in phases


class TestExtractionCrop:
    """The extraction may keep only the output ROI (cel_results.py:25-54)."""

    def test_zoi_outside_crop_is_refused(self):
        b = _Bundle(_D0)
        b.roi = {"xmin": -0.02, "xmax": 0.05, "ymin": -0.05, "ymax": 0.05}
        with pytest.raises(RuntimeError, match="not contained"):
            sample_zoi(b, _ZOI, 0.01, _ELEM)

    def test_zoi_inside_crop_is_accepted(self):
        b = _Bundle(_D0)
        b.roi = {"xmin": -0.1, "xmax": 0.1, "ymin": -0.1, "ymax": 0.1}
        sample_zoi(b, _ZOI, 0.01, _ELEM)

    def test_crop_failure_is_a_failed_run_not_a_crash(self):
        def run(cfg):
            b = _Bundle(_dims_of(cfg))
            b.roi = {"xmin": 0.0, "xmax": 0.01, "ymin": 0.0, "ymax": 0.01}
            return b
        res = run_domain_independence(
            run, _Cfg(), _ZOI, _D0, 0.01, _ELEM, thresholds={"Vx": 50.0},
            step_elems=5, n_max=3, m_ratios=2, order=("l_wp",))
        assert res.runs and not any(r.job_ok for r in res.runs)
        assert "not contained" in res.runs[0].error
        assert res.per_dimension["l_wp"].status == "not_converged"


def test_cost_fn_receives_dims_and_host_time():
    seen = []

    def cost(bundle, dims, host_s):
        seen.append((bundle is not None, dims, host_s))
        return {"n": len(seen)}
    res = run_domain_independence(
        _runner(fail=lambda d: abs(d.l_wp - 0.15) < 1e-9), _Cfg(), _ZOI, _D0,
        0.01, _ELEM, thresholds={"Vx": 50.0}, step_elems=5, n_max=3,
        m_ratios=2, order=("l_wp",), cost_fn=cost)
    assert len(seen) == res.n_runs
    assert all(h >= 0.0 for (_b, _d, h) in seen)
    assert [b for (b, _d, _h) in seen].count(False) == 1   # failed run too
    assert res.runs[0].cost == {"n": 1}
