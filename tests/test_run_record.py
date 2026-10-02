# -*- coding: utf-8 -*-
"""Tests for gui.sensitivity.run_record (lot L2: run safeguards and cost)."""
from __future__ import annotations

import numpy as np
import pytest

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.run_record import (
    CostRecord, GuardSettings, cost_record, euler_element_count,
    evaluate_guards, guard_reasons, make_guard_fn, missing_outputs,
    windowed_energy_ratio, DEFAULT_RHG_MAX,
)


class _Info:
    def __init__(self, fields, n):
        self.field_variables = fields
        self.n_elements = n


class _Bundle:
    """History channels given as a dict name -> array; fields declared."""

    def __init__(self, hist, fields=("EVF", "TEMP", "V1", "V2"),
                 history_time=None, n_eul=1200, n_tool=300):
        self._h = {k: np.asarray(v, dtype=float) for k, v in hist.items()}
        self._fields = list(fields)
        self._ht = history_time
        self._n = {"EULER": n_eul, "TOOL": n_tool}

    @property
    def instance_names(self):
        return ["EULER", "TOOL"]

    def instance(self, name):
        if name == "EULER":
            return _Info(self._fields, self._n["EULER"])
        return _Info([], self._n["TOOL"])

    def history(self, name):
        if name not in self._h:
            raise KeyError(name)
        return self._h[name]

    @property
    def history_time(self):
        return np.zeros(0) if self._ht is None else np.asarray(self._ht)


def _full(t, ke, ie, ae, ae_t=None):
    return {"ENERGY_TIME": t, "ALLKE": ke, "ALLIE": ie, "ALLAE": ae,
            "ALLAE_TIME": t if ae_t is None else ae_t,
            "RF1_RP": np.ones_like(t), "RF2_RP": np.ones_like(t)}


T = np.linspace(0.0, 1.0, 11)          # window (0.3, 1.0) keeps 0.3 .. 1.0


class TestEnergyRatio:
    def test_ratio_of_sums_restricted_to_window(self):
        ke = np.where(T < 0.3, 1000.0, 1.0)      # huge transient outside T
        ie = np.where(T < 0.3, 1e-9, 100.0)
        b = _Bundle(_full(T, ke, ie, ie * 0.02))
        v, why = windowed_energy_ratio(b, "ALLKE", "ALLIE", "ENERGY_TIME",
                                       "ENERGY_TIME", (0.3, 1.0))
        assert why == "" and v == pytest.approx(0.01)

    def test_own_time_bases(self):
        # ALLAE sampled twice as densely: each restricted with its own time.
        t2 = np.linspace(0.0, 1.0, 21)
        b = _Bundle(_full(T, np.ones(11), np.full(11, 10.0),
                          np.full(21, 0.2), ae_t=t2))
        v, _ = windowed_energy_ratio(b, "ALLAE", "ALLIE", "ALLAE_TIME",
                                     "ENERGY_TIME", (0.3, 1.0))
        n_ae = int(np.sum(t2 >= 0.3))
        n_ie = int(np.sum(T >= 0.3))
        assert v == pytest.approx(0.2 * n_ae / (10.0 * n_ie))

    def test_old_bundle_falls_back_on_history_time(self):
        h = {"ALLKE": np.ones(11), "ALLIE": np.full(11, 100.0)}
        v, why = windowed_energy_ratio(_Bundle(h, history_time=T), "ALLKE",
                                       "ALLIE", "ENERGY_TIME", "ENERGY_TIME")
        assert why == "" and v == pytest.approx(0.01)

    def test_no_time_base_is_not_evaluable(self):
        h = {"ALLKE": np.ones(11), "ALLIE": np.full(11, 100.0)}
        v, why = windowed_energy_ratio(_Bundle(h), "ALLKE", "ALLIE",
                                       "ENERGY_TIME", "ENERGY_TIME")
        assert v is None and "time" in why

    def test_non_positive_denominator(self):
        b = _Bundle(_full(T, np.ones(11), np.zeros(11), np.zeros(11)))
        v, why = windowed_energy_ratio(b, "ALLKE", "ALLIE", "ENERGY_TIME",
                                       "ENERGY_TIME")
        assert v is None and why


class TestGuards:
    def test_all_pass(self):
        b = _Bundle(_full(T, np.full(11, 0.5), np.full(11, 100.0),
                          np.full(11, 1.0)))
        g = evaluate_guards(b)
        assert g["outputs"] == (0.0, True)
        assert g["R_K"][0] == pytest.approx(0.005) and g["R_K"][1]
        assert g["R_HG"][0] == pytest.approx(0.01) and g["R_HG"][1]

    def test_rhg_default_threshold_is_5_percent(self):
        assert DEFAULT_RHG_MAX == 0.05
        b = _Bundle(_full(T, np.full(11, 0.5), np.full(11, 100.0),
                          np.full(11, 6.0)))
        assert evaluate_guards(b)["R_HG"][1] is False
        loose = GuardSettings(rhg_max=0.1)
        assert evaluate_guards(b, loose)["R_HG"][1] is True

    def test_missing_allae_fails_with_reason(self):
        h = _full(T, np.full(11, 0.5), np.full(11, 100.0), np.ones(11))
        del h["ALLAE"]
        b = _Bundle(h)
        g = evaluate_guards(b)
        assert g["R_HG"] == (None, False)
        assert g["outputs"][1] is False
        assert "ALLAE" in missing_outputs(b, GuardSettings())
        assert "R_HG" in guard_reasons(b)

    def test_missing_field_fails_outputs(self):
        b = _Bundle(_full(T, np.full(11, 0.5), np.full(11, 100.0),
                          np.ones(11)), fields=("EVF", "TEMP", "V1"))
        assert evaluate_guards(b)["outputs"] == (1.0, False)

    def test_make_guard_fn(self):
        fn = make_guard_fn(GuardSettings(rk_max=1e-6))
        b = _Bundle(_full(T, np.full(11, 0.5), np.full(11, 100.0),
                          np.ones(11)))
        assert fn(b)["R_K"][1] is False


class TestCost:
    def test_model_element_count_from_dims(self):
        d = DomainDims(h_wp=0.3, h_void=0.2, l_wp=0.5, l_void=0.2)
        assert euler_element_count(d, 0.01) == 70 * 50

    def test_cost_record_from_sta_and_bundle(self, tmp_path):
        sta = tmp_path / "job.sta"
        sta.write_text(
            "  100  1.0E-06 1.0E-06  00:00:10 6.0E-10   1  1.0E-06  1.0E-01\n"
            "  250  2.0E-06 2.0E-06  00:02:00 5.0E-10   1  1.0E-06  1.0E-01\n",
            encoding="latin-1")
        d = DomainDims(h_wp=0.1, h_void=0.1, l_wp=0.1, l_void=0.1)
        b = _Bundle({}, n_eul=123, n_tool=45)
        c = cost_record(b, sta, host_wall_s=180.0, n_cpu=4, dims=d,
                        elem_size=0.01)
        assert c.n_elem_euler == 400                 # model count, not crop
        assert c.n_elem_euler_extracted == 123
        assert c.n_elem_tool_extracted == 45
        assert c.n_inc == 250
        assert c.dt_stable_first == pytest.approx(6e-10)
        assert c.dt_stable_min == pytest.approx(5e-10)
        assert c.t_wall_solver_s == 120.0
        assert c.c_cpu_s == pytest.approx(480.0)     # Eq. 11 on solver time
        assert c.t_wall_host_s == 180.0
        assert set(c.as_dict()) >= {"c_cpu_s", "n_inc"}

    def test_cost_record_without_sta(self):
        c = cost_record(None, None, host_wall_s=5.0, n_cpu=2)
        assert isinstance(c, CostRecord)
        assert c.c_cpu_s is None and c.t_wall_host_s == 5.0
