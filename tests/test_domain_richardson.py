# -*- coding: utf-8 -*-
"""Richardson test on the domain (gui.sensitivity.domain_richardson)."""
from __future__ import annotations

import json
import math

import pytest

from gui.core.domain_sizing import DomainDims
from gui.core.model_config import ModelConfig
from gui.sensitivity import domain_richardson as dr

BASE = DomainDims(h_wp=0.1, h_void=0.08, l_wp=0.2, l_void=0.12)


def test_series_scales_every_side_and_snaps_up():
    doms, scales = dr.scaled_series(BASE, 1.5, 3, elem_size=0.002)
    assert doms[0].l_wp == pytest.approx(0.2)
    assert doms[1].l_wp == pytest.approx(0.3)
    assert doms[2].l_wp == pytest.approx(0.45)
    for d in doms:
        for s in dr.SIDES:
            n = getattr(d, s) / 0.002
            assert n == pytest.approx(round(n))
    assert scales[0] == 1.0
    assert scales[1] == pytest.approx(1.5, rel=2e-2)
    assert scales[2] == pytest.approx(2.25, rel=2e-2)


def test_series_rejects_bad_input():
    with pytest.raises(ValueError):
        dr.scaled_series(BASE, 1.0)
    with pytest.raises(ValueError):
        dr.scaled_series(BASE, 1.5, 2)
    with pytest.raises(ValueError):
        dr.scaled_series(DomainDims(0.0, 1, 1, 1), 1.5)


def _power(s, f_inf=10.0, c=4.0, p=2.0):
    return f_inf + c / s ** p


def test_power_law_is_extrapolated():
    scales = [1.0, 1.5, 2.25]
    sc = [{"Vx": _power(s)} for s in scales]
    per_q, ref, kind, dev, passing = dr.evaluate(sc, scales, {"Vx": 0.5},
                                                 ["Vx"])
    assert per_q["Vx"].p == pytest.approx(2.0, rel=1e-6)
    assert kind["Vx"] == "richardson"
    assert ref["Vx"] == pytest.approx(10.0, rel=1e-9)
    # |f - 10| / 0.5: 8, 3.56, 1.58 -> none passes.
    assert dev[0]["Vx"] == pytest.approx(8.0)
    assert passing is None
    _, _, _, _, passing = dr.evaluate(sc, scales, {"Vx": 2.0}, ["Vx"])
    assert passing == 1


def test_low_order_falls_back_to_largest_domain():
    scales = [1.0, 1.5, 2.25]
    sc = [{"T": _power(s, p=0.2)} for s in scales]
    per_q, ref, kind, dev, _ = dr.evaluate(sc, scales, {"T": 1.0}, ["T"])
    assert per_q["T"].p < 0.5
    assert kind["T"] == "largest"
    assert ref["T"] == sc[2]["T"]
    assert dev[2]["T"] == 0.0


def test_non_monotone_series_falls_back():
    scales = [1.0, 1.5, 2.25]
    sc = [{"Fc": 1.0}, {"Fc": 2.0}, {"Fc": 1.5}]
    _, ref, kind, _, _ = dr.evaluate(sc, scales, {"Fc": 1.0}, ["Fc"])
    assert kind["Fc"] == "largest" and ref["Fc"] == 1.5


def test_nan_counts_as_failing():
    scales = [1.0, 1.5, 2.25]
    sc = [{"Vx": float("nan")}, {"Vx": 1.0}, {"Vx": 1.0}]
    _, _, _, dev, passing = dr.evaluate(sc, scales, {"Vx": 1.0}, ["Vx"])
    assert dev[0]["Vx"] == math.inf
    assert passing == 1


def _fake_run(monkeypatch, calls):
    def scalars(bundle, *a, **k):
        s = bundle["l_wp"] / BASE.l_wp
        return {"V1": _power(s), "Fc": 0.002 * _power(s, 50.0, 8.0, 1.0)}
    monkeypatch.setattr(dr, "zoi_scalars", scalars)

    def run(cfg):
        calls.append({s: getattr(cfg.euler_geometry, s) for s in dr.SIDES})
        return {"l_wp": cfg.euler_geometry.l_wp}
    return run


def test_run_series_on_fake_solver(monkeypatch):
    calls = []
    cfg = ModelConfig()
    before = cfg.euler_geometry.l_wp
    res = dr.run_domain_richardson(
        _fake_run(monkeypatch, calls), cfg, (0, 1, 0, 1), BASE, 0.002,
        tolerances={"V1": 0.5, "Fc": 1.0}, field_vars=("V1",),
        force_channels={"Fc": "RF1_RP"})
    assert res.stopped_by == "ok"
    assert len(calls) == 3
    assert calls[2]["h_void"] > calls[1]["h_void"] > calls[0]["h_void"]
    assert cfg.euler_geometry.l_wp == before          # model untouched
    # Force divided by h (per unit width): 50 + 8/s.
    assert res.scalars[0]["Fc"] == pytest.approx(58.0)
    assert res.reference_kind == {"V1": "richardson", "Fc": "richardson"}
    rec = json.loads(json.dumps(res.record()))
    assert rec["power_law"] is True
    assert rec["per_quantity"]["V1"]["reference"] == pytest.approx(10.0,
                                                                   rel=1e-2)


def test_run_stops_on_missing_run_or_cancel(monkeypatch):
    res = dr.run_domain_richardson(lambda c: None, ModelConfig(),
                                   (0, 1, 0, 1), BASE, 0.002)
    assert res.stopped_by == "no_bundle" and not res.per_quantity
    res = dr.run_domain_richardson(lambda c: None, ModelConfig(),
                                   (0, 1, 0, 1), BASE, 0.002,
                                   should_cancel=lambda: True)
    assert res.stopped_by == "cancelled"


def test_tab_button_and_result_file(qapp, tmp_path, monkeypatch):
    from gui.tabs.optimization_tab import OptimizationTab
    tab = OptimizationTab(ModelConfig())
    assert tab.btn_dom_rich.text().startswith("Richardson test")
    calls = []
    res = dr.run_domain_richardson(
        _fake_run(monkeypatch, calls), ModelConfig(), (0, 1, 0, 1), BASE,
        0.002, tolerances={"V1": 0.5}, field_vars=("V1",),
        force_channels={})
    tab._pending_dr = (tmp_path, {})
    tab._on_dr_done(res)
    rec = json.loads((tmp_path / "domain_richardson.json").read_text())
    assert rec["study"] == "domain_richardson"
    assert "power law" in tab.lbl_status.text()
    tab.le_dom_rich_ratio.setText("1")
    warned = []
    monkeypatch.setattr("gui.tabs.optimization_tab.QMessageBox.warning",
                        lambda *a, **k: warned.append(a[1]))
    monkeypatch.setattr(tab, "thresholds_complete", lambda: True)
    monkeypatch.setattr(tab, "window", lambda: (0.3, 1.0))
    monkeypatch.setattr(tab, "_shares_or_warn", lambda t: [1, 1, 1])
    assert tab._domain_richardson_spec() is None
    assert warned == ["Richardson test on the domain"]
    tab.le_dom_rich_ratio.setText("1.5")
    spec = tab._domain_richardson_spec()
    assert spec["ratio"] == 1.5 and spec["n_levels"] == 3
