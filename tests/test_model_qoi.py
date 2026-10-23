# -*- coding: utf-8 -*-
"""Sensitivity QoI computed as the Model tab studies (gui.sensitivity.
model_qoi) and their eps_q-normalised columns (runner_core.add_eps_ratios)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gui.sensitivity import model_qoi as mq
from gui.sensitivity import runner_core as rc
from gui.sensitivity.domain_independence import field_scalars
from gui.sensitivity.zoi_sampling import history_window_mean, window_mask


class _Bundle:
    """2 x 2 elements of 1 mm, 5 frames over [0, 1] s, RF history."""

    def __init__(self, roi=None, elem=1.0):
        self._c = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0],
                            [0.5, 1.5, 0], [1.5, 1.5, 0]], dtype=float)
        self.times = np.linspace(0.0, 1.0, 5)
        t = self.times[:, None]
        self._f = {"V1": 100.0 * t + np.array([0.0, 10.0, 20.0, 30.0]),
                   "V2": -50.0 + 0.0 * t + np.zeros(4),
                   "TEMP": 20.0 + 100.0 * t + np.zeros(4),
                   "EVF": np.ones((5, 4))}
        self._f["EVF"][3, 1] = 0.2            # element 1 void at frame 3
        self.history_time = np.linspace(0.0, 1.0, 11)
        self._h = {"RF1_RP": 2.0 * self.history_time,
                   "RF2_RP": np.full(11, -3.0)}
        self.model_config = {"mesh": {"elem_size": elem}}
        self.roi = roi
        self.instance_names = ["EULER"]

    def instance(self, name):
        return SimpleNamespace(field_variables=list(self._f))

    def element_centroids_init(self, inst):
        return self._c

    def field(self, inst, var):
        return self._f[var]

    def history(self, var):
        return self._h[var]


ZOI = (0.0, 2.0, 0.0, 1.0)            # the two bottom elements
WINDOW = (0.5, 1.0)                   # frames 2, 3, 4


def _qois(bundle, **kw):
    specs = mq.make_specs(ZOI, WINDOW, 0.5, **kw)
    return {s.id: s.fn(bundle, None, 0.0) for s in specs}


def test_fields_match_the_studies_scalar():
    b = _Bundle()
    tmask = window_mask(b.times, *WINDOW)
    ref = field_scalars(b, "EULER", ZOI, ["Vx", "Vy", "T", "EVF"], tmask,
                        0.5, 1.0)
    got = _qois(b, qois=("Vx", "Vy", "T", "EVF"))
    for q in ("Vx", "Vy", "T", "EVF"):
        assert got[q] == pytest.approx(ref[q])
    # By hand: element 0 -> mean(50, 75, 100) = 75; element 1 is void at
    # frame 3 -> mean(60, 110) = 85; equal areas -> 80.
    assert got["Vx"] == pytest.approx(80.0)
    assert got["EVF"] == pytest.approx((1.0 + (1 + 0.2 + 1) / 3) / 2)


def test_forces_are_window_means_per_width():
    b = _Bundle(elem=0.5)
    got = _qois(b, qois=("Fc", "Ff"))
    assert got["Fc"] == pytest.approx(
        history_window_mean(b, "RF1_RP", *WINDOW) / 0.5)
    assert got["Fc"] == pytest.approx(np.mean(2.0 * np.linspace(.5, 1, 6))
                                      / 0.5)
    assert got["Ff"] == pytest.approx(-6.0)


def test_specs_carry_eps_key_and_unit():
    specs = mq.make_specs(ZOI, WINDOW)
    assert [s.id for s in specs] == list(mq.MODEL_QOIS)
    assert all(s.eps_key == s.id for s in specs)
    assert {s.id: s.unit for s in specs}["Fc"] == "N/mm"


def test_zoi_outside_extracted_roi_gives_nan_in_the_runner():
    b = _Bundle(roi={"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 2.0})
    specs = mq.make_specs(ZOI, WINDOW, qois=("Vx", "Fc"))
    with pytest.raises(RuntimeError):
        specs[0].fn(b, None, 0.0)
    out = rc.extract_qois(b, specs)
    assert np.isnan(out["Vx"]) and np.isfinite(out["Fc"])


def test_eps_ratio_jacobian():
    plan = SimpleNamespace(param_paths=["a", "b"], deltas=[0.5, 2.0])
    spec = SimpleNamespace(id="Vx", eps_key="Vx")
    analyses = {"Vx": {"a": {"dQdx": -40.0, "scheme_used": "central"},
                       "b": {"dQdx": 1.0, "scheme_used": "forward"}}}
    added = rc.add_eps_ratios(analyses, [spec], plan, "jacobian",
                              {"Vx": 10.0})
    col = "Vx" + rc.EPS_SUFFIX
    assert added == [col]
    assert analyses[col]["a"]["sensitivity"] == pytest.approx(2.0)
    assert analyses[col]["b"]["sensitivity"] == pytest.approx(0.2)
    assert analyses[col]["b"]["scheme_used"] == "forward"


def test_eps_ratio_morris_and_skips():
    plan = SimpleNamespace(param_paths=["a", "b"])
    specs = [SimpleNamespace(id="Fc", eps_key="Fc"),
             SimpleNamespace(id="T_max", eps_key=""),
             SimpleNamespace(id="Vy", eps_key="Vy")]
    analyses = {"Fc": {"names": ["a", "b"], "mu_star": [20.0, 5.0],
                       "sigma": [10.0, 0.0], "mu": [-20.0, 5.0],
                       "mu_star_conf": [2.0, 1.0]},
                "T_max": {"names": ["a", "b"], "mu_star": [1.0, 2.0]},
                "Vy": {"error": "too few complete trajectories"}}
    added = rc.add_eps_ratios(analyses, specs, plan, "morris",
                              {"Fc": 10.0, "Vy": 10.0})
    col = "Fc" + rc.EPS_SUFFIX
    assert added == [col]
    assert analyses[col]["mu_star"] == pytest.approx([2.0, 0.5])
    assert analyses[col]["sigma"] == pytest.approx([1.0, 0.0])
    assert rc.morris_ranking(SimpleNamespace(analyses=analyses), col)[0][0] \
        == "a"


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------
def _settings(**over):
    st = {"eps": {"Vx": 10.0, "Vy": 10.0, "T": 10.0, "EVF": 0.1,
                  "Fc": 10.0, "Ff": 10.0},
          "window": (0.3, 1.0), "zoi": None, "evf_threshold": 0.5}
    st.update(over)
    return st


def _tab_with_param(cfg):
    from PySide6.QtCore import Qt
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(cfg)
    for r in range(tab.table.rowCount()):
        it = tab.table.item(r, 0)
        if it is not None and it.flags() & Qt.ItemIsUserCheckable:
            it.setCheckState(Qt.Checked)
            break
    return tab


def _inside_zoi(cfg):
    b = cfg.bbox
    return (float(b.xmin), 0.5 * (float(b.xmin) + float(b.xmax)),
            float(b.ymin), 0.5 * (float(b.ymin) + float(b.ymax)))


def test_tab_defaults_follow_the_model_tab(qapp):
    from gui.core.model_config import ModelConfig
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig())
    assert not any(cb.isEnabled() for cb in tab._model_qoi_checks.values())
    assert tab._qoi_checks["Fx_mean"].isChecked()
    assert "Fc" not in tab._qoi_checks           # the Model-tab Fc only
    tab.set_model_settings_getter(lambda: _settings())
    assert all(cb.isChecked() for cb in tab._model_qoi_checks.values())
    assert not any(cb.isChecked() for cb in tab._qoi_checks.values())


def test_tab_generates_with_model_settings(qapp):
    from gui.core.model_config import ModelConfig
    cfg = ModelConfig()
    tab = _tab_with_param(cfg)
    tab.set_model_settings_getter(lambda: _settings(zoi=_inside_zoi(cfg)))
    tab._on_generate()
    assert tab.plan is not None, tab.status.text()
    assert [q.id for q in tab.selected_qois] == list(mq.MODEL_QOIS)
    rec = tab._plan_record([], 1)
    assert rec["model_qoi"]["eps"]["Fc"] == 10.0
    assert rec["model_qoi"]["window"] == [0.3, 1.0]


def test_tab_refuses_zoi_outside_roi_and_missing_eps(qapp):
    from gui.core.model_config import ModelConfig
    cfg = ModelConfig()
    tab = _tab_with_param(cfg)
    b = cfg.bbox
    far = (float(b.xmax) + 1.0, float(b.xmax) + 2.0,
           float(b.ymin), float(b.ymax))
    tab.set_model_settings_getter(lambda: _settings(zoi=far))
    tab._on_generate()
    assert tab.plan is None and "not inside the ROI" in tab.status.text()
    tab.chk_full_domain.setChecked(True)          # whole domain extracted
    tab._on_generate()
    assert tab.plan is not None
    eps = _settings()["eps"]
    eps.pop("Ff")
    tab.set_model_settings_getter(
        lambda: _settings(zoi=_inside_zoi(cfg), eps=eps))
    tab.plan = None
    tab._on_generate()
    assert tab.plan is None and "Ff" in tab.status.text()
