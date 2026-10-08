# -*- coding: utf-8 -*-
"""ZOI proposed from the sensitivity maps (gui.sensitivity.zoi_proposal),
the Fc / Ff per-width QoIs, the whole-domain extraction flag and the
Sensitivity -> Model tab wiring. Offline (no Abaqus)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from gui.sensitivity import zoi_proposal as zp


def _grid(nx=6, ny=4, h=1.0):
    """Unit-square elements on an nx x ny grid: verts (n_elem, 4, 2)."""
    verts = []
    for j in range(ny):
        for i in range(nx):
            x0, y0 = i * h, j * h
            verts.append([[x0, y0], [x0 + h, y0], [x0 + h, y0 + h],
                          [x0, y0 + h]])
    return np.asarray(verts, float)


def _idx(i, j, nx=6):
    return j * nx + i


TIMES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])     # window (0.3, 1) -> 3 frames
EPS = {"Vx": 10.0, "Vy": 10.0, "T": 10.0, "EVF": 0.1}


class TestMapSStar:

    def test_mean_over_window_of_abs_change_over_eps(self):
        S = np.array([[100.0], [100.0], [-2.0], [4.0], [-6.0]])
        tmask = np.array([False, False, True, True, True])
        out = zp.map_s_star(S, delta=5.0, eps=10.0, tmask=tmask,
                            material=None)
        # |S*delta|/eps = 1, 2, 3 on the window -> mean 2
        assert out.shape == (1,)
        assert out[0] == pytest.approx(2.0)

    def test_material_mask_drops_void_frames(self):
        S = np.array([[1.0], [1.0], [3.0]])
        tmask = np.array([True, True, True])
        material = np.array([[True], [False], [True]])
        out = zp.map_s_star(S, 1.0, 1.0, tmask, material)
        assert out[0] == pytest.approx(2.0)          # mean of 1 and 3

    def test_no_material_frame_is_nan(self):
        out = zp.map_s_star(np.ones((2, 1)), 1.0, 1.0, np.array([True, True]),
                            np.zeros((2, 1), bool))
        assert np.isnan(out[0])

    def test_eps_must_be_positive(self):
        with pytest.raises(ValueError):
            zp.map_s_star(np.ones((1, 1)), 1.0, 0.0, np.array([True]), None)


class TestProposeZoi:

    def _maps(self, hot, value_hot=30.0, value_cold=5.0, var="V1"):
        """V1 map: |S*delta| = value_hot on `hot` elements, value_cold
        elsewhere (delta = 1), constant over the 5 frames."""
        verts = _grid()
        S = np.full((TIMES.size, verts.shape[0]), value_cold)
        for e in hot:
            S[:, e] = -value_hot                      # sign must not matter
        return verts, {var: {"p": S}}

    def test_rectangle_is_footprint_of_elements_above_one(self):
        hot = [_idx(2, 1), _idx(3, 2)]
        verts, maps = self._maps(hot)
        p = zp.propose_zoi(maps, {"p": 1.0}, EPS, verts, TIMES)
        assert p.bbox == (2.0, 4.0, 1.0, 3.0)
        assert p.n_selected == 2
        assert p.sides_at_extent == []
        assert p.driver[hot[0]] == ("V1", "p")
        assert p.s_star[hot[0]] == pytest.approx(3.0)
        assert p.s_star[0] == pytest.approx(0.5)

    def test_reaching_the_extracted_zone_is_flagged(self):
        verts, maps = self._maps([_idx(0, 1), _idx(2, 3)])
        p = zp.propose_zoi(maps, {"p": 1.0}, EPS, verts, TIMES)
        assert p.bbox == (0.0, 3.0, 1.0, 4.0)
        assert set(p.sides_at_extent) == {"xmin", "ymax"}

    def test_explicit_extent_is_used(self):
        verts, maps = self._maps([_idx(2, 1)])
        p = zp.propose_zoi(maps, {"p": 1.0}, EPS, verts, TIMES,
                           extent=(2.0, 10.0, -5.0, 10.0))
        assert p.sides_at_extent == ["xmin"]

    def test_nothing_above_one_gives_no_rectangle(self):
        verts, maps = self._maps([], value_cold=5.0)
        p = zp.propose_zoi(maps, {"p": 1.0}, EPS, verts, TIMES)
        assert p.bbox is None and p.n_selected == 0

    def test_field_without_eps_is_skipped(self):
        verts, maps = self._maps([_idx(2, 1)], var="V")      # |V|: no eps
        p = zp.propose_zoi(maps, {"p": 1.0}, EPS, verts, TIMES)
        assert p.skipped_fields == ["V"]
        assert p.bbox is None

    def test_frames_outside_window_do_not_count(self):
        verts = _grid()
        S = np.zeros((TIMES.size, verts.shape[0]))
        S[:2, _idx(1, 1)] = 1e6                       # transient only
        p = zp.propose_zoi({"V1": {"p": S}}, {"p": 1.0}, EPS, verts, TIMES)
        assert p.bbox is None
        assert p.n_window_frames == 3

    def test_material_mask_applies_to_velocity_not_to_evf(self):
        verts = _grid()
        n = verts.shape[0]
        e = _idx(4, 2)
        S = np.zeros((TIMES.size, n)); S[:, e] = 50.0
        evf = np.ones((TIMES.size, n)); evf[:, e] = 0.0     # void there
        p = zp.propose_zoi({"V1": {"p": S}}, {"p": 1.0}, EPS, verts, TIMES,
                           evf_base=evf)
        assert p.bbox is None                         # void: velocity ignored
        S_evf = np.zeros((TIMES.size, n)); S_evf[:, e] = 0.5
        p = zp.propose_zoi({"EVF": {"p": S_evf}}, {"p": 1.0}, EPS, verts,
                           TIMES, evf_base=evf)
        assert p.bbox == (4.0, 5.0, 2.0, 3.0)        # EVF itself not masked

    def test_delta_scales_the_change(self):
        verts, maps = self._maps([_idx(2, 1)], value_hot=4.0, value_cold=0.0)
        assert zp.propose_zoi(maps, {"p": 1.0}, EPS, verts,
                              TIMES).bbox is None     # 4 < 10
        assert zp.propose_zoi(maps, {"p": 3.0}, EPS, verts,
                              TIMES).bbox == (2.0, 3.0, 1.0, 2.0)  # 12 > 10

    def test_max_over_maps_and_driver(self):
        verts = _grid()
        n = verts.shape[0]
        a = np.zeros((TIMES.size, n)); a[:, _idx(1, 1)] = 20.0
        b = np.zeros((TIMES.size, n)); b[:, _idx(4, 2)] = 1.0
        p = zp.propose_zoi({"V2": {"p": a}, "EVF": {"q": b}},
                           {"p": 1.0, "q": 1.0}, EPS, verts, TIMES)
        assert p.driver[_idx(1, 1)] == ("V2", "p")
        assert p.driver[_idx(4, 2)] == ("EVF", "q")
        assert p.counts == {("V2", "p"): 1, ("EVF", "q"): 1}
        assert p.bbox == (1.0, 5.0, 1.0, 3.0)

    def test_empty_window_raises(self):
        verts, maps = self._maps([_idx(2, 1)])
        with pytest.raises(ValueError):
            zp.propose_zoi(maps, {"p": 1.0}, EPS, verts, TIMES,
                           window=(0.3, 0.3))

    def test_write_proposal(self, tmp_path):
        verts, maps = self._maps([_idx(2, 1)])
        p = zp.propose_zoi(maps, {"p": 1.0}, EPS, verts, TIMES)
        files = zp.write_proposal(tmp_path, p, verts, eps=EPS,
                                  window=(0.3, 1.0), deltas={"p": 1.0},
                                  extent_kind="ROI box")
        names = sorted(f.name for f in files)
        assert names == ["zoi_proposal.json", "zoi_sstar.npz",
                         "zoi_sstar.png"]
        rec = json.loads((tmp_path / "zoi_proposal.json").read_text("utf-8"))
        assert rec["zoi"] == {"xmin": 2.0, "xmax": 3.0, "ymin": 1.0,
                              "ymax": 2.0}
        assert rec["n_elements_selected"] == 1
        assert rec["elements_selected_by_driver"] == {"V1 @ p": 1}
        with np.load(tmp_path / "zoi_sstar.npz") as z:
            assert z["s_star"].shape == (verts.shape[0],)
            assert "V1__p" in z.files


# ---------------------------------------------------------------------------
# Fc / Ff QoIs per unit width, as the Model tab judges them
# ---------------------------------------------------------------------------
class _HistBundle:
    def __init__(self, t, rf1, rf2, elem_size):
        self.history_time = np.asarray(t, float)
        self._h = {"RF1_RP": np.asarray(rf1, float),
                   "RF2_RP": np.asarray(rf2, float)}
        self.history_info = SimpleNamespace(variables=list(self._h))
        self.model_config = {"mesh": {"elem_size": elem_size}}

    def history(self, var):
        return self._h[var]


class TestForceQois:

    def test_signed_window_mean_per_width(self):
        from gui.results import qoi
        t = [0.0, 0.25, 0.5, 0.75, 1.0]
        b = _HistBundle(t, [-100, -100, -2, -4, -6], [1, 1, 1, 3, 5], 0.002)
        # window t >= 0.3: RF1 mean -4 N, RF2 mean 3 N; / w = 0.002 mm
        assert qoi.qoi_Fc(b, None, 0.3) == pytest.approx(-2000.0)
        assert qoi.qoi_Ff(b, None, 0.3) == pytest.approx(1500.0)
        assert {"Fc", "Ff"} <= set(qoi.available_qoi_ids())

    def test_missing_width_is_nan(self):
        from gui.results import qoi
        b = _HistBundle([0, 1], [1, 1], [1, 1], 0.0)
        assert np.isnan(qoi.qoi_Fc(b, None, 0.0))


# ---------------------------------------------------------------------------
# Whole-domain extraction flag reaches run_simul's run_cfg
# ---------------------------------------------------------------------------
def test_full_domain_flag_in_run_params(tmp_path, monkeypatch):
    from gui.sensitivity import run_worker as rw
    from gui.core.model_config import ModelConfig
    seen = {}

    def fake_args(cmd, script, model_params, run_params):
        seen["run"] = dict(run_params)
        return ["abaqus"]

    def boom(*a, **k):
        raise OSError("no abaqus here")

    monkeypatch.setattr(rw, "build_abaqus_args", fake_args)
    monkeypatch.setattr(rw.subprocess, "Popen", boom)
    for flag in (False, True):
        w = rw.SensitivityRunWorker(None, "jacobian", [], ModelConfig(),
                                    abaqus_cmd="abaqus", abaqus_script="x.py",
                                    workdir=str(tmp_path),
                                    extract_full_domain=flag)
        assert w._abaqus_solve(ModelConfig(), 0) is None
        assert seen["run"].get("extract_full_domain", False) is flag


# ---------------------------------------------------------------------------
# Sensitivity tab: propose, then copy to the Model tab
# ---------------------------------------------------------------------------
def test_tab_proposes_and_copies_zoi(qapp, tmp_path):
    from gui.results.fake_builder import build_fake_results
    from gui.results.reader import ResultsBundle
    from gui.sensitivity import jacobian_plan as jac
    from gui.sensitivity import param_registry as pr
    from gui.sensitivity import runner_core as rc
    from gui.tabs.sensitivity_tab import SensitivityTab
    from gui.core.model_config import ModelConfig

    bundles = []
    for k in range(3):
        _, npz = build_fake_results(tmp_path / ("j%d.results.npz" % k),
                                    n_frames=3, n_grid_x=5, n_grid_y=4)
        bundles.append(ResultsBundle.load(npz))
    spec = pr.spec_for("interaction.friction_coeff")
    plan = jac.build_plan([(spec, 0.3, 0.1, False)], scheme="central")
    order = [None] * plan.n_runs
    order[0] = bundles[0]
    order[plan.idx_plus[0]] = bundles[1]
    order[plan.idx_minus[0]] = bundles[2]

    tab = SensitivityTab(ModelConfig())
    tab._field_checks["TEMP"].setChecked(True)     # fake runs have no V1
    tab.plan = plan
    tab.plan_kind = "jacobian"
    res = rc.RunResult(plan_kind="jacobian", qoi_ids=[],
                       param_paths=list(plan.param_paths),
                       Y=np.zeros((plan.n_runs, 0)), analyses={},
                       failures=[], bundles=order)
    tab._build_field_maps(res)
    assert tab.btn_zoi_propose.isEnabled()

    # Identical fake runs give zero maps: plant one sensitive element.
    path = plan.param_paths[0]
    S = np.zeros_like(np.asarray(tab._field_maps["TEMP"][path], float))
    S[:, 0] = 1e6
    tab._field_maps["TEMP"][path] = S
    tab._map_base_evf = None
    tab._run_workdir = str(tmp_path)

    # No Model-tab settings: explained, nothing proposed.
    tab._on_propose_zoi()
    assert tab._zoi_proposal is None
    tab.set_model_settings_getter(lambda: {"eps": {"Vx": 10.0},
                                           "window": (0.0, 1.0)})
    tab._on_propose_zoi()                          # T has no eps: refused
    assert tab._zoi_proposal is None
    assert "T" in tab.lbl_zoi.text()
    tab.set_model_settings_getter(lambda: {"eps": {"T": 10.0},
                                           "window": (0.0, 1.0)})
    got = []
    tab.zoiProposed.connect(got.append)
    tab._on_propose_zoi()
    prop = tab._zoi_proposal
    assert prop is not None and prop.bbox is not None
    assert prop.n_selected == 1
    assert tab.btn_zoi_apply.isEnabled()
    assert (tmp_path / "sensitivity_maps" / "zoi_proposal.json").exists()
    tab._on_apply_zoi()
    assert got == [tuple(prop.bbox)]
    for b in bundles:
        b.close()


def test_model_tab_set_zoi_and_settings(qapp):
    from gui.tabs.optimization_tab import OptimizationTab
    from gui.core.model_config import ModelConfig
    tab = OptimizationTab(ModelConfig())
    tab.set_zoi((0.1, 0.25, -0.05, 0.02))
    assert tab.zoi() == pytest.approx((0.1, 0.25, -0.05, 0.02))
    st = tab.model_settings()
    assert set(st) == {"eps", "window"}
    assert len(st["window"]) == 2
