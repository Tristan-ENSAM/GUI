# -*- coding: utf-8 -*-
"""
Sensitivity CARTOGRAPHY — the map container, its .npz format and the
viewer panel.

Covers:
  - gui.sensitivity.map_io       (reduction, colour range, save/load)
  - gui.sensitivity.runner_core.build_map_set
  - gui.widgets.sensitivity_map_panel.SensitivityMapPanel / ...Window
  - gui.tabs.sensitivity_tab wiring of the Maps tab

Offline: no Abaqus, and the bundles come from the fake builder. The Qt
tests run under the offscreen platform (see tests/conftest.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.sensitivity import map_io as mio
from gui.sensitivity import runner_core as rc
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import param_registry as pr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mapset(method="jacobian", n_frames=3, n_elem=4):
    """A small, fully populated SensitivityMapSet on a 2x2 quad mesh."""
    nodes = np.array([[0., 0.], [1., 0.], [2., 0.],
                      [0., 1.], [1., 1.], [2., 1.],
                      [0., 2.], [1., 2.], [2., 2.]])
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4],
                      [3, 4, 7, 6], [4, 5, 8, 7]])[:n_elem]
    base = np.arange(n_frames * n_elem, dtype=float).reshape(n_frames, n_elem)
    if method == "morris":
        maps = {"TEMP": {"interaction.friction_coeff": {
            "mu_star": np.abs(base), "sigma": base * 0.1, "mu": -base}}}
    else:
        maps = {"TEMP": {"interaction.friction_coeff": {"dFdtheta": base}}}
    return mio.SensitivityMapSet(
        method=method, maps=maps, field_vars=["TEMP"],
        param_paths=["interaction.friction_coeff"],
        nodes_xy=nodes, faces=faces,
        times=np.linspace(0.0, 1e-3, n_frames),
        param_labels={"interaction.friction_coeff": "Coefficient de "
                                                    "frottement µ"},
        param_units={"interaction.friction_coeff": "—"},
        field_labels={"TEMP": "TEMP"},
        meta={"n_runs": 3})


class _FakeBundle:
    def __init__(self, by_var):
        self._by_var = by_var

    def field(self, inst, var):
        return self._by_var[var]


# ---------------------------------------------------------------------------
# Display reduction
# ---------------------------------------------------------------------------
class TestReduceMap:

    def test_frame_slice_signed_and_abs(self):
        S = np.array([[1.0, -2.0], [3.0, -4.0]])
        v, txt = mio.reduce_map(S, mode="signed", frame=1)
        assert np.allclose(v, [3.0, -4.0])
        assert txt == "frame 1/1"
        v, _ = mio.reduce_map(S, mode="abs", frame=1)
        assert np.allclose(v, [3.0, 4.0])

    def test_frame_index_is_clamped(self):
        S = np.array([[1.0, 2.0], [3.0, 4.0]])
        v, txt = mio.reduce_map(S, frame=99)
        assert np.allclose(v, [3.0, 4.0]) and txt == "frame 1/1"
        v, txt = mio.reduce_map(S, frame=-5)
        assert np.allclose(v, [1.0, 2.0]) and txt == "frame 0/1"

    def test_aggregate_mean_keeps_sign_rms_does_not(self):
        S = np.array([[1.0, 2.0], [-1.0, 2.0]])
        v, txt = mio.reduce_map(S, mode="signed", aggregate=True)
        assert np.allclose(v, [0.0, 2.0])          # cancellation is meaningful
        assert "mean over 2 frames" == txt
        v, txt = mio.reduce_map(S, mode="abs", aggregate=True)
        assert np.allclose(v, [1.0, 2.0])          # RMS: no cancellation
        assert "RMS over 2 frames" == txt

    def test_nan_frames_are_ignored(self):
        S = np.array([[np.nan, 2.0], [4.0, 2.0]])
        v, _ = mio.reduce_map(S, aggregate=True)
        assert np.allclose(v, [4.0, 2.0])

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            mio.reduce_map(np.zeros((0, 3)))
        with pytest.raises(ValueError):
            mio.reduce_map(np.array([1.0, 2.0]))


class TestColorRange:

    def test_diverging_is_symmetric(self):
        vmin, vmax, cmap = mio.color_range(np.array([-1.0, 3.0]), True)
        assert (vmin, vmax) == (-3.0, 3.0) and cmap == "RdBu_r"

    def test_sequential_starts_at_zero(self):
        vmin, vmax, cmap = mio.color_range(np.array([1.0, 3.0]), False)
        assert vmin == 0.0 and vmax == 3.0 and cmap == "inferno"

    def test_all_nan_falls_back_to_a_unit_range(self):
        vmin, vmax, _ = mio.color_range(np.array([np.nan, np.nan]), True)
        assert vmin < vmax
        vmin, vmax, _ = mio.color_range(np.array([np.nan]), False)
        assert vmin < vmax

    def test_constant_field_is_not_degenerate(self):
        vmin, vmax, _ = mio.color_range(np.zeros(4), True)
        assert vmin < vmax
        vmin, vmax, _ = mio.color_range(np.zeros(4), False)
        assert vmin < vmax


class TestDisplayEntries:

    def test_jacobian_offers_signed_and_magnitude(self):
        e = mio.display_entries(mio.QUANTITIES_BY_METHOD["jacobian"])
        assert [(q, m) for q, m, _l, _d in e] == [("dFdtheta", "signed"),
                                                  ("dFdtheta", "abs")]
        assert e[0][3] is True and e[1][3] is False      # diverging flag

    def test_morris_offers_the_three_indices(self):
        e = mio.display_entries(mio.QUANTITIES_BY_METHOD["morris"])
        assert [q for q, _m, _l, _d in e][:3] == ["mu_star", "sigma", "mu"]

    def test_unknown_quantity_is_skipped(self):
        assert mio.display_entries(["nope"]) == []


# ---------------------------------------------------------------------------
# The container and its .npz format
# ---------------------------------------------------------------------------
class TestMapSet:

    def test_quantities_follow_the_method(self):
        assert _mapset("jacobian").quantities == ("dFdtheta",)
        assert _mapset("morris").quantities == ("mu_star", "sigma", "mu")

    def test_shape_helpers(self):
        ms = _mapset(n_frames=5)
        assert ms.n_frames == 5 and ms.n_elements == 4
        assert ms.is_drawable() is True
        assert ms.get("TEMP", "interaction.friction_coeff",
                      "dFdtheta").shape == (5, 4)
        assert ms.get("TEMP", "nope", "dFdtheta") is None

    def test_empty_set_is_not_drawable(self):
        ms = mio.SensitivityMapSet(method="jacobian", maps={}, field_vars=[],
                                   param_paths=[], nodes_xy=np.zeros((0, 2)),
                                   faces=np.zeros((0, 4), dtype=int))
        assert ms.is_drawable() is False
        assert ms.n_frames == 0 and ms.n_elements == 0
        assert "No sensitivity map" in mio.describe(ms)
        assert "No sensitivity map" in mio.describe(None)


class TestNpzRoundTrip:

    @pytest.mark.parametrize("method", ["jacobian", "morris"])
    def test_round_trip_preserves_everything_used_for_drawing(self, method,
                                                              tmp_path):
        ms = _mapset(method)
        out = mio.save_npz(ms, tmp_path / "study.npz")
        back = mio.load_npz(out)
        assert back.method == method
        assert back.field_vars == ms.field_vars
        assert back.param_paths == ms.param_paths
        assert back.quantities == ms.quantities
        assert back.param_label("interaction.friction_coeff") == \
            ms.param_label("interaction.friction_coeff")
        assert back.param_units == ms.param_units
        assert np.allclose(back.nodes_xy, ms.nodes_xy)
        assert np.array_equal(back.faces, ms.faces)
        assert np.allclose(back.times, ms.times)
        assert back.meta["n_runs"] == 3
        for q in ms.quantities:
            a = ms.get("TEMP", "interaction.friction_coeff", q)
            b = back.get("TEMP", "interaction.friction_coeff", q)
            assert np.allclose(a, b, atol=1e-5)      # stored as float32

    def test_extension_is_forced(self, tmp_path):
        out = mio.save_npz(_mapset(), tmp_path / "study")
        assert out.endswith(".npz")

    def test_loading_a_foreign_npz_is_a_clean_error(self, tmp_path):
        p = tmp_path / "other.npz"
        np.savez(p, something=np.zeros(3))
        with pytest.raises(mio.MapLoadError):
            mio.load_npz(p)

    def test_loading_a_missing_file_is_a_clean_error(self, tmp_path):
        with pytest.raises(mio.MapLoadError):
            mio.load_npz(tmp_path / "nope.npz")

    def test_newer_format_is_refused(self, tmp_path):
        ms = _mapset()
        out = mio.save_npz(ms, tmp_path / "study.npz")
        with np.load(out) as z:
            payload = {k: z[k] for k in z.files}
        payload["format_version"] = np.array(mio.FORMAT_VERSION + 1,
                                             dtype=np.int32)
        np.savez_compressed(out, **payload)
        with pytest.raises(mio.MapLoadError):
            mio.load_npz(out)


# ---------------------------------------------------------------------------
# Building a map set from run bundles
# ---------------------------------------------------------------------------
class TestBuildMapSet:

    def _jac_bundles(self, tmp_path, n_frames=3):
        from gui.results.fake_builder import build_fake_results
        from gui.results.reader import ResultsBundle
        plan = jac.build_plan(
            [(pr.spec_for("interaction.friction_coeff"), 0.3, 0.1, False)],
            scheme="central")
        order = [None] * plan.n_runs
        keep = []
        for k, idx in enumerate([0, plan.idx_plus[0], plan.idx_minus[0]]):
            _, npz = build_fake_results(tmp_path / ("j%d.results.npz" % k),
                                        n_frames=n_frames, n_grid_x=5,
                                        n_grid_y=4)
            b = ResultsBundle.load(npz)
            order[idx] = b
            keep.append(b)
        return plan, order, keep

    def test_jacobian_map_set_is_drawable(self, tmp_path):
        plan, order, keep = self._jac_bundles(tmp_path)
        ms = rc.build_map_set(plan, "jacobian", order, ["EVF"],
                              param_labels={"interaction.friction_coeff": "mu"},
                              meta={"scheme": "central"})
        assert ms is not None and ms.is_drawable()
        assert ms.method == "jacobian" and ms.quantities == ("dFdtheta",)
        assert ms.n_frames == 3
        assert ms.n_elements == ms.get("EVF", "interaction.friction_coeff",
                                       "dFdtheta").shape[1]
        assert ms.times.size == 3
        assert ms.meta["scheme"] == "central"
        for b in keep:
            b.close()

    def test_no_field_or_no_bundle_yields_none(self, tmp_path):
        plan, order, keep = self._jac_bundles(tmp_path)
        assert rc.build_map_set(plan, "jacobian", order, []) is None
        assert rc.build_map_set(plan, "jacobian", [], ["EVF"]) is None
        assert rc.build_map_set(plan, "jacobian", [None, None], ["EVF"]) is None
        for b in keep:
            b.close()

    def test_unreadable_field_yields_none(self, tmp_path):
        plan, order, keep = self._jac_bundles(tmp_path)
        assert rc.build_map_set(plan, "jacobian", order, ["NOPE"]) is None
        for b in keep:
            b.close()

    def test_mesh_from_bundle_matches_the_element_count(self, tmp_path):
        from gui.results.fake_builder import build_fake_results
        from gui.results.reader import ResultsBundle
        _, npz = build_fake_results(tmp_path / "m.results.npz", n_frames=2,
                                    n_grid_x=5, n_grid_y=4)
        b = ResultsBundle.load(npz)
        inst = rc.eulerian_instance(b)
        nodes_xy, faces = mio.mesh_from_bundle(b, inst)
        assert nodes_xy.shape[1] == 2
        assert faces.shape[0] == b.instance(inst).n_elements
        b.close()


# ---------------------------------------------------------------------------
# The viewer panel (headless Qt)
# ---------------------------------------------------------------------------
class TestMapPanel:

    def _panel(self, qapp, method="jacobian"):
        from gui.widgets.sensitivity_map_panel import SensitivityMapPanel
        panel = SensitivityMapPanel()
        panel.set_map_set(_mapset(method))
        return panel

    def test_selectors_are_populated(self, qapp):
        panel = self._panel(qapp)
        assert panel.cb_field.count() == 1
        assert panel.cb_field.currentData() == "TEMP"
        assert panel.cb_param.count() == 1
        assert panel.cb_param.currentData() == "interaction.friction_coeff"
        assert panel.cb_quantity.count() == 2          # signed + magnitude
        assert panel.sld_frame.maximum() == 2
        assert panel.btn_save.isEnabled() and panel.btn_detach.isEnabled()

    def test_morris_selector_lists_the_three_indices(self, qapp):
        panel = self._panel(qapp, "morris")
        labels = [panel.cb_quantity.itemText(i)
                  for i in range(panel.cb_quantity.count())]
        assert any(l.startswith("μ*") for l in labels)
        assert any(l.startswith("σ") for l in labels)

    @pytest.mark.parametrize("method", ["jacobian", "morris"])
    def test_every_control_combination_renders(self, qapp, method):
        panel = self._panel(qapp, method)
        for qi in range(panel.cb_quantity.count()):
            panel.cb_quantity.setCurrentIndex(qi)
            for agg in (True, False):
                panel.chk_aggregate.setChecked(agg)
                for f in range(panel.sld_frame.maximum() + 1):
                    panel.sld_frame.setValue(f)
                    panel.refresh()

    def test_frame_slider_follows_the_aggregate_toggle(self, qapp):
        panel = self._panel(qapp)
        panel.chk_aggregate.setChecked(True)
        assert panel.sld_frame.isEnabled() is False
        assert panel.lbl_frame.text() == "agg"
        panel.chk_aggregate.setChecked(False)
        assert panel.sld_frame.isEnabled() is True
        panel.sld_frame.setValue(1)
        assert panel.lbl_frame.text().startswith("1")
        assert "t=" in panel.lbl_frame.text()          # frame time is shown

    def test_clearing_says_why(self, qapp):
        panel = self._panel(qapp)
        panel.set_map_set(None, hint="nothing to map")
        assert panel.lbl_hint.text() == "nothing to map"
        assert panel.cb_field.count() == 0
        assert panel.btn_save.isEnabled() is False
        panel.refresh()                                # must not raise

    def test_save_then_load_through_the_panel(self, qapp, tmp_path):
        panel = self._panel(qapp, "morris")
        out = mio.save_npz(panel.map_set, tmp_path / "s.npz")
        panel.set_map_set(None)
        assert panel.load_file(out) is True
        assert panel.map_set.method == "morris"
        assert panel.cb_quantity.count() >= 3
        assert "Loaded" in panel.lbl_hint.text()

    def test_loading_a_bad_file_keeps_the_current_study(self, qapp, tmp_path):
        panel = self._panel(qapp)
        bad = tmp_path / "bad.npz"
        np.savez(bad, x=np.zeros(2))
        assert panel.load_file(bad) is False
        assert panel.map_set is not None               # unchanged
        assert panel.cb_field.count() == 1

    def test_detached_window_shows_the_same_study(self, qapp):
        panel = self._panel(qapp)
        win = panel._on_detach()
        assert win is not None
        assert win.panel.map_set is panel.map_set
        assert win.panel.btn_detach.isVisible() is False
        win.close()


# ---------------------------------------------------------------------------
# Sensitivity tab wiring
# ---------------------------------------------------------------------------
class TestSensitivityTabMaps:

    def _tab(self):
        from gui.tabs.sensitivity_tab import SensitivityTab
        from gui.core.model_config import ModelConfig
        return SensitivityTab(ModelConfig())

    def test_run_result_populates_the_panel(self, qapp, tmp_path):
        from gui.results.fake_builder import build_fake_results
        from gui.results.reader import ResultsBundle
        tab = self._tab()
        plan = jac.build_plan(
            [(pr.spec_for("interaction.friction_coeff"), 0.3, 0.1, False)],
            scheme="central")
        order = [None] * plan.n_runs
        keep = []
        for k, idx in enumerate([0, plan.idx_plus[0], plan.idx_minus[0]]):
            _, npz = build_fake_results(tmp_path / ("t%d.results.npz" % k),
                                        n_frames=3, n_grid_x=5, n_grid_y=4)
            b = ResultsBundle.load(npz)
            order[idx] = b
            keep.append(b)
        tab.plan = plan
        tab.plan_kind = "jacobian"
        tab._field_checks["EVF"].setChecked(True)
        res = rc.RunResult(plan_kind="jacobian", qoi_ids=[],
                           param_paths=list(plan.param_paths),
                           Y=np.zeros((plan.n_runs, 0)), analyses={},
                           failures=[], bundles=order)
        tab._build_field_maps(res)
        ms = tab.map_panel.map_set
        assert ms is not None and ms.method == "jacobian"
        assert tab.map_panel.cb_param.count() == 1
        assert tab.map_panel.cb_field.count() == 1
        assert ms.meta["scheme"] == "central"
        assert ms.param_units["interaction.friction_coeff"]   # unit recorded
        for b in keep:
            b.close()

    def test_no_field_ticked_clears_with_a_hint(self, qapp):
        tab = self._tab()
        res = rc.RunResult(plan_kind="jacobian", qoi_ids=[], param_paths=[],
                           Y=np.zeros((1, 0)), analyses={}, failures=[],
                           bundles=None)
        tab._build_field_maps(res)
        assert tab.map_panel.map_set is None
        assert tab._map_set is None
        assert "ROI field" in tab.map_panel.lbl_hint.text()

    def test_morris_with_a_field_is_accepted_by_the_planner(self, qapp):
        """Ticking a ROI field no longer disables itself under Morris: the
        field is not a scalar QoI there, it is what gets mapped."""
        pytest.importorskip("SALib")
        tab = self._tab()
        tab.cb_method.setCurrentIndex(1)               # Morris
        tab._field_checks["EVF"].setChecked(True)
        for r in range(tab.table.rowCount()):
            spec = tab._row_spec.get(r)
            if spec is not None and spec.path == "interaction.friction_coeff":
                tab.table.item(r, 0).setCheckState(
                    __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Checked)
                break
        tab.spin_traj.setValue(2)
        tab._on_generate()
        assert tab.plan is not None and tab.plan_kind == "morris"
        assert tab._selected_field_vars() == ["EVF"]
        assert "mapped per element" in tab.status.text()
