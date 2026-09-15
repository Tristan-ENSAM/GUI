# -*- coding: utf-8 -*-
"""Results tab, multi-run: overlay of several runs' history (stable colour per
run, prefixed legend), closing a run, and preserving the frame / instance /
field / colormap when switching runs. Plus the log energy axis of the
TimeSeriesViewer.
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.results.fake_builder import build_fake_results
from gui.tabs.results_tab import ResultsTab
from gui.widgets.time_series_viewer import TimeSeriesViewer


def _make_run(tmp_path, name, n_frames=20):
    json_path, _npz = build_fake_results(
        tmp_path / ("%s.results.json" % name), job_name=name,
        n_frames=n_frames)
    return str(json_path)


@pytest.fixture
def tab(qapp):
    return ResultsTab()


class TestMultiRunOverlay:
    def test_two_runs_overlay_with_prefixed_labels(self, tab, tmp_path):
        tab.load_bundle(_make_run(tmp_path, "runA"))
        tab.load_bundle(_make_run(tmp_path, "runB"))
        assert set(tab._runs) == {"runA", "runB"}
        labels = list(tab.ts_viewer._lines.keys())
        # every loaded run contributes its history, prefixed by run name
        assert any(L.startswith("runA\u00b7") for L in labels)
        assert any(L.startswith("runB\u00b7") for L in labels)

    def test_colour_is_stable_and_distinct_per_run(self, tab, tmp_path):
        tab.load_bundle(_make_run(tmp_path, "runA"))
        tab.load_bundle(_make_run(tmp_path, "runB"))
        cA, cB = tab._run_color("runA"), tab._run_color("runB")
        assert cA != cB
        assert tab._run_color("runA") == cA          # stable on re-query

    def test_single_run_labels_are_not_prefixed(self, tab, tmp_path):
        tab.load_bundle(_make_run(tmp_path, "solo"))
        labels = list(tab.ts_viewer._lines.keys())
        assert labels and all("\u00b7" not in L for L in labels)


class TestCloseRun:
    def test_close_removes_active_run(self, tab, tmp_path):
        tab.load_bundle(_make_run(tmp_path, "runA"))
        tab.load_bundle(_make_run(tmp_path, "runB"))   # runB active
        tab._on_close_clicked()
        assert set(tab._runs) == {"runA"}
        assert tab._active_run == "runA"

    def test_close_last_run_clears(self, tab, tmp_path):
        tab.load_bundle(_make_run(tmp_path, "solo"))
        tab._on_close_clicked()
        assert tab._runs == {}
        assert tab._active_run is None
        assert not tab.ts_viewer._lines


class TestPreserveOnRunChange:
    def test_frame_index_is_preserved(self, tab, tmp_path):
        tab.load_bundle(_make_run(tmp_path, "runA", n_frames=20))
        tab.load_bundle(_make_run(tmp_path, "runB", n_frames=20))
        tab.slider.setValue(7)                 # -> _frame_idx = 7 on runB
        assert tab._frame_idx == 7
        tab.cb_run.setCurrentText("runA")      # triggers _on_run_changed
        assert tab._frame_idx == 7             # kept, not reset to 0

    def test_field_selection_is_preserved(self, tab, tmp_path):
        tab.load_bundle(_make_run(tmp_path, "runA"))
        tab.load_bundle(_make_run(tmp_path, "runB"))
        # pick a field present in the fake bundle
        fields = [tab.cb_field.itemText(i) for i in range(tab.cb_field.count())]
        target = "TEMP" if "TEMP" in fields else fields[-1]
        tab.cb_field.setCurrentText(target)
        tab.cb_run.setCurrentText("runA")
        assert tab.cb_field.currentText() == target

    def test_colormap_is_preserved(self, tab, tmp_path):
        tab.load_bundle(_make_run(tmp_path, "runA"))
        tab.load_bundle(_make_run(tmp_path, "runB"))
        cmaps = [tab.cb_cmap.itemText(i) for i in range(tab.cb_cmap.count())]
        target = cmaps[-1]
        tab.cb_cmap.setCurrentText(target)
        tab.cb_run.setCurrentText("runA")
        assert tab.cb_cmap.currentText() == target


class TestEnergyAxis:
    def test_energy_axis_is_log(self, qapp):
        v = TimeSeriesViewer()
        assert v._ax_energy.get_yscale() == "log"

    def test_energy_flag_routes_to_energy_panel(self, qapp):
        v = TimeSeriesViewer()
        t, y = np.array([0.0, 1.0]), np.array([1.0, 2.0])
        v.add_series("runA\u00b7ALLKE", t, y, energy=True)   # prefixed name
        assert v._lines["runA\u00b7ALLKE"].axes is v._ax_energy
        v.add_series("runA\u00b7RF1_RP", t, y, energy=False)
        assert v._lines["runA\u00b7RF1_RP"].axes is v._ax
