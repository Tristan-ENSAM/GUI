# -*- coding: utf-8 -*-
"""Job and Results (Tristan 2026-11-16): CPUs limited to the machine,
working-directory button, Fc/Ff in N/mm, energy meanings and ratios, value
under the cursor, initial-run checks as a page of Results."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest


def test_cpu_count_of_machines(tmp_path):
    from gui.core import machines as m
    from gui.core.preferences import Preferences
    q = tmp_path / "q"
    q.mkdir()
    prefs = Preferences()
    prefs.machines = [{"name": "WTS", "kind": "queue",
                       "queue_dir": str(q), "abaqus_cmd": ""}]
    assert m.cpu_count(prefs, m.LOCAL_NAME) == os.cpu_count()
    assert m.cpu_count(prefs, "WTS") is None            # no agent yet
    (q / "agent.json").write_text(json.dumps({"host": "X", "cpus": 40}))
    assert m.cpu_count(prefs, "WTS") == 40
    assert m.cpu_count(prefs, "nope") is None


def test_job_cpus_limited_and_workdir_button(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    jt = w.job_tab
    assert jt.spin_cpus.maximum() == os.cpu_count()
    assert jt.btn_workdir.text() == "Open working dir"


class _Hist:
    variables = ["RF1_RP", "RF2_RP", "ALLKE", "ALLIE", "ALLAE",
                 "ENERGY_TIME", "ALLAE_TIME"]


class _Bundle:
    """Histories only (no field): RF in N, energies on their own time."""
    instance_names = []
    history_info = _Hist()

    def __init__(self, h=0.005):
        self.history_time = np.linspace(0.0, 1e-4, 11)
        self.model_config = {"mesh": {"elem_size": h}} if h else {}
        te = np.linspace(0.0, 1e-4, 6)
        self._h = {"RF1_RP": -np.full(11, 0.5), "RF2_RP": -np.full(11, 0.2),
                   "ALLKE": np.full(6, 1.0), "ALLIE": np.full(6, 10.0),
                   "ALLAE": np.full(6, 0.5), "ENERGY_TIME": te,
                   "ALLAE_TIME": te}

    def history(self, v):
        return self._h[v]

    def close(self):
        pass


def test_forces_per_width_and_energy_note(qapp):
    from gui.tabs.results_tab import ResultsTab, energy_ratio_text
    t = ResultsTab(window_getter=lambda: (0.3, 1.0))
    t._runs["r"] = _Bundle()
    t._active_run = "r"
    t._refresh_time_series()
    lines = t.ts_viewer._lines
    fc = lines["Fc = −RF1_RP / h"]
    assert np.allclose(fc.get_ydata(), 100.0)            # 0.5 N / 0.005 mm
    assert np.allclose(lines["Ff = −RF2_RP / h"].get_ydata(), 40.0)
    assert "ALLKE: kinetic energy" in lines
    assert "ALLAE: artificial strain energy (hourglass)" in lines
    note = t.ts_viewer._energy_note.get_text()
    assert "R_K" in note and "10 %" in note and "R_HG" in note and "5 %" in note
    assert "over T = [0.3, 1]" in energy_ratio_text(_Bundle(), (0.3, 1.0))


def test_forces_without_element_size_stay_in_newton(qapp):
    from gui.tabs.results_tab import ResultsTab
    t = ResultsTab()
    t._runs["r"] = _Bundle(h=None)
    t._active_run = "r"
    t._refresh_time_series()
    assert "RF1_RP [N]" in t.ts_viewer._lines


def test_field_value_under_the_cursor(qapp):
    from gui.widgets.field_viewer import FieldViewer
    fv = FieldViewer()
    nodes = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    elems = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    fv.set_mesh(nodes, elems)
    fv.set_values(np.array([3.0, 7.0]), title="TEMP   frame 0/1")
    assert fv.value_at(0.5, 0.5) == 3.0
    assert fv.value_at(1.5, 0.5) == 7.0
    assert fv.value_at(5.0, 0.5) is None
    assert "TEMP = 7" in fv.format_coord(1.5, 0.5)


def test_initial_run_checks_are_a_results_page(qapp):
    from gui.main import MainWindow
    w = MainWindow(); w._dirty = False
    names = [w.results_tab.pages.tabText(i)
             for i in range(w.results_tab.pages.count())]
    assert names == ["Fields and histories", "Initial run checks"]
    opt = [w.opt_tabs.tabText(i) for i in range(w.opt_tabs.count())]
    assert not any("Initial run" in n for n in opt)
