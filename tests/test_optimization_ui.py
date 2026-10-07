# -*- coding: utf-8 -*-
"""The restructured Optimization tab: ZOI panel + the two study launchers
(mesh GCI, domain independence). Guards must fire before any Abaqus launch.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from gui.core.model_config import ModelConfig
from gui.tabs.optimization_tab import OptimizationTab


@pytest.fixture
def tab(qapp, monkeypatch):
    t = OptimizationTab(ModelConfig())
    # Neutralise the Preferences/Abaqus path check; the input guards run first.
    monkeypatch.setattr(t, "_validate_launch", lambda: ("prefs", "wd", 1))
    return t


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: seen.append(a[2])))
    return seen


class TestRestructuredTab:
    def test_constructs_with_new_panels(self, tab):
        assert tab.btn_mesh is not None and tab.btn_domain is not None
        assert set(tab.le_zoi) == {"xmin", "xmax", "ymin", "ymax"}
        assert tab.le_gci_finest is not None
        assert tab.sp_gci_n.value() >= 3
        assert set(tab._dj_eps) == {"EVF", "TEMP", "V1", "V2", "force"}

    def test_zoi_defaults_to_roi_when_blank(self, tab):
        roi = tab.config_inputs()["roi"]
        assert tab.zoi() == pytest.approx(roi)

    def test_set_zoi_from_roi_fills_fields(self, tab):
        tab._zoi_from_roi()
        assert tab.le_zoi["xmin"].text() != ""
        assert tab.zoi() == pytest.approx(tab.config_inputs()["roi"])

    def test_domain_study_requires_the_six_absolute_tolerances(self, tab,
                                                               warnings):
        for le in tab._q_eps.values():
            le.setText("")
        tab._q_eps["Vx"].setText("1")
        tab._on_run_domain_independence()
        assert warnings and "tolerances" in warnings[0].lower()
        assert not hasattr(tab, "_di_worker")

    def test_tolerances_default_to_2pct(self, tab):
        assert tab._tolerances() == {q: 0.02 for q in
                                     ("EVF", "TEMP", "V1", "V2", "force")}

    def test_spin_boxes_follow_the_style_size_hint(self, tab, qapp):
        """The spin boxes must be as wide as the style asks (value + arrow
        buttons). A fixed 64 px hid the value in the Windows 11 style, whose
        arrows sit side by side; a larger font reproduces it here."""
        spins = [tab.sp_margin, tab.sp_gci_n, *tab._dom_spins.values()]
        for sp in spins:
            sp.setStyleSheet("font-size: 40px;")
        tab.resize(2400, 1400)
        tab.show()
        qapp.processEvents()
        try:
            for sp in spins:
                assert sp.maximumWidth() > sp.minimumWidth()   # not fixed
                assert sp.width() >= sp.sizeHint().width()
        finally:
            tab.hide()

    def test_busy_toggles_the_study_buttons(self, tab):
        tab._busy(True, "running")
        assert not tab.btn_mesh.isEnabled()
        assert not tab.btn_domain.isEnabled()
        assert tab.btn_cancel.isEnabled()
        tab._busy(False)
        assert tab.btn_mesh.isEnabled()
        assert not tab.btn_cancel.isEnabled()


class TestRunOutputWiring:
    def test_profile_name_from_getter(self, qapp):
        tab = OptimizationTab(ModelConfig(), profile_name_getter=lambda: "myprof")
        assert tab._profile_name() == "myprof"

    def test_profile_name_defaults_untitled(self, qapp):
        tab = OptimizationTab(ModelConfig())
        assert tab._profile_name() == "Untitled"

    def test_study_run_dir_creates_folder_and_config(self, qapp, tmp_path):
        from pathlib import Path
        tab = OptimizationTab(ModelConfig(), profile_name_getter=lambda: "prof")
        d = tab._study_run_dir(tmp_path, "GCI", {"finest": 0.006})
        assert Path(d).parent == Path(tmp_path)
        assert Path(d).name.startswith("prof_GCI_")
        assert (Path(d) / "config.json").exists()


# ---------------------------------------------------------------------------
# Lot L3: domain independence study wiring
# ---------------------------------------------------------------------------
def _fill_thresholds(tab):
    for q, v in (("Vx", "10"), ("Vy", "10"), ("T", "1"), ("EVF", "0.01"),
                 ("Fc", "0.5"), ("Ff", "0.5")):
        tab._q_eps[q].setText(v)


class _CaptureWorker:
    """Stands in for a QThread worker: records its kwargs, never runs."""
    last = None

    def __init__(self, parent=None, **kw):
        type(self).last = kw
        from unittest.mock import MagicMock
        self.progress = MagicMock()
        self.finished_ok = MagicMock()
        self.failed = MagicMock()

    def start(self):
        pass

    def isRunning(self):
        return False


@pytest.fixture
def launch(tab, monkeypatch, tmp_path):
    import gui.tabs.optimization_tab as ot
    monkeypatch.setattr(ot, "DomainIndependenceWorker", _CaptureWorker)
    monkeypatch.setattr(ot, "MeshGciWorker", _CaptureWorker)
    monkeypatch.setattr(tab, "_validate_launch",
                        lambda: (type("P", (), {"abaqus_cmd": "a",
                                                "abaqus_script": "s"})(),
                                 tmp_path, 4))
    monkeypatch.setattr(tab, "_start_progress", lambda: None)
    _CaptureWorker.last = None
    return tab


class TestDomainIndependenceWiring:
    def test_defaults_match_the_decisions(self, tab):
        assert tab.domain_settings() == {"dom_step_elems": 10, "dom_n_max": 8,
                                         "dom_n_hold": 1, "dom_m_ratios": 2}
        assert tab.window() == (0.3, 1.0)
        g = tab.guard_settings()
        assert (g.rk_max, g.rhg_max) == (0.01, 0.05)

    def test_window_validation(self, tab):
        tab._dom_texts["window_start"].setText("0.8")
        tab._dom_texts["window_end"].setText("0.5")
        with pytest.raises(ValueError):
            tab.window()

    def test_n_max_must_allow_the_decay_test(self, tab):
        tab._dom_spins["dom_m_ratios"].setValue(3)
        tab._dom_spins["dom_n_max"].setValue(3)
        with pytest.raises(ValueError):
            tab.domain_settings()

    def test_initial_domain_is_zoi_plus_margin_with_offset(self, tab):
        tab.cfg.elem_size = 0.01
        tab.cfg.euler_position.x0 = 1.0
        for k, v in (("xmin", "0.9"), ("xmax", "1.05"), ("ymin", "-0.08"),
                     ("ymax", "0.02")):
            tab.le_zoi[k].setText(v)
        tab.sp_margin.setValue(1)
        d = tab.compute_initial_dims()
        assert d.l_wp == pytest.approx(0.11)
        assert d.l_void == pytest.approx(0.06)
        assert d.h_wp == pytest.approx(0.09)
        assert d.h_void == pytest.approx(0.03)

    def test_launch_passes_the_study_settings(self, launch):
        tab = launch
        _fill_thresholds(tab)
        tab._dom_spins["dom_step_elems"].setValue(12)
        tab._dom_texts["rhg_max"].setText("0.07")
        tab._max["h_void"].setText("0.4")
        elem_before = tab.cfg.elem_size
        tab._on_run_domain_independence()
        kw = _CaptureWorker.last
        assert kw is not None
        assert kw["step_elems"] == 12 and kw["n_max"] == 8
        assert kw["n_hold"] == 1 and kw["m_ratios"] == 2
        assert kw["window"] == (0.3, 1.0)
        assert kw["thresholds"]["Ff"] == pytest.approx(0.5)
        assert kw["caps"] == {"h_void": 0.4}
        assert kw["initial_dims"] == tab.compute_initial_dims()
        # a deep copy with the needed history outputs forced on
        assert kw["base_cfg"] is not tab.cfg
        assert kw["base_cfg"].step.output.ho_preselect is True
        assert kw["base_cfg"].step.output.ho_rf_on_rp is True
        assert tab.cfg.elem_size == elem_before
        # safeguards built from the panel
        assert callable(kw["guard_fn"]) and callable(kw["cost_fn"])

    def test_gci_gets_a_copy_and_the_shared_window(self, launch):
        tab = launch
        tab._dom_texts["window_start"].setText("0.5")
        tab._on_run_mesh_gci()
        kw = _CaptureWorker.last
        assert kw["window"] == (0.5, 1.0)
        assert kw["base_cfg"] is not tab.cfg

    def test_progress_and_result_are_logged(self, tab):
        from gui.core.domain_sizing import DomainDims
        from gui.sensitivity.domain_independence import (
            Comparison, DimensionResult, RunRecord, StudyResult)
        rec = RunRecord(index=0, dims={"h_wp": .1, "h_void": .1, "l_wp": .1,
                                       "l_void": .1}, job_ok=True,
                        guards={"R_K": (0.001, True), "R_HG": (None, False)},
                        diagonal_ratio=120.0, diagonal_warning=True)
        tab._on_di_progress({"phase": "run", "record": rec})
        comp = Comparison("l_wp", 1, 0.1, 0.2, {"Vx": 1.0}, 0.5, "Vx", True,
                          True, mode="tail_bound")
        tab._on_di_progress({"phase": "comparison", "comparison": comp})
        d = DomainDims(.1, .1, .2, .1)
        dr = DimensionResult("l_wp", 0.1, 0.2, "tail_bound", q_crit="Vx",
                             criterion=0.4)
        tab._on_di_progress({"phase": "dimension", "result": dr})
        tab._on_di_done(StudyResult(initial=d, final=d,
                                    per_dimension={"l_wp": dr},
                                    runs=[rec], status="partial",
                                    warnings=["diag"]))
        text = tab.log.toPlainText()
        assert "R_HG=n/a FAIL" in text and "diag/h=120.0" in text
        assert "E_max=0.5 (Vx)" in text and "retained 0.2" in text
        assert "NOT converged" in text
        assert tab._last_domain_result is not None


class _GridBundle:
    """Analytic ResultsBundle over the real Eulerian grid of `dims` (element
    size h, origin at the cutting corner): every ZOI quantity carries a
    boundary influence decaying with each dimension, energies are steady."""

    def __init__(self, dims, h, lam=0.05, nt=11):
        self.dims, self.h, self.lam = dims, h, lam
        xs = np.arange(-dims.l_wp + h / 2, dims.l_void, h)
        ys = np.arange(-dims.h_wp + h / 2, dims.h_void, h)
        X, Y = np.meshgrid(xs, ys)
        self._c = np.column_stack([X.ravel(), Y.ravel(),
                                   np.full(X.size, h / 2)])
        self._mat = self._c[:, 1] < 0.0          # material below y = 0
        self.times = np.linspace(0.0, 1e-4, nt)
        self.history_time = np.linspace(0.0, 1e-4, 21)
        self.instance_names = ["EULER"]

    def instance(self, name):
        class _I:
            field_variables = ["EVF", "TEMP", "V1", "V2"]
            n_elements = len(self._c)
        return _I()

    def element_centroids_init(self, inst):
        return self._c

    def _infl(self):
        return sum(np.exp(-getattr(self.dims, n) / self.lam)
                   for n in ("h_wp", "h_void", "l_wp", "l_void"))

    def field(self, inst, var):
        nt, ne = len(self.times), len(self._c)
        if var == "EVF":
            return np.tile(self._mat.astype(float), (nt, 1))
        base = {"TEMP": 300.0, "V1": 1000.0, "V2": -200.0}[var]
        return np.full((nt, ne), base * (1.0 + self._infl()))

    def history(self, var):
        n = len(self.history_time)
        vals = {"RF1_RP": 0.8 * self.h * (1 + self._infl()),
                "RF2_RP": 0.3 * self.h * (1 + self._infl()),
                "ALLKE": 0.001, "ALLIE": 1.0, "ALLAE": 0.01}
        if var in ("ENERGY_TIME", "ALLAE_TIME"):
            return self.history_time
        return np.full(n, vals[var])


def test_end_to_end_through_the_tab(qapp, monkeypatch, tmp_path):
    """The tab launches the real worker with the real safeguards and cost on
    an analytic bundle; the study completes and the result is logged."""
    import gui.tabs.optimization_tab as ot
    from gui.core.domain_sizing import DomainDims
    tab = OptimizationTab(ModelConfig())
    tab.cfg.elem_size = 0.01
    for k, v in (("xmin", "-0.03"), ("xmax", "0.03"), ("ymin", "-0.03"),
                 ("ymax", "0.03")):
        tab.le_zoi[k].setText(v)
    tab.sp_margin.setValue(1)
    _fill_thresholds(tab)
    tab._dom_spins["dom_step_elems"].setValue(4)
    monkeypatch.setattr(tab, "_validate_launch", lambda: ("p", tmp_path, 2))
    monkeypatch.setattr(tab, "_start_progress", lambda: None)

    def fake_make(prefs, run_dir, cpus, prefix):
        def run_bundle(cfg):
            g = cfg.euler_geometry
            return _GridBundle(DomainDims(g.h_wp, g.h_void, g.l_wp, g.l_void),
                               cfg.elem_size)
        run_bundle.state = {"sta": None, "job": None}
        return run_bundle
    monkeypatch.setattr(tab, "_make_run_bundle", fake_make)
    tab._on_run_domain_independence()
    assert tab._di_worker.wait(60000)
    for _ in range(50):
        qapp.processEvents()
    res = tab._last_domain_result
    assert res is not None, tab.log.toPlainText()
    assert res.status == "converged"
    assert all(r.guards_ok for r in res.runs)
    # ZOI half-width 0.03 + 1 elem (0.01) = 0.04 mm per side -> 8 x 8 elems
    assert res.runs[0].cost.n_elem_euler == 8 * 8
    text = tab.log.toPlainText()
    assert "RESULT: every dimension independent" in text
    assert "R_HG=0.01" in text


def _analytic_tab(qapp, monkeypatch, tmp_path):
    from gui.core.domain_sizing import DomainDims
    tab = OptimizationTab(ModelConfig())
    tab.cfg.elem_size = 0.01
    tab.cfg.step.output_filter_enabled = False
    for k, v in (("xmin", "-0.03"), ("xmax", "0.03"), ("ymin", "-0.03"),
                 ("ymax", "0.03")):
        tab.le_zoi[k].setText(v)
    tab.sp_margin.setValue(1)
    _fill_thresholds(tab)
    tab._dom_spins["dom_step_elems"].setValue(4)
    monkeypatch.setattr(tab, "_validate_launch", lambda: ("p", tmp_path, 2))
    monkeypatch.setattr(tab, "_start_progress", lambda: None)

    def fake_make(prefs, run_dir, cpus, prefix):
        def run_bundle(cfg):
            g = cfg.euler_geometry
            return _GridBundle(DomainDims(g.h_wp, g.h_void, g.l_wp, g.l_void),
                               cfg.elem_size)
        run_bundle.state = {"sta": None, "job": None}
        return run_bundle
    monkeypatch.setattr(tab, "_make_run_bundle", fake_make)
    return tab


def _drain(qapp, worker):
    assert worker.wait(120000)
    for _ in range(50):
        qapp.processEvents()


def test_checks_and_exports_through_the_tab(qapp, monkeypatch, tmp_path):
    """Domain study -> exports, then interaction checks (real GCI on D*) ->
    checks exports; table and plots filled."""
    import csv
    import json
    tab = _analytic_tab(qapp, monkeypatch, tmp_path)
    assert not tab.btn_checks.isEnabled()
    tab._on_run_domain_independence()
    _drain(qapp, tab._di_worker)
    res = tab._last_domain_result
    assert res is not None and res.status == "converged"
    folder = tab._last_domain_dir
    for name in ("runs.csv", "comparisons.csv", "dimensions.csv",
                 "summary.json"):
        assert (folder / name).exists(), name
    assert tab.btn_checks.isEnabled()
    assert tab.table.rowCount() > 0
    n_runs_before = res.n_runs

    tab._on_run_interaction_checks()
    _drain(qapp, tab._checks_worker)
    chk = tab._last_checks
    assert chk is not None, tab.log.toPlainText()
    assert [c.name for c in chk.checks] == ["ms_x_mesh", "domain_combined",
                                            "ms_at_point", "mesh_x_domain"]
    # combined domain: exactly one extra run (D* reused from the study)
    assert res.n_runs == n_runs_before + 1
    assert chk.checks[2].passed is True          # f constant in h -> exact
    assert len(chk.gci_calls) == 3
    rows = list(csv.DictReader(open(folder / "checks.csv")))
    assert [r["check"] for r in rows] == ["ms_x_mesh", "domain_combined",
                                          "ms_at_point", "mesh_x_domain"]
    s = json.loads((folder / "summary.json").read_text())
    assert s["interaction_checks"]["status"] == chk.status
    assert "INTERACTION CHECKS:" in tab.log.toPlainText()


def test_gci_records_cost_and_exports(qapp, monkeypatch, tmp_path):
    tab = _analytic_tab(qapp, monkeypatch, tmp_path)
    tab.cfg.euler_geometry.h_wp = tab.cfg.euler_geometry.h_void = 0.1
    tab.cfg.euler_geometry.l_wp = tab.cfg.euler_geometry.l_void = 0.1
    tab._on_run_mesh_gci()
    _drain(qapp, tab._mesh_worker)
    assert tab._last_gci is not None, tab.log.toPlainText()
    res, calls, tol, folder = tab._last_gci
    assert len(calls) == len(res.sizes)
    assert all(c.cost.n_elem_euler for c in calls)
    assert (folder / "gci.csv").exists() and (folder / "gci_meshes.csv").exists()
    # the user's config is untouched by the study
    assert tab.cfg.elem_size == 0.01


# ---------------------------------------------------------------------------
# Config-derived tab logic (moved from the removed tests/test_domain_opt.py)
# ---------------------------------------------------------------------------
class TestTabConfigLogic:

    def _tab(self):
        from gui.tabs.optimization_tab import OptimizationTab
        from gui.core.model_config import ModelConfig
        return OptimizationTab(ModelConfig())

    def test_config_inputs(self, qapp):
        tab = self._tab()
        inp = tab.config_inputs()
        # t1 = wp_y0 - tool_y0 = 0 - (-0.05) = 0.05 (default model)
        assert inp["t1"] == pytest.approx(0.05)
        assert inp["elem"] == pytest.approx(0.005)
        assert len(inp["roi"]) == 4

    def test_initial_domain_is_the_roi_when_zoi_blank(self, qapp):
        tab = self._tab()
        c = tab.cfg
        # set a known measurement ROI (BBox) and a matching element size
        c.bbox.xmin, c.bbox.xmax = -0.20, 0.05
        c.bbox.ymin, c.bbox.ymax = -0.10, 0.15
        c.elem_size = 0.01
        tab.sp_margin.setValue(0)
        d0 = tab.compute_initial_dims()
        # domain-frame mapping: l_wp=-xmin, l_void=xmax, h_wp=-ymin, h_void=ymax
        assert d0.l_wp == pytest.approx(0.20)
        assert d0.l_void == pytest.approx(0.05)
        assert d0.h_wp == pytest.approx(0.10)
        assert d0.h_void == pytest.approx(0.15)

    def test_quantity_field_map_all_field_backed(self, qapp):
        tab = self._tab()
        m = tab.quantity_field_map()
        assert m == {"Vx": "V1", "Vy": "V2", "T": "TEMP", "EVF": "EVF"}
        # forces are always present as channels (not in the field map)
        assert tab.force_channels() == {"Fc": "RF1_RP", "Ff": "RF2_RP"}

    def test_thresholds_required_for_all_fields(self, qapp):
        tab = self._tab()
        assert tab.thresholds_complete() is False        # none set yet
        for q in ("Vx", "Vy", "T", "EVF", "Fc", "Ff"):
            tab._q_eps[q].setText("1")
        tab._q_eps["Vx"].setText("2.5")
        tab._q_eps["T"].setText("1,0")                   # comma accepted
        thr = tab.thresholds()
        assert thr["Vx"] == pytest.approx(2.5)
        assert thr["T"] == pytest.approx(1.0)
        assert tab.thresholds_complete() is True




# ---------------------------------------------------------------------------
# Step 0: mass-scaling independence study wiring
# ---------------------------------------------------------------------------
class TestMsStudyWiring:
    @pytest.fixture
    def ms_launch(self, launch, monkeypatch):
        import gui.tabs.optimization_tab as ot
        monkeypatch.setattr(ot, "MsIndependenceWorker", _CaptureWorker)
        launch.cfg.step.output_filter_enabled = True
        launch.cfg.step.output_filter_verify = False
        return launch

    def test_defaults(self, tab):
        values, elem = tab.ms_settings()
        assert values == (250.0, 500.0, 1000.0, 2000.0, 4000.0)
        assert elem == tab.cfg.elem_size

    def test_busy_includes_the_ms_button(self, tab):
        tab._busy(True, "running")
        assert not tab.btn_ms.isEnabled()
        tab._busy(False)
        assert tab.btn_ms.isEnabled()

    def test_requires_the_tolerances(self, ms_launch, warnings):
        ms_launch._on_run_ms_independence()
        assert warnings and "tolerances" in warnings[0].lower()
        assert _CaptureWorker.last is None

    def test_requires_the_output_filter(self, ms_launch, warnings):
        _fill_thresholds(ms_launch)
        ms_launch.cfg.step.output_filter_enabled = False
        ms_launch._on_run_ms_independence()
        assert warnings and "filter" in warnings[0].lower()
        assert _CaptureWorker.last is None

    def test_rejects_bad_ms_values(self, ms_launch, warnings):
        _fill_thresholds(ms_launch)
        ms_launch.le_ms_values.setText("1000, 500")
        ms_launch._on_run_ms_independence()
        assert warnings and "increasing" in warnings[0]
        assert _CaptureWorker.last is None

    def test_launch_passes_the_study_settings(self, ms_launch):
        tab = ms_launch
        _fill_thresholds(tab)
        tab.le_ms_values.setText("500 1000 2000")
        tab.le_ms_elem.setText("0.004")
        tab._on_run_ms_independence()
        kw = _CaptureWorker.last
        assert kw["ms_values"] == (500.0, 1000.0, 2000.0)
        assert kw["elem_size"] == pytest.approx(0.004)
        assert kw["thresholds"]["Ff"] == pytest.approx(0.5)
        assert kw["domain_dims"] == tab._dims_from_cfg()
        assert kw["base_cfg"] is not tab.cfg
        assert kw["base_cfg"].step.output_filter_verify is True
        assert tab.cfg.step.output_filter_verify is False
        assert callable(kw["guard_fn"]) and callable(kw["cost_fn"])

    def test_guard_fn_adds_the_filter_checks(self, ms_launch, monkeypatch):
        tab = ms_launch
        _fill_thresholds(tab)
        made = {}
        real = tab._make_run_bundle

        def capture(*a, **k):
            made["rb"] = real(*a, **k)
            return made["rb"]
        monkeypatch.setattr(tab, "_make_run_bundle", capture)
        tab._on_run_ms_independence()
        guard_fn = _CaptureWorker.last["guard_fn"]
        made["rb"].state["filter_check"] = {
            "passed": True, "filters": {"SENSORBAND": {"rel_max_dev": 0.001}},
            "reverberation": {"passed": True, "e_rev": 0.002}}
        b = _GridBundle(tab._dims_from_cfg(), 0.01)
        g = guard_fn(b)
        assert g["filter"] == (pytest.approx(0.001), True)
        assert g["reverb"] == (pytest.approx(0.002), True)
        assert g["R_K"][1] is True
        made["rb"].state["filter_check"] = None
        assert guard_fn(b)["filter"] == (None, False)

    def test_result_is_logged_tabled_and_exported(self, tab, tmp_path):
        from gui.sensitivity.domain_independence import RunRecord
        from gui.sensitivity.ms_independence import (MsComparison,
                                                     MsStudyResult)
        dims = {"h_wp": .2, "h_void": .2, "l_wp": .2, "l_void": .2}
        runs = [RunRecord(index=i, dims=dims, job_ok=True,
                          guards={"reverb": (0.002, True)})
                for i in range(3)]
        comps = [MsComparison(1, 500., 1000., {"T": 3.0}, 0.3, "T", True,
                              True, 0, 1),
                 MsComparison(2, 1000., 2000., {"T": 12.0}, 1.2, "T", True,
                              False, 1, 2)]
        res = MsStudyResult(ms_values=[500., 1000., 2000.], retained=1000.,
                            status="converged", runs=runs,
                            run_ms=[500., 1000., 2000.], comparisons=comps,
                            settings={"elem_size": 0.004,
                                      "thresholds": {"T": 10.0}})
        tab._on_ms_progress({"phase": "run", "record": runs[0], "ms": 500.})
        tab._on_ms_progress({"phase": "comparison", "comparison": comps[1]})
        tab._pending_ms_dir = tmp_path
        tab._on_ms_done(res)
        text = tab.log.toPlainText()
        assert "ms=500 | job ok | reverb=0.002" in text
        assert "ms 1000->2000" in text and "not independent" in text
        assert "largest independent ms = 1000" in text
        assert (tmp_path / "ms_comparisons.csv").exists()
        cells = [[tab.table.item(r, c).text() for c in range(3)]
                 for r in range(tab.table.rowCount())]
        assert ["ms", "ms", "1000"] in cells


def test_checks_take_the_ms_value_before_ms_star(qapp):
    from gui.sensitivity.ms_independence import MsStudyResult
    tab = OptimizationTab(ModelConfig())
    assert tab.ms_lower_for_checks() is None
    tab.cfg.step.mass_scaling_enabled = True
    tab.cfg.step.mass_scaling_factor = 1000.0
    tab._last_ms = (MsStudyResult(ms_values=[250., 500., 1000., 2000.]),
                    None)
    assert tab.ms_lower_for_checks() == 500.0
    tab.cfg.step.mass_scaling_factor = 1500.0
    assert tab.ms_lower_for_checks() is None
