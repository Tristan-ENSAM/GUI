# -*- coding: utf-8 -*-
"""Model tab: reuse of finished runs (resume, load), the record of each step
for the current model, the whole pipeline, the help toggle and the preview.

The disk tests run the real launcher of the tab (`_make_run_bundle`) with a
fake Abaqus: each "run" writes the files a finished job leaves
(<job>.results.npz, <job>.meta.json with model_config, <job>.sta with the
success mark), and loading a bundle builds the analytic grid bundle of
test_optimization_ui from the parameters saved in the meta file.
"""
from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QAbstractButton, QLabel

from gui.core.domain_sizing import DomainDims
from gui.core.model_config import ModelConfig
from gui.sensitivity import run_cache as rc
from gui.sensitivity import study_specs as ss
from gui.sensitivity import study_state as st
from gui.tabs.optimization_tab import OptimizationTab
from tests.test_optimization_ui import _GridBundle

_OK_MARK = " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"


# ---------------------------------------------------------------------------
# run_cache
# ---------------------------------------------------------------------------
def _write_run(folder, job, params, ok=True):
    folder = Path(folder)
    (folder / (job + ".results.npz")).write_bytes(b"npz")
    (folder / (job + ".meta.json")).write_text(
        json.dumps({"model_config": params}), encoding="utf-8")
    (folder / (job + ".sta")).write_text(
        "header\n" + ("x" * 200000) + "\n" + (_OK_MARK if ok else "ERROR\n"),
        encoding="latin-1")


class TestRunCache:
    def test_key_ignores_int_float_and_key_order(self):
        a = {"mesh": {"elem_size": 1, "n": 2.0}, "b": (1, 2)}
        b = {"b": [1.0, 2.0], "mesh": {"n": 2, "elem_size": 1.0}}
        assert rc.params_key(a) == rc.params_key(b)
        assert rc.params_key(a) != rc.params_key({"mesh": {"elem_size": 2}})
        # bools are not numbers
        assert rc.params_key({"x": True}) != rc.params_key({"x": 1.0})

    def test_finished_runs_only(self, tmp_path):
        _write_run(tmp_path, "GCI_run000", {"a": 1})
        _write_run(tmp_path, "GCI_run001", {"a": 2}, ok=False)
        (tmp_path / "GCI_run002.meta.json").write_text(
            json.dumps({"model_config": {"a": 3}}))          # no npz/sta
        assert rc.run_completed(tmp_path, "GCI_run000")      # mark in tail
        assert not rc.run_completed(tmp_path, "GCI_run001")
        assert not rc.run_completed(tmp_path, "GCI_run002")
        cache = rc.RunCache([tmp_path])
        assert len(cache) == 1
        hit = cache.lookup({"a": 1.0})
        assert hit is not None and hit.job == "GCI_run000"
        assert cache.lookup({"a": 2}) is None
        assert cache.add_folder(tmp_path) == 0               # once

    def test_new_jobs_are_numbered_after_the_existing_ones(self, tmp_path):
        assert rc.next_job_index(tmp_path, "ms") == 0
        (tmp_path / "ms_run000.sta").write_text("")
        (tmp_path / "ms_run007.log").write_text("")
        (tmp_path / "msx_run050.sta").write_text("")         # other prefix
        assert rc.next_job_index(tmp_path, "ms") == 8

    def test_remove_job_files(self, tmp_path):
        _write_run(tmp_path, "d_run000", {"a": 1})
        (tmp_path / "d_run0001.sta").write_text("")           # other job
        assert rc.remove_job_files(tmp_path, "d_run000") == []
        assert [p.name for p in tmp_path.iterdir()] == ["d_run0001.sta"]

    def test_closest_reports_the_differences(self, tmp_path):
        _write_run(tmp_path, "r0", {"mesh": {"elem_size": 0.01},
                                    "mat": {"A": 880.0}})
        cache = rc.RunCache([tmp_path])
        run, diffs = cache.closest({"mesh": {"elem_size": 0.02},
                                    "mat": {"A": 900.0}},
                                   ignore=["mesh.elem_size"])
        assert run.job == "r0"
        assert diffs == [("mat.A", 880.0, 900.0)]


# ---------------------------------------------------------------------------
# study_state
# ---------------------------------------------------------------------------
def _params(**over):
    p = ModelConfig().to_params_dict()
    for dotted, v in over.items():
        node = p
        keys = dotted.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = v
    return p


class TestStudyState:
    def test_model_key_ignores_the_sized_quantities(self):
        base = st.model_key(_params())
        assert st.model_key(_params(mesh__elem_size=0.123)) == base
        assert st.model_key(_params(
            step__mass_scaling_factor_eulerian=16000.0)) == base
        assert st.model_key(_params(
            geometry__euler__geometry__l_wp=1.5)) == base
        assert st.model_key(_params(
            interaction__friction_coeff=0.42)) != base

    def _steps(self, mk="m", sk="s"):
        rec = lambda step, value, inputs: st.make_record(  # noqa: E731
            "done", value, "/f/" + step, mk, sk, inputs)
        return {"ms": rec("ms", 1000.0, {"h": 0.004}),
                "mesh": rec("mesh", 0.002, {"ms": 1000.0}),
                "domain": rec("domain", [0.1, 0.2, 0.3, 0.4],
                              {"ms": 1000.0, "h": 0.002}),
                "checks": rec("checks", "accepted",
                              {"ms": 1000.0, "h": 0.002,
                               "dims": [0.1, 0.2, 0.3, 0.4]})}

    def test_states(self):
        steps = self._steps()
        for step in st.STEPS:
            assert st.step_state(steps, step, "m", "s")[0] == "done"
        assert st.step_state(steps, "ms", "other", "s")[0] == "stale"
        assert st.step_state(steps, "ms", "m", "other")[0] == "stale"
        assert st.step_state({}, "mesh", "m", "s")[0] == "none"
        # step 0 redone with another result: every later step is upstream
        steps["ms"]["value"] = 2000.0
        assert st.step_state(steps, "mesh", "m", "s")[0] == "upstream"
        assert st.step_state(steps, "checks", "m", "s")[0] == "upstream"
        # an interrupted step 1 made at the old ms* is upstream too (no
        # resume of a study made on an old result) ...
        steps["mesh"]["status"] = "interrupted"
        assert st.step_state(steps, "mesh", "m", "s")[0] == "upstream"
        # ... and resumable when it ran at the current ms*
        steps["mesh"]["inputs"]["ms"] = 2000.0
        assert st.step_state(steps, "mesh", "m", "s")[0] == "interrupted"

    def test_display_rounding_still_matches(self):
        # the model tabs show 6 significant digits
        assert st.values_match("mesh", 0.000333333, 1.0 / 3000)
        assert st.values_match("domain", [0.1, 0.2, 0.3, 0.4],
                               [0.1000001, 0.2, 0.3, 0.4])
        assert not st.values_match("ms", 1000.0, 1001.0)

    def test_missing_prerequisites_name_the_step_and_the_tab(self):
        steps = self._steps()
        mv = {"ms": 1000.0, "h": 0.004, "dims": [0.1, 0.2, 0.3, 0.4]}
        out = st.missing_prerequisites(steps, "domain", "m", "s", mv)
        assert len(out) == 1 and "Mesh tab" in out[0] and "h = 0.002" in out[0]
        out = st.missing_prerequisites({}, "mesh", "m", "s", mv)
        assert out == ["Step 0 (mass scaling) is not done yet."]


# ---------------------------------------------------------------------------
# study_specs
# ---------------------------------------------------------------------------
class TestStudySpecs:
    def _folder(self, tmp_path, study, params):
        d = tmp_path / ("Untitled_%s_x" % study)
        d.mkdir()
        (d / "config.json").write_text(json.dumps(
            {"study": study, "parameters": params}))
        return d

    def test_older_gci_folder_is_completed_from_its_runs(self, tmp_path):
        spec = {"zoi": {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1},
                "window": [0.3, 1.0], "finest_elem_size": 0.001, "ratio": 2,
                "n_meshes": 4, "grid_step": 0.001,
                "tolerances": {"TEMP": 10, "V1": 10},
                "domain_dims": {"h_wp": 1, "h_void": 1, "l_wp": 1,
                                "l_void": 1}}
        d = self._folder(tmp_path, "GCI", spec)
        _write_run(d, "GCI_run000", _params(
            step__mass_scaling_enabled=True,
            step__mass_scaling_factor_eulerian=16000.0,
            step__output_filter_verify=False))
        step, out = ss.read_study_folder(d, {"rk_max": 0.05, "rhg_max": 0.05,
                                             "base_ms": [False, 1.0]})
        assert step == "mesh"
        assert out["base_ms"] == [True, 16000.0]
        assert out["filter_verify"] is False
        assert out["rk_max"] == 0.05
        assert ss.eps_of_spec(out) == {"T": 10.0, "Vx": 10.0}

    def test_not_a_study_folder(self, tmp_path):
        with pytest.raises(ValueError, match="no config.json"):
            ss.read_study_folder(tmp_path)
        d = self._folder(tmp_path, "sensitivity", {})
        with pytest.raises(ValueError, match="not a step"):
            ss.read_study_folder(d)
        d2 = tmp_path / "x"
        d2.mkdir()
        (d2 / "config.json").write_text(json.dumps(
            {"study": "massscaling", "parameters": {"zoi": [0, 1, 0, 1]}}))
        with pytest.raises(ValueError, match="lacks"):
            ss.read_study_folder(d2)

    def test_checks_config_round_trip(self, tmp_path):
        spec = {"h_star": 0.002, "gci_plan": {}, "gci_tolerances": {},
                "zoi": [0, 1, 0, 1], "thresholds_abs": {"T": 1},
                "window": [0.3, 1], "rk_max": 0.05, "rhg_max": 0.05,
                "base_ms": [True, 1000.0], "d_star": [1, 1, 1, 1]}
        assert ss.write_checks_config(tmp_path, spec)
        assert ss.read_checks_config(tmp_path) == spec
        assert ss.read_checks_config(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# The tab, without runs
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def answers(monkeypatch):
    """Questions answer with `answers.next` (once) or their first choice;
    information boxes are collected."""
    box = SimpleNamespace(asked=[], informed=[], next=None)

    def ask(self, title, text, choices, default=None):
        box.asked.append((title, text, [k for k, _l in choices]))
        if box.next is not None:
            key, box.next = box.next, None
            return key
        return choices[0][0]
    monkeypatch.setattr(OptimizationTab, "_ask", ask)
    monkeypatch.setattr(OptimizationTab, "_inform",
                        lambda self, title, text: box.informed.append(text))
    return box


@pytest.fixture
def tab(qapp):
    return OptimizationTab(ModelConfig())


def test_help_is_hidden_until_asked(tab):
    assert tab._hints and all(h.isHidden() for h in tab._hints)
    tab.cb_help.setChecked(True)
    assert not any(h.isHidden() for h in tab._hints)
    tab.cb_help.setChecked(False)
    assert all(h.isHidden() for h in tab._hints)


def test_no_text_of_the_tab_mentions_a_paper(tab):
    texts = []
    for w in tab.findChildren(QLabel) + tab.findChildren(QAbstractButton):
        texts += [w.text(), w.toolTip()]
    from PySide6.QtWidgets import QLineEdit
    for w in tab.findChildren(QLineEdit):
        texts += [w.toolTip(), w.placeholderText()]
    bad = [t for t in texts if "paper" in t.lower() or "jmpt" in t.lower()]
    assert bad == []


def test_every_step_has_a_status_line(tab):
    for step in st.STEPS:
        assert "Not done yet" in tab._step_status[step].text()
    assert not tab._btn_apply["ms"].isVisibleTo(tab)


def _record(tab, step, value, inputs=None, status="done", **extra):
    mkey, skey = tab._keys()
    tab.cfg.optimization.steps[step] = st.make_record(
        status, value, "/nowhere/" + step, mkey, skey, inputs or {},
        model_params=tab._model_params(), **extra)


def test_use_in_model_writes_the_result_and_reloads_the_tabs(tab):
    reloaded = []
    tab.set_model_refresher(lambda step: reloaded.append(step))
    _record(tab, "ms", 16000.0)
    tab._refresh_step_status()
    assert "ms = 16000" in tab._step_status["ms"].text()
    assert "Use in model" in tab._step_status["ms"].text() or \
        tab._btn_apply["ms"].isEnabled()
    tab._btn_apply["ms"].click()
    assert tab.cfg.step.mass_scaling_enabled is True
    assert tab.cfg.step.mass_scaling_factor == 16000.0
    assert reloaded == ["ms"]
    assert "The model uses it" in tab._step_status["ms"].text()
    assert not tab._btn_apply["ms"].isEnabled()
    # writing the result does not invalidate the step
    assert tab._step_state("ms")[0] == "done"
    # a step is done only on the results of the steps before it
    _record(tab, "domain", [0.11, 0.22, 0.33, 0.44],
            {"ms": 16000.0, "h": float(tab.cfg.elem_size)})
    assert tab._step_state("domain")[0] == "prereq"
    tab._refresh_step_status()
    text = tab._step_status["domain"].text()
    assert "Step 1 (element size) is not done yet" in text
    assert not tab._btn_apply["domain"].isVisibleTo(tab)
    _record(tab, "mesh", float(tab.cfg.elem_size), {"ms": 16000.0})
    assert tab._step_state("domain")[0] == "done"
    tab._on_apply_step("domain")
    g = tab.cfg.euler_geometry
    assert (g.h_wp, g.h_void, g.l_wp, g.l_void) == (0.11, 0.22, 0.33, 0.44)


def test_a_model_change_makes_the_record_stale_and_says_what(tab):
    _record(tab, "ms", 1000.0)
    tab.cfg.interaction.friction_coeff = 0.42
    tab._refresh_step_status()
    assert "another model" in tab._step_status["ms"].text()
    assert "friction_coeff" in tab._step_status["ms"].toolTip()


def test_running_a_step_out_of_order_asks_first(tab, answers, monkeypatch):
    started = []
    monkeypatch.setattr(tab, "_validate_launch",
                        lambda *a: started.append(1) or None)
    answers.next = "cancel"
    tab._on_run_domain_independence()
    title, text, keys = answers.asked[-1]
    assert title == "Order of the steps"
    assert "Step 0 (mass scaling) is not done yet." in text
    assert "Step 1 (element size) is not done yet." in text
    assert keys == ["run", "cancel"]
    assert started == []                      # declined: nothing launched
    tab._on_run_domain_independence()          # "Run anyway"
    assert started == [1]


def test_ms_study_runs_on_the_coarsest_mesh_of_step_1(tab):
    tab.le_gci_finest.setText("0.0005")
    assert tab.ms_settings()[1] == pytest.approx(0.004)
    tab.le_ms_elem.setText("0.002")
    assert tab.ms_settings()[1] == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
def _legend(tab):
    leg = tab.preview._ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def test_preview_legend_names_every_element_once(tab):
    tab.le_zoi["xmin"].setText("-0.05")
    tab.le_zoi["xmax"].setText("0.05")
    tab.le_zoi["ymin"].setText("-0.05")
    tab.le_zoi["ymax"].setText("0.0")
    tab._max["l_wp"].setText("0.5")
    tab._draw_preview()
    labels = _legend(tab)
    assert "ROI (results written here)" in labels
    assert "ZOI (comparison zone)" in labels
    assert any(l.startswith("Step 2 starting domain (") for l in labels)
    assert "Largest size allowed (step 2)" in labels   # a single cap
    assert "Eulerian domain (Geometry tab)" in labels
    assert not any(l.startswith("Tool RP") for l in labels)
    assert any(l.startswith("ZOI sampling points") for l in labels)
    assert len(labels) == len(set(labels))
    assert tab.preview._ax.get_title() == ""
    # the view holds the cap line
    assert tab.preview._ax.get_xlim()[0] < -0.5


def test_preview_thins_the_sampling_points(tab):
    tab._zoi_from_roi()
    tab.le_grid_step.setText("0.0005")                 # ~360 000 points
    tab._draw_preview()
    ax = tab.preview._ax
    pts = sum(len(c.get_offsets()) for c in ax.collections)
    assert pts <= 2 * 2000
    assert any("shown" in l for l in _legend(tab))


def test_preview_warns_when_the_zoi_leaves_the_roi(tab):
    roi = tab.config_inputs()["roi"]
    tab.le_zoi["xmin"].setText("%g" % (roi[0] - 1.0))
    tab._draw_preview()
    texts = [t.get_text() for t in tab.preview._ax.texts]
    assert any("beyond the ROI" in t for t in texts)


# ---------------------------------------------------------------------------
# With runs on disk (fake Abaqus)
# ---------------------------------------------------------------------------
_FC_OK = {"passed": True, "filters": {"SENSORBAND": {"rel_max_dev": 0.001}},
          "reverberation": {"passed": True, "e_rev": 0.002}}


class _ModelBundle(_GridBundle):
    """The analytic bundle of the run's own parameters, with a temperature
    that jumps by 50 K from ms = 2000 on (so ms* = 1000)."""
    LAM = 0.02          # decay length of the boundary influence [mm]

    def __init__(self, params):
        g = params["geometry"]["euler"]["geometry"]
        s = params["step"]
        self.ms = (float(s["mass_scaling_factor_eulerian"])
                   if s["mass_scaling_enabled"] else 1.0)
        super().__init__(DomainDims(g["h_wp"], g["h_void"], g["l_wp"],
                                    g["l_void"]),
                         float(params["mesh"]["elem_size"]), lam=self.LAM)

    def field(self, inst, var):
        out = super().field(inst, var)
        if var == "TEMP" and self.ms >= 2000:
            out = out + 50.0
        return out


@pytest.fixture
def world(qapp, monkeypatch, tmp_path):
    import subprocess
    import gui.core.filter_check as fcm
    import gui.tabs.optimization_tab as ot
    launched = []
    # behaviour["fn"](params) -> "ok" | "abort" (Abaqus stops the analysis)
    # | "raise" (Abaqus cannot be started)
    behaviour = {"fn": None}

    class FakeProc:
        def __init__(self, args, cwd=None, stdout=None, stderr=None):
            _tag, params, run = args
            how = behaviour["fn"](params) if behaviour["fn"] else "ok"
            if how == "raise":
                raise OSError("agent down")
            launched.append(run["job_name"])
            self.returncode, self.pid = 0, 0
            self.stdout = io.BytesIO(b"")
            if how == "abort":
                (Path(cwd) / (run["job_name"] + ".sta")).write_text(
                    "***ERROR\n THE ANALYSIS HAS NOT BEEN COMPLETED\n")
                self.returncode = 1
                return
            _write_run(cwd, run["job_name"], params)

        def poll(self):
            return 0

        def wait(self):
            return 0

    class FakeBundles:
        @staticmethod
        def load(path):
            meta = Path(str(path)[:-len(".results.npz")] + ".meta.json")
            return _ModelBundle(json.loads(meta.read_text())["model_config"])

    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    monkeypatch.setattr(ot, "build_abaqus_args",
                        lambda cmd, script, params, run: ("fake", params, run))
    monkeypatch.setattr(ot, "ResultsBundle", FakeBundles)
    monkeypatch.setattr(fcm, "check_bundle", lambda *a, **k: dict(_FC_OK))
    monkeypatch.setattr(fcm, "format_report", lambda fc: "")
    prefs = SimpleNamespace(abaqus_cmd="abaqus", abaqus_script="run.py",
                            execution_mode="local",
                            default_workdir=str(tmp_path))
    t = OptimizationTab(ModelConfig(), prefs_getter=lambda: prefs,
                        cpus_getter=lambda: 2)
    validate = []
    monkeypatch.setattr(t, "_validate_launch",
                        lambda *a: validate.append(a) or (prefs, tmp_path, 2))
    monkeypatch.setattr(t, "_start_progress", lambda: None)
    t.cfg.elem_size = 0.01
    t.cfg.step.output_filter_enabled = True
    for k, v in (("xmin", "-0.03"), ("xmax", "0.03"), ("ymin", "-0.03"),
                 ("ymax", "0.03")):
        t.le_zoi[k].setText(v)
    t.sp_margin.setValue(1)
    for q, v in (("Vx", "10"), ("Vy", "10"), ("T", "1"), ("EVF", "0.01"),
                 ("Fc", "0.5"), ("Ff", "0.5")):
        t._q_eps[q].setText(v)
    t._dom_spins["dom_step_elems"].setValue(4)
    return SimpleNamespace(tab=t, launched=launched, wd=tmp_path,
                           behaviour=behaviour, validate=validate)


def _wait(qapp, tab, until, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        qapp.processEvents()
        if until():
            for _ in range(20):
                qapp.processEvents()
            return
        time.sleep(0.005)
    raise AssertionError("timed out\n" + tab.log.toPlainText()[-4000:])


def _idle(tab):
    return lambda: not tab._is_busy and tab._active is None


def test_domain_study_resume_load_and_other_model(qapp, world, answers):
    tab = world.tab
    tab._on_run_domain_independence()
    _wait(qapp, tab, _idle(tab))
    rec = tab.cfg.optimization.steps["domain"]
    assert rec["status"] == "done", tab.log.toPlainText()[-3000:]
    folder = Path(rec["folder"])
    final = tab._last_domain_result.final
    n_runs = len(world.launched)
    assert n_runs == tab._last_domain_result.n_runs
    assert rec["value"] == [final.h_wp, final.h_void, final.l_wp,
                            final.l_void]

    # Load: the whole result comes back without a single run.
    tab.forget_results()
    assert tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert len(world.launched) == n_runs
    assert tab._last_domain_result.final == final
    assert tab.cfg.optimization.steps["domain"]["status"] == "done"
    assert "already computed: reused" in tab.log.toPlainText()

    # Two runs of the folder of record never finished: its result stays
    # (done, but steps 0 and 1 are not done here), the user is offered to
    # compute them.
    jobs = sorted(p.name[:-len(".meta.json")]
                  for p in folder.glob("domainsizing_run*.meta.json"))
    for job in jobs[-2:]:
        (folder / (job + ".sta")).write_text("STOPPED\n")
    answers.next = "later"
    before = dict(tab.cfg.optimization.steps["domain"])
    tab.forget_results()
    tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert "are missing" in answers.asked[-1][1]
    assert "stays recorded" in answers.asked[-1][1]
    assert tab.cfg.optimization.steps["domain"] == before
    assert tab._step_state("domain")[0] == "prereq"
    # Without a result of record, the same folder is an unfinished study.
    del tab.cfg.optimization.steps["domain"]
    answers.next = "later"
    tab.forget_results()
    tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert "is not finished" in answers.asked[-1][1]
    assert len(world.launched) == n_runs
    assert tab._step_state("domain")[0] == "interrupted"
    # Run the step again: resume, only the two missing runs are launched.
    tab._on_run_domain_independence()
    _wait(qapp, tab, _idle(tab))
    assert answers.asked[-1][2] == ["resume", "new", "cancel"]
    assert len(world.launched) == n_runs + 2
    assert tab._last_domain_result.final == final
    assert tab.cfg.optimization.steps["domain"]["status"] == "done"
    # (steps 0 and 1 are not done here: the step says so)
    assert tab._step_state("domain")[0] == "prereq"
    # The new jobs do not reuse the names of the earlier ones.
    assert world.launched[-2:] == ["domainsizing_run%03d" % i
                                   for i in (n_runs, n_runs + 1)]
    # The launch check looked at the study folder, not the working dir.
    assert world.validate[-1] == (folder,)

    # The same folder for another model: nothing is launched, the user is
    # told what differs.
    tab.cfg.interaction.friction_coeff += 0.1
    tab.forget_results()
    tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert len(world.launched) == n_runs + 2
    assert "another model" in world.tab.lbl_status.text()
    assert "friction_coeff" in answers.informed[-1]


def test_run_all_steps_chains_the_results(qapp, world, answers,
                                          monkeypatch):
    # A boundary influence that dies out fast, so that the four sides
    # grown together (final check) stay within eps_q as well.
    monkeypatch.setattr(_ModelBundle, "LAM", 0.005)
    tab = world.tab
    # A ZOI inside the material (y < 0 in the analytic bundle) and a plan
    # of 3 meshes 0.01 / 0.02 / 0.04: step 0 runs on 0.04.
    tab.le_zoi["ymin"].setText("-0.09")
    tab.le_zoi["ymax"].setText("-0.01")
    tab.sp_gci_n.setValue(3)
    tab._on_run_all()
    assert answers.asked[-1][0] == "Run all steps"
    _wait(qapp, tab, lambda: not tab._pipeline and not tab._is_busy
          and tab._active is None)
    steps = tab.cfg.optimization.steps
    log = tab.log.toPlainText()
    assert [steps.get(s, {}).get("status") for s in st.STEPS] == \
        ["done"] * 4, log[-4000:]
    # each result was written into the model before the next step
    assert tab.cfg.step.mass_scaling_enabled is True
    assert tab.cfg.step.mass_scaling_factor == 1000.0
    assert tab.cfg.elem_size == steps["mesh"]["value"]
    assert steps["mesh"]["value"] in (pytest.approx(0.02),
                                      pytest.approx(0.04))
    g = tab.cfg.euler_geometry
    assert st.dims_close([g.h_wp, g.h_void, g.l_wp, g.l_void],
                         steps["domain"]["value"])
    assert steps["mesh"]["inputs"]["ms"] == 1000.0
    assert steps["domain"]["inputs"] == {"ms": 1000.0,
                                         "h": steps["mesh"]["value"]}
    assert steps["ms"]["inputs"]["h"] == pytest.approx(0.04)
    assert steps["checks"]["inputs"]["dims"] == steps["domain"]["value"]
    # the checks ran the step-1 plan again on D*
    assert tab._last_checks.gci_calls[0].elem_size == pytest.approx(0.01)
    assert "ALL STEPS DONE" in log
    for step in st.STEPS:
        assert tab._step_state(step)[0] == "done"

    # Run again: everything is done for this model, nothing is launched.
    n = len(world.launched)
    tab._on_run_all()
    _wait(qapp, tab, lambda: not tab._pipeline and not tab._is_busy)
    assert len(world.launched) == n
    assert "already done for this model" in tab.log.toPlainText()

    # The plan of step 1 does not follow h* written into the Mesh tab.
    assert tab.le_gci_finest.text() == "0.01"
    assert tab._gci_plan_sizes()[0] == pytest.approx(0.01)

    # Every study folder holds the runs its replay needs: step 1 copied
    # the run it took from step 0, so step 0's folder can go.
    assert "copied into this study" in log
    import shutil
    shutil.rmtree(steps["ms"]["folder"])
    n_asked = len(answers.asked)
    tab.forget_results()
    assert tab.open_study(steps["mesh"]["folder"])
    _wait(qapp, tab, _idle(tab))
    assert len(answers.asked) == n_asked
    assert steps["mesh"]["status"] == "done"
    assert len(world.launched) == n

    # Opening an unfinished copy of the step-2 study does not replace the
    # result of step 2, and puts the panel back when not resumed.
    dom = Path(steps["domain"]["folder"])
    copy = dom.parent / (dom.name + "_copy")
    shutil.copytree(dom, copy)
    for meta in sorted(copy.glob("domainsizing_run*.meta.json"))[-2:]:
        (copy / (meta.name[:-len(".meta.json")] + ".sta")).write_text("x")
    panel = [le.text() for le in tab._opt_line_edits()]
    tab.le_grid_step.setText("")            # (what the copy was made with)
    answers.next = "later"
    tab.open_study(copy)
    _wait(qapp, tab, _idle(tab))
    assert "is not finished" in answers.asked[-1][1]
    assert "keeps its current result" in answers.asked[-1][1]
    assert Path(steps["domain"]["folder"]) == dom
    assert steps["domain"]["status"] == "done"
    assert [le.text() for le in tab._opt_line_edits()] == panel
    for step in st.STEPS:
        assert tab._step_state(step)[0] == "done"
    assert len(world.launched) == n


def test_ms_study_extends_the_values_until_one_fails(qapp, world):
    tab = world.tab
    tab.le_ms_values.setText("125, 250")
    tab.le_ms_elem.setText("0.02")
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    rec = tab.cfg.optimization.steps["ms"]
    assert rec["status"] == "done", tab.log.toPlainText()[-3000:]
    assert rec["value"] == 1000.0
    assert rec["values"][:4] == [125.0, 250.0, 500.0, 1000.0]
    cfg = json.loads((Path(rec["folder"]) / "config.json").read_text())
    assert cfg["parameters"]["ms_values"][2] == 500.0
    # 125, 250 then 500, 1000, 2000: each ms run once
    ms_jobs = [j for j in world.launched if j.startswith("ms_")]
    assert len(ms_jobs) == 5


# ---------------------------------------------------------------------------
# Failures, cancels and guards
# ---------------------------------------------------------------------------
def _ms_of(params):
    s = params["step"]
    return (float(s["mass_scaling_factor_eulerian"])
            if s["mass_scaling_enabled"] else 1.0)


def test_a_run_that_cannot_start_is_not_a_result(qapp, world, answers):
    tab = world.tab
    world.behaviour["fn"] = lambda p: "raise" if _ms_of(p) >= 2000 else "ok"
    tab.le_ms_values.setText("500, 1000, 2000, 4000")
    tab.le_ms_elem.setText("0.02")
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    rec = tab.cfg.optimization.steps["ms"]
    assert rec["status"] == "interrupted", tab.log.toPlainText()[-2000:]
    assert "could not be started" in rec["message"]
    assert tab.cfg.step.mass_scaling_enabled is False     # nothing applied
    # Once Abaqus starts again, the study is resumed where it stopped.
    world.behaviour["fn"] = None
    n = len(world.launched)
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    assert answers.asked[-1][2] == ["resume", "new", "cancel"]
    rec = tab.cfg.optimization.steps["ms"]
    assert (rec["status"], rec["value"]) == ("done", 1000.0)
    assert len(world.launched) == n + 1                   # only ms = 2000


def test_an_analysis_abaqus_stopped_is_replayed_as_failed(qapp, world,
                                                          answers):
    tab = world.tab
    world.behaviour["fn"] = lambda p: "abort" if _ms_of(p) >= 2000 else "ok"
    tab.le_ms_values.setText("500, 1000, 2000")
    tab.le_ms_elem.setText("0.02")
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    rec = tab.cfg.optimization.steps["ms"]
    assert (rec["status"], rec["value"]) == ("done", 1000.0)
    folder = Path(rec["folder"])
    assert list(folder.glob("ms_run*.failed.json"))
    # Loading the study meets the same failure: no "not finished".
    n, n_asked = len(world.launched), len(answers.asked)
    tab.forget_results()
    assert tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert len(answers.asked) == n_asked
    assert len(world.launched) == n
    rec = tab.cfg.optimization.steps["ms"]
    assert (rec["status"], rec["value"]) == ("done", 1000.0)
    assert "counted as a failed run again" in tab.log.toPlainText()


def test_a_folder_is_judged_on_its_own_runs(qapp, world, answers):
    tab = world.tab
    tab._on_run_domain_independence()                    # model B, folder X
    _wait(qapp, tab, _idle(tab))
    assert tab.cfg.optimization.steps["domain"]["status"] == "done"
    tab.cfg.interaction.friction_coeff += 0.1            # model A
    tab.le_ms_values.setText("500, 1000, 2000")
    tab.le_ms_elem.setText("0.02")
    tab._on_run_ms_independence()                        # folder Y
    _wait(qapp, tab, _idle(tab))
    rec_ms = dict(tab.cfg.optimization.steps["ms"])
    assert rec_ms["status"] == "done"
    tab.cfg.interaction.friction_coeff -= 0.1            # model B again
    n_asked = len(answers.asked)
    tab.forget_results()
    tab.open_study(rec_ms["folder"])
    _wait(qapp, tab, _idle(tab))
    assert len(answers.asked) == n_asked                 # no "resume?"
    assert "friction_coeff" in answers.informed[-1]
    assert tab.cfg.optimization.steps["ms"] == rec_ms


def test_a_cancel_at_the_end_never_extends_the_ms_values(qapp, world,
                                                         monkeypatch):
    tab = world.tab
    tab.le_ms_values.setText("125, 250")
    tab.le_ms_elem.setText("0.02")
    extended = []
    monkeypatch.setattr(tab, "_extend_ms",
                        lambda act: extended.append(act) or True)
    done = tab._on_ms_done

    def cancel_then_done(res):
        tab._cancel_evt.set()             # Cancel clicked at the last moment
        done(res)
    monkeypatch.setattr(tab, "_on_ms_done", cancel_then_done)
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    assert extended == []
    assert tab.cfg.optimization.steps["ms"]["status"] == "interrupted"


def test_the_sampling_step_set_in_the_tab_is_a_setting(tab):
    k0 = tab._keys()[1]
    tab.le_grid_step.setText("0.002")
    assert tab._keys()[1] != k0
    tab.le_grid_step.setText("")
    assert tab._keys()[1] == k0
    # blank follows the element size, which h* changes: no new key
    tab.cfg.elem_size = 0.0123
    assert tab._keys()[1] == k0


def test_a_zero_eps_is_left_out_of_the_settings():
    a = ss.comparison_settings((0, 1, 0, 1), {"T": 0.0, "Vx": 10, "Ff": -1},
                               (0.3, 1.0), 0.05, 0.05)
    b = ss.comparison_settings((0, 1, 0, 1), {"Vx": 10.0}, (0.3, 1.0),
                               0.05, 0.05)
    assert a == b and a["eps"] == {"Vx": 10.0}


def test_the_pipeline_never_reruns_a_step_in_a_loop(tab):
    mkey, _skey = tab._keys()
    rec = st.make_record("done", 1000.0, "/f", mkey, "other settings", {})
    tab.cfg.optimization.steps["ms"] = rec
    tab._pipeline, tab._pipeline_done = True, set()
    tab._pipeline_after("ms", rec)
    assert tab._pipeline is False
    assert "does not count as done" in tab.log.toPlainText()


def test_a_failed_record_of_another_model_is_not_called_done(tab):
    _record(tab, "ms", None, status="failed", message="first comparison")
    tab.cfg.interaction.friction_coeff = 0.42
    tab._refresh_step_status()
    assert tab._step_status["ms"].text().startswith(
        "Failed for another model")
    out = st.missing_prerequisites(tab._steps(), "mesh", *tab._keys(),
                                   tab._model_values())
    assert out == ["Step 0 (mass scaling) is failed for another model or "
                   "other settings."]


def test_outcome_of_cancelled_or_failed_studies():
    out = OptimizationTab._outcome(
        "domain", SimpleNamespace(status="partial"), None, True)
    assert out[0] == "interrupted"
    out = OptimizationTab._outcome("mesh", None, "no bundle", False,
                                   ["GCI_run001"])
    assert out[0] == "failed" and "GCI_run001" in out[2]


def test_home_keeps_the_overlays_in_view(tab):
    tab._max["l_wp"].setText("5")
    tab._draw_preview()
    tab.preview._ax.set_xlim(-0.1, 0.1)                  # zoomed in
    tab.preview.fit_view()                               # Home button
    assert tab.preview._ax.get_xlim()[0] < -5


def test_odd_numbers_never_break_the_preview(tab):
    tab._max["l_wp"].setText("inf")
    tab.le_grid_step.setText("1e-310")
    tab.le_zoi["xmin"].setText("nan")
    tab._draw_preview()                                  # no exception
    assert "l_wp" not in tab.caps()
    assert all(map(math.isfinite, tab.zoi()))


def test_no_other_profile_while_a_study_runs(qapp, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from gui.main import MainWindow
    w = MainWindow()
    w._dirty = False
    told = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: told.append(a[1])))

    def no_dialog(*a, **k):
        raise AssertionError("the file dialog must not open")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(no_dialog))
    cfg = w.cfg
    w.optimization_tab._pipeline = True              # between two steps
    w.file_open()
    w.file_new()
    assert w.cfg is cfg
    assert told == ["Open profile", "New profile"]
    w.optimization_tab._pipeline = False


def _drop_last_runs(folder, prefix, n=2):
    """Make the last `n` runs of a study folder unfinished (their .sta
    removed: a hard-linked copy elsewhere keeps its own)."""
    jobs = sorted(p.name[:-len(".meta.json")]
                  for p in Path(folder).glob("%s_run*.meta.json" % prefix))
    for job in jobs[-n:]:
        (Path(folder) / (job + ".sta")).unlink()
    return jobs[-n:]


def test_a_stopped_resume_keeps_the_result_of_record(qapp, world, answers):
    tab = world.tab
    tab._on_run_domain_independence()
    _wait(qapp, tab, _idle(tab))
    rec = dict(tab.cfg.optimization.steps["domain"])
    assert rec["status"] == "done"
    folder = Path(rec["folder"])
    _drop_last_runs(folder, "domainsizing")
    exports = {p.name: p.read_bytes() for p in folder.iterdir()
               if p.suffix in (".csv", ".md", ".txt")}
    # "Resume now", but Abaqus cannot be started: the result stays, and
    # the export files of that result are not rewritten.
    world.behaviour["fn"] = lambda p: "raise"
    answers.next = "resume"
    tab.forget_results()
    tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert "could not be started" in tab.log.toPlainText()
    assert tab.cfg.optimization.steps["domain"] == rec
    assert {p.name: p.read_bytes() for p in folder.iterdir()
            if p.suffix in (".csv", ".md", ".txt")} == exports
    # Same with a Cancel during the resume.
    world.behaviour["fn"] = lambda p: tab._cancel_evt.set() or "ok"
    answers.next = "resume"
    tab.forget_results()
    tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert tab.cfg.optimization.steps["domain"] == rec
    # Resumed to the end, the folder is complete again.
    world.behaviour["fn"] = None
    answers.next = "resume"
    tab.forget_results()
    tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    now = tab.cfg.optimization.steps["domain"]
    assert (now["status"], now["value"]) == ("done", rec["value"])


def test_a_resume_that_does_not_start_puts_the_panel_back(
        qapp, world, answers, monkeypatch):
    tab = world.tab
    tab._q_eps["T"].setText("2")
    tab._on_run_domain_independence()                    # folder B, T = 2
    _wait(qapp, tab, _idle(tab))
    folder_b = Path(tab.cfg.optimization.steps["domain"]["folder"])
    _drop_last_runs(folder_b, "domainsizing")
    tab._q_eps["T"].setText("1")
    _record(tab, "domain", [0.1, 0.1, 0.1, 0.1], {"ms": 1.0, "h": 0.01})
    rec_a = dict(tab.cfg.optimization.steps["domain"])
    # The launch check refuses (Abaqus path not set...).
    monkeypatch.setattr(tab, "_validate_launch", lambda *a: None)
    answers.next = "resume"
    tab.forget_results()
    tab.open_study(folder_b)
    _wait(qapp, tab, _idle(tab))
    assert "keeps its current result" in answers.asked[-1][1]
    assert tab._q_eps["T"].text() == "1"
    assert tab.cfg.optimization.steps["domain"] == rec_a
    # It starts, but no run can be made: same.
    monkeypatch.setattr(tab, "_validate_launch",
                        lambda *a: (tab._prefs_getter(), world.wd, 2))
    world.behaviour["fn"] = lambda p: "raise"
    answers.next = "resume"
    tab.forget_results()
    tab.open_study(folder_b)
    _wait(qapp, tab, _idle(tab))
    assert "could not be started" in tab.log.toPlainText()
    assert tab._q_eps["T"].text() == "1"
    assert tab.cfg.optimization.steps["domain"] == rec_a


def test_a_folder_of_failed_runs_is_judged_on_them(qapp, world, answers):
    tab = world.tab
    world.behaviour["fn"] = lambda p: "abort"
    tab.le_ms_values.setText("500, 1000")
    tab.le_ms_elem.setText("0.02")
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    rec = tab.cfg.optimization.steps["ms"]
    assert rec["status"] == "failed", tab.log.toPlainText()[-2000:]
    folder = Path(rec["folder"])
    assert list(folder.glob("ms_run*.failed.json"))
    assert not list(folder.glob("ms_run*.meta.json"))
    tab.cfg.interaction.friction_coeff += 0.1
    n_asked = len(answers.asked)
    tab.forget_results()
    tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert len(answers.asked) == n_asked                 # no "resume?"
    assert "friction_coeff" in answers.informed[-1]


def test_the_checks_sample_like_the_domain_study(qapp, world):
    tab = world.tab
    tab._on_run_domain_independence()                    # sampling step blank
    _wait(qapp, tab, _idle(tab))
    tab.le_grid_step.setText("0.003")
    spec = tab._checks_spec(tab._last_domain_result)
    assert spec["grid_step_set"] is None
    assert ss.settings_of_spec(spec)["grid"] is None


def test_an_older_folder_with_a_blank_sampling_step():
    # No "grid_step_set": the step used equals the element size a blank
    # field gave, so the field is taken as blank.
    assert ss.grid_set_of_spec({"elem_size": 0.004,
                                "grid_step": 0.004}) is None
    assert ss.grid_set_of_spec({"finest_elem_size": 0.0005,
                                "grid_step": 0.0005}) is None
    assert ss.grid_set_of_spec({"h_star": 0.002, "gci_plan": {
        "grid_step": 0.002}}) is None
    assert ss.grid_set_of_spec({"elem_size": 0.004,
                                "grid_step": 0.002}) == 0.002
    # An older ms or GCI folder does not record the blank-field value.
    assert ss.grid_set_of_spec({"ms_values": [250, 500], "elem_size": 0.08,
                                "grid_step": 0.01}) is None
    assert ss.grid_set_of_spec({"grid_step_set": None,
                                "grid_step": 0.002}) is None


def test_quitting_during_a_study_asks_and_stops_it(qapp, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox
    from gui.main import MainWindow
    w = MainWindow()
    w._dirty = False
    answer = {"v": QMessageBox.No}
    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: asked.append(a[1]) or answer["v"]))
    stopped = []
    monkeypatch.setattr(w.optimization_tab, "shutdown",
                        lambda *a: stopped.append(1) or True)
    w.optimization_tab._pipeline = True              # between two steps
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert not ev.isAccepted() and stopped == []
    assert asked == ["Model tab study running"]
    answer["v"] = QMessageBox.Yes
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted() and stopped == [1]
    w.optimization_tab._pipeline = False


def test_shutdown_stops_the_study_without_recording_it(qapp, world):
    import threading
    tab = world.tab
    entered, release = threading.Event(), threading.Event()

    def slow(params):
        entered.set()
        release.wait(5)
        return "ok"
    world.behaviour["fn"] = slow
    tab._on_run_domain_independence()
    assert entered.wait(10)
    threading.Timer(0.3, release.set).start()
    assert tab.shutdown(10000)
    for _ in range(50):
        qapp.processEvents()
    assert "domain" not in tab.cfg.optimization.steps
    assert not tab.is_running()


def test_a_cancelled_rerun_keeps_the_done_result(qapp, world, answers):
    tab = world.tab
    tab.le_ms_values.setText("500, 1000, 2000")
    tab.le_ms_elem.setText("0.02")
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    rec = tab.cfg.optimization.steps["ms"]
    assert (rec["status"], rec["value"]) == ("done", 1000.0)
    _record(tab, "mesh", 0.01, {"ms": 1000.0})
    # Step 0 again with one more value, cancelled at its first new run.
    tab.le_ms_values.setText("250, 500, 1000, 2000")

    def cancel(params):                  # what the Cancel button sets
        tab._ms_worker.cancel()
        tab._cancel_evt.set()
        return "ok"
    world.behaviour["fn"] = cancel
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    now = tab.cfg.optimization.steps["ms"]
    assert (now["status"], now["value"]) == ("done", 1000.0)
    assert now["attempt"]["status"] == "interrupted"
    assert now["attempt"]["folder"] != now["folder"]
    assert tab._step_state("mesh")[0] == "done"        # not "upstream"
    assert "A later study" in tab._step_status["ms"].text()
    assert now["attempt"]["folder"] in tab._known_folders()
    # The step's button offers to resume that study; it ends with a result.
    world.behaviour["fn"] = None
    answers.next = "resume"
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    assert "keeps its result (ms = 1000)" in answers.asked[-1][1]
    now = tab.cfg.optimization.steps["ms"]
    assert (now["status"], now["value"]) == ("done", 1000.0)
    assert "attempt" not in now
    assert now["values"][:4] == [250.0, 500.0, 1000.0, 2000.0]


def test_nothing_starts_between_two_pipeline_steps(tab, answers):
    tab._pipeline = True
    tab._sync_run_buttons()
    assert not tab.btn_ms.isEnabled() and not tab.btn_open.isEnabled()
    assert tab.btn_cancel.isEnabled()
    tab._on_run_ms_independence()
    tab._on_run_all()
    tab._on_open_study()
    assert answers.asked == []
    tab._is_busy = True                     # a study runs
    assert tab._start_ms({}, "/x", None, 1) is False
    assert "not started" in tab.log.toPlainText()
    tab._is_busy = False
    tab._pipeline_stop("test")
    assert tab.btn_ms.isEnabled() and not tab.btn_cancel.isEnabled()


def test_the_safe_answer_is_the_default_button(tab, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    seen = []
    monkeypatch.setattr(QMessageBox, "exec",
                        lambda box: seen.append(box.defaultButton().text()))
    from gui.tabs.model_steps import ModelStepsMixin
    ask = ModelStepsMixin._ask           # (the module's fixture patches it)
    ask(tab, "t", "x", [("run", "Run anyway"), ("cancel", "Cancel")],
        default="cancel")
    ask(tab, "t", "x", [("resume", "Resume"), ("new", "Start a new study")])
    assert seen == ["Cancel", "Resume"]


def test_domain_sizes_written_never_lose_an_element():
    from decimal import Decimal
    for h in (0.000625, 0.00141421, 0.001125):
        for k in range(1, 4000):
            v = st.round6_up(k * h)
            assert Decimal(str(v)) // Decimal(str(h)) == k, (h, k, v)
            assert v == float("%g" % v)              # what the tab shows


def test_plain_words_in_step_messages(tab):
    bad = SimpleNamespace(status="rejected", checks=[SimpleNamespace(
        name="domain_combined", passed=False, details={"action": "redo"},
        conclusion="")])
    msg = OptimizationTab._outcome("checks", bad, None, False)[2]
    assert "domain_combined" not in msg and "grown on all sides" in msg
    _record(tab, "ms", None, status="interrupted",
            message="a run could not be made (agent down)")
    tab._refresh_step_status()
    assert "agent down" in tab._step_status["ms"].text()


def test_h_star_rounded_by_the_tabs_is_found_in_the_plan():
    from gui.sensitivity.interaction_checks import mesh_domain_check
    gci = SimpleNamespace(sizes=[0.001, 0.001 * math.sqrt(2), 0.002],
                          recommended_size=None, in_asymptotic_range=True,
                          per_quantity={}, scalars={})
    res = mesh_domain_check(gci, float("%g" % (0.001 * math.sqrt(2))), {})
    assert "not in the GCI plan" not in res.conclusion
    assert res.details["h_star"] == 0.00141421


# ---------------------------------------------------------------------------
# Results kept, folders, quit (third review)
# ---------------------------------------------------------------------------
def _cancel_study(tab, attr):
    """A behaviour that clicks Cancel during the next launched run."""
    def cancel(params):
        getattr(tab, attr).cancel()
        tab._cancel_evt.set()
        return "ok"
    return cancel


def test_an_extended_resume_of_a_read_back_keeps_the_result(qapp, world,
                                                            answers):
    tab = world.tab
    # An ms folder made with T eps = 2, its ms = 250 run missing.
    tab._q_eps["T"].setText("2")
    world.behaviour["fn"] = lambda p: "raise" if _ms_of(p) >= 250 else "ok"
    tab.le_ms_values.setText("125, 250")
    tab.le_ms_elem.setText("0.02")
    tab._on_run_ms_independence()
    _wait(qapp, tab, _idle(tab))
    folder = Path(tab.cfg.optimization.steps["ms"]["folder"])
    # A valid result of step 0 for the panel (T eps = 1).
    tab._q_eps["T"].setText("1")
    _record(tab, "ms", 1000.0, {"h": 0.02})
    rec = dict(tab.cfg.optimization.steps["ms"])
    # Open the folder, resume it: every comparison passes, the study is
    # extended, and the extension is cancelled.

    def cancel_in_extension(p):
        if _ms_of(p) >= 500:
            tab._ms_worker.cancel()
            tab._cancel_evt.set()
        return "ok"
    world.behaviour["fn"] = cancel_in_extension
    answers.next = "resume"
    tab.forget_results()
    tab.open_study(folder)
    _wait(qapp, tab, _idle(tab))
    assert "not bracketed yet" in tab.log.toPlainText()
    assert tab._q_eps["T"].text() == "1"
    assert tab.cfg.optimization.steps["ms"] == rec


def test_reading_back_the_study_of_record_keeps_the_later_study(
        qapp, world, answers):
    tab = world.tab
    tab._on_run_domain_independence()
    _wait(qapp, tab, _idle(tab))
    tab._dom_spins["dom_step_elems"].setValue(3)
    world.behaviour["fn"] = _cancel_study(tab, "_di_worker")
    tab._on_run_domain_independence()
    _wait(qapp, tab, _idle(tab))
    world.behaviour["fn"] = None
    now = tab.cfg.optimization.steps["domain"]
    att = dict(now["attempt"])
    assert att["status"] == "interrupted" and att["folder"] != now["folder"]
    tab.forget_results()
    assert tab.open_study(now["folder"])
    _wait(qapp, tab, _idle(tab))
    after = tab.cfg.optimization.steps["domain"]
    assert after["status"] == "done" and after.get("attempt") == att


def test_two_studies_in_the_same_second_get_two_folders(tmp_path,
                                                        monkeypatch):
    from datetime import datetime
    import gui.core.run_output as ro
    when = datetime(2026, 10, 9, 12, 0, 0)
    a = ro.create_study_dir(tmp_path, "p", "GCI", {"a": 1}, when)
    b = ro.create_study_dir(tmp_path, "p", "GCI", {"b": 2}, when)
    assert a != b and b.name == a.name + "_2"
    assert json.loads((a / "config.json").read_text())["parameters"] == \
        {"a": 1}


def test_the_checks_use_the_result_the_step_keeps(qapp, world, answers,
                                                  monkeypatch):
    tab = world.tab
    tab._on_run_domain_independence()
    _wait(qapp, tab, _idle(tab))
    rec = dict(tab.cfg.optimization.steps["domain"])
    assert tab._step_state("domain")[0] == "prereq"  # steps 0, 1 not done
    # Step 2 again with another growth: it does not converge (failed).
    tab._dom_spins["dom_step_elems"].setValue(1)
    tab._dom_spins["dom_m_ratios"].setValue(1)
    tab._dom_spins["dom_n_max"].setValue(2)
    tab._on_run_domain_independence()
    _wait(qapp, tab, _idle(tab))
    now = tab.cfg.optimization.steps["domain"]
    assert now["folder"] == rec["folder"]
    assert now["attempt"]["status"] == "failed"
    started = []
    monkeypatch.setattr(tab, "_start_checks",
                        lambda spec, study, folder, *a, **k:
                        started.append((spec, Path(folder))) or False)
    tab._on_run_interaction_checks()                 # "Run anyway"
    _wait(qapp, tab, lambda: not tab._is_busy and tab._active is None
          and started, timeout=30)
    spec, folder = started[-1]
    assert rc.same_folder(folder, rec["folder"])
    assert spec["d_star"] == rec["value"]


def test_a_cancelled_rerun_of_the_checks_keeps_their_files(
        qapp, world, answers, monkeypatch):
    monkeypatch.setattr(_ModelBundle, "LAM", 0.005)
    tab = world.tab
    tab.le_zoi["ymin"].setText("-0.09")
    tab.le_zoi["ymax"].setText("-0.01")
    tab.sp_gci_n.setValue(3)
    tab._on_run_all()
    _wait(qapp, tab, lambda: not tab._pipeline and not tab._is_busy
          and tab._active is None)
    rec = dict(tab.cfg.optimization.steps["checks"])
    assert rec["status"] == "done"
    dom = Path(rec["folder"])
    files = {p.name: p.read_bytes() for p in dom.iterdir()
             if p.suffix in (".csv", ".json") and not p.name.startswith(
                 ("domainsizing_run", "checks_run"))}
    assert "checks.csv" in files and "checks_config.json" in files
    # The checks again with another ms (set by hand), cancelled.
    tab.cfg.step.mass_scaling_factor = 2000.0
    world.behaviour["fn"] = _cancel_study(tab, "_checks_worker")
    tab._on_run_interaction_checks()                 # "Run anyway"
    _wait(qapp, tab, _idle(tab))
    world.behaviour["fn"] = None
    now = tab.cfg.optimization.steps["checks"]
    assert (now["status"], now["value"]) == ("done", "accepted")
    assert now["attempt"]["status"] == "interrupted"
    assert "resume" not in tab._step_status["checks"].text()
    assert {p.name: p.read_bytes() for p in dom.iterdir()
            if p.name in files} == files
    # The folder still reads back the accepted checks, with no question.
    tab.cfg.step.mass_scaling_factor = 1000.0
    n = len(answers.asked)
    tab.forget_results()
    tab.open_study(dom)
    _wait(qapp, tab, _idle(tab))
    for _ in range(100):                     # the checks read back next
        qapp.processEvents()
    _wait(qapp, tab, _idle(tab))
    assert len(answers.asked) == n
    assert tab.cfg.optimization.steps["checks"]["status"] == "done"


def test_a_declined_quit_stops_nothing(qapp, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox
    from gui.main import MainWindow
    w = MainWindow()
    w._dirty = False
    asked = []

    def question(parent, title, *a, **k):
        asked.append(title)
        return (QMessageBox.Yes if title == "Sensitivity campaign running"
                else QMessageBox.No)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    stopped = []
    monkeypatch.setattr(w.sensitivity_tab, "is_running", lambda: True)
    monkeypatch.setattr(w.sensitivity_tab, "shutdown",
                        lambda *a: stopped.append("sens") or True)
    monkeypatch.setattr(w.optimization_tab, "shutdown",
                        lambda *a: stopped.append("model") or True)
    w.optimization_tab._pipeline = True
    ev = QCloseEvent()
    w.closeEvent(ev)
    w.optimization_tab._pipeline = False
    assert asked == ["Sensitivity campaign running",
                     "Model tab study running"]
    assert not ev.isAccepted() and stopped == []


def test_a_declined_start_leaves_the_plan_as_it_was(tab, answers,
                                                    monkeypatch):
    monkeypatch.setattr(tab, "_validate_launch", lambda *a: None)
    answers.next = "cancel"                 # order guard: Cancel
    tab._on_run_mesh_gci()
    assert answers.asked[-1][0] == "Order of the steps"
    assert tab.le_gci_finest.text() == ""


def test_dims_shown_rounded_down_are_not_the_result(tab):
    h = 0.00141421
    tab.cfg.elem_size = h
    d_star = [37 * h, 11 * h, 53 * h, 7 * h]
    _record(tab, "domain", d_star, {"ms": 1.0, "h": h})
    g = tab.cfg.euler_geometry
    g.h_wp, g.h_void, g.l_wp, g.l_void = (float("%g" % v) for v in d_star)
    assert not st.in_model("domain", tab._model_values()["dims"], d_star)
    tab._apply_step_value("domain", d_star)
    assert st.in_model("domain", tab._model_values()["dims"], d_star)
    n = [round(e / h) for e in tab.cfg.effective_euler_dims()]
    assert n == [37, 11, 53, 7]
