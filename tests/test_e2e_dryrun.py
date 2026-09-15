# -*- coding: utf-8 -*-
"""
End-to-end dry-run of the sensitivity pipeline WITHOUT Abaqus.

These tests close the gap between the mock-based unit tests and the real
Abaqus path. They exercise:

  * eulerian_instance / Field-QoI SSD against a *real* ResultsBundle
    (built by fake_builder) — this is what would have caught the
    property-vs-method regression that the method-style mock hid;
  * the real SensitivityRunWorker driving a real subprocess (a fake
    Abaqus 'command', tests/abaqus_stub.py) that writes a real .sta and a
    real (.json + .npz) bundle, then loading it back through
    ResultsBundle — i.e. the launch + stream + reload contract;
  * a failed run surfacing as a runDone(ok=False) + a recorded failure;
  * Cancel terminating the whole process tree (POSIX-only check);
  * the .sta parser on a stub-written file;
  * the live wall-clock estimate and the results table in SensitivityTab;
  * the CSV export of the sensitivity table / field-SSD ranking.

Run:  QT_QPA_PLATFORM=offscreen python -m pytest tests/test_e2e_dryrun.py -q
"""
from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.core.model_config import ModelConfig
from gui.core.preferences import Preferences
from gui.core.sta_parser import parse_sta
from gui.results.fake_builder import build_fake_results
from gui.results.reader import ResultsBundle
from gui.results.qoi import QoISpec
from gui.sensitivity import param_registry as pr
from gui.sensitivity import jacobian_plan as jac
from gui.sensitivity import runner_core as rc
from gui.sensitivity import export_results as xr
from gui.sensitivity.run_worker import SensitivityRunWorker


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _stub_launcher(tmp_path: Path, env_lines: str = "") -> Path:
    """Write an executable shell launcher that runs the stub by file path
    (so the cwd stays the workdir and the bundle lands there)."""
    launcher = tmp_path / "abaqus_stub.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n%s"
        'exec "%s" "%s" "$@"\n'
        % (env_lines, sys.executable, _ROOT / "tests" / "abaqus_stub.py"),
        encoding="ascii")
    launcher.chmod(0o755)
    return launcher


def _jacobian_1param(scheme="forward"):
    cfg = ModelConfig()
    sE = pr.spec_for("euler_material.E")
    x0 = pr.get_display(cfg, sE, "C")
    plan = jac.build_plan([(sE, x0, 0.1 * x0, False)], scheme=scheme)
    return cfg, plan


# ---------------------------------------------------------------------------
# A. Real bundle: eulerian_instance + Field QoI (the regression guard)
# ---------------------------------------------------------------------------
def test_eulerian_instance_on_real_bundle(tmp_path):
    j, n = build_fake_results(tmp_path / "b.results.npz",
                              n_frames=4, n_grid_x=6, n_grid_y=4)
    b = ResultsBundle.load(n)
    # The real ResultsBundle exposes instance_names as a PROPERTY (a list).
    assert not callable(b.instance_names)
    assert rc.eulerian_instance(b) == "Euler"


def test_field_qoi_on_real_bundles(tmp_path):
    """Drive run_plan with a solve_fn returning real ResultsBundles whose
    fields differ with the input (field_scale), and check the Field-QoI
    columns are present and finite/positive — the path that silently
    vanished before the fix."""
    cfg, plan = _jacobian_1param("central")  # base, +, -  -> 3 runs

    def solve_fn(c, i):
        sE = pr.spec_for("euler_material.E")
        e = pr.get_display(c, sE, "C")
        out = tmp_path / ("run%03d.results.npz" % i)
        _, npz = build_fake_results(out, cfg=None, n_frames=4,
                                    n_grid_x=6, n_grid_y=4,
                                    job_name="run%03d" % i,
                                    field_scale=1.0 + 1e-3 * e)
        return ResultsBundle.load(npz)

    res = rc.run_plan(plan, "jacobian", [], solve_fn, cfg,
                      field_vars=["EVF", "TEMP"])
    assert "EVF [field]" in res.qoi_ids
    assert "TEMP [field]" in res.qoi_ids
    path = plan.param_paths[0]
    sEVF = res.analyses["EVF [field]"][path]["sensitivity"]
    sTEMP = res.analyses["TEMP [field]"][path]["sensitivity"]
    assert np.isfinite(sEVF) and sEVF >= 0.0
    assert np.isfinite(sTEMP) and sTEMP >= 0.0


# ---------------------------------------------------------------------------
# C. Worker + real subprocess (the launch / stream / reload contract)
# ---------------------------------------------------------------------------
def test_worker_dryrun_with_stub_subprocess(qapp, tmp_path):
    cfg, plan = _jacobian_1param("forward")   # 2 runs
    launcher = _stub_launcher(tmp_path)
    workdir = tmp_path / "wd"
    workdir.mkdir()

    worker = SensitivityRunWorker(
        plan, "jacobian",
        qoi_specs=[QoISpec("Fx_mean", "Fx_mean", "N",
                           lambda b, inst, w: float(
                               np.nanmean(np.abs(b.history("RF1_RP")))))],
        base_cfg=cfg,
        abaqus_cmd=str(launcher), abaqus_script="ignored.py",
        workdir=str(workdir), cpus=1, field_vars=["EVF", "TEMP"])

    logs, done, finished = [], [], {}
    worker.log.connect(logs.append)
    worker.runDone.connect(lambda i, ok: done.append((i, ok)))
    worker.finished.connect(lambda r: finished.__setitem__("r", r))
    worker.run()   # synchronous (no QThread): runs both profiles inline

    res = finished["r"]
    text = "".join(logs)
    assert "[STUB] COMPLETED" in text           # subprocess output streamed
    assert all(ok for _, ok in done)            # every run reported ok
    assert not res.failures
    assert res.Y.shape == (plan.n_runs, 1)
    assert np.all(np.isfinite(res.Y))           # Fx_mean computed from .npz
    assert "EVF [field]" in res.qoi_ids         # field QoI on real bundles
    # the bundle the worker reloaded actually exists on disk
    assert (workdir / "sens_run000.results.npz").exists()
    assert (workdir / "sens_run000.sta").exists()


def test_worker_dryrun_failed_run_is_reported(qapp, tmp_path):
    cfg, plan = _jacobian_1param("forward")
    launcher = _stub_launcher(tmp_path, env_lines="export STUB_FAIL=1\n")
    workdir = tmp_path / "wd_fail"
    workdir.mkdir()

    worker = SensitivityRunWorker(
        plan, "jacobian", qoi_specs=[], base_cfg=cfg,
        abaqus_cmd=str(launcher), abaqus_script="ignored.py",
        workdir=str(workdir), cpus=1)
    done, finished = [], {}
    worker.runDone.connect(lambda i, ok: done.append((i, ok)))
    worker.finished.connect(lambda r: finished.__setitem__("r", r))
    worker.run()

    assert done and all(ok is False for _, ok in done)
    assert finished["r"].failures            # recorded as failures


# ---------------------------------------------------------------------------
# E. Cancel terminates the whole process tree (POSIX-only check)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.name == "nt",
                    reason="taskkill /T path not testable here; verify on PC")
def test_cancel_kills_process_tree(qapp, tmp_path):
    cfg, plan = _jacobian_1param("forward")
    pidfile = tmp_path / "child.pid"
    launcher = _stub_launcher(
        tmp_path,
        env_lines=("export STUB_SPAWN_CHILD=1\nexport STUB_SLEEP=30\n"
                   'export STUB_CHILD_PIDFILE="%s"\n' % pidfile))
    workdir = tmp_path / "wd_cancel"
    workdir.mkdir()

    worker = SensitivityRunWorker(
        plan, "jacobian", qoi_specs=[], base_cfg=cfg,
        abaqus_cmd=str(launcher), abaqus_script="ignored.py",
        workdir=str(workdir), cpus=1)

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    # Wait until the stub has spawned its child and recorded the PID.
    for _ in range(100):
        if pidfile.exists():
            break
        time.sleep(0.1)
    assert pidfile.exists(), "stub never spawned its child"
    child_pid = int(pidfile.read_text())
    # The child must be alive right now.
    os.kill(child_pid, 0)

    worker.cancel()
    t.join(timeout=10)
    assert not t.is_alive(), "worker did not return after cancel"

    # The child must no longer be running. In this container PID 1 does not
    # reap orphans, so a killed child lingers as a zombie (state 'Z') and
    # its PID is still listed — "killed" means gone OR zombie, not "alive".
    def _dead_or_zombie(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return True                      # gone entirely
        try:
            stat = Path("/proc/%d/stat" % pid).read_text()
            return stat.split(")")[-1].split()[0] == "Z"
        except Exception:
            return False                     # exists and not a zombie
    time.sleep(0.5)
    assert _dead_or_zombie(child_pid), "Cancel left the child process running"


# ---------------------------------------------------------------------------
# F. .sta parser on a stub-written file
# ---------------------------------------------------------------------------
def test_sta_parser_on_stub_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tests.abaqus_stub import _write_sta
    _write_sta("job", n_frames=8, sim_time=5e-4)
    snap = parse_sta(tmp_path / "job.sta")
    assert snap.is_ready()
    assert snap.frame_current == 8 and snap.frame_total == 8
    assert snap.fraction() == pytest.approx(1.0)
    assert snap.wall_time is not None and snap.inc_number is not None


# ---------------------------------------------------------------------------
# G. SensitivityTab: live estimate + results table (headless)
# ---------------------------------------------------------------------------
def test_tab_live_estimate_and_table(qapp, tmp_path):
    from gui.tabs.sensitivity_tab import SensitivityTab
    tab = SensitivityTab(ModelConfig(),
                         prefs_getter=lambda: Preferences(),
                         cpus_getter=lambda: 1)

    # A partial .sta (half done) so the estimate extrapolates a total.
    sta = tmp_path / "sensitivity_run000.sta"
    sta.write_text(
        "  Output Field Frame Number    4, of    8, at step time 2.500E-04\n"
        "   1234  2.500E-04 2.500E-04  00:00:30 5.6E-10       16760  9.0E-6  1.6E-1\n",
        encoding="latin-1")
    tab._run_workdir = tmp_path
    tab._running_index = 0
    tab._run_total = 2
    tab._run_durations = []
    tab._failed_live = []
    tab._run_clock0 = time.monotonic()
    tab.progress.setRange(0, 2)
    tab.progress.setValue(0)
    tab._poll_sta()
    msg = tab.status.text()
    assert "frame 4/8" in msg and "/run" in msg and "remaining" in msg \
        and "total" in msg
    # Explicit per-frame / per-run / total arithmetic: 30 s over 4 frames
    # -> 7.5 s/frame ; 8 frames/run -> 60 s/run ; 2 runs -> 120 s total.
    assert tab._per_frame_sec == pytest.approx(7.5)
    assert tab._per_run_sec == pytest.approx(60.0)
    # Smooth bar (0..1000): half of run 0 of 2 done -> 250.
    assert tab.progress.maximum() == 1000
    assert tab.progress.value() == pytest.approx(250, abs=1)

    # Build a real RunResult and check the results table fills in.
    cfg = ModelConfig()
    sE = pr.spec_for("euler_material.E")
    x0 = pr.get_display(cfg, sE, "C")
    plan = jac.build_plan([(sE, x0, 0.1 * x0, False)], scheme="forward")
    qoi = [QoISpec("Q", "Q", "-", fn=lambda b, inst, w: b["Q"])]
    res = rc.run_plan(plan, "jacobian", qoi,
                      lambda c, i: {"Q": pr.get_display(c, sE, "C")}, cfg)
    tab._last_result = res
    tab._show_results(res)
    assert tab.results_table.rowCount() == 1
    assert tab.results_table.columnCount() == 1 + len(res.qoi_ids)


# ---------------------------------------------------------------------------
# H. CSV export of the sensitivity table / ranking
# ---------------------------------------------------------------------------
def test_export_csv(tmp_path):
    cfg = ModelConfig()
    sE = pr.spec_for("euler_material.E")
    sMu = pr.spec_for("interaction.friction_coeff")
    x0E = pr.get_display(cfg, sE, "C")
    x0M = pr.get_display(cfg, sMu, "C")
    plan = jac.build_plan([(sE, x0E, 1.0, False), (sMu, x0M, 0.01, False)],
                          scheme="forward")
    # Q = 2*E + 100*mu  ->  dQ/dE = 2, dQ/dmu = 100 (mu ranks first)
    qoi = [QoISpec("Q", "Q", "-", fn=lambda b, inst, w: b["Q"])]
    res = rc.run_plan(plan, "jacobian", qoi,
                      lambda c, i: {"Q": 2.0 * pr.get_display(c, sE, "C")
                                    + 100.0 * pr.get_display(c, sMu, "C")}, cfg)

    text = xr.result_to_csv(res, label_for=lambda p: pr.spec_for(p).label)
    lines = text.strip().splitlines()
    assert lines[0].startswith("qoi,parameter,label,sensitivity")
    # within QoI 'Q', mu (|100|) sorts before E (|2|)
    body = [ln for ln in lines[1:] if ln.startswith("Q,")]
    assert body[0].split(",")[1] == "interaction.friction_coeff"
    assert body[1].split(",")[1] == "euler_material.E"

    out = tmp_path / "out.csv"
    xr.write_csv(res, out, label_for=lambda p: pr.spec_for(p).label)
    assert out.exists() and out.read_text(encoding="utf-8-sig").count("\n") >= 3


# ---------------------------------------------------------------------------
# H. to_params_dict serialisation contract: repr() must be literal-only so
#    that run_simul.py's ast.literal_eval(repr(dict)) round-trips exactly.
# ---------------------------------------------------------------------------
def test_to_params_dict_is_literal_only():
    import ast
    cfg = ModelConfig()
    # Exercise a non-default unit system (more code paths in the dict build).
    from gui.core.unit_system import UnitSystem
    cfg.units = UnitSystem(mass="g", length="mm", time="ms", temp="K")
    d = cfg.to_params_dict()
    # literal_eval refuses anything that isn't a Python literal (no dataclass,
    # Decimal, numpy scalar, etc.) — this is exactly the contract run_simul
    # relies on. It must round-trip to an equal dict.
    back = ast.literal_eval(repr(d))
    assert back == d
