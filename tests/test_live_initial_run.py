# -*- coding: utf-8 -*-
"""Live initial run: snapshot retries, the worker loop (launch, periodic
reads of the running .odb, final bundle) and the tab wiring. Abaqus is
faked; nothing here was run with Abaqus."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest

from gui.core.live_run import LiveRunWorker, live_bundle_path, snapshot_args
from gui.core.model_config import ModelConfig

_SNAP = Path(__file__).resolve().parents[1] / "abaqus_scripts" / "live_snapshot.py"


def _load_snapshot():
    spec = importlib.util.spec_from_file_location("live_snapshot", _SNAP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_snapshot_retries_then_writes():
    mod = _load_snapshot()
    calls = []

    def extract(job, cfg, out_base=None):
        calls.append(out_base)
        if len(calls) < 2:
            raise RuntimeError("Object is not available in DB")
    ok, err = mod.snapshot("j", {}, tries=3, wait=0, extract=extract,
                           sleep=lambda s: None)
    assert ok and err == "" and calls == ["j.live", "j.live"]


def test_snapshot_gives_up_after_the_tries():
    mod = _load_snapshot()

    def extract(job, cfg, out_base=None):
        raise RuntimeError("locked")
    ok, err = mod.snapshot("j", {}, tries=2, wait=0, extract=extract,
                           sleep=lambda s: None)
    assert not ok and "locked" in err


def test_snapshot_args():
    a = snapshot_args("abaqus", "/s", "initial_run000", "/r/c.json")
    assert a[:3] == ["abaqus", "python", str(Path("/s") / "live_snapshot.py")]
    assert a[3:5] == ["initial_run000", "/r/c.json"]


class _FakeProc:
    """Runs for `n` polls; writes the .odb at once and the final bundle at
    the end."""
    def __init__(self, run_dir, job, n):
        self.run_dir, self.job, self.n = Path(run_dir), job, n
        self.stdout = None
        self.returncode = None
        (self.run_dir / (job + ".odb")).write_bytes(b"odb")

    def poll(self):
        self.n -= 1
        if self.n > 0:
            return None
        (self.run_dir / (self.job + ".results.npz")).write_bytes(b"x")
        self.returncode = 0
        return 0

    def wait(self, timeout=None):
        return 0


def _worker(tmp_path, n_polls=8, interval=1.0, fail_reads=0):
    run_dir = tmp_path / "initial_run000"
    clock = {"t": 0.0}
    reads = {"n": 0}

    def popen(args, cwd=None, **kw):
        assert "noGUI=" in args[2]
        return _FakeProc(cwd, "initial_run000", n_polls)

    class _Done:
        def __init__(self, rc):
            self.returncode, self.stdout = rc, b"[LIVE] done"

    def run_cmd(argv, cwd=None, **kw):
        reads["n"] += 1
        if reads["n"] <= fail_reads:
            return _Done(3)
        live_bundle_path(cwd, "initial_run000").write_bytes(b"live")
        return _Done(0)

    def sleep(s):
        clock["t"] += s

    w = LiveRunWorker(ModelConfig(), abaqus_cmd="abaqus",
                      abaqus_script=str(tmp_path / "run_simul.py"),
                      run_dir=run_dir, job_name="initial_run000",
                      interval=interval, popen=popen, run_cmd=run_cmd,
                      clock=lambda: clock["t"], sleep=sleep)
    return w, reads


def test_worker_reads_while_running_and_reports_the_final_bundle(tmp_path):
    w, reads = _worker(tmp_path, n_polls=8, interval=1.0)
    snaps, done = [], []
    w.snapshot.connect(lambda p, s: snaps.append(p))
    w.finished.connect(done.append)
    w.run()
    # 0.5 s per poll, a read every 1 s while it runs
    assert reads["n"] >= 2 and len(snaps) == reads["n"]
    assert snaps[0].endswith("initial_run000.live.results.npz")
    assert done == [str(tmp_path / "initial_run000" /
                        "initial_run000.results.npz")]
    assert (tmp_path / "initial_run000" /
            "initial_run000.live_cfg.json").is_file()


def test_failed_read_is_logged_not_fatal(tmp_path):
    w, reads = _worker(tmp_path, n_polls=8, interval=1.0, fail_reads=1)
    logs, snaps = [], []
    w.log.connect(logs.append)
    w.snapshot.connect(lambda p, s: snaps.append(p))
    w.run()
    assert any("[live read] failed" in x for x in logs)
    assert len(snaps) == reads["n"] - 1


def test_tab_reanalyses_each_snapshot(qapp, tmp_path, monkeypatch):
    from gui.tabs.optimization_tab import OptimizationTab
    from gui.tabs.initial_run_tab import InitialRunTab
    opt = OptimizationTab(ModelConfig())
    tab = InitialRunTab(opt)
    seen = []
    monkeypatch.setattr(tab.prelim, "analyse_file", seen.append)
    w, _reads = _worker(tmp_path)
    tab._worker = w
    w.n_snapshots = 1
    tab._on_snapshot(str(tmp_path / "a.live.results.npz"), 1.2)
    assert seen == [str(tmp_path / "a.live.results.npz")]
    assert "Read 1" in tab.lbl_live.text()
    tab._on_finished(str(tmp_path / "a.results.npz"))
    assert seen[-1] == str(tmp_path / "a.results.npz")
    assert tab.btn_launch.isEnabled() and not tab.btn_stop.isEnabled()
    assert not tab.is_running()


def test_launch_refuses_remote_mode(qapp, monkeypatch):
    from gui.tabs.optimization_tab import OptimizationTab
    from gui.tabs.initial_run_tab import InitialRunTab
    import gui.core.remote_exec as rx
    monkeypatch.setattr(rx, "is_remote", lambda p: True)
    tab = InitialRunTab(OptimizationTab(ModelConfig()),
                        prefs_getter=lambda: object())
    assert "remote mode" in tab._launch_problem(object())


# ---- automatic stop (PR 4) ------------------------------------------------
def _st(ts, t_end=1e-3, length=5e-5):
    from gui.sensitivity import preliminary_checks as pc
    st = pc.StationarityResult(length=length, tau=0.1, t_end=t_end)
    for q in pc.QUANTITIES:
        st.per_quantity[q] = pc.Settling(q, None, None, None, 1.0, ts, 0.0)
    return st


def _rc(t_first):
    from gui.sensitivity import preliminary_checks as pc
    return pc.Recontact(None, None, t_first, 3, (0.0, 0.1))


def test_stop_proposal_rules():
    from gui.sensitivity.preliminary_checks import stop_proposal
    # settled at 0.3 ms, 0.7 ms >= 2 L afterwards: stop at the end
    p = stop_proposal(_st(3e-4), _rc(None))
    assert p.reason == "stable" and p.t_stop == 1e-3
    assert p.window == pytest.approx((0.3, 1.0))
    # settled part shorter than 2 L: go on
    assert stop_proposal(_st(9.5e-4), None) is None
    # one quantity never settles: go on
    st = _st(3e-4)
    st.per_quantity["Vx"].t_s = None
    assert stop_proposal(st, None) is None
    # re-contact after t_s: Ttot = t_first, T = [t_s/t_first, 1]
    p = stop_proposal(_st(3e-4), _rc(6e-4))
    assert p.reason == "recontact" and p.t_stop == 6e-4
    assert p.window == pytest.approx((0.5, 1.0)) and p.note == ""
    p = stop_proposal(_st(5.5e-4), _rc(6e-4))
    assert "shorter than 2 L" in p.note
    # re-contact before the run settles: stop, nothing for Step
    p = stop_proposal(_st(None), _rc(2e-4))
    assert p.reason == "recontact" and p.window is None
    # rules off
    assert stop_proposal(_st(3e-4), _rc(6e-4), on_recontact=False).reason \
        == "stable"
    assert stop_proposal(_st(3e-4), _rc(6e-4), on_stable=False,
                         on_recontact=False) is None


def test_stopped_worker_keeps_its_last_read(tmp_path):
    w, reads = _worker(tmp_path, n_polls=1000, interval=1.0)
    done = []
    w.finished.connect(done.append)
    orig = w._snapshot

    def snap(cfg_path):
        orig(cfg_path)
        (w.run_dir / "initial_run000.live.meta.json").write_text("{}")
        if w.n_snapshots == 2:
            w._stop = True          # as stop(), without abaqus terminate
            w.stopped = True
    w._snapshot = snap
    w.run()
    final = tmp_path / "initial_run000" / "initial_run000.results.npz"
    assert done == [str(final)]
    assert final.read_bytes() == b"live"
    assert (final.parent / "initial_run000.meta.json").is_file()


def test_tab_stops_and_writes_step(qapp, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gui.core.model_config import ModelConfig
    from gui.tabs.optimization_tab import OptimizationTab
    from gui.tabs.initial_run_tab import InitialRunTab
    from gui.tabs.step_tab import StepTab
    cfg = ModelConfig()
    opt = OptimizationTab(cfg)
    step = StepTab(cfg)
    tab = InitialRunTab(opt, step_tab=step)
    stops = []
    monkeypatch.setattr(tab, "_on_stop", lambda: stops.append(1))
    tab._worker = SimpleNamespace(stopped=False)        # running
    res = {"st": _st(3e-4, t_end=8e-4), "rc": _rc(None),
           "fc": SimpleNamespace(dt=2e-6)}
    tab.cb_stop_stable.setChecked(False)
    tab._on_analysed(res)
    assert stops == [] and tab.btn_apply.isEnabled()
    assert "settled" in tab.lbl_stop.text()
    tab.cb_stop_stable.setChecked(True)
    tab._on_analysed(res)
    assert stops == [1]
    assert cfg.step.sim_time == pytest.approx(8e-4)
    assert cfg.step.n_frames == 400                     # 8e-4 / 2e-6
    assert opt.window() == pytest.approx((0.375, 1.0))
    assert "Written to Step" in tab.lbl_stop.text()
    tab._on_analysed(res)                               # once only
    assert stops == [1]
