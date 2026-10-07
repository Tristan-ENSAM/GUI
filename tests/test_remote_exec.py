# -*- coding: utf-8 -*-
"""Remote execution through the shared-folder queue (gui.core.remote_exec).

The agent runs a stand-in for Abaqus (a small Python script that writes the
files a real run leaves behind), so the whole protocol -- submit, claim, live
mirror, copy-back, cancel, version check -- is exercised without Abaqus and
without a second PC: the "shared drive" and the compute PC's local folder are
two temporary directories."""
from __future__ import annotations

import io
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from gui.core import remote_exec as rx
from gui.core.preferences import Preferences

FAKE_ABAQUS = textwrap.dedent('''
    import sys, time
    job, duration = sys.argv[1], float(sys.argv[2])
    open(job + ".gui.log", "w").write("building model\\n")
    open(job + ".sta", "w").write("STEP TOTAL\\n")
    open(job + ".odb", "wb").write(b"x" * 1000)
    time.sleep(duration)
    open(job + ".results.npz", "wb").write(b"npz")
    open(job + ".meta.json", "w").write("{}")
    print("licence banner")
''')


@pytest.fixture
def setup(tmp_path):
    scripts = tmp_path / "abaqus_scripts"
    scripts.mkdir()
    (scripts / "run_simul.py").write_text("# generator v1\n")
    fake = tmp_path / "fake_abaqus.py"
    fake.write_text(FAKE_ABAQUS)
    shared = tmp_path / "Z"
    queue = shared / "queue"
    return {"scripts": scripts, "fake": fake, "queue": queue,
            "shared": shared, "local_root": tmp_path / "C_local"}


def _agent(setup, duration=0.3, **kw):
    def build_args(cmd, script, model_params, run_params):
        return [sys.executable, str(setup["fake"]), run_params["job_name"],
                str(model_params.get("duration", duration))]
    return rx.RemoteAgent(setup["queue"], "abaqus.bat",
                          str(setup["scripts"] / "run_simul.py"),
                          local_root=str(setup["local_root"]),
                          build_args=build_args, out=io.StringIO(), **kw)


def _run_agent(agent, stop):
    while not stop.is_set():
        agent.step()
        time.sleep(0.05)


@pytest.fixture
def running_agent(setup):
    agent = _agent(setup)
    stop = threading.Event()
    th = threading.Thread(target=_run_agent, args=(agent, stop), daemon=True)
    th.start()
    yield agent
    stop.set()
    th.join(timeout=5)


def _submit(setup, job="job1", model=None, check_agent=False):
    run_dir = setup["shared"] / "study"
    return rx.RemoteProcess(setup["queue"], run_dir, model or {},
                            {"cpus": 16, "job_name": job}, setup["scripts"],
                            check_agent=check_agent), run_dir


def test_round_trip_copies_results_back_and_keeps_odb_local(setup,
                                                            running_agent):
    proc, run_dir = _submit(setup)
    assert proc.wait(timeout=30) == 0
    assert (run_dir / "job1.results.npz").read_bytes() == b"npz"
    for name in ("job1.meta.json", "job1.sta", "job1.gui.log"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "job1.odb").exists()
    local = rx.local_run_dir(setup["local_root"], run_dir)
    assert (local / "job1.odb").is_file()
    assert b"licence banner" in proc.stdout.read()
    assert proc.stdout.read() == b""
    # The outcome file is consumed; nothing is left in the queue.
    for sub in ("pending", "running", "done", "cancel"):
        assert list((setup["queue"] / sub).iterdir()) == [], sub


def test_live_mirror_of_sta_and_log_while_running(setup, running_agent):
    proc, run_dir = _submit(setup, model={"duration": 4.0})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (run_dir / "job1.sta").exists():
        time.sleep(0.1)
    assert (run_dir / "job1.sta").exists()
    assert proc.poll() is None                  # still running
    assert not (run_dir / "job1.results.npz").exists()
    assert proc.wait(timeout=30) == 0


def test_runs_execute_one_at_a_time_in_order(setup, running_agent):
    p1, _ = _submit(setup, job="a", model={"duration": 1.0})
    time.sleep(0.01)
    p2, run_dir = _submit(setup, job="b", model={"duration": 0.1})
    assert p2.wait(timeout=30) == 0
    assert p1.returncode == 0 or p1.poll() == 0
    a = rx.local_run_dir(setup["local_root"], run_dir)
    assert (a / "a.results.npz").stat().st_mtime \
        <= (a / "b.results.npz").stat().st_mtime


def test_script_version_mismatch_is_refused(setup, running_agent):
    (setup["scripts"] / "run_simul.py").write_text("# generator v2\n")
    proc, run_dir = _submit(setup)
    assert proc.wait(timeout=30) == 1
    assert "scripts differ" in proc.error
    assert not (run_dir / "job1.results.npz").exists()


def test_line_endings_do_not_change_the_fingerprint(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "x.py").write_bytes(b"line1\nline2\n")
    (b / "x.py").write_bytes(b"line1\r\nline2\r\n")
    assert rx.scripts_fingerprint(a) == rx.scripts_fingerprint(b)


def test_cancel_before_claim_withdraws_the_request(setup):
    proc, _ = _submit(setup)                    # no agent running
    proc.cancel()
    assert proc.poll() == 1 and proc.cancelled
    assert list((setup["queue"] / "pending").iterdir()) == []
    assert list((setup["queue"] / "cancel").iterdir()) == []


def test_cancel_while_running_terminates_then_reports(setup):
    calls = []
    agent = _agent(setup, terminate=lambda j: calls.append("terminate") or False,
                   kill_tree=lambda j: (calls.append("kill"),
                                        j["proc"].kill()))
    stop = threading.Event()
    th = threading.Thread(target=_run_agent, args=(agent, stop), daemon=True)
    th.start()
    try:
        proc, _ = _submit(setup, model={"duration": 30.0})
        deadline = time.monotonic() + 10
        while agent.job is None and time.monotonic() < deadline:
            time.sleep(0.05)
        proc.cancel()
        proc.wait(timeout=20)
        assert proc.cancelled
        assert proc.returncode != 0
        assert calls == ["terminate", "kill"]
    finally:
        stop.set()
        th.join(timeout=5)


def test_cancel_by_job_name_for_the_job_tab_client(setup):
    proc, _ = _submit(setup, job="Cutting_job")
    other, _ = _submit(setup, job="x-Cutting_job")
    assert rx.request_cancel_by_job(setup["queue"], "Cutting_job")
    assert proc.wait(timeout=5) == 1 and proc.cancelled
    assert other.poll() is None                 # different job untouched


def test_agent_restart_fails_the_interrupted_run(setup):
    proc, _ = _submit(setup)
    pend = setup["queue"] / "pending" / (proc.id + ".json")
    pend.rename(setup["queue"] / "running" / pend.name)  # claimed, agent died
    _agent(setup).recover()
    assert proc.wait(timeout=5) == 1
    assert "restarted" in proc.error


def test_submit_refuses_when_no_agent_heartbeat(setup, monkeypatch):
    monkeypatch.setattr(rx, "ALIVE_TIMEOUT", 1.0)
    with pytest.raises(rx.RemoteError, match="not running"):
        _submit(setup, check_agent=True)
    assert list((setup["queue"] / "pending").iterdir()) == []


def test_agent_alive_sees_the_heartbeat(setup, running_agent):
    hb = rx.agent_alive(setup["queue"], timeout=10)
    assert hb is not None and hb["scripts_fingerprint"]
    assert rx.agent_alive(setup["queue"].parent / "nothing", timeout=1) is None


def test_local_run_dir_replaces_the_drive(tmp_path):
    assert rx.local_run_dir(tmp_path, "/a/b") == tmp_path / "a" / "b"


def test_launch_problems_remote_requires_workdir_on_shared_drive(tmp_path):
    scripts = tmp_path / "abaqus_scripts"
    scripts.mkdir()
    (scripts / "run_simul.py").write_text("")
    prefs = Preferences(abaqus_cmd="missing.bat",
                        abaqus_script=str(scripts / "run_simul.py"),
                        execution_mode="remote",
                        remote_queue_dir=r"Z:\ABQ\queue")
    probs = rx.launch_problems(prefs, r"C:\TEMP\wd")
    assert any("shared drive Z:" in p for p in probs)
    # The local Abaqus command is not needed in remote mode.
    assert not any("Abaqus command" in p for p in probs)
    assert rx.launch_problems(prefs, r"z:\ABQ\wd") == []
    prefs.execution_mode = "local"
    assert any("Abaqus command" in p
               for p in rx.launch_problems(prefs, r"C:\TEMP\wd"))
    prefs.execution_mode, prefs.remote_queue_dir = "remote", ""
    assert any("no queue folder" in p
               for p in rx.launch_problems(prefs, r"Z:\wd"))


def test_job_tab_client_command_round_trip(setup, running_agent):
    """The Job tab runs `python -m gui.core.remote_exec submit` as a child."""
    import subprocess
    prefs = Preferences(abaqus_script=str(setup["scripts"] / "run_simul.py"),
                        execution_mode="remote",
                        remote_queue_dir=str(setup["queue"]))
    run_dir = setup["shared"] / "jobtab"
    program, args, root = rx.submit_command(
        prefs, run_dir, {}, {"cpus": 1, "job_name": "Cutting_job"})
    import os
    env = dict(os.environ, PYTHONPATH=root)
    out = subprocess.run([program] + args, capture_output=True, env=env,
                         timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert b"licence banner" in out.stdout
    assert (run_dir / "Cutting_job.results.npz").is_file()
