# -*- coding: utf-8 -*-
"""Remote execution through the shared-folder queue.

Client: gui.core.remote_exec. Agent: gui.core.remote_agent, the standalone
script the GUI copies into the queue folder and the compute PC runs with the
Python bundled with Abaqus. A stand-in for abaqus.bat (a small Python script
that writes the files a real run leaves behind) replaces Abaqus, and the
"shared drive" and the compute PC's local folder are two temporary
directories, so the whole protocol is exercised on one machine."""
from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from gui.core import remote_agent as ra
from gui.core import remote_exec as rx
from gui.core.preferences import Preferences

# Mimics `abaqus.bat cae noGUI=<run_simul.py> -- --model_cfg <repr>
# --run_cfg <repr>` run in the job folder.
FAKE_ABAQUS = textwrap.dedent('''\
    #!%s
    import ast, os, sys, time
    a = sys.argv[1:]
    if a and a[0] == "terminate":
        sys.exit(1)
    script = a[1].split("=", 1)[1]
    assert os.path.isfile(script), script
    model = ast.literal_eval(a[a.index("--model_cfg") + 1])
    job = ast.literal_eval(a[a.index("--run_cfg") + 1])["job_name"]
    open(job + ".gui.log", "w").write("building model with %%s\\n" %% script)
    open(job + ".sta", "w").write("STEP TOTAL\\n")
    open(job + ".odb", "wb").write(b"x" * 1000)
    time.sleep(float(model.get("duration", 0.3)))
    open(job + ".results.npz", "wb").write(b"npz")
    open(job + ".meta.json", "w").write("{}")
    print("licence banner")
''') % sys.executable


@pytest.fixture
def setup(tmp_path):
    scripts = tmp_path / "abaqus_scripts"
    scripts.mkdir()
    (scripts / "run_simul.py").write_text("# generator v1\n")
    (scripts / "cel_common.py").write_text("# helper\n")
    fake = tmp_path / "abaqus"
    fake.write_text(FAKE_ABAQUS)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    shared = tmp_path / "Z"
    return {"scripts": scripts, "abaqus": str(fake), "queue": shared / "queue",
            "shared": shared, "gui_wd": tmp_path / "gui_pc_wd",
            "local_root": str(tmp_path / "C_local")}


def _agent(setup, **kw):
    return ra.Agent(str(setup["queue"]), setup["abaqus"],
                    local_root=setup["local_root"], out=io.StringIO(), **kw)


def _loop(agent, stop):
    while not stop.is_set():
        agent.step()
        time.sleep(0.05)


@pytest.fixture
def running_agent(setup):
    agent = _agent(setup)
    stop = threading.Event()
    th = threading.Thread(target=_loop, args=(agent, stop), daemon=True)
    th.start()
    yield agent
    stop.set()
    th.join(timeout=5)


def _submit(setup, job="job1", model=None, check_agent=False, sub="study"):
    run_dir = setup["gui_wd"] / sub      # on the GUI PC, not on Z:
    proc = rx.RemoteProcess(setup["queue"], run_dir, model or {},
                            {"cpus": 16, "job_name": job}, setup["scripts"],
                            remote_abaqus_cmd=setup["abaqus"],
                            check_agent=check_agent)
    return proc, run_dir


def test_round_trip_copies_results_back_and_keeps_odb_local(setup,
                                                            running_agent):
    proc, run_dir = _submit(setup)
    assert proc.wait(timeout=30) == 0
    assert (run_dir / "job1.results.npz").read_bytes() == b"npz"
    for name in ("job1.meta.json", "job1.sta", "job1.gui.log"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "job1.odb").exists()
    local = Path(ra.local_run_dir(setup["local_root"], str(proc.transit)))
    assert (local / "job1.odb").is_file()
    # Z: was only a transit area: nothing of the run is left on it.
    assert not proc.transit.exists()
    assert list((setup["queue"] / "runs").iterdir()) == []
    assert b"licence banner" in proc.stdout.read()
    assert proc.stdout.read() == b""
    for sub in ("pending", "running", "done", "cancel"):
        assert list((setup["queue"] / sub).iterdir()) == [], sub


def test_the_agent_runs_the_scripts_copied_from_the_gui_pc(setup,
                                                           running_agent):
    proc, run_dir = _submit(setup)
    assert proc.wait(timeout=30) == 0
    staged = Path(rx.stage_scripts(setup["queue"], setup["scripts"]))
    assert staged.parent == setup["queue"] / "scripts"
    assert (staged / "cel_common.py").is_file()
    assert str(staged / "run_simul.py") in (run_dir / "job1.gui.log").read_text()
    # A new version of the generator goes to a new folder; the old one stays
    # for runs already queued with it.
    (setup["scripts"] / "run_simul.py").write_text("# generator v2\n")
    staged2 = Path(rx.stage_scripts(setup["queue"], setup["scripts"]))
    assert staged2 != staged and staged.is_dir()


def test_live_mirror_of_sta_and_log_while_running(setup, running_agent):
    proc, run_dir = _submit(setup, model={"duration": 4.0})
    deadline = time.monotonic() + 10
    # The GUI's run loops poll every 0.4 s; polling is what brings the live
    # files from the transit folder to the local run folder.
    while proc.poll() is None and not (run_dir / "job1.sta").exists() \
            and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (run_dir / "job1.sta").exists()
    assert proc.poll() is None
    assert not (run_dir / "job1.results.npz").exists()
    assert proc.wait(timeout=30) == 0


def test_runs_execute_one_at_a_time_in_order(setup, running_agent):
    p1, _ = _submit(setup, job="a", model={"duration": 1.0})
    time.sleep(0.01)
    p2, run_dir = _submit(setup, job="b", model={"duration": 0.1})
    assert p2.wait(timeout=30) == 0
    assert p1.poll() == 0
    on_cpc = [Path(ra.local_run_dir(setup["local_root"], str(p.transit)))
              / (j + ".results.npz") for p, j in ((p1, "a"), (p2, "b"))]
    assert on_cpc[0].stat().st_mtime <= on_cpc[1].stat().st_mtime
    assert (run_dir / "a.results.npz").is_file()


def test_line_endings_do_not_change_the_fingerprint(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "x.py").write_bytes(b"line1\nline2\n")
    (b / "x.py").write_bytes(b"line1\r\nline2\r\n")
    assert rx.scripts_fingerprint(a) == rx.scripts_fingerprint(b)


def test_deploy_writes_the_agent_and_a_crlf_launcher(setup):
    bat = rx.deploy_queue(setup["queue"], r"D:\SIMULIA\Commands\abaqus.bat")
    assert bat == setup["queue"] / "start_agent.bat"
    data = bat.read_bytes()
    assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")
    assert b'set "ABQ=D:\\SIMULIA\\Commands\\abaqus.bat"' in data
    assert b'call "%ABQ%" python "%~dp0remote_agent.py"' in data
    assert (setup["queue"] / "remote_agent.py").read_bytes() \
        == rx.AGENT_SOURCE.read_bytes()


def test_cancel_before_claim_withdraws_the_request(setup):
    proc, _ = _submit(setup)
    proc.cancel()
    assert proc.poll() == 1 and proc.cancelled
    assert list((setup["queue"] / "pending").iterdir()) == []
    assert list((setup["queue"] / "cancel").iterdir()) == []


def test_cancel_while_running_terminates_then_kills(setup):
    calls = []
    agent = _agent(setup,
                   terminate=lambda j: calls.append("terminate") or False,
                   kill=lambda j: (calls.append("kill"), j["proc"].kill()))
    stop = threading.Event()
    th = threading.Thread(target=_loop, args=(agent, stop), daemon=True)
    th.start()
    try:
        proc, _ = _submit(setup, model={"duration": 30.0})
        deadline = time.monotonic() + 10
        while agent.job is None and time.monotonic() < deadline:
            time.sleep(0.05)
        proc.cancel()
        proc.wait(timeout=20)
        assert proc.cancelled and proc.returncode != 0
        assert calls == ["terminate", "kill"]
    finally:
        stop.set()
        th.join(timeout=5)


def test_cancel_by_job_name_for_the_job_tab_client(setup):
    proc, _ = _submit(setup, job="Cutting_job")
    other, _ = _submit(setup, job="x-Cutting_job")
    assert rx.request_cancel_by_job(setup["queue"], "Cutting_job")
    assert proc.wait(timeout=5) == 1 and proc.cancelled
    assert other.poll() is None


def test_agent_restart_fails_the_interrupted_run(setup):
    proc, _ = _submit(setup)
    pend = setup["queue"] / "pending" / (proc.id + ".json")
    pend.rename(setup["queue"] / "running" / pend.name)
    _agent(setup).recover()
    assert proc.wait(timeout=5) == 1
    assert "restarted" in proc.error


def test_submit_refuses_when_no_agent_heartbeat(setup, monkeypatch):
    monkeypatch.setattr(rx, "ALIVE_TIMEOUT", 1.0)
    with pytest.raises(rx.RemoteError, match="start_agent.bat"):
        _submit(setup, check_agent=True)
    assert list((setup["queue"] / "pending").iterdir()) == []


def test_local_run_dir_replaces_the_drive(tmp_path):
    root = str(tmp_path)
    assert Path(ra.local_run_dir(root, "/a/b")) == tmp_path / "a" / "b"
    assert Path(ra.local_run_dir(root, "a\\b")) == tmp_path / "a" / "b"


def test_launch_problems(tmp_path):
    scripts = tmp_path / "abaqus_scripts"
    scripts.mkdir()
    (scripts / "run_simul.py").write_text("")
    prefs = Preferences(abaqus_cmd="missing.bat",
                        abaqus_script=str(scripts / "run_simul.py"),
                        execution_mode="remote",
                        remote_queue_dir=r"Z:\ABQ\queue")
    # The working directory stays on the GUI PC; no local Abaqus needed.
    assert rx.launch_problems(prefs, r"C:\TEMP\wd") == []
    prefs.remote_queue_dir = r"Z:\my queue"
    assert any("spaces" in p for p in rx.launch_problems(prefs, r"Z:\wd"))
    prefs.execution_mode = "local"
    assert any("Abaqus command" in p
               for p in rx.launch_problems(prefs, r"C:\TEMP\wd"))


def test_deployed_agent_end_to_end_as_a_separate_process(setup):
    """The file the GUI deploys, started the way start_agent.bat starts it,
    serves a submission checked against its heartbeat."""
    rx.deploy_queue(setup["queue"], setup["abaqus"])
    agent = subprocess.Popen(
        [sys.executable, str(setup["queue"] / "remote_agent.py"),
         "--abaqus", setup["abaqus"], "--local-root", setup["local_root"]],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        proc, run_dir = _submit(setup, check_agent=True)
        assert proc.wait(timeout=60) == 0
        assert (run_dir / "job1.results.npz").is_file()
    finally:
        agent.kill()
        agent.wait()


def test_job_tab_client_command_round_trip(setup, running_agent):
    prefs = Preferences(abaqus_script=str(setup["scripts"] / "run_simul.py"),
                        execution_mode="remote",
                        remote_queue_dir=str(setup["queue"]),
                        remote_abaqus_cmd=setup["abaqus"])
    run_dir = setup["gui_wd"] / "jobtab"
    program, args, root = rx.submit_command(
        prefs, run_dir, {}, {"cpus": 1, "job_name": "Cutting_job"})
    env = dict(os.environ, PYTHONPATH=root)
    out = subprocess.run([program] + args, capture_output=True, env=env,
                         timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert b"licence banner" in out.stdout
    assert (run_dir / "Cutting_job.results.npz").is_file()
