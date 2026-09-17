# -*- coding: utf-8 -*-
"""
Clean job termination: `abaqus terminate job=<name>`.

WHY IT MATTERS: terminating through Abaqus stops the analysis AND releases its
licence tokens. A hard taskkill/SIGKILL leaves them checked out until the
FlexNet server reclaims them -- on a shared pool that penalises everyone else.
So the clean route is tried first and the process kill is only the fallback.
"""
from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from gui.sensitivity.run_worker import (abaqus_terminate_job,
                                        build_abaqus_args,
                                        kill_process_tree_by_pid)


def _script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


class TestAbaqusTerminateJob:
    def test_no_cid_means_nothing_to_terminate(self, tmp_path):
        # The .cid carries the host/port used to signal the job. No .cid means
        # the solver never started (or already exited): do not spawn anything.
        assert abaqus_terminate_job("abaqus", "J", tmp_path) is False

    @pytest.mark.parametrize("cmd,job", [("", "J"), ("abaqus", "")])
    def test_missing_arguments_are_refused(self, cmd, job, tmp_path):
        (tmp_path / "J.cid").write_text("host:1\n")
        assert abaqus_terminate_job(cmd, job, tmp_path) is False

    def test_unusable_command_does_not_raise(self, tmp_path):
        (tmp_path / "J.cid").write_text("host:1\n")
        # Must degrade to False so the caller falls back to killing the tree.
        assert abaqus_terminate_job("/nonexistent/abaqus", "J", tmp_path) is False

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell script stub")
    def test_issues_the_documented_command_from_the_job_directory(self, tmp_path):
        (tmp_path / "J.cid").write_text("host:1\n")
        exe = _script(tmp_path / "abq.sh",
                      '#!/bin/sh\necho "$@" > "$PWD/called.txt"\nexit 0\n')
        assert abaqus_terminate_job(str(exe), "J", tmp_path) is True
        # `job=` requires the CWD to be the job's working directory.
        assert (tmp_path / "called.txt").read_text().strip() == "terminate job=J"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell script stub")
    def test_nonzero_exit_reports_failure(self, tmp_path):
        (tmp_path / "J.cid").write_text("host:1\n")
        exe = _script(tmp_path / "fail.sh", "#!/bin/sh\nexit 1\n")
        assert abaqus_terminate_job(str(exe), "J", tmp_path) is False


class TestBuildAbaqusArgs:
    """The launch contract, shared by the Job tab (preview AND real launch)
    and the sensitivity worker. Three call sites used to spell it out
    separately; this pins the shape so they cannot drift apart again."""

    def test_the_documented_command_shape(self):
        args = build_abaqus_args("abq.bat", "run_simul.py",
                                 {"a": 1}, {"job_name": "J"})
        assert args == ["abq.bat", "cae", "noGUI=run_simul.py", "--",
                        "--model_cfg", "{'a': 1}",
                        "--run_cfg", "{'job_name': 'J'}"]

    def test_the_program_is_index_zero(self):
        # QProcess.start() wants program and arguments separately, so the Job
        # tab passes args[0] and args[1:]. Guard that split staying valid.
        args = build_abaqus_args("abq.bat", "s.py", {}, {})
        assert args[0] == "abq.bat"
        assert args[1] == "cae"

    def test_config_crosses_as_a_literal_repr(self):
        """run_simul.parse_arguments reads both dicts back with
        ast.literal_eval, so what is written must survive that round trip."""
        import ast
        model = {"geometry": {"bbox": {"xmin": -0.5}}, "flag": True}
        run = {"cpus": 4, "job_name": "J", "write_inp_only": False}
        args = build_abaqus_args("abq", "s.py", model, run)
        assert ast.literal_eval(args[args.index("--model_cfg") + 1]) == model
        assert ast.literal_eval(args[args.index("--run_cfg") + 1]) == run


class TestKillProcessTreeByPid:
    """The Job tab's fallback when `abaqus terminate` cannot answer.

    Abaqus spawns the solver as its own process, so killing only the launcher
    leaves standard.exe/explicit.exe orphaned with their licence tokens held.
    """

    def test_refuses_a_null_pid(self):
        assert kill_process_tree_by_pid(0) is False

    def test_posix_declines_so_the_caller_falls_back(self, monkeypatch):
        # os.getpgid() on a QProcess child returns the GUI's OWN group (Qt
        # does not put it in a new one), so killpg would kill the GUI. The
        # function must decline rather than guess.
        monkeypatch.setattr(os, "name", "posix")
        called = []
        monkeypatch.setattr("subprocess.run",
                            lambda *a, **k: called.append(a))
        assert kill_process_tree_by_pid(4321) is False
        assert called == []

    def test_windows_issues_taskkill_with_the_tree_flag(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        seen = {}

        def _fake_run(args, **kwargs):
            seen["args"] = args
            return None

        monkeypatch.setattr("subprocess.run", _fake_run)
        assert kill_process_tree_by_pid(4321) is True
        # /T is what makes it a TREE kill -- without it the solver survives.
        assert seen["args"] == ["taskkill", "/F", "/T", "/PID", "4321"]

    def test_a_failing_taskkill_reports_false(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")

        def _boom(*a, **k):
            raise OSError("taskkill missing")

        monkeypatch.setattr("subprocess.run", _boom)
        assert kill_process_tree_by_pid(4321) is False


class TestWorkerTracksCurrentJob:
    def test_cancel_without_a_running_job_is_harmless(self):
        # cancel() must be safe before anything has started.
        from gui.sensitivity.run_worker import SensitivityRunWorker
        w = SensitivityRunWorker(
            plan=[], plan_kind="oat", qoi_specs=[], base_cfg=None,
            abaqus_cmd="abaqus", abaqus_script="s.py", workdir=".")
        assert w._current_job is None
        w.cancel()
        assert w._cancel is True


# --------------------------------------------------------------------------
# The cancel paths are ASYNCHRONOUS since M3 was reopened: the blocking calls
# run on a daemon thread and the result comes back through a queued Qt signal.
# Tests must therefore pump the event loop instead of reading state straight
# after the call -- which is also the property under test: the slot returns
# before the blocking work is done.
# --------------------------------------------------------------------------

def _pump(qapp, predicate, timeout=5.0):
    """Spin the event loop until `predicate()` is true, or give up."""
    import time as _t
    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        _t.sleep(0.01)
    qapp.processEvents()
    return predicate()


class _FakePopen:
    """Stands in for the Abaqus launcher. `returncode` is the attribute the
    cancel path reads instead of calling poll(), to avoid two threads reaping
    the same child."""

    def __init__(self, returncode=None):
        self.pid = 4321
        self.returncode = returncode


class TestOptimizationTabCancelsTheRunInFlight:
    """The Optimization studies cancel the way the campaigns do: `abaqus
    terminate` first, process tree only as a fallback, neither on the GUI
    thread.

    Before this, _on_cancel called proc.terminate() on the `abaqus cae`
    launcher. That reaches the launcher alone -- the solver behind it survives
    (M1) and the licence stays checked out -- and the test that triggered it
    sat inside `iter(proc.stdout.readline, b"")`, a loop no line ever wakes,
    because `abaqus cae noGUI=` produces no stdout (M7). The button therefore
    appeared to do nothing until the run ended on its own.
    """

    @pytest.fixture
    def tab(self, qapp):
        from gui.core.model_config import ModelConfig
        from gui.tabs.optimization_tab import OptimizationTab
        return OptimizationTab(ModelConfig())

    def _arm(self, tab, job="GCI_run002", returncode=None):
        proc = _FakePopen(returncode)
        tab._current_job = job
        tab._current_proc = proc
        tab._current_abaqus_cmd = "abaqus"
        tab._current_run_dir = "/wd"
        return proc

    def test_idle_cancel_touches_nothing(self, tab, monkeypatch):
        # No study running: no job name, no process. Must not raise, and must
        # not spawn a terminate for a job that does not exist.
        calls = []
        monkeypatch.setattr("gui.tabs.optimization_tab.abaqus_terminate_job",
                            lambda *a: calls.append(a) or True)
        tab._on_cancel()
        assert calls == []
        assert tab._cancel_evt.is_set()

    def test_the_slot_returns_before_the_blocking_call_finishes(self, tab,
                                                               monkeypatch,
                                                               qapp):
        """The whole point of M3's correction: `abaqus terminate` can sit for
        20 s, and the window must survive it."""
        import time as _t
        started = threading.Event()
        release = threading.Event()

        def _slow(*_a):
            started.set()
            release.wait(5.0)
            return True

        monkeypatch.setattr("gui.tabs.optimization_tab.abaqus_terminate_job",
                            _slow)
        self._arm(tab)
        t0 = _t.monotonic()
        tab._on_cancel()
        elapsed = _t.monotonic() - t0
        # The slot returned while the "subprocess" is still blocked.
        assert started.wait(2.0)
        assert elapsed < 1.0, "the cancel slot blocked the GUI thread"
        release.set()

    def test_the_clean_route_is_asked_first(self, tab, monkeypatch, qapp):
        seen, killed = [], []
        monkeypatch.setattr("gui.tabs.optimization_tab.abaqus_terminate_job",
                            lambda *a: seen.append(a) or True)
        monkeypatch.setattr("gui.tabs.optimization_tab.kill_process_tree_by_pid",
                            lambda pid: killed.append(pid))
        self._arm(tab)
        tab._on_cancel()
        assert _pump(qapp, lambda: bool(seen))
        assert seen == [("abaqus", "GCI_run002", "/wd")]
        # Abaqus accepted: the tree is spared, and the kill is only armed on a
        # timer that this test deliberately does not wait out.
        assert killed == []

    def test_falls_back_to_the_tree_when_abaqus_stays_silent(self, tab,
                                                             monkeypatch,
                                                             qapp):
        killed = []
        monkeypatch.setattr("gui.tabs.optimization_tab.abaqus_terminate_job",
                            lambda *a: False)
        monkeypatch.setattr("gui.tabs.optimization_tab.kill_process_tree_by_pid",
                            lambda pid: killed.append(pid) or True)
        self._arm(tab, job="domainsizing_run000")
        tab._on_cancel()
        assert _pump(qapp, lambda: bool(killed))
        assert killed == [4321]

    def test_an_already_finished_run_is_not_killed(self, tab, monkeypatch,
                                                   qapp):
        """PID reuse is the hazard: the run exited between the click and the
        timer, so taskkill would aim at whatever inherited the number."""
        killed = []
        monkeypatch.setattr("gui.tabs.optimization_tab.kill_process_tree_by_pid",
                            lambda pid: killed.append(pid) or True)
        proc = _FakePopen(returncode=0)     # already reaped
        tab._kill_if_alive(proc)
        qapp.processEvents()
        assert killed == []

    def test_the_label_no_longer_promises_a_deferred_cancel(self, tab):
        tab._on_cancel()
        text = tab.lbl_status.text()
        assert "after the current run" not in text
        assert "current run" in text

    def test_the_log_tail_emits_only_new_bytes(self, tab, tmp_path):
        # Same contract as the campaign worker: the panel must not replay what
        # it already shows, and a non-ASCII byte must not stop the live log.
        log = tmp_path / "GCI_run000.gui.log"
        log.write_bytes(b"[STAGE] SOLVE_START\n")
        offset = tab._emit_log_tail(log, 0)
        assert "[STAGE] SOLVE_START" in tab.log.toPlainText()
        with open(log, "ab") as handle:
            handle.write(b"\xe9chec\n")
        tab._emit_log_tail(log, offset)
        panel = tab.log.toPlainText()
        assert "chec" in panel
        assert panel.count("[STAGE] SOLVE_START") == 1

    def test_an_absent_log_is_not_an_error(self, tab, tmp_path):
        assert tab._emit_log_tail(tmp_path / "nope.gui.log", 0) == 0


class TestJobTabCancelDoesNotFreezeTheWindow:
    """M3: the Job tab's Cancel ran `abaqus terminate` (subprocess timeout
    20 s), then waitForFinished(10000), then waitForFinished(2000) -- all on
    the GUI thread, so the window was dead for up to ~32 s. The subprocess
    calls now go off-thread and the two waits are single-shot timers."""

    @pytest.fixture
    def tab(self, qapp):
        from gui.core.model_config import ModelConfig
        from gui.core.preferences import Preferences
        from gui.tabs.job_tab import JobTab
        return JobTab(ModelConfig(), lambda: Preferences())

    def test_terminate_runs_off_the_gui_thread(self, tab, monkeypatch, qapp):
        gui_thread = threading.current_thread().ident
        ran_on = {}
        monkeypatch.setattr(
            "gui.tabs.job_tab.abaqus_terminate_job",
            lambda *a: ran_on.setdefault("id", threading.current_thread().ident))
        tab._pipeline = {"job_name": "J", "workdir": "/wd"}
        monkeypatch.setattr(tab, "_get_prefs",
                            lambda: type("P", (), {"abaqus_cmd": "abaqus"})())
        # Drive the stage directly: _cancel_run's QMessageBox needs a user.
        from gui.core.async_call import run_async
        run_async(lambda: ran_on.setdefault(
            "id", threading.current_thread().ident), lambda _r: None, tab)
        assert _pump(qapp, lambda: "id" in ran_on)
        assert ran_on["id"] != gui_thread

    def test_a_finished_process_is_not_killed(self, tab, monkeypatch, qapp):
        killed = []
        monkeypatch.setattr("gui.tabs.job_tab.kill_process_tree_by_pid",
                            lambda pid: killed.append(pid) or True)
        tab._proc = None                    # pipeline already cleaned up
        tab._force_kill()
        qapp.processEvents()
        assert killed == []
        assert "CANCELLED by user" in tab.txt_output.toPlainText()

    def test_the_tree_kill_is_tried_before_the_single_process(self, tab,
                                                              monkeypatch,
                                                              qapp):
        from PySide6.QtCore import QProcess
        killed, fallback = [], []

        class _P:
            def state(self):
                return QProcess.Running
            def processId(self):
                return 777
            def terminate(self):
                fallback.append("terminate")

        tab._proc = _P()
        monkeypatch.setattr("gui.tabs.job_tab.kill_process_tree_by_pid",
                            lambda pid: killed.append(pid) or True)
        tab._force_kill()
        assert _pump(qapp, lambda: bool(killed))
        assert killed == [777]
        # taskkill answered, so the single-process escalation stays unused.
        assert fallback == []

    def test_posix_falls_back_to_terminate_then_kill(self, tab, monkeypatch,
                                                     qapp):
        from PySide6.QtCore import QProcess
        events = []

        class _P:
            def state(self):
                return QProcess.Running
            def processId(self):
                return 778
            def terminate(self):
                events.append("terminate")
            def kill(self):
                events.append("kill")

        tab._proc = _P()
        # kill_process_tree_by_pid returns False off Windows.
        monkeypatch.setattr("gui.tabs.job_tab.kill_process_tree_by_pid",
                            lambda pid: False)
        tab._force_kill()
        assert _pump(qapp, lambda: "terminate" in events)
        tab._escalate_kill()                # the 2 s timer, fired by hand
        assert events == ["terminate", "kill"]
