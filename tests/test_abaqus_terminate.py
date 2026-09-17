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


class TestOptimizationTabCancelsTheRunInFlight:
    """The Optimization studies now cancel the way the sensitivity campaigns
    do: `abaqus terminate` first, process tree only as a fallback.

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

    class _Proc:
        def __init__(self, alive=True):
            self.pid = 4321
            self._alive = alive
            self.waited = None
        def poll(self):
            return None if self._alive else 0
        def wait(self, timeout=None):
            self.waited = timeout
            if self._alive:
                raise RuntimeError("still running")
            return 0

    def test_idle_cancel_touches_nothing(self, tab, monkeypatch):
        # No study running: no job name, no process. Must not raise, and must
        # not spawn a terminate for a job that does not exist.
        calls = []
        monkeypatch.setattr("gui.tabs.optimization_tab.abaqus_terminate_job",
                            lambda *a: calls.append(a) or True)
        tab._on_cancel()
        assert calls == []
        assert tab._cancel_evt.is_set()

    def test_clean_route_first_and_no_kill_when_it_answers(self, tab,
                                                           monkeypatch):
        proc = self._Proc(alive=False)      # exits when asked politely
        tab._current_job = "GCI_run002"
        tab._current_proc = proc
        tab._current_abaqus_cmd = "abaqus"
        tab._current_run_dir = "/wd"
        seen, killed = [], []
        monkeypatch.setattr("gui.tabs.optimization_tab.abaqus_terminate_job",
                            lambda *a: seen.append(a) or True)
        monkeypatch.setattr("gui.tabs.optimization_tab.kill_process_tree_by_pid",
                            lambda pid: killed.append(pid))
        tab._on_cancel()
        assert seen == [("abaqus", "GCI_run002", "/wd")]
        # The licence-friendly route worked, so the tree is left alone.
        assert killed == []
        assert proc.waited == 10.0

    def test_falls_back_to_the_tree_when_abaqus_stays_silent(self, tab,
                                                             monkeypatch):
        proc = self._Proc(alive=True)       # ignores the terminate
        tab._current_job = "domainsizing_run000"
        tab._current_proc = proc
        tab._current_abaqus_cmd = "abaqus"
        tab._current_run_dir = "/wd"
        killed = []
        monkeypatch.setattr("gui.tabs.optimization_tab.abaqus_terminate_job",
                            lambda *a: False)
        monkeypatch.setattr("gui.tabs.optimization_tab.kill_process_tree_by_pid",
                            lambda pid: killed.append(pid))
        tab._on_cancel()
        assert killed == [4321]

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
