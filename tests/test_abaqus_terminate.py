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
