# -*- coding: utf-8 -*-
"""
Background worker that runs a sensitivity plan through Abaqus.

It reuses the exact launch mechanism of the Job tab: for each profile it
calls `<abaqus_cmd> cae noGUI=<run_simul.py> -- --model_cfg <repr> --run_cfg
<repr>` in the working directory, streams the merged output, then loads the
`<job>.results.npz` bundle. The orchestration itself is delegated to
`runner_core.run_plan`, so this class only provides the per-profile
`solve_fn` (Abaqus subprocess) plus Qt signals.

Runs sequentially (one simulation at a time). Designed to live in a
QThread (moveToThread) so the streamed subprocess never blocks the UI.
`solve_fn` is injectable, so the orchestration + signal plumbing can be
exercised in tests without Abaqus.
"""
from __future__ import annotations

from pathlib import Path
import os
import signal
import subprocess
import time
import logging

from PySide6.QtCore import QObject, Signal

from gui.sensitivity import runner_core as rc
from gui.core.logging_util import log_swallowed
from gui.results.reader import ResultsBundle


def _popen_group_kwargs() -> dict:
    """Popen kwargs that put the child in its own process group/session so
    we can later terminate the *whole* tree (Abaqus 'cae' spawns the actual
    solver as a child — killing only the parent would leak the solver)."""
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP only exists on Windows.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_tree(proc: "subprocess.Popen", grace: float = 2.0) -> None:
    """Terminate `proc` and every process it spawned. Best-effort and
    cross-platform.

    POSIX: signal the whole process group, first with SIGTERM (lets the
    solver clean up), then escalate to SIGKILL if it has not exited within
    `grace` seconds — some children ignore SIGTERM, and SIGKILL cannot be
    caught or ignored.

    NOTE: the Windows branch (taskkill /T) cannot be exercised in the
    Linux dev/CI environment — confirm it on the remote PC."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        return
    # POSIX
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        log_swallowed("sending SIGTERM to the run process group",
                      level=logging.DEBUG)
    deadline = time.monotonic() + max(0.0, grace)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        log_swallowed("sending SIGKILL to the run process group",
                      level=logging.DEBUG)
        try:
            proc.kill()
        except Exception:
            pass


class SensitivityRunWorker(QObject):
    progress = Signal(int, int)     # (done, total)
    log = Signal(str)               # live output chunk
    runDone = Signal(int, bool)     # (run_index, ok)
    finished = Signal(object)       # rc.RunResult
    failed = Signal(str)            # fatal error before/around the loop

    def __init__(self, plan, plan_kind, qoi_specs, base_cfg, *,
                 abaqus_cmd: str, abaqus_script: str, workdir: str,
                 cpus: int = 1, warmup_frac: float = 0.0,
                 job_prefix: str = "sens", keep_bundles: bool = False,
                 field_vars=None, field_metric: str = "ssd",
                 solve_fn=None, parent=None):
        super().__init__(parent)
        self._plan = plan
        self._plan_kind = plan_kind
        self._qoi_specs = qoi_specs
        self._base_cfg = base_cfg
        self._abaqus_cmd = abaqus_cmd
        self._abaqus_script = abaqus_script
        self._workdir = Path(workdir)
        self._cpus = int(cpus)
        self._warmup = float(warmup_frac)
        self._job_prefix = job_prefix
        self._keep_bundles = keep_bundles
        self._field_vars = list(field_vars) if field_vars else None
        self._field_metric = field_metric
        self._solve_fn = solve_fn          # injected (tests); else Abaqus
        self._cancel = False
        self._proc = None                  # current subprocess.Popen

    # -- control -------------------------------------------------------
    def cancel(self):
        self._cancel = True
        p = self._proc
        if p is not None:
            _terminate_process_tree(p)

    # -- entry point (run inside the QThread) --------------------------
    def run(self):
        try:
            solve = self._solve_fn or self._abaqus_solve
            result = rc.run_plan(
                self._plan, self._plan_kind, self._qoi_specs, solve,
                self._base_cfg, warmup_frac=self._warmup,
                progress=lambda d, t: self.progress.emit(d, t),
                should_cancel=lambda: self._cancel,
                keep_bundles=self._keep_bundles,
                field_vars=self._field_vars, field_metric=self._field_metric)
            self.finished.emit(result)
        except Exception as e:                          # pragma: no cover
            self.failed.emit("%s" % e)

    # -- the Abaqus per-profile solve (default solve_fn) ---------------
    def _abaqus_solve(self, cfg, i):
        job_name = "%s_run%03d" % (self._job_prefix, i)
        out_path = self._workdir / ("%s.results.npz" % job_name)
        try:
            if out_path.exists():
                out_path.unlink()           # avoid reading a stale bundle
        except Exception:
            log_swallowed("removing stale bundle %s" % out_path,
                          level=logging.DEBUG)

        model_params = cfg.to_params_dict()
        run_params = {"cpus": self._cpus, "job_name": job_name}
        args = [self._abaqus_cmd, "cae",
                "noGUI=%s" % self._abaqus_script, "--",
                "--model_cfg", repr(model_params),
                "--run_cfg", repr(run_params)]

        self.log.emit("\n%s\n[run %d] %s\n%s\n"
                      % ("-" * 60, i + 1, job_name, "-" * 60))
        try:
            self._proc = subprocess.Popen(
                args, cwd=str(self._workdir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                **_popen_group_kwargs())
        except Exception as e:
            self.log.emit("[run %d] failed to start Abaqus: %s\n" % (i + 1, e))
            self.runDone.emit(i, False)
            return None

        # Stream merged output live (Abaqus uses cp1252 on Windows).
        try:
            for raw in iter(self._proc.stdout.readline, b""):
                if self._cancel:
                    break
                self.log.emit(raw.decode("cp1252", errors="replace"))
        except Exception:
            log_swallowed("streaming Abaqus stdout for run %d" % (i + 1),
                          level=logging.DEBUG)
        self._proc.wait()
        rc_code = self._proc.returncode
        self._proc = None

        if self._cancel:
            self.runDone.emit(i, False)
            return None
        if rc_code != 0 or not out_path.exists():
            self.log.emit("[run %d] no results bundle (returncode=%s)\n"
                          % (i + 1, rc_code))
            self.runDone.emit(i, False)
            return None
        try:
            bundle = ResultsBundle.load(out_path)
        except Exception as e:
            self.log.emit("[run %d] could not load results: %s\n" % (i + 1, e))
            self.runDone.emit(i, False)
            return None
        self.runDone.emit(i, True)
        return bundle
