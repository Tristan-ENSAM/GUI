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
import threading
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


SCRIPT_LOG_SUFFIX = ".gui.log"


def script_log_path(workdir, job_name: str) -> Path:
    """Where run_simul.py writes this job's diagnostics.

    Mirrors `run_simul.log_path_for` on the Abaqus side. The two cannot share
    one implementation -- that module is written for Abaqus' Python 2.7 and
    lives outside the GUI package -- so a test pins them to the same answer
    instead. Both the Job tab and the sensitivity worker read through here, so
    the rule is stated once on this side.
    """
    return Path(workdir) / ("%s%s" % (job_name, SCRIPT_LOG_SUFFIX))


def build_abaqus_args(abaqus_cmd: str, abaqus_script: str,
                      model_params: dict, run_params: dict) -> list:
    """The exact argv that runs `run_simul.py` under Abaqus/CAE.

    Single source of truth for the launch contract, which three call sites
    used to spell out independently: the Job tab's dry-run preview, the Job
    tab's real launch, and the sensitivity worker. They cannot be allowed to
    drift -- a change made in one of them would silently break the others,
    which is exactly how the two cancel paths ended up behaving differently.

    Both dicts cross the process boundary as `repr()` and are read back with
    `ast.literal_eval` (run_simul.parse_arguments), so they must contain
    literals only -- see ModelConfig.to_params_dict.

    Returns the FULL argv, `abaqus_cmd` included at index 0. QProcess takes
    the program separately from its arguments, so that caller passes
    `args[0]` and `args[1:]`.
    """
    return [abaqus_cmd, "cae", "noGUI=%s" % abaqus_script, "--",
            "--model_cfg", repr(model_params),
            "--run_cfg", repr(run_params)]


def abaqus_terminate_job(abaqus_cmd: str, job_name: str, workdir,
                         timeout: float = 20.0) -> bool:
    """Ask Abaqus to stop `job_name` cleanly: ``abaqus terminate job=<name>``.

    WHY THIS BEFORE KILLING THE PROCESS TREE: terminating through Abaqus stops
    the analysis executable AND RELEASES ITS LICENCE TOKENS. A hard taskkill /
    SIGKILL leaves the tokens checked out until the FlexNet server reclaims
    them, which on a shared licence pool penalises everyone else.

    The command must run FROM THE JOB'S WORKING DIRECTORY: it reads
    ``<job_name>.cid`` there to find the host and port used to signal the job.
    No .cid means the solver never started (or has already exited), so there is
    nothing to terminate.

    Returns True if Abaqus accepted the request. Best-effort: any failure
    returns False so the caller can fall back to killing the process tree.
    """
    if not abaqus_cmd or not job_name:
        return False
    workdir = Path(workdir)
    if not (workdir / ("%s.cid" % job_name)).is_file():
        # Nothing to signal: the analysis is not running.
        return False
    try:
        completed = subprocess.run(
            [abaqus_cmd, "terminate", "job=%s" % job_name],
            cwd=str(workdir), capture_output=True, timeout=timeout, check=False)
        return completed.returncode == 0
    except Exception:
        log_swallowed("asking Abaqus to terminate job %r" % job_name,
                      level=logging.DEBUG)
        return False


def kill_process_tree_by_pid(pid: int) -> bool:
    """Kill `pid` AND every process it spawned, addressing it by PID only.

    For callers holding a QProcess rather than a Popen: QProcess.terminate()
    and .kill() reach only the direct child, so on Windows they stop
    `abaqus.bat`/`cae.exe` and leave the solver processes it spawned
    (pre, standard.exe, explicit.exe, package) running.

    Windows only -- returns False everywhere else, and the caller must then
    fall back to its own single-process kill. The POSIX kill-a-whole-group
    route used by `_terminate_process_tree` is NOT reusable here: it relies on
    Popen(start_new_session=True) having put the child in its own process
    group, which QProcess does not do. os.getpgid() on a QProcess child
    returns the GUI's OWN group, so killpg would take the GUI down with it.
    """
    if not pid:
        return False
    if os.name != "nt":
        return False
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                       capture_output=True, check=False)
        return True
    except Exception:
        log_swallowed("killing the process tree of pid %r" % pid,
                      level=logging.DEBUG)
        return False


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
                log_swallowed("terminating the run process (Windows fallback)",
                              level=logging.DEBUG)
        return
    # POSIX
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            log_swallowed("terminating the run process (no process group)",
                          level=logging.DEBUG)
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
            log_swallowed("killing the run process (fallback)",
                          level=logging.DEBUG)


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
        self._current_job = None           # job name of the run in flight

    # -- control -------------------------------------------------------
    def cancel(self):
        """Stop the campaign, and the run currently in flight.

        Two stages, in this order:
          1. ``abaqus terminate job=<name>`` -- the clean route: it stops the
             solver AND releases the licence tokens.
          2. kill the process tree -- the fallback, for when Abaqus does not
             answer (no .cid yet, job already finishing, hung solver). This
             leaves the tokens checked out, hence the ordering.

        Returns immediately. This method is invoked directly from the Cancel
        slot, so it executes on the GUI THREAD even though the worker lives in
        another one -- and the two stages block for up to 30 s together
        (subprocess timeout 20 s, then the grace wait). Running them inline is
        finding M3: a frozen window for the whole duration. They go to a
        daemon thread instead; the flag below is what actually stops the
        campaign, and it is set synchronously so the run loop sees it at once.
        """
        self._cancel = True
        job = self._current_job
        if not job:
            return
        self.log.emit("[CANCEL] asking Abaqus to terminate job %s\n" % job)
        threading.Thread(target=self._cancel_blocking,
                         args=(job, self._proc), daemon=True).start()

    def _cancel_blocking(self, job: str, proc) -> None:
        """The blocking half of cancel(), off the GUI thread.

        `proc` is passed in rather than read from self: by the time this runs,
        the run loop may have moved on and cleared the attribute, and killing
        the NEXT run's process would be worse than killing nothing.
        """
        if abaqus_terminate_job(self._abaqus_cmd, job, self._workdir):
            if proc is not None:
                try:
                    # Give the solver a moment to unwind before force-killing.
                    proc.wait(timeout=10.0)
                    return
                except Exception:
                    log_swallowed("waiting for the terminated job to exit",
                                  level=logging.DEBUG)
        if proc is not None and proc.returncode is None:
            _terminate_process_tree(proc)

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

    def _emit_log_tail(self, log_path, offset: int) -> int:
        """Emit whatever run_simul.py appended since `offset`; return the new
        offset. Reads from a byte position rather than re-reading, so a long
        campaign does not replay what the panel already shows. Latin-1 decodes
        any byte: a decode error here would silently stop the live log."""
        try:
            with open(log_path, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
        except OSError:
            return offset          # not created yet, or already gone
        if chunk:
            self.log.emit(chunk.decode("latin-1", errors="replace"))
        return offset

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
        args = build_abaqus_args(self._abaqus_cmd, self._abaqus_script,
                                 model_params, run_params)

        self.log.emit("\n%s\n[run %d] %s\n%s\n"
                      % ("-" * 60, i + 1, job_name, "-" * 60))
        # Published BEFORE the process starts so cancel() can name the job to
        # `abaqus terminate` even if the click lands during start-up.
        self._current_job = job_name
        try:
            self._proc = subprocess.Popen(
                args, cwd=str(self._workdir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                **_popen_group_kwargs())
        except Exception as e:
            self.log.emit("[run %d] failed to start Abaqus: %s\n" % (i + 1, e))
            self._current_job = None
            self.runDone.emit(i, False)
            return None

        # Stream the run live by TAILING THE SCRIPT'S LOG, not its stdout.
        # `abaqus cae noGUI=` runs run_simul.py inside a separate kernel
        # process (ABQcaeK.exe) whose stdout reaches nobody -- not this pipe,
        # not even a console redirection. Reading proc.stdout here used to
        # yield nothing but the licence banner, so a campaign showed no sign
        # of what each run was doing. run_simul tees everything into
        # <job>.gui.log; the Job tab tails the same file.
        log_path = script_log_path(self._workdir, job_name)
        offset = 0
        while self._proc.poll() is None:
            if self._cancel:
                break
            offset = self._emit_log_tail(log_path, offset)
            time.sleep(0.4)
        # Final drain: the lines written since the last tick are the ones that
        # explain how the run ended.
        self._emit_log_tail(log_path, offset)
        # Whatever the launcher itself put on stdout (licence banner, a fatal
        # error before the script starts). Read once, after exit.
        try:
            rest = self._proc.stdout.read()
            if rest:
                self.log.emit(rest.decode("cp1252", errors="replace"))
        except Exception:
            log_swallowed("reading Abaqus launcher output for run %d" % (i + 1),
                          level=logging.DEBUG)
        self._proc.wait()
        rc_code = self._proc.returncode
        self._proc = None
        self._current_job = None

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
