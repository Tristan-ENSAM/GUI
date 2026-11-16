# -*- coding: utf-8 -*-
"""One local run read while it runs (Optimization > Initial run).

The worker launches the model like any GUI run (run_simul.py under
`abaqus cae noGUI`), and every `interval` seconds, once the .odb exists,
runs `abaqus python live_snapshot.py` in the job folder: it writes
<job>.live.results.npz, the results so far, which the tab re-analyses.
Route A of the design doc; feasibility measured on CL-WTS-1 (2026-11-16:
5 reads of 6 during the run, ~1 s per read on a small model, one transient
"Object is not available in DB" error), not on the office PC yet.

A stopped run (Stop button or an automatic stop rule of the tab) gets its
final results from the last snapshot taken after the stop: the live bundle
is copied to <job>.results.npz / <job>.meta.json, the names of a finished
run, so the studies and the Results tab can read it. Hypothesis, not
checked with Abaqus: the .odb of a terminated job is readable (the last
snapshot then fails and the run has no final results).

Local runs only (remote mode: a later PR, the agent of the compute PC
would make the snapshots). The interval is a choice of the design doc
(30 s); the time of one read on a large model is not measured.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from gui.core.logging_util import log_swallowed
from gui.sensitivity.run_worker import (
    _drain_pipe, _popen_group_kwargs, _terminate_process_tree,
    abaqus_terminate_job, build_abaqus_args, script_log_path)

LIVE_SUFFIX = ".live.results.npz"
DEFAULT_INTERVAL_S = 30.0


def live_bundle_path(run_dir, job_name) -> Path:
    return Path(run_dir) / (job_name + LIVE_SUFFIX)


def snapshot_args(abaqus_cmd: str, script_dir, job_name: str,
                  cfg_path, tries: int = 3, wait: float = 5.0) -> list:
    """argv of one snapshot (`abaqus python live_snapshot.py ...`)."""
    return [abaqus_cmd, "python", str(Path(script_dir) / "live_snapshot.py"),
            job_name, str(cfg_path), "--tries", str(int(tries)),
            "--wait", "%g" % wait]


class LiveRunWorker(QObject):
    log = Signal(str)
    # (path of the live bundle, seconds the snapshot took)
    snapshot = Signal(str, float)
    # path of the final <job>.results.npz, or "" (failed, or stopped with
    # no results read); `stopped` tells a stopped run from a finished one
    finished = Signal(str)

    def __init__(self, cfg, *, abaqus_cmd: str, abaqus_script: str,
                 run_dir, job_name: str, cpus: int = 1,
                 interval: float = DEFAULT_INTERVAL_S,
                 popen=None, run_cmd=None, clock=None, sleep=None,
                 parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._abaqus_cmd = abaqus_cmd
        self._abaqus_script = abaqus_script
        self._script_dir = Path(abaqus_script).parent
        self.run_dir = Path(run_dir)
        self.job_name = job_name
        self._cpus = int(cpus)
        self.interval = float(interval)
        self._popen = popen or subprocess.Popen       # injected in tests
        self._run_cmd = run_cmd or subprocess.run
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._stop = False
        self.stopped = False
        self._proc = None
        self.n_snapshots = 0

    # -- control (GUI thread) -------------------------------------------
    def stop(self):
        """Stop the run: `abaqus terminate`, then the process tree (off the
        GUI thread). The results read so far stay in the live bundle."""
        self._stop = True
        self.stopped = True
        proc = self._proc
        self.log.emit("[STOP] asking Abaqus to terminate job %s\n"
                      % self.job_name)
        threading.Thread(target=self._stop_blocking, args=(proc,),
                         daemon=True).start()

    def _stop_blocking(self, proc):
        if abaqus_terminate_job(self._abaqus_cmd, self.job_name,
                                self.run_dir):
            if proc is not None:
                try:
                    proc.wait(timeout=10.0)
                    return
                except Exception:
                    log_swallowed("waiting for the terminated job",
                                  level=logging.DEBUG)
        if proc is not None and proc.poll() is None:
            _terminate_process_tree(proc)

    # -- thread entry ---------------------------------------------------
    def run(self):
        final = self.run_dir / (self.job_name + ".results.npz")
        try:
            self._run()
        except Exception as e:      # never leave the tab waiting
            self.log.emit("[live run] %s\n" % e)
        if self._stop:
            final = self._promote_live()
        self.finished.emit(str(final) if final is not None
                           and final.is_file() else "")

    def _promote_live(self):
        """Stopped run: the last live bundle becomes <job>.results.npz (and
        its .meta.json), or None when no bundle was read."""
        import shutil
        live = live_bundle_path(self.run_dir, self.job_name)
        meta = Path(str(live).replace(".results.npz", ".meta.json"))
        if not live.is_file():
            return None
        final = self.run_dir / (self.job_name + ".results.npz")
        try:
            shutil.copyfile(live, final)
            if meta.is_file():
                shutil.copyfile(meta, self.run_dir
                                / (self.job_name + ".meta.json"))
        except OSError as e:
            self.log.emit("[live run] cannot keep the last results: %s\n"
                          % e)
            return None
        return final

    def _run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        model_params = self._cfg.to_params_dict()
        cfg_path = self.run_dir / (self.job_name + ".live_cfg.json")
        cfg_path.write_text(json.dumps(model_params), encoding="utf-8")
        args = build_abaqus_args(self._abaqus_cmd, self._abaqus_script,
                                 model_params,
                                 {"cpus": self._cpus,
                                  "job_name": self.job_name})
        self._proc = self._popen(args, cwd=str(self.run_dir),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 **_popen_group_kwargs())
        out_chunks = []
        if getattr(self._proc, "stdout", None) is not None:
            threading.Thread(target=_drain_pipe,
                             args=(self._proc.stdout, out_chunks),
                             daemon=True).start()
        log_path = script_log_path(self.run_dir, self.job_name)
        odb = self.run_dir / (self.job_name + ".odb")
        offset = 0
        next_read = self._clock() + self.interval
        while self._proc.poll() is None and not self._stop:
            offset = self._tail(log_path, offset)
            if self._clock() >= next_read and odb.is_file():
                self._snapshot(cfg_path)
                next_read = self._clock() + self.interval
            self._sleep(0.5)
        self._tail(log_path, offset)
        # A stopped run: one last snapshot of what it wrote before the stop.
        if self._stop:
            try:
                self._proc.wait(timeout=60)
            except Exception:
                pass
            if odb.is_file():
                self._snapshot(cfg_path)

    def _tail(self, log_path, offset):
        try:
            with open(log_path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
        except OSError:
            return offset
        if chunk:
            self.log.emit(chunk.decode("utf-8", errors="replace"))
        return offset + len(chunk)

    def _snapshot(self, cfg_path):
        t0 = self._clock()
        argv = snapshot_args(self._abaqus_cmd, self._script_dir,
                             self.job_name, cfg_path)
        try:
            done = self._run_cmd(argv, cwd=str(self.run_dir),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=900,
                                  **_popen_group_kwargs())
            rc, text = done.returncode, done.stdout.decode(
                "utf-8", errors="replace")
        except Exception as e:
            rc, text = -1, str(e)
        dt = self._clock() - t0
        path = live_bundle_path(self.run_dir, self.job_name)
        if rc == 0 and path.is_file():
            self.n_snapshots += 1
            self.snapshot.emit(str(path), float(dt))
        else:
            tail = "\n".join(text.strip().splitlines()[-3:])
            self.log.emit("[live read] failed after %.0f s (code %s): %s\n"
                          % (dt, rc, tail))
