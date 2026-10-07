# -*- coding: utf-8 -*-
"""
Remote execution of Abaqus runs through a job queue on a shared drive.

Why a queue folder and not SSH/sockets: the GUI PC and the compute PC share a
network drive (Z:) but the user is not administrator on the compute PC, so no
SSH server, no Windows share of his own and no firewall rule can be added.
A plain folder both machines can read and write needs none of that.

Two sides:

* AGENT, on the compute PC: gui/core/remote_agent.py, a standalone script run
  with the Python bundled with Abaqus, so NOTHING is installed there. The GUI
  copies it into the queue folder with a ready-made start_agent.bat
  (`deploy_queue`); on the compute PC the user double-clicks
  Z:\\...\\queue\\start_agent.bat in a Remote Desktop session and leaves it open.
  It runs the queued runs one at a time in a LOCAL folder of the compute PC
  and copies .sta/.gui.log back live and the results at the end.

* CLIENT, in the GUI (this module, Python 3.9 compatible): `RemoteProcess`
  stands in for the `subprocess.Popen` of a local run (poll/wait/returncode/
  stdout), so the existing run loops -- which already tail <job>.gui.log,
  poll <job>.sta and load <job>.results.npz from the run folder -- work
  unchanged as long as that run folder is on the shared drive.

The model generator (abaqus_scripts/*.py) also comes from the GUI PC: each
submission copies it to scripts/<hash>/ in the queue folder (once per
version) and the agent runs that copy, so the two PCs cannot run different
versions of it.

Queue layout (all under the queue folder)::

    pending/<id>.json   request, written by the client
    running/<id>.json   the same request, once the agent has claimed it
    cancel/<id>         cancel request, written by the client
    done/<id>.json      outcome, written by the agent, read and removed by the client
    agent.json          agent heartbeat (a counter that changes every ~2 s)
    scripts/<hash>/     model generator copied from the GUI PC
    remote_agent.py, start_agent.bat   the agent, written by the GUI

Every request and outcome is written to a temporary name first and renamed into place, so a
reader never sees half a request or half an outcome. The claim is a rename of
pending/<id>.json, and a cancel before the claim is a removal of the same
file: whichever happens first wins, the other gets FileNotFoundError.

What comes back to the shared run folder (see RESULT_SUFFIXES in remote_agent.py): the results
bundle, its metadata, the .sta, the script log, and the small Abaqus text
files. The .odb and the restart/scratch files stay on the compute PC, in the
local folder named in the outcome and in the run log.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import ntpath
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

HEARTBEAT_PERIOD = 2.0          # s, agent heartbeat
ALIVE_TIMEOUT = 15.0            # s, client wait for a heartbeat change. Kept
                                # well above the period: Windows caches file
                                # metadata on network drives for up to ~10 s.
AGENT_SOURCE = Path(__file__).with_name("remote_agent.py")


class RemoteError(RuntimeError):
    """The remote run could not be submitted (agent not running, bad setup)."""


# ---------------------------------------------------------------------------
# helpers shared by both sides
# ---------------------------------------------------------------------------
def scripts_fingerprint(scripts_dir) -> str:
    """Hash of the Abaqus-side scripts (*.py in `scripts_dir`): names the
    folder they are staged in, so a new version gets a new folder. Line
    endings are normalised (a CRLF/LF checkout is the same version)."""
    h = hashlib.sha256()
    d = Path(scripts_dir)
    for p in sorted(d.glob("*.py"), key=lambda q: q.name.lower()):
        h.update(p.name.lower().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp-%s" % uuid.uuid4().hex[:8])
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(str(tmp), str(path))


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _queue_dirs(queue) -> dict:
    q = Path(queue)
    dirs = {name: q / name for name in ("pending", "running", "cancel", "done")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def check_remote_setup(queue_dir: str, workdir: str, scripts_dir) -> list:
    """Problems that make a remote run impossible, as user-readable lines."""
    problems = []
    if not (queue_dir or "").strip():
        problems.append("Remote execution is on but no queue folder is set "
                        "(Preferences > Execution).")
        return problems
    # ntpath, not Path: the check is about Windows drives whatever the OS.
    q_drive = ntpath.splitdrive(queue_dir)[0]
    wd_drive = ntpath.splitdrive(str(workdir))[0]
    if q_drive and q_drive.upper() != wd_drive.upper():
        problems.append(
            "In remote mode the working directory must be on the shared drive "
            "%s (it is %s), otherwise the results cannot come back."
            % (q_drive, workdir))
    if " " in queue_dir:
        problems.append("The queue folder path must not contain spaces "
                        "(Abaqus noGUI= does not accept them): %s" % queue_dir)
    if not Path(scripts_dir).is_dir():
        problems.append("Abaqus scripts folder not found: %s" % scripts_dir)
    return problems


def launch_problems(prefs, workdir) -> list:
    """Pre-flight for a run, local or remote, as user-readable lines. In
    remote mode the local Abaqus command is not needed (the compute PC uses
    its own) but the local scripts are, for the version check."""
    problems = []
    if not Path(prefs.abaqus_script).exists():
        problems.append("Script not found: %s" % prefs.abaqus_script)
    if is_remote(prefs):
        problems += check_remote_setup(prefs.remote_queue_dir, str(workdir),
                                       Path(prefs.abaqus_script).parent)
    elif not Path(prefs.abaqus_cmd).exists():
        problems.append("Abaqus command not found: %s" % prefs.abaqus_cmd)
    return problems


def stage_scripts(queue, scripts_dir) -> str:
    """Copy the model generator (scripts_dir/*.py) to scripts/<hash>/ in the
    queue folder, once per version; return that folder. Written to a
    temporary folder then renamed, so the agent never sees half a copy."""
    fp = scripts_fingerprint(scripts_dir)
    dest = Path(queue) / "scripts" / fp
    if not (dest / "run_simul.py").is_file():
        tmp = dest.with_name(fp + ".tmp-%s" % uuid.uuid4().hex[:8])
        tmp.mkdir(parents=True)
        for p in Path(scripts_dir).glob("*.py"):
            shutil.copyfile(str(p), str(tmp / p.name))
        try:
            os.rename(str(tmp), str(dest))
        except OSError:          # another submission staged it meanwhile
            shutil.rmtree(str(tmp), ignore_errors=True)
    return str(dest)


START_AGENT_BAT = r"""@echo off
rem Written by the GUI (gui/core/remote_exec.py) - edit the Preferences, not
rem this file. Double-click it on the COMPUTE PC, in its Remote Desktop
rem session, and leave the window open. Close Remote Desktop with the cross
rem (disconnect): signing out would stop the agent.
rem Nothing to install: the agent runs with the Python bundled with Abaqus.
set "ABQ={abaqus}"
if not exist "%ABQ%" (
  echo Abaqus not found on this PC: %ABQ%
  echo Set "Abaqus command on the compute PC" in the GUI Preferences.
  pause
  exit /b 2
)
call "%ABQ%" python "%~dp0remote_agent.py" --abaqus "%ABQ%"
pause
"""


def deploy_queue(queue, remote_abaqus_cmd: str) -> Path:
    """Create the queue folder and (re)write the agent and its launcher in it
    when they differ from this GUI's version. Returns the launcher's path."""
    q = Path(queue)
    _queue_dirs(q)
    agent = q / "remote_agent.py"
    src = AGENT_SOURCE.read_bytes()
    if not agent.is_file() or agent.read_bytes() != src:
        tmp = agent.with_name("remote_agent.py.tmp-%s" % uuid.uuid4().hex[:8])
        tmp.write_bytes(src)
        os.replace(str(tmp), str(agent))
    bat = q / "start_agent.bat"
    text = START_AGENT_BAT.replace("{abaqus}", remote_abaqus_cmd or "")
    data = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("cp1252")
    if not bat.is_file() or bat.read_bytes() != data:
        bat.write_bytes(data)
    return bat


def agent_alive(queue, timeout: Optional[float] = None) -> Optional[dict]:
    """The agent heartbeat if the agent is running (its counter changed within
    `timeout`, default ALIVE_TIMEOUT), else None. Returns as soon as a change
    is seen."""
    if timeout is None:
        timeout = ALIVE_TIMEOUT
    hb_path = Path(queue) / "agent.json"
    first = _read_json(hb_path)
    seq0 = first.get("seq") if first else None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.5)
        hb = _read_json(hb_path)
        if hb is not None and hb.get("seq") != seq0:
            return hb
    return None


# ---------------------------------------------------------------------------
# client side
# ---------------------------------------------------------------------------
class _RemoteStdout:
    """Popen.stdout look-alike. The launcher output (licence banner, a fatal
    error before the script starts) only exists once the agent reports the
    outcome, so read()/read1() wait for it, return it once, then b""."""

    def __init__(self, proc: "RemoteProcess"):
        self._proc = proc
        self._given = False

    def read(self, n: int = -1) -> bytes:
        while self._proc.poll() is None:
            time.sleep(0.5)
        if self._given:
            return b""
        self._given = True
        return self._proc.launcher_output

    read1 = read


class RemoteProcess:
    """A run submitted to the remote agent, used where a Popen was.

    Supports what the GUI's run loops use: poll(), wait(timeout),
    returncode, stdout.read(), pid (None: there is no local process).
    cancel() replaces `abaqus terminate` + tree kill: the agent does both on
    its side."""

    pid = None

    def __init__(self, queue, run_dir, model_params: dict, run_params: dict,
                 scripts_dir, remote_abaqus_cmd: Optional[str] = None,
                 check_agent: bool = True):
        self.queue = Path(queue)
        self.dirs = _queue_dirs(self.queue)
        if remote_abaqus_cmd is not None:
            bat = deploy_queue(self.queue, remote_abaqus_cmd)
        else:
            bat = self.queue / "start_agent.bat"
        staged = stage_scripts(self.queue, scripts_dir)
        if check_agent and agent_alive(self.queue) is None:
            raise RemoteError(
                "the remote agent is not running (no heartbeat in %s). On the "
                "compute PC, double-click %s and leave its window open."
                % (self.queue, bat))
        self.job_name = str(run_params.get("job_name", "job"))
        now = time.time()        # ids sort in submission order (ms)
        self.id = "%s%03d-%s-%s" % (time.strftime("%Y%m%d-%H%M%S",
                                                  time.localtime(now)),
                                    int(now * 1000) % 1000,
                                    uuid.uuid4().hex[:6], self.job_name)
        self.returncode = None
        self.cancelled = False
        self.error = ""
        self.local_run_dir = ""
        self.launcher_output = b""
        self.stdout = _RemoteStdout(self)
        self._lock = threading.Lock()
        request = {
            "id": self.id,
            "job_name": self.job_name,
            "run_dir": str(run_dir),
            "model_params": repr(model_params),
            "run_params": repr(run_params),
            "scripts_dir": staged,
            "client_host": socket.gethostname(),
            "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json_atomic(self.dirs["pending"] / (self.id + ".json"), request)

    # -- Popen-like API ----------------------------------------------------
    def poll(self):
        with self._lock:
            if self.returncode is not None:
                return self.returncode
            done = self.dirs["done"] / (self.id + ".json")
            out = _read_json(done) if done.exists() else None
            if out is None:
                return None
            self.cancelled = bool(out.get("cancelled"))
            self.error = out.get("error") or ""
            self.local_run_dir = out.get("local_run_dir") or ""
            text = out.get("launcher_output") or ""
            if self.error:
                text += "\n[REMOTE] %s\n" % self.error
            if self.local_run_dir:
                text += ("\n[REMOTE] job files (.odb ...) kept on %s in %s\n"
                         % (out.get("agent_host", "the compute PC"),
                            self.local_run_dir))
            self.launcher_output = text.encode("cp1252", errors="replace")
            rc = out.get("returncode")
            self.returncode = int(rc) if rc is not None else 1
            try:
                done.unlink()
            except OSError:
                pass
            return self.returncode

    def wait(self, timeout: Optional[float] = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                raise subprocess.TimeoutExpired("remote:%s" % self.id, timeout)
            time.sleep(0.5)
        return self.returncode

    def cancel(self) -> None:
        """Stop the run. Before the agent has claimed it, the request is simply
        withdrawn; after, the agent terminates Abaqus and reports back."""
        if self.poll() is not None:
            return
        try:
            (self.dirs["cancel"] / self.id).write_text("cancel")
        except OSError:
            log.warning("could not write the cancel request for %s", self.id)
        try:
            (self.dirs["pending"] / (self.id + ".json")).unlink()
        except FileNotFoundError:
            return              # claimed: the agent will answer in done/
        except OSError:
            return
        with self._lock:        # withdrawn before the claim
            self.cancelled = True
            self.returncode = 1
            self.launcher_output = b"[REMOTE] cancelled before it started\n"
        try:
            (self.dirs["cancel"] / self.id).unlink()
        except OSError:
            pass

    terminate = cancel
    kill = cancel


def submit_remote(prefs, run_dir, model_params: dict, run_params: dict,
                  check_agent: bool = True) -> RemoteProcess:
    """Submit one run with the settings of `prefs` (Preferences)."""
    return RemoteProcess(prefs.remote_queue_dir, run_dir, model_params,
                         run_params, Path(prefs.abaqus_script).parent,
                         remote_abaqus_cmd=getattr(prefs, "remote_abaqus_cmd",
                                                   None),
                         check_agent=check_agent)


def is_remote(prefs) -> bool:
    return getattr(prefs, "execution_mode", "local") == "remote"


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _cmd_submit(ns) -> int:
    """Blocking client used by the Job tab (through QProcess): submit, wait,
    print the launcher output, exit with the remote return code. Cancelled
    by writing the cancel file (request_cancel), not by killing this process."""
    try:
        proc = RemoteProcess(ns.queue, ns.run_dir,
                             ast.literal_eval(ns.model_cfg),
                             ast.literal_eval(ns.run_cfg), ns.scripts_dir,
                             remote_abaqus_cmd=ns.remote_abaqus or None)
    except Exception as e:
        print("[REMOTE] %s" % e)
        return 3
    print("[REMOTE] submitted %s to %s" % (proc.id, ns.queue))
    sys.stdout.flush()
    rc = proc.wait()
    sys.stdout.write(proc.launcher_output.decode("cp1252", errors="replace"))
    if proc.cancelled:
        print("[REMOTE] cancelled")
    return rc


def submit_command(prefs, run_dir, model_params: dict,
                   run_params: dict) -> tuple:
    """(program, args, env_pythonpath) to run `_cmd_submit` as a child process
    (the Job tab drives a QProcess). The console interpreter is used even when
    the GUI runs under pythonw.exe, whose output would otherwise be lost."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        exe = exe.parent / "python.exe"
    args = ["-u", "-m", "gui.core.remote_exec", "submit",
            "--queue", str(prefs.remote_queue_dir),
            "--run-dir", str(run_dir),
            "--scripts-dir", str(Path(prefs.abaqus_script).parent),
            "--model-cfg", repr(model_params),
            "--run-cfg", repr(run_params),
            "--remote-abaqus", str(getattr(prefs, "remote_abaqus_cmd", ""))]
    repo_root = str(Path(__file__).resolve().parents[2])
    return str(exe), args, repo_root


def request_cancel_by_job(queue, job_name: str) -> bool:
    """Cancel the queued or running request(s) of `job_name` (Job tab path,
    where the RemoteProcess lives in the child client process). Returns True
    if a request was found."""
    dirs = _queue_dirs(queue)
    found = False
    for sub in ("pending", "running"):
        for p in dirs[sub].glob("*.json"):
            rid = p.stem
            # id = <date>-<time>-<hex>-<job name>
            if rid.split("-", 3)[-1] != job_name:
                continue
            found = True
            try:
                (dirs["cancel"] / rid).write_text("cancel")
            except OSError:
                continue
            if sub == "pending":
                try:
                    p.unlink()
                except OSError:
                    continue        # claimed meanwhile: the agent answers
                _write_json_atomic(dirs["done"] / (rid + ".json"), {
                    "returncode": 1, "cancelled": True,
                    "launcher_output": "[REMOTE] cancelled before it started\n"})
                try:
                    (dirs["cancel"] / rid).unlink()
                except OSError:
                    pass
    return found


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m gui.core.remote_exec")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("submit", help="submit one run and wait (Job tab)")
    s.add_argument("--queue", required=True)
    s.add_argument("--run-dir", required=True)
    s.add_argument("--scripts-dir", required=True)
    s.add_argument("--model-cfg", required=True)
    s.add_argument("--run-cfg", required=True)
    s.add_argument("--remote-abaqus", default="")
    ns = ap.parse_args(argv)
    if ns.cmd == "submit":
        return _cmd_submit(ns)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
