# -*- coding: utf-8 -*-
"""
Remote execution of Abaqus runs through a job queue on a shared drive.

Why a queue folder and not SSH/sockets: the GUI PC and the compute PC share a
network drive (Z:) but the user is not administrator on the compute PC, so no
SSH server, no Windows share of his own and no firewall rule can be added.
A plain folder both machines can read and write needs none of that.

Two sides, one module (Qt-free, Python 3.9 compatible -- the compute PC only
has Anaconda Python 3.9):

* AGENT, on the compute PC, started by hand in its Remote Desktop session::

      python -m gui.core.remote_exec agent --queue Z:\\ABQ_remote\\queue

  It takes the oldest request, runs Abaqus in a LOCAL folder of the compute PC
  (never on the network drive: the solver's scratch and .odb I/O stay local),
  copies <job>.sta and <job>.gui.log to the shared run folder every couple of
  seconds while it runs, then copies the result files back and publishes the
  outcome. One run at a time, in submission order.

* CLIENT, in the GUI: `RemoteProcess` stands in for the `subprocess.Popen` of
  a local run (poll/wait/returncode/stdout), so the existing run loops --
  which already tail <job>.gui.log, poll <job>.sta and load
  <job>.results.npz from the run folder -- work unchanged as long as that
  run folder is on the shared drive.

Queue layout (all under the queue folder)::

    pending/<id>.json   request, written by the client
    running/<id>.json   the same request, once the agent has claimed it
    cancel/<id>         cancel request, written by the client
    done/<id>.json      outcome, written by the agent, read and removed by the client
    agent.json          agent heartbeat (a counter that changes every ~2 s)

Every file is written to a temporary name first and renamed into place, so a
reader never sees half a request or half an outcome. The claim is a rename of
pending/<id>.json, and a cancel before the claim is a removal of the same
file: whichever happens first wins, the other gets FileNotFoundError.

What comes back to the shared run folder (see RESULT_SUFFIXES): the results
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
MIRROR_PERIOD = 2.0             # s, live copy of .sta / .gui.log
CANCEL_GRACE = 10.0             # s, after `abaqus terminate` before a tree kill
DEFAULT_LOCAL_ROOT = r"C:\TEMP\ABQ_remote"

# Copied back while the job runs (the GUI shows progress from them).
LIVE_SUFFIXES = (".sta", ".gui.log")
# Copied back once the job has ended.
RESULT_SUFFIXES = (".results.npz", ".meta.json", ".sta", ".gui.log",
                   ".msg", ".dat", ".log", ".inp")


class RemoteError(RuntimeError):
    """The remote run could not be submitted (agent not running, bad setup)."""


# ---------------------------------------------------------------------------
# helpers shared by both sides
# ---------------------------------------------------------------------------
def scripts_fingerprint(scripts_dir) -> str:
    """Hash of the Abaqus-side scripts (*.py in `scripts_dir`).

    Both PCs must run the same model generator, otherwise a run submitted from
    one checkout is built by another version of the code. Line endings are
    normalised so a CRLF/LF checkout difference does not count as a change."""
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


def local_run_dir(local_root, shared_run_dir) -> Path:
    """Folder of the compute PC where a run of `shared_run_dir` executes:
    the shared path with its drive replaced by `local_root`
    (Z:\\ABQ\\study_x -> C:\\TEMP\\ABQ_remote\\ABQ\\study_x)."""
    p = Path(shared_run_dir)
    parts = p.parts[1:] if p.anchor else p.parts
    return Path(local_root).joinpath(*parts)


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
                 scripts_dir, check_agent: bool = True):
        self.queue = Path(queue)
        self.dirs = _queue_dirs(self.queue)
        if check_agent and agent_alive(self.queue) is None:
            raise RemoteError(
                "the remote agent is not running (no heartbeat in %s). Start "
                "run_remote_agent.bat on the compute PC." % self.queue)
        self.job_name = str(run_params.get("job_name", "job"))
        self.id = "%s-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"),
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
            "scripts_fingerprint": scripts_fingerprint(scripts_dir),
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
                         check_agent=check_agent)


def is_remote(prefs) -> bool:
    return getattr(prefs, "execution_mode", "local") == "remote"


# ---------------------------------------------------------------------------
# agent side
# ---------------------------------------------------------------------------
def _copy_if_changed(src: Path, dst: Path, seen: dict) -> None:
    try:
        st = src.stat()
    except OSError:
        return
    key = (st.st_size, st.st_mtime)
    if seen.get(src.name) == key:
        return
    try:
        shutil.copyfile(str(src), str(dst))
        seen[src.name] = key
    except OSError:
        pass                    # file busy: next tick


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp-%s" % uuid.uuid4().hex[:8])
    shutil.copyfile(str(src), str(tmp))
    os.replace(str(tmp), str(dst))


def _default_build_args(abaqus_cmd, abaqus_script, model_params, run_params):
    from gui.sensitivity.run_worker import build_abaqus_args
    return build_abaqus_args(abaqus_cmd, abaqus_script, model_params,
                             run_params)


class RemoteAgent:
    """The compute-PC side. `build_args`, `terminate` and `kill_tree` are
    injectable so the loop can be tested without Abaqus."""

    def __init__(self, queue, abaqus_cmd: str, abaqus_script: str,
                 local_root: str = DEFAULT_LOCAL_ROOT, build_args=None,
                 terminate=None, kill_tree=None, out=None):
        self.queue = Path(queue)
        self.dirs = _queue_dirs(self.queue)
        self.abaqus_cmd = abaqus_cmd
        self.abaqus_script = abaqus_script
        self.local_root = Path(local_root)
        self.build_args = build_args or _default_build_args
        self._terminate = terminate
        self._kill_tree = kill_tree
        self.out = out or sys.stdout
        self.host = socket.gethostname()
        self.fingerprint = scripts_fingerprint(Path(abaqus_script).parent)
        self.seq = 0
        self._last_hb = 0.0
        self.job = None         # dict describing the run in flight

    # -- small utilities ---------------------------------------------------
    def say(self, msg: str) -> None:
        self.out.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        self.out.flush()

    def heartbeat(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_hb < HEARTBEAT_PERIOD:
            return
        self._last_hb = now
        self.seq += 1
        try:
            _write_json_atomic(self.queue / "agent.json", {
                "seq": self.seq, "host": self.host, "pid": os.getpid(),
                "scripts_fingerprint": self.fingerprint,
                "busy_job": self.job["id"] if self.job else None,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        except OSError as e:
            self.say("WARNING: heartbeat not written (%s)" % e)

    def _publish(self, rid: str, outcome: dict) -> None:
        outcome.setdefault("agent_host", self.host)
        _write_json_atomic(self.dirs["done"] / (rid + ".json"), outcome)
        for p in (self.dirs["running"] / (rid + ".json"),
                  self.dirs["cancel"] / rid):
            try:
                p.unlink()
            except OSError:
                pass

    def recover(self) -> None:
        """Runs left in running/ by an agent that died: report them failed."""
        for p in sorted(self.dirs["running"].glob("*.json")):
            rid = p.stem
            self.say("run %s was interrupted by an agent restart" % rid)
            self._publish(rid, {"returncode": 1, "cancelled": False,
                                "error": "the remote agent was restarted "
                                         "while this run was in progress"})

    # -- one step of the loop ---------------------------------------------
    def step(self) -> None:
        self.heartbeat()
        if self.job is None:
            self._start_next()
        else:
            self._follow()

    def run_forever(self, period: float = 1.0) -> None:
        self.recover()
        self.heartbeat(force=True)
        self.say("agent ready on %s, queue %s, local runs in %s"
                 % (self.host, self.queue, self.local_root))
        self.say("scripts fingerprint %s (%s)"
                 % (self.fingerprint, Path(self.abaqus_script).parent))
        while True:
            try:
                self.step()
            except Exception as e:         # keep the agent alive
                self.say("ERROR in agent loop: %r" % e)
            time.sleep(period)

    def _start_next(self) -> None:
        pending = sorted(self.dirs["pending"].glob("*.json"))
        for p in pending:
            rid = p.stem
            claimed = self.dirs["running"] / p.name
            try:
                os.replace(str(p), str(claimed))
            except OSError:
                continue        # withdrawn by the client, or not readable yet
            req = _read_json(claimed)
            if req is None:
                self._publish(rid, {"returncode": 1, "error":
                                    "unreadable request"})
                continue
            self._launch(rid, req)
            return

    def _launch(self, rid: str, req: dict) -> None:
        job = req.get("job_name", "job")
        shared = Path(req["run_dir"])
        if req.get("scripts_fingerprint") != self.fingerprint:
            msg = ("the Abaqus scripts differ between the two PCs (GUI %s, "
                   "compute PC %s). Pull the same version of the repository "
                   "on both, restart the agent, and run again."
                   % (req.get("scripts_fingerprint"), self.fingerprint))
            self.say("refused %s: %s" % (job, msg))
            self._publish(rid, {"returncode": 1, "error": msg})
            return
        local = local_run_dir(self.local_root, shared)
        try:
            local.mkdir(parents=True, exist_ok=True)
            shared.mkdir(parents=True, exist_ok=True)
            low = job.lower() + "."
            for f in local.iterdir():   # a previous run of the same name
                if f.is_file() and f.name.lower().startswith(low):
                    f.unlink()
            args = self.build_args(self.abaqus_cmd, self.abaqus_script,
                                   ast.literal_eval(req["model_params"]),
                                   ast.literal_eval(req["run_params"]))
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            proc = subprocess.Popen(args, cwd=str(local),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, **kwargs)
        except Exception as e:
            self.say("could not start %s: %r" % (job, e))
            self._publish(rid, {"returncode": 1, "error":
                                "could not start Abaqus on the compute PC: %s"
                                % e, "local_run_dir": str(local)})
            return
        chunks = []

        def _drain():
            try:
                for chunk in iter(lambda: proc.stdout.read(65536), b""):
                    chunks.append(chunk)
            except Exception:
                pass
        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()
        self.job = {"id": rid, "name": job, "proc": proc, "local": local,
                    "shared": shared, "chunks": chunks, "reader": reader,
                    "seen": {}, "last_mirror": 0.0, "cancel_at": None,
                    "killed": False}
        self.say("started %s (from %s) in %s"
                 % (job, req.get("client_host", "?"), local))

    def _mirror(self, suffixes) -> None:
        j = self.job
        for suf in suffixes:
            src = j["local"] / (j["name"] + suf)
            _copy_if_changed(src, j["shared"] / src.name, j["seen"])

    def _follow(self) -> None:
        j = self.job
        now = time.monotonic()
        if now - j["last_mirror"] >= MIRROR_PERIOD:
            j["last_mirror"] = now
            self._mirror(LIVE_SUFFIXES)
        proc = j["proc"]
        if proc.poll() is None:
            if j["cancel_at"] is None and (self.dirs["cancel"] / j["id"]).exists():
                j["cancel_at"] = now
                self.say("cancel requested for %s" % j["name"])
                if not self._do_terminate(j):
                    self._do_kill(j)
            elif (j["cancel_at"] is not None and not j["killed"]
                  and now - j["cancel_at"] > CANCEL_GRACE):
                self._do_kill(j)
            return
        self._finish()

    def _do_terminate(self, j) -> bool:
        if self._terminate is not None:
            return bool(self._terminate(j))
        from gui.sensitivity.run_worker import abaqus_terminate_job
        return abaqus_terminate_job(self.abaqus_cmd, j["name"], j["local"])

    def _do_kill(self, j) -> None:
        j["killed"] = True
        if self._kill_tree is not None:
            self._kill_tree(j)
            return
        from gui.sensitivity.run_worker import _terminate_process_tree
        _terminate_process_tree(j["proc"])

    def _finish(self) -> None:
        j = self.job
        j["reader"].join(timeout=5.0)
        copied = []
        for suf in RESULT_SUFFIXES:
            src = j["local"] / (j["name"] + suf)
            if src.is_file():
                try:
                    _copy_atomic(src, j["shared"] / src.name)
                    copied.append(src.name)
                except OSError as e:
                    self.say("could not copy back %s: %s" % (src.name, e))
        rc = j["proc"].returncode
        cancelled = j["cancel_at"] is not None
        self._publish(j["id"], {
            "returncode": rc if rc is not None else 1,
            "cancelled": cancelled,
            "launcher_output": b"".join(j["chunks"]).decode(
                "cp1252", errors="replace"),
            "local_run_dir": str(j["local"]),
            "copied": copied})
        self.say("%s %s (rc=%s), copied back: %s"
                 % ("cancelled" if cancelled else "finished", j["name"], rc,
                    ", ".join(copied) or "nothing"))
        self.job = None


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _cmd_agent(ns) -> int:
    from gui.core.preferences import load_preferences
    prefs = load_preferences()
    cmd = ns.abaqus_cmd or prefs.abaqus_cmd
    script = ns.abaqus_script or prefs.abaqus_script
    queue = ns.queue or getattr(prefs, "remote_queue_dir", "")
    if not queue:
        print("No queue folder: pass --queue Z:\\...\\queue", file=sys.stderr)
        return 2
    for label, p in (("Abaqus command", cmd), ("Abaqus script", script)):
        if not Path(p).exists():
            print("%s not found: %s (set it in the GUI Preferences on this "
                  "PC, or pass it on the command line)" % (label, p),
                  file=sys.stderr)
            return 2
    RemoteAgent(queue, cmd, script, ns.local_root).run_forever()
    return 0


def _cmd_submit(ns) -> int:
    """Blocking client used by the Job tab (through QProcess): submit, wait,
    print the launcher output, exit with the remote return code. Cancelled
    by writing the cancel file (request_cancel), not by killing this process."""
    try:
        proc = RemoteProcess(ns.queue, ns.run_dir,
                             ast.literal_eval(ns.model_cfg),
                             ast.literal_eval(ns.run_cfg), ns.scripts_dir)
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
            "--run-cfg", repr(run_params)]
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
    a = sub.add_parser("agent", help="run the agent on the compute PC")
    a.add_argument("--queue", default="")
    a.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    a.add_argument("--abaqus-cmd", default="")
    a.add_argument("--abaqus-script", default="")
    s = sub.add_parser("submit", help="submit one run and wait (Job tab)")
    s.add_argument("--queue", required=True)
    s.add_argument("--run-dir", required=True)
    s.add_argument("--scripts-dir", required=True)
    s.add_argument("--model-cfg", required=True)
    s.add_argument("--run-cfg", required=True)
    ns = ap.parse_args(argv)
    if ns.cmd == "agent":
        return _cmd_agent(ns)
    if ns.cmd == "submit":
        return _cmd_submit(ns)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
