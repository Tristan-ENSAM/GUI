# -*- coding: utf-8 -*-
"""
Remote-execution agent, run on the COMPUTE PC with the Python bundled with
Abaqus -- nothing to install there:

    "C:\\SIMULIA\\Commands\\abaqus.bat" python remote_agent.py --abaqus <abaqus.bat>

The GUI copies this file and a ready-made start_agent.bat into the queue
folder on the shared drive (gui.core.remote_exec.deploy_queue), so on the
compute PC the user only double-clicks Z:\\...\\queue\\start_agent.bat.

STANDALONE ON PURPOSE: it imports nothing from the GUI package and runs
under Python 2.7 (the interpreter of Abaqus 2022 and earlier) as well as
Python 3 (later Abaqus releases, and the tests). Hence no f-strings, no
pathlib, no os.replace, no subprocess.run, no FileNotFoundError.

What it does: takes the oldest request in pending/, runs Abaqus in a LOCAL
folder of the compute PC (the solver's scratch and .odb I/O never go over the
network), copies <job>.sta and <job>.gui.log to the shared run folder every
couple of seconds, then copies the result files back and writes the outcome
in done/. One run at a time, in submission order. The model generator
(run_simul.py and its modules) is the copy the GUI put in the queue folder
for this request, so both PCs always run the same scripts.

Protocol: see gui/core/remote_exec.py (client side).
"""
from __future__ import print_function

import argparse
import ast
import errno
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid

HEARTBEAT_PERIOD = 2.0          # s
MIRROR_PERIOD = 2.0             # s, live copy of .sta / .gui.log
CANCEL_GRACE = 10.0             # s, after `abaqus terminate` before a tree kill
DEFAULT_LOCAL_ROOT = r"C:\TEMP\ABQ_remote"

LIVE_SUFFIXES = (".sta", ".gui.log")
RESULT_SUFFIXES = (".results.npz", ".meta.json", ".sta", ".gui.log",
                   ".msg", ".dat", ".log", ".inp")
SUBDIRS = ("pending", "running", "cancel", "done")


# ---------------------------------------------------------------------------
# small file helpers (Python 2 and 3)
# ---------------------------------------------------------------------------
def _makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _remove(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _replace(src, dst):
    """os.replace for Python 2: on Windows os.rename refuses an existing
    destination, so remove it first. Readers tolerate the instant in between
    (a missing file reads as "not there yet")."""
    try:
        os.rename(src, dst)
    except OSError:
        _remove(dst)
        os.rename(src, dst)


def _tmp_name(path):
    return "%s.tmp-%s" % (path, uuid.uuid4().hex[:8])


def write_json_atomic(path, data):
    tmp = _tmp_name(path)
    text = json.dumps(data, indent=1)
    if not isinstance(text, type(u"")):
        text = text.decode("utf-8")
    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    _replace(tmp, path)


def read_json(path):
    try:
        with io.open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (IOError, OSError, ValueError):
        return None


def local_run_dir(local_root, shared_run_dir):
    """Z:\\ABQ\\study_x -> <local_root>\\ABQ\\study_x (drive replaced)."""
    rest = os.path.splitdrive(shared_run_dir)[1]
    parts = [p for p in re.split(r"[\\/]", rest) if p]
    return os.path.join(local_root, *parts) if parts else local_root


def _copy_atomic(src, dst):
    tmp = _tmp_name(dst)
    shutil.copyfile(src, tmp)
    _replace(tmp, dst)


# ---------------------------------------------------------------------------
# Abaqus process control
# ---------------------------------------------------------------------------
def build_abaqus_args(abaqus_cmd, script, model_params_repr, run_params_repr):
    """Same argv as gui.sensitivity.run_worker.build_abaqus_args; the dicts
    arrive already as repr() strings and are passed through untouched."""
    return [abaqus_cmd, "cae", "noGUI=%s" % script, "--",
            "--model_cfg", model_params_repr, "--run_cfg", run_params_repr]


def abaqus_terminate(abaqus_cmd, job_name, workdir, timeout=20.0):
    """`abaqus terminate job=<name>` from the job folder: stops the solver and
    releases the licence tokens. False when there is nothing to signal (no
    .cid) or Abaqus did not accept."""
    if not os.path.isfile(os.path.join(workdir, job_name + ".cid")):
        return False
    try:
        p = subprocess.Popen([abaqus_cmd, "terminate", "job=%s" % job_name],
                             cwd=workdir, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    except OSError:
        return False
    deadline = time.time() + timeout
    while p.poll() is None and time.time() < deadline:
        time.sleep(0.2)
    if p.poll() is None:
        try:
            p.kill()
        except OSError:
            pass
        return False
    return p.returncode == 0


def kill_tree(proc):
    """Kill the launcher AND the solver processes it spawned."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return
        except OSError:
            pass
    else:
        try:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        proc.kill()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------
class Agent(object):
    """`build_args`, `terminate` and `kill` are injectable for the tests."""

    def __init__(self, queue, abaqus_cmd, local_root=DEFAULT_LOCAL_ROOT,
                 build_args=None, terminate=None, kill=None, out=None):
        self.queue = queue
        self.abaqus_cmd = abaqus_cmd
        self.local_root = local_root
        self.build_args = build_args or build_abaqus_args
        self._terminate = terminate
        self._kill = kill
        self.out = out or sys.stdout
        self.host = socket.gethostname()
        self.seq = 0
        self._last_hb = 0.0
        self.job = None
        for d in SUBDIRS:
            _makedirs(self.sub(d))

    def sub(self, name, *rest):
        return os.path.join(self.queue, name, *rest)

    def say(self, msg):
        self.out.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        self.out.flush()

    def heartbeat(self, force=False):
        now = time.time()
        if not force and now - self._last_hb < HEARTBEAT_PERIOD:
            return
        self._last_hb = now
        self.seq += 1
        try:
            write_json_atomic(os.path.join(self.queue, "agent.json"), {
                "seq": self.seq, "host": self.host, "pid": os.getpid(),
                "python": sys.version.split()[0],
                "abaqus_cmd": self.abaqus_cmd,
                "busy_job": self.job["id"] if self.job else None,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        except (IOError, OSError) as e:
            self.say("WARNING: heartbeat not written (%s)" % e)

    def publish(self, rid, outcome):
        outcome.setdefault("agent_host", self.host)
        write_json_atomic(self.sub("done", rid + ".json"), outcome)
        _remove(self.sub("running", rid + ".json"))
        _remove(self.sub("cancel", rid))

    def recover(self):
        """Runs left in running/ by an agent that died: report them failed."""
        for name in sorted(os.listdir(self.sub("running"))):
            if name.endswith(".json"):
                rid = name[:-5]
                self.say("run %s was interrupted by an agent restart" % rid)
                self.publish(rid, {"returncode": 1, "cancelled": False,
                                   "error": "the remote agent was restarted "
                                            "while this run was in progress"})

    def step(self):
        self.heartbeat()
        if self.job is None:
            self._start_next()
        else:
            self._follow()

    def run_forever(self, period=1.0):
        self.recover()
        self.heartbeat(force=True)
        self.say("agent ready on %s (Python %s)" % (self.host,
                                                    sys.version.split()[0]))
        self.say("queue %s" % self.queue)
        self.say("Abaqus %s, local runs in %s" % (self.abaqus_cmd,
                                                   self.local_root))
        self.say("leave this window open; close Remote Desktop with the cross")
        while True:
            try:
                self.step()
            except Exception as e:          # keep the agent alive
                self.say("ERROR in agent loop: %r" % (e,))
            time.sleep(period)

    # -- one run ---------------------------------------------------------
    def _start_next(self):
        names = sorted(n for n in os.listdir(self.sub("pending"))
                       if n.endswith(".json"))
        for name in names:
            rid = name[:-5]
            claimed = self.sub("running", name)
            try:
                os.rename(self.sub("pending", name), claimed)
            except OSError:
                continue        # withdrawn by the client meanwhile
            req = read_json(claimed)
            if req is None:
                self.publish(rid, {"returncode": 1,
                                   "error": "unreadable request"})
                continue
            self._launch(rid, req)
            return

    def _launch(self, rid, req):
        job = req.get("job_name", "job")
        shared = req["run_dir"]
        local = local_run_dir(self.local_root, shared)
        script = os.path.join(req["scripts_dir"], "run_simul.py")
        try:
            if not os.path.isfile(script):
                raise IOError("model generator not found: %s" % script)
            _makedirs(local)
            _makedirs(shared)
            low = job.lower() + "."
            for f in os.listdir(local):     # a previous run of the same name
                p = os.path.join(local, f)
                if f.lower().startswith(low) and os.path.isfile(p):
                    _remove(p)
            args = self.build_args(self.abaqus_cmd, script,
                                   req["model_params"], req["run_params"])
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["preexec_fn"] = os.setsid
            proc = subprocess.Popen(args, cwd=local, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, **kwargs)
        except Exception as e:
            self.say("could not start %s: %r" % (job, e))
            self.publish(rid, {"returncode": 1, "local_run_dir": local,
                               "error": "could not start Abaqus on the "
                                        "compute PC: %s" % (e,)})
            return
        chunks = []

        def _drain():
            try:
                for chunk in iter(lambda: proc.stdout.read(65536), b""):
                    chunks.append(chunk)
            except Exception:
                pass
        reader = threading.Thread(target=_drain)
        reader.daemon = True
        reader.start()
        self.job = {"id": rid, "name": job, "proc": proc, "local": local,
                    "shared": shared, "chunks": chunks, "reader": reader,
                    "seen": {}, "last_mirror": 0.0, "cancel_at": None,
                    "killed": False}
        self.say("started %s (from %s) in %s"
                 % (job, req.get("client_host", "?"), local))

    def _mirror(self):
        j = self.job
        for suf in LIVE_SUFFIXES:
            src = os.path.join(j["local"], j["name"] + suf)
            try:
                st = os.stat(src)
            except OSError:
                continue
            key = (st.st_size, st.st_mtime)
            if j["seen"].get(suf) == key:
                continue
            try:
                shutil.copyfile(src, os.path.join(j["shared"],
                                                  j["name"] + suf))
                j["seen"][suf] = key
            except (IOError, OSError):
                pass            # busy: next tick

    def _follow(self):
        j = self.job
        now = time.time()
        if now - j["last_mirror"] >= MIRROR_PERIOD:
            j["last_mirror"] = now
            self._mirror()
        if j["proc"].poll() is None:
            if j["cancel_at"] is None \
                    and os.path.exists(self.sub("cancel", j["id"])):
                j["cancel_at"] = now
                self.say("cancel requested for %s" % j["name"])
                if not self._do_terminate(j):
                    self._do_kill(j)
            elif j["cancel_at"] is not None and not j["killed"] \
                    and now - j["cancel_at"] > CANCEL_GRACE:
                self._do_kill(j)
            return
        self._finish()

    def _do_terminate(self, j):
        if self._terminate is not None:
            return bool(self._terminate(j))
        return abaqus_terminate(self.abaqus_cmd, j["name"], j["local"])

    def _do_kill(self, j):
        j["killed"] = True
        if self._kill is not None:
            self._kill(j)
        else:
            kill_tree(j["proc"])

    def _finish(self):
        j = self.job
        j["reader"].join(5.0)
        copied = []
        for suf in RESULT_SUFFIXES:
            src = os.path.join(j["local"], j["name"] + suf)
            if os.path.isfile(src):
                try:
                    _copy_atomic(src, os.path.join(j["shared"],
                                                   j["name"] + suf))
                    copied.append(j["name"] + suf)
                except (IOError, OSError) as e:
                    self.say("could not copy back %s: %s"
                             % (j["name"] + suf, e))
        rc = j["proc"].returncode
        cancelled = j["cancel_at"] is not None
        self.publish(j["id"], {
            "returncode": rc if rc is not None else 1,
            "cancelled": cancelled,
            "launcher_output": b"".join(j["chunks"]).decode(
                "cp1252", "replace"),
            "local_run_dir": j["local"],
            "copied": copied})
        self.say("%s %s (rc=%s), copied back: %s"
                 % ("cancelled" if cancelled else "finished", j["name"], rc,
                    ", ".join(copied) or "nothing"))
        self.job = None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Abaqus remote-execution agent")
    ap.add_argument("--abaqus", required=True,
                    help="abaqus.bat of this PC")
    ap.add_argument("--queue", default="",
                    help="queue folder (default: the folder of this file)")
    ap.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    ns = ap.parse_args(argv)
    queue = ns.queue or os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(ns.abaqus):
        print("Abaqus command not found: %s" % ns.abaqus)
        print("Set 'Abaqus command on the compute PC' in the GUI Preferences "
              "(it rewrites start_agent.bat).")
        return 2
    Agent(queue, ns.abaqus, ns.local_root).run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
