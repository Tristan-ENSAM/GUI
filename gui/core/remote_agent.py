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
in done/. One run at a time, in submission order.

Delivery through Remote Desktop: when the request names the GUI PC's run
folder as seen from this session (client_dir, e.g. \\\\tsclient\\C\\TEMP\\...,
which exists while the user's Remote Desktop window is open with the C:
drive redirected) and the token file the GUI left there is visible, the
results AND the .odb are moved straight into that folder (copied, size
checked, then deleted here), so they never use the small shared drive.
Otherwise (window closed, drive not redirected, another PC connected, copy
cut by a lost connection) the results go through the shared drive as before,
so a study keeps going, and the .odb waits here in a backlog
(<local_root>/pending_deliveries.json, kept across agent restarts): every
BACKLOG_PERIOD the agent retries it, and moves it as soon as the GUI PC's
folder and token are visible again, i.e. when the user reconnects.

The model generator (run_simul.py and its modules) is the copy the GUI put
in the queue folder for this request, so both PCs always run the same
scripts.

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
BACKLOG_PERIOD = 30.0           # s, retry of the .odb not yet delivered
MIRROR_PERIOD = 2.0             # s, live copy of .sta / .gui.log
CANCEL_GRACE = 10.0             # s, after `abaqus terminate` before a tree kill
DEFAULT_LOCAL_ROOT = r"C:\TEMP\ABQ_remote"

LIVE_SUFFIXES = (".sta", ".gui.log")
RESULT_SUFFIXES = (".results.npz", ".meta.json", ".sta", ".gui.log",
                   ".msg", ".dat", ".log", ".inp")
# Also delivered when the GUI PC's folder is reachable (not through Z:).
DELIVERY_EXTRA_SUFFIXES = (".odb",)
SUBDIRS = ("pending", "running", "cancel", "done")
# CAE session journals (abaqus.rpy, abaqus.rpy.N): useless, removed from the
# run's local folder once Abaqus has exited (same rule as
# cel_common.is_replay_file; this agent imports nothing).
REPLAY_FILE = re.compile(r"^abaqus\.rpy(\.\d+)?$", re.IGNORECASE)


def token_name(rid):
    """File the client leaves in its run folder: seeing it through
    client_dir proves that folder is the submitting PC's, not the C: of
    whichever PC is connected to this session."""
    return ".remote-%s.token" % rid


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


def move_files(src_dir, names, dest_dir):
    """Copy each file to dest_dir, check its size there, and only once ALL of
    them are copied delete the sources. Raises (sources untouched) if any
    copy fails or comes out with the wrong size."""
    for name in names:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dest_dir, name)
        _copy_atomic(src, dst)
        if os.path.getsize(dst) != os.path.getsize(src):
            raise IOError("size mismatch after copying %s" % name)
    for name in names:
        _remove(os.path.join(src_dir, name))
    try:
        os.rmdir(src_dir)           # only if nothing else is left
    except OSError:
        pass


def _copy_atomic(src, dst):
    tmp = _tmp_name(dst)
    try:
        shutil.copyfile(src, tmp)
        _replace(tmp, dst)
    except Exception:
        _remove(tmp)
        raise


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
def _has_year(line):
    for tok in line.replace(".", " ").replace("-", " ").split():
        if len(tok) == 4 and tok.isdigit() and tok[:2] in ("19", "20"):
            return True
    return False


def parse_release(text):
    """Release named by `abaqus information=release` output: its first line
    naming Abaqus and a year, plus its build line when there is one; same
    rule as gui.core.run_db.parse_release (this file is standalone)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    head = next((ln for ln in lines
                 if "abaqus" in ln.lower() and _has_year(ln)), "")
    if not head:
        return ""
    build = next((ln for ln in lines if "build" in ln.lower()), "")
    return head if not build or build == head else "%s | %s" % (head, build)


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
        # `abaqus information=release`, read once at start (read_release):
        # the GUI files the runs of this PC under it (gui.core.run_db).
        self.abaqus_release = ""
        self.seq = 0
        self._last_hb = 0.0
        self.job = None
        for d in SUBDIRS:
            _makedirs(self.sub(d))
        self._backlog_lock = threading.Lock()
        self._backlog_path = os.path.join(local_root,
                                          "pending_deliveries.json")
        self.backlog = read_json(self._backlog_path) or []
        self._last_backlog = 0.0
        self._backlog_thread = None

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
                "abaqus_release": self.abaqus_release,
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
        now = time.time()
        if self.backlog and now - self._last_backlog >= BACKLOG_PERIOD \
                and not (self._backlog_thread
                         and self._backlog_thread.is_alive()):
            self._last_backlog = now
            self._backlog_thread = threading.Thread(target=self.retry_backlog)
            self._backlog_thread.daemon = True
            self._backlog_thread.start()
        if self.job is None:
            self._start_next()
        else:
            self._follow()

    def read_release(self):
        """Read `abaqus information=release` (a few seconds) and publish it
        with the heartbeat; '' when it cannot be read."""
        try:
            proc = subprocess.Popen([self.abaqus_cmd, "information=release"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            out = proc.communicate()[0]
            self.abaqus_release = parse_release(
                out.decode("cp1252", "replace"))
        except Exception as e:
            self.say("WARNING: Abaqus release not read (%r)" % (e,))
            return
        if self.abaqus_release:
            self.say("Abaqus release: %s" % self.abaqus_release)
        self.heartbeat(force=True)

    def run_forever(self, period=1.0):
        t = threading.Thread(target=self.read_release)
        t.daemon = True
        t.start()
        self.recover()
        self.heartbeat(force=True)
        self.say("agent ready on %s (Python %s)" % (self.host,
                                                    sys.version.split()[0]))
        self.say("queue %s" % self.queue)
        self.say("Abaqus %s, local runs in %s" % (self.abaqus_cmd,
                                                   self.local_root))
        self.say("leave this window open; close Remote Desktop with the cross")
        self.say("results and .odb go straight to the GUI PC when a run ends "
                 "while Remote Desktop is open with its drive C: redirected")
        if self.backlog:
            self.say("%d .odb waiting for the GUI PC (delivered when Remote "
                     "Desktop is open again)" % len(self.backlog))
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
                    "shared": shared, "client_dir": req.get("client_dir") or "",
                    "delivery": None, "chunks": chunks, "reader": reader,
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
        if j["delivery"] is None:
            # Copying a multi-GB .odb takes a while: a thread does it so the
            # heartbeat keeps going (other GUIs would think the agent dead).
            j["delivery"] = {"done": False}
            th = threading.Thread(target=self._deliver, args=(j,))
            th.daemon = True
            th.start()
            return
        if j["delivery"]["done"]:
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

    def _present(self, j, suffixes):
        return [j["name"] + suf for suf in suffixes
                if os.path.isfile(os.path.join(j["local"], j["name"] + suf))]

    def _copy_results(self, j, dest, suffixes):
        copied = []
        for name in self._present(j, suffixes):
            _copy_atomic(os.path.join(j["local"], name),
                         os.path.join(dest, name))
            copied.append(name)
        return copied

    # -- backlog of .odb not delivered yet -------------------------------
    def _save_backlog(self):
        try:
            _makedirs(self.local_root)
            write_json_atomic(self._backlog_path, self.backlog)
        except (IOError, OSError) as e:
            self.say("WARNING: backlog not saved (%s)" % e)

    def _add_backlog(self, entry):
        with self._backlog_lock:
            self.backlog.append(entry)
            self._save_backlog()

    def retry_backlog(self):
        """Move every waiting .odb whose GUI PC folder and token are visible.
        Never overwrites a file already there (a later run of the same job
        may have written it): such an entry stays and is reported."""
        with self._backlog_lock:
            todo = list(self.backlog)
        for e in todo:
            names = [n for n in e["files"]
                     if os.path.isfile(os.path.join(e["local"], n))]
            done = not names
            token = os.path.join(e["client_dir"], token_name(e["id"]))
            if names and os.path.isfile(token):
                clash = [n for n in names if os.path.exists(
                    os.path.join(e["client_dir"], n))]
                if clash:
                    if not e.get("clash_reported"):
                        e["clash_reported"] = True
                        self.say("%s already exists in %s: kept here in %s"
                                 % (", ".join(clash), e["client_dir"],
                                    e["local"]))
                    continue
                try:
                    self.say("moving waiting %s to %s"
                             % (", ".join(names), e["client_dir"]))
                    move_files(e["local"], names, e["client_dir"])
                    done = True
                    self.say("delivered %s" % ", ".join(names))
                except Exception as ex:
                    self.say("waiting %s not delivered yet (%s)"
                             % (", ".join(names), ex))
            if done:
                _remove(token)      # the client left it for this delivery
                with self._backlog_lock:
                    self.backlog = [b for b in self.backlog
                                    if b["id"] != e["id"]]
                    self._save_backlog()

    def _remove_replay_files(self, j):
        try:
            names = os.listdir(j["local"])
        except OSError:
            return
        for name in names:
            if REPLAY_FILE.match(name):
                _remove(os.path.join(j["local"], name))

    def _deliver(self, j):
        d = j["delivery"]
        d.update(copied=[], delivered_to="", note="", pending=[])
        self._remove_replay_files(j)
        try:
            client = j["client_dir"]
            if client:
                if os.path.isfile(os.path.join(client, token_name(j["id"]))):
                    self.say("moving results and .odb of %s to %s"
                             % (j["name"], client))
                    try:
                        names = self._present(
                            j, RESULT_SUFFIXES + DELIVERY_EXTRA_SUFFIXES)
                        move_files(j["local"], names, client)
                        d["copied"] = names
                        d["delivered_to"] = client
                    except Exception as e:
                        d["note"] = "copy to %s failed (%s)" % (client, e)
                else:
                    d["note"] = ("%s not reachable from this session (Remote "
                                 "Desktop window closed, drive C: not "
                                 "redirected, or another PC connected)"
                                 % client)
                if d["note"]:
                    self.say("%s: %s; using the shared drive"
                             % (j["name"], d["note"]))
            if not d["delivered_to"]:
                for suf in RESULT_SUFFIXES:
                    try:
                        d["copied"] += self._copy_results(j, j["shared"],
                                                          (suf,))
                    except (IOError, OSError) as e:
                        self.say("could not copy back %s: %s"
                                 % (j["name"] + suf, e))
                if client:
                    d["pending"] = self._present(j, DELIVERY_EXTRA_SUFFIXES)
                    if d["pending"]:
                        self._add_backlog({
                            "id": j["id"], "name": j["name"],
                            "local": j["local"], "client_dir": client,
                            "files": d["pending"]})
                        self.say("%s waits here until Remote Desktop is open "
                                 "again" % ", ".join(d["pending"]))
        except Exception as e:      # never leave the run unanswered
            d["note"] = (d["note"] + "; " if d["note"] else "") + repr(e)
        finally:
            d["done"] = True

    def _finish(self):
        j = self.job
        d = j["delivery"]
        j["reader"].join(5.0)
        rc = j["proc"].returncode
        cancelled = j["cancel_at"] is not None
        self.publish(j["id"], {
            "returncode": rc if rc is not None else 1,
            "cancelled": cancelled,
            "launcher_output": b"".join(j["chunks"]).decode(
                "cp1252", "replace"),
            "local_run_dir": j["local"],
            "copied": d["copied"],
            "delivered_to": d["delivered_to"],
            "delivery_note": d["note"],
            "pending_delivery": d["pending"]})
        self.say("%s %s (rc=%s), %s: %s"
                 % ("cancelled" if cancelled else "finished", j["name"], rc,
                    ("moved to " + d["delivered_to"]) if d["delivered_to"]
                    else "copied back",
                    ", ".join(d["copied"]) or "nothing"))
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
