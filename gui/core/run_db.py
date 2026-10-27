# -*- coding: utf-8 -*-
"""Simulation database: finished runs kept in one folder (an external drive
to start with) and reused by every later study, whatever its folder.

Layout of the database folder (Preferences > Simulation database)::

    <root>/
      README.txt                     what this folder is (written once)
      index.csv, index.xlsx          one line per run (rebuilt; never read)
      runs/<k2>/<k16>__<name>/       one finished run
        <job>.results.npz, <job>.meta.json, <job>.sta   (always)
        <job>.inp, .msg, .dat, .gui.log                   (when present)
        <job>.odb                                         (only if asked)
        params.json                  model_config: the identity of the run
        provenance.json              where/how it ran (see `provenance`)
        COMPLETE                     written last: the copy is whole

`<k16>` is the first 16 hex digits of the run's parameter key
(gui.sensitivity.run_cache.params_key), `<k2>` its first two, so a lookup
only lists one small folder; `<name>` is the readable name of the run
(``{PC}__{profile}__{job}``, gui.core.run_output fields) and is never used
to find it. The full key of params.json is compared on every lookup. A run
folder without COMPLETE (copy cut short) is ignored and replaced by the
next archive of that run. A run already in the database is never
overwritten. Only plain files: the drive stays readable without the GUI.

Reuse across Abaqus versions: two versions do not give the same results
(2022 GA on CL-WTS-1 vs 2022 HF8 on the GUI PC), so by default a run is
reused only when it was made with the same Abaqus release as the machine
that would run it; when either release is unknown, only a run made on that
same PC (hypothesis: one PC = one Abaqus install) is reused. `any_release`
lifts the filter.

Pure host-side module (no Qt). Command line::

    python -m gui.core.run_db import <folder> --db <root> [--with-odb]
    python -m gui.core.run_db index --db <root>
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from gui.sensitivity.run_cache import (
    CachedRun, flatten, params_key, run_completed)

RUNS = "runs"
COMPLETE = "COMPLETE"
PARAMS = "params.json"
PROVENANCE = "provenance.json"
KEY_CHARS = 16
SEP = "__"
# Files of a run copied into the database (the bundle first: run_completed).
BUNDLE_SUFFIXES = (".results.npz", ".meta.json", ".sta")
EXTRA_SUFFIXES = (".inp", ".msg", ".dat", ".gui.log")
ODB_SUFFIX = ".odb"
UNKNOWN = "unknown"

README = """Simulation database of the Abaqus CEL cutting GUI.

runs/<2 hex>/<16 hex>__<PC>__<profile>__<job>/ is one finished run:
  params.json      all the model parameters (the identity of the run)
  provenance.json  PC, Abaqus release, CPUs, origin, date
  <job>.results.npz / .meta.json / .sta / .inp / ...  the run's files
  COMPLETE         the copy is whole (a folder without it is ignored)
index.csv / index.xlsx list the runs; they are rebuilt by
  python -m gui.core.run_db index --db <this folder>
Do not rename the run folders: the GUI finds a run by its 16 hex digits.
"""


# ---------------------------------------------------------------------------
# Abaqus release
# ---------------------------------------------------------------------------
_RELEASE_CACHE: Dict[str, str] = {}


def parse_release(text: str) -> str:
    """The release of `abaqus information=release` output: its first line
    naming Abaqus and a year (e.g. 'Abaqus 2022.HF8'), followed by its
    build line when there is one; '' when none is found. The exact output
    format was not checked on both installs: the whole line is kept, so two
    releases compare equal only when Abaqus prints the same text."""
    # One rule for both sides: the agent's copy (standalone module).
    from gui.core.remote_agent import parse_release as _parse
    return _parse(text)


def local_release(abaqus_cmd: str, timeout: float = 120.0) -> str:
    """Release of the Abaqus behind `abaqus_cmd` on this PC (run once per
    command and kept); '' when it cannot be read."""
    if not abaqus_cmd:
        return ""
    if abaqus_cmd in _RELEASE_CACHE:
        return _RELEASE_CACHE[abaqus_cmd]
    try:
        out = subprocess.run([abaqus_cmd, "information=release"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=timeout).stdout
        rel = parse_release(out.decode("cp1252", errors="replace"))
    except (OSError, subprocess.SubprocessError, ValueError):
        rel = ""
    _RELEASE_CACHE[abaqus_cmd] = rel
    return rel


def machine_release(prefs, name: str = "") -> str:
    """Abaqus release of machine `name` ("" = default): read on this PC for
    the local machine, from the queue's agent.json (written by the agent
    at start) for a queue machine; '' when unknown."""
    from gui.core import machines as M
    m = M.find(prefs, name or M.default_name(prefs))
    if m is None:
        return ""
    if m.kind == M.LOCAL:
        return local_release(getattr(prefs, "abaqus_cmd", "") or "")
    try:
        data = json.loads((Path(m.queue_dir) / "agent.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    rel = data.get("abaqus_release") if isinstance(data, dict) else None
    return str(rel) if rel else ""


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def gui_commit() -> str:
    """Git commit of the GUI checkout, or '' (not a checkout, no git)."""
    root = Path(__file__).resolve().parents[2]
    head = root / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            p = root / ".git" / ref[4:].strip()
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
            packed = root / ".git" / "packed-refs"
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref[4:].strip():
                    return parts[0]
            return ""
        return ref
    except OSError:
        return ""


def provenance(host: str = "", release: str = "", cpus=None,
               source_folder="", source_job: str = "",
               profile: str = "", note: str = "") -> dict:
    """What is known of how a run was made. Unknown fields hold 'unknown',
    never a guess."""
    return {
        "host": host or UNKNOWN,
        "abaqus_release": release or UNKNOWN,
        "cpus": int(cpus) if cpus else UNKNOWN,
        "profile": profile or UNKNOWN,
        "source_folder": str(source_folder) if source_folder else UNKNOWN,
        "source_job": source_job or UNKNOWN,
        "gui_commit": gui_commit() or UNKNOWN,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "archived_by_host": socket.gethostname(),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@dataclass
class DbRun:
    folder: Path
    job: str
    key: str
    params: dict = field(repr=False)
    prov: dict = field(repr=False, default_factory=dict)

    def as_cached(self) -> CachedRun:
        """The run as a RunCache entry (same files, same names)."""
        return CachedRun(folder=self.folder, job=self.job, key=self.key,
                         params=self.params)

    @property
    def host(self) -> str:
        return str(self.prov.get("host") or UNKNOWN)

    @property
    def release(self) -> str:
        return str(self.prov.get("abaqus_release") or UNKNOWN)


def _known(v) -> bool:
    return bool(v) and v != UNKNOWN


def release_accepted(run: DbRun, host: str = "", release: str = "",
                     any_release: bool = False) -> bool:
    """May `run` stand for a run of Abaqus `release` on PC `host`? Same
    release when both are known; else same PC; else no."""
    if any_release:
        return True
    if _known(release) and _known(run.release):
        return run.release == release
    return _known(host) and _known(run.host) and \
        run.host.lower() == host.lower()


def _field(text) -> str:
    from gui.core.run_output import safe_token
    return safe_token(text) or "x"


class RunDatabase:
    """A simulation database folder (see the module docstring)."""

    def __init__(self, root):
        self.root = Path(root)

    # -- state ---------------------------------------------------------
    def available(self) -> bool:
        """The folder is there (drive plugged in)."""
        try:
            return self.root.is_dir()
        except OSError:
            return False

    def _bucket(self, key: str) -> Path:
        return self.root / RUNS / key[:2]

    def _read_run(self, folder: Path) -> Optional[DbRun]:
        if not (folder / COMPLETE).is_file():
            return None
        try:
            params = json.loads((folder / PARAMS).read_text(encoding="utf-8"))
            prov = json.loads((folder / PROVENANCE).read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(params, dict) or not isinstance(prov, dict):
            return None
        job = str(prov.get("job") or "")
        if not job or not run_completed(folder, job):
            return None
        return DbRun(folder=folder, job=job, key=params_key(params),
                     params=params, prov=prov)

    def _candidates(self, key: str) -> List[Path]:
        head = key[:KEY_CHARS] + SEP
        bucket = self._bucket(key)
        try:
            names = os.listdir(str(bucket))
        except OSError:
            return []
        return [bucket / n for n in sorted(names) if n.startswith(head)]

    def find(self, params: dict) -> List[DbRun]:
        """Every complete run of the database with exactly these
        parameters."""
        key = params_key(params)
        out = []
        for folder in self._candidates(key):
            run = self._read_run(folder)
            if run is not None and run.key == key:
                out.append(run)
        return out

    def lookup(self, params: dict, host: str = "", release: str = "",
               any_release: bool = False) -> Optional[DbRun]:
        """A complete run with these parameters that may stand for a run
        on `host` with Abaqus `release` (release_accepted), or None."""
        for run in self.find(params):
            if release_accepted(run, host, release, any_release):
                return run
        return None

    def runs(self) -> List[DbRun]:
        """Every complete run of the database."""
        out = []
        base = self.root / RUNS
        try:
            buckets = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            return out
        for b in buckets:
            try:
                folders = sorted(p for p in b.iterdir() if p.is_dir())
            except OSError:
                continue
            for f in folders:
                run = self._read_run(f)
                if run is not None:
                    out.append(run)
        return out

    # -- writing -------------------------------------------------------
    def _ensure_root(self) -> None:
        (self.root / RUNS).mkdir(parents=True, exist_ok=True)
        readme = self.root / "README.txt"
        if not readme.exists():
            try:
                readme.write_text(README, encoding="utf-8")
            except OSError:
                pass

    def archive(self, src_folder, job: str, prov: dict,
                keep_odb: bool = False) -> Optional[DbRun]:
        """Copy the finished run `job` of `src_folder` into the database.
        Returns the database run (the one already there when this exact run
        with the same Abaqus release and PC is in it), or None when `job` is
        not a finished run or the copy failed (nothing complete left
        behind). `prov` comes from `provenance`."""
        src = Path(src_folder)
        if not run_completed(src, job):
            return None
        try:
            meta = json.loads((src / (job + ".meta.json")).read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return None
        params = meta.get("model_config") if isinstance(meta, dict) else None
        if not isinstance(params, dict):
            return None
        prov = dict(prov or {})
        prov["job"] = job
        for run in self.find(params):
            if run.release == prov.get("abaqus_release", UNKNOWN) \
                    and run.host == prov.get("host", UNKNOWN):
                return run
        key = params_key(params)
        name = SEP.join([key[:KEY_CHARS], _field(prov.get("host")),
                         _field(prov.get("profile")), _field(job)])
        try:
            self._ensure_root()
            dst = self._bucket(key) / name
            n = 1
            while dst.exists() and (dst / COMPLETE).exists():
                n += 1                      # same run, other PC/release
                dst = self._bucket(key) / ("%s-%d" % (name, n))
            dst.mkdir(parents=True, exist_ok=True)
            suffixes = list(BUNDLE_SUFFIXES) + list(EXTRA_SUFFIXES)
            if keep_odb:
                suffixes.append(ODB_SUFFIX)
            for suf in suffixes:
                s = src / (job + suf)
                if not s.is_file():
                    if suf in BUNDLE_SUFFIXES:
                        return None
                    continue
                d = dst / (job + suf)
                shutil.copy2(str(s), str(d))
                if d.stat().st_size != s.stat().st_size:
                    raise OSError("size mismatch copying %s" % s.name)
            prov["odb_kept"] = bool(keep_odb and (dst / (job + ODB_SUFFIX))
                                    .is_file())
            (dst / PARAMS).write_text(json.dumps(params, indent=1),
                                      encoding="utf-8")
            (dst / PROVENANCE).write_text(
                json.dumps(prov, indent=1, default=str), encoding="utf-8")
            (dst / COMPLETE).write_text(prov.get("archived_at", ""),
                                        encoding="utf-8")
        except OSError:
            return None
        return self._read_run(dst)

    # -- index ---------------------------------------------------------
    def rebuild_index(self, write_xlsx: bool = True) -> Path:
        """Write index.csv (+ index.xlsx) with one line per run: its folder,
        provenance and every model parameter that differs between runs."""
        runs = self.runs()
        flat = [flatten(r.params) for r in runs]
        keys = sorted(set().union(*flat)) if flat else []
        varying = [k for k in keys
                   if len({repr(f.get(k)) for f in flat}) > 1]
        prov_cols = ["host", "abaqus_release", "cpus", "profile",
                     "source_job", "source_folder", "archived_at",
                     "odb_kept", "gui_commit"]
        self._ensure_root()
        path = self.root / "index.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["folder", "key"] + prov_cols + varying)
            for r, f in zip(runs, flat):
                w.writerow([str(r.folder.relative_to(self.root)), r.key]
                           + [r.prov.get(c, "") for c in prov_cols]
                           + [f.get(k, "") for k in varying])
        if write_xlsx:
            from gui.core.xlsx_writer import excel_copy
            excel_copy(path)
        return path

    # -- import --------------------------------------------------------
    def import_folder(self, folder, keep_odb: bool = False,
                      recursive: bool = True, log=print) -> Dict[str, int]:
        """Archive every finished run found in `folder` (and its sub-folders).
        The PC is read from a folder named {PC}__... (gui.core.run_output),
        else 'unknown'; the Abaqus release is not known for runs made
        before the database. Returns counts {added, present, skipped}."""
        counts = {"added": 0, "present": 0, "skipped": 0}
        root = Path(folder)
        dirs = [root] + (sorted(p for p in root.rglob("*") if p.is_dir())
                         if recursive else [])
        for d in dirs:
            try:
                metas = sorted(d.glob("*.meta.json"))
            except OSError:
                continue
            for meta in metas:
                job = meta.name[:-len(".meta.json")]
                if not run_completed(d, job):
                    counts["skipped"] += 1
                    continue
                before = None
                try:
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    before = len(self.find(data.get("model_config") or {}))
                except (OSError, ValueError, AttributeError):
                    pass
                host, profile = _host_profile_of(d)
                run = self.archive(d, job, provenance(
                    host=host, profile=profile, source_folder=d,
                    source_job=job, note="imported"), keep_odb=keep_odb)
                if run is None:
                    counts["skipped"] += 1
                    log("skipped %s in %s (copy failed)" % (job, d))
                elif before is not None and len(self.find(run.params)) \
                        == before:
                    counts["present"] += 1
                else:
                    counts["added"] += 1
                    log("added %s from %s" % (job, d))
        return counts


def _host_profile_of(folder: Path):
    """(PC, profile) from a folder named {PC}__{profile}__..., else
    ('', ''), and the profile of config.json when it has one."""
    parts = folder.name.split(SEP)
    host = parts[0] if len(parts) >= 3 else ""
    profile = parts[1] if len(parts) >= 3 else ""
    try:
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        profile = str(cfg.get("profile") or profile)
    except (OSError, ValueError, AttributeError):
        pass
    return host, profile


def from_prefs(prefs) -> Optional[RunDatabase]:
    """The database set in Preferences, or None (not set)."""
    root = str(getattr(prefs, "run_db_dir", "") or "").strip()
    return RunDatabase(root) if root else None


def attach_to_cache(cache, prefs, machine: str, host: str) -> str:
    """Make `cache` (a RunCache) look runs up in the database of `prefs`
    for a study on `machine` (PC `host`). Returns a line for the study log
    ('' when no database is set)."""
    db = from_prefs(prefs)
    if db is None:
        return ""
    if not db.available():
        return ("Simulation database %s not reachable (drive not plugged "
                "in?): the study runs without it" % db.root)
    cache.attach_db(db, host=host,
                    release_getter=lambda: machine_release(prefs, machine),
                    any_release=bool(getattr(prefs, "run_db_any_release",
                                             False)))
    return "Simulation database %s: its runs are reused when they match" \
        % db.root


def archive_run(prefs, machine: str, host: str, folder, job: str, cpus=None,
                profile: str = "") -> str:
    """Copy a run that just finished into the database of `prefs` (when one
    is set and archiving is on). Returns a line for the log ('' when there
    is nothing to say). Reading the Abaqus release can take a few seconds:
    call it off the GUI thread."""
    db = from_prefs(prefs)
    if db is None or not getattr(prefs, "run_db_archive", True):
        return ""
    if not db.available():
        return ("[%s] simulation database %s not reachable: run not "
                "archived" % (job, db.root))
    prov = provenance(host=host, release=machine_release(prefs, machine),
                      cpus=cpus, source_folder=folder, source_job=job,
                      profile=profile)
    run = db.archive(folder, job, prov,
                     keep_odb=bool(getattr(prefs, "run_db_keep_odb", False)))
    if run is None:
        return "[%s] could not be copied into the simulation database" % job
    return "[%s] in the simulation database: %s" % (job, run.folder.name)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m gui.core.run_db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="copy the finished runs of a folder")
    p.add_argument("folder")
    p.add_argument("--db", required=True)
    p.add_argument("--with-odb", action="store_true")
    p = sub.add_parser("index", help="rebuild index.csv / index.xlsx")
    p.add_argument("--db", required=True)
    ns = ap.parse_args(list(argv) if argv is not None else None)
    db = RunDatabase(ns.db)
    if ns.cmd == "import":
        c = db.import_folder(ns.folder, keep_odb=ns.with_odb)
        print("added %(added)d, already there %(present)d, skipped "
              "%(skipped)d" % c)
    print("index: %s" % db.rebuild_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
