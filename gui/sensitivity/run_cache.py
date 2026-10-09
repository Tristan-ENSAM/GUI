# -*- coding: utf-8 -*-
"""Reuse of finished Abaqus runs: resume an interrupted study, load a study.

Every finished run of a study leaves, in the study folder,
``<job>.results.npz`` + ``<job>.meta.json`` + ``<job>.sta``. The meta file
holds ``model_config``, the exact ``ModelConfig.to_params_dict()`` the run was
launched with. A run is therefore identified by the CONTENT of its
parameters, not by its job name: when a study asks for a configuration whose
parameters match a finished run, the saved bundle is returned instead of
launching Abaqus again.

The study cores are deterministic for given bundles (same calls in the same
order), so re-running a study over its own folder REPLAYS it: the runs
already done are reloaded, the first missing one is launched and the study
continues (resume). With launching forbidden, the replay rebuilds the full
in-memory result of a finished study without any Abaqus run (load); the
first missing run then tells why the folder is not enough (study not
finished, or made for another model).

A run counts as finished only if its .sta reports a successful analysis and
its bundle files exist: an interrupted job never has both. A run whose
analysis Abaqus stopped (its .sta says so) leaves ``<job>.failed.json`` with
its parameters instead, so the replay meets the same failure again rather
than a missing run. Matching uses a
canonical JSON form of the parameters (ints and floats unified, key order
ignored), so a match is exact up to the 12 significant digits of
``to_params_dict``. Older study folders work too: their runs already carry
``model_config``.

Pure host-side module (no Qt).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Last line of an Abaqus .sta (abaqus_scripts/cel_model.py,
# _check_job_succeeded): one or the other once the solver has ended.
SUCCESS_MARK = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
NOT_COMPLETED_MARK = "THE ANALYSIS HAS NOT BEEN COMPLETED"
_META_SUFFIX = ".meta.json"
_NPZ_SUFFIX = ".results.npz"
_FAILED_SUFFIX = ".failed.json"
_STA_TAIL = 65536          # bytes of the .sta read to find the marks


# ---------------------------------------------------------------------------
# Canonical form and keys
# ---------------------------------------------------------------------------
def canonical(obj):
    """JSON-like copy with tuples as lists, every non-bool number as float
    (so 1 and 1.0 match) and dict keys as strings."""
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return float(obj)
    try:                                   # numpy scalars and the like
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def params_key(params: dict) -> str:
    """Stable hash of a parameter dict (see `canonical`)."""
    text = json.dumps(canonical(params), sort_keys=True,
                      separators=(",", ":"), allow_nan=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def flatten(d: dict, prefix: str = "") -> Dict[str, object]:
    """{"a": {"b": 1}} -> {"a.b": 1.0} (canonical leaves)."""
    out: Dict[str, object] = {}
    for k, v in canonical(d).items():
        name = "%s%s" % (prefix, k)
        if isinstance(v, dict):
            out.update(flatten(v, name + "."))
        else:
            out[name] = v
    return out


def diff_params(a: dict, b: dict, ignore: Iterable[str] = ()
                ) -> List[Tuple[str, object, object]]:
    """Differing leaves of two parameter dicts, as (dotted key, a, b),
    sorted by key; keys listed in `ignore` are skipped."""
    fa, fb = flatten(a), flatten(b)
    skip = set(ignore)
    out = []
    for k in sorted(set(fa) | set(fb)):
        if k in skip:
            continue
        va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
        if va != vb:
            out.append((k, va, vb))
    return out


# ---------------------------------------------------------------------------
# Job files
# ---------------------------------------------------------------------------
_JOB_RE_CACHE: Dict[str, "re.Pattern"] = {}


def _job_re(prefix: str):
    pat = _JOB_RE_CACHE.get(prefix)
    if pat is None:
        pat = re.compile(r"^%s_run(\d+)\." % re.escape(prefix), re.IGNORECASE)
        _JOB_RE_CACHE[prefix] = pat
    return pat


def next_job_index(folder, prefix: str) -> int:
    """First run index after every `<prefix>_runNNN.*` file of `folder`, so
    a study resumed in its folder never reuses the name of an earlier run."""
    folder = Path(folder)
    pat = _job_re(prefix)
    last = -1
    try:
        names = [p.name for p in folder.iterdir()]
    except OSError:
        return 0
    for name in names:
        m = pat.match(name)
        if m:
            last = max(last, int(m.group(1)))
    return last + 1


def job_files(folder, job: str) -> List[Path]:
    """Files of `job` in `folder` (`<job>.<anything>`, case-insensitive)."""
    folder = Path(folder)
    head = job.lower() + "."
    try:
        return sorted(p for p in folder.iterdir()
                      if p.is_file() and p.name.lower().startswith(head))
    except OSError:
        return []


def remove_job_files(folder, job: str) -> List[Path]:
    """Delete the files of `job` (left over by an interrupted run) before a
    launch under that name; returns the files that could not be removed."""
    stuck = []
    for p in job_files(folder, job):
        try:
            p.unlink()
        except OSError:
            stuck.append(p)
    return stuck


def _sta_tail(path) -> Optional[str]:
    """The end of a .sta (the marks are its last line; a long run's .sta
    can be several MB), or None when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - _STA_TAIL))
            return handle.read().decode("latin-1", errors="replace")
    except OSError:
        return None


def sta_outcome(folder, job: str) -> str:
    """How the analysis of `job` ended, from its .sta: "success",
    "not_completed" (Abaqus stopped the analysis) or "unknown" (no .sta, or
    no final mark: the solver did not start, was killed, or the file did
    not come back)."""
    text = _sta_tail(Path(folder) / (job + ".sta"))
    if text is None:
        return "unknown"
    if SUCCESS_MARK in text:
        return "success"
    if NOT_COMPLETED_MARK in text:
        return "not_completed"
    return "unknown"


def run_completed(folder, job: str) -> bool:
    """True when `job` finished: its .sta reports a successful analysis and
    its bundle (.results.npz + .meta.json) exists."""
    folder = Path(folder)
    if not (folder / (job + _NPZ_SUFFIX)).exists():
        return False
    if not (folder / (job + _META_SUFFIX)).exists():
        return False
    return sta_outcome(folder, job) == "success"


def write_failed_marker(folder, job: str, params: dict, reason: str) -> bool:
    """Record that the analysis of `job` (made with `params`) did not
    complete, so a replay of the study meets the same failure; best
    effort."""
    from datetime import datetime
    payload = {"model_config": params, "reason": str(reason),
               "when": datetime.now().isoformat(timespec="seconds")}
    try:
        (Path(folder) / (job + _FAILED_SUFFIX)).write_text(
            json.dumps(payload, indent=1, default=str), encoding="utf-8")
        return True
    except OSError:
        return False


def copy_run(src_folder, src_job: str, dst_folder, dst_job: str) -> bool:
    """Put the files of a finished run (.results.npz, .meta.json, .sta)
    into another study folder under `dst_job`: a hard link when the two
    folders share a disk, else a copy. False (nothing left behind) when it
    fails."""
    import os
    import shutil
    done = []
    try:
        for suffix in (_NPZ_SUFFIX, _META_SUFFIX, ".sta"):
            src = Path(src_folder) / (src_job + suffix)
            dst = Path(dst_folder) / (dst_job + suffix)
            try:
                os.link(str(src), str(dst))
            except OSError:
                shutil.copy2(str(src), str(dst))
            done.append(dst)
        return True
    except OSError:
        for p in done:
            try:
                p.unlink()
            except OSError:
                pass
        return False


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@dataclass
class CachedRun:
    folder: Path
    job: str
    key: str
    params: dict = field(repr=False)

    @property
    def npz(self) -> Path:
        return self.folder / (self.job + _NPZ_SUFFIX)

    @property
    def meta(self) -> Path:
        return self.folder / (self.job + _META_SUFFIX)

    @property
    def sta(self) -> Path:
        return self.folder / (self.job + ".sta")


def same_folder(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return Path(a) == Path(b)


class RunCache:
    """Index of the finished runs of one or more study folders, by the key
    of their parameters, and of the runs whose analysis did not complete."""

    def __init__(self, folders: Sequence = ()):
        self._index: Dict[str, List[CachedRun]] = {}
        self._failed_index: Dict[str, List[CachedRun]] = {}
        self.runs: List[CachedRun] = []
        self.failed: List[CachedRun] = []
        self.folders: List[Path] = []
        for f in folders:
            if f:
                self.add_folder(f)

    def __len__(self) -> int:
        return len(self.runs)

    def add_folder(self, folder) -> int:
        """Index the finished runs of `folder`; returns how many were added.
        A folder already indexed is skipped."""
        folder = Path(folder)
        try:
            resolved = folder.resolve()
        except OSError:
            resolved = folder
        if resolved in self.folders:
            return 0
        self.folders.append(resolved)
        n = 0
        try:
            metas = sorted(folder.glob("*" + _META_SUFFIX))
        except OSError:
            return 0
        for meta in metas:
            job = meta.name[:-len(_META_SUFFIX)]
            if self._add(folder, job)[1]:
                n += 1
        try:
            failed = sorted(folder.glob("*" + _FAILED_SUFFIX))
        except OSError:
            failed = []
        for marker in failed:
            self.add_failed(folder, marker.name[:-len(_FAILED_SUFFIX)])
        return n

    def add_failed(self, folder, job: str) -> Optional[CachedRun]:
        """Index a run whose analysis did not complete (its marker)."""
        folder = Path(folder)
        try:
            data = json.loads((folder / (job + _FAILED_SUFFIX)).read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return None
        params = data.get("model_config") if isinstance(data, dict) else None
        if not isinstance(params, dict):
            return None
        run = CachedRun(folder=folder, job=job, key=params_key(params),
                        params=params)
        for other in self._failed_index.get(run.key, []):
            if other.folder == run.folder and other.job == run.job:
                return other
        self._failed_index.setdefault(run.key, []).append(run)
        self.failed.append(run)
        return run

    def lookup_failed(self, params: dict, folder) -> Optional[CachedRun]:
        """A run of `folder` with these parameters whose analysis did not
        complete, or None. Only the study's own folder counts: a failure
        seen by another study is tried again."""
        for run in self._failed_index.get(params_key(params), []):
            if same_folder(run.folder, folder):
                return run
        return None

    def add_run(self, folder, job: str) -> Optional[CachedRun]:
        """Index one finished run (after a launch, or while scanning)."""
        return self._add(folder, job)[0]

    def _add(self, folder, job: str) -> Tuple[Optional[CachedRun], bool]:
        folder = Path(folder)
        if not run_completed(folder, job):
            return None, False
        try:
            data = json.loads((folder / (job + _META_SUFFIX)).read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return None, False
        params = data.get("model_config") if isinstance(data, dict) else None
        if not isinstance(params, dict):
            return None, False
        run = CachedRun(folder=folder, job=job, key=params_key(params),
                        params=params)
        for other in self._index.get(run.key, []):
            if other.folder == run.folder and other.job == run.job:
                return other, False
        self._index.setdefault(run.key, []).append(run)
        self.runs.append(run)
        return run, True

    def lookup(self, params: dict, prefer=None) -> Optional[CachedRun]:
        """A finished run with exactly these parameters (one in `prefer`,
        a folder, when there is one), or None."""
        hits = self._index.get(params_key(params))
        if not hits:
            return None
        if prefer is not None:
            for run in hits:
                if same_folder(run.folder, prefer):
                    return run
        return hits[0]

    def closest(self, params: dict, ignore: Iterable[str] = (),
                runs: Optional[Sequence[CachedRun]] = None
                ) -> Tuple[Optional[CachedRun], List[Tuple[str, object, object]]]:
        """The run (of `runs`, default every indexed run) whose parameters
        differ from `params` in the fewest leaves (keys in `ignore` not
        counted), with the differences (run value first, requested value
        second)."""
        best, best_diff = None, None
        for run in (self.runs if runs is None else runs):
            d = diff_params(run.params, params, ignore=ignore)
            if best_diff is None or len(d) < len(best_diff):
                best, best_diff = run, d
        return best, (best_diff or [])
