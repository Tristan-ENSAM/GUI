# -*- coding: utf-8 -*-
"""
run_progress — the three stages of one run, as the GUI's progress bars see
them: the solver (from <job>.sta), the extraction of the results (from the
[PROGRESS] lines of <job>.gui.log) and, in remote mode, the transfer of the
files to the run folder (from <job>.transfer.json).

Each reader takes files only, so it can be called from the GUI thread's
polling timer whatever thread runs the job, and never raises: a file that is
missing or half written reads as "nothing yet".

The [PROGRESS] lines are written by abaqus_scripts/cel_results.py for the bar
only; `LogLineFilter` keeps them out of the visible log.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROGRESS_TAG = "[PROGRESS]"          # same tag as cel_results.PROGRESS_TAG

_EXTRACT_LINE = re.compile(
    r"^\[PROGRESS\] extract (?P<frac>[\d.]+)(?: \| (?P<detail>.*))?$")
_STAGE_LINE = re.compile(r"^\[STAGE\] (?P<stage>\w+)")


def transfer_manifest_name(job: str) -> str:
    """File the remote side writes in the run folder before it copies the
    results there: {"files": {name: size in bytes}}. Same name as in
    remote_agent.transfer_manifest_name."""
    return "%s.transfer.json" % job


class LogLineFilter:
    """Drop the [PROGRESS] lines from a log read by chunks.

    A chunk read from a growing file can end in the middle of a line, so the
    unfinished last line is held back until the next chunk (or `flush`)."""

    def __init__(self):
        self._rest = ""

    def feed(self, text: str) -> str:
        text = self._rest + text
        cut = text.rfind("\n") + 1
        self._rest = text[cut:]
        return self._keep(text[:cut])

    def flush(self) -> str:
        text, self._rest = self._rest, ""
        return self._keep(text)

    @staticmethod
    def _keep(text: str) -> str:
        if PROGRESS_TAG not in text:
            return text
        return "".join(line for line in text.splitlines(True)
                       if not line.lstrip().startswith(PROGRESS_TAG))


@dataclass
class StageProgress:
    """One bar: `fraction` in [0, 1] (None = not started) and a short text."""
    fraction: Optional[float] = None
    text: str = ""
    done: bool = False


def extraction_progress(log_path) -> StageProgress:
    """Extraction stage from the script log: not started before
    [STAGE] EXTRACT_START, done at [STAGE] EXTRACT_DONE, else the last
    [PROGRESS] extract line."""
    out = StageProgress()
    try:
        with open(log_path, "r", encoding="latin-1", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        m = _STAGE_LINE.match(line)
        if m:
            stage = m.group("stage")
            if stage == "EXTRACT_START":
                out = StageProgress(0.0, "opening the results")
            elif stage == "EXTRACT_DONE":
                out = StageProgress(1.0, "done", done=True)
            continue
        m = _EXTRACT_LINE.match(line)
        if m and out.fraction is not None and not out.done:
            try:
                frac = max(0.0, min(1.0, float(m.group("frac"))))
            except ValueError:
                continue
            out = StageProgress(frac, (m.group("detail") or "").strip())
    return out


def transfer_progress(run_dir, job: str) -> StageProgress:
    """Transfer stage from the manifest the remote side leaves in the run
    folder: bytes already there (finished files, or the temporary copy being
    written, <name>.tmp-xxxxxxxx) over the bytes announced."""
    run_dir = Path(run_dir)
    manifest = run_dir / transfer_manifest_name(job)
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        since = manifest.stat().st_mtime
        files = {str(k): int(v) for k, v in (data.get("files") or {}).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return StageProgress()
    total = sum(max(v, 0) for v in files.values())
    if total <= 0:
        return StageProgress(0.0, "starting")
    got = 0
    current = ""
    for name, size in files.items():
        final = run_dir / name
        try:
            st = final.stat()
            if st.st_size == size and st.st_mtime >= since - 2.0:
                got += size
                continue
        except OSError:
            pass
        # The copy in progress: the newest temporary file of that name
        # written after the manifest (older ones are leftovers).
        best = 0
        for tmp in run_dir.glob(name + ".tmp-*"):
            try:
                st = tmp.stat()
            except OSError:
                continue
            if st.st_mtime >= since - 2.0:
                best = max(best, min(st.st_size, size))
        if best:
            current = name
        got += best
    frac = got / float(total)
    text = "%s / %s" % (_fmt_bytes(got), _fmt_bytes(total))
    if current:
        text += " · " + current
    return StageProgress(frac, text)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0 or unit == "GB":
            return ("%.0f %s" if unit == "B" else "%.1f %s") % (n, unit)
        n /= 1024.0
    return "%.1f GB" % n
