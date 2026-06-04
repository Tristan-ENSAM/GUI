# -*- coding: utf-8 -*-
"""
sta_parser — extract progress information from an Abaqus Explicit .sta
file as it is being written by the solver.

The .sta file is text-based and updated by Abaqus at every output frame
during a long Explicit run. We read it to drive a progress bar in the
GUI without having to capture the verbose console stream of the solver
(which is too chatty for a long run — millions of increments).

The relevant rows we look for:

  Output Field Frame Number   7, of  500, at step time 7.000E-06

  12479  7.000E-06 7.000E-06  00:07:22 5.604E-10       16760  9.030E-06  1.659E-01
  ^      ^         ^          ^        ^               ^      ^          ^
  inc#   STEPtime  TOTALtime  walltime stable_dt       critEl  KE         TE

The progress bar uses (step_time / sim_time), which corresponds 1:1 to
"frame number / n_frames". We keep both available because the frame
number is what the Output Field row tells us natively.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Match the increment row. The .sta uses fixed-width columns padded by
# spaces — we keep the regex tolerant to extra whitespace because some
# Abaqus versions pad slightly differently. The wall-time field is
# HH:MM:SS, possibly with leading zeros.
_INC_ROW = re.compile(
    r"^\s*"
    r"(?P<inc>\d+)\s+"                                # increment number
    r"(?P<step_time>[\d.+\-eE]+)\s+"                  # step time (s)
    r"(?P<total_time>[\d.+\-eE]+)\s+"                 # total time (s)
    r"(?P<wall_time>\d+:\d+:\d+)\s+"                  # wall time HH:MM:SS
    r"(?P<dt>[\d.+\-eE]+)\s+"                         # stable dt
    r"(?P<crit_elem>\d+)\s+"                          # critical element label
    r"(?P<kinetic_energy>[\d.+\-eE]+)\s+"             # kinetic energy
    r"(?P<total_energy>[\d.+\-eE]+)"                  # total energy
    r"\s*$"
)

# Match the "Output Field Frame Number" row. This is the most reliable
# progress signal: it explicitly tells us where we are relative to the
# total number of requested frames.
_FRAME_ROW = re.compile(
    r"^\s*Output Field Frame Number\s+"
    r"(?P<current>\d+)"
    r"\s*,\s*of\s+"
    r"(?P<total>\d+)"
    r"\s*,\s*at\s+step\s+time\s+"
    r"(?P<step_time>[\d.+\-eE]+)"
)


@dataclass
class StaProgress:
    """A snapshot of the parsed .sta state at one moment in time.

    Fields are Optional because some may be unavailable depending on
    where we are in the file (e.g. the very first lines of a .sta only
    contain the preprocessor header — no progress yet).

    Use `is_ready()` to check whether the snapshot carries meaningful
    progress information.
    """
    frame_current:  Optional[int]   = None    # last "Output Field Frame Number"
    frame_total:    Optional[int]   = None    # total frames requested
    step_time:      Optional[float] = None    # current step time (s)
    inc_number:     Optional[int]   = None    # last increment number
    wall_time:      Optional[str]   = None    # HH:MM:SS
    stable_dt:      Optional[float] = None
    critical_elem:  Optional[int]   = None
    kinetic_energy: Optional[float] = None
    total_energy:   Optional[float] = None

    def is_ready(self) -> bool:
        """True if at least one progress signal has been parsed."""
        return self.frame_current is not None or self.inc_number is not None

    def fraction(self) -> Optional[float]:
        """Return the fraction of the run completed in [0, 1], or None
        if unknown. Prefer the explicit "frame number / total" ratio
        when available; otherwise fall back to step_time / sim_time
        (the caller can compute that one when it knows sim_time)."""
        if (self.frame_current is not None and self.frame_total is not None
                and self.frame_total > 0):
            return float(self.frame_current) / float(self.frame_total)
        return None


def parse_sta(sta_path: str | Path) -> StaProgress:
    """Read a .sta file and return the most-recent progress snapshot.

    Implementation:
      - We read the whole file every call. .sta files for our cutting
        simulations stay small (one short line per frame + a fixed
        header), so reading hundreds of times for the duration of the
        run is cheap compared to actual Abaqus work.
      - We scan the lines in order and update the snapshot's fields as
        we encounter matching rows. This makes us robust to the order
        in which Abaqus prints things (the increment row and the
        "Output Field Frame Number" row alternate, but the exact
        ordering within a frame group varies slightly).

    If the file does not exist (the solver hasn't created it yet), we
    return an empty StaProgress, not an error — the GUI's polling timer
    is allowed to call this before the solver has produced anything.
    """
    p = Path(sta_path)
    snap = StaProgress()
    if not p.exists():
        return snap

    try:
        # Latin-1 covers everything Abaqus puts in .sta (ASCII + maybe
        # accented warning messages on French installs). Reading as
        # text is fine — these files are at most a few hundred lines.
        with open(p, "r", encoding="latin-1", errors="replace") as f:
            for line in f:
                # The frame row is the more specific match — try it first.
                m = _FRAME_ROW.match(line)
                if m:
                    snap.frame_current = int(m.group("current"))
                    snap.frame_total   = int(m.group("total"))
                    # The step_time on this row is redundant with the
                    # inc row's, but useful when the .sta is truncated
                    # mid-frame (no inc row yet).
                    snap.step_time     = float(m.group("step_time"))
                    continue

                m = _INC_ROW.match(line)
                if m:
                    snap.inc_number     = int(m.group("inc"))
                    snap.step_time      = float(m.group("step_time"))
                    snap.wall_time      = m.group("wall_time")
                    snap.stable_dt      = float(m.group("dt"))
                    snap.critical_elem  = int(m.group("crit_elem"))
                    snap.kinetic_energy = float(m.group("kinetic_energy"))
                    snap.total_energy   = float(m.group("total_energy"))
                    continue
    except OSError:
        # File vanished between exists() and open() — return whatever we
        # have so far (likely the empty snapshot).
        pass

    return snap
