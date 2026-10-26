# -*- coding: utf-8 -*-
"""Per-study run output organisation.

Each study launch (mesh GCI, domain sizing, sensitivity, and later inverse
identification) gets its OWN timestamped folder in the working directory,
named ``{profile}_{PREFIX}_{machine}_{YYYY-MM-DD_HH-MM-SS}`` (machine left out
when unknown), containing a ``config.json``
that records the study type, timestamp and parameters. Every sub-run of the
study writes inside that folder and its files are prefixed by PREFIX, so GCI,
domain-sizing and sensitivity runs never collide and stay grouped.

Run (job) names are ``{PREFIX}_run{NNN}`` followed by a tag that says what the
run varies (``ms_run003_ms25600``, ``GCI_run002_h0p5um``,
``domainsizing_run004_D250x250x50x50``, ``checks_run001_ms25600_h1um``),
kept to JOB_NAME_MAX characters. Sensitivity runs keep the short
``sensitivity_runNNN`` (many varied parameters would make long names). The number stays: resume and the run cache
rely on it; older folders without the tag are still read.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


# Longest job name written. The Abaqus 2022 limit is not known here (not
# checked); 38 characters is a cautious bound.
JOB_NAME_MAX = 38


def safe_token(text) -> str:
    """`text` with every run of characters other than letters, digits and
    '-' replaced by one '-' (file and job names stay portable)."""
    return re.sub(r"[^A-Za-z0-9-]+", "-", str(text or "")).strip("-")


def num_token(x) -> str:
    """A number without '.' or '+': 25600 -> '25600', 0.5 -> '0p5',
    2.5e-05 -> '2p5em05'."""
    t = "%g" % float(x)
    return t.replace(".", "p").replace("+", "").replace("-", "m")


def _um(mm) -> str:
    """A length in mm as a token in µm (3 significant digits)."""
    return num_token(float("%.3g" % (float(mm) * 1000.0)))


def study_folder_name(profile_name: Optional[str], prefix: str,
                      when: Optional[datetime] = None,
                      machine: Optional[str] = None) -> str:
    """`{profile}_{PREFIX}_{machine}_{YYYY-MM-DD_HH-MM-SS}` (profile defaults
    to 'Untitled'; no machine part when `machine` is empty)."""
    when = when or datetime.now()
    parts = [profile_name or "Untitled", prefix]
    m = safe_token(machine)
    if m:
        parts.append(m)
    parts.append(when.strftime("%Y-%m-%d_%H-%M-%S"))
    return "_".join(parts)


def model_run_tag(job_prefix: str, cfg) -> str:
    """What a run of a Model sizing study varies, from its ModelConfig:
    ms (mass-scaling study), h (GCI), D = l_wp x h_wp x l_void x h_void
    (domain), ms and h (final checks). '' for any other prefix."""
    try:
        st = cfg.step
        ms = (float(st.mass_scaling_factor)
              if getattr(st, "mass_scaling_enabled", False) else 1.0)
        h = float(cfg.elem_size)
        g = cfg.euler_geometry
    except (AttributeError, TypeError, ValueError):
        return ""
    if job_prefix == "ms":
        return "ms" + num_token(ms)
    if job_prefix == "GCI":
        return "h%sum" % _um(h)
    if job_prefix == "domainsizing":
        return "D" + "x".join(_um(v) for v in (g.l_wp, g.h_wp,
                                                 g.l_void, g.h_void))
    if job_prefix == "checks":
        return "ms%s_h%sum" % (num_token(ms), _um(h))
    return ""


def job_name(prefix: str, i: int, tag: str = "") -> str:
    """`{prefix}_run{NNN}_{tag}`, the tag cut so the name has at most
    JOB_NAME_MAX characters (the numbered part is never cut)."""
    base = "%s_run%03d" % (prefix, i)
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", tag or "").strip("-_")
    if not tag:
        return base
    room = JOB_NAME_MAX - len(base) - 1
    if room <= 0:
        return base
    return "%s_%s" % (base, tag[:room].rstrip("-_"))


def create_study_dir(workdir, profile_name: Optional[str], prefix: str,
                     study_config: dict,
                     when: Optional[datetime] = None,
                     machine: Optional[str] = None) -> Path:
    """Create the study folder in `workdir` and write its config.json.

    Returns the folder Path. The folder is created (raising on failure, e.g. a
    bad working directory); the config.json write is best-effort so a JSON
    hiccup never blocks the study.
    """
    when = when or datetime.now()
    name = study_folder_name(profile_name, prefix, when, machine)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    # A new folder every time: two studies started within the same second
    # must not share one (the second would overwrite the first's files).
    run_dir, n = Path(workdir) / name, 1
    while True:
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            n += 1
            run_dir = Path(workdir) / ("%s_%d" % (name, n))
    payload = {
        "study": prefix,
        "created_at": when.isoformat(timespec="seconds"),
        "profile": profile_name or "Untitled",
        "parameters": study_config,
    }
    try:
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    except OSError:
        pass
    return run_dir
