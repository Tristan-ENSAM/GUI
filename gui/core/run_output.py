# -*- coding: utf-8 -*-
"""Per-study run output organisation.

Each study launch (mesh GCI, domain sizing, sensitivity, and later inverse
identification) gets its OWN timestamped folder in the working directory,
named ``{profile}_{PREFIX}_{YYYY-MM-DD_HH-MM-SS}``, containing a ``config.json``
that records the study type, timestamp and parameters. Every sub-run of the
study writes inside that folder and its files are prefixed by PREFIX, so GCI,
domain-sizing and sensitivity runs never collide and stay grouped.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def study_folder_name(profile_name: Optional[str], prefix: str,
                      when: Optional[datetime] = None) -> str:
    """`{profile}_{PREFIX}_{YYYY-MM-DD_HH-MM-SS}` (profile defaults to 'Untitled')."""
    when = when or datetime.now()
    return "%s_%s_%s" % (profile_name or "Untitled", prefix,
                         when.strftime("%Y-%m-%d_%H-%M-%S"))


def create_study_dir(workdir, profile_name: Optional[str], prefix: str,
                     study_config: dict,
                     when: Optional[datetime] = None) -> Path:
    """Create the study folder in `workdir` and write its config.json.

    Returns the folder Path. The folder is created (raising on failure, e.g. a
    bad working directory); the config.json write is best-effort so a JSON
    hiccup never blocks the study.
    """
    when = when or datetime.now()
    name = study_folder_name(profile_name, prefix, when)
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
