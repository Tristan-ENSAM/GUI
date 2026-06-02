# -*- coding: utf-8 -*-
"""
User-level preferences, persisted between sessions.

These are SEPARATE from the per-project profile (the .acpf file): they
represent user-machine settings (where Abaqus lives, default working
directory, K vs °C display, ...) and follow the user, not the project.

Storage location:
  - Windows : %APPDATA%/AbqCuttingGUI/preferences.json
  - Linux   : ~/.config/AbqCuttingGUI/preferences.json
  - macOS   : ~/Library/Application Support/AbqCuttingGUI/preferences.json

The file is created lazily on first save; missing or malformed files
fall back to defaults silently so the GUI is always usable.

Designed to be extensible — when we add remote/HPC execution, we'll
introduce more fields (ssh user, queue name, modules to load, ...).
"""
from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path


# ----------------------------------------------------------------------------
# Storage location
# ----------------------------------------------------------------------------
def _prefs_dir() -> Path:
    """Per-OS conventional location for application config files."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", str(Path.home()))
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:  # linux / freebsd / etc.
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "AbqCuttingGUI"


def prefs_path() -> Path:
    return _prefs_dir() / "preferences.json"


# ----------------------------------------------------------------------------
# Dataclass
# ----------------------------------------------------------------------------
@dataclass
class Preferences:
    """All user-level preferences. Defaults aim at a typical Windows install
    of Abaqus 2022+; any path can be overridden by the user via the
    Preferences dialog or by editing preferences.json directly."""

    # ---- Abaqus executable and scripts ----
    abaqus_cmd:            str = r"C:\SIMULIA\Commands\abaqus.bat"
    abaqus_script:         str = r"C:\TEMP\Abaqus_script\abq_odb_generator.py"
    abaqus_extract_script: str = r"C:\TEMP\Abaqus_script\abq_extract.py"

    # ---- Default working directory for Abaqus jobs ----
    default_workdir: str = r"C:\TEMP\Abaqus_wd"

    # ---- Display preferences (mirror cfg.ui — but the user-level
    #      preference is the FALLBACK applied to new profiles. The cfg.ui
    #      on a loaded profile still wins so a saved file's display is
    #      preserved when reopened.) ----
    temp_unit_default: str = "C"     # "C" | "K"


# ----------------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------------
def load_preferences() -> Preferences:
    """Read prefs from disk; fall back to defaults on any error.

    Tolerant to:
      - missing file (first launch)
      - missing keys (older versions of the prefs file)
      - extra keys from a newer version (silently dropped)
      - malformed JSON (returns defaults + warning to stderr)
    """
    path = prefs_path()
    if not path.exists():
        return Preferences()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[preferences] WARNING: failed to read {path}: {e}",
              file=sys.stderr)
        return Preferences()
    if not isinstance(data, dict):
        return Preferences()

    prefs = Preferences()
    known = {f.name for f in fields(Preferences)}
    for k, v in data.items():
        if k in known:
            setattr(prefs, k, v)
    return prefs


def save_preferences(prefs: Preferences) -> None:
    """Persist prefs to disk. Creates parent directory if needed."""
    path = prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(prefs), f, indent=2, ensure_ascii=False)
