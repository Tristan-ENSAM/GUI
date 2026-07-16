# -*- coding: utf-8 -*-
"""
Material preset library.

Two layers stacked, with user-defined presets taking precedence:
  1. Built-in presets shipped with the app, in `gui/presets/materials.json`.
     Read-only — overwritten on every update.
  2. User-defined presets in `materials_user.json` next to the launcher,
     created on first "Save as preset..." action. Persists across updates.

A preset is just the same dict shape as `cfg.tool_material` /
`cfg.euler_material` (Abaqus internal units), so applying one is a single
dict replacement.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from gui.core.logging_util import log_swallowed


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _bundled_path() -> Path:
    """Path to the bundled materials.json next to this module."""
    return Path(__file__).resolve().parent.parent / "presets" / "materials.json"


def _user_path() -> Path:
    """Path to the user's editable presets file. We put it next to the
    application launcher (the parent of the `gui` package), not inside
    the package — so it survives updates and is easy to find for the user."""
    return Path(__file__).resolve().parent.parent.parent / "materials_user.json"


def profiles_dir() -> Path:
    """Dedicated folder holding one JSON file per saved material profile.
    Lives next to the launcher so the user can browse/delete files there
    directly. Created on first access."""
    d = Path(__file__).resolve().parent.parent.parent / "material_profiles"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_swallowed("creating the material_profiles directory")
    return d


def save_profile_file(path, material: dict) -> Path:
    """Write a material dict to `path` as JSON (overwriting). Returns the
    path actually written (with a .json suffix enforced)."""
    import json
    p = Path(path)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(material, f, indent=2, ensure_ascii=False)
    return p


def load_profile_file(path) -> dict:
    """Read a material profile JSON file and return the material dict.
    Raises ValueError if the file is not a JSON object."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("A material profile must be a JSON object.")
    return data


# ---------------------------------------------------------------------------
# PresetLibrary
# ---------------------------------------------------------------------------
class PresetLibrary:
    """Reads bundled + user presets; can write user presets back to disk."""

    def __init__(self):
        self._bundled: dict = {"tool": {}, "workpiece": {}}
        self._user:    dict = {"tool": {}, "workpiece": {}}
        self.reload()

    # ----- I/O -----
    def reload(self):
        """Re-read both JSON files. Tolerant to missing/malformed files —
        a corrupt user file shouldn't crash the GUI; we just ignore it and
        the user gets only the built-in presets until they fix the file."""
        self._bundled = self._safe_load(_bundled_path(), required=True)
        self._user    = self._safe_load(_user_path(),    required=False)

    @staticmethod
    def _safe_load(path: Path, required: bool) -> dict:
        if not path.exists():
            if required:
                # No bundled file means the install is broken — return an
                # empty library rather than crashing, the GUI still works.
                print(f"[presets] WARNING: bundled file missing at {path}",
                      file=sys.stderr)
            return {"tool": {}, "workpiece": {}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[presets] WARNING: failed to read {path}: {e}",
                  file=sys.stderr)
            return {"tool": {}, "workpiece": {}}

        # Normalise: ensure both keys exist even if the file omits one
        out = {"tool": dict(data.get("tool", {})),
               "workpiece": dict(data.get("workpiece", {}))}
        return out

    def _save_user(self):
        """Persist the user library to disk."""
        path = _user_path()
        payload = {
            "format_version": 1,
            "tool":      self._user.get("tool", {}),
            "workpiece": self._user.get("workpiece", {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ----- query -----
    def list_presets(self, kind: str) -> list[str]:
        """Return all preset names for `kind` in ('tool', 'workpiece'),
        bundled first, then user-only. Duplicates (same name in both
        layers) are listed once, with the user version winning."""
        if kind not in ("tool", "workpiece"):
            return []
        bundled_names = list(self._bundled.get(kind, {}).keys())
        user_names    = list(self._user.get(kind, {}).keys())
        # Stable order: bundled in their JSON order, then any user-only
        seen = set(bundled_names)
        extra = [n for n in user_names if n not in seen]
        return bundled_names + extra

    def is_user_defined(self, kind: str, name: str) -> bool:
        return name in self._user.get(kind, {})

    def get(self, kind: str, name: str) -> dict | None:
        """Return a deep-ish copy of the preset dict (user layer wins).
        None if not found."""
        if kind not in ("tool", "workpiece"):
            return None
        # User overrides bundled
        for src in (self._user, self._bundled):
            if name in src.get(kind, {}):
                return dict(src[kind][name])
        return None

    # ----- mutate (user layer only) -----
    def save_user_preset(self, kind: str, name: str, material: dict):
        """Add or overwrite a preset in the user library, then persist."""
        if kind not in ("tool", "workpiece"):
            raise ValueError(f"Unknown preset kind: {kind!r}")
        if not name or not name.strip():
            raise ValueError("Preset name cannot be empty")
        self._user.setdefault(kind, {})[name] = dict(material)
        self._save_user()

    def delete_user_preset(self, kind: str, name: str) -> bool:
        """Remove a user-defined preset; return True if removed.
        Bundled presets cannot be deleted (the file is read-only)."""
        if kind not in ("tool", "workpiece"):
            return False
        if name in self._user.get(kind, {}):
            del self._user[kind][name]
            self._save_user()
            return True
        return False
