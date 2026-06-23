# -*- coding: utf-8 -*-
"""
Read/write experimental field files (DIC velocity, later IRT temperature).

Pairing (gui/results/FORMAT.md, experimental section):
  <stem>.npz   raw arrays   (x, y, t, V1, V2, Vmag, valid  for DIC)
  <stem>.json  metadata     (modality, source, units, params, provenance)

The two files share the same stem; the .npz is the data, the .json the
provenance/parameters. Coordinates x, y are in mm in the MODEL frame and the
field arrays are shaped (n_frames, n_points) so that
gui.sensitivity.field_metrics can compare them with numerical fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import numpy as np

FORMAT_VERSION = 1


def _json_path(npz_path) -> Path:
    return Path(npz_path).with_suffix(".json")


def save_dic_field(npz_path, x, y, t, V1, V2, Vmag, valid,
                   meta: Optional[dict] = None, extra: Optional[dict] = None,
                   units: Optional[dict] = None) -> Path:
    """Write a DIC velocity field as <stem>.npz + <stem>.json. `extra` holds
    additional (n_frames, n_points) arrays (e.g. strain / strain-rate fields)
    saved alongside V1/V2/Vmag; `units` maps array names to units (stored in
    the json). Returns the .npz path."""
    p = Path(npz_path)
    if p.suffix.lower() != ".npz":
        p = p.with_suffix(".npz")
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    t = np.asarray(t, np.float64)
    arrays = {"x": x, "y": y, "t": t,
              "V1": np.asarray(V1, np.float32), "V2": np.asarray(V2, np.float32),
              "Vmag": np.asarray(Vmag, np.float32), "valid": np.asarray(valid, bool)}
    if extra:
        for k, v in extra.items():
            arrays[k] = np.asarray(v, np.float32)
    np.savez_compressed(p, **arrays)

    info = {
        "format_version": FORMAT_VERSION,
        "modality": "dic",
        "units": units or {"x": "mm", "y": "mm", "t": "s", "V": "mm/s"},
        "fields": sorted(k for k in arrays if k not in ("x", "y", "t", "valid")),
        "n_points": int(x.size),
        "n_frames": int(t.size),
    }
    if meta:
        info.update(meta)
    with open(_json_path(p), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return p


def load_dic_field(npz_path) -> dict:
    """Load a DIC field. Returns a dict with the arrays plus a 'meta' key (the
    parsed .json, or {} if absent)."""
    p = Path(npz_path)
    with np.load(p, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    jp = _json_path(p)
    out["meta"] = {}
    if jp.exists():
        try:
            with open(jp, "r", encoding="utf-8") as f:
                out["meta"] = json.load(f)
        except Exception:
            out["meta"] = {}
    return out
