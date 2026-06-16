# -*- coding: utf-8 -*-
"""
ExperimentSession — one acquired cutting test ("essai").

This is the experimental counterpart of ModelConfig: it describes a single
planing/orthogonal-cutting test and everything derived from it. Unlike the
numerical config it lives in its OWN file (one .json per test), because an
experiment is not tied to a single model configuration — one config may be
compared against several tests.

The three raw streams (visible camera, IR camera, force signal) are assumed
to be already time-synchronised by a hardware trigger during acquisition, so
the session only needs each stream's frame rate (fps) and a common trigger
offset; a frame's time is t = trigger_offset_s + frame_index / fps.

Calibration objects, the reference geometry and the produced/imported field
files are filled in by the later sub-tabs (Calibration, Alignment, DIC, IRT);
for now they are kept as plain containers so the schema can grow without
breaking the JSON round-trip.

The .npz field files referenced here follow the contract documented in
gui/results/FORMAT.md (experimental section): coordinates x, y in mm in the
MODEL frame, a time vector t in s, and the field arrays shaped
(n_frames, n_points), so that gui.sensitivity.field_metrics can compare them
with the numerical fields after resampling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields, is_dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Raw-stream descriptors
# ---------------------------------------------------------------------------
@dataclass
class StreamCfg:
    """An image stream (visible or IR): the cutting acquisition + the
    matching no-load ('a vide') acquisition used to estimate the noise
    floor, plus the frame rate."""
    path:        str   = ""          # cutting acquisition (dir / tiff / video / .npz)
    noload_path: str   = ""          # no-load acquisition (for noise)
    fps:         float = 1000.0      # frames per second


@dataclass
class ForceCfg:
    """The cutting-force signal: file + sampling rate + column mapping.
    Convention (model side): Fc = RF1, Ff = RF2."""
    path:        str   = ""
    noload_path: str   = ""
    fps:         float = 50000.0     # samples per second
    # Column indices in the imported text/CSV file. col_t = -1 means the file
    # has no time column and time is derived from fps.
    col_t:       int   = -1
    col_fc:      int   = 0
    col_ff:      int   = 1


# ---------------------------------------------------------------------------
# Top-level session
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSession:
    name:                  str   = "experiment"
    material:              str   = ""
    cutting_speed_nominal: float = 0.0      # mm/s (nominal, metadata only)
    notes:                 str   = ""
    # Common t = 0 set by the hardware trigger (all streams share it).
    trigger_offset_s:      float = 0.0

    visible: StreamCfg = field(default_factory=lambda: StreamCfg(fps=20000.0))
    ir:      StreamCfg = field(default_factory=lambda: StreamCfg(fps=1000.0))
    forces:  ForceCfg  = field(default_factory=ForceCfg)

    # Field files produced (DIC/IRT tabs) or imported. Paths to .npz; the
    # paired .json (same stem) carries the computation parameters.
    dic_field_path: str = ""
    irt_field_path: str = ""

    # Filled by later tabs — kept as open dicts for forward compatibility.
    visible_calibration: dict = field(default_factory=dict)
    ir_calibration:      dict = field(default_factory=dict)
    reference_geometry:  dict = field(default_factory=dict)

    # -- serialisation ----------------------------------------------------
    def to_json_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Optional[dict]) -> "ExperimentSession":
        s = cls()
        if not isinstance(data, dict):
            return s
        _apply(s, data)
        return s

    def save(self, path) -> Path:
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, ensure_ascii=False)
        return p

    @classmethod
    def load(cls, path) -> "ExperimentSession":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_dict(data)

    # -- convenience ------------------------------------------------------
    def frame_time(self, stream: str, index: int) -> float:
        """Time (s) of frame `index` of a given stream, relative to the
        common trigger origin."""
        cfg = getattr(self, stream)
        fps = float(cfg.fps) if cfg.fps else 1.0
        return self.trigger_offset_s + index / fps


def _apply(obj, data: dict) -> None:
    """Recursively copy known keys from `data` into the dataclass `obj`,
    ignoring unknown keys and tolerating missing nested blocks (so older or
    partial files load with defaults)."""
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            continue
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(value, dict):
            _apply(cur, value)
        else:
            setattr(obj, key, value)
