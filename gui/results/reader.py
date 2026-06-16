# -*- coding: utf-8 -*-
"""
ResultsBundle — read the (.json + .npz) pair produced by the Abaqus
extractor (or by the fake-builder used for tests).

Design:
  - The bundle is loaded eagerly on construction (both .json and .npz)
    so we fail fast if anything is wrong. The .npz arrays are MEMORY-
    MAPPED, so opening a multi-GB run is cheap until you actually
    request a slice.
  - Field arrays are exposed via `bundle.field(instance, var)` which
    returns a numpy view. Use `field(...)[frame_idx]` to get a single
    snapshot.
  - The bundle is read-only. Writing is handled by the extractor or
    the fake-builder, not here.
  - Errors during load raise `ResultsLoadError`. Don't let numpy or
    json blow up in the caller's face — wrap and re-raise with a
    helpful message.
"""
from __future__ import annotations
from pathlib import Path
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from gui.core.logging_util import log_swallowed


class ResultsLoadError(RuntimeError):
    """Raised when a (.json + .npz) pair cannot be loaded — wrong path,
    version mismatch, malformed schema, missing companion file..."""


@dataclass
class InstanceInfo:
    """Lightweight summary of one instance (Euler, Tool, Workpiece).
    Mirrors the corresponding sub-dict of the .json's `instances` map."""
    name:                  str
    kind:                  str               # "eulerian" | "lagrangian"
    element_type:          str               # "EC3D8RT" | "C3D8RT" | ...
    n_nodes:               int
    n_elements:            int
    n_frames:              int
    field_variables:       list[str]
    has_displacements:     bool = False


@dataclass
class HistoryInfo:
    """Summary of the history-output block."""
    n_samples: int       = 0
    variables: list[str] = field(default_factory=list)


class ResultsBundle:
    """A loaded results bundle: metadata + memory-mapped arrays.

    Usage:
        bundle = ResultsBundle.load("path/to/Cutting_job.results.npz")
        bundle.times                          # (n_frames,)
        bundle.instance_names                 # ["Euler"]
        info = bundle.instance("Euler")       # InstanceInfo
        peeq = bundle.field("Euler", "PEEQ")  # (n_frames, n_elements), memory-mapped
        peeq_f3 = peeq[3]                     # one frame, shape (n_elements,)
        rf1 = bundle.history("RF1_RP")        # (n_samples,)
        bundle.close()                        # release the .npz handle
    """

    def __init__(self, meta: dict, arr: np.lib.npyio.NpzFile,
                 json_path: Path, npz_path: Path):
        # Private constructor — use ResultsBundle.load(...) instead.
        self._meta       = meta
        self._arr        = arr
        self._json_path  = json_path
        self._npz_path   = npz_path

        # Pre-compute and cache the InstanceInfo objects so callers can
        # iterate cheaply without poking inside the raw meta dict.
        self._instances: dict[str, InstanceInfo] = {}
        for name, info in self._meta.get("instances", {}).items():
            self._instances[name] = InstanceInfo(
                name=name,
                kind=info.get("kind", "eulerian"),
                element_type=info.get("element_type", "?"),
                n_nodes=int(info.get("n_nodes", 0)),
                n_elements=int(info.get("n_elements", 0)),
                n_frames=int(info.get("n_frames", 0)),
                field_variables=list(info.get("field_variables", [])),
                has_displacements=bool(info.get("has_displacements", False)),
            )

        ho = self._meta.get("history", {})
        self._history_info = HistoryInfo(
            n_samples=int(ho.get("n_samples", 0)),
            variables=list(ho.get("variables", [])),
        )

    # =====================================================================
    # Construction
    # =====================================================================
    @classmethod
    def load(cls, path: str | Path) -> "ResultsBundle":
        """Load a bundle from a .npz path (the .json is found beside it).

        Accepts either:
          - path/to/<name>.results.npz   (the canonical naming)
          - path/to/<name>.results.json
          - path/to/<name>.results       (both extensions auto-appended)
        """
        p = Path(path)
        # Resolve the (.json, .npz) pair from whatever the user passed.
        if p.suffix == ".npz":
            npz_path  = p
            json_path = p.with_suffix(".json")
        elif p.suffix == ".json":
            json_path = p
            npz_path  = p.with_suffix(".npz")
        else:
            # Treat as a basename and append both extensions
            json_path = p.with_suffix(".json")
            npz_path  = p.with_suffix(".npz")

        if not json_path.exists():
            raise ResultsLoadError(
                f"Metadata file not found: {json_path}\n"
                "A results bundle is a (.json + .npz) PAIR — both must be present."
            )
        if not npz_path.exists():
            raise ResultsLoadError(
                f"Array file not found: {npz_path}\n"
                "A results bundle is a (.json + .npz) PAIR — both must be present."
            )

        # Load metadata
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ResultsLoadError(
                f"Could not parse {json_path}:\n  {type(e).__name__}: {e}"
            ) from e

        # Version check (we accept anything up to our current schema)
        v = meta.get("format_version")
        if v is None:
            raise ResultsLoadError(
                f"{json_path} is missing 'format_version' — is this really a results bundle?"
            )
        if int(v) > cls.SUPPORTED_VERSION:
            raise ResultsLoadError(
                f"{json_path} declares format_version={v}, but this version "
                f"of the reader only understands up to {cls.SUPPORTED_VERSION}.\n"
                "Update the GUI to read this file."
            )

        # Memory-map the array bundle (cheap, lazy)
        try:
            arr = np.load(npz_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as e:
            raise ResultsLoadError(
                f"Could not load arrays from {npz_path}:\n  {type(e).__name__}: {e}"
            ) from e

        return cls(meta, arr, json_path, npz_path)

    SUPPORTED_VERSION = 1

    def close(self):
        """Release the underlying .npz file handle. Idempotent."""
        if self._arr is not None:
            try:
                self._arr.close()
            except Exception:
                log_swallowed("closing the .npz handle", level=logging.DEBUG)
            self._arr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __repr__(self):
        return (
            f"<ResultsBundle job_name={self.job_name!r} "
            f"step={self.step_name!r} "
            f"instances={list(self._instances)} "
            f"n_frames={len(self.times)}>"
        )

    # =====================================================================
    # Properties (read-only views into metadata)
    # =====================================================================
    @property
    def meta(self) -> dict:
        """Raw metadata dict (read-only — don't mutate)."""
        return self._meta

    @property
    def job_name(self) -> str:
        return self._meta.get("job_name", "<unknown>")

    @property
    def step_name(self) -> str:
        return self._meta.get("step_name", "Cut")

    @property
    def source_odb(self) -> str:
        return self._meta.get("source_odb", "<unknown>")

    @property
    def times(self) -> np.ndarray:
        """Frame times (n_frames,) in seconds."""
        return np.asarray(self._arr["times"])

    @property
    def n_frames(self) -> int:
        return int(self.times.shape[0])

    @property
    def roi(self) -> Optional[dict]:
        """ROI bbox used at extraction, or None if no filtering was applied."""
        r = self._meta.get("roi", {})
        if not r.get("applied", False):
            return None
        return {
            "xmin": float(r["xmin"]), "xmax": float(r["xmax"]),
            "ymin": float(r["ymin"]), "ymax": float(r["ymax"]),
            "zmin": float(r["zmin"]), "zmax": float(r["zmax"]),
        }

    @property
    def model_config(self) -> dict:
        """The snapshot of cfg.to_params_dict() that produced this run."""
        return self._meta.get("model_config", {})

    # =====================================================================
    # Instance access
    # =====================================================================
    @property
    def instance_names(self) -> list[str]:
        return list(self._instances.keys())

    def instance(self, name: str) -> InstanceInfo:
        """Get the InstanceInfo for an instance. Raises KeyError if unknown."""
        try:
            return self._instances[name]
        except KeyError:
            raise KeyError(
                f"No such instance {name!r}. Available: {self.instance_names}"
            )

    def nodes_init(self, instance: str) -> np.ndarray:
        """Initial node positions: (n_nodes, 3), float32, memory-mapped."""
        return np.asarray(self._arr[f"{instance}__nodes_init"])

    def elements(self, instance: str) -> np.ndarray:
        """Element connectivity: (n_elements, 8), int32, 0-based node indices."""
        return np.asarray(self._arr[f"{instance}__elements"])

    def element_centroids_init(self, instance: str) -> np.ndarray:
        """Initial element centroids: (n_elements, 3), float32, memory-mapped."""
        return np.asarray(self._arr[f"{instance}__element_centroids_init"])

    def displacements(self, instance: str) -> Optional[np.ndarray]:
        """(n_frames, n_nodes, 3) displacement array, or None if not stored.
        Only present for Lagrangian instances where the mesh moves."""
        key = f"{instance}__displacements"
        if key not in self._arr.files:
            return None
        return np.asarray(self._arr[key])

    def field(self, instance: str, var: str) -> np.ndarray:
        """(n_frames, n_elements) array for a given field variable.
        Memory-mapped — slicing returns small numpy arrays without
        copying the whole field."""
        key = f"{instance}__fields__{var}"
        if key not in self._arr.files:
            avail = self.instance(instance).field_variables
            raise KeyError(
                f"No field {var!r} on instance {instance!r}. Available: {avail}"
            )
        return np.asarray(self._arr[key])

    # =====================================================================
    # History access
    # =====================================================================
    @property
    def history_info(self) -> HistoryInfo:
        return self._history_info

    @property
    def history_time(self) -> np.ndarray:
        """(n_samples,) — time stamps of the history-output samples.
        Distinct from `times`: history is typically sampled at a fixed
        time interval (every 1e-6 s by default), much finer than the
        field-output frames."""
        key = "history__time"
        if key not in self._arr.files:
            return np.zeros((0,))
        return np.asarray(self._arr[key])

    def history(self, var: str) -> np.ndarray:
        """(n_samples,) — history values for `var`."""
        key = f"history__{var}"
        if key not in self._arr.files:
            avail = self._history_info.variables
            raise KeyError(
                f"No history variable {var!r}. Available: {avail}"
            )
        return np.asarray(self._arr[key])
