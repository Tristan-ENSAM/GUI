# -*- coding: utf-8 -*-
"""
Sensitivity MAPS — container, display reduction and .npz persistence.

A sensitivity map keeps the element axis that the scalar indices average
away: for one output field (EVF, V, TEMP...) and one perturbed parameter
(A, B, n, C, m, mu...), it holds an array of shape (n_frames, n_elements).

What a map means depends on the method that produced it, and the two are
NOT interchangeable — the map always uses the same scheme as the study it
came from:

  * Jacobian (finite differences) -> quantity "dFdtheta": the signed local
    derivative dF/dtheta per element (same FD scheme as the scalar run).
  * Morris (global screening)     -> quantities "mu_star", "sigma", "mu":
    the Morris indices formed element by element from the elementary
    effects (same definition and the same dimensionless grid step as the
    scalar SALib analysis — see field_metrics.elementwise_morris_stats).

This module is Qt-free and numpy-only, so the maths and the file format
are unit-testable without a display. The widget in
gui/widgets/sensitivity_map_panel.py renders what `reduce_map` returns.

File format (.npz, version 1)
-----------------------------
  format_version : () int
  meta           : () unicode, a JSON blob (method, parameters, labels,
                   units, and whatever the producer recorded about the run)
  nodes_xy       : (n_nodes, 2) float  — 2D node coordinates of the mesh
  faces          : (n_elements, n_loc) int — node indices, one polygon per
                   element, already ordered for drawing
  times          : (n_frames,) float   — frame times, empty if unknown
  map__<v>__<p>__<quantity> : (n_frames, n_elements) float, where <v> and
                   <p> index meta["field_vars"] and meta["param_paths"]

Everything is saved with allow_pickle=False, so a bundle written here can
be re-read by any numpy without executing anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as _dc_field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

FORMAT_VERSION = 1

# What each stored quantity is called, how it is offered in the UI, and
# whether it is a signed (diverging) or a non-negative (sequential) field.
#   entry = (quantity key, mode, label, diverging)
# `mode` is applied at display time: "signed" shows the array as stored,
# "abs" shows its magnitude.
_DISPLAY_ENTRIES = {
    "dFdtheta": [("signed", "dF/dθ (signed)", True),
                 ("abs", "|dF/dθ|", False)],
    "mu_star":  [("signed", "μ* (mean |EE|)", False)],
    "sigma":    [("signed", "σ (std EE)", False)],
    "mu":       [("signed", "μ (mean EE, signed)", True),
                 ("abs", "|μ|", False)],
}

# Display order of the quantities a method produces.
QUANTITIES_BY_METHOD = {
    "jacobian": ("dFdtheta",),
    "morris": ("mu_star", "sigma", "mu"),
}


def display_entries(quantities):
    """[(quantity, mode, label, diverging)] for the quantities available,
    in the order of `quantities`. Unknown keys are skipped."""
    out = []
    for q in quantities:
        for mode, label, diverging in _DISPLAY_ENTRIES.get(q, []):
            out.append((q, mode, label, diverging))
    return out


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------
@dataclass
class SensitivityMapSet:
    """Every map of one sensitivity study, plus the mesh to draw them on."""
    method: str                       # "jacobian" | "morris"
    maps: dict                        # {var: {param_path: {quantity: array}}}
    field_vars: list                  # ordered output fields
    param_paths: list                 # ordered perturbed parameters
    nodes_xy: np.ndarray              # (n_nodes, 2)
    faces: np.ndarray                 # (n_elements, n_loc)
    times: np.ndarray = _dc_field(default_factory=lambda: np.zeros(0))
    param_labels: dict = _dc_field(default_factory=dict)   # path -> label
    param_units: dict = _dc_field(default_factory=dict)    # path -> unit
    field_labels: dict = _dc_field(default_factory=dict)   # var  -> label
    meta: dict = _dc_field(default_factory=dict)           # free-form

    # -- convenience ----------------------------------------------------
    @property
    def quantities(self) -> tuple:
        """The quantity keys this set actually carries, in display order."""
        want = QUANTITIES_BY_METHOD.get(self.method, ())
        have = set()
        for per_param in self.maps.values():
            for per_q in per_param.values():
                have.update(per_q)
        ordered = [q for q in want if q in have]
        ordered += sorted(q for q in have if q not in want)
        return tuple(ordered)

    @property
    def n_frames(self) -> int:
        for per_param in self.maps.values():
            for per_q in per_param.values():
                for arr in per_q.values():
                    a = np.asarray(arr)
                    if a.ndim == 2 and a.size:
                        return int(a.shape[0])
        return 0

    @property
    def n_elements(self) -> int:
        return int(np.asarray(self.faces).shape[0]) if len(self.faces) else 0

    def get(self, var, param_path, quantity):
        """The (n_frames, n_elements) array, or None if absent."""
        arr = self.maps.get(var, {}).get(param_path, {}).get(quantity)
        return None if arr is None else np.asarray(arr, dtype=float)

    def param_label(self, path) -> str:
        return self.param_labels.get(path, path)

    def field_label(self, var) -> str:
        return self.field_labels.get(var, var)

    def is_drawable(self) -> bool:
        return (len(np.asarray(self.nodes_xy)) > 0
                and len(np.asarray(self.faces)) > 0
                and bool(self.maps))


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def mesh_from_bundle(bundle, instance):
    """(nodes_xy, faces) for one instance of a results bundle.

    The stored meshes are 3D hexes with a single element through the
    thickness; the drawable footprint is each element's projected face,
    whose nodes are angle-ordered around the centroid so the polygon is
    not self-crossing. Same construction as the Results tab.

    Raises whatever the bundle raises if the instance has no mesh."""
    info = bundle.instance(instance)
    nodes_xy = np.asarray(bundle.nodes_init(info.name), dtype=float)[:, :2]
    conn = np.asarray(bundle.elements(info.name))
    P = nodes_xy[conn]                               # (n_elem, n_loc, 2)
    c = P.mean(axis=1, keepdims=True)
    ang = np.arctan2(P[:, :, 1] - c[:, :, 1], P[:, :, 0] - c[:, :, 0])
    order = np.argsort(ang, axis=1)
    faces = np.take_along_axis(conn, order, axis=1)
    return nodes_xy, faces


# ---------------------------------------------------------------------------
# Display reduction
# ---------------------------------------------------------------------------
def reduce_map(S, mode="signed", frame=None, aggregate=False):
    """Reduce a (n_frames, n_elements) map to the (n_elements,) vector the
    viewer colours, plus a short text saying how time was handled.

    mode      : "signed" keeps the stored sign, "abs" takes the magnitude.
    frame     : frame index to show when `aggregate` is False (clamped).
    aggregate : reduce all frames to one map — time-MEAN in "signed" mode
                (cancellation is meaningful: it shows the net effect), time-
                RMS in "abs" mode (magnitudes do not cancel).

    Returns (values, frame_text). Raises ValueError on a non-2D map."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.size == 0:
        raise ValueError("a map must be a non-empty (n_frames, n_elements) "
                         "array; got shape %r" % (S.shape,))
    with np.errstate(invalid="ignore"):
        if aggregate:
            if mode == "abs":
                values = np.sqrt(np.nanmean(S * S, axis=0))
                text = "RMS over %d frames" % S.shape[0]
            else:
                values = np.nanmean(S, axis=0)
                text = "mean over %d frames" % S.shape[0]
        else:
            f = int(np.clip(0 if frame is None else frame, 0, S.shape[0] - 1))
            row = S[f]
            values = np.abs(row) if mode == "abs" else row
            text = "frame %d/%d" % (f, S.shape[0] - 1)
    return values, text


def color_range(values, diverging):
    """(vmin, vmax, cmap) for a reduced map.

    A diverging quantity gets a symmetric range around zero so the colour
    says which way the output moves; a non-negative one gets a sequential
    ramp from zero. An all-NaN map falls back to a unit range rather than
    raising — the viewer then shows an empty mesh."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if diverging:
        m = float(np.max(np.abs(finite))) if finite.size else 1.0
        if m == 0.0:
            m = 1.0
        return -m, m, "RdBu_r"
    vmax = float(np.max(finite)) if finite.size else 1.0
    vmin = min(0.0, float(np.min(finite)) if finite.size else 0.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax, "inferno"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _map_key(v_idx, p_idx, quantity):
    return "map__%d__%d__%s" % (v_idx, p_idx, quantity)


def save_npz(mapset: SensitivityMapSet, path) -> str:
    """Write `mapset` to `path` (.npz, compressed). Returns the path.

    The whole study goes in one file: every (field, parameter, quantity)
    map, the mesh, the frame times and the metadata needed to label them.
    `load_npz` reads it back into an equivalent SensitivityMapSet."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    meta = {
        "format_version": FORMAT_VERSION,
        "method": mapset.method,
        "field_vars": list(mapset.field_vars),
        "param_paths": list(mapset.param_paths),
        "param_labels": dict(mapset.param_labels),
        "param_units": dict(mapset.param_units),
        "field_labels": dict(mapset.field_labels),
        "quantities": list(mapset.quantities),
        "created": datetime.now().isoformat(timespec="seconds"),
        "run": dict(mapset.meta),
    }
    payload = {
        "format_version": np.array(FORMAT_VERSION, dtype=np.int32),
        "meta": np.array(json.dumps(meta, ensure_ascii=False, default=str)),
        "nodes_xy": np.asarray(mapset.nodes_xy, dtype=np.float32),
        "faces": np.asarray(mapset.faces, dtype=np.int32),
        "times": np.asarray(mapset.times, dtype=np.float64).ravel(),
    }
    for vi, var in enumerate(mapset.field_vars):
        for pi, path_ in enumerate(mapset.param_paths):
            for q, arr in mapset.maps.get(var, {}).get(path_, {}).items():
                a = np.asarray(arr, dtype=np.float32)
                if a.ndim == 2 and a.size:
                    payload[_map_key(vi, pi, q)] = a
    np.savez_compressed(str(path), **payload)
    return str(path)


class MapLoadError(RuntimeError):
    """Raised when a .npz is not a sensitivity-map bundle this code reads."""


def load_npz(path) -> SensitivityMapSet:
    """Read a .npz written by `save_npz` back into a SensitivityMapSet.

    Raises MapLoadError (never a bare numpy/JSON error) when the file is
    not a map bundle, or declares a newer format than this code knows."""
    path = Path(path)
    try:
        arr = np.load(str(path), allow_pickle=False)
    except Exception as e:
        raise MapLoadError("Could not open %s: %s" % (path, e)) from e
    try:
        if "meta" not in arr.files or "nodes_xy" not in arr.files:
            raise MapLoadError(
                "%s is not a sensitivity-map bundle (no 'meta'/'nodes_xy'). "
                "Load a file written by 'Save maps (.npz)'." % path.name)
        version = int(arr["format_version"]) if "format_version" in arr.files \
            else 1
        if version > FORMAT_VERSION:
            raise MapLoadError(
                "%s declares format_version=%d; this version reads up to %d."
                % (path.name, version, FORMAT_VERSION))
        try:
            meta = json.loads(str(arr["meta"]))
        except Exception as e:
            raise MapLoadError("Could not parse the metadata of %s: %s"
                               % (path.name, e)) from e
        field_vars = list(meta.get("field_vars", []))
        param_paths = list(meta.get("param_paths", []))
        maps: dict = {}
        for key in arr.files:
            if not key.startswith("map__"):
                continue
            try:
                _, vi, pi, quantity = key.split("__", 3)
                var = field_vars[int(vi)]
                ppath = param_paths[int(pi)]
            except (ValueError, IndexError):
                continue                   # a key we cannot place: ignore it
            maps.setdefault(var, {}).setdefault(ppath, {})[quantity] = \
                np.asarray(arr[key], dtype=float)
        if not maps:
            raise MapLoadError("%s holds no sensitivity map." % path.name)
        return SensitivityMapSet(
            method=str(meta.get("method", "jacobian")),
            maps=maps,
            field_vars=[v for v in field_vars if v in maps],
            param_paths=param_paths,
            nodes_xy=np.asarray(arr["nodes_xy"], dtype=float),
            faces=np.asarray(arr["faces"], dtype=int),
            times=np.asarray(arr["times"], dtype=float)
            if "times" in arr.files else np.zeros(0),
            param_labels=dict(meta.get("param_labels", {})),
            param_units=dict(meta.get("param_units", {})),
            field_labels=dict(meta.get("field_labels", {})),
            meta=dict(meta.get("run", {})),
        )
    finally:
        try:
            arr.close()
        except Exception:
            pass


def describe(mapset: Optional[SensitivityMapSet]) -> str:
    """One-line summary for the UI hint label."""
    if mapset is None or not mapset.is_drawable():
        return "No sensitivity map loaded."
    method = {"jacobian": "Jacobian (finite differences)",
              "morris": "Morris (global screening)"}.get(mapset.method,
                                                         mapset.method)
    return ("%s — %d parameter(s) × %d field(s), %d frame(s), "
            "%d elements." % (method, len(mapset.param_paths),
                              len(mapset.field_vars), mapset.n_frames,
                              mapset.n_elements))
