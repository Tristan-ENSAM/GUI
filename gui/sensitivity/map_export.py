# -*- coding: utf-8 -*-
"""
Write the per-element sensitivity maps of a Jacobian run to disk.

The Maps tab shows dF/dparam per element and per frame (see
runner_core.jacobian_field_maps). This module saves the same data next to
the runs, in a sub-folder of the study directory, so a campaign can be
reviewed without the GUI:

  sensitivity_maps/
    maps_index.csv                 one row per (field, parameter) map
    frame_times.csv                frame index -> simulation time [s]
    mesh_nodes.csv                 node index, x, y           (ROI mesh)
    mesh_elements.csv              element index, node indices of its face
    map_<FIELD>_p<NN>_<path>.csv   element, x_c, y_c, time_mean, time_rms,
                                   then one signed column per frame
    map_<FIELD>_p<NN>_<path>_mean.png   time-mean, signed (diverging)
    map_<FIELD>_p<NN>_<path>_rms.png    time-RMS, magnitude (sequential)

The aggregates are the ones the Maps tab uses: time-mean keeps the sign,
time-RMS is a magnitude. Pure functions (matplotlib Agg, no Qt, no pyplot)
so they are unit-testable and safe outside the GUI thread.
"""
from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path

import numpy as np

from gui.core.logging_util import log_swallowed

MAPS_SUBDIR = "sensitivity_maps"

# Unit of each exported field as stored in the results bundle. The model runs
# in the Abaqus consistent system mm / s / °C (gui.core.units), V is the
# magnitude of the nodal velocity and EVF a volume fraction.
FIELD_UNITS = {"EVF": "—", "V": "mm/s", "TEMP": "°C"}


def element_faces(nodes, elems):
    """2D footprint of a hex mesh, as drawn by the Results / Maps viewers:
    project the nodes on (x, y) and angle-order each element's nodes around
    its centroid. Returns (nodes_xy (n_nodes, 2), face_idx (n_elem, n_loc))."""
    nodes_xy = np.asarray(nodes)[:, :2]
    conn = np.asarray(elems)
    P = nodes_xy[conn]                                   # (n_elem, n_loc, 2)
    c = P.mean(axis=1, keepdims=True)
    ang = np.arctan2(P[:, :, 1] - c[:, :, 1], P[:, :, 0] - c[:, :, 0])
    order = np.argsort(ang, axis=1)
    face_idx = np.take_along_axis(conn, order, axis=1)
    return nodes_xy, face_idx


def time_aggregates(S):
    """(time-mean signed, time-RMS) of an (n_frames, n_elem) map; NaN-safe
    (an element that is NaN on every frame stays NaN)."""
    S = np.asarray(S, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(S, axis=0)
        rms = np.sqrt(np.nanmean(S * S, axis=0))
    return mean, rms


def map_unit(field_unit: str, param_unit: str) -> str:
    """Unit of dF/dparam, e.g. 'mm/s / MPa'. '—' marks dimensionless."""
    f = field_unit or "—"
    p = param_unit or "—"
    if p == "—":
        return f
    if f == "—":
        return "1/%s" % p
    return "%s / %s" % (f, p)


def _safe(text: str) -> str:
    """File-name-safe version of a parameter path or field name."""
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_"
                   for ch in str(text))


def _render_png(path, verts, values, signed, title, cbar_label):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib import colormaps

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if signed:
        m = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmin, vmax, cmap_name = -m, m, "RdBu_r"
    else:
        vmin = 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        cmap_name = "inferno"
    if vmax <= vmin:
        vmax = vmin + 1e-12
    cmap = colormaps[cmap_name].copy()
    cmap.set_bad(alpha=0.0)                 # NaN elements left blank

    fig = Figure(figsize=(8, 5), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    polys = PolyCollection(verts, array=np.ma.masked_invalid(values),
                           cmap=cmap, edgecolors="none")
    polys.set_clim(vmin, vmax)
    ax.add_collection(polys)
    xy = verts.reshape(-1, 2)
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(polys, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(str(path))


def write_maps(out_dir, maps, nodes_xy, face_idx, *, param_info,
               centroids_xy=None, frame_times=None, scheme="",
               deltas=None, field_units=None, images=True):
    """Write every (field, parameter) map to `out_dir` (created if needed).

    maps        : {field_var: {param_path: S (n_frames, n_elem)}}
    nodes_xy    : (n_nodes, 2) ; face_idx : (n_elem, n_loc) — see
                  element_faces
    param_info  : {param_path: (label, unit)} — unit of the plan's
                  displayed values (dF/dparam is per that unit)
    centroids_xy: (n_elem, 2) element centroids; default = face vertex mean
    frame_times : (n_frames,) simulation times, optional
    deltas      : {param_path: FD step}, recorded in the index
    images      : False to skip the PNGs (tables only)

    Returns the list of written file Paths. A map whose size does not match
    the mesh is skipped (logged), never written half-way.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    field_units = dict(FIELD_UNITS, **(field_units or {}))
    deltas = deltas or {}
    nodes_xy = np.asarray(nodes_xy, dtype=float)
    face_idx = np.asarray(face_idx)
    verts = nodes_xy[face_idx]                          # (n_elem, n_loc, 2)
    n_elem = verts.shape[0]
    if centroids_xy is None:
        centroids_xy = verts.mean(axis=1)
    centroids_xy = np.asarray(centroids_xy, dtype=float)[:, :2]
    written = []

    # --- shared tables: mesh + frame times ---------------------------------
    p = out / "mesh_nodes.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["node", "x_mm", "y_mm"])
        for i, (x, y) in enumerate(nodes_xy):
            w.writerow([i, repr(float(x)), repr(float(y))])
    written.append(p)
    p = out / "mesh_elements.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["element"] + ["n%d" % j for j in range(face_idx.shape[1])])
        for i, row in enumerate(face_idx):
            w.writerow([i] + [int(v) for v in row])
    written.append(p)
    if frame_times is not None:
        p = out / "frame_times.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s"])
            for i, t in enumerate(np.asarray(frame_times, dtype=float)):
                w.writerow([i, repr(float(t))])
        written.append(p)

    # --- one table (+ images) per map ---------------------------------------
    index_rows = []
    paths_order = list(param_info)
    for var, per in maps.items():
        for path, S in per.items():
            S = np.asarray(S, dtype=float)
            if S.ndim != 2 or S.shape[1] != n_elem:
                logging.getLogger(__name__).warning(
                    "map %s @ %s skipped: shape %s vs %d mesh elements",
                    var, path, S.shape, n_elem)
                continue
            num = paths_order.index(path) + 1 if path in paths_order else 0
            stem = "map_%s_p%02d_%s" % (_safe(var), num, _safe(path))
            label, punit = param_info.get(path, (path, "—"))
            unit = map_unit(field_units.get(var, ""), punit)
            mean, rms = time_aggregates(S)

            p_csv = out / (stem + ".csv")
            header = ",".join(["element", "x_c_mm", "y_c_mm", "time_mean",
                               "time_rms"]
                              + ["frame_%03d" % k for k in range(S.shape[0])])
            table = np.column_stack([np.arange(n_elem), centroids_xy[:, :2],
                                     mean, rms, S.T])
            # NaN (element without data, e.g. empty Eulerian cell) is written
            # as the literal "nan".
            with open(p_csv, "w", newline="", encoding="utf-8-sig") as f:
                np.savetxt(f, table, delimiter=",", header=header,
                           comments="",
                           fmt=["%d"] + ["%.9g"] * (table.shape[1] - 1))
            written.append(p_csv)

            pngs = []
            if images:
                for suffix, vals, signed, what in (
                        ("_mean.png", mean, True,
                         "time mean over %d frames (signed)" % S.shape[0]),
                        ("_rms.png", rms, False,
                         "time RMS over %d frames (magnitude)" % S.shape[0])):
                    p_png = out / (stem + suffix)
                    try:
                        _render_png(p_png, verts, vals, signed,
                                    "d%s/d(%s) — %s" % (var, label, what),
                                    "d%s/d(%s)  [%s]" % (var, label, unit))
                        written.append(p_png)
                        pngs.append(p_png.name)
                    except Exception:
                        log_swallowed("rendering map image %s" % p_png,
                                      level=logging.WARNING)
            index_rows.append({
                "field": var, "parameter": path, "label": label,
                "map_unit": unit, "scheme": scheme,
                "delta": _num(deltas.get(path, float("nan"))),
                "delta_unit": punit,
                "n_frames": S.shape[0], "n_elements": n_elem,
                "table": p_csv.name, "images": ";".join(pngs)})

    p = out / "maps_index.csv"
    cols = ["field", "parameter", "label", "map_unit", "scheme", "delta",
            "delta_unit", "n_frames", "n_elements", "table", "images"]
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(index_rows)
    written.append(p)
    return written


def _num(v) -> str:
    """Compact CSV number; NaN -> empty cell (Excel-friendly)."""
    v = float(v)
    return "" if not np.isfinite(v) else repr(v)
