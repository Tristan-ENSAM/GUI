# -*- coding: utf-8 -*-
"""
Export result fields to plain .txt — one file per (instance, quantity).

Layout (as requested):
  * one row  = one element/node (row header = its 0-based index),
  * one col  = one time step (column header = the time in seconds).

So each file is a (n_elements, n_frames) matrix with a header row of times
and a leading index column. Tab-separated, no comment prefix, so it loads
directly in Excel or numpy (np.genfromtxt).
"""
from __future__ import annotations
import os
import numpy as np


def _write_matrix_txt(path, matrix, times, index_label="index"):
    """matrix: (n_rows, n_frames). times: (n_frames,). Writes a header row
    'index t0 t1 ...' then one row per element/node prefixed by its index."""
    matrix = np.asarray(matrix)
    n_rows, n_cols = matrix.shape
    times = np.asarray(times).ravel()
    if times.shape[0] != n_cols:
        # Be forgiving: fall back to a plain 0..n_cols-1 column index.
        times = np.arange(n_cols, dtype=float)
    idx = np.arange(n_rows, dtype=np.int64).reshape(-1, 1)
    data = np.hstack([idx.astype(np.float64), matrix.astype(np.float64)])
    header = index_label + "\t" + "\t".join("%.9g" % t for t in times)
    fmt = ["%d"] + ["%.7g"] * n_cols
    np.savetxt(path, data, fmt=fmt, delimiter="\t", header=header, comments="")
    return path


def export_instance_fields(bundle, instance, outdir):
    """Export every field of `instance` to '<instance>__<var>.txt'.
    Each field array is (n_frames, n_elem) and is transposed so that a row
    is an element and a column is a time step. Returns the list of paths."""
    os.makedirs(outdir, exist_ok=True)
    times = bundle.times
    info = bundle.instance(instance)
    written = []
    for var in info.field_variables:
        field = np.asarray(bundle.field(instance, var))     # (n_frames, n_elem)
        matrix = field.T                                     # (n_elem, n_frames)
        path = os.path.join(outdir, "%s__%s.txt" % (instance, var))
        _write_matrix_txt(path, matrix, times, index_label="elem_index")
        written.append(path)
    return written


def export_bundle(bundle, outdir):
    """Export the fields of all instances. Also writes, per instance, a
    '<instance>__element_centroids.txt' (index, x, y, z) so the element
    index can be mapped back to a position. Returns the list of paths."""
    os.makedirs(outdir, exist_ok=True)
    written = []
    for inst in bundle.instance_names:
        written += export_instance_fields(bundle, inst, outdir)
        # centroids (index -> x,y,z), best-effort
        try:
            cen = np.asarray(bundle.element_centroids_init(inst))
            if cen.size:
                idx = np.arange(cen.shape[0], dtype=np.int64).reshape(-1, 1)
                data = np.hstack([idx.astype(np.float64), cen.astype(np.float64)])
                p = os.path.join(outdir, "%s__element_centroids.txt" % inst)
                np.savetxt(p, data, fmt=["%d", "%.7g", "%.7g", "%.7g"],
                           delimiter="\t", header="elem_index\tx\ty\tz",
                           comments="")
                written.append(p)
        except Exception:
            pass
    return written
