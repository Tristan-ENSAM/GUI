# -*- coding: utf-8 -*-
"""ZOI sampling helpers shared by the sizing studies.

Gathered here by lot L6 of the correction report, when the modules that used
to host them were removed (mesh_opt, domain_opt, domain_convergence: the
abandoned mesh/domain pipeline and the previous domain study). The code is
unchanged; only the location moved.

* roi_grid           - fixed regular grid of evaluation points over a box;
* nearest_samples    - element field resampled on points by nearest centroid;
* element_centroids_xy - initial (x, y) centroids of an instance;
* window_mask        - frames inside the settled time window T;
* history_window_mean - windowed mean of a history channel.

Pure host-side Python (CPython 3.x).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Fixed grid + nearest-neighbour resampling
# ---------------------------------------------------------------------------
def roi_grid(roi, step: float) -> np.ndarray:
    """Regular grid of evaluation points covering the box.

    roi = (xmin, xmax, ymin, ymax); `step` is the spacing (mesh-independent).
    Points sit at xmin + k*step (and similarly in y), always including a point
    at or before xmax/ymax. Returns (N_p, 2)."""
    xmin, xmax, ymin, ymax = roi
    if step <= 0:
        raise ValueError("step must be > 0")
    nx = max(1, int(math.floor((xmax - xmin) / step + 1e-9)) + 1)
    ny = max(1, int(math.floor((ymax - ymin) / step + 1e-9)) + 1)
    xs = xmin + step * np.arange(nx)
    ys = ymin + step * np.arange(ny)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])


def element_centroids_xy(bundle, inst) -> np.ndarray:
    """(n_elements, 2) initial element centroids (x, y) of an instance."""
    c = np.asarray(bundle.element_centroids_init(inst), dtype=float)
    return c[:, :2]


def _nearest_indices(centroids: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid for each point. Uses scipy's cKDTree when
    available, else a vectorised brute-force fallback."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(centroids).query(points)[1]
    except Exception:
        diff = points[:, None, :] - centroids[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        return np.argmin(d2, axis=1)


def nearest_samples(bundle, var, inst, points, frames=None) -> np.ndarray:
    """Resample element field `var` onto `points` (N_p, 2) by nearest centroid.
    Returns (N_t, N_p). `frames` optionally selects frames by index."""
    centroids = element_centroids_xy(bundle, inst)
    idx = _nearest_indices(centroids, np.asarray(points, dtype=float))
    f = np.asarray(bundle.field(inst, var), dtype=float)
    if frames is not None:
        f = f[frames]
    if f.ndim == 1:
        f = f[None, :]
    return f[:, idx]


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------
def window_mask(times: np.ndarray, w_start: float = 0.3,
                w_end: float = 1.0) -> np.ndarray:
    """Boolean mask of frames whose time lies in [w_start, w_end] * t_end.

    `times` are frame times (monotonically increasing). w_start/w_end are two
    fractions in [0, 1] delimiting the settled window: (0.3, 1.0) discards the
    first 30 % transient and keeps the rest. w_start <= w_end is required.
    """
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return np.zeros(0, dtype=bool)
    if not (0.0 <= w_start <= w_end <= 1.0):
        raise ValueError(
            "window must satisfy 0 <= w_start <= w_end <= 1, got (%r, %r)"
            % (w_start, w_end))
    t_end = t[-1]
    if t_end <= 0.0:
        return np.zeros(t.shape, dtype=bool)
    return (t >= w_start * t_end) & (t <= w_end * t_end)


def history_window_mean(bundle, channel: str,
                        w_start: float, w_end: float) -> Optional[float]:
    """Windowed mean of a scalar history channel, or None if unavailable.

    Uses the HISTORY time base (bundle.history_time), which is sampled
    independently from the field frames.
    """
    try:
        y = np.asarray(bundle.history(channel), dtype=float)
        t = np.asarray(bundle.history_time, dtype=float)
    except Exception:
        return None
    if y.size == 0 or t.size != y.size:
        return None
    m = window_mask(t, w_start, w_end)
    if not m.any():
        return None
    return float(np.nanmean(y[m]))
