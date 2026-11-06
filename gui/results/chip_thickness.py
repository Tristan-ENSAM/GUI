# -*- coding: utf-8 -*-
"""Chip thickness e(t) of a run, measured like the high-speed camera images
of an orthogonal cutting test.

Experimental method (A. Pouliquen, PhD thesis, ENSAM 2022, section III.2.3,
Fig. III.21): a line of pixels parallel to the cutting direction, at a fixed
height y0 above the machined surface, crosses the chip; on each image e is
the distance along that line between the tool contour (rake face) and the
free surface of the chip. The signal e(t) is then averaged over the test.

Simulation counterpart (choices of 2026-11-06, to be confirmed):

* the line is the row of Eulerian elements whose centroid is nearest to
  y = y_surface + y0, y_surface the top of the workpiece
  (geometry.euler.workpiece_position.y0) and y0 the height above it
  (default: the uncut chip thickness t1 = workpiece y0 - tool y0);
* the material flows in +x towards the tool (cel_model.py, v1 =
  +cutting speed), so the chip is the material run of that row with the
  LARGEST x (the one against the rake face); a run is a set of consecutive
  elements with EVF >= threshold;
* e = h * (sum of EVF over the run + EVF of the element on each side of it),
  so the partly filled cells at the two interfaces count by their fill;
* a frame with no material run on the row, or whose run reaches an end of
  the extracted row (the chip may go on beyond the ROI), gives NaN;
* the QoI is the mean of e over the frames of the window T that are not NaN
  (NaN when none is).

Hypothesis (not verified on runs): at y0 = t1 the line crosses the formed
chip, above the end of the primary shear zone and below its curl.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

EVF_VAR = "EVF"


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def surface_and_t1(model_config) -> Tuple[Optional[float], Optional[float]]:
    """(y of the workpiece top, t1) from a run's model config, in either the
    run-parameters layout (geometry.euler.workpiece_position) or the profile
    layout (workpiece.position); (None, None) when absent."""
    mc = model_config if isinstance(model_config, dict) else {}
    wp = (_get(mc, "geometry", "euler", "workpiece_position", "y0")
          if _get(mc, "geometry") is not None
          else _get(mc, "workpiece", "position", "y0"))
    tool = (_get(mc, "geometry", "tool", "position", "y0")
            if _get(mc, "geometry") is not None
            else _get(mc, "tool", "position", "y0"))
    try:
        ys = float(wp)
    except (TypeError, ValueError):
        return None, None
    try:
        t1 = ys - float(tool)
    except (TypeError, ValueError):
        t1 = None
    return ys, (t1 if t1 is not None and t1 > 0 else None)


def row_indices(centroids: np.ndarray, y_line: float, h: float) -> np.ndarray:
    """Indices of the elements of the row nearest to y_line, sorted by x."""
    y = centroids[:, 1]
    if y.size == 0:
        return np.zeros(0, dtype=int)
    y_row = y[np.argmin(np.abs(y - y_line))]
    idx = np.nonzero(np.abs(y - y_row) <= 0.25 * h)[0]
    return idx[np.argsort(centroids[idx, 0])]


def width_on_row(evf: np.ndarray, h: float, threshold: float = 0.5) -> float:
    """Width of the chip on one row of EVF values sorted by x (see the module
    docstring); NaN when there is no run or the run touches an end."""
    evf = np.asarray(evf, dtype=float)
    mat = evf >= threshold
    if not mat.any():
        return float("nan")
    hi = int(np.nonzero(mat)[0][-1])            # last material element
    lo = hi
    while lo - 1 >= 0 and mat[lo - 1]:
        lo -= 1
    if lo == 0 or hi == evf.size - 1:
        return float("nan")                     # truncated by the row ends
    fill = float(np.sum(evf[lo:hi + 1])) + float(evf[lo - 1]) \
        + float(evf[hi + 1])
    return float(h) * fill


def thickness_series(bundle, inst: str, h: float,
                     y0: Optional[float] = None,
                     threshold: float = 0.5) -> np.ndarray:
    """e(t) for every frame of the bundle (NaN where not measurable).
    Raises ValueError when the line cannot be placed."""
    ys, t1 = surface_and_t1(getattr(bundle, "model_config", None))
    if ys is None:
        raise ValueError("the run has no workpiece position")
    if y0 is None:
        if t1 is None:
            raise ValueError("the run has no uncut chip thickness t1")
        y0 = t1
    y_line = ys + float(y0)
    cent = np.asarray(bundle.element_centroids_init(inst), dtype=float)[:, :2]
    crop = getattr(bundle, "roi", None)
    if isinstance(crop, dict) and not (
            crop["ymin"] - 1e-12 <= y_line <= crop["ymax"] + 1e-12):
        raise ValueError("the chip-thickness line y = %.4g mm is outside the "
                         "extracted ROI" % y_line)
    idx = row_indices(cent, y_line, h)
    if idx.size < 3:
        raise ValueError("no element row at y = %.4g mm" % y_line)
    if abs(cent[idx[0], 1] - y_line) > h:
        raise ValueError("the chip-thickness line y = %.4g mm is outside the "
                         "Eulerian domain" % y_line)
    f = np.asarray(bundle.field(inst, EVF_VAR), dtype=float)
    if f.ndim == 1:
        f = f[None, :]
    rows = f[:, idx]
    return np.array([width_on_row(r, h, threshold) for r in rows])


def mean_thickness(bundle, inst: str, h: float, tmask: np.ndarray,
                   y0: Optional[float] = None,
                   threshold: float = 0.5) -> float:
    """Mean of e(t) over the frames of `tmask` that are measurable."""
    e = thickness_series(bundle, inst, h, y0, threshold)
    tmask = np.asarray(tmask, dtype=bool)
    if tmask.shape != e.shape:
        raise ValueError("window mask and frames differ in length")
    sel = e[tmask]
    sel = sel[np.isfinite(sel)]
    return float(np.mean(sel)) if sel.size else float("nan")
