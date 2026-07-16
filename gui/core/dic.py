# -*- coding: utf-8 -*-
"""
Digital Image Correlation — velocity fields from a visible image sequence.

This module holds the *engines* (pure, no Qt). The first engine is a **local
subset** DIC: a fixed grid of points is defined in the reference frame, and
for each consecutive image pair the local displacement of each subset is found
by normalised cross-correlation (ZNCC, via cv2.matchTemplate) with parabolic
sub-pixel peak refinement. A global Q4 (q4dic) engine is planned separately.

Design choices (see the DIC tab / FORMAT.md):
  - **Eulerian, fixed grid**: the same grid of points is used for every pair,
    so velocity is reported at fixed spatial points — the natural counterpart
    of the CEL Eulerian velocity field used in the inverse identification.
  - **Incremental**: displacement is measured between frame i and i+1, so the
    velocity is instantaneous; n_frames = n_images - 1.
  - velocity (mm/s): V = displacement_px * mm_per_px * fps, with the y axis
    flipped to the model frame (image y is down, model y is up). Coordinates
    x, y are mapped to the model frame (origin at image centre) like the
    Alignment tab, so DIC and the model share one frame.

The result arrays follow gui/results/FORMAT.md (experimental DIC section):
  x, y : (n_points,) mm in the model frame
  t    : (n_frames,) s, midpoint of each image pair, relative to the trigger
  V1, V2, Vmag : (n_frames, n_points) mm/s
  valid: (n_frames, n_points) bool (ZNCC peak >= threshold)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import time
import numpy as np

from gui.core.alignment import pixel_to_model


@dataclass
class DicParams:
    """Local-DIC settings. `subset` and `search` are half-irrelevant names on
    purpose: `subset` is the FULL square subset side (px, made odd), `search`
    is the half-width (px) of the search window added around the subset."""
    engine: str = "local"        # "local" (here) | "global" (q4dic, later)
    subset: int = 31             # subset side in px (forced odd, >=5)
    step: int = 16               # grid spacing in px
    search: int = 16             # half search window in px
    zncc_min: float = 0.5        # validity threshold on the ZNCC peak
    subpixel: bool = True

    def to_json_dict(self) -> dict:
        return asdict(self)


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a.astype(np.float32)


def make_grid(roi: Tuple[float, float, float, float], step: int,
              margin: int = 0) -> np.ndarray:
    """Regular grid of point centres (px) inside `roi`=(x,y,w,h). `margin`
    insets the grid from the ROI border (use subset//2 + search to keep
    subsets fully inside the image). Returns (n_points, 2) float array."""
    x, y, w, h = roi
    x0 = x + margin; x1 = x + w - margin
    y0 = y + margin; y1 = y + h - margin
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 2), float)
    xs = np.arange(x0, x1 + 1e-9, step, dtype=float)
    ys = np.arange(y0, y1 + 1e-9, step, dtype=float)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _subpixel_parabola(c: np.ndarray, iy: int, ix: int) -> Tuple[float, float]:
    """Parabolic sub-pixel refinement of a correlation peak at integer (iy,ix)
    within the 2D map `c`. Returns (sy, sx) sub-pixel offsets in (-1, 1)."""
    sy = sx = 0.0
    if 0 < iy < c.shape[0] - 1:
        a, b, d = c[iy - 1, ix], c[iy, ix], c[iy + 1, ix]
        den = (a - 2 * b + d)
        if abs(den) > 1e-12:
            sy = 0.5 * (a - d) / den
    if 0 < ix < c.shape[1] - 1:
        a, b, d = c[iy, ix - 1], c[iy, ix], c[iy, ix + 1]
        den = (a - 2 * b + d)
        if abs(den) > 1e-12:
            sx = 0.5 * (a - d) / den
    return float(np.clip(sy, -1, 1)), float(np.clip(sx, -1, 1))


def correlate_local(ref: np.ndarray, cur: np.ndarray, points: np.ndarray,
                    subset: int = 31, search: int = 16, zncc_min: float = 0.5,
                    subpixel: bool = True):
    """Local subset ZNCC displacement of each point from `ref` to `cur`.

    Returns (disp, valid, score):
      disp  : (n_points, 2) displacement (dx, dy) in pixels (image axes)
      valid : (n_points,) bool, ZNCC peak >= zncc_min and subset in-bounds
      score : (n_points,) ZNCC peak value
    """
    import cv2
    R = _to_gray_f32(ref)
    C = _to_gray_f32(cur)
    H, W = R.shape
    s = int(subset) | 1            # force odd
    half = s // 2
    sr = int(search)
    pts = np.asarray(points, float).reshape(-1, 2)
    n = len(pts)
    disp = np.full((n, 2), np.nan)
    valid = np.zeros(n, bool)
    score = np.full(n, np.nan)
    for k in range(n):
        px, py = pts[k]
        ix, iy = int(round(px)), int(round(py))
        # reference subset (template) fully inside ref
        if ix - half < 0 or iy - half < 0 or ix + half >= W or iy + half >= H:
            continue
        tmpl = R[iy - half:iy + half + 1, ix - half:ix + half + 1]
        # search window in cur, clamped to image
        x0 = max(0, ix - half - sr); x1 = min(W, ix + half + 1 + sr)
        y0 = max(0, iy - half - sr); y1 = min(H, iy + half + 1 + sr)
        win = C[y0:y1, x0:x1]
        if win.shape[0] < s or win.shape[1] < s:
            continue
        corr = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, maxloc = cv2.minMaxLoc(corr)
        cx, cy = maxloc            # top-left of best match within `corr`
        sy = sx = 0.0
        if subpixel:
            sy, sx = _subpixel_parabola(corr, cy, cx)
        # displacement = (matched top-left in cur) - (template top-left in ref)
        match_x = x0 + cx + sx
        match_y = y0 + cy + sy
        ref_x = ix - half
        ref_y = iy - half
        disp[k] = (match_x - ref_x, match_y - ref_y)
        score[k] = peak
        valid[k] = peak >= zncc_min
    return disp, valid, score


def grid_from_points(x: np.ndarray, y: np.ndarray):
    """If (x, y) form a regular grid, return (ux, uy, ix, iy) with ux, uy the
    sorted unique axes and ix, iy the column/row index of each point; else
    None."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size == 0:
        return None
    ux = np.unique(np.round(x, 6)); uy = np.unique(np.round(y, 6))
    if ux.size * uy.size != x.size:
        return None
    ix = np.searchsorted(ux, np.round(x, 6))
    iy = np.searchsorted(uy, np.round(y, 6))
    return ux, uy, ix, iy


# von Mises equivalent (2D, incompressible plane assumption: e_zz = -(exx+eyy)).
# Documented so the convention is explicit and can be changed if needed.
def _equiv(exx, eyy, exy):
    ezz = -(exx + eyy)
    return np.sqrt(2.0 / 3.0 * (exx ** 2 + eyy ** 2 + ezz ** 2 + 2.0 * exy ** 2))


def compute_dic_fields(frames, points: np.ndarray, params: DicParams,
                       fps: float, mm_per_px: float, img_w: int, img_h: int,
                       trigger_offset_s: float = 0.0, progress=None,
                       point_keep=None, on_frame=None,
                       mask_per_frame: bool = False, mask_params: dict = None):
    """Incremental local DIC + derived fields on the (regular) measurement grid.

    Returns a dict:
      x, y      : (n_points,) mm, model frame
      t         : (n_frames,) s, pair midpoints
      valid     : (n_frames, n_points) bool
      grid      : (nx, ny) or None
      fields    : {name: (n_frames, n_points)} with
                  Ux, Uy, Umag     incremental displacement (mm)
                  Vx, Vy, Vmag     velocity (mm/s)
                  Exx_dot, Eyy_dot, Exy_dot, Eeq_dot   strain rate (1/s)
                  Exx, Eyy, Exy, Eeq                   cumulative strain (-)
      units     : {name: unit}

    Strain is the small-strain symmetric gradient of the displacement field
    accumulated over pairs at fixed (Eulerian) points — an approximation when
    material flows through the grid; strain rate is that increment over dt.
    Gradients need a regular grid; if the points are not a grid, only the
    displacement/velocity fields are returned (strain fields are NaN).

    ``progress(i_done, n_pairs)`` is the simple progress callback. ``on_frame``
    is an optional detailed callback invoked once per pair with a dict::

        {'index': i, 'n_pairs': N, 'n_valid': k, 'n_total': m,
         'mean_zncc': float | None, 'elapsed_s': float, 'frame_s': float}

    meant to drive a status log and an ETA in the UI; it does not affect the
    computation.

    Masking
    -------
    ``point_keep`` is a static (n_pts,) keep-mask applied to every frame (the
    legacy behaviour). When ``mask_per_frame`` is True the keep-mask is instead
    recomputed on EACH reference frame with ``point_mask`` using ``mask_params``
    (``min_intensity`` and an optional ``win``; intensity-only, the same
    criterion shown in the Search-ROI preview). A point is then valid only on
    the frames where it sits on bright-enough material; on the frames where it
    is masked out its displacement/velocity are NaN (a temporal hole) and it
    does not count as valid.

    Because the measurement grid is Eulerian (fixed points, material flows
    through), the time-CUMULATED strain has no physical meaning once material
    leaves the field of view, so this engine reports only the INSTANTANEOUS
    strain rates (``Exx_dot``/``Eyy_dot``/``Exy_dot``/``Eeq_dot``); it does not
    output cumulated ``Exx``/``Eyy``/``Exy``/``Eeq`` fields.
    """
    n_img = len(frames)
    pts = np.asarray(points, float).reshape(-1, 2)
    n_pts = len(pts)
    n_pairs = max(0, n_img - 1)
    dt = 1.0 / fps if fps else 1.0

    xy = np.array([pixel_to_model(p[0], p[1], img_w, img_h, mm_per_px)
                   for p in pts]) if n_pts else np.empty((0, 2))
    x_mm = xy[:, 0] if n_pts else np.empty(0)
    y_mm = xy[:, 1] if n_pts else np.empty(0)
    grid = grid_from_points(x_mm, y_mm)

    names = ["Ux", "Uy", "Umag", "Vx", "Vy", "Vmag",
             "Exx_dot", "Eyy_dot", "Exy_dot", "Eeq_dot", "ZNCC"]
    fields = {k: np.full((n_pairs, n_pts), np.nan) for k in names}
    valid = np.zeros((n_pairs, n_pts), bool)
    t = np.zeros(n_pairs)

    keep_static = (np.ones(n_pts, bool) if point_keep is None
                   else np.asarray(point_keep, bool))
    mp = mask_params or {}
    mask_win = int(mp.get("win", max(5, int(params.subset) // 2)))
    mask_min_int = float(mp.get("min_intensity", 0.0))

    if grid is not None:
        ux, uy, ix, iy = grid
        gx_mm = float(np.mean(np.diff(ux))) if ux.size > 1 else 1.0
        gy_mm = float(np.mean(np.diff(uy))) if uy.size > 1 else 1.0

    t_start = time.perf_counter()
    for i in range(n_pairs):
        t_frame0 = time.perf_counter()
        # Keep-mask for this pair: recomputed on the reference frame i when
        # mask_per_frame is on, otherwise the static keep-mask.
        if mask_per_frame and n_pts:
            keep = point_mask(frames[i], pts, win=mask_win,
                              min_intensity=mask_min_int)
        else:
            keep = keep_static
        disp, ok, score = correlate_local(
            frames[i], frames[i + 1], pts,
            subset=params.subset, search=params.search,
            zncc_min=params.zncc_min, subpixel=params.subpixel)
        ok = ok & keep                          # masked-out points are invalid
        # ZNCC peak as the per-point DIC quality/score (kept even where the
        # correlation is below threshold, so low-quality zones are visible);
        # NaN only where masked out.
        fields["ZNCC"][i] = np.where(keep, score, np.nan)
        ux_mm = disp[:, 0] * mm_per_px
        uy_mm = -disp[:, 1] * mm_per_px
        ux_mm = np.where(ok, ux_mm, np.nan)
        uy_mm = np.where(ok, uy_mm, np.nan)
        fields["Ux"][i] = ux_mm
        fields["Uy"][i] = uy_mm
        fields["Umag"][i] = np.hypot(ux_mm, uy_mm)
        fields["Vx"][i] = ux_mm / dt
        fields["Vy"][i] = uy_mm / dt
        fields["Vmag"][i] = np.hypot(ux_mm, uy_mm) / dt
        valid[i] = ok
        t[i] = trigger_offset_s + (i + 0.5) * dt

        if grid is not None:
            ux_g = np.full((uy.size, ux.size), np.nan)
            uy_g = np.full((uy.size, ux.size), np.nan)
            ux_g[iy, ix] = ux_mm
            uy_g[iy, ix] = uy_mm
            dUx_dy, dUx_dx = np.gradient(ux_g, gy_mm, gx_mm)
            dUy_dy, dUy_dx = np.gradient(uy_g, gy_mm, gx_mm)
            dexx = dUx_dx; deyy = dUy_dy
            dexy = 0.5 * (dUx_dy + dUy_dx)
            # Instantaneous strain rates only; cumulated strain is omitted (no
            # physical meaning on an Eulerian grid as material leaves the FOV).
            fields["Exx_dot"][i] = (dexx / dt)[iy, ix]
            fields["Eyy_dot"][i] = (deyy / dt)[iy, ix]
            fields["Exy_dot"][i] = (dexy / dt)[iy, ix]
            fields["Eeq_dot"][i] = (_equiv(dexx, deyy, dexy) / dt)[iy, ix]

        if progress is not None:
            progress(i + 1, n_pairs)
        if on_frame is not None:
            now = time.perf_counter()
            n_total = int(keep.sum())
            n_valid = int(ok.sum())
            zncc_valid = score[ok]
            mean_zncc = float(np.mean(zncc_valid)) if zncc_valid.size else None
            on_frame({"index": i, "n_pairs": n_pairs,
                      "n_valid": n_valid, "n_total": n_total,
                      "mean_zncc": mean_zncc,
                      "elapsed_s": now - t_start,
                      "frame_s": now - t_frame0})

    units = {"Ux": "mm", "Uy": "mm", "Umag": "mm",
             "Vx": "mm/s", "Vy": "mm/s", "Vmag": "mm/s",
             "Exx_dot": "1/s", "Eyy_dot": "1/s", "Exy_dot": "1/s", "Eeq_dot": "1/s",
             "ZNCC": "-"}
    return {"x": x_mm, "y": y_mm, "t": t, "valid": valid,
            "grid": (None if grid is None else (grid[0].size, grid[1].size)),
            "fields": fields, "units": units}


def point_mask(image: np.ndarray, points: np.ndarray, win: int = 15,
               min_intensity: float = 0.0):
    """Keep-mask for measurement points based on the reference frame: a point
    is kept if its local window has mean intensity >= `min_intensity` (drops
    the dark scene background / out-of-material regions). Returns a
    (n_points,) bool array.

    The texture (local std) criterion was removed: masking is intensity-only,
    consistent across the local and global engines.
    """
    g = _to_gray_f32(image)
    H, W = g.shape
    half = max(1, int(win) // 2)
    pts = np.asarray(points, float).reshape(-1, 2)
    keep = np.ones(len(pts), bool)
    for k in range(len(pts)):
        ix, iy = int(round(pts[k, 0])), int(round(pts[k, 1]))
        x0, x1 = max(0, ix - half), min(W, ix + half + 1)
        y0, y1 = max(0, iy - half), min(H, iy + half + 1)
        patch = g[y0:y1, x0:x1]
        if patch.size == 0 or patch.mean() < min_intensity:
            keep[k] = False
    return keep


def velocity_fields(frames, points: np.ndarray, params: DicParams,
                    fps: float, mm_per_px: float, img_w: int, img_h: int,
                    trigger_offset_s: float = 0.0):
    """Run incremental local DIC over a list/sequence of frames and return the
    velocity field arrays (model frame, mm/s) per FORMAT.md.

    `frames` is any sequence indexable as frames[i] -> 2D/3D image array.
    Returns dict with x, y, t, V1, V2, Vmag, valid.
    """
    n_img = len(frames)
    pts = np.asarray(points, float).reshape(-1, 2)
    n_pts = len(pts)
    n_pairs = max(0, n_img - 1)

    # fixed grid mapped to the model frame (origin at image centre, y up)
    xy = np.array([pixel_to_model(p[0], p[1], img_w, img_h, mm_per_px)
                   for p in pts]) if n_pts else np.empty((0, 2))
    x_mm = xy[:, 0] if n_pts else np.empty(0)
    y_mm = xy[:, 1] if n_pts else np.empty(0)

    V1 = np.full((n_pairs, n_pts), np.nan)
    V2 = np.full((n_pairs, n_pts), np.nan)
    Vmag = np.full((n_pairs, n_pts), np.nan)
    valid = np.zeros((n_pairs, n_pts), bool)
    t = np.zeros(n_pairs)

    dt = 1.0 / fps if fps else 1.0
    for i in range(n_pairs):
        disp, ok, _ = correlate_local(
            frames[i], frames[i + 1], pts,
            subset=params.subset, search=params.search,
            zncc_min=params.zncc_min, subpixel=params.subpixel)
        vx = disp[:, 0] * mm_per_px / dt            # +x is +x in both frames
        vy = -disp[:, 1] * mm_per_px / dt           # image y down -> model y up
        V1[i] = vx
        V2[i] = vy
        Vmag[i] = np.hypot(vx, vy)
        valid[i] = ok
        t[i] = trigger_offset_s + (i + 0.5) * dt    # midpoint of the pair

    return {"x": x_mm, "y": y_mm, "t": t,
            "V1": V1, "V2": V2, "Vmag": Vmag, "valid": valid}
