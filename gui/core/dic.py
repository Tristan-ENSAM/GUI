# -*- coding: utf-8 -*-
"""
Digital Image Correlation — velocity fields from a visible image sequence.

This module holds the *engines* (pure, no Qt). The first engine is a **local
subset** DIC: a fixed grid of points is defined in the reference frame, and
for each consecutive image pair the local displacement of each subset is found
by normalised cross-correlation (ZNCC, via cv2.matchTemplate) at the integer
level, then refined to sub-pixel (by default an iterative Gauss-Newton on the
images; see ``DicParams.subpixel_method``). The global Q4 engine lives in
``gui.core.dic_global``.

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
from typing import Tuple
import time
import numpy as np

from gui.core.alignment import pixel_to_model


SUBPIXEL_METHODS = ("icgn", "gauss", "parabola")


@dataclass
class DicParams:
    """Local-DIC settings. `subset` is the FULL square subset side (px, odd,
    >= 5; an even value is rejected, not silently changed); `search` is the
    HALF-width (px) of the search range: displacements are searched in
    [-search, +search] and a correlation peak on the border of that range is
    reported invalid (the true displacement may lie beyond it).

    `subpixel_method` (see ``correlate_local``):
      - "icgn"     : iterative Gauss-Newton refinement on the images
                     (translation, ZNSSD, cubic-spline interpolation);
      - "gauss"    : 3-point Gaussian fit of the correlation peak;
      - "parabola" : 3-point parabolic fit (legacy; biased towards integer
                     displacements, see the DIC audit).
    `min_std_rel`: a subset (or its match) whose grey-level std is below
    ``min_std_rel * ptp(reference frame)`` is rejected (textureless or
    saturated: ZNCC is undefined there)."""
    engine: str = "local"        # "local" (here) | "global" (q4dic, later)
    subset: int = 31             # subset side in px (odd, >=5)
    step: int = 16               # grid spacing in px
    search: int = 16             # half search range in px
    zncc_min: float = 0.5        # validity threshold on the ZNCC score
    subpixel: bool = True
    subpixel_method: str = "icgn"
    min_std_rel: float = 1e-3

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
    subsets fully inside the image). Returns (n_points, 2) float array.

    The centres are INTEGER pixel indices (pixel centres): a fractional ROI
    (drawn with the mouse) is snapped inwards, so the subset actually
    correlated and the exported coordinate are the same point."""
    x, y, w, h = roi
    x0 = np.ceil(x + margin - 1e-9); x1 = np.floor(x + w - margin + 1e-9)
    y0 = np.ceil(y + margin - 1e-9); y1 = np.floor(y + h - margin + 1e-9)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 2), float)
    xs = np.arange(x0, x1 + 1e-9, step, dtype=float)
    ys = np.arange(y0, y1 + 1e-9, step, dtype=float)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def pixel_centres(points: np.ndarray) -> np.ndarray:
    """Integer pixel centres used as subset centres (round half up, the same
    rule for every point, unlike Python's round-half-to-even)."""
    return np.floor(np.asarray(points, float).reshape(-1, 2) + 0.5)


def _subpixel_parabola(c: np.ndarray, iy: int, ix: int) -> Tuple[float, float]:
    """Parabolic sub-pixel refinement of a correlation peak at integer (iy,ix)
    within the 2D map `c`. Returns (sy, sx) sub-pixel offsets in [-0.5, 0.5]
    for a genuine maximum. Legacy estimator: for a Gaussian-like peak of std
    s it returns k(s)*u for a small offset u, k = q / (2 s^2 (1 - q)),
    q = exp(-1/(2 s^2)) < 1 (k = 0.88 for s = 1.4 px): peak locking."""
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


def _subpixel_gauss(c: np.ndarray, iy: int, ix: int) -> Tuple[float, float]:
    """3-point Gaussian fit (vertex of the parabola through the LOG of the
    correlation values): exact for a Gaussian peak. Falls back to the
    parabola along an axis where a value is <= 0 (log undefined)."""
    def one(a, b, d):
        if min(a, b, d) > 0:
            a, b, d = np.log(a), np.log(b), np.log(d)
        den = a - 2 * b + d
        return 0.5 * (a - d) / den if abs(den) > 1e-12 else 0.0
    sy = sx = 0.0
    if 0 < iy < c.shape[0] - 1:
        sy = one(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix])
    if 0 < ix < c.shape[1] - 1:
        sx = one(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1])
    return float(np.clip(sy, -1, 1)), float(np.clip(sx, -1, 1))


def _icgn_translation(tmpl: np.ndarray, coeffs: np.ndarray, cx: float,
                      cy: float, u0: Tuple[float, float], max_iter: int = 20,
                      tol: float = 1e-4):
    """Sub-pixel translation by Gauss-Newton on the ZNSSD criterion, with the
    gradient of the reference subset kept fixed (inverse-compositional
    Hessian, additive translation update). The deformed image is sampled by
    cubic B-spline interpolation (``coeffs`` = ``spline_filter(cur, 3)``).

    ``(cx, cy)`` is the subset centre (integer px) in the reference, ``u0``
    the integer-peak displacement. Returns (u, zncc, ok): u = (dx, dy) px,
    zncc the ZNCC at the solution, ok False when the iteration fails
    (singular Hessian, divergence beyond 1 px from u0, no convergence)."""
    from scipy.ndimage import map_coordinates
    half = tmpl.shape[0] // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    f = tmpl.astype(float)
    fm = f - f.mean()
    fn = float(np.sqrt((fm ** 2).sum()))
    gy, gx = np.gradient(f)
    J = np.column_stack([gx.ravel(), gy.ravel()])
    Hs = J.T @ J
    if fn <= 0 or abs(np.linalg.det(Hs)) < 1e-12:
        return np.array(u0, float), float("nan"), False
    Hinv = np.linalg.inv(Hs)
    u = np.array(u0, float)
    zn = float("nan")
    for _ in range(max_iter):
        g = map_coordinates(coeffs, [cy + yy + u[1], cx + xx + u[0]],
                            order=3, mode="mirror", prefilter=False)
        gm = g - g.mean()
        gn = float(np.sqrt((gm ** 2).sum()))
        if gn <= 0:
            return u, float("nan"), False
        zn = float((fm * gm).sum() / (fn * gn))
        du = Hinv @ (J.T @ ((fm / fn - gm / gn).ravel() * fn))
        u = u + du
        if not np.all(np.isfinite(u)) or np.max(np.abs(u - np.asarray(u0))) > 1.0:
            return u, zn, False
        if float(np.hypot(*du)) < tol:
            g = map_coordinates(coeffs, [cy + yy + u[1], cx + xx + u[0]],
                                order=3, mode="mirror", prefilter=False)
            gm = g - g.mean()
            gn = float(np.sqrt((gm ** 2).sum()))
            zn = float((fm * gm).sum() / (fn * gn)) if gn > 0 else float("nan")
            return u, zn, True
    return u, zn, False


def correlate_local(ref: np.ndarray, cur: np.ndarray, points: np.ndarray,
                    subset: int = 31, search: int = 16, zncc_min: float = 0.5,
                    subpixel: bool = True, subpixel_method: str = "icgn",
                    min_std_rel: float = 1e-3, return_info: bool = False):
    """Local subset ZNCC displacement of each point from `ref` to `cur`.

    Conventions: image axes (x = column, y = row, y downward); a point is a
    pixel centre, rounded half up (``pixel_centres``); ``disp`` is the motion
    of the material from `ref` to `cur` (content moving right -> dx > 0).

    Integer search: ``cv2.matchTemplate(cur_window, ref_subset,
    TM_CCOEFF_NORMED)`` (= ZNCC) over displacements in [-search, +search].
    Sub-pixel: see ``DicParams.subpixel_method``. A point is INVALID when
      - its subset (or the search window) is not fully inside the image,
      - the subset or its match is textureless (std < min_std_rel * ptp(ref);
        ZNCC is undefined there and OpenCV returns 1 for two flat patches),
      - the integer peak lies on the border of the search range (the true
        displacement may be beyond ``search``; no sub-pixel is possible),
      - the ICGN refinement fails (``subpixel_method='icgn'``),
      - the ZNCC score is below ``zncc_min``.

    Returns (disp, valid, score):
      disp  : (n_points, 2) displacement (dx, dy) in pixels (image axes);
              NaN where the correlation could not be computed
      valid : (n_points,) bool
      score : (n_points,) ZNCC (at the refined position for 'icgn', else the
              integer peak)
    With ``return_info=True`` a 4th item is returned: a dict of (n_points,)
    bool arrays 'edge' (peak on the search border), 'flat' (textureless) and
    'icgn_failed'.
    """
    import cv2
    s = int(subset)
    if s < 5 or s % 2 == 0:
        raise ValueError("subset must be an odd integer >= 5 (got %r)" % subset)
    if subpixel_method not in SUBPIXEL_METHODS:
        raise ValueError("subpixel_method must be one of %s" % (SUBPIXEL_METHODS,))
    R = _to_gray_f32(ref)
    C = _to_gray_f32(cur)
    H, W = R.shape
    half = s // 2
    sr = int(search)
    if sr < 1:
        raise ValueError("search must be >= 1 px")
    min_std = float(min_std_rel) * float(np.ptp(R))
    use_icgn = subpixel and subpixel_method == "icgn"
    coeffs = None
    if use_icgn:
        from scipy.ndimage import spline_filter
        coeffs = spline_filter(C.astype(np.float64), order=3, mode="mirror")
    centres = pixel_centres(points)
    n = len(centres)
    disp = np.full((n, 2), np.nan)
    valid = np.zeros(n, bool)
    score = np.full(n, np.nan)
    edge = np.zeros(n, bool)
    flat = np.zeros(n, bool)
    failed = np.zeros(n, bool)
    for k in range(n):
        ix, iy = int(centres[k, 0]), int(centres[k, 1])
        # reference subset (template) fully inside ref
        if ix - half < 0 or iy - half < 0 or ix + half >= W or iy + half >= H:
            continue
        tmpl = R[iy - half:iy + half + 1, ix - half:ix + half + 1]
        if float(tmpl.std()) <= min_std:
            flat[k] = True
            continue
        # search window in cur, clamped to image
        x0 = max(0, ix - half - sr); x1 = min(W, ix + half + 1 + sr)
        y0 = max(0, iy - half - sr); y1 = min(H, iy + half + 1 + sr)
        win = C[y0:y1, x0:x1]
        if win.shape[0] < s or win.shape[1] < s:
            continue
        corr = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, maxloc = cv2.minMaxLoc(corr)
        cx, cy = maxloc            # top-left of best match within `corr`
        if float(win[cy:cy + s, cx:cx + s].std()) <= min_std:
            flat[k] = True
            continue
        # integer displacement = matched top-left in cur - template top-left
        dx0 = x0 + cx - (ix - half)
        dy0 = y0 + cy - (iy - half)
        edge[k] = (cx == 0 or cy == 0 or cx == corr.shape[1] - 1
                   or cy == corr.shape[0] - 1)
        zn = float(peak)
        if edge[k] or not subpixel:
            dx, dy = float(dx0), float(dy0)
        elif use_icgn:
            u, zi, ok = _icgn_translation(tmpl, coeffs, ix, iy, (dx0, dy0))
            if ok:
                dx, dy = float(u[0]), float(u[1])
                zn = zi
            else:
                failed[k] = True
                dx, dy = float(dx0), float(dy0)
        else:
            fn_ = _subpixel_gauss if subpixel_method == "gauss" else _subpixel_parabola
            sy, sx = fn_(corr, cy, cx)
            dx, dy = dx0 + sx, dy0 + sy
        disp[k] = (dx, dy)
        score[k] = zn
        valid[k] = (zn >= zncc_min) and not edge[k] and not failed[k]
    if return_info:
        return disp, valid, score, {"edge": edge, "flat": flat,
                                    "icgn_failed": failed}
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
         'mean_zncc': float | None, 'n_edge': e, 'n_flat': f,
         'elapsed_s': float, 'frame_s': float}

    where ``n_edge`` counts the (unmasked) points whose correlation peak hit
    the border of the search range (displacement possibly >= ``search``:
    increase it) and ``n_flat`` the textureless ones.

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
    # Export the centres actually correlated (integer pixel centres).
    pts = pixel_centres(points)
    n_pts = len(pts)
    n_pairs = max(0, n_img - 1)
    dt = 1.0 / fps if fps else 1.0

    xy = np.array([pixel_to_model(p[0], p[1], img_w, img_h, mm_per_px)
                   for p in pts]) if n_pts else np.empty((0, 2))
    x_mm = xy[:, 0] if n_pts else np.empty(0)
    y_mm = xy[:, 1] if n_pts else np.empty(0)
    grid = grid_from_points(x_mm, y_mm)
    # Finite-difference strain rates need >= 2 points along both axes (a
    # one-row/one-column grid made np.gradient raise); otherwise they stay NaN.
    strain_grid = (grid is not None and grid[0].size >= 2
                   and grid[1].size >= 2)

    names = ["Ux", "Uy", "Umag", "Vx", "Vy", "Vmag",
             "Exx_dot", "Eyy_dot", "Exy_dot", "Eeq_dot", "ZNCC"]
    fields = {k: np.full((n_pairs, n_pts), np.nan) for k in names}
    valid = np.zeros((n_pairs, n_pts), bool)
    t = np.zeros(n_pairs)

    keep_static = (np.ones(n_pts, bool) if point_keep is None
                   else np.asarray(point_keep, bool))
    mp = mask_params or {}
    # The mask window is the subset itself (the pixels the correlation uses).
    mask_win = int(mp.get("win", int(params.subset)))
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
        disp, ok, score, cinfo = correlate_local(
            frames[i], frames[i + 1], pts,
            subset=params.subset, search=params.search,
            zncc_min=params.zncc_min, subpixel=params.subpixel,
            subpixel_method=getattr(params, "subpixel_method", "icgn"),
            min_std_rel=getattr(params, "min_std_rel", 1e-3),
            return_info=True)
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

        if strain_grid:
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
                      "n_edge": int((cinfo["edge"] & keep).sum()),
                      "n_flat": int((cinfo["flat"] & keep).sum()),
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
    the dark scene background / out-of-material regions). ``win`` is the FULL
    window side (pass the subset size so the test covers the pixels that the
    correlation uses). Returns a (n_points,) bool array.

    The texture (local std) criterion was removed: masking is intensity-only,
    consistent across the local and global engines.
    """
    g = _to_gray_f32(image)
    H, W = g.shape
    half = max(1, int(win) // 2)
    pts = pixel_centres(points)
    keep = np.ones(len(pts), bool)
    for k in range(len(pts)):
        ix, iy = int(pts[k, 0]), int(pts[k, 1])
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
    pts = pixel_centres(points)
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
            zncc_min=params.zncc_min, subpixel=params.subpixel,
            subpixel_method=getattr(params, "subpixel_method", "icgn"),
            min_std_rel=getattr(params, "min_std_rel", 1e-3))
        vx = disp[:, 0] * mm_per_px / dt            # +x is +x in both frames
        vy = -disp[:, 1] * mm_per_px / dt           # image y down -> model y up
        V1[i] = vx
        V2[i] = vy
        Vmag[i] = np.hypot(vx, vy)
        valid[i] = ok
        t[i] = trigger_offset_s + (i + 0.5) * dt    # midpoint of the pair

    return {"x": x_mm, "y": y_mm, "t": t,
            "V1": V1, "V2": V2, "Vmag": Vmag, "valid": valid}
