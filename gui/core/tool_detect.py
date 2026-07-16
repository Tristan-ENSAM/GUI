# -*- coding: utf-8 -*-
"""
Automatic tool detection for the Alignment tab.

Pipeline (all pure, OpenCV-based, testable):
  1. optional crop to a manual search ROI (reduces the search zone),
  2. threshold the (gray) image to boost contrast for contour finding,
  3. find the largest external contour,
  4. fit a quadrilateral to it (approxPolyDP -> 4 vertices, else minAreaRect).

The four quad corners are returned in FULL-image pixel coordinates, ordered.
The UI then lets the user (a) override corners manually and (b) pick which
quad edge is the rake face and which is the flank face; the intersection of
those two segments is the tool tip (`segment_intersection`).
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from gui.core.calibration import _to_gray

Point = Tuple[float, float]


def threshold_image(gray: np.ndarray, threshold: Optional[float] = None,
                    invert: bool = False) -> np.ndarray:
    """Binarise a gray image (uint8 0/255). `threshold=None` uses Otsu's
    automatic threshold. `invert` swaps foreground/background (use it when the
    tool is darker / lighter than the background)."""
    import cv2
    g = _to_gray(gray)
    if threshold is None:
        _, bw = cv2.threshold(g, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(g, float(threshold), 255, cv2.THRESH_BINARY)
    if invert:
        bw = 255 - bw
    return bw


def _order_quad(quad: np.ndarray) -> np.ndarray:
    """Order 4 corners counter-clockwise starting from the one closest to the
    top-left, for a stable, reproducible corner order."""
    q = np.asarray(quad, float).reshape(-1, 2)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    return q[np.argsort(ang)]


def detect_tool_quad(image: np.ndarray, roi: Optional[Tuple] = None,
                     threshold: Optional[float] = None,
                     invert: bool = False) -> Optional[np.ndarray]:
    """Detect the tool silhouette as a quadrilateral.

    `roi` = (x, y, w, h) in pixels restricts the search (recommended).
    Returns a (4, 2) array of corner pixels in FULL-image coordinates,
    ordered, or None if nothing was found."""
    import cv2
    gray = _to_gray(image)
    ox, oy = 0, 0
    if roi is not None:
        ox, oy, rw, rh = [int(round(v)) for v in roi]
        ox = max(0, ox); oy = max(0, oy)
        sub = gray[oy:oy + rh, ox:ox + rw]
    else:
        sub = gray
    if sub.size == 0:
        return None
    bw = threshold_image(sub, threshold, invert)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    quad = None
    for f in np.linspace(0.01, 0.12, 12):
        approx = cv2.approxPolyDP(c, f * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(float)
            break
    if quad is None:                       # fall back to a rotated rectangle
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(float)
    quad += np.array([ox, oy], float)      # back to full-image coordinates
    return _order_quad(quad)


def segment_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """Intersection of the infinite lines (p1,p2) and (p3,p4). Returns None if
    the segments are (near) parallel."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def nearest_corner(quad: np.ndarray, px: float, py: float) -> int:
    """Index of the quad corner nearest to (px, py)."""
    q = np.asarray(quad, float).reshape(-1, 2)
    d = np.hypot(q[:, 0] - px, q[:, 1] - py)
    return int(np.argmin(d))


def nearest_edge(quad: np.ndarray, px: float, py: float) -> int:
    """Index i of the quad edge (corner i -> corner i+1) nearest to (px, py)."""
    q = np.asarray(quad, float).reshape(-1, 2)
    best, best_d = 0, float("inf")
    for i in range(4):
        a = q[i]; b = q[(i + 1) % 4]
        ab = b - a
        t = np.clip(np.dot(np.array([px, py]) - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        proj = a + t * ab
        d = np.hypot(px - proj[0], py - proj[1])
        if d < best_d:
            best_d, best = d, i
    return best


# ---------------------------------------------------------------------------
# Semi-automatic edge fitting (user gives a rough line, we snap it to the edge)
# ---------------------------------------------------------------------------
def gradient_magnitude(image: np.ndarray, blur: float = 2.0) -> np.ndarray:
    """Sobel gradient magnitude of the (optionally smoothed) gray image."""
    import cv2
    g = _to_gray(image).astype(np.float32)
    if blur and blur > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(blur))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    return cv2.magnitude(gx, gy)


def _bilinear(im: np.ndarray, x: float, y: float) -> float:
    h, w = im.shape
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    return float((im[y0, x0] * (1 - fx) + im[y0, x0 + 1] * fx) * (1 - fy)
                 + (im[y0 + 1, x0] * (1 - fx) + im[y0 + 1, x0 + 1] * fx) * fy)


def snap_line_to_edge(image: np.ndarray, p1: Point, p2: Point,
                      search: int = 25, n: int = 40, blur: float = 2.0,
                      binary_threshold: Optional[float] = None):
    """Refine a rough line p1->p2 by snapping it onto the strongest image
    edge: at `n` samples along the line, search +/- `search` px perpendicular
    for the maximum gradient, then robustly fit a line to those points.

    When ``binary_threshold`` is given, the gradient is computed on the
    BINARISED image (``gray >= threshold``) rather than the smoothed grey
    image, so the fit locks onto the sharp tool/background boundary instead of
    internal speckle texture.

    Returns (q1, q2, snapped_pts): the refined endpoints (p1, p2 projected on
    the fitted line) and the (m, 2) snapped edge points. Falls back to the
    input line if too few points are found. Averages out serration because a
    whole line is fitted."""
    import cv2
    if binary_threshold is not None:
        g = _to_gray(image).astype(np.float64)
        # Smooth BEFORE thresholding, then take the gradient of the binary mask.
        if blur > 0:
            import scipy.ndimage as _ndi
            g = _ndi.gaussian_filter(g, blur)
        binimg = (g >= float(binary_threshold)).astype(np.float64) * 255.0
        gmag = gradient_magnitude(binimg, 0.0)
    else:
        gmag = gradient_magnitude(image, blur)
    a = np.array(p1, float); b = np.array(p2, float)
    d = b - a
    L = float(np.hypot(*d))
    if L < 1e-6:
        return tuple(a), tuple(b), np.empty((0, 2))
    d /= L
    nrm = np.array([-d[1], d[0]])
    ts = np.linspace(-search, search, int(2 * search) + 1)
    pts = []
    for i in range(n):
        c = a + d * (L * i / (n - 1))
        # Sample the gradient along the perpendicular search line.
        vals = np.array([_bilinear(gmag, *(c + nrm * t)) for t in ts])
        vmax = float(vals.max())
        if vmax <= 0:
            continue
        # Use the gradient-weighted centroid of the strong-response region so
        # the snapped point sits ON the edge (the centre of the gradient band),
        # not at the first maximum nor at the border of the search window.
        thr = 0.5 * vmax
        w = np.where(vals >= thr, vals, 0.0)
        if w.sum() <= 0:
            continue
        t_star = float((ts * w).sum() / w.sum())
        x, y = c + nrm * t_star
        pts.append((x, y))
    pts = np.asarray(pts, np.float32)
    if len(pts) < 2:
        return tuple(a), tuple(b), pts
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()

    def proj(p):
        ap = np.array(p, float) - np.array([x0, y0])
        t = ap[0] * vx + ap[1] * vy
        return (float(x0 + t * vx), float(y0 + t * vy))

    return proj(a), proj(b), pts
