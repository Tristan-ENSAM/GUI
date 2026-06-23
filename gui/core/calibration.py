# -*- coding: utf-8 -*-
"""
Visible-camera calibration (full Zhang) behind a pluggable backend.

The DIC side of the project needs two engines (a local subset engine and the
global q4dic engine); calibration follows the same philosophy: a backend
interface so the OpenCV implementation here can later be swapped for / joined
by a q4dic-based one.

A `TargetSpec` describes the user-selectable calibration target (checkerboard
or circle grid, its dimensions and physical spacing). `OpenCVCalibrator`
runs the full Zhang procedure over several poses (cv2.calibrateCamera) to get
the camera matrix + distortion, then computes the pixel->mm homography and
the mm/px scale on the chosen reference view (the cutting plane). Results are
returned as a `CalibrationResult` and stored in
`ExperimentSession.visible_calibration` (see gui/results/FORMAT.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


# ---------------------------------------------------------------------------
# Target description (user-selectable)
# ---------------------------------------------------------------------------
@dataclass
class TargetSpec:
    """A calibration target.

    pattern:
      - 'checkerboard'  : `cols` x `rows` INNER corners, `spacing_mm` = square side
      - 'circles'       : symmetric circle grid, `cols` x `rows` circles
      - 'circles_asym'  : asymmetric circle grid (OpenCV convention)
    spacing_mm: physical distance between adjacent features (mm)."""
    pattern:    str   = "checkerboard"
    cols:       int   = 9
    rows:       int   = 6
    spacing_mm: float = 1.0

    def object_points(self) -> np.ndarray:
        """(N, 3) world coordinates (z = 0) of the target features, in mm,
        in detection order. Asymmetric circle grids use the staggered layout
        expected by cv2.findCirclesGrid."""
        if self.pattern == "circles_asym":
            pts = []
            for r in range(self.rows):
                for c in range(self.cols):
                    x = (2 * c + (r % 2)) * self.spacing_mm
                    y = r * self.spacing_mm
                    pts.append((x, y, 0.0))
            return np.asarray(pts, np.float32)
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        grid = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        objp[:, :2] = grid * self.spacing_mm
        return objp

    def to_dict(self) -> dict:
        return {"pattern": self.pattern, "cols": self.cols,
                "rows": self.rows, "spacing_mm": self.spacing_mm}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TargetSpec":
        d = d or {}
        return cls(pattern=d.get("pattern", "checkerboard"),
                   cols=int(d.get("cols", 9)), rows=int(d.get("rows", 6)),
                   spacing_mm=float(d.get("spacing_mm", 1.0)))


@dataclass
class CalibrationResult:
    camera_matrix:    list           # 3x3
    dist_coeffs:      list           # [k1, k2, p1, p2, k3]
    scale_mm_per_px:  float
    homography:       list           # 3x3, maps pixels -> mm in the ref plane
    reproj_rms_px:    float
    image_size:       list           # [width, height]
    reference_index:  int
    n_views_used:     int
    target:           dict
    images:           list = field(default_factory=list)
    units:            str = "mm"

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "scale_mm_per_px": self.scale_mm_per_px,
            "homography": self.homography,
            "reproj_rms_px": self.reproj_rms_px,
            "image_size": self.image_size,
            "reference_index": self.reference_index,
            "n_views_used": self.n_views_used,
            "target": self.target,
            "images": self.images,
            "units": self.units,
        }


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class VisibleCalibrationBackend:
    """Interface a calibration engine must implement. Concrete engines:
    OpenCVCalibrator (here); a q4dic-based one can be added later."""
    name = "base"

    def detect(self, image: np.ndarray, target: TargetSpec):
        """Return (N, 2) sub-pixel image points in detection order, or None
        if the target was not found."""
        raise NotImplementedError

    def calibrate(self, images: List[np.ndarray], target: TargetSpec,
                  reference_index: int = 0) -> CalibrationResult:
        raise NotImplementedError


def _to_gray(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    a = a.astype(np.float64)
    mx = a.max() if a.size else 1.0
    if mx <= 1.0 + 1e-9:        # float image in [0,1]
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


class OpenCVCalibrator(VisibleCalibrationBackend):
    name = "opencv"

    def __init__(self):
        try:
            import cv2  # noqa: F401
        except Exception as e:                       # pragma: no cover
            raise RuntimeError(
                "OpenCV is required for visible calibration. Install it with "
                "`pip install opencv-python`. (%s)" % e)

    # -- detection --------------------------------------------------------
    def detect(self, image, target):
        import cv2
        gray = _to_gray(image)
        size = (target.cols, target.rows)
        if target.pattern == "checkerboard":
            ok, corners = cv2.findChessboardCorners(
                gray, size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok:
                return None
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            return corners.reshape(-1, 2)
        flag = (cv2.CALIB_CB_ASYMMETRIC_GRID if target.pattern == "circles_asym"
                else cv2.CALIB_CB_SYMMETRIC_GRID)
        ok, centers = cv2.findCirclesGrid(gray, size, flags=flag)
        if not ok:
            return None
        return centers.reshape(-1, 2)

    # -- calibration ------------------------------------------------------
    def calibrate(self, images, target, reference_index=0):
        import cv2
        objp = target.object_points()
        obj_points, img_points, used = [], [], []
        image_size = None
        for i, im in enumerate(images):
            gray = _to_gray(im)
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])   # (w, h)
            pts = self.detect(im, target)
            if pts is None:
                continue
            obj_points.append(objp)
            img_points.append(pts.astype(np.float32).reshape(-1, 1, 2))
            used.append(i)
        if len(used) < 3:
            raise RuntimeError(
                "Full Zhang calibration needs the target detected in at least "
                "3 views (got %d). Add more / clearer target images." % len(used))

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None)

        # Reference view = the cutting plane. Map it to the position in `used`.
        if reference_index in used:
            ref_pos = used.index(reference_index)
        else:
            ref_pos = 0
            reference_index = used[0]

        # pixel -> mm homography on the (undistorted) reference view.
        ref_img_pts = img_points[ref_pos].reshape(-1, 2)
        und = cv2.undistortPoints(ref_img_pts.reshape(-1, 1, 2), K, dist,
                                  P=K).reshape(-1, 2)
        obj_xy = objp[:, :2].astype(np.float32)
        Hpx2mm, _ = cv2.findHomography(und, obj_xy)
        scale = _scale_mm_per_px(und, obj_xy)

        return CalibrationResult(
            camera_matrix=np.asarray(K).tolist(),
            dist_coeffs=np.asarray(dist).reshape(-1).tolist(),
            scale_mm_per_px=float(scale),
            homography=np.asarray(Hpx2mm).tolist(),
            reproj_rms_px=float(rms),
            image_size=list(image_size),
            reference_index=int(reference_index),
            n_views_used=len(used),
            target=target.to_dict(),
        )


def _scale_mm_per_px(img_pts_undist: np.ndarray, obj_xy_mm: np.ndarray) -> float:
    """Median mm-per-pixel estimated from consecutive feature pairs
    (robust to a few outliers): median over i of |Δmm_i| / |Δpx_i|."""
    dpx = np.linalg.norm(np.diff(img_pts_undist, axis=0), axis=1)
    dmm = np.linalg.norm(np.diff(obj_xy_mm, axis=0), axis=1)
    m = dpx > 1e-9
    if not np.any(m):
        return float("nan")
    return float(np.median(dmm[m] / dpx[m]))
