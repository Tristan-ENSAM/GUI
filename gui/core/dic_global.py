# -*- coding: utf-8 -*-
"""
Global Q4 Digital Image Correlation (Correli-Q4 style) engine.

This is the *global* counterpart of the local subset engine in ``gui.core.dic``.
Instead of correlating independent subsets, a single Q4 finite-element mesh is
laid over the ROI; the unknowns are the nodal displacements, and the whole
grey-level residual ``f(x) - g(x + u(x))`` is minimised at once by a
Gauss-Newton / modified Newton-Raphson iteration (Besnard, Hild & Roux, 2006,
Experimental Mechanics 46(6):789-803).

This module is the pure (no Qt) numerical core: mesh, Q4 bilinear shape
functions, bicubic image interpolation, element/global assembly, the iterative
solver (two variants), and the strain post-processing. The orchestration that
turns a frame *sequence* into the field arrays consumed by the existing viewer
(``gui.widgets.dic_field_viewer`` via ``gui.core.exp_field_io``) lives in
``compute_dic_global_fields`` at the bottom and mirrors the public shape of
``gui.core.dic.compute_dic_fields``.

Algorithmic references (verified against the user's personal ``q4dic``
re-implementation, modules ``mesh.py``/``solver.py``/``postprocessing.py``,
whose low-level routines are covered by passing unit tests):
  - Q4 shape functions, structured mesh, pixel-wise quadrature: q4dic/mesh.py
  - 'standard' and 'hild' Newton-Raphson variants, analytic covariance
    ``Cov = 2 sigma_f^2 [H]^-1``: q4dic/solver.py
  - strain = gradient of the shape functions at Gauss points: q4dic/postprocessing.py

Conventions reused from the GUI_Abaqus side (NOT from q4dic):
  - The solver works internally in *image pixel* coordinates (origin at the
    top-left, x to the right along columns, y downward along rows) exactly like
    the local engine and ``cv2``. The conversion to the *model* frame (origin at
    the image centre, y up) is applied only on output, through
    ``gui.core.alignment.pixel_to_model`` -- identical to ``dic.compute_dic_fields``.
  - The shear strain is the *tensorial* component ``eps_xy = 0.5 (du_x/dy +
    du_y/dx)`` and the von Mises equivalent uses the plane incompressibility
    closure ``e_zz = -(e_xx + e_yy)`` -- both identical to ``dic._equiv`` so the
    local and global engines feed the viewer with the same definitions.

NOTE (scope of this batch): the mesh is built on the full ROI bounding box. The
material/tool mask and the per-element validity/convection handling (q4dic
``segmentation.py``) are deferred to a later batch, as agreed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Sequence
import time
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from gui.core.alignment import pixel_to_model


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class DicGlobalParams:
    """Global Q4-DIC settings.

    Attributes
    ----------
    elem_size : int
        Target Q4 element side in pixels. The number of elements is
        ``floor(roi_side / elem_size)`` in each direction and the ROI is
        trimmed so the mesh fits an integer number of elements (q4dic
        convention).
    variant : str
        Newton-Raphson assembly variant:
          - ``'standard'``: the grey-level gradient is taken on the deformed
            image ``g`` at ``x + u^k``; ``[H]`` is reassembled every iteration.
          - ``'hild'``: the gradient is taken once on the reference image ``f``;
            ``[H]`` is assembled a single time and only the right-hand side is
            updated. Cheaper for long sequences (Hild & Roux).
        Both variants converge to the same displacement (verified); ``'hild'``
        is an approximation that trades a fixed Hessian for speed.
    max_iter : int
        Maximum Newton-Raphson iterations per image pair.
    tol : float
        Convergence threshold on the nodal correction norm ``||dU||`` (pixels).
    incremental : bool
        Sequence correlation pattern (see ``compute_dic_global_fields``):
          - ``True``  (Eulerian incremental): reference = frame i, deformed =
            frame i+1; each pair measures the increment i -> i+1. Mirrors the
            local engine's instantaneous-velocity convention.
          - ``False`` (Lagrangian total): reference = frame 0 fixed, deformed =
            frame i+1; each pair measures the total displacement since frame 0.
    u_init_previous : bool
        If ``True`` the previous pair's solution initialises the next one
        (faster when the motion varies slowly). If ``False`` every pair starts
        from zero (more robust, slower). Independent of ``incremental``.
    reg_rel : float
        Relative Tikhonov-like floor added to ``[H]`` before solving, scaled by
        ``trace([H]) / n_dof``. Purely numerical conditioning (NOT a mechanical
        regularisation); q4dic uses 1e-6.
    pyramid_levels : int
        Number of Gaussian-pyramid levels for the multi-scale solve (Besnard,
        Hild & Roux). ``1`` means single-scale (the native resolution only, the
        previous behaviour). With ``L`` levels the correlation is first solved
        on the coarsest image (downsampled by 2**(L-1)), then the displacement
        is rescaled and refined level by level down to the native resolution.
        This widens the displacement-capture range and speeds convergence for
        large inter-frame motion. A single setting applies to every image pair.
    pyramid_sigma : float
        Standard deviation of the Gaussian anti-aliasing filter applied before
        each 2x downsampling (q4dic default 1.0). Ignored when
        ``pyramid_levels == 1``.
    mask_enabled : bool
        When True, Q4 elements whose material coverage is below
        ``coverage_threshold`` are excluded from the solve (material mask). The
        material mask is the intensity threshold ``gray >= mask_min_intensity``,
        recomputed on each pair's reference frame. Nodes touching only excluded
        elements are constrained (their displacement is NaN on output).
    mask_min_intensity : float
        Grey-level threshold defining the material mask (intensity-only).
    coverage_threshold : float
        Minimum fraction of material pixels in an element's bounding box for
        the element to be kept (0..1, default 0.5).
    convect : bool
        When True the mesh is convected (Lagrangian): each pair solves on the
        mesh displaced by the cumulated displacement of the previous pairs, so
        the nodes follow the material. Elements that fold over (det(J) <= 0
        after convection) are excluded like out-of-material elements. The output
        fields are still reported at the reference (frame-0) node positions.
    tool_polygon : list of (x, y), optional
        Pixel vertices of a closed polygon covering the tool. Pixels inside it
        are treated as non-material (excluded), in addition to the intensity
        threshold. Enables element exclusion even when ``mask_enabled`` is off.
    """
    engine: str = "global"
    elem_size: int = 24
    variant: str = "standard"      # 'standard' | 'hild'
    max_iter: int = 30
    tol: float = 1e-4
    incremental: bool = True
    u_init_previous: bool = False
    reg_rel: float = 1e-6
    pyramid_levels: int = 1        # 1 = single-scale (unchanged behaviour)
    pyramid_sigma: float = 1.0
    mask_enabled: bool = False
    mask_min_intensity: float = 0.0
    coverage_threshold: float = 0.5
    convect: bool = False
    tool_polygon: Optional[list] = None

    def to_json_dict(self) -> dict:
        return asdict(self)


def _to_gray_f64(img: np.ndarray) -> np.ndarray:
    """Image to float64 grayscale (mean of RGB if needed). Matches the local
    engine's ``_to_gray_f32`` but keeps float64 for the spline/solver."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a.astype(np.float64)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation (consistent with the ZNCC criterion;
    q4dic/preprocessing.normalize_image). Falls back to mean subtraction only
    when the image is flat."""
    g = _to_gray_f64(img)
    mu = float(g.mean())
    sd = float(g.std())
    if sd < 1e-12:
        return g - mu
    return (g - mu) / sd


# =============================================================================
# Q4 bilinear shape functions  (ported from q4dic/mesh.py)
# =============================================================================

def shape_functions(xi: float, eta: float) -> np.ndarray:
    """Bilinear Q4 shape functions at natural coordinates (xi, eta) in
    [-1, 1]^2. Local node numbering (q4dic convention)::

        4 --- 3
        |     |
        1 --- 2

    Returns a (4,) array [N1, N2, N3, N4]."""
    return np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    ])


def shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
    """Derivatives of the Q4 shape functions w.r.t. natural coordinates.

    Returns a (2, 4) array with row 0 = dN/dxi, row 1 = dN/deta."""
    return np.array([
        [-0.25 * (1 - eta), 0.25 * (1 - eta),
         0.25 * (1 + eta), -0.25 * (1 + eta)],
        [-0.25 * (1 - xi), -0.25 * (1 + xi),
         0.25 * (1 + xi), 0.25 * (1 - xi)],
    ])


def shape_functions_grid(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Vectorised shape functions over many points. ``xi``/``eta`` are (n,)
    arrays; returns a (4, n) array (row i = N_{i+1} at every point)."""
    return np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    ])


def gauss_points_2d(n_gauss: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre points/weights on [-1, 1]^2. Returns (pts, weights) with
    pts of shape (n_gauss^2, 2) and weights of shape (n_gauss^2,)."""
    pts_1d, w_1d = np.polynomial.legendre.leggauss(n_gauss)
    xi, eta = np.meshgrid(pts_1d, pts_1d)
    w_xi, w_eta = np.meshgrid(w_1d, w_1d)
    return (np.column_stack([xi.ravel(), eta.ravel()]),
            (w_xi * w_eta).ravel())


# =============================================================================
# Structured Q4 mesh  (ported from q4dic/mesh.py:Q4Mesh)
# =============================================================================

class Q4Mesh:
    """Structured Q4 mesh on an axis-aligned rectangle in pixel coordinates.

    Nodes are numbered row by row, left to right, bottom to top (in pixel
    space, "bottom" = smaller y). Each node carries 2 DOF (ux, uy); element
    DOF order is [ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4].

    Parameters
    ----------
    x0, y0, x1, y1 : float
        ROI corners in pixels (x0 < x1, y0 < y1).
    n_elem_x, n_elem_y : int
        Number of elements along x and y.
    """

    def __init__(self, x0: float, y0: float, x1: float, y1: float,
                 n_elem_x: int, n_elem_y: int):
        if n_elem_x < 1 or n_elem_y < 1:
            raise ValueError("n_elem_x and n_elem_y must be >= 1")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("need x1 > x0 and y1 > y0")
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.n_elem_x = int(n_elem_x)
        self.n_elem_y = int(n_elem_y)
        self.elem_size_x = (x1 - x0) / n_elem_x
        self.elem_size_y = (y1 - y0) / n_elem_y
        self.nodes = self._generate_nodes()
        self.connectivity = self._generate_connectivity()
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.connectivity.shape[0]
        self.n_dof = 2 * self.n_nodes

    def _generate_nodes(self) -> np.ndarray:
        nx = self.n_elem_x + 1
        ny = self.n_elem_y + 1
        xs = np.linspace(self.x0, self.x1, nx)
        ys = np.linspace(self.y0, self.y1, ny)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _generate_connectivity(self) -> np.ndarray:
        nx = self.n_elem_x + 1
        conn = []
        for ie in range(self.n_elem_y):
            for je in range(self.n_elem_x):
                n1 = ie * nx + je
                n2 = ie * nx + je + 1
                n3 = (ie + 1) * nx + je + 1
                n4 = (ie + 1) * nx + je
                conn.append([n1, n2, n3, n4])
        return np.array(conn, dtype=int)

    def dof_indices(self, elem_idx: int) -> np.ndarray:
        """Global DOF indices (8,) for an element: node k -> ux=2k, uy=2k+1."""
        nodes = self.connectivity[elem_idx]
        dofs = np.empty(8, dtype=int)
        for i, n in enumerate(nodes):
            dofs[2 * i] = 2 * n
            dofs[2 * i + 1] = 2 * n + 1
        return dofs

    def physical_to_natural(self, x: np.ndarray, y: np.ndarray,
                            elem_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Map physical (x, y) to natural (xi, eta) for an axis-aligned element.
        Uses the element's min corner so xi, eta span [-1, 1] across it."""
        node_coords = self.nodes[self.connectivity[elem_idx]]
        x_min = node_coords[:, 0].min()
        y_min = node_coords[:, 1].min()
        xi = 2.0 * (x - x_min) / self.elem_size_x - 1.0
        eta = 2.0 * (y - y_min) / self.elem_size_y - 1.0
        return xi, eta

    def get_pixel_points_in_element(self, elem_idx: int
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integer pixel coordinates contained in an element. Pixel-wise
        quadrature (q4dic/Besnard) is more faithful to the discrete image than
        Gauss quadrature for the grey-level residual."""
        node_coords = self.nodes[self.connectivity[elem_idx]]
        x_min, x_max = node_coords[:, 0].min(), node_coords[:, 0].max()
        y_min, y_max = node_coords[:, 1].min(), node_coords[:, 1].max()
        x_px = np.arange(np.ceil(x_min), np.floor(x_max) + 1, dtype=float)
        y_px = np.arange(np.ceil(y_min), np.floor(y_max) + 1, dtype=float)
        xx, yy = np.meshgrid(x_px, y_px)
        return xx.ravel(), yy.ravel()

    def build_shape_matrix_at_pixels(self, elem_idx: int
                                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shape matrix [N] (2, 8, n_pixels) at all pixels of an element, with
        the pixel coordinates. Row 0 maps nodal DOF to ux, row 1 to uy."""
        x_pts, y_pts = self.get_pixel_points_in_element(elem_idx)
        xi, eta = self.physical_to_natural(x_pts, y_pts, elem_idx)
        N = shape_functions_grid(xi, eta)             # (4, n_pts)
        n_pts = x_pts.size
        N_mat = np.zeros((2, 8, n_pts))
        for i in range(4):
            N_mat[0, 2 * i, :] = N[i]
            N_mat[1, 2 * i + 1, :] = N[i]
        return N_mat, x_pts, y_pts


def build_mesh_on_roi(roi: Tuple[float, float, float, float],
                      elem_size: int) -> Q4Mesh:
    """Build a Q4 mesh over an ROI = (x, y, w, h) in pixels, trimming the ROI
    so it holds an integer number of ``elem_size`` elements (q4dic convention,
    pipeline.run_dic). Raises ValueError if the ROI is smaller than one element."""
    x, y, w, h = roi
    n_elem_x = int(w // elem_size)
    n_elem_y = int(h // elem_size)
    if n_elem_x < 1 or n_elem_y < 1:
        raise ValueError(
            "ROI too small for elem_size=%d (need at least one element per "
            "direction; ROI is %.0fx%.0f px)" % (elem_size, w, h))
    x1 = x + n_elem_x * elem_size
    y1 = y + n_elem_y * elem_size
    return Q4Mesh(x0=float(x), y0=float(y), x1=float(x1), y1=float(y1),
                  n_elem_x=n_elem_x, n_elem_y=n_elem_y)


def material_mask_intensity(image: np.ndarray, min_intensity: float
                            ) -> np.ndarray:
    """Binary material mask from a simple grey-level threshold:
    ``mask = gray(image) >= min_intensity``. Drops the dark background / tool.
    Returns a (H, W) bool array. Intensity-only (no texture criterion), to
    match the local engine's masking."""
    g = _to_gray_f64(image)
    return g >= float(min_intensity)


def polygon_mask(shape: Tuple[int, int], polygon: Sequence
                 ) -> np.ndarray:
    """Rasterise a closed polygon to a boolean mask of the given (H, W) shape.

    ``polygon`` is a sequence of (x, y) vertices in pixel coordinates (x =
    column, y = row). Pixels strictly inside (or on the boundary of) the polygon
    are True. Returns an all-False mask if the polygon has fewer than 3 vertices.
    Uses matplotlib's even-odd point-in-polygon test.
    """
    H, W = shape
    poly = np.asarray(polygon, float).reshape(-1, 2)
    if poly.shape[0] < 3:
        return np.zeros((H, W), bool)
    from matplotlib.path import Path
    path = Path(poly)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = path.contains_points(pts, radius=0.5)
    return inside.reshape(H, W)


def material_mask(image: np.ndarray, min_intensity: float,
                  tool_polygon: Optional[Sequence] = None) -> np.ndarray:
    """Combined material mask: bright-enough pixels that are NOT inside the tool
    polygon. ``material = (gray >= min_intensity) AND NOT inside(tool_polygon)``.
    The tool polygon (pixel vertices) is optional; when omitted only the
    intensity criterion applies."""
    mat = material_mask_intensity(image, min_intensity)
    if tool_polygon is not None and len(tool_polygon) >= 3:
        tool = polygon_mask(mat.shape, tool_polygon)
        mat = mat & ~tool
    return mat


def element_coverage_mask(mesh: Q4Mesh, material_mask: np.ndarray,
                          coverage_threshold: float = 0.5) -> np.ndarray:
    """Per-element validity from material coverage (Besnard/Hild segmentation).

    For each element, the fraction of material pixels inside its bounding box
    is compared to ``coverage_threshold``; the element is kept when
    ``frac >= coverage_threshold``. Returns a (n_elements,) bool array.
    """
    H, W = material_mask.shape
    valid = np.zeros(mesh.n_elements, dtype=bool)
    for e in range(mesh.n_elements):
        coords = mesh.nodes[mesh.connectivity[e]]
        x0 = int(np.clip(np.floor(coords[:, 0].min()), 0, W - 1))
        x1 = int(np.clip(np.ceil(coords[:, 0].max()), 0, W - 1))
        y0 = int(np.clip(np.floor(coords[:, 1].min()), 0, H - 1))
        y1 = int(np.clip(np.ceil(coords[:, 1].max()), 0, H - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        frac = material_mask[y0:y1 + 1, x0:x1 + 1].mean()
        valid[e] = frac >= coverage_threshold
    return valid


def active_nodes_from_elements(mesh: Q4Mesh, valid_elements: np.ndarray
                               ) -> np.ndarray:
    """Boolean (n_nodes,) array of nodes belonging to at least one valid
    element. Nodes that touch only excluded elements are 'orphans' (False);
    their DOF are unconstrained by any image data."""
    active = np.zeros(mesh.n_nodes, dtype=bool)
    for e in range(mesh.n_elements):
        if valid_elements[e]:
            active[mesh.connectivity[e]] = True
    return active


def convect_mesh(mesh: Q4Mesh, U: np.ndarray) -> Q4Mesh:
    """Return a copy of ``mesh`` with its nodes displaced by ``U`` (pixels).

    The topology (connectivity, element counts) is preserved; only the node
    coordinates move, so the new mesh follows the material (Lagrangian
    convection, q4dic/segmentation.convect_mesh). NaN entries in ``U`` (orphan
    nodes) leave the corresponding node in place.
    """
    new = Q4Mesh.__new__(Q4Mesh)
    new.x0, new.y0, new.x1, new.y1 = mesh.x0, mesh.y0, mesh.x1, mesh.y1
    new.n_elem_x, new.n_elem_y = mesh.n_elem_x, mesh.n_elem_y
    new.elem_size_x, new.elem_size_y = mesh.elem_size_x, mesh.elem_size_y
    new.connectivity = mesh.connectivity
    new.n_nodes, new.n_elements, new.n_dof = (
        mesh.n_nodes, mesh.n_elements, mesh.n_dof)
    dx = np.nan_to_num(U[0::2], nan=0.0)
    dy = np.nan_to_num(U[1::2], nan=0.0)
    new.nodes = mesh.nodes.copy()
    new.nodes[:, 0] += dx
    new.nodes[:, 1] += dy
    return new


def check_jacobian(mesh: Q4Mesh) -> np.ndarray:
    """Per-element validity from the sign of the Jacobian determinant at the
    element centre (xi=eta=0). For a well-formed Q4 element det(J) > 0; a
    convected mesh whose element folds over gives det(J) <= 0 (q4dic
    segmentation.check_jacobian). Returns a (n_elements,) bool array."""
    dN = shape_function_derivatives(0.0, 0.0)        # (2, 4)
    valid = np.ones(mesh.n_elements, dtype=bool)
    for e in range(mesh.n_elements):
        coords = mesh.nodes[mesh.connectivity[e]]    # (4, 2)
        J = dN @ coords                              # (2, 2)
        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if det_J <= 0:
            valid[e] = False
    return valid


# =============================================================================
# Bicubic image interpolation  (wrapper over scipy, q4dic/preprocessing.py)
# =============================================================================

class BicubicInterpolator:
    """Sub-pixel grey-level interpolator built once per image, evaluated at the
    deformed positions x + u during the iteration.

    Wraps ``scipy.interpolate.RectBivariateSpline`` with kx=ky=3 over the pixel
    grid (axis 0 = rows = y, axis 1 = cols = x), as in q4dic/preprocessing.py.
    """

    def __init__(self, img: np.ndarray):
        g = _to_gray_f64(img)
        H, W = g.shape
        self.H, self.W = H, W
        self._spline = RectBivariateSpline(
            np.arange(H, dtype=np.float64),
            np.arange(W, dtype=np.float64),
            g, kx=3, ky=3)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Grey level at sub-pixel positions (x = columns, y = rows)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return self._spline.ev(y.ravel(), x.ravel()).reshape(x.shape)

    def gradient(self, x: np.ndarray, y: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Spatial gradient (dg/dx, dg/dy) at (x, y). dg/dx is the derivative
        along columns, dg/dy along rows (q4dic convention)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        dg_dy = self._spline.ev(y.ravel(), x.ravel(), dx=1, dy=0).reshape(x.shape)
        dg_dx = self._spline.ev(y.ravel(), x.ravel(), dx=0, dy=1).reshape(x.shape)
        return dg_dx, dg_dy


def estimate_sigma_f(f1: np.ndarray, f2: np.ndarray) -> float:
    """Image noise std from two static frames: sigma_f = std(f1 - f2)/sqrt(2)
    (q4dic/solver.estimate_sigma_f). Both should be normalised the same way."""
    diff = _to_gray_f64(f1) - _to_gray_f64(f2)
    return float(np.std(diff) / np.sqrt(2.0))


# =============================================================================
# Element / global assembly  (ported from q4dic/solver.py)
# =============================================================================

def assemble_element(elem_idx: int, mesh: Q4Mesh,
                     interp_g: BicubicInterpolator, U: np.ndarray,
                     f: np.ndarray,
                     interp_f: Optional[BicubicInterpolator] = None,
                     variant: str = "standard"
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Element contribution to [H] (8, 8), h (8,), and the summed squared
    residual over the element's pixels.

    ``variant='standard'`` takes the gradient on the deformed image at x+u^k
    (Hessian changes each iteration); ``variant='hild'`` takes it on the
    reference image f at x (fixed Hessian) and requires ``interp_f``.
    """
    N_mat, x_pts, y_pts = mesh.build_shape_matrix_at_pixels(elem_idx)
    dof_ids = mesh.dof_indices(elem_idx)
    U_elem = U[dof_ids]

    ux_pts = N_mat[0].T @ U_elem
    uy_pts = N_mat[1].T @ U_elem
    x_def = x_pts + ux_pts
    y_def = y_pts + uy_pts

    g_vals = interp_g.evaluate(x_def, y_def)

    H_img, W_img = f.shape
    x_int = np.clip(np.round(x_pts).astype(int), 0, W_img - 1)
    y_int = np.clip(np.round(y_pts).astype(int), 0, H_img - 1)
    f_vals = f[y_int, x_int]

    residual = f_vals - g_vals

    if variant == "hild":
        if interp_f is None:
            raise ValueError("variant='hild' requires interp_f")
        dg_dx, dg_dy = interp_f.gradient(x_pts, y_pts)
    else:
        dg_dx, dg_dy = interp_g.gradient(x_def, y_def)

    # psi_i . grad  ->  (8, n_pts)
    psi_dot_grad = (N_mat[0] * dg_dx[np.newaxis, :]
                    + N_mat[1] * dg_dy[np.newaxis, :])

    H_elem = psi_dot_grad @ psi_dot_grad.T
    h_elem = psi_dot_grad @ residual
    return H_elem, h_elem, float(np.sum(residual ** 2))


def assemble_global(mesh: Q4Mesh, interp_g: BicubicInterpolator,
                    U: np.ndarray, f: np.ndarray,
                    interp_f: Optional[BicubicInterpolator] = None,
                    variant: str = "standard",
                    valid_elements: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Assemble the global Hessian [H] (n_dof, n_dof), rhs h (n_dof,) and the
    total squared residual by summing element contributions. When
    ``valid_elements`` is given, elements flagged False are skipped (material
    coverage masking)."""
    n_dof = mesh.n_dof
    H_global = np.zeros((n_dof, n_dof))
    h_global = np.zeros(n_dof)
    total_residual = 0.0
    for e in range(mesh.n_elements):
        if valid_elements is not None and not valid_elements[e]:
            continue
        H_e, h_e, res_sq = assemble_element(
            e, mesh, interp_g, U, f, interp_f, variant)
        dof_ids = mesh.dof_indices(e)
        H_global[np.ix_(dof_ids, dof_ids)] += H_e
        h_global[dof_ids] += h_e
        total_residual += res_sq
    return H_global, h_global, total_residual


def assemble_h_only(mesh: Q4Mesh, interp_g: BicubicInterpolator,
                    U: np.ndarray, f: np.ndarray,
                    interp_f: BicubicInterpolator,
                    valid_elements: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, float]:
    """Right-hand side h (n_dof,) and total squared residual, for the 'hild'
    variant where [H] is held fixed: the gradient is on f (fixed), only the
    residual f - g(x+u^k) changes between iterations. Skips invalid elements
    when ``valid_elements`` is given."""
    n_dof = mesh.n_dof
    h_global = np.zeros(n_dof)
    total_residual = 0.0
    H_img, W_img = f.shape
    for e in range(mesh.n_elements):
        if valid_elements is not None and not valid_elements[e]:
            continue
        N_mat, x_pts, y_pts = mesh.build_shape_matrix_at_pixels(e)
        dof_ids = mesh.dof_indices(e)
        U_elem = U[dof_ids]
        ux_pts = N_mat[0].T @ U_elem
        uy_pts = N_mat[1].T @ U_elem
        x_def = x_pts + ux_pts
        y_def = y_pts + uy_pts
        g_vals = interp_g.evaluate(x_def, y_def)
        x_int = np.clip(np.round(x_pts).astype(int), 0, W_img - 1)
        y_int = np.clip(np.round(y_pts).astype(int), 0, H_img - 1)
        residual = f[y_int, x_int] - g_vals
        df_dx, df_dy = interp_f.gradient(x_pts, y_pts)
        psi_dot_gradf = (N_mat[0] * df_dx[np.newaxis, :]
                         + N_mat[1] * df_dy[np.newaxis, :])
        h_global[dof_ids] += psi_dot_gradf @ residual
        total_residual += float(np.sum(residual ** 2))
    return h_global, total_residual


def compute_uncertainty(H_global: np.ndarray, sigma_f: float, n_dof: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic displacement covariance ``Cov = 2 sigma_f^2 [H]^-1`` and per-DOF
    std ``sqrt(diag(Cov))`` (Hild & Roux). A small relative floor is added to
    [H] for conditioning. Even DOF = ux, odd DOF = uy."""
    reg = 1e-10 * np.trace(H_global) / max(n_dof, 1)
    H_reg = H_global + reg * np.eye(n_dof)
    try:
        H_inv = np.linalg.inv(H_reg)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H_reg)
    cov = 2.0 * sigma_f ** 2 * H_inv
    sigma_u = np.sqrt(np.abs(np.diag(cov)))
    return cov, sigma_u


# =============================================================================
# Newton-Raphson solver  (ported from q4dic/solver.py:newton_raphson)
# =============================================================================

def _regularise_and_constrain(H_global: np.ndarray, reg_rel: float, n_dof: int,
                              orphan_dof: Optional[np.ndarray]) -> np.ndarray:
    """Return the regularised Hessian used for the linear solve.

    Adds the relative diagonal floor ``reg_rel * trace([H])/n_dof`` for
    conditioning. For orphan DOF (nodes touching only excluded elements) the
    row/column is zeroed and a unit pivot is set, so the linear solve yields
    ``dU = 0`` there (the RHS is also zeroed by the caller). This keeps the
    system non-singular without letting orphan DOF pollute their neighbours.
    """
    reg = reg_rel * np.trace(H_global) / max(n_dof, 1)
    H_reg = H_global + reg * np.eye(n_dof)
    if orphan_dof is not None and orphan_dof.any():
        H_reg = H_reg.copy()
        H_reg[orphan_dof, :] = 0.0
        H_reg[:, orphan_dof] = 0.0
        H_reg[orphan_dof, orphan_dof] = 1.0
    return H_reg


def newton_raphson(mesh: Q4Mesh, interp_g: BicubicInterpolator, f: np.ndarray,
                   U_init: Optional[np.ndarray] = None, max_iter: int = 30,
                   tol: float = 1e-4, variant: str = "standard",
                   sigma_f: Optional[float] = None, reg_rel: float = 1e-6,
                   valid_elements: Optional[np.ndarray] = None
                   ) -> Dict[str, object]:
    """Modified Newton-Raphson / Gauss-Newton minimisation of the global
    grey-level residual.

    Parameters
    ----------
    mesh : Q4Mesh
    interp_g : BicubicInterpolator
        Interpolator of the deformed image g.
    f : np.ndarray
        Reference image (normalised, full integer grid).
    U_init : np.ndarray, optional
        Initial nodal displacement (n_dof,); zeros if None.
    max_iter, tol : int, float
        Iteration cap and threshold on ``||dU||`` (pixels).
    variant : str
        'standard' or 'hild' (see DicGlobalParams).
    sigma_f : float, optional
        If given, the analytic covariance is computed at convergence.
    reg_rel : float
        Relative conditioning floor on [H].

    Returns
    -------
    dict with keys:
        'U'            : (n_dof,) nodal displacement (pixels)
        'residuals'    : list of normalised residuals per iteration
        'corrections'  : list of ||dU|| per iteration
        'n_iter'       : iterations performed
        'converged'    : bool
        'cov'          : (n_dof, n_dof) or None
        'sigma_u'      : (n_dof,) or None
    """
    if variant not in ("standard", "hild"):
        raise ValueError("variant must be 'standard' or 'hild'")
    n_dof = mesh.n_dof
    U = (np.zeros(n_dof) if U_init is None
         else np.nan_to_num(np.asarray(U_init, float), nan=0.0).copy())

    residuals: List[float] = []
    corrections: List[float] = []
    cov = None
    sigma_u = None

    # Orphan DOF: nodes touching only excluded elements get no image data, so
    # their rows/cols in [H] are empty. We constrain them (dU forced to 0) and
    # mark their displacement NaN on output (Option A: explicit "no data").
    orphan_dof = None
    if valid_elements is not None:
        active_nodes = active_nodes_from_elements(mesh, valid_elements)
        orphan_nodes = ~active_nodes
        orphan_dof = np.zeros(n_dof, dtype=bool)
        orphan_dof[0::2] = orphan_nodes
        orphan_dof[1::2] = orphan_nodes

    interp_f = BicubicInterpolator(f) if variant == "hild" else None

    # 'hild': assemble [H] once.
    H_fixed = None
    H_fixed_reg = None
    if variant == "hild":
        H_fixed, _, _ = assemble_global(mesh, interp_g, U, f, interp_f, "hild",
                                        valid_elements=valid_elements)
        H_fixed_reg = _regularise_and_constrain(
            H_fixed, reg_rel, n_dof, orphan_dof)

    n_pixels = sum(mesh.get_pixel_points_in_element(e)[0].size
                   for e in range(mesh.n_elements)
                   if valid_elements is None or valid_elements[e])

    converged = False
    H_global = None
    for k in range(max_iter):
        if variant == "hild":
            h_global, total_res = assemble_h_only(
                mesh, interp_g, U, f, interp_f, valid_elements=valid_elements)
            H_global = H_fixed
            H_reg = H_fixed_reg
        else:
            H_global, h_global, total_res = assemble_global(
                mesh, interp_g, U, f, interp_f, "standard",
                valid_elements=valid_elements)
            H_reg = _regularise_and_constrain(
                H_global, reg_rel, n_dof, orphan_dof)

        if orphan_dof is not None:
            h_global = h_global.copy()
            h_global[orphan_dof] = 0.0

        residuals.append(float(np.sqrt(total_res / max(n_pixels, 1))))

        try:
            dU = np.linalg.solve(H_reg, h_global)
        except np.linalg.LinAlgError:
            break

        U = U + dU
        corr = float(np.linalg.norm(dU))
        corrections.append(corr)
        if corr < tol:
            converged = True
            break

    if sigma_f is not None:
        H_for_cov = H_fixed if variant == "hild" else H_global
        if H_for_cov is not None:
            cov, sigma_u = compute_uncertainty(H_for_cov, sigma_f, n_dof)

    if orphan_dof is not None:
        U = U.copy()
        U[orphan_dof] = np.nan          # orphan nodes carry no measurement
        if sigma_u is not None:
            sigma_u = sigma_u.copy()
            sigma_u[orphan_dof] = np.nan

    return {"U": U, "residuals": residuals, "corrections": corrections,
            "n_iter": len(corrections), "converged": converged,
            "cov": cov, "sigma_u": sigma_u}


# =============================================================================
# Multi-scale (Gaussian pyramid)  (ported from q4dic preprocessing + solver)
# =============================================================================

def build_gaussian_pyramid(img: np.ndarray, n_levels: int,
                           sigma: float = 1.0) -> List[np.ndarray]:
    """Gaussian image pyramid (Besnard, Hild & Roux multi-scale DIC).

    Level 0 is the native image; level k is downsampled by 2**k. Each level is
    Gaussian-smoothed (anti-aliasing) before 2x decimation. Returns a list with
    ``pyramid[0]`` native and ``pyramid[-1]`` coarsest.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")
    base = _to_gray_f64(img)
    pyramid = [base]
    current = base
    for _ in range(n_levels - 1):
        smoothed = gaussian_filter(current, sigma=sigma)
        downsampled = smoothed[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled
    return pyramid


def multiscale_newton_raphson(mesh: Q4Mesh, f: np.ndarray, g: np.ndarray,
                              n_levels: int, sigma: float = 1.0,
                              U_init: Optional[np.ndarray] = None,
                              max_iter: int = 30, tol: float = 1e-4,
                              variant: str = "standard",
                              sigma_f: Optional[float] = None,
                              reg_rel: float = 1e-6,
                              valid_elements: Optional[np.ndarray] = None
                              ) -> Dict[str, object]:
    """Coarse-to-fine Q4-DIC solve over a Gaussian pyramid.

    The correlation is first solved on the coarsest level (large physical
    motion becomes a small pixel motion there), then the displacement is
    rescaled and refined level by level down to the native resolution. The
    returned ``U`` is at the NATIVE scale (level 0), so it is a drop-in
    replacement for :func:`newton_raphson` for the displacement; the analytic
    uncertainty is computed only at the native level.

    Parameters
    ----------
    mesh : Q4Mesh
        Mesh defined at the NATIVE (level-0) pixel scale.
    f, g : np.ndarray
        Reference and deformed images (any scale/normalisation; both are
        normalised consistently per level here).
    n_levels : int
        Pyramid levels (``1`` falls back to a single-scale solve identical to
        :func:`newton_raphson`).
    sigma : float
        Anti-aliasing Gaussian sigma (ignored when ``n_levels == 1``).
    U_init : np.ndarray, optional
        Initial native-scale nodal displacement; zeros if None. It is divided
        down to the coarsest scale to seed the first level.

    Returns
    -------
    dict with the same keys as :func:`newton_raphson` plus ``'history'`` (a
    per-level list of {'level', 'residuals', 'corrections', 'converged'}).
    """
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")
    if n_levels == 1:
        sol = newton_raphson(
            mesh=mesh, interp_g=BicubicInterpolator(normalize_image(g)),
            f=normalize_image(f), U_init=U_init, max_iter=max_iter, tol=tol,
            variant=variant, sigma_f=sigma_f, reg_rel=reg_rel,
            valid_elements=valid_elements)
        sol["history"] = [{"level": 0, "residuals": sol["residuals"],
                           "corrections": sol["corrections"],
                           "converged": sol["converged"]}]
        return sol

    # Per-level normalisation keeps the ZNCC-consistent contrast at each scale.
    pyr_f = [normalize_image(im) for im in
             build_gaussian_pyramid(f, n_levels, sigma)]
    pyr_g = [normalize_image(im) for im in
             build_gaussian_pyramid(g, n_levels, sigma)]

    U = None
    history: List[dict] = []
    cov = None
    sigma_u = None
    last_residuals: List[float] = []
    last_corrections: List[float] = []
    converged = False

    for level in range(n_levels - 1, -1, -1):
        scale = 2 ** level
        mesh_level = Q4Mesh(
            x0=mesh.x0 / scale, y0=mesh.y0 / scale,
            x1=mesh.x1 / scale, y1=mesh.y1 / scale,
            n_elem_x=mesh.n_elem_x, n_elem_y=mesh.n_elem_y)

        if U is None:
            # Seed the coarsest level. A caller-provided native-scale init is
            # brought down to this level's pixel scale by /scale.
            U_lvl = (np.zeros(mesh_level.n_dof) if U_init is None
                     else np.asarray(U_init, float) / scale)
        else:
            # U already carries the previous (coarser) level's solution rescaled
            # to the current finer scale (the *2 done at the end of that level),
            # so it is used as-is here. NOTE: q4dic divides it again by 2 here
            # (solver.py:494) which cancels its own end-of-level *2 (line 521);
            # that double scaling is a bug, so we deliberately do NOT replicate
            # it -- the displacement must grow by 2x from coarse to fine.
            # Orphan nodes carry NaN; reset them to 0 so the next level's init
            # stays finite (they will be re-flagged NaN by that level's solve).
            U_lvl = np.nan_to_num(U, nan=0.0)

        sigma_f_level = sigma_f if level == 0 else None
        sol = newton_raphson(
            mesh=mesh_level, interp_g=BicubicInterpolator(pyr_g[level]),
            f=pyr_f[level], U_init=U_lvl, max_iter=max_iter, tol=tol,
            variant=variant, sigma_f=sigma_f_level, reg_rel=reg_rel,
            valid_elements=valid_elements)

        U = sol["U"]
        last_residuals = sol["residuals"]
        last_corrections = sol["corrections"]
        converged = sol["converged"]
        if level == 0:
            cov = sol["cov"]
            sigma_u = sol["sigma_u"]
        history.append({"level": level, "residuals": sol["residuals"],
                        "corrections": sol["corrections"],
                        "converged": sol["converged"]})
        if level > 0:
            U = U * 2.0              # coarse -> finer next level: 2x the pixels

    return {"U": U, "residuals": last_residuals,
            "corrections": last_corrections,
            "n_iter": len(last_corrections), "converged": converged,
            "cov": cov, "sigma_u": sigma_u, "history": history}


# =============================================================================
# Strain post-processing  (ported from q4dic/postprocessing.py)
# =============================================================================

def _equiv(exx, eyy, exy):
    """von Mises equivalent strain, 2D with plane incompressibility closure
    e_zz = -(exx + eyy). IDENTICAL to gui.core.dic._equiv so both engines share
    the convention."""
    ezz = -(exx + eyy)
    return np.sqrt(2.0 / 3.0 * (exx ** 2 + eyy ** 2 + ezz ** 2 + 2.0 * exy ** 2))


def compute_strains_at_nodes(U: np.ndarray, mesh: Q4Mesh
                             ) -> Dict[str, np.ndarray]:
    """Strain fields at the mesh nodes from the nodal displacement.

    Strains are evaluated at the 2x2 Gauss points of each element by
    differentiating the Q4 shape functions, then averaged onto the nodes
    (simple element-contribution averaging, q4dic/strains_gauss_to_nodes). The
    shear is tensorial ``eps_xy = 0.5(du_x/dy + du_y/dx)`` and ``eps_vm`` uses
    the e_zz closure, matching the local engine.

    Returns a dict of (n_nodes,) arrays: 'eps_xx', 'eps_yy', 'eps_xy', 'eps_vm'.
    All in pixel-based (dimensionless) strain; the mapping to the model frame
    does not change strain values (uniform scale, axis flip only changes signs
    of cross terms, handled in the field assembler).
    """
    gauss_pts, _ = gauss_points_2d(2)
    jxx = mesh.elem_size_x / 2.0           # rectangular-element Jacobian
    jyy = mesh.elem_size_y / 2.0

    acc = {k: np.zeros(mesh.n_nodes) for k in ("eps_xx", "eps_yy", "eps_xy")}
    count = np.zeros(mesh.n_nodes)

    for e in range(mesh.n_elements):
        node_ids = mesh.connectivity[e]
        dof_ids = mesh.dof_indices(e)
        U_elem = U[dof_ids]
        ux_nodes = U_elem[0::2]
        uy_nodes = U_elem[1::2]

        # Average strain over the element's Gauss points (constant gradient
        # within a rectangular Q4 only at the centre; averaging the Gauss
        # points gives the element-mean strain, then spread to its 4 nodes).
        exx_e = eyy_e = exy_e = 0.0
        for (xi_g, eta_g) in gauss_pts:
            dN = shape_function_derivatives(xi_g, eta_g)
            dN_dx = dN[0] / jxx
            dN_dy = dN[1] / jyy
            exx_e += dN_dx @ ux_nodes
            eyy_e += dN_dy @ uy_nodes
            exy_e += 0.5 * (dN_dy @ ux_nodes + dN_dx @ uy_nodes)
        ng = gauss_pts.shape[0]
        exx_e /= ng
        eyy_e /= ng
        exy_e /= ng

        for n in node_ids:
            acc["eps_xx"][n] += exx_e
            acc["eps_yy"][n] += eyy_e
            acc["eps_xy"][n] += exy_e
            count[n] += 1

    count = np.where(count == 0, 1, count)
    exx = acc["eps_xx"] / count
    eyy = acc["eps_yy"] / count
    exy = acc["eps_xy"] / count
    return {"eps_xx": exx, "eps_yy": eyy, "eps_xy": exy,
            "eps_vm": _equiv(exx, eyy, exy)}


def nodal_displacements(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split the DOF vector into (ux, uy) per node (pixels)."""
    return U[0::2].copy(), U[1::2].copy()


# =============================================================================
# Sequence orchestration  ->  field arrays for the existing viewer
# =============================================================================

def compute_dic_global_fields(frames: Sequence, roi: Tuple[float, float, float, float],
                              params: DicGlobalParams, fps: float, mm_per_px: float,
                              img_w: int, img_h: int, trigger_offset_s: float = 0.0,
                              sigma_f: Optional[float] = None, progress=None,
                              on_frame=None
                              ) -> Dict[str, object]:
    """Run global Q4-DIC over a frame sequence and return field arrays shaped
    like ``gui.core.dic.compute_dic_fields`` so the existing viewer and
    ``exp_field_io`` work unchanged.

    The measurement points are the MESH NODES (not a subset grid). Output arrays
    are (n_pairs, n_nodes); node coordinates x, y are in the model frame (mm).

    Parameters mirror ``compute_dic_fields`` where they overlap. ``sigma_f`` (if
    given) enables the analytic per-node displacement uncertainty, returned as
    extra fields ``sigma_Ux``/``sigma_Uy`` (mm).

    ``progress(i_done, n_pairs)`` is the simple progress callback shared with the
    local engine. ``on_frame(info)`` is an optional detailed callback invoked
    once per completed pair with a dict::

        {'index': i, 'n_pairs': N, 'n_iter': k, 'residual': r,
         'converged': bool, 'elapsed_s': float, 'frame_s': float}

    where ``frame_s`` is the wall-clock time of that pair and ``elapsed_s`` the
    cumulative time since the start. It is meant to drive a status log and an
    ETA in the UI; it has no effect on the computation.

    Returns a dict with keys: x, y, t, valid, grid (None for an FE mesh),
    fields, units, plus 'mesh' (the Q4Mesh) for downstream use.

    Notes
    -----
    - Displacement sign: image y points down, model y points up, so
      ``Uy_model = -uy_pixel * mm_per_px`` (as in the local engine).
    - Strain components: a y-axis flip negates the cross derivative once, so the
      tensorial ``eps_xy`` changes sign while ``eps_xx``/``eps_yy`` are
      unchanged; ``eps_vm`` is invariant. We negate ``Exy`` on output to express
      it in the model frame, consistent with the displacement convention.
    - Only the INSTANTANEOUS strain rates (``Exx_dot``/``Eyy_dot``/``Exy_dot``/
      ``Eeq_dot``) are produced; cumulated strain is omitted to match the local
      engine (no physical meaning once material leaves the field of view).
    - 'incremental' vs total and the U_init strategy follow ``params`` (they
      still set which reference frame each pair uses).
    """
    n_img = len(frames)
    if n_img < 2:
        raise ValueError("need a sequence of at least 2 frames")
    if params.variant not in ("standard", "hild"):
        raise ValueError("params.variant must be 'standard' or 'hild'")

    mesh = build_mesh_on_roi(roi, params.elem_size)
    n_nodes = mesh.n_nodes
    n_pairs = n_img - 1
    dt = 1.0 / fps if fps else 1.0

    # Node coordinates -> model frame (mm). pixel_to_model flips y.
    xy = np.array([pixel_to_model(nx, ny, img_w, img_h, mm_per_px)
                   for nx, ny in mesh.nodes])
    x_mm = xy[:, 0]
    y_mm = xy[:, 1]

    names = ["Ux", "Uy", "Umag", "Vx", "Vy", "Vmag",
             "Exx_dot", "Eyy_dot", "Exy_dot", "Eeq_dot", "residual"]
    want_sigma = sigma_f is not None
    if want_sigma:
        names += ["sigma_Ux", "sigma_Uy"]
    fields = {k: np.full((n_pairs, n_nodes), np.nan) for k in names}
    valid = np.zeros((n_pairs, n_nodes), bool)
    t = np.zeros(n_pairs)

    f0_raw = frames[0] if not params.incremental else None
    U_prev: Optional[np.ndarray] = None
    t_start = time.perf_counter()
    use_pyramid = int(getattr(params, "pyramid_levels", 1)) > 1
    tool_poly = getattr(params, "tool_polygon", None)
    has_tool = tool_poly is not None and len(tool_poly) >= 3
    use_mask = bool(getattr(params, "mask_enabled", False)) or has_tool
    convect = bool(getattr(params, "convect", False))
    U_cumul = np.zeros(mesh.n_dof)             # accumulated nodal displacement

    for i in range(n_pairs):
        t_frame0 = time.perf_counter()
        if params.incremental:
            f_raw = frames[i]
            g_raw = frames[i + 1]
        else:
            f_raw = f0_raw
            g_raw = frames[i + 1]

        # Lagrangian convection: solve on the mesh displaced by the cumulated
        # displacement of the previous pairs, so nodes follow the material.
        mesh_calc = convect_mesh(mesh, U_cumul) if convect else mesh

        # Per-pair element validity: material coverage (intensity threshold
        # AND outside the tool polygon) and, when convecting, the Jacobian sign.
        valid_elements = None
        if use_mask:
            min_int = (params.mask_min_intensity
                       if getattr(params, "mask_enabled", False) else 0.0)
            mat = material_mask(f_raw, min_int, tool_poly if has_tool else None)
            valid_elements = element_coverage_mask(
                mesh_calc, mat, params.coverage_threshold)
        if convect:
            jac_ok = check_jacobian(mesh_calc)
            valid_elements = jac_ok if valid_elements is None else (
                valid_elements & jac_ok)

        U_init = U_prev if (params.u_init_previous and U_prev is not None) else None

        if use_pyramid:
            # The multi-scale solver normalises each pyramid level itself.
            sol = multiscale_newton_raphson(
                mesh=mesh_calc, f=f_raw, g=g_raw,
                n_levels=int(params.pyramid_levels),
                sigma=float(params.pyramid_sigma), U_init=U_init,
                max_iter=params.max_iter, tol=params.tol,
                variant=params.variant, sigma_f=sigma_f,
                reg_rel=params.reg_rel, valid_elements=valid_elements)
        else:
            sol = newton_raphson(
                mesh=mesh_calc, interp_g=BicubicInterpolator(normalize_image(g_raw)),
                f=normalize_image(f_raw), U_init=U_init,
                max_iter=params.max_iter, tol=params.tol, variant=params.variant,
                sigma_f=sigma_f, reg_rel=params.reg_rel,
                valid_elements=valid_elements)
        U = sol["U"]                            # displacement of this pair
        U_prev = U
        if convect:
            # Accumulate for the next pair's convection (orphan NaN -> 0).
            U_cumul = U_cumul + np.nan_to_num(U, nan=0.0)

        ux_px, uy_px = nodal_displacements(U)
        ux_mm = ux_px * mm_per_px
        uy_mm = -uy_px * mm_per_px              # image y down -> model y up

        fields["Ux"][i] = ux_mm
        fields["Uy"][i] = uy_mm
        fields["Umag"][i] = np.hypot(ux_mm, uy_mm)
        fields["Vx"][i] = ux_mm / dt
        fields["Vy"][i] = uy_mm / dt
        fields["Vmag"][i] = np.hypot(ux_mm, uy_mm) / dt

        st = compute_strains_at_nodes(U, mesh_calc)
        exx = st["eps_xx"]
        eyy = st["eps_yy"]
        exy = -st["eps_xy"]                     # model-frame sign (see notes)
        # Instantaneous strain rates only; cumulated strain is omitted to match
        # the local engine (no physical meaning once material leaves the FOV).
        fields["Exx_dot"][i] = exx / dt
        fields["Eyy_dot"][i] = eyy / dt
        fields["Exy_dot"][i] = exy / dt
        fields["Eeq_dot"][i] = _equiv(exx, eyy, exy) / dt

        res = sol["residuals"][-1] if sol["residuals"] else np.nan
        fields["residual"][i] = res

        if want_sigma and sol["sigma_u"] is not None:
            su = sol["sigma_u"]
            fields["sigma_Ux"][i] = su[0::2] * mm_per_px
            fields["sigma_Uy"][i] = su[1::2] * mm_per_px

        valid[i] = sol["converged"]
        t[i] = trigger_offset_s + (i + 0.5) * dt

        if progress is not None:
            progress(i + 1, n_pairs)
        if on_frame is not None:
            now = time.perf_counter()
            on_frame({"index": i, "n_pairs": n_pairs,
                      "n_iter": sol["n_iter"],
                      "residual": float(res) if np.isfinite(res) else None,
                      "converged": bool(sol["converged"]),
                      "elapsed_s": now - t_start,
                      "frame_s": now - t_frame0})

    units = {"Ux": "mm", "Uy": "mm", "Umag": "mm",
             "Vx": "mm/s", "Vy": "mm/s", "Vmag": "mm/s",
             "Exx_dot": "1/s", "Eyy_dot": "1/s", "Exy_dot": "1/s", "Eeq_dot": "1/s",
             "residual": "-"}
    if want_sigma:
        units["sigma_Ux"] = "mm"
        units["sigma_Uy"] = "mm"

    return {"x": x_mm, "y": y_mm, "t": t, "valid": valid, "grid": None,
            "fields": fields, "units": units, "mesh": mesh}
