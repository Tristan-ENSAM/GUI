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

Functional actually minimised: C(U) = sum_p [f(x_p) - g(x_p + u(x_p))]^2 over
the pixels p owned by the valid elements (each pixel counted once), f and g
normalised (zero mean, unit std) with statistics taken on the ROI. The system
is assembled as sparse matrices (H = G^T G, h = G^T r) and solved by sparse LU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Sequence
import time
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from gui.core.alignment import pixel_to_model

# Default dilation (px) of the saturation mask: pixels next to a saturated
# area are contaminated too (cubic-spline ringing across the clipped edge,
# plus the motion of the pair). Chosen from a measurement, not a literature
# value: a stationary saturated half-image, 1.5 px speckle, 0.3 px motion,
# gave a max error on the neighbouring textured nodes of 0.084 / 0.046 /
# 0.010 / 0.0027 / 0.0023 px for 0 / 1 / 2 / 3 / 5 px (0.0025 px without
# any saturation). Increase it for larger motions per pair.
SAT_MARGIN_PX = 3


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
    min_std_rel : float
        Texture threshold: an element whose grey-level std over its pixels is
        <= ``min_std_rel * ptp(reference frame)`` (flat, saturated) is
        excluded like a masked element (its orphan nodes -> NaN, invalid).
        0 disables it. Default 1e-3 (design choice, same rule as the local
        engine).
    saturation_level : float or None
        Grey level at or above which a pixel is saturated (e.g. 4095 for a
        12-bit camera); None disables the saturation mask. Saturated pixels
        of the reference OR the deformed image, dilated by
        ``saturation_margin`` px, are left out of the residual and of the
        normalisation statistics; an element whose remaining fraction is
        below ``coverage_threshold`` is excluded.
    grey_correction : bool
        Estimate a global grey-level gain/offset (a, b) over the ROI with the
        displacement (two extra unknowns): r = f - [(1+a) g(x+u) + b].
    init_search : int
        When > 0, each pair is initialised by a local ZNCC correlation at the
        nodes (subset = element size, half search range ``init_search`` px),
        extending the capture range beyond the Gauss-Newton basin (about the
        speckle size, x2 per pyramid level). 0 = no local initialisation.
    convect : bool
        Requires ``incremental=True`` (the total pattern already measures
        from frame 0; accumulating its totals was inconsistent).
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
    min_std_rel: float = 1e-3
    init_search: int = 0
    grey_correction: bool = True
    saturation_level: Optional[float] = None
    saturation_margin: int = SAT_MARGIN_PX

    def to_json_dict(self) -> dict:
        return asdict(self)


def _to_gray_f64(img: np.ndarray) -> np.ndarray:
    """Image to float64 grayscale (mean of RGB if needed). Matches the local
    engine's ``_to_gray_f32`` but keeps float64 for the spline/solver."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a.astype(np.float64)


def roi_stats(img: np.ndarray, region=None,
              pixel_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """(mean, std) of the grey levels of ``img`` over ``region``, restricted
    to the usable pixels of ``pixel_mask`` (bool image, True = usable) when
    given (e.g. saturated pixels left out of the statistics)."""
    g = _to_gray_f64(img)
    if pixel_mask is not None:
        m = np.asarray(pixel_mask, bool)
        if region is not None:
            m = m[region]
            g = g[region]
        s = g[m]
    else:
        s = g if region is None else g[region]
    if s.size == 0:
        s = _to_gray_f64(img)
    return float(s.mean()), float(s.std())


def saturation_mask(image: np.ndarray, level: float,
                    margin: int = SAT_MARGIN_PX) -> np.ndarray:
    """Bool image, True where the grey level is >= ``level`` (saturated:
    the sensor clipped, the pixel carries no displacement information),
    dilated by ``margin`` px. For a 12-bit camera ``level`` = 4095."""
    g = _to_gray_f64(image)
    m = g >= float(level)
    if margin > 0 and m.any():
        from scipy.ndimage import binary_dilation
        m = binary_dilation(m, iterations=int(margin))
    return m


def _downsample_mask(mask: np.ndarray, n_levels: int) -> List[np.ndarray]:
    """Usable-pixel mask per pyramid level: a coarse pixel is usable only if
    the native pixels feeding it (Gaussian sigma ~1 px before each 2x
    decimation) are usable -> erosion by 2 px, then decimation."""
    from scipy.ndimage import binary_erosion
    out = [np.asarray(mask, bool)]
    cur = out[0]
    for _ in range(n_levels - 1):
        cur = binary_erosion(cur, iterations=2, border_value=1)[::2, ::2]
        out.append(cur)
    return out


def normalize_with(img: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Affine grey-level map (img - mu) / sd (mean subtraction if sd ~ 0).
    Apply the SAME (mu, sd) -- taken on the reference ROI -- to f and g:
    separate maps would create a grey-level mismatch whenever the content
    of the region differs between the two images (i.e. under motion)."""
    g = _to_gray_f64(img)
    if sd < 1e-12:
        return g - mu
    return (g - mu) / sd


def normalize_image(img: np.ndarray, region=None) -> np.ndarray:
    """Zero-mean, unit-std normalisation (a global affine grey-level
    correction; the functional minimised is then the SSD of the normalised
    images, not a per-element ZNSSD). ``region`` (tuple of slices or bool
    mask) restricts the mean/std to the correlated area: statistics over the
    whole image changed the grey levels inside the ROI whenever content
    changed OUTSIDE it (chip, tool, background), which biased the solution.
    Falls back to mean subtraction only when the region is flat."""
    g = _to_gray_f64(img)
    s = g if region is None else g[region]
    if s.size == 0:
        s = g
    mu = float(s.mean())
    sd = float(s.std())
    if sd < 1e-12:
        return g - mu
    return (g - mu) / sd


def roi_region(mesh: "Q4Mesh", shape: Tuple[int, int], margin: float = 0.0):
    """Slices of the bounding box of the mesh nodes, expanded by ``margin``
    px and clipped to the image: the area used for the normalisation
    statistics (use the same area for f and g)."""
    H, W = shape[:2]
    m = int(np.ceil(max(0.0, float(margin))))
    y0 = max(0, int(np.floor(mesh.nodes[:, 1].min())) - m)
    y1 = min(H, int(np.ceil(mesh.nodes[:, 1].max())) + 1 + m)
    x0 = max(0, int(np.floor(mesh.nodes[:, 0].min())) - m)
    x1 = min(W, int(np.ceil(mesh.nodes[:, 0].max())) + 1 + m)
    return (slice(y0, y1), slice(x0, x1))


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


def inverse_bilinear(coords: np.ndarray, x: np.ndarray, y: np.ndarray,
                     n_iter: int = 20, tol: float = 1e-12):
    """Natural coordinates (xi, eta) of physical points (x, y) in a general
    Q4 whose nodes ``coords`` (4, 2) follow the 1-2-3-4 numbering; Newton
    iterations on x(xi, eta) = x. Returns (xi, eta, inside) with inside =
    |xi|, |eta| <= 1 (+1e-9). Exact in one step for a parallelogram."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xi = np.zeros_like(x)
    eta = np.zeros_like(x)
    cx = coords[:, 0]
    cy = coords[:, 1]
    for _ in range(n_iter):
        N = shape_functions_grid(xi, eta)
        rx = N.T @ cx - x
        ry = N.T @ cy - y
        dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        a = dxi.T @ cx
        b = deta.T @ cx
        c = dxi.T @ cy
        d = deta.T @ cy
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        sxi = (d * rx - b * ry) / det
        seta = (-c * rx + a * ry) / det
        xi = xi - sxi
        eta = eta - seta
        if np.max(np.abs(sxi), initial=0.0) < tol and \
                np.max(np.abs(seta), initial=0.0) < tol:
            break
    inside = (np.abs(xi) <= 1 + 1e-9) & (np.abs(eta) <= 1 + 1e-9)
    return xi, eta, inside


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
        # Rectangular, axis-aligned elements (False once convected): selects
        # the affine (exact) or the isoparametric (Newton) inverse mapping.
        self.axis_aligned = True
        self._px_cache = {}

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
        """Map physical (x, y) to natural (xi, eta) in an element. Exact
        affine map for an axis-aligned rectangle (min corner -> (-1, -1));
        isoparametric Newton inversion for a general (convected) Q4."""
        node_coords = self.nodes[self.connectivity[elem_idx]]
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if not self.axis_aligned:
            xi, eta, _ = inverse_bilinear(node_coords, x, y)
            return xi, eta
        x_min = node_coords[:, 0].min()
        y_min = node_coords[:, 1].min()
        sx = node_coords[:, 0].max() - x_min
        sy = node_coords[:, 1].max() - y_min
        xi = 2.0 * (x - x_min) / sx - 1.0
        eta = 2.0 * (y - y_min) / sy - 1.0
        return xi, eta

    def _owned_pixels(self, elem_idx: int):
        """(x, y, xi, eta) of the integer pixels OWNED by the element, cached.

        Every pixel of the meshed area belongs to exactly one element: an
        element owns the half-open cell [-1, 1) x [-1, 1) in natural
        coordinates, closed (<= 1) on the last column / row of the mesh.
        (A closed cell counted the pixels of shared edges twice and of shared
        corners four times, i.e. a weighted functional.) Axis-aligned
        elements use the exact affine map; general (convected) quads the
        isoparametric inversion, restricted to pixels inside the quad."""
        hit = self._px_cache.get(elem_idx)
        if hit is not None:
            return hit
        c = self.nodes[self.connectivity[elem_idx]]
        ie, je = divmod(int(elem_idx), self.n_elem_x)
        last_x = je == self.n_elem_x - 1
        last_y = ie == self.n_elem_y - 1
        x_px = np.arange(np.ceil(c[:, 0].min() - 1e-9),
                         np.floor(c[:, 0].max() + 1e-9) + 1, dtype=float)
        y_px = np.arange(np.ceil(c[:, 1].min() - 1e-9),
                         np.floor(c[:, 1].max() + 1e-9) + 1, dtype=float)
        xx, yy = np.meshgrid(x_px, y_px)
        xx = xx.ravel(); yy = yy.ravel()
        xi, eta = self.physical_to_natural(xx, yy, elem_idx)
        tol = 1e-9
        keep = (xi >= -1 - tol) & (eta >= -1 - tol)
        keep &= (xi <= 1 + tol) if last_x else (xi < 1 - tol)
        keep &= (eta <= 1 + tol) if last_y else (eta < 1 - tol)
        out = (xx[keep], yy[keep], np.clip(xi[keep], -1, 1),
               np.clip(eta[keep], -1, 1))
        self._px_cache[elem_idx] = out
        return out

    def get_pixel_points_in_element(self, elem_idx: int
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integer pixel coordinates owned by an element (each pixel of the
        mesh belongs to exactly one element, see ``_owned_pixels``).
        Pixel-wise quadrature (q4dic/Besnard) is more faithful to the discrete
        image than Gauss quadrature for the grey-level residual."""
        x, y, _, _ = self._owned_pixels(elem_idx)
        return x, y

    def build_shape_matrix_at_pixels(self, elem_idx: int
                                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shape matrix [N] (2, 8, n_pixels) at all pixels of an element, with
        the pixel coordinates. Row 0 maps nodal DOF to ux, row 1 to uy."""
        x_pts, y_pts, xi, eta = self._owned_pixels(elem_idx)
        N = shape_functions_grid(xi, eta)             # (4, n_pts)
        n_pts = x_pts.size
        N_mat = np.zeros((2, 8, n_pts))
        for i in range(4):
            N_mat[0, 2 * i, :] = N[i]
            N_mat[1, 2 * i + 1, :] = N[i]
        return N_mat, x_pts, y_pts

    def pixel_operator(self, valid_elements: Optional[np.ndarray] = None):
        """Vectorised pixel-level operator over the owned pixels of the valid
        elements (cached per mask): dict with
          x, y   : (n_pix,) integer pixel coordinates (float arrays)
          elem   : (n_pix,) owning element
          Bx, By : sparse (n_pix, n_dof) with u_x(pix) = Bx @ U,
                   u_y(pix) = By @ U (Q4 shape functions)
        Same mathematics as ``build_shape_matrix_at_pixels`` for every
        element, stacked, so the Gauss-Newton system is assembled as
        H = G^T G, h = G^T r with G = diag(gx) Bx + diag(gy) By."""
        import scipy.sparse as sp
        key = None if valid_elements is None else \
            np.asarray(valid_elements, bool).tobytes()
        cache = self._px_cache.setdefault("_ops", {})
        if key in cache:
            return cache[key]
        xs, ys, es, rows, cols, vals = [], [], [], [], [], []
        n0 = 0
        for e in range(self.n_elements):
            if valid_elements is not None and not valid_elements[e]:
                continue
            x, y, xi, eta = self._owned_pixels(e)
            n = x.size
            if n == 0:
                continue
            N = shape_functions_grid(xi, eta)            # (4, n)
            nodes = self.connectivity[e]
            r = np.arange(n0, n0 + n)
            for a in range(4):
                rows.append(r); cols.append(np.full(n, nodes[a])); vals.append(N[a])
            xs.append(x); ys.append(y); es.append(np.full(n, e))
            n0 += n
        if n0 == 0:
            op = {"x": np.empty(0), "y": np.empty(0), "elem": np.empty(0, int),
                  "Bx": sp.csr_matrix((0, self.n_dof)),
                  "By": sp.csr_matrix((0, self.n_dof)), "n_pix": 0}
        else:
            rows = np.concatenate(rows); nodes_ = np.concatenate(cols)
            vals = np.concatenate(vals)
            Bx = sp.csr_matrix((vals, (rows, 2 * nodes_)), shape=(n0, self.n_dof))
            By = sp.csr_matrix((vals, (rows, 2 * nodes_ + 1)), shape=(n0, self.n_dof))
            op = {"x": np.concatenate(xs), "y": np.concatenate(ys),
                  "elem": np.concatenate(es), "Bx": Bx, "By": By, "n_pix": n0}
        cache[key] = op
        return op


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

    For each element, the fraction of material pixels among the pixels it
    OWNS (the pixels its residual uses, also for a convected quad) is
    compared to ``coverage_threshold``; the element is kept when
    ``frac >= coverage_threshold``. Returns a (n_elements,) bool array.
    """
    H, W = material_mask.shape
    valid = np.zeros(mesh.n_elements, dtype=bool)
    for e in range(mesh.n_elements):
        x, y = mesh.get_pixel_points_in_element(e)
        inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
        if not inside.any():
            continue
        frac = material_mask[y[inside].astype(int), x[inside].astype(int)].mean()
        valid[e] = frac >= coverage_threshold
    return valid


def element_texture_mask(mesh: Q4Mesh, image: np.ndarray,
                         min_std_rel: float = 1e-3) -> np.ndarray:
    """Per-element texture validity: an element whose grey-level std over
    its owned pixels is <= ``min_std_rel * ptp(image)`` carries no image
    information (flat, saturated) and is excluded like a masked element
    (its nodes become orphans -> NaN, invalid) instead of being "measured"
    as its initial guess. Returns a (n_elements,) bool array."""
    g = _to_gray_f64(image)
    H, W = g.shape
    thr = float(min_std_rel) * float(np.ptp(g))
    ok = np.zeros(mesh.n_elements, dtype=bool)
    for e in range(mesh.n_elements):
        x, y = mesh.get_pixel_points_in_element(e)
        inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
        if inside.sum() < 2:
            continue
        v = g[y[inside].astype(int), x[inside].astype(int)]
        ok[e] = float(v.std()) > thr
    return ok


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
    # General quads from now on: isoparametric inverse map, fresh pixel cache.
    new.axis_aligned = False
    new._px_cache = {}
    return new


def check_jacobian(mesh: Q4Mesh) -> np.ndarray:
    """Per-element validity from the sign of the Jacobian determinant. For a
    Q4, det(J) is linear in xi and in eta (no xi*eta term), so its minimum
    over the element is reached at a corner: the element is valid iff
    det(J) > 0 at its 4 corners (the centre alone misses corner fold-overs).
    Returns a (n_elements,) bool array."""
    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    dNs = [shape_function_derivatives(xi, eta) for xi, eta in corners]
    valid = np.ones(mesh.n_elements, dtype=bool)
    for e in range(mesh.n_elements):
        coords = mesh.nodes[mesh.connectivity[e]]    # (4, 2)
        for dN in dNs:
            J = dN @ coords                          # (2, 2)
            if J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0] <= 0:
                valid[e] = False
                break
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


def compute_uncertainty(H_global, sigma_f: float, n_dof: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic displacement covariance ``Cov = 2 sigma_f^2 [H]^-1`` and per-DOF
    std ``sqrt(diag(Cov))`` (Hild & Roux). ``H_global`` may be dense or
    sparse. A small relative floor is added to [H] for conditioning. Even
    DOF = ux, odd DOF = uy. ``sigma_f`` must be in the grey-level units of
    the images the solver used (normalised images: sigma_GL / std)."""
    import scipy.sparse as sp
    if sp.issparse(H_global):
        H_global = H_global.toarray()
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

def _regularise_and_constrain(H_global, reg_rel: float, n_dof: int,
                              orphan_dof: Optional[np.ndarray]):
    """Return the regularised Hessian used for the linear solve (dense or
    sparse, same type as the input).

    Adds the relative diagonal floor ``reg_rel * trace([H])/n_dof`` for
    conditioning. It damps the STEP only: at the fixed point dU = 0, which
    requires h = 0 (exact stationarity of the cost), so it does not bias the
    solution. For orphan DOF (nodes touching only excluded elements) the
    row/column is zeroed and a unit pivot is set, so the linear solve yields
    ``dU = 0`` there (the RHS is also zeroed by the caller).
    """
    import scipy.sparse as sp
    if sp.issparse(H_global):
        reg = reg_rel * float(H_global.diagonal().sum()) / max(n_dof, 1)
        H_reg = (H_global + reg * sp.identity(n_dof, format="csc")).tocsc()
        if orphan_dof is not None and orphan_dof.any():
            keep = sp.diags((~orphan_dof).astype(float))
            H_reg = (keep @ H_reg @ keep
                     + sp.diags(orphan_dof.astype(float))).tocsc()
        return H_reg
    reg = reg_rel * np.trace(H_global) / max(n_dof, 1)
    H_reg = H_global + reg * np.eye(n_dof)
    if orphan_dof is not None and orphan_dof.any():
        H_reg = H_reg.copy()
        H_reg[orphan_dof, :] = 0.0
        H_reg[:, orphan_dof] = 0.0
        H_reg[orphan_dof, orphan_dof] = 1.0
    return H_reg


def _orphan_dof(mesh: Q4Mesh, valid_elements: Optional[np.ndarray]):
    if valid_elements is None:
        return None
    orphan_nodes = ~active_nodes_from_elements(mesh, valid_elements)
    orphan = np.zeros(mesh.n_dof, dtype=bool)
    orphan[0::2] = orphan_nodes
    orphan[1::2] = orphan_nodes
    return orphan


def newton_raphson(mesh: Q4Mesh, interp_g: BicubicInterpolator, f: np.ndarray,
                   U_init: Optional[np.ndarray] = None, max_iter: int = 30,
                   tol: float = 1e-4, variant: str = "standard",
                   sigma_f: Optional[float] = None, reg_rel: float = 1e-6,
                   valid_elements: Optional[np.ndarray] = None,
                   min_step: float = 1.0 / 16.0,
                   grey_correction: bool = True,
                   line_search: bool = True,
                   pixel_mask: Optional[np.ndarray] = None
                   ) -> Dict[str, object]:
    """Gauss-Newton minimisation of the global grey-level residual

        C(U, a, b) = sum_{pixels p owned by valid elements} r_p^2,
        r_p = f(x_p) - [(1 + a) g(x_p + u(x_p)) + b],  u(x) = sum_a N_a(x) u_a,

    each pixel counted once. (a, b) is a global grey-level gain/offset
    correction over the ROI (``grey_correction=True``; a = b = 0 fixed
    otherwise): it absorbs the brightness/contrast mismatch left by the
    normalisation, exactly (two extra unknowns of the same least-squares
    problem, not an ad hoc rescaling).

    Linearisation (``variant='standard'``, exact Gauss-Newton): the columns
    of -J_r are (1+a) grad g(x+u) . N for U, g(x+u) for a and 1 for b, so
    (J^T J) dz = -J^T r reads H dz = h with H = G^T G, h = G^T r.
    ``variant='hild'``: grad g(x+u) and g(x+u) replaced by grad f(x) and
    f(x) in G (constant matrix, factorised once; modified Gauss-Newton,
    Correli-Q4-like).

    ``pixel_mask`` (bool image of f's shape, True = usable): pixels flagged
    False (e.g. saturated, see ``saturation_mask``) are left out of the
    residual; an element left without any pixel is treated as excluded.

    Step control: the cost at the new iterate is checked at the next
    assembly; if it increased, the step is halved (backtracking) down to
    ``min_step``, below which the solve stops (``stop_reason='stagnation'``,
    not converged). Convergence: max nodal correction |dU|_node < ``tol``
    (px), independent of the mesh size.

    Returns a dict with keys:
        'U'              : (n_dof,) nodal displacement (px); NaN on orphans
        'residuals'      : RMS residual at each accepted iterate (before its step)
        'corrections'    : max nodal |dU| (px) of each applied step
        'n_iter'         : steps applied
        'converged'      : bool
        'stop_reason'    : 'converged' | 'max_iter' | 'stagnation' | 'singular'
        'n_backtracks'   : int
        'grey_ab'        : (a, b) grey-level gain/offset at the solution
        'residual_final' : RMS residual at the returned U
        'elem_rms'       : (n_elements,) RMS residual per element at the
                           returned U (NaN for excluded elements)
        'cov', 'sigma_u' : analytic covariance / std (sigma_f given) or None
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    if variant not in ("standard", "hild"):
        raise ValueError("variant must be 'standard' or 'hild'")
    n_dof = mesh.n_dof
    U = (np.zeros(n_dof) if U_init is None
         else np.nan_to_num(np.asarray(U_init, float), nan=0.0).copy())

    op = mesh.pixel_operator(valid_elements)
    if pixel_mask is not None and op["n_pix"]:
        pm = np.asarray(pixel_mask, bool)
        keep = pm[np.clip(op["y"].astype(int), 0, pm.shape[0] - 1),
                  np.clip(op["x"].astype(int), 0, pm.shape[1] - 1)]
        if not keep.all():
            op = {"x": op["x"][keep], "y": op["y"][keep],
                  "elem": op["elem"][keep], "Bx": op["Bx"][keep],
                  "By": op["By"][keep], "n_pix": int(keep.sum())}
            # elements left without pixels carry no data -> excluded
            has_px = np.bincount(op["elem"], minlength=mesh.n_elements) > 0
            valid_elements = has_px if valid_elements is None else (
                np.asarray(valid_elements, bool) & has_px)
    orphan_dof = _orphan_dof(mesh, valid_elements)
    n_ab = 2 if grey_correction else 0
    n_z = n_dof + n_ab
    orphan_z = None
    if orphan_dof is not None:
        orphan_z = np.concatenate([orphan_dof, np.zeros(n_ab, bool)])
    xp, yp, Bx, By = op["x"], op["y"], op["Bx"], op["By"]
    n_pixels = max(op["n_pix"], 1)
    H_img, W_img = f.shape
    f_vals = f[np.clip(yp.astype(int), 0, H_img - 1),
               np.clip(xp.astype(int), 0, W_img - 1)]
    ones = np.ones(op["n_pix"])

    def split(z):
        if n_ab:
            return z[:n_dof], z[n_dof], z[n_dof + 1]
        return z, 0.0, 0.0

    def residual(z):
        Uv, a_, b_ = split(z)
        xd = xp + Bx @ Uv
        yd = yp + By @ Uv
        gv = interp_g.evaluate(xd, yd)
        return f_vals - ((1.0 + a_) * gv + b_), xd, yd, gv

    def build_G(gx, gy, gv, a_):
        G = (1.0 + a_) * (sp.diags(gx) @ Bx + sp.diags(gy) @ By)
        if n_ab:
            G = sp.hstack([G, sp.csr_matrix(gv[:, None]),
                           sp.csr_matrix(ones[:, None])])
        return G.tocsr()

    z = np.concatenate([U, np.zeros(n_ab)])
    lu_fixed = None
    H_fixed = None
    G_fixed = None
    if variant == "hild":
        gx, gy = BicubicInterpolator(f).gradient(xp, yp)
        G_fixed = build_G(gx, gy, f_vals, 0.0)
        H_fixed = (G_fixed.T @ G_fixed).tocsc()
        try:
            lu_fixed = spl.splu(_regularise_and_constrain(
                H_fixed, reg_rel, n_z, orphan_z))
        except RuntimeError:
            lu_fixed = None

    residuals: List[float] = []
    corrections: List[float] = []
    converged = False
    stop_reason = "max_iter"
    n_backtracks = 0
    H_global = H_fixed
    z_acc = z.copy()
    C_acc = None
    dz_last = None
    step = 1.0
    k = 0
    while k < max_iter:
        r, xd, yd, gv = residual(z)
        C = float(r @ r)
        if line_search and C_acc is not None and C > C_acc * (1.0 + 1e-12):
            # the last step increased the cost: backtrack
            if step * 0.5 < min_step:
                z = z_acc
                stop_reason = "stagnation"
                break
            step *= 0.5
            n_backtracks += 1
            z = z_acc + step * dz_last
            corrections[-1] = corrections[-1] * 0.5
            continue
        z_acc = z.copy()
        C_acc = C
        residuals.append(float(np.sqrt(C / n_pixels)))
        if variant == "hild":
            h_global = G_fixed.T @ r
        else:
            gx, gy = interp_g.gradient(xd, yd)
            G = build_G(gx, gy, gv, split(z)[1])
            H_global = (G.T @ G).tocsc()
            h_global = G.T @ r
        if orphan_z is not None:
            h_global = np.where(orphan_z, 0.0, h_global)
        try:
            if lu_fixed is not None:
                dz = lu_fixed.solve(h_global)
            else:
                dz = spl.spsolve(_regularise_and_constrain(
                    H_global, reg_rel, n_z, orphan_z), h_global)
        except RuntimeError:
            dz = np.full(n_z, np.nan)
        if not np.all(np.isfinite(dz)):
            stop_reason = "singular"
            break
        step = 1.0
        dz_last = dz
        z = z_acc + dz
        k += 1
        dU = dz[:n_dof]
        corr = float(np.max(np.hypot(dU[0::2], dU[1::2]), initial=0.0))
        corrections.append(corr)
        if corr < tol:
            converged = True
            stop_reason = "converged"
            break

    # Final residual and per-element RMS at the RETURNED solution.
    r, _, _, _ = residual(z)
    U, a_fin, b_fin = split(z)
    U = U.copy()
    residual_final = float(np.sqrt(float(r @ r) / n_pixels))
    elem_rms = np.full(mesh.n_elements, np.nan)
    if op["n_pix"]:
        ssq = np.bincount(op["elem"], weights=r * r, minlength=mesh.n_elements)
        cnt = np.bincount(op["elem"], minlength=mesh.n_elements)
        has = cnt > 0
        elem_rms[has] = np.sqrt(ssq[has] / cnt[has])

    cov = None
    sigma_u = None
    if sigma_f is not None and H_global is not None:
        # covariance of all unknowns; the displacement block is kept (it
        # accounts for the correlation with the grey-level unknowns)
        cov_z, su_z = compute_uncertainty(H_global, sigma_f, n_z)
        cov = cov_z[:n_dof, :n_dof]
        sigma_u = su_z[:n_dof]

    if orphan_dof is not None:
        U = U.copy()
        U[orphan_dof] = np.nan          # orphan nodes carry no measurement
        if sigma_u is not None:
            sigma_u = sigma_u.copy()
            sigma_u[orphan_dof] = np.nan

    return {"U": U, "residuals": residuals, "corrections": corrections,
            "n_iter": len(corrections), "converged": converged,
            "stop_reason": stop_reason, "n_backtracks": n_backtracks,
            "grey_ab": (float(a_fin), float(b_fin)),
            "residual_final": residual_final, "elem_rms": elem_rms,
            "cov": cov, "sigma_u": sigma_u}


# =============================================================================
# Multi-scale (Gaussian pyramid)  (ported from q4dic preprocessing + solver)
# =============================================================================

# Smallest element side (px) allowed at the coarsest pyramid level: a 1 px
# element holds about one pixel for its 8 DOF. Design choice, not a
# literature value (3 px elements still capture a large translation in
# tests/test_dic_global.py::TestPyramid).
MIN_COARSE_ELEM_PX = 2.0


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
                              valid_elements: Optional[np.ndarray] = None,
                              grey_correction: bool = True,
                              pixel_mask: Optional[np.ndarray] = None
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
    # Normalisation: ONE affine grey-level map, with the statistics of the
    # reference image over the meshed area, applied to f and g. Content
    # changes outside the ROI no longer alter the grey levels inside it, and
    # f and g keep the same map (grey-level conservation preserved). A real
    # brightness/contrast change between the frames is estimated by the
    # solver (grey_correction).
    if n_levels == 1:
        mu, sd = roi_stats(f, roi_region(mesh, np.shape(f)), pixel_mask)
        sol = newton_raphson(
            mesh=mesh,
            interp_g=BicubicInterpolator(normalize_with(g, mu, sd)),
            f=normalize_with(f, mu, sd),
            U_init=U_init, max_iter=max_iter, tol=tol,
            variant=variant, sigma_f=sigma_f, reg_rel=reg_rel,
            valid_elements=valid_elements, grey_correction=grey_correction,
            pixel_mask=pixel_mask)
        sol["history"] = [{"level": 0, "residuals": sol["residuals"],
                           "corrections": sol["corrections"],
                           "converged": sol["converged"]}]
        return sol
    coarse = min(mesh.elem_size_x, mesh.elem_size_y) / 2 ** (n_levels - 1)
    if coarse < MIN_COARSE_ELEM_PX:
        raise ValueError(
            "pyramid too deep: %d levels shrink the %g px elements to %.2g px "
            "at the coarsest level (minimum %g px)"
            % (n_levels, min(mesh.elem_size_x, mesh.elem_size_y), coarse,
               MIN_COARSE_ELEM_PX))

    raw_f = build_gaussian_pyramid(f, n_levels, sigma)
    raw_g = build_gaussian_pyramid(g, n_levels, sigma)
    pyr_mask = (_downsample_mask(pixel_mask, n_levels)
                if pixel_mask is not None else [None] * n_levels)
    pyr_f, pyr_g = [], []
    for lv in range(n_levels):
        sc = 2 ** lv
        m_lv = mesh if lv == 0 else _scaled_mesh(mesh, 1.0 / sc)
        mu, sd = roi_stats(raw_f[lv], roi_region(m_lv, raw_f[lv].shape),
                           pyr_mask[lv])
        pyr_f.append(normalize_with(raw_f[lv], mu, sd))
        pyr_g.append(normalize_with(raw_g[lv], mu, sd))

    U = None
    history: List[dict] = []
    cov = None
    sigma_u = None
    last_residuals: List[float] = []
    last_corrections: List[float] = []
    converged = False

    for level in range(n_levels - 1, -1, -1):
        scale = 2 ** level
        mesh_level = mesh if level == 0 else _scaled_mesh(mesh, 1.0 / scale)

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
            valid_elements=valid_elements,
            # grey-level gain/offset at the native level only: on the smooth
            # coarse levels its columns are nearly collinear with the
            # displacement ones (ill-conditioned; made the coarse solve
            # diverge in tests)
            grey_correction=grey_correction and level == 0,
            pixel_mask=pyr_mask[level])

        U = sol["U"]
        last_residuals = sol["residuals"]
        last_corrections = sol["corrections"]
        converged = sol["converged"]
        if level == 0:
            cov = sol["cov"]
            sigma_u = sol["sigma_u"]
        history.append({"level": level, "residuals": sol["residuals"],
                        "corrections": sol["corrections"],
                        "converged": sol["converged"],
                        "n_backtracks": sol.get("n_backtracks", 0)})
        if level > 0:
            U = U * 2.0              # coarse -> finer next level: 2x the pixels

    return {"U": U, "residuals": last_residuals,
            "corrections": last_corrections,
            "n_iter": len(last_corrections), "converged": converged,
            "stop_reason": sol.get("stop_reason"),
            "n_backtracks": sum(h.get("n_backtracks", 0) for h in history),
            "residual_final": sol.get("residual_final"),
            "elem_rms": sol.get("elem_rms"),
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


def compute_strains_at_nodes(U: np.ndarray, mesh: Q4Mesh,
                             valid_elements: Optional[np.ndarray] = None,
                             return_elements: bool = False
                             ) -> Dict[str, np.ndarray]:
    """Strain fields at the mesh nodes from the nodal displacement.

    Strains are evaluated at the 2x2 Gauss points of each element by
    differentiating the Q4 shape functions with the element's TRUE Jacobian
    (dN/dx = J^-1 dN/dxi, J = dN/dxi . node coords; exact for convected,
    non-rectangular quads, identical to elem_size/2 for rectangles), averaged
    over the element, then averaged onto the nodes over the elements that
    touch them. Elements flagged False in ``valid_elements`` are left out of
    the nodal average (they carry no measurement).

    The shear is tensorial ``eps_xy = 0.5(du_x/dy + du_y/dx)`` and ``eps_vm``
    uses the e_zz closure, matching the local engine. Pixel-based
    (dimensionless) strain; the axis flip to the model frame is handled by
    the field assembler.

    Returns a dict of (n_nodes,) arrays 'eps_xx', 'eps_yy', 'eps_xy',
    'eps_vm' (NaN at nodes with no valid element); with
    ``return_elements=True`` also 'elem_eps_xx', 'elem_eps_yy',
    'elem_eps_xy' (n_elements,): the element means before nodal smoothing.
    """
    gauss_pts, _ = gauss_points_2d(2)
    dNs = [shape_function_derivatives(xi, eta) for xi, eta in gauss_pts]
    ne = mesh.n_elements
    e_xx = np.full(ne, np.nan)
    e_yy = np.full(ne, np.nan)
    e_xy = np.full(ne, np.nan)
    for e in range(ne):
        if valid_elements is not None and not valid_elements[e]:
            continue
        nodes = mesh.connectivity[e]
        coords = mesh.nodes[nodes]
        ux = U[2 * nodes]
        uy = U[2 * nodes + 1]
        sxx = syy = sxy = 0.0
        for dN in dNs:
            J = dN @ coords                       # (2, 2) d(x,y)/d(xi,eta)
            dNxy = np.linalg.solve(J, dN)         # (2, 4) rows: d/dx, d/dy
            sxx += dNxy[0] @ ux
            syy += dNxy[1] @ uy
            sxy += 0.5 * (dNxy[1] @ ux + dNxy[0] @ uy)
        e_xx[e] = sxx / len(dNs)
        e_yy[e] = syy / len(dNs)
        e_xy[e] = sxy / len(dNs)

    acc = {k: np.zeros(mesh.n_nodes) for k in ("eps_xx", "eps_yy", "eps_xy")}
    count = np.zeros(mesh.n_nodes)
    for e in range(ne):
        if not np.isfinite(e_xx[e]):
            continue
        for n in mesh.connectivity[e]:
            acc["eps_xx"][n] += e_xx[e]
            acc["eps_yy"][n] += e_yy[e]
            acc["eps_xy"][n] += e_xy[e]
            count[n] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        exx = np.where(count > 0, acc["eps_xx"] / count, np.nan)
        eyy = np.where(count > 0, acc["eps_yy"] / count, np.nan)
        exy = np.where(count > 0, acc["eps_xy"] / count, np.nan)
    out = {"eps_xx": exx, "eps_yy": eyy, "eps_xy": exy,
           "eps_vm": _equiv(exx, eyy, exy)}
    if return_elements:
        out.update({"elem_eps_xx": e_xx, "elem_eps_yy": e_yy,
                    "elem_eps_xy": e_xy})
    return out


def nodal_displacements(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split the DOF vector into (ux, uy) per node (pixels)."""
    return U[0::2].copy(), U[1::2].copy()


# =============================================================================
# Sequence orchestration  ->  field arrays for the existing viewer
# =============================================================================

def eval_q4_field(mesh: Q4Mesh, U: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear evaluation of the nodal field ``U`` (n_dof,) at points
    ``pts`` (n, 2) of an axis-aligned mesh, inside the Q4 approximation
    space (no external interpolation). Points slightly outside the mesh use
    the nearest border element (linear extrapolation). Returns (n, 2)."""
    if not mesh.axis_aligned:
        raise ValueError("eval_q4_field needs an axis-aligned mesh")
    pts = np.asarray(pts, float).reshape(-1, 2)
    ex, ey = mesh.elem_size_x, mesh.elem_size_y
    # Points farther than one element outside the mesh (or non-finite, e.g.
    # from a diverged solve) get NaN instead of a wild extrapolation.
    ok = (np.all(np.isfinite(pts), axis=1)
          & (pts[:, 0] >= mesh.x0 - ex) & (pts[:, 0] <= mesh.x1 + ex)
          & (pts[:, 1] >= mesh.y0 - ey) & (pts[:, 1] <= mesh.y1 + ey))
    out = np.full((pts.shape[0], 2), np.nan)
    if not ok.any():
        return out
    q = pts[ok]
    je = np.clip(np.floor((q[:, 0] - mesh.x0) / ex).astype(int), 0, mesh.n_elem_x - 1)
    ie = np.clip(np.floor((q[:, 1] - mesh.y0) / ey).astype(int), 0, mesh.n_elem_y - 1)
    e = ie * mesh.n_elem_x + je
    x_min = mesh.x0 + je * ex
    y_min = mesh.y0 + ie * ey
    xi = 2.0 * (q[:, 0] - x_min) / ex - 1.0
    eta = 2.0 * (q[:, 1] - y_min) / ey - 1.0
    N = shape_functions_grid(xi, eta)                       # (4, n)
    nodes = mesh.connectivity[e]                            # (n, 4)
    out[ok, 0] = np.sum(N.T * U[2 * nodes], axis=1)
    out[ok, 1] = np.sum(N.T * U[2 * nodes + 1], axis=1)
    return out


def eulerian_midpoint_displacement(mesh: Q4Mesh, dU: np.ndarray,
                                   n_iter: int = 10) -> np.ndarray:
    """Increment attributed to the FIXED node positions at the pair midpoint.

    ``dU`` is the incremental field on a fixed (non-convected) mesh: dU(x_n)
    is the motion, between frames i and i+1, of the material point located
    at x_n in frame i. The Eulerian increment at x_n and t_{i+1/2} is
    dU(X) with X + dU(X)/2 = x_n (the material point that is at x_n half-way
    through the pair), solved by fixed-point iterations within the Q4 space.
    To first order it equals dU(x_n) - 0.5 grad(dU) . dU. Returns (n_nodes, 2)
    in px (NaN where dU is NaN)."""
    X = mesh.nodes.copy()
    for _ in range(n_iter):
        d = eval_q4_field(mesh, dU, X)
        X = mesh.nodes - 0.5 * np.nan_to_num(d, nan=0.0)
    return eval_q4_field(mesh, dU, X)


def _scaled_mesh(mesh: Q4Mesh, factor: float) -> Q4Mesh:
    """Copy of ``mesh`` with every coordinate multiplied by ``factor`` (same
    topology). Keeps convected (general) geometries general."""
    if mesh.axis_aligned:
        return Q4Mesh(x0=mesh.x0 * factor, y0=mesh.y0 * factor,
                      x1=mesh.x1 * factor, y1=mesh.y1 * factor,
                      n_elem_x=mesh.n_elem_x, n_elem_y=mesh.n_elem_y)
    new = convect_mesh(mesh, np.zeros(mesh.n_dof))
    new.nodes = mesh.nodes * factor
    new.x0, new.y0, new.x1, new.y1 = (mesh.x0 * factor, mesh.y0 * factor,
                                      mesh.x1 * factor, mesh.y1 * factor)
    new.elem_size_x = mesh.elem_size_x * factor
    new.elem_size_y = mesh.elem_size_y * factor
    return new


def _local_init(mesh: Q4Mesh, f_raw, g_raw, search: int) -> np.ndarray:
    """Initial nodal displacement from a local ZNCC correlation at the nodes
    (subset = element size, odd, >= 5; Gaussian sub-pixel). Nodes where it
    fails get the median of the valid ones (0 if none)."""
    from gui.core.dic import correlate_local
    subset = max(5, int(round(min(mesh.elem_size_x, mesh.elem_size_y))) | 1)
    d, ok, _ = correlate_local(f_raw, g_raw, mesh.nodes, subset=subset,
                               search=int(search), zncc_min=0.5,
                               subpixel_method="gauss")
    U0 = np.zeros(mesh.n_dof)
    if ok.any():
        fill = np.nanmedian(d[ok], axis=0)
        d = np.where(ok[:, None], d, fill[None, :])
        U0[0::2] = d[:, 0]
        U0[1::2] = d[:, 1]
    return U0


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
    are (n_pairs, n_nodes); node coordinates x, y are in the model frame (mm),
    at the reference (frame-0) node positions.

    Kinematic description of the per-pair fields (``description`` key):
      - incremental, fixed mesh  -> 'eulerian_fixed_nodes': dU(x_n) is the
        motion over the pair of the material point at x_n in frame i. The
        extra fields Vx_eul/Vy_eul give the velocity attributed to the fixed
        node at the pair midpoint (see ``eulerian_midpoint_displacement``),
        the quantity to compare with a velocity on a fixed spatial grid.
      - incremental, convected   -> 'lagrangian_reference_nodes': values
        labelled by the reference node, measured at its convected position.
      - total                    -> 'lagrangian_frame0': Ux/Uy are the total
        displacement from frame 0; velocities and strain rates come from the
        difference of consecutive totals (they were total / dt before).

    ``progress(i_done, n_pairs)`` / ``on_frame(info)`` as before; ``info``
    also carries 'stop_reason' and 'n_backtracks'.

    Returns a dict with keys: x, y, t, valid (per node: converged AND finite
    displacement), grid (None), fields, units, mesh, description, and the
    mesh / diagnostics needed for export: nodes_px (n_nodes, 2),
    connectivity (n_elem, 4), elem_size, n_iter, converged, stop_reason,
    residual_final (n_pairs,), residual_elem (n_pairs, n_elem), elem_fields
    (dict of (n_pairs, n_elem) element-mean strain rates, unsmoothed).
    """
    n_img = len(frames)
    if n_img < 2:
        raise ValueError("need a sequence of at least 2 frames")
    if params.variant not in ("standard", "hild"):
        raise ValueError("params.variant must be 'standard' or 'hild'")
    convect = bool(getattr(params, "convect", False))
    if convect and not params.incremental:
        raise ValueError("convect requires the incremental pattern (the total "
                         "pattern already measures from frame 0)")

    mesh = build_mesh_on_roi(roi, params.elem_size)
    n_nodes = mesh.n_nodes
    n_elem = mesh.n_elements
    n_pairs = n_img - 1
    dt = 1.0 / fps if fps else 1.0

    # Node coordinates -> model frame (mm). pixel_to_model flips y.
    xy = np.array([pixel_to_model(nx, ny, img_w, img_h, mm_per_px)
                   for nx, ny in mesh.nodes])
    x_mm = xy[:, 0]
    y_mm = xy[:, 1]

    eulerian = params.incremental and not convect
    description = ("eulerian_fixed_nodes" if eulerian else
                   "lagrangian_reference_nodes" if convect else
                   "lagrangian_frame0")
    names = ["Ux", "Uy", "Umag", "Vx", "Vy", "Vmag",
             "Exx_dot", "Eyy_dot", "Exy_dot", "Eeq_dot", "residual"]
    if eulerian:
        names += ["Vx_eul", "Vy_eul"]
    want_sigma = sigma_f is not None
    if want_sigma:
        names += ["sigma_Ux", "sigma_Uy"]
    fields = {k: np.full((n_pairs, n_nodes), np.nan) for k in names}
    elem_fields = {k: np.full((n_pairs, n_elem), np.nan)
                   for k in ("Exx_dot", "Eyy_dot", "Exy_dot")}
    residual_elem = np.full((n_pairs, n_elem), np.nan)
    residual_final = np.full(n_pairs, np.nan)
    n_iter = np.zeros(n_pairs, int)
    converged = np.zeros(n_pairs, bool)
    stop_reason = [""] * n_pairs
    valid = np.zeros((n_pairs, n_nodes), bool)
    t = np.zeros(n_pairs)

    f0_raw = frames[0] if not params.incremental else None
    U_prev: Optional[np.ndarray] = None
    U_total_prev = np.zeros(mesh.n_dof)        # total pattern: previous total
    t_start = time.perf_counter()
    tool_poly = getattr(params, "tool_polygon", None)
    has_tool = tool_poly is not None and len(tool_poly) >= 3
    use_mask = bool(getattr(params, "mask_enabled", False)) or has_tool
    min_std_rel = float(getattr(params, "min_std_rel", 0.0) or 0.0)
    init_search = int(getattr(params, "init_search", 0) or 0)
    U_cumul = np.zeros(mesh.n_dof)             # accumulated nodal displacement

    for i in range(n_pairs):
        t_frame0 = time.perf_counter()
        if params.incremental:
            f_raw = frames[i]
            g_raw = frames[i + 1]
        else:
            f_raw = f0_raw
            g_raw = frames[i + 1]

        # Lagrangian convection (incremental only): solve on the mesh
        # displaced by the cumulated displacement, so nodes follow the material.
        mesh_calc = convect_mesh(mesh, U_cumul) if convect else mesh

        # Per-pair element validity: material coverage (intensity threshold
        # AND outside the tool polygon), texture, and the Jacobian sign.
        valid_elements = None
        if use_mask:
            min_int = (params.mask_min_intensity
                       if getattr(params, "mask_enabled", False) else 0.0)
            mat = material_mask(f_raw, min_int, tool_poly if has_tool else None)
            valid_elements = element_coverage_mask(
                mesh_calc, mat, params.coverage_threshold)
        pix_mask = None
        sat_level = getattr(params, "saturation_level", None)
        if sat_level is not None:
            margin = int(getattr(params, "saturation_margin", SAT_MARGIN_PX))
            sat = (saturation_mask(f_raw, sat_level, margin)
                   | saturation_mask(g_raw, sat_level, margin))
            if sat.any():
                pix_mask = ~sat
                cov = element_coverage_mask(mesh_calc, pix_mask,
                                            params.coverage_threshold)
                valid_elements = cov if valid_elements is None else (
                    valid_elements & cov)
        if min_std_rel > 0:
            tex = element_texture_mask(mesh_calc, f_raw, min_std_rel)
            valid_elements = tex if valid_elements is None else (valid_elements & tex)
        if convect:
            jac_ok = check_jacobian(mesh_calc)
            valid_elements = jac_ok if valid_elements is None else (
                valid_elements & jac_ok)

        if init_search > 0:
            U_init = _local_init(mesh_calc, f_raw, g_raw, init_search)
        elif params.u_init_previous and U_prev is not None:
            U_init = U_prev
        else:
            U_init = None

        # One code path for single- and multi-scale (normalisation on the ROI).
        sol = multiscale_newton_raphson(
            mesh=mesh_calc, f=f_raw, g=g_raw,
            n_levels=max(1, int(getattr(params, "pyramid_levels", 1))),
            sigma=float(getattr(params, "pyramid_sigma", 1.0)), U_init=U_init,
            max_iter=params.max_iter, tol=params.tol, variant=params.variant,
            sigma_f=sigma_f, reg_rel=params.reg_rel,
            valid_elements=valid_elements,
            grey_correction=bool(getattr(params, "grey_correction", True)),
            pixel_mask=pix_mask)
        U = sol["U"]                            # measured displacement
        U_prev = U
        if params.incremental:
            dU_pair = U                         # increment i -> i+1
        else:
            dU_pair = U - U_total_prev          # total(i+1) - total(i)
            U_total_prev = np.nan_to_num(U, nan=0.0)
        if convect:
            U_cumul = U_cumul + np.nan_to_num(U, nan=0.0)

        ux_px, uy_px = nodal_displacements(U)
        dux_px, duy_px = nodal_displacements(dU_pair)
        ux_mm = ux_px * mm_per_px
        uy_mm = -uy_px * mm_per_px              # image y down -> model y up
        vx = dux_px * mm_per_px / dt
        vy = -duy_px * mm_per_px / dt
        fields["Ux"][i] = ux_mm
        fields["Uy"][i] = uy_mm
        fields["Umag"][i] = np.hypot(ux_mm, uy_mm)
        fields["Vx"][i] = vx
        fields["Vy"][i] = vy
        fields["Vmag"][i] = np.hypot(vx, vy)
        if eulerian:
            d_eul = eulerian_midpoint_displacement(mesh, dU_pair)
            fields["Vx_eul"][i] = d_eul[:, 0] * mm_per_px / dt
            fields["Vy_eul"][i] = -d_eul[:, 1] * mm_per_px / dt

        # Strain RATES from the increment of the pair.
        st = compute_strains_at_nodes(dU_pair, mesh_calc, valid_elements,
                                      return_elements=True)
        exx = st["eps_xx"]
        eyy = st["eps_yy"]
        exy = -st["eps_xy"]                     # model-frame sign (see notes)
        fields["Exx_dot"][i] = exx / dt
        fields["Eyy_dot"][i] = eyy / dt
        fields["Exy_dot"][i] = exy / dt
        fields["Eeq_dot"][i] = _equiv(exx, eyy, exy) / dt
        elem_fields["Exx_dot"][i] = st["elem_eps_xx"] / dt
        elem_fields["Eyy_dot"][i] = st["elem_eps_yy"] / dt
        elem_fields["Exy_dot"][i] = -st["elem_eps_xy"] / dt

        # Residual at the returned U: per element, and per node (mean of the
        # adjacent measured elements).
        erms = sol.get("elem_rms")
        if erms is not None:
            residual_elem[i] = erms
            acc = np.zeros(n_nodes)
            cnt = np.zeros(n_nodes)
            for e in range(n_elem):
                if np.isfinite(erms[e]):
                    acc[mesh.connectivity[e]] += erms[e]
                    cnt[mesh.connectivity[e]] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                fields["residual"][i] = np.where(cnt > 0, acc / cnt, np.nan)
        res = sol.get("residual_final")
        residual_final[i] = res if res is not None else np.nan

        if want_sigma and sol["sigma_u"] is not None:
            su = sol["sigma_u"]
            fields["sigma_Ux"][i] = su[0::2] * mm_per_px
            fields["sigma_Uy"][i] = su[1::2] * mm_per_px

        converged[i] = bool(sol["converged"])
        n_iter[i] = int(sol["n_iter"])
        stop_reason[i] = str(sol.get("stop_reason"))
        valid[i] = converged[i] & np.isfinite(ux_px) & np.isfinite(uy_px)
        t[i] = trigger_offset_s + (i + 0.5) * dt

        if progress is not None:
            progress(i + 1, n_pairs)
        if on_frame is not None:
            now = time.perf_counter()
            on_frame({"index": i, "n_pairs": n_pairs,
                      "n_iter": sol["n_iter"],
                      "residual": (float(residual_final[i])
                                   if np.isfinite(residual_final[i]) else None),
                      "converged": bool(sol["converged"]),
                      "stop_reason": stop_reason[i],
                      "n_backtracks": int(sol.get("n_backtracks", 0) or 0),
                      "elapsed_s": now - t_start,
                      "frame_s": now - t_frame0})

    units = {"Ux": "mm", "Uy": "mm", "Umag": "mm",
             "Vx": "mm/s", "Vy": "mm/s", "Vmag": "mm/s",
             "Exx_dot": "1/s", "Eyy_dot": "1/s", "Exy_dot": "1/s", "Eeq_dot": "1/s",
             "residual": "-"}
    if eulerian:
        units["Vx_eul"] = "mm/s"
        units["Vy_eul"] = "mm/s"
    if want_sigma:
        units["sigma_Ux"] = "mm"
        units["sigma_Uy"] = "mm"

    return {"x": x_mm, "y": y_mm, "t": t, "valid": valid, "grid": None,
            "fields": fields, "units": units, "mesh": mesh,
            "description": description,
            "nodes_px": mesh.nodes.copy(),
            "connectivity": mesh.connectivity.copy(),
            "elem_size": float(params.elem_size),
            "n_iter": n_iter, "converged": converged,
            "stop_reason": stop_reason,
            "residual_final": residual_final, "residual_elem": residual_elem,
            "elem_fields": elem_fields}
