# -*- coding: utf-8 -*-
"""Validation battery of the global Q4 engine (audit A1-A12, tests A-M).

Synthetic images are generated WITHOUT an interpolation kernel where
possible (exact Fourier shifts; per-row / per-column 1-D Fourier shifts for
u_x(y) / u_y(x); an analytic inverse + 5th-order B-spline for the stretch),
so failures are attributable to the engine.

Tolerance origins (see the audit report):
  (N) numerical identity (machine precision / truncation order);
  (R) regression: value measured on this code with a documented margin;
  (S) statistical, from the predicted or empirical scatter.
Specification tolerances for the inverse identification are NOT set here.
"""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from gui.core import dic_global as dg
from gui.core.exp_field_io import save_dic_field, load_dic_field


def speckle(H=128, W=128, radius=1.5, seed=0):
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    g = np.exp(-2 * (np.pi * radius) ** 2 * (fx ** 2 + fy ** 2))
    img = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((H, W))) * g))
    img = (img - img.min()) / (img.max() - img.min())
    return img * 4095.0


def fshift(img, ux, uy):
    H, W = img.shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img)
                                * np.exp(-2j * np.pi * (fx * ux + fy * uy))))


def shift_rows(img, u_row):
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft(np.fft.fft(img, axis=1)
                               * np.exp(-2j * np.pi * fx * u_row[:, None]), axis=1))


def solve(f, g, roi, elem, **kw):
    mesh = dg.build_mesh_on_roi(roi, elem)
    sol = dg.multiscale_newton_raphson(mesh, f, g, n_levels=kw.pop("levels", 1), **kw)
    return mesh, sol


def run_seq(frames, roi, W, H, **kw):
    return dg.compute_dic_global_fields(frames, roi, dg.DicGlobalParams(**kw),
                                        fps=1.0, mm_per_px=1.0, img_w=W, img_h=H)


REF = speckle()
ROI = (16, 16, 96, 96)


# --- A: zero displacement -------------------------------------------------
def test_A_zero_displacement_exact():
    mesh, s = solve(REF, REF.copy(), ROI, 24)
    assert s["converged"]
    assert np.abs(s["U"]).max() < 1e-10                      # (N)


# --- B: integer translation, signs and axes --------------------------------
@pytest.mark.parametrize("ux,uy", [(2, 0), (-2, 0), (0, 2), (0, -1), (2, -2)])
def test_B_integer_translation(ux, uy):
    mesh, s = solve(REF, fshift(REF, ux, uy), ROI, 24)
    u, v = dg.nodal_displacements(s["U"])
    assert np.abs(u - ux).max() < 1e-3 and np.abs(v - uy).max() < 1e-3   # (R: 0.0000)


def test_B_model_frame_signs():
    """+x image motion -> +Ux model; +y image (down) motion -> -Uy model."""
    r = run_seq([REF, fshift(REF, 0.5, 0.5)], ROI, 128, 128, elem_size=24)
    assert np.nanmean(r["fields"]["Ux"][0]) == pytest.approx(0.5, abs=2e-3)
    assert np.nanmean(r["fields"]["Uy"][0]) == pytest.approx(-0.5, abs=2e-3)


# --- C: sub-pixel translation --------------------------------------------
@pytest.mark.parametrize("u", [0.1, 0.25, 0.5, -0.3, 0.75])
def test_C_subpixel_translation(u):
    ref = speckle(radius=2.0, seed=3)
    mesh, s = solve(ref, fshift(ref, u, -0.5 * u), ROI, 24)
    ux, uy = dg.nodal_displacements(s["U"])
    assert abs(ux.mean() - u) < 5e-4 and abs(uy.mean() + 0.5 * u) < 5e-4  # (R: <=2e-4)


# --- D: affine field ------------------------------------------------------
def test_D_affine_field_reproduced():
    H, W = REF.shape
    e = 0.004; xc = W / 2
    fy = np.fft.fftfreq(H)[:, None]
    cur = np.real(np.fft.ifft(np.fft.fft(REF, axis=0)
                              * np.exp(-2j * np.pi * fy * (e * (np.arange(W) - xc))[None, :]), axis=0))
    mesh, s = solve(REF, cur, ROI, 16)
    ux, uy = dg.nodal_displacements(s["U"])
    assert np.abs(uy - e * (mesh.nodes[:, 0] - xc)).max() < 3e-3          # (R: 1.6e-3)
    st = dg.compute_strains_at_nodes(s["U"], mesh)
    assert np.nanmean(st["eps_xy"]) == pytest.approx(e / 2, rel=0.01)      # (R: 0.25 %)


def test_D2_convected_mesh_partition_and_mapping():
    mesh = dg.build_mesh_on_roi(ROI, 24)
    U = np.zeros(mesh.n_dof)
    U[0::2] = 0.3 * (mesh.nodes[:, 1] - mesh.nodes[:, 1].min())            # simple shear
    mc = dg.convect_mesh(mesh, U)
    pts = np.concatenate([np.column_stack(mc.get_pixel_points_in_element(e))
                          for e in range(mc.n_elements)])
    _, counts = np.unique(pts, axis=0, return_counts=True)
    assert counts.max() == 1                                               # (N)
    for e in range(mc.n_elements):
        x, y, xi, eta = mc._owned_pixels(e)
        N = dg.shape_functions_grid(xi, eta)
        c = mc.nodes[mc.connectivity[e]]
        assert np.allclose(N.T @ c[:, 0], x, atol=1e-8)                   # (N)
        assert np.allclose(N.T @ c[:, 1], y, atol=1e-8)


def test_D3_strains_true_jacobian():
    """Affine displacement on a convected (sheared) mesh: exact strain."""
    mesh = dg.convect_mesh(dg.build_mesh_on_roi(ROI, 24),
                           np.tile([0.0, 0.0], dg.build_mesh_on_roi(ROI, 24).n_nodes))
    U0 = np.zeros(mesh.n_dof)
    U0[0::2] = 3.0 * (mesh.nodes[:, 1] - 64) / 96          # shear the geometry
    mc = dg.convect_mesh(dg.build_mesh_on_roi(ROI, 24), U0)
    A = np.array([[0.01, 0.004], [-0.002, 0.006]])          # du = A x
    U = np.zeros(mc.n_dof)
    U[0::2] = mc.nodes @ A[0]
    U[1::2] = mc.nodes @ A[1]
    st = dg.compute_strains_at_nodes(U, mc)
    assert np.allclose(st["eps_xx"], A[0, 0]) and np.allclose(st["eps_yy"], A[1, 1])
    assert np.allclose(st["eps_xy"], 0.5 * (A[0, 1] + A[1, 0]))           # (N)


def test_D4_jacobian_checked_at_corners():
    mesh = dg.build_mesh_on_roi(ROI, 24)
    mc = dg.convect_mesh(mesh, np.zeros(mesh.n_dof))
    n = mc.connectivity[0][2]                   # node 3 of element 0
    mc.nodes[n] = mc.nodes[mc.connectivity[0][0]] + [2.0, 2.0]   # corner fold
    ok = dg.check_jacobian(mc)
    assert not ok[0]


# --- E/F: shear band and mesh convergence ---------------------------------
def test_E_shear_band_plateaus_and_gradient():
    H, W = 256, 128
    ref = speckle(H, W, radius=1.5, seed=7)
    yc, A, w = 128.3, 1.0, 4.0
    u_row = A * 0.5 * (1 + np.tanh((np.arange(H) - yc) / w))
    res = {}
    for elem in (4, 16):
        mesh, s = solve(ref, shift_rows(ref, u_row), (16, 16, 96, 224), elem, max_iter=60)
        ux = s["U"][0::2]
        ys = np.unique(mesh.nodes[:, 1])
        um = np.array([ux[mesh.nodes[:, 1] == v].mean() for v in ys])
        jump = um[ys > yc + 40].mean() - um[ys < yc - 40].mean()
        gmax = np.max(np.gradient(um, ys))
        res[elem] = (jump, gmax / (A / (2 * w)))
        assert jump == pytest.approx(A, rel=0.005)                          # (R: 1.000-1.002)
    # gradient restitution improves with smaller elements (R: 80 % vs 30 %)
    assert res[4][1] > 0.7 and res[16][1] < 0.45 and res[4][1] > res[16][1]


# --- G: initialisation / basin --------------------------------------------
def test_G_local_init_extends_basin_and_failure_is_flagged():
    H = W = 192
    ref = speckle(H, W, radius=1.5, seed=9)
    roi = (40, 40, 112, 112)
    frames = [ref, fshift(ref, 6.0, 1.8)]
    r0 = run_seq(frames, roi, W, H, elem_size=16)
    assert not r0["converged"][0]                          # flagged, not silent
    assert not r0["valid"][0].any()
    r1 = run_seq(frames, roi, W, H, elem_size=16, init_search=10)
    assert r1["converged"][0]
    assert np.nanmean(r1["fields"]["Ux"][0]) == pytest.approx(6.0, abs=2e-3)   # (R: 0.000)


def test_G_pyramid_depth_guard():
    mesh = dg.build_mesh_on_roi(ROI, 8)
    with pytest.raises(ValueError):
        dg.multiscale_newton_raphson(mesh, REF, REF, n_levels=4)   # 8/8 = 1 px


# --- H: temporal consistency ------------------------------------------------
def test_H1_velocity_same_in_incremental_and_total():
    frames = [fshift(REF, 0.4 * k, 0) for k in range(4)]
    vi = run_seq(frames, ROI, 128, 128, elem_size=24, incremental=True)["fields"]["Vx"]
    rt = run_seq(frames, ROI, 128, 128, elem_size=24, incremental=False)
    vt = rt["fields"]["Vx"]
    assert np.allclose(np.nanmean(vi, axis=1), 0.4, atol=3e-3)           # (R)
    assert np.allclose(np.nanmean(vt, axis=1), 0.4, atol=3e-3)
    # total displacement = sum of increments
    assert np.nanmean(rt["fields"]["Ux"][-1]) == pytest.approx(1.2, abs=3e-3)


def test_H2_convect_requires_incremental():
    with pytest.raises(ValueError):
        run_seq([REF, REF], ROI, 128, 128, elem_size=24, incremental=False,
                convect=True)


def test_H3_eulerian_vs_lagrangian_description():
    """Stretch u_x(X) = a_k (X - xc): the fixed mesh measures the Eulerian
    increment, the convected mesh the Lagrangian one."""
    H = W = 160
    ref = speckle(H, W, radius=2.0, seed=4)
    xc = (W - 1) / 2.0
    a = [0.0, 0.01, 0.02]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    frames = [map_coordinates(ref, [yy, xc + (xx - xc) / (1 + ak)], order=5,
                              mode="mirror") for ak in a]
    roi = (24, 24, 112, 112)
    re_ = run_seq(frames, roi, W, H, elem_size=16)
    rl = run_seq(frames, roi, W, H, elem_size=16, convect=True)
    Xn = re_["nodes_px"][:, 0]
    eul = (a[2] - a[1]) * (Xn - xc) / (1 + a[1])
    lag = (a[2] - a[1]) * (Xn - xc)
    assert re_["description"] == "eulerian_fixed_nodes"
    assert rl["description"] == "lagrangian_reference_nodes"
    assert np.nanmax(np.abs(re_["fields"]["Ux"][1] - eul)) < 2e-3        # (R: 5e-4)
    assert np.nanmax(np.abs(rl["fields"]["Ux"][1] - lag)) < 2e-3
    assert np.nanmax(np.abs(re_["fields"]["Ux"][1] - lag)) > 4e-3        # distinguishable


def test_H4_eulerian_midpoint_velocity():
    """On a fixed mesh, dU(X) with X + dU(X)/2 = x_n: for u = a (x - xc)
    the midpoint increment is a (x_n - xc) / (1 + a/2)."""
    mesh = dg.build_mesh_on_roi(ROI, 16)
    a, xc = 0.05, 64.0
    dU = np.zeros(mesh.n_dof)
    dU[0::2] = a * (mesh.nodes[:, 0] - xc)
    d = dg.eulerian_midpoint_displacement(mesh, dU)
    assert np.allclose(d[:, 0], a * (mesh.nodes[:, 0] - xc) / (1 + a / 2), atol=1e-9)   # (N)


# --- I: gradient / Jacobian ----------------------------------------------
def test_I_gradient_matches_finite_differences():
    """C = sum r^2 (grey correction off): dC/dU = -2 h, h = G^T r. Checked
    on the operator used by the solver."""
    import scipy.sparse as sp
    mesh = dg.build_mesh_on_roi(ROI, 32)
    f = dg.normalize_image(REF)
    ig = dg.BicubicInterpolator(dg.normalize_image(fshift(REF, 0.3, -0.2)))
    op = mesh.pixel_operator()
    fv = f[op["y"].astype(int), op["x"].astype(int)]

    def parts(U):
        xd = op["x"] + op["Bx"] @ U; yd = op["y"] + op["By"] @ U
        r = fv - ig.evaluate(xd, yd)
        gx, gy = ig.gradient(xd, yd)
        G = sp.diags(gx) @ op["Bx"] + sp.diags(gy) @ op["By"]
        return r, G

    U0 = np.full(mesh.n_dof, 0.1)
    r, G = parts(U0)
    h = G.T @ r
    for i in (0, 1, mesh.n_dof // 2, mesh.n_dof - 1):
        e = np.zeros(mesh.n_dof); e[i] = 1e-4
        rp, _ = parts(U0 + e); rm, _ = parts(U0 - e)
        fd = (rp @ rp - rm @ rm) / (2e-4)
        assert fd == pytest.approx(-2 * h[i], rel=1e-6)                   # (N)


def test_I2_each_pixel_counted_once():
    for elem in (16, 24, 32):
        mesh = dg.build_mesh_on_roi((10, 10, 96, 96), elem)
        pts = np.concatenate([np.column_stack(mesh.get_pixel_points_in_element(e))
                              for e in range(mesh.n_elements)])
        _, c = np.unique(pts, axis=0, return_counts=True)
        assert c.max() == 1                                                # (N)
        span = mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0) + 1
        assert len(pts) == int(span[0] * span[1])


def test_I3_vectorised_operator_matches_element_assembly():
    """The sparse G^T G / G^T r assembly equals the element-by-element one."""
    import scipy.sparse as sp
    mesh = dg.build_mesh_on_roi(ROI, 24)
    f = dg.normalize_image(REF)
    ig = dg.BicubicInterpolator(dg.normalize_image(fshift(REF, 0.3, 0.1)))
    U = np.full(mesh.n_dof, 0.2)
    Hd, hd, _ = dg.assemble_global(mesh, ig, U, f)
    op = mesh.pixel_operator()
    xd = op["x"] + op["Bx"] @ U; yd = op["y"] + op["By"] @ U
    r = f[op["y"].astype(int), op["x"].astype(int)] - ig.evaluate(xd, yd)
    gx, gy = ig.gradient(xd, yd)
    G = sp.diags(gx) @ op["Bx"] + sp.diags(gy) @ op["By"]
    assert np.allclose((G.T @ G).toarray(), Hd, rtol=1e-10, atol=1e-8)   # (N)
    assert np.allclose(G.T @ r, hd, rtol=1e-10, atol=1e-8)


# --- J: optimiser ---------------------------------------------------------
def test_J_cost_decreases_and_converges():
    mesh = dg.build_mesh_on_roi(ROI, 16)
    f = dg.normalize_image(REF)
    s = dg.newton_raphson(mesh, dg.BicubicInterpolator(
        dg.normalize_image(fshift(REF, 0.8, 0.3))), f, max_iter=20)
    assert s["converged"] and s["n_iter"] <= 10
    assert all(b <= a * (1 + 1e-12) for a, b in zip(s["residuals"], s["residuals"][1:]))
    assert s["residual_final"] <= s["residuals"][-1] + 1e-12
    assert np.isfinite(s["elem_rms"]).all()


# --- K: robustness --------------------------------------------------------
def test_K1_content_change_outside_roi_has_no_effect():
    g = fshift(REF, 0.3, 0)
    g2 = g.copy(); g2[:, :10] = 4095.0; g2[:10, :] = 4095.0    # outside ROI
    _, s1 = solve(REF, g, ROI, 24)
    _, s2 = solve(REF, g2, ROI, 24)
    # Not exactly zero: the cubic interpolating spline of g is not local
    # (its coefficients depend on the whole image). (R) measured 5e-6 px;
    # 9.4e-3 px with the former whole-image normalisation.
    assert np.abs(s1["U"] - s2["U"]).max() < 1e-4


def test_K1b_illumination_change_absorbed():
    """g = 1.2 g0 + 300 (brightness/contrast change): grey correction."""
    g = 1.2 * fshift(REF, 0.3, 0) + 300.0
    _, s = solve(REF, g, ROI, 24)
    assert np.nanmean(s["U"][0::2]) == pytest.approx(0.3, abs=2e-3)       # (R)
    a, b = s["grey_ab"]
    assert abs(a) > 0.05                       # the gain was estimated


def test_K2_textureless_elements_excluded():
    img = REF.copy(); img[:, 64:] = 4095.0
    g = fshift(REF, 0.3, 0); g[:, 64:] = 4095.0
    r = run_seq([img, g], ROI, 128, 128, elem_size=24)
    nodes = r["nodes_px"]
    flat = nodes[:, 0] > 64 + 1e-9
    assert r["converged"][0]
    assert not r["valid"][0][flat].any()
    assert np.isnan(r["fields"]["Ux"][0][nodes[:, 0] > 88]).all()


def test_K3_nan_nodes_are_never_valid():
    img = REF.copy(); img[:, 70:] = 0.0
    p = dict(elem_size=24, mask_enabled=True, mask_min_intensity=0.5 * REF.mean())
    r = run_seq([img, fshift(img, 0.3, 0)], ROI, 128, 128, **p)
    nan = np.isnan(r["fields"]["Ux"][0])
    assert nan.any() and not (nan & r["valid"][0]).any()


# --- M: analytic uncertainty -----------------------------------------------
def test_M_predicted_scatter_matches_empirical():
    rng = np.random.default_rng(3)
    H = W = 128; roi = (16, 16, 96, 96); sig = 20.0
    ref = speckle(H, W, radius=1.5, seed=3)
    mesh = dg.build_mesh_on_roi(roi, 24)
    Us = []
    for _ in range(20):
        f = ref + rng.normal(0, sig, ref.shape)
        g = fshift(ref, 0.3, 0) + rng.normal(0, sig, ref.shape)
        mu, sd = dg.roi_stats(f, dg.roi_region(mesh, f.shape))
        s = dg.newton_raphson(mesh, dg.BicubicInterpolator(dg.normalize_with(g, mu, sd)),
                              dg.normalize_with(f, mu, sd), sigma_f=sig / sd)
        Us.append(s["U"]); pred = s["sigma_u"]
    emp = np.array(Us).std(axis=0, ddof=1)
    # (R) measured ratio 1.03-1.06 after the pixel-partition fix (1.17-1.22
    # before); margin for 20 draws
    assert 0.85 < emp.mean() / pred.mean() < 1.15


# --- export (C8) ----------------------------------------------------------
def test_export_single_npz_with_mesh(tmp_path):
    r = run_seq([REF, fshift(REF, 0.3, 0), fshift(REF, 0.6, 0)], ROI, 128, 128,
                elem_size=24)
    mesh = {"nodes_px": r["nodes_px"], "connectivity": r["connectivity"],
            "n_iter": r["n_iter"], "converged": r["converged"],
            "residual_elem": r["residual_elem"]}
    f = r["fields"]
    p = save_dic_field(tmp_path / "q4_dic.npz", r["x"], r["y"], r["t"],
                       f["Vx"], f["Vy"], f["Vmag"], r["valid"],
                       meta={"description": r["description"]},
                       extra={k: v for k, v in f.items() if k not in ("Vx", "Vy", "Vmag")},
                       mesh=mesh)
    d = load_dic_field(p)
    assert d["mesh_connectivity"].dtype.kind == "i"
    assert d["mesh_connectivity"].shape == (r["connectivity"].shape[0], 4)
    assert np.array_equal(d["mesh_nodes_px"], r["nodes_px"])
    assert d["mesh_residual_elem"].shape == (2, r["connectivity"].shape[0])
    assert d["meta"]["description"] == "eulerian_fixed_nodes"
    assert "mesh_connectivity" not in d["meta"]["fields"]
    assert "Vx_eul" in d


# --- saturation mask ------------------------------------------------------
def _stationary_saturated_half():
    img = REF.copy(); img[:, 64:] = 4095.0
    g = fshift(REF, 0.3, 0); g[:, 64:] = 4095.0
    return img, g


@pytest.mark.parametrize("levels", [1, 2])
def test_S1_saturation_mask_removes_edge_bias(levels):
    """A saturated area that does not move while the texture does: without
    the mask the neighbouring textured nodes are biased (measured 0.084 px);
    with the mask (level 4095, margin 3) they match the unsaturated case."""
    img, g = _stationary_saturated_half()
    off = run_seq([img, g], ROI, 128, 128, elem_size=24, pyramid_levels=levels)
    on = run_seq([img, g], ROI, 128, 128, elem_size=24, pyramid_levels=levels,
                 saturation_level=4095.0)
    tex = on["nodes_px"][:, 0] <= 40
    err_off = np.nanmax(np.abs(off["fields"]["Ux"][0][tex] - 0.3))
    err_on = np.nanmax(np.abs(on["fields"]["Ux"][0][tex] - 0.3))
    assert err_off > 0.05                        # the bias exists without mask
    assert err_on < 0.005                        # (R: 0.0027; 0.0025 unsaturated)
    assert on["converged"][0] and on["valid"][0][tex].all()


def test_S2_saturation_mask_helpers():
    img = np.zeros((20, 20)); img[10, 10] = 4095.0
    m = dg.saturation_mask(img, 4095.0, margin=2)
    assert m[10, 10] and m[10, 12] and m[12, 10] and not m[10, 13]
    assert not dg.saturation_mask(img, 4095.0, margin=0)[10, 11]
    # statistics without the saturated pixels
    a = REF.copy(); a[:, 64:] = 4095.0
    region = (slice(16, 112), slice(16, 112))
    mu_all, _ = dg.roi_stats(a, region)
    mu_ok, _ = dg.roi_stats(a, region, ~dg.saturation_mask(a, 4095.0, 0))
    assert mu_ok == pytest.approx(REF[16:112, 16:64].mean(), rel=1e-9)
    assert mu_all > mu_ok


def test_S3_pixel_mask_drops_pixels_and_empty_elements():
    mesh = dg.build_mesh_on_roi(ROI, 24)
    f = dg.normalize_image(REF)
    g = dg.normalize_image(fshift(REF, 0.3, 0))
    pm = np.ones(REF.shape, bool); pm[:, 64:] = False       # right half unusable
    s = dg.newton_raphson(mesh, dg.BicubicInterpolator(g), f, pixel_mask=pm)
    ux = s["U"][0::2]
    right = mesh.nodes[:, 0] > 64
    assert np.isnan(ux[right]).all()                          # orphans -> NaN
    assert np.nanmax(np.abs(ux[~right] - 0.3)) < 5e-3
    assert np.isnan(s["elem_rms"][[e for e in range(mesh.n_elements)
                                   if mesh.nodes[mesh.connectivity[e]][:, 0].min() >= 64]]).all()


def test_S4_scattered_saturation_keeps_nodes():
    """Isolated saturated speckle highlights (2 % of the pixels): the mask
    only drops those pixels, no node is lost and the error does not grow."""
    H = W = 192
    ref = speckle(H, W, radius=1.5, seed=1)
    clip = np.percentile(ref, 98)
    f = np.minimum(ref, clip); g = np.minimum(fshift(ref, 0.3, 0.1), clip)
    r = run_seq([f, g], (16, 16, 160, 160), W, H, elem_size=16,
                saturation_level=float(clip))
    v = r["valid"][0]
    assert v.mean() > 0.99
    assert np.nanmean(np.abs(r["fields"]["Ux"][0][v] - 0.3)) < 2e-3      # (R: 0.0008)
