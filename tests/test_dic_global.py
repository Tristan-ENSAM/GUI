# -*- coding: utf-8 -*-
"""
Unit tests for the global Q4-DIC engine (gui.core.dic_global).

These tests are self-contained (synthetic speckle + known transforms) and do
not require Qt. They mirror the validated q4dic test suite (shape-function
identities, sub-pixel translation, imposed uniaxial strain, standard/hild
agreement) and add coverage for the sequence orchestration that produces the
viewer field arrays.
"""
import numpy as np
import pytest

from gui.core import dic_global as dg


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _speckle(H=128, W=128, n=320, seed=0):
    """Random Gaussian-blob speckle, normalised to [0, 1]."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    img = np.zeros((H, W), float)
    for _ in range(n):
        cx = rng.integers(5, W - 5)
        cy = rng.integers(5, H - 5)
        r = rng.uniform(2.0, 4.0)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    return (img - img.min()) / (img.max() - img.min())


def _shift_fft(img, ux, uy):
    """Sub-pixel rigid translation by (ux, uy) px via the Fourier shift
    theorem (band-limited, no interpolation bias from the generator)."""
    H, W = img.shape
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    F = np.fft.fft2(img)
    phase = np.exp(-2j * np.pi * (FX * ux + FY * uy))
    return np.real(np.fft.ifft2(F * phase))


@pytest.fixture(scope="module")
def speckle():
    return dg.normalize_image(_speckle(128, 128, seed=42))


@pytest.fixture(scope="module")
def mesh128():
    return dg.Q4Mesh(x0=10, y0=10, x1=118, y1=118, n_elem_x=6, n_elem_y=6)


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------

class TestShapeFunctions:

    def test_partition_of_unity(self):
        for xi, eta in [(-1, -1), (0, 0), (0.5, -0.3), (1, 1)]:
            assert abs(dg.shape_functions(xi, eta).sum() - 1.0) < 1e-12

    def test_kronecker_delta(self):
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for j, (xi, eta) in enumerate(corners):
            N = dg.shape_functions(xi, eta)
            for i in range(4):
                assert abs(N[i] - (1.0 if i == j else 0.0)) < 1e-12

    def test_derivatives_finite_difference(self):
        eps = 1e-6
        xi0, eta0 = 0.3, -0.2
        dN_an = dg.shape_function_derivatives(xi0, eta0)[0]
        dN_fd = (dg.shape_functions(xi0 + eps, eta0)
                 - dg.shape_functions(xi0 - eps, eta0)) / (2 * eps)
        assert np.allclose(dN_fd, dN_an, atol=1e-8)

    def test_gauss_weights_sum(self):
        _, w = dg.gauss_points_2d(2)
        assert abs(w.sum() - 4.0) < 1e-12


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

class TestMesh:

    def test_dimensions(self, mesh128):
        assert mesh128.n_nodes == 49
        assert mesh128.n_elements == 36
        assert mesh128.n_dof == 98

    def test_dof_indices(self, mesh128):
        d = mesh128.dof_indices(0)
        assert d.shape == (8,)
        assert d[0] == 2 * mesh128.connectivity[0][0]
        assert d[1] == 2 * mesh128.connectivity[0][0] + 1

    def test_natural_coords_center(self, mesh128):
        coords = mesh128.nodes[mesh128.connectivity[0]]
        xc = np.array([coords[:, 0].mean()])
        yc = np.array([coords[:, 1].mean()])
        xi, eta = mesh128.physical_to_natural(xc, yc, 0)
        assert abs(xi[0]) < 1e-10 and abs(eta[0]) < 1e-10

    def test_build_mesh_on_roi_trims(self):
        # ROI 100 px wide, elem 24 -> 4 elements -> trimmed to 96 px.
        m = dg.build_mesh_on_roi((10, 20, 100, 100), elem_size=24)
        assert m.n_elem_x == 4 and m.n_elem_y == 4
        assert abs(m.x1 - (10 + 96)) < 1e-9
        assert abs(m.y1 - (20 + 96)) < 1e-9

    def test_build_mesh_on_roi_too_small(self):
        with pytest.raises(ValueError):
            dg.build_mesh_on_roi((0, 0, 10, 10), elem_size=24)

    def test_invalid_mesh_args(self):
        with pytest.raises(ValueError):
            dg.Q4Mesh(0, 0, 10, 10, 0, 1)
        with pytest.raises(ValueError):
            dg.Q4Mesh(10, 0, 0, 10, 1, 1)


# ---------------------------------------------------------------------------
# Interpolator
# ---------------------------------------------------------------------------

class TestInterpolator:

    def test_evaluate_matches_pixels(self, speckle):
        interp = dg.BicubicInterpolator(speckle)
        ys, xs = np.array([20.0, 40.0]), np.array([30.0, 50.0])
        vals = interp.evaluate(xs, ys)
        # At integer positions the spline must reproduce the pixel values.
        assert np.allclose(vals, speckle[ys.astype(int), xs.astype(int)],
                           atol=1e-6)

    def test_gradient_shapes(self, speckle):
        interp = dg.BicubicInterpolator(speckle)
        x = np.array([10.5, 20.3])
        y = np.array([10.2, 30.8])
        dx, dy = interp.gradient(x, y)
        assert dx.shape == x.shape and dy.shape == y.shape


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class TestSolver:

    @pytest.mark.parametrize("variant", ["standard", "hild"])
    def test_translation_subpixel(self, speckle, mesh128, variant):
        ux0, uy0 = 1.7, -0.8
        g = dg.normalize_image(_shift_fft(speckle, ux0, uy0))
        sol = dg.newton_raphson(mesh128, dg.BicubicInterpolator(g), speckle,
                                max_iter=30, tol=1e-4, variant=variant)
        ux, uy = dg.nodal_displacements(sol["U"])
        assert sol["converged"]
        assert abs(ux.mean() - ux0) < 0.05
        assert abs(uy.mean() - uy0) < 0.05

    def test_standard_hild_agree(self, speckle, mesh128):
        g = dg.normalize_image(_shift_fft(speckle, 1.3, 0.6))
        ig = dg.BicubicInterpolator(g)
        s_std = dg.newton_raphson(mesh128, ig, speckle, variant="standard")
        s_hild = dg.newton_raphson(mesh128, ig, speckle, variant="hild")
        assert np.allclose(s_std["U"], s_hild["U"], atol=0.02)

    def test_uniform_strain(self, speckle, mesh128):
        """Imposed uniaxial strain eps_xx = 0.01 about the image centre."""
        from scipy.ndimage import map_coordinates
        eps = 0.01
        H, W = speckle.shape
        cx = W / 2.0
        xx, yy = np.meshgrid(np.arange(W, dtype=float),
                             np.arange(H, dtype=float))
        # g(x) = f(x - eps*(x-cx)); sample f at the pre-image positions.
        x_src = xx - eps * (xx - cx)
        g = map_coordinates(speckle, [yy.ravel(), x_src.ravel()],
                            order=3, mode="nearest").reshape(H, W)
        g = dg.normalize_image(g)
        sol = dg.newton_raphson(mesh128, dg.BicubicInterpolator(g), speckle,
                                max_iter=40, tol=1e-4, variant="standard")
        st = dg.compute_strains_at_nodes(sol["U"], mesh128)
        assert abs(st["eps_xx"].mean() - eps) < 1e-3
        assert abs(st["eps_yy"].mean()) < 1e-3

    def test_sigma_f_scaling(self, speckle, mesh128):
        """Cov = 2 sigma_f^2 [H]^-1: doubling sigma_f doubles sigma_u."""
        g = dg.normalize_image(_shift_fft(speckle, 0.5, 0.5))
        ig = dg.BicubicInterpolator(g)
        s1 = dg.newton_raphson(mesh128, ig, speckle, sigma_f=0.01)
        s2 = dg.newton_raphson(mesh128, ig, speckle, sigma_f=0.02)
        ratio = np.mean(s2["sigma_u"]) / np.mean(s1["sigma_u"])
        assert abs(ratio - 2.0) < 1e-6

    def test_bad_variant_raises(self, speckle, mesh128):
        with pytest.raises(ValueError):
            dg.newton_raphson(mesh128, dg.BicubicInterpolator(speckle),
                              speckle, variant="nope")


# ---------------------------------------------------------------------------
# von Mises convention
# ---------------------------------------------------------------------------

class TestEquiv:

    def test_uniaxial_value(self):
        # exx=0.01, others 0, ezz=-0.01 -> sqrt(2/3*(2*0.01^2)) = 0.0115470
        assert abs(float(dg._equiv(0.01, 0.0, 0.0)) - 0.0115470) < 1e-6

    def test_matches_local_engine(self):
        from gui.core.dic import _equiv as local_equiv
        for exx, eyy, exy in [(0.01, 0.0, 0.0), (0.02, -0.01, 0.005),
                              (-0.03, 0.02, -0.01)]:
            assert abs(float(dg._equiv(exx, eyy, exy))
                       - float(local_equiv(exx, eyy, exy))) < 1e-12


# ---------------------------------------------------------------------------
# Sequence orchestration
# ---------------------------------------------------------------------------

class TestSequence:

    def _make_sequence(self, n=4, ux=0.6, uy=-0.4, seed=1):
        base = _speckle(96, 96, seed=seed)
        frames = [base]
        for k in range(1, n):
            frames.append(_shift_fft(base, ux * k, uy * k))
        return frames

    def test_shapes_and_keys(self):
        frames = self._make_sequence(n=4)
        params = dg.DicGlobalParams(elem_size=24, variant="standard")
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 72, 72), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=96, img_h=96)
        n_pairs = len(frames) - 1
        n_nodes = res["mesh"].n_nodes
        for k in ("Ux", "Uy", "Vx", "Exx_dot", "Eeq_dot", "residual"):
            assert res["fields"][k].shape == (n_pairs, n_nodes)
        # Cumulated strain removed for symmetry with the local engine.
        for k in ("Exx", "Eyy", "Exy", "Eeq"):
            assert k not in res["fields"]
        assert res["x"].shape == (n_nodes,)
        assert res["valid"].shape == (n_pairs, n_nodes)
        assert res["grid"] is None

    def test_incremental_sign_and_magnitude(self):
        # Pure +x image shift of 0.6 px/frame, mm_per_px=0.01 -> Ux ~ +0.006 mm,
        # Uy ~ 0. Model x is +x (cutting dir), so Ux stays positive.
        frames = self._make_sequence(n=3, ux=0.6, uy=0.0)
        params = dg.DicGlobalParams(elem_size=24, variant="standard",
                                    incremental=True)
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 72, 72), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=96, img_h=96)
        ux = np.nanmean(res["fields"]["Ux"][0])
        uy = np.nanmean(res["fields"]["Uy"][0])
        assert abs(ux - 0.006) < 1e-3
        assert abs(uy) < 5e-4

    def test_uy_sign_flip(self):
        # Image shift downward (+uy in pixels) must give NEGATIVE model Uy.
        frames = self._make_sequence(n=2, ux=0.0, uy=0.8)
        params = dg.DicGlobalParams(elem_size=24, variant="standard")
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 72, 72), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=96, img_h=96)
        uy = np.nanmean(res["fields"]["Uy"][0])
        assert uy < 0

    def test_total_vs_incremental(self):
        # Total mode pair i measures displacement since frame 0; for a constant
        # per-frame shift, pair 1 (frames 0->2) is ~2x pair 0 (0->1).
        frames = self._make_sequence(n=3, ux=0.5, uy=0.0)
        p_tot = dg.DicGlobalParams(elem_size=24, incremental=False)
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 72, 72), params=p_tot,
            fps=1000.0, mm_per_px=0.01, img_w=96, img_h=96)
        ux0 = np.nanmean(res["fields"]["Ux"][0])
        ux1 = np.nanmean(res["fields"]["Ux"][1])
        assert ux1 > 1.8 * ux0

    def test_sigma_fields_present(self):
        frames = self._make_sequence(n=2)
        params = dg.DicGlobalParams(elem_size=24)
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 72, 72), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=96, img_h=96, sigma_f=0.01)
        assert "sigma_Ux" in res["fields"]
        assert res["fields"]["sigma_Ux"].shape == res["fields"]["Ux"].shape
        assert res["units"]["sigma_Ux"] == "mm"

    def test_too_few_frames(self):
        with pytest.raises(ValueError):
            dg.compute_dic_global_fields(
                [_speckle(96, 96)], roi=(10, 10, 72, 72),
                params=dg.DicGlobalParams(), fps=1000.0, mm_per_px=0.01,
                img_w=96, img_h=96)

    def test_on_frame_callback(self):
        frames = self._make_sequence(n=4)
        infos = []
        dg.compute_dic_global_fields(
            frames, roi=(10, 10, 72, 72), params=dg.DicGlobalParams(elem_size=24),
            fps=1000.0, mm_per_px=0.01, img_w=96, img_h=96,
            on_frame=lambda d: infos.append(d))
        assert len(infos) == len(frames) - 1
        first = infos[0]
        for key in ("index", "n_pairs", "n_iter", "residual",
                    "converged", "elapsed_s", "frame_s"):
            assert key in first
        # Cumulative time is monotone non-decreasing.
        assert infos[-1]["elapsed_s"] >= infos[0]["elapsed_s"]

    def test_max_iter_tol_respected(self):
        """max_iter caps the iteration count; a tight tol needs more iters than
        a loose one for the same pair."""
        frames = self._make_sequence(n=2, ux=0.7, uy=0.0)
        infos_capped = []
        dg.compute_dic_global_fields(
            frames, roi=(10, 10, 72, 72),
            params=dg.DicGlobalParams(elem_size=24, max_iter=2, tol=1e-8),
            fps=1000.0, mm_per_px=0.01, img_w=96, img_h=96,
            on_frame=lambda d: infos_capped.append(d))
        assert infos_capped[0]["n_iter"] <= 2


# ---------------------------------------------------------------------------
# Multi-scale Gaussian pyramid
# ---------------------------------------------------------------------------

class TestPyramid:

    def test_build_pyramid_shapes(self):
        img = _speckle(160, 160, seed=5)
        pyr = dg.build_gaussian_pyramid(img, n_levels=3, sigma=1.0)
        assert len(pyr) == 3
        # Level 0 native, each next level halved (floor).
        assert pyr[0].shape == (160, 160)
        assert pyr[1].shape == (80, 80)
        assert pyr[2].shape == (40, 40)

    def test_build_pyramid_one_level_is_native(self):
        img = _speckle(96, 96, seed=6)
        pyr = dg.build_gaussian_pyramid(img, n_levels=1)
        assert len(pyr) == 1
        assert pyr[0].shape == img.shape

    def test_build_pyramid_bad_levels(self):
        with pytest.raises(ValueError):
            dg.build_gaussian_pyramid(_speckle(32, 32), n_levels=0)

    def test_single_level_matches_newton(self):
        """n_levels=1 must reproduce the plain newton_raphson result."""
        f = _speckle(128, 128, seed=7)
        g = _shift_fft(f, 1.1, -0.6)
        mesh = dg.Q4Mesh(10, 10, 118, 118, 6, 6)
        s_ms = dg.multiscale_newton_raphson(mesh, f, g, n_levels=1)
        s_nr = dg.newton_raphson(mesh, dg.BicubicInterpolator(dg.normalize_image(g)),
                                 dg.normalize_image(f))
        assert np.allclose(s_ms["U"], s_nr["U"], atol=1e-9)

    def test_pyramid_matches_mono_small_displacement(self):
        f = _speckle(160, 160, seed=8)
        g = _shift_fft(f, 1.3, -0.7)
        mesh = dg.Q4Mesh(20, 20, 140, 140, 5, 5)
        mono = dg.newton_raphson(mesh, dg.BicubicInterpolator(dg.normalize_image(g)),
                                 dg.normalize_image(f))
        pyr = dg.multiscale_newton_raphson(mesh, f, g, n_levels=3, sigma=1.0)
        ux_m, _ = dg.nodal_displacements(mono["U"])
        ux_p, _ = dg.nodal_displacements(pyr["U"])
        assert abs(ux_m.mean() - ux_p.mean()) < 0.02

    def test_pyramid_captures_large_displacement(self):
        """The key benefit: a large shift is recovered by the pyramid. We also
        check it beats the single-scale solve on a displacement large enough
        that the latter cannot converge within max_iter."""
        f = _speckle(160, 160, n=400, seed=9)
        g = _shift_fft(f, 9.0, 0.0)
        mesh = dg.Q4Mesh(20, 20, 140, 140, 5, 5)
        mono = dg.newton_raphson(mesh, dg.BicubicInterpolator(dg.normalize_image(g)),
                                 dg.normalize_image(f), max_iter=20, tol=1e-4)
        pyr = dg.multiscale_newton_raphson(mesh, f, g, n_levels=4, sigma=1.0,
                                           max_iter=20, tol=1e-4)
        ux_p, _ = dg.nodal_displacements(pyr["U"])
        ux_m, _ = dg.nodal_displacements(mono["U"])
        # Pyramid recovers the true 9 px shift...
        assert abs(ux_p.mean() - 9.0) < 0.1
        # ...and does strictly better than the single-scale estimate.
        assert abs(ux_p.mean() - 9.0) < abs(ux_m.mean() - 9.0)
        assert [h["level"] for h in pyr["history"]] == [3, 2, 1, 0]

    def test_pyramid_through_sequence_api(self):
        """compute_dic_global_fields honours pyramid_levels/pyramid_sigma."""
        f = _speckle(120, 120, seed=10)
        frames = [f, _shift_fft(f, 4.0, 0.0)]
        params = dg.DicGlobalParams(elem_size=24, pyramid_levels=3,
                                    pyramid_sigma=1.0)
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 96, 96), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120)
        # 4 px shift at 0.01 mm/px -> ~0.04 mm; mono-scale would miss it.
        assert abs(np.nanmean(res["fields"]["Ux"][0]) - 0.04) < 2e-3

    def test_params_in_json(self):
        p = dg.DicGlobalParams(pyramid_levels=3, pyramid_sigma=1.5)
        d = p.to_json_dict()
        assert d["pyramid_levels"] == 3
        assert d["pyramid_sigma"] == 1.5


# ---------------------------------------------------------------------------
# Material mask & Q4 element exclusion (coverage)
# ---------------------------------------------------------------------------

class TestElementMask:

    def _half_dark(self, H=120, W=120, seed=1):
        """Speckle whose right half is set to zero (out-of-material)."""
        img = _speckle(H, W, n=400, seed=seed).astype(float)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img[:, W // 2:] = 0.0
        return img

    def test_material_mask_intensity(self):
        img = self._half_dark()
        mask = dg.material_mask_intensity(img, min_intensity=20)
        # Left half is material, right half is not.
        assert mask[:, :60].mean() > 0.5
        assert mask[:, 60:].mean() < 0.05

    def test_element_coverage_excludes_dark(self):
        img = self._half_dark()
        mask = dg.material_mask_intensity(img, 20)
        mesh = dg.build_mesh_on_roi((10, 10, 100, 100), 20)
        valid = dg.element_coverage_mask(mesh, mask, 0.5)
        # Some kept, some excluded; not all, not none.
        assert 0 < valid.sum() < mesh.n_elements

    def test_coverage_threshold_monotone(self):
        img = self._half_dark()
        mask = dg.material_mask_intensity(img, 20)
        mesh = dg.build_mesh_on_roi((10, 10, 100, 100), 20)
        low = dg.element_coverage_mask(mesh, mask, 0.2).sum()
        high = dg.element_coverage_mask(mesh, mask, 0.8).sum()
        # A stricter coverage requirement keeps no more elements.
        assert high <= low

    def test_active_nodes(self):
        mesh = dg.build_mesh_on_roi((10, 10, 100, 100), 20)
        valid = np.zeros(mesh.n_elements, bool)
        valid[0] = True                       # keep a single element
        active = dg.active_nodes_from_elements(mesh, valid)
        # Exactly the 4 nodes of element 0 are active.
        assert active.sum() == 4
        assert set(np.where(active)[0]) == set(mesh.connectivity[0])

    def test_orphan_nodes_are_nan(self):
        img = self._half_dark()
        g = np.roll(img, 1, axis=0)
        mesh = dg.build_mesh_on_roi((10, 10, 100, 100), 20)
        mask = dg.material_mask_intensity(img, 20)
        valid = dg.element_coverage_mask(mesh, mask, 0.5)
        active = dg.active_nodes_from_elements(mesh, valid)
        sol = dg.newton_raphson(
            mesh, dg.BicubicInterpolator(dg.normalize_image(g)),
            dg.normalize_image(img), valid_elements=valid)
        ux, uy = dg.nodal_displacements(sol["U"])
        # Orphan nodes -> NaN; active nodes -> finite.
        assert np.all(np.isnan(ux[~active]))
        assert np.all(np.isfinite(ux[active]))

    def test_mask_through_sequence_api(self):
        img = self._half_dark()
        frames = [img, np.roll(img, 1, axis=0)]
        params = dg.DicGlobalParams(elem_size=20, mask_enabled=True,
                                    mask_min_intensity=20,
                                    coverage_threshold=0.5)
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 100, 100), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120)
        ux = res["fields"]["Ux"][0]
        # Right-side (dark) nodes are NaN; left-side material nodes are finite.
        assert np.isnan(ux).any()
        assert np.isfinite(ux).any()

    def test_mask_disabled_no_nan(self):
        """With mask disabled the full mesh solves (no orphan NaN)."""
        f = _speckle(120, 120, seed=2)
        frames = [f, _shift_fft(f, 0.5, 0.0)]
        params = dg.DicGlobalParams(elem_size=20, mask_enabled=False)
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 100, 100), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120)
        assert np.isfinite(res["fields"]["Ux"][0]).all()

    def test_mask_params_in_json(self):
        p = dg.DicGlobalParams(mask_enabled=True, mask_min_intensity=30,
                               coverage_threshold=0.6)
        d = p.to_json_dict()
        assert d["mask_enabled"] is True
        assert d["mask_min_intensity"] == 30
        assert d["coverage_threshold"] == 0.6


# ---------------------------------------------------------------------------
# Lagrangian mesh convection
# ---------------------------------------------------------------------------

class TestConvection:

    def test_convect_mesh_moves_nodes(self):
        mesh = dg.build_mesh_on_roi((20, 20, 100, 100), 20)
        U = np.tile([5.0, -3.0], mesh.n_nodes)
        m2 = dg.convect_mesh(mesh, U)
        assert np.allclose(m2.nodes[:, 0], mesh.nodes[:, 0] + 5.0)
        assert np.allclose(m2.nodes[:, 1], mesh.nodes[:, 1] - 3.0)
        # Topology preserved.
        assert np.array_equal(m2.connectivity, mesh.connectivity)

    def test_convect_mesh_ignores_nan(self):
        mesh = dg.build_mesh_on_roi((20, 20, 60, 60), 20)
        U = np.zeros(mesh.n_dof)
        U[0] = np.nan                          # orphan node x
        m2 = dg.convect_mesh(mesh, U)
        assert np.isfinite(m2.nodes).all()     # NaN treated as 0 shift

    def test_check_jacobian_ok_for_regular_mesh(self):
        mesh = dg.build_mesh_on_roi((20, 20, 100, 100), 20)
        assert dg.check_jacobian(mesh).all()

    def test_check_jacobian_detects_fold(self):
        mesh = dg.build_mesh_on_roi((0, 0, 40, 40), 20)
        # Fold element 0 by swapping two of its nodes' positions drastically:
        # push node 0 far to the right past node 1.
        U = np.zeros(mesh.n_dof)
        n0 = mesh.connectivity[0][0]
        U[2 * n0] = 1000.0                     # huge x shift of one corner
        folded = dg.convect_mesh(mesh, U)
        assert not dg.check_jacobian(folded).all()

    def test_convection_sequence_runs(self):
        f = _speckle(140, 140, seed=1)
        frames = [f, _shift_fft(f, 2.0, 0.0), _shift_fft(f, 4.0, 0.0)]
        params = dg.DicGlobalParams(elem_size=20, incremental=True, convect=True)
        res = dg.compute_dic_global_fields(
            frames, roi=(20, 20, 100, 100), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=140, img_h=140)
        # Rigid 2 px/frame -> ~0.02 mm per pair.
        assert abs(np.nanmean(res["fields"]["Ux"][0]) - 0.02) < 2e-3

    def test_convection_output_coords_are_reference(self):
        """Fields are reported at the reference (frame-0) node positions, so the
        output coordinates match the non-convected run."""
        f = _speckle(140, 140, seed=2)
        frames = [f, _shift_fft(f, 2.0, 0.0)]
        roi = (20, 20, 100, 100)
        r_conv = dg.compute_dic_global_fields(
            frames, roi=roi, params=dg.DicGlobalParams(elem_size=20, convect=True),
            fps=1000.0, mm_per_px=0.01, img_w=140, img_h=140)
        r_plain = dg.compute_dic_global_fields(
            frames, roi=roi, params=dg.DicGlobalParams(elem_size=20, convect=False),
            fps=1000.0, mm_per_px=0.01, img_w=140, img_h=140)
        assert np.allclose(r_conv["x"], r_plain["x"])
        assert np.allclose(r_conv["y"], r_plain["y"])

    def test_convect_in_json(self):
        p = dg.DicGlobalParams(convect=True)
        assert p.to_json_dict()["convect"] is True


# ---------------------------------------------------------------------------
# Tool polygon mask
# ---------------------------------------------------------------------------

class TestToolPolygon:

    def test_polygon_mask_square(self):
        m = dg.polygon_mask((50, 50), [(10, 10), (30, 10), (30, 30), (10, 30)])
        assert m[20, 20]            # inside
        assert not m[5, 5]          # outside
        assert 350 < m.sum() < 500  # ~20x20 area (boundary inclusive)

    def test_polygon_mask_too_few_vertices(self):
        m = dg.polygon_mask((20, 20), [(1, 1), (2, 2)])
        assert not m.any()

    def test_material_mask_excludes_tool(self):
        img = np.full((50, 50), 200.0)
        poly = [(10, 10), (30, 10), (30, 30), (10, 30)]
        mat = dg.material_mask(img, 50, poly)
        assert not mat[20, 20]      # tool interior excluded
        assert mat[5, 5]            # bright material kept elsewhere

    def test_material_mask_no_polygon(self):
        img = np.full((20, 20), 200.0)
        assert dg.material_mask(img, 50, None).all()

    def test_tool_polygon_through_sequence(self):
        f = _speckle(120, 120, seed=1)
        frames = [f, _shift_fft(f, 0.5, 0.0)]
        poly = [(70, 0), (120, 0), (120, 50), (70, 50)]   # top-right corner
        params = dg.DicGlobalParams(elem_size=20, tool_polygon=poly)
        res = dg.compute_dic_global_fields(
            frames, roi=(10, 10, 100, 100), params=params,
            fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120)
        # Corner nodes inside the tool polygon are excluded -> NaN.
        assert np.isnan(res["fields"]["Ux"][0]).any()
        assert np.isfinite(res["fields"]["Ux"][0]).any()

    def test_tool_polygon_in_json(self):
        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        p = dg.DicGlobalParams(tool_polygon=poly)
        assert p.to_json_dict()["tool_polygon"] == poly
