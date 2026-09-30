# -*- coding: utf-8 -*-
"""Validation battery for the local ZNCC engine (gui.core.dic).

Images are generated WITHOUT an interpolation kernel (exact Fourier shift for
translations, per-row / per-column 1-D Fourier shift for u_x(y) shear and
u_y(x) stretch), so a failure is attributable to the DIC engine, not to the
synthetic-image generator.

Tolerances are specification choices (agreed in the DIC audit), not
literature values: TOL_MEAN = 0.005 px systematic error, TOL_SLOPE = 0.02.
Speckle: Gaussian-filtered white noise (filter std RADIUS px), 12-bit range.
"""
import numpy as np
import pytest

from gui.core.dic import (correlate_local, make_grid, compute_dic_fields,
                          point_mask, pixel_centres, DicParams)

cv2 = pytest.importorskip("cv2")

RADIUS = 1.5
SS, SR = 11, 3
TOL_MEAN = 0.005
TOL_SLOPE = 0.02


def speckle(H=160, W=160, radius=RADIUS, seed=0):
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    g = np.exp(-2 * (np.pi * radius) ** 2 * (fx ** 2 + fy ** 2))
    img = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((H, W))) * g))
    img = (img - img.min()) / (img.max() - img.min())
    return img * 4095.0


def fshift(img, ux, uy):
    """Exact band-limited translation: content moves by (+ux, +uy) px."""
    H, W = img.shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img)
                                * np.exp(-2j * np.pi * (fx * ux + fy * uy))))


def shift_rows(img, u_row):
    """Row j translated by u_row[j] along x (exact): u_x = u_row(y)."""
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft(np.fft.fft(img, axis=1)
                               * np.exp(-2j * np.pi * fx * u_row[:, None]), axis=1))


def measure(ref, cur, ss=SS, sr=SR, step=8, method="icgn"):
    H, W = ref.shape
    pts = make_grid((0, 0, W - 1, H - 1), step, margin=ss // 2 + sr + 4)
    d, ok, sc = correlate_local(ref, cur, pts, subset=ss, search=sr,
                                zncc_min=-1.0, subpixel_method=method)
    return pts, d, ok, sc


REF = speckle()


# ---------------------------------------------------------------------------
# Translations and conventions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ux,uy", [(2, 0), (-2, 0), (0, 2), (0, -1), (2, -2)])
def test_integer_translation(ux, uy):
    _, d, ok, _ = measure(REF, fshift(REF, ux, uy))
    assert ok.all()
    assert np.nanmean(d[:, 0]) == pytest.approx(ux, abs=1e-3)
    assert np.nanmean(d[:, 1]) == pytest.approx(uy, abs=1e-3)
    assert np.nanstd(d[:, 0]) < 0.01          # no per-point peak-fit scatter


@pytest.mark.parametrize("u", [0.1, 0.25, 0.4, 0.6, 0.9])
@pytest.mark.parametrize("axis", ["x", "y", "diag"])
def test_subpixel_translation(u, axis):
    ux, uy = {"x": (u, 0.0), "y": (0.0, u), "diag": (u, u)}[axis]
    _, d, _, _ = measure(REF, fshift(REF, ux, uy))
    assert np.nanmean(d[:, 0]) == pytest.approx(ux, abs=TOL_MEAN)
    assert np.nanmean(d[:, 1]) == pytest.approx(uy, abs=TOL_MEAN)


@pytest.mark.parametrize("u", [0.1, 0.3, 1.3])
def test_sign_symmetry(u):
    """Content moving +x gives a positive displacement (image axes), and
    E[u_est(+u)] = -E[u_est(-u)]."""
    _, dp, _, _ = measure(REF, fshift(REF, u, 0))
    _, dm, _, _ = measure(REF, fshift(REF, -u, 0))
    assert np.nanmean(dp[:, 0]) > 0 > np.nanmean(dm[:, 0])
    assert np.nanmean(dp[:, 0]) + np.nanmean(dm[:, 0]) == pytest.approx(0, abs=TOL_MEAN)


def test_linearity_small_increments():
    """U_mes = a U_ref + b over 0.02..0.3 px (regime of the shear test)."""
    us = np.linspace(0.02, 0.3, 8)
    m = [np.nanmean(measure(REF, fshift(REF, u, 0))[1][:, 0]) for u in us]
    a, b = np.polyfit(us, m, 1)
    assert a == pytest.approx(1.0, abs=TOL_SLOPE)
    assert b == pytest.approx(0.0, abs=TOL_MEAN)


def test_peak_locking():
    """Systematic error over a full period [0, 1) below TOL_MEAN."""
    us = np.linspace(0.0, 1.0, 11)[:-1]
    err = [np.nanmean(measure(REF, fshift(REF, u, 0))[1][:, 0]) - u for u in us]
    assert np.max(np.abs(err)) < TOL_MEAN


def test_no_spurious_uy_for_pure_x():
    _, d, _, _ = measure(REF, fshift(REF, 0.3, 0))
    assert abs(np.nanmean(d[:, 1])) < TOL_MEAN
    assert np.nanstd(d[:, 1]) < 0.01


def test_affine_field():
    """u_y = e (x - xc) (exact per-column shift): slope 1 at subset centres."""
    H, W = REF.shape
    e = 0.004
    xc = W / 2.0
    uy_col = e * (np.arange(W) - xc)
    fy = np.fft.fftfreq(H)[:, None]
    cur = np.real(np.fft.ifft(np.fft.fft(REF, axis=0)
                              * np.exp(-2j * np.pi * fy * uy_col[None, :]), axis=0))
    pts, d, _, _ = measure(REF, cur)
    a, _ = np.polyfit(e * (pts[:, 0] - xc), d[:, 1], 1)
    assert a == pytest.approx(1.0, abs=TOL_SLOPE)


@pytest.mark.parametrize("A,w", [(0.2, 3.0), (0.2, 8.0)])
def test_shear_band_slope(A, w):
    H, W = REF.shape
    y = np.arange(H, dtype=float)
    u_row = A * 0.5 * (1 + np.tanh((y - H / 2) / w))
    pts, d, _, _ = measure(REF, shift_rows(REF, u_row), step=6)
    half = SS // 2
    iy = pts[:, 1].astype(int)
    u_sub = np.array([u_row[j - half:j + half + 1].mean() for j in iy])
    a, _ = np.polyfit(u_sub, d[:, 0], 1)
    assert a == pytest.approx(1.0, abs=TOL_SLOPE)
    assert np.nanmean(np.abs(d[:, 1])) < 0.01


def test_legacy_parabola_still_available():
    """The legacy estimator stays selectable and shows the peak-locking bias
    of the audit: ~0.885 * u for a fine (1 px) speckle."""
    fine = speckle(radius=1.0, seed=1)
    _, d, _, _ = measure(fine, fshift(fine, 0.1, 0), method="parabola")
    assert np.nanmean(d[:, 0]) < 0.095
    _, d, _, _ = measure(fine, fshift(fine, 0.1, 0), method="icgn")
    assert np.nanmean(d[:, 0]) == pytest.approx(0.1, abs=TOL_MEAN)


def test_gauss_estimator_small_bias():
    _, d, _, _ = measure(REF, fshift(REF, 0.1, 0), method="gauss")
    assert np.nanmean(d[:, 0]) == pytest.approx(0.1, abs=0.01)


# ---------------------------------------------------------------------------
# Guards (F2 search border, F3 even subset, F4 textureless)
# ---------------------------------------------------------------------------
def test_peak_on_search_border_is_invalid():
    """sr=1: a 0.7 px shift puts the integer peak on the border of the
    search range -> invalid (not silently returned as 1 px)."""
    pts = make_grid((0, 0, 159, 159), 8, margin=12)
    d, ok, sc, info = correlate_local(REF, fshift(REF, 0.7, 0), pts,
                                      subset=11, search=1, zncc_min=0.5,
                                      return_info=True)
    assert info["edge"].all() and not ok.any()
    # with a sufficient search range the same shift is measured
    d, ok, _ = correlate_local(REF, fshift(REF, 0.7, 0), pts, subset=11,
                               search=3, zncc_min=0.5)
    assert ok.all()
    assert np.nanmean(d[:, 0]) == pytest.approx(0.7, abs=TOL_MEAN)


def test_flat_region_not_valid():
    a = REF.copy(); a[:, 80:] = 4095.0           # saturated in both images
    pts = np.array([[120.0, 80.0], [40.0, 80.0]])
    d, ok, sc, info = correlate_local(a, a, pts, subset=SS, search=SR,
                                      zncc_min=0.5, return_info=True)
    assert info["flat"][0] and not ok[0]
    assert ok[1] and not info["flat"][1]


@pytest.mark.parametrize("ss", [10, 4])
def test_even_or_small_subset_rejected(ss):
    with pytest.raises(ValueError):
        correlate_local(REF, REF, np.array([[80.0, 80.0]]), subset=ss, search=2)


# ---------------------------------------------------------------------------
# Coordinates (F5 grid snapping, F6 model origin) and mask window (F7)
# ---------------------------------------------------------------------------
def test_grid_is_integer_for_fractional_roi():
    pts = make_grid((40.37, 10.5, 100.0, 40.0), 5, margin=8)
    assert np.array_equal(pts, np.round(pts))
    assert set(np.diff(np.unique(pts[:, 0]))) == {5.0}
    assert pts[:, 0].min() >= 40.37 + 8 and pts[:, 0].max() <= 140.37 - 8


def test_pixel_centres_round_half_up():
    c = pixel_centres(np.array([[40.5, 41.5], [0.49, 2.5]]))
    assert c.tolist() == [[41.0, 42.0], [0.0, 3.0]]


def test_exported_coordinates_are_the_correlated_centres():
    """compute_dic_fields exports the integer centres, in the model frame
    with the origin at the geometric image centre ((W-1)/2, (H-1)/2)."""
    H, W = REF.shape
    # 79.6 -> 80, 79.4 -> 79 ; second point 3 columns / 2 rows away
    pts = np.array([[79.6, 79.4], [83.0, 81.0]])
    res = compute_dic_fields([REF, fshift(REF, 0.2, 0)], pts,
                             DicParams(subset=11, search=3), fps=1.0,
                             mm_per_px=1.0, img_w=W, img_h=H)
    assert res["x"][0] == pytest.approx(80 - (W - 1) / 2.0)
    assert res["y"][0] == pytest.approx((H - 1) / 2.0 - 79)
    assert res["fields"]["Ux"][0, 0] == pytest.approx(0.2, abs=0.01)


def test_mask_window_is_the_subset():
    """point_mask(win=subset) tests the subset's pixels: a subset whose
    border half lies on a dark background is dropped (it was kept when the
    window was subset // 2)."""
    img = np.full((100, 100), 3000.0)
    img[:, 55:] = 0.0                            # dark right part
    pt = np.array([[50.0, 50.0]])                # subset 21 -> cols 40..60
    assert not point_mask(img, pt, win=21, min_intensity=2500)[0]
    assert point_mask(img, pt, win=21 // 2, min_intensity=2500)[0]
