# -*- coding: utf-8 -*-
"""Local DIC: prediction from the previous pair (DicParams.predict) and
filling of rejected points from their neighbours (DicParams.fill_invalid)."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from gui.core import dic
from tests.test_dic_tab_global import _speckle, _shift


def _seq(shifts, H=120, W=120, flat=None, seed=1):
    base = _speckle(H=H, W=W, n=350, seed=seed).astype(float)
    if flat is not None:
        y0, y1, x0, x1 = flat
        base[y0:y1, x0:x1] = 128.0
    pos = np.cumsum([0] + list(shifts))
    return [_shift(base, p, 0).astype(np.uint8) for p in pos]


def _run(frames, predict, fill, search=6, W=120, H=120):
    pts = dic.make_grid((0, 0, W, H), 10, margin=10 + search + 2)
    p = dic.DicParams(subset=21, step=10, search=search, predict=predict,
                      fill_invalid=fill)
    return dic.compute_dic_fields(frames, pts, p, fps=1.0, mm_per_px=1.0,
                                  img_w=W, img_h=H)


class TestPredict:
    def test_prediction_tracks_motion_beyond_search(self):
        shifts = [3, 6, 9]                    # 6 and 9 px > search - 1
        frames = _seq(shifts)
        plain = _run(frames, predict=False, fill=False)
        pred = _run(frames, predict=True, fill=False)
        assert plain["valid"][2].mean() < 0.05
        assert pred["valid"].mean(axis=1).min() > 0.95
        for i, s in enumerate(shifts):
            ux = pred["fields"]["Ux"][i][pred["valid"][i]]
            assert np.median(np.abs(ux - s)) < 0.05

    def test_wrong_prediction_falls_back_to_zero_search(self):
        # Motion stops: the prediction (8 px) is wrong by more than search,
        # the retry around zero must recover every point.
        shifts = [5, 0]
        frames = _seq(shifts)
        plain = _run(frames, predict=False, fill=False)
        pred = _run(frames, predict=True, fill=False)
        assert pred["valid"][1].sum() >= plain["valid"][1].sum()
        assert pred["valid"][1].mean() > 0.95

    def test_default_is_off(self):
        assert dic.DicParams().predict is False
        assert dic.DicParams().fill_invalid is False


class TestFill:
    def test_textureless_patch_is_filled_and_flagged(self):
        frames = _seq([2, 2], flat=(45, 85, 45, 85))
        no = _run(frames, predict=False, fill=False)
        yes = _run(frames, predict=False, fill=True)
        assert "Filled" not in no["fields"]
        filled = np.nan_to_num(yes["fields"]["Filled"]) > 0.5
        assert filled.any()
        # valid keeps meaning "measured": identical with and without fill
        assert np.array_equal(yes["valid"], no["valid"])
        assert not np.any(filled & yes["valid"])
        # filled values come from the neighbours (uniform 2 px translation)
        ux = yes["fields"]["Ux"]
        assert np.all(np.abs(ux[filled] - 2.0) < 0.05)
        assert yes["units"]["Filled"] == "-"

    def test_isolated_point_without_neighbours_is_not_filled(self):
        g = dic.grid_from_points(np.array([0., 1., 2.]), np.array([0., 0., 0.]))
        med, cnt = dic.neighbour_median(np.array([1., 5., 3.]),
                                        np.array([True, False, True]), g)
        assert cnt.tolist() == [0, 2, 0]
        assert med[1] == pytest.approx(2.0)
        assert np.isnan(med[0])


def test_neighbour_median_vector_values():
    xs, ys = np.meshgrid([0., 1., 2.], [0., 1., 2.])
    g = dic.grid_from_points(xs.ravel(), ys.ravel())
    vals = np.stack([np.arange(9.), -np.arange(9.)], axis=1)
    avail = np.ones(9, bool)
    avail[4] = False                                   # centre
    med, cnt = dic.neighbour_median(vals, avail, g)
    assert cnt[4] == 8
    assert med[4] == pytest.approx([4.0, -4.0])


def test_tab_passes_options_and_shows_filled(qapp):
    from gui.core.experiment_session import ExperimentSession
    from gui.tabs.dic_tab import DICTab, _shown_valid
    tab = DICTab(ExperimentSession(name="t"))
    p = tab._params()
    assert p.predict is True and p.fill_invalid is True     # UI defaults
    tab.chk_predict.setChecked(False)
    tab.chk_fill.setChecked(False)
    p = tab._params()
    assert p.predict is False and p.fill_invalid is False
    valid = np.array([[True, False, False]])
    shown = _shown_valid(valid, {"Filled": np.array([[0.0, 1.0, np.nan]])})
    assert shown.tolist() == [[True, True, False]]
    assert _shown_valid(valid, {}).tolist() == valid.tolist()
