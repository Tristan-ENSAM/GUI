# -*- coding: utf-8 -*-
"""
Test the per-frame status log + ETA for the LOCAL DIC engine, both at the
engine level (compute_dic_fields on_frame callback) and through the threaded
worker (_DicWorker.sig_log).
"""
import numpy as np
from PySide6.QtCore import Qt

from gui.core import dic as dic_engine
from gui.core.dic import make_grid, DicParams
from gui.core.sequence_io import ImageSequence
from gui.tabs.dic_tab import _DicWorker


def _speckle(H=120, W=120, n=250, seed=0):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    img = np.zeros((H, W), float)
    for _ in range(n):
        cx = rng.integers(5, W - 5)
        cy = rng.integers(5, H - 5)
        r = rng.uniform(2.0, 4.0)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _frames():
    base = _speckle(seed=1)
    return [base, np.roll(base, 2, axis=1), np.roll(base, 4, axis=1)]


def test_engine_on_frame_local():
    frames = _frames()
    pts = make_grid((20, 20, 80, 80), 12, margin=25)
    infos = []
    dic_engine.compute_dic_fields(
        frames, pts, DicParams(engine="local", subset=31, search=12,
                               zncc_min=0.4, subpixel=True),
        fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120,
        on_frame=lambda d: infos.append(d))
    assert len(infos) == len(frames) - 1
    for key in ("index", "n_pairs", "n_valid", "n_total",
                "mean_zncc", "elapsed_s", "frame_s"):
        assert key in infos[0]
    assert infos[0]["n_total"] == len(pts)


def test_local_worker_emits_log(qapp):
    frames = _frames()
    pts = make_grid((20, 20, 80, 80), 12, margin=25)
    seq = ImageSequence.from_array(np.asarray(frames), fps=1000.0)
    w = _DicWorker(seq, pts, DicParams(engine="local", subset=31, search=12,
                                       zncc_min=0.4, subpixel=True),
                   1000.0, 0.01, 120, 120, 0.0, None)
    msgs, done, failed = [], [], []
    w.sig_log.connect(lambda m: msgs.append(m), Qt.QueuedConnection)
    w.sig_done.connect(lambda *_: done.append(True), Qt.QueuedConnection)
    w.sig_failed.connect(lambda m: failed.append(m), Qt.QueuedConnection)
    w.start()
    w.wait()
    for _ in range(50):
        qapp.processEvents()
    assert not failed, failed
    assert done
    assert len(msgs) == len(frames) - 1
    assert "valid" in msgs[0] and "ZNCC" in msgs[0] and "ETA" in msgs[0]


def _frames_darkening():
    """Frame 0 full speckle; frame 1 right half darkened (out of material)."""
    base = _speckle(seed=1)
    f1 = base.copy()
    f1[:, 60:] = 2
    return [base, f1, base]


class TestPerFrameMaskCompute:

    def test_no_cumulated_strain_fields(self):
        """The local engine reports only instantaneous strain rates; the
        cumulated Exx/Eyy/Exy/Eeq fields are gone (no meaning on an Eulerian
        grid as material leaves the FOV)."""
        frames = _frames()
        pts = make_grid((20, 20, 80, 80), 12, margin=25)
        res = dic_engine.compute_dic_fields(
            frames, pts, DicParams(engine="local", subset=31, search=12,
                                   zncc_min=0.4, subpixel=True),
            fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120)
        for k in ("Exx", "Eyy", "Exy", "Eeq"):
            assert k not in res["fields"]
        for k in ("Exx_dot", "Eyy_dot", "Exy_dot", "Eeq_dot"):
            assert k in res["fields"]
        assert "Eeq" not in res["units"]

    def test_per_frame_mask_drops_points_when_material_leaves(self):
        frames = _frames_darkening()
        pts = make_grid((20, 20, 80, 80), 12, margin=25)
        res = dic_engine.compute_dic_fields(
            frames, pts, DicParams(engine="local", subset=31, search=12,
                                   zncc_min=0.4, subpixel=True),
            fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120,
            mask_per_frame=True,
            mask_params={"min_intensity": 25})
        # pair 0 ref = frame 0 (material everywhere); pair 1 ref = frame 1
        # (right half dark) -> fewer valid points.
        assert res["valid"][1].sum() < res["valid"][0].sum()

    def test_per_frame_mask_makes_nan_holes(self):
        frames = _frames_darkening()
        pts = make_grid((20, 20, 80, 80), 12, margin=25)
        res = dic_engine.compute_dic_fields(
            frames, pts, DicParams(engine="local", subset=31, search=12,
                                   zncc_min=0.4, subpixel=True),
            fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120,
            mask_per_frame=True,
            mask_params={"min_intensity": 25})
        # Masked-out points on pair 1 carry NaN displacement (temporal hole).
        assert np.isnan(res["fields"]["Ux"][1]).any()

    def test_static_mask_still_works(self):
        """With mask_per_frame off, the legacy static point_keep is honoured."""
        frames = _frames()
        pts = make_grid((20, 20, 80, 80), 12, margin=25)
        keep = np.ones(len(pts), bool)
        keep[: len(pts) // 2] = False
        res = dic_engine.compute_dic_fields(
            frames, pts, DicParams(engine="local", subset=31, search=12,
                                   zncc_min=0.4, subpixel=True),
            fps=1000.0, mm_per_px=0.01, img_w=120, img_h=120,
            point_keep=keep, mask_per_frame=False)
        # The statically-dropped half is NaN on every frame.
        assert np.all(np.isnan(res["fields"]["Ux"][0][:len(pts) // 2]))


class TestWorkerPerFrameMask:

    def test_worker_passes_per_frame_mask(self, qapp):
        frames = _frames_darkening()
        pts = make_grid((20, 20, 80, 80), 12, margin=25)
        seq = ImageSequence.from_array(np.asarray(frames), fps=1000.0)
        w = _DicWorker(seq, pts, DicParams(engine="local", subset=31, search=12,
                                           zncc_min=0.4, subpixel=True),
                       1000.0, 0.01, 120, 120, 0.0, None,
                       mask_per_frame=True,
                       mask_params={"min_intensity": 25})
        res, done, failed = [], [], []
        w.sig_done.connect(lambda r: (res.append(r), done.append(True)),
                           Qt.QueuedConnection)
        w.sig_failed.connect(lambda m: failed.append(m), Qt.QueuedConnection)
        w.start()
        w.wait()
        for _ in range(50):
            qapp.processEvents()
        assert not failed, failed
        assert done
        assert res[0]["valid"][1].sum() < res[0]["valid"][0].sum()
