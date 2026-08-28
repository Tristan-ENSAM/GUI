# -*- coding: utf-8 -*-
"""
Unit tests for the results layer (gui.results): ResultsBundle reader, the QoI
reductions, and the plain-txt exporter.

These tests are pure Python + filesystem (no Qt, no Abaqus). They build a
synthetic (.json + .npz) bundle with gui.results.fake_builder.build_fake_results
— which follows FORMAT.md exactly — and exercise the public API against it,
plus the error/robustness paths.

Values are never hard-coded "magic numbers": every QoI assertion recomputes the
expected value independently from the bundle's own arrays, so the test checks
that the function does what it documents (e.g. max|RF1|), not that an arbitrary
constant matches.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from gui.results.fake_builder import build_fake_results
from gui.results.reader import (ResultsBundle, ResultsLoadError, InstanceInfo,
                                HistoryInfo)
from gui.results import qoi as qoi_mod
from gui.results import export_txt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# Small but non-degenerate fake bundle: 6 frames, 8x6 grid -> 48 elements.
N_FRAMES = 6
N_GX, N_GY = 8, 6
N_ELEM = N_GX * N_GY                          # 48
N_NODES = (N_GX + 1) * (N_GY + 1) * 2         # 126


@pytest.fixture()
def fake_paths(tmp_path):
    """Build a fake bundle and return (json_path, npz_path)."""
    json_path, npz_path = build_fake_results(
        tmp_path / "fake_job.results.npz",
        n_frames=N_FRAMES, n_grid_x=N_GX, n_grid_y=N_GY, seed=7)
    return json_path, npz_path


@pytest.fixture()
def bundle(fake_paths):
    """A loaded ResultsBundle, closed after the test."""
    b = ResultsBundle.load(fake_paths[1])
    yield b
    b.close()


# ---------------------------------------------------------------------------
# Builder sanity
# ---------------------------------------------------------------------------
def test_fake_builder_writes_pair(fake_paths):
    json_path, npz_path = fake_paths
    assert json_path.exists() and npz_path.exists()
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["format_version"] == 1
    assert meta["instances"]["Euler"]["n_elements"] == N_ELEM


# ---------------------------------------------------------------------------
# ResultsBundle.load — path resolution
# ---------------------------------------------------------------------------
class TestLoadPaths:

    def test_load_from_npz(self, fake_paths):
        b = ResultsBundle.load(fake_paths[1]); b.close()

    def test_load_from_json(self, fake_paths):
        b = ResultsBundle.load(fake_paths[0]); b.close()

    def test_load_from_bare_stem(self, tmp_path):
        # When the files are named "<stem>.json"/"<stem>.npz" (no ".results"
        # infix), loading by the bare stem works: load() appends both suffixes.
        build_fake_results(tmp_path / "plain.npz",
                           n_frames=N_FRAMES, n_grid_x=N_GX, n_grid_y=N_GY)
        b = ResultsBundle.load(tmp_path / "plain"); b.close()

    def test_load_from_canonical_results_stem(self, fake_paths):
        # The canonical ".results" stem now resolves correctly: load() appends
        # ".json"/".npz" to the full name (string concatenation), so
        # ".../fake_job.results" -> ".../fake_job.results.json"/.npz.
        # (Previously, with_suffix replaced ".results" and this failed.)
        stem = fake_paths[1].with_suffix("")          # ".../fake_job.results"
        b = ResultsBundle.load(stem)
        assert b.job_name == "fake_job"
        b.close()


# ---------------------------------------------------------------------------
# ResultsBundle — metadata properties
# ---------------------------------------------------------------------------
class TestBundleMeta:

    def test_basic_properties(self, bundle):
        assert bundle.job_name == "fake_job"
        assert bundle.step_name == "Cut"
        assert isinstance(bundle.model_config, dict)
        assert "fake" in bundle.source_odb

    def test_times_and_frames(self, bundle):
        t = bundle.times
        assert t.shape == (N_FRAMES,)
        assert bundle.n_frames == N_FRAMES
        # Monotonically non-decreasing, starts at 0.
        assert t[0] == pytest.approx(0.0)
        assert np.all(np.diff(t) >= 0)

    def test_roi(self, bundle):
        roi = bundle.roi
        assert roi is not None
        for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            assert k in roi
        assert roi["xmax"] > roi["xmin"]

    def test_repr(self, bundle):
        r = repr(bundle)
        assert "fake_job" in r and "Euler" in r


# ---------------------------------------------------------------------------
# ResultsBundle — instances
# ---------------------------------------------------------------------------
class TestBundleInstances:

    def test_instance_names(self, bundle):
        assert bundle.instance_names == ["Euler"]

    def test_instance_info(self, bundle):
        info = bundle.instance("Euler")
        assert isinstance(info, InstanceInfo)
        assert info.kind == "eulerian"
        assert info.element_type == "EC3D8RT"
        assert info.n_frames == N_FRAMES
        assert info.n_elements == N_ELEM
        assert info.has_displacements is False
        assert set(info.field_variables) == {"PEEQ", "TEMP", "S_VM", "EVF"}

    def test_unknown_instance_raises(self, bundle):
        with pytest.raises(KeyError):
            bundle.instance("DoesNotExist")

    def test_geometry_arrays_shapes(self, bundle):
        assert bundle.nodes_init("Euler").shape == (N_NODES, 3)
        assert bundle.elements("Euler").shape == (N_ELEM, 8)
        assert bundle.element_centroids_init("Euler").shape == (N_ELEM, 3)

    def test_displacements_absent(self, bundle):
        # Fake Eulerian bundle stores no displacements.
        assert bundle.displacements("Euler") is None


# ---------------------------------------------------------------------------
# ResultsBundle — fields
# ---------------------------------------------------------------------------
class TestBundleFields:

    def test_field_shape(self, bundle):
        for var in ("PEEQ", "TEMP", "S_VM", "EVF"):
            arr = bundle.field("Euler", var)
            assert arr.shape == (N_FRAMES, N_ELEM)

    def test_unknown_field_raises(self, bundle):
        with pytest.raises(KeyError):
            bundle.field("Euler", "NOPE")

    def test_evf_in_unit_range(self, bundle):
        evf = bundle.field("Euler", "EVF")
        assert evf.min() >= 0.0 and evf.max() <= 1.0


# ---------------------------------------------------------------------------
# ResultsBundle — history
# ---------------------------------------------------------------------------
class TestBundleHistory:

    def test_history_info(self, bundle):
        hi = bundle.history_info
        assert isinstance(hi, HistoryInfo)
        assert set(hi.variables) == {"RF1_RP", "RF2_RP"}
        assert hi.n_samples > 0

    def test_history_arrays(self, bundle):
        t = bundle.history_time
        rf1 = bundle.history("RF1_RP")
        assert t.ndim == 1 and rf1.ndim == 1
        assert t.shape == rf1.shape

    def test_unknown_history_raises(self, bundle):
        with pytest.raises(KeyError):
            bundle.history("NOPE")


# ---------------------------------------------------------------------------
# ResultsBundle — lifecycle
# ---------------------------------------------------------------------------
class TestBundleLifecycle:

    def test_close_idempotent(self, fake_paths):
        b = ResultsBundle.load(fake_paths[1])
        b.close()
        b.close()      # must not raise

    def test_context_manager(self, fake_paths):
        with ResultsBundle.load(fake_paths[1]) as b:
            assert b.n_frames == N_FRAMES


# ---------------------------------------------------------------------------
# ResultsBundle — error paths
# ---------------------------------------------------------------------------
class TestLoadErrors:

    def test_missing_json(self, fake_paths, tmp_path):
        # npz present, json removed.
        fake_paths[0].unlink()
        with pytest.raises(ResultsLoadError):
            ResultsBundle.load(fake_paths[1])

    def test_missing_npz(self, fake_paths):
        fake_paths[1].unlink()
        with pytest.raises(ResultsLoadError):
            ResultsBundle.load(fake_paths[0])

    def test_missing_format_version(self, fake_paths):
        meta = json.loads(fake_paths[0].read_text(encoding="utf-8"))
        del meta["format_version"]
        fake_paths[0].write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ResultsLoadError):
            ResultsBundle.load(fake_paths[1])

    def test_future_version_rejected(self, fake_paths):
        meta = json.loads(fake_paths[0].read_text(encoding="utf-8"))
        meta["format_version"] = ResultsBundle.SUPPORTED_VERSION + 1
        fake_paths[0].write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ResultsLoadError):
            ResultsBundle.load(fake_paths[1])

    def test_malformed_json(self, fake_paths):
        fake_paths[0].write_text("{ this is not valid json", encoding="utf-8")
        with pytest.raises(ResultsLoadError):
            ResultsBundle.load(fake_paths[1])


# ---------------------------------------------------------------------------
# QoI — values recomputed independently from the bundle arrays
# ---------------------------------------------------------------------------
class TestQoI:

    def test_registry_ids(self):
        ids = qoi_mod.available_qoi_ids()
        assert set(ids) == {"Fx_max", "Fx_mean", "Fy_max", "Fy_mean",
                            "T_max", "PEEQ_max"}

    def test_all_qois_finite(self, bundle):
        out = qoi_mod.compute_qois(bundle)
        assert set(out) == set(qoi_mod.available_qoi_ids())
        assert all(np.isfinite(v) for v in out.values())

    def test_force_qois_match_recomputation(self, bundle):
        rf1 = np.abs(np.asarray(bundle.history("RF1_RP"), float))
        rf2 = np.abs(np.asarray(bundle.history("RF2_RP"), float))
        out = qoi_mod.compute_qois(bundle, ["Fx_max", "Fx_mean", "Fy_max"])
        assert out["Fx_max"] == pytest.approx(float(rf1.max()), rel=1e-6)
        assert out["Fx_mean"] == pytest.approx(float(rf1.mean()), rel=1e-6)
        assert out["Fy_max"] == pytest.approx(float(rf2.max()), rel=1e-6)

    def test_field_qois_match_recomputation(self, bundle):
        temp = np.asarray(bundle.field("Euler", "TEMP"), float)
        peeq = np.asarray(bundle.field("Euler", "PEEQ"), float)
        out = qoi_mod.compute_qois(bundle, ["T_max", "PEEQ_max"])
        assert out["T_max"] == pytest.approx(float(np.nanmax(temp)), rel=1e-6)
        assert out["PEEQ_max"] == pytest.approx(float(np.nanmax(peeq)), rel=1e-6)

    def test_warmup_frac_skips_head(self, bundle):
        rf1 = np.abs(np.asarray(bundle.history("RF1_RP"), float))
        frac = 0.5
        start = int(round(frac * rf1.size))
        expected = float(rf1[start:].mean())
        out = qoi_mod.compute_qois(bundle, ["Fx_mean"], warmup_frac=frac)
        assert out["Fx_mean"] == pytest.approx(expected, rel=1e-6)

    def test_qoi_spec_unknown_raises(self):
        with pytest.raises(KeyError):
            qoi_mod.qoi_spec("not_a_qoi")

    def test_missing_field_helper_returns_nan(self, bundle):
        # Robustness primitive: a field that does not exist -> nan, no raise.
        val = qoi_mod._field_global_max(bundle, "NONEXISTENT_FIELD", None)
        assert np.isnan(val)

    def test_missing_history_helper_returns_none(self, bundle):
        assert qoi_mod._history_abs(bundle, "NONEXISTENT_HIST", 0.0) is None

    def test_compute_from_path_matches_bundle(self, fake_paths, bundle):
        from_path = qoi_mod.compute_qois_from_path(fake_paths[1])
        direct = qoi_mod.compute_qois(bundle)
        assert set(from_path) == set(direct)
        for k in direct:
            assert from_path[k] == pytest.approx(direct[k], rel=1e-6, nan_ok=True)

    def test_compute_from_missing_path_all_nan(self, tmp_path):
        out = qoi_mod.compute_qois_from_path(tmp_path / "does_not_exist.results.npz")
        assert set(out) == set(qoi_mod.available_qoi_ids())
        assert all(np.isnan(v) for v in out.values())


# ---------------------------------------------------------------------------
# export_txt
# ---------------------------------------------------------------------------
class TestExportTxt:

    def test_export_instance_fields(self, bundle, tmp_path):
        outdir = tmp_path / "export"
        paths = export_txt.export_instance_fields(bundle, "Euler", str(outdir))
        # One file per field variable.
        info = bundle.instance("Euler")
        assert len(paths) == len(info.field_variables)
        for p in paths:
            assert p.endswith(".txt")
            data = np.genfromtxt(p, delimiter="\t", skip_header=1)
            # (n_elem, 1 index column + n_frames).
            assert data.shape == (N_ELEM, 1 + N_FRAMES)

    def test_export_roundtrip_values(self, bundle, tmp_path):
        outdir = tmp_path / "export"
        export_txt.export_instance_fields(bundle, "Euler", str(outdir))
        peeq_path = outdir / "Euler__PEEQ.txt"
        data = np.genfromtxt(peeq_path, delimiter="\t", skip_header=1)
        # Drop the index column; the stored matrix is field.T (n_elem, n_frames).
        reloaded = data[:, 1:]
        original = np.asarray(bundle.field("Euler", "PEEQ"), float)   # (n_frames, n_elem)
        assert np.allclose(reloaded.T, original, rtol=1e-5, atol=1e-6)

    def test_export_header_label(self, bundle, tmp_path):
        outdir = tmp_path / "export"
        export_txt.export_instance_fields(bundle, "Euler", str(outdir))
        first_line = (outdir / "Euler__PEEQ.txt").read_text().splitlines()[0]
        assert first_line.split("\t")[0] == "elem_index"

    def test_export_bundle_includes_centroids(self, bundle, tmp_path):
        outdir = tmp_path / "export_all"
        paths = export_txt.export_bundle(bundle, str(outdir))
        names = {p.rsplit("/", 1)[-1].replace("\\", "/").split("/")[-1] for p in paths}
        # Field files + a centroids file for the Euler instance.
        assert any(n.endswith("__element_centroids.txt") for n in names)
        cen_path = outdir / "Euler__element_centroids.txt"
        cen = np.genfromtxt(cen_path, delimiter="\t", skip_header=1)
        assert cen.shape == (N_ELEM, 4)        # index + x + y + z

    def test_write_matrix_txt_time_length_fallback(self, tmp_path):
        # When times length != n_cols, the writer falls back to a 0..n-1 index
        # header instead of crashing.
        path = str(tmp_path / "m.txt")
        matrix = np.arange(12, dtype=float).reshape(3, 4)   # 3 rows, 4 cols
        export_txt._write_matrix_txt(path, matrix, times=np.array([0.0, 1.0]))
        data = np.genfromtxt(path, delimiter="\t", skip_header=1)
        assert data.shape == (3, 1 + 4)
