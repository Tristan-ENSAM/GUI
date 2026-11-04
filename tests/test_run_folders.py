# -*- coding: utf-8 -*-
"""One folder per run (2026-11-04): a run writes all its files into
<study>/<job>/; study folders made before, with the files directly in the
study folder ("flat"), are still read, resumed and reused."""
from __future__ import annotations

import json
from pathlib import Path

from gui.sensitivity import run_cache as rc
from gui.sensitivity import study_specs as ss

_OK = " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"


def _write(d: Path, job: str, params: dict, ok=True):
    d.mkdir(parents=True, exist_ok=True)
    (d / (job + ".results.npz")).write_bytes(b"npz")
    (d / (job + ".meta.json")).write_text(
        json.dumps({"model_config": params}), encoding="utf-8")
    (d / (job + ".sta")).write_text("h\n" + (_OK if ok else "ERROR\n"),
                                    encoding="latin-1")


def test_job_dir_new_and_flat(tmp_path):
    study = tmp_path / "study"
    _write(study / "ms_run000_ms1", "ms_run000_ms1", {"a": 1})
    _write(study, "ms_run001", {"a": 2})                    # flat, older
    assert rc.job_dir(study, "ms_run000_ms1") == study / "ms_run000_ms1"
    assert rc.job_dir(study, "ms_run001") == study
    # a job with no file yet gets its own folder
    assert rc.job_dir(study, "ms_run002") == study / "ms_run002"
    assert rc.run_completed(study, "ms_run000_ms1")
    assert rc.run_completed(study, "ms_run001")


def test_cache_indexes_both_layouts(tmp_path):
    study = tmp_path / "study"
    _write(study / "GCI_run000_h1um", "GCI_run000_h1um", {"h": 1})
    _write(study, "GCI_run001", {"h": 2})
    _write(study / "GCI_run002", "GCI_run002", {"h": 3}, ok=False)
    cache = rc.RunCache([study])
    assert sorted(r.job for r in cache.runs) == ["GCI_run000_h1um",
                                                "GCI_run001"]
    hit = cache.lookup({"h": 1})
    assert hit.folder == study.resolve() or hit.folder == study
    assert hit.npz == study / "GCI_run000_h1um" / "GCI_run000_h1um.results.npz"
    assert cache.lookup({"h": 2}).npz == study / "GCI_run001.results.npz"
    assert rc.jobs_of(study) == ["GCI_run000_h1um", "GCI_run001",
                                 "GCI_run002"]


def test_next_index_counts_run_folders(tmp_path):
    study = tmp_path / "s"
    (study / "ms_run004_ms200").mkdir(parents=True)
    (study / "ms_run002.sta").write_text("")
    (study / "msx_run009").mkdir()                           # other prefix
    assert rc.next_job_index(study, "ms") == 5


def test_copy_run_into_a_run_folder(tmp_path):
    old = tmp_path / "old"
    _write(old, "ms_run000", {"a": 1})                        # flat source
    new = tmp_path / "new"
    new.mkdir()
    assert rc.copy_run(old, "ms_run000", new, "ms_run003_ms1")
    d = new / "ms_run003_ms1"
    assert sorted(p.name for p in d.iterdir()) == [
        "ms_run003_ms1.meta.json", "ms_run003_ms1.results.npz",
        "ms_run003_ms1.sta"]
    assert rc.run_completed(new, "ms_run003_ms1")


def test_failed_marker_and_stale_files(tmp_path):
    study = tmp_path / "s"
    d = rc.new_job_dir(study, "ms_run001_ms51200")
    (d / "ms_run001_ms51200.sta").write_text("THE ANALYSIS HAS NOT BEEN "
                                             "COMPLETED\n")
    assert rc.sta_outcome(study, "ms_run001_ms51200") == "not_completed"
    assert rc.write_failed_marker(study, "ms_run001_ms51200", {"a": 1}, "x")
    assert (d / "ms_run001_ms51200.failed.json").exists()
    cache = rc.RunCache([study])
    assert cache.lookup_failed({"a": 1}, study) is not None
    (study / "ms_run001_ms51200.log").write_text("")          # flat leftover
    files = rc.job_files(study, "ms_run001_ms51200")
    assert {p.name for p in files} == {"ms_run001_ms51200.sta",
                                       "ms_run001_ms51200.failed.json",
                                       "ms_run001_ms51200.log"}
    assert rc.remove_job_files(study, "ms_run001_ms51200") == []
    assert rc.job_files(study, "ms_run001_ms51200") == []


def test_first_finished_params_reads_run_folders(tmp_path):
    study = tmp_path / "s"
    _write(study / "ms_run000_ms1", "ms_run000_ms1", {"step": 1})
    assert ss.first_finished_params(study, "ms") == {"step": 1}
