# -*- coding: utf-8 -*-
"""Simulation database (gui.core.run_db): archive, lookup, release rule,
import, index, and the RunCache fallback."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from gui.core import run_db
from gui.core.preferences import Preferences
from gui.sensitivity.run_cache import RunCache, params_key

SUCCESS = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"


def _run(folder: Path, job: str, params: dict, odb: bool = False,
         done: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (job + ".results.npz")).write_bytes(b"npz")
    (folder / (job + ".meta.json")).write_text(
        json.dumps({"model_config": params}), encoding="utf-8")
    (folder / (job + ".sta")).write_text(SUCCESS if done else "running\n")
    (folder / (job + ".inp")).write_text("*Heading\n")
    if odb:
        (folder / (job + ".odb")).write_bytes(b"odb" * 10)


P1 = {"mesh": {"elem_size": 0.002}, "step": {"ms": 25600}}
P2 = {"mesh": {"elem_size": 0.001}, "step": {"ms": 25600}}


def _prov(host="CL-WTS-1", release="Abaqus 2022", **kw):
    return run_db.provenance(host=host, release=release, cpus=20, **kw)


def test_archive_then_lookup(tmp_path):
    src = tmp_path / "study"
    _run(src, "ms_run000_ms25600", P1, odb=True)
    db = run_db.RunDatabase(tmp_path / "db")
    run = db.archive(src, "ms_run000_ms25600", _prov(profile="Ducobu_2017"))
    assert run is not None
    key = params_key(P1)
    # runs/<2 hex>/<16 hex>__<PC>__<profile>__<job>
    assert run.folder.parent == tmp_path / "db" / "runs" / key[:2]
    assert run.folder.name == \
        "%s__CL-WTS-1__Ducobu-2017__ms-run000-ms25600" % key[:16]
    for name in ("ms_run000_ms25600.results.npz",
                 "ms_run000_ms25600.meta.json", "ms_run000_ms25600.sta",
                 "ms_run000_ms25600.inp", "params.json", "provenance.json",
                 "COMPLETE"):
        assert (run.folder / name).is_file(), name
    assert not (run.folder / "ms_run000_ms25600.odb").exists()  # not asked
    assert run.prov["odb_kept"] is False
    assert (tmp_path / "db" / "README.txt").is_file()
    hit = db.lookup(P1, host="CL-WTS-1", release="Abaqus 2022")
    assert hit is not None and hit.job == "ms_run000_ms25600"
    assert db.lookup(P2, host="CL-WTS-1", release="Abaqus 2022") is None


def test_archive_keeps_odb_when_asked_and_never_duplicates(tmp_path):
    src = tmp_path / "s"
    _run(src, "job", P1, odb=True)
    db = run_db.RunDatabase(tmp_path / "db")
    a = db.archive(src, "job", _prov(), keep_odb=True)
    assert (a.folder / "job.odb").is_file() and a.prov["odb_kept"] is True
    b = db.archive(src, "job", _prov(), keep_odb=True)
    assert b.folder == a.folder
    assert len(db.runs()) == 1
    # the same parameters from another PC / release: a second entry
    c = db.archive(src, "job", _prov(host="CL-CHENEVEZ-01",
                                     release="Abaqus 2022.HF8"))
    assert c.folder != a.folder
    assert len(db.find(P1)) == 2


def test_unfinished_run_is_not_archived(tmp_path):
    src = tmp_path / "s"
    _run(src, "job", P1, done=False)
    db = run_db.RunDatabase(tmp_path / "db")
    assert db.archive(src, "job", _prov()) is None
    assert db.runs() == []


def test_incomplete_copy_is_ignored(tmp_path):
    src = tmp_path / "s"
    _run(src, "job", P1)
    db = run_db.RunDatabase(tmp_path / "db")
    run = db.archive(src, "job", _prov())
    (run.folder / "COMPLETE").unlink()
    assert db.find(P1) == []
    # the next archive of that run completes the same folder
    again = db.archive(src, "job", _prov())
    assert again is not None and again.folder == run.folder


def test_release_rule():
    run = run_db.DbRun(folder=Path("."), job="j", key="k", params={},
                       prov={"host": "CL-WTS-1", "abaqus_release": "2022 GA"})
    acc = run_db.release_accepted
    assert acc(run, "CL-CHENEVEZ-01", "2022 GA")          # same release
    assert not acc(run, "CL-WTS-1", "2022.HF8")           # other release
    assert acc(run, "CL-WTS-1", "2022.HF8", any_release=True)
    assert acc(run, "cl-wts-1", "")                       # unknown: same PC
    assert not acc(run, "CL-CHENEVEZ-01", "")             # unknown, other PC
    unknown = run_db.DbRun(folder=Path("."), job="j", key="k", params={},
                           prov={"host": "unknown",
                                 "abaqus_release": "unknown"})
    assert not acc(unknown, "CL-WTS-1", "2022 GA")
    assert acc(unknown, "CL-WTS-1", "2022 GA", any_release=True)


def test_parse_release_and_agent_copy_agree():
    from gui.core import remote_agent
    text = ("\nAbaqus JOB information\nAbaqus 2022.HF8\n"
            "Build ID: 2023_03_27-10.12.34 xyz\nmore\n")
    for parse in (run_db.parse_release, remote_agent.parse_release):
        assert parse(text) == "Abaqus 2022.HF8 | Build ID: 2023_03_27-10.12.34 xyz"
        assert parse("nothing here") == ""
        assert parse("Abaqus 2022") == "Abaqus 2022"


def test_run_cache_falls_back_on_the_database(tmp_path):
    src = tmp_path / "old"
    _run(src, "GCI_run001_h2um", P1)
    db = run_db.RunDatabase(tmp_path / "db")
    db.archive(src, "GCI_run001_h2um", _prov())
    cache = RunCache([tmp_path / "new"])
    assert cache.lookup(P1) is None                       # no database yet
    calls = []

    def rel():
        calls.append(1)
        return "Abaqus 2022"
    cache.attach_db(db, host="CL-WTS-1", release_getter=rel)
    hit = cache.lookup(P1)
    assert hit is not None and hit.job == "GCI_run001_h2um"
    assert hit.npz.is_file() and hit.sta.is_file()
    cache.lookup(P2)
    assert calls == [1]                                   # read once
    other = RunCache([])
    other.attach_db(db, host="CL-CHENEVEZ-01",
                    release_getter=lambda: "Abaqus 2022.HF8")
    assert other.lookup(P1) is None                       # other release


def test_unplugged_database_is_skipped(tmp_path):
    db = run_db.RunDatabase(tmp_path / "not_plugged")
    cache = RunCache([])
    prefs = replace(Preferences(), run_db_dir=str(tmp_path / "not_plugged"))
    line = run_db.attach_to_cache(cache, prefs, "This PC", "PC")
    assert "not reachable" in line and cache.db is None
    assert not db.available()
    assert "not reachable" in run_db.archive_run(prefs, "This PC", "PC",
                                                 tmp_path, "job")
    assert run_db.attach_to_cache(cache, Preferences(), "This PC", "PC") == ""


def test_archive_run_from_prefs(tmp_path, monkeypatch):
    src = tmp_path / "CL-WTS-1__Ducobu-2017__job__T"
    _run(src, "T", P1, odb=True)
    (tmp_path / "db").mkdir()
    prefs = replace(Preferences(), run_db_dir=str(tmp_path / "db"),
                    run_db_keep_odb=True)
    monkeypatch.setattr(run_db, "machine_release", lambda p, m: "Abaqus 2022")
    line = run_db.archive_run(prefs, "Compute PC", "CL-WTS-1", src, "T", 20,
                              "Ducobu_2017")
    assert "in the simulation database" in line
    (run,) = run_db.RunDatabase(tmp_path / "db").runs()
    assert run.prov["cpus"] == 20 and run.prov["abaqus_release"] == \
        "Abaqus 2022"
    assert (run.folder / "T.odb").is_file()
    off = replace(prefs, run_db_archive=False)
    assert run_db.archive_run(off, "Compute PC", "CL-WTS-1", src, "T") == ""


def test_import_folder_and_index(tmp_path):
    work = tmp_path / "ABQ_wd"
    _run(work / "CL-WTS-1__Ducobu-2017__ms__2026-10-26_17-10-56",
         "ms_run000_ms1", P1)
    _run(work / "Ducobu_2017_massscaling_2026-10-07_17-01-22", "ms_run001",
         P2)
    _run(work / "half", "ms_run002", {"x": 1}, done=False)
    db = run_db.RunDatabase(tmp_path / "db")
    counts = db.import_folder(work, log=lambda *_: None)
    assert counts == {"added": 2, "present": 0, "skipped": 1}
    assert db.import_folder(work, log=lambda *_: None)["present"] == 2
    hosts = sorted(r.host for r in db.runs())
    assert hosts == ["CL-WTS-1", "unknown"]               # never guessed
    assert all(r.release == "unknown" for r in db.runs())
    idx = db.rebuild_index()
    text = idx.read_text(encoding="utf-8")
    assert "mesh.elem_size" in text                       # varies
    assert "step.ms" not in text                          # same everywhere
    assert (tmp_path / "db" / "index.xlsx").is_file()


def test_command_line(tmp_path, capsys):
    _run(tmp_path / "wd" / "a", "j", P1)
    assert run_db.main(["import", str(tmp_path / "wd"), "--db",
                        str(tmp_path / "db")]) == 0
    out = capsys.readouterr().out
    assert "added 1" in out and "index.csv" in out
