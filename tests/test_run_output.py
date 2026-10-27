# -*- coding: utf-8 -*-
"""Per-study run output: timestamped folder name and config.json."""
from __future__ import annotations

import json
from datetime import datetime

from gui.core.run_output import study_folder_name, create_study_dir


def test_folder_name_format():
    when = datetime(2026, 9, 10, 13, 45, 7)
    assert study_folder_name("myprofile", "GCI", when, "CL-WTS-1") \
        == "CL-WTS-1__myprofile__gci__2026-09-10_13-45-07"


def test_folder_name_defaults_to_untitled():
    name = study_folder_name(None, "domainsizing", datetime(2026, 1, 2, 3, 4, 5))
    assert name == "unknown-PC__Untitled__domain__2026-01-02_03-04-05"


def test_create_study_dir_writes_config(tmp_path):
    when = datetime(2026, 9, 10, 13, 45, 7)
    d = create_study_dir(tmp_path, "prof", "GCI",
                         {"finest": 0.006, "ratio": 2}, when=when)
    assert d == tmp_path / "unknown-PC__prof__gci__2026-09-10_13-45-07"
    assert d.is_dir()
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["study"] == "GCI"
    assert cfg["profile"] == "prof"
    assert cfg["parameters"] == {"finest": 0.006, "ratio": 2}
    assert cfg["created_at"].startswith("2026-09-10T13:45:07")


def test_create_study_dir_serialises_none(tmp_path):
    d = create_study_dir(tmp_path, "p", "domainsizing", {"min_elem_size": None})
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["parameters"]["min_elem_size"] is None


# ---- names that say what a run varies, machine in the folder name -------
from types import SimpleNamespace

from gui.core import run_output as ro
from gui.sensitivity.run_cache import next_job_index


def _cfg(ms=25600.0, on=True, h=0.0005, dims=(0.25, 0.25, 0.05, 0.05)):
    l_wp, h_wp, l_void, h_void = dims
    return SimpleNamespace(
        step=SimpleNamespace(mass_scaling_factor=ms, mass_scaling_enabled=on),
        elem_size=h,
        euler_geometry=SimpleNamespace(l_wp=l_wp, h_wp=h_wp, l_void=l_void,
                                       h_void=h_void))


def test_folder_name_carries_the_machine():
    from datetime import datetime
    when = datetime(2026, 10, 26, 9, 12, 0)
    # machine first, then profile, then type, then date; '_' and spaces
    # inside a field become '-', so '__' only ever separates two fields
    assert ro.study_folder_name("Ducobu_2017", "massscaling", when,
                                "CL-WTS-1") == \
        "CL-WTS-1__Ducobu-2017__ms__2026-10-26_09-12-00"
    assert ro.study_folder_name("P", "GCI", when, "Compute PC (20 cores)") \
        == "Compute-PC-20-cores__P__gci__2026-10-26_09-12-00"
    assert ro.study_folder_name("P", "GCI", when) == \
        "unknown-PC__P__gci__2026-10-26_09-12-00"
    assert ro.study_folder_name("a__b", "sensitivity", when, "PC") == \
        "PC__a-b__sensitivity__2026-10-26_09-12-00"
    assert ro.study_folder_name("P", "domainsizing", when, "PC").split(
        "__") == ["PC", "P", "domain", "2026-10-26_09-12-00"]


def test_folder_names_sort_by_machine_profile_type_date():
    from datetime import datetime
    names = [ro.study_folder_name(p, t, datetime(2026, 10, d), m)
             for m, p, t, d in [("CL-WTS-1", "B", "GCI", 1),
                                ("CL-CHENEVEZ-01", "A", "massscaling", 2),
                                ("CL-WTS-1", "A", "massscaling", 3),
                                ("CL-WTS-1", "A", "massscaling", 1)]]
    assert sorted(names) == [names[1], names[3], names[2], names[0]]


def test_job_folder_name():
    assert ro.job_folder_name("Ducobu_2017", "Cutting_job", "CL-WTS-1") == \
        "CL-WTS-1__Ducobu-2017__job__Cutting-job"
    assert ro.job_folder_name(None, "", None) == \
        "unknown-PC__Untitled__job__job"


def test_model_run_tags():
    c = _cfg()
    assert ro.job_name("ms", 3, ro.model_run_tag("ms", c)) == \
        "ms_run003_ms25600"
    assert ro.model_run_tag("ms", _cfg(on=False)) == "ms1"
    assert ro.job_name("GCI", 2, ro.model_run_tag("GCI", c)) == \
        "GCI_run002_h0p5um"
    assert ro.job_name("domainsizing", 4,
                       ro.model_run_tag("domainsizing", c)) == \
        "domainsizing_run004_D250x250x50x50"
    assert ro.model_run_tag("checks", _cfg(h=0.001)) == "ms25600_h1um"
    assert ro.model_run_tag("sensitivity", c) == ""


def test_job_name_is_cut_and_safe():
    name = ro.job_name("domainsizing", 12,
                       "D12345x12345x12345x12345_extra")
    assert len(name) <= ro.JOB_NAME_MAX
    assert name.startswith("domainsizing_run012_D")
    assert "." not in ro.job_name("GCI", 0, "h0.5um")
    assert ro.job_name("sensitivity", 3) == "sensitivity_run003"


def test_next_index_sees_tagged_and_old_names(tmp_path):
    for n in ("ms_run000.meta.json", "ms_run001_ms250.sta",
              "ms_run004_ms2000.results.npz", "GCI_run009_h1um.sta"):
        (tmp_path / n).write_text("")
    assert next_job_index(tmp_path, "ms") == 5
    assert next_job_index(tmp_path, "GCI") == 10
