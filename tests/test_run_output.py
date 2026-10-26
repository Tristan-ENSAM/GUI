# -*- coding: utf-8 -*-
"""Per-study run output: timestamped folder name and config.json."""
from __future__ import annotations

import json
from datetime import datetime

from gui.core.run_output import study_folder_name, create_study_dir


def test_folder_name_format():
    when = datetime(2026, 9, 10, 13, 45, 7)
    assert study_folder_name("myprofile", "GCI", when) \
        == "myprofile_GCI_2026-09-10_13-45-07"


def test_folder_name_defaults_to_untitled():
    name = study_folder_name(None, "domainsizing", datetime(2026, 1, 2, 3, 4, 5))
    assert name == "Untitled_domainsizing_2026-01-02_03-04-05"


def test_create_study_dir_writes_config(tmp_path):
    when = datetime(2026, 9, 10, 13, 45, 7)
    d = create_study_dir(tmp_path, "prof", "GCI",
                         {"finest": 0.006, "ratio": 2}, when=when)
    assert d == tmp_path / "prof_GCI_2026-09-10_13-45-07"
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
    assert ro.study_folder_name("Ducobu_2017", "massscaling", when,
                                "CL-WTS-1") == \
        "Ducobu_2017_massscaling_CL-WTS-1_2026-10-26_09-12-00"
    assert ro.study_folder_name("P", "GCI", when, "Compute PC (20 cores)") \
        == "P_GCI_Compute-PC-20-cores_2026-10-26_09-12-00"
    assert ro.study_folder_name("P", "GCI", when) == \
        "P_GCI_2026-10-26_09-12-00"


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
