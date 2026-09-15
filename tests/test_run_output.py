# -*- coding: utf-8 -*-
"""Per-study run output: timestamped folder name and config.json."""
from __future__ import annotations

import json
from datetime import datetime

from gui.core.run_output import study_folder_name, create_study_dir


def test_folder_name_format():
    when = datetime(2026, 9, 10, 13, 45, 7)
    assert study_folder_name("myprofile", "GCI", when) \
        == "myprofile_GCI_20260910_134507"


def test_folder_name_defaults_to_untitled():
    name = study_folder_name(None, "domainsizing", datetime(2026, 1, 2, 3, 4, 5))
    assert name == "Untitled_domainsizing_20260102_030405"


def test_create_study_dir_writes_config(tmp_path):
    when = datetime(2026, 9, 10, 13, 45, 7)
    d = create_study_dir(tmp_path, "prof", "GCI",
                         {"finest": 0.006, "ratio": 2}, when=when)
    assert d == tmp_path / "prof_GCI_20260910_134507"
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
