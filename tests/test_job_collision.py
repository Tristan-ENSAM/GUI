# -*- coding: utf-8 -*-
"""
Launching a job whose name already exists in the working directory.

Abaqus refuses to overwrite an existing job and dies, so the collision has to
be caught BEFORE launching. Windows paths are case-insensitive, so "Job1" and
"JOB1" collide too -- the match is therefore case-insensitive on the stem.
"""
from __future__ import annotations

import pytest

from gui.tabs.job_tab import _existing_job_files, _remove_job_files


@pytest.fixture
def workdir(tmp_path):
    for n in ("Job1.odb", "Job1.sta", "JOB1.dat", "job1.msg",
              "Job10.odb", "Other.odb"):
        (tmp_path / n).write_text("x")
    return tmp_path


class TestExistingJobFiles:
    def test_matches_ignoring_case(self, workdir):
        names = {f.name for f in _existing_job_files(workdir, "Job1")}
        assert {"Job1.odb", "Job1.sta", "JOB1.dat", "job1.msg"} == names

    def test_does_not_match_a_longer_stem(self, workdir):
        # "Job10" merely starts with "Job1": the dot separator must be required.
        names = {f.name for f in _existing_job_files(workdir, "Job1")}
        assert "Job10.odb" not in names

    def test_does_not_match_other_jobs(self, workdir):
        names = {f.name for f in _existing_job_files(workdir, "Job1")}
        assert "Other.odb" not in names

    def test_empty_when_nothing_matches(self, workdir):
        assert _existing_job_files(workdir, "Nothing") == []

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert _existing_job_files(tmp_path / "nope", "Job1") == []


class TestRemoveJobFiles:
    def test_removes_and_reports_no_failure(self, workdir):
        files = _existing_job_files(workdir, "Job1")
        assert _remove_job_files(files) == []
        assert _existing_job_files(workdir, "Job1") == []

    def test_leaves_other_jobs_alone(self, workdir):
        _remove_job_files(_existing_job_files(workdir, "Job1"))
        remaining = {f.name for f in workdir.iterdir()}
        assert remaining == {"Job10.odb", "Other.odb"}

    def test_reports_files_it_could_not_remove(self, tmp_path):
        missing = tmp_path / "gone.odb"      # never created -> unlink raises
        assert _remove_job_files([missing]) == [missing]
