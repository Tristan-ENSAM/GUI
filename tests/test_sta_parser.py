# -*- coding: utf-8 -*-
"""
Unit tests for gui.core.sta_parser.

Pure logic + filesystem (no Qt, no Abaqus). The example rows in the module
docstring are used as fixtures. Covers the empty/missing-file path, parsing
of the "Output Field Frame Number" row and the increment row, the last-wins
behaviour when several frames are present, and the StaProgress helpers.
"""
from __future__ import annotations

import pytest

from gui.core.sta_parser import parse_sta, StaProgress


# Representative .sta rows (from the module docstring).
FRAME_ROW = "Output Field Frame Number   7, of  500, at step time 7.000E-06\n"
INC_ROW = ("  12479  7.000E-06 7.000E-06  00:07:22 5.604E-10       "
           "16760  9.030E-06  1.659E-01\n")
HEADER = "Abaqus/Explicit 2022                      DATE 01-Jan-2026\n"


# ---------------------------------------------------------------------------
# StaProgress helpers
# ---------------------------------------------------------------------------
class TestStaProgress:

    def test_empty_not_ready(self):
        s = StaProgress()
        assert s.is_ready() is False
        assert s.fraction() is None

    def test_ready_on_frame(self):
        s = StaProgress(frame_current=7, frame_total=500)
        assert s.is_ready() is True
        assert s.fraction() == pytest.approx(7 / 500)

    def test_ready_on_inc(self):
        s = StaProgress(inc_number=100)
        assert s.is_ready() is True

    def test_fraction_none_without_total(self):
        assert StaProgress(frame_current=7).fraction() is None

    def test_fraction_guards_zero_total(self):
        assert StaProgress(frame_current=7, frame_total=0).fraction() is None


# ---------------------------------------------------------------------------
# parse_sta
# ---------------------------------------------------------------------------
class TestParseSta:

    def test_missing_file_returns_empty(self, tmp_path):
        s = parse_sta(tmp_path / "nope.sta")
        assert isinstance(s, StaProgress)
        assert s.is_ready() is False

    def test_parse_frame_row(self, tmp_path):
        p = tmp_path / "job.sta"
        p.write_text(HEADER + FRAME_ROW, encoding="latin-1")
        s = parse_sta(p)
        assert s.frame_current == 7
        assert s.frame_total == 500
        assert s.step_time == pytest.approx(7.0e-6)
        assert s.fraction() == pytest.approx(7 / 500)

    def test_parse_inc_row(self, tmp_path):
        p = tmp_path / "job.sta"
        p.write_text(HEADER + INC_ROW, encoding="latin-1")
        s = parse_sta(p)
        assert s.inc_number == 12479
        assert s.step_time == pytest.approx(7.0e-6)
        assert s.wall_time == "00:07:22"
        assert s.stable_dt == pytest.approx(5.604e-10)
        assert s.critical_elem == 16760
        assert s.kinetic_energy == pytest.approx(9.030e-6)
        assert s.total_energy == pytest.approx(1.659e-1)

    def test_last_frame_wins(self, tmp_path):
        p = tmp_path / "job.sta"
        rows = (HEADER
                + "Output Field Frame Number   1, of  500, at step time 1.000E-06\n"
                + "Output Field Frame Number   2, of  500, at step time 2.000E-06\n"
                + "Output Field Frame Number   9, of  500, at step time 9.000E-06\n")
        p.write_text(rows, encoding="latin-1")
        s = parse_sta(p)
        assert s.frame_current == 9          # most recent snapshot
        assert s.fraction() == pytest.approx(9 / 500)

    def test_mixed_rows(self, tmp_path):
        p = tmp_path / "job.sta"
        p.write_text(HEADER + FRAME_ROW + INC_ROW, encoding="latin-1")
        s = parse_sta(p)
        # Both kinds parsed into the same snapshot.
        assert s.frame_current == 7 and s.inc_number == 12479
        assert s.is_ready() is True
