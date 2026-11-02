# -*- coding: utf-8 -*-
"""Separate Simulation / Extraction / Transfer bars of a run.

Origin (Tristan, 2026-10-28): the bar of a finished run stayed at 50 % and
only jumped to 100 % once the results were extracted. Cause: from frame 1000
on, Abaqus writes "Output Field Frame Number1000" with no space, which the
.sta parser did not match, so a 2000-frame run stopped at frame 999/2000.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from gui.core.run_progress import (LogLineFilter, extraction_progress,
                                   transfer_manifest_name, transfer_progress)
from gui.core.sta_parser import parse_sta


def test_four_digit_frame_numbers_are_read(tmp_path):
    sta = tmp_path / "J.sta"
    sta.write_text(
        "Output Field Frame Number 999, of 2000, at step time 4.995E-04\n"
        "Output Field Frame Number1000, of 2000, at step time 5.000E-04\n"
        "Output Field Frame Number2000, of 2000, at step time 1.000E-03\n",
        encoding="latin-1")
    snap = parse_sta(sta)
    assert (snap.frame_current, snap.frame_total) == (2000, 2000)
    assert snap.fraction() == pytest.approx(1.0)


class TestLogLineFilter:
    def test_progress_lines_are_dropped(self):
        f = LogLineFilter()
        out = f.feed("  field 'EVF'...\n"
                     "[PROGRESS] extract 0.0100 | EVF 20/2001 frames\n"
                     "    EVF: 2001 frames read in 0h03m28s\n")
        assert "[PROGRESS]" not in out
        assert "field 'EVF'" in out and "frames read in" in out

    def test_a_line_cut_between_two_reads_is_not_leaked(self):
        f = LogLineFilter()
        assert f.feed("ok\n[PROGR") == "ok\n"
        assert f.feed("ESS] extract 0.5 | x\nnext\n") == "next\n"

    def test_flush_gives_the_unfinished_last_line(self):
        f = LogLineFilter()
        assert f.feed("no newline yet") == ""
        assert f.flush() == "no newline yet"


class TestExtractionProgress:
    def test_not_started_before_the_stage(self, tmp_path):
        log = tmp_path / "J.gui.log"
        log.write_text("[STAGE] SOLVE_START\n", encoding="latin-1")
        assert extraction_progress(log).fraction is None
        assert extraction_progress(tmp_path / "missing.log").fraction is None

    def test_last_progress_line_wins(self, tmp_path):
        log = tmp_path / "J.gui.log"
        log.write_text("[STAGE] SOLVE_DONE\n[STAGE] EXTRACT_START\n"
                       "[PROGRESS] extract 0.1000 | EVF 200/2001 frames\n"
                       "[PROGRESS] extract 0.4200 | V 900/2001 frames | "
                       "0h05m00s elapsed | ~0h07m00s left\n",
                       encoding="latin-1")
        p = extraction_progress(log)
        assert p.fraction == pytest.approx(0.42)
        assert "V 900/2001" in p.text and not p.done

    def test_done_at_extract_done(self, tmp_path):
        log = tmp_path / "J.gui.log"
        log.write_text("[STAGE] EXTRACT_START\n"
                       "[PROGRESS] extract 0.9000 | x\n"
                       "[STAGE] EXTRACT_DONE\n", encoding="latin-1")
        p = extraction_progress(log)
        assert p.done and p.fraction == 1.0


class TestTransferProgress:
    def test_no_manifest_no_progress(self, tmp_path):
        assert transfer_progress(tmp_path, "J").fraction is None

    def test_bytes_arrived_over_bytes_announced(self, tmp_path):
        (tmp_path / transfer_manifest_name("J")).write_text(
            json.dumps({"files": {"J.results.npz": 100, "J.odb": 300}}))
        (tmp_path / "J.results.npz").write_bytes(b"x" * 100)
        (tmp_path / "J.odb.tmp-1234abcd").write_bytes(b"x" * 50)
        p = transfer_progress(tmp_path, "J")
        assert p.fraction == pytest.approx(150 / 400.0)
        assert "J.odb" in p.text

    def test_leftover_temporary_files_do_not_count(self, tmp_path):
        old = tmp_path / "J.odb.tmp-deadbeef"
        old.write_bytes(b"x" * 300)
        past = time.time() - 3600
        os.utime(old, (past, past))
        (tmp_path / transfer_manifest_name("J")).write_text(
            json.dumps({"files": {"J.odb": 300}}))
        assert transfer_progress(tmp_path, "J").fraction == 0.0


def test_remote_copy_back_announces_then_removes_its_manifest(tmp_path):
    """The client's final copy (shared-drive path) writes the manifest the
    transfer bar reads, and removes it once the files are there."""
    from gui.core import remote_exec as rx
    seen = []
    proc = rx.RemoteProcess.__new__(rx.RemoteProcess)
    proc.job_name = "J"
    proc.transit = tmp_path / "transit"
    proc.transit.mkdir()
    (proc.transit / "J.results.npz").write_bytes(b"x" * 10)
    proc.run_dir = tmp_path / "run"
    proc._seen = {}
    real_copy = rx.RemoteProcess._copy_back

    def spy(self, suffixes, only_changed):
        m = self.run_dir / transfer_manifest_name("J")
        seen.append(json.loads(m.read_text()))
        real_copy(self, suffixes, only_changed)
    rx.RemoteProcess._copy_back = spy
    try:
        proc._bring_back((".results.npz", ".sta"), only_changed=False)
    finally:
        rx.RemoteProcess._copy_back = real_copy
    assert seen == [{"files": {"J.results.npz": 10}}]
    assert (proc.run_dir / "J.results.npz").read_bytes() == b"x" * 10
    assert not (proc.run_dir / transfer_manifest_name("J")).exists()


def test_agent_and_gui_agree_on_the_manifest_name():
    from gui.core import remote_agent
    assert remote_agent.transfer_manifest_name("J") == \
        transfer_manifest_name("J")


def test_bars_reset_per_stage(qapp, tmp_path):
    """The solver bar reaches 100 % on its own; the extraction bar starts
    from 0 instead of the single bar waiting at the solver's value."""
    from gui.widgets.run_progress import RunProgressBars
    sta = tmp_path / "J.sta"
    sta.write_text(
        "Output Field Frame Number2000, of 2000, at step time 1.000E-03\n",
        encoding="latin-1")
    log = tmp_path / "J.gui.log"
    log.write_text("[STAGE] EXTRACT_START\n"
                   "[PROGRESS] extract 0.2500 | EVF 500/2001 frames\n",
                   encoding="latin-1")
    bars = RunProgressBars()
    bars.start(remote=False)
    assert bars.bars["transfer"].isHidden()
    bars.poll(sta_path=sta, log_path=log, run_dir=tmp_path, job="J")
    assert bars.bars["simulation"].value() == 100
    assert bars.bars["extraction"].value() == 25
    bars.start(remote=True)
    assert bars.bars["simulation"].value() == 0
    assert not bars.bars["transfer"].isHidden()
