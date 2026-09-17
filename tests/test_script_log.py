# -*- coding: utf-8 -*-
"""The file-based diagnostic channel that replaces the lost stdout (M7).

WHY THIS FILE EXISTS
--------------------
`abaqus cae noGUI=` runs run_simul.py inside a separate kernel process
(ABQcaeK.exe) whose stdout reaches nobody: not the Job tab, not even a plain
`cmd` redirection. A probe established this, and the file channel is what made
the probe able to report at all. Sixty-nine diagnostic messages were being
written for no reader, including the one that finally diagnosed M4:

    [WARNING] MASSEUL/VOLEUL history not created: Invalid variables are
    specified in an output request.

So run_simul.py now tees stdout/stderr into `<job>.gui.log` and the Job tab
tails it. These tests cover both halves of that contract, headless:
run_simul.py imports only the standard library at module level, and the tail
logic is exercised against a real file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_RUN_SIMUL = _REPO / "abaqus_scripts" / "run_simul.py"


@pytest.fixture(scope="module")
def run_simul():
    """Import abaqus_scripts/run_simul.py directly.

    It is Python 2.7 source for Abaqus, but the module level holds only
    sys/os/argparse/ast, so it imports fine under the test interpreter. Only
    main() needs Abaqus, and nothing here calls it.
    """
    spec = importlib.util.spec_from_file_location("run_simul_under_test",
                                                  _RUN_SIMUL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules.pop("run_simul_under_test", None)
    return module


class TestLogPath:
    """Both sides must derive the SAME path or the GUI tails nothing."""

    def test_uses_the_job_name_and_the_agreed_suffix(self, run_simul, tmp_path):
        got = run_simul.log_path_for("Cutting_job", str(tmp_path))
        assert Path(got) == tmp_path / "Cutting_job.gui.log"

    def test_matches_what_the_job_tab_builds(self, run_simul, tmp_path):
        # job_tab._launch_abaqus does: wd / f"{job_name}.gui.log"
        job_name = "TEST_0"
        gui_side = tmp_path / ("%s.gui.log" % job_name)
        script_side = Path(run_simul.log_path_for(job_name, str(tmp_path)))
        assert gui_side == script_side

    def test_defaults_to_the_working_directory(self, run_simul, monkeypatch,
                                               tmp_path):
        # The GUI sets the child's cwd to the working directory, so an absent
        # workdir argument must still land beside the job.
        monkeypatch.chdir(tmp_path)
        assert Path(run_simul.log_path_for("J")).parent == tmp_path


class TestTee:
    """The tee must reach the file even when the original stream misbehaves —
    that stream is the one that goes nowhere under Abaqus."""

    def test_writes_to_both_targets(self, run_simul, tmp_path):
        target = tmp_path / "log.txt"
        captured = []

        class _Fake:
            def write(self, text):
                captured.append(text)

            def flush(self):
                pass

        with open(target, "w") as handle:
            tee = run_simul._Tee(handle, _Fake())
            tee.write("[META] hello\n")
        assert target.read_text() == "[META] hello\n"
        assert captured == ["[META] hello\n"]

    def test_a_broken_original_stream_does_not_lose_the_file(self, run_simul,
                                                             tmp_path):
        target = tmp_path / "log.txt"

        class _Broken:
            def write(self, text):
                raise IOError("stdout is not connected")

            def flush(self):
                raise IOError("nor can it flush")

        with open(target, "w") as handle:
            tee = run_simul._Tee(handle, _Broken())
            tee.write("survives\n")
            tee.flush()
        assert target.read_text() == "survives\n"

    def test_flushes_every_write(self, run_simul, tmp_path):
        """The GUI tails this file DURING the run; buffered writes would make
        the live log arrive only at the end, which is the whole problem."""
        target = tmp_path / "log.txt"
        with open(target, "w") as handle:
            tee = run_simul._Tee(handle, handle)
            tee.write("visible immediately\n")
            # Read it back while the writer is still open.
            assert "visible immediately" in target.read_text()


class TestJobTabTailsTheLog:
    """The GUI half: append only what is new, never replay the panel."""

    @pytest.fixture
    def tab(self, qapp, tmp_path):
        from gui.core.model_config import ModelConfig
        from gui.core.preferences import Preferences
        from gui.tabs.job_tab import JobTab
        t = JobTab(ModelConfig(), lambda: Preferences())
        t._pipeline = {"log_path": tmp_path / "J.gui.log"}
        t._log_offset = 0
        return t

    def test_absent_log_is_not_an_error(self, tab):
        # The script has not created it yet on the first ticks.
        tab._poll_script_log()
        assert tab.txt_output.toPlainText() == ""

    def test_appends_only_the_new_bytes(self, tab):
        log = tab._pipeline["log_path"]
        log.write_text("[META] first\n", encoding="latin-1")
        tab._poll_script_log()
        assert "[META] first" in tab.txt_output.toPlainText()

        with open(log, "a", encoding="latin-1") as handle:
            handle.write("[STAGE] second\n")
        tab._poll_script_log()

        panel = tab.txt_output.toPlainText()
        assert "[STAGE] second" in panel
        # The whole point of the offset: "first" must appear exactly once.
        assert panel.count("[META] first") == 1

    def test_a_quiet_tick_appends_nothing(self, tab):
        log = tab._pipeline["log_path"]
        log.write_text("only line\n", encoding="latin-1")
        tab._poll_script_log()
        before = tab.txt_output.toPlainText()
        tab._poll_script_log()
        assert tab.txt_output.toPlainText() == before

    def test_non_ascii_does_not_break_the_live_log(self, tab):
        """Abaqus messages are not guaranteed ASCII; a decode error here used
        to be the kind of thing that silently kills a polling loop."""
        log = tab._pipeline["log_path"]
        log.write_bytes(b"\xe9chec du solveur\n")
        tab._poll_script_log()
        assert "chec du solveur" in tab.txt_output.toPlainText()
