# -*- coding: utf-8 -*-
"""Machine list of the Preferences (gui.core.machines), the Preferences
dialog's Machines group, the tabs' machine choice and its record in the
study folder."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import pytest

from gui.core import machines as M
from gui.core.preferences import Preferences
from gui.sensitivity.study_specs import record_study_machine


def _prefs(**kw):
    return replace(Preferences(), **kw)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def test_fresh_prefs_have_only_this_pc():
    p = _prefs()
    assert M.names(p) == [M.LOCAL_NAME]
    assert M.default_name(p) == M.LOCAL_NAME
    q = M.prefs_for_machine(p)
    assert q.execution_mode == "local"


def test_old_remote_prefs_become_a_machine_and_the_default():
    p = _prefs(execution_mode="remote", remote_queue_dir=r"Z:\ABQ_queue",
               remote_abaqus_cmd=r"C:\SIMULIA\Commands\abaqus.bat")
    assert M.names(p) == [M.LOCAL_NAME, M.MIGRATED_NAME]
    assert M.default_name(p) == M.MIGRATED_NAME
    q = M.prefs_for_machine(p)
    assert q.execution_mode == "remote"
    assert q.remote_queue_dir == r"Z:\ABQ_queue"
    assert M.machine_name(q) == M.MIGRATED_NAME
    # Written out, the list keeps the same machine and default.
    w = M.migrate(p)
    assert w.machines == [{"name": M.MIGRATED_NAME, "kind": "queue",
                           "queue_dir": r"Z:\ABQ_queue",
                           "abaqus_cmd": r"C:\SIMULIA\Commands\abaqus.bat"}]
    assert w.default_machine == M.MIGRATED_NAME


def test_old_local_prefs_with_a_queue_folder_keep_this_pc_default():
    p = _prefs(execution_mode="local", remote_queue_dir=r"Z:\q")
    assert M.default_name(p) == M.LOCAL_NAME
    assert M.prefs_for_machine(p, M.MIGRATED_NAME).execution_mode == "remote"


def test_empty_list_means_no_queue_machine():
    # A list emptied in the dialog is not rebuilt from the old fields.
    p = _prefs(machines=[], remote_queue_dir=r"Z:\q", execution_mode="local")
    assert M.names(p) == [M.LOCAL_NAME]


def test_each_machine_gets_its_own_queue():
    p = _prefs(machines=[
        {"name": "CL-WTS-1", "kind": "queue", "queue_dir": r"Z:\a",
         "abaqus_cmd": "a.bat"},
        {"name": "Other", "kind": "queue", "queue_dir": r"Z:\b",
         "abaqus_cmd": "b.bat"}], default_machine="CL-WTS-1",
        execution_mode="remote", remote_queue_dir=r"Z:\a")
    a = M.prefs_for_machine(p, "CL-WTS-1")
    b = M.prefs_for_machine(p, "Other")
    loc = M.prefs_for_machine(p, M.LOCAL_NAME)
    assert (a.remote_queue_dir, a.remote_abaqus_cmd) == (r"Z:\a", "a.bat")
    assert (b.remote_queue_dir, b.remote_abaqus_cmd) == (r"Z:\b", "b.bat")
    assert loc.execution_mode == "local"
    assert M.machine_name(b) == "Other"
    # The user's preferences are not modified.
    assert p.remote_queue_dir == r"Z:\a"


def test_unknown_machine_gives_none():
    assert M.prefs_for_machine(_prefs(), "Gone") is None


def test_bad_entries_are_dropped():
    p = _prefs(machines=[
        "junk", {"kind": "queue"}, {"name": "X", "kind": "ssh"},
        {"name": M.LOCAL_NAME, "kind": "queue"},
        {"name": "A", "queue_dir": "q1"}, {"name": "A", "queue_dir": "q2"}])
    assert M.names(p) == [M.LOCAL_NAME, "A"]
    assert M.find(p, "A").queue_dir == "q1"


def test_unknown_default_falls_back():
    p = _prefs(machines=[{"name": "A", "queue_dir": "q"}],
               default_machine="Gone", execution_mode="local")
    assert M.default_name(p) == M.LOCAL_NAME


# ---------------------------------------------------------------------------
# study folder record
# ---------------------------------------------------------------------------
def test_record_study_machine(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"study": "ms"}))
    assert record_study_machine(tmp_path, "CL-WTS-1") == []
    assert record_study_machine(tmp_path, "CL-WTS-1") == ["CL-WTS-1"]
    assert record_study_machine(tmp_path, "Cassiopee") == ["CL-WTS-1"]
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["machines"] == ["CL-WTS-1", "Cassiopee"]
    assert data["study"] == "ms"


def test_record_study_machine_without_config(tmp_path):
    assert record_study_machine(tmp_path, "X") is None


# ---------------------------------------------------------------------------
# widgets
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_machine_combo_keeps_choice_and_flags_missing(qapp):
    from gui.widgets.machine_combo import MachineCombo
    box = {"p": _prefs(machines=[{"name": "A", "queue_dir": "qa"}],
                       default_machine="A")}
    cmb = MachineCombo(lambda: box["p"])
    assert cmb.count() == 3                  # Default (A), This PC, A
    assert cmb.itemText(0) == "Default (A)"
    assert cmb.run_prefs().remote_queue_dir == "qa"
    cmb.set_machine(M.LOCAL_NAME)
    assert cmb.run_prefs().execution_mode == "local"
    cmb.set_machine("A")
    box["p"] = _prefs(machines=[])
    cmb.refresh()
    assert cmb.machine() == "A"
    assert cmb.currentText() == "A (missing)"
    assert cmb.run_prefs() is None


def test_preferences_dialog_round_trip(qapp):
    from gui.widgets.preferences_dialog import PreferencesDialog
    p = _prefs(execution_mode="remote", remote_queue_dir=r"Z:\q",
               remote_abaqus_cmd="r.bat")
    dlg = PreferencesDialog(None, p)
    assert dlg.tbl_machines.rowCount() == 1
    assert dlg.cmb_default_machine.currentText() == M.MIGRATED_NAME
    dlg._on_add_machine()
    r = dlg.tbl_machines.rowCount() - 1
    from PySide6.QtWidgets import QTableWidgetItem
    dlg.tbl_machines.setItem(r, 1, QTableWidgetItem(r"Z:\other"))
    assert dlg.machine_problems() == []
    out = dlg.result_prefs()
    assert [m["name"] for m in out.machines] == [M.MIGRATED_NAME, "Machine 1"]
    assert out.default_machine == M.MIGRATED_NAME
    assert out.execution_mode == "remote"
    # Default set to this PC: the execution fields say local.
    dlg.cmb_default_machine.setCurrentText(M.LOCAL_NAME)
    assert dlg.result_prefs().execution_mode == "local"
    # Duplicate names and a missing queue folder are refused.
    dlg.tbl_machines.setItem(r, 0, QTableWidgetItem(M.MIGRATED_NAME))
    dlg.tbl_machines.setItem(r, 1, QTableWidgetItem(""))
    probs = dlg.machine_problems()
    assert any("Two machines" in s for s in probs)
    assert any("no queue folder" in s for s in probs)
