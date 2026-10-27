# -*- coding: utf-8 -*-
"""
"Machine" choice of a tab that launches Abaqus runs (Job, Sensitivity,
Optimization > Model): the machines of the Preferences (gui.core.machines).

The first entry, "Default (<name>)", stores "" and follows the default
machine of the Preferences; the others store the machine's name. A name
that is no longer in the Preferences (a profile saved with a machine since
removed) stays listed as "(missing)" so the choice is not silently changed;
`run_prefs` then returns None and the tab refuses to launch.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox

from gui.core.machines import (LOCAL_NAME, default_name, host_name, names,
                                prefs_for_machine)


class MachineCombo(QComboBox):
    machineChanged = Signal(str)     # "" = the default machine

    def __init__(self, prefs_getter, parent=None):
        super().__init__(parent)
        self._prefs_getter = prefs_getter
        self._machine = ""
        self.setToolTip(
            "Where this tab's runs go. The machines are set in\n"
            "Preferences - Settings - Machines. A study keeps the machine\n"
            "it started with.")
        self.currentIndexChanged.connect(self._on_index)
        self.refresh()

    def _prefs(self):
        return self._prefs_getter() if self._prefs_getter else None

    def refresh(self) -> None:
        """Rebuild the list from the Preferences, keeping the choice."""
        prefs = self._prefs()
        all_names = names(prefs) if prefs is not None else []
        self.blockSignals(True)
        self.clear()
        self.addItem("Default (%s)" % (default_name(prefs) if prefs is not None
                                       else "?"), "")
        for n in all_names:
            self.addItem(n, n)
        if self._machine and self._machine not in all_names:
            self.addItem("%s (missing)" % self._machine, self._machine)
        i = self.findData(self._machine)
        self.setCurrentIndex(max(i, 0))
        self.blockSignals(False)

    def _on_index(self, i: int) -> None:
        if i < 0:
            return
        self._machine = self.itemData(i) or ""
        self.machineChanged.emit(self._machine)

    def machine(self) -> str:
        return self._machine

    def set_machine(self, name: str) -> None:
        self._machine = name or ""
        self.refresh()

    def machine_label(self) -> str:
        """The machine the runs would go to now, by name."""
        prefs = self._prefs()
        if prefs is None:
            return "?"
        return self._machine or default_name(prefs)

    def machine_host(self) -> str:
        """The Windows name of the PC the runs would go to now, for folder
        names; the machine's name in Preferences when its agent has not
        told it yet ("" for this PC when even that fails)."""
        prefs = self._prefs()
        if prefs is None:
            return ""
        label = self.machine_label()
        return host_name(prefs, self._machine) or (
            "" if label == LOCAL_NAME else label)

    def run_prefs(self):
        """The Preferences set for the chosen machine; None when there are
        no Preferences or the machine is missing."""
        prefs = self._prefs()
        if prefs is None:
            return None
        return prefs_for_machine(prefs, self._machine)
