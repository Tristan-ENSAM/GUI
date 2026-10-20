# -*- coding: utf-8 -*-
"""Results tab: each history channel is plotted against its own time base.

cel_results.py writes ALLKE/ALLIE on history__ENERGY_TIME and ALLAE on
history__ALLAE_TIME, while history__time is the RF one. When the two are not
sampled alike (e.g. 23668 RF samples vs 201 energy samples), plotting the
energies against history__time raised a ValueError that aborted the run
switch, so the field viewer stayed empty.
"""
import numpy as np

from gui.tabs.results_tab import _history_time_for


class _Bundle:
    def __init__(self, arrays):
        self._a = arrays

    def history(self, var):
        return self._a[var]


def test_energy_uses_its_own_time_base():
    rf_t = np.linspace(0, 1, 23668)
    en_t = np.linspace(0, 1, 201)
    b = _Bundle({"ENERGY_TIME": en_t, "ALLAE_TIME": en_t})
    assert np.array_equal(_history_time_for(b, "ALLKE", rf_t, 201), en_t)
    assert _history_time_for(b, "ALLAE", rf_t, 201).size == 201
    assert _history_time_for(b, "RF1_RP", rf_t, 23668).size == 23668


def test_no_matching_time_base_returns_none():
    b = _Bundle({})
    assert _history_time_for(b, "ALLKE", np.zeros(10), 5) is None
    # Older bundles without ENERGY_TIME: generic time when lengths match.
    assert _history_time_for(b, "ALLKE", np.zeros(5), 5).size == 5
