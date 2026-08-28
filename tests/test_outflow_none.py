# -*- coding: utf-8 -*-
"""The NONE Eulerian-outflow option (UI availability + config serialisation).
The run_simul behaviour for NONE (dropping OUTFLOW from the BC definition) runs
under Abaqus and is not exercised here."""
from __future__ import annotations


from gui.core.model_config import ModelConfig


def test_none_is_an_outflow_option(qapp):
    from gui.tabs.bcs_tab import BCsTab
    values = [v for _label, v in BCsTab.OUTFLOW_OPTIONS]
    assert "NONE" in values
    # the previous options are still present
    for v in ("FREE", "NONREFLECTING", "EQUILIBRIUM", "ZERO_PRESSURE"):
        assert v in values


def test_none_round_trips_through_params():
    c = ModelConfig()
    c.bcs.eulerian_outflow_right = "NONE"
    assert c.to_params_dict()["bcs"]["eulerian_outflow_right"] == "NONE"
