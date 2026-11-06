# -*- coding: utf-8 -*-
"""The "Show:" filter of the parameter tables (Explore, Sensitivity,
Identify): All / Physical / Ticked, then one entry per category of the
table (the bold header rows), indented under a separator."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from gui.sensitivity import param_registry as pr

CATEGORY_PREFIX = "cat:"


def fill_filter_combo(cb: QComboBox) -> None:
    """Fill `cb` with the general filters and the registry categories."""
    cb.clear()
    for label, key in (("All", "all"), ("Physical", "physical"),
                       ("Ticked", "ticked")):
        cb.addItem(label, key)
    cb.insertSeparator(cb.count())
    for category in pr.registry_by_category():
        cb.addItem("    " + category, CATEGORY_PREFIX + category)
    cb.setToolTip(
        "All: every parameter of the model. Physical: without the "
        "numerical settings and the outputs. Ticked: the parameters of "
        "the plan. Below: the parameters of one category of the table.")


def keep(spec, mode: str, ticked: bool) -> bool:
    """Whether the row of `spec` passes the filter `mode`."""
    if mode == "physical":
        return spec.kind == "physical"
    if mode == "ticked":
        return bool(ticked)
    if mode and mode.startswith(CATEGORY_PREFIX):
        return spec.category == mode[len(CATEGORY_PREFIX):]
    return True
