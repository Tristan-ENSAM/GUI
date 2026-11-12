# -*- coding: utf-8 -*-
"""Help tab: the glossary of the abbreviations used in the GUI.

Compare runs and the run names use short symbols (Tristan 2026-11-12: names
with abbreviations, explained in a glossary under Help). The parameter
rows come from gui.sensitivity.param_registry (SYMBOLS, labels, units), so
the glossary follows the registry; the other rows are written here.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHeaderView, QLabel, QLineEdit, QTabWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from gui.sensitivity import param_registry as pr

COLUMNS = ("Symbol", "Meaning", "Unit", "Group")

# Quantities compared between runs (Model sizing, Explore, Sensitivity,
# Identify, Compare runs).
QOI_ROWS = [
    ("Vx", "Velocity along the cutting direction, mean over the ZOI and T",
     "mm/s"),
    ("Vy", "Velocity normal to the cutting direction, mean over the ZOI "
     "and T", "mm/s"),
    ("T", "Temperature of the workpiece material, mean over the ZOI and T",
     "K"),
    ("EVF", "Eulerian volume fraction (share of material in an element)",
     "–"),
    ("Fc", "Cutting force per unit width (RF1 of the tool / width), mean "
     "over T", "N/mm"),
    ("Ff", "Feed (thrust) force per unit width (RF2 / width), mean over T",
     "N/mm"),
    ("e", "Chip thickness on the line y0, mean over T", "mm"),
]

# Settings and results of the studies.
STUDY_ROWS = [
    ("ZOI", "Comparison zone: where two runs are compared", "mm"),
    ("ROI", "Output zone written by each run (matched to DIC / IRT)", "mm"),
    ("T (window)", "Comparison window: part of the simulated time over "
     "which the means are taken", "fraction of Ttot"),
    ("ε_q", "Admitted deviation of quantity q", "unit of q"),
    ("E_max", "Largest |mean(a) − mean(b)| / ε_q over the "
     "quantities; a pair passes when E_max < 1", "–"),
    ("ms*", "Mass-scaling factor chosen by the ms study", "–"),
    ("h*", "Element size chosen by the mesh study (GCI)", "mm"),
    ("D", "Eulerian domain: h_wp/h_void/l_wp/l_void", "mm"),
    ("D*", "Domain chosen by the domain study", "mm"),
    ("GCI", "Grid convergence index (Richardson extrapolation on the mesh)",
     "–"),
    ("p", "Observed order of convergence (Richardson)", "–"),
    ("t_s", "Settling time of a quantity (stationarity check P0)", "s"),
    ("L", "Length of the moving mean of the stationarity check", "µs"),
    ("τ", "Settling band of the stationarity check, in ε_q", "–"),
    ("C_CPU", "Cost of a run: number of CPUs × solver wall time",
     "core·s"),
    ("R_K", "Kinetic energy over internal energy (quasi-static guard)",
     "–"),
    ("R_HG", "Hourglass energy over internal energy (guard)", "–"),
]

# Parts of the run and folder names.
NAME_ROWS = [
    ("<study>_runNNN", "Run number NNN of a study or plan folder", ""),
    ("ms_run", "Run of the mass-scaling study (tag msN = its factor)", ""),
    ("GCI_run", "Run of the mesh study (tag hNum = element size in µm)",
     ""),
    ("domainsizing_run", "Run of the domain study (tag D<l_wp>x<h_wp>x"
     "<l_void>x<h_void> in µm)", ""),
    ("checks_run", "Run of the final checks (tag msN_hNum)", ""),
    ("sweep_run / grid_run", "Run of an Explore plan (sweep: one parameter "
     "at a time; grid: all combinations)", ""),
    ("sensitivity_run", "Run of a Sensitivity plan (Jacobian, Morris)", ""),
    ("identify_run", "Run of an identification", ""),
    ("(Ref)", "The reference run of a plan (every parameter at its model "
     "value)", ""),
]


def glossary_rows(temp_unit: str = "C"):
    """[(symbol, meaning, unit, group)], parameters first (registry order)."""
    rows = []
    for spec in pr.REGISTRY:
        rows.append((pr.symbol_of(spec.path), spec.label,
                     spec.unit_str(temp_unit), spec.category))
    rows += [(s, m, u, "Quantities compared") for s, m, u in QOI_ROWS]
    rows += [(s, m, u, "Studies") for s, m, u in STUDY_ROWS]
    rows += [(s, m, u, "Run names") for s, m, u in NAME_ROWS]
    return rows


class GlossaryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        intro = QLabel(
            "Abbreviations used in the run names, in Compare runs and in the "
            "tables. Type in the box to filter.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.le_search = QLineEdit()
        self.le_search.setPlaceholderText("Filter (symbol, meaning, group)")
        self.le_search.textChanged.connect(self._filter)
        lay.addWidget(self.le_search)
        rows = glossary_rows()
        self.table = QTableWidget(len(rows), len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                it = QTableWidgetItem(text)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(r, c, it)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        lay.addWidget(self.table, 1)

    def _filter(self, text):
        words = text.strip().lower()
        for r in range(self.table.rowCount()):
            hay = " ".join(self.table.item(r, c).text().lower()
                           for c in range(self.table.columnCount()))
            self.table.setRowHidden(r, bool(words) and words not in hay)


class HelpTab(QTabWidget):
    """Top-level Help tab; one sub-tab today: Glossary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.glossary = GlossaryPanel()
        self.addTab(self.glossary, "Glossary")
