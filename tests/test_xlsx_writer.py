# -*- coding: utf-8 -*-
"""Tests for gui.core.xlsx_writer (Excel copies of the CSV exports)."""
from __future__ import annotations

import csv
import zipfile

from gui.core.xlsx_writer import convert_paths, csv_to_xlsx, write_xlsx


def _sheet(path):
    with zipfile.ZipFile(path) as z:
        assert "xl/workbook.xml" in z.namelist()
        return z.read("xl/worksheets/sheet1.xml").decode("utf-8")


def test_types_numbers_booleans_text_and_blanks(tmp_path):
    p = write_xlsx(tmp_path / "a.xlsx",
                   [["h_mm", "recommended", "note", "x"],
                    ["0.0005", "True", "a<b & c", ""],
                    ["1.089e-08", "False", "nan", "1_0"]])
    xml = _sheet(p)
    assert '<c r="A2"><v>0.0005</v></c>' in xml
    assert '<c r="A3"><v>1.089e-08</v></c>' in xml
    assert '<c r="B2" t="b"><v>1</v></c>' in xml
    assert '<c r="B3" t="b"><v>0</v></c>' in xml
    assert "a&lt;b &amp; c" in xml
    assert 'r="D2"' not in xml                       # empty cell skipped
    assert "<t xml:space=\"preserve\">nan</t>" in xml  # not a number
    assert "<t xml:space=\"preserve\">1_0</t>" in xml


def test_column_letters_past_z(tmp_path):
    xml = _sheet(write_xlsx(tmp_path / "w.xlsx", [list(range(30))]))
    assert 'r="Z1"' in xml and 'r="AA1"' in xml and 'r="AD1"' in xml


def test_csv_left_unchanged_and_converted(tmp_path):
    c = tmp_path / "gci.csv"
    with open(c, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["h_mm", "f_EVF"], ["0.001", "0.81"]])
    before = c.read_bytes()
    x = csv_to_xlsx(c)
    assert x == tmp_path / "gci.xlsx" and c.read_bytes() == before
    assert '<v>0.81</v>' in _sheet(x)


def test_convert_folder(tmp_path):
    sub = tmp_path / "study"
    sub.mkdir()
    for n in ("gci.csv", "gci_meshes.csv"):
        (sub / n).write_text("a,b\n1,2\n", encoding="utf-8")
    out = convert_paths([str(tmp_path)])
    assert sorted(p.name for p in out) == ["gci.xlsx", "gci_meshes.xlsx"]


def test_study_export_writes_sibling_xlsx(tmp_path):
    from gui.sensitivity.study_export import write_csv
    p = write_csv(tmp_path / "runs.csv", [{"a": 1.5, "b": None}])
    assert p == tmp_path / "runs.csv"
    assert (tmp_path / "runs.xlsx").exists()
