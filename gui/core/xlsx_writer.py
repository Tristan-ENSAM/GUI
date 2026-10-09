# -*- coding: utf-8 -*-
"""Excel copies (.xlsx) of the CSV files the GUI exports.

Why: the CSV exports use a comma separator and a dot decimal mark (the
format numpy, pandas and csv.DictReader read everywhere). Excel opens a CSV
by double-click with the Windows regional settings, so on a French Windows
(list separator ";" and decimal mark ",") every line lands in column A.
Changing the CSV format would only move the problem to another locale and
break the readers of the files already written. An .xlsx stores numbers as
numbers, so it opens correctly by double-click whatever the locale.

So every CSV export keeps its CSV unchanged and gets a sibling .xlsx of the
same name (gci.csv -> gci.xlsx). Cells: "True"/"False" -> booleans, values
that parse as finite floats -> numbers, empty -> empty cell, anything else
-> text.

Standard library only (zipfile + XML by hand): no openpyxl needed.

Existing CSVs (written before this module) can be converted without
rerunning anything:

    python -m gui.core.xlsx_writer <folder or file.csv> [...]

A folder converts every *.csv below it.
"""
from __future__ import annotations

import csv
import logging
import math
import sys
import zipfile
from pathlib import Path
from typing import List, Sequence
from xml.sax.saxutils import escape

log = logging.getLogger(__name__)

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
    'content-types">'
    '<Default Extension="rels" ContentType="application/'
    'vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/xl/workbook.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="'
    'application/vnd.openxmlformats-officedocument.spreadsheetml.'
    'worksheet+xml"/>'
    '</Types>')
_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
    'relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
    'officeDocument/2006/relationships/officeDocument" '
    'Target="xl/workbook.xml"/>'
    '</Relationships>')
_WB_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
    'relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
    'officeDocument/2006/relationships/worksheet" '
    'Target="worksheets/sheet1.xml"/>'
    '</Relationships>')
_NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_NS_R = ('xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/'
         'relationships"')


def _col(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell(ref: str, value: str) -> str:
    if value == "":
        return ""
    if value in ("True", "False"):
        return '<c r="%s" t="b"><v>%d</v></c>' % (ref, value == "True")
    try:
        x = None if "_" in value else float(value)
    except ValueError:
        x = None
    if x is not None and math.isfinite(x):
        return '<c r="%s"><v>%s</v></c>' % (ref, repr(x))
    return ('<c r="%s" t="inlineStr"><is><t xml:space="preserve">%s</t>'
            '</is></c>' % (ref, escape(value)))


def _sheet_name(name: str) -> str:
    for ch in '[]:*?/\\':
        name = name.replace(ch, "_")
    return name[:31] or "Sheet1"


def write_xlsx(path, rows: Sequence[Sequence[object]],
               sheet: str = "Sheet1") -> Path:
    """Write `rows` (first row = header) to a one-sheet .xlsx."""
    path = Path(path)
    lines = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(_cell("%s%d" % (_col(c), r),
                              "" if v is None else str(v))
                        for c, v in enumerate(row))
        lines.append('<row r="%d">%s</row>' % (r, cells))
    pane = ('<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" '
            'topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>' if rows else "")
    sheet_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                 '<worksheet %s>%s<sheetData>%s</sheetData></worksheet>'
                 % (_NS, pane, "".join(lines)))
    wb_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
              '<workbook %s %s><sheets><sheet name="%s" sheetId="1" '
              'r:id="rId1"/></sheets></workbook>'
              % (_NS, _NS_R, escape(_sheet_name(sheet), {'"': "&quot;"})))
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("xl/workbook.xml", wb_xml)
        z.writestr("xl/_rels/workbook.xml.rels", _WB_RELS)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    tmp.replace(path)
    return path


def csv_to_xlsx(csv_path, xlsx_path=None) -> Path:
    """Convert one CSV (comma separator, dot decimals) to an .xlsx."""
    csv_path = Path(csv_path)
    if xlsx_path is None:
        xlsx_path = csv_path.with_suffix(".xlsx")
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    return write_xlsx(xlsx_path, rows, sheet=Path(xlsx_path).stem)


def excel_copy(csv_path) -> None:
    """Best-effort sibling .xlsx of an exported CSV: a failure is logged and
    never breaks the CSV export itself."""
    try:
        csv_to_xlsx(csv_path)
    except Exception:
        log.warning("could not write the Excel copy of %s", csv_path,
                    exc_info=True)


def convert_paths(paths: Sequence[str]) -> List[Path]:
    """CLI helper: convert the given CSV files and every CSV below the given
    folders. Returns the .xlsx paths written."""
    out = []
    for p in map(Path, paths):
        files = sorted(p.rglob("*.csv")) if p.is_dir() else [p]
        for f in files:
            out.append(csv_to_xlsx(f))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for x in convert_paths(sys.argv[1:]):
        print(x)
