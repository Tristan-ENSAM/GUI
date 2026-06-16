# -*- coding: utf-8 -*-
"""
Export a sensitivity RunResult to CSV.

One row per (QoI, parameter). Scalar-QoI rows carry the full Jacobian
breakdown (sensitivity, raw dQ/dx, whether it was normalised, the base
point x0 and the base QoI value Q0); field-discrepancy QoI rows (ids
ending in " [field]") carry the SSD-based sensitivity, and the parallel
relative-change rows (ids ending in " \u0394% (rel)") carry the field's
relative change in percent, weighted over nodes and frames. The file is
sorted within each QoI by descending |sensitivity| — i.e. it doubles as
the ranking the optimisation step will consume.

Pure functions (no Qt) so they are unit-testable; the tab wires a
"Save results…" button that calls `write_csv`.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Callable, Optional

_COLUMNS = ["qoi", "parameter", "label", "sensitivity",
            "abs_sensitivity", "dQdx", "normalized", "x0", "Q0"]


def _is_field_qoi(qoi_id: str) -> bool:
    return qoi_id.endswith("[field]")


def result_rows(result, label_for: Optional[Callable[[str], str]] = None):
    """Flatten a RunResult into a list of dict rows, one per (QoI, param),
    sorted within each QoI by descending |sensitivity| (NaN last)."""
    rows = []
    for qid in result.qoi_ids:
        analysis = result.analyses.get(qid, {})
        if not isinstance(analysis, dict):
            continue
        block = []
        for path in result.param_paths:
            d = analysis.get(path)
            if not isinstance(d, dict) or "sensitivity" not in d:
                continue
            sens = float(d.get("sensitivity", float("nan")))
            row = {
                "qoi": qid,
                "parameter": path,
                "label": (label_for(path) if label_for else path),
                "sensitivity": sens,
                "abs_sensitivity": abs(sens),
                "dQdx": d.get("dQdx", ""),
                "normalized": d.get("normalized", ""),
                "x0": d.get("x0", ""),
                "Q0": d.get("Q0", ""),
            }
            block.append(row)
        block.sort(key=lambda r: (math.isnan(r["abs_sensitivity"]),
                                  -r["abs_sensitivity"]))
        rows.extend(block)
    return rows


def result_to_csv(result, label_for: Optional[Callable[[str], str]] = None
                  ) -> str:
    """Return the CSV text for a RunResult."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for row in result_rows(result, label_for):
        out = dict(row)
        # Stringify floats compactly; leave blanks as-is.
        for k in ("sensitivity", "abs_sensitivity", "dQdx", "x0", "Q0"):
            v = out.get(k, "")
            if isinstance(v, float):
                out[k] = "" if math.isnan(v) else repr(v)
        w.writerow(out)
    return buf.getvalue()


def write_csv(result, path, label_for: Optional[Callable[[str], str]] = None
              ) -> str:
    """Write the CSV to `path` (utf-8-sig so Excel shows accents). Returns
    the path written."""
    text = result_to_csv(result, label_for)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(text)
    return str(path)
