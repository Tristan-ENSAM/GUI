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

Morris results have their own columns (mu_star, sigma, mu, mu_star_conf and
the trajectories used / total), sorted by mu_star; a QoI that could not be
analysed gets one row per parameter with the reason in `note`.

Pure functions (no Qt) so they are unit-testable; the tab wires a
"Save results…" button that calls `write_csv`.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Callable, Optional

from gui.core.xlsx_writer import excel_copy

_COLUMNS = ["qoi", "parameter", "label", "sensitivity",
            "abs_sensitivity", "dQdx", "elasticity", "normalized",
            "raw_fallback", "scheme_used", "x0", "Q0"]
_MORRIS_COLUMNS = ["qoi", "parameter", "label", "mu_star", "sigma", "mu",
                   "mu_star_conf", "trajectories_used", "trajectories_total",
                   "note"]
_FLOAT_KEYS = ("sensitivity", "abs_sensitivity", "dQdx", "elasticity",
               "x0", "Q0", "mu_star", "sigma", "mu", "mu_star_conf")


def _morris_rows(result, label_for):
    rows = []
    for qid in result.qoi_ids:
        a = result.analyses.get(qid, {})
        if not isinstance(a, dict):
            continue
        used = a.get("n_used", "")
        total = a.get("n_trajectories", "")
        if "error" in a:
            for path in result.param_paths:
                rows.append({"qoi": qid, "parameter": path,
                             "label": label_for(path) if label_for else path,
                             "trajectories_used": used,
                             "trajectories_total": total,
                             "note": "not analysed: %s" % a["error"]})
            continue
        names = list(a.get("names", []))
        block = []
        for k, path in enumerate(names):
            def at(key):
                arr = a.get(key)
                try:
                    return float(arr[k])
                except (TypeError, IndexError, ValueError):
                    return float("nan")
            block.append({"qoi": qid, "parameter": path,
                          "label": label_for(path) if label_for else path,
                          "mu_star": at("mu_star"), "sigma": at("sigma"),
                          "mu": at("mu"), "mu_star_conf": at("mu_star_conf"),
                          "trajectories_used": used,
                          "trajectories_total": total, "note": ""})
        block.sort(key=lambda r: (math.isnan(r["mu_star"]), -r["mu_star"]))
        rows.extend(block)
    return rows


def result_rows(result, label_for: Optional[Callable[[str], str]] = None):
    """Flatten a RunResult into a list of dict rows, one per (QoI, param),
    sorted within each QoI by descending |sensitivity| (NaN last) -- or by
    descending mu_star for a Morris result."""
    if getattr(result, "plan_kind", "jacobian") == "morris":
        return _morris_rows(result, label_for)
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
                "elasticity": d.get("elasticity", ""),
                "raw_fallback": d.get("raw_fallback", ""),
                "scheme_used": d.get("scheme_used", ""),
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
    cols = (_MORRIS_COLUMNS
            if getattr(result, "plan_kind", "jacobian") == "morris"
            else _COLUMNS)
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for row in result_rows(result, label_for):
        out = dict(row)
        # Stringify floats compactly; leave blanks as-is.
        for k in _FLOAT_KEYS:
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
    if str(path).lower().endswith(".csv"):
        excel_copy(path)
    return str(path)
