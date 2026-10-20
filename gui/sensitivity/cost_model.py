# -*- coding: utf-8 -*-
"""Predicted solver time of a run: t = a + b * N_inc (decision of 2026-10-20).

On one mesh and one domain, the solver wall time of the runs of a study
follows t = a + b * N_inc: `a` is the fixed part (start-up, output frames),
`b` the time per increment. The fit is a least squares on the RELATIVE
residual, iterated: a run more than `tol` away from the fit is left out and
the fit is redone. Measured on CL-WTS-1 (ms studies of 2026-10-15/19, 20 CPU):
4 um a = 103 s, b = 4.6 ms, every run within 7 %; 2 um a = 820 s,
b = 11.2 ms, two runs 2.7 and 5.9 times slower (another process on the
shared machine), the others within -12 / +20 %.

The prediction says what a run costs on a free machine; a run far above it
was slowed by something else, and its C_CPU is not representative.

N_inc of an ms value not run yet: the stable increment grows as sqrt(ms) on
a fixed mesh, so N_inc(ms) = N_inc(ms_ref) * sqrt(ms_ref / ms).

Pure module (no Qt).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# A run more than 30 % away from the fit is left out of it (2026-10-20: the
# runs on a free machine were within -12 / +20 %).
OUTLIER_TOL = 0.30


@dataclass
class CostFit:
    a: float                      # fixed part [s]
    b: float                      # time per increment [s]
    kept: List[bool] = field(default_factory=list)   # runs used in the fit

    def predict(self, n_inc: float) -> float:
        return self.a + self.b * float(n_inc)


def fit_cost(n_inc: Sequence[float], t: Sequence[float],
             tol: float = OUTLIER_TOL) -> Optional[CostFit]:
    """Fit t = a + b * N_inc on the runs of one mesh and domain. None with
    fewer than two usable runs (or two with the same N_inc)."""
    n = np.asarray([float(v) if v is not None else np.nan for v in n_inc])
    tt = np.asarray([float(v) if v is not None else np.nan for v in t])
    ok = np.isfinite(n) & np.isfinite(tt) & (n > 0) & (tt > 0)
    keep = ok.copy()
    fit = None
    for _ in range(20):
        if keep.sum() < 2 or np.unique(n[keep]).size < 2:
            break
        A = np.vstack([np.ones(keep.sum()), n[keep]]).T / tt[keep, None]
        a, b = np.linalg.lstsq(A, np.ones(keep.sum()), rcond=None)[0]
        fit = (float(a), float(b))
        pred = a + b * n
        with np.errstate(divide="ignore", invalid="ignore"):
            new = ok & (pred > 0) & (np.abs(tt / pred - 1.0) <= tol)
        if (new == keep).all():
            break
        keep = new
    if fit is None:
        return None
    return CostFit(a=fit[0], b=fit[1], kept=[bool(k) for k in keep])


def n_inc_at_ms(ms: float, ref_ms: Sequence[float],
                ref_n_inc: Sequence[float]) -> Optional[float]:
    """N_inc expected at `ms` from the runs (ref_ms, ref_n_inc) of the same
    mesh and domain: the run of the nearest ms (in log), scaled by
    sqrt(ms_ref / ms)."""
    best = None
    for m, n in zip(ref_ms, ref_n_inc):
        if not (m and n and m > 0 and n > 0 and ms > 0):
            continue
        d = abs(math.log(m / ms))
        if best is None or d < best[0]:
            best = (d, m, n)
    if best is None:
        return None
    return best[2] * math.sqrt(best[1] / ms)


def remaining_from_progress(elapsed_s: Optional[float],
                            fraction: Optional[float]) -> Optional[float]:
    """Remaining solver time of a running job, from its elapsed wall time
    and the fraction of the simulated time done (linear in the frames)."""
    if elapsed_s is None or fraction is None or not (0.02 <= fraction < 1.0):
        return None
    return float(elapsed_s) * (1.0 - fraction) / fraction


def fmt_duration(seconds: Optional[float]) -> str:
    """'45 s', '12 min', '3 h 05', '2 d 04 h'."""
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    s = max(0.0, float(seconds))
    if s < 90:
        return "%d s" % round(s)
    if s < 5400:
        return "%d min" % round(s / 60.0)
    h = s / 3600.0
    if h < 48:
        return "%d h %02d" % (int(h), round((h - int(h)) * 60) % 60)
    d = int(h // 24)
    return "%d d %02d h" % (d, round(h - 24 * d) % 24)
