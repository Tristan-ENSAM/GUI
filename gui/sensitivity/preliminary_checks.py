# -*- coding: utf-8 -*-
"""Preliminary checks on ONE finished run, before the sizing studies.

They fix what the studies take for granted (thread of 2026-10-23, report
reports/sequence-preliminaire-zoi-temps.md), from the saved results only:

P0  Simulated time and window T. For Fc, Ff (history) and the area means of
    Vx, Vy, T, EVF over a zone (per frame), the trailing moving mean over a
    length L is computed; a quantity is settled from the first time t_s after
    which its moving mean stays within a band of width tau * eps_q (range of
    the moving mean from t_s to the end <= tau * eps_q). The run is settled
    from the largest t_s; the proposed window is [t_s, t_end]. It is flagged
    too short when t_end - t_s < 2 L (choice: two moving-mean lengths).

P1  Output frames. The field filter is a Butterworth of order 2
    (abaqus_scripts/cel_model.py, ButterworthFilter order=2): content above
    the Nyquist frequency of the frames is attenuated, not removed, and
    aliases. The attenuation at that Nyquist frequency is
    A = 10 log10(1 + (f_N / f_c)^(2 n)). A subsampling test also compares the
    window means over the zone with every frame and with one frame in k.

Chip re-contact. With a too high Eulerian domain the chip can curl back
    onto the uncut surface. The uncut surface is taken per column from the
    first frame (top material element, EVF >= threshold, upstream of the tool
    tip minus a margin); the cells k element sizes above that surface cell,
    gap <= k < gap + band, are watched, and the first frame where one of them
    holds material (EVF >= threshold) is reported.

Pure host-side Python; nothing is run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from gui.sensitivity.domain_independence import (
    FIELD_QUANTITIES, FORCE_QUANTITIES, MASKED_QUANTITIES)
from gui.sensitivity.runner_core import eulerian_instance
from gui.sensitivity.zoi_sampling import (
    _spacing, area_mean, element_centroids_xy, window_mask, zoi_area_weights)

QUANTITIES = ("Vx", "Vy", "T", "EVF", "Fc", "Ff")
FILTER_ORDER = 2          # cel_model.py ButterworthFilter(order=2)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def _cfg(bundle, *keys, default=None):
    d = getattr(bundle, "model_config", None) or {}
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def elem_size(bundle) -> float:
    try:
        return float(_cfg(bundle, "mesh", "elem_size"))
    except (TypeError, ValueError):
        return float("nan")


def zone_series(bundle, zone, evf_threshold: float = 0.5,
                quantities: Sequence[str] = ("Vx", "Vy", "T", "EVF")
                ) -> Dict[str, np.ndarray]:
    """{q: (n_frames,)} area mean over the zone at each frame. Vx, Vy, T only
    over the elements holding material at that frame (EVF >= threshold);
    NaN at a frame with no such element. Missing fields are left out."""
    inst = eulerian_instance(bundle)
    if not inst:
        raise RuntimeError("No Eulerian instance (EVF carrier) in the run.")
    h = elem_size(bundle)
    idx, w = zoi_area_weights(element_centroids_xy(bundle, inst), zone,
                              h if math.isfinite(h) else None)
    if idx.size == 0:
        raise RuntimeError("The zone x[%g, %g] y[%g, %g] holds no element of "
                           "the extracted results." % tuple(zone))
    evf = np.asarray(bundle.field(inst, FIELD_QUANTITIES["EVF"]),
                     dtype=float)[:, idx]
    material = evf >= float(evf_threshold)
    out = {}
    for q in quantities:
        try:
            v = np.asarray(bundle.field(inst, FIELD_QUANTITIES[q]),
                           dtype=float)[:, idx]
        except KeyError:
            continue
        ok = np.isfinite(v)
        if q in MASKED_QUANTITIES:
            ok &= material
        ww = np.where(ok, w[None, :], 0.0)
        s = ww.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[q] = np.where(s > 0, (ww * np.where(ok, v, 0.0)).sum(axis=1)
                              / np.where(s > 0, s, 1.0), np.nan)
    return out


def force_series(bundle) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """(history time, {Fc, Ff: F / w in N/mm}); w = element size of the run."""
    t = np.asarray(bundle.history_time, dtype=float)
    w = elem_size(bundle)
    out = {}
    for q, ch in FORCE_QUANTITIES.items():
        try:
            f = np.asarray(bundle.history(ch), dtype=float)
        except Exception:
            continue
        if f.size == t.size and math.isfinite(w) and w > 0:
            out[q] = f / w
    return t, out


# ---------------------------------------------------------------------------
# P0: moving mean and settling time
# ---------------------------------------------------------------------------
def moving_mean(t, y, length: float) -> np.ndarray:
    """Trailing moving mean of the samples in [t_i - length, t_i]; NaN until
    a full length is available (t_i - t_0 < length) and where y is NaN."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, y, 0.0))])
    n = np.concatenate([[0], np.cumsum(ok)])
    lo = np.searchsorted(t, t - float(length), side="left")
    hi = np.arange(1, t.size + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (c[hi] - c[lo]) / cnt
    m[(cnt == 0) | (t - t[0] < float(length) - 1e-15 * max(1.0, t[-1]))] = \
        np.nan
    return m


def settling_index(m, tol: float) -> Optional[int]:
    """First index i with a finite m[i] such that max(m[i:]) - min(m[i:])
    <= tol over the finite values; None if there is none."""
    m = np.asarray(m, dtype=float)
    fin = np.isfinite(m)
    if not fin.any():
        return None
    hi = np.where(fin, m, -np.inf)[::-1]
    lo = np.where(fin, m, np.inf)[::-1]
    smax = np.maximum.accumulate(hi)[::-1]
    smin = np.minimum.accumulate(lo)[::-1]
    good = fin & (smax - smin <= float(tol))
    if not good.any():
        return None
    return int(np.argmax(good))


@dataclass
class Settling:
    quantity: str
    times: np.ndarray
    values: np.ndarray
    mean: np.ndarray                  # trailing moving mean
    tol: float                        # tau * eps_q
    t_s: Optional[float]              # None: not settled in the run
    final: float                      # last finite moving-mean value


@dataclass
class StationarityResult:
    length: float
    tau: float
    per_quantity: Dict[str, Settling] = field(default_factory=dict)
    t_end: float = float("nan")

    @property
    def t_s(self) -> Optional[float]:
        """Largest settling time; None when one quantity never settles."""
        ts = [s.t_s for s in self.per_quantity.values()]
        if not ts or any(v is None for v in ts):
            return None
        return max(ts)

    @property
    def window(self) -> Optional[Tuple[float, float]]:
        """Proposed window T as fractions of the simulated time."""
        ts = self.t_s
        if ts is None or not self.t_end > 0:
            return None
        return (ts / self.t_end, 1.0)

    @property
    def too_short(self) -> bool:
        ts = self.t_s
        return ts is None or (self.t_end - ts) < 2.0 * self.length


def stationarity(bundle, zone, length: float, tau: float, eps: dict,
                 evf_threshold: float = 0.5,
                 quantities: Sequence[str] = QUANTITIES
                 ) -> StationarityResult:
    """P0 on one run (see the module docstring). `length` in seconds,
    `tau` a fraction of eps_q (the band is tau * eps_q wide)."""
    res = StationarityResult(length=float(length), tau=float(tau))
    series = {}
    fields = [q for q in quantities if q in FIELD_QUANTITIES]
    if fields:
        times = np.asarray(bundle.times, dtype=float)
        for q, v in zone_series(bundle, zone, evf_threshold, fields).items():
            series[q] = (times, v)
        res.t_end = float(times[-1]) if times.size else float("nan")
    if any(q in FORCE_QUANTITIES for q in quantities):
        ht, forces = force_series(bundle)
        for q, v in forces.items():
            if q in quantities:
                series[q] = (ht, v)
        if ht.size and not math.isfinite(res.t_end):
            res.t_end = float(ht[-1])
    for q in quantities:
        if q not in series or not (eps.get(q) or 0) > 0:
            continue
        t, v = series[q]
        m = moving_mean(t, v, length)
        tol = float(tau) * float(eps[q])
        i = settling_index(m, tol)
        fin = m[np.isfinite(m)]
        res.per_quantity[q] = Settling(
            q, t, v, m, tol, None if i is None else float(t[i]),
            float(fin[-1]) if fin.size else float("nan"))
    return res


# ---------------------------------------------------------------------------
# P1: frames
# ---------------------------------------------------------------------------
def butterworth_attenuation_db(f: float, fc: float,
                               order: int = FILTER_ORDER) -> float:
    """Attenuation of a Butterworth at frequency f: 10 log10(1+(f/fc)^2n)."""
    return 10.0 * math.log10(1.0 + (float(f) / float(fc)) ** (2 * order))


def frame_interval_for(fc: float, atten_db: float,
                       order: int = FILTER_ORDER) -> float:
    """Largest frame interval whose Nyquist frequency sees `atten_db` of
    attenuation: f_N = fc (10^(A/10) - 1)^(1/2n), dt = 1 / (2 f_N)."""
    r = (10.0 ** (float(atten_db) / 10.0) - 1.0) ** (1.0 / (2 * order))
    return 1.0 / (2.0 * r * float(fc))


@dataclass
class FrameCheck:
    dt: float                        # median frame interval [s]
    f_nyquist: float                 # 1 / (2 dt) [Hz]
    cutoff: Optional[float]          # field filter cutoff [Hz], None if off
    atten_db: Optional[float]        # attenuation at f_nyquist
    subsampling: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # k -> {q: |mean(1 frame in k) - mean(all)| / eps_q} over the window


def frame_check(bundle, zone, window, eps: dict, factors=(2, 4, 8),
                evf_threshold: float = 0.5) -> FrameCheck:
    """P1 on one run: attenuation of the field filter at the Nyquist
    frequency of the frames, and the subsampling test over the window."""
    times = np.asarray(bundle.times, dtype=float)
    d = np.diff(times)
    dt = float(np.median(d[d > 0])) if (d > 0).any() else float("nan")
    fn = 0.5 / dt if dt > 0 else float("nan")
    fc = None
    if _cfg(bundle, "step", "output_filter_enabled", default=False):
        try:
            fc = float(_cfg(bundle, "step", "output_filter_cutoff_hz"))
        except (TypeError, ValueError):
            fc = None
        if fc is not None and not fc > 0:
            fc = None
    att = (butterworth_attenuation_db(fn, fc)
           if fc is not None and math.isfinite(fn) else None)
    out = FrameCheck(dt=dt, f_nyquist=fn, cutoff=fc, atten_db=att)

    inst = eulerian_instance(bundle)
    tmask = window_mask(times, *window)
    if not inst or not tmask.any():
        return out
    h = elem_size(bundle)
    idx, w = zoi_area_weights(element_centroids_xy(bundle, inst), zone,
                              h if math.isfinite(h) else None)
    if idx.size == 0:
        return out
    evf = np.asarray(bundle.field(inst, FIELD_QUANTITIES["EVF"]),
                     dtype=float)[:, idx]
    frames = np.nonzero(tmask)[0]
    ref = {}
    vals = {}
    for q in ("Vx", "Vy", "T", "EVF"):
        e = eps.get(q)
        if not (e or 0) > 0:
            continue
        try:
            v = np.asarray(bundle.field(inst, FIELD_QUANTITIES[q]),
                           dtype=float)[:, idx]
        except KeyError:
            continue
        vals[q] = v
        mat = (evf >= evf_threshold) if q in MASKED_QUANTITIES else None
        ref[q] = area_mean(v[frames], w,
                           None if mat is None else mat[frames])
    for k in factors:
        sub = frames[::int(k)]
        row = {}
        for q, v in vals.items():
            mat = ((evf >= evf_threshold) if q in MASKED_QUANTITIES
                   else None)
            m = area_mean(v[sub], w, None if mat is None else mat[sub])
            row[q] = abs(m - ref[q]) / float(eps[q])
        out.subsampling[int(k)] = row
    return out


# ---------------------------------------------------------------------------
# Chip re-contact on the uncut surface
# ---------------------------------------------------------------------------
@dataclass
class Recontact:
    times: np.ndarray
    band_max_evf: np.ndarray          # max EVF in the band at each frame
    t_first: Optional[float]          # first frame with EVF >= threshold
    n_columns: int                    # element columns watched
    x_range: Tuple[float, float]      # watched x range [mm]
    note: str = ""


def chip_recontact(bundle, x_tip: float, margin: float, gap: float = 1.5,
                   band: float = 3.0, evf_threshold: float = 0.5
                   ) -> Recontact:
    """Watch a band above the uncut surface (see the module docstring).
    `x_tip` and `margin` in model length units; `gap` and `band` in element
    sizes. Columns whose surface or band leaves the extracted zone are not
    watched."""
    inst = eulerian_instance(bundle)
    if not inst:
        raise RuntimeError("No Eulerian instance (EVF carrier) in the run.")
    times = np.asarray(bundle.times, dtype=float)
    c = element_centroids_xy(bundle, inst)
    evf = np.asarray(bundle.field(inst, FIELD_QUANTITIES["EVF"]),
                     dtype=float)
    dx, dy = _spacing(c[:, 0]), _spacing(c[:, 1])
    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise RuntimeError("The element spacing cannot be read.")
    x_hi = float(x_tip) - float(margin)
    key = np.round((c[:, 0] - c[:, 0].min()) / dx).astype(int)
    xs = np.unique(key)
    y_top = float(np.max(c[:, 1]))
    k_min = int(math.ceil(float(gap) - 1e-9))
    k_max = int(math.ceil(float(gap) + float(band) - 1e-9)) - 1
    if k_max < k_min:
        raise ValueError("the band must hold at least one cell")
    watch = []
    used_x = []
    for ix in xs:
        col = np.nonzero(key == ix)[0]
        x = float(np.mean(c[col, 0]))
        if x > x_hi:
            continue
        full = col[evf[0, col] >= evf_threshold]
        if full.size == 0:
            continue
        ys = float(np.max(c[full, 1]))
        # Cells k element sizes above the surface cell, gap <= k < gap+band.
        off = np.round((c[col, 1] - ys) / dy)
        if ys + k_max * dy > y_top + 1e-6 * dy:
            continue                  # band leaves the extracted zone
        sel = col[(off >= k_min) & (off <= k_max)]
        if sel.size:
            watch.append(sel)
            used_x.append(x)
    if not watch:
        return Recontact(times, np.full(times.shape, np.nan), None, 0,
                         (float("nan"), float("nan")),
                         note="no column upstream of the tool can be watched "
                              "in the extracted zone")
    sel = np.concatenate(watch)
    band_max = np.max(evf[:, sel], axis=1)
    hit = np.nonzero(band_max >= evf_threshold)[0]
    note = ""
    if hit.size and hit[0] == 0:
        note = "material already in the band at the first frame"
    return Recontact(times, band_max,
                     float(times[hit[0]]) if hit.size else None,
                     len(watch), (min(used_x), max(used_x)), note=note)
