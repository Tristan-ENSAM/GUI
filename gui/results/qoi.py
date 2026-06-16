# -*- coding: utf-8 -*-
"""
Quantities of Interest (QoI) — reduce a results bundle to a handful of
scalars for the sensitivity study.

A QoI is a single number computed from a `ResultsBundle` (see
`gui.results.reader`). The Morris study runs N simulations, loads each
bundle, computes the same QoI on all of them, and feeds the resulting
vectors to SALib.

The four QoI requested for the cutting model
--------------------------------------------
  - Fx_max   : peak cutting-force magnitude   = max |RF1_RP|       [N]
  - Fx_mean  : mean cutting-force magnitude    = mean |RF1_RP|      [N]
  - T_max    : peak temperature anywhere/anytime in the instance    [°C]
  - PEEQ_max : peak equivalent plastic strain anywhere/anytime       [—]

A bonus `Fy_max` (= max |RF2_RP|) is included because it is free.

Conventions / assumptions (validate these)
-------------------------------------------
  - RF1_RP / RF2_RP are the X / Y reaction forces at the tool reference
    point (FORMAT.md). The cutting force is the reaction on the tool. Its
    SIGN depends on the kinematic setup, so we reduce on the ABSOLUTE
    value: "max"/"mean" mean peak/mean magnitude. This is unit- and
    sign-convention-free, which is what matters for screening (Morris
    ranks relative influence; a consistent definition across runs is the
    only requirement).
  - Force unit is N (the model's t-mm-s-MPa-°C system, see gui.core.units).
  - T_max and PEEQ_max are taken over the WHOLE field array
    (n_frames × n_elements) of the chosen instance — i.e. the hottest
    element at the hottest frame, and likewise for PEEQ.
  - `warmup_frac` lets the mean ignore an initial fraction of the signal
    (the tool entering the material). Default 0.0 → use everything.

Robustness
----------
Sensitivity runs WILL have failures (diverged jobs, missing fields). A
single bad run must not crash the whole batch, so every QoI returns
`float('nan')` (and records a note) when its source data is missing,
rather than raising. Callers check `np.isfinite(...)` and drop NaN rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import logging
import numpy as np

from gui.results.reader import ResultsBundle
from gui.core.logging_util import log_swallowed


# ---------------------------------------------------------------------------
# QoI spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QoISpec:
    """One scalar quantity of interest."""
    id: str                       # machine id, e.g. "Fx_max"
    label: str                    # human label for the UI
    unit: str                     # display unit
    fn: Callable[["ResultsBundle", Optional[str], float], float]
    # `fn(bundle, instance, warmup_frac) -> float` (nan on missing data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pick_field_instance(bundle: ResultsBundle, var: str) -> Optional[str]:
    """Return the first instance that carries field `var`, or None."""
    for name in bundle.instance_names:
        try:
            if var in bundle.instance(name).field_variables:
                return name
        except KeyError:
            continue
    return None


def _history_abs(bundle: ResultsBundle, var: str,
                 warmup_frac: float) -> Optional[np.ndarray]:
    """Return |history(var)| past the warmup fraction, or None if absent."""
    if var not in bundle.history_info.variables:
        return None
    try:
        sig = np.abs(np.asarray(bundle.history(var), dtype=np.float64))
    except KeyError:
        return None
    if sig.size == 0:
        return None
    if warmup_frac and warmup_frac > 0.0:
        start = int(round(warmup_frac * sig.size))
        start = min(max(start, 0), sig.size - 1)
        sig = sig[start:]
    return sig


def _field_global_max(bundle: ResultsBundle, var: str,
                      instance: Optional[str]) -> float:
    """max over (frames × elements) of `var` on `instance` (auto if None)."""
    inst = instance or _pick_field_instance(bundle, var)
    if inst is None:
        return float("nan")
    try:
        arr = bundle.field(inst, var)
    except KeyError:
        return float("nan")
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(np.asarray(arr, dtype=np.float64)))


# ---------------------------------------------------------------------------
# QoI functions
# ---------------------------------------------------------------------------
def qoi_Fx_max(bundle, instance=None, warmup_frac=0.0) -> float:
    sig = _history_abs(bundle, "RF1_RP", warmup_frac)
    return float("nan") if sig is None else float(np.nanmax(sig))


def qoi_Fx_mean(bundle, instance=None, warmup_frac=0.0) -> float:
    sig = _history_abs(bundle, "RF1_RP", warmup_frac)
    return float("nan") if sig is None else float(np.nanmean(sig))


def qoi_Fy_max(bundle, instance=None, warmup_frac=0.0) -> float:
    sig = _history_abs(bundle, "RF2_RP", warmup_frac)
    return float("nan") if sig is None else float(np.nanmax(sig))


def qoi_Fy_mean(bundle, instance=None, warmup_frac=0.0) -> float:
    sig = _history_abs(bundle, "RF2_RP", warmup_frac)
    return float("nan") if sig is None else float(np.nanmean(sig))


def qoi_T_max(bundle, instance=None, warmup_frac=0.0) -> float:
    return _field_global_max(bundle, "TEMP", instance)


def qoi_PEEQ_max(bundle, instance=None, warmup_frac=0.0) -> float:
    return _field_global_max(bundle, "PEEQ", instance)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REGISTRY: list[QoISpec] = [
    QoISpec("Fx_max",   "Force de coupe max  (|RF1| max)",   "N",  qoi_Fx_max),
    QoISpec("Fx_mean",  "Force de coupe moyenne (|RF1| moy)", "N",  qoi_Fx_mean),
    QoISpec("Fy_max",   "Force d'avance max  (|RF2| max)",   "N",  qoi_Fy_max),
    QoISpec("Fy_mean",  "Force d'avance moyenne (|RF2| moy)", "N",  qoi_Fy_mean),
    QoISpec("T_max",    "Température max",                    "°C", qoi_T_max),
    QoISpec("PEEQ_max", "PEEQ max",                          "—",  qoi_PEEQ_max),
]

_BY_ID: dict[str, QoISpec] = {q.id: q for q in REGISTRY}


def qoi_spec(qoi_id: str) -> QoISpec:
    try:
        return _BY_ID[qoi_id]
    except KeyError:
        raise KeyError(f"Unknown QoI id {qoi_id!r}. Known: {list(_BY_ID)}")


def available_qoi_ids() -> list[str]:
    return [q.id for q in REGISTRY]


def compute_qois(bundle: ResultsBundle,
                 qoi_ids: Optional[list[str]] = None,
                 instance: Optional[str] = None,
                 warmup_frac: float = 0.0) -> dict[str, float]:
    """Compute a set of QoI on a bundle.

    Parameters
    ----------
    bundle:      a loaded ResultsBundle.
    qoi_ids:     which QoI to compute (default: all of REGISTRY).
    instance:    instance name for field-based QoI (auto-detected if None).
    warmup_frac: fraction of the history signal to skip before averaging.

    Returns a dict {qoi_id: value}. Missing/failed QoI come back as
    `float('nan')` — never raises on a single bad QoI.
    """
    ids = qoi_ids if qoi_ids is not None else available_qoi_ids()
    out: dict[str, float] = {}
    for qid in ids:
        spec = qoi_spec(qid)
        try:
            out[qid] = float(spec.fn(bundle, instance, warmup_frac))
        except Exception:
            # Last-resort guard: a QoI fn should already return nan on
            # missing data, but never let one bad run abort the batch.
            log_swallowed("computing QoI %r" % qid, level=logging.DEBUG)
            out[qid] = float("nan")
    return out


def compute_qois_from_path(npz_or_json_path,
                           qoi_ids: Optional[list[str]] = None,
                           instance: Optional[str] = None,
                           warmup_frac: float = 0.0) -> dict[str, float]:
    """Convenience: load a bundle from disk, compute QoI, close it.

    Returns all-NaN for the requested QoI if the bundle can't be loaded
    (so a missing/failed run becomes a clean NaN row, not an exception).
    """
    ids = qoi_ids if qoi_ids is not None else available_qoi_ids()
    try:
        bundle = ResultsBundle.load(npz_or_json_path)
    except Exception:
        log_swallowed("loading results bundle %r" % str(npz_or_json_path))
        return {qid: float("nan") for qid in ids}
    try:
        return compute_qois(bundle, ids, instance, warmup_frac)
    finally:
        bundle.close()
