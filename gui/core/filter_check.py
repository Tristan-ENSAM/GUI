# -*- coding: utf-8 -*-
"""Offline verification of Abaqus's runtime Butterworth output filters.

Abaqus/Explicit filters output at the SOLVER increment and warns in the .sta
when cutoff/(1/dt) < 1e-3 ("may produce incorrect results in the filtered
output"). Rather than trusting or fearing that warning, each run with the
filter on is checked: the RAW tool-RP forces (written unfiltered at every
increment) are filtered offline with the same Butterworth and compared with
the series Abaqus filtered at runtime (SENSORBAND for the force filter,
CAMERABAND for the camera filter, both written by cel_model when
step.output_filter_verify is on).

Offline filter, mirroring what the Analysis Guide states Abaqus does
("Filtering Output and Operating on Output in Abaqus/Explicit"):
  * the raw series is remapped by quadratic interpolation to a constant
    increment (here the median solver increment), because the IIR filter
    needs a constant sampling;
  * causal Butterworth of the same order (scipy.signal.butter, sos form),
    pre-charged with the first raw value (START CONDITION=DC, the default);
  * a cutoff at or above half the sampling frequency is not filtered,
    exactly as Abaqus does.

Metric, per filter and force component:
    e = max |x_abaqus(t) - x_offline(t)| / max |x_offline(t)|
over the Abaqus-filtered sample times. A filter passes when e <= tolerance
for both components. DEFAULT_TOLERANCE = 1 % is a choice, not an Abaqus
figure: the bilinear discretisation used by scipy and Abaqus's own (not
documented) need not match exactly, and the remapping step differs slightly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gui.core.model_config import OUTPUT_FILTER_ORDER

DEFAULT_TOLERANCE = 0.01

# npz tags written by cel_results, and the model_config key of each cutoff.
FILTERS = (
    ("SENSORBAND", "force", "output_filter_cutoff_history_hz"),
    ("CAMERABAND", "camera", "output_filter_cutoff_hz"),
)
RAW_TAG = "RAW"
COMPONENTS = ("RF1", "RF2")


def _uniform(t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                   float]:
    """Remap (t, x) to a constant step (median increment), quadratic."""
    from scipy.interpolate import interp1d
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    t, idx = np.unique(t, return_index=True)
    x = x[idx]
    if t.size < 3:
        raise ValueError("raw series too short (%d samples)" % t.size)
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("non-increasing time base")
    n = int(np.floor((t[-1] - t[0]) / dt)) + 1
    tu = t[0] + dt * np.arange(n)
    xu = interp1d(t, x, kind="quadratic", assume_sorted=True)(tu)
    return tu, xu, dt


def offline_butterworth(t_raw, x_raw, cutoff_hz: float,
                        order: int = OUTPUT_FILTER_ORDER
                        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Filter the raw series offline. Returns (t, y, dt) on the constant
    time base; y is the remapped raw series when cutoff >= Nyquist."""
    from scipy import signal
    tu, xu, dt = _uniform(t_raw, x_raw)
    fs = 1.0 / dt
    if cutoff_hz >= 0.5 * fs:
        return tu, xu, dt
    sos = signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    zi = signal.sosfilt_zi(sos) * xu[0]
    y, _ = signal.sosfilt(sos, xu, zi=zi)
    return tu, y, dt


def compare_filtered(t_raw, x_raw, t_abq, x_abq, cutoff_hz: float,
                     order: int = OUTPUT_FILTER_ORDER) -> Dict[str, float]:
    """Relative max deviation between the Abaqus-filtered series and the
    offline filter of the raw one, plus the normalised cutoff fc*dt."""
    tu, y, dt = offline_butterworth(t_raw, x_raw, cutoff_hz, order)
    t_abq = np.asarray(t_abq, dtype=float)
    x_abq = np.asarray(x_abq, dtype=float)
    inside = (t_abq >= tu[0]) & (t_abq <= tu[-1])
    if not inside.any():
        raise ValueError("no Abaqus-filtered sample inside the raw time span")
    ref = np.interp(t_abq[inside], tu, y)
    scale = float(np.max(np.abs(ref)))
    dev = float(np.max(np.abs(x_abq[inside] - ref)))
    rel = dev / scale if scale > 0 else (0.0 if dev == 0 else float("inf"))
    return {"rel_max_dev": rel, "abs_max_dev": dev, "scale": scale,
            "ratio": float(cutoff_hz) * dt, "dt": dt}


def check_arrays(arrays, cutoffs: Dict[str, float],
                 tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Run the comparison on a mapping of npz-like arrays.

    cutoffs: {"SENSORBAND": Hz, "CAMERABAND": Hz} (missing/0 -> skipped).
    Returns {"tolerance", "filters": {tag: {...}}, "passed": bool|None}."""
    keys = set(getattr(arrays, "files", None) or arrays.keys())
    out = {"tolerance": tolerance, "filters": {}, "passed": None}

    def _get(tag, comp):
        k = "filtercheck__%s__%s" % (tag, comp)
        return np.asarray(arrays[k]) if k in keys else None

    t_raw = _get(RAW_TAG, "time")
    if t_raw is None:
        out["error"] = "raw force series missing from the bundle"
        return out
    verdicts = []
    for tag, label, _key in FILTERS:
        fc = float(cutoffs.get(tag) or 0.0)
        t_f = _get(tag, "time")
        if fc <= 0 or t_f is None:
            continue
        res = {"label": label, "cutoff_hz": fc}
        try:
            for comp in COMPONENTS:
                res[comp] = compare_filtered(t_raw, _get(RAW_TAG, comp),
                                             t_f, _get(tag, comp), fc)
            worst = max(res[c]["rel_max_dev"] for c in COMPONENTS)
            res["rel_max_dev"] = worst
            res["ratio"] = res[COMPONENTS[0]]["ratio"]
            res["passed"] = bool(worst <= tolerance)
        except Exception as e:                      # reported, not raised
            res["error"] = str(e)
            res["passed"] = None
        out["filters"][tag] = res
        verdicts.append(res["passed"])
    if verdicts and all(v is not None for v in verdicts):
        out["passed"] = all(verdicts)
    return out


def _cutoffs_from_meta(meta: dict) -> Dict[str, float]:
    step = (meta.get("model_config") or {}).get("step") or {}
    return {tag: float(step.get(key) or 0.0) for tag, _l, key in FILTERS}


def meta_path_for(npz_path) -> Path:
    p = Path(npz_path)
    stem = p.name[:-len(".results.npz")] if p.name.endswith(
        ".results.npz") else p.stem
    return p.with_name(stem + ".meta.json")


def check_bundle(npz_path, tolerance: float = DEFAULT_TOLERANCE,
                 write_meta: bool = True) -> Optional[dict]:
    """Check a results bundle; store the result under "filter_check" in its
    .meta.json. None when the run did not request the verification."""
    npz_path = Path(npz_path)
    if not npz_path.name.endswith(".results.npz") or not npz_path.exists():
        return None                 # e.g. a write-.inp-only run
    meta_path = meta_path_for(npz_path)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    step = (meta.get("model_config") or {}).get("step") or {}
    if not (step.get("output_filter_enabled")
            and step.get("output_filter_verify")):
        return None
    with np.load(npz_path) as arrays:
        res = check_arrays(arrays, _cutoffs_from_meta(meta), tolerance)
    if write_meta:
        meta["filter_check"] = res
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return res


def format_report(res: Optional[dict]) -> str:
    """Human-readable block for the job output."""
    if res is None:
        return ""
    lines = ["[FILTER CHECK] Abaqus runtime filters vs offline Butterworth "
             "(order %d, tolerance %.3g %%)"
             % (OUTPUT_FILTER_ORDER, 100.0 * res["tolerance"])]
    if res.get("error"):
        lines.append("  not evaluated: %s" % res["error"])
    for tag, r in res.get("filters", {}).items():
        if r.get("error"):
            lines.append("  %-10s (%s, fc = %.6g Hz): not evaluated: %s"
                         % (tag, r["label"], r["cutoff_hz"], r["error"]))
            continue
        lines.append(
            "  %-10s (%s, fc = %.6g Hz, fc*dt = %.3g): max deviation "
            "%.3g %% -> %s"
            % (tag, r["label"], r["cutoff_hz"], r["ratio"],
               100.0 * r["rel_max_dev"], "OK" if r["passed"] else "FAILED"))
    if not res.get("filters") and not res.get("error"):
        lines.append("  no filtered force series in the bundle")
    return "\n".join(lines) + "\n"
