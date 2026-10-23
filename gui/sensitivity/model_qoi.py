# -*- coding: utf-8 -*-
"""QoI of the Sensitivity tab computed exactly as the Model tab studies do.

The sizing studies (ms, GCI, domain) judge a run on six scalars, compared
with the admitted deviations eps_q of the Model tab:

* Vx, Vy, T, EVF : area mean over the ZOI rectangle, each element
  time-averaged over its frames inside the window T (Vx, Vy, T only over
  its material frames, EVF >= threshold), weighted by its area inside the
  ZOI (`zoi_sampling.zoi_area_means`, the scalar of the "mean" metric);
* Fc, Ff : mean of the tool reaction force RF1 / RF2 over the history
  samples inside the window T, divided by the simulated width w (the
  element size of the run), in N/mm.

`make_specs` returns them as `QoISpec`s whose function closes over the ZOI,
the window and the EVF threshold, so the runner reduces each run with the
same settings as the Model tab. Each spec carries `eps_key`, the eps_q it
is judged against (see `runner_core.add_eps_ratios`).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from gui.results.qoi import QoISpec
from gui.sensitivity.domain_independence import (
    FIELD_QUANTITIES, FORCE_QUANTITIES, MASKED_QUANTITIES, zoi_within_crop)
from gui.sensitivity.runner_core import eulerian_instance
from gui.sensitivity.zoi_sampling import (
    history_window_mean, window_mask, zoi_area_means)

# Order and labels of the Model tab (eps_q row).
MODEL_QOIS: Tuple[str, ...] = ("Vx", "Vy", "T", "EVF", "Fc", "Ff")
LABELS = {"Vx": "Vx (ZOI mean)", "Vy": "Vy (ZOI mean)",
          "T": "T (ZOI mean)", "EVF": "φ = EVF (ZOI mean)",
          "Fc": "Fc = RF1/w (mean)", "Ff": "Ff = RF2/w (mean)"}
UNITS = {"Vx": "mm/s", "Vy": "mm/s", "T": "°C", "EVF": "–",
         "Fc": "N/mm", "Ff": "N/mm"}


def _elem_size(bundle) -> float:
    try:
        return float(bundle.model_config["mesh"]["elem_size"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return float("nan")


def field_mean(bundle, q: str, zoi, window, evf_threshold: float) -> float:
    """Area mean of field quantity q over the ZOI and the window T. NaN when
    the field or the Eulerian instance is missing; raises RuntimeError when
    the ZOI leaves the extracted ROI (the edge elements would be read)."""
    inst = eulerian_instance(bundle)
    if not inst:
        return float("nan")
    crop = getattr(bundle, "roi", None)
    if isinstance(crop, dict) and not zoi_within_crop(zoi, crop):
        raise RuntimeError("the ZOI is not inside the extracted ROI")
    tmask = window_mask(np.asarray(bundle.times, dtype=float), *window)
    if not tmask.any():
        return float("nan")
    var = FIELD_QUANTITIES[q]
    masked = [var] if q in MASKED_QUANTITIES else []
    h = _elem_size(bundle)
    means = zoi_area_means(bundle, inst, zoi, [var], masked=masked,
                           tmask=tmask, evf_var=FIELD_QUANTITIES["EVF"],
                           evf_threshold=evf_threshold,
                           elem_size=h if np.isfinite(h) else None)
    return float(means[var])


def force_mean(bundle, q: str, window) -> float:
    """Mean of F/w over the history samples inside the window T (N/mm)."""
    w = _elem_size(bundle)
    if not (np.isfinite(w) and w > 0):
        return float("nan")
    m = history_window_mean(bundle, FORCE_QUANTITIES[q], *window)
    return float("nan") if m is None else float(m) / w


def make_specs(zoi: Sequence[float], window: Sequence[float],
               evf_threshold: float = 0.5,
               qois: Sequence[str] = MODEL_QOIS) -> list:
    """QoISpecs of the Model-tab quantities `qois` with these settings."""
    zoi = tuple(float(v) for v in zoi)
    window = (float(window[0]), float(window[1]))
    thr = float(evf_threshold)
    specs = []
    for q in qois:
        if q in FIELD_QUANTITIES:
            def fn(bundle, instance=None, warmup_frac=0.0, _q=q):
                return field_mean(bundle, _q, zoi, window, thr)
        elif q in FORCE_QUANTITIES:
            def fn(bundle, instance=None, warmup_frac=0.0, _q=q):
                return force_mean(bundle, _q, window)
        else:
            raise KeyError("unknown Model-tab quantity %r" % q)
        specs.append(QoISpec(q, LABELS[q], UNITS[q], fn, eps_key=q))
    return specs
