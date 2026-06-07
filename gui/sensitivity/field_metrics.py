# -*- coding: utf-8 -*-
"""
Field discrepancy metrics — reduce a full field (n_frames, n_elements) to
a single scalar by comparing two states element-by-element, frame-by-frame.

Two uses:
  * Jacobian field sensitivity: how much does perturbing a parameter move
    the WHOLE field?  J_i = SSD( F(x0+delta_i), F(x0) )  (optionally /delta).
  * Identification objective (later): SSD between the simulated field and
    the experimental field measured on the planing rig.

All functions are NaN-safe (NaNs are ignored) and operate on arrays that
are already aligned (same instance, same elements, same frames). If the
two fields differ in shape, only the overlapping leading region is used.
"""
from __future__ import annotations
import numpy as np


def _align(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        # Compare only the overlapping leading block (robust to a run that
        # produced fewer frames). Callers should normally pass aligned data.
        sl = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(a.shape, b.shape))
        a, b = a[sl], b[sl]
    return a, b


def field_ssd(field_a, field_b) -> float:
    """Sum of squared differences over all elements and time steps:
        SSD = Σ_e Σ_t (a - b)^2 .
    NaNs in either field are ignored."""
    a, b = _align(field_a, field_b)
    d = a - b
    return float(np.nansum(d * d))


def field_l2(field_a, field_b) -> float:
    """Euclidean (L2) norm of the difference: sqrt(SSD)."""
    return float(np.sqrt(field_ssd(field_a, field_b)))


def field_rmse(field_a, field_b) -> float:
    """Root-mean-square error over the finite entries."""
    a, b = _align(field_a, field_b)
    d = (a - b).ravel()
    m = np.isfinite(d)
    if not m.any():
        return float("nan")
    d = d[m]
    return float(np.sqrt(np.mean(d * d)))


def jacobian_field_sensitivity(base_field, pert_field, delta=None,
                               metric="ssd"):
    """Field sensitivity of one parameter for one field variable.

    base_field / pert_field : (n_frames, n_elements) arrays for the base
    run and the perturbed run (x0 + delta along this parameter).

    metric:
      'ssd'  -> Σ (pert - base)^2                (always >= 0)
      'l2'   -> sqrt(SSD)
      'rmse' -> RMS of (pert - base)
    If `delta` is given, the result is divided by |delta| ('l2'/'rmse') or
    delta^2 ('ssd'), giving a per-unit-parameter rate. Otherwise the raw
    discrepancy is returned."""
    if metric == "ssd":
        val = field_ssd(pert_field, base_field)
        if delta:
            val /= float(delta) ** 2
    elif metric == "l2":
        val = field_l2(pert_field, base_field)
        if delta:
            val /= abs(float(delta))
    elif metric == "rmse":
        val = field_rmse(pert_field, base_field)
        if delta:
            val /= abs(float(delta))
    else:
        raise ValueError("metric must be 'ssd', 'l2' or 'rmse'")
    return float(val)
