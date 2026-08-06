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


def field_rel_change_pct(base_field, pert_field) -> float:
    """Relative change of the field, in percent, weighted (averaged) over
    nodes/points AND frames:

        dF% = 100 * RMS_{nodes,frames}(pert - base) / RMS_{nodes,frames}(base)
            = 100 * sqrt(mean((pert - base)^2)) / sqrt(mean(base^2))

    Using the mean (not the sum) makes the result independent of the number
    of nodes and frames -- each node and each frame is weighted equally. The
    same finite mask is applied to numerator and denominator (only paired,
    finite entries count). Returns NaN if the base field has no finite
    energy on the ROI. Note the count cancels in the ratio, so this equals
    the relative L2 norm ||pert-base|| / ||base||."""
    a, b = _align(pert_field, base_field)   # a = pert, b = base
    d = (a - b).ravel()
    bb = b.ravel()
    m = np.isfinite(d) & np.isfinite(bb)
    if not m.any():
        return float("nan")
    d = d[m]
    bb = bb[m]
    denom = np.sqrt(np.mean(bb * bb))
    if denom == 0.0:
        return float("nan")
    return float(100.0 * np.sqrt(np.mean(d * d)) / denom)


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


def elementwise_signed_sensitivity(base_field, plus_field, minus_field,
                                   delta, scheme="central"):
    """Per-element, per-frame SIGNED finite-difference sensitivity dF/dparam.

    Unlike jacobian_field_sensitivity (which reduces the whole field to one
    scalar over the ROI), this keeps the element axis: it returns an array
    S of shape (n_frames, n_elements) where

        central  : S = (plus  - minus) / (2*delta)
        forward  : S = (plus  - base ) /    delta
        backward : S = (base  - minus) /    delta

    base/plus/minus are (n_frames, n_elements) field arrays for the base run,
    the +delta run and the -delta run respectively; only the ones required by
    `scheme` need to be provided (the others may be None). `delta` is the FD
    step in displayed units (must be non-zero). The result is NaN-safe: where
    a required input is NaN (e.g. a folded/empty element), S is NaN; if a
    required array is missing entirely, an all-NaN array is returned (shaped
    from whatever input is available).

    The sign is meaningful: S > 0 means the field increases with the
    parameter at that element, S < 0 means it decreases. For a magnitude map
    take abs(S); to reduce over time use the mean of S (signed) or the RMS of
    S (magnitude)."""
    d = float(delta)
    if d == 0.0:
        raise ValueError("delta must be non-zero")

    def _arr(x):
        return None if x is None else np.asarray(x, dtype=float)

    base = _arr(base_field)
    plus = _arr(plus_field)
    minus = _arr(minus_field)

    if scheme == "central":
        a, b, denom = plus, minus, 2.0 * d
    elif scheme == "forward":
        a, b, denom = plus, base, d
    elif scheme == "backward":
        a, b, denom = base, minus, d
    else:
        raise ValueError("scheme must be 'central', 'forward' or 'backward'")

    # If a required operand is missing, return an all-NaN field shaped from
    # any array we do have.
    if a is None or b is None:
        ref = next((x for x in (base, plus, minus) if x is not None), None)
        if ref is None:
            return np.empty((0, 0), dtype=float)
        return np.full(ref.shape, np.nan, dtype=float)

    a, b = _align(a, b)
    return (a - b) / denom
