# -*- coding: utf-8 -*-
"""Richardson test on the Eulerian domain (decision of 2026-11-06).

Question tested
---------------
Does the influence of the domain boundaries on the ZOI decrease as a power
of 1/L, like a discretisation error decreases as a power of h? If it does,
a Richardson extrapolation to an infinite domain gives a reference for D*
(as for h*); if it does not, the reference stays the largest domain tested,
the rule of the mesh study when p < 0.5.

Series
------
Three domains D_1 = D, D_2 = r D, D_3 = r^2 D: the four sides (l_wp, h_wp,
l_void, h_void) multiplied by the same factor (one factor for all sides; a
factor per side would multiply the runs by 4), each rounded UP to a whole
number of elements. The sides are measured from the tool tip (frame of
cel_model.py: the domain is (-l_wp, -h_wp) -> (l_void, h_void)), so a larger
factor keeps the ZOI inside. Mesh and mass scaling are those of the model.

The "step" of the Richardson law is x = 1/s, s the scale of a domain with
respect to D: the geometric mean of its four side ratios (the rounding to
whole elements makes them differ slightly). The largest domain is the
"finest" level: r21 = s_3/s_2, r32 = s_2/s_1.

Per quantity (same scalar as the domain and GCI studies: the area mean over
the ZOI, each element time-averaged over its material frames, EVF >=
threshold; forces: mean of F/h over T), mesh_gci.quantity_gci gives the
observed order p, the extrapolated value f_inf and the reliability flag
(monotone series, p >= 0.5, r^p - 1 >= 0.1). The reference is f_inf when
reliable, else the value on the largest domain (fallback). The deviation of
each domain is |f_q(D_k) - ref_q| / eps_q, eps_q times the domain share of
the error budget.

This module only measures: it does not change D*. Whether the extrapolated
reference is adopted for the domain step is decided on its results.

Hypothesis (not verified on runs): a boundary effect may as well decay
exponentially with the distance, in which case no constant p exists; the
three-level p cannot tell the two apart, the asymptotic ratio (~1 in the
asymptotic range) is reported as an indicator.

Pure host-side module (CPython 3.x): `run_bundle(cfg)` is injected and must
return a ResultsBundle-like object or None.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.mesh_gci import QuantityGci, quantity_gci, zoi_scalars

SIDES = ("l_wp", "h_wp", "l_void", "h_void")
DEFAULT_RATIO = 1.5
DEFAULT_LEVELS = 3
FIELD_VARS = ("EVF", "TEMP", "V1", "V2")
FORCE_CHANNELS = {"Fc": "RF1_RP", "Ff": "RF2_RP"}


def _snap_up(value: float, h: float) -> float:
    n = math.ceil(float(value) / h - 1e-9)
    return float(max(1, n) * h)


def scaled_series(dims: DomainDims, ratio: float = DEFAULT_RATIO,
                  n: int = DEFAULT_LEVELS, elem_size: Optional[float] = None
                  ) -> Tuple[List[DomainDims], List[float]]:
    """(domains, scales): D, rD, r^2 D, ... (n levels, smallest first), each
    side rounded up to whole elements when `elem_size` is given; scales[k]
    is the geometric mean of the side ratios of level k to level 0 (1 for
    level 0)."""
    if ratio <= 1.0:
        raise ValueError("the ratio of the domain series must be > 1")
    if n < 3:
        raise ValueError("a Richardson test needs at least 3 domains")
    base = {s: float(getattr(dims, s)) for s in SIDES}
    if any(not v > 0 for v in base.values()):
        raise ValueError("every side of the domain must be > 0")
    h = float(elem_size) if elem_size else None
    out, scales = [], []
    for k in range(n):
        f = float(ratio) ** k
        vals = {s: (_snap_up(base[s] * f, h) if h else base[s] * f)
                for s in SIDES}
        out.append(DomainDims(**vals))
    first = out[0]
    for d in out:
        logs = [math.log(getattr(d, s) / getattr(first, s)) for s in SIDES]
        scales.append(math.exp(sum(logs) / len(logs)))
    return out, scales


@dataclass
class DomainRichardsonResult:
    ratio: float = DEFAULT_RATIO
    domains: List[DomainDims] = field(default_factory=list)  # small -> large
    scales: List[float] = field(default_factory=list)
    scalars: List[Dict[str, float]] = field(default_factory=list)
    per_quantity: Dict[str, QuantityGci] = field(default_factory=dict)
    reference: Dict[str, float] = field(default_factory=dict)
    # "richardson" (extrapolated) or "largest" (fallback) per quantity.
    reference_kind: Dict[str, str] = field(default_factory=dict)
    # deviation[k][q] = |f_q(D_k) - ref_q| / eps_q.
    deviation: List[Dict[str, float]] = field(default_factory=list)
    # Smallest domain of the series whose every deviation is <= 1, or None.
    smallest_passing: Optional[int] = None
    tolerances: Dict[str, float] = field(default_factory=dict)
    stopped_by: str = ""          # "ok" | "cancelled" | "no_bundle"
    message: str = ""

    @property
    def power_law(self) -> bool:
        """True when every quantity has a reliable extrapolation."""
        return bool(self.reference_kind) and all(
            v == "richardson" for v in self.reference_kind.values())

    def record(self) -> dict:
        def num(v):
            return None if v is None or not math.isfinite(v) else float(v)
        return {
            "study": "domain_richardson", "ratio": self.ratio,
            "stopped_by": self.stopped_by, "message": self.message,
            "domains": [{s: getattr(d, s) for s in SIDES}
                        for d in self.domains],
            "scales": list(self.scales),
            "scalars": [{q: num(v) for q, v in sc.items()}
                        for sc in self.scalars],
            "tolerances": dict(self.tolerances),
            "per_quantity": {
                q: {"p": num(g.p), "f_largest": num(g.f_fine),
                    "f_extrapolated": num(g.f_extrapolated),
                    "asymptotic_ratio": num(g.asymptotic_ratio),
                    "monotonic": bool(g.monotonic),
                    "reliable": bool(g.reliable),
                    "reference": num(self.reference.get(q)),
                    "reference_kind": self.reference_kind.get(q)}
                for q, g in self.per_quantity.items()},
            "deviation": [{q: num(v) for q, v in d.items()}
                          for d in self.deviation],
            "smallest_passing": self.smallest_passing,
            "power_law": self.power_law}


def evaluate(scalars: Sequence[Dict[str, float]], scales: Sequence[float],
             tolerances: Dict[str, float],
             quantities: Sequence[str]) -> Tuple[Dict, Dict, Dict, List,
                                                 Optional[int]]:
    """(per_quantity, reference, reference_kind, deviation, smallest_passing)
    from the scalars of the series (smallest domain first)."""
    n = len(scalars)
    if n < 3 or len(scales) != n:
        raise ValueError("need the scalars and scales of >= 3 domains")
    # Richardson on the three largest domains: level 1 = largest ("finest").
    i1, i2, i3 = n - 1, n - 2, n - 3
    r21 = scales[i1] / scales[i2]
    r32 = scales[i2] / scales[i3]
    per_q, ref, kind = {}, {}, {}
    for q in quantities:
        f1 = scalars[i1].get(q, float("nan"))
        f2 = scalars[i2].get(q, float("nan"))
        f3 = scalars[i3].get(q, float("nan"))
        g = quantity_gci(f1, f2, f3, r21, r32)
        per_q[q] = g
        if g.reliable:
            ref[q], kind[q] = g.f_extrapolated, "richardson"
        else:
            ref[q], kind[q] = f1, "largest"
    dev = []
    for sc in scalars:
        d = {}
        for q in quantities:
            tol = tolerances.get(q)
            if not tol:
                continue
            v, r = sc.get(q, float("nan")), ref.get(q, float("nan"))
            d[q] = (abs(v - r) / tol if math.isfinite(v) and math.isfinite(r)
                    else float("inf"))
        dev.append(d)
    passing = None
    if tolerances:
        for k, d in enumerate(dev):
            if d and all(v <= 1.0 for v in d.values()):
                passing = k
                break
    return per_q, ref, kind, dev, passing


def run_domain_richardson(
        run_bundle: Callable, base_cfg, zoi: Sequence[float],
        base_dims: DomainDims, elem_size: float, *,
        ratio: float = DEFAULT_RATIO, n_levels: int = DEFAULT_LEVELS,
        tolerances: Optional[Dict[str, float]] = None,
        grid_step: float = 0.001, window: Tuple[float, float] = (0.3, 1.0),
        evf_threshold: float = 0.5,
        field_vars: Sequence[str] = FIELD_VARS,
        force_channels: Optional[Dict[str, str]] = None,
        should_cancel: Optional[Callable] = None,
        progress_cb: Optional[Callable] = None) -> DomainRichardsonResult:
    """Run the series D, rD, r^2 D at the model's mesh and mass scaling and
    evaluate the Richardson test (see the module docstring)."""
    if force_channels is None:
        force_channels = dict(FORCE_CHANNELS)
    tol = {q: float(t) for q, t in (tolerances or {}).items()
           if t is not None and float(t) > 0}
    h = float(elem_size)
    domains, scales = scaled_series(base_dims, ratio, n_levels, h)
    res = DomainRichardsonResult(ratio=float(ratio), domains=domains,
                                 scales=scales, tolerances=tol)
    for k, d in enumerate(domains):
        if should_cancel is not None and should_cancel():
            res.stopped_by = "cancelled"
            return res
        cfg = copy.deepcopy(base_cfg)
        cfg.elem_size = h
        for s in SIDES:
            setattr(cfg.euler_geometry, s, float(getattr(d, s)))
        bundle = run_bundle(cfg)
        if bundle is None:
            if should_cancel is not None and should_cancel():
                res.stopped_by = "cancelled"
                return res
            res.stopped_by = "no_bundle"
            res.message = "domain %d of %d gave no result" % (k + 1,
                                                              len(domains))
            return res
        sc = zoi_scalars(bundle, zoi, grid_step, field_vars, window=window,
                         evf_threshold=evf_threshold,
                         force_channels=force_channels, reduction="area",
                         elem_size=h)
        for label in force_channels:        # force per unit width (N/mm)
            v = sc.get(label)
            if v is not None and math.isfinite(v) and h > 0:
                sc[label] = v / h
        res.scalars.append(sc)
        if progress_cb:
            progress_cb({"phase": "domain_richardson", "level": k,
                         "dims": d, "scale": scales[k], "scalars": sc})
    quantities = list(field_vars) + list(force_channels)
    (res.per_quantity, res.reference, res.reference_kind, res.deviation,
     res.smallest_passing) = evaluate(res.scalars, scales, tol, quantities)
    res.stopped_by = "ok"
    return res
