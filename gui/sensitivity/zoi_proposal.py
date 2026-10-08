# -*- coding: utf-8 -*-
"""ZOI proposed from the Jacobian sensitivity maps.

The ZOI is the zone where the sizing studies (ms, GCI, domain) measure the
field QoIs. It is justified here as the zone where the fields depend on the
model parameters: outside it, a plausible change of any parameter moves no
field by more than the admitted deviation eps_q, so the convergence of the
fields there does not matter for what the model is used for.

For a field q (Vx = V1, Vy = V2, T = TEMP, EVF) and a parameter p of the plan:

    S*_qp(e) = mean_t |S_qp(e, t) * delta_p| / eps_q

S_qp is the per-element map dq/dp (runner_core.jacobian_field_maps), delta_p
the plan's FD step: S * delta is the field change the runs simulated, (q+ -
q-)/2 for the central scheme, not a linear extrapolation. The mean runs over
the frames of the window T; for Vx, Vy and T it only keeps the frames where
the base run is material at e (EVF >= evf_threshold, as in the domain study),
EVF itself is not masked. The mean of |.| over time is the per-point term of
the MAD the ms and domain studies use (Eq. 5).

    S*(e) = max over (q, p) of S*_qp(e)

The proposed ZOI is the smallest axis-aligned rectangle that contains the
footprint of every element with S*(e) >= 1. It is a proposal: the user copies
it into the Model tab.

Pure numpy (no Qt), unit-testable.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gui.sensitivity.zoi_sampling import window_mask

# Bundle field variable -> name of its eps_q in the Model tab.
FIELD_TO_EPS: Dict[str, str] = {"V1": "Vx", "V2": "Vy", "TEMP": "T",
                                "EVF": "EVF"}
# Variables masked by the material indicator (EVF itself is not).
_MASKED = ("V1", "V2", "TEMP")


@dataclass
class ZoiProposal:
    s_star: np.ndarray                     # (n_elem,) max over (field, param)
    driver: List[Optional[Tuple[str, str]]]  # (field, param) of the max, per element
    per_map: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    bbox: Optional[Tuple[float, float, float, float]] = None  # xmin, xmax, ymin, ymax
    n_selected: int = 0
    # Sides of the bbox lying on the edge of the extracted zone: the ZOI is
    # then cut by the extraction, not by the physics.
    sides_at_extent: List[str] = field(default_factory=list)
    skipped_fields: List[str] = field(default_factory=list)   # no eps_q
    n_window_frames: int = 0
    # (field, param) -> number of elements where that map alone is >= 1
    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)


def map_s_star(S, delta: float, eps: float, tmask: np.ndarray,
               material: Optional[np.ndarray]) -> np.ndarray:
    """S*_qp per element: mean over the selected frames of |S*delta| / eps.

    S (n_frames, n_elem); tmask (n_frames,) frames of the window T;
    material (n_frames, n_elem) bool or None (no mask). An element with no
    selected frame is NaN."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2:
        raise ValueError("S must be (n_frames, n_elem)")
    if not (eps > 0):
        raise ValueError("eps must be > 0")
    A = np.abs(S[tmask] * float(delta)) / float(eps)
    if material is not None:
        A = np.where(np.asarray(material, bool)[tmask], A, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(A, axis=0)


def propose_zoi(maps: Dict[str, Dict[str, np.ndarray]],
                deltas: Dict[str, float],
                eps: Dict[str, float],
                verts: np.ndarray,
                times: Sequence[float],
                window: Tuple[float, float] = (0.3, 1.0),
                evf_base: Optional[np.ndarray] = None,
                evf_threshold: float = 0.5,
                extent: Optional[Tuple[float, float, float, float]] = None,
                threshold: float = 1.0) -> ZoiProposal:
    """Compute S* and the proposed ZOI rectangle.

    maps    : {field_var: {param_path: S (n_frames, n_elem)}}
    deltas  : {param_path: FD step of the plan}
    eps     : {eps name (Vx, Vy, T, EVF): eps_q}, absolute, in the field unit
    verts   : (n_elem, n_loc, 2) element footprints [mm] (map_export.element_faces)
    times   : frame times; window: T as fractions of the simulated time
    evf_base: (n_frames, n_elem) EVF of the base run, for the material mask
    extent  : (xmin, xmax, ymin, ymax) of the extracted zone, to flag a ZOI
              that reaches it; default = the footprint of all elements
    """
    verts = np.asarray(verts, dtype=float)
    n_elem = verts.shape[0]
    tmask = window_mask(np.asarray(times, float), *window)
    if not tmask.any():
        raise ValueError("no frame in the window T")
    material = None
    if evf_base is not None:
        evf = np.asarray(evf_base, dtype=float)
        with np.errstate(invalid="ignore"):
            material = evf >= float(evf_threshold)

    per_map: Dict[Tuple[str, str], np.ndarray] = {}
    skipped: List[str] = []
    for var, per in maps.items():
        name = FIELD_TO_EPS.get(var)
        e = eps.get(name) if name else None
        if e is None or not (e > 0):
            skipped.append(var)
            continue
        mask = material if var in _MASKED else None
        for path, S in per.items():
            d = deltas.get(path)
            S = np.asarray(S, dtype=float)
            if d is None or S.ndim != 2 or S.shape[1] != n_elem:
                continue
            per_map[(var, path)] = map_s_star(S, d, e, tmask, mask)

    s_star = np.full(n_elem, np.nan)
    driver: List[Optional[Tuple[str, str]]] = [None] * n_elem
    counts: Dict[Tuple[str, str], int] = {}
    if per_map:
        keys = list(per_map)
        stack = np.vstack([per_map[k] for k in keys])        # (n_maps, n_elem)
        filled = np.where(np.isfinite(stack), stack, -np.inf)
        best = np.argmax(filled, axis=0)
        has = np.isfinite(stack).any(axis=0)
        s_star = np.where(has, filled[best, np.arange(n_elem)], np.nan)
        driver = [keys[best[i]] if has[i] else None for i in range(n_elem)]
        for k in keys:
            with np.errstate(invalid="ignore"):
                counts[k] = int(np.sum(per_map[k] >= threshold))

    with np.errstate(invalid="ignore"):
        sel = np.isfinite(s_star) & (s_star >= threshold)
    out = ZoiProposal(s_star=s_star, driver=driver, per_map=per_map,
                      n_selected=int(sel.sum()), skipped_fields=skipped,
                      n_window_frames=int(tmask.sum()), counts=counts)
    if not sel.any():
        return out
    pts = verts[sel].reshape(-1, 2)
    bbox = (float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max()))
    out.bbox = bbox
    allp = verts.reshape(-1, 2)
    if extent is None:
        extent = (float(allp[:, 0].min()), float(allp[:, 0].max()),
                  float(allp[:, 1].min()), float(allp[:, 1].max()))
    # Tolerance: half the median element width, so a bbox on the last row of
    # elements counts as touching the edge.
    widths = verts[:, :, 0].max(axis=1) - verts[:, :, 0].min(axis=1)
    tol = 0.5 * float(np.median(widths)) if widths.size else 0.0
    for side, b, x in zip(("xmin", "xmax", "ymin", "ymax"), bbox, extent):
        if abs(b - x) <= tol:
            out.sides_at_extent.append(side)
    return out


def proposal_record(p: ZoiProposal, *, eps, window, evf_threshold,
                    threshold=1.0, deltas=None, extent_kind="") -> dict:
    """JSON-ready summary of a proposal (written next to the maps)."""
    drivers = {}
    if p.bbox is not None:
        with np.errstate(invalid="ignore"):
            sel = np.isfinite(p.s_star) & (p.s_star >= threshold)
        for i in np.flatnonzero(sel):
            k = p.driver[i]
            if k is not None:
                drivers["%s @ %s" % k] = drivers.get("%s @ %s" % k, 0) + 1
    finite = p.s_star[np.isfinite(p.s_star)]
    return {
        "rule": "smallest rectangle containing every element with "
                "max_(q,p) mean_T |dq/dp * delta_p| / eps_q >= %g" % threshold,
        "zoi": (None if p.bbox is None else
                dict(zip(("xmin", "xmax", "ymin", "ymax"), p.bbox))),
        "n_elements_selected": p.n_selected,
        "n_elements": int(p.s_star.size),
        "s_star_max": (float(finite.max()) if finite.size else None),
        "sides_at_extent": list(p.sides_at_extent),
        "extracted_zone": extent_kind,
        "eps_q": dict(eps), "window_T": list(window),
        "n_window_frames": p.n_window_frames,
        "evf_threshold": float(evf_threshold),
        "deltas": dict(deltas or {}),
        "skipped_fields_no_eps": list(p.skipped_fields),
        "elements_selected_by_map": {"%s @ %s" % k: v
                                     for k, v in p.counts.items()},
        "elements_selected_by_driver": drivers,
    }


def write_proposal(out_dir, p: ZoiProposal, verts, *, eps, window,
                   deltas=None, evf_threshold: float = 0.5,
                   extent_kind: str = "", images: bool = True):
    """Write the proposal next to the maps:

      zoi_proposal.json   rule, rectangle, eps_q, T, counts (proposal_record)
      zoi_sstar.npz       s_star (n_elem,), one S*_qp array per map
                          (key "<field>__<param path>"), the rectangle
      zoi_sstar.png       S* with the rectangle drawn (images=True)

    Returns the written Paths."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = proposal_record(p, eps=eps, window=window,
                          evf_threshold=evf_threshold, deltas=deltas,
                          extent_kind=extent_kind)
    written = []
    pj = out / "zoi_proposal.json"
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    written.append(pj)

    arrays = {"s_star": np.asarray(p.s_star, float),
              "bbox": np.asarray(p.bbox if p.bbox else [np.nan] * 4, float)}
    for (var, path), a in p.per_map.items():
        arrays["%s__%s" % (var, path)] = np.asarray(a, float)
    pn = out / "zoi_sstar.npz"
    np.savez_compressed(pn, **arrays)
    written.append(pn)

    if images:
        pp = out / "zoi_sstar.png"
        _render_s_star(pp, np.asarray(verts, float), p)
        written.append(pp)
    return written


def _render_s_star(path, verts, p: ZoiProposal):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle
    from matplotlib import colormaps

    vals = np.asarray(p.s_star, float)
    finite = vals[np.isfinite(vals)]
    vmax = max(1.0, float(finite.max())) if finite.size else 1.0
    cmap = colormaps["inferno"].copy()
    cmap.set_bad(alpha=0.0)
    fig = Figure(figsize=(8, 5), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    polys = PolyCollection(verts, array=np.ma.masked_invalid(vals),
                           cmap=cmap, edgecolors="none")
    polys.set_clim(0.0, vmax)
    ax.add_collection(polys)
    xy = verts.reshape(-1, 2)
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    if p.bbox is not None:
        x0, x1, y0, y1 = p.bbox
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="#22c55e", linewidth=1.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("S* = max_(q,p) mean_T |dq/dp · δ_p| / ε_q ; "
                 "proposed ZOI (green): S* ≥ 1", fontsize=9)
    cb = fig.colorbar(polys, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label("S* [-]", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(path))
