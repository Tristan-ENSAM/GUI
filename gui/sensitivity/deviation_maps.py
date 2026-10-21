# -*- coding: utf-8 -*-
"""Deviation maps between two runs of a sizing study (ms, mesh, domain).

Purpose
-------
The sizing studies reduce the comparison of two runs to one number per
quantity, E_q (mean over the ZOI points and the frames of the window T of
|q_a - q_b|, gui.sensitivity.domain_independence). This module keeps the
detail that number hides: WHERE in the ZOI and WHEN in T the two runs
differ, so that a deviation can be traced to a zone (chip free surface,
tool contact, shear zone...) or to a time behaviour (constant offset,
oscillation out of phase).

For one pair (reference run a, compared run b) and one quantity q:

* frame map   : |q_b - q_a| / eps_q per ZOI point, at one frame of T;
* mean map    : the time mean over T of that, per point (over the frames
  where the point is compared). Its contribution to E_q of the studies;
* bias map    : |mean_T q_b - mean_T q_a| / eps_q per point, i.e. the
  difference of the TIME MEANS. A deviation that only comes from
  fluctuations out of phase between the two runs is in the mean map and
  not in the bias map;
* E_inst      : mean over the compared (point, frame) samples of
  |q_b - q_a| / eps_q, each sample weighted once: the E_q / eps_q of the
  ms and domain studies;
* E_bias      : mean over the points of the bias map;
* e_frame     : spatial mean per frame of |q_b - q_a| / eps_q (how the
  deviation evolves in T);
* point curves: q_a(t) and q_b(t) at one ZOI point.

Forces (Fc, Ff, per unit width F/h) have no map: E_inst is the mean over the
history samples of |F_b - F_a| / eps, E_bias = |mean F_b - mean F_a| / eps,
and the curves are the two force histories.

Sampling (fact, same rules as the studies)
------------------------------------------
* same element size in both runs (ms study, domain study): nearest element
  centroid, as `sample_zoi` of the domain and ms studies -> E_inst equals
  the E_q / eps_q they report;
* different element sizes (mesh study): linear interpolation of the element
  centroids (the bilinear sampling of gui.sensitivity.mesh_gci). The mesh
  study itself compares one space-time mean per quantity, not this
  pointwise metric: the maps show where its meshes differ point by point.
* Vx, Vy and T are masked by EVF >= evf_threshold of EACH run: a (point,
  frame) sample counts only where both runs are material (EVF is not
  masked);
* frames are paired by index and forces of the reference are interpolated on
  the history times of the compared run (`align_samples` of the ms study);
  frames offset by more than half a frame interval are refused.

Pure host-side module (no Qt).
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gui.sensitivity.domain_independence import (
    ALL_QUANTITIES, FIELD_QUANTITIES, FORCE_QUANTITIES, MASKED_QUANTITIES,
    ZoiSample, field_scalars, run_scalar, zoi_within_crop)
from gui.sensitivity.ms_independence import align_samples
from gui.sensitivity.run_cache import RunCache, CachedRun
from gui.sensitivity.runner_core import eulerian_instance
from gui.sensitivity.study_specs import GCI_TO_EPS, read_study_folder
from gui.sensitivity.zoi_sampling import (
    _nearest_indices, element_centroids_xy, roi_grid, window_mask)

# Relative difference under which two element sizes count as equal.
_SAME_SIZE_RTOL = 1e-9


# ---------------------------------------------------------------------------
# Runs of study folders
# ---------------------------------------------------------------------------
def run_ms(params: dict) -> float:
    """Mass-scaling factor in effect in a run's parameters (1 when off)."""
    step = (params or {}).get("step") or {}
    if not step.get("mass_scaling_enabled"):
        return 1.0
    try:
        return float(step.get("mass_scaling_factor_eulerian", 1.0))
    except (TypeError, ValueError):
        return 1.0


def run_elem_size(params: dict) -> float:
    try:
        return float(((params or {}).get("mesh") or {}).get("elem_size"))
    except (TypeError, ValueError):
        return float("nan")


def run_domain(params: dict) -> Dict[str, float]:
    geo = (((params or {}).get("geometry") or {}).get("euler") or {}
           ).get("geometry") or {}
    out = {}
    for k in ("h_wp", "h_void", "l_wp", "l_void"):
        try:
            out[k] = float(geo.get(k))
        except (TypeError, ValueError):
            out[k] = float("nan")
    return out


def run_label(run: CachedRun) -> str:
    """'ms 25600 | h 0.002 mm | D 0.2/0.2/0.2/0.2 | ms_run008 (folder)'."""
    p = run.params
    d = run_domain(p)
    return ("ms %g | h %g mm | D %g/%g/%g/%g | %s (%s)"
            % (run_ms(p), run_elem_size(p), d["h_wp"], d["h_void"],
               d["l_wp"], d["l_void"], run.job, run.folder.name))


def list_runs(folders: Sequence) -> List[CachedRun]:
    """Finished runs of the folders (the runs a study replay would reuse),
    sorted by element size (fine first), then ms, then domain size."""
    cache = RunCache([f for f in folders if f])

    def key(r: CachedRun):
        d = run_domain(r.params)
        return (run_elem_size(r.params), run_ms(r.params),
                sum(v for v in d.values() if math.isfinite(v)), r.job)
    return sorted(cache.runs, key=key)


def default_reference(runs: Sequence[CachedRun], step: Optional[str]) -> int:
    """Index of the natural reference: the smallest ms (ms study), the finest
    mesh (mesh study), the largest domain (domain study). -1 if no run."""
    if not runs:
        return -1
    if step == "mesh":
        vals = [run_elem_size(r.params) for r in runs]
        return int(np.nanargmin(vals))
    if step == "domain":
        vals = [sum(run_domain(r.params).values()) for r in runs]
        return int(np.nanargmax(vals))
    vals = [run_ms(r.params) for r in runs]
    return int(np.argmin(vals))


@dataclass
class CompareSettings:
    """The comparison settings of a study folder (config.json)."""
    step: Optional[str]
    zoi: Tuple[float, float, float, float]
    grid_step: float
    window: Tuple[float, float]
    thresholds: Dict[str, float]
    evf_threshold: float = 0.5


def _zoi_tuple(z) -> Tuple[float, float, float, float]:
    if isinstance(z, dict):
        return tuple(float(z[k]) for k in ("xmin", "xmax", "ymin", "ymax"))
    return tuple(float(v) for v in z)


def settings_of_folder(folder) -> CompareSettings:
    """Comparison settings of a study folder; raises ValueError with a
    message for the user."""
    # The safeguard limits play no role here: defaults for older folders.
    step, spec = read_study_folder(folder, {"rk_max": 0.05, "rhg_max": 0.05})
    if spec.get("thresholds_abs"):
        thr = {q: float(v) for q, v in spec["thresholds_abs"].items()
               if v is not None and q in ALL_QUANTITIES}
    else:                                   # mesh study: GCI names
        thr = {GCI_TO_EPS[g]: float(v)
               for g, v in (spec.get("tolerances") or {}).items()
               if v is not None and g in GCI_TO_EPS}
    grid = spec.get("grid_step")
    if grid is None:
        grid = spec.get("elem_size") or spec.get("finest_elem_size")
    return CompareSettings(
        step=step, zoi=_zoi_tuple(spec["zoi"]), grid_step=float(grid),
        window=tuple(float(v) for v in (spec.get("window") or (0.3, 1.0))),
        thresholds=thr,
        evf_threshold=float(spec.get("evf_threshold", 0.5)))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def grid_shape(zoi, grid_step: float) -> Tuple[int, int]:
    """(ny, nx) of `roi_grid(zoi, grid_step)` (row-major, x fastest)."""
    pts = roi_grid(zoi, grid_step)
    nx = int(np.unique(np.round(pts[:, 0], 12)).size)
    return pts.shape[0] // nx, nx


def _sampler(bundle, inst, points, linear: bool):
    """f(var) -> (n_frames, Np) values of `var` on `points`."""
    centroids = element_centroids_xy(bundle, inst)
    if linear:
        from gui.sensitivity.mesh_gci import apply_weights, interp_weights
        weights = interp_weights(centroids, points)
        return lambda var: apply_weights(
            np.asarray(bundle.field(inst, var), dtype=float), weights)
    idx = _nearest_indices(centroids, np.asarray(points, dtype=float))
    return lambda var: np.asarray(
        np.asarray(bundle.field(inst, var))[:, idx], dtype=float)


def sample_run(bundle, zoi, grid_step: float, elem_size: float,
               window=(0.3, 1.0), evf_threshold: float = 0.5,
               quantities: Sequence[str] = ALL_QUANTITIES,
               linear: bool = False) -> ZoiSample:
    """Windowed ZOI sample of one run (as `sample_zoi` of the studies, with
    linear interpolation when `linear`)."""
    inst = eulerian_instance(bundle)
    if not inst:
        raise RuntimeError("No Eulerian instance (EVF carrier) in the bundle.")
    crop = getattr(bundle, "roi", None)
    if isinstance(crop, dict) and not zoi_within_crop(zoi, crop):
        raise RuntimeError(
            "The ZOI is not contained in the extracted ROI of run %s."
            % getattr(bundle, "job_name", "?"))
    points = roi_grid(zoi, grid_step)
    times = np.asarray(bundle.times, dtype=float)
    tmask = window_mask(times, *window)
    if not tmask.any():
        raise RuntimeError("The time window contains no field frame.")
    out = ZoiSample(field_times=times[tmask])
    want = [q for q in quantities if q in FIELD_QUANTITIES]
    if want:
        get = _sampler(bundle, inst, points, linear)
        evf = get(FIELD_QUANTITIES["EVF"])[tmask]
        material = evf >= float(evf_threshold)
        for q in want:
            if q == "EVF":
                out.fields[q] = evf
                continue
            try:
                arr = get(FIELD_QUANTITIES[q])[tmask]
            except KeyError:
                continue                  # field not written by this run
            if q in MASKED_QUANTITIES:
                arr = np.where(material, arr, np.nan)
            out.fields[q] = arr
        sc = field_scalars(bundle, inst, zoi, [q for q in want
                                               if q in out.fields],
                           tmask, evf_threshold, elem_size)
        out.scalars = {q: v for q, v in sc.items()}
    forces = [q for q in quantities if q in FORCE_QUANTITIES]
    if forces:
        try:
            ht = np.asarray(bundle.history_time, dtype=float)
        except Exception:
            ht = np.zeros(0)
        hmask = window_mask(ht, *window) if ht.size else np.zeros(0, bool)
        out.force_times = ht[hmask] if ht.size else ht
        for q in forces:
            try:
                f = np.asarray(bundle.history(FORCE_QUANTITIES[q]),
                               dtype=float)
            except Exception:
                continue
            if f.size != ht.size or not hmask.any():
                continue
            out.forces[q] = f[hmask] / float(elem_size)
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
@dataclass
class QuantityDeviation:
    """Deviation of one quantity between the reference a and the run b."""
    quantity: str
    eps: float
    is_field: bool
    times: np.ndarray                     # (Nt,) frame or history times
    a: np.ndarray                         # (Nt, Np) fields, (Nt,) forces
    b: np.ndarray
    e_inst: float = float("nan")          # pointwise E_q / eps_q
    e_mean: float = float("nan")          # |m_q(b) - m_q(a)| / eps_q: the
                                          # metric that decides the studies
    e_bias: float = float("nan")
    e_frame: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def frame_map(self, k: int) -> np.ndarray:
        """|b - a| / eps at frame k, (Np,) (NaN where not compared)."""
        return np.abs(self.b[k] - self.a[k]) / self.eps

    def mean_map(self) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.abs(self.b - self.a), axis=0) / self.eps

    def bias_map(self) -> np.ndarray:
        both = np.isfinite(self.a) & np.isfinite(self.b)
        a = np.where(both, self.a, np.nan)
        b = np.where(both, self.b, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.abs(np.nanmean(b, axis=0) - np.nanmean(a, axis=0)
                          ) / self.eps

    def n_compared(self) -> np.ndarray:
        """Number of frames compared per point (both runs material)."""
        return (np.isfinite(self.a) & np.isfinite(self.b)).sum(axis=0)


@dataclass
class PairDeviation:
    """All quantities of one pair, on the ZOI grid of the settings."""
    settings: CompareSettings
    points: np.ndarray                    # (Np, 2)
    shape: Tuple[int, int]                # (ny, nx)
    linear: bool
    quantities: Dict[str, QuantityDeviation] = field(default_factory=dict)
    frame_offset_over_interval: float = float("nan")
    label_a: str = ""
    label_b: str = ""

    def nearest_point(self, x: float, y: float) -> int:
        d = (self.points[:, 0] - x) ** 2 + (self.points[:, 1] - y) ** 2
        return int(np.argmin(d))

    def summary_rows(self) -> List[Dict[str, object]]:
        rows = []
        for q, d in self.quantities.items():
            rows.append({"quantity": q, "eps": d.eps,
                         "E_mean_over_eps": d.e_mean,
                         "E_inst_over_eps": d.e_inst,
                         "E_bias_over_eps": d.e_bias,
                         "bias_share": (d.e_bias / d.e_inst
                                        if d.e_inst and math.isfinite(d.e_inst)
                                        else None)})
        return rows

    def map_rows(self) -> List[Dict[str, object]]:
        """One row per ZOI point: mean and bias maps of every field."""
        rows = []
        maps = {q: (d.mean_map(), d.bias_map(), d.n_compared())
                for q, d in self.quantities.items() if d.is_field}
        for i, (x, y) in enumerate(self.points):
            r = {"x_mm": float(x), "y_mm": float(y)}
            for q, (m, b, n) in maps.items():
                r["%s_mean_over_eps" % q] = float(m[i])
                r["%s_bias_over_eps" % q] = float(b[i])
                r["%s_n_frames" % q] = int(n[i])
            rows.append(r)
        return rows


def _mean_abs(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    return float(np.mean(np.abs(d))) if d.size else float("nan")


def compare_samples(sa: ZoiSample, sb: ZoiSample, settings: CompareSettings,
                    linear: bool = False) -> PairDeviation:
    """PairDeviation of b against the reference a (samples of `sample_run`)."""
    a2, b2, info = align_samples(sa, sb)
    points = roi_grid(settings.zoi, settings.grid_step)
    out = PairDeviation(settings=settings, points=points,
                        shape=grid_shape(settings.zoi, settings.grid_step),
                        linear=linear,
                        frame_offset_over_interval=float(info.get(
                            "frame_offset_over_interval", float("nan"))))
    for q, eps in settings.thresholds.items():
        if not eps or eps <= 0:
            continue
        if q in FIELD_QUANTITIES and q in a2.fields and q in b2.fields:
            a = np.asarray(a2.fields[q], dtype=float)
            b = np.asarray(b2.fields[q], dtype=float)
            d = QuantityDeviation(q, float(eps), True,
                                  np.asarray(b2.field_times, float), a, b)
            diff = np.abs(b - a)
            d.e_inst = _mean_abs(diff) / eps
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d.e_frame = np.nanmean(diff, axis=1) / eps
            bias = d.bias_map()
            bias = bias[np.isfinite(bias)]
            d.e_bias = float(np.mean(bias)) if bias.size else float("nan")
        elif q in FORCE_QUANTITIES and q in a2.forces and q in b2.forces:
            a = np.asarray(a2.forces[q], dtype=float)
            b = np.asarray(b2.forces[q], dtype=float)
            d = QuantityDeviation(q, float(eps), False,
                                  np.asarray(b2.force_times, float), a, b)
            d.e_inst = _mean_abs(b - a) / eps
            d.e_bias = (abs(float(np.mean(b)) - float(np.mean(a))) / eps
                        if a.size else float("nan"))
            d.e_frame = np.abs(b - a) / eps
        else:
            continue
        ma, mb = run_scalar(sa, q), run_scalar(sb, q)
        d.e_mean = (abs(mb - ma) / eps if math.isfinite(ma)
                    and math.isfinite(mb) else float("nan"))
        out.quantities[q] = d
    return out


def compare_runs(bundle_a, bundle_b, settings: CompareSettings,
                 elem_a: float, elem_b: float) -> PairDeviation:
    """PairDeviation of run b against the reference run a. Nearest-centroid
    sampling when both runs have the same element size, linear otherwise.
    Forces per unit width use each run's own element size."""
    linear = not math.isclose(float(elem_a), float(elem_b),
                              rel_tol=_SAME_SIZE_RTOL)
    q = tuple(settings.thresholds)
    kw = dict(window=settings.window, evf_threshold=settings.evf_threshold,
              quantities=q, linear=linear)
    sa = sample_run(bundle_a, settings.zoi, settings.grid_step, elem_a, **kw)
    sb = sample_run(bundle_b, settings.zoi, settings.grid_step, elem_b, **kw)
    return compare_samples(sa, sb, settings, linear=linear)


def write_pair_exports(folder, pair: PairDeviation, stem: str) -> List[Path]:
    """<stem>_summary.csv and <stem>_maps.csv (+ .xlsx copies)."""
    from gui.sensitivity.study_export import write_csv
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    head = {"reference": pair.label_a, "compared": pair.label_b,
            "sampling": "linear" if pair.linear else "nearest centroid",
            "frame_offset_over_interval": pair.frame_offset_over_interval}
    summary = [dict(head, **r) for r in pair.summary_rows()]
    return [write_csv(folder / (stem + "_summary.csv"), summary),
            write_csv(folder / (stem + "_maps.csv"), pair.map_rows())]
