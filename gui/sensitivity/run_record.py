# -*- coding: utf-8 -*-
"""Per-run safeguards and cost descriptors (correction report, lot L2).

Safeguards (paper Eq. 9-10, decisions D3, D4, D6 of 2026-10-01/02)
--------------------------------------------------------------------
Each run of a sizing study is checked against:

* ``outputs`` - every output the study needs is present in the bundle
  (Eulerian fields EVF, TEMP, V1, V2; RP forces RF1_RP, RF2_RP; energies
  ALLKE, ALLIE, ALLAE). A safeguard that cannot be evaluated counts as a
  failure (report, Part B, T5);
* ``R_K`` - kinetic over internal energy, aggregated over the window T::

      R_K = sum_{t in T} ALLKE(t) / sum_{t in T} ALLIE(t)

  The ratio of sums avoids the division by ALLIE ~ 0 at the start of the step
  that made the mean of the per-sample ratio explode (D3-a; same form as
  gui.sensitivity.domain_opt.mean_ke_ie_ratio, here restricted to T);
* ``R_HG`` - artificial (hourglass-control) over internal energy, same form
  with ALLAE (D4: ALLAE is the artificial strain energy of the constraints
  that remove singular modes, such as hourglass control, per the Abaqus
  documentation quoted by Tristan).

Default thresholds:

* G_HG,max = 0.05 - decision of the author (2026-10-02), modifiable; no
  external source.
* G_K,max = 0.01 - taken from the energy-guard default of
  ModelConfig.mass_scaling_bounds (gui/core/model_config.py:822). That value
  was set for the mass-scaling window and is NOT validated for this study:
  to be confirmed.

Open hypotheses (to check on the first real run, report T5): H1 the Abaqus
documentation on EC3D8R holds for EC3D8RT; H2 PRESELECT contains ALLAE; H3
with the pure viscous hourglass form, the hourglass work appears in ALLAE.

Energy time base
----------------
The energies are restricted to T with their own time vector:
``ENERGY_TIME`` for ALLKE/ALLIE and ``ALLAE_TIME`` for ALLAE, written by
abaqus_scripts/cel_results.py. Bundles older than that fall back on
``history_time`` when the lengths match; otherwise the safeguard is reported
as not evaluable (failure).

Cost (paper Eq. 11, decision D7)
--------------------------------
N_elem (Eulerian and tool instances, from the bundle), N_inc and the stable
increment (first / minimum / last) and the solver wall time from the .sta,
the host wall time (launch to bundle, CAE pre-processing and extraction
included), N_CPU, and C_CPU = N_CPU * t_wall,solver. Eq. (11) uses the
SOLVER time (D7); the host time is recorded for information.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from gui.core.sta_parser import parse_sta
from gui.sensitivity.domain_convergence import window_mask
from gui.sensitivity.runner_core import eulerian_instance

REQUIRED_FIELDS = ("EVF", "TEMP", "V1", "V2")
REQUIRED_HISTORY = ("RF1_RP", "RF2_RP", "ALLKE", "ALLIE", "ALLAE")

DEFAULT_RK_MAX = 0.01      # model_config.py:822 (mass-scaling guard default)
DEFAULT_RHG_MAX = 0.05     # author's decision, 2026-10-02

# Guard result: (value or None when not evaluable, ok, reason)
GuardValue = Tuple[Optional[float], bool]


# ---------------------------------------------------------------------------
# Energy ratios
# ---------------------------------------------------------------------------
def _channel(bundle, name: str) -> Optional[np.ndarray]:
    try:
        v = np.asarray(bundle.history(name), dtype=float)
    except Exception:
        return None
    return v if v.size else None


def _time_for(bundle, time_channel: str, n: int) -> Optional[np.ndarray]:
    """Time vector of length n for an energy channel (own time, else the
    generic history time when it has the same length)."""
    t = _channel(bundle, time_channel)
    if t is not None and t.size == n:
        return t
    try:
        ht = np.asarray(bundle.history_time, dtype=float)
    except Exception:
        return None
    return ht if ht.size == n else None


def windowed_energy_ratio(bundle, num: str, den: str, num_time: str,
                          den_time: str,
                          window: Tuple[float, float] = (0.3, 1.0)
                          ) -> Tuple[Optional[float], str]:
    """sum_T num / sum_T den, each channel restricted to the window T with
    its own time vector. Returns (value, "") or (None, reason)."""
    a = _channel(bundle, num)
    b = _channel(bundle, den)
    if a is None:
        return None, "%s absent" % num
    if b is None:
        return None, "%s absent" % den
    ta = _time_for(bundle, num_time, a.size)
    tb = _time_for(bundle, den_time, b.size)
    if ta is None or tb is None:
        return None, "no time base to restrict %s/%s to T" % (num, den)
    ma = window_mask(ta, *window)
    mb = window_mask(tb, *window)
    if not ma.any() or not mb.any():
        return None, "window T contains no energy sample"
    s_den = float(np.nansum(b[mb]))
    if not s_den > 0.0:
        return None, "sum of %s over T is not positive" % den
    return float(np.nansum(a[ma])) / s_den, ""


# ---------------------------------------------------------------------------
# Safeguards
# ---------------------------------------------------------------------------
@dataclass
class GuardSettings:
    """Thresholds and window of the run safeguards (all modifiable)."""
    rk_max: float = DEFAULT_RK_MAX
    rhg_max: float = DEFAULT_RHG_MAX
    window: Tuple[float, float] = (0.3, 1.0)
    required_fields: Sequence[str] = REQUIRED_FIELDS
    required_history: Sequence[str] = REQUIRED_HISTORY


def missing_outputs(bundle, settings: GuardSettings) -> list:
    """Names of the required outputs absent from the bundle."""
    missing = []
    inst = eulerian_instance(bundle)
    avail = []
    if inst:
        try:
            avail = list(bundle.instance(inst).field_variables)
        except Exception:
            avail = []
    for v in settings.required_fields:
        if v not in avail:
            missing.append(v)
    for h in settings.required_history:
        if _channel(bundle, h) is None:
            missing.append(h)
    return missing


def evaluate_guards(bundle, settings: Optional[GuardSettings] = None
                    ) -> Dict[str, Tuple[Optional[float], bool]]:
    """{name: (value, ok)} for the safeguards of one run.

    Keys: "outputs" (number of missing outputs), "R_K", "R_HG". A safeguard
    that cannot be evaluated is (None, False)."""
    s = settings or GuardSettings()
    out: Dict[str, Tuple[Optional[float], bool]] = {}
    miss = missing_outputs(bundle, s)
    out["outputs"] = (float(len(miss)), not miss)
    rk, _ = windowed_energy_ratio(bundle, "ALLKE", "ALLIE", "ENERGY_TIME",
                                  "ENERGY_TIME", s.window)
    out["R_K"] = (rk, rk is not None and rk < s.rk_max)
    rhg, _ = windowed_energy_ratio(bundle, "ALLAE", "ALLIE", "ALLAE_TIME",
                                   "ENERGY_TIME", s.window)
    out["R_HG"] = (rhg, rhg is not None and rhg < s.rhg_max)
    return out


def guard_reasons(bundle, settings: Optional[GuardSettings] = None
                  ) -> Dict[str, str]:
    """Human-readable reason for every safeguard that is not evaluable."""
    s = settings or GuardSettings()
    reasons: Dict[str, str] = {}
    miss = missing_outputs(bundle, s)
    if miss:
        reasons["outputs"] = "missing: " + ", ".join(miss)
    _v, why = windowed_energy_ratio(bundle, "ALLKE", "ALLIE", "ENERGY_TIME",
                                    "ENERGY_TIME", s.window)
    if why:
        reasons["R_K"] = why
    _v, why = windowed_energy_ratio(bundle, "ALLAE", "ALLIE", "ALLAE_TIME",
                                    "ENERGY_TIME", s.window)
    if why:
        reasons["R_HG"] = why
    return reasons


def make_guard_fn(settings: Optional[GuardSettings] = None
                  ) -> Callable[[object], Dict[str, Tuple[Optional[float], bool]]]:
    """guard_fn(bundle) for gui.sensitivity.domain_independence."""
    s = settings or GuardSettings()
    return lambda bundle: evaluate_guards(bundle, s)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
@dataclass
class CostRecord:
    """Cost descriptors of one run (None = not available).

    n_elem_euler is the MODEL count of Eulerian elements, computed from the
    domain dimensions (structured mesh of uniform size h, one element through
    the thickness: cel_model.py:298 and :412-418). The *_extracted counts are
    what the bundle holds, which is only the output-ROI crop when the ROI
    filter is active (cel_results.py:25-54): they are NOT the model cost."""
    n_elem_euler: Optional[int] = None
    n_elem_euler_extracted: Optional[int] = None
    n_elem_tool_extracted: Optional[int] = None
    n_inc: Optional[int] = None
    dt_stable_first: Optional[float] = None
    dt_stable_min: Optional[float] = None
    dt_stable_last: Optional[float] = None
    t_wall_solver_s: Optional[float] = None
    t_wall_host_s: Optional[float] = None
    n_cpu: Optional[int] = None
    c_cpu_s: Optional[float] = None          # Eq. (11): N_CPU * t_wall,solver

    def as_dict(self) -> dict:
        return asdict(self)


def _instance_elements(bundle) -> Tuple[Optional[int], Optional[int]]:
    """(N_elem Eulerian, N_elem of the other instance) from the bundle."""
    eul = eulerian_instance(bundle)
    n_eul = n_other = None
    try:
        names = bundle.instance_names
        names = names() if callable(names) else names
        for name in names:
            n = int(getattr(bundle.instance(name), "n_elements", 0) or 0)
            if name == eul:
                n_eul = n
            elif n_other is None:
                n_other = n
    except Exception:
        pass
    return n_eul, n_other


def euler_element_count(dims, elem_size: float) -> int:
    """Number of Eulerian elements of the model for `dims` and size h.

    The dimensions are floored to whole elements by cel_common.discretize
    before meshing (cel_model.py:173-177); this mirrors it with a rounding
    tolerance so that values already on the grid are not lost to round-off."""
    h = float(elem_size)
    nx = int(math.floor((dims.l_wp + dims.l_void) / h + 1e-6))
    ny = int(math.floor((dims.h_wp + dims.h_void) / h + 1e-6))
    return max(0, nx) * max(0, ny)


def cost_record(bundle=None, sta_path=None, host_wall_s: Optional[float] = None,
                n_cpu: Optional[int] = None, dims=None,
                elem_size: Optional[float] = None) -> CostRecord:
    """Assemble the cost of one run from what is available.

    dims, elem_size : the run's domain and Eulerian element size, used for
        the model element count (the bundle count may be an ROI crop)."""
    rec = CostRecord(t_wall_host_s=host_wall_s,
                     n_cpu=None if n_cpu is None else int(n_cpu))
    if dims is not None and elem_size:
        rec.n_elem_euler = euler_element_count(dims, elem_size)
    if bundle is not None:
        rec.n_elem_euler_extracted, rec.n_elem_tool_extracted = \
            _instance_elements(bundle)
    if sta_path is not None and Path(sta_path).exists():
        snap = parse_sta(sta_path)
        rec.n_inc = snap.inc_number
        rec.dt_stable_first = snap.stable_dt_first
        rec.dt_stable_min = snap.stable_dt_min
        rec.dt_stable_last = snap.stable_dt
        rec.t_wall_solver_s = snap.wall_time_seconds()
    if rec.n_cpu is not None and rec.t_wall_solver_s is not None:
        rec.c_cpu_s = float(rec.n_cpu) * rec.t_wall_solver_s
    return rec
