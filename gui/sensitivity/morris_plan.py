# -*- coding: utf-8 -*-
"""
Morris sensitivity plan — the sampling engine (Lot 2b).

Turns a user selection of variable parameters (+ min/max in *displayed*
engineering units) into:
  1. a SALib Morris problem and trajectory sample matrix X, and
  2. one ready-to-run ModelConfig per sample row ("profile").

After the runs, `analyze` reduces the QoI vectors to Morris indices
(mu_star, sigma) per parameter.

Units
-----
The UI lets the user enter bounds in the SAME engineering units shown in
the Materials / BCs tabs (e.g. E in GPa, cutting speed in m/min). SALib
samples within those displayed bounds; each sampled value is written back
into a ModelConfig via `param_registry.apply_display`, which converts to
the Abaqus-internal stored value the solver expects.

Cost
----
Morris runs ``N * (k + 1)`` simulations for ``k`` parameters and ``N``
trajectories. `n_runs()` reports this up front so the UI can warn before
launching a multi-hour batch.

SALib API (verified against SALib 1.5.2)
----------------------------------------
  SALib.sample.morris.sample(problem, N, num_levels=4, seed=None) -> X
  SALib.analyze.morris.analyze(problem, X, Y, num_levels=4, seed=None)
      -> {'names', 'mu', 'mu_star', 'sigma', 'mu_star_conf'}
  problem = {'num_vars': k, 'names': [...], 'bounds': [[lo, hi], ...]}
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional
import numpy as np

from gui.sensitivity import param_registry as pr


# ---------------------------------------------------------------------------
# Plan container
# ---------------------------------------------------------------------------
@dataclass
class MorrisPlan:
    """A generated Morris sampling plan, in DISPLAYED units."""
    specs: list                      # list[ParamSpec], the varied parameters
    bounds: list                     # list[(lo, hi)] in displayed units
    N: int                           # number of trajectories
    num_levels: int                  # Morris grid levels
    problem: dict                    # SALib problem dict
    X: np.ndarray                    # (n_runs, k) sample, displayed units
    temp_unit: str = "C"
    seed: Optional[int] = None
    # UnitSystem the displayed values are expressed in (snapshot taken at
    # generation). None = the active system at conversion time (legacy).
    unit_system: object = None

    @property
    def param_paths(self) -> list:
        return [s.path for s in self.specs]

    @property
    def n_runs(self) -> int:
        return int(self.X.shape[0])

    @property
    def k(self) -> int:
        return len(self.specs)


def n_runs(k: int, N: int) -> int:
    """Number of model evaluations a Morris design needs: N*(k+1)."""
    return int(N * (k + 1))


# ---------------------------------------------------------------------------
# Build the plan
# ---------------------------------------------------------------------------
def build_plan(selected, N, num_levels=4, seed=None, temp_unit="C",
               unit_system=None):
    """Build a Morris plan.

    Parameters
    ----------
    selected : list of (ParamSpec, lo, hi)
        The parameters to vary and their min/max in DISPLAYED units.
    N : int
        Number of Morris trajectories (total runs = N*(k+1)).
    num_levels : int
        Number of grid levels (Morris ``p``); 4 is the common default.
    seed : int | None
        RNG seed for reproducibility. When None a seed is drawn here and
        stored on the plan, so every plan can be regenerated exactly.
    unit_system : UnitSystem | None
        The system the displayed bounds are expressed in (default: a
        snapshot of the active one).

    Returns a MorrisPlan. Raises ValueError on an empty/invalid selection.
    """
    if not selected:
        raise ValueError("Select at least one parameter to vary.")
    specs, bounds = [], []
    for spec, lo, hi in selected:
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            raise ValueError(
                "Parameter %r: max (%g) must be greater than min (%g)."
                % (spec.path, hi, lo))
        specs.append(spec)
        bounds.append((lo, hi))

    problem = {
        "num_vars": len(specs),
        "names": [s.path for s in specs],
        "bounds": [[lo, hi] for (lo, hi) in bounds],
    }
    # Imported lazily so the rest of the GUI works even if SALib isn't
    # installed yet (the Sensitivity tab surfaces a clear message instead).
    from SALib.sample.morris import sample as morris_sample
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
    if unit_system is None:
        unit_system = pr.current_system(temp_unit)
    X = morris_sample(problem, N=int(N), num_levels=int(num_levels), seed=seed)
    return MorrisPlan(specs=specs, bounds=bounds, N=int(N),
                      num_levels=int(num_levels), problem=problem,
                      X=np.asarray(X, dtype=float), temp_unit=temp_unit,
                      seed=int(seed), unit_system=unit_system)


# ---------------------------------------------------------------------------
# Materialise the plan into ModelConfig profiles
# ---------------------------------------------------------------------------
def plan_to_configs(base_cfg, plan: MorrisPlan):
    """Return one deep-copied ModelConfig per sample row, with the varied
    parameters set to their sampled (displayed -> stored) values. All other
    parameters keep the base_cfg value."""
    configs = []
    for row in plan.X:
        cfg = copy.deepcopy(base_cfg)
        for spec, value in zip(plan.specs, row):
            pr.apply_display(cfg, spec, float(value), plan.temp_unit,
                             system=plan.unit_system)
        configs.append(cfg)
    return configs


def profile_table(plan: MorrisPlan):
    """A plain table (list of dict) of the plan for display/export:
    one row per run, columns = parameter display values (in display units)."""
    rows = []
    for i, row in enumerate(plan.X):
        d = {"run": i + 1}
        for spec, value in zip(plan.specs, row):
            d[spec.path] = float(value)
        rows.append(d)
    return rows


# ---------------------------------------------------------------------------
# Analyse (after the runs)
# ---------------------------------------------------------------------------
def analyze(plan: MorrisPlan, Y, num_resamples=100, conf_level=0.95,
            seed=None):
    """Compute Morris indices for one QoI vector Y (length == n_runs).

    Returns the SALib dict: {'names', 'mu', 'mu_star', 'sigma',
    'mu_star_conf'}. NaN entries in Y (failed runs) are not allowed by
    SALib, so callers should drop/repair them first (see analyze_safe)."""
    from SALib.analyze.morris import analyze as morris_analyze
    Y = np.asarray(Y, dtype=float)
    return morris_analyze(plan.problem, plan.X, Y,
                          num_resamples=num_resamples,
                          conf_level=conf_level,
                          num_levels=plan.num_levels, seed=seed)


def analyze_safe(plan: MorrisPlan, Y):
    """Like analyze(), but tolerant of failed runs: NaN/inf QoI values are
    replaced by the mean of the finite ones so SALib doesn't choke. Returns
    (result_dict, n_bad) where n_bad is the number of repaired runs. If too
    many runs failed (< 2 finite), returns (None, n_bad)."""
    Y = np.asarray(Y, dtype=float).copy()
    bad = ~np.isfinite(Y)
    n_bad = int(bad.sum())
    finite = Y[~bad]
    if finite.size < 2:
        return None, n_bad
    if n_bad:
        Y[bad] = float(np.mean(finite))
    return analyze(plan, Y), n_bad


def analyze_complete(plan: MorrisPlan, Y):
    """Morris indices from the COMPLETE trajectories only.

    A trajectory is k+1 consecutive runs; each elementary effect is the
    difference between two neighbouring runs of one trajectory. A failed
    (or cancelled, never-run) run therefore spoils its whole trajectory.
    Unlike analyze_safe, which fills missing runs with the mean of the others
    and so fabricates elementary effects, this drops every trajectory that
    holds a non-finite QoI value and analyses the remaining ones.

    Returns (result_dict | None, info) with
        info = {"n_trajectories", "n_used", "n_dropped", "n_bad_runs"}.
    result_dict is None when fewer than 2 complete trajectories remain:
    SALib's sigma is a sample standard deviation (ddof=1) and needs 2.
    """
    Y = np.asarray(Y, dtype=float)
    size = plan.k + 1
    n_traj = Y.size // size
    finite = np.isfinite(Y[:n_traj * size]).reshape(n_traj, size)
    keep = finite.all(axis=1)
    info = {"n_trajectories": int(n_traj), "n_used": int(keep.sum()),
            "n_dropped": int(n_traj - keep.sum()),
            "n_bad_runs": int((~np.isfinite(Y)).sum())}
    if keep.sum() < 2:
        return None, info
    rows = np.repeat(keep, size)
    from SALib.analyze.morris import analyze as morris_analyze
    res = morris_analyze(plan.problem, plan.X[:n_traj * size][rows],
                         Y[:n_traj * size][rows],
                         num_levels=plan.num_levels, seed=plan.seed)
    return res, info
