# -*- coding: utf-8 -*-
"""Search engine of the Identify tab (design validated 2026-11-06).

Problem
-------
Find the values of k parameters p (each within [lo, hi]) whose QoI q(p)
match reference values q_ref (measured, or given by a reference model),
each QoI being judged against its admitted deviation eps_q.

Cost
----
    J(p) = sum_q w_q ((q(p) - q_ref) / eps_q)^2 / sum_q w_q

the weighted mean of the squared deviations in units of eps_q (w_q = 1 by
default). Identify succeeds when EVERY QoI with a weight > 0 is within its
eps_q: |q - q_ref| <= eps_q (same rule as Model sizing). A run budget also
stops the search.

Strategies
----------
* ``fixed_point``: one parameter at a time, the others held, by a bounded
  golden-section search on J over [lo, hi] (about 10 runs per parameter to
  bring the interval to 1 %: 0.618^10 ~ 0.8 %), cycles until the QoI are
  within eps_q.
* ``lm``: Levenberg-Marquardt on the residuals r_q = sqrt(w_q / W)
  (q - q_ref) / eps_q, Jacobian by finite differences (k + 1 runs per
  iteration forward, 2k + 1 central), steps projected on the bounds.

Stagnation (rule of 2026-11-06)
-------------------------------
A parameter is judged by the effect of its last step on the QoI, not by
the size of the step: p_j stagnates when max_q |S_qj dp_j| / eps_q < tau
for `n_stag` iterations in a row, S_qj = dq/dp_j the measured sensitivity
(LM: the Jacobian of the iteration; fixed point: the slope of the last
bracket of the 1-D search) and dp_j its last step. A stagnating parameter
is frozen at its value and leaves the plan. When no free parameter is left
and the QoI are not all within eps_q, one last iteration is run with every
parameter free (a parameter frozen early may have to move once the others
have moved). tau = 0.1 and n_stag = 2 are choices (validated by Tristan),
not sourced values.

The engine never runs a simulation itself: it calls
``evaluate(points) -> [ {qid: value} or None ]`` with a list of parameter
vectors (a list, so that independent runs, e.g. those of a finite-
difference Jacobian, can be launched together). A failed run is None.
Every evaluation is memoised on its parameter vector, and `prior`
evaluations (the history of an interrupted identification) are reused
without new runs: the search is deterministic, so resuming replays it.

Pure module (numpy only, no Qt).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

STRATEGIES = ("fixed_point", "lm")
FD_SCHEMES = ("forward", "central")
DEFAULT_TAU = 0.1          # choice validated 2026-11-06
DEFAULT_N_STAG = 2         # choice validated 2026-11-06
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0      # 0.618...

# Status of a finished search
CONVERGED = "converged"        # every QoI within eps_q
BUDGET = "budget"              # run budget spent
STALLED = "stalled"            # every parameter frozen, last free pass done
NO_PROGRESS = "no_progress"    # LM: no step lowers J
FAILED = "failed"              # the start point could not be evaluated


@dataclass
class Param:
    """One parameter to identify, in the units the evaluator expects."""
    path: str
    lo: float
    hi: float
    start: Optional[float] = None      # None: middle of [lo, hi]
    scale: Optional[float] = None      # None: hi - lo

    def __post_init__(self):
        self.lo, self.hi = float(self.lo), float(self.hi)
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)) \
                or self.hi <= self.lo:
            raise ValueError("%s: Max must be greater than Min." % self.path)
        if self.start is None:
            self.start = 0.5 * (self.lo + self.hi)
        self.start = float(self.start)
        if not self.lo <= self.start <= self.hi:
            raise ValueError("%s: the start point %g is outside [%g, %g]."
                             % (self.path, self.start, self.lo, self.hi))
        if self.scale is None:
            self.scale = self.hi - self.lo
        self.scale = float(self.scale)
        if not self.scale > 0:
            raise ValueError("%s: the scale must be > 0." % self.path)


@dataclass
class Target:
    """One QoI to match: reference value, admitted deviation, weight."""
    qid: str
    ref: float
    eps: float
    weight: float = 1.0

    def __post_init__(self):
        self.ref, self.eps, self.weight = (float(self.ref), float(self.eps),
                                           float(self.weight))
        if not self.eps > 0:
            raise ValueError("%s: eps_q must be > 0." % self.qid)
        if self.weight < 0:
            raise ValueError("%s: the weight must be >= 0." % self.qid)


@dataclass
class Evaluation:
    """One run of the search."""
    run: int                         # 1-based, in launch order
    x: List[float]
    q: Optional[Dict[str, float]]    # None: failed run
    J: float
    phase: str                       # start | fd | step | line
    iteration: int
    reused: bool = False             # read from `prior`, not launched

    def to_dict(self, paths) -> dict:
        return {"run": self.run, "x": dict(zip(paths, self.x)),
                "q": self.q, "J": self.J, "phase": self.phase,
                "iteration": self.iteration, "reused": self.reused}


@dataclass
class Iteration:
    """State after one iteration (LM step or fixed-point cycle)."""
    iteration: int
    x: List[float]
    J: float
    max_ratio: float                 # max_q |q - q_ref| / eps_q
    free: List[str]
    frozen: List[str]
    runs: int                        # runs launched so far
    effect: Dict[str, float] = field(default_factory=dict)
    # effect[path] = max_q |S_qj dp_j| / eps_q of this iteration's step


@dataclass
class IdentifyResult:
    status: str
    x: List[float]                   # best point found
    q: Optional[Dict[str, float]]
    J: float
    max_ratio: float
    n_runs: int                      # runs launched (not reused)
    evaluations: List[Evaluation]
    iterations: List[Iteration]
    frozen: List[str]
    message: str = ""

    def record(self, params: Sequence[Param],
               targets: Sequence[Target]) -> dict:
        """JSON-ready history (parameters, J, deviations per QoI), saved
        with the identification so it can be resumed."""
        paths = [p.path for p in params]
        return {
            "status": self.status, "message": self.message,
            "x": dict(zip(paths, self.x)), "q": self.q, "J": self.J,
            "max_ratio": self.max_ratio, "n_runs": self.n_runs,
            "frozen": list(self.frozen),
            "params": [{"path": p.path, "lo": p.lo, "hi": p.hi,
                        "start": p.start, "scale": p.scale} for p in params],
            "targets": [{"qid": t.qid, "ref": t.ref, "eps": t.eps,
                         "weight": t.weight} for t in targets],
            "evaluations": [e.to_dict(paths) for e in self.evaluations],
            "iterations": [{
                "iteration": it.iteration, "x": dict(zip(paths, it.x)),
                "J": it.J, "max_ratio": it.max_ratio, "free": it.free,
                "frozen": it.frozen, "runs": it.runs, "effect": it.effect}
                for it in self.iterations],
        }


def cost(q: Optional[Dict[str, float]], targets: Sequence[Target]) -> float:
    """J: weighted mean of ((q - q_ref) / eps_q)^2; inf for a failed run or
    a missing QoI."""
    r = residuals(q, targets)
    return float(np.sum(r * r)) if np.all(np.isfinite(r)) else math.inf


def residuals(q: Optional[Dict[str, float]],
              targets: Sequence[Target]) -> np.ndarray:
    """r_q = sqrt(w_q / W) (q - q_ref) / eps_q, so that sum r^2 = J."""
    W = sum(t.weight for t in targets) or 1.0
    out = np.full(len(targets), np.nan)
    if q is None:
        return out
    for i, t in enumerate(targets):
        v = q.get(t.qid)
        if v is None or not math.isfinite(float(v)):
            continue
        out[i] = math.sqrt(t.weight / W) * (float(v) - t.ref) / t.eps
    return out


def max_ratio(q: Optional[Dict[str, float]],
              targets: Sequence[Target]) -> float:
    """max_q |q - q_ref| / eps_q over the targets with a weight > 0."""
    if q is None:
        return math.inf
    worst = 0.0
    for t in targets:
        if t.weight <= 0:
            continue
        v = q.get(t.qid)
        if v is None or not math.isfinite(float(v)):
            return math.inf
        worst = max(worst, abs(float(v) - t.ref) / t.eps)
    return worst


def converged(q, targets) -> bool:
    """Every QoI with a weight > 0 within its eps_q."""
    return max_ratio(q, targets) <= 1.0


def step_effect(S: np.ndarray, dx: np.ndarray,
                targets: Sequence[Target]) -> np.ndarray:
    """max_q |S_qj dx_j| / eps_q per parameter j (S: n_q x k, dq/dp)."""
    eps = np.array([t.eps for t in targets], float)
    with np.errstate(invalid="ignore"):
        E = np.abs(S * dx[None, :]) / eps[:, None]
    E = np.where(np.isfinite(E), E, 0.0)
    return E.max(axis=0) if E.size else np.zeros(len(dx))


def identifiability(S: np.ndarray, params: Sequence[Param],
                    targets: Sequence[Target]) -> dict:
    """Conditioning of the sensitivities, shown before a search: S (n_q x
    k, dq/dp) normalised to (dq / eps_q) per (dp / scale). A large
    condition number, or two parameters with nearly proportional effects
    (|cos| close to 1), means they cannot be told apart with these QoI.
    Returns {"cond": float, "pairs": [(path_i, path_j, cos)] sorted by
    |cos| decreasing}."""
    eps = np.array([t.eps for t in targets], float)
    sc = np.array([p.scale for p in params], float)
    A = np.asarray(S, float) * sc[None, :] / eps[:, None]
    A = np.where(np.isfinite(A), A, 0.0)
    sv = np.linalg.svd(A, compute_uv=False) if A.size else np.array([])
    cond = (float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0
            else math.inf)
    norms = np.linalg.norm(A, axis=0)
    pairs = []
    for i in range(A.shape[1]):
        for j in range(i + 1, A.shape[1]):
            if norms[i] > 0 and norms[j] > 0:
                c = float(A[:, i] @ A[:, j] / (norms[i] * norms[j]))
                pairs.append((params[i].path, params[j].path, c))
    pairs.sort(key=lambda t: -abs(t[2]))
    return {"cond": cond, "pairs": pairs}


def golden_runs(tol: float) -> int:
    """Runs of a golden-section search to bring [lo, hi] to tol of its
    width: 2 to start, then one per reduction by 0.618."""
    tol = float(tol)
    if not 0 < tol < 1:
        raise ValueError("The tolerance must be between 0 and 1.")
    return 2 + max(0, int(math.ceil(math.log(tol) / math.log(GOLDEN))) - 1)


class _Budget(Exception):
    """The run budget is spent."""


class Identification:
    """One identification: parameters, targets, an evaluator and settings.
    `run()` returns an IdentifyResult."""

    def __init__(self, params: Sequence[Param], targets: Sequence[Target],
                 evaluate: Callable[[List[List[float]]], list],
                 strategy: str = "lm", budget: int = 50,
                 tau: float = DEFAULT_TAU, n_stag: int = DEFAULT_N_STAG,
                 fd_scheme: str = "forward", fd_step: float = 0.01,
                 line_tol: float = 0.01, max_iter: int = 50,
                 lm_lambda: float = 1e-2, lm_retries: int = 4,
                 prior: Optional[Sequence[dict]] = None,
                 progress: Optional[Callable[[Iteration], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None):
        if not params:
            raise ValueError("Select at least one parameter to identify.")
        if not targets or not any(t.weight > 0 for t in targets):
            raise ValueError("Give at least one QoI with a weight > 0.")
        if strategy not in STRATEGIES:
            raise ValueError("strategy must be one of %s" % (STRATEGIES,))
        if fd_scheme not in FD_SCHEMES:
            raise ValueError("fd_scheme must be one of %s" % (FD_SCHEMES,))
        if int(budget) < 1:
            raise ValueError("The run budget must be at least 1.")
        if not 0 < fd_step < 0.5:
            raise ValueError("The finite-difference step must be between 0 "
                             "and 0.5 of the scale.")
        self.params = list(params)
        self.targets = list(targets)
        self._evaluate = evaluate
        self.strategy = strategy
        self.budget = int(budget)
        self.tau = float(tau)
        self.n_stag = int(n_stag)
        self.fd_scheme = fd_scheme
        self.fd_step = float(fd_step)
        self.line_tol = float(line_tol)
        self.max_iter = int(max_iter)
        self.lm_lambda = float(lm_lambda)
        self.lm_retries = int(lm_retries)
        self._progress = progress
        self._cancelled = cancelled or (lambda: False)
        self.lo = np.array([p.lo for p in self.params])
        self.hi = np.array([p.hi for p in self.params])
        self.scale = np.array([p.scale for p in self.params])
        self._memo: Dict[tuple, Optional[dict]] = {}
        self._prior: Dict[tuple, Optional[dict]] = {}
        for e in prior or ():
            xs = e.get("x") or {}
            try:
                key = self._key([float(xs[p.path]) for p in self.params])
            except (KeyError, TypeError, ValueError):
                continue
            self._prior[key] = e.get("q")
        self.evaluations: List[Evaluation] = []
        self.iterations: List[Iteration] = []
        self.n_launched = 0
        self._iteration = 0
        self._stag = np.zeros(len(self.params), int)
        self._frozen = np.zeros(len(self.params), bool)

    # ---------------------------------------------------------------
    # evaluations
    # ---------------------------------------------------------------
    @staticmethod
    def _key(x) -> tuple:
        return tuple(float("%.12g" % v) for v in x)

    def _eval(self, points, phase) -> list:
        """q of each point (memoised; prior evaluations are reused, the
        others are launched together). Raises _Budget when the budget would
        be exceeded before all are known."""
        points = [np.clip(np.asarray(p, float), self.lo, self.hi)
                  for p in points]
        todo, seen = [], set()
        for p in points:
            k = self._key(p)
            if k in self._memo or k in seen:
                continue
            seen.add(k)
            if k in self._prior:
                q = self._prior[k]
                self._memo[k] = q
                self._log(p, q, phase, reused=True)
            else:
                todo.append(p)
        if todo:
            if self._cancelled():
                raise _Budget("cancelled")
            if self.n_launched + len(todo) > self.budget:
                raise _Budget("budget")
            res = list(self._evaluate([list(map(float, p)) for p in todo]))
            if len(res) != len(todo):
                raise RuntimeError("The evaluator returned %d results for "
                                   "%d points." % (len(res), len(todo)))
            for p, q in zip(todo, res):
                self.n_launched += 1
                q = None if q is None else {k: float(v) for k, v in q.items()}
                self._memo[self._key(p)] = q
                self._log(p, q, phase)
        return [self._memo[self._key(p)] for p in points]

    def _log(self, x, q, phase, reused=False):
        self.evaluations.append(Evaluation(
            run=len(self.evaluations) + 1, x=[float(v) for v in x], q=q,
            J=cost(q, self.targets), phase=phase,
            iteration=self._iteration, reused=reused))

    def _J(self, x, phase) -> float:
        return cost(self._eval([x], phase)[0], self.targets)

    # ---------------------------------------------------------------
    # run
    # ---------------------------------------------------------------
    def run(self) -> IdentifyResult:
        x = np.array([p.start for p in self.params], float)
        try:
            q0 = self._eval([x], "start")[0]
        except _Budget as e:
            return self._result(FAILED, x, None,
                                "No run could be launched (%s)." % e)
        if q0 is None:
            return self._result(FAILED, x, None,
                                "The run at the start point failed.")
        self._record(x, q0, {})
        if converged(q0, self.targets):
            return self._result(CONVERGED, x, q0,
                                "The start point is within eps_q.")
        step = self._lm_iteration if self.strategy == "lm" \
            else self._fixed_point_cycle
        last_free_pass = False
        lam = [self.lm_lambda]
        status, msg = STALLED, ""
        try:
            while True:
                if self._iteration >= self.max_iter:
                    status, msg = BUDGET, ("Stopped after %d iterations."
                                           % self.max_iter)
                    break
                if self._frozen.all():
                    if last_free_pass:
                        status = STALLED
                        msg = ("Every parameter stagnates and the last pass "
                               "with all of them free did not bring the QoI "
                               "within eps_q.")
                        break
                    # one last iteration with every parameter free
                    last_free_pass = True
                    self._frozen[:] = False
                    self._stag[:] = 0
                self._iteration += 1
                x, q, ok = step(x, q0, lam, final=last_free_pass)
                q0 = q
                if converged(q, self.targets):
                    status, msg = CONVERGED, "Every QoI is within eps_q."
                    break
                if not ok:
                    status = NO_PROGRESS
                    msg = "No step lowers the cost any more."
                    break
                if last_free_pass:
                    # the final pass is one iteration only
                    self._frozen[:] = True
        except _Budget as e:
            status = BUDGET
            msg = ("Search cancelled." if str(e) == "cancelled" else
                   "The run budget (%d runs) is spent." % self.budget)
        best = self._best()
        return self._result(status, best[0], best[1], msg)

    def _best(self):
        """Point of lowest J among the evaluations (the search point on
        ties)."""
        best = None
        for e in self.evaluations:
            if e.q is not None and (best is None or e.J < best.J):
                best = e
        if best is None:
            x = np.array([p.start for p in self.params])
            return x, None
        return np.array(best.x), best.q

    def _record(self, x, q, effect):
        paths = [p.path for p in self.params]
        it = Iteration(
            iteration=self._iteration, x=[float(v) for v in x],
            J=cost(q, self.targets), max_ratio=max_ratio(q, self.targets),
            free=[p for p, f in zip(paths, self._frozen) if not f],
            frozen=[p for p, f in zip(paths, self._frozen) if f],
            runs=self.n_launched,
            effect={paths[j]: float(v) for j, v in effect.items()})
        self.iterations.append(it)
        if self._progress is not None:
            self._progress(it)

    def _result(self, status, x, q, msg) -> IdentifyResult:
        paths = [p.path for p in self.params]
        return IdentifyResult(
            status=status, x=[float(v) for v in x], q=q,
            J=cost(q, self.targets), max_ratio=max_ratio(q, self.targets),
            n_runs=self.n_launched, evaluations=list(self.evaluations),
            iterations=list(self.iterations),
            frozen=[p for p, f in zip(paths, self._frozen) if f],
            message=msg)

    # ---------------------------------------------------------------
    # stagnation
    # ---------------------------------------------------------------
    def _update_stagnation(self, idx, effect, final):
        """Count the iterations in a row whose step moved no QoI by tau
        eps_q; freeze the parameter after n_stag of them."""
        for j in idx:
            if effect[j] < self.tau:
                self._stag[j] += 1
            else:
                self._stag[j] = 0
            if not final and self._stag[j] >= self.n_stag:
                self._frozen[j] = True

    # ---------------------------------------------------------------
    # Levenberg-Marquardt
    # ---------------------------------------------------------------
    def _jacobian(self, x, q, idx):
        """dq/dp (n_q x len(idx)) by finite differences of fd_step x scale,
        turned inwards at a bound."""
        h = self.fd_step * self.scale[idx]
        pts, signs = [], []
        for n, j in enumerate(idx):
            hj = h[n]
            if self.fd_scheme == "central":
                lo_ok = x[j] - hj >= self.lo[j]
                hi_ok = x[j] + hj <= self.hi[j]
                if lo_ok and hi_ok:
                    for s in (+1, -1):
                        p = x.copy(); p[j] += s * hj; pts.append(p)
                    signs.append("c")
                    continue
            s = +1 if x[j] + hj <= self.hi[j] else -1
            p = x.copy(); p[j] += s * hj; pts.append(p)
            signs.append(s)
        res = self._eval(pts, "fd")
        qv = np.array([q[t.qid] for t in self.targets], float)
        S = np.full((len(self.targets), len(idx)), np.nan)
        i = 0
        for n, j in enumerate(idx):
            if signs[n] == "c":
                qp, qm = res[i], res[i + 1]
                i += 2
                if qp is not None and qm is not None:
                    S[:, n] = [(qp[t.qid] - qm[t.qid]) / (2 * h[n])
                               for t in self.targets]
            else:
                qs = res[i]
                i += 1
                if qs is not None:
                    S[:, n] = [(qs[t.qid] - qv[m]) / (signs[n] * h[n])
                               for m, t in enumerate(self.targets)]
        return S

    def _lm_iteration(self, x, q, lam, final=False):
        idx = [j for j in range(len(self.params)) if not self._frozen[j]]
        S = self._jacobian(x, q, idx)
        W = sum(t.weight for t in self.targets) or 1.0
        wq = np.array([math.sqrt(t.weight / W) / t.eps
                       for t in self.targets])
        # residual Jacobian in normalised variables u = x / scale
        A = S * wq[:, None] * self.scale[idx][None, :]
        A = np.where(np.isfinite(A), A, 0.0)
        r = residuals(q, self.targets)
        J0 = float(r @ r)
        g = A.T @ r
        H = A.T @ A
        D = np.diag(np.maximum(np.diag(H), 1e-12))
        for _ in range(self.lm_retries + 1):
            try:
                du = -np.linalg.solve(H + lam[0] * D, g)
            except np.linalg.LinAlgError:
                du = -np.linalg.lstsq(H + lam[0] * D, g, rcond=None)[0]
            xn = x.copy()
            xn[idx] = np.clip(x[idx] + du * self.scale[idx],
                              self.lo[idx], self.hi[idx])
            if np.allclose(xn, x, rtol=0, atol=1e-15):
                break
            qn = self._eval([xn], "step")[0]
            Jn = cost(qn, self.targets)
            if Jn < J0:
                lam[0] = max(lam[0] / 3.0, 1e-9)
                dx = xn[idx] - x[idx]
                eff = step_effect(S, dx, self.targets)
                effect = {j: float(eff[n]) for n, j in enumerate(idx)}
                self._update_stagnation(idx, np.array(
                    [effect.get(j, 0.0) for j in range(len(self.params))]),
                    final)
                self._record(xn, qn, effect)
                return xn, qn, True
            lam[0] *= 3.0
        # no step lowers J: every free parameter has stagnated this turn
        self._record(x, q, {j: 0.0 for j in idx})
        return x, q, False

    # ---------------------------------------------------------------
    # one parameter at a time + fixed point
    # ---------------------------------------------------------------
    def _line_search(self, x, j):
        """Golden-section search of J over [lo_j, hi_j], the others held.
        Returns (best x, its q, slope dq/dp_j of the last bracket)."""
        a, b = self.lo[j], self.hi[j]
        width0 = b - a

        def at(v):
            p = x.copy(); p[j] = v
            return p
        c = b - GOLDEN * (b - a)
        d = a + GOLDEN * (b - a)
        qc, qd = self._eval([at(c), at(d)], "line")
        fc, fd = cost(qc, self.targets), cost(qd, self.targets)
        while (b - a) > self.line_tol * width0:
            if fc <= fd:
                b, d, qd, fd = d, c, qc, fc
                c = b - GOLDEN * (b - a)
                qc = self._eval([at(c)], "line")[0]
                fc = cost(qc, self.targets)
            else:
                a, c, qc, fc = c, d, qd, fd
                d = a + GOLDEN * (b - a)
                qd = self._eval([at(d)], "line")[0]
                fd = cost(qd, self.targets)
        slope = np.full(len(self.targets), np.nan)
        if qc is not None and qd is not None and d != c:
            slope = np.array([(qd[t.qid] - qc[t.qid]) / (d - c)
                              for t in self.targets])
        # best of the bracket and the current point
        q_cur = self._memo.get(self._key(np.clip(x, self.lo, self.hi)))
        cands = [(cost(q_cur, self.targets), x[j], q_cur),
                 (fc, c, qc), (fd, d, qd)]
        f, v, qv = min(cands, key=lambda t: t[0])
        return at(v), qv, slope

    def _fixed_point_cycle(self, x, q, lam, final=False):
        idx = [j for j in range(len(self.params)) if not self._frozen[j]]
        effect = {}
        for j in idx:
            xn, qn, slope = self._line_search(x, j)
            dx = xn[j] - x[j]
            eff = step_effect(slope[:, None], np.array([dx]),
                              self.targets)[0]
            effect[j] = float(eff)
            x, q = xn, qn
            if converged(q, self.targets):
                break
        self._update_stagnation(list(effect), np.array(
            [effect.get(j, 0.0) for j in range(len(self.params))]), final)
        self._record(x, q, effect)
        return x, q, True
