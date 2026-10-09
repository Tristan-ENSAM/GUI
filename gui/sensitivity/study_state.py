# -*- coding: utf-8 -*-
"""Which sizing steps are done for the current model, and in what order.

The Model tab sizes three numerical choices in a fixed order:

    step 0  mass scaling factor ms*      (needs nothing)
    step 1  element size h*              (runs at ms*)
    step 2  Eulerian domain D*           (runs at ms* and h*)
    step 3  final checks at (ms*, h*, D*)

Each finished step leaves a record in ``cfg.optimization.steps`` (saved in
the .acpf profile)::

    {"status": "done" | "failed" | "interrupted",
     "value": ms* | h* | [h_wp, h_void, l_wp, l_void] | "accepted",
     "folder": study folder, "finished_at": ISO time,
     "model_key": ..., "settings_key": ...,
     "inputs": {"ms": ..., "h": ..., "dims": [...]},   # what the step used
     "message": short text, "values": [...]}            # optional extras

A record is valid for the CURRENT model when

* its ``model_key`` matches: the model parameters EXCEPT the three sized
  quantities (and the outputs the studies force on) are unchanged, so
  writing ms*, h* or D* into the model never invalidates a step;
* its ``settings_key`` matches: same comparison settings (ZOI, eps_q,
  window T, sampling step as set in the tab, safeguards);
* the steps before it are valid and it ran with their results (step 1 at
  ms*, step 2 at ms* and h*, step 3 at ms*, h* and D*).

Pure module (no Qt).
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from gui.sensitivity.run_cache import diff_params, params_key

STEPS = ("ms", "mesh", "domain", "checks")
STEP_TITLES = {"ms": "Step 0 (mass scaling)", "mesh": "Step 1 (element size)",
               "domain": "Step 2 (Eulerian domain)",
               "checks": "Step 3 (final checks)"}
DIM_NAMES = ("h_wp", "h_void", "l_wp", "l_void")

# Parameters set by the sizing itself (or forced on by every study): they
# are left out of the model key, so applying a result keeps the steps valid.
AXIS_KEYS = (
    "geometry.euler.geometry.h_wp", "geometry.euler.geometry.h_void",
    "geometry.euler.geometry.l_wp", "geometry.euler.geometry.l_void",
    "mesh.elem_size",
    "step.mass_scaling_enabled", "step.mass_scaling_factor_eulerian",
    "step.mass_scaling_factor_tool",
    "step.output_filter_verify", "step.output.ho_preselect",
    "step.output.ho_rf_on_rp",
)


def _drop(params: dict, dotted: Sequence[str]) -> dict:
    import copy
    out = copy.deepcopy(params)
    for key in dotted:
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.get(p) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return out


def model_key(params: dict) -> str:
    """Key of the model, the sized quantities left out."""
    return params_key(_drop(params, AXIS_KEYS))


def model_differences(old_params: dict, new_params: dict
                      ) -> List[Tuple[str, object, object]]:
    """What changed in the model since `old_params` (sized quantities
    ignored), as (dotted key, old, new)."""
    return diff_params(old_params, new_params, ignore=AXIS_KEYS)


def settings_key(settings: dict) -> str:
    """Key of the shared comparison settings."""
    return params_key(settings)


def make_record(status: str, value=None, folder=None, model_key_: str = "",
                settings_key_: str = "", inputs: Optional[dict] = None,
                message: str = "", when: Optional[datetime] = None,
                **extra) -> dict:
    rec = {"status": status, "value": value,
           "folder": str(folder) if folder else "",
           "finished_at": (when or datetime.now()).isoformat(
               timespec="seconds"),
           "model_key": model_key_, "settings_key": settings_key_,
           "inputs": dict(inputs or {}), "message": message}
    rec.update(extra)
    return rec


def _close(a, b, rel=1e-5) -> bool:
    """Equal up to rel: the tabs show (and write back) values with 6
    significant digits, so a result written into the model may come back
    rounded there."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-15)


def dims_close(a, b) -> bool:
    try:
        return len(a) == len(b) == 4 and all(_close(x, y) for x, y in zip(a, b))
    except TypeError:
        return False


def values_match(step: str, a, b) -> bool:
    """Equality of two step results (dims compared component-wise)."""
    if step == "domain":
        return dims_close(a, b)
    if step == "checks":
        return a == b
    return _close(a, b)


# State of a step for the current model:
#   "none"         never run
#   "done"         valid result for this model
#   "failed"       last run failed (for this model)
#   "interrupted"  last run stopped before the end (resumable)
#   "stale"        made for another model or other settings
#   "upstream"     made with a result of an earlier step that changed since
#   "prereq"       done, but an earlier step is not done for this model
#                  (the step was run out of order)
STATE_LABELS = {"none": "not done yet", "done": "done",
                "failed": "failed", "interrupted": "interrupted",
                "stale": "done for another model or other settings",
                "upstream": "made with an earlier result of a previous step",
                "prereq": "done, but an earlier step is not done for this "
                          "model"}


def state_label(state: str, rec: Optional[dict] = None) -> str:
    """STATE_LABELS, with a stale record named by what it was (a failed or
    interrupted study is not 'done for another model')."""
    status = (rec or {}).get("status")
    if state == "stale" and status in ("failed", "interrupted"):
        return "%s for another model or other settings" % status
    return STATE_LABELS.get(state, state)


def step_state(steps: Dict[str, dict], step: str, mkey: str, skey: str
               ) -> Tuple[str, Optional[dict]]:
    """(state, record) of `step` for the model/settings keys given."""
    rec = (steps or {}).get(step)
    if not rec:
        return "none", None
    if rec.get("model_key") != mkey or rec.get("settings_key") != skey:
        return "stale", rec
    status = rec.get("status")
    if status not in ("done", "failed", "interrupted"):
        status = "failed"
    # A done step needs every earlier step done ("prereq" otherwise) and
    # must have run with their results; a failed or interrupted one is
    # "upstream" only when an earlier step is done with a result it did not
    # use (resuming it would finish a study made on an old result).
    need = {"ms": "ms", "mesh": "h", "domain": "dims"}
    for up in STEPS[:STEPS.index(step)]:
        st, up_rec = step_state(steps, up, mkey, skey)
        if st != "done":
            if status == "done":
                return "prereq", rec
            continue
        used = (rec.get("inputs") or {}).get(need[up])
        if used is None or not values_match(up, used, up_rec.get("value")):
            return "upstream", rec
    return status, rec


def first_step_not_done(steps: Dict[str, dict], step: str, mkey: str,
                        skey: str) -> Optional[Tuple[str, str, Optional[dict]]]:
    """(earlier step, its state, its record) of the first step before
    `step` that is not done for this model, or None."""
    for up in STEPS[:STEPS.index(step)]:
        st, rec = step_state(steps, up, mkey, skey)
        if st != "done":
            return up, st, rec
    return None


def done_value(steps: Dict[str, dict], step: str, mkey: str, skey: str):
    """The valid result of `step`, or None."""
    st, rec = step_state(steps, step, mkey, skey)
    return rec.get("value") if st == "done" else None


def missing_prerequisites(steps: Dict[str, dict], step: str, mkey: str,
                          skey: str, model_values: dict) -> List[str]:
    """Why `step` should not run yet, as user-facing sentences (empty when
    every earlier step is done for this model and its result is in the
    model). `model_values` = {"ms": ..., "h": ..., "dims": [...]} as the
    model holds them now."""
    out: List[str] = []
    i = STEPS.index(step)
    holds = {"ms": ("ms", "the mass scaling factor (Step tab)"),
             "mesh": ("h", "the element size (Mesh tab)"),
             "domain": ("dims", "the Eulerian domain (Geometry tab)")}
    for up in STEPS[:i]:
        st, rec = step_state(steps, up, mkey, skey)
        if st != "done":
            out.append("%s is %s." % (STEP_TITLES[up], state_label(st, rec)))
            continue
        name, where = holds[up]
        if not values_match(up, model_values.get(name), rec.get("value")):
            out.append("The model does not use the result of %s: %s is %s, "
                       "the step found %s." % (
                           STEP_TITLES[up], where,
                           format_value(up, model_values.get(name)),
                           format_value(up, rec.get("value"))))
    return out


def format_value(step: str, value) -> str:
    if value is None:
        return "n/a"
    if step == "ms":
        return "ms = %g" % float(value)
    if step == "mesh":
        return "h = %g mm" % float(value)
    if step == "domain":
        try:
            return ("l_wp %g, h_wp %g, h_void %g, l_void %g mm"
                    % (value[2], value[0], value[1], value[3]))
        except (TypeError, IndexError):
            return str(value)
    return str(value)
