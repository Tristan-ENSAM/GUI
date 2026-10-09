# -*- coding: utf-8 -*-
"""Study folders <-> study settings ("specs"), for resuming and loading.

A spec is the dict of inputs one sizing study was launched with. It is
written as the ``parameters`` of the study folder's ``config.json`` (same
keys as before this module existed, plus ``base_ms`` = [enabled, factor] of
the mass scaling the runs used, ``filter_verify`` and ``cache_folders``, the
other study folders whose runs it may reuse), so an older folder reads back
too: what it lacks is filled from its runs (``base_ms`` and
``filter_verify`` from the first finished run) or from the current tab
(``rk_max``/``rhg_max`` for an older GCI folder). The final checks keep
theirs in ``checks_config.json`` inside the domain study folder.

Pure module (no Qt).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

from gui.sensitivity.run_cache import run_completed

# config.json "study" prefix -> step name; step -> folder prefix, job prefix
STEP_OF_PREFIX = {"massscaling": "ms", "GCI": "mesh", "domainsizing": "domain"}
PREFIX_OF_STEP = {v: k for k, v in STEP_OF_PREFIX.items()}
JOB_PREFIX = {"ms": "ms", "mesh": "GCI", "domain": "domainsizing",
              "checks": "checks"}
CHECKS_CONFIG = "checks_config.json"
ZOI_KEYS = ("xmin", "xmax", "ymin", "ymax")
DIM_KEYS = ("h_wp", "h_void", "l_wp", "l_void")
# GCI quantity name <-> comparison-settings name
GCI_TO_EPS = {"EVF": "EVF", "TEMP": "T", "V1": "Vx", "V2": "Vy",
              "Fc": "Fc", "Ff": "Ff"}


def _r(x: float) -> float:
    return float("%.9g" % float(x))


def zoi_tuple(spec: dict) -> Tuple[float, float, float, float]:
    z = spec["zoi"]
    if isinstance(z, dict):
        return tuple(float(z[k]) for k in ZOI_KEYS)
    return tuple(float(v) for v in z)


def comparison_settings(zoi, eps: Dict[str, float], window, rk_max: float,
                        rhg_max: float, grid_step=None) -> dict:
    """The shared comparison settings a step result depends on (the key of
    study_state.settings_key). An ε_q <= 0 is left out, as the studies
    ignore it. `grid_step` is the ZOI sampling step as SET in the tab (None
    when blank: it then follows the element size of each study)."""
    out_eps = {}
    for k, v in sorted((eps or {}).items()):
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            out_eps[str(k)] = _r(v)
    return {"zoi": [_r(v) for v in zoi],
            "eps": out_eps,
            "window": [_r(v) for v in window],
            "rk_max": _r(rk_max), "rhg_max": _r(rhg_max),
            "grid": None if grid_step is None else _r(grid_step)}


def eps_of_spec(spec: dict) -> Dict[str, float]:
    if "thresholds_abs" in spec:
        return {str(k): float(v) for k, v in spec["thresholds_abs"].items()}
    return {GCI_TO_EPS.get(k, k): float(v)
            for k, v in (spec.get("tolerances") or {}).items()}


def grid_set_of_spec(spec: dict):
    """The sampling step as set in the tab when the study started (None:
    blank). An older spec has only the step used; it is taken as blank
    when it equals the element size a blank field gave that study (the
    element size of the study, the finest mesh of a GCI plan, h* of the
    final checks), as set otherwise. That is an assumption: an older
    folder does not say whether the field was blank."""
    if "grid_step_set" in spec:
        return spec["grid_step_set"]
    plan = spec.get("gci_plan") if isinstance(spec.get("gci_plan"),
                                               dict) else {}
    used = spec.get("grid_step", plan.get("grid_step"))
    if used is None:
        return None
    blank = (spec.get("elem_size") or spec.get("finest_elem_size")
             or spec.get("h_star"))
    try:
        if blank is not None and math.isclose(float(used), float(blank),
                                              rel_tol=1e-9, abs_tol=1e-15):
            return None
        return float(used)
    except (TypeError, ValueError):
        return None


def settings_of_spec(spec: dict) -> dict:
    return comparison_settings(zoi_tuple(spec), eps_of_spec(spec),
                               spec["window"], spec["rk_max"],
                               spec["rhg_max"], grid_set_of_spec(spec))


def first_finished_params(folder, job_prefix: str) -> Optional[dict]:
    """model_config of the first finished `<job_prefix>_runNNN` of
    `folder`, or None."""
    folder = Path(folder)
    metas = sorted(folder.glob("%s_run*.meta.json" % job_prefix))
    for meta in metas:
        job = meta.name[:-len(".meta.json")]
        if not run_completed(folder, job):
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        mc = data.get("model_config") if isinstance(data, dict) else None
        if isinstance(mc, dict):
            return mc
    return None


def base_ms_of_params(params: dict) -> Optional[list]:
    step = (params or {}).get("step") or {}
    if "mass_scaling_enabled" not in step:
        return None
    return [bool(step.get("mass_scaling_enabled")),
            float(step.get("mass_scaling_factor_eulerian", 1.0))]


def filter_verify_of_params(params: dict) -> Optional[bool]:
    step = (params or {}).get("step") or {}
    if "output_filter_verify" not in step:
        return None
    return bool(step.get("output_filter_verify"))


def ms_of(base_ms) -> float:
    """Mass-scaling factor in effect for [enabled, factor] (1 when off)."""
    try:
        return float(base_ms[1]) if base_ms[0] else 1.0
    except (TypeError, IndexError, ValueError):
        return 1.0


def rewrite_study_config(folder, spec: dict) -> bool:
    """Replace the ``parameters`` of the folder's config.json (e.g. after
    the ms values were extended); best effort."""
    path = Path(folder) / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        data["parameters"] = spec
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        return True
    except (OSError, ValueError):
        return False


def write_checks_config(folder, spec: dict, when=None) -> bool:
    """Settings of the final checks, next to the domain study they check
    (the checks runs live in that folder); best effort."""
    from datetime import datetime
    payload = {"study": "checks",
               "created_at": (when or datetime.now()).isoformat(
                   timespec="seconds"),
               "parameters": spec}
    try:
        (Path(folder) / CHECKS_CONFIG).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        return True
    except OSError:
        return False


def read_checks_config(folder) -> Optional[dict]:
    """The checks settings saved in a domain study folder, or None."""
    try:
        data = json.loads((Path(folder) / CHECKS_CONFIG).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return None
    spec = data.get("parameters") if isinstance(data, dict) else None
    need = ("h_star", "gci_plan", "gci_tolerances", "zoi", "thresholds_abs",
            "window", "rk_max", "rhg_max", "base_ms")
    if not isinstance(spec, dict) or any(spec.get(k) is None for k in need):
        return None
    return spec


def read_study_folder(folder, defaults: Optional[dict] = None
                      ) -> Tuple[str, dict]:
    """(step, spec) of a study folder; raises ValueError with a message
    for the user. `defaults` fills what an older config.json lacks
    (rk_max, rhg_max, base_ms)."""
    folder = Path(folder)
    path = folder / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise ValueError("%s has no config.json: it is not a study folder "
                         "of the Model tab." % folder.name)
    except ValueError:
        raise ValueError("%s: config.json is not valid JSON." % folder.name)
    prefix = data.get("study") if isinstance(data, dict) else None
    step = STEP_OF_PREFIX.get(prefix)
    if step is None:
        raise ValueError("%s is a '%s' study, not a step of the Model tab "
                         "(mass scaling, mesh convergence or domain)."
                         % (folder.name, prefix))
    spec = dict(data.get("parameters") or {})
    defaults = dict(defaults or {})
    for k in ("rk_max", "rhg_max"):
        if spec.get(k) is None and k in defaults:
            spec[k] = defaults[k]
    if step in ("mesh", "domain") and (spec.get("base_ms") is None
                                       or spec.get("filter_verify") is None):
        # An older folder: what the runs were made with is in their meta.
        params = first_finished_params(folder, JOB_PREFIX[step])
        if spec.get("base_ms") is None:
            bm = base_ms_of_params(params) if params else None
            spec["base_ms"] = bm if bm is not None else defaults.get("base_ms")
        if spec.get("filter_verify") is None:
            fv = filter_verify_of_params(params) if params else None
            spec["filter_verify"] = (fv if fv is not None
                                     else defaults.get("filter_verify"))
    required = {"ms": ("zoi", "elem_size", "ms_values", "grid_step",
                       "thresholds_abs", "window", "domain_dims"),
                "mesh": ("zoi", "window", "finest_elem_size", "ratio",
                         "n_meshes", "grid_step", "tolerances",
                         "domain_dims"),
                "domain": ("zoi", "elem_size", "margin_elems",
                           "euler_offset", "grid_step", "thresholds_abs",
                           "window", "step_elems", "n_max", "n_hold",
                           "m_ratios", "initial_dims")}[step]
    missing = [k for k in required + ("rk_max", "rhg_max")
               if spec.get(k) is None]
    if missing:
        raise ValueError("%s: config.json lacks %s." % (
            folder.name, ", ".join(missing)))
    spec.setdefault("evf_threshold", 0.5)
    return step, spec
