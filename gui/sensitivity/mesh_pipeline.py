# -*- coding: utf-8 -*-
"""
Full mesh + domain convergence pipeline (Model tab).

Sequences the six steps agreed with the user, each reusing an already-tested
component, with proper "holding" of previously-identified values:

  1. identify workpiece element size   (vary elem_size, tool seed held)
  2. identify tool-tip element size     (vary tool_elem_size, elem_size = wp*)
  3. identify Eulerian domain size      (grow dims, elem_size=wp*, tool=tool*)
  4. verify wp element size             (perturb wp* by +/-1 increment)
  5. verify tool element size           (perturb tool* by +/-1 increment)
  6. verify domain size                 (perturb each dim by +/-1 element)

Steps 1-2 compare DIFFERENT meshes -> nearest-neighbour resampling on a fixed
ROI grid (step = finest wp size). Step 3 keeps the mesh fixed, so the anchored
Eulerian centroids coincide and the exact centroid comparison is used. The
verification passes reuse the same fixed configuration (wp*, tool*, domain*).

Pure module: `run_bundle(cfg) -> ResultsBundle | None` is injected, so the whole
sequence is unit-testable without Abaqus.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from gui.core.domain_sizing import DomainDims
from gui.sensitivity.domain_opt import (
    make_sample_fn, DomainOptimizer, verify_domain, OptimizeResult,
    DEFAULT_ORDER, DEFAULT_QUANTITIES)
from gui.sensitivity.mesh_opt import (
    make_mesh_sample_fn, refine_until_stable, verify_stability, finest_size,
    MeshConvResult)


@dataclass
class PipelineResult:
    wp_elem: float
    tool_elem: float
    domain: DomainDims
    wp_conv: Optional[MeshConvResult]
    tool_conv: Optional[MeshConvResult]
    domain_result: Optional[OptimizeResult]
    wp_verify: Optional[dict]
    tool_verify: Optional[dict]
    domain_verify: Optional[dict]
    n_runs: int
    grid_step: float
    ms_factor: Optional[float] = None       # identified mass-scaling factor
    ms_result: Optional[object] = None      # MassScalingResult (step 0)


def _apply(cfg, **attrs):
    c = copy.deepcopy(cfg)
    for k, v in attrs.items():
        setattr(c, k, v)
    return c


def run_mesh_domain_pipeline(
        base_cfg, run_bundle, roi, quantity_field_map, thresholds,
        wp_start: float, tool_start: float, initial_domain: DomainDims,
        factor: float = 0.5, max_steps: int = 20,
        wp_factor: float = 0.5, tool_factor: float = 0.5,
        domain_grow_factor: float = 2.0,
        wp_min: Optional[float] = None, tool_min: Optional[float] = None,
        caps: Optional[Dict[str, float]] = None, order=DEFAULT_ORDER,
        include_tool: bool = True, do_verify: bool = True,
        include_wp: bool = True, include_domain: bool = True,
        max_doublings: int = 20, force_channels: Optional[Dict[str, str]] = None,
        grid_step: Optional[float] = None,
        identify_ms: bool = False, ms_start: float = 1.0,
        ms_factor_growth: float = 2.0, ms_max_factor: Optional[float] = None,
        ms_guard_threshold: Optional[float] = None,
        ms_bisection_resolution: float = 0.0,
        wp_bisection_resolution: float = 0.0,
        tool_bisection_resolution: float = 0.0,
        domain_bisection_resolution: float = 0.0,
        progress_cb: Optional[Callable[[dict], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None) -> PipelineResult:
    """Run the pipeline. `thresholds`/`quantity_field_map` are shared across all
    steps. `force_channels` adds normalized reaction forces as compared
    quantities. `grid_step` is the fixed ROI-grid spacing for every comparison
    (defaults to the finest wp size). When `identify_ms` is True, an extra
    STEP 0 identifies the mass-scaling factor first (criterion on Vx/Vy, energy
    guard `ms_guard_threshold`), and holds it for every later step. Returns a
    PipelineResult; raises whatever `run_bundle` raises."""
    from gui.sensitivity.mass_scaling import (identify_mass_scaling,
                                              make_mass_scaling_sample_fn)
    force_channels = dict(force_channels or {})
    quantities = (tuple(quantity_field_map.keys())
                  + tuple(force_channels.keys()))

    # Wrap run_bundle to count total Abaqus calls and to allow cancellation.
    counter = {"n": 0}

    def counted_run_bundle(cfg):
        counter["n"] += 1
        return run_bundle(cfg)

    def emit(phase, **extra):
        if progress_cb is not None:
            d = {"phase": phase, "n_runs": counter["n"]}
            d.update(extra)
            progress_cb(d)

    def cancelled():
        return should_cancel is not None and should_cancel()

    # Fixed ROI grid step for all comparisons: user-set spacing, or the finest
    # workpiece size (so the grid resolves the finest Eulerian mesh).
    if grid_step is None:
        grid_step = finest_size(wp_start, factor, max_steps, wp_min)

    # -- Step 0: mass-scaling factor (optional) ------------------------------
    ms_factor = None
    ms_result = None
    if identify_ms and not cancelled():
        vel_map = {q: quantity_field_map[q] for q in ("Vx", "Vy")
                   if q in quantity_field_map}
        vel_thr = {q: thresholds[q] for q in ("Vx", "Vy") if q in thresholds}
        if not vel_map or not vel_thr:
            raise ValueError("mass-scaling identification needs Vx/Vy in the "
                             "quantities and their thresholds.")
        if ms_guard_threshold is None:
            raise ValueError("mass-scaling identification needs an energy "
                             "guard threshold (mean ALLKE/ALLIE).")
        emit("ms_start")
        sf_ms = make_mass_scaling_sample_fn(
            base_cfg, counted_run_bundle, roi, grid_step, vel_map)
        ms_result = identify_mass_scaling(
            sf_ms, vel_thr, ms_guard_threshold, ms_start,
            factor=ms_factor_growth, max_factor=ms_max_factor,
            bisection_resolution=ms_bisection_resolution,
            progress_cb=lambda e: emit("ms", sub=e),
            should_cancel=should_cancel)
        ms_factor = ms_result.identified
        emit("ms_done", identified=ms_factor,
             guard=ms_result.guard_at_identified,
             converged=ms_result.velocity_converged,
             limited_by=ms_result.limited_by)
        # hold the identified factor for every later step (workpiece AND tool)
        base_cfg = _apply(base_cfg)
        base_cfg.step.mass_scaling_enabled = True
        base_cfg.step.mass_scaling_factor = float(ms_factor)

    # -- Step 1: workpiece element size (optional) ---------------------------
    wp = float(base_cfg.elem_size)
    wp_conv = None
    if include_wp and not cancelled():
        emit("wp_start")
        sf_wp = make_mesh_sample_fn(
            base_cfg, counted_run_bundle, roi, grid_step, quantity_field_map,
            size_attr="elem_size",
            held_sizes={"tool_elem_size": float(base_cfg.tool_elem_size)},
            force_channels=force_channels)
        wp_conv = refine_until_stable(
            sf_wp, thresholds, wp_start, quantities, wp_factor, wp_min,
            max_steps, bisection_resolution=wp_bisection_resolution,
            progress_cb=lambda e: emit("wp", sub=e),
            should_cancel=should_cancel)
        wp = wp_conv.identified
        emit("wp_done", identified=wp)

    # -- Step 2: tool-tip element size ---------------------------------------
    tool = float(base_cfg.tool_elem_size)
    tool_conv = None
    if include_tool and not cancelled():
        emit("tool_start")
        sf_tool = make_mesh_sample_fn(
            base_cfg, counted_run_bundle, roi, grid_step, quantity_field_map,
            size_attr="tool_elem_size", held_sizes={"elem_size": wp},
            force_channels=force_channels)
        tool_conv = refine_until_stable(
            sf_tool, thresholds, tool_start, quantities, tool_factor, tool_min,
            max_steps, bisection_resolution=tool_bisection_resolution,
            progress_cb=lambda e: emit("tool", sub=e),
            should_cancel=should_cancel)
        tool = tool_conv.identified
        emit("tool_done", identified=tool)

    # -- Step 3: Eulerian domain size (optional) -----------------------------
    domain_result = None
    domain = initial_domain
    if include_domain and not cancelled():
        emit("domain_start")
        cfg_mesh_fixed = _apply(base_cfg, elem_size=wp, tool_elem_size=tool)
        sf_dom = make_sample_fn(cfg_mesh_fixed, counted_run_bundle, roi=roi,
                                elem_size=wp,
                                quantity_field_map=quantity_field_map,
                                force_channels=force_channels,
                                grid_step=grid_step)
        optimizer = DomainOptimizer(sf_dom, thresholds=thresholds,
                                    elem_size=wp, caps=caps, order=order,
                                    quantities=quantities,
                                    max_doublings=max_doublings,
                                    bisect_resolution=domain_bisection_resolution,
                                    grow_factor=domain_grow_factor)
        domain_result = optimizer.optimize(
            initial_domain, progress_cb=lambda e: emit("domain", sub=e),
            should_cancel=should_cancel)
        domain = domain_result.dims
        emit("domain_done", dims=domain)

    # -- Steps 4-6: verification ---------------------------------------------
    wp_verify = tool_verify = domain_verify = None
    if do_verify and not cancelled():
        cfg_final = _apply(base_cfg, elem_size=wp, tool_elem_size=tool)
        cfg_final.euler_geometry.h_wp = float(domain.h_wp)
        cfg_final.euler_geometry.h_void = float(domain.h_void)
        cfg_final.euler_geometry.l_wp = float(domain.l_wp)
        cfg_final.euler_geometry.l_void = float(domain.l_void)

        if include_wp and not cancelled():
            emit("verify_wp_start")
            sf_vwp = make_mesh_sample_fn(
                cfg_final, counted_run_bundle, roi, grid_step,
                quantity_field_map, size_attr="elem_size",
                held_sizes={"tool_elem_size": tool},
                force_channels=force_channels)
            wp_verify = verify_stability(sf_vwp, wp, thresholds, quantities,
                                         wp_factor,
                                         step=wp_bisection_resolution)
            emit("verify_wp_done", stable=wp_verify["stable"])

        if include_tool and not cancelled():
            emit("verify_tool_start")
            sf_vtool = make_mesh_sample_fn(
                cfg_final, counted_run_bundle, roi, grid_step,
                quantity_field_map, size_attr="tool_elem_size",
                held_sizes={"elem_size": wp}, force_channels=force_channels)
            tool_verify = verify_stability(sf_vtool, tool, thresholds,
                                           quantities, tool_factor,
                                           step=tool_bisection_resolution)
            emit("verify_tool_done", stable=tool_verify["stable"])

        if include_domain and not cancelled():
            emit("verify_domain_start")
            sf_vdom = make_sample_fn(cfg_final, counted_run_bundle, roi=roi,
                                     elem_size=wp,
                                     quantity_field_map=quantity_field_map,
                                     force_channels=force_channels,
                                     grid_step=grid_step)
            domain_verify = verify_domain(sf_vdom, domain, thresholds,
                                          elem_size=wp, quantities=quantities,
                                          order=order,
                                          step=domain_bisection_resolution)
            emit("verify_domain_done", stable=domain_verify["stable"])

    emit("done")
    return PipelineResult(
        wp_elem=wp, tool_elem=tool, domain=domain, wp_conv=wp_conv,
        tool_conv=tool_conv, domain_result=domain_result, wp_verify=wp_verify,
        tool_verify=tool_verify, domain_verify=domain_verify,
        n_runs=counter["n"], grid_step=grid_step,
        ms_factor=ms_factor, ms_result=ms_result)
