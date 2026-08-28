# -*- coding: utf-8 -*-
"""Abaqus CEL model construction and job execution.

This module is Abaqus-specific (Python 2.7). Pure-Python helpers shared with
the Python 3 GUI stay in :mod:`cel_common`.

The public workflow is intentionally small:
    prepare_parameters -> build_model -> create_job -> run_job

``build_model`` delegates each Abaqus concern to a named function so model
construction can be navigated from the editor outline.
"""

import os
import sys

from abaqus import *
from abaqusConstants import *
from step import *
from sketch import *
from load import *
from part import *
from mesh import *
from interaction import *
from regionToolset import *

from cel_common import (cfg_get, discretize, resolve_mapping,
                        resolve_tool_translation)

_resolve_tool_translation = resolve_tool_translation
_resolve_mapping = resolve_mapping

# ---------------------------------------------------------------------------
# Abaqus symbolic-constant mappings
# ---------------------------------------------------------------------------
# Tangential contact formulation
_TANGENTIAL_MAP = {
    "penalty":      PENALTY,
    "rough":        ROUGH,
    "frictionless": FRICTIONLESS,
}
# Normal contact pressure-overclosure law
_PRESSURE_OVERCLOSURE_MAP = {
    "hard":        HARD,
    "exponential": EXPONENTIAL,
    "linear":      LINEAR,
    "tabular":     TABULAR,
}
# Eulerian inflow BC types
_INFLOW_MAP = {
    "FREE": FREE,
    "NONE": NONE,
    "VOID": VOID,
}
# Eulerian outflow BC types
_OUTFLOW_MAP = {
    "FREE":          FREE,
    "NONREFLECTING": NON_REFLECTING,
    "EQUILIBRIUM":   EQUILIBRIUM,
    "ZERO_PRESSURE": ZERO_PRESSURE,
    "NONE":          NONE,
}
# EulerianBC `definition` argument (which fluxes are specified)
_BC_DEFINITION_MAP = {
    "inflow":  INFLOW,
    "outflow": OUTFLOW,
    "both":    BOTH,
}


def prepare_parameters(model_cfg, run_cfg):
    """Read/validate GUI dictionaries and return normalized parameters."""
    elem_size = float(cfg_get(model_cfg, "mesh.elem_size", 0.01))

    cutting_speed = float(cfg_get(model_cfg, "bcs.cutting_speed", 1000.0))
    # `sim_time` and `n_frames` are owned by the Step tab.
    sim_time      = float(cfg_get(model_cfg, "step.sim_time", 0.0001))
    n_frames      = int(  cfg_get(model_cfg, "step.n_frames", 1))

    # -----------------------------------------------------------------------------
    # Output (FROZEN -- no longer selectable from the GUI)
    # -----------------------------------------------------------------------------
    # The extraction pipeline only reads EVF, TEMP and V. The rest is kept so
    # the ODB can be opened by hand in Abaqus/CAE to diagnose "what went wrong"
    # (damage, contact, element deletion, volumetric waves...).
    #
    # Deliberately NOT requested:
    #   U  -- nodal displacements are meaningless on an Eulerian instance: the
    #         mesh is fixed and the material flows through it.
    #   A, RF, NT, VP, P, HFL, HP, MFL, SDV -- unused, and every extra variable
    #         inflates the ODB and the write time.
    fo_variables = (
        'EVF',      # Eulerian volume fraction -- material tracking (extracted)
        'TEMP',     # temperature (extracted)
        'V',        # velocity (extracted; the DIC comparison variable)
        'S',        # stress tensor
        'PEEQ',     # equivalent plastic strain
        'ERV',      # VOLUMETRIC STRAIN RATE -- shows the dilatational waves
        'SDEG',     # stiffness degradation (damage evolution)
        'DMICRT',   # damage initiation criteria
        'STATUS',   # element status (1 active / 0 deleted)
        'CSTRESS',  # contact stresses
    )

    # History output is likewise frozen: reaction forces at the tool RP
    # (cutting forces), PRESELECT (carries ALLKE/ALLIE for the energy guard),
    # and the Eulerian mass/volume per material instance.
    ho_n_intervals = n_frames

    # -----------------------------------------------------------------------------
    # Output filters (Step tab). TWO cutoffs, because the two observables have
    # DIFFERENT measurement bandwidths:
    #   * FIELD  -> the DIC chain. Correlating two images separated by dt and
    #     dividing by dt is already a moving average over that interval, which
    #     dominates the exposure blur. Cascade of both -> ~25.6 kHz at 60 kfps
    #     with a 5 us exposure.
    #   * HISTORY -> forces are sampled far faster (500 kHz), so their runtime
    #     filter only needs to prevent ALIASING at the output rate; the real
    #     sensor bandwidth is applied afterwards in post-processing. Keeping
    #     this cutoff HIGH matters: an IIR filter needs cutoff/(1/dt) > 1e-3,
    #     so a low cutoff would force an unreachable mass-scaling factor.
    # -----------------------------------------------------------------------------
    filter_enabled = bool(cfg_get(model_cfg, "step.output_filter_enabled", False))
    filter_cutoff = float(cfg_get(model_cfg, "step.output_filter_cutoff_hz", 0.0))
    filter_cutoff_history = float(
        cfg_get(model_cfg, "step.output_filter_cutoff_history_hz", 0.0))

    # -----------------------------------------------------------------------------
    # Mass scaling (Step tab > Mass scaling)
    # -----------------------------------------------------------------------------
    # Mass scaling for CEL is done by scaling the Eulerian material density
    # (Abaqus' native mass scaling does not apply to Eulerian EC3D8R elements).
    # rho_eff = factor * rho ; Cp_eff = Cp / factor, so rho*Cp (and thus the
    # temperature) is preserved. Matches the manual reference workflow where
    # both rho and Cp are scaled by hand. Pass PHYSICAL rho/Cp here so the
    # factor scales them once (do not pre-scale by hand as well).
    # Both factors default to 1.0 (no scaling).
    ms_enabled = bool(cfg_get(model_cfg, "step.mass_scaling_enabled", False))
    ms_eul     = float(cfg_get(model_cfg, "step.mass_scaling_factor_eulerian", 1.0))
    ms_tool    = float(cfg_get(model_cfg, "step.mass_scaling_factor_tool",     1.0))
    if not ms_enabled:
        ms_eul  = 1.0
        ms_tool = 1.0

    h_tool      = float(cfg_get(model_cfg, "geometry.tool.geometry.h_tool",      0.3))
    l_tool      = float(cfg_get(model_cfg, "geometry.tool.geometry.l_tool",      0.5))
    r_tool      = float(cfg_get(model_cfg, "geometry.tool.geometry.r_tool",      0.01))
    rake_angle  = float(cfg_get(model_cfg, "geometry.tool.geometry.rake_angle",  40.0))
    clear_angle = float(cfg_get(model_cfg, "geometry.tool.geometry.clear_angle", 10.0))
    tool_x0     = float(cfg_get(model_cfg, "geometry.tool.position.x0", 0.0))
    tool_y0     = float(cfg_get(model_cfg, "geometry.tool.position.y0", -0.05))

    egeom  = cfg_get(model_cfg, "geometry.euler.geometry", {})
    h_wp   = float(egeom.get("h_wp",   0.3))
    h_void = float(egeom.get("h_void", 0.2))
    l_wp   = float(egeom.get("l_wp",   0.5))
    l_void = float(egeom.get("l_void", 0.2))
    if bool(egeom.get("discretize", True)):
        h_wp   = discretize(h_wp,   elem_size)
        h_void = discretize(h_void, elem_size)
        l_wp   = discretize(l_wp,   elem_size)
        l_void = discretize(l_void, elem_size)

    mesh_tx = float(cfg_get(model_cfg, "geometry.euler.position.x0",            0.0))
    mesh_ty = float(cfg_get(model_cfg, "geometry.euler.position.y0",            0.0))
    wp_tx   = float(cfg_get(model_cfg, "geometry.euler.workpiece_position.x0",  0.0))
    wp_ty   = float(cfg_get(model_cfg, "geometry.euler.workpiece_position.y0",  0.0))

    tool_tx, tool_ty, _tool_engages, _tool_reason = _resolve_tool_translation(
        h_tool, l_tool, r_tool, rake_angle, clear_angle, tool_x0, tool_y0, wp_ty)
    if not _tool_engages:
        raise ValueError(
            "Invalid tool/workpiece configuration: {} (tool_x0={}, tool_y0={}, "
            "workpiece y0={}). The tool does not cut — check the tool position, "
            "depth, and angles before running.".format(
                _tool_reason, tool_x0, tool_y0, wp_ty))


    margin = elem_size / 2
    xmin = float(cfg_get(model_cfg, "geometry.bbox.xmin", -0.5))
    xmax = float(cfg_get(model_cfg, "geometry.bbox.xmax",  0.5))
    ymin = float(cfg_get(model_cfg, "geometry.bbox.ymin", -0.5))
    ymax = float(cfg_get(model_cfg, "geometry.bbox.ymax",  0.5))
    zmin = float(cfg_get(model_cfg, "geometry.bbox.zmin",  0.0))
    zmax = float(cfg_get(model_cfg, "geometry.bbox.zmax",  margin))

    emat = cfg_get(model_cfg, "materials.euler", {})
    tmat = cfg_get(model_cfg, "materials.tool",  {})

    # Eulerian material default values
    emat.setdefault("rho",      4.44e-09);  emat.setdefault("E",        109000.0)
    emat.setdefault("nu",       0.34);      emat.setdefault("k",        10.0)
    emat.setdefault("Cp",       674000000.0); emat.setdefault("alpha",  1e-05)
    emat.setdefault("beta",     0.9)
    emat.setdefault("A",        812.0);     emat.setdefault("B",        844.0)
    emat.setdefault("n",        0.261);     emat.setdefault("C",        0.015)
    emat.setdefault("m",        1.0);       emat.setdefault("Tm",       1620.0)
    emat.setdefault("Tr",       20.0);      emat.setdefault("eps_dot0", 0.05)
    emat.setdefault("D1",       0.245);     emat.setdefault("D2",       0.081)
    emat.setdefault("D3",       1.276);     emat.setdefault("D4",      -0.028)
    emat.setdefault("D5",       3.87);      emat.setdefault("eps0",     0.05)
    emat.setdefault("Gf",       18.5)

    # Tool material default values
    tmat.setdefault("rho",   1.19e-08);  tmat.setdefault("E",     534000.0)
    tmat.setdefault("nu",    0.22);      tmat.setdefault("k",     50.0)
    tmat.setdefault("Cp",    400000000.0); tmat.setdefault("alpha", 1e-05)

    # -----------------------------------------------------------------------------
    # Interaction parameters (formerly hardcoded as PENALTY mu=0.3, HARD, no heat)
    # -----------------------------------------------------------------------------
    inter = cfg_get(model_cfg, "interaction", {}) or {}
    inter_tangential       = str(  inter.get("tangential_formulation", "penalty"))   # "penalty" | "rough" | "frictionless"
    inter_friction         = float(inter.get("friction_coeff",         0.3))
    inter_slip_tol         = float(inter.get("slip_tolerance",         0.005))
    inter_pressure         = str(  inter.get("pressure_overclosure",   "hard"))      # "hard" | "exponential" | "linear" | "tabular"
    inter_heat_gen         = bool( inter.get("heat_generation",        False))
    inter_heat_to_slave    = float(inter.get("heat_fraction_to_slave",  0.5))
    inter_heat_to_master   = float(inter.get("heat_fraction_to_master", 0.5))

    # -----------------------------------------------------------------------------
    # Boundary / initial conditions parameters
    # -----------------------------------------------------------------------------
    bcs                  = cfg_get(model_cfg, "bcs", {}) or {}
    bcs_cutting_speed    = float(bcs.get("cutting_speed",    cutting_speed))
    bcs_cutting_faces    = list(bcs.get("cutting_velocity_faces", ["eul_bot"]))      # subset of {eul_left, eul_right, eul_top, eul_bot}
    bcs_initial_velocity = float(bcs.get("initial_velocity", cutting_speed))
    bcs_ambient_temp     = float(bcs.get("ambient_temperature", 20.0))               # in °C — absoluteZero is set to -273.15 below

    # Per-face Eulerian BC settings — read once into a flat dict.
    # face_enabled_X: if False, no EulerianBC is created on that face.
    # eulerian_bc_mode_X: "inflow" | "outflow" | "both" — controls `definition`.
    # eulerian_inflow_X / eulerian_outflow_X: Abaqus symbolic-constant *names*.
    EUL_FACES = ("left", "right", "top", "bottom")
    eulbc_cfg = {}
    for _f in EUL_FACES:
        eulbc_cfg[_f] = {
            "enabled":  bool(bcs.get("face_enabled_"      + _f, False)),
            "mode":     str( bcs.get("eulerian_bc_mode_"  + _f, "both")),
            "inflow":   str( bcs.get("eulerian_inflow_"   + _f, "FREE")),
            "outflow":  str( bcs.get("eulerian_outflow_"  + _f, "FREE")),
        }

    # Element types are FROZEN (EC3D8RT for the workpiece, C3D8RT for the tool) and
    # no longer read from the config — see the ElemType() calls in the meshing
    # section. The per-body element config and its mapping tables were removed.

    job_name = cfg_get(run_cfg, "job_name", "default_name")
    cpus     = int(cfg_get(run_cfg, "cpus",    4))
    domains  = cfg_get(run_cfg, "domains", cpus)
    # When True, build the model and write the .inp deck only — no solver run,
    # no .odb, no results extraction. Lets the user inspect/keep the input file.
    write_inp_only = bool(cfg_get(run_cfg, "write_inp_only", False))
    # Tool mesh parameters are normalized here as well so model-building
    # functions do not reach back into the raw GUI dictionary.
    tool_elem_size = float(cfg_get(model_cfg, "mesh.tool_elem_size", 0.005))
    inter_elem_size = float(cfg_get(model_cfg, "mesh.inter_elem_size", 0.02))
    max_elem_size = float(cfg_get(model_cfg, "mesh.max_elem_size", 0.05))

    # Keep the parameter carrier Python-2.7-friendly and dependency-free.
    p = locals().copy()
    p.pop("model_cfg", None)
    p.pop("run_cfg", None)
    return p


def create_parts(model, p):
    """Create Eulerian, volume-fraction workpiece and cutting-tool parts."""
    elem_size = p["elem_size"]
    l_wp = p["l_wp"]
    h_wp = p["h_wp"]
    l_void = p["l_void"]
    h_void = p["h_void"]
    h_tool = p["h_tool"]
    l_tool = p["l_tool"]
    r_tool = p["r_tool"]
    rake_angle = p["rake_angle"]
    clear_angle = p["clear_angle"]
    #%%% Euler part
    eul_sketch = model.ConstrainedSketch(name='eul_sketch', sheetSize=5)
    eul_sketch.rectangle(point1=(-l_wp, -h_wp), point2=(l_void, h_void))
    eulPart = model.Part(name='Euler', dimensionality=THREE_D, type=EULERIAN)
    eulPart.BaseSolidExtrude(sketch=eul_sketch, depth=elem_size)

    #%%% Workpiece part (for VolFraction)
    wp_sketch = model.ConstrainedSketch(name='wp_sketch', sheetSize=5)
    wp_sketch.rectangle(point1=(-l_wp, -h_wp), point2=(0, 0))
    wpPart = model.Part(name='Workpiece', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    wpPart.BaseSolidExtrude(sketch=wp_sketch, depth=elem_size)

    #%%% Tool part
    tool_sketch = model.ConstrainedSketch(name="tool_sketch", sheetSize=5)
    tool_origin = tool_sketch.Spot(point=(0.0, 0.0))
    tool_sketch.FixedConstraint(entity=tool_origin)
    tool_l1 = tool_sketch.Line(point1=(0.0, 0.0), point2=(0.0, h_tool))
    tool_l2 = tool_sketch.Line(point1=(0.0, h_tool), point2=(l_tool, h_tool))
    tool_l3 = tool_sketch.Line(point1=(l_tool, h_tool), point2=(l_tool, 0.0))
    tool_l4 = tool_sketch.Line(point1=(l_tool, 0.0), point2=(0.0, 0.0))
    tool_sketch.HorizontalConstraint(entity=tool_l2)
    tool_sketch.VerticalConstraint(entity=tool_l3)
    tool_sketch.FilletByRadius(
        radius=r_tool,
        curve1=tool_l1,
        nearPoint1=(-0.1 * l_tool, 0.1 * h_tool),
        curve2=tool_l4,
        nearPoint2=(0.1 * l_tool, -0.1 * h_tool),)
    tool_sketch.CoincidentConstraint(entity1=tool_l1, entity2=tool_origin)
    tool_sketch.CoincidentConstraint(entity1=tool_l4, entity2=tool_origin)
    tool_sketch.ObliqueDimension(vertex1=tool_sketch.vertices[2], vertex2=tool_sketch.vertices[3], textPoint=(l_tool / 2, 1.1 * h_tool), value=l_tool)
    tool_sketch.ObliqueDimension(vertex1=tool_sketch.vertices[3], vertex2=tool_sketch.vertices[4], textPoint=(1.1 * l_tool, h_tool / 2), value=h_tool)
    tool_sketch.AngularDimension(line1=tool_l3, line2=tool_l4, textPoint=(l_tool / 2, h_tool / 2), value=90 + clear_angle)
    tool_sketch.AngularDimension(line1=tool_l1, line2=tool_l2, textPoint=(l_tool / 2, h_tool / 2), value=90 + rake_angle)
    toolPart = model.Part(name="Tool", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    toolPart.BaseSolidExtrude(sketch=tool_sketch, depth=elem_size)
    return eulPart, wpPart, toolPart


def create_materials(model, p):
    """Create Eulerian/workpiece and tool material definitions."""
    emat = p["emat"]
    tmat = p["tmat"]
    ms_eul = p["ms_eul"]
    ms_tool = p["ms_tool"]
    #%%% Materials
    EulerMat = model.Material(name='Euler')
    EulerMat.Density(table=((float(emat["rho"]) * ms_eul,),))
    EulerMat.Elastic(table=((float(emat["E"]), float(emat["nu"])),))
    EulerMat.Conductivity(table=((float(emat["k"]),),))
    EulerMat.SpecificHeat(table=((float(emat["Cp"]) / ms_eul,),), law=CONSTANTPRESSURE)
    EulerMat.Expansion(table=((float(emat["alpha"]),),))
    EulerMat.InelasticHeatFraction(fraction=float(emat["beta"]))

    EulerMat.Plastic(
        hardening=JOHNSON_COOK,
        table=((float(emat["A"]), float(emat["B"]), float(emat["n"]),
                float(emat["m"]), float(emat["Tm"]), float(emat["Tr"])),)
    ).RateDependent(
        type=JOHNSON_COOK,
        table=((float(emat["C"]), float(emat["eps_dot0"])),)
    )

    # EulerMat.JohnsonCookDamageInitiation(
    #     table=((float(emat["D1"]), float(emat["D2"]), float(emat["D3"]),
    #             float(emat["D4"]), float(emat["D5"]),
    #             float(emat["Tm"]), float(emat["Tr"]), float(emat["eps0"])),)
    # ).DamageEvolution(type=ENERGY, softening=EXPONENTIAL, table=((float(emat["Gf"]),),))

    ToolMat = model.Material(name='Tool')
    ToolMat.Density(table=((float(tmat["rho"]) * ms_tool,),))
    ToolMat.Elastic(table=((float(tmat["E"]), float(tmat["nu"])),))
    ToolMat.Conductivity(table=((float(tmat["k"]),),))
    ToolMat.SpecificHeat(table=((float(tmat["Cp"]) / ms_tool,),))
    ToolMat.Expansion(table=((float(tmat["alpha"]),),))


def create_sections(model, eulPart, toolPart):
    """Create and assign Eulerian and solid sections."""
    #%%% Sections
    model.EulerianSection(name='Euler', data={'euler-1': 'Euler'})
    model.HomogeneousSolidSection(name='Tool', material='Tool')
    eulPart.SectionAssignment(region=Region(cells=eulPart.cells), sectionName='Euler')
    toolPart.SectionAssignment(region=Region(cells=toolPart.cells), sectionName='Tool')


def create_assembly(model, parts, p):
    """Instantiate, position and configure model parts in the root assembly."""
    eulPart, wpPart, toolPart = parts
    tool_tx = p["tool_tx"]
    tool_ty = p["tool_ty"]
    wp_tx = p["wp_tx"]
    wp_ty = p["wp_ty"]
    mesh_tx = p["mesh_tx"]
    mesh_ty = p["mesh_ty"]
    #%%% Assembly
    assembly = model.rootAssembly
    eul_instance  = assembly.Instance(name='Euler',     part=eulPart,  dependent=OFF)
    wp_instance   = assembly.Instance(name='Workpiece', part=wpPart,   dependent=OFF)
    tool_instance = assembly.Instance(name='Tool',      part=toolPart, dependent=OFF)

    assembly.translate(instanceList=('Tool',),      vector=(tool_tx, tool_ty, 0.0))
    assembly.translate(instanceList=('Workpiece',), vector=(wp_tx,   wp_ty,   0.0))
    assembly.translate(instanceList=('Euler',),     vector=(mesh_tx, mesh_ty, 0.0))

    wp_instance.translateTo(movableList=(wp_instance.faces[2], ),
                            fixedList=(tool_instance.faces[6], ),
                            direction=(1.0, 0.0, 0.0),
                            clearance=0.0)
    assembly.excludeFromSimulation(instances=(wp_instance,), exclude=True)
    return assembly, eul_instance, wp_instance, tool_instance


def create_mesh(assembly, eul_instance, tool_instance, p):
    """Mesh Eulerian domain and cutting tool; return the body sets used later."""
    elem_size = p["elem_size"]
    tool_elem_size = p["tool_elem_size"]
    inter_elem_size = p["inter_elem_size"]
    max_elem_size = p["max_elem_size"]
    #%%% Mesh
    assembly.seedPartInstance(regions=(eul_instance,), size=elem_size)

    eul_elemType = ElemType(elemCode=EC3D8RT, elemLibrary=EXPLICIT,
        secondOrderAccuracy=OFF, hourglassControl=DEFAULT)

    eul_set = assembly.Set(name='Euler', cells=eul_instance.cells)
    assembly.setElementType(regions=eul_set, elemTypes=(eul_elemType, eul_elemType, eul_elemType))
    assembly.setMeshControls(regions=eul_instance.cells, elemShape=HEX, technique=STRUCTURED)

    nose_mesh_edges = [13, 14]
    # tool_elem_size normalized in prepare_parameters()
    assembly.seedEdgeBySize(edges=[tool_instance.edges[i] for i in nose_mesh_edges],
                            size=tool_elem_size)
    width_mesh_edge = [1]
    assembly.seedEdgeByNumber(edges=[tool_instance.edges[i] for i in width_mesh_edge], number=1)

    # inter_elem_size normalized in prepare_parameters()
    # max_elem_size normalized in prepare_parameters()

    # Border edges (bias 1): from inter_elem_size (junction) out to max_elem_size.
    assembly.seedEdgeByBias(
        biasMethod=SINGLE,
        end1Edges=[tool_instance.edges[4],tool_instance.edges[6]],
        end2Edges=[tool_instance.edges[7],tool_instance.edges[9]],
        minSize=inter_elem_size,
        maxSize=max_elem_size)

    # Rake + clearance faces (bias 2): from the nose seed out to inter_elem_size.
    assembly.seedEdgeByBias(
        biasMethod=SINGLE,
        end1Edges=[tool_instance.edges[0],tool_instance.edges[2]],
        end2Edges=[tool_instance.edges[10],tool_instance.edges[12]],
        minSize=tool_elem_size,
        maxSize=inter_elem_size)

    tool_elemType = ElemType(elemCode=C3D8RT, elemLibrary=EXPLICIT,
        secondOrderAccuracy=OFF, hourglassControl=DEFAULT)

    tool_set = assembly.Set(name='Tool', cells=tool_instance.cells)
    assembly.setElementType(regions=tool_set, elemTypes=(tool_elemType, tool_elemType, tool_elemType))
    assembly.setMeshControls(regions=tool_instance.cells, elemShape=HEX, technique=SWEEP)

    assembly.generateMesh(regions=(eul_instance, tool_instance))
    return eul_set, tool_set


def create_sets_and_fields(assembly, eul_instance, wp_instance, tool_instance, p):
    """Create reference point, extraction ROI sets and Eulerian volume-fraction field."""
    margin = p["margin"]
    xmin = p["xmin"]
    xmax = p["xmax"]
    ymin = p["ymin"]
    ymax = p["ymax"]
    zmin = p["zmin"]
    zmax = p["zmax"]
    #%%% Sets + fields
    ref_point  = assembly.ReferencePoint(point=tool_instance.vertices[4])
    RP         = assembly.Set(name='RP', referencePoints=(assembly.referencePoints[ref_point.id],))
    tool_nodes = assembly.Set(name='tool_nodes', nodes=tool_instance.nodes)
    tool_elem  = assembly.Set(name='tool_elem',  elements=tool_instance.elements)

    assembly.DiscreteFieldByVolumeFraction(name='VolFraction', description='',
                                           eulerianInstance=eul_instance, referenceInstance=wp_instance)

    eul_nodes = assembly.Set(name='eul_nodes', nodes=eul_instance.nodes)

    roi_nodes = eul_instance.nodes.getByBoundingBox(
        xMin=xmin - margin, xMax=xmax + margin,
        yMin=ymin - margin, yMax=ymax + margin,
        zMin=zmin - margin, zMax=zmax + margin
    )
    # ROI (region of interest) sets used by extraction: one node set and one
    # element set covering the user's ROI on the z = 0 face. They are created
    # here so they appear in the model/.inp; extraction re-derives the labels
    # from the ODB and stops with an error if either is empty.
    # (There is a single region of interest; 'ROI_node' and 'ROI_elem' are its
    # two declinations, not two different zones.)
    roi_elems = eul_instance.elements.getByBoundingBox(
        xMin=xmin - margin, xMax=xmax + margin,
        yMin=ymin - margin, yMax=ymax + margin,
        zMin=zmin - margin, zMax=zmax + margin
    )
    assembly.Set(name='ROI_node', nodes=roi_nodes)
    assembly.Set(name='ROI_elem', elements=roi_elems)
    return RP, tool_elem


def create_interaction(model, RP, tool_elem, p):
    """Create contact property, general contact and rigid-tool constraint."""
    inter_tangential = p["inter_tangential"]
    inter_friction = p["inter_friction"]
    inter_slip_tol = p["inter_slip_tol"]
    inter_pressure = p["inter_pressure"]
    inter_heat_gen = p["inter_heat_gen"]
    inter_heat_to_slave = p["inter_heat_to_slave"]
    inter_heat_to_master = p["inter_heat_to_master"]
    #%%% Contact
    # All knobs (tangential formulation, friction coefficient, slip tolerance,
    # pressure-overclosure law, heat generation + fractions) come from the GUI's
    # Interaction tab.
    IntProp = model.ContactProperty(name='IntProp')

    # Tangential behavior
    _tang_kind = _resolve_mapping(_TANGENTIAL_MAP, inter_tangential, "tangential_formulation")
    if _tang_kind == PENALTY:
        IntProp.TangentialBehavior(
            formulation=PENALTY,
            table=((inter_friction,),),
            fraction=inter_slip_tol,
        )
    elif _tang_kind == ROUGH:
        # "Rough" disallows slip; friction coeff is ignored by Abaqus.
        IntProp.TangentialBehavior(formulation=ROUGH)
    else:
        # FRICTIONLESS — no friction, no slip tolerance.
        IntProp.TangentialBehavior(formulation=FRICTIONLESS)

    # Normal behavior (pressure-overclosure law)
    IntProp.NormalBehavior(
        pressureOverclosure=_resolve_mapping(_PRESSURE_OVERCLOSURE_MAP, inter_pressure, "pressure_overclosure"),
    )

    # Optional heat-generation interaction property
    if inter_heat_gen:
        IntProp.HeatGeneration(
            conversionFraction=1.0,
            slaveFraction=inter_heat_to_slave,
        )
        # `slaveFraction` covers most cases; the master fraction is implicitly
        # (1 - slaveFraction) in Abaqus's API. If `inter_heat_to_slave` and
        # `inter_heat_to_master` don't sum to 1, the GUI value of
        # `heat_fraction_to_master` is ignored — Abaqus only stores one of them.

    model.ContactExp(name='Contact', createStepName='Initial',
                       contactPropertyAssignments=((GLOBAL, SELF, 'IntProp'),))
    model.RigidBody(name='Rigid_Tool', refPointRegion=RP, bodyRegion=tool_elem)


def create_step(model, assembly, RP, p):
    """Create the thermo-mechanical Explicit step and the FROZEN output requests."""
    sim_time = p["sim_time"]
    fo_variables = p["fo_variables"]
    n_frames = p["n_frames"]
    ho_n_intervals = p["ho_n_intervals"]
    #%%% Step
    model.TempDisplacementDynamicsStep(name='Cut', previous='Initial',
                                       timePeriod=sim_time, nlgeom=ON,
                                       linearBulkViscosity=0.06,
                                       quadBulkViscosity=1.2,
                                       improvedDtMethod=ON)

    # -----------------------------------------------------------------------
    # Output filters. Abaqus filters at the SOLVER increment, BEFORE writing to
    # the ODB -- the only stage where aliasing can still be prevented (once
    # aliased data is written, no post-processing recovers it).
    # TWO cutoffs because the two observables have different bandwidths; see
    # prepare_parameters(). The filtered variables are stored under SUFFIXED
    # names (V -> V_CAMERABAND), not under the bare name.
    # -----------------------------------------------------------------------
    fo_filter = None
    ho_filter = None
    if p.get("filter_enabled"):
        if p.get("filter_cutoff", 0.0) > 0:
            model.ButterworthFilter(name='CameraBand',
                                    cutoffFrequency=p["filter_cutoff"], order=2)
            fo_filter = 'CameraBand'
            print("[META] field_filter=CameraBand cutoff=%g Hz" % p["filter_cutoff"])
        if p.get("filter_cutoff_history", 0.0) > 0:
            model.ButterworthFilter(name='SensorBand',
                                    cutoffFrequency=p["filter_cutoff_history"],
                                    order=2)
            ho_filter = 'SensorBand'
            print("[META] history_filter=SensorBand cutoff=%g Hz"
                  % p["filter_cutoff_history"])
        sys.stdout.flush()

    # ---- Field output (frozen variable list, see prepare_parameters) -------
    if fo_filter is None:
        model.FieldOutputRequest(
            name='F-Output-1', createStepName='Cut',
            variables=fo_variables, numIntervals=n_frames)
    else:
        model.FieldOutputRequest(
            name='F-Output-1', createStepName='Cut',
            variables=fo_variables, numIntervals=n_frames,
            filter=fo_filter)

    # ---- History output ---------------------------------------------------
    # RF on the tool RP = the cutting forces. Filtered only to avoid aliasing
    # at the output rate; the real sensor bandwidth is applied afterwards in
    # post-processing (a low runtime cutoff would demand an unreachable
    # mass-scaling factor).
    if ho_filter is None:
        model.HistoryOutputRequest(
            name='H-Output-1', createStepName='Cut',
            region=RP, variables=('RF1', 'RF2',),
            numIntervals=ho_n_intervals)
    else:
        model.HistoryOutputRequest(
            name='H-Output-1', createStepName='Cut',
            region=RP, variables=('RF1', 'RF2',),
            numIntervals=ho_n_intervals, filter=ho_filter)

    # PRESELECT carries ALLKE/ALLIE, which the mass-scaling energy guard reads.
    # Deliberately NOT filtered: the guard must see the true energy balance,
    # not a band-limited version of it.
    model.HistoryOutputRequest(
        name='H-Output-2', createStepName='Cut',
        variables=PRESELECT, numIntervals=ho_n_intervals)

    # Eulerian mass/volume per material instance, over the whole Eulerian
    # domain: a conservation check (is material leaving the domain, or being
    # lost numerically?) that the model had no indicator for. Cheap: two
    # scalars per sample. Not filtered -- a conservation check must see the
    # raw balance.
    try:
        model.HistoryOutputRequest(
            name='H-Output-3', createStepName='Cut',
            region=assembly.sets['Euler'],
            variables=('MASSEUL', 'VOLEUL'),
            numIntervals=ho_n_intervals)
    except Exception as exc:
        # Non-fatal: losing the conservation check must not lose the run.
        print("[WARNING] MASSEUL/VOLEUL history not created: %s" % exc)
        sys.stdout.flush()


def create_boundary_conditions(model, assembly, eul_instance, eul_set, tool_set, RP, p):
    """Create Eulerian face BCs, cutting velocity, plane strain and initial fields."""
    mesh_tx = p["mesh_tx"]
    mesh_ty = p["mesh_ty"]
    l_wp = p["l_wp"]
    l_void = p["l_void"]
    h_wp = p["h_wp"]
    h_void = p["h_void"]
    elem_size = p["elem_size"]
    EUL_FACES = p["EUL_FACES"]
    eulbc_cfg = p["eulbc_cfg"]
    bcs_cutting_faces = p["bcs_cutting_faces"]
    bcs_cutting_speed = p["bcs_cutting_speed"]
    bcs_initial_velocity = p["bcs_initial_velocity"]
    bcs_ambient_temp = p["bcs_ambient_temp"]
    #%%% Boundary conditions

    #%%%% Eulerian face Sets (4 faces of the Eulerian box)
    # We identify the 4 lateral faces of the Eulerian rectangle by their
    # geometric location (getByBoundingBox) rather than by Abaqus's internal
    # face index, so the code stays correct even if Abaqus reorders the faces.
    # Face coordinates: the sketch ran from (-l_wp, -h_wp) to (l_void, h_void),
    # translated by (mesh_tx, mesh_ty). The extruded depth is along +z.
    _x_left   = mesh_tx - l_wp
    _x_right  = mesh_tx + l_void
    _y_bottom = mesh_ty - h_wp
    _y_top    = mesh_ty + h_void
    _eps      = 1e-6 * max(l_wp + l_void, h_wp + h_void)
    _z_pad    = elem_size

    _eul_face_bbox = {
        "left":   dict(xMin=_x_left  - _eps, xMax=_x_left  + _eps,
                       yMin=_y_bottom - _eps, yMax=_y_top    + _eps,
                       zMin=-_z_pad,          zMax=elem_size + _z_pad),
        "right":  dict(xMin=_x_right - _eps, xMax=_x_right + _eps,
                       yMin=_y_bottom - _eps, yMax=_y_top    + _eps,
                       zMin=-_z_pad,          zMax=elem_size + _z_pad),
        "bottom": dict(xMin=_x_left   - _eps, xMax=_x_right + _eps,
                       yMin=_y_bottom - _eps, yMax=_y_bottom + _eps,
                       zMin=-_z_pad,          zMax=elem_size + _z_pad),
        "top":    dict(xMin=_x_left   - _eps, xMax=_x_right + _eps,
                       yMin=_y_top    - _eps, yMax=_y_top    + _eps,
                       zMin=-_z_pad,          zMax=elem_size + _z_pad),
    }

    eul_face_surfaces = {}  # face_key -> Surface
    for _f in EUL_FACES:
        _faces_obj = eul_instance.faces.getByBoundingBox(**_eul_face_bbox[_f])
        if len(_faces_obj) > 0:
            eul_face_surfaces[_f] = assembly.Surface(
                name='eul_' + _f, side1Faces=_faces_obj)

    #%%%% Eulerian inflow / outflow BCs
    # One BC per enabled face. The `definition` argument controls which fluxes
    # are specified (INFLOW / OUTFLOW / BOTH) and we always pass both
    # inflowType and outflowType — Abaqus ignores the one not selected by
    # `definition`.
    for _f in EUL_FACES:
        _c = eulbc_cfg[_f]
        if not _c["enabled"] or _f not in eul_face_surfaces:
            continue
        model.EulerianBC(
            name='eulbc_' + _f,
            createStepName='Cut',
            region=eul_face_surfaces[_f],
            definition=_resolve_mapping(_BC_DEFINITION_MAP, _c["mode"], "eulerian_boundary.mode"),
            inflowType=_resolve_mapping(_INFLOW_MAP, _c["inflow"], "inflow"),
            outflowType=_resolve_mapping(_OUTFLOW_MAP, _c["outflow"], "outflow"),
        )

    #%%%% Cutting-velocity BC
    # Applied to the union of Eulerian faces listed in bcs.cutting_velocity_faces.
    # Mapping from the GUI's face IDs (eul_left, eul_right, eul_top, eul_bot)
    # to our `eul_face_surfaces` dict (left, right, top, bottom):
    _VCUT_FACE_KEY_MAP = {
        "eul_left":  "left",
        "eul_right": "right",
        "eul_top":   "top",
        "eul_bot":   "bottom",
    }
    # Build a single FaceArray by adding face arrays together (Abaqus's
    # FaceArray supports the `+` operator). Starting from an empty FaceArray
    # is fiddly, so we initialise with the first face and then add the rest.
    _vcut_face_array = None
    for _fid in bcs_cutting_faces:
        _fkey = _VCUT_FACE_KEY_MAP.get(_fid)
        if not _fkey:
            continue
        _bbox = _eul_face_bbox.get(_fkey)
        if _bbox is None:
            continue
        _faces_obj = eul_instance.faces.getByBoundingBox(**_bbox)
        if len(_faces_obj) == 0:
            continue
        if _vcut_face_array is None:
            _vcut_face_array = _faces_obj
        else:
            _vcut_face_array = _vcut_face_array + _faces_obj

    # model.SmoothStepAmplitude(
    #     name='Smooth-Step', 
    #     timeSpan=STEP, 
    #     data=((0.0, 0.0), (6e-07, 1.0))
    #     )

    if _vcut_face_array is not None and len(_vcut_face_array) > 0:
        work_sides = assembly.Set(
            name='work_sides',
            faces=_vcut_face_array,
        )
        cut_BC = model.VelocityBC(
            name='Cutting_speed', createStepName='Initial',
            region=work_sides, v1=SET,
        )
        cut_BC.setValuesInStep(stepName='Cut', v1=bcs_cutting_speed)
        # cut_BC.setValuesInStep(stepName='Cut', v1=bcs_cutting_speed, amplitude='Smooth-Step')
    else:
        # No cutting face selected — create no velocity BC. The user will see
        # this in the .msg log; it's a valid (if unusual) configuration.
        cut_BC = None

    #%%%% Tool fixity + plane strain on the Eulerian mesh
    model.VelocityBC(name='Tool_fix', createStepName='Initial', region=RP,
                       v1=SET, v2=SET, v3=SET, vr1=SET, vr2=SET, vr3=SET)
    model.VelocityBC(name='Plane_strain', createStepName='Initial',
                       region=eul_set, v3=SET)

    #%%%% Initial Eulerian velocity (predefined field)
    # Independent from the cutting-velocity BC — set by `bcs.initial_velocity`.
    model.Velocity(
        name='Init_speed',
        region=eul_set, field='',
        distributionType=MAGNITUDE,
        velocity1=bcs_initial_velocity,
    )

    rgn = Region(cells=eul_instance.cells)
    model.MaterialAssignment(name='Material', instanceList=(eul_instance,),
                               useFields=True, fieldList=((rgn, ("VolFraction",)),))

    #%%%% Initial temperature (both bodies)
    model.Temperature(name='Init_temp',
                        createStepName='Initial', region=eul_set,
                        magnitudes=bcs_ambient_temp)
    model.Temperature(name='Init_temp_tool',
                        createStepName='Initial', region=tool_set,
                        magnitudes=bcs_ambient_temp)


def build_model(model_cfg, run_cfg):
    """Build the full Abaqus model and return model + normalized parameters."""
    p = prepare_parameters(model_cfg, run_cfg)
    model = mdb.Model(name=p["job_name"], absoluteZero=-273.15)

    parts = create_parts(model, p)
    create_materials(model, p)
    create_sections(model, parts[0], parts[2])
    assembly, eul_instance, wp_instance, tool_instance = create_assembly(model, parts, p)
    eul_set, tool_set = create_mesh(assembly, eul_instance, tool_instance, p)
    RP, tool_elem = create_sets_and_fields(assembly, eul_instance, wp_instance, tool_instance, p)
    create_interaction(model, RP, tool_elem, p)
    create_step(model, assembly, RP, p)
    create_boundary_conditions(model, assembly, eul_instance, eul_set, tool_set, RP, p)
    return model, p


def create_job(model, p):
    """Create the Abaqus Job object without submitting it."""
    job_name = p["job_name"]
    cpus = p["cpus"]
    domains = p["domains"]
    job = mdb.Job(
        name=job_name,
        model=model.name,
        type=ANALYSIS,
        numCpus=cpus,
        numDomains=domains,
        explicitPrecision=DOUBLE,
        nodalOutputPrecision=FULL,
    )
    return job


def _check_job_succeeded(job_name):
    """Raise RuntimeError unless the .sta reports a successful analysis.

    WHY NOT job.status: under ``abaqus cae noGUI=`` the Job.messages list is
    never populated ("Job messages are not returned if a script is run without
    the Abaqus/CAE GUI"), and Job.status is documented as NONE whenever
    messages is empty. So `job.status != COMPLETED` is True for EVERY run in
    this execution mode, successful or not -- it blocked extraction on jobs
    that had actually completed.

    The .sta file is written by the solver itself and ends with either
    "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" or "THE ANALYSIS HAS NOT BEEN
    COMPLETED", which is a reliable ground truth in both GUI and noGUI modes.
    A missing .sta means the job died before the solver started (preprocessing
    error), which is also a failure."""
    sta_path = job_name + ".sta"
    if not os.path.isfile(sta_path):
        print("[STAGE] SOLVE_FAILED")
        sys.stdout.flush()
        raise RuntimeError(
            "Abaqus job '%s' produced no .sta file: the analysis did not "
            "start (check the .dat for preprocessing errors). Results "
            "extraction skipped." % job_name)
    handle = open(sta_path, "rb")
    try:
        text = handle.read()
    finally:
        handle.close()
    # Bytes on both Python 2.7 (Abaqus) and 3.x; decode defensively because
    # the .sta is not guaranteed to be pure ASCII.
    if not isinstance(text, str):
        text = text.decode("latin-1", "replace")
    if "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" not in text:
        print("[STAGE] SOLVE_FAILED")
        sys.stdout.flush()
        tail = text.strip().split("\n")[-15:]
        raise RuntimeError(
            "Abaqus job '%s' did not complete successfully. Last lines of "
            "the .sta:\n%s\nResults extraction skipped."
            % (job_name, "\n".join(tail)))


# ---------------------------------------------------------------------------
# Working-directory cleanup
# ---------------------------------------------------------------------------
# An Abaqus/Explicit job leaves ~20 scratch files behind (.com .prt .res .mdl
# .stt .abq .pac .sel .lck, per-CPU .1/.2/..., SMA* directories). On a
# parameter sweep they pile up and dominate the disk usage of the working
# directory.
#
# EXTENSIONS KEPT. The first three are the ones asked for; the last two are
# the pipeline's own deliverables and must never be removed -- without them
# the GUI has no results at all.
_KEEP_EXTENSIONS = set([
    ".inp",     # the deck: what was actually sent to the solver
    ".odb",     # the results database
    ".sta",     # status: increments, stable dt, critical element, completion
    ".npz",     # <job>.results.npz  -- extracted fields (pipeline output)
    ".json",    # <job>.meta.json    -- config snapshot + result descriptors
])

# Diagnostics. NOT in the list above because they are not "results", but they
# are the files that actually let you find out WHY a run misbehaved: .dat holds
# preprocessing errors and the input warnings, .msg holds distortion warnings
# and floating-point errors. They are small (a few hundred kB). Set
# KEEP_DIAGNOSTICS = False if you really want them gone.
_DIAGNOSTIC_EXTENSIONS = set([".dat", ".msg"])
KEEP_DIAGNOSTICS = True


def cleanup_working_directory(job_name, keep_diagnostics=None, verbose=True):
    """Delete the Abaqus scratch files of `job_name`, keeping the essentials.

    Only touches entries whose name starts with `job_name` -- other jobs in the
    same working directory are left alone.

    MUST be called only after a SUCCESSFUL run and after extraction: the
    extractor needs the .odb, and on a failure the scratch files are exactly
    what you need to diagnose it.

    Never raises: losing a cleanup must not lose a completed run."""
    if keep_diagnostics is None:
        keep_diagnostics = KEEP_DIAGNOSTICS
    keep = set(_KEEP_EXTENSIONS)
    if keep_diagnostics:
        keep |= _DIAGNOSTIC_EXTENSIONS

    removed, freed, failed = 0, 0, 0
    try:
        entries = os.listdir(".")
    except Exception as exc:
        print("[WARNING] cleanup skipped (cannot list directory): %s" % exc)
        sys.stdout.flush()
        return
    for name in entries:
        if not name.startswith(job_name):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in keep:
            continue
        # A .lck file means the .odb is still held open; removing it while a
        # writer is alive can corrupt the database. Leave it.
        if ext == ".lck":
            continue
        try:
            if os.path.isdir(name):
                import shutil
                size = 0
                for root, _dirs, files in os.walk(name):
                    for f in files:
                        try:
                            size += os.path.getsize(os.path.join(root, f))
                        except Exception:
                            pass
                shutil.rmtree(name)
            else:
                size = os.path.getsize(name)
                os.remove(name)
            removed += 1
            freed += size
        except Exception:
            # File locked by the solver or the OS: skip it silently, it is
            # scratch either way.
            failed += 1
    if verbose:
        print("[META] cleanup removed=%d freed=%.1f MB kept=%s%s"
              % (removed, freed / 1048576.0, sorted(keep),
                 (" skipped=%d" % failed) if failed else ""))
        sys.stdout.flush()


def run_job(job, p):
    """Write the input deck or execute the solver.

    Returns True when a completed ODB is ready for extraction and False
    for write-inp-only mode. Raises RuntimeError on solver failure.
    """
    job_name = p["job_name"]
    sim_time = p["sim_time"]
    n_frames = p["n_frames"]
    write_inp_only = p["write_inp_only"]

    print("[META] job_name=%s" % job_name)
    print("[META] sim_time=%g" % sim_time)
    print("[META] n_frames=%d" % n_frames)
    if write_inp_only:
        print("[STAGE] WRITE_INP_START")
        sys.stdout.flush()
        job.writeInput(consistencyChecking=OFF)
        print("[STAGE] INP_WRITTEN")
        print("[OK] Wrote %s.inp (no analysis run)." % job_name)
        sys.stdout.flush()
        return False

    print("[STAGE] SOLVE_START")
    sys.stdout.flush()
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    _check_job_succeeded(job_name)
    print("[STAGE] SOLVE_DONE")
    sys.stdout.flush()
    return True
