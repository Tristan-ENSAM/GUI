# -*- coding: utf-8 -*-

import sys
import os
import argparse
import ast
from decimal import Decimal, getcontext

from abaqus import *
from abaqusConstants import *
from step import *
from sketch import *
from load import *
from part import *
from mesh import *
from interaction import *
import regionToolset

getcontext().prec = 28

# =============================================================================
#%% ARGUMENT PARSER
# =============================================================================
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model_cfg", required=True, help="Python dict literal for model")
_parser.add_argument("--run_cfg", required=True, help="Python dict literal for job run")
_args, _unknown = _parser.parse_known_args(
    sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
)

MODEL_CFG = ast.literal_eval(_args.model_cfg)
RUN_CFG   = ast.literal_eval(_args.run_cfg)

if not isinstance(MODEL_CFG, dict):
    raise TypeError("--model_cfg must evaluate to a dict")
if not isinstance(RUN_CFG, dict):
    raise TypeError("--run_cfg must evaluate to a dict")

# =============================================================================
#%% UTILITIES
# =============================================================================
def cfg_get(cfg, path, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def discretize(dim, element_size):
    """Rounds dim down to the nearest multiple of element_size (floor).
    Uses Decimal to avoid IEEE 754 floating point errors."""
    d  = Decimal(str(dim))
    es = Decimal(str(element_size))
    if es <= 0:
        raise ValueError("element_size must be > 0, got: {}".format(element_size))
    n = d // es
    if n <= 0:
        raise ValueError(
            "discretize({}, {}) -> {} elements: dim too small relative to element_size".format(dim, element_size, int(n))
        )
    return float(n * es)

# =============================================================================
# %% PARAMETERS EXTRACTION
# =============================================================================
elem_size = float(cfg_get(MODEL_CFG, "mesh.elem_size", 0.01))

cutting_speed = float(cfg_get(MODEL_CFG, "bcs.cutting_speed", 1000.0))
# `sim_time` and `n_frames` are now owned by the Step tab. They are
# mirrored into `process` for backwards compatibility, but the preferred
# read path is `step`. Try step first, fall back to process.
sim_time      = float(cfg_get(MODEL_CFG, "step.sim_time",
                              cfg_get(MODEL_CFG, "step.sim_time",  0.0001)))
n_frames      = int(  cfg_get(MODEL_CFG, "step.n_frames",
                              cfg_get(MODEL_CFG, "step.n_frames",  1)))

# -----------------------------------------------------------------------------
# Output selection (Step tab > Field output / History output)
# -----------------------------------------------------------------------------
step_output = cfg_get(MODEL_CFG, "step.output", {}) or {}

# Field-output variables: keep only those whose flag is True. Order matters
# only for readability of the .inp file.
_FO_PAIRS = [
    ("fo_S",       "S"),
    ("fo_PEEQ",    "PEEQ"),
    ("fo_VP",      "VP"),
    ("fo_P",       "P"),
    ("fo_ERV",     "ERV"),
    ("fo_TEMP",    "TEMP"),
    ("fo_HFL",     "HFL"),
    ("fo_HP",      "HP"),
    ("fo_EVF",     "EVF"),
    ("fo_MFL",     "MFL"),
    ("fo_A",       "A"),
    ("fo_V",       "V"),
    ("fo_DMICRT",  "DMICRT"),
    ("fo_SDEG",    "SDEG"),
    ("fo_STATUS",  "STATUS"),
    ("fo_SDV",     "SDV"),
    ("fo_CSTRESS", "CSTRESS"),
    ("fo_U",       "U"),
    ("fo_RF",      "RF"),
    ("fo_NT",      "NT"),
]
fo_variables = tuple(
    abq_id for attr, abq_id in _FO_PAIRS
    if bool(step_output.get(attr, True))
)

# History output
ho_preselect      = bool( step_output.get("ho_preselect",     True))
ho_rf_on_rp       = bool( step_output.get("ho_rf_on_rp",      True))
# Number of history-output intervals. Defaults to the field-output
# n_frames so a freshly-loaded profile without this key behaves like
# "synced with field output".
ho_n_intervals    = int(  step_output.get("ho_n_intervals",   n_frames))

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
ms_enabled = bool(cfg_get(MODEL_CFG, "step.mass_scaling_enabled", False))
ms_eul     = float(cfg_get(MODEL_CFG, "step.mass_scaling_factor_eulerian", 1.0))
ms_tool    = float(cfg_get(MODEL_CFG, "step.mass_scaling_factor_tool",     1.0))
if not ms_enabled:
    ms_eul  = 1.0
    ms_tool = 1.0

# Time scaling was removed from the GUI. Keep kt = 1.0 (no effect) so any
# legacy profile that still carries the flag has no influence on Cp.
kt = 1.0

h_tool      = float(cfg_get(MODEL_CFG, "geometry.tool.geometry.h_tool",      0.3))
l_tool      = float(cfg_get(MODEL_CFG, "geometry.tool.geometry.l_tool",      0.5))
r_tool      = float(cfg_get(MODEL_CFG, "geometry.tool.geometry.r_tool",      0.01))
rake_angle  = float(cfg_get(MODEL_CFG, "geometry.tool.geometry.rake_angle",  40.0))
clear_angle = float(cfg_get(MODEL_CFG, "geometry.tool.geometry.clear_angle", 10.0))
tool_tx     = float(cfg_get(MODEL_CFG, "geometry.tool.position.x0", 0.0))
tool_ty     = float(cfg_get(MODEL_CFG, "geometry.tool.position.y0", -0.05))

egeom  = cfg_get(MODEL_CFG, "geometry.euler.geometry", {})
h_wp   = float(egeom.get("h_wp",   0.3))
h_void = float(egeom.get("h_void", 0.2))
l_wp   = float(egeom.get("l_wp",   0.5))
l_void = float(egeom.get("l_void", 0.2))
if bool(egeom.get("discretize", True)):
    h_wp   = discretize(h_wp,   elem_size)
    h_void = discretize(h_void, elem_size)
    l_wp   = discretize(l_wp,   elem_size)
    l_void = discretize(l_void, elem_size)

mesh_tx = float(cfg_get(MODEL_CFG, "geometry.euler.position.x0",            0.0))
mesh_ty = float(cfg_get(MODEL_CFG, "geometry.euler.position.y0",            0.0))
wp_tx   = float(cfg_get(MODEL_CFG, "geometry.euler.workpiece_position.x0",  0.0))
wp_ty   = float(cfg_get(MODEL_CFG, "geometry.euler.workpiece_position.y0",  0.0))

margin = elem_size / 2
xmin = float(cfg_get(MODEL_CFG, "geometry.bbox.xmin", -0.5))
xmax = float(cfg_get(MODEL_CFG, "geometry.bbox.xmax",  0.5))
ymin = float(cfg_get(MODEL_CFG, "geometry.bbox.ymin", -0.5))
ymax = float(cfg_get(MODEL_CFG, "geometry.bbox.ymax",  0.5))
zmin = float(cfg_get(MODEL_CFG, "geometry.bbox.zmin",  0.0))
zmax = float(cfg_get(MODEL_CFG, "geometry.bbox.zmax",  margin))

emat = cfg_get(MODEL_CFG, "materials.euler", {})
tmat = cfg_get(MODEL_CFG, "materials.tool",  {})

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
inter = cfg_get(MODEL_CFG, "interaction", {}) or {}
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
bcs                  = cfg_get(MODEL_CFG, "bcs", {}) or {}
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

# -----------------------------------------------------------------------------
# Element-type configurations (per body)
# -----------------------------------------------------------------------------
def _elem_cfg(path):
    """Return the dict for a body's element config, with sensible defaults."""
    d = cfg_get(MODEL_CFG, path, {}) or {}
    return {
        # common
        "thermally_coupled":     bool( d.get("thermally_coupled", True)),
        "second_order_accuracy": bool( d.get("second_order_accuracy", False)),
        "hourglass_control":     str(  d.get("hourglass_control", "default")),   # default | relax_stiffness | stiffness | viscous | combined
        "disp_scale":            float(d.get("displacement_hourglass_scale_factor",  1.0)),
        "linbv_scale":           float(d.get("linear_bulk_viscosity_scale_factor",   1.0)),
        "qbv_scale":             float(d.get("quadratic_bulk_viscosity_scale_factor", 1.0)),
        "svw_scale":             float(d.get("stiffness_viscous_weight_factor",       0.5)),
        # lagrangian only
        "reduced_integration":   bool( d.get("reduced_integration",   True)),
        "kinematic_split":       str(  d.get("kinematic_split",       "average_strain")),  # average_strain | orthogonal | centroid
        "distortion_control":    str(  d.get("distortion_control_mode", "use_default")),    # use_default | yes | no
        "length_ratio":          float(d.get("length_ratio", 0.1)),
        "element_deletion":      str(  d.get("element_deletion_mode", "use_default")),       # use_default | yes | no
        "max_deg_mode":          str(  d.get("max_degradation_mode", "use_default")),        # use_default | specify
        "max_deg_value":         float(d.get("max_degradation_value", 0.0)),
        "lkc_mode":              str(  d.get("linear_kinematic_conversion_mode", "use_default")),
        "lkc_value":             float(d.get("linear_kinematic_conversion_value", 0.0)),
    }

eul_cfg  = _elem_cfg("mesh.euler_element")
tool_cfg = _elem_cfg("mesh.tool_element")

job_name = cfg_get(RUN_CFG, "job_name", "default_name")
cpus     = int(cfg_get(RUN_CFG, "cpus",    4))
domains  = cfg_get(RUN_CFG, "domains", cpus)
# When True, build the model and write the .inp deck only — no solver run,
# no .odb, no results extraction. Lets the user inspect/keep the input file.
write_inp_only = bool(cfg_get(RUN_CFG, "write_inp_only", False))


# =============================================================================
# %% SYMBOLIC-CONSTANT MAPPING TABLES
# =============================================================================
# Translate GUI string identifiers into Abaqus symbolic constants. Centralised
# here so any future renaming on the GUI side only needs one lookup update.

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
# Hourglass-control formulations
_HOURGLASS_MAP = {
    "default":         DEFAULT,
    "relax_stiffness": RELAX_STIFFNESS,
    "stiffness":       STIFFNESS,
    "viscous":         VISCOUS,
    "combined":        COMBINED,
}
# Lagrangian kinematic splits (C3D8* solids)
_KINEMATIC_SPLIT_MAP = {
    "average_strain": AVERAGE_STRAIN,
    "orthogonal":     ORTHOGONAL,
    "centroid":       CENTROID,
}
# Three-state radio (Use default / Yes / No)
_USE_DEFAULT_YES_NO_MAP = {
    "use_default": DEFAULT,
    "yes":         ON,
    "no":          OFF,
}


def _build_elem_type_kwargs(cfg, family):
    """Build the kwargs dict for ElemType() based on a GUI element config.

    `family` is "eulerian" or "lagrangian". Returns a kwargs dict ready to
    be splatted into ElemType(...). The elemCode is set here too so the
    caller doesn't need to recompute it.

    Notes:
      - Eulerian family always uses EC3D8R or EC3D8RT (reduced integration
        is implicit in the Eulerian element catalog).
      - Lagrangian family uses C3D8T (full integration) or C3D8RT (reduced),
        ALWAYS thermally coupled (we never expose C3D8 / C3D8R).
    """
    if family == "eulerian":
        # EC3D8R / EC3D8RT depending on thermal toggle
        elem_code = EC3D8RT if cfg["thermally_coupled"] else EC3D8R
    else:
        # Lagrangian explicit family — always thermal
        elem_code = C3D8RT if cfg["reduced_integration"] else C3D8T

    kwargs = {
        "elemCode":           elem_code,
        "elemLibrary":        EXPLICIT,
        "secondOrderAccuracy": ON if cfg["second_order_accuracy"] else OFF,
        "hourglassControl":   _HOURGLASS_MAP.get(cfg["hourglass_control"], DEFAULT),
        # Scaling factors. Abaqus accepts these unconditionally; the active
        # ones depend on the hourglass formulation (the inactive ones are
        # ignored by the solver). Passing them always keeps the code simple.
        "displacementHourglassScaleFactor": cfg["disp_scale"],
        "linearBulkViscosityScaleFactor":   cfg["linbv_scale"],
        "quadBulkViscosityScaleFactor":     cfg["qbv_scale"],
        "stiffnessViscousWeightFactor":     cfg["svw_scale"],
    }

    if family == "lagrangian" and cfg["reduced_integration"]:
        # Kinematic split is only meaningful for reduced-integration elements.
        kwargs["kinematicSplit"] = _KINEMATIC_SPLIT_MAP.get(
            cfg["kinematic_split"], AVERAGE_STRAIN)

    if family == "lagrangian":
        # Distortion control (use_default / yes / no)
        kwargs["distortionControl"] = _USE_DEFAULT_YES_NO_MAP.get(
            cfg["distortion_control"], DEFAULT)
        if cfg["distortion_control"] == "yes":
            kwargs["lengthRatio"] = cfg["length_ratio"]
        # Element deletion (use_default / yes / no)
        kwargs["elemDeletion"] = _USE_DEFAULT_YES_NO_MAP.get(
            cfg["element_deletion"], DEFAULT)
        # Max degradation
        if cfg["max_deg_mode"] == "specify":
            kwargs["maxDegradation"] = cfg["max_deg_value"]
        # Linear kinematic conversion is exposed in the dialog as
        # `linearKinematicCtrl` in Abaqus's API in some versions; older
        # versions don't accept it. Wrap in try/except at call site rather
        # than here, so the rest of the kwargs apply even on old Abaqus.

    return kwargs


# =============================================================================
#%% MODEL CONSTRUCTION
# =============================================================================
# Temperature unit: °C. absoluteZero is set to -273.15 so the GUI's °C
# values map directly to Abaqus magnitudes without conversion.
myModel = mdb.Model(name=job_name, absoluteZero=-273.15)

#%%% Euler part
eul_sketch = myModel.ConstrainedSketch(name='eul_sketch', sheetSize=5)
eul_sketch.rectangle(point1=(-l_wp, -h_wp), point2=(l_void, h_void))
eulPart = myModel.Part(name='Euler', dimensionality=THREE_D, type=EULERIAN)
eulPart.BaseSolidExtrude(sketch=eul_sketch, depth=elem_size)

#%%% Workpiece part (for VolFraction)
wp_sketch = myModel.ConstrainedSketch(name='wp_sketch', sheetSize=5)
wp_sketch.rectangle(point1=(-l_wp, -h_wp), point2=(0, 0))
wpPart = myModel.Part(name='Workpiece', dimensionality=THREE_D, type=DEFORMABLE_BODY)
wpPart.BaseSolidExtrude(sketch=wp_sketch, depth=elem_size)

#%%% Tool part
tool_sketch = myModel.ConstrainedSketch(name="tool_sketch", sheetSize=5)
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
toolPart = myModel.Part(name="Tool", dimensionality=THREE_D, type=DEFORMABLE_BODY)
toolPart.BaseSolidExtrude(sketch=tool_sketch, depth=elem_size)

#%%% Materials
# Mass scaling for CEL is applied by scaling the Eulerian material density
# (Abaqus' native mass scaling does not apply to Eulerian EC3D8R elements).
# rho_eff = factor * rho ; Cp_eff = Cp / factor, so the volumetric heat
# capacity rho*Cp is preserved and the temperature stays physical. This
# matches the manual reference workflow, where BOTH rho and Cp are scaled
# by the same factor by hand.
# IMPORTANT: pass PHYSICAL rho/Cp in the config and let the factor below do
# the scaling ONCE. Do not pre-scale by hand AND set the factor, or the
# material would be scaled twice (rho*f^2) and the solve would diverge.
EulerMat = myModel.Material(name='Euler')
EulerMat.Density(table=((float(emat["rho"]) * ms_eul,),))
EulerMat.Elastic(table=((float(emat["E"]), float(emat["nu"])),))
EulerMat.Conductivity(table=((float(emat["k"]),),))
EulerMat.SpecificHeat(table=((float(emat["Cp"]) / (ms_eul * kt),),), law=CONSTANTPRESSURE)
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

ToolMat = myModel.Material(name='Tool')
ToolMat.Density(table=((float(tmat["rho"]) * ms_tool,),))
ToolMat.Elastic(table=((float(tmat["E"]), float(tmat["nu"])),))
ToolMat.Conductivity(table=((float(tmat["k"]),),))
ToolMat.SpecificHeat(table=((float(tmat["Cp"]) / (ms_tool * kt),),))
ToolMat.Expansion(table=((float(tmat["alpha"]),),))

#%%% Sections
myModel.EulerianSection(name='Euler', data={'euler-1': 'Euler'})
myModel.HomogeneousSolidSection(name='Tool', material='Tool')
eulPart.SectionAssignment(region=Region(cells=eulPart.cells), sectionName='Euler')
toolPart.SectionAssignment(region=Region(cells=toolPart.cells), sectionName='Tool')

#%%% Assembly
assembly = myModel.rootAssembly
eul_instance  = assembly.Instance(name='Euler',     part=eulPart,  dependent=OFF)
wp_instance   = assembly.Instance(name='Workpiece', part=wpPart,   dependent=OFF)
tool_instance = assembly.Instance(name='Tool',      part=toolPart, dependent=OFF)

assembly.translate(instanceList=('Tool',),      vector=(tool_tx, tool_ty, 0.0))
assembly.translate(instanceList=('Workpiece',), vector=(wp_tx,   wp_ty,   0.0))
assembly.translate(instanceList=('Euler',),     vector=(mesh_tx, mesh_ty, 0.0))

# wp_instance.translateTo(movableList=(wp_instance.faces[2], ),
#                         fixedList=(tool_instance.faces[6], ),
#                         direction=(1.0, 0.0, 0.0),
#                         clearance=0.0)
    
assembly.excludeFromSimulation(instances=(wp_instance,), exclude=True)

#%%% Mesh
assembly.seedPartInstance(regions=(eul_instance,), size=elem_size)
# Eulerian element type, fully driven by the GUI's Mesh > Element Type tab.
# Falls back gracefully if a particular kwarg isn't accepted by the running
# Abaqus version (e.g. older versions may not honour every scaling factor
# name) — we retry with a reduced kwargs dict in that case.
_eul_elem_kwargs = _build_elem_type_kwargs(eul_cfg, family="eulerian")
try:
    eul_elemType = ElemType(**_eul_elem_kwargs)
except TypeError:
    # Strip any kwarg the running Abaqus doesn't recognise.
    eul_elemType = ElemType(
        elemCode=_eul_elem_kwargs["elemCode"], elemLibrary=EXPLICIT)
eul_set = assembly.Set(name='Euler', cells=eul_instance.cells)
assembly.setElementType(regions=eul_set, elemTypes=(eul_elemType, eul_elemType, eul_elemType))
assembly.setMeshControls(regions=eul_instance.cells, elemShape=HEX, technique=STRUCTURED)

nose_mesh_edges = [13, 14]
# Tool-nose seed size, now configurable (was hard-coded 0.001) so the
# mesh-convergence study can drive it; falls back to 0.001 for old configs.
tool_elem_size = float(cfg_get(MODEL_CFG, "mesh.tool_elem_size", 0.005))
assembly.seedEdgeBySize(edges=[tool_instance.edges[i] for i in nose_mesh_edges],
                        size=tool_elem_size)
width_mesh_edge = [1]
assembly.seedEdgeByNumber(edges=[tool_instance.edges[i] for i in width_mesh_edge], number=1)

assembly.seedEdgeByBias(
    biasMethod=SINGLE,
    end1Edges=[tool_instance.edges[4],tool_instance.edges[6]],
    end2Edges=[tool_instance.edges[7],tool_instance.edges[9]],
    minSize=0.02,
    maxSize=0.05)

assembly.seedEdgeByBias(
    biasMethod=SINGLE,
    end1Edges=[tool_instance.edges[0],tool_instance.edges[2]],
    end2Edges=[tool_instance.edges[10],tool_instance.edges[12]],
    minSize=0.005,
    maxSize=0.02)

tool_elem_kwargs = _build_elem_type_kwargs(tool_cfg, family="lagrangian")
try:
    tool_elemType = ElemType(**tool_elem_kwargs)
except TypeError:
    tool_elemType = ElemType(
        elemCode=tool_elem_kwargs["elemCode"], elemLibrary=EXPLICIT)
tool_set = assembly.Set(name='Tool', cells=tool_instance.cells)
assembly.setElementType(regions=tool_set, elemTypes=(tool_elemType,))
assembly.setMeshControls(regions=tool_instance.cells, elemShape=HEX, technique=SWEEP)

assembly.generateMesh(regions=(eul_instance, tool_instance))

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
assembly.Set(name='ROI', nodes=roi_nodes)

# ZOI (zone of interest) sets used by extraction: one element set and one
# node set covering the user's ROI on the z = 0 face. They are also created
# here so they appear in the model/.inp; extraction re-derives the labels
# from the ODB and stops with an error if either is empty.
roi_elems = eul_instance.elements.getByBoundingBox(
    xMin=xmin - margin, xMax=xmax + margin,
    yMin=ymin - margin, yMax=ymax + margin,
    zMin=zmin - margin, zMax=zmax + margin
)
assembly.Set(name='ZOI_nodes', nodes=roi_nodes)
assembly.Set(name='ZOI_elems', elements=roi_elems)

#%%% Contact
# All knobs (tangential formulation, friction coefficient, slip tolerance,
# pressure-overclosure law, heat generation + fractions) come from the GUI's
# Interaction tab.
IntProp = myModel.ContactProperty(name='IntProp')

# Tangential behavior
_tang_kind = _TANGENTIAL_MAP.get(inter_tangential, PENALTY)
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
    pressureOverclosure=_PRESSURE_OVERCLOSURE_MAP.get(inter_pressure, HARD),
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

myModel.ContactExp(name='Contact', createStepName='Initial',
                   contactPropertyAssignments=((GLOBAL, SELF, 'IntProp'),))
myModel.RigidBody(name='Rigid_Tool', refPointRegion=RP, bodyRegion=tool_elem)

#%%% Step
myModel.TempDisplacementDynamicsStep(name='Cut', previous='Initial',
                                      timePeriod=sim_time, nlgeom=ON,
                                      improvedDtMethod=ON)

# Field output: only the variables ticked in the GUI's Step tab.
# `fo_variables` is built above from `step.output.fo_*` flags. If the user
# unticked everything, we fall back to a minimal set so the Abaqus solver
# doesn't choke (an empty variables tuple is invalid).
if not fo_variables:
    fo_variables = ('S', 'U')
myModel.FieldOutputRequest(
    name='F-Output-1', createStepName='Cut',
    variables=fo_variables, numIntervals=n_frames,
)

# History output: optional RF on the tool RP + optional PRESELECT.
# Both use numIntervals (number of equally-spaced samples) rather than
# timeInterval (seconds between samples). When the user keeps the
# "Sync with field output" toggle on in the Step tab, ho_n_intervals
# equals n_frames, so each history sample lines up 1:1 with a field
# frame — directly usable to overlay forces against PEEQ/TEMP in the
# Results tab without interpolation.
if ho_rf_on_rp:
    myModel.HistoryOutputRequest(
        name='H-Output-1', createStepName='Cut',
        region=RP, variables=('RF1', 'RF2',),
        numIntervals=ho_n_intervals,
    )
if ho_preselect:
    myModel.HistoryOutputRequest(
        name='H-Output-2', createStepName='Cut',
        variables=PRESELECT,
        numIntervals=ho_n_intervals,
    )
# ALLKE/ALLIE are already produced by the PRESELECT whole-model history
# (ho_preselect); no dedicated energy request is needed. The mass-scaling guard
# extracts them from the PRESELECT history region.
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
    # ============================================================ #
    # !!! VERIFY ON FIRST RUN !!!                                   #
    # If `eul_face_surfaces[_f]` is missing or contains the wrong   #
    # face for one of left/right/top/bottom, check the bbox above.  #
    # The tolerance `_eps` may need adjusting, or Abaqus may have   #
    # split/merged the face during meshing.                         #
    # ============================================================ #

#%%%% Eulerian inflow / outflow BCs
# One BC per enabled face. The `definition` argument controls which fluxes
# are specified (INFLOW / OUTFLOW / BOTH) and we always pass both
# inflowType and outflowType — Abaqus ignores the one not selected by
# `definition`.
for _f in EUL_FACES:
    _c = eulbc_cfg[_f]
    if not _c["enabled"] or _f not in eul_face_surfaces:
        continue
    myModel.EulerianBC(
        name='eulbc_' + _f,
        createStepName='Cut',
        region=eul_face_surfaces[_f],
        definition=_BC_DEFINITION_MAP.get(_c["mode"], BOTH),
        inflowType=_INFLOW_MAP.get(_c["inflow"],   FREE),
        outflowType=_OUTFLOW_MAP.get(_c["outflow"], FREE),
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

if _vcut_face_array is not None and len(_vcut_face_array) > 0:
    work_sides = assembly.Set(
        name='work_sides',
        faces=_vcut_face_array,
    )
    cut_BC = myModel.VelocityBC(
        name='Cutting_speed', createStepName='Initial',
        region=work_sides, v1=SET,
    )
    cut_BC.setValuesInStep(stepName='Cut', v1=bcs_cutting_speed)
else:
    # No cutting face selected — create no velocity BC. The user will see
    # this in the .msg log; it's a valid (if unusual) configuration.
    cut_BC = None

#%%%% Tool fixity + plane strain on the Eulerian mesh
myModel.VelocityBC(name='Tool_fix', createStepName='Initial', region=RP,
                   v1=SET, v2=SET, v3=SET, vr1=SET, vr2=SET, vr3=SET)
myModel.VelocityBC(name='Plane_strain', createStepName='Initial',
                   region=eul_set, v3=SET)

#%%%% Initial Eulerian velocity (predefined field)
# Independent from the cutting-velocity BC — set by `bcs.initial_velocity`.
myModel.Velocity(
    name='Init_speed',
    region=eul_set, field='',
    distributionType=MAGNITUDE,
    velocity1=bcs_initial_velocity,
)

# myModel.Stress(name='Init_stress', region=eul_set, distributionType=UNIFORM,
#                sigma11=0.0, sigma22=0.0, sigma33=0.0, sigma12=0.0, sigma13=0.0, sigma23=0.0)

rgn = regionToolset.Region(cells=eul_instance.cells)
myModel.MaterialAssignment(name='Material', instanceList=(eul_instance,),
                           useFields=True, fieldList=((rgn, ("VolFraction",)),))

#%%%% Initial temperature (both bodies)
# Value comes from bcs.ambient_temperature, in °C (absoluteZero=-273.15
# was set on the Model, so a 20 °C value here means magnitudes=20).
myModel.Temperature(name='Init_temp',
                    createStepName='Initial', region=eul_set,
                    magnitudes=bcs_ambient_temp)
myModel.Temperature(name='Init_temp_tool',
                    createStepName='Initial', region=tool_set,
                    magnitudes=bcs_ambient_temp)
# =============================================================================
#%% JOB SUBMISSION
# =============================================================================
myJob = mdb.Job(
    name=job_name,
    model=myModel.name,
    type=ANALYSIS,
    numCpus=cpus,
    numDomains=domains,
    explicitPrecision=DOUBLE,
    nodalOutputPrecision=FULL,
)
# Run the solver. submit() writes the .inp itself before launching the
# analysis, so we don't call writeInput() separately. waitForCompletion()
# blocks until the explicit solver subprocess is fully done — this is
# critical: it ensures the .odb on disk is complete and ready to read
# by the extraction block below.
#
# We deliberately print just two short markers around the solver call:
# the per-increment chatter Abaqus dumps to stdout in interactive mode
# is too noisy for a long Explicit run (hundreds of thousands of incs).
# The GUI tracks progress by polling the .sta file written by the solver,
# which has one summary row per output frame — exactly what we need.
print("[META] job_name=%s" % job_name)
print("[META] sim_time=%g" % sim_time)
print("[META] n_frames=%d" % n_frames)
if write_inp_only:
    # Write the .inp deck and stop — no analysis, no .odb, no extraction.
    print("[STAGE] WRITE_INP_START")
    sys.stdout.flush()
    myJob.writeInput(consistencyChecking=OFF)
    print("[STAGE] INP_WRITTEN")
    print("[OK] Wrote %s.inp (no analysis run)." % job_name)
    sys.stdout.flush()
    sys.exit(0)
print("[STAGE] SOLVE_START")
sys.stdout.flush()
myJob.submit(consistencyChecking=OFF)
myJob.waitForCompletion()
print("[STAGE] SOLVE_DONE")
sys.stdout.flush()


# =============================================================================
#%% RESULTS EXTRACTION
# =============================================================================
# Open the .odb just produced by myJob.submit() and dump a (.json + .npz)
# bundle following gui/results/FORMAT.md. By running this step *inside*
# the same Abaqus python invocation we get two guarantees that the
# previous standalone extract_odb.py couldn't offer:
#   1. The .odb is complete: waitForCompletion() above blocked until
#      every frame was written. A separate process spawned afterwards
#      could (and did) race against the still-running solver in some
#      edge cases.
#   2. We always have access to MODEL_CFG (already parsed above), so
#      the ROI and the cfg snapshot embed naturally without any side
#      file gymnastics.

print("\n" + "=" * 72)
print("[STAGE] EXTRACT_START")
print("EXTRACTING results from %s.odb" % job_name)
print("=" * 72)
sys.stdout.flush()

import json as _json
from odbAccess import openOdb as _openOdb
from abaqusConstants import CENTROID as _CENTROID
import numpy as _np


def _vprint(msg):
    """stdout-and-flush — so the GUI's output panel streams progress."""
    print(msg)
    sys.stdout.flush()


def _bbox_of_array(arr):
    """Return (min, max) tuples for a (N, 3) numpy array."""
    return arr.min(axis=0), arr.max(axis=0)


def _resolve_roi():
    """Read bbox from MODEL_CFG. Return a dict {xmin,xmax,...} or None
    if degenerate (the ZOI/ROI is the bbox of the user's region of
    interest, used to crop the extracted fields)."""
    bb = cfg_get(MODEL_CFG, "geometry.bbox", {}) or {}
    try:
        xmin = float(bb.get("xmin", 0.0)); xmax = float(bb.get("xmax", 0.0))
        ymin = float(bb.get("ymin", 0.0)); ymax = float(bb.get("ymax", 0.0))
        zmin = float(bb.get("zmin", 0.0)); zmax = float(bb.get("zmax", 0.0))
    except (TypeError, ValueError):
        return None
    # An ROI only filters when it has POSITIVE in-plane extent (x AND y).
    # The model is a thin plane-strain slab, so the default bbox only
    # encodes a tiny z-thickness (e.g. {0,0,0,0,0,1e-4}). The old test
    # ("all six == 0 -> keep all") let that default through as an *active*
    # ROI, which selected the empty region x==0,y==0 and dropped every
    # instance from the bundle (nothing to show in the Results tab).
    if xmax <= xmin or ymax <= ymin:
        return None
    return {"xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax}


def _in_bbox(c, roi):
    """Inclusive in-plane (x, y) bounding-box test. z is intentionally
    ignored: the model is a thin plane-strain slab, so filtering on the
    small slab thickness would exclude every element."""
    return (roi["xmin"] <= c[0] <= roi["xmax"]
            and roi["ymin"] <= c[1] <= roi["ymax"])


def _extract_instance_geometry(inst, roi):
    """Walk every element of an ODB instance, keep those whose initial
    centroid is in roi (or all if roi is None). Returns:
        nodes_init    (n_nodes, 3)  float32, kept nodes only, re-indexed
        elements      (n_elem, 8)   int32, 0-based connectivity into nodes_init
        centroids     (n_elem, 3)   float32, initial centroids of kept elements
        kept_node_ids list[int]     original Abaqus node labels
        kept_elem_ids list[int]     original Abaqus element labels
        full_bbox     tuple         ((xmin,ymin,zmin), (xmax,ymax,zmax)) of
                                    every element in this instance — printed
                                    as a debug hint so the user can sanity-check
                                    the ROI against the actual mesh extent.
    """
    n_total_nodes = len(inst.nodes)
    all_coords = _np.zeros((n_total_nodes, 3), dtype=_np.float32)
    label_to_idx = {}
    for i in range(n_total_nodes):
        node = inst.nodes[i]
        all_coords[i] = node.coordinates
        label_to_idx[node.label] = i

    n_total_elems = len(inst.elements)
    all_centroids = _np.zeros((n_total_elems, 3), dtype=_np.float32)
    elem_connectivities = []
    elem_labels = []
    elem_kinds = []  # for filtering
    for i in range(n_total_elems):
        elem = inst.elements[i]
        if elem.type not in ("EC3D8R", "EC3D8RT", "C3D8R", "C3D8RT",
                              "C3D8", "C3D8T"):
            elem_connectivities.append(None)
            elem_labels.append(elem.label)
            elem_kinds.append(elem.type)
            continue
        conn = [label_to_idx[lbl] for lbl in elem.connectivity]
        all_centroids[i] = all_coords[conn].mean(axis=0)
        elem_connectivities.append(conn)
        elem_labels.append(elem.label)
        elem_kinds.append(elem.type)

    # Compute the mesh's actual bbox (over all hex elements) so the user
    # can verify their ROI is in the right ballpark.
    valid_mask = _np.array(
        [conn is not None for conn in elem_connectivities], dtype=bool
    )
    if valid_mask.any():
        valid_centroids = all_centroids[valid_mask]
        full_bbox = (valid_centroids.min(axis=0), valid_centroids.max(axis=0))
    else:
        full_bbox = (_np.zeros(3), _np.zeros(3))

    # Now apply the ROI filter
    kept_elements = []
    kept_centroids = []
    kept_elem_ids = []
    for i, conn in enumerate(elem_connectivities):
        if conn is None:
            continue
        if roi is not None and not _in_bbox(all_centroids[i], roi):
            continue
        kept_elements.append(conn)
        kept_centroids.append(all_centroids[i])
        kept_elem_ids.append(elem_labels[i])

    if len(kept_elements) == 0:
        return (_np.zeros((0, 3), dtype=_np.float32),
                _np.zeros((0, 8), dtype=_np.int32),
                _np.zeros((0, 3), dtype=_np.float32),
                [], [], full_bbox)

    touched = set()
    for conn in kept_elements:
        for n in conn:
            touched.add(n)
    touched_sorted = sorted(touched)
    new_idx_of = {}
    for new_i, old_i in enumerate(touched_sorted):
        new_idx_of[old_i] = new_i

    nodes_init = all_coords[touched_sorted]
    elements_arr = _np.zeros((len(kept_elements), 8), dtype=_np.int32)
    for i, conn in enumerate(kept_elements):
        for j in range(8):
            elements_arr[i, j] = new_idx_of[conn[j]]
    centroids = _np.asarray(kept_centroids, dtype=_np.float32)

    idx_to_label = {}
    for lbl, idx in label_to_idx.items():
        idx_to_label[idx] = lbl
    kept_node_ids = [idx_to_label[old_i] for old_i in touched_sorted]

    return (nodes_init, elements_arr, centroids,
            kept_node_ids, kept_elem_ids, full_bbox)


def _reduce_VM(vals, comp_labels):
    """von Mises reduction of a stress tensor (missing comps treated 0)."""
    idx = dict(zip(comp_labels, range(len(comp_labels))))
    def col(name):
        return vals[:, idx[name]] if name in idx else 0.0
    s11, s22, s33 = col("S11"), col("S22"), col("S33")
    s12, s13, s23 = col("S12"), col("S13"), col("S23")
    return _np.sqrt(0.5 * (
        (s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
        + 6.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )).astype(_np.float32)


def _reduce_identity(vals, comp_labels):
    if vals.ndim == 2 and vals.shape[1] == 1:
        return vals[:, 0].astype(_np.float32)
    return vals.astype(_np.float32)


_TENSOR_REDUCERS = {
    "MISES": ("S", _reduce_VM),
    "S_VM":  ("S", _reduce_VM),   # backward-compatible alias
}

# Native stress invariants (preferred over recombining components).
_STRESS_INVARIANT = {"MISES": "MISES", "S_VM": "MISES", "S_P": "PRESS"}


def _read_data(v):
    """Read a field value, tolerant of double-precision ODBs (this model
    runs in double precision, so `value.data` raises and we fall back to
    `value.dataDouble`)."""
    try:
        return v.data
    except Exception:
        return v.dataDouble


def _resolve_fo_name(step, abq_var, inst_name, root_assembly, max_probe=3):
    """Find the real fieldOutputs key for `abq_var` on `inst_name`.

    Abaqus/CEL stores per-material results on an Eulerian instance under a
    suffixed name (PEEQ -> 'PEEQ_ASSEMBLY_EULER_EULER-1', S -> 'S_...',
    TEMP -> 'TEMP_...', EVF -> 'EVF_...' material + 'EVF_VOID'), while the
    bare names may carry only the Lagrangian tool's values. Pick the
    candidate that actually has values on the target instance: exact name,
    then a material-suffixed name, then '*_VOID' as last resort. Returns
    the key or None.
    """
    inst = root_assembly.instances[inst_name]
    n = len(step.frames)
    probe = [0]
    if n > 2:
        probe.append(n // 2)
    probe.append(n - 1)

    def _has_inst_values(fo):
        try:
            return len(fo.getSubset(region=inst).values) > 0
        except Exception:
            return False

    for fi in probe[:max_probe]:
        keys = list(step.frames[fi].fieldOutputs.keys())
        exact     = [k for k in keys if k == abq_var]
        suffixed  = [k for k in keys if k.startswith(abq_var + "_")]
        suff_mat  = [k for k in suffixed if not k.endswith("_VOID")]
        suff_void = [k for k in suffixed if k.endswith("_VOID")]
        for k in exact:
            if _has_inst_values(step.frames[fi].fieldOutputs[k]):
                return k
        # CEL material-suffixed names referencing THIS instance: trust name
        # (getSubset/.values are unreliable for tensor fields like S).
        for k in (suff_mat + suff_void):
            if inst_name in k:
                return k
        for k in (suff_mat + suff_void):
            if _has_inst_values(step.frames[fi].fieldOutputs[k]):
                return k
    return None


def _extract_field(step, var, inst_name, kept_elem_ids, root_assembly):
    if var in _TENSOR_REDUCERS:
        abq_var, reducer = _TENSOR_REDUCERS[var]
    else:
        abq_var, reducer = var, _reduce_identity
    inv_name = _STRESS_INVARIANT.get(var)

    key = _resolve_fo_name(step, abq_var, inst_name, root_assembly)
    if key is None:
        # Not present on this instance (e.g. PEEQ/S/EVF on the rigid tool).
        raise KeyError(abq_var)

    kept_set = set(kept_elem_ids)
    elem_id_to_pos = {}
    for pos, lbl in enumerate(kept_elem_ids):
        elem_id_to_pos[lbl] = pos
    n_elems = len(kept_elem_ids)
    n_frames = len(step.frames)
    out = _np.zeros((n_frames, n_elems), dtype=_np.float32)

    invariant = None
    if inv_name is not None:
        try:
            import abaqusConstants as _abqc
            invariant = getattr(_abqc, inv_name, None)
        except Exception:
            invariant = None

    inst = root_assembly.instances[inst_name]
    # Decide once whether restricting to the instance yields values; some
    # CEL tensor fields (S) don't subset by instance cleanly, so fall back
    # to the full field + element-label filtering (the suffixed field only
    # carries this instance's elements anyway).
    use_region = False
    probe_fi = (n_frames // 2) if n_frames > 1 else 0
    try:
        if len(step.frames[probe_fi].fieldOutputs[key].getSubset(region=inst).values) > 0:
            use_region = True
    except Exception:
        use_region = False

    for fi in range(n_frames):
        try:
            fo = step.frames[fi].fieldOutputs[key]
        except KeyError:
            continue
        if use_region:
            try:
                fo = fo.getSubset(region=inst)
            except (AttributeError, KeyError):
                pass
        # von Mises / pressure: prefer the native scalar invariant.
        src = fo
        used_invariant = False
        if invariant is not None:
            try:
                src = fo.getScalarField(invariant=invariant)
                used_invariant = True
            except Exception:
                src, used_invariant = fo, False
        if not used_invariant:
            try:
                src = src.getSubset(position=_CENTROID)
            except Exception:
                pass
        comp_labels = [] if used_invariant else (
            list(src.componentLabels) if src.componentLabels else [])
        vals_list = []
        labels_list = []
        for v in src.values:
            lbl = v.elementLabel
            if lbl in kept_set:
                if comp_labels:
                    vals_list.append(list(_read_data(v)))
                else:
                    vals_list.append([float(_read_data(v))])
                labels_list.append(lbl)
        if not vals_list:
            continue
        vals = _np.asarray(vals_list, dtype=_np.float32)
        if comp_labels:
            scalars = reducer(vals, comp_labels)
        else:
            scalars = _reduce_identity(vals, comp_labels)
        for k, lbl in enumerate(labels_list):
            pos = elem_id_to_pos.get(lbl)
            if pos is not None:
                out[fi, pos] = scalars[k]
    return out


def _extract_displacements(step, inst_name, kept_node_ids, root_assembly):
    kept_set = set(kept_node_ids)
    node_id_to_pos = {}
    for pos, lbl in enumerate(kept_node_ids):
        node_id_to_pos[lbl] = pos
    n_nodes = len(kept_node_ids)
    n_frames = len(step.frames)
    out = _np.zeros((n_frames, n_nodes, 3), dtype=_np.float32)
    for fi in range(n_frames):
        try:
            fo = step.frames[fi].fieldOutputs["U"]
        except KeyError:
            continue
        try:
            fo = fo.getSubset(region=root_assembly.instances[inst_name])
        except (AttributeError, KeyError):
            pass
        for v in fo.values:
            lbl = v.nodeLabel
            if lbl in kept_set:
                pos = node_id_to_pos[lbl]
                d = _read_data(v)
                out[fi, pos, 0] = d[0]
                out[fi, pos, 1] = d[1]
                out[fi, pos, 2] = d[2] if len(d) > 2 else 0.0
    return out


_NODAL_VECTOR_VARS = ("V",)


def _extract_nodal_vector_to_elem(step, var, inst_name, kept_node_ids,
                                  elements, root_assembly):
    """Read a NODAL vector field (e.g. V); return per-element fields
    {var+'1': Vx, var+'2': Vy, var: magnitude}, each (n_frames, n_elem),
    by averaging each signed component over an element's nodes. Raises
    KeyError if absent on the instance."""
    try:
        from abaqusConstants import NODAL
    except Exception:
        NODAL = None
    inst = root_assembly.instances[inst_name]
    key = _resolve_fo_name(step, var, inst_name, root_assembly)
    if key is None:
        raise KeyError(var)
    label_to_local = {}
    for i, lbl in enumerate(kept_node_ids):
        label_to_local[int(lbl)] = i
    n_nodes = len(kept_node_ids)
    elements = _np.asarray(elements)
    n_elem = elements.shape[0]
    n_frames = len(step.frames)
    v1_out = _np.zeros((n_frames, n_elem), dtype=_np.float32)
    v2_out = _np.zeros((n_frames, n_elem), dtype=_np.float32)
    for fi in range(n_frames):
        try:
            fo = step.frames[fi].fieldOutputs[key]
        except KeyError:
            continue
        sub = fo
        try:
            sub = fo.getSubset(region=inst)
        except (AttributeError, KeyError):
            pass
        if NODAL is not None:
            try:
                sub = sub.getSubset(position=NODAL)
            except Exception:
                pass
        nodal_v1 = _np.zeros(n_nodes, dtype=_np.float32)
        nodal_v2 = _np.zeros(n_nodes, dtype=_np.float32)
        for v in sub.values:
            j = label_to_local.get(int(v.nodeLabel))
            if j is not None:
                d = _read_data(v)
                nodal_v1[j] = d[0]
                nodal_v2[j] = d[1] if len(d) > 1 else 0.0
        v1_out[fi] = nodal_v1[elements].mean(axis=1)
        v2_out[fi] = nodal_v2[elements].mean(axis=1)
    mag = _np.sqrt(v1_out * v1_out + v2_out * v2_out).astype(_np.float32)
    return {var + "1": v1_out, var + "2": v2_out, var: mag}


def _extract_history_rf(step):
    """Return (time, rf1, rf2) or (None, None, None)."""
    for region_key, region in step.historyRegions.items():
        outputs = region.historyOutputs
        if "RF1" in outputs and "RF2" in outputs:
            rf1_pairs = outputs["RF1"].data
            rf2_pairs = outputs["RF2"].data
            t = _np.asarray([p[0] for p in rf1_pairs], dtype=_np.float64)
            rf1 = _np.asarray([p[1] for p in rf1_pairs], dtype=_np.float32)
            rf2 = _np.asarray([p[1] for p in rf2_pairs], dtype=_np.float32)
            return t, rf1, rf2
    return None, None, None


def _extract_history_energy(step):
    """Return (time, allke, allie) for the whole-model energies, or
    (None, None, None). ALLKE/ALLIE live in the whole-model / assembly history
    region (no element/RP region)."""
    for region_key, region in step.historyRegions.items():
        outputs = region.historyOutputs
        if "ALLKE" in outputs and "ALLIE" in outputs:
            ke_pairs = outputs["ALLKE"].data
            ie_pairs = outputs["ALLIE"].data
            t = _np.asarray([p[0] for p in ke_pairs], dtype=_np.float64)
            allke = _np.asarray([p[1] for p in ke_pairs], dtype=_np.float32)
            allie = _np.asarray([p[1] for p in ie_pairs], dtype=_np.float32)
            return t, allke, allie
    return None, None, None


# --- The extraction itself ---
_roi = _resolve_roi()
if _roi is None:
    _vprint("ROI: none (keeping all elements)")
else:
    _vprint("ROI: x[%g,%g] y[%g,%g] z[%g,%g]" % (
        _roi["xmin"], _roi["xmax"],
        _roi["ymin"], _roi["ymax"],
        _roi["zmin"], _roi["zmax"]))

_field_vars = ["EVF", "TEMP", "V"]
_vprint("Fields requested: " + ", ".join(_field_vars))

_odb_path = job_name + ".odb"
_vprint("Opening ODB: " + _odb_path)
_odb = _openOdb(_odb_path, readOnly=True)
try:
    _step = _odb.steps["Cut"]
    _frames = _step.frames
    _n_frames = len(_frames)
    _times = _np.asarray([fr.frameValue for fr in _frames], dtype=_np.float64)
    _vprint("Step 'Cut' has %d frames, t in [%g, %g]"
            % (_n_frames, _times[0], _times[-1]))

    _npz_payload = {"times": _times}
    _instances_meta = {}

    for _inst_name in _odb.rootAssembly.instances.keys():
        _inst = _odb.rootAssembly.instances[_inst_name]
        _n_elem_total = len(_inst.elements)
        if _n_elem_total == 0:
            continue
        _vprint("\nInstance %s: %d nodes, %d elements"
                % (_inst_name, len(_inst.nodes), _n_elem_total))

        _elem_type = _inst.elements[0].type
        _kind = "eulerian" if _elem_type.startswith("EC") else "lagrangian"

        # The ROI/ZOI applies ONLY to the Eulerian instance (cutting zone).
        # Lagrangian instances (e.g. the TOOL) are always kept whole.
        _inst_roi = _roi if _kind == "eulerian" else None
        (_nodes_init, _elements, _centroids,
         _kept_node_ids, _kept_elem_ids, _full_bbox) = \
            _extract_instance_geometry(_inst, _inst_roi)

        # Print the full mesh bbox — invaluable when the ROI filter
        # rejects everything, since it tells the user whether the
        # ROI numbers are in the right unit/range.
        _vprint("  mesh bbox: x[%g,%g] y[%g,%g] z[%g,%g]"
                % (_full_bbox[0][0], _full_bbox[1][0],
                   _full_bbox[0][1], _full_bbox[1][1],
                   _full_bbox[0][2], _full_bbox[1][2]))
        _n_kept_elem = _elements.shape[0]
        _n_kept_node = _nodes_init.shape[0]
        _vprint("  ZOI after ROI (%s): ZOI_nodes=%d, ZOI_elems=%d"
                % (_kind, _n_kept_node, _n_kept_elem))

        # The ROI defines the ZOI sets on the Eulerian (cutting) instance.
        # If the ROI selects nothing there, extraction cannot proceed: tell
        # the user to size a larger ROI and stop with a non-zero exit code.
        if _kind == "eulerian" and (_n_kept_elem == 0 or _n_kept_node == 0):
            _vprint("")
            _vprint("[ERROR] The ROI selects an empty zone of interest on the "
                    "Eulerian instance")
            _vprint("        (ZOI_nodes=%d, ZOI_elems=%d)."
                    % (_n_kept_node, _n_kept_elem))
            _vprint("        Mesh bbox is x[%g,%g] y[%g,%g]; your ROI is "
                    "x[%g,%g] y[%g,%g]."
                    % (_full_bbox[0][0], _full_bbox[1][0],
                       _full_bbox[0][1], _full_bbox[1][1],
                       (_roi or {}).get("xmin", 0.0), (_roi or {}).get("xmax", 0.0),
                       (_roi or {}).get("ymin", 0.0), (_roi or {}).get("ymax", 0.0)))
            _vprint("        Increase the ROI (Geometry tab) so it overlaps "
                    "the mesh, then re-run.")
            sys.stdout.flush()
            _odb.close()
            sys.exit(2)
        if _n_kept_elem == 0:
            continue

        _npz_payload["%s__nodes_init" % _inst_name] = _nodes_init
        _npz_payload["%s__elements" % _inst_name] = _elements
        _npz_payload["%s__element_centroids_init" % _inst_name] = _centroids

        _stored_vars = []
        for _var in _field_vars:
            _vprint("  field '%s'..." % _var)
            if _var in _NODAL_VECTOR_VARS:
                if _kind != "eulerian":
                    _vprint("    nodal field, skipping on this instance.")
                    continue
                try:
                    _vf = _extract_nodal_vector_to_elem(
                        _step, _var, _inst_name, _kept_node_ids,
                        _elements, _odb.rootAssembly)
                except KeyError:
                    _vprint("    not available, skipping.")
                    continue
                for _vk in (_var + "1", _var + "2", _var):
                    _npz_payload["%s__fields__%s" % (_inst_name, _vk)] = _vf[_vk]
                    _stored_vars.append(_vk)
                continue
            try:
                _arr = _extract_field(_step, _var, _inst_name,
                                       _kept_elem_ids, _odb.rootAssembly)
            except KeyError:
                _vprint("    not available, skipping.")
                continue
            _npz_payload["%s__fields__%s" % (_inst_name, _var)] = _arr
            _stored_vars.append(_var)

        _has_disp = False
        if _kind == "lagrangian":
            try:
                _disp = _extract_displacements(_step, _inst_name,
                                                _kept_node_ids, _odb.rootAssembly)
                _npz_payload["%s__displacements" % _inst_name] = _disp
                _has_disp = True
                _vprint("  displacements stored.")
            except Exception as _e:
                _vprint("  displacement extraction failed: %s" % _e)

        _instances_meta[_inst_name] = {
            "kind":              _kind,
            "element_type":      _elem_type,
            "n_nodes":           int(_n_kept_node),
            "n_elements":        int(_n_kept_elem),
            "n_frames":          int(_n_frames),
            "field_variables":   _stored_vars,
            "has_displacements": _has_disp,
        }

    _vprint("\nExtracting history...")
    _h_t, _rf1, _rf2 = _extract_history_rf(_step)
    _history_vars = []
    if _h_t is not None:
        _npz_payload["history__time"] = _h_t
        _npz_payload["history__RF1_RP"] = _rf1
        _npz_payload["history__RF2_RP"] = _rf2
        _history_vars = ["RF1_RP", "RF2_RP"]
        _vprint("  history: %d samples, RF1/RF2 stored" % len(_h_t))
    else:
        _vprint("  no RP history found.")

    _he_t, _allke, _allie = _extract_history_energy(_step)
    if _he_t is not None:
        if "history__time" not in _npz_payload:
            _npz_payload["history__time"] = _he_t
        _npz_payload["history__ALLKE"] = _allke
        _npz_payload["history__ALLIE"] = _allie
        _history_vars = _history_vars + ["ALLKE", "ALLIE"]
        _vprint("  history: %d samples, ALLKE/ALLIE stored" % len(_he_t))
    else:
        _vprint("  no energy history found.")

    # Metadata
    from datetime import datetime as _datetime
    _meta = {
        "format_version": 1,
        "saved_at":       _datetime.now().isoformat(),
        "source_odb":     os.path.abspath(_odb_path),
        "job_name":       job_name,
        "step_name":      "Cut",
        "times":          _times.tolist(),
        "roi": {
            "applied": _roi is not None,
            "xmin": (_roi["xmin"] if _roi else 0.0),
            "xmax": (_roi["xmax"] if _roi else 0.0),
            "ymin": (_roi["ymin"] if _roi else 0.0),
            "ymax": (_roi["ymax"] if _roi else 0.0),
            "zmin": (_roi["zmin"] if _roi else 0.0),
            "zmax": (_roi["zmax"] if _roi else 0.0),
        },
        "model_config": MODEL_CFG,
        "instances":    _instances_meta,
        "history": {
            "n_samples": int(len(_h_t)) if _h_t is not None else 0,
            "variables": _history_vars,
        },
    }

    _out_npz = job_name + ".results.npz"
    _out_json = job_name + ".results.json"
    _vprint("\nWriting %s ..." % _out_npz)
    _np.savez_compressed(_out_npz, **_npz_payload)
    _vprint("Writing %s ..." % _out_json)
    _f = open(_out_json, "w")
    try:
        _json.dump(_meta, _f, indent=2)
    finally:
        _f.close()

    _vprint("\nDone. Results bundle ready:")
    _vprint("  " + os.path.abspath(_out_npz))
    _vprint("  " + os.path.abspath(_out_json))
finally:
    _odb.close()

print("[STAGE] EXTRACT_DONE")
sys.stdout.flush()
