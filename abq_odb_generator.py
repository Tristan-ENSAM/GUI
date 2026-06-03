# -*- coding: utf-8 -*-

import sys
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
elem_size = float(cfg_get(MODEL_CFG, "euler.elem_size", 0.01))

cutting_speed = float(cfg_get(MODEL_CFG, "process.cutting_speed", 1000.0))
# `sim_time` and `n_frames` are now owned by the Step tab. They are
# mirrored into `process` for backwards compatibility, but the preferred
# read path is `step`. Try step first, fall back to process.
sim_time      = float(cfg_get(MODEL_CFG, "step.sim_time",
                              cfg_get(MODEL_CFG, "process.sim_time",  0.0001)))
n_frames      = int(  cfg_get(MODEL_CFG, "step.n_frames",
                              cfg_get(MODEL_CFG, "process.n_frames",  1)))

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
# When enabled: rho_eff = factor * rho ; Cp_eff = Cp / factor.
# Both factors default to 1.0 (no scaling) and are applied directly at
# material-write time below (see emat / tmat handling).
ms_enabled = bool(cfg_get(MODEL_CFG, "step.mass_scaling_enabled", False))
ms_eul     = float(cfg_get(MODEL_CFG, "step.mass_scaling_factor_eulerian", 1.0))
ms_tool    = float(cfg_get(MODEL_CFG, "step.mass_scaling_factor_tool",     1.0))
if not ms_enabled:
    ms_eul  = 1.0
    ms_tool = 1.0

h_tool      = float(cfg_get(MODEL_CFG, "tool.geometry.h_tool",      0.3))
l_tool      = float(cfg_get(MODEL_CFG, "tool.geometry.l_tool",      0.5))
r_tool      = float(cfg_get(MODEL_CFG, "tool.geometry.r_tool",      0.01))
rake_angle  = float(cfg_get(MODEL_CFG, "tool.geometry.rake_angle",  40.0))
clear_angle = float(cfg_get(MODEL_CFG, "tool.geometry.clear_angle", 10.0))
tool_tx     = float(cfg_get(MODEL_CFG, "tool.position.x0", 0.0))
tool_ty     = float(cfg_get(MODEL_CFG, "tool.position.y0", -0.05))

egeom  = cfg_get(MODEL_CFG, "euler.geometry", {})
h_wp   = float(egeom.get("h_wp",   0.3))
h_void = float(egeom.get("h_void", 0.2))
l_wp   = float(egeom.get("l_wp",   0.5))
l_void = float(egeom.get("l_void", 0.2))
if bool(egeom.get("discretize", True)):
    h_wp   = discretize(h_wp,   elem_size)
    h_void = discretize(h_void, elem_size)
    l_wp   = discretize(l_wp,   elem_size)
    l_void = discretize(l_void, elem_size)

mesh_tx = float(cfg_get(MODEL_CFG, "euler.position.x0",            0.0))
mesh_ty = float(cfg_get(MODEL_CFG, "euler.position.y0",            0.0))
wp_tx   = float(cfg_get(MODEL_CFG, "euler.workpiece_position.x0",  0.0))
wp_ty   = float(cfg_get(MODEL_CFG, "euler.workpiece_position.y0",  0.0))

margin = elem_size / 2
xmin = float(cfg_get(MODEL_CFG, "bbox.xmin", -0.5))
xmax = float(cfg_get(MODEL_CFG, "bbox.xmax",  0.5))
ymin = float(cfg_get(MODEL_CFG, "bbox.ymin", -0.5))
ymax = float(cfg_get(MODEL_CFG, "bbox.ymax",  0.5))
zmin = float(cfg_get(MODEL_CFG, "bbox.zmin",  0.0))
zmax = float(cfg_get(MODEL_CFG, "bbox.zmax",  margin))

emat = cfg_get(MODEL_CFG, "euler.material", {})
tmat = cfg_get(MODEL_CFG, "tool.material",  {})

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

eul_cfg  = _elem_cfg("euler.element")
tool_cfg = _elem_cfg("tool.element")

job_name = cfg_get(RUN_CFG, "job_name", "default_name")
cpus     = int(cfg_get(RUN_CFG, "cpus",    4))
domains  = cfg_get(RUN_CFG, "domains", cpus)


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
# Mass scaling (CEL): rho_eff = factor * rho ; Cp_eff = Cp / factor.
# When ms_enabled is False the factors above were forced to 1.0, so the
# arithmetic below is a no-op and the materials are unchanged.
EulerMat = myModel.Material(name='Euler')
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

ToolMat = myModel.Material(name='Tool')
ToolMat.Density(table=((float(tmat["rho"]) * ms_tool,),))
ToolMat.Elastic(table=((float(tmat["E"]), float(tmat["nu"])),))
ToolMat.Conductivity(table=((float(tmat["k"]),),))
ToolMat.SpecificHeat(table=((float(tmat["Cp"]) / ms_tool,),))
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
assembly.seedEdgeBySize(edges=[tool_instance.edges[i] for i in nose_mesh_edges], size=0.001)
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
    minSize=0.001,
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
        region=work_sides, v1=SET, v2=SET,
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
myJob.writeInput()
# myJob.submit(consistencyChecking=OFF)
# myJob.waitForCompletion()
