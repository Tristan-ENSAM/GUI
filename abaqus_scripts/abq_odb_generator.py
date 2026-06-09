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
sim_time      = float(cfg_get(MODEL_CFG, "process.sim_time",      0.0001))
n_frames      = int(  cfg_get(MODEL_CFG, "process.n_frames",      1))

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

job_name = cfg_get(RUN_CFG, "job_name", "default_name")
cpus     = int(cfg_get(RUN_CFG, "cpus",    4))
domains  = cfg_get(RUN_CFG, "domains", cpus)


# =============================================================================
#%% MODEL CONSTRUCTION
# =============================================================================
myModel = mdb.Model(name=job_name, absoluteZero=0)
#-273.15 => °C

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
EulerMat = myModel.Material(name='Euler')
EulerMat.Density(table=((float(emat["rho"]),),))
EulerMat.Elastic(table=((float(emat["E"]), float(emat["nu"])),))
EulerMat.Conductivity(table=((float(emat["k"]),),))
EulerMat.SpecificHeat(table=((float(emat["Cp"]),),), law=CONSTANTPRESSURE)
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
ToolMat.Density(table=((float(tmat["rho"]),),))
ToolMat.Elastic(table=((float(tmat["E"]), float(tmat["nu"])),))
ToolMat.Conductivity(table=((float(tmat["k"]),),))
ToolMat.SpecificHeat(table=((float(tmat["Cp"]),),))
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
eul_elemType = ElemType(elemCode=EC3D8RT, elemLibrary=EXPLICIT,)
#, secondOrderAccuracy=ON, hourglassControl=VISCOUS
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

tool_elemType = ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT, kinematicSplit=ORTHOGONAL)
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
IntProp = myModel.ContactProperty(name='IntProp')
IntProp.TangentialBehavior(formulation=PENALTY, table=((0.3,),), fraction=0.005) 
IntProp.NormalBehavior(pressureOverclosure=HARD)
myModel.ContactExp(name='Contact', createStepName='Initial',
                   contactPropertyAssignments=((GLOBAL, SELF, 'IntProp'),))
myModel.RigidBody(name='Rigid_Tool', refPointRegion=RP, bodyRegion=tool_elem)

#%%% Step
myModel.TempDisplacementDynamicsStep(name='Cut', previous='Initial', timePeriod=sim_time, nlgeom=ON, improvedDtMethod=ON)
# myModel.FieldOutputRequest(name='F-Output-1', createStepName='Cut',
#                            variables=('S', 'ERV', 'U', 'V', 'NT', 'EVF', 'PEEQ', 'SDEG', 'DMICRT',), 
#                            numIntervals=n_frames,
#                            timeMarks=ON,
#                            position=NODES)
myModel.FieldOutputRequest(name='F-Output-1', createStepName='Cut', 
                           variables=('S', 'PEEQ', 'ERV', 'U', 'V', 'A', 'RF', 'P', 'HP', 'VP', 
                                      'CSTRESS', 'SDEG', 'DMICRT', 'NT', 'TEMP', 'HFL', 'MFL', 'EVF', 'SDV', 
                                      'STATUS',), numIntervals=n_frames)
# PRESELECT, ALL
#'CSTRESS', 'S', 'U', 'V', 'NT', 'PEEQ', 'EVF', 'SDEG'
myModel.HistoryOutputRequest(name='H-Output-1', createStepName='Cut',
                             region=RP, variables=('RF1', 'RF2',),
                             numIntervals=n_frames)

myModel.HistoryOutputRequest(name='H-Output-2', 
                              createStepName='Cut', variables=PRESELECT) #PRESELECT => ALLEN
#%%% Boundary conditions

#%%%% Eulerian boundaries
# inflow_surf  = assembly.Surface(name='inflow_surf',  side1Faces=eul_instance.faces[0:1])
# myModel.EulerianBC(name='left_flow',  createStepName='Cut', region=inflow_surf,
#                     definition=BOTH, inflowType=FREE, outflowType=NON_REFLECTING)

# slide_surf = assembly.Surface(name='slide_surf',  side1Faces=eul_instance.faces[3:6])
# myModel.EulerianBC(name='slide',  createStepName='Cut', region=slide_surf,
#                     definition=BOTH, inflowType=NONE, outflowType=NONE)

# outflow_surf = assembly.Surface(name='outflow_surf', side1Faces=eul_instance.faces[2:3])
# myModel.EulerianBC(name='right_flow', createStepName='Cut', region=outflow_surf,
#                     definition=BOTH, inflowType=FREE, outflowType=NON_REFLECTING)

#NON_REFLECTING, EQUILIBRIUM
#%%%% Velocity conditions
work_sides   = assembly.Set(name='work_sides',   faces= eul_instance.faces[0:1]+eul_instance.faces[3:4])
# eul_instance.faces[0:1]+eul_instance.faces[3:4]+eul_instance.faces[2:3]
cut_BC = myModel.VelocityBC(name='Cutting_speed', createStepName='Initial', region=work_sides, v1=SET, v2=SET)
#v1=SET, v2=SET, v3=SET, vr1=SET, vr2=SET, vr3=SET
myModel.VelocityBC(name='Tool_fix', createStepName='Initial', region=RP,
                   v1=SET, v2=SET, v3=SET, vr1=SET, vr2=SET, vr3=SET)
myModel.Velocity(name='Init_speed', region=eul_set, field='', 
                  distributionType=MAGNITUDE, velocity1=cutting_speed,)
myModel.VelocityBC(name='Plane_strain', createStepName='Initial', region=eul_set, v3=SET)
# vr1=SET, vr2=SET, vr3=SET => non pris en compte pour ces éléments par Abaqus

# myModel.Stress(name='Init_stress', region=eul_set, distributionType=UNIFORM,
#                sigma11=0.0, sigma22=0.0, sigma33=0.0, sigma12=0.0, sigma13=0.0, sigma23=0.0)

rgn = regionToolset.Region(cells=eul_instance.cells)
myModel.MaterialAssignment(name='Material', instanceList=(eul_instance,),
                           useFields=True, fieldList=((rgn, ("VolFraction",)),))

myModel.Temperature(name='Init_temp',  createStepName='Initial', region=eul_set,  magnitudes=300)
myModel.Temperature(name='Init_temp_tool', createStepName='Initial', region=tool_set, magnitudes=300)

cut_BC.setValuesInStep(stepName='Cut', v1=cutting_speed)
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
# myJob.writeInput()
myJob.submit(consistencyChecking=OFF)
myJob.waitForCompletion()
