# -*- coding: utf-8 -*-
"""Abaqus API introspection script for the GUI_Abaqus reliability review
(see _review/REVIEW.md).

WHY THIS SCRIPT EXISTS
-----------------------
The review of abaqus_scripts/cel_model.py and abaqus_scripts/cel_results.py
inventories every Abaqus Scripting Interface symbol the project relies on
(methods, keyword arguments, symbolic constants). None of it can be verified
from the review environment, which has no Abaqus install. This script checks
those symbols against the REAL installation it runs on.

v2 -- WHAT CHANGED AND WHY
---------------------------
v1 checked module-level names (`sketch.ConstrainedSketch`, `part.Part`,
`material.Material`) and reported them MISSING. That was a defect in the
CHECK, not a finding about the project: cel_model.py never calls those names.
It calls `model.ConstrainedSketch(...)`, `model.Part(...)`,
`model.Material(...)` -- methods grafted onto the Model object. v1 also never
checked the assembly, Material-instance, Job or ODB APIs at all, which is
most of what the project actually uses.

v3 -- THREE DEFECTS THE FIRST v2 RUN EXPOSED
---------------------------------------------
1. It reported Model.RigidBody / FieldOutputRequest / HistoryOutputRequest
   MISSING because it imported only `abaqus`. Those methods are grafted onto
   Model by importing the CAE modules, which cel_model.py does at its top
   (`from step import *`, `from interaction import *`, ...). v3 replicates
   those imports before introspecting. The ODB proves all three work: it
   carries 501 field-output frames, RF1/RF2 on the rigid-body reference node.
2. `model.Material(name="_check_api_mat")` raised "invalid name" -- Abaqus
   rejects a leading underscore. All throwaway names now start with a letter.
3. The FieldValue probe used `keys[0]`, which landed on a contact output
   (CPRESS) rather than on anything the project extracts. v3 probes the
   variables cel_results.py actually reads (EVF/TEMP/V on the Eulerian
   instance), which is the only way to settle whether `_read_data`'s
   `data` -> `dataDouble` fallback is sound.

v2/v3 instantiate a THROWAWAY in-memory model and introspect the real
objects, then delete it. They still:
  * builds no geometry, meshes nothing, submits nothing,
  * writes exactly one file (the report),
  * deletes the temporary model and job from the mdb before exiting.

Existence still does not prove correct usage: a FOUND row means the method is
there, not that the keyword arguments passed by cel_model.py are right.
Signatures are printed whenever Abaqus exposes them (many CAE methods are
extension types with no introspectable signature -- those print as such).

USAGE (on the Abaqus PC, from the repository root):

    abaqus cae noGUI=_review/check_api.py

To ALSO verify the ODB extraction API and the history-output variable names
against a real result file, pass one you already have:

    abaqus cae noGUI=_review/check_api.py -- --odb C:\\TEMP\\ABQ_wd\\myjob.odb

That is strongly recommended: the ODB branch is the only way to confirm
`getScalarField`/`getSubset`/`dataDouble` and to see whether MASSEUL/VOLEUL,
RF1/RF2 and ALLKE/ALLIE really landed in the history regions. The ODB is
opened READ-ONLY and closed again.

Writes "_review/check_api_report.txt" (falling back to the current directory)
and touches nothing else.

Python 2.7 (Abaqus' bundled interpreter): no f-strings, no annotations.
"""
import inspect
import sys

REPORT_LINES = []

# Abaqus rejects names with a leading underscore ("invalid name"), which is
# what broke the Material check on the first v2 run. Letters only.
_TMP_MODEL = "CheckApiTmpModel"
_TMP_JOB = "CheckApiTmpJob"


def log(line=""):
    REPORT_LINES.append(line)
    print(line)


def _signature(obj):
    """Best-effort call signature; Abaqus CAE methods are often extension
    types with none."""
    try:
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            return str(inspect.getargspec(obj))
        if inspect.isclass(obj):
            return str(inspect.getargspec(obj.__init__))
    except TypeError:
        return "(extension type - no introspectable signature)"
    except Exception:
        return "(signature unavailable)"
    return ""


def check_attr(owner, owner_name, attr_name):
    """Report whether `owner.attr_name` exists, with its signature if any."""
    try:
        ok = hasattr(owner, attr_name)
    except Exception as exc:
        log("  [ERROR]   %s.%s -- hasattr raised: %s"
            % (owner_name, attr_name, exc))
        return False
    if not ok:
        log("  [MISSING] %s.%s" % (owner_name, attr_name))
        return False
    log("  [FOUND]   %s.%s %s"
        % (owner_name, attr_name, _signature(getattr(owner, attr_name))))
    return True


def check_const(module, module_name, name):
    if hasattr(module, name):
        log("  [FOUND]   %s.%s = %r" % (module_name, name, getattr(module, name)))
        return True
    log("  [MISSING] %s.%s" % (module_name, name))
    return False


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


# ---------------------------------------------------------------------------
# What cel_model.py / cel_results.py actually use
# ---------------------------------------------------------------------------
CONSTANTS = [
    "THREE_D", "EULERIAN", "DEFORMABLE_BODY", "OFF", "ON",
    "EC3D8RT", "C3D8RT", "EXPLICIT", "DEFAULT", "HEX", "STRUCTURED",
    "SWEEP", "SINGLE", "PENALTY", "ROUGH", "FRICTIONLESS", "HARD",
    "EXPONENTIAL", "LINEAR", "TABULAR", "GLOBAL", "SELF",
    "JOHNSON_COOK", "CONSTANTPRESSURE", "ENERGY", "MAGNITUDE", "SET",
    "FREE", "NONE", "VOID", "NON_REFLECTING", "EQUILIBRIUM",
    "ZERO_PRESSURE", "INFLOW", "OUTFLOW", "BOTH", "ANALYSIS", "DOUBLE",
    "FULL", "CENTROID", "NODAL", "MISES", "PRESS", "PRESELECT",
]

# model.<name>(...) in cel_model.py
MODEL_METHODS = [
    "ConstrainedSketch", "Part", "Material", "EulerianSection",
    "HomogeneousSolidSection", "ContactProperty", "ContactExp", "RigidBody",
    "TempDisplacementDynamicsStep", "ButterworthFilter", "FieldOutputRequest",
    "HistoryOutputRequest", "EulerianBC", "VelocityBC", "Velocity",
    "MaterialAssignment", "Temperature", "rootAssembly",
]

# model.rootAssembly.<name>(...) in cel_model.py
ASSEMBLY_METHODS = [
    "Instance", "translate", "excludeFromSimulation", "seedPartInstance",
    "Set", "Surface", "setElementType", "setMeshControls", "seedEdgeBySize",
    "seedEdgeByNumber", "seedEdgeByBias", "generateMesh", "ReferencePoint",
    "referencePoints", "DiscreteFieldByVolumeFraction", "sets", "instances",
]

# material.<name>(...) in cel_model.create_materials
MATERIAL_METHODS = [
    "Density", "Elastic", "Conductivity", "SpecificHeat", "Expansion",
    "InelasticHeatFraction", "Plastic", "JohnsonCookDamageInitiation",
]

# contactProperty.<name>(...) in cel_model.create_interaction
CONTACT_PROP_METHODS = ["TangentialBehavior", "NormalBehavior", "HeatGeneration"]

# job.<name>(...) in cel_model.run_job
JOB_METHODS = ["writeInput", "submit", "waitForCompletion", "status", "messages"]

# sketch.<name>(...) in cel_model.create_parts
SKETCH_METHODS = [
    "rectangle", "Spot", "FixedConstraint", "Line", "HorizontalConstraint",
    "VerticalConstraint", "FilletByRadius", "CoincidentConstraint",
    "ObliqueDimension", "AngularDimension",
]


def _parse_odb_path(argv):
    """`--odb <path>` after the lone `--`, same convention as run_simul.py."""
    args = argv
    if "--" in args:
        args = args[args.index("--") + 1:]
    for i, a in enumerate(args):
        if a == "--odb" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--odb="):
            return a.split("=", 1)[1]
    return None


def check_constants():
    section("abaqusConstants used by cel_model.py / cel_results.py")
    import abaqusConstants
    missing = [n for n in CONSTANTS
               if not check_const(abaqusConstants, "abaqusConstants", n)]
    return missing


def import_cae_modules():
    """Import exactly what cel_model.py imports.

    This is not cosmetic. Abaqus grafts Model methods onto the Model object as
    these modules load: without `step`, `model.FieldOutputRequest` does not
    exist; without `interaction`, neither does `model.RigidBody`. The first v2
    run reported all three MISSING for exactly this reason -- a defect in the
    check, not in the project.
    """
    section("Importing the CAE modules cel_model.py imports")
    for name in ("step", "sketch", "load", "part", "mesh", "interaction",
                 "regionToolset", "material", "section", "assembly",
                 "connectorBehavior"):
        try:
            __import__(name)
            log("  [OK]      import %s" % name)
        except Exception as exc:
            # Only the first seven are imported by cel_model.py; the rest are
            # opportunistic, so a failure here is informational.
            log("  [SKIP]    import %s -- %s" % (name, exc))


def check_model_side(mdb):
    """Instantiate a throwaway model and introspect the objects the project
    really talks to. Everything is removed again in the finally block."""
    from abaqusConstants import JOHNSON_COOK, THREE_D, DEFORMABLE_BODY

    section("Model methods (model.<X>) -- the API cel_model.py actually calls")
    log("  Checked on a throwaway in-memory model, not on a module-level")
    log("  class: `model.Part(...)` is a Model METHOD, which is why v1's")
    log("  `part.Part` check was meaningless.")
    log("")
    if _TMP_MODEL in mdb.models.keys():
        del mdb.models[_TMP_MODEL]
    model = mdb.Model(name=_TMP_MODEL, absoluteZero=-273.15)
    try:
        for name in MODEL_METHODS:
            check_attr(model, "Model", name)

        section("rootAssembly methods (assembly.<X>)")
        assembly = model.rootAssembly
        for name in ASSEMBLY_METHODS:
            check_attr(assembly, "rootAssembly", name)

        section("Material methods (material.<X>)")
        try:
            mat = model.Material(name="CheckApiMat")
            for name in MATERIAL_METHODS:
                check_attr(mat, "Material", name)
            # .Plastic(...).RateDependent(...) is chained in create_materials:
            # confirm RateDependent hangs off the Plastic object it returns.
            try:
                plastic = mat.Plastic(
                    hardening=JOHNSON_COOK,
                    table=((1.0, 1.0, 1.0, 1.0, 1000.0, 20.0),))
                check_attr(plastic, "Plastic (returned object)", "RateDependent")
            except Exception as exc:
                log("  [ERROR]   Plastic(hardening=JOHNSON_COOK, ...): %s" % exc)
        except Exception as exc:
            log("  [ERROR]   model.Material(...) raised: %s" % exc)

        section("ContactProperty methods (IntProp.<X>)")
        try:
            prop = model.ContactProperty(name="CheckApiProp")
            for name in CONTACT_PROP_METHODS:
                check_attr(prop, "ContactProperty", name)
        except Exception as exc:
            log("  [ERROR]   model.ContactProperty(...) raised: %s" % exc)

        section("ConstrainedSketch methods (sketch.<X>)")
        try:
            sk = model.ConstrainedSketch(name="CheckApiSketch", sheetSize=5)
            for name in SKETCH_METHODS:
                check_attr(sk, "ConstrainedSketch", name)
        except Exception as exc:
            log("  [ERROR]   model.ConstrainedSketch(...) raised: %s" % exc)

        section("Part methods (part.<X>)")
        try:
            prt = model.Part(name="CheckApiPart", dimensionality=THREE_D,
                             type=DEFORMABLE_BODY)
            for name in ("BaseSolidExtrude", "SectionAssignment", "cells"):
                check_attr(prt, "Part", name)
        except Exception as exc:
            log("  [ERROR]   model.Part(...) raised: %s" % exc)

        section("Job methods (job.<X>) -- created, never submitted")
        try:
            if _TMP_JOB in mdb.jobs.keys():
                del mdb.jobs[_TMP_JOB]
            job = mdb.Job(name=_TMP_JOB, model=_TMP_MODEL)
            for name in JOB_METHODS:
                check_attr(job, "Job", name)
            # cel_model._check_job_succeeded avoids job.status in noGUI mode
            # because messages is empty there. Record what they read as.
            try:
                log("  job.status  = %r" % job.status)
                log("  job.messages = %r (empty list expected under noGUI)"
                    % (list(job.messages),))
            except Exception as exc:
                log("  (reading job.status/messages raised: %s)" % exc)
            del mdb.jobs[_TMP_JOB]
        except Exception as exc:
            log("  [ERROR]   mdb.Job(...) raised: %s" % exc)
    finally:
        try:
            del mdb.models[_TMP_MODEL]
            log("")
            log("  (throwaway model %r deleted)" % _TMP_MODEL)
        except Exception as exc:
            log("  [WARNING] could not delete %r: %s" % (_TMP_MODEL, exc))


def check_odb_side(odb_path):
    section("ODB extraction API (cel_results.py) -- %s" % odb_path)
    from odbAccess import openOdb
    odb = openOdb(odb_path, readOnly=True)
    try:
        check_attr(odb, "Odb", "steps")
        check_attr(odb, "Odb", "rootAssembly")
        check_attr(odb, "Odb", "close")
        log("  steps: %r" % list(odb.steps.keys()))

        step_name = "Cut" if "Cut" in odb.steps.keys() else list(odb.steps.keys())[0]
        step = odb.steps[step_name]
        log("  using step %r with %d frames" % (step_name, len(step.frames)))

        log("")
        log("-- instances --")
        for iname in odb.rootAssembly.instances.keys():
            inst = odb.rootAssembly.instances[iname]
            log("  %s: %d nodes, %d elements"
                % (iname, len(inst.nodes), len(inst.elements)))
            if len(inst.elements):
                log("    first element type: %s" % inst.elements[0].type)

        log("")
        log("-- fieldOutputs keys (frame -1): what EVF/TEMP/V resolve to --")
        if len(step.frames):
            frame = step.frames[-1]
            keys = sorted(frame.fieldOutputs.keys())
            for k in keys:
                log("    %s" % k)
            for base in ("EVF", "TEMP", "V"):
                hits = [k for k in keys if k == base or k.startswith(base + "_")]
                log("  resolver candidates for %-4s : %r" % (base, hits))

            # Probe the variables cel_results.py ACTUALLY extracts, on the
            # Eulerian instance. The first v2 run probed keys[0] -- a contact
            # output -- and concluded nothing useful about _read_data().
            eul = None
            for iname in odb.rootAssembly.instances.keys():
                inst = odb.rootAssembly.instances[iname]
                if len(inst.elements) and inst.elements[0].type.startswith("EC"):
                    eul = inst
                    break
            log("")
            log("-- FieldValue.data vs .dataDouble on the EXTRACTED variables --")
            log("   cel_results._read_data() does `try: v.data / except:")
            log("   v.dataDouble`, on the premise that a double-precision ODB")
            log("   makes .data raise. If .data works and .dataDouble is")
            log("   absent, that premise is wrong and the fallback is dead")
            log("   code that would raise AttributeError if ever reached.")
            if eul is None:
                log("   [SKIP] no Eulerian (EC*) instance found in this ODB")
            for base in ("EVF", "TEMP", "V"):
                hits = [k for k in keys if k == base or k.startswith(base + "_")]
                if not hits:
                    log("  %-5s: no candidate key in this ODB" % base)
                    continue
                key = hits[0]
                try:
                    fo = frame.fieldOutputs[key]
                    if eul is not None:
                        try:
                            fo = fo.getSubset(region=eul)
                        except Exception:
                            pass
                    if not len(fo.values):
                        log("  %-5s (%s): no values on this instance"
                            % (base, key))
                        continue
                    v = fo.values[0]
                    has_data = hasattr(v, "data")
                    has_double = hasattr(v, "dataDouble")
                    log("  %-5s (%s): data=%s  dataDouble=%s"
                        % (base, key, has_data, has_double))
                    if has_data:
                        try:
                            log("        v.data reads OK -> %r" % (v.data,))
                        except Exception as exc:
                            log("        v.data RAISES: %s -- the fallback is"
                                " load-bearing here" % exc)
                except Exception as exc:
                    log("  %-5s (%s): probe failed: %s" % (base, key, exc))

            if keys:
                fo0 = frame.fieldOutputs[keys[0]]
                log("")
                log("-- FieldOutput methods (on %r) --" % keys[0])
                for name in ("getSubset", "getScalarField", "componentLabels",
                             "values"):
                    check_attr(fo0, "FieldOutput", name)

        log("")
        log("-- historyRegions: verifies RF1/RF2, ALLKE/ALLIE, MASSEUL/VOLEUL --")
        log("   (these are OUTPUT VARIABLE NAMES -- no hasattr can check them;")
        log("    only a real ODB shows whether Abaqus accepted the request)")
        all_hist = []
        for rkey in step.historyRegions.keys():
            region = step.historyRegions[rkey]
            names = sorted(region.historyOutputs.keys())
            all_hist.extend(names)
            log("  region %r: %r" % (rkey, names))
        log("")
        log("  VERDICT on the history variables the model requests:")
        for base in ("RF1", "RF2", "ALLKE", "ALLIE", "MASSEUL", "VOLEUL"):
            hits = [n for n in all_hist
                    if n == base or n.startswith(base + "_")]
            log("    %-8s : %s" % (base, hits if hits else "ABSENT"))
        log("")
        log("  MASSEUL/VOLEUL are requested by cel_model.py:615-624 inside a")
        log("  try/except that only prints a warning. If they read ABSENT")
        log("  above, the Eulerian mass/volume conservation check does not")
        log("  exist in the results, whatever the model script intended.")
    finally:
        odb.close()
        log("")
        log("  (ODB closed)")


def main():
    log("Abaqus API introspection report (check_api.py v2)")
    log("Python: %s" % sys.version)

    try:
        from abaqus import mdb
    except Exception as exc:
        log("[FATAL] cannot import abaqus: %s" % exc)
        _write_report()
        return

    missing_consts = check_constants()

    # MUST come before check_model_side: Model methods only exist once the
    # CAE modules that define them are imported.
    import_cae_modules()

    try:
        check_model_side(mdb)
    except Exception as exc:
        log("[ERROR] model-side introspection aborted: %s: %s"
            % (type(exc).__name__, exc))

    odb_path = _parse_odb_path(sys.argv)
    if odb_path:
        try:
            check_odb_side(odb_path)
        except Exception as exc:
            log("[ERROR] ODB introspection aborted: %s: %s"
                % (type(exc).__name__, exc))
    else:
        section("ODB extraction API -- SKIPPED")
        log("  Re-run with an existing result file to cover cel_results.py:")
        log("    abaqus cae noGUI=_review/check_api.py -- --odb <path>.odb")
        log("  Without it, getSubset/getScalarField/dataDouble and the")
        log("  MASSEUL/VOLEUL history names stay UNVERIFIED.")

    section("CLI options -- NOT checked by this script")
    log("  This script covers the Python Scripting Interface only. Verify the")
    log("  launcher options separately with `abaqus help`:")
    log("    abaqus cae noGUI=<script> -- --model_cfg <repr> --run_cfg <repr>")
    log("    abaqus terminate job=<name>")
    log("    abaqus job=<name> continue cpus=<n>")

    section("SUMMARY")
    missing = [l for l in REPORT_LINES if l.strip().startswith("[MISSING]")]
    errors = [l for l in REPORT_LINES if l.strip().startswith("[ERROR]")]
    log("  constants missing : %d" % len(missing_consts))
    log("  [MISSING] rows    : %d" % len(missing))
    log("  [ERROR] rows      : %d" % len(errors))
    if missing or errors:
        log("")
        log("  Every one of these needs a look -- either the project calls")
        log("  something that does not exist on this install, or this script")
        log("  is checking the wrong thing (as v1 did).")
        for l in missing + errors:
            log("   %s" % l.strip())
    else:
        log("  Every symbol checked exists on this installation.")

    _write_report()


def _write_report():
    try:
        handle = open("_review/check_api_report.txt", "w")
    except Exception:
        handle = open("check_api_report.txt", "w")
    try:
        handle.write("\n".join(REPORT_LINES))
        handle.write("\n")
    finally:
        handle.close()
    print("")
    print("Report written.")


if __name__ == "__main__":
    main()
