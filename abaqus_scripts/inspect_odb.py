# -*- coding: utf-8 -*-
"""
inspect_odb.py — list the field outputs actually present in an ODB.

Run under Abaqus Python (2.7):

    abaqus python inspect_odb.py --odb C:\\TEMP\\ABQ_wd\\Cutting_job.odb --step Cut

For the chosen step it prints, for the first frame, an intermediate frame
and the last frame: the available fieldOutputs keys and, for each key, the
instances that actually carry values. This is the ground truth used to
debug "field X not available" messages from the extractor — in particular
it shows whether element results (PEEQ, S, EVF) are simply missing from
frame 0 (normal in Abaqus/Explicit) but present later.
"""
from __future__ import print_function
import argparse
from odbAccess import openOdb


def instances_with_values(fo):
    names = set()
    try:
        for v in fo.values:
            inst = getattr(v, "instance", None)
            if inst is not None:
                names.add(inst.name)
    except Exception as e:
        return "<error reading values: %s>" % e
    return ", ".join(sorted(names)) if names else "(no element/node values)"


def dump_frame(step, fi, label):
    n = len(step.frames)
    if n == 0:
        print("  (step has no frames)")
        return
    if fi < 0:
        fi = n + fi
    fi = max(0, min(fi, n - 1))
    frame = step.frames[fi]
    print("\n--- %s: frame %d  (t = %g) ---" % (label, fi, frame.frameValue))
    keys = list(frame.fieldOutputs.keys())
    print("  fieldOutputs keys (%d): %s" % (len(keys), ", ".join(keys)))
    for k in keys:
        fo = frame.fieldOutputs[k]
        comp = list(fo.componentLabels) if fo.componentLabels else []
        comp_str = (" components=%s" % comp) if comp else ""
        print("    %-10s ->  instances: %s%s"
              % (k, instances_with_values(fo), comp_str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odb", required=True)
    ap.add_argument("--step", default="Cut")
    args = ap.parse_args()

    odb = openOdb(args.odb, readOnly=True)
    try:
        print("Steps in ODB: %s" % list(odb.steps.keys()))
        if args.step not in odb.steps:
            print("Step '%s' not found." % args.step)
            return
        step = odb.steps[args.step]
        nf = len(step.frames)
        print("Step '%s': %d frames" % (args.step, nf))
        print("Instances: %s" % list(odb.rootAssembly.instances.keys()))

        dump_frame(step, 0,        "FIRST frame")
        if nf > 2:
            dump_frame(step, nf // 2, "MIDDLE frame")
        dump_frame(step, -1,       "LAST frame")
    finally:
        odb.close()


if __name__ == "__main__":
    main()
