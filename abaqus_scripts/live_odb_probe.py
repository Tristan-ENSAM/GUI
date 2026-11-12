# -*- coding: utf-8 -*-
"""Can the .odb of a RUNNING Abaqus/Explicit job be read? (feasibility test)

The planned Initial run tab re-reads the results while the run is going
(stationarity, chip re-contact, subsampling) and stops it when a rule is met.
That needs the .odb to open while the solver is still writing it. This script
checks it, nothing else: it changes no file and writes only its own report.

Usage, while a job runs (any short CEL job of the GUI):

    abaqus python live_odb_probe.py <job>.odb [--every 30] [--count 10]
                                              [--step Cut] [--report probe.txt]

Every `--every` seconds, `--count` times, it opens the .odb in read-only
mode and reports:
  - whether it opened (and the error if not),
  - how many field frames the step has and the time of the last one,
  - the number of points and last time of the RF1/RF2 history,
  - whether the last frame's EVF (or the first field there) can be read,
  - how long the whole read took.
Then it closes the .odb. The report is printed and written to --report
(default: <job>.live_probe.txt next to the .odb).

Run it once on the office PC (Abaqus 2022.HF8) and once on CL-WTS-1
(Abaqus 2022 GA). The run must keep going while the script reads it; check
the end of the .sta file afterwards for an error at the read times.
"""
from __future__ import print_function

import os
import sys
import time

try:
    from odbAccess import openOdb
except ImportError:  # outside `abaqus python`
    openOdb = None


def _parse(argv):
    opts = {"every": 30.0, "count": 10, "step": "Cut", "report": None}
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--every", "--count", "--step", "--report"):
            if i + 1 >= len(argv):
                raise SystemExit("%s needs a value" % a)
            v = argv[i + 1]
            key = a[2:]
            opts[key] = (float(v) if key == "every" else
                         int(v) if key == "count" else v)
            i += 2
        else:
            args.append(a)
            i += 1
    if len(args) != 1:
        raise SystemExit(__doc__)
    return args[0], opts


def _history(step):
    """(n points, last time) of the first region holding RF1 and RF2, or
    (0, None)."""
    for _k, region in step.historyRegions.items():
        outs = region.historyOutputs
        keys = [k for k in outs.keys() if k.upper().startswith("RF1")]
        if keys and any(k.upper().startswith("RF2") for k in outs.keys()):
            data = outs[keys[0]].data
            return len(data), (data[-1][0] if data else None)
    return 0, None


def _last_field(frame):
    """(name, n values) of EVF in the frame, else of its first field."""
    fos = frame.fieldOutputs
    names = list(fos.keys())
    if not names:
        return None, 0
    name = "EVF" if "EVF" in names else names[0]
    fo = fos[name]
    try:
        n = sum(len(b.data) for b in fo.bulkDataBlocks)
    except Exception:
        n = len(fo.values)
    return name, n


def probe_once(path, step_name):
    t0 = time.time()
    line = {"when": time.strftime("%H:%M:%S")}
    try:
        odb = openOdb(path, readOnly=True)
    except Exception as e:
        line["error"] = "open failed: %s" % (e,)
        line["seconds"] = time.time() - t0
        return line
    try:
        step = odb.steps[step_name] if step_name in odb.steps.keys() \
            else odb.steps[odb.steps.keys()[-1]]
        frames = step.frames
        line["frames"] = len(frames)
        line["t_last_frame"] = frames[-1].frameValue if len(frames) else None
        line["history_points"], line["t_last_history"] = _history(step)
        if len(frames):
            line["field"], line["field_values"] = _last_field(frames[-1])
    except Exception as e:
        line["error"] = "read failed: %s" % (e,)
    finally:
        try:
            odb.close()
        except Exception as e:
            line["close_error"] = str(e)
    line["seconds"] = time.time() - t0
    return line


def _fmt(line):
    if "error" in line:
        return "%s  ERROR %s  (%.1f s)" % (line["when"], line["error"],
                                          line["seconds"])
    return ("%s  frames %s, last frame t = %s s, history %s points up to "
            "t = %s s, %s: %s values, read in %.1f s%s"
            % (line["when"], line.get("frames"), line.get("t_last_frame"),
               line.get("history_points"), line.get("t_last_history"),
               line.get("field"), line.get("field_values"), line["seconds"],
               ("  (close: %s)" % line["close_error"])
               if "close_error" in line else ""))


def main(argv):
    path, opts = _parse(argv)
    if openOdb is None:
        raise SystemExit("Run this with `abaqus python`, not a plain Python.")
    path = os.path.abspath(path)
    report = opts["report"] or (os.path.splitext(path)[0]
                                + ".live_probe.txt")
    lck = os.path.splitext(path)[0] + ".lck"
    out = ["live_odb_probe: %s" % path,
           "job running at start (.lck present): %s" % os.path.exists(lck),
           "every %g s, %d reads, step %r" % (opts["every"], opts["count"],
                                             opts["step"])]
    for line in out:
        print(line)
    ok = 0
    for i in range(opts["count"]):
        res = probe_once(path, opts["step"])
        res_line = "%2d  %s  [.lck %s]" % (i + 1, _fmt(res),
                                            "present" if os.path.exists(lck)
                                            else "gone")
        print(res_line)
        sys.stdout.flush()
        out.append(res_line)
        ok += "error" not in res
        if i + 1 < opts["count"]:
            time.sleep(opts["every"])
    out.append("reads that worked: %d / %d" % (ok, opts["count"]))
    print(out[-1])
    with open(report, "w") as f:
        f.write("\n".join(out) + "\n")
    print("report written to %s" % report)


if __name__ == "__main__":
    main(sys.argv[1:])
