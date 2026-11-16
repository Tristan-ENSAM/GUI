# -*- coding: utf-8 -*-
"""Snapshot of a RUNNING job: its results so far, as a GUI bundle.

Run by the Initial run tab with `abaqus python`, in the folder of the job:

    abaqus python live_snapshot.py <job> <model_cfg.json> [--tries 3]
                                                          [--wait 5]

It opens <job>.odb read-only while the solver writes it (route A, tested on
CL-WTS-1 / Abaqus 2022 GA on 2026-11-16: 5 reads of 6 worked during the run,
one failed with "Object is not available in DB" and the next one worked),
and writes <job>.live.results.npz + <job>.live.meta.json with the same
extraction as the end of a run (cel_results.extract_results). A failed read
is tried again after --wait seconds, --tries times in all.

Exit code: 0 written, 3 every try failed, 2 bad arguments.
"""
from __future__ import print_function

import json
import os
import sys
import time


def _script_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def _parse(argv):
    opts = {"tries": 3, "wait": 5.0}
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--tries", "--wait"):
            if i + 1 >= len(argv):
                raise SystemExit(2)
            opts[a[2:]] = (int(argv[i + 1]) if a == "--tries"
                           else float(argv[i + 1]))
            i += 2
        else:
            args.append(a)
            i += 1
    if len(args) != 2:
        print(__doc__)
        raise SystemExit(2)
    return args[0], args[1], opts


def snapshot(job, model_cfg, tries=3, wait=5.0, extract=None, sleep=None):
    """Write the live bundle of `job`; returns (ok, last error text)."""
    if extract is None:
        _script_dir()
        from cel_results import extract_results as extract
    sleep = sleep or time.sleep
    err = ""
    for k in range(max(1, int(tries))):
        try:
            extract(job, model_cfg, out_base=job + ".live")
            return True, ""
        except Exception as e:      # the solver is writing the .odb
            err = str(e)
            print("[LIVE] read %d of %d failed: %s" % (k + 1, tries, err))
            sys.stdout.flush()
            if k + 1 < tries:
                sleep(wait)
    return False, err


def main(argv):
    job, cfg_path, opts = _parse(argv)
    with open(cfg_path) as f:
        model_cfg = json.load(f)
    t0 = time.time()
    ok, err = snapshot(job, model_cfg, opts["tries"], opts["wait"])
    print("[LIVE] %s in %.1f s%s" % ("written" if ok else "FAILED",
                                    time.time() - t0,
                                    "" if ok else ": " + err))
    sys.stdout.flush()
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main(sys.argv[1:])
