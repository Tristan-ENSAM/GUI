# -*- coding: utf-8 -*-
"""
Fake Abaqus command — a stand-in for ``abaqus.bat`` used by the dry-run
harness so the *whole* sensitivity pipeline can be exercised without
Abaqus installed.

It mimics the contract of ``abaqus_scripts/run_simul.py`` as seen from
``gui.sensitivity.run_worker``:

  * invoked as  ``<this> cae noGUI=<script> -- --model_cfg <repr>
    --run_cfg <repr>``  (the leading ``cae``/``noGUI=``/``--`` tokens are
    ignored, exactly like run_simul.py which uses add_help=False +
    parse_known_args),
  * reads ``job_name`` from ``--run_cfg``,
  * writes ``<job_name>.sta`` and the ``<job_name>.results.{npz,json}``
    bundle in the *current working directory* (the worker sets cwd to the
    workdir), then exits 0.

Field amplitudes are scaled by a deterministic function of ``--model_cfg``
so that two runs with different inputs yield *different* fields (a
field-discrepancy SSD QoI is then non-zero).

Environment switches (used by tests):
  STUB_FAIL=1        exit non-zero before writing the bundle.
  STUB_NO_BUNDLE=1   exit 0 but write no bundle (simulates a crashed solve).
  STUB_SLEEP=<sec>   sleep this long before finishing (simulates a long run).
  STUB_SPAWN_CHILD=1 spawn a child that sleeps STUB_SLEEP seconds, to test
                     that Cancel kills the *whole* process tree.
  STUB_CHILD_PIDFILE=<path>  write the spawned child's PID here.

Usage as a module:  python -m tests.abaqus_stub cae noGUI=x -- \
    --model_cfg "{...}" --run_cfg "{'job_name': 'sens_run000', 'cpus': 1}"
"""
from __future__ import annotations

import argparse
import ast
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Make the repo importable when run as a loose script (shebang launcher).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gui.results.fake_builder import build_fake_results


def _collect_numbers(obj, out):
    """Recursively gather all numeric leaves of a nested dict/list."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def _field_scale_from_cfg(model_cfg) -> float:
    """A bounded, deterministic [0.95, 1.05] scale derived from the cfg, so
    different inputs give different fields (non-zero SSD), same input gives
    the same field (reproducible)."""
    nums = []
    _collect_numbers(model_cfg, nums)
    h = sum((i + 1) * v for i, v in enumerate(nums))
    return 1.0 + 0.05 * math.sin(h)


def _write_sta(job_name: str, n_frames: int = 8, sim_time: float = 5e-4):
    """Write a small but format-faithful .sta (header + per-frame rows) that
    gui.core.sta_parser can parse. One 'Output Field Frame Number' row and
    one increment row per frame; wall-clock grows monotonically."""
    lines = [
        "  Abaqus/Explicit (fake stub)\n",
        "\n",
        "  STEP  TOTAL    STABLE   CRITICAL  KINETIC      TOTAL\n",
        "\n",
    ]
    inc = 0
    for k in range(1, n_frames + 1):
        st = sim_time * k / n_frames
        wall = k * 7  # seconds; -> HH:MM:SS below
        hh, rem = divmod(wall, 3600)
        mm, ss = divmod(rem, 60)
        inc += 1234
        lines.append(
            "  Output Field Frame Number %4d, of %4d, at step time %.3E\n"
            % (k, n_frames, st))
        lines.append(
            "  %6d  %.3E %.3E  %02d:%02d:%02d %.3E       %5d  %.3E  %.3E\n"
            % (inc, st, st, hh, mm, ss, 5.6e-10, 16760, 9.03e-6, 1.66e-1))
    Path("%s.sta" % job_name).write_text("".join(lines), encoding="latin-1")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Abaqus passes everything after the lone '--' to the noGUI script;
    # reproduce that so 'cae'/'noGUI=...'/'--' don't confuse the parser.
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model_cfg", default="{}")
    ap.add_argument("--run_cfg", default="{}")
    args, _unknown = ap.parse_known_args(argv)

    try:
        model_cfg = ast.literal_eval(args.model_cfg)
    except Exception:
        model_cfg = {}
    try:
        run_cfg = ast.literal_eval(args.run_cfg)
    except Exception:
        run_cfg = {}
    job_name = str(run_cfg.get("job_name", "stub_job"))

    print("[STUB] Abaqus/Explicit stand-in starting", flush=True)
    print("[STUB] job_name=%s cpus=%s" % (job_name, run_cfg.get("cpus")),
          flush=True)

    # Optionally spawn a child so Cancel can be checked to kill the tree.
    child = None
    sleep_s = float(os.environ.get("STUB_SLEEP", "0") or "0")
    if os.environ.get("STUB_SPAWN_CHILD") == "1":
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import time,sys; time.sleep(float(sys.argv[1]))",
             str(max(sleep_s, 30.0))])
        pidfile = os.environ.get("STUB_CHILD_PIDFILE")
        if pidfile:
            Path(pidfile).write_text(str(child.pid), encoding="ascii")
        print("[STUB] spawned child pid=%d" % child.pid, flush=True)

    if sleep_s > 0:
        # Stream a few lines while "working" so the worker's reader has
        # something to consume; remain responsive to termination.
        t_end = time.monotonic() + sleep_s
        while time.monotonic() < t_end:
            print("[STUB] ... solving ...", flush=True)
            time.sleep(0.2)

    if os.environ.get("STUB_FAIL") == "1":
        print("[STUB] forced failure (STUB_FAIL=1)", flush=True)
        return 1

    _write_sta(job_name)
    print("[STUB] wrote %s.sta" % job_name, flush=True)

    if os.environ.get("STUB_NO_BUNDLE") == "1":
        print("[STUB] no bundle written (STUB_NO_BUNDLE=1)", flush=True)
        return 0

    fs = _field_scale_from_cfg(model_cfg)
    build_fake_results("%s.results.npz" % job_name,
                       cfg=model_cfg if isinstance(model_cfg, dict) else None,
                       n_frames=6, n_grid_x=10, n_grid_y=8,
                       job_name=job_name, field_scale=fs)
    print("[STUB] wrote %s.results.npz/.json (field_scale=%.4f)"
          % (job_name, fs), flush=True)
    print("[STUB] COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
