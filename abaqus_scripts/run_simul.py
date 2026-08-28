# -*- coding: utf-8 -*-
"""Entry point for the CEL Abaqus workflow.

The GUI launches this file with ``abaqus cae noGUI=run_simul.py -- ...``.
This module intentionally contains orchestration only:
  1. parse MODEL_CFG / RUN_CFG,
  2. build the Abaqus model,
  3. run (or only write) the job,
  4. extract results after a successful solve.
"""

import sys
import os
import argparse
import ast


def _ensure_script_dir():
    """Put the directory containing this script on sys.path.

    Abaqus noGUI execution does not guarantee that ``__file__`` is defined or
    that the noGUI script directory is importable.  Keep the same robust lookup
    strategy as the former monolithic script so sibling modules can be imported.
    """
    here = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        for arg in sys.argv:
            if arg.startswith('noGUI=') or arg.endswith('run_simul.py'):
                candidate = arg.split('=', 1)[-1]
                if os.path.isfile(candidate):
                    here = os.path.dirname(os.path.abspath(candidate))
                    break
    if here is None:
        raise ImportError(
            'cannot locate Abaqus script directory: run_simul.py, '
            'cel_model.py, cel_results.py and cel_common.py must be colocated')
    if here not in sys.path:
        sys.path.insert(0, here)
    return here


def parse_arguments(argv=None):
    """Parse and validate the two configuration dictionaries from the GUI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_cfg", required=True,
                        help="Python dict literal for model")
    parser.add_argument("--run_cfg", required=True,
                        help="Python dict literal for job run")

    raw_argv = sys.argv if argv is None else argv
    args, unknown = parser.parse_known_args(
        raw_argv[raw_argv.index("--") + 1:] if "--" in raw_argv
        else raw_argv[1:])

    model_cfg = ast.literal_eval(args.model_cfg)
    run_cfg = ast.literal_eval(args.run_cfg)

    if not isinstance(model_cfg, dict):
        raise TypeError("--model_cfg must evaluate to a dict")
    if not isinstance(run_cfg, dict):
        raise TypeError("--run_cfg must evaluate to a dict")

    return model_cfg, run_cfg


def main():
    _ensure_script_dir()

    # Imports are intentionally delayed until the Abaqus script directory is on
    # sys.path; both modules depend on the Abaqus Python environment.
    from cel_model import (build_model, create_job, run_job,
                           cleanup_working_directory)
    from cel_results import extract_results

    model_cfg, run_cfg = parse_arguments()
    model, params = build_model(model_cfg, run_cfg)
    job = create_job(model, params)

    if run_job(job, params):
        extract_results(params["job_name"], model_cfg)
        # Only after a successful run AND after extraction: the extractor needs
        # the .odb, and on a failure the scratch files are what you diagnose
        # with. run_job() raises on failure, and write_inp_only returns False,
        # so neither path reaches this line.
        if bool(run_cfg.get("cleanup_working_dir", True)):
            cleanup_working_directory(params["job_name"])


if __name__ == "__main__":
    main()
