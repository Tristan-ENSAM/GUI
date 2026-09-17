# -*- coding: utf-8 -*-
"""Drop-in wrapper around run_simul.py that captures what M7 loses.

WHY (findings M7 and M4 in _review/REVIEW.md)
----------------------------------------------
M7 is established: `abaqus cae noGUI=` runs the script -- probe_marker.txt
proves it, under Python 2.7.15 with the right working directory -- but
neither its stdout nor its stderr reaches the caller. Not the Job tab, and
not even a plain `cmd` redirection. The launcher actually spawns
`ABQcaeK.exe -cae -noGUI <script>` as a separate kernel process, and that
process's output goes nowhere we can read.

The immediate casualty is M4: `cel_model.create_step` wraps its
MASSEUL/VOLEUL history request in a try/except that only prints
`[WARNING] MASSEUL/VOLEUL history not created: <message>`. That message is
the one piece of evidence still missing to fix M4, and it travels on exactly
the channel M7 destroys.

This wrapper redirects stdout and stderr to a FILE, then runs the real
pipeline unchanged. It imports the production `run_simul` rather than
copying any of it, so what it captures is what really happens -- no
reimplementation to drift out of sync.

It doubles as a prototype of fix (a) for M7: if the log lands correctly,
routing diagnostics to a file beside the job is a workable answer, and the
GUI already knows how to tail one (JobTab._poll_sta watches the .sta every
800 ms).

USAGE
  1. Preferences -> Settings, point "generator script" at THIS file:
         C:\\GUI_Abaqus\\_review\\run_simul_logged.py
  2. Click "Write .inp only" (builds the model, no solver, a few seconds).
  3. Send back  <working directory>\\run_simul_stdout.log
     i.e. normally C:\\TEMP\\ABQ_wd\\run_simul_stdout.log
  4. Put the real run_simul.py back in Preferences.

The log holds everything the pipeline prints, so search it for MASSEUL: the
line is either there with Abaqus' own wording, or genuinely absent -- which
would mean the request never raised and the deck loses it later instead.
Either answer settles M4.

Python 2.7 (Abaqus' interpreter): no f-strings, no annotations.
"""
import os
import sys
import traceback


_LOG_NAME = "run_simul_stdout.log"


def _script_dir():
    """Directory holding this wrapper, robust to __file__ being undefined
    under noGUI execution -- the same defensive lookup run_simul.py uses."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        for arg in sys.argv:
            if arg.endswith("run_simul_logged.py"):
                return os.path.dirname(os.path.abspath(arg))
    return os.getcwd()


def main():
    here = _script_dir()
    # abaqus_scripts/ sits beside _review/ in the repository.
    scripts = os.path.join(os.path.dirname(here), "abaqus_scripts")
    if os.path.isdir(scripts) and scripts not in sys.path:
        sys.path.insert(0, scripts)

    log_path = os.path.join(os.getcwd(), _LOG_NAME)
    # Unbuffered, so a hard crash mid-run still leaves everything written.
    handle = open(log_path, "w", 0)
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = handle
    sys.stderr = handle
    try:
        handle.write("=== run_simul_logged wrapper ===\n")
        handle.write("cwd:     %s\n" % os.getcwd())
        handle.write("scripts: %s\n" % scripts)
        handle.write("argv:    %r\n" % (sys.argv,))
        handle.write("=" * 60 + "\n")
        try:
            import run_simul
            run_simul.main()
            handle.write("\n=== run_simul.main() returned normally ===\n")
        except SystemExit as exc:
            # cel_results calls sys.exit(2) when the ROI selects nothing.
            handle.write("\n=== SystemExit: %r ===\n" % (exc.code,))
        except Exception:
            handle.write("\n=== run_simul.main() RAISED ===\n")
            traceback.print_exc(file=handle)
    finally:
        try:
            handle.flush()
        finally:
            handle.close()
            sys.stdout, sys.stderr = saved_out, saved_err


if __name__ == "__main__":
    main()
