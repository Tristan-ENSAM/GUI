# -*- coding: utf-8 -*-
"""Which output channel of an Abaqus noGUI script actually reaches its caller?

WHY THIS EXISTS (finding M7 in _review/REVIEW.md)
-------------------------------------------------
Two `Write .inp only` runs produced DIFFERENT input decks -- so the generator
script really did run and really did change -- yet IDENTICAL Job-tab output
panels, neither carrying a single line that `run_simul.py` prints. Nothing the
script writes to stdout reaches the GUI, although the panel is labelled "Live
output". That also hides the `[WARNING] MASSEUL/VOLEUL history not created:`
message, which is the one piece of evidence still missing to fix finding M4.

Two explanations remain, and they call for opposite fixes:
  A. Abaqus/CAE does forward its interpreter's stdout to whoever launched it,
     and the GUI's QProcess capture is what drops it -> fix the GUI.
  B. Abaqus/CAE does not forward it at all -> no GUI change can help, and the
     diagnostics must be written to a file beside the job instead.

This script separates the two. It writes the SAME marker to three channels,
so whichever arrives tells us which ones work. It builds nothing, runs no
solver, needs no configuration, and takes a second.

USAGE -- run BOTH, then compare.

1) From a cmd window, straight to a file:

       cd C:\\TEMP\\ABQ_wd
       C:\\SIMULIA\\Commands\\abaqus.bat cae noGUI=C:\\GUI_Abaqus\\_review\\stdout_probe.py > probe_console.txt 2>&1

2) Through the GUI, to exercise the exact path that loses the output:
   Preferences -> Settings, point "generator script" at this file, click
   "Write .inp only", then copy the output panel. Put the real
   `run_simul.py` back afterwards. The GUI will report a failure because no
   .inp appears -- that is expected and harmless; only the panel text matters.

READING THE RESULT
  * `[PROBE] stdout` in probe_console.txt but NOT in the GUI panel
        -> explanation A: Abaqus forwards it, the GUI drops it.
  * `[PROBE] stdout` in neither
        -> explanation B: it never leaves Abaqus.
  * `[PROBE] stderr` arriving where stdout does not
        -> the GUI merges channels (MergedChannels), so this would point at
           stdout buffering rather than at the pipe itself.
  * probe_marker.txt is written either way: it proves the script ran at all,
    which keeps a silent failure from being mistaken for a lost message.

Python 2.7 (Abaqus' interpreter): no f-strings, no annotations.
"""
import os
import sys


def main():
    # 1. stdout, flushed exactly the way run_simul.py flushes it.
    print("[PROBE] stdout line 1")
    print("[PROBE] stdout line 2 (flushed below, as cel_model does)")
    sys.stdout.flush()

    # 2. stderr. The Job tab sets QProcess.MergedChannels, so if this arrives
    #    while stdout does not, the pipe is fine and stdout is being buffered
    #    somewhere upstream.
    sys.stderr.write("[PROBE] stderr line\n")
    sys.stderr.flush()

    # 3. A file in the CURRENT working directory -- the channel that cannot
    #    fail, and the fallback the project would use if stdout is
    #    unreachable. The GUI sets cwd to the working directory.
    marker = os.path.join(os.getcwd(), "probe_marker.txt")
    handle = open(marker, "w")
    try:
        handle.write("the probe ran\n")
        handle.write("cwd: %s\n" % os.getcwd())
        handle.write("python: %s\n" % sys.version.replace("\n", " "))
        handle.write("argv: %r\n" % (sys.argv,))
    finally:
        handle.close()

    print("[PROBE] wrote %s" % marker)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
