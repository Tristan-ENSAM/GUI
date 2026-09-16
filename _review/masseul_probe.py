# -*- coding: utf-8 -*-
"""Find out, empirically, what Abaqus accepts in place of MASSEUL/VOLEUL.

WHY (finding M4 in _review/REVIEW.md)
--------------------------------------
`cel_model.create_step` asks for an Eulerian mass/volume conservation check:

    model.HistoryOutputRequest(name='H-Output-3', createStepName='Cut',
                               region=assembly.sets['Euler'],
                               variables=('MASSEUL', 'VOLEUL'), ...)

Abaqus refuses it, and the message -- recovered only after working around
finding M7 -- is:

    Invalid variables are specified in an output request.  An output request
    cannot be created in a step where some variables are invalid.

So the REGION is not the problem; the VARIABLE NAMES are. The request has
been failing silently since commit e9e967f, which is why no ODB has ever
carried the conservation check the model believes it has.

What the correct names are is exactly the sort of thing this review refuses
to answer from memory -- naming an Abaqus identifier from memory is what
created the bug in the first place. This script therefore does not propose an
answer: it TRIES a list of candidates against the real model and reports
which ones Abaqus accepts. Every line of its report is a fact.

It builds the real model through cel_model.build_model, so what it tests is
the actual step, the actual Eulerian set and the actual assembly -- not a
simplified stand-in that might accept or reject things differently. It never
submits anything: no solver, no licence beyond CAE, a few seconds.

USAGE -- same shape as run_simul_logged.py:
  1. Preferences -> Settings, point "generator script" at this file.
  2. Click "Write .inp only".
  3. Send back  <working directory>\\masseul_probe.log
  4. Put the real run_simul.py back.

The candidate list below is a list of THINGS TO TEST, not of
recommendations. Whatever comes back FOUND is what the fix will use; if
nothing does, the honest outcome is to drop the request rather than keep a
guard that never guarded anything.

Python 2.7 (Abaqus' interpreter): no f-strings, no annotations.
"""
import os
import sys
import traceback


_LOG_NAME = "masseul_probe.log"

# Candidates to TEST. Their presence here asserts nothing about their
# validity -- that is the whole point of running the probe.
_VARIABLE_CANDIDATES = [
    ("MASSEUL", "VOLEUL"),   # what the code asks for today, the baseline
    ("MASSEUL",),            # is exactly one of the two at fault?
    ("VOLEUL",),
    ("EVOL",),               # element volume, a documented element variable
    ("MASS",),               # element mass, likewise
    ("EVF",),                # Eulerian volume fraction: known-good control.
                             # If THIS is rejected too, the region or the
                             # request shape is wrong after all and the
                             # variable names are a red herring.
]


def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        for arg in sys.argv:
            if arg.endswith("masseul_probe.py"):
                return os.path.dirname(os.path.abspath(arg))
    return os.getcwd()


def _try_request(model, log, index, region, variables, n_intervals):
    """Attempt one HistoryOutputRequest; report the verdict, never raise."""
    name = "Probe_%d" % index
    try:
        model.HistoryOutputRequest(
            name=name, createStepName='Cut', region=region,
            variables=variables, numIntervals=n_intervals)
    except Exception as exc:
        log.write("  [REJECTED] %-24r %s\n" % (variables, exc))
        return False
    log.write("  [ACCEPTED] %-24r\n" % (variables,))
    # Remove it again so each candidate is judged on its own, and so the
    # deck this run writes is not polluted by the probe.
    try:
        del model.historyOutputRequests[name]
    except Exception:
        log.write("      (note: could not delete %s afterwards)\n" % name)
    return True


def main():
    here = _script_dir()
    scripts = os.path.join(os.path.dirname(here), "abaqus_scripts")
    if os.path.isdir(scripts) and scripts not in sys.path:
        sys.path.insert(0, scripts)

    log = open(os.path.join(os.getcwd(), _LOG_NAME), "w", 0)
    try:
        log.write("=== masseul_probe ===\n")
        log.write("cwd: %s\n\n" % os.getcwd())

        import run_simul
        from cel_model import build_model

        model_cfg, run_cfg = run_simul.parse_arguments()
        model, params = build_model(model_cfg, run_cfg)
        assembly = model.rootAssembly
        n_intervals = params["ho_n_intervals"]

        log.write("model built: %s\n" % model.name)
        log.write("assembly sets: %r\n\n" % sorted(assembly.sets.keys()))

        # --- 1. the variable names, on the region the code already uses ----
        log.write("1. HistoryOutputRequest(region=assembly.sets['Euler'])\n")
        try:
            region = assembly.sets['Euler']
        except Exception as exc:
            log.write("  [FATAL] no assembly set 'Euler': %s\n" % exc)
            region = None
        if region is not None:
            for i, variables in enumerate(_VARIABLE_CANDIDATES):
                _try_request(model, log, i, region, variables, n_intervals)

        # --- 2. does the integrated-output route even exist here? ----------
        # The CAE documentation maps integrated output onto the SAME history
        # output request, with the domain set to an integrated output section
        # rather than to a region. If that argument exists, the fix may lie
        # that way; if it does not, that avenue is closed.
        log.write("\n2. integrated-output route\n")
        log.write("  mdb.models[...].IntegratedOutputSection exists: %s\n"
                  % hasattr(model, "IntegratedOutputSection"))
        try:
            model.HistoryOutputRequest(
                name="Probe_integrated", createStepName='Cut',
                integratedOutputSection="does_not_exist",
                variables=("SOF",), numIntervals=n_intervals)
            log.write("  [ACCEPTED] integratedOutputSection= is a valid "
                      "argument (the section name was bogus on purpose)\n")
            try:
                del model.historyOutputRequests["Probe_integrated"]
            except Exception:
                pass
        except Exception as exc:
            # A complaint about the missing SECTION means the ARGUMENT is
            # valid; a complaint about the keyword itself means it is not.
            log.write("  [INFO] %s\n" % exc)
            log.write("  ^ read this closely: an error about the missing\n"
                      "    section means the argument exists; an error about\n"
                      "    an unexpected keyword means it does not.\n")

        log.write("\n=== done ===\n")
    except Exception:
        log.write("\n=== probe itself failed ===\n")
        traceback.print_exc(file=log)
    finally:
        try:
            log.flush()
        finally:
            log.close()


if __name__ == "__main__":
    main()
