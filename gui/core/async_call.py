# -*- coding: utf-8 -*-
"""Run one blocking call off the GUI thread, and come back on it.

WHY THIS EXISTS
---------------
Cancelling an Abaqus run is two blocking calls deep:

    abaqus_terminate_job(...)   # subprocess.run(..., timeout=20.0)
    proc.wait(timeout=10.0)     # or QProcess.waitForFinished(10000)

Both used to run inside the Cancel slot, i.e. on the GUI thread, so the window
stopped repainting for up to ~32 s in the Job tab (20 + 10 + 2) and ~30 s in
the two study tabs. That is finding M3 of _review/REVIEW.md, first closed
without correction, then reopened when Tristan asked for the reserve to be
lifted.

`run_async` moves the blocking call to a daemon thread and delivers its result
back through a Qt signal, which Qt queues onto the receiver's thread. The
caller therefore writes ordinary GUI code in the callback -- touching widgets
is safe there -- while nothing blocks the event loop.

WAITING, NOT BLOCKING. The second stage (give the solver a moment to unwind
before force-killing it) is deliberately NOT done here with a wait() on a
worker thread: a QProcess may only be touched from the thread that owns it,
and two threads wait()ing the same Popen race for its exit status. Callers use
a single-shot QTimer instead, which costs the event loop nothing.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from gui.core.logging_util import log_swallowed


class _AsyncCall(QObject):
    """Carries one result from the worker thread back to the GUI thread.

    It is parented to the calling widget so Qt keeps it alive until the signal
    has been delivered; a local variable would be collected first and the
    result would vanish silently.
    """

    done = Signal(object)


def run_async(fn, on_done, parent):
    """Call `fn()` on a daemon thread; call `on_done(result)` on `parent`'s thread.

    `parent` is required, and must be a QObject living in the GUI thread: it
    owns the carrier object and decides which thread the callback runs on.

    An exception inside `fn` is logged and delivered as a None result rather
    than killing the thread silently, so a caller that falls back on failure
    (kill the process tree when `abaqus terminate` did not answer) still gets
    its turn. Returns the carrier, which callers normally ignore.
    """
    carrier = _AsyncCall(parent)
    carrier.done.connect(on_done)

    def _work():
        try:
            result = fn()
        except Exception:
            log_swallowed("running %r off the GUI thread" % getattr(
                fn, "__name__", fn))
            result = None
        carrier.done.emit(result)

    threading.Thread(target=_work, daemon=True).start()
    return carrier
