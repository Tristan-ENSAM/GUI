# -*- coding: utf-8 -*-
"""
Crash / fault diagnostics for the GUI.

Aimed at diagnosing *native* crashes that produce NO Python traceback on their
own — typically the Windows heap-corruption abort reported as exit code
0xC0000374 / -1073740940. Such a crash kills the interpreter at the C level, so
the usual Python machinery never runs; the only way to get a clue is to arm
low-level hooks *before* the crash happens.

What install_crash_diagnostics() sets up:
  1. Python logging to a timestamped file (kept even if the console closes).
  2. faulthandler dumping every thread's traceback to a dedicated file on a
     fatal native signal (access violation, abort, ...). The file handle is
     kept open for the whole process on purpose.
  3. sys.excepthook and threading.excepthook, so uncaught Python exceptions on
     the main thread AND on QThread workers are logged (a worker exception can
     leave Qt in a state that later corrupts the heap).
  4. A Qt message handler: Qt warnings — especially the cross-thread
     "Cannot create/destroy children ... in a different thread" family, a
     classic heap-corruption cause — are routed to the log instead of vanishing.
  5. A dump of the native-library versions (PySide6 / numpy / scipy /
     matplotlib) and numpy's build config: a mismatch between binary wheels
     (e.g. two MKL builds) is a prime suspect for heap corruption.

Call install_crash_diagnostics() as EARLY as possible in main(), before the
QApplication is created. It is defensive: any failure while arming a hook is
logged and swallowed so it can never prevent the GUI from starting.
"""
from __future__ import annotations

import faulthandler
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

# Keep the faulthandler output file alive for the whole process. If this handle
# were garbage-collected, faulthandler would write to a closed file at crash
# time and we would lose the trace.
_FAULT_FILE = None


def _resolve_log_dir(log_dir) -> Path:
    """Pick a writable directory for the logs, falling back gracefully."""
    candidates = []
    if log_dir:
        candidates.append(Path(log_dir))
    candidates.append(Path.home() / ".gui_abaqus_logs")
    candidates.append(Path.cwd() / "gui_logs")
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            # writability probe
            probe = c / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return c
        except Exception:
            continue
    return Path.cwd()


def install_crash_diagnostics(log_dir=None):
    """Arm the crash diagnostics. Returns (log_path, fault_path).

    log_dir : preferred directory for the logs (e.g. the Preferences working
              directory). Falls back to ~/.gui_abaqus_logs then the CWD.
    """
    global _FAULT_FILE
    d = _resolve_log_dir(log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = d / ("gui_debug_%s.log" % stamp)
    fault_path = d / ("gui_fault_%s.log" % stamp)

    # 1) Python logging -> file + stderr.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    log = logging.getLogger("gui.debug")

    # 2) faulthandler -> dedicated file, all threads.
    try:
        _FAULT_FILE = open(fault_path, "w")
        faulthandler.enable(file=_FAULT_FILE, all_threads=True)
        log.info("faulthandler armed -> %s", fault_path)
    except Exception:
        # last resort: dump to stderr instead of a file
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            log.warning("Could not enable faulthandler", exc_info=True)

    # 3) uncaught exceptions on the main thread and on worker threads.
    def _excepthook(exc_type, exc, tb):
        log.error("UNCAUGHT EXCEPTION (main thread)",
                  exc_info=(exc_type, exc, tb))
        try:
            sys.__excepthook__(exc_type, exc, tb)
        except Exception:
            pass
    sys.excepthook = _excepthook

    if hasattr(threading, "excepthook"):
        def _thread_excepthook(args):
            name = getattr(getattr(args, "thread", None), "name", "?")
            log.error("UNCAUGHT EXCEPTION (thread %s)", name,
                      exc_info=(args.exc_type, args.exc_value,
                                args.exc_traceback))
        threading.excepthook = _thread_excepthook

    # 4) Qt message handler (cross-thread widget warnings, etc.).
    try:
        from PySide6.QtCore import qInstallMessageHandler, QtMsgType
        _level = {
            QtMsgType.QtDebugMsg: logging.DEBUG,
            QtMsgType.QtInfoMsg: logging.INFO,
            QtMsgType.QtWarningMsg: logging.WARNING,
            QtMsgType.QtCriticalMsg: logging.ERROR,
            QtMsgType.QtFatalMsg: logging.CRITICAL,
        }
        qt_log = logging.getLogger("qt")

        def _qt_handler(mode, context, message):
            lvl = _level.get(mode, logging.INFO)
            # context.file/line are often empty; include when present
            where = ""
            try:
                if context is not None and context.file:
                    where = " (%s:%d)" % (context.file, context.line)
            except Exception:
                pass
            qt_log.log(lvl, "%s%s", message, where)
        qInstallMessageHandler(_qt_handler)
        log.info("Qt message handler installed")
    except Exception:
        log.warning("Could not install the Qt message handler", exc_info=True)

    # 5) native-library versions + numpy build config.
    _log_environment(log)

    log.info("Crash diagnostics installed.")
    log.info("  general log : %s", log_path)
    log.info("  fault  log  : %s", fault_path)
    # Make the paths obvious on the console too.
    try:
        sys.stderr.write("\n[DEBUG] Logs for this session:\n"
                         "  %s\n  %s\n\n" % (log_path, fault_path))
        sys.stderr.flush()
    except Exception:
        pass
    return log_path, fault_path


def _log_environment(log):
    import platform
    try:
        log.info("Python %s", sys.version.replace("\n", " "))
        log.info("Platform %s", platform.platform())
        log.info("Executable %s", sys.executable)
    except Exception:
        pass
    for mod in ("PySide6", "numpy", "scipy", "matplotlib"):
        try:
            m = __import__(mod)
            log.info("%-11s %s", mod, getattr(m, "__version__", "?"))
        except Exception as e:
            log.info("%-11s NOT importable: %s", mod, e)
    # numpy build config often reveals a conflicting BLAS/MKL.
    try:
        import io
        import contextlib
        import numpy as np
        if hasattr(np, "show_config"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                np.show_config()
            cfg = buf.getvalue().strip()
            if cfg:
                log.info("numpy build config:\n%s", cfg)
    except Exception:
        pass
