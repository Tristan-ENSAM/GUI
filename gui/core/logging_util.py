# -*- coding: utf-8 -*-
"""Small logging helper so that *swallowed* exceptions stop being invisible.

Historically the codebase used many bare ``except Exception: pass`` blocks.
A few of those are legitimately tolerant (NaN-safe reductions, optional
fields), but a swallow that hides a real bug turns a crash into a *wrong
result* — which is far worse. This module gives one helper, `log_swallowed`,
to keep the tolerant behaviour while making the error visible in the log.

Usage::

    from gui.core.logging_util import log_swallowed
    try:
        risky()
    except Exception:
        log_swallowed("parsing optional foo")   # behaviour unchanged, now logged

Set the env var ``GUI_ABAQUS_LOG=DEBUG`` (or INFO/WARNING) to control the
level; default is WARNING so swallowed errors surface during normal use.
"""
from __future__ import annotations

import logging
import os

_LEVEL = os.environ.get("GUI_ABAQUS_LOG", "WARNING").upper()

logger = logging.getLogger("gui_abaqus")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "[%(levelname)s] gui_abaqus: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(getattr(logging, _LEVEL, logging.WARNING))
    logger.propagate = False


def log_swallowed(context: str, level: int = logging.WARNING) -> None:
    """Log the exception currently being handled, with a short context.

    Call this from inside an ``except`` block. It records the traceback at
    DEBUG and a one-line summary at `level` (WARNING by default). It never
    raises, so control flow at the call site is unchanged."""
    try:
        logger.log(level, "swallowed exception while %s", context,
                   exc_info=logger.isEnabledFor(logging.DEBUG))
    except Exception:
        pass
