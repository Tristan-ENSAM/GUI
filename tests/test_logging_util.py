# -*- coding: utf-8 -*-
"""
Unit tests for gui.core.logging_util.log_swallowed.

The helper is used throughout the codebase to keep tolerant `except` blocks
non-fatal while making the swallowed error visible in the log. These tests
pin its contract:
  - it records a message containing the supplied context,
  - it honours the requested level,
  - it never raises, so control flow at the call site is unchanged.

The gui_abaqus logger sets propagate=False, so pytest's caplog (which listens
on the root logger) would not see its records; we attach a temporary handler
to the logger directly instead.
"""
from __future__ import annotations

import logging
import pytest

from gui.core import logging_util as lu


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture()
def capture():
    h = _Capture()
    lu.logger.addHandler(h)
    prev = lu.logger.level
    lu.logger.setLevel(logging.DEBUG)        # let DEBUG records through
    try:
        yield h
    finally:
        lu.logger.removeHandler(h)
        lu.logger.setLevel(prev)


def test_logs_context_at_warning(capture):
    try:
        raise ValueError("boom")
    except Exception:
        lu.log_swallowed("doing the risky thing")
    assert len(capture.records) == 1
    rec = capture.records[0]
    assert rec.levelno == logging.WARNING
    assert "doing the risky thing" in rec.getMessage()


def test_logs_at_requested_level(capture):
    try:
        raise RuntimeError("x")
    except Exception:
        lu.log_swallowed("debug-level swallow", level=logging.DEBUG)
    assert capture.records and capture.records[0].levelno == logging.DEBUG


def test_never_raises_outside_except():
    # Called with no active exception: must still not raise.
    lu.log_swallowed("no active exception here")


def test_does_not_propagate_exception(capture):
    # The helper must not re-raise the exception being handled.
    ran = False
    try:
        raise KeyError("k")
    except Exception:
        lu.log_swallowed("swallowing a KeyError")
        ran = True
    assert ran is True
