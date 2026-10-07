# -*- coding: utf-8 -*-
"""The Optimization log must only be touched by the GUI thread.

run_bundle runs inside the study workers' threads and logs through
OptimizationTab._log_ui. Appending to the QPlainTextEdit from there raced the
GUI thread and crashed the GUI with an access violation (0xC0000005) when the
GCI study was launched on 2026-10-07.
"""
from __future__ import annotations

import threading

from gui.core.model_config import ModelConfig
from gui.tabs.optimization_tab import OptimizationTab


def test_log_from_worker_thread_is_appended_on_gui_thread(qapp, monkeypatch):
    tab = OptimizationTab(ModelConfig())
    gui_thread = threading.get_ident()
    seen = []
    real_append = tab.log.appendPlainText
    monkeypatch.setattr(
        tab.log, "appendPlainText",
        lambda text: (seen.append(threading.get_ident()), real_append(text)))

    worker = threading.Thread(target=tab._log_ui, args=("from worker\n",))
    worker.start()
    worker.join()
    assert seen == []                  # nothing touched from the worker
    qapp.processEvents()
    assert seen == [gui_thread]
    assert tab.log.toPlainText().endswith("from worker")


def test_log_from_gui_thread_is_immediate(qapp):
    tab = OptimizationTab(ModelConfig())
    tab._log_ui("now\n")
    assert tab.log.toPlainText().endswith("now")
