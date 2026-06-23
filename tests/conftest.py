# -*- coding: utf-8 -*-
"""Shared pytest fixtures."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _qt_cleanup():
    """Destroy top-level widgets (and their embedded matplotlib canvases)
    after every test. Without this, accumulated FigureCanvasQTAgg widgets make
    Qt abort at teardown (SIGABRT/segfault) under the offscreen platform."""
    yield
    app = QApplication.instance()
    if app is None:
        return
    import gc
    app.processEvents()
    # deleteLater (not close) so MainWindow's closeEvent — which may pop a
    # modal "save changes?" dialog — is never triggered under offscreen.
    for w in list(app.topLevelWidgets()):
        w.deleteLater()
    app.processEvents()
    gc.collect()
    app.processEvents()
