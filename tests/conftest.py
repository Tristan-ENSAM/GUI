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


def setup_settings(cfg):
    """Setup-tab settings (eps_q, window T, ZOI inside the ROI, EVF
    threshold), as the main window gives them to Explore / Sensitivity:
    the plans use the QoI of the sizing studies (QoI sort of 2026-11-06)."""
    b = cfg.bbox
    zoi = (float(b.xmin), 0.5 * (float(b.xmin) + float(b.xmax)),
           float(b.ymin), 0.5 * (float(b.ymin) + float(b.ymax)))
    return {"eps": {"Vx": 10.0, "Vy": 10.0, "T": 10.0, "EVF": 0.1,
                    "Fc": 10.0, "Ff": 10.0},
            "window": (0.3, 1.0), "zoi": zoi, "evf_threshold": 0.5}


def _connected_tab(cfg=None, **kw):
    """A SensitivityTab wired to Setup settings, as in the main window."""
    from gui.core.model_config import ModelConfig
    from gui.tabs.sensitivity_tab import SensitivityTab
    cfg = cfg if cfg is not None else ModelConfig()
    tab = SensitivityTab(cfg, **kw)
    tab.set_model_settings_getter(lambda: setup_settings(tab.cfg))
    return tab

