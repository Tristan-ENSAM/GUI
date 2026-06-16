# -*- coding: utf-8 -*-
"""
Analysis tab — CEL only.

The project now targets the Coupled Eulerian-Lagrangian formulation
exclusively (the Lagrangian path was removed). The formulation is fixed to
"CEL"; this tab is informational. `analysisChanged` is still emitted on
apply so the geometry preview and other tabs refresh consistently.
"""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QScrollArea, QFrame,
)

from gui.core.model_config import ModelConfig


class AnalysisTab(QWidget):
    analysisChanged = Signal()

    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        # Hard-fix the formulation: this build is CEL-only.
        self.cfg.analysis.formulation = "CEL"

        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(12, 12, 12, 12)
        inner_lay.addWidget(self._build_formulation_group())
        inner_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _build_formulation_group(self) -> QGroupBox:
        g = QGroupBox("Formulation")
        lay = QVBoxLayout(g)
        title = QLabel("CEL - Coupled Eulerian-Lagrangian")
        title.setStyleSheet("font-weight: bold; color: #1f4060;")
        lay.addWidget(title)
        info = QLabel(
            "The workpiece is Eulerian (material flows through a fixed mesh),\n"
            "best for large deformations and chip formation without mesh\n"
            "distortion. This build supports the CEL formulation only."
        )
        info.setStyleSheet("color: #555;")
        lay.addWidget(info)
        return g

    # =====================================================================
    # External hooks
    # =====================================================================
    def apply_from_cfg(self):
        """Loading a profile can't change the formulation anymore - we just
        re-assert CEL and notify, so dependent tabs refresh."""
        self.cfg.analysis.formulation = "CEL"
        self.analysisChanged.emit()
