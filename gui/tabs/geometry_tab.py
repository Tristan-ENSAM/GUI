# -*- coding: utf-8 -*-
"""
Geometry tab: edit tool, workpiece, eulerian-domain and bbox parameters,
with a live 2D preview and a panel of derived quantities
(effective dims after discretization, element count, stable dt estimate).
"""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, QSplitter, QGridLayout,
    QScrollArea, QFrame, QCheckBox,
)

from gui.core.model_config import ModelConfig
from gui.widgets.param_field import NumField, PairRow
from gui.widgets.geometry_preview import GeometryPreview


class GeometryTab(QWidget):
    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        # ------------ left: parameter form (in a scroll area) ------------
        form = QWidget()
        form_lay = QVBoxLayout(form)
        form_lay.setContentsMargins(4, 4, 4, 4)
        form_lay.setSpacing(4)

        form_lay.addWidget(self._build_tool_group())
        form_lay.addWidget(self._build_workpiece_group())
        form_lay.addWidget(self._build_euler_group())
        # _build_mesh_seeds_group returns None now (the options moved to
        # the Mesh and BCs tabs). Keep the call to avoid breaking other
        # code paths that might rely on side effects, but skip the addWidget.
        _maybe_widget = self._build_mesh_seeds_group()
        if _maybe_widget is not None:
            form_lay.addWidget(_maybe_widget)
        # The ZOI (= extraction zone) group is added by the main window
        # (add_extra_group): its fields are owned by the Optimize tabs.
        form_lay.addStretch()
        self._form_lay = form_lay

        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QFrame.Shape.NoFrame)
        form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        form_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        form_scroll.setWidget(form)
        form_scroll.setMinimumWidth(340)

        # ------------ right: preview + info ------------
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.chk_ref = QCheckBox("Show reference image (watermark)")
        self.chk_ref.setEnabled(False)
        self.chk_ref.setToolTip("Overlay the alignment reference image behind "
                                "the sketch (set it from the Alignment tab's "
                                "'Write to Simulate').")
        self.chk_ref.toggled.connect(self._on_toggle_reference)
        right_lay.addWidget(self.chk_ref)
        self.preview = GeometryPreview()
        # Two views: the model to scale, and a sketch of the parameters.
        from PySide6.QtWidgets import QTabWidget
        from gui.widgets.geometry_sketch import GeometrySketch
        self.sketch = GeometrySketch()
        self.views = QTabWidget()
        self.views.addTab(self.preview, "Model (to scale)")
        self.views.addTab(self.sketch, "Parameter sketch")
        right_lay.addWidget(self.views, stretch=1)
        right_lay.addWidget(self._build_info_panel())

        # ------------ splitter ------------
        split = QSplitter(Qt.Horizontal)
        split.addWidget(form_scroll)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 840])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)

        # initial draw
        self._refresh()

    def add_extra_group(self, widget) -> None:
        """Show `widget` (a group owned by another tab, e.g. the comparison
        settings of Optimization) at the end of this tab's form."""
        lay = self._form_lay
        lay.insertWidget(max(0, lay.count() - 1), widget)

    # =====================================================================
    # Group builders
    # =====================================================================
    # Dense form (Tristan 2026-11-16): two fields per row, standard symbols
    # (ISO 3002: rake angle γ, clearance angle α, cutting-edge radius r_β,
    # depth of cut t1; h is the element size), the full name in each tooltip and on the parameter
    # sketch.
    @staticmethod
    def _grid_group(title: str, rows) -> QGroupBox:
        """A group with `rows` of up to two NumFields per row."""
        g = QGroupBox(title)
        grid = QGridLayout(g)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(0)
        for r, row in enumerate(rows):
            for c, w in enumerate(row):
                if w is not None:
                    grid.addWidget(w, r, c)
        grid.setColumnStretch(2, 1)
        return g

    @staticmethod
    def _field(label, value, unit, tip, **kw):
        return NumField(label, value, unit, compact=True, tip=tip,
                        label_width=44, **kw)

    def _build_tool_group(self) -> QGroupBox:
        tg = self.cfg.tool_geometry
        tp = self.cfg.tool_position
        f = self._field
        self.f_h_tool = f("h<sub>tool</sub>", tg.h_tool, "mm", "Tool height")
        self.f_l_tool = f("l<sub>tool</sub>", tg.l_tool, "mm", "Tool length")
        self.f_r_tool = f("r<sub>β</sub>", tg.r_tool, "mm",
                          "Cutting-edge radius r_β (ISO 3002): radius "
                          "of the rounded edge between the rake and the "
                          "clearance faces", minimum=0.0)
        self.f_rake = f("γ", tg.rake_angle, "°",
                        "Rake angle γ (ISO 3002): angle between the "
                        "rake face and the normal to the cut surface")
        self.f_clear = f("α", tg.clear_angle, "°",
                         "Clearance angle α (ISO 3002): angle between "
                         "the clearance face and the cut surface")
        self.f_tx = f("x<sub>0</sub>", tp.x0, "mm",
                      "Tool position x0: x of the leading point of the cutting edge "
                      "(first point the material touches)")
        self.f_ty = f("y<sub>0</sub>", tp.y0, "mm",
                      "Tool position y0: y of the lowest point of the "
                      "cutting edge (the cutting line y = y0)")
        self.lbl_depth = QLabel("h = —")
        self.lbl_depth.setToolTip(
            "Depth of cut h (uncut chip thickness t1 of orthogonal cutting) "
            "= y0 of the workpiece − y0 of the tool.")
        g = self._grid_group("Tool", [
            (self.f_h_tool, self.f_l_tool),
            (self.f_rake, self.f_clear),
            (self.f_r_tool, None),
            (self.f_tx, self.f_ty),
        ])
        g.layout().addWidget(self.lbl_depth, 2, 1)
        for w in (self.f_h_tool, self.f_l_tool, self.f_r_tool,
                  self.f_rake, self.f_clear, self.f_tx, self.f_ty):
            w.valueChanged.connect(self._on_change)
        return g

    def _build_workpiece_group(self) -> QGroupBox:
        eg = self.cfg.euler_geometry
        wp = self.cfg.wp_position
        f = self._field
        self.f_h_wp = f("h<sub>wp</sub>", eg.h_wp, "mm", "Workpiece height",
                        minimum=0.0)
        self.f_l_wp = f("l<sub>wp</sub>", eg.l_wp, "mm", "Workpiece length",
                        minimum=0.0)
        self.f_wp_x = f("x<sub>0</sub>", wp.x0, "mm",
                        "Workpiece position x0: x of its right face")
        self.f_wp_y = f("y<sub>0</sub>", wp.y0, "mm",
                        "Workpiece position y0: y of its top face")
        g = self._grid_group("Workpiece", [(self.f_h_wp, self.f_l_wp),
                                           (self.f_wp_x, self.f_wp_y)])
        self._workpiece_group = g
        for w in (self.f_h_wp, self.f_l_wp, self.f_wp_x, self.f_wp_y):
            w.valueChanged.connect(self._on_change)
        return g

    def _build_euler_group(self) -> QGroupBox:
        eg = self.cfg.euler_geometry
        ep = self.cfg.euler_position
        f = self._field
        self.f_h_void = f("h<sub>void</sub>", eg.h_void, "mm",
                          "Height of the empty part of the Eulerian domain "
                          "above the workpiece", minimum=0.0)
        self.f_l_void = f("l<sub>void</sub>", eg.l_void, "mm",
                          "Length of the empty part of the Eulerian domain "
                          "right of the workpiece", minimum=0.0)
        self.f_ex = f("x<sub>0</sub>", ep.x0, "mm",
                      "Eulerian domain position x0 (x of the workpiece "
                      "right face inside it)")
        self.f_ey = f("y<sub>0</sub>", ep.y0, "mm",
                      "Eulerian domain position y0 (y of the workpiece top "
                      "face inside it)")
        g = self._grid_group("Eulerian domain", [(self.f_h_void, self.f_l_void),
                                                 (self.f_ex, self.f_ey)])
        self._euler_group = g
        for w in (self.f_h_void, self.f_l_void, self.f_ex, self.f_ey):
            w.valueChanged.connect(self._on_change)
        return g

    def _build_mesh_seeds_group(self) -> QGroupBox | None:
        """Geometry tab no longer owns any preview options (mesh options in
        the Mesh tab, BC options in the BCs/ICs tab)."""
        return None

    def _build_info_panel(self) -> QGroupBox:
        g = QGroupBox("Derived quantities")
        grid = QGridLayout(g)

        self.lbl_eff_dims = QLabel("—")
        self.lbl_eff_es   = QLabel("—")
        self.lbl_nel      = QLabel("—")
        self.lbl_dt       = QLabel("—")
        for lbl in (self.lbl_eff_dims, self.lbl_eff_es, self.lbl_nel, self.lbl_dt):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Labels for the row titles, kept as attributes because their text
        # changes between CEL and Lagrangian.
        self.lbl_eff_dims_title = QLabel("Effective (h_wp, h_void, l_wp, l_void):")
        self.lbl_nel_title      = QLabel("Eulerian element count (~):")

        grid.addWidget(self.lbl_eff_dims_title, 0, 0)
        grid.addWidget(self.lbl_eff_dims,       0, 1)
        grid.addWidget(QLabel("Effective elem_size:"), 1, 0)
        grid.addWidget(self.lbl_eff_es,                1, 1)
        grid.addWidget(self.lbl_nel_title, 2, 0)
        grid.addWidget(self.lbl_nel,       2, 1)
        grid.addWidget(QLabel("Stable Δt estimate:"), 3, 0)
        grid.addWidget(self.lbl_dt,                   3, 1)
        grid.setColumnStretch(1, 1)
        return g

    # =====================================================================
    # Sync widgets -> cfg, then refresh preview & info panel
    # =====================================================================
    def _on_change(self, *_):
        self._pull_from_widgets()
        self._refresh()

    def _pull_from_widgets(self):
        c = self.cfg
        c.tool_geometry.h_tool      = self.f_h_tool.value()
        c.tool_geometry.l_tool      = self.f_l_tool.value()
        c.tool_geometry.r_tool      = self.f_r_tool.value()
        c.tool_geometry.rake_angle  = self.f_rake.value()
        c.tool_geometry.clear_angle = self.f_clear.value()
        c.tool_position.x0          = self.f_tx.value()
        c.tool_position.y0          = self.f_ty.value()

        c.wp_position.x0 = self.f_wp_x.value()
        c.wp_position.y0 = self.f_wp_y.value()

        c.euler_geometry.h_wp       = self.f_h_wp.value()
        c.euler_geometry.h_void     = self.f_h_void.value()
        c.euler_geometry.l_wp       = self.f_l_wp.value()
        c.euler_geometry.l_void     = self.f_l_void.value()
        c.euler_position.x0         = self.f_ex.value()
        c.euler_position.y0         = self.f_ey.value()
        # elem_size and discretize are owned by the Mesh tab — not pulled here.

        # cfg.bbox (the ZOI) is written by the ZOI fields of the Optimize
        # tabs, shown in this tab's form.

    def apply_from_cfg(self):
        """Inverse of `_pull_from_widgets`: push cfg values back into the
        widgets. Called when a profile is loaded from disk so the UI
        reflects the freshly-loaded config without triggering a redraw
        per field.

        NumField.set_value / BoolField.set_value both block signals
        internally, so we can call this safely without firing
        `valueChanged` cascades. We then trigger a single explicit
        refresh at the end."""
        c = self.cfg
        self.f_h_tool.set_value(c.tool_geometry.h_tool)
        self.f_l_tool.set_value(c.tool_geometry.l_tool)
        self.f_r_tool.set_value(c.tool_geometry.r_tool)
        self.f_rake.set_value(c.tool_geometry.rake_angle)
        self.f_clear.set_value(c.tool_geometry.clear_angle)
        self.f_tx.set_value(c.tool_position.x0)
        self.f_ty.set_value(c.tool_position.y0)

        self.f_h_wp.set_value(c.euler_geometry.h_wp)
        self.f_l_wp.set_value(c.euler_geometry.l_wp)
        self.f_wp_x.set_value(c.wp_position.x0)
        self.f_wp_y.set_value(c.wp_position.y0)

        self.f_h_void.set_value(c.euler_geometry.h_void)
        self.f_l_void.set_value(c.euler_geometry.l_void)
        self.f_ex.set_value(c.euler_position.x0)
        self.f_ey.set_value(c.euler_position.y0)
        # elem_size / discretize are pushed by MeshTab.apply_from_cfg instead.


        self._refresh()

    def set_reference_overlay(self, image, mm_per_px):
        """Receive the alignment reference image + scale and show it as a
        watermark behind the sketch (enables the toggle)."""
        self.preview.set_reference_overlay(image, mm_per_px, show=True)
        self.chk_ref.setEnabled(True)
        self.chk_ref.blockSignals(True)
        self.chk_ref.setChecked(True)
        self.chk_ref.blockSignals(False)

    def _on_toggle_reference(self, on: bool):
        self.preview.set_reference_visible(on)

    def _refresh(self):
        # Geometry tab shows the pure geometry only: no mesh overlay (Mesh
        # tab owns that), no BC overlay (BCs/ICs tab owns that).
        self.preview.update_from_config(
            self.cfg,
            show_mesh=False,
            show_bcs=False,
            show_cut_line=True,
        )
        depth = self.cfg.wp_position.y0 - self.cfg.tool_position.y0
        self.lbl_depth.setText("t<sub>1</sub> = %.6g mm" % depth)

        h_wp, h_void, l_wp, l_void = self.cfg.effective_euler_dims()

        self.lbl_eff_dims_title.setText("Effective (h_wp, h_void, l_wp, l_void):")
        self.lbl_eff_dims.setText(
            f"{h_wp:.6g}, {h_void:.6g}, {l_wp:.6g}, {l_void:.6g} mm"
        )
        self.lbl_nel_title.setText("Eulerian element count (~):")

        # Effective element size: reveals what Abaqus will actually seed
        # when `discretize` is off (the user-typed value may not divide the
        # dims evenly, so Abaqus rounds the count and adjusts the size).
        es_x, es_y = self.cfg.effective_elem_sizes()
        if self.cfg.euler_geometry.discretize:
            self.lbl_eff_es.setText(
                f"{es_x:.6g} mm  (= elem_size, dims floored)"
            )
        else:
            if abs(es_x - es_y) < 1e-12:
                # Same size in both directions
                if abs(es_x - self.cfg.elem_size) < 1e-12:
                    self.lbl_eff_es.setText(
                        f"{es_x:.6g} mm  (matches elem_size exactly)"
                    )
                else:
                    self.lbl_eff_es.setText(
                        f"{es_x:.6g} mm  (elem_size = {self.cfg.elem_size:g})"
                    )
            else:
                # Anisotropic — display both, the smaller in bold mentally
                self.lbl_eff_es.setText(
                    f"x: {es_x:.6g} mm,  y: {es_y:.6g} mm  "
                    f"(elem_size = {self.cfg.elem_size:g})"
                )

        n = self.cfg.n_elements_estimate()
        self.lbl_nel.setText(f"{n:,}".replace(",", " "))
        dt = self.cfg.stable_dt_estimate()
        if dt > 0:
            self.lbl_dt.setText(f"{dt:.3e} s")
        else:
            self.lbl_dt.setText("—  (need E, ρ, elem_size > 0)")

