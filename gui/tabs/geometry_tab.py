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
    QScrollArea, QFrame,
)

from gui.core.model_config import ModelConfig
from gui.widgets.param_field import NumField, IntField, BoolField, PairRow
from gui.widgets.geometry_preview import GeometryPreview


class GeometryTab(QWidget):
    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        # ------------ left: parameter form (in a scroll area) ------------
        form = QWidget()
        form_lay = QVBoxLayout(form)
        form_lay.setContentsMargins(6, 6, 6, 6)

        form_lay.addWidget(self._build_tool_group())
        form_lay.addWidget(self._build_workpiece_group())
        form_lay.addWidget(self._build_euler_group())
        # _build_mesh_seeds_group returns None now (the options moved to
        # the Mesh and BCs tabs). Keep the call to avoid breaking other
        # code paths that might rely on side effects, but skip the addWidget.
        _maybe_widget = self._build_mesh_seeds_group()
        if _maybe_widget is not None:
            form_lay.addWidget(_maybe_widget)
        form_lay.addWidget(self._build_bbox_group())
        form_lay.addStretch()

        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QFrame.Shape.NoFrame)
        form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        form_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        form_scroll.setWidget(form)
        form_scroll.setMinimumWidth(380)

        # ------------ right: preview + info ------------
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.preview = GeometryPreview()
        right_lay.addWidget(self.preview, stretch=1)
        right_lay.addWidget(self._build_info_panel())

        # ------------ splitter ------------
        split = QSplitter(Qt.Horizontal)
        split.addWidget(form_scroll)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 800])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)

        # initial draw
        self._refresh()

    # =====================================================================
    # Group builders
    # =====================================================================
    def _build_tool_group(self) -> QGroupBox:
        g = QGroupBox("Tool")
        lay = QVBoxLayout(g)

        tg = self.cfg.tool_geometry
        tp = self.cfg.tool_position

        # Single-field rows
        self.f_h_tool = NumField("h_tool",         tg.h_tool,    "mm")
        self.f_l_tool = NumField("l_tool",         tg.l_tool,    "mm")
        self.f_r_tool = NumField("r_tool (edge)",  tg.r_tool,    "mm", minimum=0.0)
        self.f_rake   = NumField("rake angle",     tg.rake_angle, "°")
        self.f_clear  = NumField("clear angle",    tg.clear_angle, "°")
        # Paired position x0 / y0 on a single row
        self.f_tx = NumField("x0", tp.x0, "mm", compact=True)
        self.f_ty = NumField("y0", tp.y0, "mm", compact=True)
        pos_row = PairRow(self.f_tx, self.f_ty)

        for w in (self.f_h_tool, self.f_l_tool, self.f_r_tool,
                  self.f_rake, self.f_clear):
            lay.addWidget(w)
            w.valueChanged.connect(self._on_change)
        lay.addWidget(pos_row)
        self.f_tx.valueChanged.connect(self._on_change)
        self.f_ty.valueChanged.connect(self._on_change)
        return g

    def _build_workpiece_group(self) -> QGroupBox:
        g = QGroupBox("Workpiece")
        self._workpiece_group = g
        lay = QVBoxLayout(g)

        eg = self.cfg.euler_geometry
        wp = self.cfg.wp_position

        # Paired h_wp / l_wp on a single row
        self.f_h_wp = NumField("h_wp", eg.h_wp, "mm", minimum=0.0, compact=True)
        self.f_l_wp = NumField("l_wp", eg.l_wp, "mm", minimum=0.0, compact=True)
        dims_row = PairRow(self.f_h_wp, self.f_l_wp)

        # Paired wp x0 / y0
        self.f_wp_x = NumField("x0", wp.x0, "mm", compact=True)
        self.f_wp_y = NumField("y0", wp.y0, "mm", compact=True)
        pos_row = PairRow(self.f_wp_x, self.f_wp_y)

        lay.addWidget(dims_row)
        lay.addWidget(pos_row)
        for f in (self.f_h_wp, self.f_l_wp, self.f_wp_x, self.f_wp_y):
            f.valueChanged.connect(self._on_change)
        return g

    def _build_euler_group(self) -> QGroupBox:
        g = QGroupBox("Eulerian domain")
        self._euler_group = g
        lay = QVBoxLayout(g)
        eg = self.cfg.euler_geometry
        ep = self.cfg.euler_position

        # Paired h_void / l_void on a single row
        self.f_h_void = NumField("h_void", eg.h_void, "mm", minimum=0.0, compact=True)
        self.f_l_void = NumField("l_void", eg.l_void, "mm", minimum=0.0, compact=True)
        void_row = PairRow(self.f_h_void, self.f_l_void)

        # Paired euler x0 / y0
        self.f_ex = NumField("x0", ep.x0, "mm", compact=True)
        self.f_ey = NumField("y0", ep.y0, "mm", compact=True)
        pos_row = PairRow(self.f_ex, self.f_ey)

        lay.addWidget(void_row)
        lay.addWidget(pos_row)
        for f in (self.f_h_void, self.f_l_void, self.f_ex, self.f_ey):
            f.valueChanged.connect(self._on_change)
        return g

    def _build_mesh_seeds_group(self) -> QGroupBox | None:
        """Geometry tab no longer owns any preview options.

        Mesh-related options (elem_size, discretize, show_mesh) live in
        the Mesh tab. BC-related options (show_bcs, picking) live in the
        BCs/ICs tab. We keep this method returning None so the layout
        code stays similar and the change is local."""
        return None

    def _build_bbox_group(self) -> QGroupBox:
        g = QGroupBox("ROI")
        lay = QVBoxLayout(g)
        b = self.cfg.bbox

        self.f_xmin = NumField("xmin", b.xmin, "mm", compact=True)
        self.f_xmax = NumField("xmax", b.xmax, "mm", compact=True)
        self.f_ymin = NumField("ymin", b.ymin, "mm", compact=True)
        self.f_ymax = NumField("ymax", b.ymax, "mm", compact=True)
        x_row = PairRow(self.f_xmin, self.f_xmax)
        y_row = PairRow(self.f_ymin, self.f_ymax)
        for row in (x_row, y_row):
            lay.addWidget(row)
        note = QLabel("Extraction is taken on the z = 0 face (out-of-plane "
                      "bounds are fixed).")
        note.setStyleSheet("color: #888; font-style: italic;")
        lay.addWidget(note)
        for f in (self.f_xmin, self.f_xmax, self.f_ymin, self.f_ymax):
            f.valueChanged.connect(self._on_change)
        return g

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

        c.bbox.xmin = self.f_xmin.value(); c.bbox.xmax = self.f_xmax.value()
        c.bbox.ymin = self.f_ymin.value(); c.bbox.ymax = self.f_ymax.value()
        # z bounds are fixed (z = 0 face); not edited from the UI.

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

        self.f_xmin.set_value(c.bbox.xmin); self.f_xmax.set_value(c.bbox.xmax)
        self.f_ymin.set_value(c.bbox.ymin); self.f_ymax.set_value(c.bbox.ymax)
        # z bounds are fixed (z = 0 face); no widget to update.

        # Also update visibility of the Eulerian-domain group (it depends on
        # analysis.formulation, which may have changed via the loaded file).
        self._euler_group.setVisible(c.analysis.formulation != "Lagrangian")
        self._refresh()

    def _refresh(self):
        # Geometry tab shows the pure geometry only: no mesh overlay (Mesh
        # tab owns that), no BC overlay (BCs/ICs tab owns that).
        self.preview.update_from_config(
            self.cfg,
            show_mesh=False,
            show_bcs=False,
        )

        is_lagrangian = (self.cfg.analysis.formulation == "Lagrangian")
        h_wp, h_void, l_wp, l_void = self.cfg.effective_euler_dims()

        # Effective dims row: hide h_void / l_void in Lagrangian mode where
        # they are meaningless.
        if is_lagrangian:
            self.lbl_eff_dims_title.setText("Effective (h_wp, l_wp):")
            self.lbl_eff_dims.setText(f"{h_wp:.6g}, {l_wp:.6g} mm")
            self.lbl_nel_title.setText("Workpiece element count (~):")
        else:
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

    # =====================================================================
    # Reaction to the Analysis tab changing formulation
    # =====================================================================
    def on_analysis_changed(self):
        """Called by MainWindow when the user toggles formulation.
        We hide the entire Eulerian-domain group in Lagrangian mode (the
        void region and the Eulerian-domain placement are meaningless),
        and refresh the preview + derived-quantities panel."""
        is_lagrangian = (self.cfg.analysis.formulation == "Lagrangian")
        self._euler_group.setVisible(not is_lagrangian)
        self._refresh()
