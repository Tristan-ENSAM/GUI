# -*- coding: utf-8 -*-
"""
Boundary conditions and initial conditions tab.

Layout: left = parameter form, right = interactive preview (BC | IC toggle).

Left form:
  - Cutting velocity (m/min): editable.
  - Initial Eulerian velocity (m/min): editable, independent from cutting
    velocity.
  - Eulerian inflow/outflow per face (left/right/bottom/top): each face
    has an Enabled checkbox; when unchecked, the mode + inflow + outflow
    combos for that face are hidden.
  - Initial temperature (°C or K depending on Preferences).

Preview: click a face to toggle it in the cutting velocity selection
(BC view only).
"""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QSplitter, QFrame, QButtonGroup, QRadioButton, QScrollArea,
    QCheckBox,
)

from gui.core.model_config import ModelConfig
from gui.core import units
from gui.widgets.param_field import NumField
from gui.widgets.geometry_preview import GeometryPreview


def _section_header(title: str):
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "background-color: #e8eef5; color: #1f4060; "
        "font-weight: bold; padding: 2px 6px; "
        "border-left: 3px solid #1f6fb2;"
    )
    return lbl


class BCsTab(QWidget):
    bcsChanged = Signal()

    BC_MODES = [
        ("Inflow",  "inflow"),
        ("Outflow", "outflow"),
        ("Both",    "both"),
    ]
    INFLOW_OPTIONS = [
        ("FREE — material may freely flow in",        "FREE"),
        ("NONE — face is treated as a wall in",       "NONE"),
        ("VOID — material entering becomes void",     "VOID"),
    ]
    OUTFLOW_OPTIONS = [
        ("FREE — material may freely flow out",          "FREE"),
        ("NONREFLECTING — absorb outgoing waves",        "NONREFLECTING"),
        ("EQUILIBRIUM — far-field equilibrium pressure", "EQUILIBRIUM"),
        ("ZERO_PRESSURE — outflow at p = 0",             "ZERO_PRESSURE"),
        ("NONE — face is treated as a wall out",         "NONE"),
    ]
    EUL_FACES = (
        ("left",   "Left face"),
        ("right",  "Right face"),
        ("bottom", "Bottom face"),
        ("top",    "Top face"),
    )

    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        # --- left: scrollable form ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(360)

        form_wrap = QWidget()
        form_lay = QVBoxLayout(form_wrap)
        form_lay.setContentsMargins(8, 8, 8, 8)
        form_lay.setSpacing(8)
        form_lay.addWidget(self._build_kinematics_group())
        form_lay.addWidget(self._build_eulerian_group())
        form_lay.addWidget(self._build_ic_group())
        form_lay.addStretch()
        scroll.setWidget(form_wrap)
        splitter.addWidget(scroll)

        # --- right: BC/IC toggle + preview ---
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(self._build_view_switch())
        self.preview = GeometryPreview()
        self.preview.picking_enabled = True
        self.preview.bc_view_mode = "BC"
        self.preview.facePicked.connect(self._on_face_picked)
        right_lay.addWidget(self.preview)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 900])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        self._refresh_face_visibility()
        self._refresh_preview()

    # =====================================================================
    # BC/IC switch
    # =====================================================================
    def _build_view_switch(self) -> QWidget:
        """Compact BC | IC radio toggle. Lives in a thin bar above the
        preview; keep it tight so it doesn't steal vertical space."""
        bar = QWidget()
        bar.setMaximumHeight(28)
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 2, 8, 2)
        h.setSpacing(8)
        h.addWidget(QLabel("View:"))
        self.rb_bc = QRadioButton("BC")
        self.rb_ic = QRadioButton("IC")
        self.rb_bc.setToolTip("Show boundary conditions on the preview")
        self.rb_ic.setToolTip("Show initial conditions on the preview")
        self.rb_bc.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_bc)
        grp.addButton(self.rb_ic)
        self.rb_bc.toggled.connect(self._on_view_toggled)
        h.addWidget(self.rb_bc)
        h.addWidget(self.rb_ic)
        h.addStretch()
        return bar

    def _on_view_toggled(self, _checked: bool):
        self.preview.bc_view_mode = "BC" if self.rb_bc.isChecked() else "IC"
        self._refresh_preview()

    # =====================================================================
    # Cutting velocity + initial velocity
    # =====================================================================
    def _build_kinematics_group(self) -> QGroupBox:
        g = QGroupBox("Cutting kinematics")
        v = QVBoxLayout(g)
        v.setSpacing(4)

        v.addWidget(_section_header("Cutting velocity"))
        _sf = units.speed_factor()
        _su = units.speed_unit()
        v_mmin = self.cfg.bcs.cutting_speed / _sf
        self.f_vcut = NumField(
            f"Cutting speed [{_su}]", v_mmin, "",
            minimum=-1e5, maximum=1e5,
        )
        self.f_vcut.setToolTip(
            "Cutting velocity (display unit set in Preferences → Unit "
            "system). Click on Eulerian faces in the preview to toggle them\n"
            "in the application set."
        )
        v.addWidget(self.f_vcut)

        v.addWidget(_section_header("Initial Eulerian velocity (CEL only)"))
        v_init_mmin = self.cfg.bcs.initial_velocity / _sf
        self.f_v_init = NumField(
            f"Initial velocity [{_su}]", v_init_mmin, "",
            minimum=-1e5, maximum=1e5,
        )
        self.f_v_init.setToolTip(
            "Initial velocity applied to the whole Eulerian mesh at t=0.\n"
            "Independent from cutting speed."
        )
        v.addWidget(self.f_v_init)

        self.f_vcut.valueChanged.connect(self._on_change)
        self.f_v_init.valueChanged.connect(self._on_change)
        return g

    # =====================================================================
    # Eulerian BCs (4 faces, each with an Enabled checkbox)
    # =====================================================================
    def _build_eulerian_group(self) -> QGroupBox:
        g = QGroupBox("Eulerian inflow / outflow (CEL only)")
        self._eul_group = g
        v = QVBoxLayout(g)
        v.setSpacing(4)

        self._face_widgets: dict[str, dict] = {}
        for face_key, face_label in self.EUL_FACES:
            v.addWidget(self._build_face_box(face_key, face_label))

        return g

    def _build_face_box(self, face_key: str, face_label: str) -> QWidget:
        """Build a compact box for one face: a checkbox enables BC on this
        face; when checked, the Mode / Inflow / Outflow combos appear
        below."""
        box = QFrame()
        box.setFrameShape(QFrame.Shape.StyledPanel)
        box.setStyleSheet(
            "QFrame { background-color: #fafbfc; border: 1px solid #dde3e8; "
            "border-radius: 3px; padding: 2px; }"
        )
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)

        # Enable checkbox (also acts as the section title)
        enable_cb = QCheckBox(face_label)
        enable_cb.setChecked(
            getattr(self.cfg.bcs, f"face_enabled_{face_key}", False)
        )
        enable_cb.setStyleSheet(
            "QCheckBox { font-weight: bold; color: #1f4060; }"
        )
        enable_cb.toggled.connect(self._on_change)
        lay.addWidget(enable_cb)

        # Mode row
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        mode_row.addWidget(QLabel("Mode:"))
        cb_mode = QComboBox()
        for label, value in self.BC_MODES:
            cb_mode.addItem(label, value)
        idx = cb_mode.findData(getattr(self.cfg.bcs, f"eulerian_bc_mode_{face_key}", "both"))
        if idx >= 0:
            cb_mode.setCurrentIndex(idx)
        cb_mode.currentIndexChanged.connect(self._on_change)
        mode_row.addWidget(cb_mode, stretch=1)
        mode_w = QWidget(); mode_w.setLayout(mode_row)
        lay.addWidget(mode_w)

        # Inflow row
        in_row = QHBoxLayout()
        in_row.setSpacing(4)
        in_row.addWidget(QLabel("In:"))
        cb_in = QComboBox()
        for label, value in self.INFLOW_OPTIONS:
            cb_in.addItem(label, value)
        idx = cb_in.findData(getattr(self.cfg.bcs, f"eulerian_inflow_{face_key}", "FREE"))
        if idx >= 0:
            cb_in.setCurrentIndex(idx)
        cb_in.currentIndexChanged.connect(self._on_change)
        in_row.addWidget(cb_in, stretch=1)
        in_w = QWidget(); in_w.setLayout(in_row)
        lay.addWidget(in_w)

        # Outflow row
        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        out_row.addWidget(QLabel("Out:"))
        cb_out = QComboBox()
        for label, value in self.OUTFLOW_OPTIONS:
            cb_out.addItem(label, value)
        idx = cb_out.findData(getattr(self.cfg.bcs, f"eulerian_outflow_{face_key}", "FREE"))
        if idx >= 0:
            cb_out.setCurrentIndex(idx)
        cb_out.currentIndexChanged.connect(self._on_change)
        out_row.addWidget(cb_out, stretch=1)
        out_w = QWidget(); out_w.setLayout(out_row)
        lay.addWidget(out_w)

        self._face_widgets[face_key] = {
            "enable":      enable_cb,
            "mode":        cb_mode,
            "inflow":      cb_in,
            "outflow":     cb_out,
            "mode_row":    mode_w,
            "inflow_row":  in_w,
            "outflow_row": out_w,
        }
        return box

    # =====================================================================
    # IC group
    # =====================================================================
    def _build_ic_group(self) -> QGroupBox:
        g = QGroupBox("Initial conditions")
        v = QVBoxLayout(g)
        v.setSpacing(4)
        v.addWidget(_section_header("Initial temperature"))
        tu = self.cfg.ui.temp_unit
        gui_val = units.temp_from_abaqus(self.cfg.bcs.ambient_temperature, tu)
        unit_str = "K" if tu == "K" else "°C"
        self.f_T = NumField(
            f"Ambient temperature [{unit_str}]", gui_val, "", decimals=4,
        )
        self.f_T.setToolTip(
            "Initial temperature applied to BOTH the workpiece (Eulerian\n"
            "nodes) and the tool nodes."
        )
        v.addWidget(self.f_T)
        self.f_T.valueChanged.connect(self._on_change)
        return g

    # =====================================================================
    # Hooks
    # =====================================================================
    def on_analysis_changed(self):
        is_lagrangian = (self.cfg.analysis.formulation == "Lagrangian")
        self._eul_group.setVisible(not is_lagrangian)
        self._refresh_preview()

    def refresh_temp_unit(self):
        """Back-compat alias — a temp-base change is a unit-system change."""
        self.refresh_units()

    def refresh_units(self):
        tu = self.cfg.ui.temp_unit
        unit_str = "K" if tu == "K" else "°C"
        self.f_T._lbl.setText(f"Ambient temperature [{unit_str}]")
        gui_val = units.temp_from_abaqus(self.cfg.bcs.ambient_temperature, tu)
        self.f_T.blockSignals(True)
        self.f_T.set_value(gui_val)
        self.f_T.blockSignals(False)
        # Velocity unit may have changed too.
        _sf = units.speed_factor()
        _su = units.speed_unit()
        for w, internal in ((self.f_vcut, self.cfg.bcs.cutting_speed),
                            (self.f_v_init, self.cfg.bcs.initial_velocity)):
            w.blockSignals(True)
            w.set_value(internal / _sf)
            w.blockSignals(False)
        self.f_vcut._lbl.setText(f"Cutting speed [{_su}]")
        self.f_v_init._lbl.setText(f"Initial velocity [{_su}]")
        self._refresh_preview()

    def on_external_change(self):
        self._refresh_preview()

    # =====================================================================
    # Picking (BC view only): toggle face in/out of v_cut selection
    # =====================================================================
    def _on_face_picked(self, face_id: str):
        if self.preview.bc_view_mode != "BC":
            return
        if not face_id.startswith("eul_"):
            return
        current = list(self.cfg.bcs.cutting_velocity_faces or [])
        if face_id in current:
            current.remove(face_id)
        else:
            current.append(face_id)
        self.cfg.bcs.cutting_velocity_faces = current
        self._refresh_preview()
        self.bcsChanged.emit()

    # =====================================================================
    # Sync widgets <-> cfg
    # =====================================================================
    def _refresh_face_visibility(self):
        """Show / hide each face's combos depending on its 'enabled' state.
        Within enabled faces, additionally hide inflow / outflow rows
        according to mode."""
        for face_key, w in self._face_widgets.items():
            enabled = w["enable"].isChecked()
            w["mode_row"].setVisible(enabled)
            mode = w["mode"].currentData() if enabled else "both"
            w["inflow_row"].setVisible(enabled and mode in ("inflow", "both"))
            w["outflow_row"].setVisible(enabled and mode in ("outflow", "both"))

    def _on_change(self, *_):
        self._pull_from_widgets()
        self._refresh_face_visibility()
        self._refresh_preview()
        self.bcsChanged.emit()

    def _refresh_preview(self):
        self.preview.update_from_config(self.cfg, show_mesh=False, show_bcs=True)

    def _pull_from_widgets(self):
        b = self.cfg.bcs
        _sf = units.speed_factor()
        b.cutting_speed    = self.f_vcut.value()   * _sf
        b.initial_velocity = self.f_v_init.value() * _sf
        for face_key, w in self._face_widgets.items():
            setattr(b, f"face_enabled_{face_key}",       w["enable"].isChecked())
            setattr(b, f"eulerian_bc_mode_{face_key}",   w["mode"].currentData())
            setattr(b, f"eulerian_inflow_{face_key}",    w["inflow"].currentData())
            setattr(b, f"eulerian_outflow_{face_key}",   w["outflow"].currentData())
        tu = self.cfg.ui.temp_unit
        b.ambient_temperature = units.temp_to_abaqus(self.f_T.value(), tu)

    def apply_from_cfg(self):
        widgets = [self.f_vcut, self.f_v_init, self.f_T]
        for w in self._face_widgets.values():
            widgets += [w["enable"], w["mode"], w["inflow"], w["outflow"]]
        for w in widgets:
            w.blockSignals(True)
        try:
            b = self.cfg.bcs
            _sf = units.speed_factor()
            _su = units.speed_unit()
            self.f_vcut.set_value(b.cutting_speed / _sf)
            self.f_v_init.set_value(b.initial_velocity / _sf)
            self.f_vcut._lbl.setText(f"Cutting speed [{_su}]")
            self.f_v_init._lbl.setText(f"Initial velocity [{_su}]")
            for face_key, w in self._face_widgets.items():
                w["enable"].setChecked(
                    getattr(b, f"face_enabled_{face_key}", False))
                idx = w["mode"].findData(
                    getattr(b, f"eulerian_bc_mode_{face_key}", "both"))
                if idx >= 0:
                    w["mode"].setCurrentIndex(idx)
                idx = w["inflow"].findData(
                    getattr(b, f"eulerian_inflow_{face_key}", "FREE"))
                if idx >= 0:
                    w["inflow"].setCurrentIndex(idx)
                idx = w["outflow"].findData(
                    getattr(b, f"eulerian_outflow_{face_key}", "FREE"))
                if idx >= 0:
                    w["outflow"].setCurrentIndex(idx)
            tu = self.cfg.ui.temp_unit
            self.f_T.set_value(units.temp_from_abaqus(b.ambient_temperature, tu))
            unit_str = "K" if tu == "K" else "°C"
            self.f_T._lbl.setText(f"Ambient temperature [{unit_str}]")
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.on_analysis_changed()
        self._refresh_face_visibility()
        self._refresh_preview()
