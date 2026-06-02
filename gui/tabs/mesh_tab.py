# -*- coding: utf-8 -*-
"""
Mesh tab.

Owns the mesh-seed parameters (elem_size, discretize) and shows a dedicated
geometry preview with the mesh overlay always on.

The Geometry tab keeps its own optional `Show mesh in preview` toggle for
convenience, but elem_size and discretize are edited here.

ROI refinement (bias seeds in the bounding box) is sketched as a disabled
group so the user can see where it will go; the actual implementation is
deferred until the Abaqus generator can consume it.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSplitter, QGridLayout,
    QTabWidget, QComboBox, QFrame, QScrollArea,
)

from gui.core.model_config import ModelConfig
from gui.widgets.param_field import NumField, BoolField
from gui.widgets.geometry_preview import GeometryPreview


def _section_header(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "background-color: #e8eef5; color: #1f4060; "
        "font-weight: bold; padding: 3px 6px; "
        "border-left: 3px solid #1f6fb2;"
    )
    return lbl


class MeshTab(QWidget):
    """Editor for the mesh-related sub-fields of ModelConfig.

    Emits `meshChanged()` whenever a parameter changes, so MainWindow can
    refresh the Geometry preview (which shows the same numbers) and flip
    the dirty bit."""

    meshChanged = Signal()

    def __init__(self, cfg: ModelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        # --- two-column splitter: form on the left, preview on the right ---
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        # --- left: parameter form, wrapped in a scroll area so the user
        #     can resize the splitter / window without crushing the rows ---
        form = QWidget()
        form_lay = QVBoxLayout(form)
        form_lay.setContentsMargins(8, 8, 8, 8)
        form_lay.setSpacing(8)
        form_lay.addWidget(self._build_seeds_group())
        form_lay.addWidget(self._build_element_type_group())
        form_lay.addWidget(self._build_roi_refine_group())
        form_lay.addWidget(self._build_info_panel())
        form_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # The scroll bar should appear only when the contents don't fit
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(form)
        scroll.setMinimumWidth(380)
        splitter.addWidget(scroll)

        # --- right: dedicated preview with mesh always on ---
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.preview = GeometryPreview()
        right_lay.addWidget(self.preview)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 900])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # Initial render: sync element-type sub-tab state (grey-outs +
        # element-string labels) and visibility (Eulerian hidden in
        # Lagrangian, Workpiece hidden in CEL) before the first preview
        # refresh.
        self._sync_subtab_visibility()
        for body_key in ("tool", "eulerian", "workpiece"):
            self._refresh_element_subtab_state(body_key)
        self._refresh()

    # =====================================================================
    # Sub-groups
    # =====================================================================
    def _build_seeds_group(self) -> QGroupBox:
        g = QGroupBox("Global mesh seeds")
        lay = QVBoxLayout(g)

        lay.addWidget(_section_header("Element size"))
        eg = self.cfg.euler_geometry

        self.f_es = NumField("elem_size", self.cfg.elem_size, "mm",
                             minimum=1e-9)
        self.f_es.setToolTip(
            "Target element edge length. Used as the global seed size.\n"
            "Smaller = finer mesh, higher element count, smaller Δt stable.\n"
            "Typical for orthogonal cutting: 0.5 to 5 % of the chip thickness."
        )
        lay.addWidget(self.f_es)

        self.f_disc = BoolField(
            "Discretize dims to elem_size (floor)", eg.discretize,
        )
        self.f_disc.setToolTip(
            "If checked: workpiece/void dimensions are floored to a multiple\n"
            "of elem_size before meshing, so element sizes match elem_size\n"
            "exactly in both directions.\n"
            "If unchecked: Abaqus rounds the seed count per edge; actual\n"
            "element size may differ slightly (see Derived quantities)."
        )
        lay.addWidget(self.f_disc)

        for w in (self.f_es, self.f_disc):
            w.valueChanged.connect(self._on_change)
        return g

    # =====================================================================
    # Element type editor (Tool / Eulerian / Workpiece)
    # =====================================================================
    HOURGLASS_OPTIONS = [
        ("Default",          "default"),
        ("Relax stiffness",  "relax_stiffness"),
        ("Stiffness",        "stiffness"),
        ("Viscous",          "viscous"),
        ("Combined",         "combined"),
    ]

    def _build_element_type_group(self) -> QGroupBox:
        """Section for per-body element-type controls. Three sub-tabs
        (Tool / Eulerian / Workpiece). The element-type string at the
        bottom of each sub-tab updates live as the user toggles the
        options.

        The Eulerian sub-tab is hidden in Lagrangian analyses; the
        Workpiece sub-tab is hidden in CEL analyses (the workpiece is
        only a reference geometry there)."""
        g = QGroupBox("Element type")
        lay = QVBoxLayout(g)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(2)

        self._elem_widgets: dict[str, dict] = {}
        # Keep a reference + the widgets themselves so we can toggle
        # tab visibility based on cfg.analysis.formulation.
        self._sub_tabs       = QTabWidget()
        self._sub_tab_pages: dict[str, QWidget] = {}
        for body_key, label in (("tool",      "Tool"),
                                 ("eulerian",  "Eulerian"),
                                 ("workpiece", "Workpiece")):
            inner = self._build_element_subtab(body_key)
            # Wrap each page in its own scroll area so the lagrangian
            # sub-tab (which has many fields) doesn't crush the rows when
            # the window is short. The Eulerian sub-tab is short and won't
            # scroll in practice.
            page_scroll = QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setFrameShape(QFrame.Shape.NoFrame)
            page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            page_scroll.setWidget(inner)
            self._sub_tab_pages[body_key] = page_scroll
            self._sub_tabs.addTab(page_scroll, label)
        # Give the tab widget a sensible minimum height so the lagrangian
        # sub-tab is not crushed by default — the scroll area kicks in if
        # the window is shorter than this.
        self._sub_tabs.setMinimumHeight(420)
        lay.addWidget(self._sub_tabs)
        return g

    def _sync_subtab_visibility(self):
        """Hide the Eulerian sub-tab in Lagrangian analyses, the Workpiece
        sub-tab in CEL analyses. We do this by removing+inserting tabs in
        order so the user always sees a consistent left-to-right order:
            CEL        : Tool, Eulerian
            Lagrangian : Tool, Workpiece
        """
        is_lagrangian = (self.cfg.analysis.formulation == "Lagrangian")

        # Snapshot what we want to keep, in order
        if is_lagrangian:
            wanted = [("tool", "Tool"), ("workpiece", "Workpiece")]
        else:
            wanted = [("tool", "Tool"), ("eulerian", "Eulerian")]

        # Remove EVERY current tab without deleting the page widgets
        while self._sub_tabs.count() > 0:
            self._sub_tabs.removeTab(0)

        # Re-add the wanted ones in order. removeTab doesn't destroy the
        # page widget, so the cached references in _sub_tab_pages are
        # still valid and editable.
        for body_key, label in wanted:
            self._sub_tabs.addTab(self._sub_tab_pages[body_key], label)

    def _build_element_subtab(self, body_key: str) -> QWidget:
        """Build the sub-tab for a body. Eulerian has a minimal set of
        controls; Lagrangian-family bodies (Tool / Workpiece) get the
        full set matching Abaqus's Element Type dialog for explicit
        C3D8T / C3D8RT."""
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        # Tight margins/spacing: the section is already framed by the
        # parent group box and the sub-tab bar, no need for extra padding.
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        cfg_elem = self._body_element_cfg(body_key)
        is_eulerian = (body_key == "eulerian")

        widgets: dict = {}

        # ===== Eulerian-only: thermally-coupled toggle =====
        # The explicit family (Tool/Workpiece) is ALWAYS thermally coupled
        # in this analysis (no C3D8, only C3D8T / C3D8RT), so the toggle
        # is only meaningful for the Eulerian box.
        if is_eulerian:
            cb_thermal = BoolField("Thermally coupled",
                                    cfg_elem.thermally_coupled)
            cb_thermal.setToolTip(
                "Coupled temperature-displacement Eulerian element:\n"
                "  ON  -> EC3D8RT (temperature DOFs)\n"
                "  OFF -> EC3D8R  (mechanical only)"
            )
            cb_thermal.valueChanged.connect(self._on_change)
            v.addWidget(cb_thermal)
            widgets["thermal"] = cb_thermal

        # ===== Lagrangian-only: reduced integration toggle =====
        # Drives the C3D8T  vs  C3D8RT  choice for explicit-family solids.
        # When OFF, hourglass / kinematic-split / scaling factors are all
        # disabled (Abaqus does the same in the dialog).
        if not is_eulerian:
            cb_ri = BoolField("Reduced integration",
                               cfg_elem.reduced_integration)
            cb_ri.setToolTip(
                "Reduced integration:\n"
                "  ON  -> C3D8RT (1 integration point, hourglass control)\n"
                "  OFF -> C3D8T  (full integration, no hourglass)"
            )
            cb_ri.valueChanged.connect(self._on_change)
            v.addWidget(cb_ri)
            widgets["reduced_integration"] = cb_ri

        # ===== Common: second-order accuracy =====
        cb_so = BoolField("Second-order accuracy", cfg_elem.second_order_accuracy)
        cb_so.setToolTip(
            "Use second-order accurate kinematics for the element. Leave OFF\n"
            "for typical chip-formation runs."
        )
        cb_so.valueChanged.connect(self._on_change)
        v.addWidget(cb_so)
        widgets["second_order"] = cb_so

        # ===== Lagrangian-only: kinematic split =====
        # (Only meaningful for reduced-integration; greyed-out otherwise.)
        if not is_eulerian:
            cb_ksplit = self._radio_row(
                "Kinematic split:",
                [("Average strain", "average_strain"),
                 ("Orthogonal",     "orthogonal"),
                 ("Centroid",       "centroid")],
                cfg_elem.kinematic_split,
            )
            v.addWidget(cb_ksplit.container)
            widgets["kinematic_split"] = cb_ksplit

        # ===== Common: hourglass control =====
        hg_row = QHBoxLayout()
        hg_row.addWidget(QLabel("Hourglass control:"))
        cb_hg = QComboBox()
        for label, value in self.HOURGLASS_OPTIONS:
            cb_hg.addItem(label, value)
        idx = cb_hg.findData(cfg_elem.hourglass_control)
        if idx >= 0:
            cb_hg.setCurrentIndex(idx)
        cb_hg.setToolTip(
            "Hourglass-control formulation:\n"
            "  Default  : Abaqus picks (uses linear+quad bulk viscosity only)\n"
            "  Combined : enables all parameters including stiffness-viscous weight\n"
            "  Stiffness / Relax stiffness / Viscous: use all params EXCEPT\n"
            "    the stiffness-viscous weight factor."
        )
        cb_hg.currentIndexChanged.connect(self._on_change)
        hg_row.addWidget(cb_hg, stretch=1)
        hg_w = QWidget(); hg_w.setLayout(hg_row)
        v.addWidget(hg_w)
        widgets["hourglass"] = cb_hg
        widgets["hourglass_row"] = hg_w   # to grey out the whole row

        # ===== Lagrangian-only: distortion control =====
        if not is_eulerian:
            cb_dist = self._radio_row(
                "Distortion control:",
                [("Use default", "use_default"),
                 ("Yes",         "yes"),
                 ("No",          "no")],
                cfg_elem.distortion_control_mode,
            )
            v.addWidget(cb_dist.container)
            widgets["distortion_control"] = cb_dist

            # Length ratio (only when 'yes')
            f_lr = NumField("Length ratio", cfg_elem.length_ratio, "",
                             minimum=0.0, maximum=1.0, compact=True)
            f_lr.valueChanged.connect(self._on_change)
            v.addWidget(f_lr)
            widgets["length_ratio"] = f_lr

            # Element deletion
            cb_del = self._radio_row(
                "Element deletion:",
                [("Use default", "use_default"),
                 ("Yes",         "yes"),
                 ("No",          "no")],
                cfg_elem.element_deletion_mode,
            )
            v.addWidget(cb_del.container)
            widgets["element_deletion"] = cb_del

            # Max degradation (use_default / specify + value)
            cb_md, f_md = self._make_default_specify_row(
                "Max degradation:", cfg_elem.max_degradation_mode,
                cfg_elem.max_degradation_value,
                minimum=0.0, maximum=1.0,
            )
            v.addWidget(cb_md.container)
            v.addWidget(f_md)
            widgets["max_degradation"]       = cb_md
            widgets["max_degradation_value"] = f_md

            # Linear kinematic conversion
            cb_lkc, f_lkc = self._make_default_specify_row(
                "Linear kinematic conversion:",
                cfg_elem.linear_kinematic_conversion_mode,
                cfg_elem.linear_kinematic_conversion_value,
                minimum=0.0,
            )
            v.addWidget(cb_lkc.container)
            v.addWidget(f_lkc)
            widgets["linear_kinematic_conversion"]       = cb_lkc
            widgets["linear_kinematic_conversion_value"] = f_lkc

        # ===== Common: scaling factors grid =====
        v.addWidget(_section_header("Scaling factors"))
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6); grid.setVerticalSpacing(2)
        f_disp = NumField("Displacement hourglass",
                          cfg_elem.displacement_hourglass_scale_factor, "",
                          minimum=0.0, compact=True)
        f_linbv = NumField("Linear bulk viscosity",
                           cfg_elem.linear_bulk_viscosity_scale_factor, "",
                           minimum=0.0, compact=True)
        f_qbv  = NumField("Quadratic bulk viscosity",
                          cfg_elem.quadratic_bulk_viscosity_scale_factor, "",
                          minimum=0.0, compact=True)
        f_svw  = NumField("Stiffness-viscous weight",
                          cfg_elem.stiffness_viscous_weight_factor, "",
                          minimum=0.0, maximum=1.0, compact=True)

        grid.addWidget(f_disp,  0, 0)
        grid.addWidget(f_linbv, 0, 1)
        grid.addWidget(f_qbv,   1, 0)
        grid.addWidget(f_svw,   1, 1)
        v.addLayout(grid)
        for ww in (f_disp, f_linbv, f_qbv, f_svw):
            ww.valueChanged.connect(self._on_change)
        widgets.update({
            "disp_scale":  f_disp,
            "linbv_scale": f_linbv,
            "qbv_scale":   f_qbv,
            "svw_scale":   f_svw,
        })

        # ===== Live element-string preview =====
        lbl_elem = QLabel()
        lbl_elem.setStyleSheet(
            "QLabel { padding: 4px 6px; background: #f0f4f8; "
            "border: 1px solid #c0c8d0; color: #1f4060; "
            "font-family: Consolas, monospace; }"
        )
        v.addWidget(lbl_elem)
        widgets["elem_string"] = lbl_elem

        self._elem_widgets[body_key] = widgets
        return wrap

    # ---------------------------------------------------------------
    # Helpers for the Lagrangian sub-tabs
    # ---------------------------------------------------------------
    def _radio_row(self, label: str, options: list, initial: str):
        """Build a horizontal radio-button row matching Abaqus's dialog.

        Returns an object with:
          - .container : QWidget (the row, ready to addWidget)
          - .currentData() : returns the selected value
          - .setCurrentData(v) : programmatic update
          - .valueChanged : QButtonGroup.idClicked signal proxy
                           Wired in the caller to self._on_change.
        """
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel(label))
        group = QButtonGroup(container)
        buttons = []
        for (lbl, value) in options:
            rb = QRadioButton(lbl)
            rb.setProperty("data_value", value)
            if value == initial:
                rb.setChecked(True)
            group.addButton(rb)
            buttons.append(rb)
            row.addWidget(rb)
        row.addStretch()

        class _RadioRow:
            def __init__(_self):
                _self.container = container
                _self.group = group
                _self.buttons = buttons
            def currentData(_self):
                for b in _self.buttons:
                    if b.isChecked():
                        return b.property("data_value")
                return None
            def setCurrentData(_self, v):
                for b in _self.buttons:
                    if b.property("data_value") == v:
                        b.setChecked(True)
                        return
            def setEnabled(_self, enabled):
                _self.container.setEnabled(enabled)
            def blockSignals(_self, b):
                for btn in _self.buttons:
                    btn.blockSignals(b)

        rr = _RadioRow()
        # Wire each button to self._on_change
        for b in buttons:
            b.toggled.connect(self._on_change)
        return rr

    def _make_default_specify_row(self, label: str,
                                   mode_initial: str, value_initial: float,
                                   minimum: float = 0.0, maximum: float = 1e9):
        """Build a 'Use default / Specify' radio row + the value NumField
        that is enabled only when 'Specify' is selected. Returns
        (radio_row, num_field)."""
        rr = self._radio_row(
            label,
            [("Use default", "use_default"),
             ("Specify",     "specify")],
            mode_initial,
        )
        f_val = NumField("Value", value_initial, "",
                         minimum=minimum, maximum=maximum, compact=True)
        f_val.valueChanged.connect(self._on_change)
        return rr, f_val

    def _body_element_cfg(self, body_key: str):
        """Resolve a body_key ('tool'/'eulerian'/'workpiece') to its
        MeshElementCfg instance on self.cfg."""
        if body_key == "tool":      return self.cfg.tool_element
        if body_key == "eulerian":  return self.cfg.euler_element
        return self.cfg.wp_element

    def _element_string_for(self, body_key: str) -> str:
        """Compute the Abaqus element string for the body, given cfg.

        Eulerian:   EC3D8R  /  EC3D8RT  (depending on thermally_coupled)
        Lagrangian: C3D8T   /  C3D8RT   (always thermally coupled in this
                                          analysis; reduced_integration
                                          adds the R)
        """
        cfg_elem = self._body_element_cfg(body_key)
        if body_key == "eulerian":
            suffix = "T" if cfg_elem.thermally_coupled else ""
            return f"EC3D8R{suffix}"
        # Lagrangian-family: always thermally coupled
        return "C3D8RT" if cfg_elem.reduced_integration else "C3D8T"

    def _refresh_element_subtab_state(self, body_key: str):
        """Update grey-outs + element-string label based on the current
        widget values."""
        w = self._elem_widgets[body_key]
        is_eulerian = (body_key == "eulerian")

        # Hourglass-controlled grey-outs (scaling factors)
        mode = w["hourglass"].currentData()
        # Default       -> only linear/quad bulk viscosity active
        # Combined      -> all four scale factors active
        # other (stiff/relax/visc) -> all except svw
        disp_on  = (mode != "default")
        svw_on   = (mode == "combined")

        # On Lagrangian sub-tabs, the WHOLE hourglass section and the
        # whole scaling-factors / kinematic-split block is disabled if
        # reduced_integration is OFF (because C3D8T has no hourglass).
        if is_eulerian:
            hg_active = True
        else:
            hg_active = w["reduced_integration"].value()

        # Hourglass combo row
        w["hourglass_row"].setEnabled(hg_active)
        # Scaling factors
        w["disp_scale"].setEnabled(hg_active and disp_on)
        w["linbv_scale"].setEnabled(hg_active)
        w["qbv_scale"].setEnabled(hg_active)
        w["svw_scale"].setEnabled(hg_active and svw_on)

        # Lagrangian-only sub-elements
        if not is_eulerian:
            # Kinematic split also requires reduced integration
            w["kinematic_split"].setEnabled(hg_active)
            # Length ratio only enabled when distortion control == "yes"
            dc_mode = w["distortion_control"].currentData()
            w["length_ratio"].setEnabled(dc_mode == "yes")
            # Specify-value fields enabled only when their mode is "specify"
            w["max_degradation_value"].setEnabled(
                w["max_degradation"].currentData() == "specify")
            w["linear_kinematic_conversion_value"].setEnabled(
                w["linear_kinematic_conversion"].currentData() == "specify")

        # Live element-string preview
        elem = self._element_string_for(body_key)
        w["elem_string"].setText(f"Abaqus element type: {elem}")

    def _build_roi_refine_group(self) -> QGroupBox:
        """Placeholder for the future bias-seeded refinement inside the ROI
        bounding box. Disabled so the user knows where it will live."""
        g = QGroupBox("ROI refinement (coming next)")
        g.setEnabled(False)
        lay = QVBoxLayout(g)

        note = QLabel(
            "Refining the mesh inside the ROI bbox (with a bias seed toward\n"
            "the cutting region) is not exposed yet because it requires a\n"
            "matching change in abq_odb_generator.py (extra partition + bias\n"
            "edge seed). The fields will appear here once the generator can\n"
            "consume them."
        )
        note.setStyleSheet("color: #888; font-style: italic;")
        note.setWordWrap(True)
        lay.addWidget(note)
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

        self.lbl_eff_dims_title = QLabel("Effective (h_wp, h_void, l_wp, l_void):")
        self.lbl_nel_title      = QLabel("Eulerian element count (~):")

        grid.addWidget(self.lbl_eff_dims_title,        0, 0)
        grid.addWidget(self.lbl_eff_dims,              0, 1)
        grid.addWidget(QLabel("Effective elem_size:"), 1, 0)
        grid.addWidget(self.lbl_eff_es,                1, 1)
        grid.addWidget(self.lbl_nel_title,             2, 0)
        grid.addWidget(self.lbl_nel,                   2, 1)
        grid.addWidget(QLabel("Stable Δt estimate:"),  3, 0)
        grid.addWidget(self.lbl_dt,                    3, 1)
        grid.setColumnStretch(1, 1)
        return g

    # =====================================================================
    # Sync widgets <-> cfg, then refresh preview & info panel
    # =====================================================================
    def _on_change(self, *_):
        self._pull_from_widgets()
        # Refresh element-type enable/disable + element-string label for
        # each body sub-tab.
        for body_key in ("tool", "eulerian", "workpiece"):
            self._refresh_element_subtab_state(body_key)
        self._refresh()
        self.meshChanged.emit()

    def _pull_from_widgets(self):
        self.cfg.elem_size                 = self.f_es.value()
        self.cfg.euler_geometry.discretize = self.f_disc.value()
        # Per-body element configuration
        for body_key in ("tool", "eulerian", "workpiece"):
            w = self._elem_widgets.get(body_key)
            if not w:
                continue
            cfg_elem = self._body_element_cfg(body_key)
            # Common fields
            cfg_elem.second_order_accuracy = w["second_order"].value()
            cfg_elem.hourglass_control     = w["hourglass"].currentData()
            cfg_elem.displacement_hourglass_scale_factor   = w["disp_scale"].value()
            cfg_elem.linear_bulk_viscosity_scale_factor    = w["linbv_scale"].value()
            cfg_elem.quadratic_bulk_viscosity_scale_factor = w["qbv_scale"].value()
            cfg_elem.stiffness_viscous_weight_factor       = w["svw_scale"].value()
            # Eulerian-only
            if "thermal" in w:
                cfg_elem.thermally_coupled = w["thermal"].value()
            # Lagrangian-only
            if "reduced_integration" in w:
                cfg_elem.reduced_integration = w["reduced_integration"].value()
                cfg_elem.kinematic_split = w["kinematic_split"].currentData()
                cfg_elem.distortion_control_mode = w["distortion_control"].currentData()
                cfg_elem.length_ratio = w["length_ratio"].value()
                cfg_elem.element_deletion_mode = w["element_deletion"].currentData()
                cfg_elem.max_degradation_mode = w["max_degradation"].currentData()
                cfg_elem.max_degradation_value = w["max_degradation_value"].value()
                cfg_elem.linear_kinematic_conversion_mode = \
                    w["linear_kinematic_conversion"].currentData()
                cfg_elem.linear_kinematic_conversion_value = \
                    w["linear_kinematic_conversion_value"].value()

    def _refresh(self):
        # Preview always shows the mesh on this tab.
        self.preview.update_from_config(self.cfg, show_mesh=True)

        # --- Info labels (same logic as GeometryTab's panel) ---
        is_lagrangian = (self.cfg.analysis.formulation == "Lagrangian")
        h_wp, h_void, l_wp, l_void = self.cfg.effective_euler_dims()

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

        es_x, es_y = self.cfg.effective_elem_sizes()
        if self.cfg.euler_geometry.discretize:
            self.lbl_eff_es.setText(f"{es_x:.6g} mm  (= elem_size, dims floored)")
        else:
            if abs(es_x - es_y) < 1e-12:
                if abs(es_x - self.cfg.elem_size) < 1e-12:
                    self.lbl_eff_es.setText(
                        f"{es_x:.6g} mm  (matches elem_size exactly)"
                    )
                else:
                    self.lbl_eff_es.setText(
                        f"{es_x:.6g} mm  (elem_size = {self.cfg.elem_size:g})"
                    )
            else:
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
    # External hooks
    # =====================================================================
    def on_external_change(self):
        """Called by MainWindow when ANOTHER tab edits something that affects
        the mesh display here (geometry dims, formulation, materials...).
        We don't pull from widgets — the cfg is the source of truth — we
        just re-render the preview and re-sync sub-tab visibility (the
        formulation may have flipped CEL <-> Lagrangian)."""
        self._sync_subtab_visibility()
        self._refresh()

    def apply_from_cfg(self):
        """Push cfg values into the widgets (used after Open / New)."""
        self.f_es.set_value(self.cfg.elem_size)
        self.f_disc.set_value(self.cfg.euler_geometry.discretize)
        # Per-body element widgets
        for body_key in ("tool", "eulerian", "workpiece"):
            w = self._elem_widgets.get(body_key)
            if not w:
                continue
            cfg_elem = self._body_element_cfg(body_key)
            # Collect all widgets present in the sub-tab and block their
            # signals while we populate them.
            all_w = [w["second_order"], w["hourglass"],
                     w["disp_scale"], w["linbv_scale"], w["qbv_scale"],
                     w["svw_scale"]]
            if "thermal" in w:
                all_w.append(w["thermal"])
            if "reduced_integration" in w:
                all_w += [w["reduced_integration"], w["kinematic_split"],
                          w["distortion_control"], w["length_ratio"],
                          w["element_deletion"], w["max_degradation"],
                          w["max_degradation_value"],
                          w["linear_kinematic_conversion"],
                          w["linear_kinematic_conversion_value"]]
            for ww in all_w:
                ww.blockSignals(True)
            try:
                # Common
                w["second_order"].set_value(cfg_elem.second_order_accuracy)
                idx = w["hourglass"].findData(cfg_elem.hourglass_control)
                if idx >= 0:
                    w["hourglass"].setCurrentIndex(idx)
                w["disp_scale"].set_value(cfg_elem.displacement_hourglass_scale_factor)
                w["linbv_scale"].set_value(cfg_elem.linear_bulk_viscosity_scale_factor)
                w["qbv_scale"].set_value(cfg_elem.quadratic_bulk_viscosity_scale_factor)
                w["svw_scale"].set_value(cfg_elem.stiffness_viscous_weight_factor)
                # Eulerian-only
                if "thermal" in w:
                    w["thermal"].set_value(cfg_elem.thermally_coupled)
                # Lagrangian-only
                if "reduced_integration" in w:
                    w["reduced_integration"].set_value(cfg_elem.reduced_integration)
                    w["kinematic_split"].setCurrentData(cfg_elem.kinematic_split)
                    w["distortion_control"].setCurrentData(cfg_elem.distortion_control_mode)
                    w["length_ratio"].set_value(cfg_elem.length_ratio)
                    w["element_deletion"].setCurrentData(cfg_elem.element_deletion_mode)
                    w["max_degradation"].setCurrentData(cfg_elem.max_degradation_mode)
                    w["max_degradation_value"].set_value(cfg_elem.max_degradation_value)
                    w["linear_kinematic_conversion"].setCurrentData(
                        cfg_elem.linear_kinematic_conversion_mode)
                    w["linear_kinematic_conversion_value"].set_value(
                        cfg_elem.linear_kinematic_conversion_value)
            finally:
                for ww in all_w:
                    ww.blockSignals(False)
            self._refresh_element_subtab_state(body_key)
        # Sync visibility in case the loaded profile has a different
        # formulation than the previous one.
        self._sync_subtab_visibility()
        self._refresh()
