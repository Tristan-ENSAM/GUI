# -*- coding: utf-8 -*-
"""
Main entry point of the Abaqus pre-processor GUI.

Run with:
    python -m gui.main
or:
    python gui/main.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# allow running as a plain script: `python gui/main.py`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QLabel, QVBoxLayout,
    QMessageBox, QFileDialog,
)

from gui.core.model_config import ModelConfig
from gui.core.preferences import Preferences, load_preferences, save_preferences
from gui.tabs.analysis_tab import AnalysisTab
from gui.tabs.geometry_tab import GeometryTab
from gui.tabs.materials_tab import MaterialsTab
from gui.tabs.interaction_tab import InteractionTab
from gui.tabs.bcs_tab import BCsTab
from gui.tabs.mesh_tab import MeshTab
from gui.tabs.step_tab import StepTab
from gui.tabs.job_tab import JobTab
from gui.tabs.results_tab import ResultsTab
from gui.tabs.sensitivity_tab import SensitivityTab
from gui.widgets.preferences_dialog import PreferencesDialog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_TITLE = "Abaqus Cutting Pre-processor"
FILE_EXT = ".acpf"   # Abaqus Cutting Pre-processor File
FILE_FILTER = f"Profile (*{FILE_EXT} *.json);;All files (*)"


def _placeholder(text: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #888; font-style: italic;")
    lay.addStretch()
    lay.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignHCenter)
    lay.addStretch()
    return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 800)

        # ----- state -----
        self.cfg = ModelConfig()
        self.prefs: Preferences = load_preferences()
        # If the user's preferences say "default to Kelvin", apply that
        # to the freshly-created cfg's UI block. Loaded profiles override
        # this in _rebind_cfg.
        if self.prefs.temp_unit_default == "K":
            self.cfg.ui.temp_unit = "K"
        self._current_path: Path | None = None  # None == untitled
        self._dirty: bool = False               # unsaved changes?

        # ----- tabs -----
        self.analysis_tab    = AnalysisTab(self.cfg)
        self.geometry_tab    = GeometryTab(self.cfg)
        self.materials_tab   = MaterialsTab(self.cfg)
        self.interaction_tab = InteractionTab(self.cfg)
        self.bcs_tab         = BCsTab(self.cfg)
        self.mesh_tab        = MeshTab(self.cfg)
        self.step_tab        = StepTab(self.cfg)
        # JobTab takes a callable so it always sees the freshest prefs
        # (the user may edit them mid-session via Preferences dialog).
        self.job_tab         = JobTab(self.cfg, lambda: self.prefs)
        self.results_tab     = ResultsTab()
        self.sensitivity_tab = SensitivityTab(self.cfg, lambda: self.prefs)

        tabs = QTabWidget()
        tabs.addTab(self.analysis_tab,                                        "Analysis")
        tabs.addTab(self.geometry_tab,                                        "Geometry")
        tabs.addTab(self.materials_tab,                                       "Materials")
        tabs.addTab(self.interaction_tab,                                     "Interaction")
        tabs.addTab(self.bcs_tab,                                             "BCs / ICs")
        tabs.addTab(self.mesh_tab,                                            "Mesh")
        tabs.addTab(self.step_tab,                                            "Step")
        tabs.addTab(self.job_tab,                                             "Job")
        tabs.addTab(self.results_tab,                                         "Results")
        tabs.addTab(self.sensitivity_tab,                                     "Sensitivity")
        tabs.addTab(_placeholder("Optimization — coming later"),              "Optimization")
        self.setCentralWidget(tabs)

        # ----- signal wiring -----
        # Analysis -> Geometry: preview must refresh when formulation changes
        self.analysis_tab.analysisChanged.connect(self.geometry_tab.on_analysis_changed)
        # Analysis -> BCs: the Eulerian inflow/outflow group is irrelevant in
        # Lagrangian mode and must hide accordingly.
        self.analysis_tab.analysisChanged.connect(self.bcs_tab.on_analysis_changed)
        # Analysis -> Mesh: same as Geometry — formulation affects what's
        # drawn (Eulerian box vs workpiece) and the derived-quantities labels.
        self.analysis_tab.analysisChanged.connect(self.mesh_tab.on_external_change)

        # Materials -> Geometry & Mesh: stable_dt_estimate depends on E and ρ
        # of the workpiece material.
        self.materials_tab.materialsChanged.connect(self.geometry_tab._refresh)
        self.materials_tab.materialsChanged.connect(self.mesh_tab.on_external_change)

        # Mesh -> Geometry: when the user edits elem_size or discretize in
        # the Mesh tab, Geometry's preview and derived-quantities panel must
        # reflect the new mesh.
        self.mesh_tab.meshChanged.connect(self.geometry_tab._refresh)

        # Geometry/Materials/Analysis -> BCs preview: when the geometry,
        # materials or formulation changes, the BCs tab's interactive
        # preview must re-render to keep the BC overlay aligned with the
        # current model state.
        self.materials_tab.materialsChanged.connect(self.bcs_tab.on_external_change)

        # All tabs -> dirty flag: any change marks the profile dirty
        self.analysis_tab.analysisChanged.connect(self._mark_dirty)
        self.materials_tab.materialsChanged.connect(self._mark_dirty)
        self.interaction_tab.interactionChanged.connect(self._mark_dirty)
        self.bcs_tab.bcsChanged.connect(self._mark_dirty)
        self.mesh_tab.meshChanged.connect(self._mark_dirty)
        self.step_tab.stepChanged.connect(self._mark_dirty)

        # GeometryTab doesn't emit its own changed signal — we wire all its
        # NumField/BoolField signals here to also flip the dirty bit, AND
        # also nudge the Mesh tab's preview to re-render when geometry
        # dimensions change.
        self._wire_geometry_dirty_signals()
        self._wire_geometry_to_mesh_refresh()

        # ----- menu -----
        self._build_menu()

        # ----- initial title -----
        self._refresh_title()

    # =====================================================================
    # Menu construction
    # =====================================================================
    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")

        act_new = QAction("&New", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self.file_new)
        m_file.addAction(act_new)

        act_open = QAction("&Open...", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.file_open)
        m_file.addAction(act_open)

        m_file.addSeparator()

        act_save = QAction("&Save", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.file_save)
        m_file.addAction(act_save)

        act_save_as = QAction("Save &As...", self)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.triggered.connect(self.file_save_as)
        m_file.addAction(act_save_as)

        m_file.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        # --- Preferences menu ---
        m_pref = mb.addMenu("&Preferences")
        # Quick toggle for the temperature unit (the most-used preference).
        self.act_kelvin = QAction("Display temperatures in &Kelvin", self)
        self.act_kelvin.setCheckable(True)
        self.act_kelvin.setChecked(self.cfg.ui.temp_unit == "K")
        self.act_kelvin.toggled.connect(self._on_temp_unit_toggled)
        m_pref.addAction(self.act_kelvin)
        m_pref.addSeparator()
        # Full preferences dialog (Abaqus paths, default workdir, ...)
        act_settings = QAction("&Settings…", self)
        act_settings.triggered.connect(self._open_preferences_dialog)
        m_pref.addAction(act_settings)

    def _open_preferences_dialog(self):
        """Show the modal Preferences editor and persist on Accept."""
        dlg = PreferencesDialog(self, self.prefs)
        if dlg.exec() == PreferencesDialog.DialogCode.Accepted:
            self.prefs = dlg.result_prefs()
            try:
                save_preferences(self.prefs)
            except Exception as e:
                QMessageBox.warning(
                    self, "Save preferences",
                    f"Failed to save preferences:\n{type(e).__name__}: {e}",
                )

    def _on_temp_unit_toggled(self, checked: bool):
        """User flipped the Kelvin / Celsius toggle. Update cfg + refresh
        the Materials tab so the displayed values follow the new unit.
        This is a UI preference, not a physics change, so we don't mark
        the profile dirty unless the preference is saved with it (which
        it is — see ModelConfig.to_json_dict). For consistency, we DO
        mark dirty; the alternative would be 'persistent file change
        without dirty flag' which is confusing."""
        new_unit = "K" if checked else "C"
        if self.cfg.ui.temp_unit == new_unit:
            return
        self.cfg.ui.temp_unit = new_unit
        self.materials_tab.refresh_temp_unit()
        self.bcs_tab.refresh_temp_unit()
        self._mark_dirty()

    # =====================================================================
    # Dirty tracking
    # =====================================================================
    def _mark_dirty(self, *_):
        if not self._dirty:
            self._dirty = True
            self._refresh_title()

    def _mark_clean(self):
        if self._dirty:
            self._dirty = False
        self._refresh_title()

    def _wire_geometry_dirty_signals(self):
        """Hook every NumField / BoolField in GeometryTab to _mark_dirty.
        We piggyback on the existing `valueChanged` signals — they all
        already drive `_on_change` on the GeometryTab, we just add another
        slot."""
        gt = self.geometry_tab
        all_fields = [
            gt.f_h_tool, gt.f_l_tool, gt.f_r_tool, gt.f_rake, gt.f_clear,
            gt.f_tx, gt.f_ty,
            gt.f_h_wp, gt.f_l_wp, gt.f_wp_x, gt.f_wp_y,
            gt.f_h_void, gt.f_l_void, gt.f_ex, gt.f_ey,
            gt.f_xmin, gt.f_xmax, gt.f_ymin, gt.f_ymax, gt.f_zmin, gt.f_zmax,
        ]
        for f in all_fields:
            f.valueChanged.connect(self._mark_dirty)

    def _wire_geometry_to_mesh_refresh(self):
        """When any GeometryTab field changes (workpiece/void dims, tool
        geometry/position, ROI...), the Mesh tab's AND BCs tab's previews
        must re-render to keep their overlays aligned with the current
        geometry."""
        gt = self.geometry_tab
        all_fields = [
            gt.f_h_tool, gt.f_l_tool, gt.f_r_tool, gt.f_rake, gt.f_clear,
            gt.f_tx, gt.f_ty,
            gt.f_h_wp, gt.f_l_wp, gt.f_wp_x, gt.f_wp_y,
            gt.f_h_void, gt.f_l_void, gt.f_ex, gt.f_ey,
            gt.f_xmin, gt.f_xmax, gt.f_ymin, gt.f_ymax, gt.f_zmin, gt.f_zmax,
        ]
        for f in all_fields:
            f.valueChanged.connect(self.mesh_tab.on_external_change)
            f.valueChanged.connect(self.bcs_tab.on_external_change)

    # =====================================================================
    # Title
    # =====================================================================
    def _refresh_title(self):
        name = self._current_path.name if self._current_path else "Untitled"
        star = "*" if self._dirty else ""
        self.setWindowTitle(f"{star}{name} — {APP_TITLE}")

    # =====================================================================
    # File actions
    # =====================================================================
    def _confirm_discard_changes(self) -> bool:
        """If there are unsaved changes, ask the user before discarding them.
        Returns True if it's safe to proceed (saved, or user chose Discard),
        False if the user cancelled."""
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved changes",
            "The current profile has unsaved changes.\n\nSave them?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self.file_save()         # returns False if Save dialog cancelled
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False  # Cancel

    def file_new(self):
        if not self._confirm_discard_changes():
            return
        self.cfg = ModelConfig()
        # The tabs share the cfg reference — replace it everywhere
        self._rebind_cfg()
        self._current_path = None
        self._mark_clean()

    def file_open(self):
        if not self._confirm_discard_changes():
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open profile", "", FILE_FILTER,
        )
        if not path_str:
            return  # user cancelled
        path = Path(path_str)
        try:
            new_cfg = ModelConfig.load_from(path)
        except Exception as e:
            QMessageBox.critical(
                self, "Cannot open file",
                f"Failed to open {path.name}:\n\n{type(e).__name__}: {e}",
            )
            return
        self.cfg = new_cfg
        self._rebind_cfg()
        self._current_path = path
        self._mark_clean()

    def file_save(self) -> bool:
        """Save to the current path. If no current path, prompt for one.
        Returns True on success (or no-op), False if user cancelled the
        Save As dialog."""
        if self._current_path is None:
            return self.file_save_as()
        try:
            self.cfg.save_to(self._current_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Cannot save file",
                f"Failed to save {self._current_path.name}:\n\n{type(e).__name__}: {e}",
            )
            return False
        self._mark_clean()
        return True

    def file_save_as(self) -> bool:
        suggested = (
            str(self._current_path) if self._current_path
            else f"profile{FILE_EXT}"
        )
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save profile as", suggested, FILE_FILTER,
        )
        if not path_str:
            return False  # user cancelled
        path = Path(path_str)
        # Default extension if user typed none
        if path.suffix == "":
            path = path.with_suffix(FILE_EXT)
        try:
            self.cfg.save_to(path)
        except Exception as e:
            QMessageBox.critical(
                self, "Cannot save file",
                f"Failed to save {path.name}:\n\n{type(e).__name__}: {e}",
            )
            return False
        self._current_path = path
        self._mark_clean()
        return True

    # =====================================================================
    # Cfg re-binding (after New or Open)
    # =====================================================================
    def _rebind_cfg(self):
        """When the cfg is replaced wholesale (New, Open), the existing tabs
        still hold references to the OLD cfg. We update their `.cfg`
        attribute, then call `apply_from_cfg()` on each so the widgets
        reflect the new values."""
        self.analysis_tab.cfg    = self.cfg
        self.geometry_tab.cfg    = self.cfg
        self.materials_tab.cfg   = self.cfg
        self.interaction_tab.cfg = self.cfg
        self.bcs_tab.cfg         = self.cfg
        self.mesh_tab.cfg        = self.cfg
        self.step_tab.cfg        = self.cfg
        self.job_tab.cfg         = self.cfg
        # Sync the Kelvin/Celsius toggle from the loaded ui prefs (block
        # signals so we don't double-fire materialsChanged here).
        if hasattr(self, "act_kelvin"):
            self.act_kelvin.blockSignals(True)
            self.act_kelvin.setChecked(self.cfg.ui.temp_unit == "K")
            self.act_kelvin.blockSignals(False)
        # AnalysisTab refreshes first because GeometryTab.on_analysis_changed
        # depends on cfg.analysis being already-applied. The emit in
        # AnalysisTab.apply_from_cfg will trigger GeometryTab.on_analysis_changed
        # AND BCsTab.on_analysis_changed, so we don't need to manually refresh
        # those group visibilities.
        self.analysis_tab.apply_from_cfg()
        self.geometry_tab.apply_from_cfg()
        self.materials_tab.apply_from_cfg()
        self.interaction_tab.apply_from_cfg()
        self.bcs_tab.apply_from_cfg()
        self.mesh_tab.apply_from_cfg()
        self.step_tab.apply_from_cfg()

    # =====================================================================
    # Close handler — final dirty check
    # =====================================================================
    def closeEvent(self, event):
        # Block close if there are unsaved changes
        if not self._confirm_discard_changes():
            event.ignore()
            return
        # If an Abaqus run is in progress, give the user a chance to
        # confirm — closing the GUI would orphan the child process on
        # some platforms, or terminate it abruptly on others.
        proc = getattr(self.job_tab, "_proc", None)
        if proc is not None:
            from PySide6.QtCore import QProcess
            if proc.state() != QProcess.NotRunning:
                from PySide6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, "Abaqus is running",
                    "An Abaqus run is still in progress. Quitting now will\n"
                    "kill it; the .odb may be left incomplete.\n\nQuit anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return
                proc.kill()
                proc.waitForFinished(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
