# -*- coding: utf-8 -*-
"""
Model configuration dataclasses.

Mirrors the dict structure expected by abq_odb_generator.py
(MODEL_CFG keys: process, tool, euler, bbox).
"""
from __future__ import annotations
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from pathlib import Path

from gui.core.unit_system import UnitSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discretize(dim: float, element_size: float) -> float:
    """Floor `dim` to the nearest multiple of `element_size`, Decimal-safe.
    Mirrors the function in abq_odb_generator.py so the GUI preview matches
    exactly what Abaqus will build."""
    d = Decimal(str(dim))
    es = Decimal(str(element_size))
    if es <= 0:
        raise ValueError("element_size must be > 0")
    n = d // es
    if n <= 0:
        return 0.0
    return float(n * es)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------
@dataclass
class ProcessCfg:
    # `cutting_speed` is the conceptual cutting velocity of the process
    # (m/min → mm/s). The actual VelocityBC on the Eulerian face uses
    # `bcs.cutting_speed`, which can be set independently from this
    # process-level value if the user wants (in practice we keep them
    # in sync).
    cutting_speed: float = 1000.0   # mm/s


@dataclass
class OutputCfg:
    """Field-output and history-output variable selection.

    Each flag corresponds to one Abaqus output identifier. The Job
    generator only writes the IDs whose flag is True, so unwanted
    variables aren't dumped to the .odb (faster IO, smaller files).

    Mechanical / continuum (element-based):
      S       : stress tensor
      PEEQ    : equivalent plastic strain
      VP      : viscoplastic strain
      P       : pressure (hydrostatic)
      ERV     : equivalent von Mises
    Thermal:
      TEMP    : nodal temperature (output as element-averaged)
      HFL     : heat flux vector
      HP      : heat-power per unit volume
    Eulerian-specific:
      EVF     : element volume fraction
      MFL     : mass flux
      A       : nodal acceleration
      V       : nodal velocity
    Damage / failure:
      DMICRT  : damage initiation criterion
      SDEG    : stiffness degradation
      STATUS  : element status (1 = active, 0 = deleted)
      SDV     : solution-dependent state variables
    Contact:
      CSTRESS : contact stresses (CPRESS + CSHEAR)
    Reaction / kinematics on nodes (always useful):
      U       : nodal displacement
      RF      : nodal reaction force
      NT      : nodal temperature

    History output: by default we keep RF1/RF2 on the tool RP (so the
    user can plot the cutting forces) and the PRESELECT history.
    """
    # --- field output ---
    fo_S:       bool = True
    fo_PEEQ:    bool = True
    fo_VP:      bool = True
    fo_P:       bool = True
    fo_ERV:     bool = True
    fo_TEMP:    bool = True
    fo_HFL:     bool = True
    fo_HP:      bool = True
    fo_EVF:     bool = True
    fo_MFL:     bool = True
    fo_A:       bool = True
    fo_V:       bool = True
    fo_DMICRT:  bool = True
    fo_SDEG:    bool = True
    fo_STATUS:  bool = True
    fo_SDV:     bool = True
    fo_CSTRESS: bool = True
    fo_U:       bool = True
    fo_RF:      bool = True
    fo_NT:      bool = True
    # --- history output ---
    ho_preselect:   bool = True   # PRESELECT (also provides ALLKE/ALLIE for
    #                               the mass-scaling guard-rail)
    ho_rf_on_rp:    bool = True   # RF1/RF2 on the tool RP
    # Number of history-output intervals. By design we keep it equal to
    # the number of field-output frames (StepCfg.n_frames): each field
    # frame then has a matching history sample, so history-vs-field
    # correlations (forces vs PEEQ peaks, etc.) plot 1:1 with no
    # interpolation. The actual *number of samples* in the .odb is
    # ho_n_intervals + 1 (Abaqus includes both endpoints).
    # NOTE: previously this was `ho_time_interval` (seconds). The
    # migration in ModelConfig.from_json_dict handles old .acpf files
    # by dropping the old key and falling back to the default here.
    ho_n_intervals: int = 500


@dataclass
class StepCfg:
    """Step-level settings: simulation duration, sampling, outputs and
    mass scaling.

    Mass scaling
    ------------
    `mass_scaling_enabled`: whether to apply the factor to the materials.
    `mass_scaling_factor_eulerian`: multiplied with the Eulerian (workpiece)
        material density at the moment the .inp is written. Cp is divided
        by the same factor so that thermal diffusivity k/(rho*Cp) is
        preserved — only mechanical inertia is artificially boosted, the
        thermal response stays physical.
    `mass_scaling_factor_tool`: same idea for the tool. Often left at 1.0
        (the tool is rigid in CEL, so mass scaling has no effect there).

    The factor is applied directly at material-write time inside
    abq_odb_generator.py:  rho_eff = factor * rho ;  Cp_eff = Cp / factor.
    No *Mass Scaling card is emitted in the .inp — the scaling lives in
    the material definition itself.
    """
    sim_time:        float       = 5e-4
    n_frames:        int         = 500
    output:          OutputCfg   = field(default_factory=OutputCfg)

    # Mass scaling (CEL only — Lagrangian users should use Abaqus's own
    # *Mass Scaling card directly; we don't expose that here yet)
    mass_scaling_enabled:        bool  = False
    mass_scaling_factor_eulerian: float = 1.0
    mass_scaling_factor_tool:    float = 1.0

    # Time scaling (Hammelmüller & Zehetner, COMPLAS XIII). A fictitious
    # time τ = t / κ_t (κ_t > 1) speeds up the explicit solve linearly. To
    # keep the machined length constant the kinematics are accelerated and
    # the step shortened: cutting/initial velocities ×κ_t, sim_time ÷κ_t.
    # To preserve the thermal time constant the specific heat is divided by
    # κ_t as well (on top of any mass scaling). ρ and E are NOT changed by
    # time scaling. Inertial-force effect equals mass scaling with κ_m=κ_t².
    # VALIDITY: only when strain-rate dependence is negligible — with a
    # rate-dependent Johnson-Cook law (C≠0) the rate terms are distorted.
    # The velocity/time scaling is applied in ModelConfig.to_params_dict;
    # run_simul.py applies κ_t to Cp only (velocities/time arrive scaled).
    time_scaling_enabled:        bool  = False
    time_scaling_factor:         float = 1.0


@dataclass
class AnalysisCfg:
    """High-level analysis-type settings.

    `formulation` switches between two Abaqus generator scripts:
      - "CEL"        -> abq_odb_generator.py        (Eulerian workpiece,
                                                     ExplicitDynamicsStep,
                                                     VolFraction predefined field)
      - "Lagrangian" -> abq_lagrangian_generator.py (deformable workpiece mesh,
                                                     TempDisplacementDynamicsStep,
                                                     element deletion via JC damage)

    The remaining fields are only used in Lagrangian mode (ignored in CEL):

      - `tool_motion`     : which body moves.
            "tool_moves"      : workpiece bottom fixed, cutting_speed applied
                                to the tool RP (the workpiece stays still).
            "workpiece_moves" : tool RP fixed, cutting_speed applied to the
                                workpiece bottom (same kinematics as the
                                current CEL setup).
      - `tool_rigid`      : if True, the tool is a RigidBody driven by the RP
                            (same as your CEL); if False, the tool is a
                            deformable elastic body (allows realistic tool
                            heating, but ~2-3x slower).
      - `element_deletion`: if True, fully-damaged elements are removed from
                            the mesh, which lets the chip form by material
                            separation (continuum approach). Disable only for
                            debugging — without deletion the workpiece will
                            entangle.
      - `rp_location`     : which corner of the tool carries the Reference
                            Point. Default "TR" (top-right): far from the
                            plastically active cutting edge, numerically
                            cleaner for applying the velocity BC.
    """
    formulation:      str  = "CEL"           # "CEL" | "Lagrangian"
    tool_motion:      str  = "workpiece_moves"  # "tool_moves" | "workpiece_moves"
    tool_rigid:       bool = True
    element_deletion: bool = True
    rp_location:      str  = "TR"            # "TR" | "BR" | "centroid"


@dataclass
class UICfg:
    """User-interface preferences that don't affect the physics but do affect
    how values are displayed in the GUI (and saved alongside the profile so
    a reopened file uses the same display settings).

    `temp_unit` switches the display of Tm, Tr and similar temperature fields
    between Celsius and Kelvin. The internal Abaqus value is always in °C
    (matching abq_odb_generator.py's expectations)."""
    temp_unit: str = "C"     # "C" | "K"


@dataclass
class InteractionCfg:
    """Contact / interaction parameters for the tool-workpiece pair.

    Mirrors what abq_odb_generator.py builds via `ContactProperty`:
        - TangentialBehavior(formulation, table=((mu,),), fraction=slip_frac)
        - NormalBehavior(pressureOverclosure=HARD)
        - (optional) HeatGeneration(slaveFraction, masterFraction)

    The mapping from GUI strings to Abaqus symbolic constants happens at
    serialisation time (in the Abaqus generator).
    """
    # --- Tangential behaviour ---
    tangential_formulation: str   = "penalty"     # "penalty" | "rough" | "frictionless"
    friction_coeff:         float = 0.3           # μ (only used if formulation == "penalty")
    slip_tolerance:         float = 0.005         # `fraction` arg to TangentialBehavior

    # --- Normal behaviour ---
    pressure_overclosure: str = "hard"            # "hard" | "exponential" | "linear"
    # (parameters of soft overclosure are exposed later if you need them)

    # --- Heat generation (friction-to-thermal coupling) ---
    heat_generation:          bool  = False        # turn the *Gap heat generation on
    heat_fraction_to_slave:   float = 0.5
    heat_fraction_to_master:  float = 0.5


@dataclass
class BCsCfg:
    """Boundary conditions and initial conditions.

    Cutting velocity (`cutting_speed`) is applied to the faces listed in
    `cutting_velocity_faces` (subset of the 4 Eulerian faces). Initial
    Eulerian velocity (`initial_velocity`) is an independent value
    applied as a predefined velocity field on the whole Eulerian mesh.
    Both stored in mm/s; the BCs tab converts to/from m/min for display.

    Each Eulerian face also has:
      - face_enabled_{side}: bool. When True, this face carries an
        inflow/outflow EulerianBC; when False, the face has no BC and
        the inflow/outflow combos are hidden in the UI.
      - eulerian_bc_mode_{side}: "inflow" | "outflow" | "both"
      - eulerian_inflow_{side}, eulerian_outflow_{side}: Abaqus symbolic
        constant names.

    Defaults: all face_enabled_* are False (the user explicitly enables
    each face).

    `ambient_temperature` is the initial temperature applied to both
    workpiece (Eulerian) nodes and tool nodes. Stored in °C.
    """
    # Cutting kinematics
    cutting_speed:          float    = 1000.0
    cutting_velocity_faces: list     = field(default_factory=lambda: ["eul_bot"])

    # Initial Eulerian velocity (CEL only) — independent from cutting_speed
    initial_velocity:       float    = 1000.0    # mm/s

    # Eulerian BCs (CEL only) — 4 faces, each independently enabled
    face_enabled_left:        bool = False
    face_enabled_right:       bool = False
    face_enabled_top:         bool = False
    face_enabled_bottom:      bool = False
    eulerian_bc_mode_left:    str  = "both"
    eulerian_bc_mode_right:   str  = "both"
    eulerian_bc_mode_top:     str  = "both"
    eulerian_bc_mode_bottom:  str  = "both"
    eulerian_inflow_left:     str  = "FREE"
    eulerian_outflow_left:    str  = "FREE"
    eulerian_inflow_right:    str  = "FREE"
    eulerian_outflow_right:   str  = "FREE"
    eulerian_inflow_top:      str  = "FREE"
    eulerian_outflow_top:     str  = "FREE"
    eulerian_inflow_bottom:   str  = "FREE"
    eulerian_outflow_bottom:  str  = "FREE"

    # Initial conditions
    ambient_temperature: float = 20.0   # °C (Abaqus internal)


@dataclass
class ToolGeometry:
    h_tool:      float = 0.30
    l_tool:      float = 0.15
    r_tool:      float = 0.01
    rake_angle:  float = 0.0      # deg
    clear_angle: float = 5.0      # deg


@dataclass
class ToolPosition:
    x0: float = 0.0
    y0: float = -0.05


@dataclass
class EulerGeometry:
    h_wp:       float = 0.20
    h_void:     float = 0.20
    l_wp:       float = 0.20
    l_void:     float = 0.20
    discretize: bool  = True


@dataclass
class EulerPosition:
    x0: float = 0.0
    y0: float = 0.0


@dataclass
class BBox:
    # Extraction ROI / ZOI (the geometry tab's "ROI" group edits this).
    # Default = the cutting zone of interest, so runs extract only this
    # region (faster, smaller .npz, more relevant for sensitivity/ID).
    # Widen these in the GUI to extract the full Eulerian domain.
    xmin: float = -0.25
    xmax: float = 0.05
    ymin: float = -0.25
    ymax: float = 0.05
    zmin: float = -0.001
    zmax: float = 0.001


@dataclass
class MeshElementCfg:
    """Per-body element-type configuration.

    Two flavours are supported, picked by family:
      - Eulerian (CEL volume): EC3D8R or EC3D8RT depending on
        `thermally_coupled`.
      - Explicit/Lagrangian (rigid tool, Lagrangian workpiece): in this
        GUI the explicit-family solids are ALWAYS thermally coupled —
        Abaqus has no `C3D8` in our analysis path. So the type is
        `C3D8T` (full integration) or `C3D8RT` (reduced integration)
        depending on `reduced_integration`. `thermally_coupled` is
        ignored for the explicit family but kept on the dataclass to
        keep the storage uniform.

    Common fields:
      thermally_coupled    : Eulerian only (use EC3D8RT vs EC3D8R).
      second_order_accuracy: Abaqus 'Second-order accuracy' Yes/No.
      hourglass_control    : default | relax_stiffness | stiffness |
                             viscous | combined
                             Active scaling factors:
                               default       -> lin+quad bulk viscosity
                               combined      -> all four scaling factors
                               other         -> all except svw
      displacement_hourglass_scale_factor
      linear_bulk_viscosity_scale_factor
      quadratic_bulk_viscosity_scale_factor
      stiffness_viscous_weight_factor   (only with 'combined' hourglass)

    Lagrangian-only fields (explicit family — Tool / Lagrangian workpiece):
      reduced_integration  : True  -> C3D8RT  (with hourglass control)
                             False -> C3D8T   (full integration; no
                                              hourglass, no kinematic
                                              split required)
      kinematic_split      : "average_strain" | "orthogonal" | "centroid"
                             (only meaningful for reduced-integration)
      distortion_control_mode: "use_default" | "yes" | "no"
      length_ratio          : float (only used when distortion_control_mode
                                     == "yes")
      element_deletion_mode : "use_default" | "yes" | "no"
      max_degradation_mode  : "use_default" | "specify"
      max_degradation_value : float (only used when max_degradation_mode
                                     == "specify")
      linear_kinematic_conversion_mode : "use_default" | "specify"
      linear_kinematic_conversion_value: float
    """
    # Common
    thermally_coupled:     bool  = True
    second_order_accuracy: bool  = False
    hourglass_control:     str   = "default"

    displacement_hourglass_scale_factor:   float = 1.0
    linear_bulk_viscosity_scale_factor:    float = 1.0
    quadratic_bulk_viscosity_scale_factor: float = 1.0
    stiffness_viscous_weight_factor:       float = 0.5

    # Lagrangian-only (ignored for Eulerian-family bodies)
    reduced_integration:                bool  = True
    kinematic_split:                    str   = "average_strain"
    distortion_control_mode:            str   = "use_default"
    length_ratio:                       float = 0.1
    element_deletion_mode:              str   = "use_default"
    max_degradation_mode:               str   = "use_default"
    max_degradation_value:              float = 0.0
    linear_kinematic_conversion_mode:   str   = "use_default"
    linear_kinematic_conversion_value:  float = 0.0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class JobCfg:
    """Portable job parameters saved with the profile. The working directory
    is intentionally NOT stored here: it is machine-specific (an absolute
    path) and defaults to the user-level Preferences, so it stays out of the
    portable .acpf project file."""
    job_name: str = "Cutting_job"
    cpus:     int = 4


@dataclass
class ModelConfig:
    """Top-level model config. Materials are kept as raw dicts for now
    (filled by the Materials tab later — defaults below match test.py)."""
    analysis:       AnalysisCfg    = field(default_factory=AnalysisCfg)
    ui:             UICfg          = field(default_factory=UICfg)
    units:          UnitSystem     = field(default_factory=UnitSystem)
    job:            JobCfg         = field(default_factory=JobCfg)
    process:        ProcessCfg     = field(default_factory=ProcessCfg)
    step:           StepCfg        = field(default_factory=StepCfg)
    interaction:    InteractionCfg = field(default_factory=InteractionCfg)
    bcs:            BCsCfg         = field(default_factory=BCsCfg)
    tool_geometry:  ToolGeometry   = field(default_factory=ToolGeometry)
    tool_position:  ToolPosition   = field(default_factory=ToolPosition)
    euler_geometry: EulerGeometry  = field(default_factory=EulerGeometry)
    euler_position: EulerPosition  = field(default_factory=EulerPosition)
    wp_position:    EulerPosition  = field(default_factory=EulerPosition)
    bbox:           BBox           = field(default_factory=BBox)
    elem_size:      float          = 0.005
    # Tool-nose seed size (was hard-coded to 0.001 in run_simul); exposed so
    # the mesh-convergence study can identify it. Backward-compatible default.
    tool_elem_size: float          = 0.001

    # Per-body element configuration
    tool_element:    MeshElementCfg = field(default_factory=MeshElementCfg)
    euler_element:   MeshElementCfg = field(default_factory=MeshElementCfg)
    wp_element:      MeshElementCfg = field(default_factory=MeshElementCfg)

    # Material dicts (placeholders, edited in the Materials tab later).
    # In CEL mode, `euler_material` is the workpiece material (Johnson-Cook).
    # In Lagrangian mode, it is still used as the workpiece material — the
    # name is kept for backwards compat; we expose it via to_params_dict
    # under the appropriate key depending on formulation.
    tool_material: dict = field(default_factory=lambda: {
        "rho": 1.19e-08, "E": 534000.0, "nu": 0.22,
        "k": 50.0, "Cp": 400000000.0, "alpha": 1e-05,
    })
    euler_material: dict = field(default_factory=lambda: {
        "rho": 8.960e-09, "E": 124000.0, "nu": 0.34,
        "k": 386, "Cp": 383000000.0, "alpha": 5.0e-05, "beta": 0.9,
        "A": 90, "B": 292, "n": 0.31, "C": 0.025, "m": 1.09,
        "Tm": 1356, "Tr": 300, "eps_dot0": 1.0,
        "D1": 0.54, "D2": 4.89, "D3": 3.03, "D4": 0.014,
        "D5": 1.12, "eps0": 1.0, "Gf": 5,
    })

    # Convenience alias — same dict as `euler_material` (the workpiece material
    # is the same physical entity in both formulations). Kept as a property to
    # avoid duplicating storage.
    @property
    def workpiece_material(self) -> dict:
        return self.euler_material

    # ----- Serialization -----
    def to_params_dict(self) -> dict:
        """Build the `params` dict sent to run_simul.py, organised as one
        named sub-config per GUI tab (CEL-only build).

        Layout (every value is a plain literal so repr()/literal_eval round
        trips — see run_simul.py's contract):
            analysis     : formulation + analysis options
            geometry     : tool {position, geometry}, euler {position,
                           workpiece_position, geometry}, bbox
            materials    : euler (workpiece material), tool (tool material)
            mesh         : elem_size, discretize, euler_element, tool_element
            interaction  : contact/friction/heat
            bcs          : cutting_speed, initial_velocity, ...
            step         : sim_time, n_frames, mass scaling, output
        `process` was removed: cutting_speed now comes from `bcs`, and
        sim_time / n_frames from `step`.
        """
        return {
            "analysis": asdict(self.analysis),
            "geometry": {
                "tool": {
                    "position": asdict(self.tool_position),
                    "geometry": asdict(self.tool_geometry),
                },
                "euler": {
                    "position":           asdict(self.euler_position),
                    "workpiece_position": asdict(self.wp_position),
                    "geometry":           asdict(self.euler_geometry),
                },
                "bbox": asdict(self.bbox),
            },
            "materials": {
                "euler": dict(self.euler_material),
                "tool":  dict(self.tool_material),
            },
            "mesh": {
                "elem_size":      self.elem_size,
                "tool_elem_size": self.tool_elem_size,
                "discretize":     self.euler_geometry.discretize,
                "euler_element":  asdict(self.euler_element),
                "tool_element":   asdict(self.tool_element),
            },
            "interaction": asdict(self.interaction),
            "bcs":         asdict(self.bcs),
            "step":        asdict(self.step),
        }

    # ----- Profile save/load (JSON) -----
    #
    # The JSON format is independent of `to_params_dict` (which produces an
    # asymmetric CEL-vs-Lagrangian shape for Abaqus). Save/Load must
    # round-trip every field regardless of which formulation is currently
    # active, so we emit a flat, symmetric layout that always carries all
    # values.

    FORMAT_VERSION = 1

    def to_json_dict(self) -> dict:
        """Symmetric dict representation suitable for JSON save/load.
        All fields are always present, regardless of formulation."""
        return {
            "format_version": self.FORMAT_VERSION,
            "saved_at":       datetime.now().isoformat(timespec="seconds"),
            "analysis":       asdict(self.analysis),
            "ui":             asdict(self.ui),
            "units":          self.units.to_dict(),
            "job":            asdict(self.job),
            "process":        asdict(self.process),
            "step":           asdict(self.step),
            "interaction":    asdict(self.interaction),
            "bcs":            asdict(self.bcs),
            "tool": {
                "geometry": asdict(self.tool_geometry),
                "position": asdict(self.tool_position),
                "material": dict(self.tool_material),
            },
            "workpiece": {
                # h_wp / l_wp are workpiece dimensions; h_void / l_void and
                # discretize live with the Eulerian block but historically
                # share the dataclass — we put h_wp/l_wp here for clarity.
                "h_wp":     self.euler_geometry.h_wp,
                "l_wp":     self.euler_geometry.l_wp,
                "position": asdict(self.wp_position),
                "material": dict(self.euler_material),
            },
            "eulerian": {
                "h_void":     self.euler_geometry.h_void,
                "l_void":     self.euler_geometry.l_void,
                "discretize": self.euler_geometry.discretize,
                "position":   asdict(self.euler_position),
            },
            "mesh": {
                "elem_size":     self.elem_size,
                "tool_element":  asdict(self.tool_element),
                "euler_element": asdict(self.euler_element),
                "wp_element":    asdict(self.wp_element),
            },
            "bbox": asdict(self.bbox),
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "ModelConfig":
        """Inverse of `to_json_dict`. Tolerant to missing keys: any field
        that isn't in the JSON keeps its dataclass default."""
        ver = data.get("format_version")
        if ver is not None and ver > cls.FORMAT_VERSION:
            raise ValueError(
                f"File format version {ver} is newer than this app "
                f"(supports up to v{cls.FORMAT_VERSION}). Please update."
            )

        cfg = cls()

        # Helper: set attributes on a dataclass instance from a dict, only
        # for keys that exist on the dataclass (silently drops unknowns).
        def _apply(dc, src: dict | None):
            if not src:
                return
            for k, v in src.items():
                if hasattr(dc, k):
                    setattr(dc, k, v)

        _apply(cfg.analysis,       data.get("analysis"))
        _apply(cfg.ui,             data.get("ui"))
        # Unit system (newer format). Keep it consistent with ui.temp_unit:
        #  - new profiles carry a "units" block -> it drives the temp base;
        #  - legacy profiles only have ui.temp_unit -> seed the unit system
        #    from it, so a saved Kelvin preference still flows through.
        units_data = data.get("units")
        if units_data is not None:
            cfg.units = UnitSystem.from_dict(units_data)
            cfg.ui.temp_unit = cfg.units.temp
        else:
            cfg.units = UnitSystem(temp=cfg.ui.temp_unit)
        _apply(cfg.job, data.get("job"))
        _apply(cfg.process,        data.get("process"))
        # `step` block (newer format). If absent, fall back to the legacy
        # location where sim_time/n_frames lived under `process`.
        step_data = data.get("step")
        if step_data is not None:
            # Apply scalars (sim_time, n_frames) without touching the
            # nested `output` dataclass (otherwise _apply would replace
            # the OutputCfg instance with a raw dict).
            scalar_data = {k: v for k, v in step_data.items() if k != "output"}
            _apply(cfg.step, scalar_data)
            output_data = step_data.get("output")
            if output_data is not None:
                # Migration: `ho_time_interval` (float, seconds) was
                # replaced by `ho_n_intervals` (int, number of intervals)
                # to keep the history sampling in sync with the field
                # frames. Strip the obsolete key so _apply doesn't try
                # to setattr() it on the new dataclass.
                output_data = {k: v for k, v in output_data.items()
                               if k != "ho_time_interval"}
                _apply(cfg.step.output, output_data)
        else:
            # Legacy profile: pull sim_time / n_frames from `process` if
            # they were stored there.
            proc = data.get("process") or {}
            if "sim_time" in proc:
                cfg.step.sim_time = proc["sim_time"]
            if "n_frames" in proc:
                cfg.step.n_frames = proc["n_frames"]

        _apply(cfg.interaction,    data.get("interaction"))
        _apply(cfg.bcs,            data.get("bcs"))

        tool = data.get("tool", {})
        _apply(cfg.tool_geometry, tool.get("geometry"))
        _apply(cfg.tool_position, tool.get("position"))
        if "material" in tool:
            cfg.tool_material = dict(tool["material"])

        wp = data.get("workpiece", {})
        if "h_wp" in wp: cfg.euler_geometry.h_wp = wp["h_wp"]
        if "l_wp" in wp: cfg.euler_geometry.l_wp = wp["l_wp"]
        _apply(cfg.wp_position, wp.get("position"))
        if "material" in wp:
            cfg.euler_material = dict(wp["material"])

        eul = data.get("eulerian", {})
        if "h_void"     in eul: cfg.euler_geometry.h_void     = eul["h_void"]
        if "l_void"     in eul: cfg.euler_geometry.l_void     = eul["l_void"]
        if "discretize" in eul: cfg.euler_geometry.discretize = eul["discretize"]
        _apply(cfg.euler_position, eul.get("position"))

        mesh = data.get("mesh", {})
        if "elem_size" in mesh: cfg.elem_size = mesh["elem_size"]
        _apply(cfg.tool_element,  mesh.get("tool_element"))
        _apply(cfg.euler_element, mesh.get("euler_element"))
        _apply(cfg.wp_element,    mesh.get("wp_element"))

        _apply(cfg.bbox, data.get("bbox"))

        return cfg

    def save_to(self, path: str | Path) -> None:
        """Save profile as JSON, pretty-printed."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from(cls, path: str | Path) -> "ModelConfig":
        """Load profile from JSON. Raises ValueError on malformed file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON must be an object")
        return cls.from_json_dict(data)

    # ----- Derived quantities (used by GeometryTab info panel) -----
    def effective_euler_dims(self) -> tuple[float, float, float, float]:
        """Returns (h_wp, h_void, l_wp, l_void) after optional discretization."""
        g = self.euler_geometry
        if g.discretize and self.elem_size > 0:
            return (
                discretize(g.h_wp,   self.elem_size),
                discretize(g.h_void, self.elem_size),
                discretize(g.l_wp,   self.elem_size),
                discretize(g.l_void, self.elem_size),
            )
        return (g.h_wp, g.h_void, g.l_wp, g.l_void)

    def effective_elem_sizes(self) -> tuple[float, float]:
        """Return (es_x, es_y), the element sizes that Abaqus will actually
        produce along the horizontal and vertical directions respectively.

        - If `discretize=True`: dims are floored to a multiple of `elem_size`,
          so both effective sizes equal `elem_size`.
        - If `discretize=False`: Abaqus's seedPartInstance rounds the number
          of seeds per edge to `round(L / elem_size)`, then distributes them
          evenly: actual size per direction = L / round(L / elem_size).
          The x and y effective sizes can therefore differ.
        """
        if self.elem_size <= 0:
            return (0.0, 0.0)
        g = self.euler_geometry
        if g.discretize:
            return (self.elem_size, self.elem_size)

        # Determine the rectangle that gets seeded
        if self.analysis.formulation == "Lagrangian":
            Lx, Ly = g.l_wp, g.h_wp
        else:
            Lx, Ly = (g.l_wp + g.l_void), (g.h_wp + g.h_void)

        def _per_direction(L: float) -> float:
            if L <= 0:
                return self.elem_size
            n = max(1, round(L / self.elem_size))
            return L / n

        return (_per_direction(Lx), _per_direction(Ly))

    def effective_elem_size(self) -> float:
        """Backwards-compatible scalar accessor: returns the LARGER of the
        two effective sizes (worst case in the geometric sense).

        Note: for the explicit-stability time increment, we instead want the
        SMALLER of the two — see `stable_dt_estimate()`."""
        ex, ey = self.effective_elem_sizes()
        return max(ex, ey)

    def n_elements_estimate(self) -> int:
        """Approximate element count, depending on formulation.

        CEL:        structured hex grid over the full Eulerian domain
                    (workpiece + void), one element through thickness.
                    N = nx_total * ny_total
        Lagrangian: structured hex grid over the workpiece only (no void).
                    N = nx_wp * ny_wp
        Tool elements are NOT counted (its mesh is small and uses a bias
        seed, so it adds at most a few hundred elements either way).
        """
        if self.elem_size <= 0:
            return 0
        h_wp, h_void, l_wp, l_void = self.effective_euler_dims()
        if self.analysis.formulation == "Lagrangian":
            nx = round(l_wp / self.elem_size)
            ny = round(h_wp / self.elem_size)
        else:
            nx = round((l_wp + l_void) / self.elem_size)
            ny = round((h_wp + h_void) / self.elem_size)
        return max(0, nx * ny)

    def stable_dt_estimate(self) -> float:
        """Courant-style explicit stable time increment for the workpiece:
            dt ~= L_char / c_d,  c_d = sqrt(E / rho)
        Units must be consistent with rho [t/mm^3], E [MPa]: c_d in mm/s,
        so dt in s when L in mm. NOTE: this is a first-order estimate;
        Abaqus uses a more involved formula. See Abaqus Theory Manual,
        'Stability limit for the explicit operator'.

        For an anisotropic hex element (es_x ≠ es_y), the smallest internal
        distance between neighbouring nodes drives the limit, so we use
        L_char = min(es_x, es_y). The same formula applies to both CEL and
        Lagrangian — it's the material's dilatational wave speed that
        constrains the step in both cases."""
        E   = float(self.euler_material.get("E", 0.0))
        rho = float(self.euler_material.get("rho", 0.0))
        es_x, es_y = self.effective_elem_sizes()
        L_char = min(es_x, es_y) if es_x > 0 and es_y > 0 else max(es_x, es_y)
        if E <= 0 or rho <= 0 or L_char <= 0:
            return 0.0
        c_d = (E / rho) ** 0.5
        return L_char / c_d
