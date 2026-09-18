# -*- coding: utf-8 -*-
"""
Pure geometry helpers for sizing the Eulerian domain of a CEL orthogonal-cutting
model (used by the Optimization tab). No Qt, no Abaqus — only the standard
library, so it is fully unit-testable.

Domain frame (matches abq generation in run_simul.py:396-402):
    material : x in [-l_wp, 0],  y in [-h_wp, 0]
    void     : x in [0, l_void]  (downstream / chip exit),
               y in [0, h_void]  (above the free surface, chip rise)
    free surface at y = 0, cutting plane at x ~ 0, tool tip at (0, -t1).

Sizing philosophy (agreed with the user):
  - Start from the SMALLEST domain that still encloses the primary shear band
    predicted by Merchant's theory, the extraction ROI, and (optionally) the
    tool — then grow each dimension until the ROI fields stabilise.
  - Merchant's shear angle gives the band geometry; the ROI must always be
    contained (you cannot extract a field outside the domain).

Rake-angle convention: the GUI's `rake_angle` is measured from the VERTICAL,
which is exactly the standard tool rake angle used in Merchant's relations
(a zero-rake tool has a vertical rake face). So alpha_merchant == rake_angle.

References:
  Merchant, M.E. (1945) "Mechanics of the metal cutting process".
  phi = 45 + alpha/2 - beta/2, with beta = atan(mu) the friction angle.
  Chip thickness: t2 = t1 * cos(phi - alpha) / sin(phi).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# Clamp the shear angle away from 0/90 deg so tan()/sin() stay well-behaved.
# Extreme rake/friction can push Merchant's phi out of (0, 90); the resulting
# bracket would then be pathological, but it is only a STARTING point that the
# growth step (and the ROI floor) corrects, so a wide safety clamp is enough.
_PHI_MIN_DEG = 1.0
_PHI_MAX_DEG = 89.0


@dataclass
class DomainDims:
    """Eulerian half-extents, matching EulerGeometry in model_config."""
    h_wp: float       # material depth below the free surface (y in [-h_wp, 0])
    h_void: float     # void height above the free surface (y in [0, h_void])
    l_wp: float       # material length upstream (x in [-l_wp, 0])
    l_void: float     # void length downstream (x in [0, l_void])

def merchant_shear_angle(rake_deg: float, mu: float) -> float:
    """Merchant primary shear angle phi (degrees).

        phi = 45 + rake/2 - beta/2,   beta = atan(mu)

    mu is the Coulomb friction coefficient (interaction.friction_coeff).
    The result is clamped to (1, 89) deg for numerical safety."""
    beta_deg = math.degrees(math.atan(mu))
    phi = 45.0 + 0.5 * rake_deg - 0.5 * beta_deg
    return min(max(phi, _PHI_MIN_DEG), _PHI_MAX_DEG)


def chip_thickness(t1: float, rake_deg: float, mu: float,
                   phi_deg: float | None = None) -> float:
    """Deformed chip thickness t2 from Merchant geometry.

        t2 = t1 * cos(phi - alpha) / sin(phi)

    With phi = 45 deg and alpha = 0, t2 == t1 (sanity check)."""
    if phi_deg is None:
        phi_deg = merchant_shear_angle(rake_deg, mu)
    phi = math.radians(phi_deg)
    alpha = math.radians(rake_deg)
    return t1 * math.cos(phi - alpha) / math.sin(phi)


def shear_band_bracket(t1: float, rake_deg: float, mu: float,
                       l_void_factor: float = 1.0) -> DomainDims:
    """Minimal domain (no margin, no ROI, no element snapping) that encloses
    the Merchant primary shear band and one chip thickness of void.

        l_wp   = t1 / tan(phi)            upstream reach of the shear plane
        h_wp   = t1                       depth down to the tool tip
        h_void = t2                       one chip thickness above the surface
        l_void = l_void_factor * t2       room for the chip to exit downstream
    """
    if t1 <= 0:
        raise ValueError("t1 (depth of cut) must be > 0")
    phi_deg = merchant_shear_angle(rake_deg, mu)
    phi = math.radians(phi_deg)
    t2 = chip_thickness(t1, rake_deg, mu, phi_deg)
    return DomainDims(
        h_wp=t1,
        h_void=t2,
        l_wp=t1 / math.tan(phi),
        l_void=max(0.0, l_void_factor) * t2,
    )


def _snap_up(value: float, elem_size: float) -> float:
    """Round `value` UP to a whole number of elements (so the domain still
    contains what it must). Falls back to `value` if elem_size <= 0."""
    if elem_size is None or elem_size <= 0:
        return float(value)
    n = math.ceil(value / elem_size - 1e-9)
    return float(max(1, n) * elem_size)


def initial_domain_dimensions(t1: float, rake_deg: float, mu: float,
                              elem_size: float,
                              roi: tuple | None = None,
                              tool_bbox: tuple | None = None,
                              margin_elems: int = 2,
                              l_void_factor: float = 1.0) -> DomainDims:
    """Initial Eulerian domain = envelope(Merchant bracket, ROI, tool bbox),
    plus a margin, snapped UP to whole elements.

    Parameters
    ----------
    t1, rake_deg, mu : Merchant inputs (depth of cut, rake from vertical [deg],
        Coulomb friction).
    elem_size : Eulerian element size; all returned dims are multiples of it.
    roi : optional (xmin, xmax, ymin, ymax) extraction box in the DOMAIN frame.
        The domain must contain it: l_wp >= -xmin, l_void >= xmax,
        h_wp >= -ymin, h_void >= ymax (each clamped at 0).
    tool_bbox : optional (xmin, xmax, ymin, ymax) of the tool footprint in the
        domain frame, if you want the Eulerian domain to fully contain the tool
        (in CEL the rigid tool may legitimately extend beyond it, so this is
        opt-in). Same containment mapping as `roi`.
    margin_elems : extra elements added on every side (default 2).
    l_void_factor : multiplier on the chip thickness for the downstream void.

    Returns DomainDims (each field a positive multiple of elem_size).
    """
    base = shear_band_bracket(t1, rake_deg, mu, l_void_factor=l_void_factor)
    l_wp = base.l_wp
    l_void = base.l_void
    h_wp = base.h_wp
    h_void = base.h_void

    for box in (roi, tool_bbox):
        if box is None:
            continue
        xmin, xmax, ymin, ymax = box
        l_wp = max(l_wp, -xmin if xmin < 0 else 0.0)
        l_void = max(l_void, xmax if xmax > 0 else 0.0)
        h_wp = max(h_wp, -ymin if ymin < 0 else 0.0)
        h_void = max(h_void, ymax if ymax > 0 else 0.0)

    m = max(0, int(margin_elems)) * (elem_size if elem_size and elem_size > 0
                                     else 0.0)
    return DomainDims(
        h_wp=_snap_up(h_wp + m, elem_size),
        h_void=_snap_up(h_void + m, elem_size),
        l_wp=_snap_up(l_wp + m, elem_size),
        l_void=_snap_up(l_void + m, elem_size),
    )


# --- Reverberation ceiling and dimension helpers -----------------------------
# Relocated from domain_jacobian (now removed) so the sizing helpers live beside
# DomainDims. Used by domain_convergence and the Optimization tab.
DIMENSION_NAMES = ("h_wp", "h_void", "l_wp", "l_void")

# Safety margin k between the domain cavity mode and the output-filter cutoff:
# the reverberation peak must stay k times ABOVE fc so the (2nd-order
# Butterworth) runtime filter still attenuates it strongly. k = 1 would leave
# the artefact sitting exactly at the -3 dB point.
#
# HYPOTHESIS, not a measurement: k = 3 is an inherited, unjustified safety
# coefficient (it matches ModelConfig._REVERB_MARGIN). It is exposed as a
# parameter everywhere below precisely so it can be re-calibrated against the
# margin actually measured on a production domain.
REVERB_MARGIN_K = 3.0

# Abaqus RECOMMENDS -- and does not require -- that the runtime IIR filter's
# cutoff/sampling ratio stay above this. See `filter_ratio_check`.
IIR_ADVISED_MIN_RATIO = 1.0e-3

# HARD limit, from the Abaqus doc ("Filtering Output and Operating on Output in
# Abaqus/Explicit"): "the cutoff frequency should be less than half the
# sampling frequency; otherwise, no filtering is performed".
NYQUIST_MAX_RATIO = 0.5


def diagonal(dims: DomainDims) -> float:
    """Domain diagonal -- the longest wave path, which sets the reverberation
    frequency c_eff/(2 L) (see `reverberation_frequency`)."""
    return math.hypot(dims.l_wp + dims.l_void, dims.h_wp + dims.h_void)


def dilatational_wave_speed(E: float, nu: float, rho: float) -> float:
    """Dilatational (P-wave) speed of an isotropic elastic solid:

        c_d = sqrt(E (1 - nu) / (rho (1 + nu)(1 - 2 nu)))
            = sqrt((lambda + 2 mu) / rho)

    UNITS are those of the Abaqus mm-t-s system used throughout this GUI
    (see ModelConfig.initial_stable_dt, gui/core/model_config.py:803-813):
    E in MPa = N/mm^2, rho in t/mm^3 -> c_d in mm/s.

    With E = 113800 MPa, nu = 0.342, rho = 4.43e-9 t/mm^3 (Ti-6Al-4V) this
    gives 6.3134e6 mm/s = 6313 m/s.

    Returns 0.0 when the inputs are unusable (non-positive E or rho, nu outside
    the physically admissible (-1, 0.5))."""
    try:
        E = float(E)
        nu = float(nu)
        rho = float(rho)
    except (TypeError, ValueError):
        return 0.0
    if E <= 0.0 or rho <= 0.0 or not (-1.0 < nu < 0.5):
        return 0.0
    num = E * (1.0 - nu)
    den = rho * (1.0 + nu) * (1.0 - 2.0 * nu)
    if den <= 0.0:
        return 0.0
    return math.sqrt(num / den)


def effective_wave_speed(c_d: float, mass_scaling: float = 1.0) -> float:
    """Wave speed seen by the SOLVER once mass scaling is applied.

    Mass scaling multiplies the density by ms, and c ~ 1/sqrt(rho), so

        c_eff = c_d / sqrt(ms)

    (mass scaling divides Cp by the same factor, so only inertia is scaled --
    see StepCfg's docstring, gui/core/model_config.py:124-145.)

    c_d = 6313 m/s at ms = 1000 gives c_eff = 199.65 m/s."""
    c_d = float(c_d)
    ms = float(mass_scaling)
    if c_d <= 0.0 or ms <= 0.0:
        return 0.0
    return c_d / math.sqrt(ms)


def reverberation_frequency(c_eff: float, L: float) -> float:
    """Frequency of the domain reverberation artefact, read as a CAVITY MODE:

        f_reverb = c_eff / (2 L)

    L is the domain DIAGONAL (the longest wave path, see `diagonal`); units
    must be consistent (mm and mm/s -> Hz).

    FACT (measured, 2026 campaign). Three runs at ms = 1000, identical mesh and
    source, runtime filters OFF, peak read off the detrended ALLKE spectrum:

        diagonal   peak measured   c_eff/(2 L) predicted   deviation
        0.42 mm      241.6 kHz            237.7 kHz          +1.6 %
        0.84 mm      123.1 kHz            118.8 kHz          +3.6 %
        1.26 mm       85.3 kHz             79.2 kHz          +7.6 %

    The fitted exponent is f ~ L^(-0.951) (the 1/L cavity law), and the fitted
    c_eff is 204.6 m/s against 199.65 m/s predicted from E = 113.8 GPa,
    nu = 0.342, rho = 4430 kg/m^3 (c_d = 6313 m/s) at ms = 1000 -- i.e. +2.5 %
    with NO free parameter.

    INTERPRETATION: the residual over-prediction of the measurement grows with
    L, which is consistent with the true path being slightly shorter than the
    full diagonal; it is not explained here and 8 % is the envelope over the
    three points, not a proven bound.

    Returns 0.0 when the inputs are unusable."""
    c_eff = float(c_eff)
    L = float(L)
    if c_eff <= 0.0 or L <= 0.0:
        return 0.0
    return c_eff / (2.0 * L)


def reverberation_diagonal_limit(filter_cutoff_hz: float,
                                 mass_scaling: float = 1.0,
                                 c_d: float | None = None,
                                 E: float | None = None,
                                 nu: float | None = None,
                                 rho: float | None = None,
                                 margin_k: float = REVERB_MARGIN_K) -> float:
    """Largest domain diagonal whose cavity mode stays `margin_k` times ABOVE
    the output-filter cutoff:

        c_d    = sqrt(E (1 - nu) / (rho (1 + nu)(1 - 2 nu)))
        c_eff  = c_d / sqrt(ms)
        L_max  = c_eff / (2 k fc)

    This is exactly the reverberation bound already stated by
    ModelConfig.mass_scaling_bounds (gui/core/model_config.py:831-835),
    ms < (c_d / (2 L k fc))^2, solved for L at the mass-scaling factor
    ACTUALLY used, instead of demanding that SOME ms also satisfy the IIR
    lower bound.

    WHY THIS REPLACED `diagonal_limit(elem_size)`. The old ceiling was
    L < 90.6 * elem_size. It did NOT come from reverberation: it came from
    requiring the mass-scaling window of `mass_scaling_bounds` to be non-empty,
    hence from the LOWER bound fc * dt > 1e-3. That lower bound is an Abaqus
    RECOMMENDATION, not a hard limit (see `filter_ratio_check`), and production
    runs violate it (ratio 5.447e-4) with no observed effect on the results.
    The old ceiling was over-constraining by a factor ~7 and blocked the domain
    sizing study before its first run.

    Give either `c_d` directly, or the triplet (E, nu, rho). Units follow the
    mm-t-s system: E in MPa, rho in t/mm^3, fc in Hz -> L_max in mm.

    Reference point (test): E = 113800 MPa, nu = 0.342, rho = 4.43e-9 t/mm^3,
    ms = 1000, fc = 25600 Hz, k = 3 -> L_max = 1.300 mm.

    Returns 0.0 when the inputs are unusable (so the caller can tell
    "not computable" from a real ceiling)."""
    if c_d is None:
        if E is None or nu is None or rho is None:
            return 0.0
        c_d = dilatational_wave_speed(E, nu, rho)
    c_eff = effective_wave_speed(c_d, mass_scaling)
    fc = float(filter_cutoff_hz)
    k = float(margin_k)
    if c_eff <= 0.0 or fc <= 0.0 or k <= 0.0:
        return 0.0
    return c_eff / (2.0 * k * fc)


@dataclass
class FilterRatioCheck:
    """Verdict on the runtime output filter's cutoff/sampling ratio.

    `ratio` = fc * dt, with dt the solver increment (sampling period).
    `nyquist_ok` is the HARD control; `iir_advised_ok` is ADVISORY ONLY.
    `computable` is False when dt or fc were unusable, in which case neither
    verdict means anything."""
    ratio: float
    nyquist_ok: bool
    iir_advised_ok: bool
    computable: bool = True
    message: str = ""


def filter_ratio_check(filter_cutoff_hz: float, dt: float,
                       advised_min_ratio: float = IIR_ADVISED_MIN_RATIO,
                       nyquist_max_ratio: float = NYQUIST_MAX_RATIO
                       ) -> FilterRatioCheck:
    """Check the runtime Butterworth filter against the solver increment.

    Two controls with DIFFERENT statuses -- this distinction is the point of
    the function:

    HARD (blocks) -- Nyquist. Abaqus doc, "Filtering Output and Operating on
      Output in Abaqus/Explicit": "the cutoff frequency should be less than
      half the sampling frequency; otherwise, no filtering is performed".
      So fc < 0.5 / dt is required, or the filter silently does nothing.

    ADVISORY (warns, never blocks) -- the 1e-3 ratio. The .sta message is a
      WARNING, not an error:
        "***WARNING: The cutoff frequency used with the filter ... is too low
         and it may produce incorrect results ... It is recommended that the
         ratio of cutoff frequency to sampling frequency (which is 1/time
         increment) be greater than 1e-3 in order to avoid possible
         instabilities in filters ... If none of the above are suitable for you
         then use a two-stage filtering approach: first filter the data using a
         high cutoff frequency (10 to 100 times less than the sampling
         frequency), and filter this data for a second time in a postprocessor
         (at the desired cutoff frequency)."
      The documented remedy (two-stage filtering) is repeated in `message`.
      FACT: production runs of this model sit at ratio 5.447e-4, i.e. below the
      recommendation, with no observed effect on the results."""
    fc = float(filter_cutoff_hz)
    dt = float(dt)
    if fc <= 0.0 or dt <= 0.0:
        return FilterRatioCheck(ratio=0.0, nyquist_ok=False,
                                iir_advised_ok=False, computable=False,
                                message="filter ratio not computable "
                                        "(cutoff or time increment <= 0)")
    ratio = fc * dt
    nyquist_ok = ratio < float(nyquist_max_ratio)
    advised_ok = ratio > float(advised_min_ratio)
    if not nyquist_ok:
        msg = ("BLOCKING: cutoff %.4g Hz is at or above half the sampling "
               "frequency (fc*dt = %.4g >= %.4g): Abaqus performs NO filtering "
               "at all. Lower the cutoff or lower the mass-scaling factor."
               % (fc, ratio, nyquist_max_ratio))
    elif not advised_ok:
        msg = ("WARNING (not blocking): fc*dt = %.4g is below the %.4g Abaqus "
               "RECOMMENDS for its runtime IIR filters; filter instabilities "
               "are possible. Documented remedy: two-stage filtering -- filter "
               "at runtime with a high cutoff (10 to 100 times below the "
               "sampling frequency 1/dt = %.4g Hz), then filter again in "
               "post-processing at the desired cutoff."
               % (ratio, advised_min_ratio, 1.0 / dt))
    else:
        msg = "filter ratio fc*dt = %.4g: within both controls" % ratio
    return FilterRatioCheck(ratio=ratio, nyquist_ok=nyquist_ok,
                            iir_advised_ok=advised_ok, message=msg)


# --- Adapters reading a ModelConfig-shaped object ----------------------------
# Duck-typed on purpose: this module must stay importable with the standard
# library alone (no Qt, no gui.core.model_config), so the config is read by
# attribute rather than imported. Keys read (verified in model_config.py):
#   cfg.euler_material["E"|"nu"|"rho"]          model_config.py:450, 803-805
#   cfg.step.mass_scaling_enabled / _factor     model_config.py:170, 175
#   cfg.step.output_filter_cutoff_hz            model_config.py:162
#   cfg.initial_stable_dt()                     model_config.py:791-818

def config_mass_scaling(cfg) -> float:
    """Mass-scaling factor actually applied by the exporter: the configured
    factor when mass scaling is enabled, 1.0 otherwise."""
    step = getattr(cfg, "step", None)
    if step is None or not getattr(step, "mass_scaling_enabled", False):
        return 1.0
    try:
        ms = float(getattr(step, "mass_scaling_factor", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return ms if ms > 0.0 else 1.0


def config_diagonal_limit(cfg, margin_k: float = REVERB_MARGIN_K
                          ) -> float | None:
    """`reverberation_diagonal_limit` for a ModelConfig-shaped `cfg`, in mm.

    Returns None when the configuration does not carry the material, the
    mass-scaling factor or the filter cutoff needed to compute it -- the caller
    must then treat the domain as UNBOUNDED rather than invent a ceiling."""
    mat = getattr(cfg, "euler_material", None)
    step = getattr(cfg, "step", None)
    if not isinstance(mat, dict) or step is None:
        return None
    fc = getattr(step, "output_filter_cutoff_hz", 0.0)
    try:
        fc = float(fc)
    except (TypeError, ValueError):
        return None
    limit = reverberation_diagonal_limit(
        filter_cutoff_hz=fc, mass_scaling=config_mass_scaling(cfg),
        E=mat.get("E"), nu=mat.get("nu"), rho=mat.get("rho"),
        margin_k=margin_k)
    return limit if limit > 0.0 else None


def config_filter_check(cfg) -> FilterRatioCheck | None:
    """`filter_ratio_check` for a ModelConfig-shaped `cfg`.

    The solver increment is dt = dt0 * sqrt(ms), with dt0 the un-scaled initial
    stable increment from ModelConfig.initial_stable_dt() (validated against
    two real .sta files to better than 0.03 %, see
    tests/test_mass_scaling_bounds.py:31-37). Returns None when `cfg` does not
    expose what is needed."""
    step = getattr(cfg, "step", None)
    dt0_fn = getattr(cfg, "initial_stable_dt", None)
    if step is None or not callable(dt0_fn):
        return None
    try:
        dt0 = float(dt0_fn())
        fc = float(getattr(step, "output_filter_cutoff_hz", 0.0))
    except (TypeError, ValueError):
        return None
    if dt0 <= 0.0:
        return None
    dt = dt0 * math.sqrt(config_mass_scaling(cfg))
    return filter_ratio_check(fc, dt)
