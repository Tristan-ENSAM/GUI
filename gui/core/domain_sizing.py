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

    def as_dict(self) -> dict:
        return {"h_wp": self.h_wp, "h_void": self.h_void,
                "l_wp": self.l_wp, "l_void": self.l_void}


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
