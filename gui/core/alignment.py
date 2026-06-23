# -*- coding: utf-8 -*-
"""
Pure geometry helpers for the Alignment tab.

Image/model frame convention (decided with the user):
  - origin at the IMAGE CENTRE,
  - x horizontal, pointing right  (cutting direction, tool tip towards +x),
  - y vertical,   pointing up.
Since image rows grow downward, the model y is flipped relative to pixels.

A uniform scale `mm_per_px` maps pixels to millimetres (the observed plane is
assumed imaged fronto-parallel; the value comes from the visible calibration
or from a manual override). The scale does NOT affect angles.

Angle convention matches the Abaqus tool sketch (run_simul.py): rake_angle /
clear_angle are the tilt of the rake / flank face away from the vertical
(y) axis. `line_tilt_from_vertical_deg` returns that tilt in degrees, folded
to (-90, 90], positive when the line leans towards +x going up.
"""
from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float]


def pixel_to_model(px: float, py: float, width: int, height: int,
                   mm_per_px: float) -> Point:
    """Convert an image pixel (px, py) to model millimetres, origin at the
    image centre, x right, y up."""
    x = (px - width / 2.0) * mm_per_px
    y = (height / 2.0 - py) * mm_per_px
    return (x, y)


def line_tilt_from_vertical_deg(p1_img: Point, p2_img: Point) -> float:
    """Tilt (deg) of the image line (p1->p2) from the vertical axis, in the
    model orientation (y up). 0 deg = perfectly vertical; positive = clockwise
    from the +y axis (top leaning towards +x). Independent of the mm/px scale.
    Folded to (-90, 90]. Matches the model's rake_angle (deviation of the rake
    face from vertical, run_simul.py)."""
    dx = p2_img[0] - p1_img[0]
    dy = -(p2_img[1] - p1_img[1])         # model y points up
    if dx == 0.0 and dy == 0.0:
        return float("nan")
    ang = math.degrees(math.atan2(dx, dy))   # clockwise from +y
    while ang > 90.0:
        ang -= 180.0
    while ang <= -90.0:
        ang += 180.0
    return ang


def line_tilt_from_horizontal_deg(p1_img: Point, p2_img: Point) -> float:
    """Tilt (deg) of the image line (p1->p2) from the horizontal axis, in the
    model orientation (y up). 0 deg = horizontal; positive = the line rises
    towards +x (counter-clockwise from the +x axis). Independent of the mm/px
    scale. Folded to (-90, 90]. Matches the model's clear_angle (deviation of
    the flank face from the horizontal, run_simul.py)."""
    dx = p2_img[0] - p1_img[0]
    dy = -(p2_img[1] - p1_img[1])         # model y points up
    if dx == 0.0 and dy == 0.0:
        return float("nan")
    ang = math.degrees(math.atan2(dy, dx))   # CCW from +x
    while ang > 90.0:
        ang -= 180.0
    while ang <= -90.0:
        ang += 180.0
    return ang
