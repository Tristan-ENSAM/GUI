# -*- coding: utf-8 -*-
"""
Fake results builder — generates a synthetic (.json + .npz) pair that
follows FORMAT.md exactly, without requiring Abaqus.

Use cases:
  - Develop and test the Results tab without a real .odb on hand.
  - Provide a fixture for unit tests.
  - Bootstrap the workflow before the real extractor is written.

The generated data is intentionally "fake but plausible":
  - The Eulerian grid is regular, sized from the cfg's ROI / Eulerian
    geometry.
  - Field values are smooth analytical functions of (x, y, t), with a
    moving hot spot that drifts left-to-right (a poor man's cutting
    simulation). This makes the resulting animation actually look like
    something — useful for visually validating the GUI's slider/play.
  - History RF1/RF2 oscillate around a mean, mimicking the noisy
    force signal of a real cutting run.

Usage (CLI):
    python -m gui.results.fake_builder \
        --out C:/TEMP/ABQ_wd/fake_job.results.npz \
        --n_frames 50 --n_grid 40

Or programmatically:
    from gui.results.fake_builder import build_fake_results
    build_fake_results("fake_job.results.npz", cfg=cfg)
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import argparse
import numpy as np


def build_fake_results(
    out_path: str | Path,
    *,
    cfg: dict | None = None,
    n_frames: int = 50,
    n_grid_x: int = 40,
    n_grid_y: int = 30,
    sim_time: float = 5e-4,
    job_name: str = "fake_job",
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate a fake (.json + .npz) pair.

    Parameters
    ----------
    out_path:
        Either the .npz path or the .json path (the other is auto-named).
    cfg:
        Optional snapshot of `ModelConfig.to_params_dict()`. If None,
        a minimal placeholder is written.
    n_frames:
        Number of field-output frames. The slider in the GUI will go
        from 0 to n_frames-1.
    n_grid_x, n_grid_y:
        Eulerian grid resolution (in elements) over the ROI.
    sim_time:
        Total simulation time (the last frame's time).
    job_name:
        Stored in the metadata and used in plot titles.
    seed:
        RNG seed for reproducible noise in history signals.

    Returns
    -------
    (json_path, npz_path)
    """
    out = Path(out_path)
    if out.suffix == ".npz":
        npz_path  = out
        json_path = out.with_suffix(".json")
    elif out.suffix == ".json":
        json_path = out
        npz_path  = out.with_suffix(".npz")
    else:
        json_path = out.with_suffix(".json")
        npz_path  = out.with_suffix(".npz")

    rng = np.random.default_rng(seed)

    # ----- ROI / grid setup -----
    # Pull the bbox from cfg if available, otherwise default to a typical
    # 0.4 mm x 0.3 mm cutting zone. The fake field is generated on a
    # regular grid spanning the ROI.
    bbox = (cfg or {}).get("bbox", {}) if cfg else {}
    xmin = float(bbox.get("xmin", -0.10))
    xmax = float(bbox.get("xmax",  0.30))
    ymin = float(bbox.get("ymin", -0.15))
    ymax = float(bbox.get("ymax",  0.15))
    zmin = float(bbox.get("zmin",  0.0))
    zmax = float(bbox.get("zmax",  1e-4))
    if xmax <= xmin or ymax <= ymin:
        # The bbox is empty — fall back to default to avoid a degenerate fake
        xmin, xmax = -0.10, 0.30
        ymin, ymax = -0.15, 0.15

    # Regular structured grid: nodes at the corners of n_grid_x × n_grid_y cells
    xs = np.linspace(xmin, xmax, n_grid_x + 1, dtype=np.float32)
    ys = np.linspace(ymin, ymax, n_grid_y + 1, dtype=np.float32)
    zs = np.array([zmin, zmax], dtype=np.float32)

    # Nodes: (n_x+1) * (n_y+1) * 2  in xyz order, indexed as
    # idx = iz*(n_x+1)*(n_y+1) + iy*(n_x+1) + ix
    n_nodes = (n_grid_x + 1) * (n_grid_y + 1) * 2
    n_elements = n_grid_x * n_grid_y * 1   # 1 element thick in z
    nodes_init = np.zeros((n_nodes, 3), dtype=np.float32)
    for iz, z in enumerate(zs):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                idx = iz * (n_grid_x + 1) * (n_grid_y + 1) + iy * (n_grid_x + 1) + ix
                nodes_init[idx] = (x, y, z)

    # Element connectivity (8 nodes per hex, standard Abaqus C3D8 ordering)
    elements = np.zeros((n_elements, 8), dtype=np.int32)
    centroids = np.zeros((n_elements, 3), dtype=np.float32)
    e = 0
    for iy in range(n_grid_y):
        for ix in range(n_grid_x):
            # Bottom face (z=zmin)
            n0 = 0 * (n_grid_x + 1) * (n_grid_y + 1) + (iy    ) * (n_grid_x + 1) + (ix    )
            n1 = 0 * (n_grid_x + 1) * (n_grid_y + 1) + (iy    ) * (n_grid_x + 1) + (ix + 1)
            n2 = 0 * (n_grid_x + 1) * (n_grid_y + 1) + (iy + 1) * (n_grid_x + 1) + (ix + 1)
            n3 = 0 * (n_grid_x + 1) * (n_grid_y + 1) + (iy + 1) * (n_grid_x + 1) + (ix    )
            # Top face (z=zmax)
            n4 = 1 * (n_grid_x + 1) * (n_grid_y + 1) + (iy    ) * (n_grid_x + 1) + (ix    )
            n5 = 1 * (n_grid_x + 1) * (n_grid_y + 1) + (iy    ) * (n_grid_x + 1) + (ix + 1)
            n6 = 1 * (n_grid_x + 1) * (n_grid_y + 1) + (iy + 1) * (n_grid_x + 1) + (ix + 1)
            n7 = 1 * (n_grid_x + 1) * (n_grid_y + 1) + (iy + 1) * (n_grid_x + 1) + (ix    )
            elements[e] = (n0, n1, n2, n3, n4, n5, n6, n7)
            # Centroid of the cell = average of the 8 corners
            centroids[e] = nodes_init[(n0, n1, n2, n3, n4, n5, n6, n7), :].mean(axis=0)
            e += 1

    # ----- Time sampling -----
    times = np.linspace(0.0, sim_time, n_frames, dtype=np.float64)

    # ----- Synthetic fields -----
    # All evaluated at element CENTROIDS. We use the (cx, cy) part —
    # the fake grid is 2D-like, z is just for storage.
    cx = centroids[:, 0].astype(np.float64)
    cy = centroids[:, 1].astype(np.float64)

    # Cutting "tip" moves across the ROI: from (xmin + 0.1*W, 0) at t=0
    # to (xmin + 0.8*W, 0) at t=sim_time.
    W = xmax - xmin
    tip_x_t = xmin + (0.1 + 0.7 * (times / sim_time)) * W
    tip_y   = 0.0

    fields = {}

    # PEEQ: gaussian blob centered on the moving tip, accumulating in time
    peeq = np.zeros((n_frames, n_elements), dtype=np.float32)
    sigma = 0.04 * W
    for k, tx in enumerate(tip_x_t):
        d2 = (cx - tx) ** 2 + (cy - tip_y) ** 2
        peeq[k] = (peeq[k-1] if k > 0 else 0.0) + np.exp(-d2 / (2 * sigma**2)) * 0.05
    fields["PEEQ"] = peeq

    # TEMP: similar blob but smoother and reset-ish (no accumulation), in °C
    temp = np.zeros((n_frames, n_elements), dtype=np.float32)
    ambient = 20.0
    peak    = 600.0
    for k, tx in enumerate(tip_x_t):
        d2 = (cx - tx) ** 2 + (cy - tip_y) ** 2
        # Add some persistent heating behind the tip
        trail = np.maximum(0.0, 0.6 * (tx - cx) / W)
        temp[k] = ambient + (peak - ambient) * (
            np.exp(-d2 / (2 * (1.5*sigma)**2)) + 0.3 * trail
        )
    fields["TEMP"] = temp

    # S_VM: von Mises — sharpest blob, no accumulation
    s_vm = np.zeros((n_frames, n_elements), dtype=np.float32)
    for k, tx in enumerate(tip_x_t):
        d2 = (cx - tx) ** 2 + (cy - tip_y) ** 2
        s_vm[k] = 800.0 * np.exp(-d2 / (2 * (0.7*sigma)**2))
    fields["S_VM"] = s_vm

    # EVF: 1.0 above y=0 (chip/workpiece), 0.0 below (void). The interface
    # gets perturbed by the tip passage.
    evf = np.zeros((n_frames, n_elements), dtype=np.float32)
    for k in range(n_frames):
        evf[k] = (cy > 0.0).astype(np.float32)
        # near the tip, the interface bulges
        d2 = (cx - tip_x_t[k]) ** 2 + (cy - tip_y) ** 2
        bulge = np.exp(-d2 / (2 * (1.2*sigma)**2)) * 0.3
        evf[k] = np.clip(evf[k] + bulge, 0.0, 1.0)
    fields["EVF"] = evf

    # ----- History RF -----
    # Forces oscillate around a baseline that ramps up as the tip enters
    # the material, plus high-frequency noise (chip-formation rattling).
    n_hist = max(2 * n_frames, 200)
    history_time = np.linspace(0.0, sim_time, n_hist, dtype=np.float64)
    baseline_F1 = 200.0 * np.tanh(history_time / (0.2 * sim_time))
    baseline_F2 = -50.0 * np.tanh(history_time / (0.3 * sim_time))
    noise_F1 = rng.normal(0.0, 30.0, n_hist)
    noise_F2 = rng.normal(0.0, 15.0, n_hist)
    rf1 = (baseline_F1 + noise_F1).astype(np.float32)
    rf2 = (baseline_F2 + noise_F2).astype(np.float32)

    # ----- Build the arrays dict for npz -----
    npz_payload = {
        "times": times,
        "Euler__nodes_init":                nodes_init,
        "Euler__elements":                  elements,
        "Euler__element_centroids_init":    centroids,
        "history__time": history_time,
        "history__RF1_RP": rf1,
        "history__RF2_RP": rf2,
    }
    for var, arr in fields.items():
        npz_payload[f"Euler__fields__{var}"] = arr

    # ----- Build the JSON metadata -----
    meta = {
        "format_version": 1,
        "saved_at":       datetime.now().isoformat(timespec="seconds"),
        "source_odb":     f"<fake — generated by fake_builder at "
                          f"{datetime.now().isoformat(timespec='seconds')}>",
        "job_name":       job_name,
        "step_name":      "Cut",
        "times":          times.tolist(),
        "roi": {
            "applied": True,
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax,
        },
        "model_config":   cfg or {},
        "instances": {
            "Euler": {
                "kind":               "eulerian",
                "element_type":       "EC3D8RT",
                "n_nodes":            int(n_nodes),
                "n_elements":         int(n_elements),
                "n_frames":           int(n_frames),
                "field_variables":    list(fields.keys()),
                "has_displacements":  False,
            }
        },
        "history": {
            "n_samples": int(n_hist),
            "variables": ["RF1_RP", "RF2_RP"],
        },
    }

    # ----- Write files -----
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **npz_payload)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return json_path, npz_path


def _main():
    ap = argparse.ArgumentParser(
        description="Generate a fake (.json + .npz) results bundle for testing."
    )
    ap.add_argument("--out",      type=str, required=True,
                    help="Output .npz path (the .json is named beside it).")
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--n_grid_x", type=int, default=40)
    ap.add_argument("--n_grid_y", type=int, default=30)
    ap.add_argument("--sim_time", type=float, default=5e-4)
    ap.add_argument("--job_name", type=str, default="fake_job")
    args = ap.parse_args()
    j, n = build_fake_results(
        args.out,
        n_frames=args.n_frames,
        n_grid_x=args.n_grid_x,
        n_grid_y=args.n_grid_y,
        sim_time=args.sim_time,
        job_name=args.job_name,
    )
    print(f"Wrote: {j}")
    print(f"       {n}")


if __name__ == "__main__":
    _main()
