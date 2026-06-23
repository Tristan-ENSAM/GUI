# Results file format — specification

Two files are written side-by-side after each Abaqus simulation:

```
<workdir>/<job_name>.results.json    # metadata (human-readable)
<workdir>/<job_name>.results.npz     # raw arrays (binary, compressed)
```

Both files share the same basename (`<job_name>.results`). Either is
useless without the other:

  - the .json describes the **structure** (instances, fields, time
    sampling, ROI used at extraction, snapshot of the model config),
  - the .npz holds the **values** (positions, connectivity, field
    arrays, history series).

The format is versioned via `format_version` in the .json so future
extractors can evolve without breaking older readers.


## .json — top-level keys

```json
{
  "format_version": 1,
  "saved_at":       "ISO datetime",
  "source_odb":     "<absolute path to the .odb that was extracted>",
  "job_name":       "<Cutting_job>",
  "step_name":      "Cut",

  "times":          [t0, t1, ...],          /* frame times, in seconds */

  "roi": {                                  /* the bbox used to filter */
    "applied": true,
    "xmin": float, "xmax": float,
    "ymin": float, "ymax": float,
    "zmin": float, "zmax": float
  },

  "model_config":   {... full cfg.to_params_dict() ...},

  "instances": {
    "<instance name>": {
      "kind":            "eulerian" | "lagrangian",
      "element_type":    "EC3D8RT" | "C3D8RT" | "C3D8T",
      "n_nodes":         int,
      "n_elements":      int,
      "field_variables": ["S_VM", "PEEQ", "TEMP", "EVF", ...],
      "has_displacements": false,            /* lagrangian only */
      "n_frames":        int                 /* number of time samples */
    }
  },

  "history": {
    "n_samples":  int,
    "variables":  ["RF1_RP", "RF2_RP", ...]
  }
}
```

Notes:
- `times` is duplicated in the .npz under `times` for convenience —
  the .json copy is for users who only need metadata.
- `model_config` is the full snapshot of `ModelConfig.to_params_dict()`
  at the moment the simulation was launched. Reading it back lets the
  results viewer reuse colours, geometry overlays, etc.


## .npz — key naming

Flat namespace, separator `__` (double underscore). Two levels:

```
times                                   shape (n_frames,)              float64

# Per-instance geometry & fields
<instance>__nodes_init                  shape (n_nodes, 3)             float32
<instance>__elements                    shape (n_elements, 8)          int32
<instance>__element_centroids_init      shape (n_elements, 3)          float32

# Field values: one (n_frames, n_elements) array per variable
<instance>__fields__<VAR>               shape (n_frames, n_elements)   float32

# Displacements: only present for lagrangian instances
<instance>__displacements               shape (n_frames, n_nodes, 3)   float32

# History output
history__time                           shape (n_samples,)             float64
history__<VAR>                          shape (n_samples,)             float32
```

For example, with one Eulerian instance "Euler" and history forces on
the tool RP, a CEL run with PEEQ, TEMP, S_VM, EVF gives:

```
times
Euler__nodes_init
Euler__elements
Euler__element_centroids_init
Euler__fields__PEEQ
Euler__fields__TEMP
Euler__fields__S_VM
Euler__fields__EVF
history__time
history__RF1_RP
history__RF2_RP
```


## ROI filtering at extraction time

The extractor keeps **only elements whose initial centroid lies inside
the ROI bbox**. Nodes attached to any kept element are kept too;
others are dropped. Element connectivity is re-indexed against the
surviving nodes (0-based, contiguous).

This means:
- For Eulerian instances (fixed mesh): nodes & connectivity reflect
  exactly the elements that lie in the ROI throughout the simulation.
- For Lagrangian instances (moving mesh): nodes are kept based on
  **initial** position. An element that drifts out of the ROI during
  the run is still in the dataset.


## Notation conventions

- Field variable names match Abaqus identifiers when scalar
  (`PEEQ`, `TEMP`, `EVF`, `SDEG`, `STATUS`...).
- Tensor fields are reduced to a scalar at extraction time, with a
  suffix indicating the reduction:
    - `S_VM`   = von Mises of the stress tensor
    - `S_P`    = hydrostatic pressure
  This avoids dumping 6-component tensors when most users only ever
  plot the equivalent value.
- Vector fields keep their natural components:
    - `RF1_RP`, `RF2_RP` (X- and Y- reaction force at the tool RP)

---

## Experimental field files (DIC / IRT)

The Experimental Data tab produces (or imports) one `.npz` + one `.json` per
modality per test, referenced from the `ExperimentSession`
(`gui/core/experiment_session.py`). The layout is chosen so that
`gui.sensitivity.field_metrics` can compare experimental and numerical fields
after resampling onto a common grid.

`<test>_dic.npz` (velocity, mm/s):
- `x`, `y` : `(n_points,)` coordinates in **mm, in the MODEL frame**
  (DIC node centres, mapped through the visible calibration + alignment).
- `t`      : `(n_frames,)` time in s, relative to the trigger origin.
  Velocity is instantaneous, computed between two images, so
  `n_frames = n_images - 1`.
- `V1`, `V2`, `Vmag` : `(n_frames, n_points)` velocity components + magnitude.
- `valid` : `(n_frames, n_points)` boolean correlation-validity mask.

`<test>_irt.npz` (temperature, °C): same `x`, `y`, `t`, plus
- `T` : `(n_frames, n_points)` temperature.

The paired `.json` carries the computation parameters and provenance:
modality, source (`computed`/`imported`), units, the calibration reference
used, the image→mm→model transform, and either the `q4dic` DIC settings
(ROI, subset, Q4 element size, correlation criterion, fps, trigger offset) or
the radiometric-model parameters (coefficients, emissivity, integration time,
fps). The no-load ("a vide") noise estimate is stored here once computed.

Grids differ between DIC/IRT and the FE mesh, so storing `x, y` is mandatory:
the simu↔experiment comparison resamples onto a common grid downstream.

### Visible calibration block (`session.visible_calibration`)

Written by the Calibration — visible tab (full Zhang via OpenCV):
`target` (`pattern`, `cols`, `rows`, `spacing_mm`), `images` (paths used),
`camera_matrix` (3x3), `dist_coeffs` (`[k1,k2,p1,p2,k3]`),
`scale_mm_per_px`, `homography` (3x3, pixels->mm in the reference plane),
`reproj_rms_px`, `image_size` (`[w,h]`), `reference_index` (the cutting-plane
view), `n_views_used`, `units`. Orientation/position in the model frame is
added later by the Alignment tab.

### Reference geometry block (`session.reference_geometry`)

Written by the Alignment tab and (optionally) pushed into the numerical model:
`rake_angle`, `clear_angle` (deg, tilt of the rake / flank face from the
vertical), `tool_x0`, `tool_y0`, `wp_x0`, `wp_y0` (mm). Frame convention:
origin at the image centre, x horizontal (cutting direction, tool tip towards
+x), y vertical (up). The scale mm/px comes from the visible calibration or a
manual override.
