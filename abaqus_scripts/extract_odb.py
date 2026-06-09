# -*- coding: utf-8 -*-
"""
extract_odb.py — read an Abaqus .odb and produce a (.npz + .json)
results bundle conforming to gui/results/FORMAT.md.

Runs under Abaqus Python 2.7 (`abaqus python extract_odb.py ...`) — that
is why we avoid f-strings, type hints, and Python 3-only library
features. NumPy is available in the Abaqus environment.

Usage:
    abaqus python extract_odb.py ^
        --odb     C:\\TEMP\\Abaqus_wd\\Cutting_job.odb ^
        --out     C:\\TEMP\\Abaqus_wd\\Cutting_job.results.npz ^
        --step    Cut ^
        --roi     -0.10,0.30,-0.15,0.15,0.0,0.0001 ^
        --cfg     C:\\TEMP\\Abaqus_wd\\Cutting_job.cfg.json

Arguments:
    --odb       Path to the .odb to read.
    --out       Path to the output .npz (the .json is named beside it).
    --step      Step name to extract (default "Cut").
    --roi       Comma-separated xmin,xmax,ymin,ymax,zmin,zmax. If a
                degenerate bbox is given (all zeros) the script keeps
                EVERYTHING (no ROI filtering).
    --cfg       Optional JSON file containing the model config snapshot
                to embed verbatim in the bundle's metadata. The Job tab
                writes this file alongside the .odb at run time.
    --fields    Comma-separated list of field variables to extract.
                Default: EVF,TEMP,V (Eulerian: fill fraction, element
                temperature, nodal velocity magnitude). PEEQ / MISES on
                request.
    --history   Comma-separated list of history variables to extract,
                each in the form "name@instance.set". The RP forces are
                exposed as a special shorthand "RF1_RP" / "RF2_RP".

The script writes verbose progress to stdout so the user sees what is
happening while it runs (a typical cutting run is tens of thousands of
elements times hundreds of frames — extraction takes a few seconds).
"""
from __future__ import print_function  # Python 2 compat for print(...)
import sys
import os
import json
import argparse
from datetime import datetime
import numpy as np

# These imports only work inside an Abaqus Python interpreter.
try:
    from odbAccess import openOdb
except ImportError:
    print("ERROR: this script must run under `abaqus python ...`")
    print("       (the `odbAccess` module is unavailable otherwise).")
    sys.exit(1)


# =============================================================================
# Helpers
# =============================================================================
def _vprint(msg):
    """Verbose stdout print, forcibly flushed so Abaqus's Run streams it
    line-by-line into the GUI's output panel."""
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


def parse_roi(roi_str):
    """Parse 'xmin,xmax,ymin,ymax,zmin,zmax' or return None if the bbox
    is degenerate (all-zero), meaning 'keep everything'."""
    if not roi_str:
        return None
    parts = [float(x) for x in roi_str.split(",")]
    if len(parts) != 6:
        raise ValueError("--roi expects 6 comma-separated floats")
    xmin, xmax, ymin, ymax, zmin, zmax = parts
    if (xmin == 0 and xmax == 0 and ymin == 0 and ymax == 0
            and zmin == 0 and zmax == 0):
        return None
    return {"xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax}


def parse_roi_from_cfg(cfg_path):
    """Read the bbox from a cfg snapshot .json file. Returns a dict
    in the same shape as parse_roi(), or None if the bbox is degenerate
    or the file can't be read.

    This is the preferred way to pass the ROI to the extractor — see
    the comment in job_tab.py._start_extract_phase: Abaqus's `abaqus
    python` wrapper mangles arguments that start with '-' or contain
    '=', so passing a comma-separated ROI string on the command line
    is unreliable. Reading from the cfg file sidesteps that entirely.
    """
    if not cfg_path or not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    except (IOError, ValueError):
        return None
    bb = cfg.get("bbox", {})
    if not bb:
        return None
    try:
        xmin = float(bb.get("xmin", 0.0))
        xmax = float(bb.get("xmax", 0.0))
        ymin = float(bb.get("ymin", 0.0))
        ymax = float(bb.get("ymax", 0.0))
        zmin = float(bb.get("zmin", 0.0))
        zmax = float(bb.get("zmax", 0.0))
    except (TypeError, ValueError):
        return None
    # An ROI only filters when it has POSITIVE in-plane extent (x AND y).
    # The model is a thin plane-strain slab whose default bbox only encodes
    # a tiny z-thickness (e.g. {0,0,0,0,0,1e-4}); the old "all six == 0"
    # test let that through as an active ROI and dropped every element.
    if xmax <= xmin or ymax <= ymin:
        return None
    return {"xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax}


def in_bbox(c, roi):
    """Return True iff the centroid c lies in the ROI, tested in the
    (x, y) plane only. z is intentionally ignored: the model is a thin
    plane-strain slab, so filtering on the slab thickness would exclude
    every element. Inclusive on all sides."""
    return (roi["xmin"] <= c[0] <= roi["xmax"]
            and roi["ymin"] <= c[1] <= roi["ymax"])


# =============================================================================
# Geometry extraction
# =============================================================================
def extract_instance_geometry(instance, roi):
    """Read nodes & connectivity from one ODB instance, optionally
    keep only the elements whose initial centroid is inside the ROI.

    Returns:
        nodes_init   : (n_nodes, 3) float32   — initial node positions
        elements     : (n_elements, 8) int32  — connectivity, 0-based,
                                               re-indexed against `nodes_init`
        centroids    : (n_elements, 3) float32 — initial element centroids
        kept_node_ids:  list of int           — original Abaqus node labels
        kept_elem_ids:  list of int           — original Abaqus element labels
                                               (the order matches `elements`)
    """
    # All initial coordinates: (n_total_nodes, 3) — use index = node_label - 1
    # when labels are 1..N contiguous (typical of meshes generated by Abaqus
    # itself). We don't assume that; we build a label -> index map.
    n_total_nodes = len(instance.nodes)
    all_coords = np.zeros((n_total_nodes, 3), dtype=np.float32)
    label_to_idx = {}
    for i in range(n_total_nodes):
        node = instance.nodes[i]
        all_coords[i] = node.coordinates
        label_to_idx[node.label] = i

    # Element subset. When a ROI is active, use Abaqus' C-level
    # getByBoundingBox (much faster than a Python centroid loop on tens of
    # thousands of elements) and derive the node ROI from the kept
    # elements' connectivity, so element- and node-outputs stay consistent.
    elem_seq = instance.elements
    if roi is not None:
        zmin = roi.get("zmin"); zmax = roi.get("zmax")
        zlo = -1.0e20 if zmin is None else zmin
        zhi = 1.0e20 if zmax is None else zmax
        # pad slightly so boundary nodes of the thin slab aren't excluded
        ex = 1.0e-6 * (roi["xmax"] - roi["xmin"] + 1.0)
        ey = 1.0e-6 * (roi["ymax"] - roi["ymin"] + 1.0)
        try:
            elem_seq = instance.elements.getByBoundingBox(
                xMin=roi["xmin"] - ex, xMax=roi["xmax"] + ex,
                yMin=roi["ymin"] - ey, yMax=roi["ymax"] + ey,
                zMin=zlo, zMax=zhi)
            _vprint("  ROI getByBoundingBox kept %d / %d elements"
                    % (len(elem_seq), len(instance.elements)))
        except Exception as _e:
            _vprint("  getByBoundingBox unavailable (%s); keeping all" % _e)
            elem_seq = instance.elements

    kept_elements = []      # connectivity rows referencing all_coords indices
    kept_centroids = []
    kept_elem_ids = []      # original Abaqus labels (1-based)

    for elem in elem_seq:
        if elem.type not in ("EC3D8R", "EC3D8RT", "C3D8R", "C3D8RT",
                              "C3D8", "C3D8T"):
            continue
        # 8 nodes per hex
        conn = [label_to_idx[lbl] for lbl in elem.connectivity]
        c = all_coords[conn].mean(axis=0)
        kept_elements.append(conn)
        kept_centroids.append(c)
        kept_elem_ids.append(elem.label)

    if len(kept_elements) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 8), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                [], [])

    # Re-index against the surviving nodes only — keeps the .npz small.
    # First, find the set of nodes touched by at least one kept element.
    touched = set()
    for conn in kept_elements:
        for n in conn:
            touched.add(n)
    touched_sorted = sorted(touched)
    new_idx_of = {}
    for new_i, old_i in enumerate(touched_sorted):
        new_idx_of[old_i] = new_i

    nodes_init = all_coords[touched_sorted]
    elements = np.zeros((len(kept_elements), 8), dtype=np.int32)
    for i, conn in enumerate(kept_elements):
        for j in range(8):
            elements[i, j] = new_idx_of[conn[j]]
    centroids = np.asarray(kept_centroids, dtype=np.float32)

    # Original Abaqus node labels (1-based) of the kept nodes
    kept_node_ids = []
    # We need the inverse: which Abaqus labels are at touched_sorted positions
    idx_to_label = {}
    for lbl, idx in label_to_idx.items():
        idx_to_label[idx] = lbl
    for old_i in touched_sorted:
        kept_node_ids.append(idx_to_label[old_i])

    return nodes_init, elements, centroids, kept_node_ids, kept_elem_ids


# =============================================================================
# Field extraction
# =============================================================================
# Map of "GUI-facing field name" -> "(abaqus field name, reducer)"
# Reducer is a function (values, components_labels) -> scalar np array
# of shape (n_elements,). For scalar fields the reducer is the identity.
def _reduce_VM(vals, comp_labels):
    """Reduce a stress tensor to its von Mises equivalent. Expects 6
    components in Abaqus order: S11 S22 S33 S12 S13 S23. Missing
    components are treated as zero (never raises)."""
    idx = dict(zip(comp_labels, range(len(comp_labels))))
    def col(name):
        return vals[:, idx[name]] if name in idx else 0.0
    s11, s22, s33 = col("S11"), col("S22"), col("S33")
    s12, s13, s23 = col("S12"), col("S13"), col("S23")
    return np.sqrt(0.5 * (
        (s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
        + 6.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )).astype(np.float32)


def _reduce_pressure(vals, comp_labels):
    """Hydrostatic pressure: -(s11 + s22 + s33) / 3. Missing comps = 0."""
    idx = dict(zip(comp_labels, range(len(comp_labels))))
    def col(name):
        return vals[:, idx[name]] if name in idx else 0.0
    return (-(col("S11") + col("S22") + col("S33")) / 3.0).astype(np.float32)


def _reduce_identity(vals, comp_labels):
    """Pass-through for scalar fields."""
    if vals.ndim == 2 and vals.shape[1] == 1:
        return vals[:, 0].astype(np.float32)
    return vals.astype(np.float32)


# Tensor-derived "virtual" field variables
TENSOR_REDUCERS = {
    "MISES": ("S", _reduce_VM),   # von Mises (Abaqus invariant name)
    "S_VM":  ("S", _reduce_VM),   # backward-compatible alias
    "S_P":   ("S", _reduce_pressure),
}

# Stress invariants available natively on the S field via
# getScalarField(invariant=...). Preferred over recombining components,
# which can fail on CEL per-material fields whose componentLabels are not
# reported as plain 'S11'.. . If the invariant call fails we fall back to
# the (now guard-safe) component reducers above.
_STRESS_INVARIANT = {"MISES": "MISES", "S_VM": "MISES", "S_P": "PRESS"}


def _read_data(v):
    """Read a field value's data, tolerant of double-precision ODBs.

    Abaqus raises 'Data is double precision. Use dataDouble()' on
    `value.data` when the analysis ran in double precision (this model
    does). Fall back to `dataDouble` in that case."""
    try:
        return v.data
    except Exception:
        return v.dataDouble


def _resolve_fo_name(step, abq_var, instance_name, root_assembly, max_probe=3):
    """Find the real fieldOutputs key for `abq_var` on `instance_name`.

    Abaqus/CEL stores per-material results on an Eulerian instance under a
    suffixed name, e.g.
        PEEQ -> 'PEEQ_ASSEMBLY_EULER_EULER-1'
        S    -> 'S_ASSEMBLY_EULER_EULER-1'
        TEMP -> 'TEMP_ASSEMBLY_EULER_EULER-1'   (Eulerian)
        EVF  -> 'EVF_ASSEMBLY_EULER_EULER-1'    (material)  + 'EVF_VOID'
    while the bare names (e.g. 'TEMP') may carry ONLY the Lagrangian tool's
    values. We pick the candidate key that actually has values on the
    target instance, preferring: exact name, then a material-suffixed
    name, then a '*_VOID' name as last resort.

    Returns the key string, or None if nothing carries values on the
    instance. The key is consistent across frames, so we resolve once.
    """
    inst = root_assembly.instances[instance_name]
    n = len(step.frames)
    probe = [0]
    if n > 2:
        probe.append(n // 2)
    probe.append(n - 1)

    def _has_inst_values(fo):
        try:
            return len(fo.getSubset(region=inst).values) > 0
        except Exception:
            return False

    for fi in probe[:max_probe]:
        keys = list(step.frames[fi].fieldOutputs.keys())
        exact     = [k for k in keys if k == abq_var]
        suffixed  = [k for k in keys if k.startswith(abq_var + "_")]
        suff_mat  = [k for k in suffixed if not k.endswith("_VOID")]
        suff_void = [k for k in suffixed if k.endswith("_VOID")]
        # Bare names can be shared across instances (e.g. 'TEMP' carries
        # only the Lagrangian tool), so require values ON this instance.
        for k in exact:
            if _has_inst_values(step.frames[fi].fieldOutputs[k]):
                return k
        # CEL material-suffixed names (e.g. 'S_ASSEMBLY_EULER_EULER-1') are
        # specific to the Eulerian material/instance. If the key references
        # THIS instance, trust the name: getSubset/.values can be unreliable
        # for some tensor fields (S) even though the data is present.
        for k in (suff_mat + suff_void):
            if instance_name in k:
                return k
        # Otherwise fall back to a getSubset value check (covers suffixes
        # that don't embed the instance name).
        for k in (suff_mat + suff_void):
            if _has_inst_values(step.frames[fi].fieldOutputs[k]):
                return k
    return None


def extract_field(step, var, instance_name, kept_elem_ids, root_assembly):
    """Build a (n_frames, n_kept_elements) array for `var` over the
    whole step, restricted to the elements we kept.

    Strategy:
      - Resolve the real ODB field name for `var` on this instance
        (handles CEL per-material suffixes); raise KeyError if none.
      - For each frame, read the field, restrict to the instance and to
        element-centroid values, keep only kept_elem_ids.
      - For tensor variables (S), reduce to a scalar (VM or pressure).
    """
    # Resolve the requested Abaqus variable name and reducer
    if var in TENSOR_REDUCERS:
        abq_var, reducer = TENSOR_REDUCERS[var]
    else:
        abq_var, reducer = var, _reduce_identity
    inv_name = _STRESS_INVARIANT.get(var)   # "MISES" | "PRESS" | None

    key = _resolve_fo_name(step, abq_var, instance_name, root_assembly)
    if key is None:
        # Not present on this instance (e.g. PEEQ/S/EVF on the rigid tool).
        raise KeyError(abq_var)

    kept_set = set(kept_elem_ids)
    elem_id_to_pos = {}
    for pos, lbl in enumerate(kept_elem_ids):
        elem_id_to_pos[lbl] = pos
    n_elems = len(kept_elem_ids)

    frames = step.frames
    n_frames = len(frames)
    out = np.zeros((n_frames, n_elems), dtype=np.float32)

    try:
        from abaqusConstants import CENTROID
    except Exception:
        CENTROID = None
    # Resolve the native stress invariant constant once (if requested).
    invariant = None
    if inv_name is not None:
        try:
            import abaqusConstants as _abqc
            invariant = getattr(_abqc, inv_name, None)
        except Exception:
            invariant = None

    inst = root_assembly.instances[instance_name]
    # Decide once whether restricting to the instance yields values for
    # this field. Some CEL tensor fields (e.g. S) don't subset by instance
    # cleanly even though their values belong to it; in that case we read
    # the full field and rely on element-label filtering below (the
    # suffixed field only carries this instance's elements anyway).
    use_region = False
    probe_fi = (n_frames // 2) if n_frames > 1 else 0
    try:
        if len(frames[probe_fi].fieldOutputs[key].getSubset(region=inst).values) > 0:
            use_region = True
    except Exception:
        use_region = False

    for fi in range(n_frames):
        try:
            fo = frames[fi].fieldOutputs[key]
        except KeyError:
            continue
        if use_region:
            try:
                fo = fo.getSubset(region=inst)
            except (AttributeError, KeyError):
                pass
        # von Mises / pressure: prefer the native scalar invariant (avoids
        # recombining tensor components, which can fail on CEL S fields).
        src = fo
        used_invariant = False
        if invariant is not None:
            try:
                src = fo.getScalarField(invariant=invariant)
                used_invariant = True
            except Exception:
                src, used_invariant = fo, False
        if not used_invariant and CENTROID is not None:
            try:
                src = src.getSubset(position=CENTROID)
            except Exception:
                pass
        comp_labels = [] if used_invariant else (
            list(src.componentLabels) if src.componentLabels else [])
        vals_list = []
        labels_list = []
        for v in src.values:
            lbl = v.elementLabel
            if lbl in kept_set:
                if comp_labels:
                    vals_list.append(list(_read_data(v)))
                else:
                    vals_list.append([float(_read_data(v))])
                labels_list.append(lbl)
        if not vals_list:
            continue
        vals = np.asarray(vals_list, dtype=np.float32)
        if comp_labels:
            scalars = reducer(vals, comp_labels)
        else:
            scalars = _reduce_identity(vals, comp_labels)
        for k, lbl in enumerate(labels_list):
            pos = elem_id_to_pos.get(lbl)
            if pos is not None:
                out[fi, pos] = scalars[k]

    return out


# =============================================================================
# Displacement extraction (Lagrangian instances)
# =============================================================================
def extract_displacements(step, instance_name, kept_node_ids, root_assembly):
    """Build a (n_frames, n_nodes, 3) array of U over the kept nodes
    for the given Lagrangian instance."""
    kept_set = set(kept_node_ids)
    node_id_to_pos = {}
    for pos, lbl in enumerate(kept_node_ids):
        node_id_to_pos[lbl] = pos
    n_nodes = len(kept_node_ids)

    frames = step.frames
    n_frames = len(frames)
    out = np.zeros((n_frames, n_nodes, 3), dtype=np.float32)

    inst = root_assembly.instances[instance_name]
    for fi in range(n_frames):
        try:
            fo = frames[fi].fieldOutputs["U"]
        except KeyError:
            continue
        try:
            fo = fo.getSubset(region=inst)
        except (AttributeError, KeyError):
            pass
        for v in fo.values:
            lbl = v.nodeLabel
            if lbl in kept_set:
                pos = node_id_to_pos[lbl]
                d = _read_data(v)
                out[fi, pos, 0] = d[0]
                out[fi, pos, 1] = d[1]
                out[fi, pos, 2] = d[2] if len(d) > 2 else 0.0
    return out


# Nodal vector fields (e.g. velocity V) that we display by reducing to a
# per-node magnitude and averaging onto each element.
_NODAL_VECTOR_VARS = ("V",)


def extract_nodal_vector_to_elem(step, var, instance_name,
                                 kept_node_ids, elements, root_assembly):
    """Read a NODAL vector field (e.g. V) and return per-element fields,
    obtained by averaging each nodal component over an element's nodes:

        {var+"1": Vx, var+"2": Vy, var: magnitude}

    Vx/Vy are SIGNED (they show flow direction); the magnitude is
    sqrt(Vx^2 + Vy^2) of the per-element averaged components. Each array is
    (n_frames, n_elements). Raises KeyError if `var` is absent on the
    instance.
    """
    try:
        from abaqusConstants import NODAL
    except Exception:
        NODAL = None

    inst = root_assembly.instances[instance_name]
    key = _resolve_fo_name(step, var, instance_name, root_assembly)
    if key is None:
        raise KeyError(var)

    label_to_local = {}
    for i, lbl in enumerate(kept_node_ids):
        label_to_local[int(lbl)] = i
    n_nodes = len(kept_node_ids)
    elements = np.asarray(elements)              # (n_elem, 8), local indices
    n_elem = elements.shape[0]
    n_frames = len(step.frames)
    v1_out = np.zeros((n_frames, n_elem), dtype=np.float32)
    v2_out = np.zeros((n_frames, n_elem), dtype=np.float32)

    for fi in range(n_frames):
        try:
            fo = step.frames[fi].fieldOutputs[key]
        except KeyError:
            continue
        sub = fo
        try:
            sub = fo.getSubset(region=inst)
        except (AttributeError, KeyError):
            pass
        if NODAL is not None:
            try:
                sub = sub.getSubset(position=NODAL)
            except Exception:
                pass
        nodal_v1 = np.zeros(n_nodes, dtype=np.float32)
        nodal_v2 = np.zeros(n_nodes, dtype=np.float32)
        for v in sub.values:
            j = label_to_local.get(int(v.nodeLabel))
            if j is not None:
                d = _read_data(v)
                nodal_v1[j] = d[0]
                nodal_v2[j] = d[1] if len(d) > 1 else 0.0
        # Average each (signed) component over the element's nodes.
        v1_out[fi] = nodal_v1[elements].mean(axis=1)
        v2_out[fi] = nodal_v2[elements].mean(axis=1)

    mag = np.sqrt(v1_out * v1_out + v2_out * v2_out).astype(np.float32)
    return {var + "1": v1_out, var + "2": v2_out, var: mag}


# =============================================================================
# History extraction
# =============================================================================
def extract_history_rf(step):
    """Look for the H-Output-1 history region (the tool RP) and dump its
    RF1 / RF2 time series. Returns (time, rf1, rf2) numpy arrays, or
    (None, None, None) if no such history is found."""
    # H-Output-1 stores RF1/RF2 on the tool reference point.
    # Each historyRegion has a `historyOutputs` dict keyed by variable name.
    for region_key, region in step.historyRegions.items():
        outputs = region.historyOutputs
        if "RF1" in outputs and "RF2" in outputs:
            rf1_pairs = outputs["RF1"].data    # list of (time, value) tuples
            rf2_pairs = outputs["RF2"].data
            t = np.asarray([p[0] for p in rf1_pairs], dtype=np.float64)
            rf1 = np.asarray([p[1] for p in rf1_pairs], dtype=np.float32)
            rf2 = np.asarray([p[1] for p in rf2_pairs], dtype=np.float32)
            return t, rf1, rf2
    return None, None, None


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Extract a (.npz + .json) results bundle from an Abaqus .odb"
    )
    ap.add_argument("--odb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", default="Cut")
    ap.add_argument("--roi", default=None,
                    help="xmin,xmax,ymin,ymax,zmin,zmax. Degenerate = keep all.")
    ap.add_argument("--cfg", default=None,
                    help="Optional path to a JSON file containing the "
                         "model config snapshot (embedded verbatim).")
    ap.add_argument("--fields", default="EVF,TEMP,V",
                    help="Comma-separated field variables to extract.")
    args = ap.parse_args()

    _vprint("extract_odb.py  [rev: CEL-aware, nodal-V, name-trust]")

    # ROI resolution order:
    #   1. --roi on the command line (legacy / manual use)
    #   2. cfg.bbox from the --cfg file
    #   3. None = keep everything
    # The GUI's Job tab uses option 2 (it writes the cfg snapshot before
    # launching us) so it does not have to escape minus signs through
    # Abaqus's argument wrapper.
    if args.roi:
        roi = parse_roi(args.roi)
    else:
        roi = parse_roi_from_cfg(args.cfg)
    if roi is None:
        _vprint("ROI: none (keeping all elements)")
    else:
        _vprint("ROI: x[%g,%g] y[%g,%g] z[%g,%g]" % (
            roi["xmin"], roi["xmax"], roi["ymin"], roi["ymax"],
            roi["zmin"], roi["zmax"]))

    field_vars = [v.strip() for v in args.fields.split(",") if v.strip()]
    _vprint("Fields requested: %s" % ", ".join(field_vars))

    # Resolve output paths
    out = args.out
    if out.endswith(".npz"):
        npz_path = out
        json_path = out[:-4] + ".json"
    elif out.endswith(".json"):
        json_path = out
        npz_path = out[:-5] + ".npz"
    else:
        json_path = out + ".json"
        npz_path = out + ".npz"

    # Open the ODB
    _vprint("Opening ODB: %s" % args.odb)
    odb = openOdb(args.odb, readOnly=True)
    try:
        if args.step not in odb.steps:
            available = list(odb.steps.keys())
            raise ValueError("Step '%s' not in ODB. Available: %s"
                             % (args.step, available))
        step = odb.steps[args.step]

        # Times of every frame
        frames = step.frames
        n_frames = len(frames)
        times = np.asarray([fr.frameValue for fr in frames], dtype=np.float64)
        _vprint("Step '%s' has %d frames, t in [%g, %g]"
                % (args.step, n_frames, times[0], times[-1]))

        # Walk every instance
        npz_payload = {"times": times}
        instances_meta = {}

        for inst_name in odb.rootAssembly.instances.keys():
            inst = odb.rootAssembly.instances[inst_name]
            n_elem_total = len(inst.elements)
            if n_elem_total == 0:
                continue
            _vprint("\nInstance %s: %d nodes, %d elements"
                    % (inst_name, len(inst.nodes), n_elem_total))

            # Detect kind first: any element type starting with EC = Eulerian
            elem_type = inst.elements[0].type
            kind = "eulerian" if elem_type.startswith("EC") else "lagrangian"

            # Geometry + ROI filtering. The ROI/ZOI applies ONLY to the
            # Eulerian instance (the cutting zone). Lagrangian instances
            # (e.g. the TOOL) are always extracted whole.
            inst_roi = roi if kind == "eulerian" else None
            nodes_init, elements, centroids, kept_node_ids, kept_elem_ids = \
                extract_instance_geometry(inst, inst_roi)
            n_kept_elem = elements.shape[0]
            n_kept_node = nodes_init.shape[0]
            _vprint("  kept after ROI (%s): %d nodes, %d elements"
                    % (kind, n_kept_node, n_kept_elem))
            if n_kept_elem == 0:
                continue

            # Element/node arrays
            npz_payload["%s__nodes_init" % inst_name] = nodes_init
            npz_payload["%s__elements" % inst_name] = elements
            npz_payload["%s__element_centroids_init" % inst_name] = centroids

            # Field arrays
            stored_vars = []
            for var in field_vars:
                _vprint("  field '%s'..." % var)
                # Nodal vector fields (velocity): read at nodes, reduce to a
                # per-element magnitude. Eulerian instances only (the tool is
                # rigid, its V is ~0 and not of interest).
                if var in _NODAL_VECTOR_VARS:
                    if kind != "eulerian":
                        _vprint("    nodal field, skipping on this instance.")
                        continue
                    try:
                        vfields = extract_nodal_vector_to_elem(
                            step, var, inst_name, kept_node_ids, elements,
                            odb.rootAssembly)
                    except KeyError:
                        _vprint("    not available for this instance, skipping.")
                        continue
                    for vk in (var + "1", var + "2", var):
                        npz_payload["%s__fields__%s" % (inst_name, vk)] = vfields[vk]
                        stored_vars.append(vk)
                    _vprint("    stored %s1, %s2, %s (nodal->elem)" % (var, var, var))
                    continue
                try:
                    arr = extract_field(step, var, inst_name, kept_elem_ids,
                                        odb.rootAssembly)
                except KeyError:
                    _vprint("    not available for this instance, skipping.")
                    continue
                npz_payload["%s__fields__%s" % (inst_name, var)] = arr
                stored_vars.append(var)

            # Displacements: Lagrangian instances only
            has_disp = False
            if kind == "lagrangian":
                try:
                    disp = extract_displacements(step, inst_name, kept_node_ids,
                                                 odb.rootAssembly)
                    npz_payload["%s__displacements" % inst_name] = disp
                    has_disp = True
                    _vprint("  displacements stored.")
                except Exception as e:
                    _vprint("  displacement extraction failed: %s" % e)

            instances_meta[inst_name] = {
                "kind":              kind,
                "element_type":      elem_type,
                "n_nodes":           int(n_kept_node),
                "n_elements":        int(n_kept_elem),
                "n_frames":          int(n_frames),
                "field_variables":   stored_vars,
                "has_displacements": has_disp,
            }

        # History
        _vprint("\nExtracting history...")
        h_t, rf1, rf2 = extract_history_rf(step)
        history_vars = []
        if h_t is not None:
            npz_payload["history__time"] = h_t
            npz_payload["history__RF1_RP"] = rf1
            npz_payload["history__RF2_RP"] = rf2
            history_vars = ["RF1_RP", "RF2_RP"]
            _vprint("  history: %d samples, RF1/RF2 stored" % len(h_t))
        else:
            _vprint("  no RP history found.")

        # Optional cfg snapshot
        model_config = {}
        if args.cfg and os.path.exists(args.cfg):
            try:
                with open(args.cfg, "r") as fp:
                    model_config = json.load(fp)
                _vprint("Model config snapshot loaded from %s" % args.cfg)
            except (IOError, ValueError) as e:
                _vprint("WARNING: could not read --cfg file: %s" % e)

        # Compose metadata
        meta = {
            "format_version": 1,
            "saved_at":       datetime.now().isoformat(),
            "source_odb":     os.path.abspath(args.odb),
            "job_name":       os.path.splitext(os.path.basename(args.odb))[0],
            "step_name":      args.step,
            "times":          times.tolist(),
            "roi": {
                "applied": roi is not None,
                "xmin": (roi["xmin"] if roi else 0.0),
                "xmax": (roi["xmax"] if roi else 0.0),
                "ymin": (roi["ymin"] if roi else 0.0),
                "ymax": (roi["ymax"] if roi else 0.0),
                "zmin": (roi["zmin"] if roi else 0.0),
                "zmax": (roi["zmax"] if roi else 0.0),
            },
            "model_config": model_config,
            "instances":    instances_meta,
            "history": {
                "n_samples": int(len(h_t)) if h_t is not None else 0,
                "variables": history_vars,
            },
        }

        # Write outputs
        _vprint("\nWriting %s ..." % npz_path)
        np.savez_compressed(npz_path, **npz_payload)
        _vprint("Writing %s ..." % json_path)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        _vprint("\nDone. Bundle is ready:")
        _vprint("  " + npz_path)
        _vprint("  " + json_path)

    finally:
        odb.close()


if __name__ == "__main__":
    main()
