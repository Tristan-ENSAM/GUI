# -*- coding: utf-8 -*-
"""ODB result extraction for the CEL workflow.

This module is executed inside Abaqus Python after a successful analysis.
It reads the completed ODB and writes the existing .results.npz + .meta.json
bundle consumed by the GUI.
"""

import os
import sys
import json as _json

from odbAccess import openOdb as _openOdb
from abaqusConstants import CENTROID as _CENTROID
import numpy as _np

from cel_common import cfg_get

def _vprint(msg):
    """stdout-and-flush — so the GUI's output panel streams progress."""
    print(msg)
    sys.stdout.flush()


def _bbox_of_array(arr):
    """Return (min, max) tuples for a (N, 3) numpy array."""
    return arr.min(axis=0), arr.max(axis=0)


def _resolve_roi(model_cfg):
    """Read bbox from model_cfg. Return a dict {xmin,xmax,...} or None
    if degenerate (the ROI is the bbox of the user's region of
    interest, used to crop the extracted fields)."""
    bb = cfg_get(model_cfg, "geometry.bbox", {}) or {}
    try:
        xmin = float(bb.get("xmin", 0.0)); xmax = float(bb.get("xmax", 0.0))
        ymin = float(bb.get("ymin", 0.0)); ymax = float(bb.get("ymax", 0.0))
        zmin = float(bb.get("zmin", 0.0)); zmax = float(bb.get("zmax", 0.0))
    except (TypeError, ValueError):
        return None
    # An ROI only filters when it has POSITIVE in-plane extent (x AND y).
    # The model is a thin plane-strain slab, so the default bbox only
    # encodes a tiny z-thickness (e.g. {0,0,0,0,0,1e-4}). The old test
    # ("all six == 0 -> keep all") let that default through as an *active*
    # ROI, which selected the empty region x==0,y==0 and dropped every
    # instance from the bundle (nothing to show in the Results tab).
    if xmax <= xmin or ymax <= ymin:
        return None
    return {"xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax}


def _in_bbox(c, roi):
    """Inclusive in-plane (x, y) bounding-box test. z is intentionally
    ignored: the model is a thin plane-strain slab, so filtering on the
    small slab thickness would exclude every element."""
    return (roi["xmin"] <= c[0] <= roi["xmax"]
            and roi["ymin"] <= c[1] <= roi["ymax"])


def _extract_instance_geometry(inst, roi):
    """Walk every element of an ODB instance, keep those whose initial
    centroid is in roi (or all if roi is None). Returns:
        nodes_init    (n_nodes, 3)  float32, kept nodes only, re-indexed
        elements      (n_elem, 8)   int32, 0-based connectivity into nodes_init
        centroids     (n_elem, 3)   float32, initial centroids of kept elements
        kept_node_ids list[int]     original Abaqus node labels
        kept_elem_ids list[int]     original Abaqus element labels
        full_bbox     tuple         ((xmin,ymin,zmin), (xmax,ymax,zmax)) of
                                    every element in this instance — printed
                                    as a debug hint so the user can sanity-check
                                    the ROI against the actual mesh extent.
    """
    n_total_nodes = len(inst.nodes)
    all_coords = _np.zeros((n_total_nodes, 3), dtype=_np.float32)
    label_to_idx = {}
    for i in range(n_total_nodes):
        node = inst.nodes[i]
        all_coords[i] = node.coordinates
        label_to_idx[node.label] = i

    n_total_elems = len(inst.elements)
    all_centroids = _np.zeros((n_total_elems, 3), dtype=_np.float32)
    elem_connectivities = []
    elem_labels = []
    elem_kinds = []  # for filtering
    for i in range(n_total_elems):
        elem = inst.elements[i]
        if elem.type not in ("EC3D8R", "EC3D8RT", "C3D8R", "C3D8RT",
                              "C3D8", "C3D8T"):
            elem_connectivities.append(None)
            elem_labels.append(elem.label)
            elem_kinds.append(elem.type)
            continue
        conn = [label_to_idx[lbl] for lbl in elem.connectivity]
        all_centroids[i] = all_coords[conn].mean(axis=0)
        elem_connectivities.append(conn)
        elem_labels.append(elem.label)
        elem_kinds.append(elem.type)

    # Compute the mesh's actual bbox (over all hex elements) so the user
    # can verify their ROI is in the right ballpark.
    valid_mask = _np.array(
        [conn is not None for conn in elem_connectivities], dtype=bool
    )
    if valid_mask.any():
        valid_centroids = all_centroids[valid_mask]
        full_bbox = (valid_centroids.min(axis=0), valid_centroids.max(axis=0))
    else:
        full_bbox = (_np.zeros(3), _np.zeros(3))

    # Now apply the ROI filter
    kept_elements = []
    kept_centroids = []
    kept_elem_ids = []
    for i, conn in enumerate(elem_connectivities):
        if conn is None:
            continue
        if roi is not None and not _in_bbox(all_centroids[i], roi):
            continue
        kept_elements.append(conn)
        kept_centroids.append(all_centroids[i])
        kept_elem_ids.append(elem_labels[i])

    if len(kept_elements) == 0:
        return (_np.zeros((0, 3), dtype=_np.float32),
                _np.zeros((0, 8), dtype=_np.int32),
                _np.zeros((0, 3), dtype=_np.float32),
                [], [], full_bbox)

    touched = set()
    for conn in kept_elements:
        for n in conn:
            touched.add(n)
    touched_sorted = sorted(touched)
    new_idx_of = {}
    for new_i, old_i in enumerate(touched_sorted):
        new_idx_of[old_i] = new_i

    nodes_init = all_coords[touched_sorted]
    elements_arr = _np.zeros((len(kept_elements), 8), dtype=_np.int32)
    for i, conn in enumerate(kept_elements):
        for j in range(8):
            elements_arr[i, j] = new_idx_of[conn[j]]
    centroids = _np.asarray(kept_centroids, dtype=_np.float32)

    idx_to_label = {}
    for lbl, idx in label_to_idx.items():
        idx_to_label[idx] = lbl
    kept_node_ids = [idx_to_label[old_i] for old_i in touched_sorted]

    return (nodes_init, elements_arr, centroids,
            kept_node_ids, kept_elem_ids, full_bbox)


def _reduce_VM(vals, comp_labels):
    """von Mises reduction of a stress tensor (missing comps treated 0)."""
    idx = dict(zip(comp_labels, range(len(comp_labels))))
    def col(name):
        return vals[:, idx[name]] if name in idx else 0.0
    s11, s22, s33 = col("S11"), col("S22"), col("S33")
    s12, s13, s23 = col("S12"), col("S13"), col("S23")
    return _np.sqrt(0.5 * (
        (s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
        + 6.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )).astype(_np.float32)


def _reduce_identity(vals, comp_labels):
    if vals.ndim == 2 and vals.shape[1] == 1:
        return vals[:, 0].astype(_np.float32)
    return vals.astype(_np.float32)


_TENSOR_REDUCERS = {
    "MISES": ("S", _reduce_VM),
    "S_VM":  ("S", _reduce_VM),   # S_VM is the canonical field name used by
                                   # the results format (see FORMAT.md) —
                                   # NOT a legacy alias, kept intentionally.
}

# Native stress invariants (preferred over recombining components).
_STRESS_INVARIANT = {"MISES": "MISES", "S_VM": "MISES", "S_P": "PRESS"}


def _read_data(v):
    """Read a field value, tolerant of double-precision ODBs (this model
    runs in double precision, so `value.data` raises and we fall back to
    `value.dataDouble`)."""
    try:
        return v.data
    except Exception:
        return v.dataDouble


def _resolve_fo_name(step, abq_var, inst_name, root_assembly, max_probe=3):
    """Find the real fieldOutputs key for `abq_var` on `inst_name`.

    Abaqus/CEL stores per-material results on an Eulerian instance under a
    suffixed name (PEEQ -> 'PEEQ_ASSEMBLY_EULER_EULER-1', S -> 'S_...',
    TEMP -> 'TEMP_...', EVF -> 'EVF_...' material + 'EVF_VOID'), while the
    bare names may carry only the Lagrangian tool's values. Pick the
    candidate that actually has values on the target instance: exact name,
    then a material-suffixed name, then '*_VOID' as last resort. Returns
    the key or None.
    """
    inst = root_assembly.instances[inst_name]
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
        for k in exact:
            if _has_inst_values(step.frames[fi].fieldOutputs[k]):
                return k
        # CEL material-suffixed names referencing THIS instance: trust name
        # (getSubset/.values are unreliable for tensor fields like S).
        for k in (suff_mat + suff_void):
            if inst_name in k:
                return k
        for k in (suff_mat + suff_void):
            if _has_inst_values(step.frames[fi].fieldOutputs[k]):
                return k
    return None


def _extract_field(step, var, inst_name, kept_elem_ids, root_assembly):
    if var in _TENSOR_REDUCERS:
        abq_var, reducer = _TENSOR_REDUCERS[var]
    else:
        abq_var, reducer = var, _reduce_identity
    inv_name = _STRESS_INVARIANT.get(var)

    key = _resolve_fo_name(step, abq_var, inst_name, root_assembly)
    if key is None:
        # Not present on this instance (e.g. PEEQ/S/EVF on the rigid tool).
        raise KeyError(abq_var)

    kept_set = set(kept_elem_ids)
    elem_id_to_pos = {}
    for pos, lbl in enumerate(kept_elem_ids):
        elem_id_to_pos[lbl] = pos
    n_elems = len(kept_elem_ids)
    n_frames = len(step.frames)
    # NaN so cells never written by the ODB (frame missing the variable,
    # element absent) stay distinguishable from a genuine physical zero.
    out = _np.full((n_frames, n_elems), _np.nan, dtype=_np.float32)

    invariant = None
    if inv_name is not None:
        try:
            import abaqusConstants as _abqc
            invariant = getattr(_abqc, inv_name, None)
        except Exception:
            invariant = None

    inst = root_assembly.instances[inst_name]
    # Decide once whether restricting to the instance yields values; some
    # CEL tensor fields (S) don't subset by instance cleanly, so fall back
    # to the full field + element-label filtering (the suffixed field only
    # carries this instance's elements anyway).
    use_region = False
    probe_fi = (n_frames // 2) if n_frames > 1 else 0
    try:
        if len(step.frames[probe_fi].fieldOutputs[key].getSubset(region=inst).values) > 0:
            use_region = True
    except Exception:
        use_region = False

    for fi in range(n_frames):
        try:
            fo = step.frames[fi].fieldOutputs[key]
        except KeyError:
            continue
        if use_region:
            try:
                fo = fo.getSubset(region=inst)
            except (AttributeError, KeyError):
                pass
        # von Mises / pressure: prefer the native scalar invariant.
        src = fo
        used_invariant = False
        if invariant is not None:
            try:
                src = fo.getScalarField(invariant=invariant)
                used_invariant = True
            except Exception:
                src, used_invariant = fo, False
        if not used_invariant:
            try:
                src = src.getSubset(position=_CENTROID)
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
        vals = _np.asarray(vals_list, dtype=_np.float32)
        if comp_labels:
            scalars = reducer(vals, comp_labels)
        else:
            scalars = _reduce_identity(vals, comp_labels)
        for k, lbl in enumerate(labels_list):
            pos = elem_id_to_pos.get(lbl)
            if pos is not None:
                out[fi, pos] = scalars[k]
    return out


def _extract_displacements(step, inst_name, kept_node_ids, root_assembly):
    kept_set = set(kept_node_ids)
    node_id_to_pos = {}
    for pos, lbl in enumerate(kept_node_ids):
        node_id_to_pos[lbl] = pos
    n_nodes = len(kept_node_ids)
    n_frames = len(step.frames)
    out = _np.full((n_frames, n_nodes, 3), _np.nan, dtype=_np.float32)
    for fi in range(n_frames):
        try:
            fo = step.frames[fi].fieldOutputs["U"]
        except KeyError:
            continue
        try:
            fo = fo.getSubset(region=root_assembly.instances[inst_name])
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


_NODAL_VECTOR_VARS = ("V",)


def _extract_nodal_vector_to_elem(step, var, inst_name, kept_node_ids,
                                  elements, root_assembly):
    """Read a NODAL vector field (e.g. V); return per-element fields
    {var+'1': Vx, var+'2': Vy, var: magnitude}, each (n_frames, n_elem),
    by averaging each signed component over an element's nodes. Raises
    KeyError if absent on the instance."""
    try:
        from abaqusConstants import NODAL
    except Exception:
        NODAL = None
    inst = root_assembly.instances[inst_name]
    key = _resolve_fo_name(step, var, inst_name, root_assembly)
    if key is None:
        raise KeyError(var)
    label_to_local = {}
    for i, lbl in enumerate(kept_node_ids):
        label_to_local[int(lbl)] = i
    n_nodes = len(kept_node_ids)
    elements = _np.asarray(elements)
    n_elem = elements.shape[0]
    n_frames = len(step.frames)
    v1_out = _np.full((n_frames, n_elem), _np.nan, dtype=_np.float32)
    v2_out = _np.full((n_frames, n_elem), _np.nan, dtype=_np.float32)
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
        nodal_v1 = _np.full(n_nodes, _np.nan, dtype=_np.float32)
        nodal_v2 = _np.full(n_nodes, _np.nan, dtype=_np.float32)
        for v in sub.values:
            j = label_to_local.get(int(v.nodeLabel))
            if j is not None:
                d = _read_data(v)
                nodal_v1[j] = d[0]
                nodal_v2[j] = d[1] if len(d) > 1 else 0.0
        v1_out[fi] = nodal_v1[elements].mean(axis=1)
        v2_out[fi] = nodal_v2[elements].mean(axis=1)
    mag = _np.sqrt(v1_out * v1_out + v2_out * v2_out).astype(_np.float32)
    return {var + "1": v1_out, var + "2": v2_out, var: mag}


def _extract_history_rf(step):
    """Return (time, rf1, rf2) or (None, None, None)."""
    for region_key, region in step.historyRegions.items():
        outputs = region.historyOutputs
        if "RF1" in outputs and "RF2" in outputs:
            rf1_pairs = outputs["RF1"].data
            rf2_pairs = outputs["RF2"].data
            t = _np.asarray([p[0] for p in rf1_pairs], dtype=_np.float64)
            rf1 = _np.asarray([p[1] for p in rf1_pairs], dtype=_np.float32)
            rf2 = _np.asarray([p[1] for p in rf2_pairs], dtype=_np.float32)
            return t, rf1, rf2
    return None, None, None


def _extract_history_energy(step):
    """Return (time, allke, allie) for the whole-model energies, or
    (None, None, None). ALLKE/ALLIE live in the whole-model / assembly history
    region (no element/RP region)."""
    for region_key, region in step.historyRegions.items():
        outputs = region.historyOutputs
        if "ALLKE" in outputs and "ALLIE" in outputs:
            ke_pairs = outputs["ALLKE"].data
            ie_pairs = outputs["ALLIE"].data
            t = _np.asarray([p[0] for p in ke_pairs], dtype=_np.float64)
            allke = _np.asarray([p[1] for p in ke_pairs], dtype=_np.float32)
            allie = _np.asarray([p[1] for p in ie_pairs], dtype=_np.float32)
            return t, allke, allie
    return None, None, None


def extract_results(job_name, model_cfg):
    """Extract the completed ``job_name.odb`` into the GUI result bundle."""
    print("\n" + "=" * 72)
    print("[STAGE] EXTRACT_START")
    print("EXTRACTING results from %s.odb" % job_name)
    print("=" * 72)
    sys.stdout.flush()

    _roi = _resolve_roi(model_cfg)
    if _roi is None:
        _vprint("ROI: none (keeping all elements)")
    else:
        _vprint("ROI: x[%g,%g] y[%g,%g] z[%g,%g]" % (
            _roi["xmin"], _roi["xmax"],
            _roi["ymin"], _roi["ymax"],
            _roi["zmin"], _roi["zmax"]))

    _field_vars = ["EVF", "TEMP", "V"]
    _vprint("Fields requested: " + ", ".join(_field_vars))

    _odb_path = job_name + ".odb"
    _vprint("Opening ODB: " + _odb_path)
    _odb = _openOdb(_odb_path, readOnly=True)
    try:
        _step = _odb.steps["Cut"]
        _frames = _step.frames
        _n_frames = len(_frames)
        _times = _np.asarray([fr.frameValue for fr in _frames], dtype=_np.float64)
        _vprint("Step 'Cut' has %d frames, t in [%g, %g]"
                % (_n_frames, _times[0], _times[-1]))

        _npz_payload = {"times": _times}
        _instances_meta = {}

        for _inst_name in _odb.rootAssembly.instances.keys():
            _inst = _odb.rootAssembly.instances[_inst_name]
            _n_elem_total = len(_inst.elements)
            if _n_elem_total == 0:
                continue
            _vprint("\nInstance %s: %d nodes, %d elements"
                    % (_inst_name, len(_inst.nodes), _n_elem_total))

            _elem_type = _inst.elements[0].type
            _kind = "eulerian" if _elem_type.startswith("EC") else "lagrangian"

            # The ROI applies ONLY to the Eulerian instance (cutting zone).
            # Lagrangian instances (e.g. the TOOL) are always kept whole.
            _inst_roi = _roi if _kind == "eulerian" else None
            (_nodes_init, _elements, _centroids,
             _kept_node_ids, _kept_elem_ids, _full_bbox) = \
                _extract_instance_geometry(_inst, _inst_roi)

            # Print the full mesh bbox — invaluable when the ROI filter
            # rejects everything, since it tells the user whether the
            # ROI numbers are in the right unit/range.
            _vprint("  mesh bbox: x[%g,%g] y[%g,%g] z[%g,%g]"
                    % (_full_bbox[0][0], _full_bbox[1][0],
                       _full_bbox[0][1], _full_bbox[1][1],
                       _full_bbox[0][2], _full_bbox[1][2]))
            _n_kept_elem = _elements.shape[0]
            _n_kept_node = _nodes_init.shape[0]
            _vprint("  kept in ROI (%s): ROI_node=%d, ROI_elem=%d"
                    % (_kind, _n_kept_node, _n_kept_elem))

            # The ROI defines the ROI_node / ROI_elem sets on the Eulerian
            # (cutting) instance. If the ROI selects nothing there, extraction
            # cannot proceed: tell the user to size a larger ROI and stop with a
            # non-zero exit code.
            if _kind == "eulerian" and (_n_kept_elem == 0 or _n_kept_node == 0):
                _vprint("")
                _vprint("[ERROR] The ROI selects nothing on the Eulerian "
                        "instance")
                _vprint("        (ROI_node=%d, ROI_elem=%d)."
                        % (_n_kept_node, _n_kept_elem))
                _vprint("        Mesh bbox is x[%g,%g] y[%g,%g]; your ROI is "
                        "x[%g,%g] y[%g,%g]."
                        % (_full_bbox[0][0], _full_bbox[1][0],
                           _full_bbox[0][1], _full_bbox[1][1],
                           (_roi or {}).get("xmin", 0.0), (_roi or {}).get("xmax", 0.0),
                           (_roi or {}).get("ymin", 0.0), (_roi or {}).get("ymax", 0.0)))
                _vprint("        Increase the ROI (Geometry tab) so it overlaps "
                        "the mesh, then re-run.")
                sys.stdout.flush()
                _odb.close()
                sys.exit(2)
            if _n_kept_elem == 0:
                continue

            _npz_payload["%s__nodes_init" % _inst_name] = _nodes_init
            _npz_payload["%s__elements" % _inst_name] = _elements
            _npz_payload["%s__element_centroids_init" % _inst_name] = _centroids

            _stored_vars = []
            for _var in _field_vars:
                _vprint("  field '%s'..." % _var)
                if _var in _NODAL_VECTOR_VARS:
                    if _kind != "eulerian":
                        _vprint("    nodal field, skipping on this instance.")
                        continue
                    try:
                        _vf = _extract_nodal_vector_to_elem(
                            _step, _var, _inst_name, _kept_node_ids,
                            _elements, _odb.rootAssembly)
                    except KeyError:
                        _vprint("    not available, skipping.")
                        continue
                    for _vk in (_var + "1", _var + "2", _var):
                        _npz_payload["%s__fields__%s" % (_inst_name, _vk)] = _vf[_vk]
                        _stored_vars.append(_vk)
                    continue
                try:
                    _arr = _extract_field(_step, _var, _inst_name,
                                           _kept_elem_ids, _odb.rootAssembly)
                except KeyError:
                    _vprint("    not available, skipping.")
                    continue
                _npz_payload["%s__fields__%s" % (_inst_name, _var)] = _arr
                _stored_vars.append(_var)

            _has_disp = False
            if _kind == "lagrangian":
                try:
                    _disp = _extract_displacements(_step, _inst_name,
                                                    _kept_node_ids, _odb.rootAssembly)
                    _npz_payload["%s__displacements" % _inst_name] = _disp
                    _has_disp = True
                    _vprint("  displacements stored.")
                except Exception as _e:
                    _vprint("  displacement extraction failed: %s" % _e)

            _instances_meta[_inst_name] = {
                "kind":              _kind,
                "element_type":      _elem_type,
                "n_nodes":           int(_n_kept_node),
                "n_elements":        int(_n_kept_elem),
                "n_frames":          int(_n_frames),
                "field_variables":   _stored_vars,
                "has_displacements": _has_disp,
            }

        _vprint("\nExtracting history...")
        _h_t, _rf1, _rf2 = _extract_history_rf(_step)
        _history_vars = []
        if _h_t is not None:
            _npz_payload["history__time"] = _h_t
            _npz_payload["history__RF1_RP"] = _rf1
            _npz_payload["history__RF2_RP"] = _rf2
            _history_vars = ["RF1_RP", "RF2_RP"]
            _vprint("  history: %d samples, RF1/RF2 stored" % len(_h_t))
        else:
            _vprint("  no RP history found.")

        _he_t, _allke, _allie = _extract_history_energy(_step)
        if _he_t is not None:
            if "history__time" not in _npz_payload:
                _npz_payload["history__time"] = _he_t
            _npz_payload["history__ALLKE"] = _allke
            _npz_payload["history__ALLIE"] = _allie
            _history_vars = _history_vars + ["ALLKE", "ALLIE"]
            _vprint("  history: %d samples, ALLKE/ALLIE stored" % len(_he_t))
        else:
            _vprint("  no energy history found.")

        # Metadata
        from datetime import datetime as _datetime
        _meta = {
            "format_version": 1,
            "saved_at":       _datetime.now().isoformat(),
            "source_odb":     os.path.abspath(_odb_path),
            "job_name":       job_name,
            "step_name":      "Cut",
            "times":          _times.tolist(),
            "roi": {
                "applied": _roi is not None,
                "xmin": (_roi["xmin"] if _roi else 0.0),
                "xmax": (_roi["xmax"] if _roi else 0.0),
                "ymin": (_roi["ymin"] if _roi else 0.0),
                "ymax": (_roi["ymax"] if _roi else 0.0),
                "zmin": (_roi["zmin"] if _roi else 0.0),
                "zmax": (_roi["zmax"] if _roi else 0.0),
            },
            "model_config": model_cfg,
            "instances":    _instances_meta,
            "history": {
                "n_samples": int(len(_h_t)) if _h_t is not None else 0,
                "variables": _history_vars,
            },
        }

        _out_npz  = job_name + ".results.npz"
        _out_json = job_name + ".meta.json"
        _vprint("\nWriting %s ..." % _out_npz)
        _np.savez_compressed(_out_npz, **_npz_payload)
        _vprint("Writing %s ..." % _out_json)
        _f = open(_out_json, "w")
        try:
            _json.dump(_meta, _f, indent=2)
        finally:
            _f.close()

        _vprint("\nDone. Results bundle ready:")
        _vprint("  " + os.path.abspath(_out_npz))
        _vprint("  " + os.path.abspath(_out_json))
    finally:
        _odb.close()

    print("[STAGE] EXTRACT_DONE")
    sys.stdout.flush()
