# -*- coding: utf-8 -*-
"""Model tab: which sizing steps are done for this model, resume and load of
a study, and the whole pipeline in one go.

Mixed into OptimizationTab (gui/tabs/optimization_tab.py), which owns the
widgets and the four study launchers (_start_ms, _start_gci, _start_domain,
_start_checks). This part decides what to run and records what came out:

* every study that ends leaves a record in cfg.optimization.steps (saved in
  the profile, format in gui.sensitivity.study_state), so each step shows
  whether it is done for the CURRENT model and settings, and a step can warn
  before running out of order;
* a study is resumed by running it again in its own folder: the cores replay
  the finished runs from their saved bundles (gui.sensitivity.run_cache) and
  launch the first missing one;
* a study is loaded the same way with launching forbidden, so its full
  result comes back without any Abaqus run;
* "Run all steps" chains the steps and writes each result into the model
  (Step, Mesh and Geometry tabs) before the next one starts.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from gui.core.logging_util import log_swallowed
from gui.sensitivity.run_cache import RunCache, same_folder
from gui.sensitivity.study_specs import (
    CHECKS_CONFIG, DIM_KEYS, JOB_PREFIX, ZOI_KEYS, comparison_settings,
    eps_of_spec, grid_set_of_spec, ms_of, read_checks_config,
    read_study_folder, rewrite_study_config, settings_of_spec, zoi_tuple)
from gui.sensitivity.study_state import (
    AXIS_KEYS, STATE_LABELS, STEPS, STEP_TITLES, first_step_not_done,
    format_value, make_record, missing_prerequisites, model_differences,
    in_model, model_key, round6_up, settings_key, state_label, step_state,
    values_match)

# Doublings added to the ms values when every comparison passed (the
# largest acceptable ms is then not bracketed): 4000 -> up to 128000.
MS_EXTENSION = 5
# Model quantity set by each step, and where the user sees it.
_MODEL_NAME = {"ms": "ms", "mesh": "h", "domain": "dims"}
_MODEL_WHERE = {"ms": "the Step tab (mass scaling)",
                "mesh": "the Mesh tab (element size)",
                "domain": "the Geometry tab (Eulerian part)"}
_STATE_COLOR = {"done": "#15803d", "none": "#6b7280", "failed": "#b91c1c",
                "interrupted": "#b45309", "stale": "#b45309",
                "upstream": "#b45309", "prereq": "#b45309"}


def _r6(v) -> float:
    """The value as the model tabs show it (6 significant digits), so what
    is written is what the user reads there."""
    return float("%g" % float(v))




def _when(rec) -> str:
    t = str((rec or {}).get("finished_at") or "")
    return t.replace("T", " ")[:16]


def _dims_list(d) -> list:
    if isinstance(d, dict):
        return [float(d[k]) for k in DIM_KEYS]
    return [float(v) for v in d]


def _fmt_leaf(v) -> str:
    if isinstance(v, float):
        return "%.12g" % v          # enough digits to show the change
    return str(v)


# Plain names of the final checks (their codes mean nothing to a user).
_CHECK_TITLES = {"domain_combined": "domain grown on all sides together",
                 "mesh_x_domain": "element size on the final domain",
                 "ms_x_mesh": "mass scaling against its bound (informative)",
                 "ms_at_point": "mass scaling at the final point"}


class ModelStepsMixin:
    """Step records, resume/load and the pipeline (see the module doc).
    OptimizationTab.__init__ sets: _active, _pipeline, _is_busy,
    _model_refresher, _step_status, _btn_apply."""

    # =================================================================
    # Model, settings and records
    # =================================================================
    def set_model_refresher(self, fn):
        """`fn(step)` is called after this tab wrote a result into the model
        (the main window reloads the Step, Mesh and Geometry tabs and marks
        the profile modified)."""
        self._model_refresher = fn

    def _steps(self) -> dict:
        o = self.cfg.optimization
        if not isinstance(getattr(o, "steps", None), dict):
            o.steps = {}
        return o.steps

    def _model_params(self) -> dict:
        return self.cfg.to_params_dict()

    def _base_ms(self) -> list:
        st = self.cfg.step
        return [bool(getattr(st, "mass_scaling_enabled", False)),
                float(getattr(st, "mass_scaling_factor", 1.0))]

    def _model_values(self) -> dict:
        """ms, h and D as the model holds them now."""
        g = self.cfg.euler_geometry
        return {"ms": ms_of(self._base_ms()), "h": float(self.cfg.elem_size),
                "dims": [float(g.h_wp), float(g.h_void), float(g.l_wp),
                         float(g.l_void)]}

    def _current_settings(self) -> dict:
        """The shared comparison settings of the panel; raises ValueError."""
        g = self.guard_settings()
        return comparison_settings(self.zoi(), self.thresholds(),
                                   self.window(), g.rk_max, g.rhg_max,
                                   self._grid_step_set())

    def _keys(self):
        mkey = model_key(self._model_params())
        try:
            skey = settings_key(self._current_settings())
        except ValueError:
            skey = None
        return mkey, skey

    def _step_state(self, step):
        mkey, skey = self._keys()
        return step_state(self._steps(), step, mkey, skey)

    def _spec_defaults(self) -> dict:
        """What an older study folder may lack, taken from the tab."""
        return {"rk_max": self._float_or(self._dom_texts["rk_max"], 0.05),
                "rhg_max": self._float_or(self._dom_texts["rhg_max"], 0.05),
                "base_ms": self._base_ms(),
                "filter_verify": bool(getattr(
                    self.cfg.step, "output_filter_verify", True))}

    def _known_folders(self) -> list:
        """Study folders of the records (and of the domain study in memory):
        their finished runs may be reused by a new study."""
        out = []
        for rec in self._steps().values():
            for r in (rec or {}, (rec or {}).get("attempt") or {}):
                f = r.get("folder")
                if f and str(f) not in out:
                    out.append(str(f))
        d = getattr(self, "_last_domain_dir", None)
        if d and str(d) not in out:
            out.append(str(d))
        return out

    def _cpus(self) -> int:
        try:
            return int(self._cpus_getter()) if self._cpus_getter else 1
        except Exception:
            return 1

    # =================================================================
    # Status line of each step, "Use in model"
    # =================================================================
    def _refresh_step_status(self, *_):
        labels = getattr(self, "_step_status", None)
        if not labels:
            return
        steps = self._steps()
        mkey, skey = self._keys()
        mv = self._model_values()
        for step in STEPS:
            st, rec = step_state(steps, step, mkey, skey)
            text, tip = self._status_text(step, st, rec, mv, skey)
            lab = labels.get(step)
            if lab is not None:
                lab.setText(text)
                lab.setToolTip(tip)
                lab.setStyleSheet("color:%s;" % _STATE_COLOR.get(st,
                                                                 "#6b7280"))
            btn = self._btn_apply.get(step)
            if btn is not None:
                held = (st == "done" and in_model(
                    step, mv[_MODEL_NAME[step]], rec.get("value")))
                btn.setVisible(st == "done")
                btn.setEnabled(st == "done" and not held
                               and not (self._is_busy or self._pipeline))
                btn.setText("In the model" if held else "Use in model")
        if hasattr(self, "btn_checks"):
            self.btn_checks.setEnabled(
                not (self._is_busy or self._pipeline)
                and self._checks_available())

    def _status_text(self, step, st, rec, mv, skey):
        """(label text, tooltip) of a step's status line."""
        tip = ""
        if st == "none":
            text = "Not done yet for this model."
        elif st == "done":
            if step == "checks":
                text = "Done (%s): every check passed." % _when(rec)
            else:
                val = rec.get("value")
                text = "Done (%s): %s." % (_when(rec), format_value(step, val))
                name = _MODEL_NAME[step]
                if in_model(step, mv[name], val):
                    text += " The model uses it."
                else:
                    text += (" The model still has %s: click 'Use in model'."
                             % format_value(step, mv[name]))
            if rec.get("message"):
                text += " Note: %s." % rec["message"]
        elif st == "failed":
            text = "Failed (%s): %s." % (_when(rec), rec.get("message")
                                         or "see the Log tab")
        elif st == "interrupted":
            msg = rec.get("message") or ""
            text = "Interrupted (%s)%s: run it again to resume it; the " \
                "finished runs are reused." % (
                    _when(rec), "" if msg in ("", "cancelled")
                    else " (%s)" % msg)
        elif st == "stale":
            what = {"failed": "Failed", "interrupted": "Interrupted"}.get(
                rec.get("status"), "Done")
            text = ("%s for another model or other comparison settings "
                    "(%s): run it again." % (what, _when(rec)))
            tip = self._stale_details(rec)
        elif st == "prereq":
            text = "Done (%s)%s. %s" % (
                _when(rec), "" if step == "checks" else
                ": %s" % format_value(step, rec.get("value")),
                self._prereq_sentence(step, rec))
        else:                                   # upstream
            text = ("Made with an earlier result of a previous step (%s): "
                    "run it again." % _when(rec))
        att = (rec or {}).get("attempt")
        if st in ("done", "prereq") and att:
            msg = att.get("message") or ""
            text += " A later study (%s) %s%s." % (
                _when(att), STATE_LABELS.get(att.get("status"), "stopped"),
                ": click the step's button to resume it"
                if self._resumable_attempt(step, rec) is not None else
                (": %s" % msg if msg not in ("", "cancelled") else ""))
            if att.get("folder"):
                tip = (tip + "\n" if tip else "") + \
                    "Later study: %s" % att["folder"]
        if skey is None:
            text += " (Comparison settings incomplete.)"
        if rec and rec.get("folder"):
            tip = (tip + "\n" if tip else "") + "Study folder: %s" % rec["folder"]
        return text, tip

    def _prereq_sentence(self, step, rec) -> str:
        """Why a done step does not count yet (state "prereq") and when it
        will: the earlier step must be done and give the value this result
        was made with."""
        up = first_step_not_done(self._steps(), step, *self._keys())
        if up is None:
            return ""
        up_step, up_st, up_rec = up
        text = "%s is %s%s: run it first." % (
            STEP_TITLES[up_step], state_label(up_st, up_rec),
            "" if up_st == "stale" else " for this model")
        used = (rec.get("inputs") or {}).get(_MODEL_NAME[up_step])
        if used is not None:
            text += (" This result was made with %s: it counts only if %s "
                     "finds that value, otherwise run it again." % (
                         format_value(up_step, used), STEP_TITLES[up_step]))
        return text

    def _stale_details(self, rec) -> str:
        lines = []
        old = rec.get("model_params")
        if isinstance(old, dict):
            diffs = model_differences(old, self._model_params())
            for k, a, b in diffs[:10]:
                lines.append("  %s: %s then, %s now" % (k, _fmt_leaf(a),
                                                        _fmt_leaf(b)))
            if len(diffs) > 10:
                lines.append("  ... and %d more" % (len(diffs) - 10))
            if diffs:
                lines.insert(0, "Model changes since this study:")
        old_s = rec.get("settings")
        try:
            cur_s = self._current_settings()
        except ValueError:
            cur_s = None
        if isinstance(old_s, dict) and cur_s is not None:
            names = {"zoi": "ZOI", "eps": "ε_q", "window": "time window T",
                     "rk_max": "G_K", "rhg_max": "G_HG",
                     "grid": "ZOI sampling step"}
            changed = [names.get(k, k) for k in sorted(set(old_s) | set(cur_s))
                       if old_s.get(k) != cur_s.get(k)]
            if changed:
                lines.append("Comparison settings changed: %s"
                             % ", ".join(changed))
        return "\n".join(lines)

    def _on_apply_step(self, step):
        st, rec = self._step_state(step)
        if st == "done":
            self._apply_step_value(step, rec.get("value"))

    def _apply_step_value(self, step, value):
        """Write a step result into the model (cfg) and have the main
        window reload the tabs that show it."""
        c = self.cfg
        if step == "ms":
            c.step.mass_scaling_enabled = True
            c.step.mass_scaling_factor = _r6(value)
        elif step == "mesh":
            c.elem_size = _r6(value)
        elif step == "domain":
            g = c.euler_geometry
            # Rounded up: D* is a whole number of elements.
            g.h_wp, g.h_void, g.l_wp, g.l_void = (round6_up(v) for v in value)
        else:
            return
        self._log_ui("[MODEL] %s written in %s" % (format_value(step, value),
                                                   _MODEL_WHERE[step]))
        if self._model_refresher is not None:
            try:
                self._model_refresher(step)
            except Exception:
                log_swallowed("refreshing the model tabs", level=logging.DEBUG)
        self.changed.emit()
        self._draw_preview()
        self._refresh_step_status()

    # =================================================================
    # Questions to the user (patched in the tests)
    # =================================================================
    def _ask(self, title, text, choices, default=None):
        """Modal question; `choices` = [(key, label), ...]. The default
        button (Enter) is `default`, else the first choice. Returns the key
        clicked, or None."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(title)
        box.setText(text)
        keys, buttons = {}, []
        for i, (key, label) in enumerate(choices):
            if key in ("cancel", "later"):
                role = QMessageBox.RejectRole       # also Esc
            elif i == 0:
                role = QMessageBox.AcceptRole
            else:
                role = QMessageBox.ActionRole
            b = box.addButton(label, role)
            keys[b] = key
            buttons.append((key, b))
        if buttons:
            box.setDefaultButton(next((b for k, b in buttons if k == default),
                                      buttons[0][1]))
        box.exec()
        return keys.get(box.clickedButton())

    def _inform(self, title, text):
        QMessageBox.information(self, title, text)

    def _confirm_prerequisites(self, step) -> bool:
        mkey, skey = self._keys()
        missing = missing_prerequisites(self._steps(), step, mkey, skey,
                                        self._model_values())
        if not missing:
            return True
        mv = self._model_values()
        now = ", ".join(format_value(up, mv[_MODEL_NAME[up]])
                        for up in STEPS[:STEPS.index(step)])
        text = ("%s should run after the earlier steps, on their results:\n\n"
                "• %s\n\nRun it anyway? It runs with the values the model "
                "has now (%s): its result counts only if the earlier steps "
                "give these same values, otherwise it has to be run again."
                % (STEP_TITLES[step], "\n• ".join(missing), now))
        return self._ask("Order of the steps", text,
                         [("run", "Run anyway"), ("cancel", "Cancel")],
                         default="cancel") == "run"

    # =================================================================
    # Starting a study: new, resume, load
    # =================================================================
    def _launch_new(self, step, then=None, confirm=True) -> bool:
        """Start a new study of `step` from the panel's settings. False when
        nothing started (a warning was shown or the user declined)."""
        if step == "checks":
            return self._run_checks("new", then=then)
        spec = {"ms": self._ms_spec, "mesh": self._gci_spec,
                "domain": self._domain_spec}[step]()
        if spec is None:
            return False
        if confirm and not self._confirm_prerequisites(step):
            return False
        if self._is_busy:          # something started during the question
            return False
        val = self._validate_launch()
        if val is None:
            return False
        prefs, wd, cpus = val
        spec["cache_folders"] = self._known_folders()
        prefix = {"ms": "massscaling", "mesh": "GCI",
                  "domain": "domainsizing"}[step]
        run_dir = self._study_run_dir(wd, prefix, spec)
        started = self._start_step(step, spec, run_dir, prefs, cpus, "new",
                                   then)
        if started and (step == "mesh" or (
                step == "ms" and self._float_or(self.le_ms_elem, None)
                is None)):
            self._freeze_plan_start()     # the plan this study used
        return started

    def _start_step(self, step, spec, folder, prefs, cpus, mode, then=None,
                    **extra) -> bool:
        start = {"ms": self._start_ms, "mesh": self._start_gci,
                 "domain": self._start_domain}[step]
        return bool(start(spec, Path(folder), prefs, cpus, mode=mode,
                          then=then, **extra))

    def _begin_study(self, step, spec, run_dir, prefs, cpus, mode, then,
                     **extra):
        """Common start of every study: the run cache, the context read back
        by _after_study, the log header. Returns the run launcher."""
        self._cancel_evt.clear()
        self._study_offline = (mode == "load")
        if step == "domain":
            self._last_domain_ok = False    # usable once it ends well
        self._study_cache = RunCache(
            [run_dir] + [f for f in (spec.get("cache_folders") or []) if f])
        run_bundle = self._make_run_bundle(prefs, run_dir, cpus,
                                           JOB_PREFIX[step])
        params = self._model_params()
        try:
            settings = settings_of_spec(spec)
        except (KeyError, TypeError, ValueError):
            settings = {}
        self._active = {
            "step": step, "mode": mode, "spec": spec,
            "folder": Path(run_dir), "then": then, "prefs": prefs,
            "cpus": cpus, "run_bundle": run_bundle, "model_params": params,
            "model_key": model_key(params), "settings": settings,
            "settings_key": settings_key(settings),
            "inputs": self._inputs_of(step, spec)}
        self._active.update(extra)
        if mode != "load" and "prev_state" not in extra:
            st0, rec0 = step_state(self._steps(), step,
                                   self._active["model_key"],
                                   self._active["settings_key"])
            if st0 in ("done", "prereq"):
                # Kept if this study ends without a result (_after_study).
                self._active["kept_done"] = rec0
        if mode == "new" and then is None and step != "checks":
            self.log.clear()
        self.tabs.setCurrentIndex(0)
        if mode == "resume":
            self._log_ui("RESUMING the study in %s: its finished runs are "
                         "reused, the missing ones are launched" % run_dir)
        elif mode == "load":
            self._log_ui("LOADING %s: no Abaqus run, the finished runs are "
                         "read back" % run_dir)
        n = len(self._study_cache)
        if n:
            self._log_ui("  %d finished run(s) available for reuse" % n)
        return run_bundle

    @staticmethod
    def _inputs_of(step, spec) -> dict:
        """Results of the earlier steps a study ran with (study_state)."""
        try:
            if step == "ms":
                return {"h": float(spec["elem_size"]),
                        "dims": _dims_list(spec["domain_dims"])}
            if step == "mesh":
                return {"ms": ms_of(spec.get("base_ms")),
                        "dims": _dims_list(spec["domain_dims"])}
            if step == "domain":
                return {"ms": ms_of(spec.get("base_ms")),
                        "h": float(spec["elem_size"])}
            return {"ms": ms_of(spec.get("base_ms")),
                    "h": float(spec["h_star"]),
                    "dims": _dims_list(spec["d_star"])}
        except (KeyError, TypeError, ValueError):
            return {}

    def _offer_resume(self, step) -> bool:
        """When the last study of `step` was interrupted for this model, ask
        whether to resume it. True when the click was handled here."""
        st, rec = self._step_state(step)
        kept = None
        if st in ("done", "prereq"):
            att = self._resumable_attempt(step, rec)
            if att is None:
                return False
            kept, rec, st = rec, att, "interrupted"
        if st != "interrupted" or not rec.get("folder"):
            return False
        folder = Path(rec["folder"])
        if not self._resumable_folder(step, folder):
            return False
        choice = self._ask(
            STEP_TITLES[step],
            "The last %s was interrupted (%s, folder %s).%s\n\nResume it? "
            "Its finished runs are reused and it goes on with the settings "
            "it was started with." % (
                STEP_TITLES[step], _when(rec), folder.name,
                "" if kept is None else " The step keeps its result (%s) "
                "unless this study ends with one." % (
                    "every check passed" if step == "checks" else
                    format_value(step, kept.get("value")))),
            [("resume", "Resume"), ("new", "Start a new study"),
             ("cancel", "Cancel")])
        if self._is_busy:          # something started during the question
            return True
        if choice == "resume":
            self._resume_step(step, folder)
            return True
        return choice != "new"

    def _resumable_folder(self, step, folder) -> bool:
        """The folder still holds the study's settings (and, for the final
        checks, is the folder of step 2's result)."""
        folder = Path(folder)
        marker = CHECKS_CONFIG if step == "checks" else "config.json"
        if not (folder / marker).exists():
            return False
        if step == "checks":
            dom = self._domain_record_folder()
            return dom is not None and same_folder(dom, folder)
        return True

    def _resumable_attempt(self, step, rec):
        """The later study noted next to a valid result of `step` when it
        can be resumed: interrupted, made for this model and these settings
        on the current results of the earlier steps, its folder still
        there. None otherwise."""
        att = (rec or {}).get("attempt")
        if (not att or att.get("status") != "interrupted"
                or att.get("resumable") is False or not att.get("folder")):
            return None
        mkey, skey = self._keys()
        if step_state(dict(self._steps(), **{step: att}), step, mkey,
                      skey)[0] != "interrupted":
            return None
        return att if self._resumable_folder(step, att["folder"]) else None

    def _resume_step(self, step, folder, then=None, **extra) -> bool:
        """Run the study of `folder` again (its finished runs are reused).
        `extra` goes into the study context (_begin_study)."""
        if self._is_busy:
            return False
        folder = Path(folder)
        if step == "checks":
            spec = read_checks_config(folder)
            if spec is None:
                QMessageBox.warning(self, STEP_TITLES[step],
                                    "%s holds no saved settings for the final "
                                    "checks: run them again." % folder.name)
                return False
            return self._run_checks("resume", then=then, spec=spec, **extra)
        try:
            found, spec = read_study_folder(folder, self._spec_defaults())
        except ValueError as e:
            QMessageBox.warning(self, STEP_TITLES[step], str(e))
            return False
        if found != step:
            QMessageBox.warning(self, STEP_TITLES[step],
                                "%s is not a study of this step." % folder.name)
            return False
        val = self._validate_launch(folder)
        if val is None:
            return False
        prefs, _wd, cpus = val
        return self._start_step(step, spec, folder, prefs, cpus, "resume",
                                then, **extra)

    # ---- loading -------------------------------------------------------
    def _on_open_study(self):
        if self.is_running():
            return
        prefs = self._prefs_getter() if self._prefs_getter else None
        start = str(getattr(prefs, "default_workdir", "") or "")
        folder = QFileDialog.getExistingDirectory(
            self, "Open a study folder (read back, no Abaqus run)", start)
        if folder:
            self.open_study(folder)

    def open_study(self, folder, restore=True, then=None) -> bool:
        """Read back the study in `folder` without any Abaqus run (replay of
        its finished runs). With `restore`, the panel takes the settings the
        study was made with first. A step-2 folder also reads back its final
        checks."""
        if self._is_busy:
            self._inform("Open a study", "A study is running: wait for it "
                                         "to end or cancel it first.")
            return False
        folder = Path(folder)
        try:
            step, spec = read_study_folder(folder, self._spec_defaults())
        except ValueError as e:
            QMessageBox.warning(self, "Open a study", str(e))
            return False
        # What the step holds now, and the panel as it is: a load that does
        # not end with a result gives both back.
        prev_state, prev_rec = self._step_state(step)
        panel = self._panel_snapshot() if restore else None
        if restore:
            self._restore_settings(step, spec)
        if then is None and step == "domain":
            then = self._after_domain_load
        if self._start_step(step, spec, folder, None, self._cpus(), "load",
                            then, prev_state=prev_state, prev_rec=prev_rec,
                            panel=panel):
            return True
        self._panel_restore(panel)
        return False

    def _panel_snapshot(self):
        spins = [self.sp_margin, self.sp_gci_n] + list(self._dom_spins.values())
        return ([(le, le.text()) for le in self._opt_line_edits()],
                [(sp, int(sp.value())) for sp in spins])

    def _panel_restore(self, snap):
        """Put the panel back as `snap` (from _panel_snapshot) had it."""
        if not snap:
            return
        les, spins = snap
        changed = False
        for le, text in les:
            if le.text() != text:
                le.setText(text)
                changed = True
        for sp, v in spins:
            if int(sp.value()) != v:
                sp.setValue(v)
                changed = True
        if changed:
            self._log_ui("The settings of the tab were put back as they were "
                         "before the study was opened.")
        self._refresh_step_status()

    def _restore_settings(self, step, spec):
        """Put the settings of a study back in the panel (only those that
        differ), so its record matches the panel."""
        changed = []

        def same(a, b):
            return (a is not None and b is not None
                    and math.isclose(float(a), float(b), rel_tol=1e-9,
                                     abs_tol=1e-15))

        def put(le, value, name, current=None):
            if value is None:
                return
            cur = self._float_or(le, None) if current is None else current
            if same(cur, value):
                return
            le.setText("%.9g" % float(value))
            changed.append(name)

        for k, v, cur in zip(ZOI_KEYS, zoi_tuple(spec), self.zoi()):
            put(self.le_zoi[k], v, "ZOI " + k, current=cur)
        for q, v in eps_of_spec(spec).items():
            if q in self._q_eps:
                put(self._q_eps[q], v, "ε_q " + q)
        w = spec.get("window") or [None, None]
        put(self._dom_texts["window_start"], w[0], "T start")
        put(self._dom_texts["window_end"], w[1], "T end")
        put(self._dom_texts["rk_max"], spec.get("rk_max"), "G_K")
        put(self._dom_texts["rhg_max"], spec.get("rhg_max"), "G_HG")
        grid = grid_set_of_spec(spec)
        if grid is None:
            if self.le_grid_step.text().strip():
                self.le_grid_step.setText("")       # blank: the element size
                changed.append("ZOI sampling step")
        else:
            put(self.le_grid_step, grid, "ZOI sampling step",
                current=self._grid_step_set())
        if step == "ms":
            vals = [float(v) for v in spec.get("ms_values") or []]
            try:
                cur_vals = list(self.ms_settings()[0])
            except ValueError:
                cur_vals = []
            if vals and (len(vals) != len(cur_vals) or not all(
                    same(a, b) for a, b in zip(vals, cur_vals))):
                self.le_ms_values.setText(", ".join("%g" % v for v in vals))
                changed.append("ms values")
            try:
                cur_elem = self.ms_settings()[1]
            except ValueError:
                cur_elem = None
            put(self.le_ms_elem, spec.get("elem_size"), "ms element size",
                current=cur_elem)
        elif step == "mesh":
            put(self.le_gci_finest, spec.get("finest_elem_size"),
                "finest element size",
                current=self._float_or(self.le_gci_finest, None))
            put(self.le_gci_ratio, spec.get("ratio"), "refinement ratio")
            n = spec.get("n_meshes")
            if n is not None and int(n) != int(self.sp_gci_n.value()):
                self.sp_gci_n.setValue(int(n))
                changed.append("number of meshes")
            minh = spec.get("min_elem_size")
            if minh is None:
                if self.le_gci_min.text().strip():
                    self.le_gci_min.setText("")
                    changed.append("smallest finest size")
            else:
                put(self.le_gci_min, minh, "smallest finest size")
        elif step == "domain":
            ints = (("margin_elems", self.sp_margin, "margin"),
                    ("step_elems", self._dom_spins["dom_step_elems"],
                     "growth step"),
                    ("n_max", self._dom_spins["dom_n_max"],
                     "max comparisons"),
                    ("n_hold", self._dom_spins["dom_n_hold"], "passes"),
                    ("m_ratios", self._dom_spins["dom_m_ratios"], "m"))
            for key, sp, name in ints:
                v = spec.get(key)
                if v is not None and int(v) != int(sp.value()):
                    sp.setValue(int(v))
                    changed.append(name)
            caps = spec.get("caps") or {}
            for d, le in self._max.items():
                v = caps.get(d)
                if v is None:
                    if le.text().strip():
                        le.setText("")
                        changed.append("largest %s" % d)
                else:
                    put(le, v, "largest %s" % d)
        if changed:
            self._log_ui("Settings of the study put back in the tab: %s"
                         % ", ".join(changed))
        self._refresh_step_status()

    def _after_domain_load(self, _step, _rec):
        """After a step-2 folder was read back: read back its final checks
        too, when it has some."""
        study, folder = self._domain_in_memory()
        if study is None or folder is None or self.is_running():
            return
        spec = read_checks_config(folder)
        if spec is None:
            if any(Path(folder).glob("checks_run*.meta.json")):
                self._log_ui("The final checks of this folder have no saved "
                             "settings (older study): run them again.")
            return
        self._start_checks(spec, study, Path(folder), None, self._cpus(),
                           mode="load", then=None)

    # =================================================================
    # End of a study: record, extend, report a short load
    # =================================================================
    def _after_study(self, step, res=None, error=None):
        """Called by every done/fail handler. Records the outcome of the
        study started by _begin_study (no-op without one)."""
        act = getattr(self, "_active", None)
        if not act or act.get("step") != step:
            return
        self._active = None
        self._study_offline = False
        state = getattr(act.get("run_bundle"), "state", None) or {}
        if state.get("miss") is not None:
            self._on_load_miss(act, state["miss"])
            return
        cancelled = self._cancel_evt.is_set()
        if state.get("launch_error"):
            # Not a result of the model: the study stays resumable.
            status, value, extra = "interrupted", None, {}
            message = ("a run could not be made (%s): fix the cause, then run "
                       "the step again to resume it" % state["launch_error"])
        else:
            status, value, message, extra = self._outcome(
                step, res, error, cancelled, state.get("analysis_failed"))
        if step == "mesh" and act.get("plan"):
            extra["plan"] = act["plan"]      # rerun on D* by the checks
        if status == "unbracketed":
            if cancelled:
                # Never extend (launch more runs) after a Cancel.
                status, message = "interrupted", "cancelled"
            elif (act["mode"] != "load" and not act.get("extended")
                    and self._extend_ms(act)):
                return
            else:
                status = "failed"
        if status == "interrupted" and act["mode"] == "load":
            self._log_ui("Loading stopped (%s): the previous state of %s is "
                         "kept." % (message, STEP_TITLES[step]))
            if self._record_kept(act)[1]:
                self._panel_restore(act.get("panel"))
            self._finish(act, None)
            return
        if status != "done" and (act["mode"] == "load" or (
                act["mode"] == "resume" and "prev_state" in act)):
            # A study read back (or resumed from a read-back) that did not
            # succeed does not replace a valid result of the step, unless it
            # is the study of that result and it failed.
            kept, restore = self._record_kept(act)
            if kept is not None and (
                    status == "interrupted" or not same_folder(
                        kept.get("folder") or ".", act["folder"])):
                self._log_ui("%s %s: %s. The current result of the step "
                             "(%s, folder %s) is kept." % (
                                 Path(act["folder"]).name,
                                 STATE_LABELS.get(status, status), message,
                                 format_value(step, kept.get("value")),
                                 Path(kept.get("folder") or "").name))
                if restore:
                    self._panel_restore(act.get("panel"))
                self._finish(act, None)
                return
        kept = act.get("kept_done")
        if (status != "done" and kept is not None
                and self._steps().get(step) is kept):
            # The step was done for this model and these settings before
            # this study: it keeps that result, this study is noted next to
            # it (resumable when interrupted, its runs reusable).
            attempt = make_record(status, value, act["folder"],
                                  act["model_key"], act["settings_key"],
                                  act["inputs"], message,
                                  settings=act["settings"], **extra)
            before = act.get("checks_config_before")
            if before is not None and same_folder(kept.get("folder") or ".",
                                                  act["folder"]):
                # Same folder as the result kept: its settings file goes
                # back, so the folder still reads back that result. This
                # run is not resumable (running the checks again reuses
                # its finished runs).
                try:
                    (Path(act["folder"]) / CHECKS_CONFIG).write_text(
                        before, encoding="utf-8")
                except OSError:
                    log_swallowed("putting back %s" % CHECKS_CONFIG)
                attempt["resumable"] = False
            kept["attempt"] = attempt
            self.changed.emit()
            self._log_ui("[%s] %s%s. The step keeps its result (%s, folder "
                         "%s)." % (
                             STEP_TITLES[step],
                             STATE_LABELS.get(status, status),
                             ": %s" % message if message else "",
                             "every check passed" if step == "checks" else
                             format_value(step, kept.get("value")),
                             Path(kept.get("folder") or "").name))
            self._refresh_step_status()
            self._draw_preview()
            self._finish(act, attempt)
            return
        if step == "domain" and status in ("done", "failed"):
            # The step-2 study in memory is the one of record: the final
            # checks may use it.
            self._last_domain_ok = True
        rec = make_record(status, value, act["folder"], act["model_key"],
                          act["settings_key"], act["inputs"], message,
                          model_params=act["model_params"],
                          settings=act["settings"], **extra)
        old = self._steps().get(step) or {}
        if (act["mode"] == "load" and old.get("folder") == rec["folder"]
                and old.get("status") == status
                and values_match(step, old.get("value"), value)):
            # The study of record read back: the same record.
            rec["finished_at"] = old.get("finished_at", rec["finished_at"])
            if old.get("attempt"):
                rec["attempt"] = old["attempt"]
        self._steps()[step] = rec
        self.changed.emit()
        self._log_ui("[%s] %s%s" % (
            STEP_TITLES[step], STATE_LABELS.get(status, status),
            (": %s" % format_value(step, value)) if status == "done"
            and step != "checks" else (": %s" % message if message else "")))
        now = self._step_state(step)[0]
        if (now == "done" and step in _MODEL_NAME and not self._pipeline
                and not in_model(step, self._model_values()[
                    _MODEL_NAME[step]], value)):
            self._log_ui("  next: 'Use in model' writes it in %s"
                         % _MODEL_WHERE[step])
        elif now == "prereq" and not self._pipeline:
            self._log_ui("  " + self._prereq_sentence(step, rec))
        self._refresh_step_status()
        self._draw_preview()
        self._finish(act, rec)

    def _finish(self, act, rec):
        then = act.get("then")
        if then is not None:
            step = act["step"]
            QTimer.singleShot(0, lambda: then(step, rec))

    def _record_kept(self, act):
        """(record, put the panel back) when a study read back (or resumed
        from a read-back) gives no result. The record is the valid result
        of the step that stays, or None when there is none: valid for the
        settings the study was opened with (now in the panel, which then
        stays as it is), else valid before the study was opened (the panel
        then goes back as it was)."""
        step = act["step"]
        steps = self._steps()
        now = step_state(steps, step, act["model_key"],
                         act["settings_key"])[0]
        if now in ("done", "prereq"):
            return steps.get(step), False
        if "prev_state" in act:
            prev_state, prev_rec = act.get("prev_state"), act.get("prev_rec")
        else:
            prev_state, prev_rec = self._step_state(step)
        if prev_state in ("done", "prereq") and prev_rec:
            return prev_rec, True
        return None, True

    def _exports_allowed(self, step=None, res=None) -> bool:
        """Whether the study that is ending (`res`: its result) writes its
        export files: not when it was only read back, nor, when its folder
        holds the files of a valid result of the step, unless it ends with
        a result itself."""
        act = getattr(self, "_active", None)
        if not act:
            return True
        if act.get("mode") == "load":
            return False
        kept = act.get("prev_rec") if "prev_state" in act else \
            act.get("kept_done")
        if not kept or not same_folder(kept.get("folder") or ".",
                                       act["folder"]):
            return True
        state = getattr(act.get("run_bundle"), "state", None) or {}
        if self._cancel_evt.is_set() or state.get("launch_error"):
            return False
        try:
            return self._outcome(step or act["step"], res, None, False,
                                 state.get("analysis_failed"))[0] == "done"
        except Exception:
            return False

    @staticmethod
    def _outcome(step, res, error, cancelled, analysis_failed=None):
        """(status, value, message, extra fields) of a study that ended.
        `analysis_failed`: jobs whose analysis did not complete."""
        if res is None:
            if cancelled:
                return "interrupted", None, "cancelled", {}
            if analysis_failed:
                # The same runs fail again on a resume: not resumable.
                return ("failed", None, "the analysis of %s did not complete "
                        "(see its .msg file in the study folder)"
                        % ", ".join(analysis_failed), {})
            return ("interrupted", None, "stopped by an error: %s"
                    % (error or "unknown"), {})
        if step == "ms":
            values = [float(v) for v in (res.ms_values or [])]
            if cancelled and res.status != "converged":
                # A run stopped by the Cancel says nothing about ms.
                return "interrupted", None, "cancelled", {}
            if res.status == "converged" and res.retained is not None:
                return "done", float(res.retained), "", {"values": values}
            if res.status == "upper_end":
                return ("unbracketed", None,
                        "every comparison passed up to ms = %g, so the "
                        "largest acceptable ms is not bracketed: add larger "
                        "values" % values[-1], {"values": values})
            if res.status == "below_range":
                return ("failed", None, "the first comparison already "
                        "failed: start the ms values lower", {})
            return "interrupted", None, "cancelled", {}
        if step == "mesh":
            if res.stopped_by == "cancelled" or cancelled:
                return "interrupted", None, "cancelled", {}
            h = res.recommended_size
            if h is None:
                return ("failed", None, "no mesh of the plan stays within "
                        "ε_q of the reference value: start the plan at "
                        "a finer mesh", {})
            msg = ("" if res.in_asymptotic_range else
                   "not every quantity is in the asymptotic range, the GCI "
                   "values are indicative")
            return "done", float(h), msg, {}
        if step == "domain":
            if cancelled and res.status != "converged":
                return "interrupted", None, "cancelled", {}
            if res.status == "converged":
                d = res.final
                return ("done", [float(d.h_wp), float(d.h_void),
                                 float(d.l_wp), float(d.l_void)], "", {})
            if res.status == "partial":
                return ("failed", None, "at least one side did not converge "
                        "(largest tested size kept): raise 'max comparisons "
                        "per side' or the largest size allowed (Advanced "
                        "parameters)", {})
            if res.status == "zoi_outside":
                return ("failed", None, "the ZOI is not inside the starting "
                        "domain", {})
            return "interrupted", None, "cancelled", {}
        # checks
        if res.status == "accepted":
            return "done", "accepted", "", {}
        if res.status == "rejected":
            bad = ["%s: %s" % (_CHECK_TITLES.get(c.name, c.name),
                               c.details.get("action") or c.conclusion)
                   for c in res.checks if c.passed is False]
            return ("failed", None, "a check failed (%s)" % "; ".join(bad),
                    {"actions": bad})
        if res.status == "incomplete":
            names = [_CHECK_TITLES.get(c.name, c.name)
                     for c in res.checks if c.passed is None]
            return ("failed", None, "a check could not be evaluated (%s)"
                    % ", ".join(names), {})
        return "interrupted", None, "cancelled", {}

    def _extend_ms(self, act) -> bool:
        """Every ms comparison passed: add values by doubling the last one
        and go on in the same folder (the finished runs are replayed)."""
        spec = dict(act["spec"])
        vals = [float(v) for v in spec.get("ms_values") or []]
        if not vals:
            return False
        add = [vals[-1] * 2.0 ** k for k in range(1, MS_EXTENSION + 1)]
        spec["ms_values"] = vals + add
        rewrite_study_config(act["folder"], spec)
        self._log_ui("=" * 68)
        self._log_ui("Every comparison passed up to ms = %g: the largest "
                     "acceptable ms is not bracketed yet. The study goes on "
                     "with %s (the runs already done are reused)."
                     % (vals[-1], ", ".join("%g" % v for v in add)))
        prefs, cpus = act.get("prefs"), act.get("cpus")
        if prefs is None:
            val = self._validate_launch(act["folder"])
            if val is None:
                return False
            prefs, _wd, cpus = val
        keep = {k: act[k] for k in ("prev_state", "prev_rec", "panel")
                if k in act}
        return bool(self._start_ms(spec, act["folder"], prefs, cpus,
                                   mode="resume", then=act.get("then"),
                                   extended=True, **keep))

    def _on_load_miss(self, act, miss):
        """Loading stopped on a run the folder does not have: either the
        study is not finished (same model) or it was made for another
        model. Judged on the folder's own runs of this study only (runs
        borrowed from other folders or of another study kept there say
        nothing about it)."""
        step, folder = act["step"], act["folder"]
        # The partial replay is no result: nothing in memory may use it.
        if step == "ms":
            self._last_ms = None
        elif step == "mesh":
            self._last_gci = None
        elif step == "domain":
            self._last_domain_ok = False
        cache = getattr(self, "_study_cache", None)
        head = JOB_PREFIX[step].lower() + "_run"

        def mine(runs):
            return [r for r in runs if same_folder(r.folder, folder)
                    and r.job.lower().startswith(head)]
        own = mine(cache.runs) if cache is not None else []
        # Runs whose analysis did not complete say which model the folder
        # was made for too.
        pool = own + (mine(cache.failed) if cache is not None else [])
        n_own = len(own)
        run, diffs = (cache.closest(miss, ignore=AXIS_KEYS, runs=pool)
                      if pool else (None, []))
        title = "Open a study"
        name = Path(folder).name
        if run is not None and diffs:
            lines = ["  %s: %s in the study, %s now" % (k, _fmt_leaf(a),
                                                        _fmt_leaf(b))
                     for k, a, b in diffs[:8]]
            if len(diffs) > 8:
                lines.append("  ... and %d more" % (len(diffs) - 8))
            text = ("%s was made for another model. Its runs differ from the "
                    "current model in:\n%s\n\nOpen the profile it was made "
                    "with (File > Open), or set these values back, then open "
                    "the study again." % (name, "\n".join(lines)))
            self._log_ui(text)
            self.lbl_status.setStyleSheet("color: #b91c1c;")
            self.lbl_status.setText("%s: study made for another model"
                                    % STEP_TITLES[step])
            self._panel_restore(act.get("panel"))
            self._inform(title, text)
            self._finish(act, None)
            return
        # Not finished for this model. A valid result of the step is never
        # replaced by it unless the resumed study ends with a result.
        kept, restore = self._record_kept(act)
        keep = kept is not None
        prev_rec = kept or {}
        panel = act.get("panel") if restore else None
        same = keep and same_folder(prev_rec.get("folder") or ".", folder)
        rec = None
        if not keep:
            rec = make_record("interrupted", None, folder, act["model_key"],
                              act["settings_key"], act["inputs"],
                              "not finished (%d finished runs)" % n_own,
                              model_params=act["model_params"],
                              settings=act["settings"])
            self._steps()[step] = rec
            self.changed.emit()
            self._refresh_step_status()
        if same:
            text = ("Some runs of the %s in %s are missing (%d finished "
                    "run(s) found; the others were removed, or did not "
                    "complete before this version kept track of it). Its "
                    "result stays recorded.\n\nCompute the missing runs now "
                    "with Abaqus? The finished ones are reused."
                    % (STEP_TITLES[step], name, n_own))
            self._log_ui("%s: some runs are missing (%d finished runs); the "
                         "result of the step is kept." % (name, n_own))
            status_text = "%s: runs missing in its folder" % STEP_TITLES[step]
        else:
            text = ("The %s in %s is not finished: %d run(s) are done.\n\n"
                    "Resume it now? The missing runs are launched with "
                    "Abaqus; the finished ones are reused." % (
                        STEP_TITLES[step], name, n_own))
            if keep:
                text += ("\n\nUntil it ends, the step keeps its current "
                         "result (%s, folder %s)." % (
                             format_value(step, prev_rec.get("value")),
                             Path(prev_rec.get("folder") or "").name))
            self._log_ui("%s is not finished (%d finished runs)%s." % (
                name, n_own, ": run the step again to resume it"
                if rec is not None else ""))
            status_text = "%s: study not finished" % STEP_TITLES[step]
        self.lbl_status.setStyleSheet("color: #b45309;")
        self.lbl_status.setText(status_text)
        if self._ask(title, text, [("resume", "Resume now"),
                                   ("later", "Not now")],
                     default="later") == "resume":
            # With a valid result kept, the resumed study replaces it only
            # if it ends with a result (_after_study).
            extra = ({"prev_state": "done", "prev_rec": kept, "panel": panel}
                     if keep else {})
            if self._resume_step(step, folder, then=act.get("then"),
                                 **extra):
                return
        if rec is None:
            self._panel_restore(panel)
        self._finish(act, rec)

    # =================================================================
    # Final checks: the domain study they need
    # =================================================================
    def _domain_record_folder(self):
        """Folder of the step-2 study valid for this model, or None."""
        st, rec = self._step_state("domain")
        if st not in ("done", "prereq") or not rec.get("folder"):
            return None
        f = Path(rec["folder"])
        return f if (f / "config.json").exists() else None

    def _domain_in_memory(self):
        """(StudyResult, folder) of the domain study held in memory, when
        the checks can use it: it has runs, and it is the study of record
        when one is valid."""
        res = getattr(self, "_last_domain_result", None)
        if (res is None or not res.runs
                or res.status not in ("converged", "partial")
                or not getattr(self, "_last_domain_ok", True)):
            return None, None
        folder = getattr(self, "_last_domain_dir", None)
        rec_folder = self._domain_record_folder()
        if rec_folder is not None:
            try:
                if folder is None or (Path(folder).resolve()
                                      != rec_folder.resolve()):
                    return None, None
            except OSError:
                return None, None
        return res, folder

    def _checks_available(self) -> bool:
        return (self._domain_in_memory()[0] is not None
                or self._domain_record_folder() is not None)

    def _gci_plan_for_checks(self, h_star) -> dict:
        """The mesh plan the checks run again on D*: the one of step 1 when
        step 1 is done for this model, else the panel's."""
        st, rec = self._step_state("mesh")
        plan = (rec or {}).get("plan") if st == "done" else None
        if isinstance(plan, dict) and plan.get("finest_elem_size"):
            return {"finest_elem_size": float(plan["finest_elem_size"]),
                    "ratio": float(plan.get("ratio") or 2.0),
                    "n_meshes": int(plan.get("n_meshes") or 3),
                    "min_elem_size": plan.get("min_elem_size")}
        return {"finest_elem_size": self._float_or(self.le_gci_finest, h_star),
                "ratio": self._float_or(self.le_gci_ratio, 2.0),
                "n_meshes": int(self.sp_gci_n.value()),
                "min_elem_size": self._float_or(self.le_gci_min, None)}

    def _run_checks(self, mode="new", then=None, spec=None, **extra) -> bool:
        """Start the final checks on the domain study of record; it is read
        back from its folder first when it is not in memory."""
        if self._is_busy:
            return False
        study, folder = self._domain_in_memory()
        if study is None:
            dom = self._domain_record_folder()
            if dom is None:
                QMessageBox.warning(self, "Interaction checks",
                                    "Run a domain study first.")
                return False
            self._log_ui("Reading back the step-2 study (%s) for the "
                         "checks…" % dom.name)

            def after(_step, _rec):
                # Nothing more after a Cancel; `then` hears of every stop.
                if (not self._cancel_evt.is_set() and not self._is_busy
                        and self._domain_in_memory()[0] is not None
                        and self._run_checks(mode, then, spec, **extra)):
                    return
                if then is not None:
                    then("checks", None)
            return self.open_study(dom, restore=False, then=after)
        if spec is None:
            spec = self._checks_spec(study)
            if spec is None:
                return False
        if mode == "load":
            prefs, wd, cpus = None, None, self._cpus()
        else:
            val = self._validate_launch(folder)
            if val is None:
                return False
            prefs, wd, cpus = val
        return bool(self._start_checks(spec, study, Path(folder or wd), prefs,
                                       cpus, mode=mode, then=then, **extra))

    # =================================================================
    # Whole pipeline
    # =================================================================
    def _on_run_all(self):
        if self.is_running():
            return
        if not self.thresholds_complete():
            QMessageBox.warning(self, "Run all steps",
                                "Set the six admitted deviations ε_q "
                                "(Vx, Vy, T, EVF, Fc, Ff) first.")
            return
        try:
            self._current_settings()
        except ValueError as e:
            QMessageBox.warning(self, "Run all steps", str(e))
            return
        lines = []
        for step in STEPS:
            st, _rec = self._step_state(step)
            what = {"done": "already done for this model: skipped",
                    "interrupted": "interrupted: resumed"}.get(st, "run")
            lines.append("%s: %s" % (STEP_TITLES[step], what))
        text = ("The steps run one after the other:\n\n%s\n\nEach result is "
                "written into the model (Step, Mesh and Geometry tabs) before "
                "the next step starts. The pipeline stops at the first step "
                "that does not succeed. Start?" % "\n".join(lines))
        if self._ask("Run all steps", text,
                     [("start", "Start"), ("cancel", "Cancel")]) != "start":
            return
        if self.is_running():      # something started during the question
            return
        self._pipeline = True
        self._sync_run_buttons()
        self._pipeline_done = set()     # steps done or reported as done
        self.log.clear()
        self._log_ui("=" * 68)
        self._log_ui("RUN ALL STEPS")
        self._log_ui("=" * 68)
        self._pipeline_next()

    def _pipeline_next(self, *_):
        if not self._pipeline:
            return
        steps = self._steps()
        for step in STEPS:
            mkey, skey = self._keys()
            st, rec = step_state(steps, step, mkey, skey)
            if st == "done":
                if step in _MODEL_NAME and not in_model(
                        step, self._model_values()[_MODEL_NAME[step]],
                        rec.get("value")):
                    self._apply_step_value(step, rec.get("value"))
                if step not in self._pipeline_done:
                    self._pipeline_done.add(step)
                    self._log_ui("%s already done for this model%s: "
                                 "skipped" % (
                                     STEP_TITLES[step], "" if step == "checks"
                                     else " (%s)" % format_value(
                                         step, rec.get("value"))))
                continue
            self.lbl_status.setStyleSheet("color: #1d4ed8;")
            self.lbl_status.setText("Run all steps: %s…"
                                    % STEP_TITLES[step])
            if not self._pipeline_start(step, st, rec):
                self._pipeline_stop("%s could not start (see the message)"
                                    % STEP_TITLES[step])
            return
        self._pipeline_stop(None)

    def _pipeline_start(self, step, st, rec) -> bool:
        if st == "interrupted" and rec.get("folder"):
            folder = Path(rec["folder"])
            marker = CHECKS_CONFIG if step == "checks" else "config.json"
            ok = (folder / marker).exists()
            if ok and step == "checks":
                dom = self._domain_record_folder()
                ok = dom is not None and dom.resolve() == folder.resolve()
            if ok:
                return self._resume_step(step, folder,
                                         then=self._pipeline_after)
        return self._launch_new(step, then=self._pipeline_after,
                                confirm=False)

    def _pipeline_after(self, step, rec):
        if not self._pipeline:
            return
        if rec is None or rec.get("status") != "done":
            status = (rec or {}).get("status") or "stopped"
            why = (rec or {}).get("message") or STATE_LABELS.get(status,
                                                                 status)
            self._pipeline_stop("%s: %s" % (STEP_TITLES[step], why))
            return
        self._pipeline_done.add(step)
        if step in _MODEL_NAME:
            self._apply_step_value(step, rec.get("value"))
        st = self._step_state(step)[0]
        if st != "done":
            # Never run a step again in a loop: its record should be done
            # for the current model and settings right after it ended.
            self._pipeline_stop("%s ended, but its result does not count as "
                                "done for the current model and settings "
                                "(%s)" % (STEP_TITLES[step], state_label(st)))
            return
        QTimer.singleShot(0, self._pipeline_next)

    def _pipeline_stop(self, reason):
        self._pipeline = False
        self._sync_run_buttons()
        self._log_ui("=" * 68)
        if reason is None:
            mv = self._model_values()
            self._log_ui("ALL STEPS DONE: %s, %s, %s; final checks passed"
                         % (format_value("ms", mv["ms"]),
                            format_value("mesh", mv["h"]),
                            format_value("domain", mv["dims"])))
            self.lbl_status.setStyleSheet("color: #15803d;")
            self.lbl_status.setText("All steps done: the model is sized.")
        else:
            self._log_ui("RUN ALL STEPS STOPPED: %s" % reason)
            self.lbl_status.setStyleSheet("color: #b45309;")
            self.lbl_status.setText("Run all steps stopped: %s" % reason)
        self._refresh_step_status()
