# Abaqus validation checklist

Parts of this project cannot be exercised in the headless dev/CI environment
because they need a real Abaqus install (model generation, the solver, the
`.odb`, and Windows-specific process control). Everything else is covered by
the automated tests (`python -m pytest tests/`, ~28 tests, headless via
`QT_QPA_PLATFORM=offscreen`).

This checklist captures what must be confirmed **once on the Abaqus PC** after
any change to `abaqus_scripts/run_simul.py`, the launch commands, or the
cancel logic. Tick each item.

## Terminology (ROI vs ZOI)

Two distinct zones, different grids, defined in different tabs:

- **ROI** — model output set, edited in the **Geometry** tab, matched to the
  real measurement fields (DIC / IRT) for the sim-vs-experiment comparison.
  Materialised in the Abaqus model as `ROI_node` / `ROI_elem` (EULER instance).
- **ZOI** — measurement zone defined by the user in the **Optimization** tab to
  size the Eulerian domain. Own grid, host-side sampling, not an Abaqus set.
  May default to the ROI but is a separate object.

## 0. Environment

- [ ] `pip install -r requirements.txt` in the GUI Python (3.11+).
- [ ] (Optional) `pip install SALib` if you want the Morris method.
- [ ] In **Preferences → Settings…**, set the Abaqus command (`abaqus.bat`)
      and the generator script path (`abaqus_scripts/run_simul.py`). Paths
      must be free of spaces (Abaqus CLI limitation).
- [ ] Set a default working directory that exists and is writable.

## 1. Dry-run (no Abaqus needed)

- [ ] Job tab → **Generate command (dry-run)**: the printed `model_params`
      dict is literal-only and the command list looks right (cmd, `cae`,
      `noGUI=run_simul.py`, `--model_cfg`, `--run_cfg`).

## 2. Write .inp only

- [ ] Job tab → **Write .inp only**: a `<job>.inp` appears in the working
      directory, no solver runs, no `.odb`/`.results.npz` is produced, and
      the log ends with `[OK] Wrote <job>.inp`.
- [ ] Open the `.inp` and sanity-check: element types (C3D8T/EC3D8RT),
      the cutting/initial velocities, the step time, and the material
      density/specific-heat values (see scaling checks below).

## 3. Single full run

- [ ] Job tab → **Run Abaqus**: live log streams; the progress bar advances
      as `.sta` frames appear; on success a `<job>.results.npz` is written.
- [ ] Results tab → **Load results…**: fields and history load; QoI
      (Fc=RF1, Ff=RF2, peak temperature) look physical.

## 4. Mass scaling vs time scaling (thermal time constant)

For the SAME physical case, compare the `.inp` material cards:

- [ ] Mass scaling κ_m only: Eulerian density = ρ·κ_m and specific heat =
      Cp/κ_m (ρ·Cp preserved); velocities and step time unchanged.
- [ ] Time scaling κ_t only: cutting & initial velocities ×κ_t, step time
      ÷κ_t (machined length v·t unchanged), specific heat = Cp/κ_t, density
      unchanged.
- [ ] Both together: density = ρ·κ_m, specific heat = Cp/(κ_m·κ_t).
- [ ] Result check: contact force / temperature fields match the unscaled
      reference within the paper's tolerance for κ_t ≤ 20 (Hammelmüller &
      Zehetner). Expect divergence if the workpiece is rate-dependent
      (Johnson-Cook C ≠ 0) — the Step tab warns about this.

## 5. Sensitivity (minimal)

- [ ] Tick one material parameter, one scalar QoI, **forward** scheme →
      2 runs; **central** → 3 runs. Confirm the run count matches the cost
      label and that the live estimate (`~/frame`, `~/run`, total) is sane.
- [ ] (If SALib installed) Switch method to **Morris**, N=10 → runs =
      N×(k+1). Confirm μ*/σ table fills in.

## 6. Cancel (Windows-specific — untested off-Windows)

- [ ] Start a run, then **Cancel**: the Abaqus process tree is killed
      (`taskkill /F /T`), no orphaned `standard.exe`/`explicit.exe` remain
      in Task Manager. (The POSIX kill-tree path is tested in CI; the
      Windows path is not.)

## 7. Profile round-trip

- [ ] Set a non-default unit system (Preferences → Unit system…), job name
      and CPUs, save the profile, reopen it: unit system, job name and CPUs
      are restored (the working directory resets to the Preferences default
      by design — it is machine-specific).

## 8. Model sizing studies (Optimization > Model) — first real runs

The study engines (`gui/sensitivity/domain_independence.py`, `mesh_gci.py`,
`interaction_checks.py`, `run_record.py`, `study_export.py`) are unit-tested
headless on analytic bundles; what follows needs a real Abaqus install.

Extraction (abaqus_scripts/cel_results.py):

- [ ] The extraction log prints `ALLAE stored` (hypothesis H2: PRESELECT
      contains ALLAE). If it prints the ALLAE warning instead, R_HG cannot be
      evaluated and every run fails its safeguards.
- [ ] R_HG = ΣALLAE/ΣALLIE over T is plausible for EC3D8RT with the default
      (pure viscous) hourglass control (hypotheses H1, H3 of the report).
- [ ] The bundle holds `history__ENERGY_TIME` and `history__ALLAE_TIME`.

Domain study:

- [ ] The ZOI lies inside the extraction ROI (Geometry tab); otherwise every
      run is refused with "not contained in the extracted ROI".
- [ ] The run log shows, per run, `job ok`, the safeguards (outputs, R_K,
      R_HG) and C_CPU; per comparison, E_q, E_max, q_crit and the mode
      (`tail_bound` / `successive`).
- [ ] C_CPU is filled (the job's .sta is found and its wall time parsed).
- [ ] The domain diagonal only produces warnings; it never stops the study.
- [ ] The study folder holds runs.csv, comparisons.csv, dimensions.csv and
      summary.json.

Mesh GCI study:

- [ ] gci.csv and gci_meshes.csv are written; gci_meshes.csv has a cost per
      mesh; the user's element size is unchanged after the study.

Interaction checks:

- [ ] "Run interaction checks" runs one combined-domain run plus the GCI plan
      on D*, then writes checks.csv and updates summary.json.
- [ ] A failed combined check logs the D11-a action (redo the domain study
      with changed settings).
