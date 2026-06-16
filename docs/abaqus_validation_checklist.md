# Abaqus validation checklist

Parts of this project cannot be exercised in the headless dev/CI environment
because they need a real Abaqus install (model generation, the solver, the
`.odb`, and Windows-specific process control). Everything else is covered by
the automated tests (`python -m pytest tests/`, ~28 tests, headless via
`QT_QPA_PLATFORM=offscreen`).

This checklist captures what must be confirmed **once on the Abaqus PC** after
any change to `abaqus_scripts/run_simul.py`, the launch commands, or the
cancel logic. Tick each item.

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
