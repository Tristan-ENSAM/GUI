# GUI Abaqus — pre-processor for CEL/Lagrangian cutting simulations

Qt-based GUI to set up Abaqus orthogonal-cutting simulations:

- CEL or Lagrangian formulation
- Tool / workpiece / Eulerian domain geometry
- Materials (JC plasticity, JC damage, thermal/elastic properties)
- Interaction (tangential formulation, friction, heat generation)
- Boundary & initial conditions (cutting velocity on Eulerian faces,
  per-face inflow/outflow BCs, initial temperature)
- Mesh seeds + per-body element-type (C3D8T / C3D8RT for the Lagrangian
  family, EC3D8R / EC3D8RT for the Eulerian box, with hourglass control,
  distortion control, etc.)
- Job parameters (name, CPUs, working directory)
- Dry-run that prints the exact subprocess command + parameter dict
  that Abaqus would receive

## Running

On Windows:
```
run_gui.bat
```

Debug mode (verbose stdout):
```
run_gui_debug.bat
```

## Running the computations on another PC (remote agent)

The GUI can stay on your PC while Abaqus runs on a compute PC, as long as both
see a common network drive (e.g. `Z:`). No admin rights, SSH or open port are
needed: runs go through a queue folder on that drive
(`gui/core/remote_exec.py`).

1. Same version of this repository on both PCs (`git pull` on both). A run is
   refused if the `abaqus_scripts/` differ.
2. On the compute PC (in its Remote Desktop session): set the Abaqus command
   and script in the GUI Preferences of that PC, then start
   `run_remote_agent.bat --queue Z:\<folder>\queue`. Leave the window open and
   close Remote Desktop with the cross (disconnect, not "Sign out").
3. On your PC: Preferences > Execution > tick "Run Abaqus on the compute PC",
   Queue folder = the same `Z:\<folder>\queue`, and set the default working
   directory on `Z:` too.

Runs execute one at a time in submission order in a local folder of the
compute PC (`C:\TEMP\ABQ_remote\<same path as on Z:>`). The `.sta` and the
script log are copied to `Z:` every 2 s; at the end the results bundle,
`.meta.json`, `.msg`, `.dat`, `.log` and `.inp` are copied back. The `.odb`
stays on the compute PC. Cancel works as usual (the agent runs
`abaqus terminate`). Resume (`continue`) is local-only.

## Layout

```
gui/
├── main.py                # MainWindow, tab wiring, profile save/load
├── core/
│   ├── model_config.py    # All dataclasses + JSON (de)serialisation
│   ├── units.py
│   ├── presets.py
│   └── preferences.py
├── presets/materials.json # Default material library
├── tabs/                  # One file per top-level tab
│   ├── analysis_tab.py
│   ├── geometry_tab.py
│   ├── materials_tab.py
│   ├── interaction_tab.py
│   ├── bcs_tab.py
│   ├── mesh_tab.py
│   └── job_tab.py
└── widgets/
    ├── param_field.py     # Custom NumField / IntField / BoolField / PairRow
    ├── geometry_preview.py
    └── preferences_dialog.py
```

## Notes

- The Abaqus generator (`cel_model.py`) lives outside this
  repository; the GUI only formats parameters and prints the command.
- `materials_user.json` and `preferences.json` (per-user state) are
  ignored by git.
