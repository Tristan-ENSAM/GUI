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
see a common network drive (e.g. `Z:`). Nothing is installed on the compute
PC, and no admin rights, SSH or open port are needed: runs go through a queue
folder on that drive (`gui/core/remote_exec.py`), and the agent there runs
with the Python bundled with Abaqus (`gui/core/remote_agent.py`).

1. On your PC: Preferences > Execution > tick "Run Abaqus on the compute PC",
   set the Queue folder (e.g. `Z:\ABQ_remote\queue`, no spaces) and the path
   of `abaqus.bat` on the compute PC. On OK the GUI writes `remote_agent.py` and `start_agent.bat`
   into the queue folder.
2. On the compute PC (Remote Desktop): double-click
   `Z:\ABQ_remote\queue\start_agent.bat` and leave the window open. Close
   Remote Desktop with the cross (disconnect), not "Sign out".

Runs execute one at a time in submission order in a local folder of the
compute PC (`C:\TEMP\ABQ_remote\...`, one folder per run), with the model
generator (`abaqus_scripts/*.py`) copied from your PC for each version. `Z:`
is only a transit area: the `.sta` and the script log reach your working
directory every 2 s, and at the end the results bundle, `.meta.json`, `.msg`,
`.dat`, `.log` and `.inp` are moved to it and the run's transit folder on
`Z:` is deleted. The `.odb` stays on the compute PC, in that local folder
(clean it there when its disk fills up). Cancel works as usual (the agent
runs `abaqus terminate`). Resume (`continue`) is local-only.

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
