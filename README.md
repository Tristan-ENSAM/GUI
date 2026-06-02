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

- The Abaqus generator (`abq_odb_generator.py`) lives outside this
  repository; the GUI only formats parameters and prints the command.
- `materials_user.json` and `preferences.json` (per-user state) are
  ignored by git.
