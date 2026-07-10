# OpenDiHu MU Remodeling Bridge

This bridge exports a remodeled motor-unit/fiber state to OpenDiHu-friendly files.

## Files

- `export_opendihu_mu_json.py`  
  Runs the remodeling model and exports JSON + OpenDiHu files.

- `opendihu_mu_fiber_config.json`  
  Rich fiber/MU table including denervation state, fiber subtype, CSA, radius, and Cm.

- `MU_fibre_distribution_remodeled.txt`  
  OpenDiHu `fiberDistributionFile`.

- `MU_firing_times_remodeled.txt`  
  OpenDiHu `firingTimesFile`.

- `opendihu_json_variables_snippet.py`  
  Snippet to paste or import into your OpenDiHu variables/config file.

## Run

```bash
python export_opendihu_mu_json.py --scenario aging_type2 --grid-n 37 --n-mu 25 --out export_37x37
```

Use `--grid-n 37` if you run the standard `left_biceps_brachii_37x37fibers.bin` example.

## In OpenDiHu variables.py

Set:

```python
fiber_distribution_file = "export_37x37/MU_fibre_distribution_remodeled.txt"
firing_times_file       = "export_37x37/MU_firing_times_remodeled.txt"
```

Then load the JSON bridge functions:

```python
import os
os.environ["MU_REMODELING_JSON"] = "export_37x37/opendihu_mu_fiber_config.json"
exec(open("export_37x37/opendihu_json_variables_snippet.py").read())
```

The functions `get_am`, `get_cm`, `get_specific_states_call_frequency`, etc. will use fiber-specific and MU-specific properties from JSON.
