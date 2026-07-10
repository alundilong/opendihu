
# -------------------------------------------------------------------------
# OpenDiHu variables.py-style JSON bridge for remodeled MU/fiber states
# -------------------------------------------------------------------------
# Put this near the top of your variables.py / ramp_emg.py file, after imports.
# Then replace get_am/get_cm/get_specific_states_* by the functions below.

import json
import os

remodeling_json_file = os.environ.get(
    "MU_REMODELING_JSON",
    "opendihu_mu_fiber_config.json"
)

with open(remodeling_json_file, "r") as f:
    remodeling_cfg = json.load(f)

motor_units = remodeling_cfg["motor_units"]
fiber_records = remodeling_cfg["fibers"]
fibers_by_no = {int(f["fiber_no"]): f for f in fiber_records}
dummy_denervated_mu_id = int(remodeling_cfg["metadata"]["dummy_denervated_mu_id"])

def _mu(mu_no):
    return motor_units[int(mu_no) % len(motor_units)]

def _fiber(fiber_no):
    return fibers_by_no[int(fiber_no)]

def get_am(fiber_no, mu_no):
    # Prefer the fiber-specific radius because it can include atrophy/hypertrophy.
    f = _fiber(fiber_no)
    r_cm = float(f.get("radius_um", _mu(mu_no)["radius_um"])) * 1e-4
    return 2.0 / r_cm

def get_cm(fiber_no, mu_no):
    # Prefer fiber-specific Cm if stored; otherwise use MU-level Cm.
    f = _fiber(fiber_no)
    return float(f.get("cm", _mu(mu_no)["cm"]))

def get_conductivity(fiber_no, mu_no):
    return Conductivity

def get_specific_states_call_frequency(fiber_no, mu_no):
    # Return [ms^-1]. Hz * 1e-3 = stimulations per ms.
    if int(mu_no) == dummy_denervated_mu_id:
        return 0.0
    return float(_mu(mu_no)["stimulation_frequency_hz"]) * 1e-3

def get_specific_states_frequency_jitter(fiber_no, mu_no):
    if int(mu_no) == dummy_denervated_mu_id:
        return [0.0 for _ in range(100)]
    return _mu(mu_no).get("jitter", [0.0 for _ in range(100)])

def get_specific_states_call_enable_begin(fiber_no, mu_no):
    # Return [ms].
    if int(mu_no) == dummy_denervated_mu_id:
        return 1e12
    return float(_mu(mu_no)["activation_start_time_s"]) * 1e3
