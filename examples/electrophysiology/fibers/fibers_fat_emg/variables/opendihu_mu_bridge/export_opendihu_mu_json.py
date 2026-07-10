#!/usr/bin/env python3
"""
Export a remodeled motor-unit/fiber state to OpenDiHu-friendly files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Any

import numpy as np


TYPE_TARGETS = {
    0: ("type_I_slow", 0.00, 0.58),
    1: ("type_IIa_fast_intermediate", 0.60, 0.80),
    2: ("type_IIx_fast", 1.00, 1.00),
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def class_from_phenotype(x: float) -> int:
    if x < 0.30:
        return 0
    if x < 0.80:
        return 1
    return 2


def radius_from_csa(base_radius_um: float, csa_norm: float) -> float:
    # CSA ~ pi*r^2, so r scales with sqrt(CSA).
    return float(base_radius_um * math.sqrt(max(csa_norm, 0.05)))


def make_motor_unit_table(state, n_mus_total: int, dummy_mu_id: int) -> list[dict[str, Any]]:
    motor_units = []
    for mu_no in range(n_mus_total):
        if mu_no == dummy_mu_id:
            motor_units.append({
                "mu_id": mu_no,
                "alive": False,
                "name": "dummy_denervated_never_fires",
                "mn_type": "denervated",
                "phenotype_target": -1.0,
                "radius_um": 35.0,
                "cm": 0.58,
                "activation_start_time_s": 1e9,
                "stimulation_frequency_hz": 0.0,
                "jitter": [0.0 for _ in range(100)],
            })
            continue

        mu_type = int(state.mu_type[mu_no])
        name, target, cm = TYPE_TARGETS[mu_type]
        alive = bool(state.mu_alive[mu_no])

        if mu_type == 0:
            radius_um = 42.0
            freq = 18.0
            start = 0.0
        elif mu_type == 1:
            radius_um = 48.0
            freq = 14.0
            start = 3.0
        else:
            radius_um = 55.0
            freq = 10.0
            start = 8.0

        if not alive:
            freq = 0.0
            start = 1e9

        motor_units.append({
            "mu_id": mu_no,
            "alive": alive,
            "name": name,
            "mn_type": int(mu_type),
            "phenotype_target": target,
            "radius_um": radius_um,
            "cm": cm,
            "activation_start_time_s": start,
            "stimulation_frequency_hz": freq,
            "jitter": [0.0 for _ in range(100)],
        })
    return motor_units


def export_state_to_json_and_opendihu_files(state, out_dir: str, dummy_mu_id: int | None = None) -> dict[str, str]:
    ensure_dir(out_dir)
    n_y, n_x = state.mu_id.shape
    n_fibers = n_x * n_y
    n_real_mus = len(state.mu_type)
    if dummy_mu_id is None:
        dummy_mu_id = n_real_mus
    n_mus_total = dummy_mu_id + 1

    motor_units = make_motor_unit_table(state, n_mus_total, dummy_mu_id)
    fibers = []
    distribution = []

    for i in range(n_y):
        for j in range(n_x):
            fiber_no = i * n_x + j      # row-major indexing
            original_mu = int(state.mu_id[i, j])
            denervated = original_mu < 0
            mu_id = dummy_mu_id if denervated else original_mu
            phenotype = float(state.fiber_type[i, j])
            subtype = class_from_phenotype(phenotype)
            csa_norm = float(state.csa[i, j])
            base_radius = motor_units[mu_id]["radius_um"] if mu_id != dummy_mu_id else 35.0
            radius_um = radius_from_csa(base_radius, csa_norm)
            cm = float(motor_units[mu_id]["cm"])
            am_cm_inv = 2.0 / (radius_um * 1e-4)  # r [um] -> [cm]

            distribution.append(mu_id)
            fibers.append({
                "fiber_no": fiber_no,
                "grid_i_y": i,
                "grid_j_x": j,
                "mu_id": mu_id,
                "original_mu_id": original_mu,
                "denervated": bool(denervated),
                "phenotype_value": phenotype,
                "subtype_id": subtype,
                "subtype_name": TYPE_TARGETS[subtype][0] if not denervated else "denervated",
                "csa_norm": csa_norm,
                "radius_um": radius_um,
                "cm": cm,
                "am_cm_inv": am_cm_inv,
            })

    cfg = {
        "metadata": {
            "description": "Remodeled MU/fiber state exported for OpenDiHu.",
            "fiber_order": "row-major over 2D fiber grid: fiber_no = grid_i_y*n_fibers_x + grid_j_x",
            "n_fibers_x": n_x,
            "n_fibers_y": n_y,
            "n_fibers": n_fibers,
            "n_real_motor_units": n_real_mus,
            "dummy_denervated_mu_id": dummy_mu_id,
            "notes": [
                "Use MU_fibre_distribution_remodeled.txt as OpenDiHu fiberDistributionFile.",
                "Denervated fibers are assigned to dummy_denervated_mu_id.",
                "Set the dummy MU firing column to zero in the firingTimesFile.",
                "Use opendihu_json_variables_snippet.py to define get_am/get_cm/get_specific_states_* callbacks."
            ],
        },
        "motor_units": motor_units,
        "fibers": fibers,
    }

    json_file = os.path.join(out_dir, "opendihu_mu_fiber_config.json")
    with open(json_file, "w") as f:
        json.dump(cfg, f, indent=2)

    distribution_file = os.path.join(out_dir, "MU_fibre_distribution_remodeled.txt")
    with open(distribution_file, "w") as f:
        f.write(" ".join(str(x) for x in distribution) + "\n")

    firing_file = os.path.join(out_dir, "MU_firing_times_remodeled.txt")
    n_rows = 200
    with open(firing_file, "w") as f:
        for _ in range(n_rows):
            row = ["1" if mu["alive"] and mu["mu_id"] != dummy_mu_id else "0" for mu in motor_units]
            f.write(" ".join(row) + "\n")

    snippet_file = os.path.join(out_dir, "opendihu_json_variables_snippet.py")
    with open(snippet_file, "w") as f:
        f.write(SNIPPET_TEXT)

    return {
        "json_file": json_file,
        "distribution_file": distribution_file,
        "firing_file": firing_file,
        "snippet_file": snippet_file,
    }


SNIPPET_TEXT = '\n# -------------------------------------------------------------------------\n# OpenDiHu variables.py-style JSON bridge for remodeled MU/fiber states\n# -------------------------------------------------------------------------\n# Put this near the top of your variables.py / ramp_emg.py file, after imports.\n# Then replace get_am/get_cm/get_specific_states_* by the functions below.\n\nimport json\nimport os\n\nremodeling_json_file = os.environ.get(\n    "MU_REMODELING_JSON",\n    "opendihu_mu_fiber_config.json"\n)\n\nwith open(remodeling_json_file, "r") as f:\n    remodeling_cfg = json.load(f)\n\nmotor_units = remodeling_cfg["motor_units"]\nfiber_records = remodeling_cfg["fibers"]\nfibers_by_no = {int(f["fiber_no"]): f for f in fiber_records}\ndummy_denervated_mu_id = int(remodeling_cfg["metadata"]["dummy_denervated_mu_id"])\n\ndef _mu(mu_no):\n    return motor_units[int(mu_no) % len(motor_units)]\n\ndef _fiber(fiber_no):\n    return fibers_by_no[int(fiber_no)]\n\ndef get_am(fiber_no, mu_no):\n    # Prefer the fiber-specific radius because it can include atrophy/hypertrophy.\n    f = _fiber(fiber_no)\n    r_cm = float(f.get("radius_um", _mu(mu_no)["radius_um"])) * 1e-4\n    return 2.0 / r_cm\n\ndef get_cm(fiber_no, mu_no):\n    # Prefer fiber-specific Cm if stored; otherwise use MU-level Cm.\n    f = _fiber(fiber_no)\n    return float(f.get("cm", _mu(mu_no)["cm"]))\n\ndef get_conductivity(fiber_no, mu_no):\n    return Conductivity\n\ndef get_specific_states_call_frequency(fiber_no, mu_no):\n    # Return [ms^-1]. Hz * 1e-3 = stimulations per ms.\n    if int(mu_no) == dummy_denervated_mu_id:\n        return 0.0\n    return float(_mu(mu_no)["stimulation_frequency_hz"]) * 1e-3\n\ndef get_specific_states_frequency_jitter(fiber_no, mu_no):\n    if int(mu_no) == dummy_denervated_mu_id:\n        return [0.0 for _ in range(100)]\n    return _mu(mu_no).get("jitter", [0.0 for _ in range(100)])\n\ndef get_specific_states_call_enable_begin(fiber_no, mu_no):\n    # Return [ms].\n    if int(mu_no) == dummy_denervated_mu_id:\n        return 1e12\n    return float(_mu(mu_no)["activation_start_time_s"]) * 1e3\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="opendihu_mu_export")
    parser.add_argument("--grid-n", type=int, default=37, help="Use 37 to match 37x37 fiber file.")
    parser.add_argument("--n-mu", type=int, default=25, help="Must be a square number for current model.")
    parser.add_argument("--scenario", default="aging_type2",
                        choices=["classic_like", "aging_type2", "exercise_rescued", "als_like_failed_rescue"])
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    from mu_digital_pathology_v2 import RemodelParams, scenario_params, run_forward

    base = RemodelParams(grid_n=args.grid_n, n_mu=args.n_mu, seed=args.seed)
    params = scenario_params(args.scenario, base)
    final_state, snapshots, history = run_forward(params)

    paths = export_state_to_json_and_opendihu_files(final_state, args.out)
    print("Exported OpenDiHu bridge files:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
