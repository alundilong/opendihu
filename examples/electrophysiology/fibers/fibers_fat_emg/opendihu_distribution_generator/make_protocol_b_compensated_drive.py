
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Protocol B compensated-drive cases from Protocol A OpenDiHu case folders.

Protocol B keeps the same MU_fibre_distribution_37x37_20.txt and
MU_firing_times_always.txt files, but modifies surviving MU stimulation_frequency
in motor_units.json to mimic compensatory neural drive after MU loss.

Expected input:
    protocol_A_cases/healthy/
    protocol_A_cases/death_25/
    protocol_A_cases/death_50/
    protocol_A_cases/death_75/

Each stage must contain:
    MU_fibre_distribution_37x37_20.txt
    MU_firing_times_always.txt
    motor_units.json

Recommended:
    motor_unit_summary.csv  # used for expansion-aware compensation if available

Example:
    python make_protocol_b_compensated_drive.py --input-dir my_protocol_A_cases --out-dir my_protocol_B_cases
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGES = [
    ("healthy", 0.00),
    ("death_25", 0.25),
    ("death_50", 0.50),
    ("death_75", 0.75),
]

REQUIRED_FILES = [
    "MU_fibre_distribution_37x37_20.txt",
    "MU_firing_times_always.txt",
    "motor_units.json",
]

OPTIONAL_COPY_FILES = [
    "generation_metadata.json",
    "mu_distribution_and_biopsy_like.png",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_motor_units(path: Path) -> Dict:
    with path.open("r") as f:
        payload = json.load(f)
    if "motor_units" not in payload:
        raise ValueError(f"{path} does not contain key 'motor_units'.")
    return payload


def write_motor_units(payload: Dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_summary(stage_dir: Path) -> pd.DataFrame | None:
    p = stage_dir / "motor_unit_summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def get_expansion_ratio(summary: pd.DataFrame | None, mu_no: int) -> float:
    if summary is None:
        return 1.0
    if "mu_id_0_based" not in summary.columns or "expansion_ratio" not in summary.columns:
        return 1.0
    row = summary.loc[summary["mu_id_0_based"] == mu_no]
    if row.empty:
        return 1.0
    val = float(row.iloc[0]["expansion_ratio"])
    if not np.isfinite(val) or val <= 0:
        return 1.0
    return val


def compute_global_factor(active_mu_fraction: float, max_global_factor: float, global_gain: float) -> float:
    active_mu_fraction = max(float(active_mu_fraction), 1e-6)
    factor = active_mu_fraction ** (-global_gain)
    return float(min(max_global_factor, factor))


def update_patch_file(stage_dir: Path, stage_name: str) -> None:
    """
    Write a small OpenDiHu variables.py patch for this stage.

    I intentionally use only old-style .format strings here to avoid f-string
    syntax problems in older Python environments.
    """

    patch = """
# -------------------------------------------------------------------------
# Patch for OpenDiHu variables.py.
# Remove/comment the original motor_units loop and use this block.
# -------------------------------------------------------------------------

import json
import os

scenario_name = "{stage_name}_protocol_B"

# Keep these definitions if your original variables.py needs them.
Conductivity = 3.828      # [mS/cm] sigma, conductivity
Am = 500.0                # [cm^-1]
Cm = 0.58                 # [uF/cm^2]

generated_input_directory = os.environ.get(
    "OPENDIHU_GENERATED_MU_DIR",
    r"{stage_dir}"
)

firing_times_file = os.path.join(generated_input_directory, "MU_firing_times_always.txt")
fiber_distribution_file = os.path.join(generated_input_directory, "MU_fibre_distribution_37x37_20.txt")
motor_units_file = os.path.join(generated_input_directory, "motor_units.json")

with open(motor_units_file, "r") as f:
    motor_units_payload = json.load(f)

motor_units = motor_units_payload["motor_units"]
n_motor_units = len(motor_units)

# The OpenDiHu callback usually receives mu_no as 0-based.
# If your callback receives 1..20, set:
#   export OPENDIHU_CALLBACK_MU_INDEX_BASE=1
callback_mu_index_base = int(os.environ.get("OPENDIHU_CALLBACK_MU_INDEX_BASE", "0"))

# timing parameters
# -----------------
end_time = 100.0                    # [ms] end time of the simulation
stimulation_frequency = 100*1e-3    # [ms^-1] sampling frequency of stimuli in firing_times_file, in stimulations per ms, number before 1e-3 factor is in Hertz.
stimulation_frequency_jitter = 0    # [-] jitter in percent of the frequency, added and substracted to the stimulation_frequency after each stimulation
dt_0D = 2.5e-3                      # [ms] timestep width of ODEs (2e-3)
dt_1D = 6.25e-4                     # [ms] timestep width of diffusion (4e-3)
dt_splitting = 2.5e-3               # [ms] overall timestep width of strang splitting (4e-3)
dt_3D = 5e-1                        # [ms] time step width of coupling, when 3D should be performed, also sampling time of monopolar EMG
output_timestep_fibers = 10          # [ms] timestep for fiber output, 0.5
output_timestep_3D_emg = 10         # [ms] timestep for output big files of 3D EMG, 100
output_timestep_surface = 10        # [ms] timestep for output surface EMG, 0.5
output_timestep_electrodes = 2e8    # [ms] timestep for python callback, which is electrode measurement output, has to be >= dt_3D

# input files
input_directory = os.path.join(os.environ["OPENDIHU_HOME"], "examples/electrophysiology/input")
fiber_file              = input_directory+"/left_biceps_brachii_37x37fibers.bin"
fat_mesh_file           = fiber_file + "_fat.bin"

# stride for sampling the 3D elements from the fiber data
# a higher number leads to less 3D elements
sampling_stride_x = 2
sampling_stride_y = 2
sampling_stride_z = 40      # good values: divisors of 1480: 1480 = 1*1480 = 2*740 = 4*370 = 5*296 = 8*185 = 10*148 = 20*74 = 37*40

# HD-EMG electrode parameters
fiber_file_for_hdemg_surface = fat_mesh_file    # use the fat mesh for placing electrodes, this option is the file of the 2D mesh on which electrode positions are set
hdemg_electrode_faces = ["1+"]                  # which faces of this 2D mesh should be considered for placing the HD-EMG electrodes (list of faces, a face is one of "0-" (left), "0+" (right), "1-" (front), "1+" (back))

# xy-direction = across muscle, z-direction = along muscle
hdemg_electrode_offset_xy = 2.0           # [cm] offset from boundary of 2D mesh where the electrode array begins
hdemg_inter_electrode_distance_z = 0.4    # [cm] distance between electrodes ("IED") in z direction (direction along muscle)
hdemg_inter_electrode_distance_xy = 0.4   # [cm] distance between electrodes ("IED") in transverse direction
hdemg_n_electrodes_z = 32           # number of electrodes in z direction (direction along muscle)
hdemg_n_electrodes_xy = 12          # number of electrode across muscle

# other options
paraview_output = True
adios_output = False
exfile_output = False
python_output = False
disable_firing_output = True
enable_surface_emg = True          # Enables the surface emg output writer
fast_monodomain_solver_optimizations = True # enable the optimizations in the fast multidomain solver
optimization_type = "vc"            # the optimization_type used in the cellml adapter, "vc" uses explicit vectorization

def _mu(mu_no):
  return motor_units[(int(mu_no) - callback_mu_index_base) % len(motor_units)]

def get_am(fiber_no, mu_no):
  r = _mu(mu_no)["radius"]*1e-4
  return 2./r

def get_cm(fiber_no, mu_no):
  return _mu(mu_no)["cm"]

def get_conductivity(fiber_no, mu_no):
  return Conductivity

def get_specific_states_call_frequency(fiber_no, mu_no):
  return _mu(mu_no)["stimulation_frequency"]*1e-3

def get_specific_states_frequency_jitter(fiber_no, mu_no):
  return _mu(mu_no)["jitter"]

def get_specific_states_call_enable_begin(fiber_no, mu_no):
  return _mu(mu_no)["activation_start_time"]*1e3
""".format(stage_name=stage_name, stage_dir=str(stage_dir.resolve()))
    (stage_dir / f"opendihu_variables_patch_protocol_B_{stage_name}.py").write_text(patch)

def create_protocol_b_stage(
    in_stage_dir: Path,
    out_stage_dir: Path,
    stage_name: str,
    kill_fraction: float,
    max_global_factor: float,
    global_gain: float,
    expansion_alpha: float,
    max_frequency: float,
    min_frequency: float,
) -> List[Dict]:
    ensure_dir(out_stage_dir)

    for fn in REQUIRED_FILES:
        src = in_stage_dir / fn
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")

    shutil.copy2(in_stage_dir / "MU_fibre_distribution_37x37_20.txt", out_stage_dir / "MU_fibre_distribution_37x37_20.txt")
    shutil.copy2(in_stage_dir / "MU_firing_times_always.txt", out_stage_dir / "MU_firing_times_always.txt")

    for fn in OPTIONAL_COPY_FILES:
        src = in_stage_dir / fn
        if src.exists():
            shutil.copy2(src, out_stage_dir / fn)

    payload = load_motor_units(in_stage_dir / "motor_units.json")
    summary = read_summary(in_stage_dir)

    motor_units = payload["motor_units"]
    active = np.array(
        [bool(mu.get("active", not mu.get("killed_or_silent", False))) for mu in motor_units],
        dtype=bool,
    )
    active_fraction = float(np.mean(active))
    global_factor = compute_global_factor(active_fraction, max_global_factor, global_gain)

    rows = []
    for idx, mu in enumerate(motor_units):
        mu_no = int(mu.get("mu_no_0_based", idx))
        old_freq = float(mu.get("stimulation_frequency", 0.0))
        killed = bool(mu.get("killed_or_silent", False)) or (not bool(mu.get("active", True)))

        expansion_ratio = get_expansion_ratio(summary, mu_no)

        if killed or old_freq <= 0:
            local_factor = 0.0
            new_freq = 0.0
            mu["active"] = False
            mu["killed_or_silent"] = True
            mu["activation_start_time"] = 1e9
        else:
            local_factor = float(expansion_ratio ** (-expansion_alpha))
            raw_new_freq = old_freq * global_factor * local_factor
            new_freq = float(np.clip(raw_new_freq, min_frequency, max_frequency))
            mu["active"] = True
            mu["killed_or_silent"] = False

        mu["baseline_stimulation_frequency"] = old_freq
        mu["protocol_B_global_factor"] = float(global_factor)
        mu["protocol_B_local_factor"] = float(local_factor)
        mu["protocol_B_expansion_ratio"] = float(expansion_ratio)
        mu["protocol_B_frequency_scale"] = float(0.0 if old_freq == 0 else new_freq / old_freq)
        mu["stimulation_frequency"] = float(new_freq)

        rows.append({
            "stage": stage_name,
            "kill_fraction": kill_fraction,
            "mu_no_0_based": mu_no,
            "active": bool(mu["active"]),
            "killed_or_silent": bool(mu["killed_or_silent"]),
            "type_id": mu.get("type_id", None),
            "type_name": mu.get("type_name", ""),
            "expansion_ratio": float(expansion_ratio),
            "baseline_frequency_hz": old_freq,
            "global_factor": float(global_factor),
            "local_factor": float(local_factor),
            "new_frequency_hz": float(new_freq),
            "frequency_scale": float(0.0 if old_freq == 0 else new_freq / old_freq),
        })

    payload["protocol"] = "Protocol B: compensated neural drive"
    payload["compensation_rule"] = {
        "global_factor": "min(max_global_factor, active_mu_fraction^(-global_gain))",
        "local_factor": "expansion_ratio^(-expansion_alpha)",
        "new_frequency": "clip(baseline_frequency * global_factor * local_factor, min_frequency, max_frequency)",
        "active_mu_fraction": active_fraction,
        "max_global_factor": max_global_factor,
        "global_gain": global_gain,
        "expansion_alpha": expansion_alpha,
        "max_frequency": max_frequency,
        "min_frequency": min_frequency,
    }

    write_motor_units(payload, out_stage_dir / "motor_units.json")
    pd.DataFrame(rows).to_csv(out_stage_dir / "protocol_B_motor_unit_summary.csv", index=False)

    metadata = {
        "stage": stage_name,
        "kill_fraction": kill_fraction,
        "input_stage_dir": str(in_stage_dir.resolve()),
        "active_mu_fraction": active_fraction,
        "global_factor": global_factor,
        "max_global_factor": max_global_factor,
        "global_gain": global_gain,
        "expansion_alpha": expansion_alpha,
        "max_frequency": max_frequency,
        "min_frequency": min_frequency,
        "note": "MU distribution and firing gate are copied from Protocol A. Only motor_units.json stimulation_frequency is modified for surviving MUs.",
    }
    with (out_stage_dir / "protocol_B_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    update_patch_file(out_stage_dir, stage_name)
    return rows


def plot_frequency_summary(all_rows: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    for stage, _ in STAGES:
        sub = all_rows[all_rows["stage"] == stage]
        ax.plot(sub["mu_no_0_based"], sub["new_frequency_hz"], marker="o", linewidth=1.2, label=stage)
    ax.set_xlabel("MU index")
    ax.set_ylabel("Protocol B stimulation frequency [Hz]")
    ax.set_title("Compensated firing rates")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    stages = []
    means = []
    stds = []
    for stage, _ in STAGES:
        sub = all_rows[(all_rows["stage"] == stage) & (all_rows["active"])]
        stages.append(stage)
        means.append(float(sub["new_frequency_hz"].mean()) if len(sub) else 0.0)
        stds.append(float(sub["new_frequency_hz"].std()) if len(sub) else 0.0)
    xpos = np.arange(len(stages))
    ax.bar(xpos, means, yerr=stds, capsize=4)
    ax.set_xticks(xpos, stages, rotation=20)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Mean ± SD among active MUs")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(out_dir / "protocol_B_frequency_summary.png", dpi=220)
    fig.savefig(out_dir / "protocol_B_frequency_summary.pdf", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for stage, _ in STAGES:
        sub = all_rows[(all_rows["stage"] == stage) & (all_rows["active"])]
        ax.scatter(sub["expansion_ratio"], sub["frequency_scale"], s=35, alpha=0.75, label=stage)
    ax.set_xlabel("MU expansion ratio")
    ax.set_ylabel("Frequency scale vs Protocol A")
    ax.set_title("Expansion-aware compensation rule")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "protocol_B_frequency_scale_vs_expansion.png", dpi=220)
    fig.savefig(out_dir / "protocol_B_frequency_scale_vs_expansion.pdf", dpi=220)
    plt.close(fig)


def write_readme(out_dir: Path, input_dir: Path) -> None:
    text = (
        "Protocol B compensated-drive case set.\n\n"
        f"Input Protocol A cases:\n  {input_dir.resolve()}\n\n"
        f"Output Protocol B cases:\n  {out_dir.resolve()}\n\n"
        "Each stage folder contains:\n"
        "  MU_fibre_distribution_37x37_20.txt\n"
        "  MU_firing_times_always.txt\n"
        "  motor_units.json\n"
        "  protocol_B_motor_unit_summary.csv\n"
        "  protocol_B_metadata.json\n"
        "  opendihu_variables_patch.py\n\n"
        "What changed from Protocol A?\n"
        "  The MU distribution file is unchanged.\n"
        "  The firing-times gate file is unchanged.\n"
        "  The surviving MUs have modified stimulation_frequency in motor_units.json.\n"
        "  Killed/silent MUs remain inactive with stimulation_frequency = 0.\n\n"
        "Use with OpenDiHu, for example:\n"
        f"  export OPENDIHU_GENERATED_MU_DIR={out_dir.resolve() / 'death_50'}\n"
    )
    (out_dir / "README_protocol_B.txt").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Protocol B compensated-drive OpenDiHu cases from Protocol A case folders.")
    parser.add_argument("--input-dir", required=True, help="Protocol A case folder containing healthy/death_25/death_50/death_75.")
    parser.add_argument("--out-dir", default="protocol_B_compensated_drive", help="Output folder for Protocol B cases.")
    parser.add_argument("--max-global-factor", type=float, default=1.8, help="Cap for global compensation factor.")
    parser.add_argument("--global-gain", type=float, default=1.0, help="Exponent in active_mu_fraction^(-global_gain).")
    parser.add_argument("--expansion-alpha", type=float, default=0.5, help="Local factor exponent: expansion_ratio^(-alpha). Use 0 to disable.")
    parser.add_argument("--max-frequency", type=float, default=35.0, help="Maximum allowed stimulation frequency [Hz].")
    parser.add_argument("--min-frequency", type=float, default=1.0, help="Minimum frequency for active surviving MUs [Hz].")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    all_rows: List[Dict] = []
    for stage, kill_fraction in STAGES:
        rows = create_protocol_b_stage(
            in_stage_dir=input_dir / stage,
            out_stage_dir=out_dir / stage,
            stage_name=stage,
            kill_fraction=kill_fraction,
            max_global_factor=args.max_global_factor,
            global_gain=args.global_gain,
            expansion_alpha=args.expansion_alpha,
            max_frequency=args.max_frequency,
            min_frequency=args.min_frequency,
        )
        all_rows.extend(rows)
        print(f"Created Protocol B stage: {out_dir / stage}")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "protocol_B_compensation_overview.csv", index=False)
    plot_frequency_summary(all_df, out_dir)
    write_readme(out_dir, input_dir)

    print(f"\nCreated Protocol B compensated-drive cases in: {out_dir}")
    print(f"Summary CSV: {out_dir / 'protocol_B_compensation_overview.csv'}")
    print(f"Figure: {out_dir / 'protocol_B_frequency_summary.png'}")


if __name__ == "__main__":
    main()
