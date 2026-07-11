#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Scenario:
    name: str
    p_death_slow: float
    p_death_2a: float
    p_death_2x: float
    p_reinn_slow: float
    p_reinn_2a: float
    p_reinn_2x: float
    sprout_radius: float
    sprout_sigma: float
    rescue_gain: float

SCENARIOS: Dict[str, Scenario] = {
    "healthy": Scenario("healthy", 0.0, 0.0, 0.0, 0.9, 0.8, 0.7, 5.0, 2.5, 0.45),
    "classic_like": Scenario("classic_like", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 999.0),
    "aging_type2": Scenario("aging_type2", 0.25, 0.60, 0.90, 0.80, 0.55, 0.30, 5.5, 2.6, 0.35),
    "exercise_rescued": Scenario("exercise_rescued", 0.30, 0.50, 0.70, 0.92, 0.78, 0.55, 7.0, 3.2, 0.55),
    "als_like_failed_rescue": Scenario("als_like_failed_rescue", 0.60, 0.70, 0.80, 0.32, 0.24, 0.18, 3.5, 1.8, 0.25),
}
MU_TYPE_NAMES = {0: "type_I_slow", 1: "type_IIa_fast_intermediate", 2: "type_IIx_fast"}
MU_CM = {0: 0.58, 1: 0.80, 2: 1.00}

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def exp_range_value(mu_no: int, n_motor_units: int, min_value: float, max_value: float, increasing: bool = True) -> float:
    if n_motor_units <= 1: return float(min_value)
    c2 = (max_value - min_value) / (1.02 ** (n_motor_units - 1) - 1)
    c1 = min_value - c2
    idx = mu_no if increasing else (n_motor_units - 1 - mu_no)
    return float(c1 + c2 * 1.02 ** idx)

def make_motor_units_json(mu_type, killed, out_file, index_base, seed, original_opendihu_style=True):
    n_motor_units = len(mu_type)
    rng = random.Random(seed)
    motor_units = []
    for mu_no in range(n_motor_units):
        type_id = int(mu_type[mu_no])
        radius = exp_range_value(mu_no, n_motor_units, 40.0, 55.0, increasing=True)
        stimulation_frequency = exp_range_value(mu_no, n_motor_units, 7.0, 24.0, increasing=False)
        activation_start_time = (0.0 if mu_no < 10 else 10.0) if original_opendihu_style else (0.0 if type_id == 0 else (5.0 if type_id == 1 else 10.0))
        active = not bool(killed[mu_no])
        if not active:
            activation_start_time = 1e9
            stimulation_frequency = 0.0
        motor_units.append({
            "mu_no_0_based": int(mu_no),
            "mu_id_in_distribution_file": int(mu_no + index_base),
            "active": bool(active),
            "killed_or_silent": bool(killed[mu_no]),
            "type_id": int(type_id),
            "type_name": MU_TYPE_NAMES[type_id],
            "radius": float(radius),
            "cm": float(MU_CM[type_id]),
            "activation_start_time": float(activation_start_time),
            "stimulation_frequency": float(stimulation_frequency),
            "jitter": [0.1 * rng.uniform(-1, 1) for _ in range(100)],
        })
    payload = {
        "format": "opendihu_motor_units_json_v1",
        "description": "motor_units list to be loaded by OpenDiHu variables.py",
        "index_base": int(index_base),
        "n_motor_units": int(n_motor_units),
        "cm_convention": {"type_I_slow": MU_CM[0], "type_IIa_fast_intermediate": MU_CM[1], "type_IIx_fast": MU_CM[2]},
        "motor_units": motor_units,
    }
    with open(out_file, "w") as f: json.dump(payload, f, indent=2)
    return motor_units

def make_mu_centers(n_x, n_y, n_mu, rng):
    n_cols = int(math.ceil(math.sqrt(n_mu * n_x / n_y)))
    n_rows = int(math.ceil(n_mu / n_cols))
    xs = np.linspace(0.5 * n_x / n_cols, n_x - 0.5 * n_x / n_cols, n_cols)
    ys = np.linspace(0.5 * n_y / n_rows, n_y - 0.5 * n_y / n_rows, n_rows)
    centers = []
    for y in ys:
        for x in xs:
            if len(centers) < n_mu: centers.append((x, y))
    centers = np.asarray(centers, dtype=float)
    centers += rng.normal(0.0, 0.35, size=centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], 0, n_x - 1); centers[:, 1] = np.clip(centers[:, 1], 0, n_y - 1)
    return centers

def make_mu_types(n_mu, rng, type1_fraction=0.50, type2a_fraction=0.35, randomize=True):
    """
    Assign MU physiological types.

    Important fix: the old version assigned MU types by sorted MU index. Because
    MU centers are placed in an ordered grid, this created an artificial spatial
    gradient in the healthy biopsy image. Here MU types are randomly permuted, so
    type I/type II centers are spatially mixed.

    type_id: 0 = type I, 1 = type IIa, 2 = type IIx.
    For red/white biopsy rendering, both type IIa and IIx are shown as type II.
    """
    type1_fraction = float(np.clip(type1_fraction, 0.0, 1.0))
    type2a_fraction = float(np.clip(type2a_fraction, 0.0, 1.0 - type1_fraction))

    n_type1 = int(round(type1_fraction * n_mu))
    n_type2a = int(round(type2a_fraction * n_mu))
    n_type2x = max(0, n_mu - n_type1 - n_type2a)

    mu_type = np.array([0] * n_type1 + [1] * n_type2a + [2] * n_type2x, dtype=int)
    if len(mu_type) < n_mu:
        mu_type = np.pad(mu_type, (0, n_mu - len(mu_type)), constant_values=2)
    mu_type = mu_type[:n_mu]

    if randomize:
        rng.shuffle(mu_type)
    return mu_type

def desired_mu_sizes(n_mu, total_fibers, exponent=1.055):
    raw = exponent ** np.arange(n_mu)
    return np.maximum(raw / raw.sum() * total_fibers, 1.0)

def initialize_distribution(n_x, n_y, n_mu, rng, type1_fraction=0.50, type2a_fraction=0.35,
                            territory_sigma_factor=0.40, repulsion_factor=0.25):
    """
    Initialize the healthy fiber-to-MU map.

    Important fixes for a normal-looking healthy biopsy:
    1) MU types are randomly mixed in space rather than sorted by MU index.
    2) MU territories are made broad/overlapping, not compact blocks.
    3) A small same-MU neighbor repulsion discourages adjacent fibers from being
       assigned to the same MU, producing a more random type I/type II mosaic.
    """
    centers = make_mu_centers(n_x, n_y, n_mu, rng)
    mu_type = make_mu_types(n_mu, rng, type1_fraction=type1_fraction, type2a_fraction=type2a_fraction, randomize=True)
    target_sizes = desired_mu_sizes(n_mu, n_x * n_y)
    counts = np.zeros(n_mu, dtype=float)
    mu_map = -np.ones((n_y, n_x), dtype=int)
    coords = [(i, j) for i in range(n_y) for j in range(n_x)]
    rng.shuffle(coords)

    sigma = float(territory_sigma_factor) * max(n_x, n_y)
    sigma = max(sigma, 1e-6)

    for i, j in coords:
        d2 = (centers[:, 0] - j) ** 2 + (centers[:, 1] - i) ** 2
        spatial = np.exp(-d2 / (2.0 * sigma ** 2))
        size_factor = (target_sizes / np.maximum(counts + 1.0, 1.0)) ** 0.75
        p = spatial * size_factor

        # Repel assignment to an MU already present in immediate neighbors.
        # This follows the idea of avoiding excessive adjacent fibers from the
        # same MU in a normal cross-section. It helps the type I/type II map look
        # random instead of block-like.
        for ii, jj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ii < n_y and 0 <= jj < n_x and mu_map[ii, jj] >= 0:
                p[int(mu_map[ii, jj])] *= float(repulsion_factor)

        if p.sum() <= 0:
            k = int(rng.integers(0, n_mu))
        else:
            k = int(rng.choice(n_mu, p=p / p.sum()))
        mu_map[i, j] = k
        counts[k] += 1
    return mu_map, centers, mu_type

def select_killed_mus(mu_map, mu_type, scenario, kill_fraction, rng):
    n_mu = len(mu_type); n_kill = int(round(np.clip(kill_fraction, 0, 0.95) * n_mu))
    killed = np.zeros(n_mu, dtype=bool)
    if n_kill <= 0: return killed
    counts = np.bincount(mu_map.ravel(), minlength=n_mu).astype(float)
    type_weight = np.zeros(n_mu, dtype=float)
    type_weight[mu_type == 0] = scenario.p_death_slow; type_weight[mu_type == 1] = scenario.p_death_2a; type_weight[mu_type == 2] = scenario.p_death_2x
    weights = type_weight * np.maximum(counts, 1.0) ** 0.25
    if weights.sum() <= 0: weights[:] = 1.0
    killed[rng.choice(n_mu, size=n_kill, replace=False, p=weights / weights.sum())] = True
    return killed

def reinnervation_competence(mu_type_id, scenario):
    return scenario.p_reinn_slow if mu_type_id == 0 else (scenario.p_reinn_2a if mu_type_id == 1 else scenario.p_reinn_2x)

def remodel_distribution(mu_map0, mu_type, killed, centers, scenario, rng):
    mu_map = mu_map0.copy()
    surviving = np.where(~killed)[0]
    if len(surviving) == 0: return mu_map
    den_coords = np.argwhere(killed[mu_map0]); rng.shuffle(den_coords)
    counts = np.bincount(mu_map.ravel(), minlength=len(mu_type)).astype(float)
    original_counts = counts.copy()
    for i, j in den_coords:
        original_mu = int(mu_map0[i, j])
        d = np.sqrt((centers[surviving, 0] - j) ** 2 + (centers[surviving, 1] - i) ** 2)
        nearby = d <= scenario.sprout_radius
        if not np.any(nearby): continue
        cand = surviving[nearby]; dist = d[nearby]
        spatial = np.exp(-(dist ** 2) / (2.0 * scenario.sprout_sigma ** 2))
        competence = np.array([reinnervation_competence(int(mu_type[k]), scenario) for k in cand])
        expansion = counts[cand] / np.maximum(original_counts[cand], 1.0)
        weights = spatial * competence * np.maximum(0.05, 1.0 / expansion)
        if weights.sum() <= 0: continue
        rescue_prob = 1.0 if scenario.name == "classic_like" else (1.0 - math.exp(-scenario.rescue_gain * weights.sum()))
        if rng.random() < rescue_prob:
            new_mu = int(rng.choice(cand, p=weights / weights.sum()))
            mu_map[i, j] = new_mu; counts[new_mu] += 1; counts[original_mu] = max(0, counts[original_mu] - 1)
    return mu_map

def compute_stage_fields(mu_map0, mu_map, mu_type, killed):
    n_y, n_x = mu_map.shape
    subtype_map = np.zeros((n_y, n_x), dtype=int); csa = np.ones((n_y, n_x), dtype=float)
    denervated = killed[mu_map]
    for i in range(n_y):
        for j in range(n_x):
            owner = int(mu_map[i, j]); owner_type = int(mu_type[owner]); subtype_map[i, j] = owner_type
            base = 0.95 if owner_type == 0 else (1.00 if owner_type == 1 else 1.08)
            orig_type = int(mu_type[int(mu_map0[i, j])])
            if owner != int(mu_map0[i, j]) and not denervated[i, j]:
                original_base = 0.95 if orig_type == 0 else (1.00 if orig_type == 1 else 1.08)
                base = 0.5 * base + 0.5 * original_base
            expanded = np.count_nonzero(mu_map == owner) / max(np.count_nonzero(mu_map0 == owner), 1)
            csa_val = base * (1.0 + 0.13 * max(0.0, expanded - 1.0))
            if denervated[i, j]: csa_val *= 0.65 if owner_type == 0 else 0.55
            csa[i, j] = csa_val
    return subtype_map, csa, denervated

def write_distribution(mu_map, out_file, index_base):
    vals = mu_map.ravel(order="C") + index_base
    with open(out_file, "w") as f: f.write(" ".join(str(int(v)) for v in vals) + "\n")

def write_firing_times(killed, out_file, n_rows, active_value=1):
    row = [str(active_value if not dead else 0) for dead in killed]
    with open(out_file, "w") as f:
        for _ in range(n_rows): f.write(" ".join(row) + "\n")

def write_summary(mu_map0, mu_map, mu_type, killed, motor_units, out_file):
    n_mu = len(mu_type); initial_counts = np.bincount(mu_map0.ravel(), minlength=n_mu); final_counts = np.bincount(mu_map.ravel(), minlength=n_mu)
    rows = []
    for k in range(n_mu):
        rows.append({
            "mu_id_0_based": k, "mu_id_1_based": k + 1,
            "mu_type_id": int(mu_type[k]), "mu_type_name": MU_TYPE_NAMES[int(mu_type[k])],
            "killed_or_silent": bool(killed[k]), "initial_fiber_count": int(initial_counts[k]),
            "final_assigned_fiber_count": int(final_counts[k]), "expansion_ratio": float(final_counts[k] / max(initial_counts[k], 1)),
            "radius_um": float(motor_units[k]["radius"]), "cm": float(motor_units[k]["cm"]),
            "activation_start_time_s": float(motor_units[k]["activation_start_time"]),
            "stimulation_frequency_hz": float(motor_units[k]["stimulation_frequency"]),
        })
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def render_biopsy_like(subtype_map, csa, denervated, seed=0, scale=8):
    rng = np.random.default_rng(seed)
    n_y, n_x = subtype_map.shape; H, W = n_y * scale, n_x * scale
    gy, gx = np.mgrid[0:n_y, 0:n_x]
    cx = np.clip(gx.astype(float) + 0.5 + rng.uniform(-0.28, 0.28, size=(n_y, n_x)), gx + 0.10, gx + 0.90)
    cy = np.clip(gy.astype(float) + 0.5 + rng.uniform(-0.28, 0.28, size=(n_y, n_x)), gy + 0.10, gy + 0.90)
    seed_xy = np.column_stack([cx.ravel(), cy.ravel()])
    weights = np.sqrt(np.clip(csa.ravel(), 0.08, 3.0)); weights[denervated.ravel()] *= 0.85
    labels = np.empty((H, W), dtype=np.int32)
    local_y = (np.arange(scale) + 0.5) / scale; local_x = (np.arange(scale) + 0.5) / scale
    lyy, lxx = np.meshgrid(local_y, local_x, indexing="ij")
    for i in range(n_y):
        for j in range(n_x):
            pi0, pi1 = i * scale, (i + 1) * scale; pj0, pj1 = j * scale, (j + 1) * scale
            ic0, ic1 = max(0, i - 2), min(n_y, i + 3); jc0, jc1 = max(0, j - 2), min(n_x, j + 3)
            cand_ii, cand_jj = np.mgrid[ic0:ic1, jc0:jc1]; cand_ii = cand_ii.ravel(); cand_jj = cand_jj.ravel()
            cand_idx = cand_ii * n_x + cand_jj
            px = j + lxx[..., None]; py = i + lyy[..., None]
            sx = seed_xy[cand_idx, 0][None, None, :]; sy = seed_xy[cand_idx, 1][None, None, :]; ww = weights[cand_idx][None, None, :]
            labels[pi0:pi1, pj0:pj1] = cand_idx[np.argmin(((px - sx) ** 2 + (py - sy) ** 2) / (ww ** 2 + 1e-8), axis=2)]
    colors = np.zeros((n_y * n_x, 3), dtype=float)
    for idx in range(n_y * n_x):
        i, j = divmod(idx, n_x); st = int(subtype_map[i, j])
        # Publication-style ATPase-like simplified coloring:
        #   red  = type I
        #   white/off-white = type II, including IIa and IIx
        #   lavender-gray = denervated/unreinnervated/remodeled
        if denervated[i, j]: base = np.array([0.60, 0.56, 0.65])
        elif st == 0: base = np.array([0.88, 0.05, 0.04])
        elif st == 1: base = np.array([0.96, 0.92, 0.86])
        else: base = np.array([1.00, 0.98, 0.94])
        base = np.clip(base + rng.normal(0.0, 0.03, size=3), 0.05, 0.98)
        if csa[i, j] < 0.7: base = 0.8 * base + 0.2 * np.array([0.44, 0.42, 0.50])
        elif csa[i, j] > 1.25: base = np.clip(base * 1.03 + 0.02, 0, 1)
        colors[idx] = base
    rgb = np.clip(colors[labels] + rng.normal(0.0, 0.04, size=(H, W, 3)), 0.0, 1.0)
    boundary = np.zeros((H, W), dtype=bool); boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]; boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return np.clip((1.0 - 0.88 * boundary[..., None]) * rgb + (0.88 * boundary[..., None]) * np.array([0.45, 0.45, 0.45]), 0.0, 1.0)

def plot_stage_panels(mu_map, subtype_map, csa, denervated, stage_title, out_file, seed=0):
    biopsy = render_biopsy_like(subtype_map, csa, denervated, seed=seed)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axes[0, 0].imshow(mu_map, cmap="tab20", interpolation="nearest"); axes[0, 0].set_title(f"{stage_title}: MU distribution"); axes[0, 0].axis("off")
    subtype_rgb = np.zeros((*subtype_map.shape, 3), dtype=float)
    subtype_rgb[subtype_map == 0] = [0.88, 0.05, 0.04]; subtype_rgb[subtype_map == 1] = [0.96, 0.92, 0.86]; subtype_rgb[subtype_map == 2] = [1.00, 0.98, 0.94]; subtype_rgb[denervated] = [0.60, 0.56, 0.65]
    axes[0, 1].imshow(subtype_rgb, interpolation="nearest"); axes[0, 1].set_title("Coarse fiber-type map"); axes[0, 1].axis("off")
    axes[1, 0].imshow(biopsy, interpolation="nearest"); axes[1, 0].set_title("Synthetic biopsy-like image"); axes[1, 0].axis("off")
    im = axes[1, 1].imshow(csa, cmap="viridis", vmin=0.4, vmax=1.6, interpolation="nearest"); axes[1, 1].set_title("Normalized CSA"); axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046); fig.savefig(out_file, dpi=220); plt.close(fig)

PATCH_TEMPLATE = """# Patch for OpenDiHu variables.py. Remove/comment the original motor_units loop.
import json
import os

scenario_name = "{stage_name}"

Conductivity = 3.828      # [mS/cm] sigma, conductivity

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

# The OpenDiHu example usually passes mu_no as a 0-based index to callbacks.
# Keep default 0. If your callback receives 1..20, set:
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

"""

def write_opendihu_variables_patch(stage_dir, stage_name, out_file):
    with open(out_file, "w") as f: f.write(PATCH_TEMPLATE.format(stage_dir=stage_dir, stage_name=stage_name))

def plot_overview(case_records, out_file):
    fig, axes = plt.subplots(4, len(case_records), figsize=(3.6 * len(case_records), 12), constrained_layout=True)
    if len(case_records) == 1: axes = np.asarray(axes).reshape(4, 1)
    for c, rec in enumerate(case_records):
        axes[0, c].imshow(rec["mu_map"], cmap="tab20", interpolation="nearest"); axes[0, c].set_title(rec["label"]); axes[0, c].axis("off")
        subtype_rgb = np.zeros((*rec["subtype_map"].shape, 3), dtype=float)
        subtype_rgb[rec["subtype_map"] == 0] = [0.05, 0.05, 0.05]; subtype_rgb[rec["subtype_map"] == 1] = [0.55, 0.55, 0.58]; subtype_rgb[rec["subtype_map"] == 2] = [0.95, 0.95, 0.92]; subtype_rgb[rec["denervated"]] = [0.58, 0.56, 0.62]
        axes[1, c].imshow(subtype_rgb, interpolation="nearest"); axes[1, c].set_title("Subtype map"); axes[1, c].axis("off")
        axes[2, c].imshow(render_biopsy_like(rec["subtype_map"], rec["csa"], rec["denervated"], seed=100 + c), interpolation="nearest"); axes[2, c].set_title("Biopsy-like"); axes[2, c].axis("off")
        axes[3, c].imshow(rec["csa"], cmap="viridis", vmin=0.4, vmax=1.6, interpolation="nearest"); axes[3, c].set_title("CSA"); axes[3, c].axis("off")
    fig.savefig(out_file, dpi=220); plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate OpenDiHu MU cases with motor_units.json.")
    parser.add_argument("--out-dir", default="opendihu_mu_case_package")
    parser.add_argument("--n-fibers-x", type=int, default=37); parser.add_argument("--n-fibers-y", type=int, default=37); parser.add_argument("--n-motor-units", type=int, default=20)
    parser.add_argument("--scenario", choices=["aging_type2", "exercise_rescued", "als_like_failed_rescue", "classic_like"], default="aging_type2")
    parser.add_argument("--seed", type=int, default=1); parser.add_argument("--index-base", type=int, choices=[0, 1], default=1); parser.add_argument("--n-firing-rows", type=int, default=200)
    parser.add_argument("--physiology-recruitment", action="store_true")
    parser.add_argument("--type1-fraction", type=float, default=0.50, help="Fraction of type I MUs/fibers in healthy map.")
    parser.add_argument("--type2a-fraction", type=float, default=0.35, help="Fraction of type IIa among all MUs; remaining non-type-I are IIx.")
    parser.add_argument("--territory-sigma-factor", type=float, default=0.40, help="Larger values make healthy MU fibers more intermingled.")
    parser.add_argument("--repulsion-factor", type=float, default=0.25, help="Penalty for assigning adjacent fibers to the same MU in healthy initialization.")
    args = parser.parse_args()
    ensure_dir(args.out_dir)
    base_rng = np.random.default_rng(args.seed)
    scenario = SCENARIOS[args.scenario]
    mu_map0, centers, mu_type = initialize_distribution(
        args.n_fibers_x, args.n_fibers_y, args.n_motor_units, base_rng,
        type1_fraction=args.type1_fraction,
        type2a_fraction=args.type2a_fraction,
        territory_sigma_factor=args.territory_sigma_factor,
        repulsion_factor=args.repulsion_factor,
    )
    stages = [("healthy", 0.00), ("death_25", 0.25), ("death_50", 0.50), ("death_75", 0.75)]
    case_records = []
    for idx, (name, kill_fraction) in enumerate(stages):
        stage_dir = os.path.abspath(os.path.join(args.out_dir, name)); ensure_dir(stage_dir)
        effective_scenario = SCENARIOS["healthy"] if name == "healthy" else scenario
        rng = np.random.default_rng(args.seed + 100 * idx)
        if kill_fraction <= 0:
            killed = np.zeros(len(mu_type), dtype=bool); mu_map = mu_map0.copy()
        else:
            killed = select_killed_mus(mu_map0, mu_type, effective_scenario, kill_fraction, rng); mu_map = remodel_distribution(mu_map0, mu_type, killed, centers, effective_scenario, rng)
        subtype_map, csa, denervated = compute_stage_fields(mu_map0, mu_map, mu_type, killed)
        write_distribution(mu_map, os.path.join(stage_dir, "MU_fibre_distribution_37x37_20.txt"), args.index_base)
        write_firing_times(killed, os.path.join(stage_dir, "MU_firing_times_always.txt"), args.n_firing_rows)
        motor_units = make_motor_units_json(mu_type, killed, os.path.join(stage_dir, "motor_units.json"), args.index_base, args.seed + 1000 * idx, original_opendihu_style=not args.physiology_recruitment)
        write_summary(mu_map0, mu_map, mu_type, killed, motor_units, os.path.join(stage_dir, "motor_unit_summary.csv"))
        plot_stage_panels(mu_map, subtype_map, csa, denervated, name.replace("_", " "), os.path.join(stage_dir, "mu_distribution_and_biopsy_like.png"), seed=args.seed + idx)
        write_opendihu_variables_patch(stage_dir, name, os.path.join(stage_dir, f"opendihu_variables_patch_{name}.py"))
        metadata = {
            "stage_name": name, "kill_fraction": kill_fraction, "n_fibers_x": args.n_fibers_x, "n_fibers_y": args.n_fibers_y,
            "n_fibers_total": args.n_fibers_x * args.n_fibers_y, "n_motor_units": args.n_motor_units,
            "type1_fraction": args.type1_fraction, "type2a_fraction": args.type2a_fraction,
            "territory_sigma_factor": args.territory_sigma_factor, "repulsion_factor": args.repulsion_factor,
            "scenario": effective_scenario.name, "seed": args.seed + 100 * idx, "index_base": args.index_base,
            "distribution_file": "MU_fibre_distribution_37x37_20.txt", "firing_file": "MU_firing_times_always.txt", "motor_units_file": "motor_units.json",
            "fiber_order": "row-major: fiber_no = y*n_fibers_x + x",
            "n_killed_or_silent_mus": int(np.count_nonzero(killed)), "n_silent_unreinnervated_fibers": int(np.count_nonzero(denervated)),
            "scenario_parameters": asdict(effective_scenario),
        }
        with open(os.path.join(stage_dir, "generation_metadata.json"), "w") as f: json.dump(metadata, f, indent=2)
        case_records.append({"label": name.replace("_", "\\n"), "mu_map": mu_map, "subtype_map": subtype_map, "csa": csa, "denervated": denervated})
    overview_file = os.path.join(args.out_dir, "all_cases_overview.png"); plot_overview(case_records, overview_file)
    with open(os.path.join(args.out_dir, "README_how_to_use_with_opendihu.txt"), "w") as f:
        f.write(f"""Generated OpenDiHu MU case package.

Each stage folder contains:
  MU_fibre_distribution_37x37_20.txt
  MU_firing_times_always.txt
  motor_units.json
  opendihu_variables_patch.py

Use a stage by setting:
  export OPENDIHU_GENERATED_MU_DIR={os.path.abspath(os.path.join(args.out_dir, 'death_50'))}

Then replace the original motor_units loop with:
  import json, os
  generated_input_directory = os.environ.get("OPENDIHU_GENERATED_MU_DIR")
  firing_times_file = os.path.join(generated_input_directory, "MU_firing_times_always.txt")
  fiber_distribution_file = os.path.join(generated_input_directory, "MU_fibre_distribution_37x37_20.txt")
  with open(os.path.join(generated_input_directory, "motor_units.json"), "r") as f:
      motor_units = json.load(f)["motor_units"]
  n_motor_units = len(motor_units)

Your existing get_am/get_cm/get_specific_states_* functions can stay unchanged
because motor_units.json uses the same keys:
  radius, cm, activation_start_time, stimulation_frequency, jitter

Or copy the full block from any stage's opendihu_variables_patch.py.
""")
    print("Created complete OpenDiHu MU case package:")
    print(f"  {args.out_dir}")
    print(f"  Overview image: {overview_file}")

if __name__ == "__main__":
    main()
