#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# OpenDiHu-aligned MU-distribution revision: exact exponential MU size law,
# independent spatial territories, rank-linked phenotype, and denervation/remodeling.

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

def make_motor_units_json(mu_type, killed, out_file, index_base, seed,
                          recruitment_rank=None, target_fiber_counts=None,
                          size_weights=None, original_opendihu_style=True):
    """Create the MU physiology table consumed by OpenDiHu.

    In the realistic version, MU biophysical/recruitment properties are tied to
    physiological recruitment rank rather than the arbitrary MU ID.  MU IDs are
    intentionally spatially randomized, so using mu_no directly would otherwise
    break the size-principle relationship.
    """
    n_motor_units = len(mu_type)
    rng = random.Random(seed)

    if recruitment_rank is None:
        recruitment_rank = np.arange(n_motor_units, dtype=int)
    recruitment_rank = np.asarray(recruitment_rank, dtype=int)

    if target_fiber_counts is None:
        target_fiber_counts = np.full(n_motor_units, np.nan)
    if size_weights is None:
        size_weights = np.full(n_motor_units, np.nan)

    motor_units = []
    for mu_no in range(n_motor_units):
        type_id = int(mu_type[mu_no])
        rank = int(recruitment_rank[mu_no])

        # Low-threshold units are smaller and generally fire faster; high-threshold
        # units are larger and have lower peak discharge rates in this simplified
        # OpenDiHu-compatible parameterization.
        radius = exp_range_value(rank, n_motor_units, 40.0, 55.0, increasing=True)
        stimulation_frequency = exp_range_value(rank, n_motor_units, 7.0, 24.0, increasing=False)

        if original_opendihu_style:
            activation_start_time = 0.0 if rank < n_motor_units / 2 else 10.0
        else:
            # Explicit type-linked recruitment timing.
            activation_start_time = 0.0 if type_id == 0 else (5.0 if type_id == 1 else 10.0)

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
            "recruitment_rank_0_based": rank,
            "recruitment_order_1_based": rank + 1,
            "target_fiber_count_healthy": None if not np.isfinite(target_fiber_counts[mu_no]) else int(target_fiber_counts[mu_no]),
            "relative_size_weight": None if not np.isfinite(size_weights[mu_no]) else float(size_weights[mu_no]),
            "radius": float(radius),
            "cm": float(MU_CM[type_id]),
            "activation_start_time": float(activation_start_time),
            "stimulation_frequency": float(stimulation_frequency),
            "jitter": [0.1 * rng.uniform(-1, 1) for _ in range(100)],
        })

    payload = {
        "format": "opendihu_motor_units_json_v2_realistic",
        "description": "Physiology-linked motor_units list for OpenDiHu; MU size and recruitment rank are decoupled from spatial MU ID.",
        "index_base": int(index_base),
        "n_motor_units": int(n_motor_units),
        "cm_convention": {
            "type_I_slow": MU_CM[0],
            "type_IIa_fast_intermediate": MU_CM[1],
            "type_IIx_fast": MU_CM[2],
        },
        "motor_units": motor_units,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    return motor_units


def make_mu_centers(n_x, n_y, n_mu, rng):
    """Place MU territory centers quasi-uniformly, then jitter them.

    The centers define only the *territory preference*.  Fibers of an MU are not
    forced into a compact block; broad Gaussian territories and neighbor
    repulsion keep healthy MU fibers intermingled.
    """
    n_cols = int(math.ceil(math.sqrt(n_mu * n_x / n_y)))
    n_rows = int(math.ceil(n_mu / n_cols))
    xs = np.linspace(0.5 * n_x / n_cols, n_x - 0.5 * n_x / n_cols, n_cols)
    ys = np.linspace(0.5 * n_y / n_rows, n_y - 0.5 * n_y / n_rows, n_rows)
    centers = []
    for y in ys:
        for x in xs:
            if len(centers) < n_mu:
                centers.append((x, y))
    centers = np.asarray(centers, dtype=float)
    centers += rng.normal(0.0, 0.35, size=centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], 0, n_x - 1)
    centers[:, 1] = np.clip(centers[:, 1], 0, n_y - 1)
    return centers


def _type_counts(n_mu, type1_fraction=0.50, type2a_fraction=0.35):
    """Return exact counts of type I, IIa and IIx MUs.

    Fractions refer to the MU pool, not directly to fiber fractions.  Type is
    linked to recruitment rank later, while MU IDs/territory centers remain
    spatially shuffled so this does not create a cross-sectional type gradient.
    """
    type1_fraction = float(np.clip(type1_fraction, 0.0, 1.0))
    type2a_fraction = float(np.clip(type2a_fraction, 0.0, 1.0 - type1_fraction))
    n_type1 = int(round(type1_fraction * n_mu))
    n_type2a = int(round(type2a_fraction * n_mu))
    n_type2x = max(0, int(n_mu) - n_type1 - n_type2a)
    # Guard against rounding overflow.
    if n_type1 + n_type2a > n_mu:
        n_type2a = max(0, n_mu - n_type1)
        n_type2x = 0
    return n_type1, n_type2a, n_type2x


def _largest_remainder_integerization(raw_counts, total_fibers):
    """Convert positive real-valued target counts to integers with exact total."""
    raw_counts = np.asarray(raw_counts, dtype=float)
    if np.any(raw_counts <= 0):
        raise ValueError("All raw MU target counts must be positive.")

    scaled = raw_counts / raw_counts.sum() * int(total_fibers)
    counts = np.floor(scaled).astype(int)
    counts = np.maximum(counts, 1)

    diff = int(total_fibers) - int(counts.sum())
    if diff > 0:
        frac = scaled - np.floor(scaled)
        order = np.argsort(-frac)
        i = 0
        while diff > 0:
            counts[int(order[i % len(order)])] += 1
            diff -= 1
            i += 1
    elif diff < 0:
        # Remove from units that are most over-rounded, while keeping >=1 fiber.
        frac = scaled - np.floor(scaled)
        order = np.argsort(frac)
        i = 0
        safety = 0
        while diff < 0:
            k = int(order[i % len(order)])
            if counts[k] > 1:
                counts[k] -= 1
                diff += 1
            i += 1
            safety += 1
            if safety > 100000:
                raise RuntimeError("Could not integerize MU targets while preserving >=1 fiber/MU.")

    if counts.sum() != int(total_fibers):
        raise RuntimeError("Failed to integerize MU targets to the requested total fiber count.")
    return counts


def make_exponential_mu_profile(n_mu, total_fibers, rng,
                                mu_size_basis=1.20,
                                type1_fraction=0.50,
                                type2a_fraction=0.35,
                                spatially_shuffle_ranks=True,
                                mu_type_mode="independent"):
    """Create an OpenDiHu-style exponential healthy MU pool.

    OpenDiHu describes MU fiber counts as exponentially distributed according to
    ``basis**x`` with lower MUs containing fewer fibers.  This function makes that
    law explicit and separates it from the spatial territory model.

    Rank-space law
    --------------
        weight(r) = mu_size_basis ** r,   r = 0, ..., n_mu-1

    The weights are normalized to exactly ``total_fibers`` and integerized by the
    largest-remainder method.  By default, recruitment ranks are then randomly
    mapped onto spatial MU IDs.  Therefore low/high MU rank does *not* imply a
    particular location in the muscle cross-section.

    MU phenotype is deliberately treated as a separate modeling choice because
    the OpenDiHu description quoted for the distribution generator specifies the
    exponential MU-size law and territory handling, but does not specify a
    fiber-type law.  ``mu_type_mode="independent"`` (default) preserves the
    previous randomly mixed type assignment.  ``mu_type_mode="rank_linked"``
    assigns low ranks to type I, intermediate ranks to IIa and high ranks to IIx.
    """
    n_mu = int(n_mu)
    total_fibers = int(total_fibers)
    if n_mu < 1:
        raise ValueError("n_mu must be >= 1")
    if total_fibers < n_mu:
        raise ValueError("Need at least one modeled fiber per MU.")
    if mu_size_basis <= 0:
        raise ValueError("mu_size_basis must be > 0.")

    ranks = np.arange(n_mu, dtype=int)
    rank_weights = np.power(float(mu_size_basis), ranks.astype(float))
    target_by_rank = _largest_remainder_integerization(rank_weights, total_fibers)

    # Map physiological rank to arbitrary spatial MU ID.
    if spatially_shuffle_ranks:
        mu_ids_by_rank = rng.permutation(n_mu)
    else:
        mu_ids_by_rank = np.arange(n_mu, dtype=int)

    recruitment_rank = np.empty(n_mu, dtype=int)
    recruitment_rank[mu_ids_by_rank] = ranks

    target_sizes = np.empty(n_mu, dtype=int)
    target_sizes[mu_ids_by_rank] = target_by_rank

    size_weights = np.empty(n_mu, dtype=float)
    size_weights[mu_ids_by_rank] = rank_weights

    n_type1, n_type2a, n_type2x = _type_counts(
        n_mu, type1_fraction=type1_fraction, type2a_fraction=type2a_fraction
    )
    if mu_type_mode == "rank_linked":
        type_by_rank = np.empty(n_mu, dtype=int)
        type_by_rank[:n_type1] = 0
        type_by_rank[n_type1:n_type1 + n_type2a] = 1
        type_by_rank[n_type1 + n_type2a:] = 2
        mu_type = np.empty(n_mu, dtype=int)
        mu_type[mu_ids_by_rank] = type_by_rank
    elif mu_type_mode == "independent":
        mu_type = np.array(
            [0] * n_type1 + [1] * n_type2a + [2] * n_type2x,
            dtype=int,
        )
        rng.shuffle(mu_type)
    else:
        raise ValueError("mu_type_mode must be 'independent' or 'rank_linked'.")

    return (
        target_sizes,
        recruitment_rank,
        size_weights,
        mu_type,
        target_by_rank,
        mu_ids_by_rank,
    )

def initialize_distribution(n_x, n_y, n_mu, rng,
                            type1_fraction=0.50, type2a_fraction=0.35,
                            mu_size_basis=1.20,
                            spatially_shuffle_ranks=True,
                            mu_type_mode="independent",
                            territory_sigma_factor=0.35,
                            repulsion_factor=0.25,
                            territory_size_scaling=0.50,
                            quota_power=1.0):
    """Initialize the healthy fiber-to-MU map using an explicit exponential law.

    The model has two deliberately separate layers:

    1. MU size/recruitment layer
       ``target_fibers(rank) ∝ mu_size_basis**rank``.
       With the default basis=1.2 this follows the convention described by the
       OpenDiHu MU distribution generator.

    2. Spatial territory layer
       Each MU receives a center and a broad Gaussian territory.  Larger MUs can
       receive broader territories, but rank is shuffled across spatial MU IDs by
       default.  Same-MU neighbor repulsion keeps healthy fibers intermingled.

    MU fiber type is separate from the OpenDiHu size law.  The default
    ``mu_type_mode=independent`` keeps types randomly mixed across MU ranks; use
    ``rank_linked`` only when that additional physiological hypothesis is desired.

    Exact target quotas are honored, so spatial assignment cannot flatten the
    prescribed exponential MU-size spectrum.
    """
    if not (0 < repulsion_factor <= 1.0):
        raise ValueError("repulsion_factor must be in (0, 1].")
    if territory_sigma_factor <= 0:
        raise ValueError("territory_sigma_factor must be > 0.")
    if quota_power <= 0:
        raise ValueError("quota_power must be > 0.")

    total_fibers = int(n_x) * int(n_y)
    centers = make_mu_centers(n_x, n_y, n_mu, rng)
    (
        target_sizes,
        recruitment_rank,
        size_weights,
        mu_type,
        target_by_rank,
        mu_ids_by_rank,
    ) = make_exponential_mu_profile(
        n_mu,
        total_fibers,
        rng,
        mu_size_basis=mu_size_basis,
        type1_fraction=type1_fraction,
        type2a_fraction=type2a_fraction,
        spatially_shuffle_ranks=spatially_shuffle_ranks,
        mu_type_mode=mu_type_mode,
    )

    mean_target = float(total_fibers) / float(n_mu)
    base_sigma = float(territory_sigma_factor) * max(n_x, n_y)
    relative_size = target_sizes.astype(float) / mean_target
    sigma_by_mu = base_sigma * np.power(relative_size, float(territory_size_scaling))
    sigma_by_mu = np.maximum(sigma_by_mu, 1e-6)

    mu_map = -np.ones((n_y, n_x), dtype=int)
    remaining = target_sizes.astype(int).copy()

    coords = [(i, j) for i in range(n_y) for j in range(n_x)]
    rng.shuffle(coords)

    for i, j in coords:
        available = remaining > 0
        if not np.any(available):
            raise RuntimeError("No MU quota remaining before all fibers were assigned.")

        d2 = (centers[:, 0] - j) ** 2 + (centers[:, 1] - i) ** 2
        spatial = np.exp(-d2 / (2.0 * sigma_by_mu ** 2))

        # Exact quota term: completed MUs cannot receive more fibers.
        quota_term = np.power(np.maximum(remaining, 0), float(quota_power))
        p = spatial * quota_term
        p[~available] = 0.0

        # Healthy MU fibers are dispersed/intermingled rather than compact blocks.
        for ii, jj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ii < n_y and 0 <= jj < n_x and mu_map[ii, jj] >= 0:
                p[int(mu_map[ii, jj])] *= float(repulsion_factor)

        if p.sum() <= 0:
            candidates = np.flatnonzero(available)
            k = int(candidates[np.argmax(remaining[candidates])])
        else:
            k = int(rng.choice(n_mu, p=p / p.sum()))

        mu_map[i, j] = k
        remaining[k] -= 1

    realized = np.bincount(mu_map.ravel(), minlength=n_mu)
    if not np.array_equal(realized, target_sizes):
        raise RuntimeError(
            f"Healthy MU quota mismatch: target={target_sizes.tolist()}, "
            f"realized={realized.tolist()}"
        )

    return (
        mu_map,
        centers,
        mu_type,
        target_sizes,
        recruitment_rank,
        size_weights,
        sigma_by_mu,
        target_by_rank,
        mu_ids_by_rank,
    )

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

def write_summary(mu_map0, mu_map, mu_type, killed, motor_units, out_file,
                  target_sizes=None, recruitment_rank=None, sigma_by_mu=None):
    n_mu = len(mu_type)
    initial_counts = np.bincount(mu_map0.ravel(), minlength=n_mu)
    final_counts = np.bincount(mu_map.ravel(), minlength=n_mu)
    target_sizes = initial_counts if target_sizes is None else np.asarray(target_sizes)
    recruitment_rank = np.arange(n_mu) if recruitment_rank is None else np.asarray(recruitment_rank)
    sigma_by_mu = np.full(n_mu, np.nan) if sigma_by_mu is None else np.asarray(sigma_by_mu)

    rows = []
    for k in range(n_mu):
        rows.append({
            "mu_id_0_based": k,
            "mu_id_1_based": k + 1,
            "mu_type_id": int(mu_type[k]),
            "mu_type_name": MU_TYPE_NAMES[int(mu_type[k])],
            "recruitment_rank_0_based": int(recruitment_rank[k]),
            "recruitment_order_1_based": int(recruitment_rank[k]) + 1,
            "target_healthy_fiber_count": int(target_sizes[k]),
            "territory_sigma_grid_units": float(sigma_by_mu[k]) if np.isfinite(sigma_by_mu[k]) else "",
            "killed_or_silent": bool(killed[k]),
            "initial_fiber_count": int(initial_counts[k]),
            "final_assigned_fiber_count": int(final_counts[k]),
            "expansion_ratio": float(final_counts[k] / max(initial_counts[k], 1)),
            "radius_um": float(motor_units[k]["radius"]),
            "cm": float(motor_units[k]["cm"]),
            "activation_start_time_s": float(motor_units[k]["activation_start_time"]),
            "stimulation_frequency_hz": float(motor_units[k]["stimulation_frequency"]),
        })
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

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


def plot_healthy_mu_size_distribution(mu_map, mu_type, recruitment_rank,
                                      mu_size_basis, out_file):
    """Validate the realized OpenDiHu-style exponential MU-size spectrum."""
    n_mu = len(mu_type)
    counts_by_mu = np.bincount(mu_map.ravel(), minlength=n_mu)
    mu_ids = np.arange(1, n_mu + 1)

    # Convert MU-ID order to recruitment-rank order.
    counts_by_rank = np.empty(n_mu, dtype=int)
    for mu_id in range(n_mu):
        counts_by_rank[int(recruitment_rank[mu_id])] = int(counts_by_mu[mu_id])
    ranks = np.arange(n_mu, dtype=int)

    # Continuous normalized basis**rank curve for comparison.
    raw = np.power(float(mu_size_basis), ranks.astype(float))
    theoretical = raw / raw.sum() * counts_by_mu.sum()

    # Fit log(count) = intercept + slope*rank as an independent diagnostic.
    log_counts = np.log(np.maximum(counts_by_rank.astype(float), 1e-12))
    slope, intercept = np.polyfit(ranks.astype(float), log_counts, 1)
    fitted = np.exp(intercept + slope * ranks)
    ss_res = float(np.sum((counts_by_rank - fitted) ** 2))
    ss_tot = float(np.sum((counts_by_rank - counts_by_rank.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    fitted_basis = math.exp(float(slope))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    axes[0].bar(mu_ids, counts_by_mu)
    axes[0].set_xlabel("Motor Unit ID")
    axes[0].set_ylabel("Modeled fibers")
    axes[0].set_title("Fiber count by spatial MU ID")
    axes[0].set_xticks(mu_ids)

    axes[1].bar(ranks + 1, counts_by_rank, label="Integerized target / realized")
    axes[1].plot(ranks + 1, theoretical, marker="o", label=f"Normalized {mu_size_basis:g}^rank law")
    axes[1].set_xlabel("MU recruitment / size rank")
    axes[1].set_ylabel("Modeled fibers")
    axes[1].set_title("Exponential MU size law")
    axes[1].legend()

    n_bins = min(10, max(5, int(round(math.sqrt(n_mu)) + 2)))
    axes[2].hist(counts_by_mu, bins=n_bins)
    axes[2].set_xlabel("Modeled fibers per MU")
    axes[2].set_ylabel("Number of MUs")
    axes[2].set_title("MU fiber-count histogram")

    stats = (
        f"Total fibers = {int(counts_by_mu.sum())}\n"
        f"MUs = {n_mu}\n"
        f"Basis requested = {mu_size_basis:.4g}\n"
        f"Basis from log-fit = {fitted_basis:.4f}\n"
        f"Rank-fit R² = {r2:.4f}\n"
        f"Min = {counts_by_rank.min()}\n"
        f"Max = {counts_by_rank.max()}\n"
        f"Max/min = {counts_by_rank.max()/max(counts_by_rank.min(), 1):.2f}"
    )
    axes[2].text(
        0.98, 0.98, stats,
        transform=axes[2].transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    fig.suptitle("OpenDiHu-aligned healthy MU fiber-count distribution")
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def write_healthy_mu_profile_csv(out_file, mu_type, recruitment_rank,
                                 target_sizes, sigma_by_mu):
    """Write the healthy rank/type/size/territory mapping for auditability."""
    rows = []
    for mu_id in range(len(mu_type)):
        rows.append({
            "mu_id_0_based": int(mu_id),
            "mu_id_1_based": int(mu_id + 1),
            "recruitment_rank_0_based": int(recruitment_rank[mu_id]),
            "recruitment_order_1_based": int(recruitment_rank[mu_id] + 1),
            "mu_type_id": int(mu_type[mu_id]),
            "mu_type_name": MU_TYPE_NAMES[int(mu_type[mu_id])],
            "healthy_target_fiber_count": int(target_sizes[mu_id]),
            "territory_sigma_grid_units": float(sigma_by_mu[mu_id]),
        })
    rows.sort(key=lambda x: x["recruitment_rank_0_based"])
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

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


# =============================================================================
# Grouped repeated-study generation
# =============================================================================
# This section extends the single-realization Protocol A generator above into a
# reproducible repeated-study generator.  Each statistical replicate is a group
# (g-1, g-2, ...).  Every group contains the same 8 paired simulation cases:
#
#   Protocol A: healthy, death_25, death_50, death_75
#   Protocol B: healthy, death_25, death_50, death_75
#
# Protocol B is derived from its Protocol A partner, so the MU distribution,
# denervation/reinnervation realization, and firing gate are exactly identical.
# Only stimulation_frequency of surviving MUs is changed.
# =============================================================================

import copy
import hashlib
import shutil
import shlex
from pathlib import Path
from datetime import datetime, timezone

GROUP_STAGES = [
    ("healthy", 0.00),
    ("death_25", 0.25),
    ("death_50", 0.50),
    ("death_75", 0.75),
]

CORE_INPUT_FILES = [
    "MU_fibre_distribution_37x37_20.txt",
    "MU_firing_times_always.txt",
    "motor_units.json",
]


def sanitize_group_number(value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError("Group number must be >= 1.")
    return value


def group_tag(group_number: int) -> str:
    return f"g-{sanitize_group_number(group_number)}"


def case_id(group_number: int, protocol: str, stage: str) -> str:
    protocol = str(protocol).upper()
    if protocol not in ("A", "B"):
        raise ValueError("protocol must be A or B")
    return f"{group_tag(group_number)}_{protocol}_{stage}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tagged_alias(path: Path, scenario_name: str) -> Path:
    """Create a group/case-tagged alias while retaining canonical runtime file."""
    if path.suffix:
        tagged_name = f"{path.stem}_{scenario_name}{path.suffix}"
    else:
        tagged_name = f"{path.name}_{scenario_name}"
    tagged = path.with_name(tagged_name)
    shutil.copy2(path, tagged)
    return tagged


def write_group_patch(stage_dir: Path, scenario_name: str, end_time_s: float) -> Path:
    """Write the OpenDiHu variables patch with group-aware scenario_name."""
    end_time_ms = float(end_time_s) * 1000.0
    text = PATCH_TEMPLATE.format(stage_dir=str(stage_dir.resolve()), stage_name=scenario_name)
    text = text.replace(
        "end_time = 100.0                    # [ms] end time of the simulation",
        f"end_time = {end_time_ms:.6g}                # [ms] end time of the simulation ({end_time_s:g} s)",
    )
    tagged = stage_dir / f"opendihu_variables_patch_{scenario_name}.py"
    tagged.write_text(text)
    # Canonical convenience copy.  Tagged file is the authoritative one.
    shutil.copy2(tagged, stage_dir / "opendihu_variables_patch.py")
    return tagged


def _read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    # Use the union of keys so Protocol A and Protocol B rows can coexist in
    # the same manifest even when one protocol has extra diagnostic columns.
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def get_expansion_ratio_from_rows(summary_rows, mu_no: int) -> float:
    for row in summary_rows:
        try:
            if int(row.get("mu_id_0_based", -1)) == int(mu_no):
                val = float(row.get("expansion_ratio", 1.0))
                if np.isfinite(val) and val > 0:
                    return val
        except (TypeError, ValueError):
            pass
    return 1.0


def compute_global_factor(active_mu_fraction: float,
                          max_global_factor: float,
                          global_gain: float) -> float:
    active_mu_fraction = max(float(active_mu_fraction), 1e-12)
    return float(min(float(max_global_factor), active_mu_fraction ** (-float(global_gain))))


def add_case_metadata_to_motor_units(payload: dict, *, group_number: int,
                                     protocol: str, stage: str,
                                     scenario_name: str, group_seed: int) -> None:
    payload["group_number"] = int(group_number)
    payload["group_tag"] = group_tag(group_number)
    payload["protocol"] = str(protocol).upper()
    payload["stage"] = stage
    payload["scenario_name"] = scenario_name
    payload["group_seed"] = int(group_seed)


def create_protocol_a_group(group_number: int, group_dir: Path, args):
    """Generate four independently remodeled Protocol A stages for one group."""
    gtag = group_tag(group_number)
    group_seed = int(args.base_seed + (group_number - 1) * args.group_seed_stride)
    protocol_dir = group_dir / "protocol_A"
    ensure_dir(str(protocol_dir))

    base_rng = np.random.default_rng(group_seed)
    scenario = SCENARIOS[args.scenario]

    (
        mu_map0, centers, mu_type, target_sizes, recruitment_rank,
        size_weights, sigma_by_mu, target_by_rank, mu_ids_by_rank
    ) = initialize_distribution(
        args.n_fibers_x, args.n_fibers_y, args.n_motor_units, base_rng,
        type1_fraction=args.type1_fraction,
        type2a_fraction=args.type2a_fraction,
        territory_sigma_factor=args.territory_sigma_factor,
        repulsion_factor=args.repulsion_factor,
        territory_size_scaling=args.territory_size_scaling,
        quota_power=args.quota_power,
        mu_size_basis=args.mu_size_basis,
        spatially_shuffle_ranks=not args.no_spatial_rank_shuffle,
        mu_type_mode=args.mu_type_mode,
    )

    # Group-level healthy-pool diagnostics.
    healthy_dist = protocol_dir / f"healthy_mu_size_distribution_{gtag}.png"
    plot_healthy_mu_size_distribution(
        mu_map0, mu_type, recruitment_rank, args.mu_size_basis, str(healthy_dist)
    )
    healthy_profile = protocol_dir / f"healthy_mu_profile_{gtag}.csv"
    write_healthy_mu_profile_csv(
        str(healthy_profile), mu_type, recruitment_rank, target_sizes, sigma_by_mu
    )

    case_records = []
    manifest_rows = []

    for idx, (stage, kill_fraction) in enumerate(GROUP_STAGES):
        sid = case_id(group_number, "A", stage)
        stage_dir = protocol_dir / stage
        ensure_dir(str(stage_dir))

        stage_seed = int(group_seed + 100 * idx)
        motor_unit_seed = int(group_seed + 1000 * idx)
        effective_scenario = SCENARIOS["healthy"] if stage == "healthy" else scenario
        rng = np.random.default_rng(stage_seed)

        if kill_fraction <= 0:
            killed = np.zeros(len(mu_type), dtype=bool)
            mu_map = mu_map0.copy()
        else:
            killed = select_killed_mus(
                mu_map0, mu_type, effective_scenario, kill_fraction, rng
            )
            mu_map = remodel_distribution(
                mu_map0, mu_type, killed, centers, effective_scenario, rng
            )

        subtype_map, csa, denervated = compute_stage_fields(
            mu_map0, mu_map, mu_type, killed
        )

        # Canonical OpenDiHu runtime input files.
        distribution_file = stage_dir / "MU_fibre_distribution_37x37_20.txt"
        firing_file = stage_dir / "MU_firing_times_always.txt"
        motor_units_file = stage_dir / "motor_units.json"
        write_distribution(mu_map, str(distribution_file), args.index_base)
        write_firing_times(killed, str(firing_file), args.n_firing_rows)

        motor_units = make_motor_units_json(
            mu_type, killed, str(motor_units_file),
            args.index_base, motor_unit_seed,
            recruitment_rank=recruitment_rank,
            target_fiber_counts=target_sizes,
            size_weights=size_weights,
            original_opendihu_style=not args.physiology_recruitment,
        )
        payload = json.loads(motor_units_file.read_text())
        add_case_metadata_to_motor_units(
            payload,
            group_number=group_number,
            protocol="A",
            stage=stage,
            scenario_name=sid,
            group_seed=group_seed,
        )
        motor_units_file.write_text(json.dumps(payload, indent=2))

        # Canonical + group-tagged audit files.
        summary_file = stage_dir / "motor_unit_summary.csv"
        write_summary(
            mu_map0, mu_map, mu_type, killed, motor_units, str(summary_file),
            target_sizes=target_sizes,
            recruitment_rank=recruitment_rank,
            sigma_by_mu=sigma_by_mu,
        )
        tagged_summary = copy_tagged_alias(summary_file, sid)

        image_file = stage_dir / "mu_distribution_and_biopsy_like.png"
        plot_stage_panels(
            mu_map, subtype_map, csa, denervated,
            sid.replace("_", " "), str(image_file), seed=stage_seed
        )
        tagged_image = copy_tagged_alias(image_file, sid)

        patch_file = write_group_patch(stage_dir, sid, args.end_time_s)

        metadata = {
            "group_number": int(group_number),
            "group_tag": gtag,
            "protocol": "A",
            "stage": stage,
            "scenario_name": sid,
            "kill_fraction": float(kill_fraction),
            "group_seed": group_seed,
            "stage_seed": stage_seed,
            "motor_unit_seed": motor_unit_seed,
            "end_time_s": float(args.end_time_s),
            "n_fibers_x": args.n_fibers_x,
            "n_fibers_y": args.n_fibers_y,
            "n_fibers_total": args.n_fibers_x * args.n_fibers_y,
            "n_motor_units": args.n_motor_units,
            "type1_fraction": args.type1_fraction,
            "type2a_fraction": args.type2a_fraction,
            "mu_size_law": "target_fiber_count proportional to mu_size_basis ** recruitment_rank",
            "mu_size_basis": args.mu_size_basis,
            "theoretical_rank_max_min_ratio": float(args.mu_size_basis ** max(args.n_motor_units - 1, 0)),
            "spatial_rank_shuffle": bool(not args.no_spatial_rank_shuffle),
            "mu_type_mode": args.mu_type_mode,
            "territory_sigma_factor": args.territory_sigma_factor,
            "territory_size_scaling": args.territory_size_scaling,
            "quota_power": args.quota_power,
            "repulsion_factor": args.repulsion_factor,
            "healthy_target_fiber_counts_by_mu_id": [int(x) for x in target_sizes],
            "healthy_target_fiber_counts_by_rank": [int(x) for x in target_by_rank],
            "recruitment_rank_0_based_by_mu_id": [int(x) for x in recruitment_rank],
            "mu_ids_by_recruitment_rank_0_based": [int(x) for x in mu_ids_by_rank],
            "scenario": effective_scenario.name,
            "index_base": args.index_base,
            "fiber_order": "row-major: fiber_no = y*n_fibers_x + x",
            "n_killed_or_silent_mus": int(np.count_nonzero(killed)),
            "n_silent_unreinnervated_fibers": int(np.count_nonzero(denervated)),
            "scenario_parameters": asdict(effective_scenario),
            "canonical_runtime_files": [p.name for p in (distribution_file, firing_file, motor_units_file)],
        }
        metadata_file = stage_dir / "generation_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        tagged_metadata = copy_tagged_alias(metadata_file, sid)

        # Tagged aliases for the three core inputs.  Canonical copies are retained
        # because OpenDiHu integrations frequently assume their historical names.
        tagged_distribution = copy_tagged_alias(distribution_file, sid)
        tagged_firing = copy_tagged_alias(firing_file, sid)
        tagged_motor_units = copy_tagged_alias(motor_units_file, sid)

        case_records.append({
            "label": sid.replace("_", "\\n"),
            "mu_map": mu_map,
            "subtype_map": subtype_map,
            "csa": csa,
            "denervated": denervated,
        })

        manifest_rows.append({
            "group_number": group_number,
            "group_tag": gtag,
            "group_seed": group_seed,
            "protocol": "A",
            "stage": stage,
            "kill_fraction": kill_fraction,
            "scenario_name": sid,
            "case_directory": str(stage_dir.resolve()),
            "patch_file": str(patch_file.resolve()),
            "distribution_file": str(distribution_file.resolve()),
            "firing_file": str(firing_file.resolve()),
            "motor_units_file": str(motor_units_file.resolve()),
            "tagged_distribution_file": str(tagged_distribution.resolve()),
            "tagged_firing_file": str(tagged_firing.resolve()),
            "tagged_motor_units_file": str(tagged_motor_units.resolve()),
            "summary_file": str(tagged_summary.resolve()),
            "metadata_file": str(tagged_metadata.resolve()),
            "image_file": str(tagged_image.resolve()),
            "stage_seed": stage_seed,
            "motor_unit_seed": motor_unit_seed,
            "n_killed_mus": int(np.count_nonzero(killed)),
            "n_denervated_fibers": int(np.count_nonzero(denervated)),
            "distribution_sha256": sha256_file(distribution_file),
            "firing_sha256": sha256_file(firing_file),
        })

    overview_file = protocol_dir / f"all_cases_overview_{gtag}_A.png"
    plot_overview(case_records, str(overview_file))
    return manifest_rows


def create_protocol_b_case(group_number: int, stage: str, kill_fraction: float,
                           in_stage_dir: Path, out_stage_dir: Path,
                           group_seed: int, args):
    """Create one paired Protocol B case from its Protocol A partner."""
    sid = case_id(group_number, "B", stage)
    ensure_dir(str(out_stage_dir))

    for filename in CORE_INPUT_FILES:
        src = in_stage_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing Protocol A input file: {src}")

    # Structural/gating inputs are copied byte-for-byte from A.
    for filename in ("MU_fibre_distribution_37x37_20.txt", "MU_firing_times_always.txt"):
        shutil.copy2(in_stage_dir / filename, out_stage_dir / filename)

    # Preserve structural summary and image for auditability.
    if (in_stage_dir / "motor_unit_summary.csv").exists():
        shutil.copy2(in_stage_dir / "motor_unit_summary.csv", out_stage_dir / "structural_motor_unit_summary.csv")
    if (in_stage_dir / "mu_distribution_and_biopsy_like.png").exists():
        shutil.copy2(in_stage_dir / "mu_distribution_and_biopsy_like.png", out_stage_dir / "mu_distribution_and_biopsy_like.png")

    payload = json.loads((in_stage_dir / "motor_units.json").read_text())
    payload = copy.deepcopy(payload)
    motor_units = payload["motor_units"]

    active_flags = np.array([
        bool(mu.get("active", not mu.get("killed_or_silent", False)))
        for mu in motor_units
    ], dtype=bool)
    active_fraction = float(np.mean(active_flags))
    global_factor = compute_global_factor(
        active_fraction, args.max_global_factor, args.global_gain
    )

    summary_rows = _read_csv_rows(in_stage_dir / "motor_unit_summary.csv")
    b_rows = []

    for idx, mu in enumerate(motor_units):
        mu_no = int(mu.get("mu_no_0_based", idx))
        old_freq = float(mu.get("stimulation_frequency", 0.0))
        killed = bool(mu.get("killed_or_silent", False)) or (not bool(mu.get("active", True)))
        expansion_ratio = get_expansion_ratio_from_rows(summary_rows, mu_no)

        if killed or old_freq <= 0:
            local_factor = 0.0
            new_freq = 0.0
            mu["active"] = False
            mu["killed_or_silent"] = True
            mu["activation_start_time"] = 1e9
        else:
            local_factor = float(expansion_ratio ** (-args.expansion_alpha))
            raw_new_freq = old_freq * global_factor * local_factor
            new_freq = float(np.clip(raw_new_freq, args.min_frequency, args.max_frequency))
            mu["active"] = True
            mu["killed_or_silent"] = False

        scale = float(0.0 if old_freq == 0 else new_freq / old_freq)
        mu["baseline_stimulation_frequency"] = old_freq
        mu["protocol_B_global_factor"] = float(global_factor)
        mu["protocol_B_local_factor"] = float(local_factor)
        mu["protocol_B_expansion_ratio"] = float(expansion_ratio)
        mu["protocol_B_frequency_scale"] = scale
        mu["stimulation_frequency"] = float(new_freq)

        b_rows.append({
            "group_number": group_number,
            "group_tag": group_tag(group_number),
            "protocol": "B",
            "stage": stage,
            "scenario_name": sid,
            "kill_fraction": kill_fraction,
            "mu_no_0_based": mu_no,
            "active": bool(mu["active"]),
            "killed_or_silent": bool(mu["killed_or_silent"]),
            "type_id": mu.get("type_id", ""),
            "type_name": mu.get("type_name", ""),
            "expansion_ratio": float(expansion_ratio),
            "baseline_frequency_hz": old_freq,
            "global_factor": float(global_factor),
            "local_factor": float(local_factor),
            "new_frequency_hz": float(new_freq),
            "frequency_scale": scale,
        })

    add_case_metadata_to_motor_units(
        payload,
        group_number=group_number,
        protocol="B",
        stage=stage,
        scenario_name=sid,
        group_seed=group_seed,
    )
    payload["protocol_description"] = "Protocol B: compensated neural drive"
    payload["compensation_rule"] = {
        "global_factor": "min(max_global_factor, active_mu_fraction^(-global_gain))",
        "local_factor": "expansion_ratio^(-expansion_alpha)",
        "new_frequency": "clip(baseline_frequency * global_factor * local_factor, min_frequency, max_frequency)",
        "active_mu_fraction": active_fraction,
        "max_global_factor": args.max_global_factor,
        "global_gain": args.global_gain,
        "expansion_alpha": args.expansion_alpha,
        "max_frequency": args.max_frequency,
        "min_frequency": args.min_frequency,
    }

    motor_units_file = out_stage_dir / "motor_units.json"
    motor_units_file.write_text(json.dumps(payload, indent=2))
    tagged_motor_units = copy_tagged_alias(motor_units_file, sid)

    summary_file = out_stage_dir / "protocol_B_motor_unit_summary.csv"
    _write_csv_rows(summary_file, b_rows)
    tagged_summary = copy_tagged_alias(summary_file, sid)

    metadata = {
        "group_number": group_number,
        "group_tag": group_tag(group_number),
        "group_seed": group_seed,
        "protocol": "B",
        "stage": stage,
        "scenario_name": sid,
        "kill_fraction": kill_fraction,
        "end_time_s": float(args.end_time_s),
        "input_protocol_A_case": str(in_stage_dir.resolve()),
        "active_mu_fraction": active_fraction,
        "global_factor": global_factor,
        "max_global_factor": args.max_global_factor,
        "global_gain": args.global_gain,
        "expansion_alpha": args.expansion_alpha,
        "max_frequency": args.max_frequency,
        "min_frequency": args.min_frequency,
        "paired_design_note": "Distribution and firing gate are byte-identical to Protocol A; only surviving-MU stimulation_frequency is changed.",
    }
    metadata_file = out_stage_dir / "protocol_B_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
    tagged_metadata = copy_tagged_alias(metadata_file, sid)

    patch_file = write_group_patch(out_stage_dir, sid, args.end_time_s)

    distribution_file = out_stage_dir / "MU_fibre_distribution_37x37_20.txt"
    firing_file = out_stage_dir / "MU_firing_times_always.txt"
    tagged_distribution = copy_tagged_alias(distribution_file, sid)
    tagged_firing = copy_tagged_alias(firing_file, sid)

    # Verify pairing: A and B must have identical structure/gating files.
    a_dist_hash = sha256_file(in_stage_dir / "MU_fibre_distribution_37x37_20.txt")
    b_dist_hash = sha256_file(distribution_file)
    a_fire_hash = sha256_file(in_stage_dir / "MU_firing_times_always.txt")
    b_fire_hash = sha256_file(firing_file)
    if a_dist_hash != b_dist_hash or a_fire_hash != b_fire_hash:
        raise RuntimeError(f"A/B pairing verification failed for {sid}.")

    return {
        "group_number": group_number,
        "group_tag": group_tag(group_number),
        "group_seed": group_seed,
        "protocol": "B",
        "stage": stage,
        "kill_fraction": kill_fraction,
        "scenario_name": sid,
        "case_directory": str(out_stage_dir.resolve()),
        "patch_file": str(patch_file.resolve()),
        "distribution_file": str(distribution_file.resolve()),
        "firing_file": str(firing_file.resolve()),
        "motor_units_file": str(motor_units_file.resolve()),
        "tagged_distribution_file": str(tagged_distribution.resolve()),
        "tagged_firing_file": str(tagged_firing.resolve()),
        "tagged_motor_units_file": str(tagged_motor_units.resolve()),
        "summary_file": str(tagged_summary.resolve()),
        "metadata_file": str(tagged_metadata.resolve()),
        "image_file": str((out_stage_dir / "mu_distribution_and_biopsy_like.png").resolve()) if (out_stage_dir / "mu_distribution_and_biopsy_like.png").exists() else "",
        "stage_seed": "paired_with_A",
        "motor_unit_seed": "paired_with_A",
        "n_killed_mus": int(np.sum(~active_flags)),
        "n_denervated_fibers": "see_paired_A_case",
        "distribution_sha256": b_dist_hash,
        "firing_sha256": b_fire_hash,
        "active_mu_fraction": active_fraction,
        "protocol_B_global_factor": global_factor,
    }, b_rows


def plot_protocol_b_frequency_summary(all_rows, out_file: Path, group_number: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    for stage, _ in GROUP_STAGES:
        sub = [r for r in all_rows if r["stage"] == stage]
        if not sub:
            continue
        x = [int(r["mu_no_0_based"]) for r in sub]
        y = [float(r["new_frequency_hz"]) for r in sub]
        ax.plot(x, y, marker="o", linewidth=1.2, label=stage)
    ax.set_xlabel("MU index")
    ax.set_ylabel("Protocol B stimulation frequency [Hz]")
    ax.set_title(f"{group_tag(group_number)}: compensated firing rates")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels, means, stds = [], [], []
    for stage, _ in GROUP_STAGES:
        vals = [
            float(r["new_frequency_hz"])
            for r in all_rows
            if r["stage"] == stage and bool(r["active"])
        ]
        labels.append(stage)
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    xpos = np.arange(len(labels))
    ax.bar(xpos, means, yerr=stds, capsize=4)
    ax.set_xticks(xpos, labels, rotation=20)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Mean ± SD among active MUs")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def write_latest_submit_helper(slurm_dir: Path, generated_groups):
    """Create a convenience script that submits all groups generated in this run."""
    slurm_dir = Path(slurm_dir).expanduser()
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for g in generated_groups:
        helper = slurm_dir / f"submit_{group_tag(g)}_all.sh"
        lines.append(f"bash {shlex.quote(str(helper.resolve()))}")
    out = slurm_dir / "submit_latest_generated.sh"
    out.write_text("\n".join(lines) + "\n")
    out.chmod(out.stat().st_mode | 0o111)
    return out


def update_study_manifest(root_dir: Path, new_rows, generated_group_numbers):
    """Append/replace generated groups in the root study manifest."""
    manifest = root_dir / "study_manifest.csv"
    old_rows = _read_csv_rows(manifest)
    generated_group_numbers = {str(int(x)) for x in generated_group_numbers}
    kept = [r for r in old_rows if str(r.get("group_number", "")) not in generated_group_numbers]
    combined = kept + list(new_rows)
    combined.sort(key=lambda r: (
        int(r["group_number"]),
        0 if r["protocol"] == "A" else 1,
        [s for s, _ in GROUP_STAGES].index(r["stage"]),
    ))
    _write_csv_rows(manifest, combined)
    return manifest



DEFAULT_HPC_MODULES = [
    "openmpi/4.1.5_gcc_9.5.0",
    "eigen3/3.4.0",
    "boost/1.82.0",
    "gcc/9.5.0",
]


def _shell_assignment(value: str) -> str:
    """Return a safe shell literal for a simple variable assignment."""
    return shlex.quote(str(value))


def _runtime_path_text(path_text: str) -> str:
    """Expand ~ but otherwise preserve user path semantics."""
    return os.path.expanduser(str(path_text))


def _format_cli_number(value) -> str:
    """Prefer integer-looking CLI values when a float is mathematically integral."""
    v = float(value)
    if v.is_integer():
        return str(int(v))
    return f"{v:g}"


def make_slurm_script_text(row: dict, copied_patch: Path, args) -> str:
    """Create one submission-ready Slurm script for an OpenDiHu case."""
    sid = str(row["scenario_name"])
    case_dir = str(row["case_directory"])
    modules = args.module if args.module else list(DEFAULT_HPC_MODULES)

    if args.run_dir:
        cd_block = f'cd {_shell_assignment(_runtime_path_text(args.run_dir))}'
    else:
        cd_block = 'cd "${SLURM_SUBMIT_DIR}"'

    module_lines = "\n".join(f"module load {shlex.quote(m)}" for m in modules)
    launcher = str(args.mpi_launcher)
    if launcher == "mpirun":
        launch_line = 'mpirun -n "${SLURM_NTASKS}" \\\n'
    else:
        launch_line = 'srun -n "${SLURM_NTASKS}" \\\n'

    executable = _runtime_path_text(args.executable)
    settings = _runtime_path_text(args.settings_script)
    patch_path = str(copied_patch.resolve())
    end_time_ms = float(args.end_time_s) * 1000.0

    return f"""#!/bin/bash -l
#SBATCH --job-name={sid}
#SBATCH --account={args.slurm_account}
#SBATCH --partition={args.slurm_partition}
#SBATCH --nodes={args.slurm_nodes}
#SBATCH --ntasks={args.slurm_ntasks}
#SBATCH --ntasks-per-node={args.slurm_ntasks_per_node}
#SBATCH --cpus-per-task={args.slurm_cpus_per_task}
#SBATCH --mem={args.slurm_mem}
#SBATCH --time={args.slurm_time}
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

{cd_block}

# Initialize Conda robustly on login and batch shells.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "${{HOME}}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${{HOME}}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${{HOME}}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${{HOME}}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script was not found." >&2
    exit 1
fi

conda activate {_shell_assignment(args.conda_env)}
{module_lines}

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export OPENDIHU_GENERATED_MU_DIR={_shell_assignment(case_dir)}

EXECUTABLE={_shell_assignment(executable)}
SETTINGS_SCRIPT={_shell_assignment(settings)}
VARIABLES_SCRIPT={_shell_assignment(patch_path)}

echo "Scenario:     {sid}"
echo "Job ID:       ${{SLURM_JOB_ID}}"
echo "Node list:    ${{SLURM_JOB_NODELIST}}"
echo "MPI ranks:    ${{SLURM_NTASKS}}"
echo "Working dir:  $(pwd)"
echo "Conda env:    ${{CONDA_DEFAULT_ENV}}"
echo "Case dir:     ${{OPENDIHU_GENERATED_MU_DIR}}"
echo "Executable:   ${{EXECUTABLE}}"
echo "Settings:     ${{SETTINGS_SCRIPT}}"
echo "Variables:    ${{VARIABLES_SCRIPT}}"

if [[ ! -x "${{EXECUTABLE}}" ]]; then
    echo "Error: OpenDiHu executable is missing or not executable: ${{EXECUTABLE}}" >&2
    exit 2
fi
if [[ ! -f "${{SETTINGS_SCRIPT}}" ]]; then
    echo "Error: settings script not found: ${{SETTINGS_SCRIPT}}" >&2
    exit 2
fi
if [[ ! -f "${{VARIABLES_SCRIPT}}" ]]; then
    echo "Error: variables patch not found: ${{VARIABLES_SCRIPT}}" >&2
    exit 2
fi

{launch_line}    "${{EXECUTABLE}}" \\
    "${{SETTINGS_SCRIPT}}" \\
    "${{VARIABLES_SCRIPT}}" \\
    --tend {_format_cli_number(end_time_ms)} \\
    --output_timestep {_format_cli_number(args.output_timestep)} \\
    --output_timestep_fibers {_format_cli_number(args.output_timestep_fibers)} \\
    --output_timestep_3D_emg {_format_cli_number(args.output_timestep_3d_emg)} \\
    --output_timestep_3D_electrodes {_format_cli_number(args.output_timestep_3d_electrodes)} \\
    --output_timestep_3D_surface {_format_cli_number(args.output_timestep_3d_surface)}
"""


def create_runtime_and_slurm_artifacts(group_number: int, group_rows, args):
    """Copy case-specific variable scripts and create one Slurm job per case."""
    patch_dir = Path(args.runtime_script_dir).expanduser()
    slurm_dir = Path(args.slurm_script_dir).expanduser()
    patch_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    gtag = group_tag(group_number)

    for row in group_rows:
        sid = str(row["scenario_name"])
        source_patch = Path(row["patch_file"])
        if not source_patch.exists():
            raise FileNotFoundError(f"Generated OpenDiHu patch does not exist: {source_patch}")

        copied_patch = patch_dir / f"opendihu_variables_patch_{sid}.py"
        shutil.copy2(source_patch, copied_patch)

        slurm_file = slurm_dir / f"slurm_{sid}.sh"
        slurm_file.write_text(make_slurm_script_text(row, copied_patch, args))
        slurm_file.chmod(slurm_file.stat().st_mode | 0o111)

        row["copied_patch_file"] = str(copied_patch.resolve())
        row["slurm_script_file"] = str(slurm_file.resolve())
        row["slurm_job_name"] = sid

        submit_lines.append(f"sbatch {shlex.quote(str(slurm_file.resolve()))}")

    submit_file = slurm_dir / f"submit_{gtag}_all.sh"
    submit_file.write_text("\n".join(submit_lines) + "\n")
    submit_file.chmod(submit_file.stat().st_mode | 0o111)
    return submit_file


def write_slurm_manifest(slurm_dir: Path, group_number: int, group_rows):
    """Write a compact job manifest next to the generated Slurm scripts."""
    rows = []
    for r in group_rows:
        rows.append({
            "group_number": r["group_number"],
            "group_tag": r["group_tag"],
            "protocol": r["protocol"],
            "stage": r["stage"],
            "scenario_name": r["scenario_name"],
            "case_directory": r["case_directory"],
            "copied_patch_file": r.get("copied_patch_file", ""),
            "slurm_script_file": r.get("slurm_script_file", ""),
            "slurm_job_name": r.get("slurm_job_name", ""),
        })
    out = slurm_dir / f"slurm_manifest_{group_tag(group_number)}.csv"
    _write_csv_rows(out, rows)
    return out


def write_group_readme(group_dir: Path, group_number: int, group_rows, args):
    gtag = group_tag(group_number)
    lines = [
        f"Repeated denervation study group: {gtag}",
        "",
        "Design:",
        "  8 paired cases per group = 4 Protocol A + 4 Protocol B",
        "  stages = healthy, death_25, death_50, death_75",
        "",
        "Pairing rule:",
        "  Protocol B copies the MU distribution and firing gate byte-for-byte",
        "  from the corresponding Protocol A stage. Only surviving-MU",
        "  stimulation_frequency is changed according to the compensated-drive rule.",
        "",
        f"Simulation duration in generated OpenDiHu patches: {args.end_time_s:g} s",
        f"Group seed: {args.base_seed + (group_number - 1) * args.group_seed_stride}",
        f"MU-size exponential basis: {args.mu_size_basis:g}",
        f"Copied runtime patches: {Path(args.runtime_script_dir).expanduser().resolve()}",
        f"Generated Slurm scripts: {Path(args.slurm_script_dir).expanduser().resolve()}",
        f"Slurm submit helper: {Path(args.slurm_script_dir).expanduser().resolve() / ('submit_' + gtag + '_all.sh')}",
        "",
        "Naming convention:",
        f"  {gtag}_A_healthy",
        f"  {gtag}_A_death_25",
        f"  {gtag}_A_death_50",
        f"  {gtag}_A_death_75",
        f"  {gtag}_B_healthy",
        f"  {gtag}_B_death_25",
        f"  {gtag}_B_death_50",
        f"  {gtag}_B_death_75",
        "",
        "Important filename policy:",
        "  Canonical OpenDiHu runtime filenames are retained inside every case folder",
        "  (MU_fibre_distribution_37x37_20.txt, MU_firing_times_always.txt,",
        "  motor_units.json) to avoid breaking existing OpenDiHu code.",
        "  Group/case-tagged aliases are created beside them for traceability.",
        "",
        "Case table:",
    ]
    for r in group_rows:
        lines.append(f"  {r['scenario_name']}: {r['case_directory']}")
    lines += [
        "",
        "Run a case by setting OPENDIHU_GENERATED_MU_DIR to its case directory",
        "and using the tagged opendihu_variables_patch_<scenario_name>.py file.",
        "",
        "For statistical analysis, use group_manifest_<g-tag>.csv or the root",
        "study_manifest.csv. Group number is the replicate/block identifier.",
    ]
    (group_dir / f"README_{gtag}.txt").write_text("\n".join(lines) + "\n")


def generate_one_group(group_number: int, root_dir: Path, args):
    group_number = sanitize_group_number(group_number)
    gtag = group_tag(group_number)
    group_dir = root_dir / gtag

    if group_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{group_dir} already exists. Use --overwrite to replace this group."
        )
    if group_dir.exists() and args.overwrite:
        shutil.rmtree(group_dir)
    ensure_dir(str(group_dir))

    # Protocol A first: one structural realization per group, four severities.
    a_rows = create_protocol_a_group(group_number, group_dir, args)

    # Protocol B is paired stage-by-stage with A.
    protocol_b_dir = group_dir / "protocol_B"
    ensure_dir(str(protocol_b_dir))
    group_seed = int(args.base_seed + (group_number - 1) * args.group_seed_stride)
    b_rows_manifest = []
    b_mu_rows = []
    for stage, kill_fraction in GROUP_STAGES:
        row, mu_rows = create_protocol_b_case(
            group_number=group_number,
            stage=stage,
            kill_fraction=kill_fraction,
            in_stage_dir=group_dir / "protocol_A" / stage,
            out_stage_dir=protocol_b_dir / stage,
            group_seed=group_seed,
            args=args,
        )
        b_rows_manifest.append(row)
        b_mu_rows.extend(mu_rows)

    b_overview = protocol_b_dir / f"protocol_B_compensation_overview_{gtag}.csv"
    _write_csv_rows(b_overview, b_mu_rows)
    plot_protocol_b_frequency_summary(
        b_mu_rows,
        protocol_b_dir / f"protocol_B_frequency_summary_{gtag}.png",
        group_number,
    )

    group_rows = a_rows + b_rows_manifest

    submit_file = create_runtime_and_slurm_artifacts(group_number, group_rows, args)
    slurm_manifest = write_slurm_manifest(Path(args.slurm_script_dir).expanduser(), group_number, group_rows)

    group_manifest = group_dir / f"group_manifest_{gtag}.csv"
    _write_csv_rows(group_manifest, group_rows)

    group_metadata = {
        "format": "opendihu_repeated_denervation_group_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "group_number": group_number,
        "group_tag": gtag,
        "group_seed": group_seed,
        "n_cases": 8,
        "protocols": ["A", "B"],
        "stages": [s for s, _ in GROUP_STAGES],
        "end_time_s": float(args.end_time_s),
        "scenario": args.scenario,
        "mu_size_basis": float(args.mu_size_basis),
        "protocol_B_parameters": {
            "max_global_factor": args.max_global_factor,
            "global_gain": args.global_gain,
            "expansion_alpha": args.expansion_alpha,
            "max_frequency": args.max_frequency,
            "min_frequency": args.min_frequency,
        },
        "pairing_verified_by_sha256": True,
        "runtime_script_directory": str(Path(args.runtime_script_dir).expanduser().resolve()),
        "slurm_script_directory": str(Path(args.slurm_script_dir).expanduser().resolve()),
        "submit_all_script": str(submit_file.resolve()),
        "slurm_manifest": str(slurm_manifest.resolve()),
        "slurm_resources": {
            "account": args.slurm_account,
            "partition": args.slurm_partition,
            "nodes": args.slurm_nodes,
            "ntasks": args.slurm_ntasks,
            "ntasks_per_node": args.slurm_ntasks_per_node,
            "cpus_per_task": args.slurm_cpus_per_task,
            "mem": args.slurm_mem,
            "time": args.slurm_time,
        },
        "manifest": group_manifest.name,
    }
    (group_dir / f"group_metadata_{gtag}.json").write_text(
        json.dumps(group_metadata, indent=2)
    )
    write_group_readme(group_dir, group_number, group_rows, args)
    return group_rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate repeated, group-tagged OpenDiHu denervation study cases. "
            "Each group contains 8 paired cases: Protocol A/B × healthy/25/50/75% MU loss."
        )
    )

    # Repetition / naming.
    parser.add_argument("--out-dir", default="opendihu_repeated_denervation_study",
                        help="Root output directory containing g-1, g-2, ...")
    parser.add_argument("--group", type=int, default=1,
                        help="First group number to generate (default: 1).")
    parser.add_argument("--n-groups", type=int, default=1,
                        help="Number of consecutive groups to generate (default: 1).")
    parser.add_argument("--base-seed", type=int, default=1,
                        help="Base seed used for g-1.")
    parser.add_argument("--group-seed-stride", type=int, default=100000,
                        help="Seed offset between groups, ensuring independent realizations.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing generated group directories.")

    # Required deployment destinations.
    parser.add_argument("--runtime-script-dir", required=True,
                        help="Required directory receiving copied case-specific OpenDiHu variables patches.")
    parser.add_argument("--slurm-script-dir", required=True,
                        help="Required directory receiving generated Slurm scripts and submission helpers.")

    # OpenDiHu/HPC runtime configuration.
    parser.add_argument("--run-dir", default=None,
                        help="Job working directory. Default: SLURM_SUBMIT_DIR.")
    parser.add_argument("--executable", default="./fibers_fat_emg",
                        help="OpenDiHu executable path relative to the job working directory unless absolute.")
    parser.add_argument("--settings-script", default="../settings_fibers_fat_emg2.py",
                        help="OpenDiHu settings script path relative to the job working directory unless absolute.")
    parser.add_argument("--conda-env", default="opendihu")
    parser.add_argument("--module", action="append", default=None,
                        help="Module to load. Repeat for multiple modules; defaults match the supplied HPC example.")
    parser.add_argument("--mpi-launcher", choices=["mpirun", "srun"], default="mpirun")

    parser.add_argument("--slurm-account", default="general")
    parser.add_argument("--slurm-partition", default="general")
    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--slurm-ntasks", type=int, default=8)
    parser.add_argument("--slurm-ntasks-per-node", type=int, default=8)
    parser.add_argument("--slurm-cpus-per-task", type=int, default=1)
    parser.add_argument("--slurm-mem", default="320G")
    parser.add_argument("--slurm-time", default="48:00:00")

    # OpenDiHu output controls from the supplied Slurm example.
    parser.add_argument("--output-timestep", type=float, default=10.0)
    parser.add_argument("--output-timestep-fibers", type=float, default=1000.0)
    parser.add_argument("--output-timestep-3d-emg", type=float, default=1000.0)
    parser.add_argument("--output-timestep-3d-electrodes", type=float, default=10000000.0)
    parser.add_argument("--output-timestep-3d-surface", type=float, default=1000.0)

    # Study timing.
    parser.add_argument("--end-time-s", type=float, default=30.0,
                        help="Simulation duration written to OpenDiHu patches [s].")
    parser.add_argument("--n-firing-rows", type=int, default=None,
                        help="Rows in always-on firing gate. Default: ceil(end_time_s*100 Hz).")

    # Structural/MU model from Protocol A generator.
    parser.add_argument("--n-fibers-x", type=int, default=37)
    parser.add_argument("--n-fibers-y", type=int, default=37)
    parser.add_argument("--n-motor-units", type=int, default=20)
    parser.add_argument("--scenario",
                        choices=["aging_type2", "exercise_rescued", "als_like_failed_rescue", "classic_like"],
                        default="aging_type2")
    parser.add_argument("--index-base", type=int, choices=[0, 1], default=1)
    parser.add_argument("--physiology-recruitment", action="store_true")
    parser.add_argument("--type1-fraction", type=float, default=0.50)
    parser.add_argument("--type2a-fraction", type=float, default=0.35)
    parser.add_argument("--territory-sigma-factor", type=float, default=0.35)
    parser.add_argument("--repulsion-factor", type=float, default=0.25)
    parser.add_argument("--territory-size-scaling", type=float, default=0.50)
    parser.add_argument("--quota-power", type=float, default=1.0)
    parser.add_argument("--mu-size-basis", type=float, default=1.20,
                        help="OpenDiHu-style exponential MU-size basis.")
    parser.add_argument("--no-spatial-rank-shuffle", action="store_true")
    parser.add_argument("--mu-type-mode", choices=["independent", "rank_linked"],
                        default="independent")

    # Protocol B compensated-drive parameters, preserved from the supplied script.
    parser.add_argument("--max-global-factor", type=float, default=1.8)
    parser.add_argument("--global-gain", type=float, default=1.0)
    parser.add_argument("--expansion-alpha", type=float, default=0.5)
    parser.add_argument("--max-frequency", type=float, default=35.0)
    parser.add_argument("--min-frequency", type=float, default=1.0)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.n_groups < 1:
        parser.error("--n-groups must be >= 1")
    if args.end_time_s <= 0:
        parser.error("--end-time-s must be > 0")
    if args.group_seed_stride <= 0:
        parser.error("--group-seed-stride must be > 0")
    if args.n_firing_rows is None:
        # The OpenDiHu patch uses 100 Hz firing-table sampling.  For the current
        # always-on gate, this mainly makes the file duration explicit/auditable.
        args.n_firing_rows = int(math.ceil(args.end_time_s * 100.0))
    if args.n_firing_rows < 1:
        parser.error("--n-firing-rows must be >= 1")
    if args.slurm_nodes < 1 or args.slurm_ntasks < 1 or args.slurm_ntasks_per_node < 1 or args.slurm_cpus_per_task < 1:
        parser.error("Slurm node/task/CPU counts must all be >= 1")
    if args.slurm_ntasks_per_node > args.slurm_ntasks:
        parser.error("--slurm-ntasks-per-node cannot exceed --slurm-ntasks")
    if args.slurm_ntasks > args.slurm_nodes * args.slurm_ntasks_per_node:
        parser.error("--slurm-ntasks exceeds nodes * ntasks-per-node")
    for name in ("output_timestep", "output_timestep_fibers", "output_timestep_3d_emg",
                 "output_timestep_3d_electrodes", "output_timestep_3d_surface"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")

    root_dir = Path(args.out_dir)
    ensure_dir(str(root_dir))

    all_new_rows = []
    generated_groups = []
    for group_number in range(args.group, args.group + args.n_groups):
        rows = generate_one_group(group_number, root_dir, args)
        all_new_rows.extend(rows)
        generated_groups.append(group_number)
        print(f"Created {group_tag(group_number)} with {len(rows)} cases")

    study_manifest = update_study_manifest(
        root_dir, all_new_rows, generated_groups
    )
    latest_submit = write_latest_submit_helper(
        Path(args.slurm_script_dir), generated_groups
    )

    study_metadata = {
        "format": "opendihu_repeated_denervation_study_v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "root_directory": str(root_dir.resolve()),
        "latest_generated_groups": generated_groups,
        "cases_per_group": 8,
        "design": "group/replicate × protocol(A,B) × denervation stage(healthy,25,50,75%)",
        "paired_protocol_design": True,
        "simulation_duration_s": float(args.end_time_s),
        "runtime_script_directory": str(Path(args.runtime_script_dir).expanduser().resolve()),
        "slurm_script_directory": str(Path(args.slurm_script_dir).expanduser().resolve()),
        "latest_submit_helper": str(latest_submit.resolve()),
        "study_manifest": study_manifest.name,
    }
    (root_dir / "study_metadata.json").write_text(json.dumps(study_metadata, indent=2))

    print("\nGeneration complete")
    print(f"  Root directory : {root_dir.resolve()}")
    print(f"  Groups         : {', '.join(group_tag(g) for g in generated_groups)}")
    print(f"  Cases/group    : 8")
    print(f"  Study manifest : {study_manifest.resolve()}")
    print(f"  Runtime scripts: {Path(args.runtime_script_dir).expanduser().resolve()}")
    print(f"  Slurm scripts  : {Path(args.slurm_script_dir).expanduser().resolve()}")
    print(f"  Submit latest  : {latest_submit.resolve()}")
    print(f"  End time       : {args.end_time_s:g} s")


if __name__ == "__main__":
    main()
