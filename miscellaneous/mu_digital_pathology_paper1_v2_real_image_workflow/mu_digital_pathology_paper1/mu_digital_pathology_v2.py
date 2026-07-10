#!/usr/bin/env python3
"""
Motor-unit digital pathology model, upgraded to a three-phenotype muscle-fiber system.

Main upgrade relative to v1
---------------------------
- Fiber phenotype is no longer binary (type I vs type II). Instead it is modeled as
  a continuous variable with three canonical anchors:
      type I   -> 0.00
      type IIa -> 0.60
      type IIx -> 1.00
- Motor-neuron phenotype is also three-state:
      0 = slow-biased (type I-like)
      1 = fast-IIa-biased
      2 = fast-IIx-biased
- Reinnervation can therefore drive incomplete and subtype-specific phenotype conversion.
- Feature extraction now reports type-I / IIa / IIx fractions and a hybrid/intermediate fraction.

This code is still a research prototype for hypothesis testing, not a clinically calibrated model.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Constants and helpers
# -----------------------------------------------------------------------------

FOUR_NB = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

FIBER_PHENO_LABELS = {
    0: "type1",
    1: "type2a",
    2: "type2x",
}

PHENO_TARGET = {
    0: 0.00,   # type I
    1: 0.60,   # type IIa
    2: 1.00,   # type IIx
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def perfect_square_root(n: int) -> int:
    r = int(round(math.sqrt(n)))
    if r * r != n:
        raise ValueError(f"n_mu={n} must be a perfect square.")
    return r


def logistic_radius_probability(distance: np.ndarray, radius: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(slope * (distance - radius)))


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def fiber_type_class_from_value(x: np.ndarray) -> np.ndarray:
    """Map continuous phenotype in [0,1] to {0:type I,1:type IIa,2:type IIx}."""
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=int)
    out[(x >= 0.30) & (x < 0.80)] = 1
    out[x >= 0.80] = 2
    return out


def hybrid_mask_from_value(x: np.ndarray, tol: float = 0.12) -> np.ndarray:
    """Hybrid/intermediate fibers: not sufficiently close to canonical subtype anchors."""
    x = np.asarray(x)
    d0 = np.abs(x - PHENO_TARGET[0])
    d1 = np.abs(x - PHENO_TARGET[1])
    d2 = np.abs(x - PHENO_TARGET[2])
    dmin = np.minimum(np.minimum(d0, d1), d2)
    return dmin > tol


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class RemodelParams:
    grid_n: int = 80
    n_mu: int = 49
    init_radius: float = 18.0
    init_slope: float = 0.10
    repulsion: float = 0.20
    seed: int = 1

    # Initial motor-neuron phenotype fractions.
    mn_slow_fraction: float = 0.45
    mn_2a_fraction: float = 0.35
    mn_2x_fraction: float = 0.20

    # Whole MU death weighting by MN phenotype.
    p_mu_death_slow: float = 0.40
    p_mu_death_2a: float = 0.50
    p_mu_death_2x: float = 0.60

    # Partial NMJ dropout probabilities by owner MN phenotype.
    p_nmj_dropout_slow: float = 0.001
    p_nmj_dropout_2a: float = 0.004
    p_nmj_dropout_2x: float = 0.007

    # Reinnervation competence by MN phenotype.
    p_reinn_slow: float = 0.75
    p_reinn_2a: float = 0.60
    p_reinn_2x: float = 0.45

    # Spatial sprouting.
    sprout_radius: float = 5.0
    sprout_sigma: float = 2.5
    complete_reinnervation: bool = False
    max_reinnervation_passes: int = 8

    # Capacity and overload.
    capacity_multiplier: float = 3.0
    overload_hypertrophy_gain: float = 0.15
    max_csa: float = 2.50

    # Dynamics.
    atrophy_rate: float = 0.10
    recovery_rate: float = 0.04
    type_conversion_rate: float = 0.04

    # Compatibility / mismatch.
    mismatch_penalty: float = 0.55
    compatibility_sigma: float = 0.30

    death_targets: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)


@dataclass
class MuscleState:
    mu_id: np.ndarray
    fiber_type: np.ndarray             # continuous phenotype in [0,1]
    csa: np.ndarray
    denervated_age: np.ndarray
    mu_type: np.ndarray                # 0 slow, 1 IIa, 2 IIx
    mu_alive: np.ndarray
    mu_centers: np.ndarray
    mu_original_size: np.ndarray
    killed_order: List[int]


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------


def _mn_type_probabilities(params: RemodelParams) -> np.ndarray:
    p = np.array([params.mn_slow_fraction, params.mn_2a_fraction, params.mn_2x_fraction], dtype=float)
    p = np.maximum(p, 0.0)
    if p.sum() <= 0:
        p[:] = 1.0
    return p / p.sum()



def initialize_muscle(params: RemodelParams) -> MuscleState:
    rng = np.random.default_rng(params.seed)
    n = params.grid_n
    n_mu = params.n_mu
    side = perfect_square_root(n_mu)

    xs = np.linspace(0.5 * n / side, n - 0.5 * n / side, side)
    ys = np.linspace(0.5 * n / side, n - 0.5 * n / side, side)
    centers = np.array([(x, y) for y in ys for x in xs], dtype=float)
    centers += rng.normal(0.0, 0.7, size=centers.shape)

    mu_type = rng.choice(3, size=n_mu, p=_mn_type_probabilities(params))

    mu_id = -np.ones((n, n), dtype=int)
    coords = [(i, j) for i in range(n) for j in range(n)]
    rng.shuffle(coords)

    for i, j in coords:
        d = np.sqrt((centers[:, 0] - j) ** 2 + (centers[:, 1] - i) ** 2)
        p = logistic_radius_probability(d, params.init_radius, params.init_slope)
        neighbor_mus = []
        for di, dj in FOUR_NB:
            ii, jj = i + di, j + dj
            if 0 <= ii < n and 0 <= jj < n and mu_id[ii, jj] >= 0:
                neighbor_mus.append(mu_id[ii, jj])
        if neighbor_mus:
            for k in set(neighbor_mus):
                p[k] *= params.repulsion
        p[p < 0.01 * p.max()] = 0.0
        if p.sum() <= 0:
            k = int(np.argmin(d))
        else:
            p = p / p.sum()
            k = int(rng.choice(n_mu, p=p))
        mu_id[i, j] = k

    fiber_type = np.zeros((n, n), dtype=float)
    for k in range(n_mu):
        mask = mu_id == k
        target = PHENO_TARGET[int(mu_type[k])]
        # Initialize around the owner-MN phenotype, but not perfectly homogeneous.
        fiber_type[mask] = np.clip(rng.normal(target, 0.06 if mu_type[k] != 0 else 0.05, size=np.count_nonzero(mask)), 0.0, 1.0)

    csa = np.ones((n, n), dtype=float)
    # Slight initial subtype-dependent CSA differences.
    cls = fiber_type_class_from_value(fiber_type)
    csa[cls == 0] *= 0.95
    csa[cls == 1] *= 1.00
    csa[cls == 2] *= 1.05
    csa += rng.normal(0.0, 0.02, size=csa.shape)
    csa = np.clip(csa, 0.7, 1.3)

    denervated_age = np.zeros((n, n), dtype=int)
    mu_alive = np.ones(n_mu, dtype=bool)
    mu_original_size = np.bincount(mu_id.ravel(), minlength=n_mu).astype(float)

    return MuscleState(
        mu_id=mu_id,
        fiber_type=fiber_type,
        csa=csa,
        denervated_age=denervated_age,
        mu_type=mu_type,
        mu_alive=mu_alive,
        mu_centers=centers,
        mu_original_size=np.maximum(mu_original_size, 1.0),
        killed_order=[],
    )


# -----------------------------------------------------------------------------
# Remodeling dynamics
# -----------------------------------------------------------------------------


def _death_weight_for_mu_type(mu_type_val: int, params: RemodelParams) -> float:
    if mu_type_val == 0:
        return params.p_mu_death_slow
    if mu_type_val == 1:
        return params.p_mu_death_2a
    return params.p_mu_death_2x



def _dropout_prob_for_mu_type(mu_type_val: int, params: RemodelParams) -> float:
    if mu_type_val == 0:
        return params.p_nmj_dropout_slow
    if mu_type_val == 1:
        return params.p_nmj_dropout_2a
    return params.p_nmj_dropout_2x



def _reinn_prob_for_mu_type(mu_type_val: int, params: RemodelParams) -> float:
    if mu_type_val == 0:
        return params.p_reinn_slow
    if mu_type_val == 1:
        return params.p_reinn_2a
    return params.p_reinn_2x



def copy_state(state: MuscleState) -> MuscleState:
    return MuscleState(
        mu_id=state.mu_id.copy(),
        fiber_type=state.fiber_type.copy(),
        csa=state.csa.copy(),
        denervated_age=state.denervated_age.copy(),
        mu_type=state.mu_type.copy(),
        mu_alive=state.mu_alive.copy(),
        mu_centers=state.mu_centers.copy(),
        mu_original_size=state.mu_original_size.copy(),
        killed_order=list(state.killed_order),
    )



def select_mu_to_kill(state: MuscleState, params: RemodelParams, rng: np.random.Generator) -> Optional[int]:
    alive = np.where(state.mu_alive)[0]
    if alive.size == 0:
        return None
    weights = np.array([_death_weight_for_mu_type(int(state.mu_type[k]), params) for k in alive], dtype=float)
    counts = np.bincount(state.mu_id[state.mu_id >= 0].ravel(), minlength=len(state.mu_alive))[alive]
    weights *= np.maximum(counts, 1) ** 0.25
    if weights.sum() <= 0:
        return int(rng.choice(alive))
    return int(rng.choice(alive, p=weights / weights.sum()))



def kill_motor_unit(state: MuscleState, k: int) -> None:
    if k is None or not state.mu_alive[k]:
        return
    mask = state.mu_id == k
    state.mu_id[mask] = -1
    state.denervated_age[mask] = 1
    state.mu_alive[k] = False
    state.killed_order.append(int(k))



def partial_nmj_dropout(state: MuscleState, params: RemodelParams, rng: np.random.Generator) -> None:
    inn = state.mu_id >= 0
    owner = np.where(inn, state.mu_id, 0)
    owner_type = state.mu_type[owner]
    p = np.vectorize(lambda t: _dropout_prob_for_mu_type(int(t), params))(owner_type)
    drop = inn & (rng.random(state.mu_id.shape) < p)
    state.mu_id[drop] = -1
    state.denervated_age[drop] = 1



def candidate_mu_weights_for_fiber(i: int, j: int, state: MuscleState, params: RemodelParams, live_counts: Optional[np.ndarray] = None) -> Dict[int, float]:
    n = state.mu_id.shape[0]
    r = int(math.ceil(params.sprout_radius))
    weights: Dict[int, float] = {}
    fiber_value = float(state.fiber_type[i, j])

    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if not (0 <= ii < n and 0 <= jj < n):
                continue
            k = int(state.mu_id[ii, jj])
            if k < 0 or not state.mu_alive[k]:
                continue
            d = math.sqrt(di * di + dj * dj)
            if d > params.sprout_radius:
                continue

            w = math.exp(-(d * d) / (2.0 * params.sprout_sigma * params.sprout_sigma))

            mn_type = int(state.mu_type[k])
            w *= _reinn_prob_for_mu_type(mn_type, params)

            target = PHENO_TARGET[mn_type]
            mismatch = abs(fiber_value - target)
            compat = math.exp(-(mismatch ** 2) / (2.0 * params.compatibility_sigma ** 2))
            w *= params.mismatch_penalty + (1.0 - params.mismatch_penalty) * compat

            if live_counts is None:
                current_count = np.count_nonzero(state.mu_id == k)
            else:
                current_count = live_counts[k]
            capacity = params.capacity_multiplier * state.mu_original_size[k]
            capacity_factor = max(0.03, 1.0 - max(0.0, current_count - state.mu_original_size[k]) / max(capacity, 1.0))
            w *= capacity_factor

            weights[k] = weights.get(k, 0.0) + w
    return weights



def reinnervate_denervated_fibers(state: MuscleState, params: RemodelParams, rng: np.random.Generator) -> int:
    total_reinn = 0
    for _ in range(params.max_reinnervation_passes):
        den_coords = np.argwhere(state.mu_id < 0)
        if den_coords.size == 0:
            break
        proposals: List[Tuple[int, int, int]] = []
        live_counts = np.bincount(state.mu_id[state.mu_id >= 0].ravel(), minlength=len(state.mu_alive)).astype(float)
        for i, j in den_coords:
            weights = candidate_mu_weights_for_fiber(int(i), int(j), state, params, live_counts=live_counts)
            if not weights:
                continue
            keys = np.array(list(weights.keys()), dtype=int)
            vals = np.array([weights[k] for k in keys], dtype=float)
            if vals.sum() <= 0:
                continue
            vals /= vals.sum()
            if params.complete_reinnervation:
                rescue = True
            else:
                raw_strength = sum(weights.values())
                rescue_prob = 1.0 - math.exp(-0.28 * raw_strength)
                rescue = rng.random() < rescue_prob
            if rescue:
                k_new = int(rng.choice(keys, p=vals))
                proposals.append((int(i), int(j), k_new))
        if not proposals:
            break
        for i, j, k_new in proposals:
            if state.mu_id[i, j] < 0 and state.mu_alive[k_new]:
                state.mu_id[i, j] = k_new
                state.denervated_age[i, j] = 0
                total_reinn += 1
        if params.complete_reinnervation and np.count_nonzero(state.mu_id < 0) == 0:
            break
    return total_reinn



def update_fiber_states(state: MuscleState, params: RemodelParams) -> None:
    den = state.mu_id < 0
    inn = ~den

    state.denervated_age[den] += 1
    state.csa[den] *= (1.0 - params.atrophy_rate)
    state.csa[den] = np.maximum(state.csa[den], 0.05)

    if np.any(inn):
        counts = np.bincount(state.mu_id[inn].ravel(), minlength=len(state.mu_alive)).astype(float)
        load_ratio = counts / np.maximum(state.mu_original_size, 1.0)
        target_csa_by_mu = 1.0 + params.overload_hypertrophy_gain * np.maximum(0.0, load_ratio - 1.0)
        target_csa_by_mu = np.minimum(target_csa_by_mu, params.max_csa)
        target = target_csa_by_mu[state.mu_id[inn]]
        state.csa[inn] += params.recovery_rate * (target - state.csa[inn])
        state.csa[inn] = np.clip(state.csa[inn], 0.05, params.max_csa)

        target_type = np.array([PHENO_TARGET[int(t)] for t in state.mu_type[state.mu_id[inn]]], dtype=float)
        state.fiber_type[inn] += params.type_conversion_rate * (target_type - state.fiber_type[inn])
        state.fiber_type[inn] = np.clip(state.fiber_type[inn], 0.0, 1.0)



def remodeling_episode(state: MuscleState, params: RemodelParams, rng: np.random.Generator, kill_one_mu: bool = True) -> Dict[str, float]:
    before_den = int(np.count_nonzero(state.mu_id < 0))
    killed = None
    if kill_one_mu:
        killed = select_mu_to_kill(state, params, rng)
        if killed is not None:
            kill_motor_unit(state, killed)
    partial_nmj_dropout(state, params, rng)
    after_dropout_den = int(np.count_nonzero(state.mu_id < 0))
    n_reinn = reinnervate_denervated_fibers(state, params, rng)
    update_fiber_states(state, params)
    after_den = int(np.count_nonzero(state.mu_id < 0))
    return {
        "killed_mu": -1 if killed is None else int(killed),
        "den_before": before_den,
        "den_after_dropout": after_dropout_den,
        "reinnervated": n_reinn,
        "den_after": after_den,
    }



def run_forward(params: RemodelParams, snapshots: Optional[Tuple[float, ...]] = None) -> Tuple[MuscleState, Dict[float, MuscleState], pd.DataFrame]:
    rng = np.random.default_rng(params.seed)
    state = initialize_muscle(params)
    if snapshots is None:
        snapshots = params.death_targets
    snapshots = tuple(sorted(snapshots))
    snapshot_states: Dict[float, MuscleState] = {0.0: copy_state(state)}
    history_rows = []
    n_mu = params.n_mu
    target_idx = 1
    max_target = max(snapshots)
    max_kills = int(round(max_target * n_mu))
    episode = 0
    while len(state.killed_order) < max_kills:
        episode += 1
        info = remodeling_episode(state, params, rng, kill_one_mu=True)
        frac_dead = len(state.killed_order) / n_mu
        feat = extract_features(state)
        row = {"episode": episode, "frac_mu_dead": frac_dead, **info, **feat}
        history_rows.append(row)
        while target_idx < len(snapshots) and frac_dead >= snapshots[target_idx] - 1e-9:
            snapshot_states[snapshots[target_idx]] = copy_state(state)
            target_idx += 1
    for s in snapshots:
        if s not in snapshot_states:
            snapshot_states[s] = copy_state(state)
    hist = pd.DataFrame(history_rows)
    return state, snapshot_states, hist


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------


def subtype_map(state: MuscleState) -> np.ndarray:
    cls = fiber_type_class_from_value(state.fiber_type)
    cls[state.mu_id < 0] = -1
    return cls



def same_type_grouping_index(tmap: np.ndarray) -> float:
    valid = tmap >= 0
    vals = tmap[valid]
    if vals.size < 2:
        return 0.0
    probs = np.array([(vals == 0).mean(), (vals == 1).mean(), (vals == 2).mean()], dtype=float)
    expected_same = float(np.sum(probs ** 2))
    same = 0
    total = 0
    n = tmap.shape[0]
    for i in range(n):
        for j in range(n):
            if tmap[i, j] < 0:
                continue
            for di, dj in [(1, 0), (0, 1)]:
                ii, jj = i + di, j + dj
                if ii < n and jj < n and tmap[ii, jj] >= 0:
                    total += 1
                    same += int(tmap[i, j] == tmap[ii, jj])
    if total == 0 or expected_same >= 1.0:
        return 0.0
    observed = same / total
    return float((observed - expected_same) / (1.0 - expected_same + 1e-12))



def connected_components_sizes(mask: np.ndarray, connectivity: int = 4) -> List[int]:
    n, m = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    sizes = []
    nb = FOUR_NB if connectivity == 4 else EIGHT_NB
    for i in range(n):
        for j in range(m):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            cnt = 0
            while stack:
                ci, cj = stack.pop()
                cnt += 1
                for di, dj in nb:
                    ii, jj = ci + di, cj + dj
                    if 0 <= ii < n and 0 <= jj < m and mask[ii, jj] and not visited[ii, jj]:
                        visited[ii, jj] = True
                        stack.append((ii, jj))
            sizes.append(cnt)
    return sizes



def mu_same_neighbor_probability(state: MuscleState) -> float:
    mu = state.mu_id
    n = mu.shape[0]
    same = 0
    total = 0
    for i in range(n):
        for j in range(n):
            if mu[i, j] < 0:
                continue
            for di, dj in [(1, 0), (0, 1)]:
                ii, jj = i + di, j + dj
                if ii < n and jj < n and mu[ii, jj] >= 0:
                    total += 1
                    same += int(mu[i, j] == mu[ii, jj])
    return float(same / total) if total else 0.0



def extract_features(state: MuscleState) -> Dict[str, float]:
    tmap = subtype_map(state)
    valid = tmap >= 0
    total = tmap.size
    den_frac = 1.0 - valid.sum() / total

    type1_mask = valid & (tmap == 0)
    type2a_mask = valid & (tmap == 1)
    type2x_mask = valid & (tmap == 2)
    hybrid_mask = valid & hybrid_mask_from_value(state.fiber_type)

    clusters = []
    for mask in [type1_mask, type2a_mask, type2x_mask]:
        clusters += connected_components_sizes(mask, connectivity=4)

    csa = state.csa
    type1_csa = csa[type1_mask]
    type2a_csa = csa[type2a_mask]
    type2x_csa = csa[type2x_mask]
    fast_mask = valid & (tmap >= 1)
    fast_csa = csa[fast_mask]

    inn_mu = state.mu_id[state.mu_id >= 0]
    live_counts = np.bincount(inn_mu, minlength=len(state.mu_alive)).astype(float)
    nonzero_live_counts = live_counts[live_counts > 0]

    features = {
        "type1_fraction": float(np.mean(type1_mask[valid])) if valid.any() else 0.0,
        "type2a_fraction": float(np.mean(type2a_mask[valid])) if valid.any() else 0.0,
        "type2x_fraction": float(np.mean(type2x_mask[valid])) if valid.any() else 0.0,
        "hybrid_fraction": float(np.mean(hybrid_mask[valid])) if valid.any() else 0.0,
        "denervated_fraction": float(den_frac),
        "grouping_index": same_type_grouping_index(tmap),
        "mu_neighbor_same": mu_same_neighbor_probability(state),
        "mean_cluster_size": float(np.mean(clusters)) if clusters else 0.0,
        "largest_cluster_fraction": float(max(clusters) / max(1, valid.sum())) if clusters else 0.0,
        "type1_csa_mean": float(np.mean(type1_csa)) if type1_csa.size else 0.0,
        "type2a_csa_mean": float(np.mean(type2a_csa)) if type2a_csa.size else 0.0,
        "type2x_csa_mean": float(np.mean(type2x_csa)) if type2x_csa.size else 0.0,
        "fast_atrophy_fraction": float(np.mean(fast_csa < 0.65)) if fast_csa.size else 0.0,
        "type2x_atrophy_fraction": float(np.mean(type2x_csa < 0.65)) if type2x_csa.size else 0.0,
        "hypertrophy_fraction": float(np.mean(csa[valid] > 1.25)) if valid.any() else 0.0,
        "csa_mean": float(np.mean(csa[valid])) if valid.any() else 0.0,
        "csa_cv": float(np.std(csa[valid]) / (np.mean(csa[valid]) + 1e-12)) if valid.any() else 0.0,
        "live_mu_count": float(np.count_nonzero(nonzero_live_counts)),
        "mean_mu_size": float(np.mean(nonzero_live_counts)) if nonzero_live_counts.size else 0.0,
        "max_mu_size": float(np.max(nonzero_live_counts)) if nonzero_live_counts.size else 0.0,
        "mu_size_gini": gini(nonzero_live_counts),
    }
    features["nn_same_type_index"] = features["grouping_index"]
    features["max_cluster_size"] = float(max(clusters)) if clusters else 0.0
    features["mu_loss_fraction"] = float(1.0 - np.count_nonzero(state.mu_alive) / len(state.mu_alive))
    ratios = []
    for k in range(len(state.mu_alive)):
        if state.mu_alive[k] and state.mu_original_size[k] > 0 and live_counts[k] > 0:
            ratios.append(live_counts[k] / state.mu_original_size[k])
    ratios = np.asarray(ratios, dtype=float)
    features["mean_mu_expansion"] = float(np.mean(ratios)) if ratios.size else 0.0
    features["max_mu_expansion"] = float(np.max(ratios)) if ratios.size else 0.0

    inn = state.mu_id >= 0
    if np.any(inn):
        target_vals = np.array([PHENO_TARGET[int(t)] for t in state.mu_type[state.mu_id[inn]]], dtype=float)
        diff = np.abs(state.fiber_type[inn] - target_vals)
        features["mismatch_fraction"] = float(np.mean(diff > 0.18))
        features["mean_conversion_gap"] = float(np.mean(diff))
    else:
        features["mismatch_fraction"] = 0.0
        features["mean_conversion_gap"] = 0.0
    return features


# -----------------------------------------------------------------------------
# ABC
# -----------------------------------------------------------------------------

FEATURES_FOR_ABC = [
    "type1_fraction",
    "type2a_fraction",
    "type2x_fraction",
    "hybrid_fraction",
    "denervated_fraction",
    "grouping_index",
    "mean_cluster_size",
    "largest_cluster_fraction",
    "type1_csa_mean",
    "type2a_csa_mean",
    "type2x_csa_mean",
    "fast_atrophy_fraction",
    "hypertrophy_fraction",
    "csa_cv",
    "mu_size_gini",
    "mean_conversion_gap",
]

FEATURE_SCALES = {
    "type1_fraction": 0.15,
    "type2a_fraction": 0.15,
    "type2x_fraction": 0.12,
    "hybrid_fraction": 0.12,
    "denervated_fraction": 0.10,
    "grouping_index": 0.20,
    "mean_cluster_size": 10.0,
    "largest_cluster_fraction": 0.15,
    "type1_csa_mean": 0.25,
    "type2a_csa_mean": 0.25,
    "type2x_csa_mean": 0.25,
    "fast_atrophy_fraction": 0.20,
    "hypertrophy_fraction": 0.20,
    "csa_cv": 0.15,
    "mu_size_gini": 0.20,
    "mean_conversion_gap": 0.12,
}

PARAMS_FOR_ABC = [
    "p_nmj_dropout_2x",
    "p_nmj_dropout_2a",
    "p_reinn_2x",
    "p_reinn_2a",
    "p_reinn_slow",
    "sprout_radius",
    "atrophy_rate",
    "type_conversion_rate",
    "capacity_multiplier",
]

PRIORS = {
    "p_nmj_dropout_2x": (0.000, 0.025),
    "p_nmj_dropout_2a": (0.000, 0.015),
    "p_reinn_2x": (0.10, 0.85),
    "p_reinn_2a": (0.10, 0.90),
    "p_reinn_slow": (0.15, 0.95),
    "sprout_radius": (2.0, 10.0),
    "atrophy_rate": (0.02, 0.22),
    "type_conversion_rate": (0.00, 0.15),
    "capacity_multiplier": (1.5, 5.0),
}


def sample_prior(rng: np.random.Generator) -> Dict[str, float]:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PRIORS.items()}



def feature_distance(sim_feat: Dict[str, float], target_feat: Dict[str, float], feature_names: List[str] = FEATURES_FOR_ABC) -> float:
    terms = []
    for f in feature_names:
        s = FEATURE_SCALES.get(f, 1.0)
        terms.append(((sim_feat.get(f, 0.0) - target_feat.get(f, 0.0)) / s) ** 2)
    return float(math.sqrt(np.mean(terms)))



def run_abc(target_feat: Dict[str, float], base_params: RemodelParams, n_samples: int = 300, keep: int = 50, seed: int = 123, verbose: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_samples):
        sampled = sample_prior(rng)
        p = replace(base_params, **sampled, seed=int(rng.integers(1, 1_000_000)))
        final_state, _, _ = run_forward(p, snapshots=(0.0, max(base_params.death_targets)))
        feat = extract_features(final_state)
        dist = feature_distance(feat, target_feat)
        rows.append({"sample": s, "distance": dist, **sampled, **{f"sim_{k}": v for k, v in feat.items()}})
        if verbose and (s + 1) % max(1, n_samples // 10) == 0:
            print(f"ABC {s+1}/{n_samples}: current best distance={min(r['distance'] for r in rows):.3f}")
    df = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    df["accepted"] = False
    df.loc[: max(0, keep - 1), "accepted"] = True
    return df


# -----------------------------------------------------------------------------
# Plotting/export
# -----------------------------------------------------------------------------


def plot_maps(snapshot_states: Dict[float, MuscleState], out_file: str, title: str = "Remodeling snapshots") -> None:
    keys = sorted(snapshot_states.keys())
    fig, axes = plt.subplots(3, len(keys), figsize=(3.2 * len(keys), 8.8), constrained_layout=True)
    if len(keys) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    color_lut = {
        -1: np.array([0.55, 0.55, 0.55]),
         0: np.array([0.05, 0.05, 0.05]),
         1: np.array([0.55, 0.55, 0.55]),
         2: np.array([0.95, 0.95, 0.95]),
    }
    for col, k in enumerate(keys):
        st = snapshot_states[k]
        tm = subtype_map(st)
        rgb = np.zeros((*tm.shape, 3), dtype=float)
        for label, color in color_lut.items():
            rgb[tm == label] = color
        axes[0, col].imshow(rgb, interpolation="nearest")
        axes[0, col].set_title(f"{int(round(100*k))}% MU death\nsubtype map")
        axes[0, col].axis("off")
        axes[1, col].imshow(st.fiber_type, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[1, col].set_title("continuous phenotype")
        axes[1, col].axis("off")
        axes[2, col].imshow(st.csa, cmap="viridis", vmin=0.2, vmax=2.0, interpolation="nearest")
        axes[2, col].set_title("fiber CSA")
        axes[2, col].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_file, dpi=220)
    plt.close(fig)



def plot_history(history: pd.DataFrame, out_file: str) -> None:
    if history.empty:
        return
    x = 100 * history["frac_mu_dead"].values
    metrics = [
        ("grouping_index", "Grouping index"),
        ("denervated_fraction", "Denervated fraction"),
        ("type2x_fraction", "Type-IIx fraction"),
        ("hybrid_fraction", "Hybrid fraction"),
        ("hypertrophy_fraction", "Hypertrophy fraction"),
        ("mean_conversion_gap", "Mean conversion gap"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, label) in zip(axes, metrics):
        if key in history:
            ax.plot(x, history[key], marker="o", markersize=3)
        ax.set_xlabel("Killed motor units (%)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    fig.savefig(out_file, dpi=220)
    plt.close(fig)



def plot_abc_results(df: pd.DataFrame, true_params: Optional[Dict[str, float]], out_file: str) -> None:
    accepted = df[df["accepted"]]
    n = len(PARAMS_FOR_ABC)
    cols = 4
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, p in zip(axes, PARAMS_FOR_ABC):
        ax.hist(df[p], bins=25, alpha=0.25, label="prior samples")
        ax.hist(accepted[p], bins=20, alpha=0.75, label="accepted")
        if true_params and p in true_params:
            ax.axvline(true_params[p], linestyle="--", linewidth=2, label="truth")
        ax.set_title(p)
        ax.grid(True, alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.savefig(out_file, dpi=220)
    plt.close(fig)



def plot_inference_scatter(df: pd.DataFrame, true_params: Optional[Dict[str, float]], out_file: str) -> None:
    accepted = df[df["accepted"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    pairs = [
        ("sprout_radius", "p_reinn_2x"),
        ("p_nmj_dropout_2x", "type_conversion_rate"),
        ("atrophy_rate", "sim_fast_atrophy_fraction"),
        ("capacity_multiplier", "sim_hypertrophy_fraction"),
    ]
    for ax, (x, y) in zip(axes.ravel(), pairs):
        ax.scatter(df[x], df[y], s=8, alpha=0.15, label="all")
        ax.scatter(accepted[x], accepted[y], s=18, alpha=0.8, label="accepted")
        if true_params and x in true_params:
            ax.axvline(true_params[x], linestyle="--", linewidth=2)
        if true_params and y in true_params:
            ax.axhline(true_params[y], linestyle="--", linewidth=2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)



def write_params(params: RemodelParams, out_file: str) -> None:
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in asdict(params).items():
            w.writerow([k, v])



def save_features(features: Dict[str, float], out_file: str) -> None:
    pd.DataFrame([features]).to_csv(out_file, index=False)


# -----------------------------------------------------------------------------
# Scenarios and CLI
# -----------------------------------------------------------------------------


def scenario_params(name: str, base: RemodelParams) -> RemodelParams:
    if name == "classic_like":
        return replace(
            base,
            mn_slow_fraction=0.50,
            mn_2a_fraction=0.30,
            mn_2x_fraction=0.20,
            p_mu_death_slow=0.5,
            p_mu_death_2a=0.5,
            p_mu_death_2x=0.5,
            p_nmj_dropout_slow=0.0,
            p_nmj_dropout_2a=0.0,
            p_nmj_dropout_2x=0.0,
            p_reinn_slow=1.0,
            p_reinn_2a=1.0,
            p_reinn_2x=1.0,
            sprout_radius=1.1,
            sprout_sigma=1.0,
            complete_reinnervation=True,
            capacity_multiplier=999.0,
            atrophy_rate=0.0,
            recovery_rate=0.0,
            overload_hypertrophy_gain=0.0,
            type_conversion_rate=1.0,
            mismatch_penalty=1.0,
        )
    if name == "aging_type2":
        return replace(
            base,
            mn_slow_fraction=0.48,
            mn_2a_fraction=0.34,
            mn_2x_fraction=0.18,
            p_mu_death_slow=0.20,
            p_mu_death_2a=0.55,
            p_mu_death_2x=0.85,
            p_nmj_dropout_slow=0.001,
            p_nmj_dropout_2a=0.006,
            p_nmj_dropout_2x=0.012,
            p_reinn_slow=0.78,
            p_reinn_2a=0.55,
            p_reinn_2x=0.28,
            sprout_radius=5.5,
            sprout_sigma=2.6,
            complete_reinnervation=False,
            capacity_multiplier=3.2,
            atrophy_rate=0.12,
            recovery_rate=0.04,
            overload_hypertrophy_gain=0.18,
            type_conversion_rate=0.035,
            mismatch_penalty=0.55,
        )
    if name == "exercise_rescued":
        return replace(
            base,
            mn_slow_fraction=0.46,
            mn_2a_fraction=0.36,
            mn_2x_fraction=0.18,
            p_mu_death_slow=0.25,
            p_mu_death_2a=0.45,
            p_mu_death_2x=0.65,
            p_nmj_dropout_slow=0.0005,
            p_nmj_dropout_2a=0.003,
            p_nmj_dropout_2x=0.006,
            p_reinn_slow=0.92,
            p_reinn_2a=0.78,
            p_reinn_2x=0.55,
            sprout_radius=7.0,
            sprout_sigma=3.2,
            complete_reinnervation=False,
            capacity_multiplier=4.5,
            atrophy_rate=0.06,
            recovery_rate=0.08,
            overload_hypertrophy_gain=0.20,
            type_conversion_rate=0.06,
            mismatch_penalty=0.65,
        )
    if name == "als_like_failed_rescue":
        return replace(
            base,
            mn_slow_fraction=0.42,
            mn_2a_fraction=0.35,
            mn_2x_fraction=0.23,
            p_mu_death_slow=0.35,
            p_mu_death_2a=0.55,
            p_mu_death_2x=0.65,
            p_nmj_dropout_slow=0.012,
            p_nmj_dropout_2a=0.016,
            p_nmj_dropout_2x=0.020,
            p_reinn_slow=0.28,
            p_reinn_2a=0.22,
            p_reinn_2x=0.18,
            sprout_radius=3.5,
            sprout_sigma=1.8,
            complete_reinnervation=False,
            capacity_multiplier=2.0,
            atrophy_rate=0.18,
            recovery_rate=0.02,
            overload_hypertrophy_gain=0.10,
            type_conversion_rate=0.02,
            mismatch_penalty=0.50,
        )
    raise ValueError(f"Unknown scenario: {name}")



def run_simulation_cli(args: argparse.Namespace) -> None:
    ensure_dir(args.out)
    base = RemodelParams(grid_n=args.grid_n, n_mu=args.n_mu, seed=args.seed)
    params = scenario_params(args.scenario, base)
    final_state, snapshots, hist = run_forward(params)
    features = extract_features(final_state)
    plot_maps(snapshots, os.path.join(args.out, "remodeling_maps.png"), title=f"Scenario: {args.scenario}")
    plot_history(hist, os.path.join(args.out, "history_plots.png"))
    hist.to_csv(os.path.join(args.out, "history.csv"), index=False)
    save_features(features, os.path.join(args.out, "final_features.csv"))
    write_params(params, os.path.join(args.out, "params.csv"))
    np.save(os.path.join(args.out, "mu_id_final.npy"), final_state.mu_id)
    np.save(os.path.join(args.out, "fiber_type_final.npy"), final_state.fiber_type)
    np.save(os.path.join(args.out, "csa_final.npy"), final_state.csa)
    print(f"Saved simulation outputs to: {args.out}")
    print(pd.Series(features).to_string())



def run_infer_demo_cli(args: argparse.Namespace) -> None:
    ensure_dir(args.out)
    base = RemodelParams(grid_n=args.grid_n, n_mu=args.n_mu, seed=args.seed, death_targets=(0.0, 0.75))
    true_params = scenario_params(args.target_scenario, base)
    target_state, target_snapshots, target_hist = run_forward(true_params, snapshots=(0.0, 0.75))
    target_feat = extract_features(target_state)
    target_dir = os.path.join(args.out, "target")
    ensure_dir(target_dir)
    plot_maps(target_snapshots, os.path.join(target_dir, "target_maps.png"), title=f"Synthetic target: {args.target_scenario}")
    plot_history(target_hist, os.path.join(target_dir, "target_history.png"))
    target_hist.to_csv(os.path.join(target_dir, "target_history.csv"), index=False)
    save_features(target_feat, os.path.join(target_dir, "target_features.csv"))
    write_params(true_params, os.path.join(target_dir, "true_params.csv"))
    inf_base = replace(base, seed=args.seed + 100)
    df = run_abc(target_feat, inf_base, n_samples=args.n_samples, keep=args.keep, seed=args.seed + 200, verbose=True)
    df.to_csv(os.path.join(args.out, "abc_samples.csv"), index=False)
    true_dict = asdict(true_params)
    plot_abc_results(df, true_dict, os.path.join(args.out, "abc_posterior_histograms.png"))
    plot_inference_scatter(df, true_dict, os.path.join(args.out, "abc_identifiability_scatter.png"))
    acc = df[df["accepted"]]
    rows = []
    for p in PARAMS_FOR_ABC:
        rows.append({
            "parameter": p,
            "true": true_dict.get(p, np.nan),
            "accepted_mean": acc[p].mean(),
            "accepted_median": acc[p].median(),
            "accepted_p05": acc[p].quantile(0.05),
            "accepted_p95": acc[p].quantile(0.95),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "posterior_summary.csv"), index=False)
    print(f"Saved inference demo outputs to: {args.out}")
    print(summary.to_string(index=False))



def run_infer_cli(args: argparse.Namespace) -> None:
    ensure_dir(args.out)
    target_feat = pd.read_csv(args.target_features).iloc[0].to_dict()
    base = RemodelParams(grid_n=args.grid_n, n_mu=args.n_mu, seed=args.seed, death_targets=(0.0, args.target_death))
    df = run_abc(target_feat, base, n_samples=args.n_samples, keep=args.keep, seed=args.seed + 200, verbose=True)
    df.to_csv(os.path.join(args.out, "abc_samples.csv"), index=False)
    plot_abc_results(df, None, os.path.join(args.out, "abc_posterior_histograms.png"))
    plot_inference_scatter(df, None, os.path.join(args.out, "abc_identifiability_scatter.png"))
    print(f"Saved ABC outputs to: {args.out}")



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Motor-unit digital pathology model v2 (type I / IIa / IIx).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="Run one forward remodeling simulation.")
    s.add_argument("--scenario", choices=["classic_like", "aging_type2", "exercise_rescued", "als_like_failed_rescue"], default="aging_type2")
    s.add_argument("--grid-n", type=int, default=80)
    s.add_argument("--n-mu", type=int, default=49)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", type=str, default="remodeling_sim_output_v2")
    s.set_defaults(func=run_simulation_cli)

    d = sub.add_parser("infer_demo", help="Generate a synthetic target and run ABC recovery.")
    d.add_argument("--target-scenario", choices=["aging_type2", "exercise_rescued", "als_like_failed_rescue"], default="aging_type2")
    d.add_argument("--grid-n", type=int, default=42)
    d.add_argument("--n-mu", type=int, default=25)
    d.add_argument("--n-samples", type=int, default=300)
    d.add_argument("--keep", type=int, default=40)
    d.add_argument("--seed", type=int, default=11)
    d.add_argument("--out", type=str, default="inference_demo_output_v2")
    d.set_defaults(func=run_infer_demo_cli)

    i = sub.add_parser("infer", help="Run ABC using a target feature CSV.")
    i.add_argument("--target-features", type=str, required=True)
    i.add_argument("--grid-n", type=int, default=42)
    i.add_argument("--n-mu", type=int, default=25)
    i.add_argument("--target-death", type=float, default=0.75)
    i.add_argument("--n-samples", type=int, default=300)
    i.add_argument("--keep", type=int, default=40)
    i.add_argument("--seed", type=int, default=11)
    i.add_argument("--out", type=str, default="abc_output_v2")
    i.set_defaults(func=run_infer_cli)
    return p



def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
