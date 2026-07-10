#!/usr/bin/env python3
"""
Denervation-reinnervation computer model inspired by:
Cohen et al., "A computer model of denervation-reinnervation in skeletal muscle",
Muscle & Nerve, 1987.

This is a clean-room Python reproduction of the algorithm described in the paper:
  - square muscle cross-section grid
  - motor units with spatial centers
  - fiber type assigned at MU level
  - initial fiber -> MU innervation based on radial probability + repulsion factor
  - progressive motor neuron death
  - complete reinnervation from four nearest living neighbors in repeated passes
  - displays similar to the paper's Figure 1
  - summary metrics: type proportions, MU density, MU radius, and a CDI-like grouping index

Important note:
The 1987 paper describes the probability function qualitatively but does not print its
exact mathematical formula, and the CDI was defined in a separate paper. Therefore this
script uses a logistic radial probability function and a nearest-neighbor CDI-like index.
The core denervation/reinnervation logic follows the paper's text.

Dependencies:
    pip install numpy matplotlib pandas

Example:
    python denervation_reinnervation_model.py --out results/default --show

    python denervation_reinnervation_model.py --out results/dataset4 \
        --type1-reinnerv-prob 0.10 --type1-denerv-prob 0.50
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DENERVATED = -1


@dataclass
class ModelParams:
    # Initial muscle organization
    grid_size: int = 80                    # 80 x 80 fibers, paper default
    n_motor_units: int = 49                # must be perfect square, paper default
    n_fiber_types: int = 2
    type1_fraction: float = 0.50           # initial percent type 1
    repulsion_factor: float = 0.20         # paper default
    slope: float = 0.10                    # paper default, used here in logistic decay
    radius: float = 18.0                   # paper default, inflection/radius in fiber diameters
    seed: int = 1

    # Denervation / reinnervation type-specific probabilities.
    # These are relative weights, not absolute probabilities.
    # For two types, index 0 = type 1, index 1 = type 2.
    type1_denerv_prob: float = 0.50
    type2_denerv_prob: float = 0.50
    type1_reinnerv_prob: float = 0.50
    type2_reinnerv_prob: float = 0.50

    # Simulation schedule
    max_death_fraction: float = 0.75       # for Figure-1-like output; can use 0.90
    snapshot_fractions: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)

    # Visualization
    highlight_mu: int = 0                  # 0 auto-selects a surviving MU with largest growth; paper showed MU No. 33

    # Reinnervation behavior
    complete_reinnervation: bool = True
    max_reinnervation_passes: int = 10000


class DenervationReinnervationModel:
    def __init__(self, params: ModelParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self.G = params.grid_size
        self.N = self.G * self.G
        self.K = params.n_motor_units

        root = int(round(math.sqrt(self.K)))
        if root * root != self.K:
            raise ValueError("n_motor_units must be a perfect square, as in the original model.")
        self.mu_grid_n = root

        # Grid coordinates: each lattice point is one muscle fiber cross-section.
        yy, xx = np.mgrid[0:self.G, 0:self.G]
        self.x = xx.ravel().astype(float)
        self.y = yy.ravel().astype(float)

        # Arrays populated by initialize()
        self.mu_centers = None             # shape (K,2)
        self.mu_type = None                # shape (K,), values 0..n_fiber_types-1
        self.mu_alive = None               # shape (K,), bool
        self.fiber_mu = None               # shape (N,), MU id, or DENERVATED

        self.initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self):
        self.mu_centers = self._make_uniform_mu_centers()
        self.mu_type = self._assign_mu_types()
        self.mu_alive = np.ones(self.K, dtype=bool)
        self.fiber_mu = np.full(self.N, DENERVATED, dtype=int)
        self._initial_innervation()

    def _make_uniform_mu_centers(self) -> np.ndarray:
        """Uniform MU centers over the square cross-section."""
        # Place centers at evenly spaced positions within the grid.
        coords = np.linspace(0.5 * self.G / self.mu_grid_n,
                             self.G - 0.5 * self.G / self.mu_grid_n,
                             self.mu_grid_n)
        cx, cy = np.meshgrid(coords, coords)
        return np.column_stack([cx.ravel(), cy.ravel()])

    def _assign_mu_types(self) -> np.ndarray:
        """Assign motor unit type randomly according to requested type proportions."""
        if self.p.n_fiber_types != 2:
            raise NotImplementedError("This reproduction currently supports two fiber/MU types.")
        probs = np.array([self.p.type1_fraction, 1.0 - self.p.type1_fraction], dtype=float)
        probs /= probs.sum()
        return self.rng.choice(2, size=self.K, p=probs)

    def _radial_probability(self, dist: np.ndarray) -> np.ndarray:
        """
        Radial density function for initial innervation.

        The paper says slope controls the drop-off and inflection/radius controls where
        probability becomes negligible, but does not print the exact formula. A logistic
        function has the same qualitative behavior.

        p(d) = 1 / (1 + exp(slope * (d - radius)))
        """
        return 1.0 / (1.0 + np.exp(self.p.slope * (dist - self.p.radius)))

    def _neighbor_mu_ids_existing(self, fiber_index: int) -> List[int]:
        """Four-neighbor MU IDs already assigned during initial setup."""
        row = fiber_index // self.G
        col = fiber_index % self.G
        ids = []
        for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= rr < self.G and 0 <= cc < self.G:
                mu = self.fiber_mu[rr * self.G + cc]
                if mu != DENERVATED:
                    ids.append(int(mu))
        return ids

    def _initial_innervation(self):
        """
        Assign each fiber to a motor neuron.

        The paper loops over fibers, computes a distance-based probability to each MU,
        reduces probability by repulsion_factor if an adjacent already-assigned fiber is
        in that MU, then samples one MU weighted by those probabilities.
        """
        order = np.arange(self.N)
        self.rng.shuffle(order)

        centers = self.mu_centers
        for idx in order:
            dx = centers[:, 0] - self.x[idx]
            dy = centers[:, 1] - self.y[idx]
            dist = np.sqrt(dx * dx + dy * dy)
            weights = self._radial_probability(dist)

            # Efficient cutoff: ignore MUs with <1% of max probability, matching paper's idea.
            max_w = weights.max()
            weights[weights < 0.01 * max_w] = 0.0

            # Repulsion: if adjacent fiber already assigned to same MU, multiply by factor.
            neighbor_mus = set(self._neighbor_mu_ids_existing(idx))
            if neighbor_mus:
                for mu in neighbor_mus:
                    weights[mu] *= self.p.repulsion_factor

            if weights.sum() <= 0:
                # Robust fallback.
                weights = np.ones(self.K)
            probs = weights / weights.sum()
            self.fiber_mu[idx] = self.rng.choice(self.K, p=probs)

    # ------------------------------------------------------------------
    # Denervation-reinnervation process
    # ------------------------------------------------------------------
    def _death_weights(self) -> np.ndarray:
        weights = np.zeros(self.K, dtype=float)
        death_probs = np.array([self.p.type1_denerv_prob, self.p.type2_denerv_prob], dtype=float)
        for k in range(self.K):
            if self.mu_alive[k]:
                weights[k] = death_probs[self.mu_type[k]]
        return weights

    def kill_one_motor_unit(self) -> Optional[int]:
        """
        Randomly kill one living motor unit weighted by type-specific death probability.
        All fibers belonging to that MU become denervated.
        """
        weights = self._death_weights()
        if weights.sum() <= 0:
            return None
        probs = weights / weights.sum()
        killed = int(self.rng.choice(self.K, p=probs))
        self.mu_alive[killed] = False
        self.fiber_mu[self.fiber_mu == killed] = DENERVATED
        return killed

    def _living_neighbor_mus(self, row: int, col: int) -> List[int]:
        ids = []
        for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= rr < self.G and 0 <= cc < self.G:
                mu = int(self.fiber_mu[rr * self.G + cc])
                if mu != DENERVATED and self.mu_alive[mu]:
                    ids.append(mu)
        return ids

    def reinnervate_completely(self) -> int:
        """
        Reinnervate denervated fibers from four-nearest living neighbors in repeated passes.

        Two-step update per pass:
          1. choose all candidate reinnervations without updating the grid
          2. apply all chosen reinnervations simultaneously

        Returns number of passes used.
        """
        rein_probs = np.array([self.p.type1_reinnerv_prob, self.p.type2_reinnerv_prob], dtype=float)
        passes = 0

        while np.any(self.fiber_mu == DENERVATED):
            passes += 1
            if passes > self.p.max_reinnervation_passes:
                raise RuntimeError("Exceeded max_reinnervation_passes. Complete reinnervation failed.")

            choices: List[Tuple[int, int]] = []
            denervated_indices = np.where(self.fiber_mu == DENERVATED)[0]

            for idx in denervated_indices:
                row = idx // self.G
                col = idx % self.G
                neigh_mus = self._living_neighbor_mus(row, col)
                if not neigh_mus:
                    continue

                # Unique neighboring MUs; weight by type-specific reinnervation probability.
                unique = np.array(sorted(set(neigh_mus)), dtype=int)
                weights = np.array([rein_probs[self.mu_type[mu]] for mu in unique], dtype=float)
                if weights.sum() <= 0:
                    continue
                chosen_mu = int(self.rng.choice(unique, p=weights / weights.sum()))
                choices.append((idx, chosen_mu))

            if not choices:
                # This can happen if all remaining denervated fibers are isolated from living fibers.
                # It should be rare for complete reinnervation unless nearly all MUs are dead.
                if self.p.complete_reinnervation:
                    raise RuntimeError(
                        "No candidates for reinnervation, but denervated fibers remain. "
                        "Try lower death fraction or allow incomplete reinnervation."
                    )
                break

            for idx, mu in choices:
                self.fiber_mu[idx] = mu

        return passes

    def denervation_reinnervation_step(self) -> Dict[str, float]:
        killed = self.kill_one_motor_unit()
        if killed is None:
            return {"killed_mu": np.nan, "passes": 0}
        passes = self.reinnervate_completely() if self.p.complete_reinnervation else 0
        return {"killed_mu": killed, "passes": passes}

    def run_to_fraction(self, death_fraction: float) -> List[Dict[str, float]]:
        """Run denervation-reinnervation until fraction of MUs killed reaches target."""
        target_dead = int(round(death_fraction * self.K))
        history = []
        while self.n_dead_mus() < target_dead:
            step_info = self.denervation_reinnervation_step()
            metrics = self.measurements()
            metrics.update(step_info)
            history.append(metrics)
        return history

    def n_dead_mus(self) -> int:
        return int(np.sum(~self.mu_alive))

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------
    def fiber_types(self) -> np.ndarray:
        """Type of each fiber equals type of its current innervating MU."""
        out = np.full(self.N, DENERVATED, dtype=int)
        inn = self.fiber_mu != DENERVATED
        out[inn] = self.mu_type[self.fiber_mu[inn]]
        return out

    def type1_percent(self) -> float:
        ft = self.fiber_types()
        valid = ft != DENERVATED
        if valid.sum() == 0:
            return np.nan
        return 100.0 * np.mean(ft[valid] == 0)

    def cdi_like(self) -> float:
        """
        A nearest-neighbor codispersion/grouping index.

        This is NOT guaranteed to be the exact CDI from Lester et al. 1983.
        It compares observed same-type nearest-neighbor frequency with the random
        expectation from global type proportions:

            index = (observed_same - expected_same) / (1 - expected_same)

        Positive values mean same-type grouping; ~0 means random mixing.
        """
        ft = self.fiber_types().reshape(self.G, self.G)
        # right and down edges to avoid double-counting
        a1 = ft[:, :-1].ravel()
        b1 = ft[:, 1:].ravel()
        a2 = ft[:-1, :].ravel()
        b2 = ft[1:, :].ravel()
        a = np.concatenate([a1, a2])
        b = np.concatenate([b1, b2])
        valid = (a != DENERVATED) & (b != DENERVATED)
        a = a[valid]
        b = b[valid]
        if len(a) == 0:
            return np.nan
        observed_same = np.mean(a == b)
        vals, counts = np.unique(ft[ft != DENERVATED], return_counts=True)
        props = counts / counts.sum()
        expected_same = np.sum(props * props)
        denom = max(1.0 - expected_same, 1e-12)
        return float((observed_same - expected_same) / denom)

    @staticmethod
    def _convex_hull(points: np.ndarray) -> np.ndarray:
        """Monotonic chain convex hull. Returns hull vertices."""
        pts = sorted(map(tuple, points))
        if len(pts) <= 1:
            return np.array(pts)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return np.array(lower[:-1] + upper[:-1])

    @staticmethod
    def _polygon_area(poly: np.ndarray) -> float:
        if len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def motor_unit_area_density_radius(self) -> pd.DataFrame:
        """
        Estimate MU area as convex hull of its fibers, density as fiber count / area,
        radius as radius of equivalent-area circle.
        """
        rows = []
        for k in range(self.K):
            idx = np.where(self.fiber_mu == k)[0]
            count = len(idx)
            if count == 0:
                rows.append({"mu": k, "alive": self.mu_alive[k], "type": self.mu_type[k],
                             "count": 0, "area": 0.0, "density": np.nan, "radius": 0.0})
                continue
            points = np.column_stack([self.x[idx], self.y[idx]])
            hull = self._convex_hull(points)
            area = self._polygon_area(hull)
            # avoid zero area for tiny MUs
            area_eff = max(area, 1.0)
            density = count / area_eff
            radius = math.sqrt(area_eff / math.pi)
            rows.append({"mu": k, "alive": self.mu_alive[k], "type": self.mu_type[k],
                         "count": count, "area": area_eff, "density": density, "radius": radius})
        return pd.DataFrame(rows)

    def measurements(self) -> Dict[str, float]:
        df = self.motor_unit_area_density_radius()
        alive = df[df["alive"]]
        return {
            "dead_mus": self.n_dead_mus(),
            "death_percent": 100.0 * self.n_dead_mus() / self.K,
            "type1_percent": self.type1_percent(),
            "cdi_like": self.cdi_like(),
            "mean_mu_area": float(alive["area"].mean()) if len(alive) else np.nan,
            "std_mu_area": float(alive["area"].std(ddof=1)) if len(alive) > 1 else np.nan,
            "mean_mu_density": float(alive["density"].mean()) if len(alive) else np.nan,
            "std_mu_density": float(alive["density"].std(ddof=1)) if len(alive) > 1 else np.nan,
            "mean_mu_radius": float(alive["radius"].mean()) if len(alive) else np.nan,
            "std_mu_radius": float(alive["radius"].std(ddof=1)) if len(alive) > 1 else np.nan,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def type_image(self) -> np.ndarray:
        return self.fiber_types().reshape(self.G, self.G)

    def mu_image(self, mu_1_based: int) -> np.ndarray:
        k = mu_1_based - 1
        img = np.zeros((self.G, self.G), dtype=float)
        img[self.fiber_mu.reshape(self.G, self.G) == k] = 1.0
        return img

    def plot_type_distribution(self, ax, title: str = "Fiber type distribution"):
        img = self.type_image()
        # type 1 = light, type 2 = dark, denervated = gray
        display = np.zeros_like(img, dtype=float)
        display[img == 0] = 1.0
        display[img == 1] = 0.0
        display[img == DENERVATED] = 0.5
        ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_motor_unit(self, ax, mu_1_based: int, title: Optional[str] = None):
        img = self.mu_image(mu_1_based)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if title is None:
            title = f"Motor unit {mu_1_based}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])


def run_single_simulation(params: ModelParams, out_dir: Path, show: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DenervationReinnervationModel(params)
    snapshots: Dict[float, DenervationReinnervationModel] = {}
    records = []

    # Save initial snapshot
    snapshots[0.0] = clone_model(model)
    records.append(model.measurements())

    # Run through MUs killed one-by-one; collect snapshots close to requested fractions.
    snapshot_targets = sorted(set(params.snapshot_fractions))
    max_target = max(snapshot_targets)
    while model.n_dead_mus() < int(round(max_target * model.K)):
        model.denervation_reinnervation_step()
        metrics = model.measurements()
        records.append(metrics)

        frac = model.n_dead_mus() / model.K
        for target in snapshot_targets:
            if target not in snapshots and frac >= target:
                snapshots[target] = clone_model(model)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "history.csv", index=False)

    # Save final MU table
    model.motor_unit_area_density_radius().to_csv(out_dir / "motor_units_final.csv", index=False)

    # Save parameter file
    pd.Series(asdict(params)).to_csv(out_dir / "params.csv")

    # Figure similar to paper Fig. 1: type distribution and selected MU at 0, 25, 50, 75%.
    # If highlight_mu <= 0, choose a living MU that gained the most fibers after reinnervation.
    highlight_mu = params.highlight_mu
    if highlight_mu <= 0:
        first_snap = snapshots[min(snapshot_targets)]
        last_snap = snapshots[max(snapshot_targets)]
        first_counts = np.bincount(first_snap.fiber_mu[first_snap.fiber_mu >= 0], minlength=first_snap.K)
        last_counts = np.bincount(last_snap.fiber_mu[last_snap.fiber_mu >= 0], minlength=last_snap.K)
        growth = last_counts - first_counts
        growth[~last_snap.mu_alive] = -10**9
        highlight_mu = int(np.argmax(growth)) + 1
        print(f"Auto-selected highlighted MU: {highlight_mu} "
              f"(fiber count {first_counts[highlight_mu-1]} -> {last_counts[highlight_mu-1]})")

    nrows = len(snapshot_targets)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(8.5, 3.6 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    panel_labels = list("ABCDEFGH")
    for r, frac in enumerate(snapshot_targets):
        snap = snapshots.get(frac)
        if snap is None:
            continue
        meas = snap.measurements()
        left_label = panel_labels[2*r] if 2*r < len(panel_labels) else ""
        right_label = panel_labels[2*r+1] if 2*r+1 < len(panel_labels) else ""
        title1 = (f"{left_label}. Fiber types after {100*frac:.0f}% MU death + complete reinnervation\n"
                  f"CDI-like={meas['cdi_like']:.3f}, type1={meas['type1_percent']:.1f}%")
        snap.plot_type_distribution(axes[r, 0], title1)
        title2 = f"{right_label}. Highlighted MU {highlight_mu}, {100*frac:.0f}% death"
        snap.plot_motor_unit(axes[r, 1], highlight_mu, title2)
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_reproduction.png", dpi=220)
    # Backward-compatible filename
    fig.savefig(out_dir / "figure_like_paper.png", dpi=220)

    # Additional diagnostic figure: one denervation event before and after reinnervation.
    # This explicitly shows the transient denervated fibers, which are absent from Fig. 1
    # because the paper displayed states after complete reinnervation.
    diag_model = DenervationReinnervationModel(params)
    while diag_model.n_dead_mus() < int(round(0.25 * diag_model.K)):
        diag_model.denervation_reinnervation_step()
    before = clone_model(diag_model)
    killed = diag_model.kill_one_motor_unit()
    denervated = clone_model(diag_model)
    diag_model.reinnervate_completely()
    after = clone_model(diag_model)

    figd, axd = plt.subplots(1, 3, figsize=(10.5, 3.5))
    before.plot_type_distribution(axd[0], "Before one MU death")
    denervated.plot_type_distribution(axd[1], f"Immediately after MU {killed+1} death\n(gray = denervated)")
    after.plot_type_distribution(axd[2], "After complete reinnervation")
    figd.tight_layout()
    figd.savefig(out_dir / "denervation_then_reinnervation.png", dpi=220)

    # History plots
    fig2, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs = axs.ravel()
    axs[0].plot(df["death_percent"], df["type1_percent"], marker="o")
    axs[0].set_xlabel("Percent motor neuron death")
    axs[0].set_ylabel("Percent type 1 fibers")

    axs[1].plot(df["death_percent"], df["cdi_like"], marker="o")
    axs[1].set_xlabel("Percent motor neuron death")
    axs[1].set_ylabel("CDI-like grouping index")

    base_density = df["mean_mu_density"].iloc[0]
    axs[2].plot(df["death_percent"], 100 * (df["mean_mu_density"] / base_density - 1.0), marker="o")
    axs[2].set_xlabel("Percent motor neuron death")
    axs[2].set_ylabel("% increase mean MU density")

    base_radius = df["mean_mu_radius"].iloc[0]
    axs[3].plot(df["death_percent"], 100 * (df["mean_mu_radius"] / base_radius - 1.0), marker="o")
    axs[3].set_xlabel("Percent motor neuron death")
    axs[3].set_ylabel("% increase mean MU radius")

    fig2.tight_layout()
    fig2.savefig(out_dir / "history_plots.png", dpi=200)

    if show:
        plt.show()
    else:
        plt.close("all")

    print(f"Saved outputs to: {out_dir.resolve()}")
    print(df.tail())


def clone_model(model: DenervationReinnervationModel) -> DenervationReinnervationModel:
    """Small manual clone for snapshots."""
    new = DenervationReinnervationModel.__new__(DenervationReinnervationModel)
    new.p = model.p
    new.rng = np.random.default_rng(0)  # not used for snapshots
    new.G = model.G
    new.N = model.N
    new.K = model.K
    new.mu_grid_n = model.mu_grid_n
    new.x = model.x.copy()
    new.y = model.y.copy()
    new.mu_centers = model.mu_centers.copy()
    new.mu_type = model.mu_type.copy()
    new.mu_alive = model.mu_alive.copy()
    new.fiber_mu = model.fiber_mu.copy()
    return new


def preset_dataset_params(dataset: int, base: ModelParams) -> ModelParams:
    """
    Parameters from paper Table 1.

    Data set columns:
      type 1 reinnervation probability, type 1 denervation probability,
      initial percent type 1, initial MU radius.

    The complementary type-2 probability is set to 100 - type1 probability.
    """
    # values are in proportions, not percent
    table = {
        1: dict(type1_reinnerv_prob=0.50, type1_denerv_prob=0.50, type1_fraction=0.50, radius=18.0),
        2: dict(type1_reinnerv_prob=0.50, type1_denerv_prob=0.50, type1_fraction=0.50, radius=36.0),
        3: dict(type1_reinnerv_prob=0.50, type1_denerv_prob=0.50, type1_fraction=0.80, radius=18.0),
        4: dict(type1_reinnerv_prob=0.10, type1_denerv_prob=0.50, type1_fraction=0.50, radius=18.0),
        5: dict(type1_reinnerv_prob=0.50, type1_denerv_prob=0.10, type1_fraction=0.50, radius=18.0),
        6: dict(type1_reinnerv_prob=0.90, type1_denerv_prob=0.10, type1_fraction=0.50, radius=18.0),
    }
    if dataset not in table:
        raise ValueError("dataset must be 1..6")
    d = table[dataset]
    p = ModelParams(**asdict(base))
    for key, value in d.items():
        setattr(p, key, value)
    p.type2_reinnerv_prob = 1.0 - p.type1_reinnerv_prob
    p.type2_denerv_prob = 1.0 - p.type1_denerv_prob
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cohen-style skeletal muscle denervation-reinnervation model")
    parser.add_argument("--out", type=str, default="denervation_results", help="Output directory")
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--n-motor-units", type=int, default=49)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--type1-fraction", type=float, default=0.50)
    parser.add_argument("--repulsion-factor", type=float, default=0.20)
    parser.add_argument("--slope", type=float, default=0.10)
    parser.add_argument("--radius", type=float, default=18.0)
    parser.add_argument("--type1-denerv-prob", type=float, default=0.50)
    parser.add_argument("--type2-denerv-prob", type=float, default=0.50)
    parser.add_argument("--type1-reinnerv-prob", type=float, default=0.50)
    parser.add_argument("--type2-reinnerv-prob", type=float, default=0.50)
    parser.add_argument("--highlight-mu", type=int, default=0, help="1-based MU number to highlight; 0 means auto-select a surviving MU with the largest growth")
    parser.add_argument("--snapshots", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75])
    parser.add_argument("--dataset", type=int, default=None, help="Use paper Table-1 preset dataset 1..6")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    p = ModelParams(
        grid_size=args.grid_size,
        n_motor_units=args.n_motor_units,
        type1_fraction=args.type1_fraction,
        repulsion_factor=args.repulsion_factor,
        slope=args.slope,
        radius=args.radius,
        seed=args.seed,
        type1_denerv_prob=args.type1_denerv_prob,
        type2_denerv_prob=args.type2_denerv_prob,
        type1_reinnerv_prob=args.type1_reinnerv_prob,
        type2_reinnerv_prob=args.type2_reinnerv_prob,
        snapshot_fractions=tuple(args.snapshots),
        max_death_fraction=max(args.snapshots),
        highlight_mu=args.highlight_mu,
    )

    if args.dataset is not None:
        p = preset_dataset_params(args.dataset, p)

    run_single_simulation(p, Path(args.out), show=args.show)


if __name__ == "__main__":
    main()
