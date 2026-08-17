#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate the healthy MU fiber-count distribution plots from an EXISTING grouped
OpenDiHu denervation study, without regenerating any simulation cases.

This script is intentionally aligned with generate_grouped_denervation_study_slurm.py.
It reads the already-written group-level file:

    g-N/protocol_A/healthy_mu_profile_g-N.csv

whose expected columns are:
    mu_id_0_based
    mu_id_1_based
    recruitment_rank_0_based
    recruitment_order_1_based
    mu_type_id
    mu_type_name
    healthy_target_fiber_count
    territory_sigma_grid_units

The original generator sorted this CSV by recruitment rank, NOT by spatial MU ID.
Therefore, this script explicitly reconstructs:

  1. counts_by_mu   -> spatial MU-ID order (left panel)
  2. counts_by_rank -> recruitment/size-rank order (middle panel)

This avoids the incorrect behavior of treating MU IDs/ranks (0..19 or 1..20) as
fiber counts.

Example
-------
python regenerate_mu_size_distribution_plot_fixed.py \
    --study-root fiber_dist_generator/opendihu_repeated_denervation_study \
    --font-size 14

Only selected groups:
python regenerate_mu_size_distribution_plot_fixed.py \
    --study-root fiber_dist_generator/opendihu_repeated_denervation_study \
    --groups 1 2 3 \
    --font-size 14

By default, each regenerated PNG/PDF is written into:
    g-N/protocol_A/

with the same base name as the original:
    healthy_mu_size_distribution_g-N.png
    healthy_mu_size_distribution_g-N.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_plotting(font_size: float) -> None:
    """Use the requested font size everywhere in the figure."""
    fs = float(font_size)
    plt.rcParams.update({
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "figure.titlesize": fs,
    })


def add_panel_label(ax, label: str, font_size: float = 16.0) -> None:
    """Add a clean journal-style panel label just inside the upper-left axes."""
    ax.text(
        0.015, 0.985, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=float(font_size),
        fontweight="bold",
        color="black",
        zorder=20,
    )


def discover_profiles(
    study_root: Path,
    requested_groups: Optional[Sequence[int]] = None,
) -> Dict[int, Path]:
    """Discover the exact group-level healthy profile files written by the generator."""
    requested = None if requested_groups is None else {int(g) for g in requested_groups}
    found: Dict[int, Path] = {}

    # Current grouped-study layout from generate_grouped_denervation_study_slurm.py
    for profile in sorted(study_root.glob("g-*/protocol_A/healthy_mu_profile_g-*.csv")):
        m = re.search(r"g-(\d+)", profile.name)
        if m is None:
            # Fall back to the directory name.
            m = re.search(r"g-(\d+)", str(profile.parent.parent))
        if m is None:
            continue

        group = int(m.group(1))
        if requested is not None and group not in requested:
            continue

        if group in found and found[group] != profile:
            raise RuntimeError(
                f"Multiple healthy profile files were found for g-{group}:\n"
                f"  {found[group]}\n"
                f"  {profile}"
            )
        found[group] = profile

    if requested is not None:
        missing = sorted(requested - set(found))
        if missing:
            raise FileNotFoundError(
                "Could not find healthy_mu_profile_g-N.csv for group(s): "
                + ", ".join(f"g-{g}" for g in missing)
            )

    if not found:
        raise FileNotFoundError(
            "No files matching g-*/protocol_A/healthy_mu_profile_g-*.csv "
            f"were found under:\n  {study_root}"
        )

    return dict(sorted(found.items()))


def read_mu_size_basis(group_dir: Path, group_number: int, fallback: float) -> float:
    """Read mu_size_basis from the existing group metadata, if available."""
    metadata_candidates = [
        group_dir / f"group_metadata_g-{group_number}.json",
        group_dir / "group_metadata.json",
    ]

    for path in metadata_candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = float(payload["mu_size_basis"])
            if np.isfinite(value) and value > 0:
                return value
        except Exception as exc:
            print(f"  Warning: could not read mu_size_basis from {path}: {exc}")

    print(
        f"  Note: group metadata with mu_size_basis was not found; "
        f"using fallback basis={fallback:g}."
    )
    return float(fallback)


def load_exact_healthy_profile(profile_csv: Path) -> pd.DataFrame:
    """Read the exact schema written by write_healthy_mu_profile_csv()."""
    df = pd.read_csv(profile_csv)

    required = {
        "mu_id_0_based",
        "recruitment_rank_0_based",
        "healthy_target_fiber_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{profile_csv} is not the expected healthy profile CSV.\n"
            f"Missing required column(s): {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    for col in [
        "mu_id_0_based",
        "recruitment_rank_0_based",
        "healthy_target_fiber_count",
    ]:
        out[col] = pd.to_numeric(out[col], errors="raise")

    out["mu_id_0_based"] = out["mu_id_0_based"].astype(int)
    out["recruitment_rank_0_based"] = out["recruitment_rank_0_based"].astype(int)
    out["healthy_target_fiber_count"] = out["healthy_target_fiber_count"].astype(int)

    n_mu = len(out)
    if n_mu < 1:
        raise ValueError(f"Healthy profile is empty: {profile_csv}")

    expected = set(range(n_mu))
    mu_ids = set(out["mu_id_0_based"].tolist())
    ranks = set(out["recruitment_rank_0_based"].tolist())

    if mu_ids != expected:
        raise ValueError(
            f"Spatial MU IDs are not a complete 0..{n_mu-1} set in {profile_csv}.\n"
            f"Found: {sorted(mu_ids)}"
        )
    if ranks != expected:
        raise ValueError(
            f"Recruitment ranks are not a complete 0..{n_mu-1} set in {profile_csv}.\n"
            f"Found: {sorted(ranks)}"
        )
    if np.any(out["healthy_target_fiber_count"].to_numpy(dtype=int) <= 0):
        raise ValueError("All healthy_target_fiber_count values must be positive.")

    return out


def reconstruct_counts(df: pd.DataFrame):
    """Reconstruct the exact quantities used by the original generator plot."""
    n_mu = len(df)

    # Left panel: arbitrary/spatial MU ID order.
    by_mu = df.sort_values("mu_id_0_based")
    counts_by_mu = by_mu["healthy_target_fiber_count"].to_numpy(dtype=int)

    # Middle panel: physiological recruitment/size-rank order.
    by_rank = df.sort_values("recruitment_rank_0_based")
    counts_by_rank = by_rank["healthy_target_fiber_count"].to_numpy(dtype=int)

    # Cross-check that these are the same multiset, only differently ordered.
    if not np.array_equal(np.sort(counts_by_mu), np.sort(counts_by_rank)):
        raise RuntimeError("MU-ID and rank reconstruction produced inconsistent fiber counts.")

    return counts_by_mu, counts_by_rank


def plot_healthy_mu_size_distribution(
    profile_df: pd.DataFrame,
    mu_size_basis: float,
    group_number: int,
    out_base: Path,
    dpi: int = 300,
) -> None:
    """Reproduce the original generator figure using the saved healthy profile."""
    counts_by_mu, counts_by_rank = reconstruct_counts(profile_df)
    n_mu = len(counts_by_mu)

    mu_ids = np.arange(1, n_mu + 1, dtype=int)
    ranks = np.arange(n_mu, dtype=int)

    total_fibers = int(counts_by_mu.sum())

    # EXACTLY the same theoretical curve definition as the original generator:
    # normalized basis**rank curve whose sum equals the realized total fiber count.
    raw = np.power(float(mu_size_basis), ranks.astype(float))
    theoretical = raw / raw.sum() * total_fibers

    # Same log-space diagnostic fit as the original generator.
    log_counts = np.log(np.maximum(counts_by_rank.astype(float), 1e-12))
    slope, intercept = np.polyfit(ranks.astype(float), log_counts, 1)
    fitted = np.exp(intercept + slope * ranks)
    ss_res = float(np.sum((counts_by_rank - fitted) ** 2))
    ss_tot = float(np.sum((counts_by_rank - counts_by_rank.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    fitted_basis = math.exp(float(slope))

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel A: fiber count by spatial MU ID
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.bar(mu_ids, counts_by_mu)
    ax.set_xlabel("Motor Unit ID")
    ax.set_ylabel("Modeled fibers")
    ax.set_title("Fiber count by spatial MU ID", fontweight="bold")
    add_panel_label(ax, "A", plt.rcParams["font.size"] + 2)
    ax.set_xticks(mu_ids)
    ax.grid(True, axis="y", alpha=0.20)

    # ------------------------------------------------------------------
    # Panel B: exponential MU-size law in rank order
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.bar(
        ranks + 1,
        counts_by_rank,
        label="Realized integer fiber counts",
        alpha=0.85,
    )
    ax.plot(
        ranks + 1,
        theoretical,
        marker="o",
        linewidth=1.8,
        label=rf"$N_r = N_{{\mathrm{{tot}}}}\,b^r / \sum_k b^k$,  $b={mu_size_basis:g}$",
    )
    ax.set_xlabel(r"MU recruitment / size rank, $r$")
    ax.set_ylabel("Modeled fibers")
    ax.set_title(r"Exponential MU size law, $N_r \propto b^r$", fontweight="bold")
    add_panel_label(ax, "B", plt.rcParams["font.size"] + 2)
    ax.set_xticks(np.arange(1, n_mu + 1, 2))
    ax.grid(True, axis="y", alpha=0.20)
    ax.legend(loc="upper left")

    # ------------------------------------------------------------------
    # Panel C: histogram + diagnostics
    # ------------------------------------------------------------------
    ax = axes[2]
    n_bins = min(10, max(5, int(round(math.sqrt(n_mu)) + 2)))
    ax.hist(counts_by_mu, bins=n_bins)
    ax.set_xlabel("Modeled fibers per MU")
    ax.set_ylabel("Number of MUs")
    ax.set_title("MU fiber-count histogram", fontweight="bold")
    add_panel_label(ax, "C", plt.rcParams["font.size"] + 2)
    ax.grid(True, axis="y", alpha=0.20)

    stats = (
        f"Total fibers = {total_fibers}\n"
        f"MUs = {n_mu}\n"
        f"Requested basis, $b$ = {mu_size_basis:.4g}\n"
        f"Log-fit basis, $\\hat{{b}}$ = {fitted_basis:.4f}\n"
        f"Rank-fit $R^2$ = {r2:.4f}\n"
        f"Min = {int(counts_by_rank.min())}\n"
        f"Max = {int(counts_by_rank.max())}\n"
        f"Max/min = {counts_by_rank.max()/max(counts_by_rank.min(), 1):.2f}"
    )
    ax.text(
        0.98,
        0.98,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=plt.rcParams["font.size"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.35", alpha=0.92),
    )

    fig.suptitle(
        f"Healthy motor-unit fiber-count distribution (g-{group_number})",
        fontweight="bold",
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Console audit: this makes a wrong parse immediately obvious.
    print(f"  MUs:               {n_mu}")
    print(f"  Total fibers:      {total_fibers}")
    print(f"  Min / max fibers:  {counts_by_rank.min()} / {counts_by_rank.max()}")
    print(f"  Basis requested:   {mu_size_basis:g}")
    print(f"  Basis log-fit:     {fitted_basis:.6f}")
    print(f"  Rank-fit R^2:      {r2:.6f}")
    print(f"  Counts by rank:    {counts_by_rank.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the original healthy MU size-distribution diagnostic figure "
            "from existing grouped-study healthy_mu_profile CSV files."
        )
    )
    parser.add_argument(
        "--study-root",
        default="opendihu_repeated_denervation_study",
        help="Root directory containing g-1/, g-2/, g-3/, ...",
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="*",
        default=None,
        help="Optional group numbers. Default: all discovered groups.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=14.0,
        help="Font size used throughout the figure. Default: 14.",
    )
    parser.add_argument(
        "--mu-size-basis",
        type=float,
        default=1.20,
        help=(
            "Fallback basis if group_metadata_g-N.json is unavailable. "
            "The metadata value is preferred. Default: 1.20."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300 dpi.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Optional common output directory. If omitted, figures overwrite/recreate "
            "healthy_mu_size_distribution_g-N.png beside each profile CSV in protocol_A/."
        ),
    )
    args = parser.parse_args()

    configure_plotting(args.font_size)

    study_root = Path(args.study_root).expanduser().resolve()
    if not study_root.exists():
        raise FileNotFoundError(f"Study root does not exist: {study_root}")

    profiles = discover_profiles(study_root, args.groups)
    common_out = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if common_out is not None:
        common_out.mkdir(parents=True, exist_ok=True)

    print(f"Study root: {study_root}")
    print(f"Groups found: {', '.join('g-'+str(g) for g in profiles)}")

    for group_number, profile_csv in profiles.items():
        print("\n" + "=" * 76)
        print(f"Rebuilding healthy MU-size distribution for g-{group_number}")
        print("=" * 76)
        print(f"  Profile: {profile_csv}")

        profile_df = load_exact_healthy_profile(profile_csv)
        group_dir = profile_csv.parent.parent
        basis = read_mu_size_basis(
            group_dir,
            group_number,
            fallback=args.mu_size_basis,
        )

        if common_out is None:
            out_base = profile_csv.parent / f"healthy_mu_size_distribution_g-{group_number}"
        else:
            out_base = common_out / f"healthy_mu_size_distribution_g-{group_number}"

        plot_healthy_mu_size_distribution(
            profile_df=profile_df,
            mu_size_basis=basis,
            group_number=group_number,
            out_base=out_base,
            dpi=args.dpi,
        )

        print(f"  PNG: {out_base}.png")
        print(f"  PDF: {out_base}.pdf")

    print("\nDone. No simulation cases were regenerated or modified.")


if __name__ == "__main__":
    main()
