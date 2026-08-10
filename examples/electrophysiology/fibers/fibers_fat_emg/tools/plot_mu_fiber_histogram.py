#!/usr/bin/env python3
"""
Plot fiber-distribution histograms from an OpenDiHu MU distribution file.

Input:
    MU_fibre_distribution_37x37_20.txt

What it shows:
    1) Bar chart of fiber count for each MU
    2) Histogram of the MU fiber counts

Example:
    python plot_mu_fiber_histogram.py \
        --input death_50/MU_fibre_distribution_37x37_20.txt \
        --output death_50/fiber_histogram.png
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def read_distribution_file(path: str) -> np.ndarray:
    """Read whitespace-separated MU assignments from file."""
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Input file is empty: {path}")

    values = np.array([int(x) for x in text.split()], dtype=int)
    return values


def infer_index_base(assignments: np.ndarray) -> int:
    """
    Infer whether MU IDs are 0-based or 1-based.
    If minimum value is 0 -> assume 0-based.
    Otherwise -> assume 1-based.
    """
    return 0 if assignments.min() == 0 else 1


def compute_counts(assignments: np.ndarray, n_motor_units: int | None, index_base: int) -> np.ndarray:
    """Convert assignments into per-MU fiber counts."""
    mu_ids_0_based = assignments - index_base

    if np.any(mu_ids_0_based < 0):
        raise ValueError(
            "Found MU IDs smaller than expected after subtracting index_base. "
            f"Check the file and index_base={index_base}."
        )

    inferred_n = int(mu_ids_0_based.max()) + 1
    if n_motor_units is None:
        n_motor_units = inferred_n
    else:
        if inferred_n > n_motor_units:
            raise ValueError(
                f"File contains MU IDs up to {inferred_n - 1 + index_base}, "
                f"but n_motor_units={n_motor_units}."
            )

    counts = np.bincount(mu_ids_0_based, minlength=n_motor_units)
    return counts


def plot_counts(counts: np.ndarray, output_path: str, title: str = "") -> None:
    """Create a 2-panel figure: bar chart + histogram."""
    mu_ids = np.arange(1, len(counts) + 1)

    mean_count = counts.mean()
    std_count = counts.std(ddof=0)
    min_count = counts.min()
    max_count = counts.max()
    total_fibers = counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel 1: bar chart per MU
    axes[0].bar(mu_ids, counts)
    axes[0].set_xlabel("Motor Unit ID")
    axes[0].set_ylabel("Number of fibers")
    axes[0].set_title("Fibers assigned to each MU")
    axes[0].set_xticks(mu_ids)

    # Panel 2: histogram of counts
    bins = np.arange(counts.min(), counts.max() + 2) - 0.5
    axes[1].hist(counts, bins=bins)
    axes[1].set_xlabel("Fibers per MU")
    axes[1].set_ylabel("Number of MUs")
    axes[1].set_title("Histogram of MU fiber counts")

    summary = (
        f"Total fibers = {total_fibers}\n"
        f"Number of MUs = {len(counts)}\n"
        f"Mean = {mean_count:.2f}\n"
        f"Std = {std_count:.2f}\n"
        f"Min = {min_count}\n"
        f"Max = {max_count}"
    )
    axes[1].text(
        0.98, 0.98, summary,
        transform=axes[1].transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    if title:
        fig.suptitle(title)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot fiber count distribution from MU_fibre_distribution_37x37_20.txt"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to MU_fibre_distribution_37x37_20.txt"
    )
    parser.add_argument(
        "--output",
        default="fiber_histogram.png",
        help="Output PNG file"
    )
    parser.add_argument(
        "--n-motor-units",
        type=int,
        default=None,
        help="Number of motor units (default: infer from file)"
    )
    parser.add_argument(
        "--index-base",
        type=int,
        choices=[0, 1],
        default=None,
        help="MU ID indexing base. Default: infer automatically."
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title"
    )

    args = parser.parse_args()

    assignments = read_distribution_file(args.input)

    index_base = args.index_base
    if index_base is None:
        index_base = infer_index_base(assignments)

    counts = compute_counts(assignments, args.n_motor_units, index_base)

    if not args.title:
        stage_name = os.path.basename(os.path.dirname(os.path.abspath(args.input)))
        title = f"Fiber distribution: {stage_name}"
    else:
        title = args.title

    plot_counts(counts, args.output, title=title)

    print("Done.")
    print(f"Input file:   {args.input}")
    print(f"Output image: {args.output}")
    print(f"Index base:   {index_base}")
    print(f"Total fibers: {counts.sum()}")
    print(f"Number of MUs:{len(counts)}")
    print(f"Mean fibers/MU: {counts.mean():.2f}")
    print(f"Min fibers/MU:  {counts.min()}")
    print(f"Max fibers/MU:  {counts.max()}")


if __name__ == "__main__":
    main()