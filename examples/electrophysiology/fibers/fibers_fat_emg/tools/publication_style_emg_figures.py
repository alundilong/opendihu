#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create publication-style composite figures from metrics_by_electrode.csv.

This v2 version fixes label/title overlap by:
  - using constrained_layout
  - moving panel labels inside each axes
  - shortening subplot titles
  - separating colorbars more cleanly
  - increasing figure size and spacing

Input:
    metrics_by_electrode.csv
produced by:
    compare_electrodes_four_cases_v4.py

Outputs:
    pubfig_emg_metrics/
        figure1_spatial_rms_and_correlation.png
        figure1_spatial_rms_and_correlation.pdf
        figure2_metric_summaries.png
        figure2_metric_summaries.pdf
        figure3_peak_to_peak_maps.png
        figure3_peak_to_peak_maps.pdf
        publication_summary_table.csv

Example:
    python publication_style_emg_figures_v2.py \
        --csv emg_overlay_by_electrode/metrics_by_electrode.csv \
        --out-dir pubfig_emg_metrics_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Journal-readable default for all plot text, including titles, axis labels,
# tick labels, legends, annotations, panel labels, and colorbars.
plt.rcParams.update({"font.size": 14})


CASES = ["healthy", "death_25", "death_50", "death_75"]
DISEASE_CASES = ["death_25", "death_50", "death_75"]
CASE_DISPLAY = {
    "healthy": "Healthy",
    "death_25": "25% MU death",
    "death_50": "50% MU death",
    "death_75": "75% MU death",
}


def infer_grid(n_electrodes: int, n_points_xy: int | None = None) -> Tuple[int, int]:
    if n_points_xy is not None and n_points_xy > 0 and n_electrodes % n_points_xy == 0:
        return int(n_points_xy), int(n_electrodes // n_points_xy)

    if n_electrodes == 384:
        return 12, 32

    factors = [f for f in range(1, n_electrodes + 1) if n_electrodes % f == 0]
    nxy = min(factors, key=lambda f: abs(f - np.sqrt(n_electrodes)))
    return int(nxy), int(n_electrodes // nxy)


def to_grid(values: np.ndarray, n_points_xy: int, n_points_z: int) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(n_points_z, n_points_xy)


def panel_label(ax, label: str) -> None:
    """
    Put panel label inside the axes to avoid overlap with neighboring subplot titles
    and colorbars.
    """
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.8),
        zorder=10,
    )


def save_png_pdf(fig, out_base: Path, dpi: int = 300) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")


def metric_array(df: pd.DataFrame, case: str, metric: str) -> np.ndarray:
    return df[f"{case}_{metric}"].to_numpy(dtype=float)


def corr_array(df: pd.DataFrame, case: str) -> np.ndarray:
    return df[f"{case}_corr_vs_healthy"].to_numpy(dtype=float)


def set_heatmap_axis_style(ax) -> None:
    ax.set_xlabel("Electrode x index")
    ax.set_ylabel("Electrode y index")


def create_figure1(df: pd.DataFrame, n_points_xy: int, n_points_z: int, out_dir: Path) -> None:
    """
    Figure 1: RMS heatmaps + delta RMS + correlation.
    v2 layout avoids title/colorbar overlap.
    """
    fig, axes = plt.subplots(
        2, 4,
        figsize=(18.0, 9.2),
        constrained_layout=True,
    )
    # Add a little extra breathing room between rows/columns.
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    rms_arrays = [metric_array(df, c, "rms") for c in CASES]
    rms_vmin = min(float(np.nanmin(v)) for v in rms_arrays)
    rms_vmax = max(float(np.nanmax(v)) for v in rms_arrays)

    # Top row: RMS heatmaps.
    im_top = None
    for i, case in enumerate(CASES):
        ax = axes[0, i]
        grid = to_grid(metric_array(df, case, "rms"), n_points_xy, n_points_z)
        im_top = ax.imshow(grid, aspect="auto", origin="upper", vmin=rms_vmin, vmax=rms_vmax)
        ax.set_title(CASE_DISPLAY[case], pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("A") + i))

    cbar1 = fig.colorbar(im_top, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=28)
    cbar1.set_label("RMS [mV]")

    # Bottom row: delta RMS for first 3 columns.
    healthy = metric_array(df, "healthy", "rms")
    diff_arrays = [metric_array(df, case, "rms") - healthy for case in DISEASE_CASES]
    diff_abs = max(
        max(abs(float(np.nanmin(v))), abs(float(np.nanmax(v))))
        for v in diff_arrays
    )
    im_diff = None
    for i, case in enumerate(DISEASE_CASES):
        ax = axes[1, i]
        grid = to_grid(metric_array(df, case, "rms") - healthy, n_points_xy, n_points_z)
        im_diff = ax.imshow(
            grid, aspect="auto", origin="upper",
            vmin=-diff_abs, vmax=diff_abs, cmap="coolwarm"
        )
        ax.set_title(f"{CASE_DISPLAY[case]} − Healthy", pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("E") + i))

    cbar2 = fig.colorbar(im_diff, ax=axes[1, 0:3], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("ΔRMS [mV]")

    # Bottom right: correlation map.
    ax_corr = axes[1, 3]
    im_corr = ax_corr.imshow(
        to_grid(corr_array(df, "death_50"), n_points_xy, n_points_z),
        aspect="auto", origin="upper", vmin=-1, vmax=1, cmap="coolwarm"
    )
    ax_corr.set_title("Correlation: 50% vs Healthy", pad=8)
    set_heatmap_axis_style(ax_corr)
    panel_label(ax_corr, "H")

    cbar3 = fig.colorbar(im_corr, ax=ax_corr, shrink=0.88, pad=0.015, aspect=28)
    cbar3.set_label("Correlation")

    save_png_pdf(fig, out_dir / "figure1_spatial_rms_and_correlation")
    plt.close(fig)


def create_figure2(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 2: line plots + boxplots + mean bars."""
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    x = df["electrode"].to_numpy(dtype=int)

    ax1 = axes[0, 0]
    for case in CASES:
        ax1.plot(x, metric_array(df, case, "rms"), linewidth=1.2, label=CASE_DISPLAY[case])
    ax1.set_title("Electrode-wise RMS")
    ax1.set_xlabel("Electrode index")
    ax1.set_ylabel("RMS [mV]")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    panel_label(ax1, "A")

    ax2 = axes[0, 1]
    for case in CASES:
        ax2.plot(x, metric_array(df, case, "peak_to_peak"), linewidth=1.2, label=CASE_DISPLAY[case])
    ax2.set_title("Electrode-wise peak-to-peak")
    ax2.set_xlabel("Electrode index")
    ax2.set_ylabel("Peak-to-peak [mV]")
    ax2.grid(True, alpha=0.25)
    panel_label(ax2, "B")

    ax3 = axes[0, 2]
    for case in CASES:
        ax3.plot(x, metric_array(df, case, "max_abs"), linewidth=1.2, label=CASE_DISPLAY[case])
    ax3.set_title("Electrode-wise max |EMG|")
    ax3.set_xlabel("Electrode index")
    ax3.set_ylabel("Max |EMG| [mV]")
    ax3.grid(True, alpha=0.25)
    panel_label(ax3, "C")

    ax4 = axes[1, 0]
    rms_data = [metric_array(df, case, "rms") for case in CASES]
    ax4.boxplot(rms_data, tick_labels=[CASE_DISPLAY[c] for c in CASES], showfliers=False)
    ax4.set_title("RMS distribution across electrodes")
    ax4.set_ylabel("RMS [mV]")
    ax4.tick_params(axis="x", rotation=20)
    ax4.grid(True, axis="y", alpha=0.25)
    panel_label(ax4, "D")

    ax5 = axes[1, 1]
    p2p_data = [metric_array(df, case, "peak_to_peak") for case in CASES]
    ax5.boxplot(p2p_data, tick_labels=[CASE_DISPLAY[c] for c in CASES], showfliers=False)
    ax5.set_title("Peak-to-peak distribution")
    ax5.set_ylabel("Peak-to-peak [mV]")
    ax5.tick_params(axis="x", rotation=20)
    ax5.grid(True, axis="y", alpha=0.25)
    panel_label(ax5, "E")

    ax6 = axes[1, 2]
    means = [np.mean(metric_array(df, case, "rms")) for case in CASES]
    stds = [np.std(metric_array(df, case, "rms")) for case in CASES]
    xpos = np.arange(len(CASES))
    ax6.bar(xpos, means, yerr=stds, capsize=4)
    ax6.set_xticks(xpos, [CASE_DISPLAY[c] for c in CASES], rotation=20)
    ax6.set_ylabel("RMS [mV]")
    ax6.set_title("Global RMS summary (mean ± SD)")
    ax6.grid(True, axis="y", alpha=0.25)
    panel_label(ax6, "F")

    save_png_pdf(fig, out_dir / "figure2_metric_summaries")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    data = [corr_array(df, case) for case in DISEASE_CASES]
    ax.boxplot(data, tick_labels=[CASE_DISPLAY[c] for c in DISEASE_CASES], showfliers=False)
    ax.set_ylabel("Correlation")
    ax.grid(True, axis="y", alpha=0.25)
    save_png_pdf(fig2, out_dir / "supplementary_correlation_boxplot")
    plt.close(fig2)


def create_figure3(df: pd.DataFrame, n_points_xy: int, n_points_z: int, out_dir: Path) -> None:
    """Figure 3: peak-to-peak maps + ratios vs healthy."""
    fig, axes = plt.subplots(
        2, 4,
        figsize=(18.0, 9.2),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    p2p_arrays = [metric_array(df, c, "peak_to_peak") for c in CASES]
    p2p_vmin = min(float(np.nanmin(v)) for v in p2p_arrays)
    p2p_vmax = max(float(np.nanmax(v)) for v in p2p_arrays)

    im_top = None
    for i, case in enumerate(CASES):
        ax = axes[0, i]
        grid = to_grid(metric_array(df, case, "peak_to_peak"), n_points_xy, n_points_z)
        im_top = ax.imshow(grid, aspect="auto", origin="upper", vmin=p2p_vmin, vmax=p2p_vmax)
        ax.set_title(CASE_DISPLAY[case], pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("A") + i))

    cbar1 = fig.colorbar(im_top, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=28)
    cbar1.set_label("Peak-to-peak [mV]")

    healthy = metric_array(df, "healthy", "peak_to_peak")
    eps = 1e-12
    ratio_arrays = [
        metric_array(df, c, "peak_to_peak") / np.maximum(healthy, eps)
        for c in DISEASE_CASES
    ]
    ratio_vmin = min(float(np.nanmin(v)) for v in ratio_arrays)
    ratio_vmax = max(float(np.nanmax(v)) for v in ratio_arrays)

    im_ratio = None
    for i, case in enumerate(DISEASE_CASES):
        ax = axes[1, i]
        ratio = metric_array(df, case, "peak_to_peak") / np.maximum(healthy, eps)
        grid = to_grid(ratio, n_points_xy, n_points_z)
        im_ratio = ax.imshow(grid, aspect="auto", origin="upper", vmin=ratio_vmin, vmax=ratio_vmax)
        ax.set_title(f"{CASE_DISPLAY[case]} / Healthy", pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("E") + i))

    cbar2 = fig.colorbar(im_ratio, ax=axes[1, 0:3], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("Peak-to-peak ratio")

    ax_line = axes[1, 3]
    x = df["electrode"].to_numpy(dtype=int)
    for case in CASES:
        ax_line.plot(x, metric_array(df, case, "peak_to_peak"), linewidth=1.0, label=CASE_DISPLAY[case])
    ax_line.set_title("Peak-to-peak across electrodes", pad=8)
    ax_line.set_xlabel("Electrode index")
    ax_line.set_ylabel("Peak-to-peak [mV]")
    ax_line.grid(True, alpha=0.25)
    ax_line.legend(loc="best")
    panel_label(ax_line, "H")

    save_png_pdf(fig, out_dir / "figure3_peak_to_peak_maps")
    plt.close(fig)


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    metrics = ["rms", "peak_to_peak", "max_abs"]

    for metric in metrics:
        row: Dict[str, float] = {"metric": metric}
        for case in CASES:
            vals = metric_array(df, case, metric)
            row[f"{case}_mean"] = float(np.mean(vals))
            row[f"{case}_std"] = float(np.std(vals))
            row[f"{case}_median"] = float(np.median(vals))
            row[f"{case}_min"] = float(np.min(vals))
            row[f"{case}_max"] = float(np.max(vals))
        rows.append(row)

    row_corr = {"metric": "corr_vs_healthy"}
    row_corr["healthy_mean"] = np.nan
    row_corr["healthy_std"] = np.nan
    row_corr["healthy_median"] = np.nan
    row_corr["healthy_min"] = np.nan
    row_corr["healthy_max"] = np.nan
    for case in DISEASE_CASES:
        vals = corr_array(df, case)
        row_corr[f"{case}_mean"] = float(np.nanmean(vals))
        row_corr[f"{case}_std"] = float(np.nanstd(vals))
        row_corr[f"{case}_median"] = float(np.nanmedian(vals))
        row_corr[f"{case}_min"] = float(np.nanmin(vals))
        row_corr[f"{case}_max"] = float(np.nanmax(vals))
    rows.append(row_corr)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-style EMG figures from metrics_by_electrode.csv.")
    parser.add_argument(
        "--csv",
        default="emg_overlay_by_electrode/metrics_by_electrode.csv",
        help="Path to metrics_by_electrode.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="pubfig_emg_metrics_v2",
        help="Output directory for publication figures",
    )
    parser.add_argument(
        "--n-points-xy",
        type=int,
        default=None,
        help="Electrodes along x direction. If omitted, inferred automatically (384 -> 12x32).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "electrode" not in df.columns:
        raise ValueError(f"{csv_path} must contain an 'electrode' column.")

    n_electrodes = len(df)
    n_points_xy, n_points_z = infer_grid(n_electrodes, args.n_points_xy)

    print(f"Loaded {csv_path}")
    print(f"Detected grid: {n_points_xy} x {n_points_z} = {n_electrodes} electrodes")

    create_figure1(df, n_points_xy, n_points_z, out_dir)
    print(f"Created {out_dir / 'figure1_spatial_rms_and_correlation.png'}")
    print(f"Created {out_dir / 'figure1_spatial_rms_and_correlation.pdf'}")

    create_figure2(df, out_dir)
    print(f"Created {out_dir / 'figure2_metric_summaries.png'}")
    print(f"Created {out_dir / 'figure2_metric_summaries.pdf'}")

    create_figure3(df, n_points_xy, n_points_z, out_dir)
    print(f"Created {out_dir / 'figure3_peak_to_peak_maps.png'}")
    print(f"Created {out_dir / 'figure3_peak_to_peak_maps.pdf'}")

    summary = create_summary_table(df)
    summary.to_csv(out_dir / "publication_summary_table.csv", index=False)
    print(f"Created {out_dir / 'publication_summary_table.csv'}")


if __name__ == "__main__":
    main()