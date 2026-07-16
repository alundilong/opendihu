#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Protocol A vs Protocol B OpenDiHu EMG results when all folders are under
the same build_release/out directory.

Your current folder structure:

    build_release/out/
        healthy/
        death_25/
        death_50/
        death_75/
        healthy_protocol_B/
        death_25_protocol_B/
        death_50_protocol_B/
        death_75_protocol_B/

Run:

    python analyze_protocol_ab_same_out.py \
        --out-root build_release/out \
        --out-dir protocol_AB_analysis

Outputs:

    protocol_AB_analysis/
        protocolA_metrics_by_electrode.csv
        protocolB_metrics_by_electrode.csv
        protocolA_global_summary.csv
        protocolB_global_summary.csv
        protocolA_vs_protocolB_global_summary.csv
        fig_protocolB_rms_maps.png/pdf
        fig_protocolB_peak_to_peak_maps.png/pdf
        fig_protocolA_vs_B_rms_compensation.png/pdf
        fig_protocolA_vs_B_peak_to_peak_compensation.png/pdf
        fig_protocolA_vs_B_global_summary.png/pdf
"""

from __future__ import print_function

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

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

PROTOCOL_A_FOLDERS = {
    "healthy": "healthy",
    "death_25": "death_25",
    "death_50": "death_50",
    "death_75": "death_75",
}

PROTOCOL_B_FOLDERS = {
    "healthy": "healthy_protocol_B",
    "death_25": "death_25_protocol_B",
    "death_50": "death_50_protocol_B",
    "death_75": "death_75_protocol_B",
}

# Publication-style metric names. Avoid str.title(), which incorrectly renders
# abbreviations such as RMS as "Rms" and produces terse labels such as "Corr".
METRIC_DISPLAY = {
    "rms": "RMS",
    "peak_to_peak": "Peak-to-peak amplitude",
    "max_abs": "Maximum absolute EMG",
    "corr_vs_healthy": "Correlation vs Healthy",
}


# -----------------------------------------------------------------------------
# Robust OpenDiHu electrodes.csv loader
# -----------------------------------------------------------------------------

def parse_float_list(tokens: List[str]) -> List[float]:
    values = []
    for token in tokens:
        token = token.strip().replace("#", "")
        if token == "":
            continue
        try:
            values.append(float(token))
        except ValueError:
            matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", token)
            for m in matches:
                try:
                    values.append(float(m))
                except ValueError:
                    pass
    return values


def parse_position_comment_line(line: str) -> Optional[np.ndarray]:
    parts = line.rstrip("\n").split(";")
    vals = parse_float_list(parts[2:])
    if len(vals) >= 3 and len(vals) % 3 == 0:
        return np.asarray(vals, dtype=float)
    return None


def infer_electrode_grid(electrode_positions, n_points, default_xy=None):
    if (
        electrode_positions is not None
        and getattr(electrode_positions, "ndim", 0) == 2
        and electrode_positions.shape[0] == n_points
        and electrode_positions.shape[1] >= 3
    ):
        z_positions = electrode_positions[:, 2]
        differences = z_positions[1:] - z_positions[:-1]
        if len(differences) > 0:
            mean_abs = np.mean(np.abs(differences)) + 1e-12
            jumps = np.where(differences > mean_abs * 3)[0]
            if len(jumps) > 0:
                n_points_xy = int(jumps[0] + 1)
                if n_points_xy > 0 and n_points % n_points_xy == 0:
                    return n_points_xy, int(n_points // n_points_xy)

    if default_xy is not None and default_xy > 0 and n_points % default_xy == 0:
        return int(default_xy), int(n_points // default_xy)

    if n_points == 384:
        return 12, 32

    factors = [f for f in range(1, n_points + 1) if n_points % f == 0]
    nxy = min(factors, key=lambda f: abs(f - np.sqrt(n_points)))
    return int(nxy), int(n_points // nxy)


def read_electrodes_csv(filename, default_xy=None):
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError("Cannot find {}".format(filename))

    t_values = []
    emg_rows = []
    position_data = None
    n_points_ref = None
    detected_format = None
    expect_position_comment_next = False

    with filename.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if raw == "":
                continue

            if raw.startswith("#"):
                if "#electrode positions" in raw:
                    expect_position_comment_next = True
                    continue

                if expect_position_comment_next:
                    pos = parse_position_comment_line(raw)
                    if pos is not None:
                        position_data = pos
                        detected_format = "comment_positions"
                    expect_position_comment_next = False
                    continue

                if position_data is None and raw.startswith("#;"):
                    pos = parse_position_comment_line(raw)
                    if pos is not None:
                        position_data = pos
                        detected_format = "comment_positions"
                    continue

                continue

            parts = raw.split(";")
            while parts and parts[-1].strip() == "":
                parts.pop()

            if len(parts) < 4:
                continue

            try:
                t = float(parts[1])
                n_points = int(float(parts[2]))
            except ValueError:
                continue

            numeric_tail = parse_float_list(parts[3:])

            if n_points_ref is None:
                n_points_ref = n_points
            elif n_points != n_points_ref:
                raise ValueError(
                    "{}, line {}: n_points changed from {} to {}".format(
                        filename, line_no, n_points_ref, n_points
                    )
                )

            if len(numeric_tail) >= 4 * n_points:
                if position_data is None:
                    position_data = np.asarray(numeric_tail[:3 * n_points], dtype=float)
                    detected_format = "transient_positions"
                values = numeric_tail[3 * n_points:3 * n_points + n_points]
            elif len(numeric_tail) >= n_points:
                values = numeric_tail[:n_points]
                if detected_format is None:
                    detected_format = "values_only"
            else:
                raise ValueError(
                    "{}, line {}: expected at least {} EMG values, got {}".format(
                        filename, line_no, n_points, len(numeric_tail)
                    )
                )

            t_values.append(t)
            emg_rows.append(values)

    if not emg_rows:
        raise ValueError("No numeric EMG rows found in {}".format(filename))

    t_array = np.asarray(t_values, dtype=float)
    emg_array = np.asarray(emg_rows, dtype=float)
    n_points = int(n_points_ref)

    if emg_array.shape[1] != n_points:
        raise ValueError(
            "{}: parsed EMG shape {}, expected second dimension {}".format(
                filename, emg_array.shape, n_points
            )
        )

    electrode_positions = None
    if position_data is not None and len(position_data) >= 3 * n_points:
        electrode_positions = np.asarray(position_data[:3 * n_points], dtype=float).reshape(n_points, 3)

    n_points_xy, n_points_z = infer_electrode_grid(electrode_positions, n_points, default_xy=default_xy)

    print(
        "    parsed {}: format={}, shape={}, grid={}x{}".format(
            filename, detected_format, emg_array.shape, n_points_xy, n_points_z
        )
    )

    return {
        "filename": str(filename),
        "t": t_array,
        "emg": emg_array,
        "positions": electrode_positions,
        "n_points": n_points,
        "n_points_xy": n_points_xy,
        "n_points_z": n_points_z,
        "detected_format": detected_format,
    }


def load_protocol_from_folder_map(out_root, folder_map, protocol_name, csv_name="electrodes.csv", default_xy=None):
    out_root = Path(out_root)
    cases = {}
    print("Loading {} from {}".format(protocol_name, out_root))

    for case in CASES:
        folder = folder_map[case]
        filename = out_root / folder / csv_name
        print("  {} ({}) -> {}".format(case, folder, filename))
        cases[case] = read_electrodes_csv(filename, default_xy=default_xy)

    ref_t = cases["healthy"]["t"]
    ref_n = cases["healthy"]["n_points"]
    ref_xy = cases["healthy"]["n_points_xy"]
    ref_z = cases["healthy"]["n_points_z"]

    for case in CASES[1:]:
        c = cases[case]
        if c["n_points"] != ref_n:
            raise ValueError("{} {}: n_points differs from healthy.".format(protocol_name, case))
        if len(c["t"]) != len(ref_t) or not np.allclose(c["t"], ref_t):
            raise ValueError("{} {}: time vector differs from healthy.".format(protocol_name, case))

    print("  detected grid: {} x {} = {} electrodes".format(ref_xy, ref_z, ref_n))
    print("  time range: [{}, {}] ms, n_time_steps={}".format(ref_t[0], ref_t[-1], len(ref_t)))
    return cases, ref_t, ref_n, ref_xy, ref_z


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def subtract_baseline(y, n_baseline=10):
    n = min(n_baseline, len(y))
    if n <= 0:
        return y
    return y - np.mean(y[:n])


def compute_metrics_for_protocol(cases, subtract_mean=False):
    n_points = cases["healthy"]["n_points"]
    rows = []
    healthy_emg = cases["healthy"]["emg"]

    for ch in range(n_points):
        row = {"electrode": ch}
        ref = healthy_emg[:, ch]
        if subtract_mean:
            ref = subtract_baseline(ref)

        for case in CASES:
            y = cases[case]["emg"][:, ch]
            if subtract_mean:
                y = subtract_baseline(y)

            row["{}_rms".format(case)] = float(np.sqrt(np.mean(y ** 2)))
            row["{}_peak_to_peak".format(case)] = float(np.max(y) - np.min(y))
            row["{}_max_abs".format(case)] = float(np.max(np.abs(y)))
            row["{}_mean".format(case)] = float(np.mean(y))

            if case != "healthy":
                denom = np.std(ref) * np.std(y)
                corr = np.nan if denom == 0 else float(
                    np.mean((ref - np.mean(ref)) * (y - np.mean(y))) / denom
                )
                row["{}_corr_vs_healthy".format(case)] = corr

        rows.append(row)

    return pd.DataFrame(rows)


def global_summary_from_metrics(df, protocol_name):
    rows = []
    for metric in ["rms", "peak_to_peak", "max_abs"]:
        for case in CASES:
            col = "{}_{}".format(case, metric)
            vals = df[col].to_numpy(dtype=float)
            rows.append({
                "protocol": protocol_name,
                "case": case,
                "metric": metric,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "median": float(np.nanmedian(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            })

    for case in DISEASE_CASES:
        col = "{}_corr_vs_healthy".format(case)
        vals = df[col].to_numpy(dtype=float)
        rows.append({
            "protocol": protocol_name,
            "case": case,
            "metric": "corr_vs_healthy",
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "median": float(np.nanmedian(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def to_grid(values, n_points_xy, n_points_z):
    arr = np.asarray(values, dtype=float)
    return arr.reshape(n_points_z, n_points_xy)


def metric_array(df, case, metric):
    return df["{}_{}".format(case, metric)].to_numpy(dtype=float)


def corr_array(df, case):
    return df["{}_corr_vs_healthy".format(case)].to_numpy(dtype=float)


def save_png_pdf(fig, out_base, dpi=300):
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", dpi=dpi, bbox_inches="tight")


def panel_label(ax, label):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.8),
        zorder=10,
    )


def set_heatmap_style(ax):
    ax.set_xlabel("Electrode x index")
    ax.set_ylabel("Electrode y index")


def create_protocol_maps(df, n_points_xy, n_points_z, out_dir, protocol_label, metric="rms"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    arrays = [metric_array(df, c, metric) for c in CASES]
    vmin = min(float(np.nanmin(v)) for v in arrays)
    vmax = max(float(np.nanmax(v)) for v in arrays)

    im_top = None
    for i, case in enumerate(CASES):
        ax = axes[0, i]
        im_top = ax.imshow(
            to_grid(metric_array(df, case, metric), n_points_xy, n_points_z),
            aspect="auto", origin="upper", vmin=vmin, vmax=vmax
        )
        ax.set_title(CASE_DISPLAY[case])
        set_heatmap_style(ax)
        panel_label(ax, chr(ord("A") + i))
    cbar1 = fig.colorbar(im_top, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=28)
    cbar1.set_label("{} [mV]".format(metric.replace("_", " ")))

    healthy = metric_array(df, "healthy", metric)
    diff_arrays = [metric_array(df, c, metric) - healthy for c in DISEASE_CASES]
    diff_abs = max(max(abs(float(np.nanmin(v))), abs(float(np.nanmax(v)))) for v in diff_arrays)

    im_diff = None
    for i, case in enumerate(DISEASE_CASES):
        ax = axes[1, i]
        im_diff = ax.imshow(
            to_grid(metric_array(df, case, metric) - healthy, n_points_xy, n_points_z),
            aspect="auto", origin="upper", vmin=-diff_abs, vmax=diff_abs, cmap="coolwarm"
        )
        ax.set_title("{} - Healthy".format(CASE_DISPLAY[case]))
        set_heatmap_style(ax)
        panel_label(ax, chr(ord("E") + i))
    cbar2 = fig.colorbar(im_diff, ax=axes[1, 0:3], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("Delta {} [mV]".format(metric.replace("_", " ")))

    ax_corr = axes[1, 3]
    im_corr = ax_corr.imshow(
        to_grid(corr_array(df, "death_50"), n_points_xy, n_points_z),
        aspect="auto", origin="upper", vmin=-1, vmax=1, cmap="coolwarm"
    )
    ax_corr.set_title("Correlation: 50% vs Healthy")
    set_heatmap_style(ax_corr)
    panel_label(ax_corr, "H")
    cbar3 = fig.colorbar(im_corr, ax=ax_corr, shrink=0.88, pad=0.015, aspect=28)
    cbar3.set_label("Correlation")

    save_png_pdf(fig, out_dir / "fig_{}_{}_maps".format(protocol_label.replace(" ", ""), metric))
    plt.close(fig)


def create_A_vs_B_compensation_maps(dfA, dfB, n_points_xy, n_points_z, out_dir, metric="rms"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    diff_arrays = [metric_array(dfB, c, metric) - metric_array(dfA, c, metric) for c in CASES]
    diff_abs = max(max(abs(float(np.nanmin(v))), abs(float(np.nanmax(v)))) for v in diff_arrays)

    im_diff = None
    for i, case in enumerate(CASES):
        ax = axes[0, i]
        diff = metric_array(dfB, case, metric) - metric_array(dfA, case, metric)
        im_diff = ax.imshow(
            to_grid(diff, n_points_xy, n_points_z),
            aspect="auto", origin="upper", vmin=-diff_abs, vmax=diff_abs, cmap="coolwarm"
        )
        ax.set_title("{}: B - A".format(CASE_DISPLAY[case]))
        set_heatmap_style(ax)
        panel_label(ax, chr(ord("A") + i))
    cbar1 = fig.colorbar(im_diff, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=28)
    cbar1.set_label("Delta {} [mV]".format(metric.replace("_", " ")))

    eps = 1e-12
    ratio_arrays = [
        metric_array(dfB, c, metric) / np.maximum(metric_array(dfA, c, metric), eps)
        for c in CASES
    ]
    concat_ratio = np.concatenate(ratio_arrays)
    ratio_min = max(0.0, float(np.nanpercentile(concat_ratio, 1.0)))
    ratio_max = float(np.nanpercentile(concat_ratio, 99.0))
    ratio_max = max(ratio_max, 1.0)

    im_ratio = None
    for i, case in enumerate(CASES):
        ax = axes[1, i]
        ratio = metric_array(dfB, case, metric) / np.maximum(metric_array(dfA, case, metric), eps)
        im_ratio = ax.imshow(
            to_grid(ratio, n_points_xy, n_points_z),
            aspect="auto", origin="upper", vmin=ratio_min, vmax=ratio_max
        )
        ax.set_title("{}: B / A".format(CASE_DISPLAY[case]))
        set_heatmap_style(ax)
        panel_label(ax, chr(ord("E") + i))
    cbar2 = fig.colorbar(im_ratio, ax=axes[1, :], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("{} ratio".format(metric.replace("_", " ")))

    save_png_pdf(fig, out_dir / "fig_protocolA_vs_B_{}_compensation".format(metric))
    plt.close(fig)


def create_global_A_vs_B_summary(summaryA, summaryB, out_dir):
    combined = pd.concat([summaryA, summaryB], ignore_index=True)

    metrics_to_plot = ["rms", "peak_to_peak", "max_abs", "corr_vs_healthy"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        sub = combined[combined["metric"] == metric]
        cases = DISEASE_CASES if metric == "corr_vs_healthy" else CASES

        x = np.arange(len(cases))
        width = 0.36
        meansA, stdsA, meansB, stdsB = [], [], [], []
        for case in cases:
            rowA = sub[(sub["protocol"] == "Protocol A") & (sub["case"] == case)]
            rowB = sub[(sub["protocol"] == "Protocol B") & (sub["case"] == case)]
            meansA.append(float(rowA["mean"].iloc[0]) if len(rowA) else np.nan)
            stdsA.append(float(rowA["std"].iloc[0]) if len(rowA) else 0.0)
            meansB.append(float(rowB["mean"].iloc[0]) if len(rowB) else np.nan)
            stdsB.append(float(rowB["std"].iloc[0]) if len(rowB) else 0.0)

        ax.bar(x - width / 2, meansA, width, yerr=stdsA, capsize=3, label="Protocol A")
        ax.bar(x + width / 2, meansB, width, yerr=stdsB, capsize=3, label="Protocol B")
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_DISPLAY[c] for c in cases], rotation=20)
        ax.set_title(METRIC_DISPLAY.get(metric, metric.replace("_", " ")))
        ax.grid(True, axis="y", alpha=0.25)
        if idx == 0:
            ax.legend()
        panel_label(ax, chr(ord("A") + idx))

    save_png_pdf(fig, out_dir / "fig_protocolA_vs_B_global_summary")
    plt.close(fig)

    combined.to_csv(out_dir / "protocolA_vs_protocolB_global_summary.csv", index=False)


def create_line_summary(dfA, dfB, out_dir):
    x = dfA["electrode"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.8))

    # Reserve a dedicated bottom margin for the shared legend. This keeps the
    # legend fully inside the exported figure without covering x-axis labels.
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.93,
        bottom=0.32,
        wspace=0.18,
    )

    for ax, metric, ylabel in zip(
        axes,
        ["rms", "peak_to_peak", "max_abs"],
        ["RMS [mV]", "Peak-to-peak [mV]", "Max |EMG| [mV]"],
    ):
        for case in CASES:
            ax.plot(x, metric_array(dfA, case, metric), linewidth=1.0, linestyle="--", label="A " + CASE_DISPLAY[case])
            ax.plot(x, metric_array(dfB, case, metric), linewidth=1.0, linestyle="-", label="B " + CASE_DISPLAY[case])
        ax.set_xlabel("Electrode index")
        ax.set_ylabel(ylabel)
        ax.set_title(METRIC_DISPLAY.get(metric, metric.replace("_", " ")))
        ax.grid(True, alpha=0.25)

    # Place one compact figure-level legend in the reserved bottom margin.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.125),
        ncol=4,
        fontsize=14,
        frameon=True,
    )
    save_png_pdf(fig, out_dir / "fig_protocolA_vs_B_line_summary")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Protocol A and B results in the same build_release/out folder.")
    parser.add_argument("--out-root", default="build_release/out", help="Root folder containing healthy/death_25/... and *_protocol_B folders.")
    parser.add_argument("--csv-name", default="electrodes.csv")
    parser.add_argument("--out-dir", default="protocol_AB_analysis")
    parser.add_argument("--n-points-xy", type=int, default=None, help="Electrodes across x direction. For 384 electrodes default is 12.")
    parser.add_argument("--subtract-mean", action="store_true", help="Subtract mean of first 10 samples from each trace before metrics.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    casesA, tA, nA, xyA, zA = load_protocol_from_folder_map(
        out_root,
        PROTOCOL_A_FOLDERS,
        "Protocol A",
        csv_name=args.csv_name,
        default_xy=args.n_points_xy,
    )
    casesB, tB, nB, xyB, zB = load_protocol_from_folder_map(
        out_root,
        PROTOCOL_B_FOLDERS,
        "Protocol B",
        csv_name=args.csv_name,
        default_xy=args.n_points_xy,
    )

    if nA != nB or xyA != xyB or zA != zB:
        raise ValueError("Protocol A and B electrode layouts differ.")
    if len(tA) != len(tB) or not np.allclose(tA, tB):
        raise ValueError("Protocol A and B time vectors differ.")

    dfA = compute_metrics_for_protocol(casesA, subtract_mean=args.subtract_mean)
    dfB = compute_metrics_for_protocol(casesB, subtract_mean=args.subtract_mean)

    dfA.to_csv(out_dir / "protocolA_metrics_by_electrode.csv", index=False)
    dfB.to_csv(out_dir / "protocolB_metrics_by_electrode.csv", index=False)

    summaryA = global_summary_from_metrics(dfA, "Protocol A")
    summaryB = global_summary_from_metrics(dfB, "Protocol B")
    summaryA.to_csv(out_dir / "protocolA_global_summary.csv", index=False)
    summaryB.to_csv(out_dir / "protocolB_global_summary.csv", index=False)

    create_protocol_maps(dfB, xyA, zA, out_dir, "ProtocolB", metric="rms")
    create_protocol_maps(dfB, xyA, zA, out_dir, "ProtocolB", metric="peak_to_peak")
    create_A_vs_B_compensation_maps(dfA, dfB, xyA, zA, out_dir, metric="rms")
    create_A_vs_B_compensation_maps(dfA, dfB, xyA, zA, out_dir, metric="peak_to_peak")
    create_global_A_vs_B_summary(summaryA, summaryB, out_dir)
    create_line_summary(dfA, dfB, out_dir)

    print("")
    print("Created Protocol A/B analysis in: {}".format(out_dir))
    print("Key files:")
    print("  {}".format(out_dir / "protocolA_metrics_by_electrode.csv"))
    print("  {}".format(out_dir / "protocolB_metrics_by_electrode.csv"))
    print("  {}".format(out_dir / "protocolA_vs_protocolB_global_summary.csv"))
    print("  {}".format(out_dir / "fig_protocolA_vs_B_rms_compensation.png"))
    print("  {}".format(out_dir / "fig_protocolA_vs_B_global_summary.png"))


if __name__ == "__main__":
    main()