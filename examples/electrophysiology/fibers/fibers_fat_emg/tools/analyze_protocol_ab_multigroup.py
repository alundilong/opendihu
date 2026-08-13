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
# Multi-group study helpers
# -----------------------------------------------------------------------------

GROUP_CASE_RE = re.compile(
    r"^g-(?P<group>\d+)_(?P<protocol>[AB])_(?P<stage>healthy|death_25|death_50|death_75)$"
)

STAGE_KILL_FRACTION = {
    "healthy": 0.00,
    "death_25": 0.25,
    "death_50": 0.50,
    "death_75": 0.75,
}


def discover_group_cases(out_root, csv_name="electrodes.csv"):
    """Discover grouped OpenDiHu cases in flat or nested directory layouts."""
    out_root = Path(out_root)
    if not out_root.exists():
        raise FileNotFoundError("Output root does not exist: {}".format(out_root))

    found = {}

    for csv_path in out_root.rglob(csv_name):
        parent = csv_path.parent

        # Flat layout:
        #   out/g-1_A_healthy/electrodes.csv
        m = GROUP_CASE_RE.match(parent.name)
        if m:
            key = (
                int(m.group("group")),
                m.group("protocol"),
                m.group("stage"),
            )
            if key in found and found[key] != csv_path:
                raise RuntimeError(
                    "Duplicate case {} discovered:\n  {}\n  {}".format(
                        key, found[key], csv_path
                    )
                )
            found[key] = csv_path
            continue

        # Nested layout:
        #   out/g-1/protocol_A/healthy/electrodes.csv
        parts = list(csv_path.parts)
        for i in range(len(parts) - 3):
            gmatch = re.match(r"^g-(\d+)$", parts[i])
            if not gmatch:
                continue
            pmatch = re.match(r"^protocol_([AB])$", parts[i + 1])
            if not pmatch:
                continue
            stage = parts[i + 2]
            if stage not in CASES:
                continue

            key = (int(gmatch.group(1)), pmatch.group(1), stage)
            if key in found and found[key] != csv_path:
                raise RuntimeError(
                    "Duplicate case {} discovered:\n  {}\n  {}".format(
                        key, found[key], csv_path
                    )
                )
            found[key] = csv_path
            break

    return found


def validate_discovered_groups(case_paths, requested_groups=None, allow_incomplete=False):
    all_groups = sorted({key[0] for key in case_paths})

    if requested_groups:
        groups = sorted(set(int(g) for g in requested_groups))
        missing_groups = [g for g in groups if g not in all_groups]
        if missing_groups:
            raise FileNotFoundError(
                "Requested groups were not discovered: {}".format(missing_groups)
            )
    else:
        groups = all_groups

    if not groups:
        raise RuntimeError("No grouped cases were discovered.")

    missing = []
    for g in groups:
        for protocol in ("A", "B"):
            for stage in CASES:
                if (g, protocol, stage) not in case_paths:
                    missing.append((g, protocol, stage))

    if missing and not allow_incomplete:
        lines = ["Missing required grouped cases:"]
        for g, protocol, stage in missing:
            lines.append("  g-{}_{}_{}".format(g, protocol, stage))
        raise FileNotFoundError("\n".join(lines))

    return groups, missing


def write_case_manifest(case_paths, groups, out_file):
    rows = []
    for g in groups:
        for protocol in ("A", "B"):
            for stage in CASES:
                path = case_paths.get((g, protocol, stage))
                rows.append({
                    "group_number": g,
                    "group_tag": "g-{}".format(g),
                    "protocol": protocol,
                    "stage": stage,
                    "kill_fraction": STAGE_KILL_FRACTION[stage],
                    "scenario_name": "g-{}_{}_{}".format(g, protocol, stage),
                    "electrodes_csv": "" if path is None else str(Path(path).resolve()),
                    "exists": bool(path is not None and Path(path).exists()),
                })
    pd.DataFrame(rows).to_csv(out_file, index=False)


def load_group_protocol(case_paths, group_number, protocol, default_xy=None):
    protocol = str(protocol).upper()
    cases = {}

    print("Loading g-{} Protocol {}".format(group_number, protocol))
    for stage in CASES:
        filename = case_paths.get((group_number, protocol, stage))
        if filename is None:
            raise FileNotFoundError(
                "Missing case g-{}_{}_{}".format(group_number, protocol, stage)
            )
        print("  {} -> {}".format(stage, filename))
        cases[stage] = read_electrodes_csv(filename, default_xy=default_xy)

    ref_t = cases["healthy"]["t"]
    ref_n = cases["healthy"]["n_points"]
    ref_xy = cases["healthy"]["n_points_xy"]
    ref_z = cases["healthy"]["n_points_z"]

    for stage in CASES[1:]:
        c = cases[stage]
        if c["n_points"] != ref_n:
            raise ValueError(
                "g-{} Protocol {} {}: n_points differs from healthy.".format(
                    group_number, protocol, stage
                )
            )
        if len(c["t"]) != len(ref_t) or not np.allclose(c["t"], ref_t):
            raise ValueError(
                "g-{} Protocol {} {}: time vector differs from healthy.".format(
                    group_number, protocol, stage
                )
            )

    return cases, ref_t, ref_n, ref_xy, ref_z


def restrict_cases_to_time_window(cases, start_ms=None, end_ms=None):
    """Restrict all cases to one common time interval."""
    ref_t = cases["healthy"]["t"]
    mask = np.ones(len(ref_t), dtype=bool)

    if start_ms is not None:
        mask &= ref_t >= float(start_ms)
    if end_ms is not None:
        mask &= ref_t <= float(end_ms)

    if not np.any(mask):
        raise ValueError(
            "Requested analysis window contains no samples: start_ms={}, end_ms={}".format(
                start_ms, end_ms
            )
        )

    result = {}
    for stage, c in cases.items():
        if len(c["t"]) != len(ref_t) or not np.allclose(c["t"], ref_t):
            raise ValueError("Time vectors differ before time-window restriction.")
        cc = dict(c)
        cc["t"] = c["t"][mask]
        cc["emg"] = c["emg"][mask, :]
        result[stage] = cc

    return result


def metrics_wide_to_long(df, group_number, protocol):
    rows = []

    for _, row in df.iterrows():
        electrode = int(row["electrode"])
        for stage in CASES:
            item = {
                "group_number": int(group_number),
                "group_tag": "g-{}".format(group_number),
                "protocol": str(protocol),
                "stage": stage,
                "kill_fraction": STAGE_KILL_FRACTION[stage],
                "electrode": electrode,
                "rms": float(row["{}_rms".format(stage)]),
                "peak_to_peak": float(row["{}_peak_to_peak".format(stage)]),
                "max_abs": float(row["{}_max_abs".format(stage)]),
                "mean": float(row["{}_mean".format(stage)]),
                "corr_vs_healthy": np.nan,
            }
            corr_col = "{}_corr_vs_healthy".format(stage)
            if corr_col in df.columns:
                item["corr_vs_healthy"] = float(row[corr_col])
            rows.append(item)

    return pd.DataFrame(rows)


def add_group_to_summary(summary, group_number, protocol_letter):
    out = summary.copy()
    out.insert(0, "group_number", int(group_number))
    out.insert(1, "group_tag", "g-{}".format(group_number))
    out["protocol_letter"] = str(protocol_letter)
    out["kill_fraction"] = out["case"].map(STAGE_KILL_FRACTION)
    return out


def build_paired_group_differences(group_summary):
    """
    Pair A/B using group as the independent simulation replicate.
    Electrode-level values are summarized within each group first.
    """
    rows = []

    for (g, stage, metric), sub in group_summary.groupby(
        ["group_number", "case", "metric"], dropna=False
    ):
        a = sub[sub["protocol"] == "Protocol A"]
        b = sub[sub["protocol"] == "Protocol B"]
        if a.empty or b.empty:
            continue

        a_mean = float(a["mean"].iloc[0])
        b_mean = float(b["mean"].iloc[0])

        rows.append({
            "group_number": int(g),
            "group_tag": "g-{}".format(int(g)),
            "stage": stage,
            "kill_fraction": STAGE_KILL_FRACTION[stage],
            "metric": metric,
            "protocol_A_mean": a_mean,
            "protocol_B_mean": b_mean,
            "B_minus_A": b_mean - a_mean,
            "B_over_A": np.nan if abs(a_mean) < 1e-15 else b_mean / a_mean,
            "percent_change_B_vs_A": (
                np.nan if abs(a_mean) < 1e-15
                else 100.0 * (b_mean - a_mean) / a_mean
            ),
        })

    return pd.DataFrame(rows)


def paired_statistics(paired_df):
    """
    Across-group paired A-vs-B statistics.

    For n=3 groups, p-values are exploratory. Descriptive paired differences,
    effect size, and CI are reported because they are more informative here.
    """
    try:
        from scipy import stats
    except Exception:
        stats = None

    rows = []

    for (stage, metric), sub in paired_df.groupby(["stage", "metric"]):
        sub = sub.sort_values("group_number")
        a = sub["protocol_A_mean"].to_numpy(dtype=float)
        b = sub["protocol_B_mean"].to_numpy(dtype=float)
        d = b - a
        n = len(d)

        mean_a = float(np.nanmean(a))
        mean_b = float(np.nanmean(b))
        mean_d = float(np.nanmean(d))
        sd_d = float(np.nanstd(d, ddof=1)) if n > 1 else np.nan
        sem_d = sd_d / np.sqrt(n) if n > 1 and np.isfinite(sd_d) else np.nan

        ci_low = np.nan
        ci_high = np.nan
        paired_t_p = np.nan
        wilcoxon_p = np.nan

        if stats is not None and n >= 2:
            try:
                tcrit = float(stats.t.ppf(0.975, df=n - 1))
                ci_low = mean_d - tcrit * sem_d
                ci_high = mean_d + tcrit * sem_d
            except Exception:
                pass

            try:
                paired_t_p = float(stats.ttest_rel(b, a, nan_policy="omit").pvalue)
            except Exception:
                pass

            try:
                if np.any(np.abs(d) > 0):
                    wilcoxon_p = float(stats.wilcoxon(b, a).pvalue)
                else:
                    wilcoxon_p = 1.0
            except Exception:
                pass

        cohen_dz = (
            np.nan if not np.isfinite(sd_d) or sd_d == 0
            else mean_d / sd_d
        )

        rows.append({
            "stage": stage,
            "kill_fraction": STAGE_KILL_FRACTION[stage],
            "metric": metric,
            "n_groups": n,
            "protocol_A_mean_across_groups": mean_a,
            "protocol_A_sd_across_groups": (
                float(np.nanstd(a, ddof=1)) if n > 1 else np.nan
            ),
            "protocol_B_mean_across_groups": mean_b,
            "protocol_B_sd_across_groups": (
                float(np.nanstd(b, ddof=1)) if n > 1 else np.nan
            ),
            "mean_B_minus_A": mean_d,
            "sd_B_minus_A": sd_d,
            "ci95_B_minus_A_low": ci_low,
            "ci95_B_minus_A_high": ci_high,
            "mean_percent_change_B_vs_A": float(
                np.nanmean(sub["percent_change_B_vs_A"].to_numpy(dtype=float))
            ),
            "cohen_dz": cohen_dz,
            "paired_t_p_exploratory": paired_t_p,
            "wilcoxon_p_exploratory": wilcoxon_p,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        stage_order = {stage: i for i, stage in enumerate(CASES)}
        metric_order = {
            "rms": 0,
            "peak_to_peak": 1,
            "max_abs": 2,
            "corr_vs_healthy": 3,
        }
        result["_stage_order"] = result["stage"].map(stage_order)
        result["_metric_order"] = result["metric"].map(metric_order).fillna(99)
        result = result.sort_values(
            ["_metric_order", "_stage_order"]
        ).drop(columns=["_stage_order", "_metric_order"])
    return result


def create_multigroup_protocol_summary(group_summary, out_dir):
    """
    Across-group mean ± SD. Error bars are across independent group realizations,
    not across electrodes.
    """
    metrics = ["rms", "peak_to_peak", "max_abs", "corr_vs_healthy"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        stages = DISEASE_CASES if metric == "corr_vs_healthy" else CASES
        x = np.arange(len(stages))

        for protocol_name in ("Protocol A", "Protocol B"):
            means = []
            stds = []
            for stage in stages:
                vals = group_summary[
                    (group_summary["protocol"] == protocol_name)
                    & (group_summary["case"] == stage)
                    & (group_summary["metric"] == metric)
                ]["mean"].to_numpy(dtype=float)

                means.append(float(np.nanmean(vals)) if len(vals) else np.nan)
                stds.append(
                    float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
                )

            ax.errorbar(
                x, means, yerr=stds,
                marker="o", capsize=4, linewidth=1.5,
                label=protocol_name,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([CASE_DISPLAY[s] for s in stages], rotation=20)
        ax.set_title(METRIC_DISPLAY.get(metric, metric))
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend()
        panel_label(ax, chr(ord("A") + idx))

    save_png_pdf(fig, out_dir / "fig_multigroup_protocol_summary")
    plt.close(fig)


def create_multigroup_paired_difference_plot(paired_df, out_dir):
    """Show B-A for every group plus the across-group mean."""
    metrics = ["rms", "peak_to_peak", "max_abs"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4))

    fig.subplots_adjust(
        left=0.06, right=0.985, top=0.92, bottom=0.30, wspace=0.20
    )

    for ax, metric in zip(axes, metrics):
        sub_metric = paired_df[paired_df["metric"] == metric]
        x = np.arange(len(CASES))

        for g in sorted(sub_metric["group_number"].unique()):
            sub_g = sub_metric[sub_metric["group_number"] == g]
            vals = []
            for stage in CASES:
                row = sub_g[sub_g["stage"] == stage]
                vals.append(
                    float(row["B_minus_A"].iloc[0]) if len(row) else np.nan
                )
            ax.plot(x, vals, marker="o", linewidth=1.2, label="g-{}".format(g))

        means = []
        for stage in CASES:
            vals = sub_metric[sub_metric["stage"] == stage][
                "B_minus_A"
            ].to_numpy(dtype=float)
            means.append(float(np.nanmean(vals)) if len(vals) else np.nan)

        ax.plot(
            x, means, marker="s", linewidth=2.2, linestyle="--",
            label="Across-group mean"
        )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_DISPLAY[s] for s in CASES], rotation=20)
        ax.set_title("B - A: {}".format(METRIC_DISPLAY.get(metric, metric)))
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=max(2, len(labels)),
    )

    save_png_pdf(fig, out_dir / "fig_multigroup_paired_B_minus_A")
    plt.close(fig)


def create_group_trajectory_plot(group_summary, out_dir):
    """Show each group's severity trajectory for Protocol A and B."""
    metrics = ["rms", "peak_to_peak", "max_abs"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4))

    fig.subplots_adjust(
        left=0.06, right=0.985, top=0.92, bottom=0.32, wspace=0.20
    )
    x = np.arange(len(CASES))

    for ax, metric in zip(axes, metrics):
        sub_metric = group_summary[group_summary["metric"] == metric]

        for protocol_name, linestyle in (
            ("Protocol A", "--"),
            ("Protocol B", "-"),
        ):
            for g in sorted(sub_metric["group_number"].unique()):
                vals = []
                for stage in CASES:
                    row = sub_metric[
                        (sub_metric["group_number"] == g)
                        & (sub_metric["protocol"] == protocol_name)
                        & (sub_metric["case"] == stage)
                    ]
                    vals.append(
                        float(row["mean"].iloc[0]) if len(row) else np.nan
                    )
                ax.plot(
                    x, vals,
                    marker="o",
                    linestyle=linestyle,
                    linewidth=1.0,
                    label="{} g-{}".format(
                        "A" if protocol_name == "Protocol A" else "B", g
                    ),
                )

        ax.set_xticks(x)
        ax.set_xticklabels([CASE_DISPLAY[s] for s in CASES], rotation=20)
        ax.set_title(METRIC_DISPLAY.get(metric, metric))
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=3,
    )
    save_png_pdf(fig, out_dir / "fig_group_trajectories")
    plt.close(fig)



# -----------------------------------------------------------------------------
# Manuscript Figures 19-22: group-level Protocol A vs Protocol B comparison
# -----------------------------------------------------------------------------


def _group_metric_values(group_summary, protocol_name, stage, metric):
    """Return one spatially averaged metric value per independent group."""
    vals = group_summary[
        (group_summary["protocol"] == protocol_name)
        & (group_summary["case"] == stage)
        & (group_summary["metric"] == metric)
    ]["mean"].to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def create_figure19_global_protocol_comparison(group_summary, out_dir):
    """Figure 19-style global A/B comparison using group as the replicate.

    Bar heights are means of the group-level spatial means. Error bars are SD
    across independent stochastic groups, not SD across electrodes.
    """
    panels = [
        ("rms", "RMS", "RMS [mV]", CASES),
        ("peak_to_peak", "Peak-to-peak amplitude", "Peak-to-peak [mV]", CASES),
        ("max_abs", "Maximum absolute EMG", "Max |EMG| [mV]", CASES),
        ("corr_vs_healthy", "Correlation vs Healthy", "Correlation", DISEASE_CASES),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    axes = axes.ravel()
    width = 0.36

    export_rows = []

    for idx, (metric, title, ylabel, stages) in enumerate(panels):
        ax = axes[idx]
        x = np.arange(len(stages))

        for j, protocol_name in enumerate(("Protocol A", "Protocol B")):
            means = []
            stds = []
            ns = []
            for stage in stages:
                vals = _group_metric_values(group_summary, protocol_name, stage, metric)
                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                ns.append(int(len(vals)))
                export_rows.append({
                    "protocol": protocol_name,
                    "stage": stage,
                    "metric": metric,
                    "n_groups": int(len(vals)),
                    "mean_across_groups": float(np.mean(vals)) if len(vals) else np.nan,
                    "sd_across_groups": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                })

            offset = (-width / 2.0) if j == 0 else (width / 2.0)
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=protocol_name,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([CASE_DISPLAY[s] for s in stages], rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        panel_label(ax, chr(ord("A") + idx))
        if idx == 0:
            ax.legend(loc="best")

    fig.suptitle(
        "Global EMG metric comparison between Protocol A and Protocol B\n"
        "mean ± SD across independent groups"
    )
    save_png_pdf(fig, out_dir / "figure19_global_protocol_A_vs_B")
    plt.close(fig)

    pd.DataFrame(export_rows).to_csv(
        out_dir / "figure19_global_protocol_A_vs_B_values.csv", index=False
    )


def _electrode_group_profile(long_df, protocol, stage, metric):
    """Mean and SD across groups at each electrode for one protocol/stage/metric."""
    sub = long_df[
        (long_df["protocol"] == protocol)
        & (long_df["stage"] == stage)
    ][["group_number", "electrode", metric]].copy()
    sub = sub[np.isfinite(sub[metric].to_numpy(dtype=float))]

    grouped = sub.groupby("electrode", sort=True)[metric]
    mean = grouped.mean()
    sd = grouped.std(ddof=1).fillna(0.0)
    return mean.index.to_numpy(dtype=int), mean.to_numpy(dtype=float), sd.to_numpy(dtype=float)


def create_figure20_electrode_wise_protocol_comparison(long_df, out_dir):
    """Figure 20-style electrode-wise A/B comparison at group level.

    Lines show the across-group mean at each electrode. Protocol A/B are
    distinguished by line style while the MU-loss condition keeps the same color.
    """
    panels = [
        ("rms", "RMS", "RMS [mV]"),
        ("peak_to_peak", "Peak-to-peak amplitude", "Peak-to-peak [mV]"),
        ("max_abs", "Maximum absolute EMG", "Max |EMG| [mV]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.6))
    # Reserve a dedicated title band above the subplot titles and a legend band below.
    fig.subplots_adjust(left=0.055, right=0.99, top=0.79, bottom=0.28, wspace=0.20)

    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    export_frames = []

    for ax, (metric, title, ylabel) in zip(axes, panels):
        for stage_idx, stage in enumerate(CASES):
            color = cycle_colors[stage_idx % len(cycle_colors)] if cycle_colors else None
            for protocol, linestyle in (("A", "--"), ("B", "-")):
                electrode, mean, sd = _electrode_group_profile(
                    long_df, protocol, stage, metric
                )
                label = "{} {}".format(protocol, CASE_DISPLAY[stage])
                ax.plot(
                    electrode,
                    mean,
                    linestyle=linestyle,
                    linewidth=1.25,
                    color=color,
                    label=label,
                )

                export_frames.append(pd.DataFrame({
                    "metric": metric,
                    "protocol": protocol,
                    "stage": stage,
                    "electrode": electrode,
                    "mean_across_groups": mean,
                    "sd_across_groups": sd,
                }))

        ax.set_xlabel("Electrode index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=4,
        fontsize=14,
        frameon=True,
    )
    fig.suptitle(
        "Electrode-wise metric comparison between Protocol A and Protocol B\n"
        "Lines show mean across independent groups",
        y=0.97,
    )
    save_png_pdf(fig, out_dir / "figure20_electrode_wise_protocol_A_vs_B")
    plt.close(fig)

    if export_frames:
        pd.concat(export_frames, ignore_index=True).to_csv(
            out_dir / "figure20_electrode_wise_protocol_A_vs_B_values.csv",
            index=False,
        )


def _paired_electrode_effects(long_df, metric):
    """Paired B-A and B/A at group × stage × electrode level."""
    sub = long_df[["group_number", "stage", "electrode", "protocol", metric]].copy()
    sub = sub[np.isfinite(sub[metric].to_numpy(dtype=float))]

    pivot = sub.pivot_table(
        index=["group_number", "stage", "electrode"],
        columns="protocol",
        values=metric,
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    if "A" not in pivot.columns or "B" not in pivot.columns:
        raise ValueError(
            "Both Protocol A and Protocol B are required to compute paired electrode effects."
        )

    pivot["B_minus_A"] = pivot["B"] - pivot["A"]
    denom = np.abs(pivot["A"].to_numpy(dtype=float))
    pivot["B_over_A"] = np.where(
        denom > 1e-15,
        pivot["B"].to_numpy(dtype=float) / pivot["A"].to_numpy(dtype=float),
        np.nan,
    )
    return pivot


def _stage_grid_from_effects(effect_df, stage, field, n_points_xy, n_points_z):
    sub = effect_df[effect_df["stage"] == stage]
    grouped = sub.groupby("electrode", sort=True)[field].mean()
    expected = int(n_points_xy) * int(n_points_z)
    if len(grouped) != expected:
        raise ValueError(
            "Cannot create group-level spatial map for {}: expected {} electrodes, got {}.".format(
                stage, expected, len(grouped)
            )
        )
    values = grouped.reindex(np.arange(expected)).to_numpy(dtype=float)
    return to_grid(values, n_points_xy, n_points_z)


def create_compensation_map_figure(
    long_df,
    out_dir,
    metric,
    metric_display,
    difference_unit,
    filename,
    n_points_xy,
    n_points_z,
):
    """Create Figure 21/22-style paired B-A and B/A maps averaged across groups."""
    effect_df = _paired_electrode_effects(long_df, metric)

    diff_grids = [
        _stage_grid_from_effects(effect_df, stage, "B_minus_A", n_points_xy, n_points_z)
        for stage in CASES
    ]
    ratio_grids = [
        _stage_grid_from_effects(effect_df, stage, "B_over_A", n_points_xy, n_points_z)
        for stage in CASES
    ]

    finite_diff = np.concatenate([g[np.isfinite(g)] for g in diff_grids])
    diff_abs = float(np.max(np.abs(finite_diff))) if finite_diff.size else 1.0
    if diff_abs <= 0:
        diff_abs = 1e-12

    finite_ratio = np.concatenate([g[np.isfinite(g)] for g in ratio_grids])
    if finite_ratio.size:
        ratio_vmin = float(np.nanmin(finite_ratio))
        ratio_vmax = float(np.nanmax(finite_ratio))
    else:
        ratio_vmin, ratio_vmax = 0.95, 1.05
    ratio_vmin = min(ratio_vmin, 1.0)
    ratio_vmax = max(ratio_vmax, 1.0)
    if ratio_vmax <= ratio_vmin:
        ratio_vmin, ratio_vmax = 0.95, 1.05

    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    im_diff = None
    for i, (stage, grid) in enumerate(zip(CASES, diff_grids)):
        ax = axes[0, i]
        im_diff = ax.imshow(
            grid,
            aspect="auto",
            origin="upper",
            vmin=-diff_abs,
            vmax=diff_abs,
            cmap="coolwarm",
        )
        ax.set_title("{}: B - A".format(CASE_DISPLAY[stage]))
        ax.set_xlabel("Electrode x index")
        ax.set_ylabel("Electrode y index")
        panel_label(ax, chr(ord("A") + i))

    cbar1 = fig.colorbar(im_diff, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=28)
    cbar1.set_label("B - A {}{}".format(metric_display, difference_unit))

    im_ratio = None
    for i, (stage, grid) in enumerate(zip(CASES, ratio_grids)):
        ax = axes[1, i]
        im_ratio = ax.imshow(
            grid,
            aspect="auto",
            origin="upper",
            vmin=ratio_vmin,
            vmax=ratio_vmax,
        )
        ax.set_title("{}: B / A".format(CASE_DISPLAY[stage]))
        ax.set_xlabel("Electrode x index")
        ax.set_ylabel("Electrode y index")
        panel_label(ax, chr(ord("E") + i))

    cbar2 = fig.colorbar(im_ratio, ax=axes[1, :], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("{} ratio (B / A)".format(metric_display))

    fig.suptitle(
        "Protocol B compensation effect relative to Protocol A for {}\n"
        "paired within group, then averaged across groups".format(metric_display)
    )
    save_png_pdf(fig, out_dir / filename)
    plt.close(fig)

    effect_df.to_csv(out_dir / (filename + "_paired_values.csv"), index=False)


def create_figure21_peak_to_peak_compensation(
    long_df, out_dir, n_points_xy, n_points_z
):
    create_compensation_map_figure(
        long_df=long_df,
        out_dir=out_dir,
        metric="peak_to_peak",
        metric_display="peak-to-peak amplitude",
        difference_unit=" [mV]",
        filename="figure21_peak_to_peak_compensation_B_vs_A",
        n_points_xy=n_points_xy,
        n_points_z=n_points_z,
    )


def create_figure22_rms_compensation(long_df, out_dir, n_points_xy, n_points_z):
    create_compensation_map_figure(
        long_df=long_df,
        out_dir=out_dir,
        metric="rms",
        metric_display="RMS amplitude",
        difference_unit=" [mV]",
        filename="figure22_rms_compensation_B_vs_A",
        n_points_xy=n_points_xy,
        n_points_z=n_points_z,
    )


def analyze_one_group(case_paths, group_number, out_dir, args):
    group_dir = out_dir / "g-{}".format(group_number)
    group_dir.mkdir(parents=True, exist_ok=True)

    casesA, tA, nA, xyA, zA = load_group_protocol(
        case_paths, group_number, "A", default_xy=args.n_points_xy
    )
    casesB, tB, nB, xyB, zB = load_group_protocol(
        case_paths, group_number, "B", default_xy=args.n_points_xy
    )

    if nA != nB or xyA != xyB or zA != zB:
        raise ValueError(
            "g-{}: Protocol A/B electrode layouts differ.".format(group_number)
        )
    if len(tA) != len(tB) or not np.allclose(tA, tB):
        raise ValueError(
            "g-{}: Protocol A/B time vectors differ.".format(group_number)
        )

    if args.analysis_start_ms is not None or args.analysis_end_ms is not None:
        casesA = restrict_cases_to_time_window(
            casesA, args.analysis_start_ms, args.analysis_end_ms
        )
        casesB = restrict_cases_to_time_window(
            casesB, args.analysis_start_ms, args.analysis_end_ms
        )

    dfA = compute_metrics_for_protocol(casesA, subtract_mean=args.subtract_mean)
    dfB = compute_metrics_for_protocol(casesB, subtract_mean=args.subtract_mean)

    dfA.to_csv(group_dir / "protocolA_metrics_by_electrode.csv", index=False)
    dfB.to_csv(group_dir / "protocolB_metrics_by_electrode.csv", index=False)

    summaryA = global_summary_from_metrics(dfA, "Protocol A")
    summaryB = global_summary_from_metrics(dfB, "Protocol B")

    summaryA.to_csv(group_dir / "protocolA_global_summary.csv", index=False)
    summaryB.to_csv(group_dir / "protocolB_global_summary.csv", index=False)

    if not args.skip_per_group_figures:
        # Preserve the old script's single-group figures, now group-specific.
        create_protocol_maps(dfA, xyA, zA, group_dir, "ProtocolA", metric="rms")
        create_protocol_maps(dfA, xyA, zA, group_dir, "ProtocolA", metric="peak_to_peak")
        create_protocol_maps(dfB, xyA, zA, group_dir, "ProtocolB", metric="rms")
        create_protocol_maps(dfB, xyA, zA, group_dir, "ProtocolB", metric="peak_to_peak")
        create_A_vs_B_compensation_maps(dfA, dfB, xyA, zA, group_dir, metric="rms")
        create_A_vs_B_compensation_maps(dfA, dfB, xyA, zA, group_dir, metric="peak_to_peak")
        create_global_A_vs_B_summary(summaryA, summaryB, group_dir)
        create_line_summary(dfA, dfB, group_dir)
    else:
        pd.concat([summaryA, summaryB], ignore_index=True).to_csv(
            group_dir / "protocolA_vs_protocolB_global_summary.csv",
            index=False,
        )

    wideA = dfA.copy()
    wideA.insert(0, "protocol", "A")
    wideA.insert(0, "group_tag", "g-{}".format(group_number))
    wideA.insert(0, "group_number", int(group_number))

    wideB = dfB.copy()
    wideB.insert(0, "protocol", "B")
    wideB.insert(0, "group_tag", "g-{}".format(group_number))
    wideB.insert(0, "group_number", int(group_number))

    longA = metrics_wide_to_long(dfA, group_number, "A")
    longB = metrics_wide_to_long(dfB, group_number, "B")

    group_summary = pd.concat(
        [
            add_group_to_summary(summaryA, group_number, "A"),
            add_group_to_summary(summaryB, group_number, "B"),
        ],
        ignore_index=True,
    )

    return {
        "wide": pd.concat([wideA, wideB], ignore_index=True),
        "long": pd.concat([longA, longB], ignore_index=True),
        "summary": group_summary,
        "grid_xy": xyA,
        "grid_z": zA,
        "n_electrodes": nA,
        "time_start_ms": float(casesA["healthy"]["t"][0]),
        "time_end_ms": float(casesA["healthy"]["t"][-1]),
        "n_time_samples": len(casesA["healthy"]["t"]),
    }


def write_analysis_readme(out_dir, groups, args):
    lines = [
        "Multi-group Protocol A/B EMG analysis",
        "",
        "Groups analyzed: {}".format(", ".join("g-{}".format(g) for g in groups)),
        "Number of groups: {}".format(len(groups)),
        "Cases analyzed: {}".format(8 * len(groups)),
        "",
        "Statistical unit:",
        "  The independent simulation replicate is group.",
        "  Electrodes are spatial measurements within a group and are not treated",
        "  as independent replicate samples in the across-group paired statistics.",
        "",
        "Analysis window:",
        "  start_ms = {}".format(args.analysis_start_ms),
        "  end_ms   = {}".format(args.analysis_end_ms),
        "",
        "Key outputs:",
        "  case_manifest.csv",
        "  all_groups_metrics_by_electrode_wide.csv",
        "  all_groups_metrics_by_electrode_long.csv",
        "  group_level_summary.csv",
        "  paired_protocol_group_differences.csv",
        "  paired_protocol_statistics.csv",
        "  fig_multigroup_protocol_summary.png/pdf",
        "  fig_multigroup_paired_B_minus_A.png/pdf",
        "  fig_group_trajectories.png/pdf",
        "",
        "Statistical caution:",
        "  With only three groups, inferential p-values have very low power.",
        "  Emphasize paired group-level differences, effect sizes, confidence",
        "  intervals, and consistency of trends across the three realizations.",
    ]
    (out_dir / "README_analysis.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze repeated Protocol A/B OpenDiHu EMG simulations across "
            "multiple independent groups (g-1, g-2, ...)."
        )
    )
    parser.add_argument(
        "--out-root",
        default="build_release/out",
        help=(
            "Root containing grouped output folders such as "
            "g-1_A_healthy, g-1_B_death_25, ..."
        ),
    )
    parser.add_argument("--csv-name", default="electrodes.csv")
    parser.add_argument(
        "--out-dir",
        default="protocol_AB_multigroup_analysis",
        help="Root analysis output directory.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="*",
        default=None,
        help="Optional group numbers to analyze. Default: auto-discover all.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow discovery to report incomplete groups; only complete groups are analyzed.",
    )
    parser.add_argument(
        "--n-points-xy",
        type=int,
        default=None,
        help="Electrodes across x direction. For 384 electrodes default is 12.",
    )
    parser.add_argument(
        "--subtract-mean",
        action="store_true",
        help="Subtract mean of first 10 samples from each trace before metrics.",
    )
    parser.add_argument(
        "--analysis-start-ms",
        type=float,
        default=None,
        help="Optional common analysis-window start time [ms].",
    )
    parser.add_argument(
        "--analysis-end-ms",
        type=float,
        default=None,
        help="Optional common analysis-window end time [ms].",
    )
    parser.add_argument(
        "--skip-per-group-figures",
        action="store_true",
        help="Compute metrics/statistics but skip the large set of per-group figures.",
    )
    parser.add_argument(
        "--skip-figures-19-22",
        action="store_true",
        help="Skip the manuscript-style group-level Protocol A/B Figures 19-22.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover grouped cases and create a manifest.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_paths = discover_group_cases(out_root, csv_name=args.csv_name)
    groups, missing = validate_discovered_groups(
        case_paths,
        requested_groups=args.groups,
        allow_incomplete=args.allow_incomplete,
    )

    print("")
    print("Discovered grouped study:")
    for g in groups:
        n_found = sum(
            (g, protocol, stage) in case_paths
            for protocol in ("A", "B")
            for stage in CASES
        )
        print("  g-{}: {}/8 cases".format(g, n_found))

    if missing:
        print("")
        print("Incomplete cases:")
        for g, protocol, stage in missing:
            print("  g-{}_{}_{}".format(g, protocol, stage))

    write_case_manifest(case_paths, groups, out_dir / "case_manifest.csv")

    if args.dry_run:
        print("")
        print("Dry run complete.")
        print("Manifest: {}".format(out_dir / "case_manifest.csv"))
        return

    complete_groups = []
    for g in groups:
        complete = all(
            (g, protocol, stage) in case_paths
            for protocol in ("A", "B")
            for stage in CASES
        )
        if complete:
            complete_groups.append(g)

    if not complete_groups:
        raise RuntimeError("No complete 8-case groups are available for analysis.")

    all_wide = []
    all_long = []
    all_group_summaries = []
    processing_rows = []
    reference_grid = None

    for g in complete_groups:
        print("")
        print("=" * 76)
        print("Processing g-{}".format(g))
        print("=" * 76)

        result = analyze_one_group(case_paths, g, out_dir, args)

        all_wide.append(result["wide"])
        all_long.append(result["long"])
        all_group_summaries.append(result["summary"])

        grid = (result["grid_xy"], result["grid_z"], result["n_electrodes"])
        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError(
                "Electrode grid differs between groups: {} vs {}".format(
                    reference_grid, grid
                )
            )

        processing_rows.append({
            "group_number": g,
            "group_tag": "g-{}".format(g),
            "n_cases": 8,
            "n_electrodes": result["n_electrodes"],
            "grid_xy": result["grid_xy"],
            "grid_z": result["grid_z"],
            "analysis_time_start_ms": result["time_start_ms"],
            "analysis_time_end_ms": result["time_end_ms"],
            "n_time_samples": result["n_time_samples"],
            "subtract_mean": bool(args.subtract_mean),
        })

    wide_df = pd.concat(all_wide, ignore_index=True)
    long_df = pd.concat(all_long, ignore_index=True)
    group_summary = pd.concat(all_group_summaries, ignore_index=True)

    wide_df.to_csv(
        out_dir / "all_groups_metrics_by_electrode_wide.csv",
        index=False,
    )
    long_df.to_csv(
        out_dir / "all_groups_metrics_by_electrode_long.csv",
        index=False,
    )
    group_summary.to_csv(
        out_dir / "group_level_summary.csv",
        index=False,
    )
    pd.DataFrame(processing_rows).to_csv(
        out_dir / "processing_summary.csv",
        index=False,
    )

    paired_df = build_paired_group_differences(group_summary)
    paired_df.to_csv(
        out_dir / "paired_protocol_group_differences.csv",
        index=False,
    )

    stats_df = paired_statistics(paired_df)
    stats_df.to_csv(
        out_dir / "paired_protocol_statistics.csv",
        index=False,
    )

    create_multigroup_protocol_summary(group_summary, out_dir)
    create_multigroup_paired_difference_plot(paired_df, out_dir)
    create_group_trajectory_plot(group_summary, out_dir)

    if not args.skip_figures_19_22:
        if reference_grid is None:
            raise RuntimeError("Electrode grid was not resolved; cannot create Figures 19-22.")
        grid_xy, grid_z, _ = reference_grid
        create_figure19_global_protocol_comparison(group_summary, out_dir)
        create_figure20_electrode_wise_protocol_comparison(long_df, out_dir)
        create_figure21_peak_to_peak_compensation(
            long_df, out_dir, grid_xy, grid_z
        )
        create_figure22_rms_compensation(
            long_df, out_dir, grid_xy, grid_z
        )

    write_analysis_readme(out_dir, complete_groups, args)

    print("")
    print("=" * 76)
    print("Multi-group analysis complete")
    print("=" * 76)
    print(
        "Groups analyzed : {}".format(
            ", ".join("g-{}".format(g) for g in complete_groups)
        )
    )
    print("Cases analyzed  : {}".format(8 * len(complete_groups)))
    print("Output directory: {}".format(out_dir.resolve()))
    print("")
    print("Key statistical files:")
    print("  {}".format(out_dir / "group_level_summary.csv"))
    print("  {}".format(out_dir / "paired_protocol_group_differences.csv"))
    print("  {}".format(out_dir / "paired_protocol_statistics.csv"))
    if not args.skip_figures_19_22:
        print("")
        print("Manuscript-style Protocol A/B figures:")
        print("  {}".format(out_dir / "figure19_global_protocol_A_vs_B.png"))
        print("  {}".format(out_dir / "figure20_electrode_wise_protocol_A_vs_B.png"))
        print("  {}".format(out_dir / "figure21_peak_to_peak_compensation_B_vs_A.png"))
        print("  {}".format(out_dir / "figure22_rms_compensation_B_vs_A.png"))

    if len(complete_groups) <= 3:
        print("")
        print(
            "Note: only {} independent groups are available. "
            "Treat inferential p-values as exploratory; emphasize paired "
            "effect sizes and consistency across groups.".format(
                len(complete_groups)
            )
        )


if __name__ == "__main__":
    main()
