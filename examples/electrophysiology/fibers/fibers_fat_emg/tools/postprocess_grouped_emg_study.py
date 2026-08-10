#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess a repeated, grouped OpenDiHu denervation study.

This script replaces the previous four-command workflow:

    python tools/compare_electrodes_four_cases.py
    python tools/compare_electrodes_four_cases_protocol_B.py
    python tools/publication_style_emg_figures.py --csv ... --out-dir ...
    python tools/publication_style_emg_figures.py --csv ... --out-dir ...

with one command that automatically discovers all statistical groups and both
Protocols A and B, processes the four denervation stages within each
(group, protocol), creates the same electrode-level metrics/figures, and then
combines all groups into analysis-ready CSV files.

Supported input layouts
-----------------------
The discovery logic is intentionally flexible. Examples that are recognized:

Flat scenario-name layout (recommended/current):
    build_release/out/g-1_A_healthy/electrodes.csv
    build_release/out/g-1_A_death_25/electrodes.csv
    ...
    build_release/out/g-1_B_death_75/electrodes.csv
    build_release/out/g-2_A_healthy/electrodes.csv
    ...

Nested layout:
    build_release/out/g-1/protocol_A/healthy/electrodes.csv
    build_release/out/g-1/protocol_B/death_50/electrodes.csv

Legacy single-group layout is also recognized:
    build_release/out/healthy/electrodes.csv
    build_release/out/death_25/electrodes.csv
    build_release/out/healthy_protocol_B/electrodes.csv
    build_release/out/death_25_protocol_B/electrodes.csv

Default output layout
---------------------
    grouped_emg_postprocessing/
      processing_manifest.csv
      all_groups_metrics_by_electrode_wide.csv
      all_groups_metrics_by_electrode_long.csv
      group_level_summary.csv
      paired_protocol_summary.csv
      aggregate_group_summary.png
      aggregate_group_summary.pdf
      g-1/
        protocol_A/
          emg_overlay_by_electrode/
            electrode_000_y00_x00.png
            ...
            metrics_by_electrode.csv
          metric_plots/
            figure1_spatial_rms_and_correlation.png/.pdf
            figure2_metric_summaries.png/.pdf
            figure3_peak_to_peak_maps.png/.pdf
            supplementary_correlation_boxplot.png/.pdf
            publication_summary_table.csv
        protocol_B/
          ...
      g-2/
        ...

Statistical interpretation
--------------------------
The script treats *group* as the independent stochastic realization/replicate.
Electrodes are preserved as spatial measurements and are NOT silently treated as
independent biological/statistical replicates.  group_level_summary.csv is
therefore especially useful for downstream between-group statistical analysis.

Example
-------
    python tools/postprocess_grouped_emg_study.py \
        --base-dir build_release/out \
        --out-dir grouped_emg_postprocessing

For a quick metrics-only pass without thousands of electrode PNGs:
    python tools/postprocess_grouped_emg_study.py \
        --base-dir build_release/out \
        --out-dir grouped_emg_postprocessing \
        --skip-electrode-plots

Select groups/protocols:
    python tools/postprocess_grouped_emg_study.py \
        --base-dir build_release/out \
        --groups 1 2 3 4 \
        --protocols A B
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams.update({"font.size": 14})

STAGES: List[Tuple[str, float]] = [
    ("healthy", 0.00),
    ("death_25", 0.25),
    ("death_50", 0.50),
    ("death_75", 0.75),
]
STAGE_NAMES = [s for s, _ in STAGES]
DISEASE_CASES = ["death_25", "death_50", "death_75"]
PROTOCOLS = ["A", "B"]

CASE_DISPLAY = {
    "healthy": "Healthy",
    "death_25": "25% MU death",
    "death_50": "50% MU death",
    "death_75": "75% MU death",
}

KILL_FRACTION = dict(STAGES)


@dataclass(frozen=True)
class CaseRef:
    group_number: int
    protocol: str
    stage: str
    csv_path: Path
    scenario_name: str

    @property
    def key(self) -> Tuple[int, str, str]:
        return (self.group_number, self.protocol, self.stage)


# -----------------------------------------------------------------------------
# Input discovery
# -----------------------------------------------------------------------------


def _normalize_path_text(path: Path) -> str:
    return "/".join(path.parts).replace("\\", "/")


def classify_case_path(csv_path: Path, base_dir: Path) -> Optional[CaseRef]:
    """Infer group/protocol/stage from a discovered electrodes.csv path."""
    try:
        rel = csv_path.relative_to(base_dir)
    except ValueError:
        rel = csv_path

    text = _normalize_path_text(rel)
    lower = text.lower()

    # 1) Current flat scenario name: g-1_A_healthy, g-12_B_death_50, etc.
    m = re.search(
        r"g[-_](\d+)[_-](?:protocol[_-]?)?([ab])[_-](healthy|death[_-]?25|death[_-]?50|death[_-]?75)",
        lower,
    )
    if m:
        group = int(m.group(1))
        protocol = m.group(2).upper()
        stage = m.group(3).replace("-", "_")
        sid = f"g-{group}_{protocol}_{stage}"
        return CaseRef(group, protocol, stage, csv_path, sid)

    # 2) Nested: .../g-1/protocol_A/healthy/electrodes.csv
    group_match = re.search(r"(?:^|/)g[-_](\d+)(?:/|$)", lower)
    protocol_match = re.search(r"(?:^|/)protocol[_-]?([ab])(?:/|$)", lower)
    stage_match = re.search(r"(?:^|/)(healthy|death[_-]?25|death[_-]?50|death[_-]?75)(?:/|$)", lower)
    if group_match and protocol_match and stage_match:
        group = int(group_match.group(1))
        protocol = protocol_match.group(1).upper()
        stage = stage_match.group(1).replace("-", "_")
        sid = f"g-{group}_{protocol}_{stage}"
        return CaseRef(group, protocol, stage, csv_path, sid)

    # 3) Group directory plus legacy B folder such as healthy_protocol_B.
    if group_match:
        group = int(group_match.group(1))
        for stage in STAGE_NAMES:
            if re.search(rf"(?:^|/){re.escape(stage)}_protocol[_-]?b(?:/|$)", lower):
                return CaseRef(group, "B", stage, csv_path, f"g-{group}_B_{stage}")
        # A case directly under g-X/<stage>/
        if stage_match:
            stage = stage_match.group(1).replace("-", "_")
            return CaseRef(group, "A", stage, csv_path, f"g-{group}_A_{stage}")

    # 4) Backward-compatible original single-group layout.
    # Only accept exact folder tokens to minimize false positives.
    for stage in STAGE_NAMES:
        if re.search(rf"(?:^|/){re.escape(stage)}_protocol[_-]?b(?:/|$)", lower):
            return CaseRef(1, "B", stage, csv_path, f"g-1_B_{stage}")
        if re.search(rf"(?:^|/){re.escape(stage)}(?:/|$)", lower):
            return CaseRef(1, "A", stage, csv_path, f"g-1_A_{stage}")

    return None


def discover_cases(base_dir: Path, csv_name: str) -> Dict[Tuple[int, str, str], CaseRef]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    found_paths = sorted(base_dir.rglob(csv_name))
    if not found_paths:
        raise FileNotFoundError(
            f'No "{csv_name}" files were found recursively under: {base_dir}'
        )

    cases: Dict[Tuple[int, str, str], CaseRef] = {}
    ignored: List[Path] = []

    for p in found_paths:
        ref = classify_case_path(p, base_dir)
        if ref is None:
            ignored.append(p)
            continue
        if ref.key in cases:
            old = cases[ref.key]
            raise RuntimeError(
                "Multiple electrode files map to the same study case:\n"
                f"  case: g-{ref.group_number}, Protocol {ref.protocol}, {ref.stage}\n"
                f"  first:  {old.csv_path}\n"
                f"  second: {ref.csv_path}\n"
                "Remove/rename duplicates or point --base-dir at a more specific output directory."
            )
        cases[ref.key] = ref

    if not cases:
        sample = "\n".join(f"  {p}" for p in found_paths[:10])
        raise RuntimeError(
            "Electrode CSV files were found, but none matched a recognized grouped case naming layout.\n"
            f"First discovered files:\n{sample}"
        )

    if ignored:
        print(f"Note: ignored {len(ignored)} unclassified {csv_name} file(s).")

    return cases


def select_and_validate_cases(
    cases: Dict[Tuple[int, str, str], CaseRef],
    requested_groups: Optional[Sequence[int]],
    requested_protocols: Sequence[str],
    allow_incomplete: bool,
) -> Tuple[List[int], List[str], Dict[Tuple[int, str, str], CaseRef]]:
    available_groups = sorted({k[0] for k in cases})
    groups = available_groups if not requested_groups else sorted(set(int(g) for g in requested_groups))
    protocols = [p.upper() for p in requested_protocols]

    missing_groups = [g for g in groups if g not in available_groups]
    if missing_groups:
        raise ValueError(
            f"Requested group(s) not found: {missing_groups}. Available groups: {available_groups}"
        )

    selected = {
        k: v for k, v in cases.items()
        if k[0] in groups and k[1] in protocols
    }

    missing: List[Tuple[int, str, str]] = []
    for g in groups:
        for p in protocols:
            for stage in STAGE_NAMES:
                if (g, p, stage) not in selected:
                    missing.append((g, p, stage))

    if missing and not allow_incomplete:
        lines = "\n".join(f"  g-{g} Protocol {p}: {stage}" for g, p, stage in missing)
        raise RuntimeError(
            "The selected study set is incomplete. Missing cases:\n"
            f"{lines}\n"
            "Use --allow-incomplete only for diagnostic/partial processing."
        )

    return groups, protocols, selected


# -----------------------------------------------------------------------------
# OpenDiHu electrodes.csv parser (preserves the behavior of the existing tools)
# -----------------------------------------------------------------------------


def parse_float_list(tokens: List[str]) -> List[float]:
    values: List[float] = []
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


def infer_electrode_grid(
    electrode_positions: Optional[np.ndarray],
    n_points: int,
    default_xy: int = 12,
) -> Tuple[int, int]:
    if (
        electrode_positions is not None
        and electrode_positions.ndim == 2
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

    if n_points == 384:
        return 12, 32

    if default_xy > 0 and n_points % default_xy == 0:
        return default_xy, int(n_points // default_xy)

    factors = [f for f in range(1, n_points + 1) if n_points % f == 0]
    n_points_xy = min(factors, key=lambda f: abs(f - np.sqrt(n_points)))
    return int(n_points_xy), int(n_points // n_points_xy)


def read_electrodes_csv(filename: Path) -> Dict[str, object]:
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find {filename}")

    t_values: List[float] = []
    emg_rows: List[List[float]] = []
    position_data: Optional[np.ndarray] = None
    n_points_ref: Optional[int] = None
    detected_format: Optional[str] = None
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
                    f"{filename}, line {line_no}: n_points changed from {n_points_ref} to {n_points}"
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
                    f"{filename}, line {line_no}: expected at least {n_points} EMG values, "
                    f"got {len(numeric_tail)} numeric fields after timestamp/t/n_points."
                )

            t_values.append(t)
            emg_rows.append(values)

    if not emg_rows or n_points_ref is None:
        raise ValueError(f"No numeric EMG rows found in {filename}")

    t_array = np.asarray(t_values, dtype=float)
    emg_array = np.asarray(emg_rows, dtype=float)
    n_points = int(n_points_ref)

    if emg_array.shape[1] != n_points:
        raise ValueError(
            f"{filename}: parsed EMG shape {emg_array.shape}, expected second dimension {n_points}"
        )

    electrode_positions = None
    if position_data is not None and len(position_data) >= 3 * n_points:
        electrode_positions = np.asarray(position_data[:3 * n_points], dtype=float).reshape(n_points, 3)

    n_points_xy, n_points_z = infer_electrode_grid(electrode_positions, n_points)

    print(
        f"      parsed {filename}: format={detected_format}, "
        f"shape={emg_array.shape}, grid={n_points_xy}x{n_points_z}"
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


def ensure_same_layout(cases: Dict[str, Dict[str, object]]) -> Tuple[np.ndarray, int, int, int]:
    names = list(cases.keys())
    if not names:
        raise ValueError("No cases supplied.")

    ref = cases[names[0]]
    ref_t = np.asarray(ref["t"], dtype=float)
    ref_n = int(ref["n_points"])
    ref_xy = int(ref["n_points_xy"])
    ref_z = int(ref["n_points_z"])

    for name in names[1:]:
        c = cases[name]
        if int(c["n_points"]) != ref_n:
            raise ValueError(f"{name}: n_points differs from reference.")
        c_t = np.asarray(c["t"], dtype=float)
        if len(c_t) != len(ref_t) or not np.allclose(c_t, ref_t):
            raise ValueError(
                f"{name}: time vector differs from reference. "
                "Resampling is intentionally not performed in this script."
            )

    return ref_t, ref_n, ref_xy, ref_z


# -----------------------------------------------------------------------------
# Electrode metrics and overlays
# -----------------------------------------------------------------------------


def subtract_baseline(y: np.ndarray, n_baseline: int = 10) -> np.ndarray:
    n = min(n_baseline, len(y))
    if n <= 0:
        return y
    return y - np.mean(y[:n])


def compute_metrics(
    cases: Dict[str, Dict[str, object]],
    channel: int,
    healthy_name: str = "healthy",
    subtract_mean: bool = False,
) -> Dict[str, float]:
    row: Dict[str, float] = {"electrode": int(channel)}
    ref = np.asarray(cases[healthy_name]["emg"], dtype=float)[:, channel]
    if subtract_mean:
        ref = subtract_baseline(ref)

    for name, c in cases.items():
        y = np.asarray(c["emg"], dtype=float)[:, channel]
        if subtract_mean:
            y = subtract_baseline(y)

        row[f"{name}_rms"] = float(np.sqrt(np.mean(y ** 2)))
        row[f"{name}_peak_to_peak"] = float(np.max(y) - np.min(y))
        row[f"{name}_max_abs"] = float(np.max(np.abs(y)))
        row[f"{name}_mean"] = float(np.mean(y))

        if name != healthy_name:
            denom = np.std(ref) * np.std(y)
            corr = np.nan if denom == 0 else float(
                np.mean((ref - np.mean(ref)) * (y - np.mean(y))) / denom
            )
            row[f"{name}_corr_vs_healthy"] = corr

    return row


def get_channel_traces(
    cases: Dict[str, Dict[str, object]],
    channel: int,
    subtract_mean: bool = False,
) -> Dict[str, np.ndarray]:
    traces: Dict[str, np.ndarray] = {}
    for name, c in cases.items():
        y = np.asarray(c["emg"], dtype=float)[:, channel]
        if subtract_mean:
            y = subtract_baseline(y)
        traces[name] = y
    return traces


def get_ylim_from_traces(traces: Dict[str, np.ndarray]) -> Tuple[float, float]:
    local_min = min(float(np.min(y)) for y in traces.values())
    local_max = max(float(np.max(y)) for y in traces.values())
    pad = 0.08 * (local_max - local_min + 1e-12)
    return local_min - pad, local_max + pad


def plot_one_electrode(
    t: np.ndarray,
    cases: Dict[str, Dict[str, object]],
    channel: int,
    n_points_xy: int,
    out_file: Path,
    group_number: int,
    protocol: str,
    subtract_mean: bool = False,
    global_ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 160,
) -> None:
    grid_y = channel // n_points_xy
    grid_x = channel % n_points_xy

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    traces = get_channel_traces(cases, channel, subtract_mean=subtract_mean)

    for name, y in traces.items():
        ax.plot(t, y, linewidth=1.2, label=CASE_DISPLAY.get(name, name))

    ax.set_ylim(*(global_ylim if global_ylim is not None else get_ylim_from_traces(traces)))
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("sEMG [mV]")
    ax.set_title(
        f"g-{group_number}, Protocol {protocol} | Electrode {channel:03d} "
        f"(y={grid_y}, x={grid_x})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def write_metrics_csv(metrics_rows: List[Dict[str, float]], out_file: Path) -> None:
    if not metrics_rows:
        return
    pd.DataFrame(metrics_rows).to_csv(out_file, index=False)


# -----------------------------------------------------------------------------
# Publication-style figures (adapted from the user's existing figure script)
# -----------------------------------------------------------------------------


def infer_grid(n_electrodes: int, n_points_xy: Optional[int] = None) -> Tuple[int, int]:
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


def create_figure1(
    df: pd.DataFrame,
    n_points_xy: int,
    n_points_z: int,
    out_dir: Path,
    group_number: int,
    protocol: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    rms_arrays = [metric_array(df, c, "rms") for c in STAGE_NAMES]
    rms_vmin = min(float(np.nanmin(v)) for v in rms_arrays)
    rms_vmax = max(float(np.nanmax(v)) for v in rms_arrays)

    im_top = None
    for i, case in enumerate(STAGE_NAMES):
        ax = axes[0, i]
        grid = to_grid(metric_array(df, case, "rms"), n_points_xy, n_points_z)
        im_top = ax.imshow(grid, aspect="auto", origin="upper", vmin=rms_vmin, vmax=rms_vmax)
        ax.set_title(CASE_DISPLAY[case], pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("A") + i))

    cbar1 = fig.colorbar(im_top, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=28)
    cbar1.set_label("RMS [mV]")

    healthy = metric_array(df, "healthy", "rms")
    diff_arrays = [metric_array(df, case, "rms") - healthy for case in DISEASE_CASES]
    diff_abs = max(
        max(abs(float(np.nanmin(v))), abs(float(np.nanmax(v))))
        for v in diff_arrays
    )
    if diff_abs <= 0:
        diff_abs = 1e-12

    im_diff = None
    for i, case in enumerate(DISEASE_CASES):
        ax = axes[1, i]
        grid = to_grid(metric_array(df, case, "rms") - healthy, n_points_xy, n_points_z)
        im_diff = ax.imshow(grid, aspect="auto", origin="upper", vmin=-diff_abs, vmax=diff_abs, cmap="coolwarm")
        ax.set_title(f"{CASE_DISPLAY[case]} − Healthy", pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("E") + i))

    cbar2 = fig.colorbar(im_diff, ax=axes[1, 0:3], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("ΔRMS [mV]")

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

    fig.suptitle(f"g-{group_number}, Protocol {protocol}: spatial EMG metrics")
    save_png_pdf(fig, out_dir / "figure1_spatial_rms_and_correlation")
    plt.close(fig)


def create_figure2(df: pd.DataFrame, out_dir: Path, group_number: int, protocol: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)
    x = df["electrode"].to_numpy(dtype=int)

    ax1 = axes[0, 0]
    for case in STAGE_NAMES:
        ax1.plot(x, metric_array(df, case, "rms"), linewidth=1.2, label=CASE_DISPLAY[case])
    ax1.set_title("Electrode-wise RMS")
    ax1.set_xlabel("Electrode index")
    ax1.set_ylabel("RMS [mV]")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    panel_label(ax1, "A")

    ax2 = axes[0, 1]
    for case in STAGE_NAMES:
        ax2.plot(x, metric_array(df, case, "peak_to_peak"), linewidth=1.2, label=CASE_DISPLAY[case])
    ax2.set_title("Electrode-wise peak-to-peak")
    ax2.set_xlabel("Electrode index")
    ax2.set_ylabel("Peak-to-peak [mV]")
    ax2.grid(True, alpha=0.25)
    panel_label(ax2, "B")

    ax3 = axes[0, 2]
    for case in STAGE_NAMES:
        ax3.plot(x, metric_array(df, case, "max_abs"), linewidth=1.2, label=CASE_DISPLAY[case])
    ax3.set_title("Electrode-wise max |EMG|")
    ax3.set_xlabel("Electrode index")
    ax3.set_ylabel("Max |EMG| [mV]")
    ax3.grid(True, alpha=0.25)
    panel_label(ax3, "C")

    ax4 = axes[1, 0]
    rms_data = [metric_array(df, case, "rms") for case in STAGE_NAMES]
    ax4.boxplot(rms_data, tick_labels=[CASE_DISPLAY[c] for c in STAGE_NAMES], showfliers=False)
    ax4.set_title("RMS distribution across electrodes")
    ax4.set_ylabel("RMS [mV]")
    ax4.tick_params(axis="x", rotation=20)
    ax4.grid(True, axis="y", alpha=0.25)
    panel_label(ax4, "D")

    ax5 = axes[1, 1]
    p2p_data = [metric_array(df, case, "peak_to_peak") for case in STAGE_NAMES]
    ax5.boxplot(p2p_data, tick_labels=[CASE_DISPLAY[c] for c in STAGE_NAMES], showfliers=False)
    ax5.set_title("Peak-to-peak distribution")
    ax5.set_ylabel("Peak-to-peak [mV]")
    ax5.tick_params(axis="x", rotation=20)
    ax5.grid(True, axis="y", alpha=0.25)
    panel_label(ax5, "E")

    ax6 = axes[1, 2]
    means = [np.mean(metric_array(df, case, "rms")) for case in STAGE_NAMES]
    stds = [np.std(metric_array(df, case, "rms")) for case in STAGE_NAMES]
    xpos = np.arange(len(STAGE_NAMES))
    ax6.bar(xpos, means, yerr=stds, capsize=4)
    ax6.set_xticks(xpos, [CASE_DISPLAY[c] for c in STAGE_NAMES], rotation=20)
    ax6.set_ylabel("RMS [mV]")
    ax6.set_title("Spatial RMS summary (mean ± SD)")
    ax6.grid(True, axis="y", alpha=0.25)
    panel_label(ax6, "F")

    fig.suptitle(f"g-{group_number}, Protocol {protocol}: electrode metric summaries")
    save_png_pdf(fig, out_dir / "figure2_metric_summaries")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    data = [corr_array(df, case) for case in DISEASE_CASES]
    ax.boxplot(data, tick_labels=[CASE_DISPLAY[c] for c in DISEASE_CASES], showfliers=False)
    ax.set_ylabel("Correlation")
    ax.set_title(f"g-{group_number}, Protocol {protocol}: correlation vs Healthy")
    ax.grid(True, axis="y", alpha=0.25)
    save_png_pdf(fig2, out_dir / "supplementary_correlation_boxplot")
    plt.close(fig2)


def create_figure3(
    df: pd.DataFrame,
    n_points_xy: int,
    n_points_z: int,
    out_dir: Path,
    group_number: int,
    protocol: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    p2p_arrays = [metric_array(df, c, "peak_to_peak") for c in STAGE_NAMES]
    p2p_vmin = min(float(np.nanmin(v)) for v in p2p_arrays)
    p2p_vmax = max(float(np.nanmax(v)) for v in p2p_arrays)

    im_top = None
    for i, case in enumerate(STAGE_NAMES):
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
    ratio_arrays = [metric_array(df, c, "peak_to_peak") / np.maximum(np.abs(healthy), eps) for c in DISEASE_CASES]
    ratio_vmin = min(float(np.nanmin(v)) for v in ratio_arrays)
    ratio_vmax = max(float(np.nanmax(v)) for v in ratio_arrays)

    im_ratio = None
    for i, case in enumerate(DISEASE_CASES):
        ax = axes[1, i]
        ratio = metric_array(df, case, "peak_to_peak") / np.maximum(np.abs(healthy), eps)
        grid = to_grid(ratio, n_points_xy, n_points_z)
        im_ratio = ax.imshow(grid, aspect="auto", origin="upper", vmin=ratio_vmin, vmax=ratio_vmax)
        ax.set_title(f"{CASE_DISPLAY[case]} / Healthy", pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("E") + i))

    cbar2 = fig.colorbar(im_ratio, ax=axes[1, 0:3], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("Peak-to-peak ratio")

    ax_line = axes[1, 3]
    x = df["electrode"].to_numpy(dtype=int)
    for case in STAGE_NAMES:
        ax_line.plot(x, metric_array(df, case, "peak_to_peak"), linewidth=1.0, label=CASE_DISPLAY[case])
    ax_line.set_title("Peak-to-peak across electrodes", pad=8)
    ax_line.set_xlabel("Electrode index")
    ax_line.set_ylabel("Peak-to-peak [mV]")
    ax_line.grid(True, alpha=0.25)
    ax_line.legend(loc="best")
    panel_label(ax_line, "H")

    fig.suptitle(f"g-{group_number}, Protocol {protocol}: peak-to-peak metrics")
    save_png_pdf(fig, out_dir / "figure3_peak_to_peak_maps")
    plt.close(fig)


def create_publication_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metrics = ["rms", "peak_to_peak", "max_abs"]

    for metric in metrics:
        row: Dict[str, object] = {"metric": metric}
        for case in STAGE_NAMES:
            vals = metric_array(df, case, metric)
            row[f"{case}_mean"] = float(np.mean(vals))
            row[f"{case}_std"] = float(np.std(vals))
            row[f"{case}_median"] = float(np.median(vals))
            row[f"{case}_min"] = float(np.min(vals))
            row[f"{case}_max"] = float(np.max(vals))
        rows.append(row)

    row_corr: Dict[str, object] = {"metric": "corr_vs_healthy"}
    for suffix in ("mean", "std", "median", "min", "max"):
        row_corr[f"healthy_{suffix}"] = np.nan
    for case in DISEASE_CASES:
        vals = corr_array(df, case)
        row_corr[f"{case}_mean"] = float(np.nanmean(vals))
        row_corr[f"{case}_std"] = float(np.nanstd(vals))
        row_corr[f"{case}_median"] = float(np.nanmedian(vals))
        row_corr[f"{case}_min"] = float(np.nanmin(vals))
        row_corr[f"{case}_max"] = float(np.nanmax(vals))
    rows.append(row_corr)

    return pd.DataFrame(rows)


def create_publication_figures(
    metrics_df: pd.DataFrame,
    n_points_xy: int,
    n_points_z: int,
    out_dir: Path,
    group_number: int,
    protocol: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    create_figure1(metrics_df, n_points_xy, n_points_z, out_dir, group_number, protocol)
    create_figure2(metrics_df, out_dir, group_number, protocol)
    create_figure3(metrics_df, n_points_xy, n_points_z, out_dir, group_number, protocol)
    summary = create_publication_summary_table(metrics_df)
    summary.insert(0, "protocol", protocol)
    summary.insert(0, "group_number", group_number)
    summary.to_csv(out_dir / "publication_summary_table.csv", index=False)


# -----------------------------------------------------------------------------
# Group/protocol processing
# -----------------------------------------------------------------------------


def compute_global_ylim(
    cases: Dict[str, Dict[str, object]],
    n_plot: int,
    subtract_mean: bool,
) -> Tuple[float, float]:
    all_vals: List[np.ndarray] = []
    for c in cases.values():
        y = np.asarray(c["emg"], dtype=float)[:, :n_plot]
        if subtract_mean:
            n = min(10, y.shape[0])
            y = y - np.mean(y[:n, :], axis=0, keepdims=True)
        all_vals.append(y.ravel())
    vals = np.concatenate(all_vals)
    ymin, ymax = float(np.min(vals)), float(np.max(vals))
    pad = 0.08 * (ymax - ymin + 1e-12)
    return ymin - pad, ymax + pad


def process_group_protocol(
    group_number: int,
    protocol: str,
    refs: Dict[str, CaseRef],
    out_root: Path,
    subtract_mean: bool,
    global_y: bool,
    max_electrodes: Optional[int],
    dpi: int,
    skip_electrode_plots: bool,
    make_pdf_overlays: bool,
    skip_publication_figures: bool,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], int, int, np.ndarray]:
    print(f"\n=== g-{group_number}, Protocol {protocol} ===")
    cases: Dict[str, Dict[str, object]] = {}
    for stage in STAGE_NAMES:
        if stage not in refs:
            continue
        print(f"  Loading {stage}: {refs[stage].csv_path}")
        cases[stage] = read_electrodes_csv(refs[stage].csv_path)

    if "healthy" not in cases:
        raise RuntimeError(f"g-{group_number}, Protocol {protocol}: healthy case is required as reference.")

    # Publication figures require all four stages. Metrics can still be produced
    # with a partial set when --allow-incomplete is used.
    t, n_points, n_points_xy, n_points_z = ensure_same_layout(cases)
    n_plot = min(n_points, max_electrodes) if max_electrodes is not None else n_points

    group_dir = out_root / f"g-{group_number}" / f"protocol_{protocol}"
    overlay_dir = group_dir / "emg_overlay_by_electrode"
    metric_plot_dir = group_dir / "metric_plots"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    metric_plot_dir.mkdir(parents=True, exist_ok=True)

    global_ylim = compute_global_ylim(cases, n_plot, subtract_mean) if global_y else None

    metrics_rows: List[Dict[str, float]] = []
    pdf_pages = None
    if make_pdf_overlays and not skip_electrode_plots:
        pdf_pages = PdfPages(overlay_dir / "all_electrodes_overlay.pdf")

    for ch in range(n_plot):
        if not skip_electrode_plots:
            static_file = overlay_dir / f"electrode_{ch:03d}_y{ch // n_points_xy:02d}_x{ch % n_points_xy:02d}.png"
            plot_one_electrode(
                t=t,
                cases=cases,
                channel=ch,
                n_points_xy=n_points_xy,
                out_file=static_file,
                group_number=group_number,
                protocol=protocol,
                subtract_mean=subtract_mean,
                global_ylim=global_ylim,
                dpi=dpi,
            )

            if pdf_pages is not None:
                traces = get_channel_traces(cases, ch, subtract_mean=subtract_mean)
                fig, ax = plt.subplots(figsize=(8.0, 4.5))
                for name, y in traces.items():
                    ax.plot(t, y, linewidth=1.2, label=CASE_DISPLAY.get(name, name))
                ax.set_ylim(*(global_ylim if global_ylim is not None else get_ylim_from_traces(traces)))
                ax.set_xlabel("Time [ms]")
                ax.set_ylabel("sEMG [mV]")
                ax.set_title(
                    f"g-{group_number}, Protocol {protocol} | Electrode {ch:03d} "
                    f"(y={ch // n_points_xy}, x={ch % n_points_xy})"
                )
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")
                fig.tight_layout()
                pdf_pages.savefig(fig)
                plt.close(fig)

        metrics_rows.append(compute_metrics(cases, ch, subtract_mean=subtract_mean))

        if (ch + 1) % 50 == 0 or ch + 1 == n_plot:
            action = "processed" if skip_electrode_plots else "plotted/processed"
            print(f"    {action} {ch + 1}/{n_plot} electrodes")

    if pdf_pages is not None:
        pdf_pages.close()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_file = overlay_dir / "metrics_by_electrode.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Created: {metrics_file}")

    if not skip_publication_figures and all(stage in cases for stage in STAGE_NAMES):
        create_publication_figures(
            metrics_df, n_points_xy, n_points_z, metric_plot_dir, group_number, protocol
        )
        print(f"  Created publication figures: {metric_plot_dir}")
    elif not skip_publication_figures:
        print("  Skipping publication figures because this group/protocol does not contain all four stages.")

    manifest_rows: List[Dict[str, object]] = []
    for stage, ref in refs.items():
        c = cases[stage]
        manifest_rows.append({
            "group_number": group_number,
            "group_tag": f"g-{group_number}",
            "protocol": protocol,
            "stage": stage,
            "kill_fraction": KILL_FRACTION[stage],
            "scenario_name": ref.scenario_name,
            "electrodes_csv": str(ref.csv_path.resolve()),
            "n_time_steps": len(np.asarray(c["t"])),
            "time_start_ms": float(np.asarray(c["t"])[0]),
            "time_end_ms": float(np.asarray(c["t"])[-1]),
            "n_electrodes": int(c["n_points"]),
            "n_points_xy": int(c["n_points_xy"]),
            "n_points_z": int(c["n_points_z"]),
            "detected_csv_format": c["detected_format"],
            "metrics_csv": str(metrics_file.resolve()),
            "metric_plots_dir": str(metric_plot_dir.resolve()),
        })

    return metrics_df, manifest_rows, n_points_xy, n_points_z, t


# -----------------------------------------------------------------------------
# Multi-group aggregation for downstream statistics
# -----------------------------------------------------------------------------


def add_metadata_to_wide(
    metrics_df: pd.DataFrame,
    group_number: int,
    protocol: str,
) -> pd.DataFrame:
    out = metrics_df.copy()
    out.insert(0, "protocol", protocol)
    out.insert(0, "group_tag", f"g-{group_number}")
    out.insert(0, "group_number", group_number)
    return out


def wide_to_long(metrics_df: pd.DataFrame, group_number: int, protocol: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in metrics_df.iterrows():
        electrode = int(row["electrode"])
        for stage in STAGE_NAMES:
            # In partial processing a stage may be absent from columns.
            rms_col = f"{stage}_rms"
            if rms_col not in metrics_df.columns:
                continue
            rec: Dict[str, object] = {
                "group_number": group_number,
                "group_tag": f"g-{group_number}",
                "protocol": protocol,
                "stage": stage,
                "kill_fraction": KILL_FRACTION[stage],
                "electrode": electrode,
                "rms": float(row[rms_col]),
                "peak_to_peak": float(row[f"{stage}_peak_to_peak"]),
                "max_abs": float(row[f"{stage}_max_abs"]),
                "mean": float(row[f"{stage}_mean"]),
                "corr_vs_healthy": np.nan,
            }
            corr_col = f"{stage}_corr_vs_healthy"
            if corr_col in metrics_df.columns:
                rec["corr_vs_healthy"] = float(row[corr_col])
            rows.append(rec)
    return pd.DataFrame(rows)


def create_group_level_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize spatial electrode distributions within each independent group."""
    records: List[Dict[str, object]] = []
    metrics = ["rms", "peak_to_peak", "max_abs", "mean", "corr_vs_healthy"]

    for (group, protocol, stage), sub in long_df.groupby(
        ["group_number", "protocol", "stage"], sort=True
    ):
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            records.append({
                "group_number": int(group),
                "group_tag": f"g-{int(group)}",
                "protocol": protocol,
                "stage": stage,
                "kill_fraction": KILL_FRACTION[stage],
                "metric": metric,
                "n_electrodes": int(len(vals)),
                "mean_across_electrodes": float(np.mean(vals)),
                "std_across_electrodes": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median_across_electrodes": float(np.median(vals)),
                "min_across_electrodes": float(np.min(vals)),
                "max_across_electrodes": float(np.max(vals)),
            })

    return pd.DataFrame(records)


def create_paired_protocol_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    """Create A/B paired group-level values; each group remains one replicate."""
    if group_summary.empty:
        return pd.DataFrame()

    base = group_summary[
        group_summary["metric"].isin(["rms", "peak_to_peak", "max_abs", "mean", "corr_vs_healthy"])
    ].copy()

    pivot = base.pivot_table(
        index=["group_number", "group_tag", "stage", "kill_fraction", "metric"],
        columns="protocol",
        values="mean_across_electrodes",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    if "A" not in pivot.columns:
        pivot["A"] = np.nan
    if "B" not in pivot.columns:
        pivot["B"] = np.nan

    pivot = pivot.rename(columns={"A": "protocol_A_mean", "B": "protocol_B_mean"})
    pivot["B_minus_A"] = pivot["protocol_B_mean"] - pivot["protocol_A_mean"]
    denom = pivot["protocol_A_mean"].abs()
    pivot["B_over_A"] = np.where(denom > 1e-15, pivot["protocol_B_mean"] / pivot["protocol_A_mean"], np.nan)
    pivot["percent_change_B_vs_A"] = np.where(
        denom > 1e-15,
        100.0 * (pivot["protocol_B_mean"] - pivot["protocol_A_mean"]) / denom,
        np.nan,
    )
    return pivot.sort_values(["group_number", "metric", "kill_fraction"]).reset_index(drop=True)


def create_aggregate_group_figure(group_summary: pd.DataFrame, out_root: Path) -> None:
    """Descriptive across-group figure; no inferential statistics are performed here."""
    metrics = [
        ("rms", "RMS [mV]"),
        ("peak_to_peak", "Peak-to-peak [mV]"),
        ("max_abs", "Max |EMG| [mV]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    x = np.arange(len(STAGE_NAMES))

    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = group_summary[group_summary["metric"] == metric]
        for protocol in PROTOCOLS:
            psub = sub[sub["protocol"] == protocol]
            means: List[float] = []
            stds: List[float] = []
            for stage in STAGE_NAMES:
                vals = psub.loc[
                    psub["stage"] == stage, "mean_across_electrodes"
                ].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            ax.errorbar(x, means, yerr=stds, marker="o", capsize=4, label=f"Protocol {protocol}")

        ax.set_xticks(x, ["Healthy", "25%", "50%", "75%"])
        ax.set_xlabel("MU-loss condition")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("Across-group descriptive summary (mean ± SD across independent groups)")
    save_png_pdf(fig, out_root / "aggregate_group_summary")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process all grouped Protocol A/B OpenDiHu electrode outputs, create "
            "per-group figures/metrics, and aggregate independent groups for statistics."
        )
    )
    parser.add_argument(
        "--base-dir",
        default="build_release/out",
        help="Root directory containing OpenDiHu output cases. Searched recursively for electrodes.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default="grouped_emg_postprocessing",
        help="Root directory for all postprocessing outputs.",
    )
    parser.add_argument("--csv-name", default="electrodes.csv", help="Electrode CSV filename to discover.")
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=None,
        help="Optional group numbers to process, e.g. --groups 1 2 3. Default: all discovered groups.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["A", "B", "a", "b"],
        default=["A", "B"],
        help="Protocols to process. Default: A B.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow missing stages/protocols. Default is strict complete-case validation.",
    )
    parser.add_argument(
        "--max-electrodes",
        type=int,
        default=None,
        help="Only process the first N electrodes (useful for testing). Default: all.",
    )
    parser.add_argument(
        "--subtract-mean",
        action="store_true",
        help="Subtract mean of the first 10 samples from each trace, matching the old optional behavior.",
    )
    parser.add_argument(
        "--global-y",
        action="store_true",
        help="Use one y-axis range across all electrodes/stages within each group/protocol.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="DPI for individual electrode PNGs.")
    parser.add_argument(
        "--skip-electrode-plots",
        action="store_true",
        help=(
            "Do not create one PNG per electrode. Metrics and publication figures are still generated. "
            "Recommended for large repeated studies when only statistics are needed."
        ),
    )
    parser.add_argument(
        "--pdf-overlays",
        action="store_true",
        help="Also create a multi-page PDF of electrode overlays for every group/protocol.",
    )
    parser.add_argument(
        "--skip-publication-figures",
        action="store_true",
        help="Skip the three publication-style figure sets for each group/protocol.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover and validate cases; do not read EMG data or create outputs.",
    )
    return parser.parse_args()


def print_discovery_table(
    groups: Sequence[int],
    protocols: Sequence[str],
    selected: Dict[Tuple[int, str, str], CaseRef],
) -> None:
    print("\nDiscovered study cases:")
    for g in groups:
        print(f"  g-{g}")
        for p in protocols:
            print(f"    Protocol {p}")
            for stage in STAGE_NAMES:
                ref = selected.get((g, p, stage))
                print(f"      {stage:8s}: {ref.csv_path if ref else '[missing]'}")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_root = Path(args.out_dir)

    all_cases = discover_cases(base_dir, args.csv_name)
    groups, protocols, selected = select_and_validate_cases(
        all_cases,
        requested_groups=args.groups,
        requested_protocols=args.protocols,
        allow_incomplete=args.allow_incomplete,
    )
    print_discovery_table(groups, protocols, selected)

    if args.dry_run:
        print("\nDry run complete. No files were written.")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    wide_frames: List[pd.DataFrame] = []
    long_frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []

    n_processed = 0
    for g in groups:
        for p in protocols:
            refs = {
                stage: selected[(g, p, stage)]
                for stage in STAGE_NAMES
                if (g, p, stage) in selected
            }
            if not refs:
                continue
            metrics_df, manifest, _, _, _ = process_group_protocol(
                group_number=g,
                protocol=p,
                refs=refs,
                out_root=out_root,
                subtract_mean=args.subtract_mean,
                global_y=args.global_y,
                max_electrodes=args.max_electrodes,
                dpi=args.dpi,
                skip_electrode_plots=args.skip_electrode_plots,
                make_pdf_overlays=args.pdf_overlays,
                skip_publication_figures=args.skip_publication_figures,
            )
            wide_frames.append(add_metadata_to_wide(metrics_df, g, p))
            long_frames.append(wide_to_long(metrics_df, g, p))
            manifest_rows.extend(manifest)
            n_processed += 1

    if not wide_frames:
        raise RuntimeError("No group/protocol datasets were processed.")

    manifest_df = pd.DataFrame(manifest_rows).sort_values(
        ["group_number", "protocol", "kill_fraction"]
    )
    manifest_df.to_csv(out_root / "processing_manifest.csv", index=False)

    wide_df = pd.concat(wide_frames, ignore_index=True)
    wide_df.to_csv(out_root / "all_groups_metrics_by_electrode_wide.csv", index=False)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.sort_values(
        ["group_number", "protocol", "kill_fraction", "electrode"]
    ).reset_index(drop=True)
    long_df.to_csv(out_root / "all_groups_metrics_by_electrode_long.csv", index=False)

    group_summary = create_group_level_summary(long_df)
    group_summary = group_summary.sort_values(
        ["metric", "group_number", "protocol", "kill_fraction"]
    ).reset_index(drop=True)
    group_summary.to_csv(out_root / "group_level_summary.csv", index=False)

    paired_summary = create_paired_protocol_summary(group_summary)
    paired_summary.to_csv(out_root / "paired_protocol_summary.csv", index=False)

    if not group_summary.empty:
        create_aggregate_group_figure(group_summary, out_root)

    print("\n============================================================")
    print("Grouped EMG postprocessing completed.")
    print(f"Processed group/protocol datasets: {n_processed}")
    print(f"Independent groups:                {len(groups)}")
    print(f"Output root:                       {out_root.resolve()}")
    print("\nMaster outputs:")
    print(f"  {out_root / 'processing_manifest.csv'}")
    print(f"  {out_root / 'all_groups_metrics_by_electrode_wide.csv'}")
    print(f"  {out_root / 'all_groups_metrics_by_electrode_long.csv'}")
    print(f"  {out_root / 'group_level_summary.csv'}")
    print(f"  {out_root / 'paired_protocol_summary.csv'}")
    print(f"  {out_root / 'aggregate_group_summary.png'}")
    print("\nFor inferential statistics, use group_number as the replicate/block.")
    print("Do not treat individual electrodes as independent biological replicates.")


if __name__ == "__main__":
    main()
