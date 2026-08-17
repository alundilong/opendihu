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
      across_group_summary.csv
      normalized_to_healthy_group_summary.csv
      paired_protocol_across_group_summary.csv
      manuscript_figures/
        01_protocol_A/
          figure_A1_group_trajectories.png/.pdf
          figure_A2_percent_change_vs_healthy.png/.pdf
          figure_A3_group_mean_rms_maps.png/.pdf
          supplementary_A3_group_sd_rms_maps.png/.pdf
          protocol_A_group_level_values.csv
          protocol_A_across_group_summary.csv
        02_protocol_B/
          figure_B1_group_trajectories.png/.pdf
          figure_B2_percent_change_vs_healthy.png/.pdf
          figure_B3_group_mean_rms_maps.png/.pdf
          supplementary_B3_group_sd_rms_maps.png/.pdf
          protocol_B_group_level_values.csv
          protocol_B_across_group_summary.csv
        03_protocol_comparison/
          figure_C1_protocol_A_vs_B_group_trajectories.png/.pdf
          figure_C2_protocol_A_vs_B_percent_change_vs_healthy.png/.pdf
          figure_C3_protocol_B_effect_percent.png/.pdf
          figure_C4_protocol_B_effect_absolute.png/.pdf
          figure_C5_protocol_A_vs_B_group_mean_rms_maps.png/.pdf
          figure_C6_group_mean_B_minus_A_rms_maps.png/.pdf
          supplementary_C5_group_sd_rms_maps.png/.pdf
          supplementary_C6_group_sd_B_minus_A_rms_maps.png/.pdf
        README_manuscript_figures.txt
      g-1/
        protocol_A/
          emg_overlay_by_electrode/
            electrode_000_y00_x00.png
            ...
            metrics_by_electrode.csv
          metric_plots/
            figure1_spatial_rms_and_psd_similarity.png/.pdf
            figure2_metric_summaries.png/.pdf
            figure3_peak_to_peak_maps.png/.pdf
            supplementary_similarity_boxplots.png/.pdf
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
PLOT_FONT_SIZE = 14
plt.rcParams.update({
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "xtick.labelsize": PLOT_FONT_SIZE,
    "ytick.labelsize": PLOT_FONT_SIZE,
    "legend.fontsize": PLOT_FONT_SIZE,
    "figure.titlesize": PLOT_FONT_SIZE,
})
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
def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation for equal-length finite vectors; NaN if undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return np.nan
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return np.nan if denom <= 0.0 else float(np.dot(x, y) / denom)

def _cosine_similarity_nonnegative(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity, primarily used for non-negative PSD/spatial metric vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return np.nan if denom <= 0.0 else float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))

def _welch_psd_numpy(
    y: np.ndarray,
    t_ms: np.ndarray,
    window_ms: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight Welch PSD using NumPy only (50% overlap, Hann window).

    The OpenDiHu electrode time vector is expressed in milliseconds in this
    postprocessor. Segment averaging deliberately suppresses sensitivity to the
    exact timing/phase of individual motor-unit discharges over a long record.
    """
    y = np.asarray(y, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)
    if len(y) != len(t_ms) or len(y) < 8:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    dt_ms = float(np.median(np.diff(t_ms)))
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    fs_hz = 1000.0 / dt_ms
    nperseg = int(round(float(window_ms) / dt_ms))
    nperseg = min(len(y), max(8, nperseg))
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)
    win_norm = float(np.sum(window ** 2))
    if win_norm <= 0.0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    starts = list(range(0, max(1, len(y) - nperseg + 1), step))
    if not starts or starts[-1] != len(y) - nperseg:
        starts.append(len(y) - nperseg)
    spectra: List[np.ndarray] = []
    for start in starts:
        seg = y[start:start + nperseg]
        if len(seg) != nperseg:
            continue
        seg = seg - np.mean(seg)
        fft = np.fft.rfft(seg * window)
        power = (np.abs(fft) ** 2) / win_norm
        spectra.append(power)
    if not spectra:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    psd = np.mean(np.vstack(spectra), axis=0)
    freq = np.fft.rfftfreq(nperseg, d=1.0 / fs_hz)
    return freq, psd

def psd_cosine_similarity(
    ref: np.ndarray,
    y: np.ndarray,
    t_ms: np.ndarray,
    window_ms: float = 1000.0,
) -> float:
    """Phase-insensitive similarity of Welch power spectra, bounded near [0, 1]."""
    f_ref, p_ref = _welch_psd_numpy(ref, t_ms, window_ms=window_ms)
    f_y, p_y = _welch_psd_numpy(y, t_ms, window_ms=window_ms)
    if len(p_ref) == 0 or len(p_ref) != len(p_y) or len(f_ref) != len(f_y):
        return np.nan
    # DC is excluded because every segment is mean-centered and DC carries no
    # useful information about EMG waveform organization here.
    if len(p_ref) > 1:
        p_ref = p_ref[1:]
        p_y = p_y[1:]
    return _cosine_similarity_nonnegative(p_ref, p_y)

def windowed_abs_correlation_similarity(
    ref: np.ndarray,
    y: np.ndarray,
    t_ms: np.ndarray,
    window_ms: float = 250.0,
) -> float:
    """Secondary local waveform diagnostic: mean |r| across non-overlapping windows.

    Unlike a single 30-s Pearson coefficient, this measure does not require
    phase alignment to persist across the entire recording. It remains a
    secondary diagnostic because stochastic discharge jitter can still reduce it.
    """
    ref = np.asarray(ref, dtype=float)
    y = np.asarray(y, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)
    if len(ref) != len(y) or len(ref) != len(t_ms) or len(ref) < 8:
        return np.nan
    dt_ms = float(np.median(np.diff(t_ms)))
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        return np.nan
    nwin = max(8, int(round(float(window_ms) / dt_ms)))
    vals: List[float] = []
    for start in range(0, len(ref) - nwin + 1, nwin):
        r = _safe_pearson(ref[start:start + nwin], y[start:start + nwin])
        if np.isfinite(r):
            vals.append(abs(float(r)))
    return float(np.mean(vals)) if vals else np.nan

def _welch_psd_matrix_numpy(
    y: np.ndarray,
    t_ms: np.ndarray,
    window_ms: float = 1000.0,
) -> np.ndarray:
    """Vectorized Welch PSD for a [time, electrode] matrix."""
    y = np.asarray(y, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)
    if y.ndim != 2 or y.shape[0] != len(t_ms) or y.shape[0] < 8:
        return np.empty((0, y.shape[1] if y.ndim == 2 else 0), dtype=float)
    dt_ms = float(np.median(np.diff(t_ms)))
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        return np.empty((0, y.shape[1]), dtype=float)
    nperseg = int(round(float(window_ms) / dt_ms))
    nperseg = min(y.shape[0], max(8, nperseg))
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)[:, None]
    win_norm = float(np.sum(window[:, 0] ** 2))
    starts = list(range(0, max(1, y.shape[0] - nperseg + 1), step))
    if not starts or starts[-1] != y.shape[0] - nperseg:
        starts.append(y.shape[0] - nperseg)
    accum: Optional[np.ndarray] = None
    n_used = 0
    for start in starts:
        seg = y[start:start + nperseg, :]
        if seg.shape[0] != nperseg:
            continue
        seg = seg - np.mean(seg, axis=0, keepdims=True)
        fft = np.fft.rfft(seg * window, axis=0)
        power = (np.abs(fft) ** 2) / win_norm
        if accum is None:
            accum = np.zeros_like(power, dtype=float)
        accum += power
        n_used += 1
    if accum is None or n_used == 0:
        return np.empty((0, y.shape[1]), dtype=float)
    psd = accum / float(n_used)
    return psd[1:, :] if psd.shape[0] > 1 else psd

def _columnwise_cosine_similarity(ref: np.ndarray, y: np.ndarray) -> np.ndarray:
    if ref.shape != y.shape or ref.ndim != 2:
        raise ValueError("PSD matrices must have the same [frequency, electrode] shape.")
    numerator = np.sum(ref * y, axis=0)
    denom = np.sqrt(np.sum(ref ** 2, axis=0) * np.sum(y ** 2, axis=0))
    out = np.full(ref.shape[1], np.nan, dtype=float)
    valid = denom > 0.0
    out[valid] = np.clip(numerator[valid] / denom[valid], -1.0, 1.0)
    return out

def _windowed_abs_corr_matrix(
    ref: np.ndarray,
    y: np.ndarray,
    t_ms: np.ndarray,
    window_ms: float,
) -> np.ndarray:
    """Mean |Pearson r| per electrode across non-overlapping local windows."""
    ref = np.asarray(ref, dtype=float)
    y = np.asarray(y, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)
    if ref.shape != y.shape or ref.ndim != 2 or ref.shape[0] != len(t_ms):
        raise ValueError("Local-correlation matrices must share the same time/electrode layout.")
    dt_ms = float(np.median(np.diff(t_ms)))
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        return np.full(ref.shape[1], np.nan, dtype=float)
    nwin = max(8, int(round(float(window_ms) / dt_ms)))
    sums = np.zeros(ref.shape[1], dtype=float)
    counts = np.zeros(ref.shape[1], dtype=int)
    for start in range(0, ref.shape[0] - nwin + 1, nwin):
        a = ref[start:start + nwin, :]
        b = y[start:start + nwin, :]
        a = a - np.mean(a, axis=0, keepdims=True)
        b = b - np.mean(b, axis=0, keepdims=True)
        numerator = np.sum(a * b, axis=0)
        denom = np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))
        valid = denom > 0.0
        r = np.full(ref.shape[1], np.nan, dtype=float)
        r[valid] = np.clip(numerator[valid] / denom[valid], -1.0, 1.0)
        finite = np.isfinite(r)
        sums[finite] += np.abs(r[finite])
        counts[finite] += 1
    out = np.full(ref.shape[1], np.nan, dtype=float)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out

def precompute_similarity_cache(
    cases: Dict[str, Dict[str, object]],
    healthy_name: str = "healthy",
    subtract_mean: bool = False,
    psd_window_ms: float = 1000.0,
    local_corr_window_ms: float = 250.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute timing-robust similarity arrays once for all electrodes.

    This avoids recomputing the Healthy PSD for every disease stage/electrode and
    keeps the 30-s postprocessing practical for 384-channel arrays.
    """
    t_ms = np.asarray(cases[healthy_name]["t"], dtype=float)
    matrices: Dict[str, np.ndarray] = {}
    for name, c in cases.items():
        y = np.asarray(c["emg"], dtype=float).copy()
        if subtract_mean:
            n = min(10, y.shape[0])
            y = y - np.mean(y[:n, :], axis=0, keepdims=True)
        matrices[name] = y
    ref = matrices[healthy_name]
    psd_ref = _welch_psd_matrix_numpy(ref, t_ms, window_ms=psd_window_ms)
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    for name, y in matrices.items():
        if name == healthy_name:
            continue
        psd_y = _welch_psd_matrix_numpy(y, t_ms, window_ms=psd_window_ms)
        cache[name] = {
            "psd_similarity_vs_healthy": _columnwise_cosine_similarity(psd_ref, psd_y),
            "local_waveform_similarity_vs_healthy": _windowed_abs_corr_matrix(
                ref, y, t_ms, window_ms=local_corr_window_ms
            ),
        }
    return cache

def compute_metrics(
    cases: Dict[str, Dict[str, object]],
    channel: int,
    healthy_name: str = "healthy",
    subtract_mean: bool = False,
    psd_window_ms: float = 1000.0,
    local_corr_window_ms: float = 250.0,
    similarity_cache: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, float]:
    row: Dict[str, float] = {"electrode": int(channel)}
    ref = np.asarray(cases[healthy_name]["emg"], dtype=float)[:, channel]
    t_ms = np.asarray(cases[healthy_name]["t"], dtype=float)
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
            if similarity_cache is not None and name in similarity_cache:
                row[f"{name}_psd_similarity_vs_healthy"] = float(
                    similarity_cache[name]["psd_similarity_vs_healthy"][channel]
                )
                row[f"{name}_local_waveform_similarity_vs_healthy"] = float(
                    similarity_cache[name]["local_waveform_similarity_vs_healthy"][channel]
                )
            else:
                row[f"{name}_psd_similarity_vs_healthy"] = psd_cosine_similarity(
                    ref, y, t_ms, window_ms=psd_window_ms
                )
                row[f"{name}_local_waveform_similarity_vs_healthy"] = windowed_abs_correlation_similarity(
                    ref, y, t_ms, window_ms=local_corr_window_ms
                )
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
def parse_electrode_position_specs(specs: Optional[Sequence[str]]) -> Optional[List[Tuple[int, int]]]:
    if specs is None:
        return None
    positions: List[Tuple[int, int]] = []
    for spec in specs:
        raw = str(spec).strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid electrode position '{raw}'. Use y,x such as 0,0 or 10,0."
            )
        try:
            y = int(parts[0])
            x = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid electrode position '{raw}'. Use integer y,x indices."
            ) from exc
        positions.append((y, x))
    unique: List[Tuple[int, int]] = []
    for p in positions:
        if p not in unique:
            unique.append(p)
    return unique
def resolve_channels_to_plot(
    requested_positions: Optional[Sequence[Tuple[int, int]]],
    n_points_xy: int,
    n_points_z: int,
    n_points_total: int,
    max_electrodes: Optional[int] = None,
) -> List[int]:
    if requested_positions:
        channels: List[int] = []
        for grid_y, grid_x in requested_positions:
            if grid_y < 0 or grid_y >= n_points_z or grid_x < 0 or grid_x >= n_points_xy:
                raise ValueError(
                    f"Requested electrode (y={grid_y}, x={grid_x}) is outside grid {n_points_z}x{n_points_xy}."
                )
            ch = grid_y * n_points_xy + grid_x
            if ch < 0 or ch >= n_points_total:
                raise ValueError(
                    f"Requested electrode (y={grid_y}, x={grid_x}) maps to invalid channel {ch}."
                )
            channels.append(int(ch))
        return channels
    n_plot = min(n_points_total, max_electrodes) if max_electrodes is not None else n_points_total
    return list(range(n_plot))
def choose_zoom_window(
    t: np.ndarray,
    traces: Dict[str, np.ndarray],
    window_ms: float = 300.0,
) -> Tuple[float, float]:
    if len(t) < 2:
        return float(t[0]), float(t[-1])
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return float(t[0]), float(t[-1])
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return float(t[0]), float(t[-1])
    window_ms = max(float(window_ms), 10.0 * dt)
    if window_ms >= duration:
        return float(t[0]), float(t[-1])
    w = max(5, int(round(window_ms / dt)))
    kernel = np.ones(w, dtype=float) / float(w)
    activity = np.zeros(len(t), dtype=float)
    for y in traces.values():
        activity += np.convolve(np.abs(np.asarray(y, dtype=float)), kernel, mode="same")
    idx = int(np.nanargmax(activity))
    center = float(t[idx])
    start = center - 0.5 * window_ms
    end = center + 0.5 * window_ms
    if start < float(t[0]):
        end += float(t[0]) - start
        start = float(t[0])
    if end > float(t[-1]):
        start -= end - float(t[-1])
        end = float(t[-1])
    start = max(start, float(t[0]))
    end = min(end, float(t[-1]))
    if end <= start:
        start = float(t[0])
        end = min(float(t[-1]), start + window_ms)
    return start, end
def get_segment_ylim(
    t: np.ndarray,
    traces: Dict[str, np.ndarray],
    start_ms: float,
    end_ms: float,
) -> Tuple[float, float]:
    mask = (t >= start_ms) & (t <= end_ms)
    if not np.any(mask):
        return get_ylim_from_traces(traces)
    seg_min = min(float(np.min(y[mask])) for y in traces.values())
    seg_max = max(float(np.max(y[mask])) for y in traces.values())
    pad = 0.12 * (seg_max - seg_min + 1e-12)
    return seg_min - pad, seg_max + pad
def validate_inset_box(left: float, bottom: float, width: float, height: float) -> Tuple[float, float, float, float]:
    vals = [float(left), float(bottom), float(width), float(height)]
    names = ["left", "bottom", "width", "height"]
    for name, val in zip(names, vals):
        if not np.isfinite(val):
            raise ValueError(f"Zoom inset {name} must be finite, got {val}.")
    left, bottom, width, height = vals
    if width <= 0 or height <= 0:
        raise ValueError("Zoom inset width and height must be positive.")
    if left < 0 or bottom < 0 or left + width > 1 or bottom + height > 1:
        raise ValueError(
            "Zoom inset box must lie inside the main axes. "
            f"Received left={left}, bottom={bottom}, width={width}, height={height}."
        )
    return left, bottom, width, height


def resolve_zoom_interval(
    t: np.ndarray,
    traces: Dict[str, np.ndarray],
    zoom_window_ms: float = 300.0,
    zoom_start_ms: Optional[float] = None,
    zoom_end_ms: Optional[float] = None,
) -> Tuple[float, float]:
    explicit_start = zoom_start_ms is not None
    explicit_end = zoom_end_ms is not None

    if explicit_start != explicit_end:
        raise ValueError(
            "--zoom-start-ms and --zoom-end-ms must be supplied together. "
            "Otherwise omit both to use automatic zoom-window selection."
        )

    if explicit_start and explicit_end:
        start = float(zoom_start_ms)
        end = float(zoom_end_ms)
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("Explicit zoom start/end times must be finite.")
        if end <= start:
            raise ValueError(
                f"Explicit zoom interval is invalid: start={start} ms, end={end} ms. "
                "Require end > start."
            )

        t_min = float(np.min(t))
        t_max = float(np.max(t))
        if start < t_min or end > t_max:
            raise ValueError(
                f"Explicit zoom interval [{start}, {end}] ms lies outside the available "
                f"signal interval [{t_min}, {t_max}] ms."
            )
        return start, end

    return choose_zoom_window(t, traces, window_ms=zoom_window_ms)


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
    add_zoom_inset: bool = True,
    zoom_window_ms: float = 300.0,
    zoom_start_ms: Optional[float] = None,
    zoom_end_ms: Optional[float] = None,
    zoom_inset_left: float = 0.64,
    zoom_inset_bottom: float = 0.57,
    zoom_inset_width: float = 0.33,
    zoom_inset_height: float = 0.38,
) -> None:
    grid_y = channel // n_points_xy
    grid_x = channel % n_points_xy
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    traces = get_channel_traces(cases, channel, subtract_mean=subtract_mean)
    for name, y in traces.items():
        ax.plot(t, y, linewidth=1.1, label=CASE_DISPLAY.get(name, name))
    main_ylim = global_ylim if global_ylim is not None else get_ylim_from_traces(traces)
    ax.set_ylim(*main_ylim)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("sEMG [mV]")
    ax.set_title(
        f"g-{group_number}, Protocol {protocol} | Electrode {channel:03d} "
        f"(y={grid_y}, x={grid_x})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    if add_zoom_inset:
        zoom_start, zoom_end = resolve_zoom_interval(
            t,
            traces,
            zoom_window_ms=zoom_window_ms,
            zoom_start_ms=zoom_start_ms,
            zoom_end_ms=zoom_end_ms,
        )
        zoom_ylim = get_segment_ylim(t, traces, zoom_start, zoom_end)
        ax.axvspan(zoom_start, zoom_end, color="gray", alpha=0.10)

        inset_box = validate_inset_box(
            zoom_inset_left, zoom_inset_bottom, zoom_inset_width, zoom_inset_height
        )
        axins = ax.inset_axes(inset_box)
        for name, y in traces.items():
            axins.plot(t, y, linewidth=1.0)
        axins.set_xlim(zoom_start, zoom_end)
        axins.set_ylim(*zoom_ylim)
        axins.grid(True, alpha=0.20)
        axins.set_title(f"Zoom ({zoom_start:.0f}–{zoom_end:.0f} ms)", fontsize=PLOT_FONT_SIZE)
        axins.tick_params(labelsize=PLOT_FONT_SIZE)
        try:
            ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8)
        except Exception:
            pass
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
def psd_similarity_array(df: pd.DataFrame, case: str) -> np.ndarray:
    return df[f"{case}_psd_similarity_vs_healthy"].to_numpy(dtype=float)

def local_waveform_similarity_array(df: pd.DataFrame, case: str) -> np.ndarray:
    return df[f"{case}_local_waveform_similarity_vs_healthy"].to_numpy(dtype=float)
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
    ax_sim = axes[1, 3]
    im_sim = ax_sim.imshow(
        to_grid(psd_similarity_array(df, "death_50"), n_points_xy, n_points_z),
        aspect="auto", origin="upper", vmin=0.0, vmax=1.0
    )
    ax_sim.set_title("PSD similarity: 50% vs Healthy", pad=8)
    set_heatmap_axis_style(ax_sim)
    panel_label(ax_sim, "H")
    cbar3 = fig.colorbar(im_sim, ax=ax_sim, shrink=0.88, pad=0.015, aspect=28)
    cbar3.set_label("PSD cosine similarity")
    fig.suptitle(f"g-{group_number}, Protocol {protocol}: spatial EMG metrics")
    save_png_pdf(fig, out_dir / "figure1_spatial_rms_and_psd_similarity")
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
    fig2, axes2 = plt.subplots(1, 2, figsize=(15.5, 5.2), constrained_layout=True)
    psd_data = [psd_similarity_array(df, case) for case in DISEASE_CASES]
    axes2[0].boxplot(psd_data, tick_labels=[CASE_DISPLAY[c] for c in DISEASE_CASES], showfliers=False)
    axes2[0].set_ylim(0.0, 1.02)
    axes2[0].set_ylabel("PSD cosine similarity")
    axes2[0].set_title("Phase-insensitive spectral similarity vs Healthy")
    axes2[0].grid(True, axis="y", alpha=0.25)
    panel_label(axes2[0], "A")
    local_data = [local_waveform_similarity_array(df, case) for case in DISEASE_CASES]
    axes2[1].boxplot(local_data, tick_labels=[CASE_DISPLAY[c] for c in DISEASE_CASES], showfliers=False)
    axes2[1].set_ylim(0.0, 1.02)
    axes2[1].set_ylabel("Mean windowed |r|")
    axes2[1].set_title("Local waveform similarity vs Healthy")
    axes2[1].grid(True, axis="y", alpha=0.25)
    panel_label(axes2[1], "B")
    fig2.suptitle(f"g-{group_number}, Protocol {protocol}: timing-robust similarity metrics")
    save_png_pdf(fig2, out_dir / "supplementary_similarity_boxplots")
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
    for similarity_metric, getter in [
        ("psd_similarity_vs_healthy", psd_similarity_array),
        ("local_waveform_similarity_vs_healthy", local_waveform_similarity_array),
    ]:
        row_sim: Dict[str, object] = {"metric": similarity_metric}
        for suffix in ("mean", "std", "median", "min", "max"):
            row_sim[f"healthy_{suffix}"] = np.nan
        for case in DISEASE_CASES:
            vals = getter(df, case)
            finite = vals[np.isfinite(vals)]
            row_sim[f"{case}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row_sim[f"{case}_std"] = float(np.std(finite)) if len(finite) else np.nan
            row_sim[f"{case}_median"] = float(np.median(finite)) if len(finite) else np.nan
            row_sim[f"{case}_min"] = float(np.min(finite)) if len(finite) else np.nan
            row_sim[f"{case}_max"] = float(np.max(finite)) if len(finite) else np.nan
        rows.append(row_sim)
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
    electrode_positions: Optional[Sequence[Tuple[int, int]]] = None,
    electrode_plot_only: bool = False,
    add_zoom_inset: bool = True,
    zoom_window_ms: float = 300.0,
    zoom_start_ms: Optional[float] = None,
    zoom_end_ms: Optional[float] = None,
    zoom_inset_left: float = 0.64,
    zoom_inset_bottom: float = 0.57,
    zoom_inset_width: float = 0.33,
    zoom_inset_height: float = 0.38,
    psd_window_ms: float = 1000.0,
    local_corr_window_ms: float = 250.0,
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
    print(
        f"  Computing timing-robust similarity metrics: PSD window={psd_window_ms:g} ms, "
        f"local correlation window={local_corr_window_ms:g} ms"
    )
    similarity_cache = precompute_similarity_cache(
        cases,
        subtract_mean=subtract_mean,
        psd_window_ms=psd_window_ms,
        local_corr_window_ms=local_corr_window_ms,
    )
    plot_channels = resolve_channels_to_plot(
        electrode_positions, n_points_xy, n_points_z, n_points, max_electrodes=max_electrodes
    )
    group_dir = out_root / f"g-{group_number}" / f"protocol_{protocol}"
    overlay_dir = group_dir / "emg_overlay_by_electrode"
    metric_plot_dir = group_dir / "metric_plots"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    metric_plot_dir.mkdir(parents=True, exist_ok=True)
    global_ylim = compute_global_ylim(cases, n_points, subtract_mean) if global_y else None
    metrics_rows: List[Dict[str, float]] = []
    pdf_pages = None
    if make_pdf_overlays and not skip_electrode_plots:
        pdf_pages = PdfPages(overlay_dir / "all_electrodes_overlay.pdf")
    for i_ch, ch in enumerate(plot_channels):
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
                add_zoom_inset=add_zoom_inset,
                zoom_window_ms=zoom_window_ms,
                zoom_start_ms=zoom_start_ms,
                zoom_end_ms=zoom_end_ms,
                zoom_inset_left=zoom_inset_left,
                zoom_inset_bottom=zoom_inset_bottom,
                zoom_inset_width=zoom_inset_width,
                zoom_inset_height=zoom_inset_height,
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
        metrics_rows.append(compute_metrics(
            cases, ch, subtract_mean=subtract_mean,
            psd_window_ms=psd_window_ms, local_corr_window_ms=local_corr_window_ms,
            similarity_cache=similarity_cache,
        ))
        processed_count = i_ch + 1
        if processed_count % 50 == 0 or processed_count == len(plot_channels):
            action = "processed" if skip_electrode_plots else "plotted/processed"
            print(f"    {action} {processed_count}/{len(plot_channels)} requested electrodes")
    if pdf_pages is not None:
        pdf_pages.close()
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_file = overlay_dir / "metrics_by_electrode.csv"
    if electrode_plot_only:
        print("  Electrode-plot-only mode: metrics CSV was not overwritten.")
        print("  Electrode-plot-only mode: skipped publication metric plots for this group/protocol.")
    else:
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
                "psd_similarity_vs_healthy": np.nan,
                "local_waveform_similarity_vs_healthy": np.nan,
            }
            psd_col = f"{stage}_psd_similarity_vs_healthy"
            if psd_col in metrics_df.columns:
                rec["psd_similarity_vs_healthy"] = float(row[psd_col])
            local_col = f"{stage}_local_waveform_similarity_vs_healthy"
            if local_col in metrics_df.columns:
                rec["local_waveform_similarity_vs_healthy"] = float(row[local_col])
            rows.append(rec)
    return pd.DataFrame(rows)
def create_group_level_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize spatial electrode distributions within each independent group."""
    records: List[Dict[str, object]] = []
    metrics = ["rms", "peak_to_peak", "max_abs", "mean", "psd_similarity_vs_healthy", "local_waveform_similarity_vs_healthy"]
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
        group_summary["metric"].isin(["rms", "peak_to_peak", "max_abs", "mean", "psd_similarity_vs_healthy", "local_waveform_similarity_vs_healthy"])
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
# Manuscript-level across-group figures and tables
# -----------------------------------------------------------------------------
MANUSCRIPT_METRICS: List[Tuple[str, str]] = [
    ("rms", "RMS [mV]"),
    ("peak_to_peak", "Peak-to-peak amplitude [mV]"),
    ("max_abs", "Maximum absolute EMG [mV]"),
]
def create_across_group_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize the independent group-level values for manuscript reporting.
    The input values are already spatial summaries within each group. Therefore,
    the SD reported here is variation *between independent stochastic groups*,
    not variation between electrodes.
    """
    rows: List[Dict[str, object]] = []
    for (protocol, stage, metric), sub in group_summary.groupby(
        ["protocol", "stage", "metric"], sort=True
    ):
        vals = sub["mean_across_electrodes"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        rows.append({
            "protocol": protocol,
            "stage": stage,
            "kill_fraction": KILL_FRACTION[stage],
            "metric": metric,
            "n_groups": int(len(vals)),
            "mean_across_groups": float(np.mean(vals)),
            "sd_across_groups": sd,
            "sem_across_groups": float(sd / np.sqrt(len(vals))) if np.isfinite(sd) else np.nan,
            "median_across_groups": float(np.median(vals)),
            "min_across_groups": float(np.min(vals)),
            "max_across_groups": float(np.max(vals)),
        })
    return pd.DataFrame(rows)
def create_normalized_to_healthy_group_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute within-group percent change from Healthy for each protocol/metric."""
    rows: List[Dict[str, object]] = []
    for (group, protocol, metric), sub in group_summary.groupby(
        ["group_number", "protocol", "metric"], sort=True
    ):
        healthy = sub[sub["stage"] == "healthy"]
        if healthy.empty:
            continue
        baseline = float(healthy["mean_across_electrodes"].iloc[0])
        for stage in STAGE_NAMES:
            row = sub[sub["stage"] == stage]
            if row.empty:
                continue
            value = float(row["mean_across_electrodes"].iloc[0])
            pct = np.nan if abs(baseline) <= 1e-15 else 100.0 * (value - baseline) / abs(baseline)
            ratio = np.nan if abs(baseline) <= 1e-15 else value / baseline
            rows.append({
                "group_number": int(group),
                "group_tag": f"g-{int(group)}",
                "protocol": protocol,
                "stage": stage,
                "kill_fraction": KILL_FRACTION[stage],
                "metric": metric,
                "healthy_value": baseline,
                "stage_value": value,
                "ratio_to_healthy": ratio,
                "percent_change_vs_healthy": pct,
            })
    return pd.DataFrame(rows)
def create_paired_across_group_summary(paired_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired Protocol B-A effects across independent groups."""
    if paired_summary.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for (stage, metric), sub in paired_summary.groupby(["stage", "metric"], sort=True):
        d = sub["B_minus_A"].to_numpy(dtype=float)
        pct = sub["percent_change_B_vs_A"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        pct = pct[np.isfinite(pct)]
        if len(d) == 0:
            continue
        sd_d = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
        sd_pct = float(np.std(pct, ddof=1)) if len(pct) > 1 else np.nan
        rows.append({
            "stage": stage,
            "kill_fraction": KILL_FRACTION[stage],
            "metric": metric,
            "n_groups": int(len(d)),
            "mean_B_minus_A": float(np.mean(d)),
            "sd_B_minus_A": sd_d,
            "median_B_minus_A": float(np.median(d)),
            "mean_percent_change_B_vs_A": float(np.mean(pct)) if len(pct) else np.nan,
            "sd_percent_change_B_vs_A": sd_pct,
            "median_percent_change_B_vs_A": float(np.median(pct)) if len(pct) else np.nan,
        })
    return pd.DataFrame(rows)
def _protocol_style(protocol: str) -> Tuple[str, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    if protocol == "A":
        return colors[0 % len(colors)], "--"
    return colors[1 % len(colors)], "-"
def _protocol_figure_prefix(protocol: str) -> str:
    protocol = str(protocol).upper()
    if protocol not in ("A", "B"):
        raise ValueError("protocol must be A or B")
    return protocol
def write_single_protocol_group_tables(
    group_summary: pd.DataFrame,
    across_group_summary: pd.DataFrame,
    out_dir: Path,
    protocol: str,
) -> None:
    """Write manuscript-facing group-level tables for one protocol."""
    protocol = str(protocol).upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    group_values = group_summary[group_summary["protocol"] == protocol].copy()
    group_values = group_values.sort_values(
        ["metric", "group_number", "kill_fraction"]
    ).reset_index(drop=True)
    group_values.to_csv(
        out_dir / f"protocol_{protocol}_group_level_values.csv", index=False
    )
    across = across_group_summary[across_group_summary["protocol"] == protocol].copy()
    across = across.sort_values(["metric", "kill_fraction"]).reset_index(drop=True)
    across.to_csv(
        out_dir / f"protocol_{protocol}_across_group_summary.csv", index=False
    )
def create_single_protocol_group_trajectories(
    group_summary: pd.DataFrame,
    out_dir: Path,
    protocol: str,
) -> None:
    """Group-level severity trajectories for one protocol only.
    This figure should precede any A-vs-B comparison in the manuscript. Each
    stochastic realization remains visible, while the bold trajectory reports
    mean ± SD across independent groups.
    """
    protocol = str(protocol).upper()
    psummary = group_summary[group_summary["protocol"] == protocol]
    if psummary.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), constrained_layout=True)
    x = np.arange(len(STAGE_NAMES))
    groups_here = sorted(psummary["group_number"].unique())
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])
    for panel, (ax, (metric, ylabel)) in enumerate(zip(axes, MANUSCRIPT_METRICS)):
        sub_metric = psummary[psummary["metric"] == metric]
        # Raw independent group realizations.
        for i, group in enumerate(groups_here):
            gsub = sub_metric[sub_metric["group_number"] == group]
            vals = []
            for stage in STAGE_NAMES:
                row = gsub[gsub["stage"] == stage]
                vals.append(
                    float(row["mean_across_electrodes"].iloc[0]) if len(row) else np.nan
                )
            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=1.15,
                markersize=4.5,
                alpha=0.48,
                color=cycle[i % len(cycle)],
                label=f"g-{int(group)}" if panel == 0 else None,
                zorder=1,
            )
        # Across-group mean ± SD.
        means, stds = [], []
        for stage in STAGE_NAMES:
            vals = sub_metric.loc[
                sub_metric["stage"] == stage, "mean_across_electrodes"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="D",
            markersize=7,
            linewidth=2.8,
            capsize=4,
            label="Across-group mean ± SD" if panel == 0 else None,
            zorder=4,
        )
        ax.set_xticks(x, ["Healthy", "25%", "50%", "75%"])
        ax.set_xlabel("Motor-unit loss condition")
        ax.set_ylabel(ylabel)
        ax.set_title(METRIC_DISPLAY_MANUSCRIPT.get(metric, metric))
        ax.grid(True, alpha=0.22)
        panel_label(ax, chr(ord("A") + panel))
    axes[0].legend(loc="best", fontsize=PLOT_FONT_SIZE)
    fig.suptitle(
        f"Protocol {protocol}: group-level EMG response to progressive motor-unit loss\n"
        "Individual stochastic realizations and across-group mean ± SD"
    )
    save_png_pdf(fig, out_dir / f"figure_{protocol}1_group_trajectories")
    plt.close(fig)
def create_single_protocol_normalized_trajectories(
    normalized_df: pd.DataFrame,
    out_dir: Path,
    protocol: str,
) -> None:
    """Within-protocol percent change relative to each group's Healthy baseline."""
    protocol = str(protocol).upper()
    psummary = normalized_df[normalized_df["protocol"] == protocol]
    if psummary.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    disease_stages = DISEASE_CASES
    x = np.arange(len(disease_stages))
    groups_here = sorted(psummary["group_number"].unique())
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), constrained_layout=True)
    for panel, (ax, (metric, _ylabel)) in enumerate(zip(axes, MANUSCRIPT_METRICS)):
        sub_metric = psummary[psummary["metric"] == metric]
        for i, group in enumerate(groups_here):
            gsub = sub_metric[sub_metric["group_number"] == group]
            vals = []
            for stage in disease_stages:
                row = gsub[gsub["stage"] == stage]
                vals.append(
                    float(row["percent_change_vs_healthy"].iloc[0])
                    if len(row) else np.nan
                )
            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=1.15,
                markersize=4.5,
                alpha=0.48,
                color=cycle[i % len(cycle)],
                label=f"g-{int(group)}" if panel == 0 else None,
                zorder=1,
            )
        means, stds = [], []
        for stage in disease_stages:
            vals = sub_metric.loc[
                sub_metric["stage"] == stage, "percent_change_vs_healthy"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="D",
            markersize=7,
            linewidth=2.8,
            capsize=4,
            label="Across-group mean ± SD" if panel == 0 else None,
            zorder=4,
        )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(x, ["25%", "50%", "75%"])
        ax.set_xlabel("Motor-unit loss condition")
        ax.set_ylabel("Change from Healthy [%]")
        ax.set_title(METRIC_DISPLAY_MANUSCRIPT.get(metric, metric))
        ax.grid(True, alpha=0.22)
        panel_label(ax, chr(ord("A") + panel))
    axes[0].legend(loc="best", fontsize=PLOT_FONT_SIZE)
    fig.suptitle(
        f"Protocol {protocol}: within-group EMG change relative to the paired Healthy case"
    )
    save_png_pdf(fig, out_dir / f"figure_{protocol}2_percent_change_vs_healthy")
    plt.close(fig)
def create_single_protocol_group_mean_spatial_maps(
    long_df: pd.DataFrame,
    out_dir: Path,
    protocol: str,
    metric: str,
    n_points_xy: int,
    n_points_z: int,
) -> None:
    """Across-group mean and SD spatial maps for one protocol only."""
    protocol = str(protocol).upper()
    p = long_df[long_df["protocol"] == protocol].copy()
    if p.empty:
        return
    expected = n_points_xy * n_points_z
    agg = (
        p.groupby(["stage", "electrode"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    arrays = []
    for stage in STAGE_NAMES:
        sub = agg[agg["stage"] == stage].sort_values("electrode")
        if len(sub) != expected:
            print(
                f"Warning: Protocol {protocol} spatial manuscript map for {metric} "
                f"expected {expected} electrodes for {stage}, found {len(sub)}; skipped."
            )
            return
        arrays.append(sub["mean"].to_numpy(dtype=float))
    vmin = min(float(np.nanmin(a)) for a in arrays)
    vmax = max(float(np.nanmax(a)) for a in arrays)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), constrained_layout=True)
    im = None
    for c, stage in enumerate(STAGE_NAMES):
        sub = agg[agg["stage"] == stage].sort_values("electrode")
        grid = sub["mean"].to_numpy(dtype=float).reshape(n_points_z, n_points_xy)
        im = axes[c].imshow(grid, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
        axes[c].set_title(CASE_DISPLAY[stage])
        axes[c].set_xlabel("Electrode x index")
        axes[c].set_ylabel("Electrode y index")
        panel_label(axes[c], chr(ord("A") + c))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.86, pad=0.015, aspect=35)
        cbar.set_label(dict(MANUSCRIPT_METRICS).get(metric, metric))
    fig.suptitle(
        f"Protocol {protocol}: across-group mean spatial "
        f"{METRIC_DISPLAY_MANUSCRIPT.get(metric, metric)}"
    )
    save_png_pdf(fig, out_dir / f"figure_{protocol}3_group_mean_{metric}_maps")
    plt.close(fig)
    if p["group_number"].nunique() > 1:
        sd_arrays = []
        for stage in STAGE_NAMES:
            sub = agg[agg["stage"] == stage].sort_values("electrode")
            sd_arrays.append(sub["std"].to_numpy(dtype=float))
        sdmax = max(float(np.nanmax(a)) for a in sd_arrays)
        fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), constrained_layout=True)
        im = None
        for c, stage in enumerate(STAGE_NAMES):
            sub = agg[agg["stage"] == stage].sort_values("electrode")
            grid = sub["std"].to_numpy(dtype=float).reshape(n_points_z, n_points_xy)
            im = axes[c].imshow(
                grid,
                aspect="auto",
                origin="upper",
                vmin=0.0,
                vmax=max(sdmax, 1e-15),
            )
            axes[c].set_title(CASE_DISPLAY[stage])
            axes[c].set_xlabel("Electrode x index")
            axes[c].set_ylabel("Electrode y index")
        if im is not None:
            cbar = fig.colorbar(im, ax=axes, shrink=0.86, pad=0.015, aspect=35)
            cbar.set_label(
                f"SD across groups: {dict(MANUSCRIPT_METRICS).get(metric, metric)}"
            )
        fig.suptitle(
            f"Protocol {protocol}: spatial variability across independent groups"
        )
        save_png_pdf(
            fig, out_dir / f"supplementary_{protocol}3_group_sd_{metric}_maps"
        )
        plt.close(fig)


def _group_mean_electrode_metric(
    long_df: pd.DataFrame,
    protocol: str,
    stage: str,
    metric: str,
) -> pd.DataFrame:
    """Return mean/SD/count across independent groups at each electrode."""
    sub = long_df[
        (long_df["protocol"] == str(protocol).upper())
        & (long_df["stage"] == stage)
    ][["group_number", "electrode", metric]].copy()
    sub = sub[np.isfinite(sub[metric].to_numpy(dtype=float))]
    if sub.empty:
        return pd.DataFrame(columns=["electrode", "mean", "std", "count"])
    return (
        sub.groupby("electrode", sort=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("electrode")
        .reset_index(drop=True)
    )


def _paired_group_electrode_difference(
    long_df: pd.DataFrame,
    protocol: str,
    stage: str,
    metric: str,
) -> pd.DataFrame:
    """Paired stage-minus-Healthy values, summarized across groups per electrode."""
    p = long_df[long_df["protocol"] == str(protocol).upper()][
        ["group_number", "stage", "electrode", metric]
    ].copy()
    stage_df = p[p["stage"] == stage].rename(columns={metric: "stage_value"})
    healthy_df = p[p["stage"] == "healthy"].rename(columns={metric: "healthy_value"})
    paired = stage_df.merge(
        healthy_df[["group_number", "electrode", "healthy_value"]],
        on=["group_number", "electrode"],
        how="inner",
        validate="one_to_one",
    )
    paired["value"] = paired["stage_value"] - paired["healthy_value"]
    return (
        paired.groupby("electrode", sort=True)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("electrode")
        .reset_index(drop=True)
    )


def _paired_group_electrode_ratio(
    long_df: pd.DataFrame,
    protocol: str,
    stage: str,
    metric: str,
    eps: float = 1e-15,
) -> pd.DataFrame:
    """Paired stage/Healthy ratio, summarized across groups per electrode."""
    p = long_df[long_df["protocol"] == str(protocol).upper()][
        ["group_number", "stage", "electrode", metric]
    ].copy()
    stage_df = p[p["stage"] == stage].rename(columns={metric: "stage_value"})
    healthy_df = p[p["stage"] == "healthy"].rename(columns={metric: "healthy_value"})
    paired = stage_df.merge(
        healthy_df[["group_number", "electrode", "healthy_value"]],
        on=["group_number", "electrode"],
        how="inner",
        validate="one_to_one",
    )
    denom = np.maximum(np.abs(paired["healthy_value"].to_numpy(dtype=float)), eps)
    paired["value"] = paired["stage_value"].to_numpy(dtype=float) / denom
    return (
        paired.groupby("electrode", sort=True)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("electrode")
        .reset_index(drop=True)
    )


def _group_mean_similarity_by_electrode(
    long_df: pd.DataFrame,
    protocol: str,
    stage: str,
    metric: str = "psd_similarity_vs_healthy",
) -> np.ndarray:
    """Mean electrode-level similarity across groups for one disease stage."""
    sub = long_df[
        (long_df["protocol"] == str(protocol).upper())
        & (long_df["stage"] == stage)
    ][["electrode", metric]].copy()
    sub = sub[np.isfinite(sub[metric].to_numpy(dtype=float))]
    if sub.empty:
        return np.asarray([], dtype=float)
    return (
        sub.groupby("electrode", sort=True)[metric]
        .mean()
        .to_numpy(dtype=float)
    )

def create_spatial_similarity_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """One spatial-map similarity value per independent group/protocol/stage.

    Cosine similarity of the RMS map is deliberately insensitive to a uniform
    amplitude scaling and therefore complements the amplitude metrics. Healthy
    self-similarity is reported as 1.0 for completeness.
    """
    rows: List[Dict[str, object]] = []
    for (group, protocol), sub in long_df.groupby(["group_number", "protocol"], sort=True):
        h = sub[sub["stage"] == "healthy"].sort_values("electrode")
        if h.empty:
            continue
        h_e = h["electrode"].to_numpy(dtype=int)
        h_rms = h["rms"].to_numpy(dtype=float)
        h_p2p = h["peak_to_peak"].to_numpy(dtype=float)
        for stage in STAGE_NAMES:
            d = sub[sub["stage"] == stage].sort_values("electrode")
            if d.empty or len(d) != len(h) or not np.array_equal(d["electrode"].to_numpy(dtype=int), h_e):
                continue
            rows.append({
                "group_number": int(group),
                "group_tag": f"g-{int(group)}",
                "protocol": str(protocol),
                "stage": stage,
                "kill_fraction": KILL_FRACTION[stage],
                "rms_map_cosine_vs_healthy": _cosine_similarity_nonnegative(
                    h_rms, d["rms"].to_numpy(dtype=float)
                ),
                "peak_to_peak_map_cosine_vs_healthy": _cosine_similarity_nonnegative(
                    h_p2p, d["peak_to_peak"].to_numpy(dtype=float)
                ),
            })
    return pd.DataFrame(rows)

def create_spatial_similarity_across_group_summary(spatial_df: pd.DataFrame) -> pd.DataFrame:
    if spatial_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for (protocol, stage), sub in spatial_df.groupby(["protocol", "stage"], sort=True):
        vals = sub["rms_map_cosine_vs_healthy"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            continue
        rows.append({
            "protocol": protocol,
            "stage": stage,
            "kill_fraction": KILL_FRACTION[stage],
            "n_groups": int(len(vals)),
            "mean_rms_map_cosine_vs_healthy": float(np.mean(vals)),
            "sd_rms_map_cosine_vs_healthy": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "min_rms_map_cosine_vs_healthy": float(np.min(vals)),
            "max_rms_map_cosine_vs_healthy": float(np.max(vals)),
        })
    return pd.DataFrame(rows)


def create_single_protocol_group_level_figure10(
    long_df: pd.DataFrame,
    out_dir: Path,
    protocol: str,
    n_points_xy: int,
    n_points_z: int,
) -> None:
    """Group-level counterpart of original Fig. 10, with Fig. 13 in panel H.

    A-D: across-group mean RMS maps.
    E-G: across-group mean of paired stage-minus-Healthy RMS maps.
    H: spatial RMS-map cosine similarity with Healthy. Each point is one
       independent stochastic group; diamonds/error bars show mean +/- SD.
    """
    protocol = str(protocol).upper()
    expected = int(n_points_xy) * int(n_points_z)
    out_dir.mkdir(parents=True, exist_ok=True)

    rms = {}
    for stage in STAGE_NAMES:
        a = _group_mean_electrode_metric(long_df, protocol, stage, "rms")
        if len(a) != expected:
            print(
                f"Warning: Protocol {protocol} group-level Fig.10 expected {expected} "
                f"electrodes for {stage}, found {len(a)}; skipped."
            )
            return
        rms[stage] = a["mean"].to_numpy(dtype=float)

    vmin = min(float(np.nanmin(v)) for v in rms.values())
    vmax = max(float(np.nanmax(v)) for v in rms.values())

    diffs = {}
    for stage in DISEASE_CASES:
        a = _paired_group_electrode_difference(long_df, protocol, stage, "rms")
        if len(a) != expected:
            print(
                f"Warning: Protocol {protocol} group-level Fig.10 difference map "
                f"for {stage} is incomplete; skipped."
            )
            return
        diffs[stage] = a["mean"].to_numpy(dtype=float)
    diff_abs = max(
        max(abs(float(np.nanmin(v))), abs(float(np.nanmax(v))))
        for v in diffs.values()
    )
    diff_abs = max(diff_abs, 1e-15)

    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    im_top = None
    for i, stage in enumerate(STAGE_NAMES):
        ax = axes[0, i]
        im_top = ax.imshow(
            rms[stage].reshape(n_points_z, n_points_xy),
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(CASE_DISPLAY[stage], pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("A") + i))
    cbar1 = fig.colorbar(im_top, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=28)
    cbar1.set_label("RMS [mV]")

    im_diff = None
    for i, stage in enumerate(DISEASE_CASES):
        ax = axes[1, i]
        im_diff = ax.imshow(
            diffs[stage].reshape(n_points_z, n_points_xy),
            aspect="auto",
            origin="upper",
            vmin=-diff_abs,
            vmax=diff_abs,
            cmap="coolwarm",
        )
        ax.set_title(f"{CASE_DISPLAY[stage]} - Healthy", pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("E") + i))
    cbar2 = fig.colorbar(im_diff, ax=axes[1, 0:3], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("Delta RMS [mV]")

    # Main similarity endpoint: spatial RMS-map cosine similarity. This is
    # insensitive to exact 30-s spike timing and uses group as the replicate.
    ax_sim = axes[1, 3]
    spatial = create_spatial_similarity_summary(
        long_df[long_df["protocol"] == protocol]
    )
    disease_spatial = spatial[spatial["stage"].isin(DISEASE_CASES)].copy()
    xpos = np.arange(len(DISEASE_CASES))
    if not disease_spatial.empty:
        means, stds = [], []
        for i, stage in enumerate(DISEASE_CASES):
            vals = disease_spatial.loc[
                disease_spatial["stage"] == stage, "rms_map_cosine_vs_healthy"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            if len(vals):
                jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax_sim.scatter(np.full(len(vals), xpos[i]) + jitter, vals, s=42, zorder=4)
        ax_sim.errorbar(
            xpos, means, yerr=stds, marker="D", markersize=7, linewidth=2.2,
            capsize=4, zorder=5, label="Mean +/- SD",
        )
        ax_sim.set_ylim(0.0, 1.02)
        ax_sim.set_xticks(xpos, ["25%", "50%", "75%"] )
        ax_sim.set_xlabel("Motor-unit loss condition")
        ax_sim.set_ylabel("RMS-map cosine similarity")
        ax_sim.set_title("Spatial similarity with Healthy", pad=8)
        ax_sim.grid(True, axis="y", alpha=0.25)
        ax_sim.legend(loc="best")
    else:
        ax_sim.text(
            0.5, 0.5, "Spatial similarity unavailable",
            ha="center", va="center", transform=ax_sim.transAxes,
        )
        ax_sim.set_axis_off()
    panel_label(ax_sim, "H")

    fig.suptitle(
        f"Protocol {protocol}: group-level spatial RMS maps and spatial similarity\n"
        "Maps show across-group means; differences and similarity are paired within group"
    )
    save_png_pdf(fig, out_dir / f"figure_{protocol}4_group_level_spatial_rms_and_similarity")
    plt.close(fig)


def create_single_protocol_group_level_figure11(
    long_df: pd.DataFrame,
    out_dir: Path,
    protocol: str,
) -> None:
    """Group-level counterpart of original Fig. 11.

    A-C: mean electrode profiles across groups with SD bands.
    D-E: distributions across electrode positions of the across-group mean field.
    F: spatial mean RMS for each independent group summarized as mean +/- SD across groups.
    """
    protocol = str(protocol).upper()
    p = long_df[long_df["protocol"] == protocol].copy()
    if p.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("rms", "Electrode-wise RMS", "RMS [mV]"),
        ("peak_to_peak", "Electrode-wise peak-to-peak", "Peak-to-peak [mV]"),
        ("max_abs", "Electrode-wise max |EMG|", "Max |EMG| [mV]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17.2, 10.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    stage_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", [f"C{i}" for i in range(10)]
    )

    # A-C: group mean electrode profile with between-group SD band.
    for panel, (metric, title, ylabel) in enumerate(metric_specs):
        ax = axes[0, panel]
        for i, stage in enumerate(STAGE_NAMES):
            agg = _group_mean_electrode_metric(long_df, protocol, stage, metric)
            if agg.empty:
                continue
            x = agg["electrode"].to_numpy(dtype=int)
            mean = agg["mean"].to_numpy(dtype=float)
            sd = agg["std"].fillna(0.0).to_numpy(dtype=float)
            color = stage_cycle[i % len(stage_cycle)]
            ax.plot(x, mean, linewidth=1.25, color=color, label=CASE_DISPLAY[stage])
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Electrode index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        panel_label(ax, chr(ord("A") + panel))
        if panel == 0:
            ax.legend(loc="best", fontsize=PLOT_FONT_SIZE)

    # D: RMS distribution over electrode positions of the group-mean field.
    ax = axes[1, 0]
    rms_box = []
    for stage in STAGE_NAMES:
        agg = _group_mean_electrode_metric(long_df, protocol, stage, "rms")
        rms_box.append(agg["mean"].to_numpy(dtype=float))
    ax.boxplot(rms_box, tick_labels=[CASE_DISPLAY[s] for s in STAGE_NAMES], showfliers=False)
    ax.set_title("RMS distribution across electrodes")
    ax.set_ylabel("RMS [mV]")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    panel_label(ax, "D")

    # E: Peak-to-peak distribution over electrode positions of group-mean field.
    ax = axes[1, 1]
    p2p_box = []
    for stage in STAGE_NAMES:
        agg = _group_mean_electrode_metric(long_df, protocol, stage, "peak_to_peak")
        p2p_box.append(agg["mean"].to_numpy(dtype=float))
    ax.boxplot(p2p_box, tick_labels=[CASE_DISPLAY[s] for s in STAGE_NAMES], showfliers=False)
    ax.set_title("Peak-to-peak distribution")
    ax.set_ylabel("Peak-to-peak [mV]")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    panel_label(ax, "E")

    # F: true group-level global RMS summary. One value per group and stage.
    ax = axes[1, 2]
    means, stds = [], []
    group_points = []
    for stage in STAGE_NAMES:
        stage_group = (
            p[p["stage"] == stage]
            .groupby("group_number", sort=True)["rms"]
            .mean()
            .to_numpy(dtype=float)
        )
        group_points.append(stage_group)
        means.append(float(np.mean(stage_group)) if len(stage_group) else np.nan)
        stds.append(float(np.std(stage_group, ddof=1)) if len(stage_group) > 1 else 0.0)
    xpos = np.arange(len(STAGE_NAMES))
    ax.bar(xpos, means, yerr=stds, capsize=4, alpha=0.82)
    # Show the independent group values explicitly on top of the summary.
    for i, vals in enumerate(group_points):
        if len(vals):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), xpos[i]) + jitter, vals, s=28, zorder=5)
    ax.set_xticks(xpos, [CASE_DISPLAY[s] for s in STAGE_NAMES], rotation=20)
    ax.set_ylabel("RMS [mV]")
    ax.set_title("Global RMS summary (mean +/- SD across groups)")
    ax.grid(True, axis="y", alpha=0.25)
    panel_label(ax, "F")

    fig.suptitle(
        f"Protocol {protocol}: group-level electrode EMG summary statistics\n"
        "Profiles and distributions use the across-group mean field; panel F uses group as the replicate"
    )
    save_png_pdf(fig, out_dir / f"figure_{protocol}5_group_level_metric_summaries")
    plt.close(fig)


def create_single_protocol_group_level_figure12(
    long_df: pd.DataFrame,
    out_dir: Path,
    protocol: str,
    n_points_xy: int,
    n_points_z: int,
) -> None:
    """Group-level counterpart of original Fig. 12.

    A-D: across-group mean peak-to-peak maps.
    E-G: mean of paired stage/Healthy ratios calculated within each group first.
    H: group mean electrode profile with SD bands.
    """
    protocol = str(protocol).upper()
    expected = int(n_points_xy) * int(n_points_z)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2p = {}
    for stage in STAGE_NAMES:
        agg = _group_mean_electrode_metric(long_df, protocol, stage, "peak_to_peak")
        if len(agg) != expected:
            print(
                f"Warning: Protocol {protocol} group-level Fig.12 expected {expected} "
                f"electrodes for {stage}, found {len(agg)}; skipped."
            )
            return
        p2p[stage] = agg["mean"].to_numpy(dtype=float)

    vmin = min(float(np.nanmin(v)) for v in p2p.values())
    vmax = max(float(np.nanmax(v)) for v in p2p.values())

    ratios = {}
    for stage in DISEASE_CASES:
        agg = _paired_group_electrode_ratio(long_df, protocol, stage, "peak_to_peak")
        if len(agg) != expected:
            print(
                f"Warning: Protocol {protocol} group-level Fig.12 ratio map "
                f"for {stage} is incomplete; skipped."
            )
            return
        ratios[stage] = agg["mean"].to_numpy(dtype=float)

    ratio_values = np.concatenate([v[np.isfinite(v)] for v in ratios.values()])
    if len(ratio_values):
        ratio_vmin = max(0.0, float(np.nanpercentile(ratio_values, 1.0)))
        ratio_vmax = float(np.nanpercentile(ratio_values, 99.0))
        if ratio_vmax <= ratio_vmin:
            ratio_vmax = ratio_vmin + 1e-12
    else:
        ratio_vmin, ratio_vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.10)

    im_top = None
    for i, stage in enumerate(STAGE_NAMES):
        ax = axes[0, i]
        im_top = ax.imshow(
            p2p[stage].reshape(n_points_z, n_points_xy),
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(CASE_DISPLAY[stage], pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("A") + i))
    cbar1 = fig.colorbar(im_top, ax=axes[0, :], shrink=0.88, pad=0.015, aspect=28)
    cbar1.set_label("Peak-to-peak [mV]")

    im_ratio = None
    for i, stage in enumerate(DISEASE_CASES):
        ax = axes[1, i]
        im_ratio = ax.imshow(
            ratios[stage].reshape(n_points_z, n_points_xy),
            aspect="auto",
            origin="upper",
            vmin=ratio_vmin,
            vmax=ratio_vmax,
        )
        ax.set_title(f"{CASE_DISPLAY[stage]} / Healthy", pad=8)
        set_heatmap_axis_style(ax)
        panel_label(ax, chr(ord("E") + i))
    cbar2 = fig.colorbar(im_ratio, ax=axes[1, 0:3], shrink=0.88, pad=0.015, aspect=28)
    cbar2.set_label("Peak-to-peak ratio")

    ax_line = axes[1, 3]
    stage_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", [f"C{i}" for i in range(10)]
    )
    for i, stage in enumerate(STAGE_NAMES):
        agg = _group_mean_electrode_metric(long_df, protocol, stage, "peak_to_peak")
        x = agg["electrode"].to_numpy(dtype=int)
        mean = agg["mean"].to_numpy(dtype=float)
        sd = agg["std"].fillna(0.0).to_numpy(dtype=float)
        color = stage_cycle[i % len(stage_cycle)]
        ax_line.plot(x, mean, linewidth=1.15, color=color, label=CASE_DISPLAY[stage])
        ax_line.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0)
    ax_line.set_title("Peak-to-peak across electrodes", pad=8)
    ax_line.set_xlabel("Electrode index")
    ax_line.set_ylabel("Peak-to-peak [mV]")
    ax_line.grid(True, alpha=0.25)
    ax_line.legend(loc="best", fontsize=PLOT_FONT_SIZE)
    panel_label(ax_line, "H")

    fig.suptitle(
        f"Protocol {protocol}: group-level peak-to-peak amplitude maps and ratios relative to Healthy\n"
        "Ratio maps are paired within group before averaging"
    )
    save_png_pdf(fig, out_dir / f"figure_{protocol}6_group_level_peak_to_peak_maps")
    plt.close(fig)


def create_manuscript_group_trajectories(group_summary: pd.DataFrame, out_dir: Path) -> None:
    """Main manuscript plot: raw group trajectories plus mean ± SD.
    Thin lines show the individual stochastic group realizations. Thick lines and
    error bars show the across-group mean ± SD. This is intentionally preferred
    over bars because the number of independent groups is small and the raw
    replicate behavior should remain visible.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), constrained_layout=True)
    x = np.arange(len(STAGE_NAMES))
    for panel, (ax, (metric, ylabel)) in enumerate(zip(axes, MANUSCRIPT_METRICS)):
        sub_metric = group_summary[group_summary["metric"] == metric]
        for protocol in PROTOCOLS:
            color, linestyle = _protocol_style(protocol)
            psub = sub_metric[sub_metric["protocol"] == protocol]
            if psub.empty:
                continue
            groups_here = sorted(psub["group_number"].unique())
            for group in groups_here:
                gsub = psub[psub["group_number"] == group]
                vals = []
                for stage in STAGE_NAMES:
                    row = gsub[gsub["stage"] == stage]
                    vals.append(float(row["mean_across_electrodes"].iloc[0]) if len(row) else np.nan)
                ax.plot(
                    x, vals, color=color, linestyle=linestyle, marker="o",
                    linewidth=0.9, markersize=4, alpha=0.30, zorder=1,
                )
            means, stds = [], []
            for stage in STAGE_NAMES:
                vals = psub.loc[
                    psub["stage"] == stage, "mean_across_electrodes"
                ].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            ax.errorbar(
                x, means, yerr=stds, color=color, linestyle=linestyle,
                marker="o", markersize=7, linewidth=2.4, capsize=4,
                label=f"Protocol {protocol}: mean ± SD", zorder=3,
            )
        ax.set_xticks(x, ["Healthy", "25%", "50%", "75%"])
        ax.set_xlabel("Motor-unit loss condition")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.replace(" [mV]", ""))
        ax.grid(True, alpha=0.22)
        panel_label(ax, chr(ord("A") + panel))
        if panel == 0:
            ax.legend(loc="best", fontsize=PLOT_FONT_SIZE)
    fig.suptitle(
        "Group-level EMG response to progressive motor-unit loss\n"
        "Thin lines: individual stochastic realizations; thick lines: mean ± SD across groups"
    )
    save_png_pdf(fig, out_dir / "figure_C1_protocol_A_vs_B_group_trajectories")
    plt.close(fig)
def create_manuscript_normalized_trajectories(normalized_df: pd.DataFrame, out_dir: Path) -> None:
    """Within-group percent change relative to the corresponding Healthy case."""
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), constrained_layout=True)
    disease_stages = DISEASE_CASES
    x = np.arange(len(disease_stages))
    for panel, (ax, (metric, _ylabel)) in enumerate(zip(axes, MANUSCRIPT_METRICS)):
        sub_metric = normalized_df[normalized_df["metric"] == metric]
        for protocol in PROTOCOLS:
            color, linestyle = _protocol_style(protocol)
            psub = sub_metric[sub_metric["protocol"] == protocol]
            if psub.empty:
                continue
            for group in sorted(psub["group_number"].unique()):
                gsub = psub[psub["group_number"] == group]
                vals = []
                for stage in disease_stages:
                    row = gsub[gsub["stage"] == stage]
                    vals.append(float(row["percent_change_vs_healthy"].iloc[0]) if len(row) else np.nan)
                ax.plot(
                    x, vals, color=color, linestyle=linestyle, marker="o",
                    linewidth=0.9, markersize=4, alpha=0.30,
                )
            means, stds = [], []
            for stage in disease_stages:
                vals = psub.loc[
                    psub["stage"] == stage, "percent_change_vs_healthy"
                ].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            ax.errorbar(
                x, means, yerr=stds, color=color, linestyle=linestyle,
                marker="o", markersize=7, linewidth=2.4, capsize=4,
                label=f"Protocol {protocol}: mean ± SD",
            )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(x, ["25%", "50%", "75%"])
        ax.set_xlabel("Motor-unit loss condition")
        ax.set_ylabel("Change from Healthy [%]")
        ax.set_title(METRIC_DISPLAY_MANUSCRIPT.get(metric, metric))
        ax.grid(True, alpha=0.22)
        panel_label(ax, chr(ord("A") + panel))
        if panel == 0:
            ax.legend(loc="best", fontsize=PLOT_FONT_SIZE)
    fig.suptitle("Within-group EMG change relative to the paired Healthy realization")
    save_png_pdf(fig, out_dir / "figure_C2_protocol_A_vs_B_percent_change_vs_healthy")
    plt.close(fig)
METRIC_DISPLAY_MANUSCRIPT = {
    "rms": "RMS",
    "peak_to_peak": "Peak-to-peak amplitude",
    "max_abs": "Maximum absolute EMG",
    "psd_similarity_vs_healthy": "PSD similarity vs Healthy",
    "local_waveform_similarity_vs_healthy": "Local waveform similarity vs Healthy",
    "mean": "Mean EMG",
}
def _create_protocol_effect_figure(
    paired_summary: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    ylabel: str,
    filename: str,
    title: str,
) -> None:
    if paired_summary.empty or "A" not in paired_summary.columns and "protocol_A_mean" not in paired_summary.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), constrained_layout=True)
    x = np.arange(len(STAGE_NAMES))
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    for panel, (ax, (metric, _metric_ylabel)) in enumerate(zip(axes, MANUSCRIPT_METRICS)):
        sub_metric = paired_summary[paired_summary["metric"] == metric]
        for i, group in enumerate(sorted(sub_metric["group_number"].unique())):
            gsub = sub_metric[sub_metric["group_number"] == group]
            vals = []
            for stage in STAGE_NAMES:
                row = gsub[gsub["stage"] == stage]
                vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
            ax.plot(
                x, vals, marker="o", linewidth=1.0, markersize=4,
                alpha=0.55, color=cycle[i % len(cycle)], label=f"g-{group}" if panel == 0 else None,
            )
        means, stds = [], []
        for stage in STAGE_NAMES:
            vals = sub_metric.loc[sub_metric["stage"] == stage, value_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        mean_color = cycle[min(3, len(cycle) - 1)]
        ax.errorbar(
            x, means, yerr=stds, marker="s", markersize=7,
            linewidth=2.6, linestyle="--", capsize=4, color=mean_color,
            label="Across-group mean ± SD" if panel == 0 else None,
            zorder=4,
        )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(x, ["Healthy", "25%", "50%", "75%"])
        ax.set_xlabel("Motor-unit loss condition")
        ax.set_ylabel(ylabel)
        ax.set_title(METRIC_DISPLAY_MANUSCRIPT[metric])
        ax.grid(True, alpha=0.22)
        panel_label(ax, chr(ord("A") + panel))
    axes[0].legend(loc="best", fontsize=PLOT_FONT_SIZE)
    fig.suptitle(title)
    save_png_pdf(fig, out_dir / filename)
    plt.close(fig)
def create_manuscript_protocol_effect_figures(paired_summary: pd.DataFrame, out_dir: Path) -> None:
    if paired_summary.empty:
        return
    _create_protocol_effect_figure(
        paired_summary,
        out_dir,
        value_col="percent_change_B_vs_A",
        ylabel="Protocol B change relative to A [%]",
        filename="figure_C3_protocol_B_effect_percent",
        title="Paired effect of increased firing rate: Protocol B relative to Protocol A",
    )
    _create_protocol_effect_figure(
        paired_summary,
        out_dir,
        value_col="B_minus_A",
        ylabel="Protocol B − Protocol A [mV]",
        filename="figure_C4_protocol_B_effect_absolute",
        title="Absolute paired Protocol B − A effect",
    )
def _common_grid_from_manifest(manifest_df: pd.DataFrame) -> Optional[Tuple[int, int]]:
    if manifest_df.empty:
        return None
    pairs = sorted({
        (int(x), int(z))
        for x, z in zip(manifest_df["n_points_xy"], manifest_df["n_points_z"])
    })
    if len(pairs) != 1:
        print(f"Warning: electrode grids differ across cases; manuscript spatial maps skipped: {pairs}")
        return None
    return pairs[0]
def _complete_spatial_data(long_df: pd.DataFrame, n_points_xy: int, n_points_z: int) -> bool:
    expected = int(n_points_xy) * int(n_points_z)
    counts = long_df.groupby(["group_number", "protocol", "stage"])["electrode"].nunique()
    if counts.empty:
        return False
    bad = counts[counts != expected]
    if len(bad):
        print(
            "Warning: manuscript spatial maps require all electrodes. "
            f"Expected {expected} electrodes per case, but found incomplete entries; maps skipped."
        )
        return False
    return True
def _spatial_metric_aggregate(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        long_df.groupby(["protocol", "stage", "electrode"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
def create_manuscript_group_mean_spatial_maps(
    long_df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    n_points_xy: int,
    n_points_z: int,
) -> None:
    protocols_present = [p for p in PROTOCOLS if p in set(long_df["protocol"])]
    if not protocols_present:
        return
    agg = _spatial_metric_aggregate(long_df, metric)
    mean_arrays: List[np.ndarray] = []
    for protocol in protocols_present:
        for stage in STAGE_NAMES:
            sub = agg[(agg["protocol"] == protocol) & (agg["stage"] == stage)].sort_values("electrode")
            if len(sub) == n_points_xy * n_points_z:
                mean_arrays.append(sub["mean"].to_numpy(dtype=float))
    if not mean_arrays:
        return
    vmin = min(float(np.nanmin(a)) for a in mean_arrays)
    vmax = max(float(np.nanmax(a)) for a in mean_arrays)
    fig, axes = plt.subplots(
        len(protocols_present), 4,
        figsize=(17.5, 4.2 * len(protocols_present)),
        squeeze=False, constrained_layout=True,
    )
    im = None
    for r, protocol in enumerate(protocols_present):
        for c, stage in enumerate(STAGE_NAMES):
            ax = axes[r, c]
            sub = agg[(agg["protocol"] == protocol) & (agg["stage"] == stage)].sort_values("electrode")
            vals = sub["mean"].to_numpy(dtype=float)
            grid = vals.reshape(n_points_z, n_points_xy)
            im = ax.imshow(grid, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
            ax.set_title(CASE_DISPLAY[stage])
            ax.set_xlabel("Electrode x index")
            ax.set_ylabel(
                f"Protocol {protocol}\nElectrode y index" if c == 0 else ""
            )
            panel_label(ax, chr(ord("A") + r * 4 + c))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.015, aspect=35)
        cbar.set_label(dict(MANUSCRIPT_METRICS).get(metric, metric))
    fig.suptitle(f"Across-group mean spatial {METRIC_DISPLAY_MANUSCRIPT.get(metric, metric)}")
    save_png_pdf(fig, out_dir / f"figure_C5_protocol_A_vs_B_group_mean_{metric}_maps")
    plt.close(fig)
    # Supplementary variability map: SD across independent groups at every electrode.
    if long_df["group_number"].nunique() > 1:
        sd_arrays = []
        for protocol in protocols_present:
            for stage in STAGE_NAMES:
                sub = agg[(agg["protocol"] == protocol) & (agg["stage"] == stage)].sort_values("electrode")
                if len(sub) == n_points_xy * n_points_z:
                    sd_arrays.append(sub["std"].to_numpy(dtype=float))
        finite_max = max(float(np.nanmax(a)) for a in sd_arrays) if sd_arrays else 0.0
        fig, axes = plt.subplots(
            len(protocols_present), 4,
            figsize=(17.5, 4.2 * len(protocols_present)),
            squeeze=False, constrained_layout=True,
        )
        im = None
        for r, protocol in enumerate(protocols_present):
            for c, stage in enumerate(STAGE_NAMES):
                ax = axes[r, c]
                sub = agg[(agg["protocol"] == protocol) & (agg["stage"] == stage)].sort_values("electrode")
                vals = sub["std"].to_numpy(dtype=float)
                grid = vals.reshape(n_points_z, n_points_xy)
                im = ax.imshow(grid, aspect="auto", origin="upper", vmin=0.0, vmax=max(finite_max, 1e-15))
                ax.set_title(CASE_DISPLAY[stage])
                ax.set_xlabel("Electrode x index")
                ax.set_ylabel(
                    f"Protocol {protocol}\nElectrode y index" if c == 0 else ""
                )
        if im is not None:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.015, aspect=35)
            cbar.set_label(f"SD across groups: {dict(MANUSCRIPT_METRICS).get(metric, metric)}")
        fig.suptitle(f"Spatial variability across independent groups: {METRIC_DISPLAY_MANUSCRIPT.get(metric, metric)}")
        save_png_pdf(fig, out_dir / f"supplementary_C5_group_sd_{metric}_maps")
        plt.close(fig)
def create_manuscript_paired_spatial_effect_maps(
    long_df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    n_points_xy: int,
    n_points_z: int,
) -> None:
    """Average the electrode-wise paired B-A effect across groups."""
    if not {"A", "B"}.issubset(set(long_df["protocol"])):
        return
    piv = long_df.pivot_table(
        index=["group_number", "stage", "electrode"],
        columns="protocol",
        values=metric,
        aggfunc="first",
    ).reset_index()
    piv.columns.name = None
    if "A" not in piv.columns or "B" not in piv.columns:
        return
    piv["B_minus_A"] = piv["B"] - piv["A"]
    agg = (
        piv.groupby(["stage", "electrode"], as_index=False)["B_minus_A"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    arrays = []
    for stage in STAGE_NAMES:
        sub = agg[agg["stage"] == stage].sort_values("electrode")
        if len(sub) == n_points_xy * n_points_z:
            arrays.append(sub["mean"].to_numpy(dtype=float))
    if len(arrays) != len(STAGE_NAMES):
        return
    vmax = max(max(abs(float(np.nanmin(a))), abs(float(np.nanmax(a)))) for a in arrays)
    vmax = max(vmax, 1e-15)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), constrained_layout=True)
    im = None
    for c, stage in enumerate(STAGE_NAMES):
        sub = agg[agg["stage"] == stage].sort_values("electrode")
        vals = sub["mean"].to_numpy(dtype=float)
        grid = vals.reshape(n_points_z, n_points_xy)
        im = axes[c].imshow(
            grid, aspect="auto", origin="upper", cmap="coolwarm",
            vmin=-vmax, vmax=vmax,
        )
        axes[c].set_title(CASE_DISPLAY[stage])
        axes[c].set_xlabel("Electrode x index")
        axes[c].set_ylabel("Electrode y index")
        panel_label(axes[c], chr(ord("A") + c))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.86, pad=0.015, aspect=35)
        cbar.set_label(f"Mean paired B − A {dict(MANUSCRIPT_METRICS).get(metric, metric)}")
    fig.suptitle(
        f"Across-group mean spatial effect of Protocol B: {METRIC_DISPLAY_MANUSCRIPT.get(metric, metric)}"
    )
    save_png_pdf(fig, out_dir / f"figure_C6_group_mean_B_minus_A_{metric}_maps")
    plt.close(fig)
    if piv["group_number"].nunique() > 1:
        sdmax = max(
            float(np.nanmax(agg.loc[agg["stage"] == stage, "std"].to_numpy(dtype=float)))
            for stage in STAGE_NAMES
        )
        fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), constrained_layout=True)
        im = None
        for c, stage in enumerate(STAGE_NAMES):
            sub = agg[agg["stage"] == stage].sort_values("electrode")
            vals = sub["std"].to_numpy(dtype=float)
            grid = vals.reshape(n_points_z, n_points_xy)
            im = axes[c].imshow(grid, aspect="auto", origin="upper", vmin=0.0, vmax=max(sdmax, 1e-15))
            axes[c].set_title(CASE_DISPLAY[stage])
            axes[c].set_xlabel("Electrode x index")
            axes[c].set_ylabel("Electrode y index")
        if im is not None:
            cbar = fig.colorbar(im, ax=axes, shrink=0.86, pad=0.015, aspect=35)
            cbar.set_label(f"SD across groups of paired B − A {dict(MANUSCRIPT_METRICS).get(metric, metric)}")
        fig.suptitle(
            f"Spatial variability of the paired Protocol B − A effect: {METRIC_DISPLAY_MANUSCRIPT.get(metric, metric)}"
        )
        save_png_pdf(fig, out_dir / f"supplementary_C6_group_sd_B_minus_A_{metric}_maps")
        plt.close(fig)
def write_manuscript_figure_readme(out_dir: Path, n_groups: int, spatial_metric: str) -> None:
    lines = [
        "Manuscript-level grouped EMG figures",
        "",
        f"Independent stochastic groups: {n_groups}",
        "Statistical replicate: group.",
        "Electrodes are spatial observations within a group and are not independent replicates.",
        "",
        "Recommended manuscript narrative order:",
        "",
        "1) Establish Protocol A behavior before any cross-protocol comparison",
        "   01_protocol_A/figure_A1_group_trajectories.png/pdf",
        "     Absolute RMS, peak-to-peak, and maximum absolute EMG across MU-loss severity.",
        "     Individual group realizations are shown together with mean ± SD across groups.",
        "   01_protocol_A/figure_A2_percent_change_vs_healthy.png/pdf",
        "     Within-group change relative to each group's own Healthy baseline.",
        f"   01_protocol_A/figure_A3_group_mean_{spatial_metric}_maps.png/pdf",
        "   01_protocol_A/figure_A4_group_level_spatial_rms_and_similarity.png/pdf",
        "   01_protocol_A/figure_A5_group_level_metric_summaries.png/pdf",
        "   01_protocol_A/figure_A6_group_level_peak_to_peak_maps.png/pdf",
        "     Across-group mean spatial phenotype for Protocol A.",
        "",
        "2) Establish Protocol B behavior independently",
        "   02_protocol_B/figure_B1_group_trajectories.png/pdf",
        "   02_protocol_B/figure_B2_percent_change_vs_healthy.png/pdf",
        f"   02_protocol_B/figure_B3_group_mean_{spatial_metric}_maps.png/pdf",
        "   02_protocol_B/figure_B4_group_level_spatial_rms_and_similarity.png/pdf",
        "   02_protocol_B/figure_B5_group_level_metric_summaries.png/pdf",
        "   02_protocol_B/figure_B6_group_level_peak_to_peak_maps.png/pdf",
        "",
        "3) Only then compare Protocol A with Protocol B",
        "   03_protocol_comparison/figure_C1_protocol_A_vs_B_group_trajectories.png/pdf",
        "   03_protocol_comparison/figure_C2_protocol_A_vs_B_percent_change_vs_healthy.png/pdf",
        "   03_protocol_comparison/figure_C3_protocol_B_effect_percent.png/pdf",
        "   03_protocol_comparison/figure_C4_protocol_B_effect_absolute.png/pdf",
        f"   03_protocol_comparison/figure_C5_protocol_A_vs_B_group_mean_{spatial_metric}_maps.png/pdf",
        f"   03_protocol_comparison/figure_C6_group_mean_B_minus_A_{spatial_metric}_maps.png/pdf",
        "",
        "Supplementary variability maps are retained in the same protocol/comparison folders.",
        "Case-level g-X/protocol_Y/metric_plots remain useful as supplementary QC figures.",
        "",
        "Reporting note:",
        "  With a small number of groups, keep all independent group trajectories visible.",
        "  Use mean ± SD to summarize between-group variability and avoid treating electrodes",
        "  as independent biological/statistical replicates.",
    ]
    (out_dir / "README_manuscript_figures.txt").write_text("\n".join(lines) + "\n")
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
        "--skip-manuscript-figures",
        action="store_true",
        help=(
            "Skip root-level manuscript figures. Across-group CSV summary tables are still written."
        ),
    )
    parser.add_argument(
        "--manuscript-spatial-metric",
        choices=["rms", "peak_to_peak", "max_abs"],
        default="rms",
        help="Metric used for manuscript group-mean and paired spatial maps. Default: rms.",
    )
    parser.add_argument(
        "--electrode-positions",
        nargs="+",
        default=None,
        help=(
            "Optional list of electrode positions in y,x format. Example: "
            "--electrode-positions 0,0 10,0 31,11. When provided, only those "
            "electrode overlay plots are updated."
        ),
    )
    parser.add_argument(
        "--electrode-plot-only",
        action="store_true",
        help=(
            "Only generate electrode overlay plots for the requested groups/protocols "
            "and selected electrode positions. Root-level aggregate and manuscript figures are skipped."
        ),
    )
    parser.add_argument(
        "--zoom-window-ms",
        type=float,
        default=300.0,
        help=(
            "Width of the automatic zoom-in inset window in ms. Used only when "
            "--zoom-start-ms/--zoom-end-ms are not supplied. Default: 300."
        ),
    )
    parser.add_argument(
        "--zoom-start-ms",
        type=float,
        default=None,
        help=(
            "Explicit start time [ms] for the zoom inset. Must be supplied together "
            "with --zoom-end-ms. If omitted, the script automatically selects a window."
        ),
    )
    parser.add_argument(
        "--zoom-end-ms",
        type=float,
        default=None,
        help=(
            "Explicit end time [ms] for the zoom inset. Must be supplied together "
            "with --zoom-start-ms. If omitted, the script automatically selects a window."
        ),
    )
    parser.add_argument(
        "--no-zoom-inset",
        action="store_true",
        help="Disable the automatic zoom-in inset on electrode overlay plots.",
    )
    parser.add_argument(
        "--zoom-inset-left",
        type=float,
        default=0.64,
        help="Left position of the zoom inset in axes coordinates [0,1]. Default: 0.64.",
    )
    parser.add_argument(
        "--zoom-inset-bottom",
        type=float,
        default=0.57,
        help="Bottom position of the zoom inset in axes coordinates [0,1]. Default: 0.57.",
    )
    parser.add_argument(
        "--zoom-inset-width",
        type=float,
        default=0.33,
        help="Width of the zoom inset in axes coordinates [0,1]. Default: 0.33.",
    )
    parser.add_argument(
        "--zoom-inset-height",
        type=float,
        default=0.38,
        help="Height of the zoom inset in axes coordinates [0,1]. Default: 0.38.",
    )
    parser.add_argument(
        "--psd-window-ms",
        type=float,
        default=1000.0,
        help=(
            "Welch segment length [ms] for phase-insensitive PSD cosine similarity. "
            "Default: 1000 ms."
        ),
    )
    parser.add_argument(
        "--local-corr-window-ms",
        type=float,
        default=250.0,
        help=(
            "Window length [ms] for the secondary local mean-|r| waveform similarity. "
            "Default: 250 ms."
        ),
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
    requested_positions = parse_electrode_position_specs(args.electrode_positions)
    if (args.zoom_start_ms is None) != (args.zoom_end_ms is None):
        raise ValueError(
            "--zoom-start-ms and --zoom-end-ms must be supplied together. "
            "Omit both to use --zoom-window-ms automatic selection."
        )
    if args.zoom_start_ms is not None and args.zoom_end_ms <= args.zoom_start_ms:
        raise ValueError("--zoom-end-ms must be greater than --zoom-start-ms.")
    validate_inset_box(
        args.zoom_inset_left,
        args.zoom_inset_bottom,
        args.zoom_inset_width,
        args.zoom_inset_height,
    )
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
    if args.electrode_plot_only:
        print("\nElectrode-plot-only mode enabled.")
        if requested_positions:
            print("Requested electrode positions: " + ", ".join(f"(y={y}, x={x})" for y, x in requested_positions))
        else:
            print("No specific electrode positions were supplied; plots will be generated for all discovered electrodes.")
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
                electrode_positions=requested_positions,
                electrode_plot_only=args.electrode_plot_only,
                add_zoom_inset=(not args.no_zoom_inset),
                zoom_window_ms=args.zoom_window_ms,
                zoom_start_ms=args.zoom_start_ms,
                zoom_end_ms=args.zoom_end_ms,
                zoom_inset_left=args.zoom_inset_left,
                zoom_inset_bottom=args.zoom_inset_bottom,
                zoom_inset_width=args.zoom_inset_width,
                zoom_inset_height=args.zoom_inset_height,
                psd_window_ms=args.psd_window_ms,
                local_corr_window_ms=args.local_corr_window_ms,
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
    if args.electrode_plot_only:
        print("\nDone. Electrode plots were updated without regenerating group/manuscript aggregates.")
        print(f"Output root: {out_root.resolve()}")
        return
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
    # Manuscript-oriented across-group summary tables. These are descriptive
    # summaries across independent stochastic groups, not across electrodes.
    across_group_summary = create_across_group_summary(group_summary)
    across_group_summary.to_csv(out_root / "across_group_summary.csv", index=False)
    normalized_group_summary = create_normalized_to_healthy_group_summary(group_summary)
    normalized_group_summary.to_csv(
        out_root / "normalized_to_healthy_group_summary.csv", index=False
    )
    paired_across_group_summary = create_paired_across_group_summary(paired_summary)
    paired_across_group_summary.to_csv(
        out_root / "paired_protocol_across_group_summary.csv", index=False
    )
    spatial_similarity_summary = create_spatial_similarity_summary(long_df)
    spatial_similarity_summary.to_csv(
        out_root / "spatial_similarity_summary.csv", index=False
    )
    spatial_similarity_across_group = create_spatial_similarity_across_group_summary(
        spatial_similarity_summary
    )
    spatial_similarity_across_group.to_csv(
        out_root / "spatial_similarity_across_group_summary.csv", index=False
    )
    manuscript_dir = out_root / "manuscript_figures"
    if not args.skip_manuscript_figures:
        manuscript_dir.mkdir(parents=True, exist_ok=True)
        protocol_dirs = {
            "A": manuscript_dir / "01_protocol_A",
            "B": manuscript_dir / "02_protocol_B",
        }
        comparison_dir = manuscript_dir / "03_protocol_comparison"
        # Manuscript sequence: characterize each protocol independently first.
        for protocol in [p for p in PROTOCOLS if p in set(protocols)]:
            pdir = protocol_dirs[protocol]
            pdir.mkdir(parents=True, exist_ok=True)
            write_single_protocol_group_tables(
                group_summary, across_group_summary, pdir, protocol
            )
            create_single_protocol_group_trajectories(
                group_summary, pdir, protocol
            )
            # Group-level counterparts of the original case-level manuscript figures.
            # Figure A/B5 corresponds to original Fig. 11 and does not require reshaping.
            create_single_protocol_group_level_figure11(
                long_df, pdir, protocol
            )
            if not normalized_group_summary.empty:
                create_single_protocol_normalized_trajectories(
                    normalized_group_summary, pdir, protocol
                )
        grid = _common_grid_from_manifest(manifest_df)
        spatial_ready = False
        if grid is not None:
            n_points_xy, n_points_z = grid
            spatial_ready = _complete_spatial_data(long_df, n_points_xy, n_points_z)
            if spatial_ready:
                for protocol in [p for p in PROTOCOLS if p in set(protocols)]:
                    create_single_protocol_group_mean_spatial_maps(
                        long_df,
                        protocol_dirs[protocol],
                        protocol,
                        args.manuscript_spatial_metric,
                        n_points_xy,
                        n_points_z,
                    )
                    # Three additional group-level figures requested for the manuscript:
                    # modified Fig.10 (with timing-robust spatial similarity in panel H), Fig.11, Fig.12.
                    create_single_protocol_group_level_figure10(
                        long_df,
                        protocol_dirs[protocol],
                        protocol,
                        n_points_xy,
                        n_points_z,
                    )
                    create_single_protocol_group_level_figure12(
                        long_df,
                        protocol_dirs[protocol],
                        protocol,
                        n_points_xy,
                        n_points_z,
                    )
        # Cross-protocol comparison comes after the standalone A and B figures.
        if {"A", "B"}.issubset(set(protocols)):
            comparison_dir.mkdir(parents=True, exist_ok=True)
            create_manuscript_group_trajectories(group_summary, comparison_dir)
            if not normalized_group_summary.empty:
                create_manuscript_normalized_trajectories(
                    normalized_group_summary, comparison_dir
                )
            if not paired_summary.empty:
                create_manuscript_protocol_effect_figures(
                    paired_summary, comparison_dir
                )
            if spatial_ready:
                create_manuscript_group_mean_spatial_maps(
                    long_df,
                    comparison_dir,
                    args.manuscript_spatial_metric,
                    n_points_xy,
                    n_points_z,
                )
                create_manuscript_paired_spatial_effect_maps(
                    long_df,
                    comparison_dir,
                    args.manuscript_spatial_metric,
                    n_points_xy,
                    n_points_z,
                )
        write_manuscript_figure_readme(
            manuscript_dir, len(groups), args.manuscript_spatial_metric
        )
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
    print(f"  {out_root / 'across_group_summary.csv'}")
    print(f"  {out_root / 'normalized_to_healthy_group_summary.csv'}")
    print(f"  {out_root / 'paired_protocol_across_group_summary.csv'}")
    print(f"  {out_root / 'spatial_similarity_summary.csv'}")
    print(f"  {out_root / 'spatial_similarity_across_group_summary.csv'}")
    if not args.skip_manuscript_figures:
        print(f"  {manuscript_dir}")
    print("\nFor inferential statistics, use group_number as the replicate/block.")
    print("Do not treat individual electrodes as independent biological replicates.")
if __name__ == "__main__":
    main()
