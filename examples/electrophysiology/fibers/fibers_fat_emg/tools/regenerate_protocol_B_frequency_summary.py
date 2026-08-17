#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate Protocol B firing-frequency summary figures from EXISTING case data.

This script does NOT regenerate Protocol B cases and does NOT run OpenDiHu.
It only reads the already-written Protocol B summary CSV files and recreates:

    protocol_B_frequency_summary_g-<N>.png
    protocol_B_frequency_summary_g-<N>.pdf

Expected grouped study layout
-----------------------------
    opendihu_repeated_denervation_study/
      g-1/
        protocol_B/
          protocol_B_compensation_overview_g-1.csv
          death_25/protocol_B_motor_unit_summary.csv
          death_50/protocol_B_motor_unit_summary.csv
          death_75/protocol_B_motor_unit_summary.csv
          healthy/protocol_B_motor_unit_summary.csv
      g-2/
        protocol_B/
          ...

The preferred input is protocol_B_compensation_overview_g-N.csv. If that file
is not present, the script automatically combines the four per-stage
protocol_B_motor_unit_summary.csv files.

Required columns
----------------
    stage
    mu_no_0_based
    active
    new_frequency_hz

Examples
--------
Regenerate all discovered groups:

    python tools/regenerate_protocol_B_frequency_summary.py \
        --study-root fiber_dist_generator/opendihu_repeated_denervation_study

Only g-1:

    python tools/regenerate_protocol_B_frequency_summary.py \
        --study-root fiber_dist_generator/opendihu_repeated_denervation_study \
        --groups 1

Use a larger font:

    python tools/regenerate_protocol_B_frequency_summary.py \
        --study-root fiber_dist_generator/opendihu_repeated_denervation_study \
        --font-size 16

Write to a separate directory instead of overwriting figures in protocol_B:

    python tools/regenerate_protocol_B_frequency_summary.py \
        --study-root fiber_dist_generator/opendihu_repeated_denervation_study \
        --out-dir regenerated_protocol_B_frequency_figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGES = ["healthy", "death_25", "death_50", "death_75"]
STAGE_DISPLAY = {
    "healthy": "Healthy",
    "death_25": "25% MU death",
    "death_50": "50% MU death",
    "death_75": "75% MU death",
}

REQUIRED_COLUMNS = {
    "stage",
    "mu_no_0_based",
    "active",
    "new_frequency_hz",
}


def configure_plot_style(font_size: float) -> None:
    """Set a journal-readable Matplotlib typography baseline."""
    fs = float(font_size)
    plt.rcParams.update({
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "figure.titlesize": fs + 1,
        "axes.linewidth": 0.9,
    })


def add_panel_label(ax, label: str) -> None:
    """Journal-style panel label in the upper-left corner."""
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=plt.rcParams["font.size"],
        bbox=dict(
            boxstyle="square,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
        ),
        zorder=20,
    )


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Robustly normalize common CSV boolean encodings."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    def one(v):
        if pd.isna(v):
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(v)
        s = str(v).strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0", ""}:
            return False
        raise ValueError("Cannot interpret active value as boolean: {!r}".format(v))

    return series.map(one).astype(bool)


def validate_summary(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "{} is missing required columns: {}\nAvailable columns: {}".format(
                source, missing, list(df.columns)
            )
        )

    out = df.copy()
    out["stage"] = out["stage"].astype(str).str.strip()
    out["mu_no_0_based"] = pd.to_numeric(out["mu_no_0_based"], errors="raise").astype(int)
    out["new_frequency_hz"] = pd.to_numeric(out["new_frequency_hz"], errors="raise").astype(float)
    out["active"] = parse_bool_series(out["active"])

    unknown = sorted(set(out["stage"]).difference(STAGES))
    if unknown:
        raise ValueError(
            "{} contains unrecognized stages: {}. Expected {}".format(
                source, unknown, STAGES
            )
        )

    # Keep canonical order and one row per stage x MU if possible.
    order = {s: i for i, s in enumerate(STAGES)}
    out["_stage_order"] = out["stage"].map(order)
    out = out.sort_values(["_stage_order", "mu_no_0_based"]).drop(columns="_stage_order")
    return out.reset_index(drop=True)


def discover_groups(study_root: Path, requested: Optional[Sequence[int]]) -> Dict[int, Path]:
    found: Dict[int, Path] = {}
    for path in sorted(study_root.glob("g-*/protocol_B")):
        if not path.is_dir():
            continue
        m = re.fullmatch(r"g-(\d+)", path.parent.name)
        if not m:
            continue
        g = int(m.group(1))
        if requested and g not in requested:
            continue
        found[g] = path

    if requested:
        missing = [int(g) for g in requested if int(g) not in found]
        if missing:
            raise FileNotFoundError(
                "Could not find protocol_B directories for groups: {} under {}".format(
                    missing, study_root
                )
            )

    if not found:
        raise FileNotFoundError(
            "No g-N/protocol_B directories were found under: {}".format(study_root)
        )
    return dict(sorted(found.items()))


def find_overview_csv(protocol_b_dir: Path, group_number: int) -> Optional[Path]:
    candidates = [
        protocol_b_dir / "protocol_B_compensation_overview_g-{}.csv".format(group_number),
        protocol_b_dir / "protocol_B_compensation_overview.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    fuzzy = sorted(protocol_b_dir.glob("protocol_B_compensation_overview*.csv"))
    return fuzzy[0] if fuzzy else None


def load_existing_summary(protocol_b_dir: Path, group_number: int) -> tuple[pd.DataFrame, List[Path]]:
    """Load overview CSV, or reconstruct it by concatenating existing stage CSVs."""
    overview = find_overview_csv(protocol_b_dir, group_number)
    if overview is not None:
        df = pd.read_csv(overview)
        return validate_summary(df, overview), [overview]

    frames: List[pd.DataFrame] = []
    sources: List[Path] = []
    for stage in STAGES:
        p = protocol_b_dir / stage / "protocol_B_motor_unit_summary.csv"
        if not p.exists():
            raise FileNotFoundError(
                "No Protocol B overview CSV was found, and the fallback stage file is missing:\n  {}".format(p)
            )
        df = pd.read_csv(p)
        # Older files normally already have stage. If absent, it is unambiguous here.
        if "stage" not in df.columns:
            df["stage"] = stage
        frames.append(df)
        sources.append(p)

    combined = pd.concat(frames, ignore_index=True)
    return validate_summary(combined, protocol_b_dir), sources


def audit_data(df: pd.DataFrame, group_number: int) -> None:
    print("  Data audit:")
    for stage in STAGES:
        sub = df[df["stage"] == stage]
        active = sub[sub["active"]]
        n_total = len(sub)
        n_active = len(active)
        mean = float(active["new_frequency_hz"].mean()) if n_active else np.nan
        sd = float(active["new_frequency_hz"].std(ddof=1)) if n_active > 1 else 0.0
        print(
            "    {:8s}: {:2d}/{:2d} active MUs, mean = {:6.2f} Hz, SD = {:6.2f} Hz".format(
                stage, n_active, n_total, mean, sd
            )
        )


def regenerate_figure(
    df: pd.DataFrame,
    group_number: int,
    out_base: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.14, wspace=0.12, hspace=0.10)

    # ------------------------------------------------------------------
    # A. Per-MU prescribed Protocol B frequencies
    # ------------------------------------------------------------------
    ax = axes[0]
    for stage in STAGES:
        sub = df[df["stage"] == stage].sort_values("mu_no_0_based")
        ax.plot(
            sub["mu_no_0_based"].to_numpy(dtype=int),
            sub["new_frequency_hz"].to_numpy(dtype=float),
            marker="o",
            markersize=5,
            linewidth=1.35,
            label=STAGE_DISPLAY[stage],
        )

    ax.set_xlabel("Motor-unit index")
    ax.set_ylabel("Protocol B stimulation frequency [Hz]")
    ax.set_title("Prescribed firing rates across motor units")
    ax.grid(True, alpha=0.22)
    # Place legend outside the plotting region so it never overlays the curves.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.20),
        ncol=2,
        frameon=True,
        borderaxespad=0.0,
    )
    add_panel_label(ax, "A")

    # Make integer MU indices easy to read without forcing every single label for larger models.
    mu_ids = np.sort(df["mu_no_0_based"].unique())
    if len(mu_ids) <= 24:
        ax.set_xticks(mu_ids[::2] if len(mu_ids) > 12 else mu_ids)

    # ------------------------------------------------------------------
    # B. Mean +/- SD among ACTIVE motor units only
    # ------------------------------------------------------------------
    ax = axes[1]
    means: List[float] = []
    stds: List[float] = []
    ns: List[int] = []

    for stage in STAGES:
        sub = df[(df["stage"] == stage) & (df["active"])]
        vals = sub["new_frequency_hz"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        means.append(float(np.mean(vals)) if len(vals) else np.nan)
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ns.append(int(len(vals)))

    xpos = np.arange(len(STAGES))
    ax.bar(xpos, means, yerr=stds, capsize=5, width=0.72)
    ax.set_xticks(xpos)
    ax.set_xticklabels([STAGE_DISPLAY[s] for s in STAGES], rotation=18, ha="right")
    ax.set_ylabel("Stimulation frequency [Hz]")
    ax.set_title("Mean ± SD among active motor units")
    ax.grid(True, axis="y", alpha=0.22)
    add_panel_label(ax, "B")

    # Expand the top y-limit to create headroom for the n-labels so they do not
    # touch or cross the top border line.
    finite_tops = [m + s for m, s in zip(means, stds) if np.isfinite(m) and np.isfinite(s)]
    if finite_tops:
        top = max(finite_tops)
        ax.set_ylim(0.0, max(top * 1.16, top + 2.0))

    # Add n above each bar with enough vertical offset and clipping disabled.
    for x, mean, sd, n in zip(xpos, means, stds, ns):
        if not np.isfinite(mean):
            continue
        y = mean + sd
        ax.annotate(
            "n={}".format(n),
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=plt.rcParams["font.size"] - 1,
            clip_on=False,
        )

    fig.suptitle("g-{}: Protocol B prescribed motor-unit firing rates".format(group_number))

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate grouped Protocol B firing-frequency summary figures from "
            "existing CSV files without regenerating simulation cases."
        )
    )
    parser.add_argument(
        "--study-root",
        required=True,
        help="Root containing g-1/protocol_B, g-2/protocol_B, ...",
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="*",
        default=None,
        help="Optional groups, e.g. --groups 1 2 3. Default: all discovered groups.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=14.0,
        help="Base font size for all figure text. Default: 14.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG/PDF export DPI. Default: 300.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Optional common output directory. If omitted, each figure is written "
            "to its existing g-N/protocol_B directory."
        ),
    )
    args = parser.parse_args()

    configure_plot_style(args.font_size)

    study_root = Path(args.study_root).resolve()
    if not study_root.exists():
        raise FileNotFoundError("Study root does not exist: {}".format(study_root))

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    groups = discover_groups(study_root, args.groups)

    print("Study root: {}".format(study_root))
    print("Groups: {}".format(", ".join("g-{}".format(g) for g in groups)))

    for g, protocol_b_dir in groups.items():
        print("\n" + "=" * 72)
        print("g-{}".format(g))
        print("=" * 72)

        df, sources = load_existing_summary(protocol_b_dir, g)
        print("  Source data:")
        for p in sources:
            print("    {}".format(p))

        audit_data(df, g)

        if out_dir is None:
            out_base = protocol_b_dir / "protocol_B_frequency_summary_g-{}".format(g)
        else:
            out_base = out_dir / "protocol_B_frequency_summary_g-{}".format(g)

        regenerate_figure(df, g, out_base, args.dpi)
        print("  Saved: {}.png".format(out_base))
        print("  Saved: {}.pdf".format(out_base))

    print("\nDone. Existing simulation cases were not modified.")


if __name__ == "__main__":
    main()
