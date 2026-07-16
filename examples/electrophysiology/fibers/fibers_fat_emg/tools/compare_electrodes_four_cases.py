#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare OpenDiHu electrode EMG traces across four remodeling cases.

This version extends v3 and can also generate animated curves for each electrode.

Supported electrodes.csv formats:
  A. transient positions in each data row
  B. position comment block at top
  C. no positions, only electrode values

Default input layout:
  build_release/out/healthy/electrodes.csv
  build_release/out/death_25/electrodes.csv
  build_release/out/death_50/electrodes.csv
  build_release/out/death_75/electrodes.csv

Outputs:
  emg_overlay_by_electrode/
    electrode_000_y00_x00.png
    electrode_000_y00_x00.gif   (if --animate is used)
    ...
    metrics_by_electrode.csv
    all_electrodes_overlay.pdf  (if --pdf is used)

Examples:
  python compare_electrodes_four_cases_v4.py
  python compare_electrodes_four_cases_v4.py --max-electrodes 12 --animate
  python compare_electrodes_four_cases_v4.py --pdf --animate --anim-step 3
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FuncAnimation, PillowWriter


# Journal-readable default for all plot text: titles, axis labels, tick labels,
# legends, annotations, and any colorbar labels added in the future.
plt.rcParams.update({"font.size": 14})


DEFAULT_CASES = [
    ("healthy", "healthy"),
    ("death_25", "death_25"),
    ("death_50", "death_50"),
    ("death_75", "death_75"),
]


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


def read_electrodes_csv(filename: Path) -> Dict[str, np.ndarray]:
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find {filename}")

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

    if not emg_rows:
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
        f"    parsed {filename}: format={detected_format}, "
        f"shape={emg_array.shape}, grid={n_points_xy}x{n_points_z}"
    )

    return {
        "filename": str(filename),
        "t": t_array,
        "emg": emg_array,
        "positions": electrode_positions,
        "n_points": np.asarray(n_points),
        "n_points_xy": np.asarray(n_points_xy),
        "n_points_z": np.asarray(n_points_z),
        "detected_format": detected_format,
    }


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


def ensure_same_layout(cases: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, int, int, int]:
    names = list(cases.keys())
    ref = cases[names[0]]
    ref_t = ref["t"]
    ref_n = int(ref["n_points"])
    ref_xy = int(ref["n_points_xy"])
    ref_z = int(ref["n_points_z"])

    for name in names[1:]:
        c = cases[name]
        if int(c["n_points"]) != ref_n:
            raise ValueError(f"{name}: n_points differs from reference.")
        if len(c["t"]) != len(ref_t) or not np.allclose(c["t"], ref_t):
            raise ValueError(
                f"{name}: time vector differs from reference. "
                "Resampling is not implemented in this plotting script."
            )

    return ref_t, ref_n, ref_xy, ref_z


def subtract_baseline(y: np.ndarray, n_baseline: int = 10) -> np.ndarray:
    n = min(n_baseline, len(y))
    if n <= 0:
        return y
    return y - np.mean(y[:n])


def compute_metrics(
    cases: Dict[str, Dict[str, np.ndarray]],
    channel: int,
    healthy_name: str = "healthy",
    subtract_mean: bool = False,
) -> Dict[str, float]:
    row = {"electrode": channel}
    ref = cases[healthy_name]["emg"][:, channel]
    if subtract_mean:
        ref = subtract_baseline(ref)

    for name, c in cases.items():
        y = c["emg"][:, channel]
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
    t: np.ndarray,
    cases: Dict[str, Dict[str, np.ndarray]],
    channel: int,
    subtract_mean: bool = False,
) -> Dict[str, np.ndarray]:
    traces = {}
    for name, c in cases.items():
        y = c["emg"][:, channel]
        if subtract_mean:
            y = subtract_baseline(y)
        traces[name] = y
    return traces


def get_ylim_from_traces(traces: Dict[str, np.ndarray]) -> Tuple[float, float]:
    local_min = min(float(np.min(y)) for y in traces.values())
    local_max = max(float(np.max(y)) for y in traces.values())
    pad = 0.08 * (local_max - local_min + 1e-12)
    return (local_min - pad, local_max + pad)


def plot_one_electrode(
    t: np.ndarray,
    cases: Dict[str, Dict[str, np.ndarray]],
    channel: int,
    n_points_xy: int,
    out_file: Path,
    subtract_mean: bool = False,
    global_ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 160,
) -> None:
    grid_y = channel // n_points_xy
    grid_x = channel % n_points_xy

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    traces = get_channel_traces(t, cases, channel, subtract_mean=subtract_mean)

    for name, y in traces.items():
        ax.plot(t, y, linewidth=1.2, label=name)

    if global_ylim is not None:
        ax.set_ylim(*global_ylim)
    else:
        ax.set_ylim(*get_ylim_from_traces(traces))

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("sEMG [mV]")
    ax.set_title(f"Electrode {channel:03d}  |  grid y={grid_y}, x={grid_x}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def animate_one_electrode(
    t: np.ndarray,
    cases: Dict[str, Dict[str, np.ndarray]],
    channel: int,
    n_points_xy: int,
    out_file: Path,
    subtract_mean: bool = False,
    global_ylim: Optional[Tuple[float, float]] = None,
    fps: int = 15,
    anim_step: int = 2,
    dpi: int = 120,
) -> None:
    grid_y = channel // n_points_xy
    grid_x = channel % n_points_xy
    traces = get_channel_traces(t, cases, channel, subtract_mean=subtract_mean)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    lines = {}
    for name in traces.keys():
        (line,) = ax.plot([], [], linewidth=1.5, label=name)
        lines[name] = line

    ax.set_xlim(float(t[0]), float(t[-1]))
    if global_ylim is not None:
        ax.set_ylim(*global_ylim)
    else:
        ax.set_ylim(*get_ylim_from_traces(traces))

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("sEMG [mV]")
    ax.set_title(f"Electrode {channel:03d}  |  grid y={grid_y}, x={grid_x}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    frame_indices = list(range(1, len(t) + 1, max(1, anim_step)))
    if frame_indices[-1] != len(t):
        frame_indices.append(len(t))

    def init():
        for line in lines.values():
            line.set_data([], [])
        return tuple(lines.values())

    def update(i_end):
        for name, y in traces.items():
            lines[name].set_data(t[:i_end], y[:i_end])
        return tuple(lines.values())

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        blit=True,
        interval=1000.0 / max(1, fps),
        repeat=False,
    )

    writer = PillowWriter(fps=fps)
    ani.save(out_file, writer=writer, dpi=dpi)
    plt.close(fig)


def write_metrics_csv(metrics_rows, out_file: Path) -> None:
    if not metrics_rows:
        return
    fieldnames = list(metrics_rows[0].keys())
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay EMG curves for each electrode across four OpenDiHu cases.")
    parser.add_argument("--base-dir", default="build_release/out", help="Folder containing healthy/death_25/death_50/death_75 subfolders.")
    parser.add_argument("--out-dir", default="emg_overlay_by_electrode", help="Output folder for figures.")
    parser.add_argument("--csv-name", default="electrodes.csv")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--max-electrodes", type=int, default=None, help="Only plot first N electrodes for testing.")
    parser.add_argument("--subtract-mean", action="store_true", help="Subtract mean of first 10 samples from every trace.")
    parser.add_argument("--global-y", action="store_true", help="Use one global y-axis range across all electrodes and cases.")
    parser.add_argument("--pdf", action="store_true", help="Also create one multi-page PDF with all electrode overlays.")

    parser.add_argument("--animate", action="store_true", help="Also create animated GIF curves for each electrode.")
    parser.add_argument("--anim-format", default="gif", choices=["gif"], help="Animation format. GIF uses PillowWriter.")
    parser.add_argument("--anim-dpi", type=int, default=120, help="DPI for animated output.")
    parser.add_argument("--anim-fps", type=int, default=15, help="Frames per second for animated output.")
    parser.add_argument("--anim-step", type=int, default=2, help="Use every Nth time sample as an animation frame to reduce file size.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_paths = {
        label: base_dir / folder / args.csv_name
        for label, folder in DEFAULT_CASES
    }

    print("Loading cases:")
    cases = {}
    for label, filename in case_paths.items():
        print(f"  {label}: {filename}")
        cases[label] = read_electrodes_csv(filename)

    t, n_points, n_points_xy, n_points_z = ensure_same_layout(cases)
    print(f"Detected electrode grid: {n_points_xy} x {n_points_z} = {n_points} electrodes")
    print(f"Time range: [{t[0]}, {t[-1]}] ms, n_time_steps={len(t)}")

    n_plot = min(n_points, args.max_electrodes) if args.max_electrodes is not None else n_points

    global_ylim = None
    if args.global_y:
        all_vals = []
        for c in cases.values():
            y = c["emg"][:, :n_plot]
            if args.subtract_mean:
                y = y - np.mean(y[:min(10, y.shape[0]), :], axis=0, keepdims=True)
            all_vals.append(y)
        all_vals = np.concatenate([v.ravel() for v in all_vals])
        ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
        pad = 0.08 * (ymax - ymin + 1e-12)
        global_ylim = (ymin - pad, ymax + pad)

    metrics_rows = []
    pdf_pages = PdfPages(out_dir / "all_electrodes_overlay.pdf") if args.pdf else None

    for ch in range(n_plot):
        static_file = out_dir / f"electrode_{ch:03d}_y{ch // n_points_xy:02d}_x{ch % n_points_xy:02d}.{args.format}"

        plot_one_electrode(
            t=t,
            cases=cases,
            channel=ch,
            n_points_xy=n_points_xy,
            out_file=static_file,
            subtract_mean=args.subtract_mean,
            global_ylim=global_ylim,
            dpi=args.dpi,
        )

        if args.animate:
            anim_file = out_dir / f"electrode_{ch:03d}_y{ch // n_points_xy:02d}_x{ch % n_points_xy:02d}.{args.anim_format}"
            animate_one_electrode(
                t=t,
                cases=cases,
                channel=ch,
                n_points_xy=n_points_xy,
                out_file=anim_file,
                subtract_mean=args.subtract_mean,
                global_ylim=global_ylim,
                fps=args.anim_fps,
                anim_step=args.anim_step,
                dpi=args.anim_dpi,
            )

        if pdf_pages is not None:
            fig, ax = plt.subplots(figsize=(8.0, 4.5))
            traces = get_channel_traces(t, cases, ch, subtract_mean=args.subtract_mean)
            for name, y in traces.items():
                ax.plot(t, y, linewidth=1.2, label=name)
            if global_ylim is not None:
                ax.set_ylim(*global_ylim)
            else:
                ax.set_ylim(*get_ylim_from_traces(traces))
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("sEMG [mV]")
            ax.set_title(f"Electrode {ch:03d}  |  grid y={ch // n_points_xy}, x={ch % n_points_xy}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)

        metrics_rows.append(compute_metrics(cases, ch, subtract_mean=args.subtract_mean))

        if (ch + 1) % 10 == 0 or ch + 1 == n_plot:
            msg = f"  plotted {ch + 1}/{n_plot} electrodes"
            if args.animate:
                msg += " (static + animation)"
            print(msg)

    if pdf_pages is not None:
        pdf_pages.close()
        print(f"Created {out_dir / 'all_electrodes_overlay.pdf'}")

    write_metrics_csv(metrics_rows, out_dir / "metrics_by_electrode.csv")
    print(f"Created electrode overlay figures in: {out_dir}")
    if args.animate:
        print("Created animated GIFs for each electrode as well.")
    print(f"Created metrics CSV: {out_dir / 'metrics_by_electrode.csv'}")


if __name__ == "__main__":
    main()