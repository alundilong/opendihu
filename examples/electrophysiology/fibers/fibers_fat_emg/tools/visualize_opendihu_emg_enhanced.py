#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced OpenDiHu EMG visualizer.

Creates:
  1. static grid plot of all electrode traces
  2. animated grid plot of all electrode traces
  3. animated electrode-map heat plot
  4. legacy VTK export of electrode positions for ParaView
  5. optional user-defined MU-axis limits for the stimulation/raster panel

Supported electrodes.csv formats:
  A. timestamp;t;n_points;p0_x;p0_y;p0_z;...;p0_value;...
  B. #electrode positions header, then timestamp;t;n_points;p0_value;...
  C. timestamp;t;n_points;p0_value;p1_value;... only

If electrode positions are missing, fallback grid positions are generated so the
VTK file can still be opened in ParaView.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter, FFMpegWriter


def parse_float_list(tokens: List[str]) -> List[float]:
    values = []
    for token in tokens:
        token = token.strip().replace("#", "")
        if token == "":
            continue
        try:
            values.append(float(token))
        except ValueError:
            for m in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", token):
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


def infer_electrode_grid(electrode_positions: Optional[np.ndarray], n_points: int, default_xy: Optional[int] = None) -> Tuple[int, int]:
    if electrode_positions is not None and electrode_positions.ndim == 2 and electrode_positions.shape[0] == n_points and electrode_positions.shape[1] >= 3:
        z_positions = electrode_positions[:, 2]
        differences = z_positions[1:] - z_positions[:-1]
        if len(differences) > 0:
            mean_abs = np.mean(np.abs(differences)) + 1e-12
            jumps = np.where(np.abs(differences) > mean_abs * 3.0)[0]
            if len(jumps) > 0:
                n_points_xy = int(jumps[0] + 1)
                if n_points_xy > 0 and n_points % n_points_xy == 0:
                    return n_points_xy, int(n_points // n_points_xy)

    if default_xy is not None and default_xy > 0 and n_points % default_xy == 0:
        return int(default_xy), int(n_points // default_xy)

    if n_points == 384:
        return 12, 32

    factors = [f for f in range(1, n_points + 1) if n_points % f == 0]
    nxy = min(factors, key=lambda f: abs(f - math.sqrt(n_points)))
    return int(nxy), int(n_points // nxy)


def fallback_grid_positions(n_points_xy: int, n_points_z: int) -> np.ndarray:
    positions = []
    for j in range(n_points_z):
        for i in range(n_points_xy):
            positions.append([float(i), 0.0, float(j)])
    return np.asarray(positions, dtype=float)


def read_electrodes_csv(filename: Path, default_xy: Optional[int] = None) -> Dict:
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
                raise ValueError(f"{filename}, line {line_no}: n_points changed from {n_points_ref} to {n_points}")

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
                raise ValueError(f"{filename}, line {line_no}: expected at least {n_points} EMG values, got {len(numeric_tail)} numeric fields.")

            t_values.append(t)
            emg_rows.append(values)

    if not emg_rows:
        raise ValueError(f"No EMG rows found in {filename}")

    t_array = np.asarray(t_values, dtype=float)
    emg_array = np.asarray(emg_rows, dtype=float)
    n_points = int(n_points_ref)

    electrode_positions = None
    positions_are_fallback = False
    if position_data is not None and len(position_data) >= 3 * n_points:
        electrode_positions = np.asarray(position_data[:3 * n_points], dtype=float).reshape(n_points, 3)

    n_points_xy, n_points_z = infer_electrode_grid(electrode_positions, n_points, default_xy=default_xy)
    if electrode_positions is None:
        electrode_positions = fallback_grid_positions(n_points_xy, n_points_z)
        positions_are_fallback = True

    print(f"Parsed {filename}")
    print(f"  format: {detected_format}")
    print(f"  EMG shape: {emg_array.shape}")
    print(f"  electrode grid: {n_points_xy} x {n_points_z} = {n_points}")
    if positions_are_fallback:
        print("  warning: electrode positions were missing; using fallback grid positions for VTK.")

    return {
        "filename": str(filename),
        "t": t_array,
        "emg": emg_array,
        "positions": electrode_positions,
        "positions_are_fallback": positions_are_fallback,
        "n_points": n_points,
        "n_points_xy": n_points_xy,
        "n_points_z": n_points_z,
        "detected_format": detected_format,
    }


def read_stimulation_log(filename: Optional[Path]) -> Dict:
    if filename is None:
        return {"mu_times": {}, "fiber_times": {}, "fiber_mu_nos": {}, "end_time_s": 0.0, "max_mu": 1}
    filename = Path(filename)
    if not filename.exists():
        print(f"Stimulation log not found: {filename}")
        return {"mu_times": {}, "fiber_times": {}, "fiber_mu_nos": {}, "end_time_s": 0.0, "max_mu": 1}

    fiber_times = {}
    fiber_mu_nos = {}
    mu_times = {}
    end_time = 0.0
    try:
        with filename.open("r") as f:
            for line in f:
                if "#" in line or line.strip() == "":
                    continue
                values = line.split(";")
                if len(values) < 3:
                    continue
                mu_no = int(values[0])
                fiber_no = int(values[1])
                times = [float(v) / 1000.0 for v in values[2:] if v.strip() != ""]
                fiber_mu_nos[fiber_no] = mu_no
                fiber_times[fiber_no] = times
                mu_times.setdefault(mu_no, set())
                for item in times:
                    end_time = max(end_time, item)
                    mu_times[mu_no].add(item)
        max_mu = max(mu_times.keys()) if mu_times else 1
        print(f"Stimulation file {filename} contains end time {end_time:.3f} s, {len(mu_times)} motor units, {len(fiber_times)} fibers.")
    except Exception:
        print(f"Failed to parse stimulation log: {filename}")
        traceback.print_exc()
        max_mu = 1
    return {"mu_times": mu_times, "fiber_times": fiber_times, "fiber_mu_nos": fiber_mu_nos, "end_time_s": end_time, "max_mu": max_mu}


def write_electrode_positions_vtk(filename: Path, positions: np.ndarray, n_points_xy: int, n_points_z: int, emg: np.ndarray, positions_are_fallback: bool = False) -> None:
    filename = Path(filename)
    n_points = positions.shape[0]
    rms = np.sqrt(np.mean(emg ** 2, axis=0))
    peak_to_peak = np.max(emg, axis=0) - np.min(emg, axis=0)
    max_abs = np.max(np.abs(emg), axis=0)
    fallback_flag = 1 if positions_are_fallback else 0

    with filename.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("OpenDiHu HD-sEMG electrode positions\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_points} float\n")
        for x, y, z in positions:
            f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
        f.write(f"\nVERTICES {n_points} {2 * n_points}\n")
        for i in range(n_points):
            f.write(f"1 {i}\n")
        f.write(f"\nPOINT_DATA {n_points}\n")

        def write_scalar(name: str, dtype: str, values):
            f.write(f"SCALARS {name} {dtype} 1\n")
            f.write("LOOKUP_TABLE default\n")
            for v in values:
                f.write(f"{v}\n")

        write_scalar("electrode_id", "int", range(n_points))
        write_scalar("grid_x", "int", [i % n_points_xy for i in range(n_points)])
        write_scalar("grid_y", "int", [i // n_points_xy for i in range(n_points)])
        write_scalar("fallback_position", "int", [fallback_flag for _ in range(n_points)])
        write_scalar("rms_mV", "float", [f"{v:.9g}" for v in rms])
        write_scalar("peak_to_peak_mV", "float", [f"{v:.9g}" for v in peak_to_peak])
        write_scalar("max_abs_mV", "float", [f"{v:.9g}" for v in max_abs])
    print(f'Created VTK file "{filename}".')


def plot_electrode_positions_preview(filename: Path, positions: np.ndarray, positions_are_fallback: bool) -> None:
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ids = np.arange(positions.shape[0])
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=ids, s=18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    title = "Electrode positions"
    if positions_are_fallback:
        title += " (fallback grid)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    print(f'Created electrode preview "{filename}".')


def compute_frame_indices(n_time_steps: int, max_frames: int) -> np.ndarray:
    max_frames = max(2, int(max_frames))
    if n_time_steps <= max_frames:
        return np.arange(1, n_time_steps + 1)
    step = int(math.ceil(n_time_steps / max_frames))
    indices = np.arange(1, n_time_steps + 1, step)
    if indices[-1] != n_time_steps:
        indices = np.append(indices, n_time_steps)
    return indices


def _resolve_ffmpeg_executable() -> str:
    """
    Resolve the FFmpeg executable used by Matplotlib.

    This respects matplotlib.rcParams["animation.ffmpeg_path"], allowing users
    to point the script to a Conda-provided FFmpeg executable when the cluster
    system FFmpeg is unsuitable.
    """
    configured = str(matplotlib.rcParams.get("animation.ffmpeg_path", "ffmpeg"))
    resolved = shutil.which(configured)

    if resolved is not None:
        return resolved

    configured_path = Path(configured).expanduser()
    if configured_path.is_file():
        return str(configured_path)

    raise RuntimeError(
        "FFmpeg was not found. Load an FFmpeg module, activate a Conda "
        "environment containing FFmpeg, or set --ffmpeg-path."
    )


def _get_ffmpeg_video_encoders(ffmpeg_executable: str) -> set[str]:
    """Return the video encoders reported by the selected FFmpeg build."""
    result = subprocess.run(
        [ffmpeg_executable, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f'Failed to query FFmpeg encoders using "{ffmpeg_executable}".\n'
            f"{result.stdout}"
        )

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        # Typical encoder entry:
        # V....D libx264  libx264 H.264 / AVC / MPEG-4 AVC ...
        if len(fields) >= 2 and len(fields[0]) == 6 and fields[0].startswith("V"):
            encoders.add(fields[1])

    return encoders


def _select_software_video_encoder(
    ffmpeg_executable: str,
    requested_codec: str = "auto",
) -> Tuple[str, List[str]]:
    """
    Select a software encoder.

    Generic codec name "h264" is deliberately not used. Some HPC FFmpeg builds
    resolve it to h264_v4l2m2m, which requires a V4L2 hardware device and fails
    on ordinary compute nodes.
    """
    encoders = _get_ffmpeg_video_encoders(ffmpeg_executable)

    if requested_codec != "auto":
        if requested_codec not in encoders:
            raise RuntimeError(
                f'FFmpeg encoder "{requested_codec}" is unavailable in '
                f'"{ffmpeg_executable}". Available H.264/MPEG-4 encoders include: '
                + ", ".join(
                    sorted(
                        name for name in encoders
                        if "264" in name.lower() or "mpeg4" in name.lower()
                    )
                )
            )

        if requested_codec == "libx264":
            return requested_codec, [
                "-preset", "medium",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
            ]

        if requested_codec == "mpeg4":
            return requested_codec, [
                "-q:v", "3",
                "-pix_fmt", "yuv420p",
            ]

        return requested_codec, ["-pix_fmt", "yuv420p"]

    if "libx264" in encoders:
        return "libx264", [
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
        ]

    # The native FFmpeg MPEG-4 encoder is broadly available and does not
    # require a hardware device. It is a practical HPC fallback when libx264
    # was not compiled into the cluster FFmpeg build.
    if "mpeg4" in encoders:
        return "mpeg4", [
            "-q:v", "3",
            "-pix_fmt", "yuv420p",
        ]

    related = sorted(
        name for name in encoders
        if "264" in name.lower() or "mpeg4" in name.lower()
    )
    raise RuntimeError(
        "No supported software MP4 encoder was found. "
        f'FFmpeg executable: "{ffmpeg_executable}". '
        f"Related encoders: {', '.join(related) if related else 'none'}. "
        "Use a Conda/cluster FFmpeg build containing libx264 or mpeg4, "
        "or request GIF explicitly with --gif."
    )


def save_animation_with_fallback(
    anim,
    out_base: Path,
    fps: int,
    dpi: int,
    prefer_mp4: bool = True,
    ffmpeg_codec: str = "auto",
) -> Path:
    """
    Save an animation.

    MP4 mode explicitly selects a software encoder instead of relying on
    FFmpeg's platform-dependent interpretation of the generic "h264" codec.
    GIF is generated only when the user explicitly passes --gif; an MP4
    encoder failure no longer starts a potentially very slow full GIF render.
    """
    out_base = Path(out_base)

    if prefer_mp4:
        mp4_file = out_base.with_suffix(".mp4")
        ffmpeg_executable = _resolve_ffmpeg_executable()
        codec, extra_args = _select_software_video_encoder(
            ffmpeg_executable,
            requested_codec=ffmpeg_codec,
        )

        print(f'Using FFmpeg: "{ffmpeg_executable}"')
        print(f"Using software video encoder: {codec}")

        writer = FFMpegWriter(
            fps=fps,
            codec=codec,
            extra_args=extra_args,
        )

        try:
            anim.save(str(mp4_file), writer=writer, dpi=dpi)
        except Exception as exc:
            try:
                mp4_file.unlink(missing_ok=True)
            except OSError:
                pass

            raise RuntimeError(
                f'Failed to save MP4 using FFmpeg encoder "{codec}". '
                "The script did not automatically start GIF generation because "
                "large EMG animations can make GIF rendering extremely slow. "
                "Run again with --gif only when GIF output is actually desired."
            ) from exc

        print(f'Created "{mp4_file}".')
        return mp4_file

    gif_file = out_base.with_suffix(".gif")
    writer = PillowWriter(fps=fps)
    anim.save(str(gif_file), writer=writer, dpi=dpi)
    print(f'Created "{gif_file}".')
    return gif_file


def plot_static_emg_grid(out_file: Path, t: np.ndarray, emg: np.ndarray, n_points_xy: int, n_points_z: int, y_limits: Tuple[float, float]) -> None:
    fig = plt.figure(figsize=(7, 12))
    ax_global = fig.add_subplot(111)
    ax_global.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    ax_global.grid(False)
    ax_global.set_xlabel("cross-fiber electrode direction")
    ax_global.set_ylabel("fiber-direction electrode direction")
    ax_global.set_title(f"sEMG for {n_points_xy} x {n_points_z} electrodes, t: [{t[0]}, {t[-1]}] ms")
    for j in range(n_points_z):
        for i in range(n_points_xy):
            ch = j * n_points_xy + i
            ax = fig.add_subplot(n_points_z, n_points_xy, ch + 1)
            ax.plot(t, emg[:, ch], linewidth=0.65)
            ax.set_ylim(*y_limits)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.axis("off")
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f'Created static EMG grid "{out_file}".')


def animate_emg_grid(out_base: Path, t: np.ndarray, emg: np.ndarray, n_points_xy: int, n_points_z: int, y_limits: Tuple[float, float], fps: int, dpi: int, max_frames: int, moving_window: bool = False, window_ms: float = 40.0, prefer_mp4: bool = True, ffmpeg_codec: str = "auto") -> Path:
    frame_indices = compute_frame_indices(emg.shape[0], max_frames)
    fig = plt.figure(figsize=(7, 12))
    axes = []
    lines = []
    for j in range(n_points_z):
        for i in range(n_points_xy):
            ch = j * n_points_xy + i
            ax = fig.add_subplot(n_points_z, n_points_xy, ch + 1)
            line, = ax.plot([], [], linewidth=0.65)
            ax.set_ylim(*y_limits)
            if moving_window:
                ax.set_xlim(t[0], min(t[-1], t[0] + window_ms))
            else:
                ax.set_xlim(t[0], t[-1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            axes.append(ax)
            lines.append(line)
    fig.subplots_adjust(hspace=0, wspace=0)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(i_end):
        if moving_window:
            current_time = t[i_end - 1]
            start_time = max(t[0], current_time - window_ms)
            start_idx = np.searchsorted(t, start_time)
            x_data = t[start_idx:i_end]
            for ch, line in enumerate(lines):
                line.set_data(x_data, emg[start_idx:i_end, ch])
                axes[ch].set_xlim(start_time, max(current_time, start_time + 1e-9))
        else:
            x_data = t[:i_end]
            for ch, line in enumerate(lines):
                line.set_data(x_data, emg[:i_end, ch])
        return lines

    anim = animation.FuncAnimation(fig, update, frames=frame_indices, init_func=init, interval=1000.0 / max(1, fps), blit=True, repeat=False)
    result = save_animation_with_fallback(anim, out_base, fps=fps, dpi=dpi, prefer_mp4=prefer_mp4, ffmpeg_codec=ffmpeg_codec)
    plt.close(fig)
    return result


def animate_emg_map(out_base: Path, t: np.ndarray, emg: np.ndarray, n_points_xy: int, n_points_z: int, y_limits: Tuple[float, float], stim: Dict, fps: int, dpi: int, max_frames: int, prefer_mp4: bool = True, stim_ylim: Optional[Tuple[float, float]] = None, ffmpeg_codec: str = "auto") -> Path:
    from matplotlib import gridspec
    frame_indices = compute_frame_indices(emg.shape[0], max_frames)
    has_stim = bool(stim.get("mu_times", {}))
    if has_stim:
        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.0])
        ax_map = fig.add_subplot(gs[0])
        ax_stim = fig.add_subplot(gs[1])
    else:
        fig, ax_map = plt.subplots(figsize=(7.5, 7))
        ax_stim = None

    data0 = emg[0, :].reshape(n_points_z, n_points_xy)
    im = ax_map.imshow(data0, origin="upper", aspect="auto", cmap="RdBu_r", vmin=y_limits[0], vmax=y_limits[1])
    cbar = fig.colorbar(im, ax=ax_map)
    cbar.set_label("sEMG [mV]")
    ax_map.set_xlabel("cross-fiber electrode index")
    ax_map.set_ylabel("fiber-direction electrode index")
    title = ax_map.set_title(f"sEMG electrode map, t = {t[0]:.3f} ms")

    time_line = None
    text_handle = None
    if has_stim and ax_stim is not None:
        mu_times = stim["mu_times"]
        fiber_times = stim["fiber_times"]
        fiber_mu_nos = stim["fiber_mu_nos"]
        end_time_s = max(stim["end_time_s"], t[-1] / 1000.0)
        max_mu = max(stim["max_mu"], 1)
        if stim_ylim is None:
            stim_ymin, stim_ymax = 0.0, float(max_mu)
        else:
            stim_ymin, stim_ymax = float(stim_ylim[0]), float(stim_ylim[1])
            if stim_ymin >= stim_ymax:
                raise ValueError(f"--stim-ylim requires MIN < MAX, got {stim_ylim}")

        for mu_no, times in mu_times.items():
            ax_stim.plot([0, end_time_s], [mu_no, mu_no], color=(0.82, 0.82, 0.82), linewidth=0.6)
        for fiber_no, times in fiber_times.items():
            mu_no = fiber_mu_nos[fiber_no]
            if len(times):
                ax_stim.plot(list(times), [mu_no for _ in times], "+", markersize=6)
        time_line, = ax_stim.plot([t[0] / 1000.0, t[0] / 1000.0], [stim_ymin, stim_ymax], color="k", linewidth=1.2)
        ax_stim.set_xlim(t[0] / 1000.0, t[-1] / 1000.0)
        ax_stim.set_ylim(stim_ymin, stim_ymax)
        ax_stim.set_ylabel("MU no.")
        ax_stim.set_xlabel("time [s]")
        text_handle = ax_stim.text(0.02, 1.05, "", transform=ax_stim.transAxes, family="monospace")

    def update(i_end):
        idx = i_end - 1
        im.set_data(emg[idx, :].reshape(n_points_z, n_points_xy))
        title.set_text(f"sEMG electrode map, t = {t[idx]:.3f} ms")
        if time_line is not None:
            tt = t[idx] / 1000.0
            time_line.set_data([tt, tt], ax_stim.get_ylim())
            if text_handle is not None:
                text_handle.set_text(f"{tt:.4f} s")
            return (im, time_line)
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000.0 / max(1, fps), blit=False, repeat=False)
    fig.tight_layout()
    result = save_animation_with_fallback(anim, out_base, fps=fps, dpi=dpi, prefer_mp4=prefer_mp4, ffmpeg_codec=ffmpeg_codec)
    plt.close(fig)
    return result


def auto_find_electrodes_csv() -> Optional[Path]:
    candidates = [
        Path("build_release/out/biceps/electrodes.csv"),
        Path("build_release/out/electrodes.csv"),
        Path("build_release/electrodes.csv"),
        Path("electrodes.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced OpenDiHu EMG visualizer with grid animation and VTK electrode export.")
    parser.add_argument("electrodes_csv", nargs="?", default=None, help="Path to electrodes.csv")
    parser.add_argument("--stimulation-log", default=None, help="Optional path to stimulation.log")
    parser.add_argument("--stim-ylim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                        help="Y-axis limits for the stimulation/MU raster subplot, e.g. --stim-ylim 0 21")
    parser.add_argument("--out-dir", default="emg_visualization", help="Output directory")
    parser.add_argument("--n-points-xy", type=int, default=None, help="Electrodes in cross-fiber direction. For 384 electrodes, default fallback is 12.")
    parser.add_argument("--fps", type=int, default=20, help="Animation FPS")
    parser.add_argument("--dpi", type=int, default=150, help="Animation DPI")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum animation frames")
    parser.add_argument("--gif", action="store_true",
                        help="Save GIF instead of MP4. GIF is not used automatically after an MP4 failure.")
    parser.add_argument("--ffmpeg-path", default=None,
                        help="Optional FFmpeg executable path, e.g. $CONDA_PREFIX/bin/ffmpeg")
    parser.add_argument("--ffmpeg-codec", choices=("auto", "libx264", "mpeg4"), default="auto",
                        help="Software MP4 encoder. 'auto' prefers libx264 and falls back to mpeg4.")
    parser.add_argument("--moving-window", action="store_true", help="Use moving time window for EMG grid trace animation")
    parser.add_argument("--window-ms", type=float, default=40.0, help="Moving-window width in ms")
    parser.add_argument("--skip-static", action="store_true", help="Skip static EMG grid plot")
    parser.add_argument("--skip-grid-animation", action="store_true", help="Skip animated EMG small-multiple trace grid")
    parser.add_argument("--skip-map-animation", action="store_true", help="Skip animated electrode-map heat plot")
    parser.add_argument("--skip-animations", action="store_true", help="Skip both animations")
    args = parser.parse_args()

    if args.ffmpeg_path:
        matplotlib.rcParams["animation.ffmpeg_path"] = str(
            Path(args.ffmpeg_path).expanduser()
        )

    if args.electrodes_csv is None:
        auto = auto_find_electrodes_csv()
        if auto is None:
            raise FileNotFoundError("No electrodes.csv path provided and none found in default locations.")
        emg_filename = auto
    else:
        emg_filename = Path(args.electrodes_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_electrodes_csv(emg_filename, default_xy=args.n_points_xy)
    t = data["t"]
    emg = data["emg"]
    positions = data["positions"]
    n_points_xy = int(data["n_points_xy"])
    n_points_z = int(data["n_points_z"])
    n_points = int(data["n_points"])

    if emg.shape[1] != n_points_xy * n_points_z:
        raise ValueError(f"EMG has {emg.shape[1]} channels but grid is {n_points_xy} x {n_points_z} = {n_points_xy*n_points_z}.")

    y_limits = (float(np.min(emg)), float(np.max(emg)))
    sampling_frequency = 1000.0 * len(t) / max(t[-1] - t[0], 1e-12)
    print(f"EMG value range [mV]: [{y_limits[0]}, {y_limits[1]}]")
    print(f"time range [ms]: [{t[0]}, {t[-1]}]")
    print(f"sampling frequency [Hz]: {sampling_frequency:.0f}")

    write_electrode_positions_vtk(out_dir / "electrode_positions.vtk", positions=positions, n_points_xy=n_points_xy, n_points_z=n_points_z, emg=emg, positions_are_fallback=bool(data["positions_are_fallback"]))
    plot_electrode_positions_preview(out_dir / "electrode_positions_preview.png", positions=positions, positions_are_fallback=bool(data["positions_are_fallback"]))

    if not args.skip_static:
        plot_static_emg_grid(out_dir / "emg_grid_plot.pdf", t=t, emg=emg, n_points_xy=n_points_xy, n_points_z=n_points_z, y_limits=y_limits)
        plot_static_emg_grid(out_dir / "emg_grid_plot.png", t=t, emg=emg, n_points_xy=n_points_xy, n_points_z=n_points_z, y_limits=y_limits)

    prefer_mp4 = not args.gif
    if not args.skip_animations:
        if not args.skip_grid_animation:
            animate_emg_grid(out_dir / "emg_grid_animation", t=t, emg=emg, n_points_xy=n_points_xy, n_points_z=n_points_z, y_limits=y_limits, fps=args.fps, dpi=args.dpi, max_frames=args.max_frames, moving_window=args.moving_window, window_ms=args.window_ms, prefer_mp4=prefer_mp4, ffmpeg_codec=args.ffmpeg_codec)
        if not args.skip_map_animation:
            stim = read_stimulation_log(Path(args.stimulation_log) if args.stimulation_log else None)
            animate_emg_map(out_dir / "emg_map_animation", t=t, emg=emg, n_points_xy=n_points_xy, n_points_z=n_points_z, y_limits=y_limits, stim=stim, fps=args.fps, dpi=args.dpi, max_frames=args.max_frames, prefer_mp4=prefer_mp4, stim_ylim=tuple(args.stim_ylim) if args.stim_ylim is not None else None, ffmpeg_codec=args.ffmpeg_codec)

    metadata = {
        "electrodes_csv": str(emg_filename),
        "stimulation_log": str(args.stimulation_log) if args.stimulation_log else None,
        "n_points": n_points,
        "n_points_xy": n_points_xy,
        "n_points_z": n_points_z,
        "time_min_ms": float(t[0]),
        "time_max_ms": float(t[-1]),
        "n_time_steps": int(len(t)),
        "sampling_frequency_hz": float(sampling_frequency),
        "emg_min_mV": y_limits[0],
        "emg_max_mV": y_limits[1],
        "positions_are_fallback": bool(data["positions_are_fallback"]),
        "detected_format": data["detected_format"],
        "stim_ylim": list(args.stim_ylim) if args.stim_ylim is not None else None,
    }
    with (out_dir / "visualization_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("")
    print(f"Created enhanced EMG visualization outputs in: {out_dir}")
    print(f"  {out_dir / 'electrode_positions.vtk'}")
    print(f"  {out_dir / 'electrode_positions_preview.png'}")
    if not args.skip_static:
        print(f"  {out_dir / 'emg_grid_plot.pdf'}")
        print(f"  {out_dir / 'emg_grid_plot.png'}")
    if not args.skip_animations:
        if not args.skip_grid_animation:
            print(f"  {out_dir / 'emg_grid_animation.mp4'} or .gif")
        if not args.skip_map_animation:
            print(f"  {out_dir / 'emg_map_animation.mp4'} or .gif")


if __name__ == "__main__":
    main()
