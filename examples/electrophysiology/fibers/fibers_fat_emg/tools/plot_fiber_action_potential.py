#!/usr/bin/env python3
"""
Plot the spatial distribution of membrane potential along a selected fiber
at multiple timesteps, stacked vertically in time order.

This is intended to create a figure where:
  - x-axis = position along the fiber
  - each curve = Vm distribution along the fiber at one timestep
  - curves are stacked vertically
  - vertical direction corresponds to time progression

Input:
  - a directory containing fibers_0000000.vtp, fibers_0000001.vtp, ...
  - or a fibers.vtp.series file
  - or a single VTP file

Outputs for every selected fiber:
  1. fiber_XXXX_stacked_profiles.png/pdf
       Spatial Vm distributions at selected timesteps, stacked vertically.
  2. fiber_XXXX_profiles_data.npz
       Numerical arrays (time, distance, Vm).
  3. fiber_XXXX_profiles_selected_times.csv
       Time values used in the stacked figure.

Example:
  python plot_fiber_spatial_profiles_stacked.py \
      build_release/out/healthy \
      --fiber-id 125 \
      --max-profiles 12 \
      --time-unit s \
      --position-axis normalized \
      --vm-range -90 40

Example with user-selected timestep indices:
  python plot_fiber_spatial_profiles_stacked.py \
      build_release/out/healthy \
      --fiber-id 125 \
      --time-indices 0 2 4 6 8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk


FILE_PATTERN = re.compile(r"^fibers_(\d+)\.vtp$")


def read_vtp(filename: Path) -> pv.PolyData:
    """Read geometry/topology and only the point array named 'solution'."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(filename))
    selection = reader.GetPointDataArraySelection()
    selection.DisableAllArrays()
    selection.EnableArray("solution")
    reader.Update()

    mesh = pv.wrap(reader.GetOutput())
    if not isinstance(mesh, pv.PolyData):
        raise TypeError(f'"{filename}" is not VTK PolyData.')
    return mesh


def discover_files(input_path: Path) -> list[tuple[Path, float | None]]:
    """Return ordered (filename, optional_series_time) pairs."""
    input_path = input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_file() and input_path.name.endswith(".vtp.series"):
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        entries = []
        for item in data.get("files", []):
            filename = (input_path.parent / item["name"]).resolve()
            if not filename.is_file():
                raise FileNotFoundError(filename)
            entries.append((filename, float(item["time"]) if "time" in item else None))

        if not entries:
            raise ValueError(f'No files listed in "{input_path}".')
        return entries

    if input_path.is_dir():
        files = []
        for filename in input_path.iterdir():
            match = FILE_PATTERN.match(filename.name)
            if match:
                files.append((int(match.group(1)), filename.resolve()))

        files.sort(key=lambda item: item[0])

        if not files:
            raise FileNotFoundError(
                f'No files matching fibers_#######.vtp in "{input_path}".'
            )

        return [(filename, None) for _, filename in files]

    if input_path.is_file() and input_path.suffix.lower() == ".vtp":
        return [(input_path, None)]

    raise ValueError("Input must be a VTP file, VTP series, or directory.")


def get_time(mesh: pv.PolyData, series_time: float | None, index: int) -> float:
    if series_time is not None:
        return series_time

    if "Time" in mesh.field_data:
        values = np.asarray(mesh.field_data["Time"]).reshape(-1)
        if values.size:
            return float(values[0])

    return float(index)


def locate_fiber(mesh: pv.PolyData, fiber_id: int) -> tuple[int, int | None]:
    """Return VTK cell index and optional MU ID."""
    if mesh.n_cells != mesh.n_lines:
        raise ValueError(
            f"Expected only line cells; found {mesh.n_cells} cells and "
            f"{mesh.n_lines} lines."
        )

    if "fiber_id" in mesh.cell_data:
        ids = np.asarray(mesh.cell_data["fiber_id"]).reshape(-1).astype(np.int64)
        matches = np.flatnonzero(ids == fiber_id)
        if len(matches) != 1:
            raise ValueError(
                f"fiber_id {fiber_id} has {len(matches)} matches. "
                f"Available range: {ids.min()}-{ids.max()}."
            )
        cell_id = int(matches[0])
    else:
        if not 0 <= fiber_id < mesh.n_lines:
            raise ValueError(f"Fiber index must be 0-{mesh.n_lines - 1}.")
        cell_id = fiber_id

    mu_id = None
    if "mu_id" in mesh.cell_data:
        mu_id = int(np.asarray(mesh.cell_data["mu_id"]).reshape(-1)[cell_id])

    return cell_id, mu_id


def load_fiber_series(
    entries: list[tuple[Path, float | None]],
    fiber_id: int,
) -> dict:
    """Extract Vm(time, position) for one fiber."""
    first_mesh = read_vtp(entries[0][0])

    if "solution" not in first_mesh.point_data:
        raise KeyError(
            f'"solution" is missing. Arrays: {list(first_mesh.point_data.keys())}'
        )

    cell_id, mu_id = locate_fiber(first_mesh, fiber_id)
    point_ids = np.asarray(first_mesh.get_cell(cell_id).point_ids, dtype=np.int64)

    if len(point_ids) < 2:
        raise ValueError(f"Fiber {fiber_id} has fewer than two points.")

    coordinates = np.asarray(first_mesh.points[point_ids], dtype=float)
    arc_length = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1)),
    ]

    if arc_length[-1] <= 0:
        raise ValueError(f"Fiber {fiber_id} has zero arc length.")

    normalized_position = arc_length / arc_length[-1]
    times = np.empty(len(entries), dtype=float)
    vm = np.empty((len(entries), len(point_ids)), dtype=np.float32)

    n_points = first_mesh.n_points
    n_lines = first_mesh.n_lines

    for i, (filename, series_time) in enumerate(entries):
        mesh = first_mesh if i == 0 else read_vtp(filename)

        if mesh.n_points != n_points or mesh.n_lines != n_lines:
            raise ValueError(f"Topology changed in {filename.name}.")

        solution = np.asarray(mesh.point_data["solution"]).reshape(-1)
        vm[i] = solution[point_ids]
        times[i] = get_time(mesh, series_time, i)

        print(
            f"fiber {fiber_id}: [{i + 1}/{len(entries)}] "
            f"{filename.name}, time={times[i]:.8g}"
        )

        if i > 0:
            del mesh

    order = np.argsort(times)

    return {
        "fiber_id": fiber_id,
        "mu_id": mu_id,
        "cell_id": cell_id,
        "point_ids": point_ids,
        "coordinates": coordinates,
        "arc_length": arc_length,
        "normalized_position": normalized_position,
        "time": times[order],
        "vm": vm[order],
    }


def scaled_time(time: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    if unit == "ms":
        return time, "Time [ms]"
    return time * 1e-3, "Time [s]"


def choose_profile_indices(
    n_times: int,
    time_indices: list[int] | None,
    max_profiles: int,
) -> np.ndarray:
    """Choose which timesteps to plot."""
    if time_indices is not None and len(time_indices) > 0:
        indices = np.asarray(time_indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= n_times):
            raise ValueError(
                f"--time-indices must be between 0 and {n_times - 1}."
            )
        # preserve user order, remove duplicates
        return np.asarray(list(dict.fromkeys(indices.tolist())), dtype=int)

    n_profiles = min(max_profiles, n_times)
    if n_profiles <= 1:
        return np.array([0], dtype=int)

    return np.linspace(0, n_times - 1, n_profiles, dtype=int)


def save_outputs(
    data: dict,
    profile_indices: np.ndarray,
    output_stem: Path,
) -> None:
    np.savez_compressed(
        str(output_stem) + "_profiles_data.npz",
        fiber_id=data["fiber_id"],
        mu_id=-1 if data["mu_id"] is None else data["mu_id"],
        cell_id=data["cell_id"],
        point_ids=data["point_ids"],
        coordinates=data["coordinates"],
        arc_length=data["arc_length"],
        normalized_position=data["normalized_position"],
        time=data["time"],
        vm=data["vm"],
        selected_profile_indices=profile_indices,
    )

    with open(str(output_stem) + "_profiles_selected_times.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["profile_no", "time_index", "time_raw"])
        for profile_no, time_index in enumerate(profile_indices):
            writer.writerow([profile_no, int(time_index), float(data["time"][time_index])])


def plot_stacked_profiles(
    data: dict,
    profile_indices: np.ndarray,
    output_stem: Path,
    time_unit: str,
    position_axis: str,
    subtract_resting: bool,
    resting_value: float | None,
    vertical_spacing: float | None,
    vm_range: list[float] | None,
    linewidth: float,
    label_every: int,
    dpi: int,
) -> None:
    """
    Plot Vm(x) at multiple times, stacked vertically.

    x-axis:
      fiber position
    y-axis:
      stacked profile number, labeled by time
    """
    raw_time = np.asarray(data["time"])
    time_plot, time_label = scaled_time(raw_time, time_unit)

    vm = np.asarray(data["vm"])
    selected_vm = vm[profile_indices, :]

    if position_axis == "normalized":
        x = np.asarray(data["normalized_position"])
        x_label = "Normalized position along fiber"
    else:
        x = np.asarray(data["arc_length"])
        x_label = "Distance along fiber"

    if subtract_resting:
        if resting_value is None:
            rest = float(np.percentile(selected_vm, 5.0))
        else:
            rest = float(resting_value)
        plot_vm = selected_vm - rest
        profile_label = "Vm - resting [mV]"
    else:
        rest = None
        plot_vm = selected_vm.copy()
        profile_label = "Vm [mV]"

    if vertical_spacing is None:
        amplitudes = np.ptp(plot_vm, axis=1)
        robust_amp = float(np.percentile(amplitudes, 75.0)) if amplitudes.size > 0 else 20.0
        spacing = max(robust_amp * 1.2, 20.0)
    else:
        spacing = float(vertical_spacing)

    n_profiles = len(profile_indices)
    offsets = np.arange(n_profiles - 1, -1, -1, dtype=float) * spacing

    fig_height = max(4.0, 0.55 * n_profiles + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    # Optional color progression by profile number using default colormap.
    color_values = np.linspace(0.15, 0.9, n_profiles)

    for i, time_index in enumerate(profile_indices):
        y = plot_vm[i, :] + offsets[i]
        ax.plot(x, y, linewidth=linewidth)

        if label_every > 0 and (i % label_every == 0 or i == n_profiles - 1):
            ax.text(
                x[-1] + 0.01 * (x[-1] - x[0] if x[-1] > x[0] else 1.0),
                offsets[i],
                f"{time_plot[time_index]:.4g}",
                va="center",
                ha="left",
                fontsize=8,
            )

    ax.set_yticks(offsets)
    ax.set_yticklabels([f"{time_plot[idx]:.4g}" for idx in profile_indices])
    ax.set_xlabel(x_label)
    ax.set_ylabel(time_label)

    title = f"Spatial action-potential profiles: fiber {data['fiber_id']}"
    if data["mu_id"] is not None:
        title += f", MU {data['mu_id']}"
    ax.set_title(title)

    if subtract_resting:
        ax.text(
            1.01,
            0.02,
            f"Resting value removed: {rest:.2f} mV",
            transform=ax.transAxes,
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
        )

    ax.grid(True, axis="x", alpha=0.25)

    # Add a small annotation clarifying the vertical stacking.
    ax.text(
        0.01,
        0.98,
        f"Profiles stacked vertically\nQuantity in each row: {profile_label}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    fig.savefig(str(output_stem) + "_stacked_profiles.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(output_stem) + "_stacked_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Vm distributions along a selected fiber at multiple timesteps, stacked vertically."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--fiber-id", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("fiber_spatial_profiles"))
    parser.add_argument("--time-unit", choices=["ms", "s"], default="ms")
    parser.add_argument("--position-axis", choices=["normalized", "distance"], default="normalized")
    parser.add_argument("--time-indices", type=int, nargs="+", default=None,
                        help="Explicit timestep indices to include, e.g. 0 2 4 6 8.")
    parser.add_argument("--max-profiles", type=int, default=12,
                        help="Maximum number of timesteps to display if --time-indices is not given.")
    parser.add_argument("--subtract-resting", action="store_true",
                        help="Subtract resting Vm before stacking (recommended).")
    parser.add_argument("--resting-value", type=float, default=None,
                        help="User-specified resting value to subtract, e.g. -75.")
    parser.add_argument("--vertical-spacing", type=float, default=None,
                        help="Vertical spacing between stacked profiles.")
    parser.add_argument("--linewidth", type=float, default=1.2)
    parser.add_argument("--label-every", type=int, default=1,
                        help="Label every Nth stacked profile on the right side.")
    parser.add_argument("--vm-range", type=float, nargs=2, default=None,
                        help="Reserved for future use; not needed for line-profile plot.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    entries = discover_files(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(entries)} timestep file(s).")

    for fiber_id in args.fiber_id:
        data = load_fiber_series(entries, fiber_id)
        profile_indices = choose_profile_indices(
            n_times=len(data["time"]),
            time_indices=args.time_indices,
            max_profiles=args.max_profiles,
        )

        output_stem = args.output_dir / f"fiber_{fiber_id:04d}"

        plot_stacked_profiles(
            data=data,
            profile_indices=profile_indices,
            output_stem=output_stem,
            time_unit=args.time_unit,
            position_axis=args.position_axis,
            subtract_resting=args.subtract_resting,
            resting_value=args.resting_value,
            vertical_spacing=args.vertical_spacing,
            vm_range=args.vm_range,
            linewidth=args.linewidth,
            label_every=args.label_every,
            dpi=args.dpi,
        )

        save_outputs(
            data=data,
            profile_indices=profile_indices,
            output_stem=output_stem,
        )

        print("")
        print(f"fiber_id: {fiber_id}")
        print(f"MU_id: {data['mu_id']}")
        print(f"points: {len(data['point_ids'])}")
        print(f"timesteps available: {len(data['time'])}")
        print(f"timesteps plotted: {len(profile_indices)}")
        print(f'Outputs: "{args.output_dir.resolve()}"')
        print("")


if __name__ == "__main__":
    main()
