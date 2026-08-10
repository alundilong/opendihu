#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create ParaView-loadable MU-assigned fiber VTP files for every generated
OpenDiHu denervation-study case.

This script combines the two operations that were previously performed by
separate scripts:

  1. Read an OpenDiHu fiber *.bin file and convert it once to a PolyData VTP
     containing one polyline cell per valid fiber, with the original fiber_id.
  2. Discover every case below a grouped study directory, read that case's
     MU_fibre_distribution_37x37_20.txt, assign MU ownership to the fibers,
     enrich the fibers with MU/case metadata, and write one VTP per case.

Expected grouped study layout
-----------------------------
<study-root>/
  g-1/
    protocol_A/
      healthy/
      death_25/
      death_50/
      death_75/
    protocol_B/
      healthy/
      death_25/
      death_50/
      death_75/
  g-2/
    ...

The script discovers cases by the canonical file name
"MU_fibre_distribution_37x37_20.txt", so group-tagged alias files do not
create duplicate output.

Example
-------
python create_mu_assigned_vtp_all_cases.py \
    --fiber-bin /path/to/left_biceps_brachii_37x37fibers.bin \
    --study-root /path/to/repeated_study_cases \
    --output-dir /path/to/paraview_mu_fibers

Optional selection example
--------------------------
python create_mu_assigned_vtp_all_cases.py \
    --fiber-bin fibers.bin \
    --study-root repeated_study_cases \
    --output-dir paraview_mu_fibers \
    --groups 1 2 3 \
    --protocols A B \
    --stages healthy death_25 death_50 death_75

Dependencies
------------
    numpy
    pyvista

The generated VTPs contain cell-data arrays that are convenient for ParaView,
including:
    fiber_id
    fiber_number
    n_points
    MU_id
    MU_id_0_based
    MU_fiber_count_model
    MU_fiber_count_vtp
    MU_fraction_model
    MU_active
    MU_killed_or_silent
    MU_type_id
    MU_recruitment_rank
    MU_stimulation_frequency_hz
    MU_expansion_ratio
    case_group_number
    case_protocol_id       (0=A, 1=B)
    case_kill_fraction

A manifest CSV is written to the output directory for auditability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DISTRIBUTION_FILENAME = "MU_fibre_distribution_37x37_20.txt"
STAGE_ORDER = {
    "healthy": 0,
    "death_25": 1,
    "death_50": 2,
    "death_75": 3,
}


@dataclass(frozen=True)
class CaseInfo:
    case_dir: Path
    distribution_file: Path
    group_number: Optional[int]
    group_tag: str
    protocol: str
    stage: str
    scenario_name: str
    kill_fraction: float


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista is required for VTP generation. Install it in your conda "
            "environment, e.g. `conda install -c conda-forge pyvista`, then rerun."
        ) from exc
    return pv


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json_if_exists(path: Path) -> Dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def safe_float(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def safe_int(value, default=-1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


# -----------------------------------------------------------------------------
# OpenDiHu fiber .bin -> base VTP
# -----------------------------------------------------------------------------

def _read_exact(file_obj, n_bytes: int, description: str) -> bytes:
    data = file_obj.read(n_bytes)
    if len(data) != n_bytes:
        raise EOFError(
            f"Unexpected end of file while reading {description}: "
            f"expected {n_bytes} bytes, got {len(data)}."
        )
    return data


def read_opendihu_fiber_bin(bin_file: Path) -> Tuple[List[np.ndarray], Dict]:
    """Read an OpenDiHu fiber binary file.

    The parsing follows the structure used by the user's existing
    my_examine_bin_fibers.py script: 32-byte header, integer header length,
    integer parameter block, then x/y/z doubles for every fiber point.

    Fibers containing two or more exact [0, 0, 0] points are marked invalid,
    matching the previous script's behavior. One isolated near-zero point is
    simply removed later when the VTP is built.
    """

    bin_file = Path(bin_file)
    if not bin_file.is_file():
        raise FileNotFoundError(f'Fiber binary file does not exist: "{bin_file}"')

    with bin_file.open("rb") as infile:
        header_raw = _read_exact(infile, 32, "32-byte OpenDiHu header")
        header_str = struct.unpack("32s", header_raw)[0].decode(
            "utf-8", errors="replace"
        ).rstrip("\x00")

        header_length_raw = _read_exact(infile, 4, "header length")
        header_length = struct.unpack("i", header_length_raw)[0]

        if header_length < 8 or header_length % 4 != 0:
            raise ValueError(
                f"Invalid OpenDiHu header length {header_length}; expected a "
                "positive multiple of 4."
            )

        n_parameter_ints = int(header_length / 4) - 1
        parameters: List[int] = []
        for i in range(n_parameter_ints):
            raw = _read_exact(infile, 4, f"header parameter {i}")
            parameters.append(struct.unpack("i", raw)[0])

        if len(parameters) < 2:
            raise ValueError(
                "OpenDiHu fiber header contains fewer than two parameters; "
                "cannot determine fiber/point counts."
            )

        n_fibers_total = int(parameters[0])
        n_points_whole_fiber = int(parameters[1])

        if n_fibers_total <= 0 or n_points_whole_fiber <= 0:
            raise ValueError(
                f"Invalid geometry dimensions: n_fibers={n_fibers_total}, "
                f"n_points_per_fiber={n_points_whole_fiber}."
            )

        is_version_2 = "version 2" in header_str.lower()
        if is_version_2:
            if len(parameters) < 4:
                raise ValueError(
                    "Version-2 OpenDiHu fiber file does not provide n_fibers_x/y."
                )
            n_fibers_x = int(parameters[2])
            n_fibers_y = int(parameters[3])
        else:
            n_fibers_x = int(round(math.sqrt(n_fibers_total)))
            n_fibers_y = n_fibers_x

        expected_values = n_fibers_total * n_points_whole_fiber * 3
        # np.fromfile reads from the current file position and is substantially
        # faster than nested struct.unpack loops for the full geometry.
        values = np.fromfile(infile, dtype=np.float64, count=expected_values)

        if values.size != expected_values:
            raise EOFError(
                f"Fiber payload is incomplete: expected {expected_values} "
                f"double values, got {values.size}."
            )

    points = values.reshape((n_fibers_total, n_points_whole_fiber, 3))

    streamlines: List[np.ndarray] = []
    invalid_fiber_ids: List[int] = []

    for fiber_id in range(n_fibers_total):
        streamline = points[fiber_id]
        exact_zero = np.all(streamline == 0.0, axis=1)

        if int(np.count_nonzero(exact_zero)) >= 2:
            streamlines.append(np.empty((0, 3), dtype=float))
            invalid_fiber_ids.append(fiber_id)
        else:
            streamlines.append(np.asarray(streamline, dtype=float))

    metadata = {
        "header": header_str,
        "header_length": int(header_length),
        "parameters": [int(x) for x in parameters],
        "n_fibers_total": int(n_fibers_total),
        "n_points_whole_fiber": int(n_points_whole_fiber),
        "n_fibers_x": int(n_fibers_x),
        "n_fibers_y": int(n_fibers_y),
        "n_invalid_fibers": int(len(invalid_fiber_ids)),
        "invalid_fiber_ids_0_based": invalid_fiber_ids,
    }

    return streamlines, metadata


def streamlines_to_polydata(streamlines: Sequence[np.ndarray]):
    """Convert streamlines to a PyVista PolyData with one polyline per fiber.

    The original zero-based fiber index is stored in cell_data['fiber_id'].
    Invalid/skipped fibers therefore do not shift the MU ownership mapping.
    """

    pv = require_pyvista()

    all_points: List[np.ndarray] = []
    lines: List[np.ndarray] = []
    fiber_ids: List[int] = []
    fiber_numbers: List[int] = []
    fiber_point_counts: List[int] = []
    point_offset = 0

    for fiber_id, streamline in enumerate(streamlines):
        if len(streamline) == 0:
            continue

        arr = np.asarray(streamline, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Fiber {fiber_id} does not contain Nx3 coordinates.")

        arr = arr[:, :3]
        # Preserve the filtering used by the existing conversion script.
        valid_mask = np.linalg.norm(arr, axis=1) >= 1e-3
        valid_points = arr[valid_mask]

        if len(valid_points) < 2:
            continue

        all_points.append(valid_points)
        point_ids = np.arange(
            point_offset,
            point_offset + len(valid_points),
            dtype=np.int64,
        )
        lines.append(np.concatenate(([len(valid_points)], point_ids)))

        fiber_ids.append(int(fiber_id))
        fiber_numbers.append(int(fiber_id + 1))
        fiber_point_counts.append(int(len(valid_points)))
        point_offset += len(valid_points)

    if not all_points:
        raise RuntimeError("No valid fiber streamlines were available for VTP output.")

    mesh = pv.PolyData()
    mesh.points = np.vstack(all_points).astype(float, copy=False)
    mesh.lines = np.concatenate(lines).astype(np.int64, copy=False)
    mesh.cell_data["fiber_id"] = np.asarray(fiber_ids, dtype=np.int32)
    mesh.cell_data["fiber_number"] = np.asarray(fiber_numbers, dtype=np.int32)
    mesh.cell_data["n_points"] = np.asarray(fiber_point_counts, dtype=np.int32)

    return mesh


def create_or_load_base_vtp(
    fiber_bin: Optional[Path],
    base_vtp: Path,
    reuse_base_vtp: bool,
    overwrite: bool,
):
    """Create the geometry VTP once, or load a cached/base VTP."""

    pv = require_pyvista()
    base_vtp = Path(base_vtp)

    if reuse_base_vtp:
        if not base_vtp.is_file():
            raise FileNotFoundError(
                f'--reuse-base-vtp was specified but file does not exist: "{base_vtp}"'
            )
        mesh = pv.read(base_vtp)
        if not isinstance(mesh, pv.PolyData):
            raise TypeError(f'Base VTP is not PolyData: "{base_vtp}"')
        return mesh, {"source": "reused_base_vtp", "base_vtp": str(base_vtp)}

    if fiber_bin is None:
        raise ValueError("--fiber-bin is required unless --reuse-base-vtp is used.")

    if base_vtp.exists() and not overwrite:
        # Existing base geometry is safe to reuse automatically.  This avoids
        # re-reading a large binary file on repeated visualization runs.
        mesh = pv.read(base_vtp)
        if not isinstance(mesh, pv.PolyData):
            raise TypeError(f'Existing base VTP is not PolyData: "{base_vtp}"')
        return mesh, {"source": "existing_base_vtp", "base_vtp": str(base_vtp)}

    streamlines, bin_metadata = read_opendihu_fiber_bin(Path(fiber_bin))
    mesh = streamlines_to_polydata(streamlines)
    ensure_dir(base_vtp.parent)
    mesh.save(base_vtp)

    metadata_file = base_vtp.with_suffix(".json")
    metadata_file.write_text(json.dumps(bin_metadata, indent=2), encoding="utf-8")

    info = dict(bin_metadata)
    info["source"] = "fiber_bin"
    info["fiber_bin"] = str(Path(fiber_bin).resolve())
    info["base_vtp"] = str(base_vtp.resolve())
    info["n_vtp_fibers"] = int(mesh.n_cells)
    return mesh, info


# -----------------------------------------------------------------------------
# MU-distribution assignment
# -----------------------------------------------------------------------------

def load_mu_distribution(filename: Path) -> np.ndarray:
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f'MU distribution file does not exist: "{filename}"')

    cleaned_lines: List[str] = []
    with filename.open("r", encoding="utf-8") as file:
        for line in file:
            cleaned_lines.append(line.split("#", maxsplit=1)[0])

    text = "\n".join(cleaned_lines)
    values = re.findall(r"[-+]?\d+", text)
    if not values:
        raise ValueError(f'No integer MU identifiers found in "{filename}".')
    return np.asarray(values, dtype=np.int64)


def convert_to_integer_indices(values: np.ndarray, field_name: str) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        raise ValueError(f'The VTP field "{field_name}" contains non-integer values.')
    return rounded.astype(np.int64)


def match_mu_ids_to_vtp(mesh, mu_distribution: np.ndarray) -> Tuple[np.ndarray, str]:
    n_cells = int(mesh.n_cells)
    n_values = int(len(mu_distribution))

    if "fiber_id" in mesh.cell_data:
        fiber_ids = convert_to_integer_indices(mesh.cell_data["fiber_id"], "fiber_id")
        if len(fiber_ids) != n_cells:
            raise ValueError(
                f'"fiber_id" has {len(fiber_ids)} values but VTP has {n_cells} cells.'
            )
        if fiber_ids.size == 0 or fiber_ids.min() < 0:
            raise ValueError('The "fiber_id" field is empty or contains negatives.')
        if fiber_ids.max() >= n_values:
            raise ValueError(
                f"Largest fiber_id={fiber_ids.max()} but distribution contains "
                f"only {n_values} ownership values."
            )
        return mu_distribution[fiber_ids], "fiber_id"

    if "fiber_number" in mesh.cell_data:
        fiber_numbers = convert_to_integer_indices(
            mesh.cell_data["fiber_number"], "fiber_number"
        )
        if fiber_numbers.size == 0 or fiber_numbers.min() < 1:
            raise ValueError('The "fiber_number" field must be one-based.')
        if fiber_numbers.max() > n_values:
            raise ValueError(
                f"Largest fiber_number={fiber_numbers.max()} but distribution "
                f"contains only {n_values} ownership values."
            )
        return mu_distribution[fiber_numbers - 1], "fiber_number"

    if n_cells == n_values:
        return mu_distribution.copy(), "cell order"

    raise ValueError(
        "Could not match MU ownership to VTP fibers. "
        f"VTP cells={n_cells}, ownership values={n_values}, "
        f"cell fields={list(mesh.cell_data.keys())}."
    )


def infer_distribution_index_base(mu_distribution: np.ndarray) -> int:
    """Infer 0- vs 1-based MU IDs from the ownership file."""
    if mu_distribution.size == 0:
        return 0
    return 0 if int(mu_distribution.min()) == 0 else 1


# -----------------------------------------------------------------------------
# Case discovery and metadata
# -----------------------------------------------------------------------------

def _metadata_candidates(case_dir: Path) -> Iterable[Dict]:
    for name in (
        "generation_metadata.json",
        "protocol_B_metadata.json",
        "motor_units.json",
    ):
        payload = read_json_if_exists(case_dir / name)
        if payload:
            yield payload


def infer_case_info(distribution_file: Path, study_root: Path) -> CaseInfo:
    case_dir = distribution_file.parent

    merged: Dict = {}
    for payload in _metadata_candidates(case_dir):
        # Earlier candidates have priority.
        for key, value in payload.items():
            merged.setdefault(key, value)

    rel_parts = list(case_dir.resolve().relative_to(study_root.resolve()).parts)

    group_number: Optional[int] = None
    group_tag = str(merged.get("group_tag", ""))
    if "group_number" in merged:
        try:
            group_number = int(merged["group_number"])
        except (TypeError, ValueError):
            pass

    if group_number is None:
        for part in rel_parts:
            m = re.fullmatch(r"g-(\d+)", part)
            if m:
                group_number = int(m.group(1))
                group_tag = part
                break

    if not group_tag and group_number is not None:
        group_tag = f"g-{group_number}"

    protocol = str(merged.get("protocol", "")).upper()
    if protocol not in {"A", "B"}:
        for part in rel_parts:
            if part.lower() == "protocol_a":
                protocol = "A"
                break
            if part.lower() == "protocol_b":
                protocol = "B"
                break

    stage = str(merged.get("stage", ""))
    if not stage:
        for candidate in reversed(rel_parts):
            if candidate in STAGE_ORDER:
                stage = candidate
                break
    if not stage:
        stage = case_dir.name

    scenario_name = str(merged.get("scenario_name", ""))
    if not scenario_name:
        if group_tag and protocol:
            scenario_name = f"{group_tag}_{protocol}_{stage}"
        elif protocol:
            scenario_name = f"{protocol}_{stage}"
        else:
            scenario_name = stage

    default_kill = {
        "healthy": 0.0,
        "death_25": 0.25,
        "death_50": 0.50,
        "death_75": 0.75,
    }.get(stage, np.nan)
    kill_fraction = safe_float(merged.get("kill_fraction", default_kill), default_kill)

    return CaseInfo(
        case_dir=case_dir,
        distribution_file=distribution_file,
        group_number=group_number,
        group_tag=group_tag,
        protocol=protocol,
        stage=stage,
        scenario_name=scenario_name,
        kill_fraction=float(kill_fraction),
    )


def discover_cases(
    study_root: Path,
    groups: Optional[Sequence[int]] = None,
    protocols: Optional[Sequence[str]] = None,
    stages: Optional[Sequence[str]] = None,
) -> List[CaseInfo]:
    study_root = Path(study_root)
    if not study_root.is_dir():
        raise NotADirectoryError(f'Study root does not exist: "{study_root}"')

    group_filter = None if not groups else {int(x) for x in groups}
    protocol_filter = None if not protocols else {str(x).upper() for x in protocols}
    stage_filter = None if not stages else {str(x) for x in stages}

    files = sorted(study_root.rglob(DISTRIBUTION_FILENAME))
    cases: List[CaseInfo] = []

    for f in files:
        case = infer_case_info(f, study_root)

        if group_filter is not None and case.group_number not in group_filter:
            continue
        if protocol_filter is not None and case.protocol not in protocol_filter:
            continue
        if stage_filter is not None and case.stage not in stage_filter:
            continue

        cases.append(case)

    def sort_key(c: CaseInfo):
        group_sort = c.group_number if c.group_number is not None else 10**9
        protocol_sort = {"A": 0, "B": 1}.get(c.protocol, 9)
        stage_sort = STAGE_ORDER.get(c.stage, 99)
        return (group_sort, protocol_sort, stage_sort, c.scenario_name)

    cases.sort(key=sort_key)
    return cases


# -----------------------------------------------------------------------------
# Optional MU-level data for richer ParaView visualization
# -----------------------------------------------------------------------------

def load_motor_units(case_dir: Path) -> Tuple[List[Dict], int]:
    payload = read_json_if_exists(case_dir / "motor_units.json")
    motor_units = payload.get("motor_units", [])
    if not isinstance(motor_units, list):
        motor_units = []
    index_base = safe_int(payload.get("index_base", 1), 1)
    return motor_units, index_base


def load_summary_by_mu(case_dir: Path) -> Dict[int, Dict[str, str]]:
    """Load whichever MU summary is available and index it by zero-based MU ID."""
    candidates = [
        case_dir / "protocol_B_motor_unit_summary.csv",
        case_dir / "motor_unit_summary.csv",
    ]

    for path in candidates:
        if not path.is_file():
            continue
        rows: Dict[int, Dict[str, str]] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "mu_id_0_based" in row:
                    mu_no = safe_int(row.get("mu_id_0_based"), -1)
                else:
                    mu_no = safe_int(row.get("mu_no_0_based"), -1)
                if mu_no >= 0:
                    rows[mu_no] = row
        return rows

    return {}


def _lookup_mu_property(
    assigned_mu_ids: np.ndarray,
    distribution_index_base: int,
    motor_units: List[Dict],
    key: str,
    default,
    dtype,
) -> np.ndarray:
    result = np.empty(len(assigned_mu_ids), dtype=dtype)
    for i, mu_id in enumerate(assigned_mu_ids):
        mu_no = int(mu_id) - int(distribution_index_base)
        if 0 <= mu_no < len(motor_units):
            value = motor_units[mu_no].get(key, default)
        else:
            value = default
        try:
            result[i] = value
        except (TypeError, ValueError, OverflowError):
            result[i] = default
    return result


def add_case_mu_fields(mesh, case: CaseInfo, mu_distribution: np.ndarray) -> Dict:
    """Add case-specific MU ownership and physiological fields to a VTP copy."""

    assigned_mu_ids, matching_method = match_mu_ids_to_vtp(mesh, mu_distribution)
    distribution_index_base = infer_distribution_index_base(mu_distribution)
    assigned_zero_based = assigned_mu_ids.astype(np.int64) - distribution_index_base

    if np.any(assigned_zero_based < 0):
        raise ValueError(
            f"Case {case.scenario_name}: inferred index base {distribution_index_base} "
            "creates negative MU indices."
        )

    # Full modeled MU counts (all ownership values, including fibers that may be
    # absent from the VTP because the geometry parser marked them invalid).
    unique_model, counts_model = np.unique(mu_distribution, return_counts=True)
    model_count_lookup = {
        int(mu): int(count) for mu, count in zip(unique_model, counts_model)
    }

    # Counts represented in the VTP itself.
    unique_vtp, counts_vtp = np.unique(assigned_mu_ids, return_counts=True)
    vtp_count_lookup = {
        int(mu): int(count) for mu, count in zip(unique_vtp, counts_vtp)
    }

    mesh.cell_data["MU_id"] = assigned_mu_ids.astype(np.int32)
    mesh.cell_data["MU_id_0_based"] = assigned_zero_based.astype(np.int32)
    mesh.cell_data["MU_fiber_count_model"] = np.asarray(
        [model_count_lookup[int(mu)] for mu in assigned_mu_ids], dtype=np.int32
    )
    mesh.cell_data["MU_fiber_count_vtp"] = np.asarray(
        [vtp_count_lookup[int(mu)] for mu in assigned_mu_ids], dtype=np.int32
    )
    mesh.cell_data["MU_fraction_model"] = np.asarray(
        [model_count_lookup[int(mu)] / len(mu_distribution) for mu in assigned_mu_ids],
        dtype=np.float64,
    )

    motor_units, motor_units_index_base = load_motor_units(case.case_dir)
    # Normally both are one-based for the current study. The distribution file
    # itself is authoritative for the fiber -> MU index conversion.
    if motor_units:
        mesh.cell_data["MU_active"] = _lookup_mu_property(
            assigned_mu_ids,
            distribution_index_base,
            motor_units,
            "active",
            False,
            np.int8,
        )
        mesh.cell_data["MU_killed_or_silent"] = _lookup_mu_property(
            assigned_mu_ids,
            distribution_index_base,
            motor_units,
            "killed_or_silent",
            False,
            np.int8,
        )
        mesh.cell_data["MU_type_id"] = _lookup_mu_property(
            assigned_mu_ids,
            distribution_index_base,
            motor_units,
            "type_id",
            -1,
            np.int32,
        )
        mesh.cell_data["MU_recruitment_rank"] = _lookup_mu_property(
            assigned_mu_ids,
            distribution_index_base,
            motor_units,
            "recruitment_rank_0_based",
            -1,
            np.int32,
        )
        mesh.cell_data["MU_stimulation_frequency_hz"] = _lookup_mu_property(
            assigned_mu_ids,
            distribution_index_base,
            motor_units,
            "stimulation_frequency",
            0.0,
            np.float64,
        )

    summary_by_mu = load_summary_by_mu(case.case_dir)
    if summary_by_mu:
        expansion = np.ones(len(assigned_mu_ids), dtype=np.float64)
        for i, mu_no in enumerate(assigned_zero_based):
            row = summary_by_mu.get(int(mu_no), {})
            expansion[i] = safe_float(row.get("expansion_ratio", 1.0), 1.0)
        mesh.cell_data["MU_expansion_ratio"] = expansion

    # Case-level numeric metadata repeated on cells makes ParaView filtering and
    # coloring easy even when multiple VTPs are loaded together.
    group_number = case.group_number if case.group_number is not None else -1
    protocol_id = {"A": 0, "B": 1}.get(case.protocol, -1)
    mesh.cell_data["case_group_number"] = np.full(
        mesh.n_cells, int(group_number), dtype=np.int32
    )
    mesh.cell_data["case_protocol_id"] = np.full(
        mesh.n_cells, int(protocol_id), dtype=np.int8
    )
    mesh.cell_data["case_kill_fraction"] = np.full(
        mesh.n_cells, float(case.kill_fraction), dtype=np.float64
    )

    return {
        "matching_method": matching_method,
        "distribution_index_base": int(distribution_index_base),
        "motor_units_index_base": int(motor_units_index_base),
        "n_distribution_values": int(len(mu_distribution)),
        "n_vtp_fibers": int(mesh.n_cells),
        "n_represented_mus": int(len(unique_vtp)),
        "mu_id_min": int(assigned_mu_ids.min()) if len(assigned_mu_ids) else None,
        "mu_id_max": int(assigned_mu_ids.max()) if len(assigned_mu_ids) else None,
    }


# -----------------------------------------------------------------------------
# Output paths, manifests, and validation
# -----------------------------------------------------------------------------

def case_output_path(case: CaseInfo, output_dir: Path, flat_output: bool) -> Path:
    filename = f"fibers_{case.scenario_name}_MU_assigned.vtp"
    if flat_output:
        return output_dir / filename

    parts: List[str] = []
    if case.group_tag:
        parts.append(case.group_tag)
    if case.protocol:
        parts.append(f"protocol_{case.protocol}")
    parts.append(case.stage)
    return output_dir.joinpath(*parts, filename)


def validate_group_case_counts(cases: Sequence[CaseInfo], strict_eight: bool) -> None:
    grouped: Dict[Optional[int], List[CaseInfo]] = {}
    for case in cases:
        grouped.setdefault(case.group_number, []).append(case)

    for group_no, group_cases in grouped.items():
        if group_no is None:
            continue
        expected = {
            (protocol, stage)
            for protocol in ("A", "B")
            for stage in STAGE_ORDER
        }
        found = {(c.protocol, c.stage) for c in group_cases}
        missing = sorted(expected - found)
        extras = sorted(found - expected)

        if missing:
            msg = f"g-{group_no} is missing expected cases: {missing}"
            if strict_eight:
                raise RuntimeError(msg)
            print(f"Warning: {msg}")
        if extras:
            print(f"Warning: g-{group_no} has additional case labels: {extras}")


def validate_protocol_pair_hashes(cases: Sequence[CaseInfo]) -> List[str]:
    """Check that A/B distributions are identical within group/stage when paired."""
    messages: List[str] = []
    lookup = {(c.group_number, c.protocol, c.stage): c for c in cases}

    group_stage_keys = sorted(
        {(c.group_number, c.stage) for c in cases if c.group_number is not None}
    )
    for group_no, stage in group_stage_keys:
        a = lookup.get((group_no, "A", stage))
        b = lookup.get((group_no, "B", stage))
        if a is None or b is None:
            continue
        ha = sha256_file(a.distribution_file)
        hb = sha256_file(b.distribution_file)
        if ha != hb:
            raise RuntimeError(
                f"Paired-distribution mismatch for g-{group_no}, {stage}:\n"
                f"  A: {a.distribution_file}\n"
                f"  B: {b.distribution_file}"
            )
        messages.append(f"g-{group_no} {stage}: A/B distribution SHA-256 match")
    return messages


def write_manifest(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one OpenDiHu fiber BIN geometry and all discovered study "
            "MU distributions into case-specific ParaView VTP files."
        )
    )

    parser.add_argument(
        "--fiber-bin",
        type=Path,
        default=None,
        help="OpenDiHu *.bin fiber geometry. Required unless --reuse-base-vtp is used.",
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="Root directory containing grouped Protocol A/B case folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Required destination directory for all case-specific VTP files.",
    )
    parser.add_argument(
        "--base-vtp",
        type=Path,
        default=None,
        help=(
            "Cache path for the geometry-only VTP. Default: "
            "<output-dir>/_base/fiber_geometry.vtp"
        ),
    )
    parser.add_argument(
        "--reuse-base-vtp",
        action="store_true",
        help="Use --base-vtp directly instead of reading --fiber-bin.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        type=int,
        default=None,
        help="Optional group numbers to process, e.g. --groups 1 2 3. Default: all.",
    )
    parser.add_argument(
        "--protocols",
        nargs="*",
        choices=["A", "B", "a", "b"],
        default=None,
        help="Optional protocols to process. Default: A and B/all discovered.",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        choices=list(STAGE_ORDER.keys()),
        default=None,
        help="Optional denervation stages. Default: all discovered.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write every VTP directly into --output-dir instead of mirroring group/protocol/stage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing case VTP outputs and regenerate the base VTP.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an existing case VTP instead of stopping. Ignored with --overwrite.",
    )
    parser.add_argument(
        "--strict-eight-per-group",
        action="store_true",
        help="Require exactly the standard 8 A/B x severity cases for each selected group.",
    )
    parser.add_argument(
        "--no-pair-check",
        action="store_true",
        help="Disable verification that paired Protocol A/B MU distributions are byte-identical.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and list cases without creating VTP files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    study_root = args.study_root.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    protocols = None
    if args.protocols:
        protocols = [p.upper() for p in args.protocols]

    cases = discover_cases(
        study_root=study_root,
        groups=args.groups,
        protocols=protocols,
        stages=args.stages,
    )

    if not cases:
        raise RuntimeError(
            f'No case files named "{DISTRIBUTION_FILENAME}" were discovered below '
            f'"{study_root}" with the requested filters.'
        )

    validate_group_case_counts(cases, strict_eight=args.strict_eight_per_group)

    print("Discovered cases:")
    for i, case in enumerate(cases, start=1):
        print(
            f"  {i:3d}. {case.scenario_name:<24s} "
            f"{case.distribution_file}"
        )
    print(f"Total cases: {len(cases)}")

    if not args.no_pair_check:
        pair_messages = validate_protocol_pair_hashes(cases)
        for msg in pair_messages:
            print(f"Pair check: {msg}")

    if args.dry_run:
        print("Dry run completed; no VTP files were written.")
        return

    base_vtp = (
        args.base_vtp.resolve()
        if args.base_vtp is not None
        else output_dir / "_base" / "fiber_geometry.vtp"
    )

    base_mesh, base_info = create_or_load_base_vtp(
        fiber_bin=args.fiber_bin,
        base_vtp=base_vtp,
        reuse_base_vtp=args.reuse_base_vtp,
        overwrite=args.overwrite,
    )

    print("")
    print("Base fiber geometry:")
    print(f"  VTP:             {base_vtp}")
    print(f"  Polyline cells:  {base_mesh.n_cells}")
    print(f"  Points:          {base_mesh.n_points}")
    if "n_fibers_total" in base_info:
        print(f"  BIN fibers:      {base_info['n_fibers_total']}")
        print(f"  Invalid fibers:  {base_info.get('n_invalid_fibers', 0)}")

    manifest_rows: List[Dict] = []
    n_created = 0
    n_skipped = 0

    for index, case in enumerate(cases, start=1):
        output_vtp = case_output_path(case, output_dir, args.flat_output)
        ensure_dir(output_vtp.parent)

        if output_vtp.exists() and not args.overwrite:
            if args.skip_existing:
                print(f"[{index}/{len(cases)}] SKIP {case.scenario_name}: {output_vtp}")
                n_skipped += 1
                manifest_rows.append({
                    "group_number": case.group_number if case.group_number is not None else "",
                    "group_tag": case.group_tag,
                    "protocol": case.protocol,
                    "stage": case.stage,
                    "scenario_name": case.scenario_name,
                    "kill_fraction": case.kill_fraction,
                    "case_directory": str(case.case_dir.resolve()),
                    "distribution_file": str(case.distribution_file.resolve()),
                    "distribution_sha256": sha256_file(case.distribution_file),
                    "output_vtp": str(output_vtp.resolve()),
                    "status": "skipped_existing",
                })
                continue
            raise FileExistsError(
                f'Output already exists: "{output_vtp}". Use --overwrite or '
                "--skip-existing."
            )

        distribution = load_mu_distribution(case.distribution_file)
        mesh = base_mesh.copy(deep=True)
        assignment_info = add_case_mu_fields(mesh, case, distribution)
        mesh.save(output_vtp)
        n_created += 1

        sidecar = {
            "group_number": case.group_number,
            "group_tag": case.group_tag,
            "protocol": case.protocol,
            "stage": case.stage,
            "scenario_name": case.scenario_name,
            "kill_fraction": case.kill_fraction,
            "case_directory": str(case.case_dir.resolve()),
            "distribution_file": str(case.distribution_file.resolve()),
            "distribution_sha256": sha256_file(case.distribution_file),
            "base_vtp": str(base_vtp.resolve()),
            "output_vtp": str(output_vtp.resolve()),
            **assignment_info,
        }
        output_vtp.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )

        manifest_rows.append({
            "group_number": case.group_number if case.group_number is not None else "",
            "group_tag": case.group_tag,
            "protocol": case.protocol,
            "stage": case.stage,
            "scenario_name": case.scenario_name,
            "kill_fraction": case.kill_fraction,
            "case_directory": str(case.case_dir.resolve()),
            "distribution_file": str(case.distribution_file.resolve()),
            "distribution_sha256": sidecar["distribution_sha256"],
            "output_vtp": str(output_vtp.resolve()),
            "matching_method": assignment_info["matching_method"],
            "distribution_index_base": assignment_info["distribution_index_base"],
            "n_distribution_values": assignment_info["n_distribution_values"],
            "n_vtp_fibers": assignment_info["n_vtp_fibers"],
            "n_represented_mus": assignment_info["n_represented_mus"],
            "status": "created",
        })

        print(
            f"[{index}/{len(cases)}] CREATED {case.scenario_name}: "
            f"{output_vtp.name} | fibers={assignment_info['n_vtp_fibers']} | "
            f"MUs={assignment_info['n_represented_mus']}"
        )

    manifest_path = output_dir / "mu_assigned_vtp_manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    run_metadata = {
        "study_root": str(study_root),
        "output_dir": str(output_dir),
        "base_vtp": str(base_vtp.resolve()),
        "fiber_bin": str(args.fiber_bin.resolve()) if args.fiber_bin is not None else None,
        "n_cases_discovered": len(cases),
        "n_created": n_created,
        "n_skipped": n_skipped,
        "groups_filter": args.groups,
        "protocols_filter": protocols,
        "stages_filter": args.stages,
        "pair_check_enabled": not args.no_pair_check,
        "base_geometry": base_info,
    }
    (output_dir / "mu_assigned_vtp_generation_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    print("")
    print("Completed MU-assigned fiber generation.")
    print(f"  Created:  {n_created}")
    print(f"  Skipped:  {n_skipped}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output:   {output_dir}")
    print("")
    print("ParaView: load any generated *_MU_assigned.vtp file and color by")
    print("  MU_id, MU_type_id, MU_active, MU_stimulation_frequency_hz,")
    print("  MU_expansion_ratio, or MU_fiber_count_model.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
