#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate MU firing rasters from OpenDiHu stimulation.log files.

IMPORTANT: OpenDiHu stimulation.log is parsed in the native format used by the
project's existing visualization script:

    MU_no ; fiber_no ; t1 ; t2 ; t3 ; ... ;

There is one row per fiber. All fibers belonging to the same MU normally carry
the same MU stimulation train. Therefore, the MU-level firing raster is formed
by taking the UNION of stimulation times across all fibers assigned to each MU.
No spike timing is reconstructed from motor_units.json.

The raster always displays all configured MUs (default: 20), including killed or
silent MUs as empty rows. Thus it is a true firing-train raster:

    MU 20 |  |   |  |     ...
    ...
    MU  2 |      |     |  ...
    MU  1 | | |      |    ...
           -------------------> time

Typical grouped output layout:
    build_release/out/g-1_A_healthy/stimulation.log
    build_release/out/g-1_A_death_25/stimulation.log
    ...

Example:
    python tools/generate_mu_firing_rasters_from_stimulation_log.py \
        --out-root build_release/out \
        --out-dir mu_firing_rasters \
        --n-motor-units 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

STAGES = ["healthy", "death_25", "death_50", "death_75"]
STAGE_DISPLAY = {
    "healthy": "Healthy",
    "death_25": "25% MU death",
    "death_50": "50% MU death",
    "death_75": "75% MU death",
}

CASE_RE = re.compile(
    r"^g[-_](?P<group>\d+)[_-](?P<protocol>[ABab])[_-]"
    r"(?P<stage>healthy|death[_-]?25|death[_-]?50|death[_-]?75)$"
)


@dataclass(frozen=True)
class CaseRef:
    group_number: int
    protocol: str
    stage: str
    scenario_name: str
    case_dir: Path
    stimulation_log: Path
    motor_units_file: Optional[Path]

    @property
    def key(self) -> Tuple[int, str, str]:
        return self.group_number, self.protocol, self.stage


def normalize_stage(s: str) -> str:
    return s.lower().replace("-", "_")


def discover_cases(out_root: Path, log_name: str) -> Dict[Tuple[int, str, str], CaseRef]:
    result: Dict[Tuple[int, str, str], CaseRef] = {}
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        m = CASE_RE.match(d.name)
        if not m:
            continue
        group = int(m.group("group"))
        protocol = m.group("protocol").upper()
        stage = normalize_stage(m.group("stage"))
        log = d / log_name
        mu_json = d / "motor_units.json"
        ref = CaseRef(
            group_number=group,
            protocol=protocol,
            stage=stage,
            scenario_name=f"g-{group}_{protocol}_{stage}",
            case_dir=d,
            stimulation_log=log,
            motor_units_file=mu_json if mu_json.exists() else None,
        )
        result[ref.key] = ref
    if not result:
        raise RuntimeError(
            f"No grouped case folders found under {out_root}. Expected names such as g-1_A_healthy."
        )
    return result


def load_n_motor_units(motor_units_file: Optional[Path], fallback: int) -> int:
    if motor_units_file is not None and motor_units_file.exists():
        try:
            data = json.loads(motor_units_file.read_text())
            mus = data.get("motor_units", [])
            if mus:
                return len(mus)
            if "n_motor_units" in data:
                return int(data["n_motor_units"])
        except Exception:
            pass
    return int(fallback)


def parse_stimulation_log_native(
    filename: Path,
    time_unit: str = "ms",
) -> Tuple[Dict[int, Set[float]], Dict[int, List[float]], Dict[int, int], int]:
    """
    Parse OpenDiHu native stimulation.log:
        MU_no;fiber_no;t1;t2;...;

    Returns
    -------
    mu_times_s:
        MU -> unique stimulation times [s], unioned across its fibers.
    fiber_times_s:
        fiber -> stimulation times [s].
    fiber_mu_no:
        fiber -> MU number exactly as stored in log.
    n_rows:
        number of parsed fiber rows.
    """
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find {filename}")

    mu_times_s: Dict[int, Set[float]] = {}
    fiber_times_s: Dict[int, List[float]] = {}
    fiber_mu_no: Dict[int, int] = {}
    n_rows = 0
    factor = 1.0 / 1000.0 if time_unit == "ms" else 1.0

    with filename.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            values = [v.strip() for v in raw.split(";")]
            while values and values[-1] == "":
                values.pop()
            if len(values) < 2:
                continue

            try:
                mu_no = int(float(values[0]))
                fiber_no = int(float(values[1]))
            except ValueError as exc:
                raise ValueError(
                    f"{filename}, line {line_no}: expected MU_no;fiber_no;times..., got: {raw[:160]}"
                ) from exc

            times: List[float] = []
            for token in values[2:]:
                if token == "":
                    continue
                try:
                    times.append(float(token) * factor)
                except ValueError as exc:
                    raise ValueError(
                        f"{filename}, line {line_no}: invalid stimulation time {token!r}"
                    ) from exc

            fiber_mu_no[fiber_no] = mu_no
            fiber_times_s[fiber_no] = times
            mu_times_s.setdefault(mu_no, set()).update(times)
            n_rows += 1

    return mu_times_s, fiber_times_s, fiber_mu_no, n_rows


def infer_logged_mu_base(mu_times: Dict[int, Set[float]], n_motor_units: int, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    ids = sorted(mu_times)
    if not ids:
        return 0
    if 0 in ids:
        return 0
    if max(ids) == n_motor_units:
        return 1
    # Existing OpenDiHu examples commonly expose internal MU numbers. If the
    # numbering is ambiguous and starts at 1 because MU 0 is silent/absent,
    # users can override with --logged-mu-index-base 1.
    return 0


def convert_to_zero_based_mu_times(
    mu_times_logged: Dict[int, Set[float]],
    n_motor_units: int,
    logged_base: int,
) -> Dict[int, List[float]]:
    result = {mu: [] for mu in range(n_motor_units)}
    for logged_mu, times in mu_times_logged.items():
        mu0 = int(logged_mu) - int(logged_base)
        if mu0 < 0 or mu0 >= n_motor_units:
            raise ValueError(
                f"Logged MU {logged_mu} resolves to {mu0}, outside 0..{n_motor_units-1}. "
                "Check --logged-mu-index-base."
            )
        result[mu0] = sorted(set(float(t) for t in times))
    return result


def filter_window(mu_times: Dict[int, List[float]], start_s: float, end_s: float) -> Dict[int, List[float]]:
    return {
        mu: [t for t in times if start_s <= t <= end_s]
        for mu, times in mu_times.items()
    }


def write_event_csv(mu_times: Dict[int, List[float]], out_file: Path, ref: CaseRef) -> None:
    rows = []
    for mu0 in sorted(mu_times):
        for k, t in enumerate(mu_times[mu0]):
            rows.append({
                "group_number": ref.group_number,
                "protocol": ref.protocol,
                "stage": ref.stage,
                "scenario_name": ref.scenario_name,
                "mu_no_0_based": mu0,
                "mu_id_1_based": mu0 + 1,
                "event_index": k,
                "event_time_s": t,
                "event_time_ms": 1000.0 * t,
            })
    pd.DataFrame(rows).to_csv(out_file, index=False)


def plot_firing_raster(
    mu_times: Dict[int, List[float]],
    out_base: Path,
    ref: CaseRef,
    start_s: float,
    end_s: float,
    dpi: int,
    reverse_mu_axis: bool,
) -> None:
    n_mu = len(mu_times)
    fig_h = max(6.0, 0.30 * n_mu + 2.0)
    fig, ax = plt.subplots(figsize=(13.0, fig_h))

    # Plot one horizontal MU row for ALL motor units, even if silent.
    for mu0 in range(n_mu):
        y = mu0 + 1
        ax.hlines(y, start_s, end_s, linewidth=0.35, alpha=0.22)
        times = mu_times.get(mu0, [])
        if times:
            ax.vlines(times, y - 0.38, y + 0.38, linewidth=0.8)

    ax.set_xlim(start_s, end_s)
    ax.set_ylim(0.5, n_mu + 0.5)
    ax.set_yticks(np.arange(1, n_mu + 1))
    ax.set_yticklabels([f"MU {i}" for i in range(1, n_mu + 1)])
    if reverse_mu_axis:
        ax.invert_yaxis()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Motor unit")
    n_active = sum(bool(v) for v in mu_times.values())
    n_events = sum(len(v) for v in mu_times.values())
    ax.set_title(
        f"{ref.scenario_name}: MU firing raster from stimulation.log\n"
        f"{n_active}/{n_mu} MUs fired, {n_events} unique MU firing events"
    )
    ax.grid(True, axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_group_overview(
    cases_data: Dict[Tuple[str, str], Tuple[Dict[int, List[float]], CaseRef]],
    group: int,
    out_base: Path,
    start_s: float,
    end_s: float,
    n_motor_units: int,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), sharex=True, sharey=True, constrained_layout=True)
    for i, protocol in enumerate(("A", "B")):
        for j, stage in enumerate(STAGES):
            ax = axes[i, j]
            item = cases_data.get((protocol, stage))
            if item is None:
                ax.axis("off")
                continue
            mu_times, ref = item
            for mu0 in range(n_motor_units):
                y = mu0 + 1
                times = mu_times.get(mu0, [])
                if times:
                    ax.vlines(times, y - 0.35, y + 0.35, linewidth=0.45)
            ax.set_title(STAGE_DISPLAY[stage])
            ax.set_xlim(start_s, end_s)
            ax.set_ylim(0.5, n_motor_units + 0.5)
            ax.set_yticks([1, 5, 10, 15, 20] if n_motor_units >= 20 else np.arange(1, n_motor_units + 1))
            if j == 0:
                ax.set_ylabel(f"Protocol {protocol}\nMU")
            if i == 1:
                ax.set_xlabel("Time [s]")
            ax.grid(True, axis="x", alpha=0.12)
    fig.suptitle(f"g-{group}: MU firing trains from stimulation.log")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate true MU firing rasters from native OpenDiHu stimulation.log files.")
    p.add_argument("--out-root", default="build_release/out")
    p.add_argument("--out-dir", default="mu_firing_rasters")
    p.add_argument("--log-name", default="stimulation.log")
    p.add_argument("--groups", nargs="+", type=int, default=None)
    p.add_argument("--protocols", nargs="+", choices=["A", "B", "a", "b"], default=["A", "B"])
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--n-motor-units", type=int, default=20)
    p.add_argument("--time-unit", choices=["ms", "s"], default="ms")
    p.add_argument("--time-start-s", type=float, default=0.0)
    p.add_argument("--time-end-s", type=float, default=30.0)
    p.add_argument("--logged-mu-index-base", type=int, choices=[0, 1], default=None)
    p.add_argument("--reverse-mu-axis", action="store_true")
    p.add_argument("--allow-missing-logs", action="store_true")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_end_s <= args.time_start_s:
        raise ValueError("--time-end-s must be greater than --time-start-s")

    out_root = Path(args.out_root)
    output_root = Path(args.out_dir)
    cases = discover_cases(out_root, args.log_name)

    groups = sorted({k[0] for k in cases}) if args.groups is None else sorted(set(args.groups))
    protocols = [p.upper() for p in args.protocols]
    stages = list(args.stages)

    selected = [
        ref for _, ref in sorted(cases.items())
        if ref.group_number in groups and ref.protocol in protocols and ref.stage in stages
    ]

    print("Selected cases:")
    missing = []
    for ref in selected:
        status = str(ref.stimulation_log) if ref.stimulation_log.exists() else "[missing]"
        print(f"  {ref.scenario_name}: {status}")
        if not ref.stimulation_log.exists():
            missing.append(ref)

    if missing and not args.allow_missing_logs:
        raise FileNotFoundError(f"{len(missing)} selected cases are missing stimulation.log")
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    summary_rows = []
    group_data: Dict[int, Dict[Tuple[str, str], Tuple[Dict[int, List[float]], CaseRef]]] = {}

    for ref in selected:
        if not ref.stimulation_log.exists():
            continue

        n_mu = load_n_motor_units(ref.motor_units_file, args.n_motor_units)
        mu_logged, fiber_times, fiber_mu, n_rows = parse_stimulation_log_native(
            ref.stimulation_log, time_unit=args.time_unit
        )
        logged_base = infer_logged_mu_base(mu_logged, n_mu, args.logged_mu_index_base)
        mu_times = convert_to_zero_based_mu_times(mu_logged, n_mu, logged_base)
        mu_times = filter_window(mu_times, args.time_start_s, args.time_end_s)

        case_out = output_root / f"g-{ref.group_number}" / f"protocol_{ref.protocol}" / ref.stage
        case_out.mkdir(parents=True, exist_ok=True)
        base = case_out / f"mu_firing_raster_{ref.scenario_name}"
        plot_firing_raster(
            mu_times,
            base,
            ref,
            args.time_start_s,
            args.time_end_s,
            args.dpi,
            args.reverse_mu_axis,
        )
        event_csv = case_out / f"mu_firing_events_{ref.scenario_name}.csv"
        write_event_csv(mu_times, event_csv, ref)

        n_active = sum(bool(v) for v in mu_times.values())
        n_events = sum(len(v) for v in mu_times.values())
        summary_rows.append({
            "group_number": ref.group_number,
            "protocol": ref.protocol,
            "stage": ref.stage,
            "scenario_name": ref.scenario_name,
            "n_motor_units_total": n_mu,
            "n_motor_units_with_firing": n_active,
            "n_silent_motor_units": n_mu - n_active,
            "n_unique_mu_firing_events": n_events,
            "n_fiber_rows_in_stimulation_log": n_rows,
            "logged_mu_index_base_used": logged_base,
            "time_start_s": args.time_start_s,
            "time_end_s": args.time_end_s,
        })
        manifest_rows.append({
            "group_number": ref.group_number,
            "protocol": ref.protocol,
            "stage": ref.stage,
            "scenario_name": ref.scenario_name,
            "stimulation_log": str(ref.stimulation_log.resolve()),
            "raster_png": str(base.with_suffix(".png").resolve()),
            "raster_pdf": str(base.with_suffix(".pdf").resolve()),
            "event_csv": str(event_csv.resolve()),
        })
        group_data.setdefault(ref.group_number, {})[(ref.protocol, ref.stage)] = (mu_times, ref)
        print(f"Created {base.with_suffix('.png')}")

    pd.DataFrame(summary_rows).to_csv(output_root / "firing_raster_summary.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_root / "firing_raster_manifest.csv", index=False)

    overview = output_root / "group_overviews"
    overview.mkdir(parents=True, exist_ok=True)
    for g, data in sorted(group_data.items()):
        n_mu_values = [len(mu_times) for mu_times, _ in data.values()]
        n_mu = max(n_mu_values) if n_mu_values else args.n_motor_units
        plot_group_overview(
            data,
            g,
            overview / f"mu_firing_overview_g-{g}",
            args.time_start_s,
            args.time_end_s,
            n_mu,
        )

    print("\nCompleted MU firing raster generation from native stimulation.log files.")
    print(f"Output root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
