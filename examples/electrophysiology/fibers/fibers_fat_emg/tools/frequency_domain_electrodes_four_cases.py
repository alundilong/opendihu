#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency-domain and time-frequency analysis of OpenDiHu electrode EMG traces
for four remodeling cases.

This script is designed as a companion to the existing time-voltage comparison
script. It reads the same electrodes.csv formats and default folder layout:

  build_release/out/healthy/electrodes.csv
  build_release/out/death_25/electrodes.csv
  build_release/out/death_50/electrodes.csv
  build_release/out/death_75/electrodes.csv

Primary analysis
----------------
1. Welch power spectral density (PSD) for each electrode and case.
2. Per-electrode spectral metrics:
     - peak / dominant frequency
     - mean frequency (MNF; spectral centroid)
     - median frequency (MDF)
     - spectral bandwidth
     - 95% power frequency
     - total spectral power
     - spectral entropy
     - user-defined band powers and relative band powers
3. Case-level summaries across electrodes.
4. Relative changes versus healthy for each electrode.
5. Case-average PSD overlays (absolute and normalized shape).
6. Optional spectrograms (time-frequency analysis) for selected/all electrodes.
7. Optional spatial heatmaps of spectral metrics across the electrode grid.

Important sampling convention
-----------------------------
OpenDiHu electrode time values are assumed to be in milliseconds, matching the
existing plotting script. Sampling frequency is inferred as:

    fs [Hz] = 1000 / median(diff(t_ms))

The script checks time-step uniformity before spectral analysis.

Dependencies
------------
  numpy
  scipy
  matplotlib

Examples
--------
Basic analysis of all electrodes:
  python frequency_domain_electrodes_four_cases.py

Test first 12 electrodes:
  python frequency_domain_electrodes_four_cases.py --max-electrodes 12

Focus plots/metrics on 20-450 Hz:
  python frequency_domain_electrodes_four_cases.py --fmin 20 --fmax 450

Create spectrograms for first 4 electrodes:
  python frequency_domain_electrodes_four_cases.py \
      --max-electrodes 4 --spectrogram

Use 250 ms Welch windows with 75% overlap:
  python frequency_domain_electrodes_four_cases.py \
      --welch-window-ms 250 --welch-overlap 0.75

Custom frequency bands:
  python frequency_domain_electrodes_four_cases.py \
      --bands "0-20,20-50,50-100,100-150,150-250,250-450"
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence, OrderedDict as OrderedDictType
from collections import OrderedDict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy.signal import welch, spectrogram, detrend as scipy_detrend
except ImportError as exc:
    raise ImportError(
        "This script requires SciPy. Install it in your conda environment with: "
        "conda install scipy"
    ) from exc


plt.rcParams.update({"font.size": 13})


DEFAULT_CASES = [
    ("healthy", "healthy"),
    ("death_25", "death_25"),
    ("death_50", "death_50"),
    ("death_75", "death_75"),
]

DEFAULT_BANDS = "0-20,20-50,50-100,100-150,150-250,250-450"
EPS = np.finfo(float).tiny


# -----------------------------------------------------------------------------
# Input parsing: intentionally kept compatible with the existing time-domain
# electrode plotting script.
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


def read_electrodes_csv(filename: Path) -> Dict[str, np.ndarray]:
    filename = Path(filename)
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
                    f"{filename}, line {line_no}: n_points changed from "
                    f"{n_points_ref} to {n_points}"
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
        electrode_positions = np.asarray(
            position_data[:3 * n_points], dtype=float
        ).reshape(n_points, 3)

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


def ensure_same_layout(
    cases: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, int, int, int]:
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
                "Resampling is intentionally not done because it can alter the spectrum."
            )

    return ref_t, ref_n, ref_xy, ref_z


# -----------------------------------------------------------------------------
# Sampling and preprocessing
# -----------------------------------------------------------------------------

def infer_sampling_frequency(
    t_ms: np.ndarray,
    max_relative_jitter: float = 1e-3,
) -> Tuple[float, float, float]:
    if len(t_ms) < 2:
        raise ValueError("At least two time samples are required for frequency analysis.")

    dt = np.diff(np.asarray(t_ms, dtype=float))
    if np.any(dt <= 0):
        raise ValueError("Time values must be strictly increasing.")

    dt_median_ms = float(np.median(dt))
    max_rel_jitter = float(np.max(np.abs(dt - dt_median_ms)) / max(abs(dt_median_ms), EPS))

    if max_rel_jitter > max_relative_jitter:
        raise ValueError(
            "The time vector is not sufficiently uniform for FFT/Welch analysis: "
            f"median dt={dt_median_ms:.9g} ms, max relative jitter={max_rel_jitter:.3e}. "
            "Resample to a uniform grid before using this script."
        )

    fs_hz = 1000.0 / dt_median_ms
    return fs_hz, dt_median_ms, max_rel_jitter


def preprocess_trace(
    y: np.ndarray,
    detrend_mode: str = "constant",
    baseline_samples: int = 0,
) -> np.ndarray:
    x = np.asarray(y, dtype=float).copy()

    if baseline_samples > 0:
        n = min(int(baseline_samples), len(x))
        if n > 0:
            x -= float(np.mean(x[:n]))

    if detrend_mode == "none":
        return x
    if detrend_mode == "constant":
        return scipy_detrend(x, type="constant")
    if detrend_mode == "linear":
        return scipy_detrend(x, type="linear")
    raise ValueError(f"Unknown detrend mode: {detrend_mode}")


def samples_from_ms(window_ms: float, fs_hz: float, n_samples: int, minimum: int = 8) -> int:
    n = int(round(window_ms * fs_hz / 1000.0))
    n = max(minimum, n)
    return min(n, n_samples)


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


# -----------------------------------------------------------------------------
# Frequency-domain computations
# -----------------------------------------------------------------------------

def compute_welch_psd(
    y: np.ndarray,
    fs_hz: float,
    window_ms: float,
    overlap_fraction: float,
    nfft_factor: float,
    detrend_mode: str,
    baseline_samples: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x = preprocess_trace(y, detrend_mode=detrend_mode, baseline_samples=baseline_samples)

    nperseg = samples_from_ms(window_ms, fs_hz, len(x), minimum=8)
    noverlap = int(round(nperseg * overlap_fraction))
    noverlap = max(0, min(noverlap, nperseg - 1))

    base_nfft = next_power_of_two(nperseg)
    nfft = int(max(nperseg, round(base_nfft * max(1.0, nfft_factor))))

    f, pxx = welch(
        x,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,  # already preprocessed explicitly
        scaling="density",
        return_onesided=True,
    )

    meta = {
        "nperseg": float(nperseg),
        "noverlap": float(noverlap),
        "nfft": float(nfft),
        "frequency_resolution_hz": float(fs_hz / nfft),
    }
    return f, pxx, meta


def restrict_frequency_range(
    f: np.ndarray,
    pxx: np.ndarray,
    fmin: float,
    fmax: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    nyquist = float(f[-1])
    upper = nyquist if fmax is None else min(float(fmax), nyquist)
    lower = max(0.0, float(fmin))

    if lower >= upper:
        raise ValueError(
            f"Invalid frequency range: fmin={lower:g} Hz, fmax={upper:g} Hz, "
            f"Nyquist={nyquist:g} Hz."
        )

    mask = (f >= lower) & (f <= upper)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Frequency range contains fewer than two PSD bins.")
    return f[mask], pxx[mask]


def integrate_power(f: np.ndarray, pxx: np.ndarray) -> float:
    if len(f) < 2:
        return 0.0
    return float(np.trapezoid(pxx, f))


def cumulative_power(f: np.ndarray, pxx: np.ndarray) -> np.ndarray:
    if len(f) == 0:
        return np.array([], dtype=float)
    if len(f) == 1:
        return np.array([0.0], dtype=float)

    df = np.diff(f)
    segment_area = 0.5 * (pxx[:-1] + pxx[1:]) * df
    return np.concatenate(([0.0], np.cumsum(segment_area)))


def frequency_at_cumulative_fraction(
    f: np.ndarray,
    pxx: np.ndarray,
    fraction: float,
) -> float:
    c = cumulative_power(f, pxx)
    if len(c) == 0 or c[-1] <= 0:
        return float("nan")
    target = fraction * c[-1]
    return float(np.interp(target, c, f))


def spectral_metrics(
    f: np.ndarray,
    pxx: np.ndarray,
    bands: OrderedDictType[str, Tuple[float, float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    total_power = integrate_power(f, pxx)
    out["total_power"] = total_power

    if total_power <= 0 or not np.isfinite(total_power):
        for key in [
            "peak_frequency_hz",
            "mean_frequency_hz",
            "median_frequency_hz",
            "spectral_bandwidth_hz",
            "frequency_95_hz",
            "spectral_entropy",
        ]:
            out[key] = float("nan")
        for label in bands:
            out[f"bandpower_{label}"] = float("nan")
            out[f"rel_bandpower_{label}"] = float("nan")
        return out

    peak_idx = int(np.argmax(pxx))
    peak_frequency = float(f[peak_idx])
    mean_frequency = float(np.trapezoid(f * pxx, f) / total_power)
    median_frequency = frequency_at_cumulative_fraction(f, pxx, 0.50)
    frequency_95 = frequency_at_cumulative_fraction(f, pxx, 0.95)

    variance = float(np.trapezoid(((f - mean_frequency) ** 2) * pxx, f) / total_power)
    spectral_bandwidth = math.sqrt(max(0.0, variance))

    # Entropy is calculated from discretized PSD-bin probability mass.
    p = np.maximum(pxx, 0.0)
    p_sum = float(np.sum(p))
    if p_sum > 0 and len(p) > 1:
        prob = p / p_sum
        entropy = -float(np.sum(prob * np.log2(prob + EPS))) / math.log2(len(prob))
    else:
        entropy = float("nan")

    out.update({
        "peak_frequency_hz": peak_frequency,
        "mean_frequency_hz": mean_frequency,
        "median_frequency_hz": median_frequency,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "frequency_95_hz": frequency_95,
        "spectral_entropy": entropy,
    })

    f_lo_available = float(f[0])
    f_hi_available = float(f[-1])

    for label, (lo, hi) in bands.items():
        lo_eff = max(float(lo), f_lo_available)
        hi_eff = min(float(hi), f_hi_available)
        if hi_eff <= lo_eff:
            bp = 0.0
        else:
            mask = (f >= lo_eff) & (f <= hi_eff)
            if np.count_nonzero(mask) >= 2:
                bp = integrate_power(f[mask], pxx[mask])
            else:
                bp = 0.0
        out[f"bandpower_{label}"] = bp
        out[f"rel_bandpower_{label}"] = bp / total_power if total_power > 0 else float("nan")

    return out


def parse_bands(text: str) -> "OrderedDict[str, Tuple[float, float]]":
    bands: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*", chunk)
        if not m:
            raise ValueError(
                f"Cannot parse band '{chunk}'. Expected format such as '20-50'."
            )
        lo = float(m.group(1))
        hi = float(m.group(2))
        if hi <= lo:
            raise ValueError(f"Band must have upper > lower: {chunk}")
        label = f"{lo:g}_{hi:g}hz"
        bands[label] = (lo, hi)

    if not bands:
        raise ValueError("No frequency bands were specified.")
    return bands


def normalized_psd(f: np.ndarray, pxx: np.ndarray) -> np.ndarray:
    area = integrate_power(f, pxx)
    if area <= 0:
        return np.zeros_like(pxx)
    return pxx / area


def psd_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        return float("nan")
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def jensen_shannon_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, bounded approximately [0, 1]."""
    pa = np.maximum(np.asarray(a, dtype=float), 0.0)
    pb = np.maximum(np.asarray(b, dtype=float), 0.0)
    sa = float(np.sum(pa))
    sb = float(np.sum(pb))
    if sa <= 0 or sb <= 0:
        return float("nan")
    pa /= sa
    pb /= sb
    m = 0.5 * (pa + pb)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / (q[mask] + EPS))))

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_psd_overlay(
    channel: int,
    n_points_xy: int,
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_file: Path,
    use_db: bool,
    dpi: int,
) -> None:
    grid_y = channel // n_points_xy
    grid_x = channel % n_points_xy

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for name, (f, pxx) in spectra.items():
        y = 10.0 * np.log10(np.maximum(pxx, EPS)) if use_db else pxx
        ax.plot(f, y, linewidth=1.35, label=name)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB re mV²/Hz]" if use_db else "PSD [mV²/Hz]")
    ax.set_title(f"Electrode {channel:03d} | grid y={grid_y}, x={grid_x}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_case_average_psd(
    avg_spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_file: Path,
    use_db: bool,
    normalize: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    for name, (f, pxx) in avg_spectra.items():
        plot_pxx = normalized_psd(f, pxx) if normalize else pxx
        if use_db:
            y = 10.0 * np.log10(np.maximum(plot_pxx, EPS))
        else:
            y = plot_pxx
        ax.plot(f, y, linewidth=1.6, label=name)

    ax.set_xlabel("Frequency [Hz]")
    if normalize:
        ax.set_ylabel("Normalized PSD [1/Hz]" if not use_db else "Normalized PSD [dB]")
        ax.set_title("Case-average normalized spectral shape")
    else:
        ax.set_ylabel("PSD [mV²/Hz]" if not use_db else "PSD [dB re mV²/Hz]")
        ax.set_title("Case-average electrode PSD")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_metric_heatmap(
    values: np.ndarray,
    n_points_xy: int,
    n_points_z: int,
    title: str,
    colorbar_label: str,
    out_file: Path,
    dpi: int,
) -> None:
    expected = n_points_xy * n_points_z
    if len(values) != expected:
        return

    grid = np.asarray(values, dtype=float).reshape(n_points_z, n_points_xy)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    im = ax.imshow(grid, origin="lower", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_xlabel("Electrode grid x")
    ax.set_ylabel("Electrode grid y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_spectrogram_one_case(
    t_ms: np.ndarray,
    y: np.ndarray,
    fs_hz: float,
    channel: int,
    n_points_xy: int,
    case_name: str,
    out_file: Path,
    fmin: float,
    fmax: Optional[float],
    window_ms: float,
    overlap_fraction: float,
    detrend_mode: str,
    baseline_samples: int,
    dpi: int,
) -> None:
    x = preprocess_trace(y, detrend_mode=detrend_mode, baseline_samples=baseline_samples)
    nperseg = samples_from_ms(window_ms, fs_hz, len(x), minimum=8)
    noverlap = int(round(nperseg * overlap_fraction))
    noverlap = max(0, min(noverlap, nperseg - 1))
    nfft = next_power_of_two(nperseg)

    f, t_s, sxx = spectrogram(
        x,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
        mode="psd",
    )

    nyquist = fs_hz / 2.0
    upper = nyquist if fmax is None else min(float(fmax), nyquist)
    mask = (f >= max(0.0, fmin)) & (f <= upper)
    f_plot = f[mask]
    s_plot = sxx[mask, :]

    # scipy returns segment-center times relative to signal start.
    time_plot_ms = float(t_ms[0]) + 1000.0 * t_s
    s_db = 10.0 * np.log10(np.maximum(s_plot, EPS))

    grid_y = channel // n_points_xy
    grid_x = channel % n_points_xy

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    mesh = ax.pcolormesh(time_plot_ms, f_plot, s_db, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("PSD [dB re mV²/Hz]")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(
        f"Spectrogram: {case_name} | electrode {channel:03d} "
        f"(y={grid_y}, x={grid_x})"
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CSV helpers and summaries
# -----------------------------------------------------------------------------

def write_rows_csv(rows: Sequence[Dict[str, object]], out_file: Path) -> None:
    if not rows:
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_case_metrics(metric_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not metric_rows:
        return []

    case_names = sorted({str(r["case"]) for r in metric_rows})
    excluded = {
        "electrode", "grid_y", "grid_x", "case", "x", "y", "z",
        "psd_corr_vs_healthy", "psd_js_divergence_vs_healthy",
    }

    numeric_keys = []
    for key in metric_rows[0].keys():
        if key in excluded:
            continue
        try:
            float(metric_rows[0][key])
            numeric_keys.append(key)
        except (TypeError, ValueError):
            pass

    rows: List[Dict[str, object]] = []
    for case in case_names:
        case_rows = [r for r in metric_rows if str(r["case"]) == case]
        summary: Dict[str, object] = {"case": case, "n_electrodes": len(case_rows)}
        for key in numeric_keys:
            vals = np.asarray([float(r[key]) for r in case_rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                summary[f"{key}_mean"] = float("nan")
                summary[f"{key}_std"] = float("nan")
                summary[f"{key}_median"] = float("nan")
            else:
                summary[f"{key}_mean"] = float(np.mean(finite))
                summary[f"{key}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
                summary[f"{key}_median"] = float(np.median(finite))
        rows.append(summary)
    return rows


def build_vs_healthy_rows(
    metric_rows: Sequence[Dict[str, object]],
    healthy_name: str = "healthy",
) -> List[Dict[str, object]]:
    by_key = {(int(r["electrode"]), str(r["case"])): r for r in metric_rows}

    metrics = [
        "peak_frequency_hz",
        "mean_frequency_hz",
        "median_frequency_hz",
        "spectral_bandwidth_hz",
        "frequency_95_hz",
        "total_power",
        "spectral_entropy",
    ]

    out: List[Dict[str, object]] = []
    for r in metric_rows:
        case = str(r["case"])
        if case == healthy_name:
            continue
        ch = int(r["electrode"])
        ref = by_key.get((ch, healthy_name))
        if ref is None:
            continue

        row: Dict[str, object] = {
            "electrode": ch,
            "grid_y": r["grid_y"],
            "grid_x": r["grid_x"],
            "case": case,
        }
        for metric in metrics:
            val = float(r[metric])
            ref_val = float(ref[metric])
            row[f"delta_{metric}"] = val - ref_val
            row[f"ratio_{metric}"] = val / ref_val if ref_val != 0 else float("nan")

        row["psd_corr_vs_healthy"] = r.get("psd_corr_vs_healthy", float("nan"))
        row["psd_js_divergence_vs_healthy"] = r.get(
            "psd_js_divergence_vs_healthy", float("nan")
        )
        out.append(row)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Frequency-domain and time-frequency analysis of OpenDiHu EMG "
            "electrode signals across four remodeling cases."
        )
    )
    parser.add_argument(
        "--base-dir",
        default="build_release/out",
        help="Folder containing healthy/death_25/death_50/death_75 subfolders.",
    )
    parser.add_argument("--csv-name", default="electrodes.csv")
    parser.add_argument(
        "--out-dir",
        default="emg_frequency_domain",
        help="Output folder.",
    )
    parser.add_argument("--max-electrodes", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=160)

    parser.add_argument(
        "--fmin",
        type=float,
        default=0.0,
        help="Minimum frequency included in plots and spectral metrics [Hz].",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Maximum frequency [Hz]. Default: Nyquist frequency.",
    )
    parser.add_argument(
        "--bands",
        default=DEFAULT_BANDS,
        help=f"Comma-separated power bands; default: {DEFAULT_BANDS}",
    )

    parser.add_argument(
        "--detrend",
        choices=["none", "constant", "linear"],
        default="constant",
        help="Preprocessing before PSD. Default removes DC mean.",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=0,
        help=(
            "Optional subtraction of the mean of the first N samples before detrending. "
            "Default 0; generally unnecessary when --detrend constant is used."
        ),
    )

    parser.add_argument(
        "--welch-window-ms",
        type=float,
        default=250.0,
        help="Welch segment length in milliseconds.",
    )
    parser.add_argument(
        "--welch-overlap",
        type=float,
        default=0.50,
        help="Welch segment overlap fraction in [0, 1).",
    )
    parser.add_argument(
        "--nfft-factor",
        type=float,
        default=1.0,
        help="Zero-padding multiplier applied to the next power-of-two NFFT.",
    )
    parser.add_argument(
        "--linear-psd",
        action="store_true",
        help="Plot PSD on a linear scale instead of dB.",
    )

    parser.add_argument(
        "--spectrogram",
        action="store_true",
        help="Also create time-frequency spectrograms for each processed electrode/case.",
    )
    parser.add_argument(
        "--spectrogram-window-ms",
        type=float,
        default=100.0,
        help="Spectrogram window length in milliseconds.",
    )
    parser.add_argument(
        "--spectrogram-overlap",
        type=float,
        default=0.75,
        help="Spectrogram overlap fraction in [0, 1).",
    )

    parser.add_argument(
        "--heatmaps",
        action="store_true",
        help=(
            "Create electrode-grid heatmaps for peak/mean/median frequency and total power. "
            "Heatmaps require all electrodes to be processed."
        ),
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Create a multi-page PDF containing all per-electrode PSD overlays.",
    )
    parser.add_argument(
        "--export-psd-csv",
        action="store_true",
        help="Export the full PSD curve for every electrode/case to CSV (can be large).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.welch_overlap < 1.0):
        parser.error("--welch-overlap must be in [0, 1).")
    if not (0.0 <= args.spectrogram_overlap < 1.0):
        parser.error("--spectrogram-overlap must be in [0, 1).")
    if args.welch_window_ms <= 0 or args.spectrogram_window_ms <= 0:
        parser.error("Window lengths must be positive.")
    if args.nfft_factor < 1.0:
        parser.error("--nfft-factor must be >= 1.0.")

    bands = parse_bands(args.bands)

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    psd_dir = out_dir / "psd_by_electrode"
    spec_dir = out_dir / "spectrograms"
    heatmap_dir = out_dir / "metric_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    psd_dir.mkdir(parents=True, exist_ok=True)
    if args.spectrogram:
        spec_dir.mkdir(parents=True, exist_ok=True)
    if args.heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    case_paths = {
        label: base_dir / folder / args.csv_name
        for label, folder in DEFAULT_CASES
    }

    print("Loading cases:")
    cases: Dict[str, Dict[str, np.ndarray]] = {}
    for label, filename in case_paths.items():
        print(f"  {label}: {filename}")
        cases[label] = read_electrodes_csv(filename)

    t_ms, n_points, n_points_xy, n_points_z = ensure_same_layout(cases)
    fs_hz, dt_ms, time_jitter = infer_sampling_frequency(t_ms)
    nyquist = fs_hz / 2.0
    effective_fmax = nyquist if args.fmax is None else min(float(args.fmax), nyquist)

    n_process = (
        min(n_points, args.max_electrodes)
        if args.max_electrodes is not None
        else n_points
    )

    print(f"Detected electrode grid: {n_points_xy} x {n_points_z} = {n_points} electrodes")
    print(f"Time range: [{t_ms[0]}, {t_ms[-1]}] ms, n_time_steps={len(t_ms)}")
    print(f"Sampling interval: {dt_ms:.9g} ms")
    print(f"Sampling frequency: {fs_hz:.6g} Hz")
    print(f"Nyquist frequency: {nyquist:.6g} Hz")
    print(f"Maximum relative time-step jitter: {time_jitter:.3e}")
    print(f"Analysis frequency range: {args.fmin:g} to {effective_fmax:g} Hz")
    if args.fmax is not None and args.fmax > nyquist:
        print(
            f"WARNING: requested fmax={args.fmax:g} Hz exceeds Nyquist; "
            f"clipped to {nyquist:g} Hz."
        )

    sampling_rows = [{
        "n_time_samples": len(t_ms),
        "time_start_ms": float(t_ms[0]),
        "time_end_ms": float(t_ms[-1]),
        "duration_ms": float(t_ms[-1] - t_ms[0]),
        "dt_median_ms": dt_ms,
        "sampling_frequency_hz": fs_hz,
        "nyquist_frequency_hz": nyquist,
        "max_relative_time_step_jitter": time_jitter,
        "analysis_fmin_hz": float(args.fmin),
        "analysis_fmax_hz": effective_fmax,
        "welch_window_ms": float(args.welch_window_ms),
        "welch_overlap_fraction": float(args.welch_overlap),
        "detrend": args.detrend,
        "baseline_samples": int(args.baseline_samples),
    }]
    write_rows_csv(sampling_rows, out_dir / "sampling_and_analysis_settings.csv")

    metric_rows: List[Dict[str, object]] = []
    psd_export_rows: List[Dict[str, object]] = []
    all_spectra_by_case: Dict[str, List[np.ndarray]] = {name: [] for name in cases}
    common_f: Optional[np.ndarray] = None
    welch_meta_printed = False

    pdf_pages = PdfPages(out_dir / "all_electrodes_psd_overlay.pdf") if args.pdf else None

    positions = cases["healthy"].get("positions")

    for ch in range(n_process):
        spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for case_name, case in cases.items():
            y = case["emg"][:, ch]
            f_full, pxx_full, welch_meta = compute_welch_psd(
                y=y,
                fs_hz=fs_hz,
                window_ms=args.welch_window_ms,
                overlap_fraction=args.welch_overlap,
                nfft_factor=args.nfft_factor,
                detrend_mode=args.detrend,
                baseline_samples=args.baseline_samples,
            )
            f, pxx = restrict_frequency_range(
                f_full, pxx_full, fmin=args.fmin, fmax=args.fmax
            )

            if not welch_meta_printed:
                print(
                    "Welch settings: "
                    f"nperseg={int(welch_meta['nperseg'])}, "
                    f"noverlap={int(welch_meta['noverlap'])}, "
                    f"nfft={int(welch_meta['nfft'])}, "
                    f"df={welch_meta['frequency_resolution_hz']:.6g} Hz"
                )
                welch_meta_printed = True

            if common_f is None:
                common_f = f
            elif len(common_f) != len(f) or not np.allclose(common_f, f):
                raise RuntimeError("Unexpected inconsistent frequency vector across traces.")

            spectra[case_name] = (f, pxx)
            all_spectra_by_case[case_name].append(pxx)

            metrics = spectral_metrics(f, pxx, bands)
            row: Dict[str, object] = {
                "electrode": ch,
                "grid_y": ch // n_points_xy,
                "grid_x": ch % n_points_xy,
                "case": case_name,
            }

            if positions is not None and len(positions) > ch:
                row["x"] = float(positions[ch, 0])
                row["y"] = float(positions[ch, 1])
                row["z"] = float(positions[ch, 2])

            row.update(metrics)
            metric_rows.append(row)

            if args.export_psd_csv:
                for freq, power in zip(f, pxx):
                    psd_export_rows.append({
                        "electrode": ch,
                        "grid_y": ch // n_points_xy,
                        "grid_x": ch % n_points_xy,
                        "case": case_name,
                        "frequency_hz": float(freq),
                        "psd_mv2_per_hz": float(power),
                    })

            if args.spectrogram:
                spec_file = (
                    spec_dir
                    / f"{case_name}_electrode_{ch:03d}_y{ch // n_points_xy:02d}_x{ch % n_points_xy:02d}.png"
                )
                plot_spectrogram_one_case(
                    t_ms=t_ms,
                    y=y,
                    fs_hz=fs_hz,
                    channel=ch,
                    n_points_xy=n_points_xy,
                    case_name=case_name,
                    out_file=spec_file,
                    fmin=args.fmin,
                    fmax=args.fmax,
                    window_ms=args.spectrogram_window_ms,
                    overlap_fraction=args.spectrogram_overlap,
                    detrend_mode=args.detrend,
                    baseline_samples=args.baseline_samples,
                    dpi=args.dpi,
                )

        # Shape similarity against healthy is calculated from normalized PSD.
        healthy_pxx = spectra["healthy"][1]
        healthy_norm = normalized_psd(spectra["healthy"][0], healthy_pxx)
        for row in metric_rows[-len(cases):]:
            case_name = str(row["case"])
            if case_name == "healthy":
                row["psd_corr_vs_healthy"] = 1.0
                row["psd_js_divergence_vs_healthy"] = 0.0
            else:
                case_norm = normalized_psd(spectra[case_name][0], spectra[case_name][1])
                row["psd_corr_vs_healthy"] = psd_correlation(healthy_norm, case_norm)
                row["psd_js_divergence_vs_healthy"] = jensen_shannon_divergence(
                    healthy_norm, case_norm
                )

        psd_file = (
            psd_dir
            / f"electrode_{ch:03d}_y{ch // n_points_xy:02d}_x{ch % n_points_xy:02d}.png"
        )
        plot_psd_overlay(
            channel=ch,
            n_points_xy=n_points_xy,
            spectra=spectra,
            out_file=psd_file,
            use_db=not args.linear_psd,
            dpi=args.dpi,
        )

        if pdf_pages is not None:
            fig, ax = plt.subplots(figsize=(8.4, 4.8))
            for name, (f, pxx) in spectra.items():
                y_plot = (
                    pxx
                    if args.linear_psd
                    else 10.0 * np.log10(np.maximum(pxx, EPS))
                )
                ax.plot(f, y_plot, linewidth=1.35, label=name)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel(
                "PSD [mV²/Hz]" if args.linear_psd else "PSD [dB re mV²/Hz]"
            )
            ax.set_title(
                f"Electrode {ch:03d} | grid y={ch // n_points_xy}, "
                f"x={ch % n_points_xy}"
            )
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)

        if (ch + 1) % 10 == 0 or ch + 1 == n_process:
            extra = " + spectrograms" if args.spectrogram else ""
            print(f"  processed {ch + 1}/{n_process} electrodes (PSD{extra})")

    if pdf_pages is not None:
        pdf_pages.close()
        print(f"Created {out_dir / 'all_electrodes_psd_overlay.pdf'}")

    # Case-average spectra.
    if common_f is None:
        raise RuntimeError("No spectra were computed.")

    avg_spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for case_name, spectra_list in all_spectra_by_case.items():
        matrix = np.vstack(spectra_list)
        avg_spectra[case_name] = (common_f, np.mean(matrix, axis=0))

    plot_case_average_psd(
        avg_spectra,
        out_dir / "case_average_psd.png",
        use_db=not args.linear_psd,
        normalize=False,
        dpi=args.dpi,
    )
    plot_case_average_psd(
        avg_spectra,
        out_dir / "case_average_normalized_psd.png",
        use_db=False,
        normalize=True,
        dpi=args.dpi,
    )

    # Average PSD values are useful for downstream plotting/statistics.
    avg_psd_rows: List[Dict[str, object]] = []
    for i, freq in enumerate(common_f):
        row: Dict[str, object] = {"frequency_hz": float(freq)}
        for case_name, (_, avg_pxx) in avg_spectra.items():
            row[f"{case_name}_mean_psd_mv2_per_hz"] = float(avg_pxx[i])
        avg_psd_rows.append(row)
    write_rows_csv(avg_psd_rows, out_dir / "case_average_psd.csv")

    # Core metrics and summaries.
    write_rows_csv(metric_rows, out_dir / "spectral_metrics_by_electrode.csv")
    summary_rows = summarize_case_metrics(metric_rows)
    write_rows_csv(summary_rows, out_dir / "spectral_metrics_summary_by_case.csv")
    vs_healthy_rows = build_vs_healthy_rows(metric_rows)
    write_rows_csv(vs_healthy_rows, out_dir / "spectral_changes_vs_healthy.csv")

    if args.export_psd_csv:
        write_rows_csv(psd_export_rows, out_dir / "all_psd_curves.csv")

    # Spatial metric maps only make sense if all electrodes are present.
    if args.heatmaps:
        if n_process != n_points:
            print(
                "WARNING: --heatmaps requested together with --max-electrodes. "
                "Heatmaps skipped because the complete grid was not processed."
            )
        else:
            heatmap_specs = [
                ("peak_frequency_hz", "Peak frequency", "Frequency [Hz]"),
                ("mean_frequency_hz", "Mean frequency", "Frequency [Hz]"),
                ("median_frequency_hz", "Median frequency", "Frequency [Hz]"),
                ("total_power", "Total spectral power", "Power [mV²]"),
            ]
            for case_name in cases:
                case_rows = [r for r in metric_rows if str(r["case"]) == case_name]
                case_rows = sorted(case_rows, key=lambda r: int(r["electrode"]))
                for key, label, cbar_label in heatmap_specs:
                    vals = np.asarray([float(r[key]) for r in case_rows], dtype=float)
                    plot_metric_heatmap(
                        values=vals,
                        n_points_xy=n_points_xy,
                        n_points_z=n_points_z,
                        title=f"{case_name}: {label}",
                        colorbar_label=cbar_label,
                        out_file=heatmap_dir / f"{case_name}_{key}.png",
                        dpi=args.dpi,
                    )

    print("\nCreated frequency-domain outputs:")
    print(f"  {out_dir / 'sampling_and_analysis_settings.csv'}")
    print(f"  {out_dir / 'spectral_metrics_by_electrode.csv'}")
    print(f"  {out_dir / 'spectral_metrics_summary_by_case.csv'}")
    print(f"  {out_dir / 'spectral_changes_vs_healthy.csv'}")
    print(f"  {out_dir / 'case_average_psd.csv'}")
    print(f"  {out_dir / 'case_average_psd.png'}")
    print(f"  {out_dir / 'case_average_normalized_psd.png'}")
    print(f"  per-electrode PSD plots: {psd_dir}")
    if args.spectrogram:
        print(f"  spectrograms: {spec_dir}")
    if args.heatmaps and n_process == n_points:
        print(f"  metric heatmaps: {heatmap_dir}")
    if args.export_psd_csv:
        print(f"  {out_dir / 'all_psd_curves.csv'}")


if __name__ == "__main__":
    main()
