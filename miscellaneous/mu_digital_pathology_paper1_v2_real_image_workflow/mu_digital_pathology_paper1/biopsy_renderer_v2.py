#!/usr/bin/env python3
"""Biopsy-like renderer for mu_digital_pathology_v2 (type I / IIa / IIx)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from mu_digital_pathology_v2 import MuscleState, subtype_map, hybrid_mask_from_value


@dataclass
class RenderParams:
    scale: int = 8
    jitter: float = 0.28
    candidate_radius: int = 2
    boundary_strength: float = 0.90
    boundary_width: int = 1
    noise_strength: float = 0.04
    seed: int = 123


@dataclass
class RenderResult:
    rgb: np.ndarray
    labels: np.ndarray
    seed_xy: np.ndarray
    seed_weight: np.ndarray


CANONICAL_COLOR = {
    -1: np.array([0.55, 0.55, 0.58]),   # denervated
     0: np.array([0.10, 0.10, 0.10]),   # type I dark
     1: np.array([0.55, 0.55, 0.58]),   # type IIa medium gray
     2: np.array([0.92, 0.92, 0.90]),   # type IIx very light
}



def _fiber_color_lut(state: MuscleState, rng: np.random.Generator) -> np.ndarray:
    n = state.mu_id.shape[0]
    n_fib = n * n
    labels = np.arange(n_fib)
    ii = labels // n
    jj = labels % n
    smap = subtype_map(state)
    rgb = np.zeros((n_fib, 3), dtype=float)
    delta = rng.normal(0.0, 1.0, size=n_fib)
    hybrid_mask = hybrid_mask_from_value(state.fiber_type)

    for idx, (i, j) in enumerate(zip(ii, jj)):
        subtype = int(smap[i, j])
        csa = float(state.csa[i, j])

        if subtype < 0:
            inten = np.clip(0.56 + 0.08 * delta[idx], 0.38, 0.72)
            base = np.array([inten, inten, inten + 0.01])
        else:
            base = CANONICAL_COLOR[subtype].copy()
            # Hybrids blend toward neighboring subtype anchors.
            if hybrid_mask[i, j]:
                val = float(state.fiber_type[i, j])
                if val < 0.30:
                    neighbor = CANONICAL_COLOR[1]
                    alpha = val / 0.30
                    base = (1.0 - alpha) * CANONICAL_COLOR[0] + alpha * neighbor
                elif val < 0.80:
                    a = (val - 0.30) / 0.50
                    # within IIa range but away from center, softly blend toward I or IIx.
                    if val < 0.60:
                        base = (1.0 - a) * CANONICAL_COLOR[0] + a * CANONICAL_COLOR[1]
                    else:
                        a2 = (val - 0.60) / 0.20
                        base = (1.0 - a2) * CANONICAL_COLOR[1] + a2 * CANONICAL_COLOR[2]
                else:
                    a = (val - 0.80) / 0.20
                    base = (1.0 - a) * CANONICAL_COLOR[1] + a * CANONICAL_COLOR[2]
            base = np.clip(base + 0.03 * delta[idx], 0.05, 0.98)

        if csa < 0.65:
            base = 0.80 * base + 0.20 * np.array([0.44, 0.42, 0.50])
        elif csa > 1.25:
            base = np.clip(base * 1.03 + 0.02, 0, 1)

        rgb[idx] = np.clip(base, 0.0, 1.0)
    return rgb



def render_biopsy_like_image(state: MuscleState, params: Optional[RenderParams] = None) -> RenderResult:
    if params is None:
        params = RenderParams()
    rng = np.random.default_rng(params.seed)
    n = state.mu_id.shape[0]
    scale = params.scale
    H = W = n * scale

    gy, gx = np.mgrid[0:n, 0:n]
    cx = gx.astype(float) + 0.5 + rng.uniform(-params.jitter, params.jitter, size=(n, n))
    cy = gy.astype(float) + 0.5 + rng.uniform(-params.jitter, params.jitter, size=(n, n))
    cx = np.clip(cx, gx + 0.10, gx + 0.90)
    cy = np.clip(cy, gy + 0.10, gy + 0.90)
    seed_xy = np.column_stack([cx.ravel(), cy.ravel()])

    weight = np.sqrt(np.clip(state.csa.ravel(), 0.08, 3.0))
    weight[state.mu_id.ravel() < 0] *= 0.88
    seed_weight = weight

    labels = np.empty((H, W), dtype=np.int32)
    local_y = (np.arange(scale) + 0.5) / scale
    local_x = (np.arange(scale) + 0.5) / scale
    lyy, lxx = np.meshgrid(local_y, local_x, indexing="ij")

    for i in range(n):
        for j in range(n):
            pi0, pi1 = i * scale, (i + 1) * scale
            pj0, pj1 = j * scale, (j + 1) * scale
            ic0 = max(0, i - params.candidate_radius)
            ic1 = min(n, i + params.candidate_radius + 1)
            jc0 = max(0, j - params.candidate_radius)
            jc1 = min(n, j + params.candidate_radius + 1)
            cand_ii, cand_jj = np.mgrid[ic0:ic1, jc0:jc1]
            cand_ii = cand_ii.ravel()
            cand_jj = cand_jj.ravel()
            cand_idx = cand_ii * n + cand_jj
            px = j + lxx[..., None]
            py = i + lyy[..., None]
            sx = seed_xy[cand_idx, 0][None, None, :]
            sy = seed_xy[cand_idx, 1][None, None, :]
            ww = seed_weight[cand_idx][None, None, :]
            d2 = (px - sx) ** 2 + (py - sy) ** 2
            score = d2 / (ww ** 2 + 1e-8)
            best = np.argmin(score, axis=2)
            labels[pi0:pi1, pj0:pj1] = cand_idx[best]

    lut = _fiber_color_lut(state, rng)
    rgb = lut[labels]
    rgb = np.clip(rgb + rng.normal(0.0, params.noise_strength, size=rgb.shape), 0.0, 1.0)

    boundary = np.zeros((H, W), dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    if params.boundary_width > 1:
        for _ in range(params.boundary_width - 1):
            b2 = boundary.copy()
            b2[:-1, :] |= boundary[1:, :]
            b2[1:, :] |= boundary[:-1, :]
            b2[:, :-1] |= boundary[:, 1:]
            b2[:, 1:] |= boundary[:, :-1]
            boundary = b2
    bound_rgb = np.array([0.46, 0.39, 0.52])
    alpha = params.boundary_strength * boundary[..., None].astype(float)
    rgb = np.clip((1.0 - alpha) * rgb + alpha * bound_rgb, 0.0, 1.0)
    return RenderResult(rgb=rgb, labels=labels, seed_xy=seed_xy, seed_weight=seed_weight)



def plot_snapshot_biopsy_panel(snapshot_states: Dict[float, MuscleState], out_file: str, title: str = "Biopsy-like remodeling") -> None:
    keys = sorted(snapshot_states.keys())
    fig, axes = plt.subplots(3, len(keys), figsize=(3.3 * len(keys), 9.0), constrained_layout=True)
    if len(keys) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    for col, frac in enumerate(keys):
        st = snapshot_states[frac]
        tm = subtype_map(st)
        coarse_rgb = np.zeros((*tm.shape, 3), dtype=float)
        for label, color in CANONICAL_COLOR.items():
            coarse_rgb[tm == label] = color
        axes[0, col].imshow(coarse_rgb, interpolation="nearest")
        axes[0, col].set_title(f"{int(round(100*frac))}% MU death\ncoarse subtype map")
        axes[0, col].axis("off")
        rr = render_biopsy_like_image(st, RenderParams(scale=8, jitter=0.30, seed=100 + col))
        axes[1, col].imshow(rr.rgb, interpolation="nearest")
        axes[1, col].set_title("synthetic ATPase-like biopsy")
        axes[1, col].axis("off")
        axes[2, col].imshow(st.csa, cmap="viridis", vmin=0.2, vmax=2.0, interpolation="nearest")
        axes[2, col].set_title("fiber CSA")
        axes[2, col].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_file, dpi=220)
    plt.close(fig)



def plot_scenario_biopsy_comparison(states: Dict[str, MuscleState], out_file: str, title: str = "Scenario comparison") -> None:
    names = list(states.keys())
    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 7.5), constrained_layout=True)
    if len(names) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, name in enumerate(names):
        st = states[name]
        rr = render_biopsy_like_image(st, RenderParams(scale=8, jitter=0.30, seed=1000 + col))
        axes[0, col].imshow(rr.rgb, interpolation="nearest")
        axes[0, col].set_title(name)
        axes[0, col].axis("off")
        tm = subtype_map(st)
        coarse_rgb = np.zeros((*tm.shape, 3), dtype=float)
        for label, color in CANONICAL_COLOR.items():
            coarse_rgb[tm == label] = color
        axes[1, col].imshow(coarse_rgb, interpolation="nearest")
        axes[1, col].set_title("coarse subtype map")
        axes[1, col].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
