#!/usr/bin/env python3
"""
Feature extraction from a real or synthetic muscle-biopsy image.

Workflow:
    biopsy image -> fiber segmentation -> subtype classification
    -> spatial/morphometric features -> target CSV for ABC inference.

This is a research baseline, not a universal pathology segmentation model.
Ordinary ATPase-like images usually cannot reveal hidden remodeling parameters
or true denervation fraction directly. The output should be interpreted as
observable image features, not ground truth remodeling parameters.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay, cKDTree
from scipy import ndimage as ndi

from skimage import io, color, exposure, filters, morphology, measure, segmentation, util, transform
from skimage.feature import peak_local_max


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def robust_float_image(img: np.ndarray) -> np.ndarray:
    """Return RGB float image in [0,1]."""
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = util.img_as_float32(img)
    return np.clip(img, 0.0, 1.0)


def maybe_resize(img: np.ndarray, max_side: int = 1400) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    return transform.resize(
        img,
        (int(round(h * scale)), int(round(w * scale))),
        preserve_range=True,
        anti_aliasing=True,
    )


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    vals = np.unique(labels)
    vals = vals[vals > 0]
    for new, old in enumerate(vals, start=1):
        out[labels == old] = new
    return out


def simple_kmeans_1d(x: np.ndarray, k: int = 3, n_iter: int = 80, seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Small dependency-free 1D k-means. Returns sorted cluster labels and centers."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int), np.array([])
    if x.size < k:
        centers = np.quantile(x, np.linspace(0, 1, min(k, x.size)))
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        return labels, centers
    centers = np.quantile(x, np.linspace(0.1, 0.9, k))
    centers = centers + rng.normal(0, 1e-6, size=k)
    for _ in range(n_iter):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new = centers.copy()
        for c in range(k):
            if np.any(labels == c):
                new[c] = np.mean(x[labels == c])
        if np.allclose(new, centers):
            break
        centers = new
    order = np.argsort(centers)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[labels].astype(int), centers[order]


def graph_connected_components(n_nodes: int, edges: np.ndarray, node_mask: Optional[np.ndarray] = None) -> List[List[int]]:
    if node_mask is None:
        active = np.ones(n_nodes, dtype=bool)
    else:
        active = node_mask.astype(bool)
    adj = [[] for _ in range(n_nodes)]
    for a, b in edges:
        if active[a] and active[b]:
            adj[a].append(b)
            adj[b].append(a)
    seen = np.zeros(n_nodes, dtype=bool)
    comps: List[List[int]] = []
    for i in range(n_nodes):
        if not active[i] or seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def gini(x: np.ndarray) -> float:
    x = np.sort(np.maximum(np.asarray(x, dtype=float), 0))
    if x.size == 0 or np.allclose(x.sum(), 0):
        return 0.0
    return float((2 * np.arange(1, len(x) + 1) @ x) / (len(x) * x.sum()) - (len(x) + 1) / len(x))


# -----------------------------------------------------------------------------
# Segmentation and classification
# -----------------------------------------------------------------------------

@dataclass
class ExtractorConfig:
    max_side: int = 1400
    min_area: int = 60
    max_area: int = 8000
    min_distance: int = 6
    tissue_threshold: float = 0.96
    n_subtypes: int = 3
    denervated_fraction_assumed: float = 0.0
    seed: int = 1


def build_tissue_mask(rgb: np.ndarray, cfg: ExtractorConfig) -> np.ndarray:
    """Estimate tissue region and remove white background."""
    hsv = color.rgb2hsv(rgb)
    gray = color.rgb2gray(rgb)
    sat = hsv[..., 1]
    mask = (gray < cfg.tissue_threshold) | (sat > 0.08)
    mask = morphology.remove_small_objects(mask, 200)
    mask = morphology.remove_small_holes(mask, 200)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    return mask


def segment_fibers(rgb: np.ndarray, tissue_mask: np.ndarray, cfg: ExtractorConfig) -> np.ndarray:
    """Marker-based watershed segmentation of fiber-like regions.

    The seed image is based on distance from high-gradient/high-saturation
    boundary-like pixels, rather than distance from the entire tissue mask. This
    is more suitable for densely packed muscle fibers where the whole field is
    tissue and individual fibers are separated by endomysial boundaries.
    """
    gray = color.rgb2gray(rgb)
    hsv = color.rgb2hsv(rgb)
    sat = hsv[..., 1]
    gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.02)

    grad = filters.sobel(gray_eq)
    grad = filters.gaussian(grad, sigma=0.8)

    # Boundary score: strong intensity gradients plus colored/dark connective
    # tissue boundaries. Percentile threshold is image-adaptive.
    score = grad + 0.35 * sat + 0.10 * (1.0 - gray)
    vals = score[tissue_mask]
    if vals.size == 0:
        return np.zeros_like(gray, dtype=np.int32)
    boundary_thr = np.quantile(vals, 0.72)
    boundary_like = tissue_mask & (score >= boundary_thr)
    boundary_like = morphology.binary_dilation(boundary_like, morphology.disk(1))

    interior = tissue_mask & (~boundary_like)
    interior = morphology.remove_small_objects(interior, max(4, cfg.min_area // 4))
    dist = ndi.distance_transform_edt(interior)

    coords = peak_local_max(
        dist,
        min_distance=cfg.min_distance,
        labels=interior.astype(np.uint8),
        exclude_border=False,
    )
    markers = np.zeros_like(gray, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    if markers.max() == 0:
        # Last-resort fallback: connected components of interior.
        labels = measure.label(interior)
    else:
        labels = segmentation.watershed(score, markers=markers, mask=tissue_mask)

    props = measure.regionprops(labels)
    keep = np.zeros(labels.max() + 1, dtype=bool)
    for p in props:
        if cfg.min_area <= p.area <= cfg.max_area:
            keep[p.label] = True
    clean = labels.copy()
    clean[~keep[labels]] = 0
    return relabel_consecutive(clean)


def extract_fiber_table(rgb: np.ndarray, labels: np.ndarray, cfg: ExtractorConfig) -> pd.DataFrame:
    gray = color.rgb2gray(rgb)
    hsv = color.rgb2hsv(rgb)
    props = measure.regionprops(labels, intensity_image=gray)
    rows = []
    for p in props:
        lab = p.label
        mask = labels == lab
        if p.area <= 0:
            continue
        rows.append({
            "fiber_id": int(lab),
            "area_px": float(p.area),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
            "mean_gray": float(gray[mask].mean()),
            "mean_r": float(rgb[..., 0][mask].mean()),
            "mean_g": float(rgb[..., 1][mask].mean()),
            "mean_b": float(rgb[..., 2][mask].mean()),
            "mean_hue": float(hsv[..., 0][mask].mean()),
            "mean_sat": float(hsv[..., 1][mask].mean()),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
            "perimeter": float(p.perimeter),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["csa_norm"] = df["area_px"] / max(df["area_px"].median(), 1e-9)

    n_types = int(cfg.n_subtypes)
    labels_k, centers = simple_kmeans_1d(df["mean_gray"].values, k=n_types, seed=cfg.seed)
    if n_types == 2:
        subtype = np.where(labels_k == 0, 0, 2)
        phenotype = np.where(labels_k == 0, 0.0, 1.0)
    else:
        subtype = labels_k
        phenotype = np.choose(labels_k, [0.0, 0.60, 1.0])
    df["subtype"] = subtype.astype(int)
    df["phenotype_value"] = phenotype.astype(float)

    x = df["mean_gray"].values
    centers = np.asarray(centers)
    d = np.abs(x[:, None] - centers[None, :])
    order = np.argsort(d, axis=1)
    d0 = d[np.arange(len(x)), order[:, 0]]
    d1 = d[np.arange(len(x)), order[:, 1]] if centers.size > 1 else np.full_like(d0, np.inf)
    ambiguity = d0 / (d1 + 1e-9)
    df["hybrid_like"] = ambiguity > 0.65
    return df


def build_neighbor_edges(fibers: pd.DataFrame, max_edge_len_factor: float = 2.8) -> np.ndarray:
    pts = fibers[["centroid_x", "centroid_y"]].values
    n = len(pts)
    if n < 3:
        return np.empty((0, 2), dtype=int)
    edges = set()
    try:
        tri = Delaunay(pts)
        for simplex in tri.simplices:
            for a, b in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                if a > b:
                    a, b = b, a
                edges.add((int(a), int(b)))
    except Exception:
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=min(7, n))
        for i in range(n):
            for j in idx[i, 1:]:
                a, b = sorted((i, int(j)))
                edges.add((a, b))
    arr = np.array(sorted(edges), dtype=int)
    if arr.size == 0:
        return arr.reshape(0, 2)
    d = np.linalg.norm(pts[arr[:, 0]] - pts[arr[:, 1]], axis=1)
    med_area = np.median(fibers["area_px"].values)
    nominal_diam = 2.0 * math.sqrt(max(med_area, 1.0) / math.pi)
    keep = d <= max_edge_len_factor * nominal_diam
    return arr[keep]


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def image_features_from_table(fibers: pd.DataFrame, edges: np.ndarray, cfg: ExtractorConfig) -> Dict[str, float]:
    if fibers.empty:
        raise ValueError("No fibers were segmented.")
    subtype = fibers["subtype"].values.astype(int)
    csa = fibers["csa_norm"].values.astype(float)
    n = len(fibers)

    p0 = np.mean(subtype == 0)
    p1 = np.mean(subtype == 1)
    p2 = np.mean(subtype == 2)

    if len(edges) > 0:
        same = subtype[edges[:, 0]] == subtype[edges[:, 1]]
        observed_same = float(np.mean(same))
    else:
        observed_same = 0.0
    probs = np.array([p0, p1, p2], dtype=float)
    expected_same = float(np.sum(probs ** 2))
    grouping_index = (observed_same - expected_same) / (1.0 - expected_same + 1e-12)

    cluster_sizes: List[int] = []
    for t in [0, 1, 2]:
        active = subtype == t
        if len(edges):
            same_edges = edges[(subtype[edges[:, 0]] == t) & (subtype[edges[:, 1]] == t)]
        else:
            same_edges = np.empty((0, 2), dtype=int)
        comps = graph_connected_components(n, same_edges, active)
        cluster_sizes.extend([len(c) for c in comps])

    type1_csa = csa[subtype == 0]
    type2a_csa = csa[subtype == 1]
    type2x_csa = csa[subtype == 2]
    fast_csa = csa[subtype >= 1]

    features = {
        "n_fibers_segmented": float(n),
        "type1_fraction": float(p0),
        "type2a_fraction": float(p1),
        "type2x_fraction": float(p2),
        "hybrid_fraction": float(np.mean(fibers.get("hybrid_like", pd.Series(False, index=fibers.index)).values.astype(bool))),
        # No direct denervation marker in ordinary ATPase images.
        "denervated_fraction": float(cfg.denervated_fraction_assumed),
        "grouping_index": float(grouping_index),
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "largest_cluster_fraction": float(max(cluster_sizes) / n) if cluster_sizes else 0.0,
        "type1_csa_mean": float(np.mean(type1_csa)) if len(type1_csa) else 0.0,
        "type2a_csa_mean": float(np.mean(type2a_csa)) if len(type2a_csa) else 0.0,
        "type2x_csa_mean": float(np.mean(type2x_csa)) if len(type2x_csa) else 0.0,
        "fast_atrophy_fraction": float(np.mean(fast_csa < 0.65)) if len(fast_csa) else 0.0,
        "type2x_atrophy_fraction": float(np.mean(type2x_csa < 0.65)) if len(type2x_csa) else 0.0,
        "hypertrophy_fraction": float(np.mean(csa > 1.25)),
        "csa_mean": float(np.mean(csa)),
        "csa_cv": float(np.std(csa) / (np.mean(csa) + 1e-12)),
        "csa_gini": gini(csa),
        # These are not directly observed in a standard biopsy image, kept for ABC compatibility.
        "mu_size_gini": 0.0,
        "mean_conversion_gap": 0.0,
        "mismatch_fraction": 0.0,
        "mu_neighbor_same": 0.0,
    }
    return features


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_diagnostics(rgb: np.ndarray, tissue_mask: np.ndarray, labels: np.ndarray, fibers: pd.DataFrame, edges: np.ndarray, out_file: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tissue_mask, cmap="gray")
    axes[0, 1].set_title("Estimated tissue mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(segmentation.mark_boundaries(rgb, labels, color=(1, 0, 0), mode="thick"))
    axes[0, 2].set_title(f"Fiber segmentation ({len(fibers)} fibers)")
    axes[0, 2].axis("off")

    class_img = np.zeros_like(rgb)
    color_lut = {0: np.array([0.05, 0.05, 0.05]), 1: np.array([0.52, 0.52, 0.55]), 2: np.array([0.95, 0.95, 0.92])}
    for _, row in fibers.iterrows():
        lab = int(row["fiber_id"])
        st = int(row["subtype"])
        class_img[labels == lab] = color_lut.get(st, np.array([1, 0, 0]))
    axes[1, 0].imshow(class_img)
    axes[1, 0].set_title("Subtype map: I / IIa / IIx-like")
    axes[1, 0].axis("off")

    csa_img = np.zeros(labels.shape, dtype=float)
    for _, row in fibers.iterrows():
        csa_img[labels == int(row["fiber_id"])] = float(row["csa_norm"])
    im = axes[1, 1].imshow(csa_img, cmap="viridis", vmin=0.3, vmax=2.0)
    axes[1, 1].set_title("Normalized CSA")
    axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].imshow(rgb)
    pts = fibers[["centroid_x", "centroid_y"]].values
    if len(edges) > 0:
        step = max(1, len(edges) // 1500)
        for a, b in edges[::step]:
            axes[1, 2].plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], "-", linewidth=0.4, alpha=0.4)
    axes[1, 2].scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.8)
    axes[1, 2].set_title("Neighbor graph for grouping statistics")
    axes[1, 2].axis("off")

    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def plot_intensity_histogram(fibers: pd.DataFrame, out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for st, name in [(0, "Type I-like"), (1, "Type IIa-like"), (2, "Type IIx-like")]:
        vals = fibers.loc[fibers["subtype"] == st, "mean_gray"]
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.55, label=name)
    ax.set_xlabel("Mean grayscale intensity")
    ax.set_ylabel("Fiber count")
    ax.set_title("Subtype classification by ATPase-like intensity")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def run_extract(args: argparse.Namespace) -> None:
    ensure_dir(args.out)
    cfg = ExtractorConfig(
        max_side=args.max_side,
        min_area=args.min_area,
        max_area=args.max_area,
        min_distance=args.min_distance,
        tissue_threshold=args.tissue_threshold,
        n_subtypes=args.n_subtypes,
        denervated_fraction_assumed=args.denervated_fraction_assumed,
        seed=args.seed,
    )
    img = io.imread(args.image)
    rgb = robust_float_image(img)
    rgb = maybe_resize(rgb, cfg.max_side)
    tissue = build_tissue_mask(rgb, cfg)
    labels = segment_fibers(rgb, tissue, cfg)
    fibers = extract_fiber_table(rgb, labels, cfg)
    if fibers.empty:
        raise RuntimeError("No fibers were segmented. Try lowering --min-area or changing --min-distance/--tissue-threshold.")
    edges = build_neighbor_edges(fibers)
    features = image_features_from_table(fibers, edges, cfg)

    fibers.to_csv(os.path.join(args.out, "fiber_table.csv"), index=False)
    pd.DataFrame(edges, columns=["fiber_index_a", "fiber_index_b"]).to_csv(os.path.join(args.out, "neighbor_edges.csv"), index=False)
    pd.DataFrame([features]).to_csv(os.path.join(args.out, "features_for_abc.csv"), index=False)
    plot_diagnostics(rgb, tissue, labels, fibers, edges, os.path.join(args.out, "diagnostic_segmentation.png"))
    plot_intensity_histogram(fibers, os.path.join(args.out, "subtype_intensity_histogram.png"))

    print(f"Saved feature extraction outputs to: {args.out}")
    print(pd.Series(features).to_string())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract subtype/spatial/CSA features from a biopsy-like muscle image.")
    p.add_argument("--image", required=True, help="Path to input biopsy image.")
    p.add_argument("--out", default="biopsy_feature_output")
    p.add_argument("--max-side", type=int, default=1400)
    p.add_argument("--min-area", type=int, default=60)
    p.add_argument("--max-area", type=int, default=8000)
    p.add_argument("--min-distance", type=int, default=6)
    p.add_argument("--tissue-threshold", type=float, default=0.96)
    p.add_argument("--n-subtypes", type=int, choices=[2, 3], default=3)
    p.add_argument("--denervated-fraction-assumed", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_extract(args)


if __name__ == "__main__":
    main()
