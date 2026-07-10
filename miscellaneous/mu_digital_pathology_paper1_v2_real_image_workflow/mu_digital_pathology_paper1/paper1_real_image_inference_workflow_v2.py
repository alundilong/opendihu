#!/usr/bin/env python3
"""
End-to-end workflow for using a biopsy-like image as a target for the v2
motor-unit digital pathology model.

Steps
-----
1. Extract subtype/spatial/CSA features from image.
2. Run ABC-style simulation-based inference against those features.
3. Generate posterior summary and posterior predictive examples.

Interpretation
--------------
The inferred parameters are plausible regimes, not ground truth. Ordinary ATPase
or myosin-stained sections do not directly encode denervation rate, sprouting
radius, or reinnervation probability.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from real_biopsy_feature_extractor import (
    ExtractorConfig,
    robust_float_image,
    maybe_resize,
    build_tissue_mask,
    segment_fibers,
    extract_fiber_table,
    build_neighbor_edges,
    image_features_from_table,
    plot_diagnostics,
    plot_intensity_histogram,
)
from skimage import io

from mu_digital_pathology_v2 import (
    RemodelParams,
    run_abc,
    run_forward,
    extract_features,
    PARAMS_FOR_ABC,
    plot_abc_results,
    plot_inference_scatter,
    plot_maps,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_feature_extraction(args: argparse.Namespace, out_dir: str) -> dict:
    ensure_dir(out_dir)
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
    rgb = robust_float_image(io.imread(args.image))
    rgb = maybe_resize(rgb, cfg.max_side)
    tissue = build_tissue_mask(rgb, cfg)
    labels = segment_fibers(rgb, tissue, cfg)
    fibers = extract_fiber_table(rgb, labels, cfg)
    if fibers.empty:
        raise RuntimeError("No fibers segmented. Adjust segmentation parameters.")
    edges = build_neighbor_edges(fibers)
    features = image_features_from_table(fibers, edges, cfg)

    fibers.to_csv(os.path.join(out_dir, "fiber_table.csv"), index=False)
    pd.DataFrame(edges, columns=["fiber_index_a", "fiber_index_b"]).to_csv(os.path.join(out_dir, "neighbor_edges.csv"), index=False)
    pd.DataFrame([features]).to_csv(os.path.join(out_dir, "features_for_abc.csv"), index=False)
    plot_diagnostics(rgb, tissue, labels, fibers, edges, os.path.join(out_dir, "diagnostic_segmentation.png"))
    plot_intensity_histogram(fibers, os.path.join(out_dir, "subtype_intensity_histogram.png"))
    return features


def apply_row_to_params(base: RemodelParams, row: pd.Series, seed: int) -> RemodelParams:
    kwargs = {}
    for p in PARAMS_FOR_ABC:
        if p in row:
            kwargs[p] = float(row[p])
    kwargs["seed"] = seed
    return replace(base, **kwargs)


def posterior_predictive(df: pd.DataFrame, base: RemodelParams, out_dir: str, n_examples: int = 6, seed: int = 999) -> pd.DataFrame:
    ensure_dir(out_dir)
    acc = df[df["accepted"]].copy()
    if acc.empty:
        acc = df.head(max(1, n_examples)).copy()
    acc = acc.head(n_examples)
    rows = []
    for idx, (_, row) in enumerate(acc.iterrows()):
        p = apply_row_to_params(base, row, seed=seed + idx)
        st, snaps, hist = run_forward(p, snapshots=(0.0, max(base.death_targets)))
        feat = extract_features(st)
        feat["posterior_example"] = idx
        feat["abc_distance"] = float(row["distance"])
        rows.append(feat)
        ex_dir = os.path.join(out_dir, f"example_{idx:02d}")
        ensure_dir(ex_dir)
        plot_maps(snaps, os.path.join(ex_dir, "posterior_predictive_maps.png"), title=f"Posterior predictive example {idx}")
        hist.to_csv(os.path.join(ex_dir, "history.csv"), index=False)
        pd.DataFrame([feat]).to_csv(os.path.join(ex_dir, "features.csv"), index=False)
    pred = pd.DataFrame(rows)
    pred.to_csv(os.path.join(out_dir, "posterior_predictive_features.csv"), index=False)
    return pred


def plot_target_vs_posterior(target_feat: dict, pred: pd.DataFrame, out_file: str) -> None:
    feature_names = [
        "type1_fraction", "type2a_fraction", "type2x_fraction", "hybrid_fraction",
        "grouping_index", "mean_cluster_size", "largest_cluster_fraction",
        "type1_csa_mean", "type2a_csa_mean", "type2x_csa_mean",
        "fast_atrophy_fraction", "hypertrophy_fraction", "csa_cv",
    ]
    vals_target = np.array([target_feat.get(f, 0.0) for f in feature_names], dtype=float)
    vals_pred = np.array([pred[f].mean() if f in pred else 0.0 for f in feature_names], dtype=float)
    vals_sd = np.array([pred[f].std() if f in pred else 0.0 for f in feature_names], dtype=float)

    x = np.arange(len(feature_names))
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(x - 0.18, vals_target, width=0.36, label="target image features")
    ax.bar(x + 0.18, vals_pred, width=0.36, yerr=vals_sd, label="posterior predictive mean")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=60, ha="right")
    ax.set_ylabel("Feature value")
    ax.set_title("Target biopsy features vs posterior predictive simulations")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real biopsy image -> features -> ABC inference -> posterior predictive checks.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="real_image_inference_v2")
    parser.add_argument("--grid-n", type=int, default=42)
    parser.add_argument("--n-mu", type=int, default=25)
    parser.add_argument("--target-death", type=float, default=0.75)
    parser.add_argument("--abc-samples", type=int, default=300)
    parser.add_argument("--abc-keep", type=int, default=40)
    parser.add_argument("--posterior-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)

    # feature extraction args
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--min-area", type=int, default=60)
    parser.add_argument("--max-area", type=int, default=8000)
    parser.add_argument("--min-distance", type=int, default=6)
    parser.add_argument("--tissue-threshold", type=float, default=0.96)
    parser.add_argument("--n-subtypes", type=int, choices=[2, 3], default=3)
    parser.add_argument("--denervated-fraction-assumed", type=float, default=0.0)
    args = parser.parse_args()

    ensure_dir(args.out)
    feat_dir = os.path.join(args.out, "image_features")
    target_feat = run_feature_extraction(args, feat_dir)

    base = RemodelParams(grid_n=args.grid_n, n_mu=args.n_mu, seed=args.seed, death_targets=(0.0, args.target_death))
    abc_dir = os.path.join(args.out, "abc")
    ensure_dir(abc_dir)
    df = run_abc(target_feat, base, n_samples=args.abc_samples, keep=args.abc_keep, seed=args.seed + 200, verbose=True)
    df.to_csv(os.path.join(abc_dir, "abc_samples.csv"), index=False)
    plot_abc_results(df, None, os.path.join(abc_dir, "abc_posterior_histograms.png"))
    plot_inference_scatter(df, None, os.path.join(abc_dir, "abc_identifiability_scatter.png"))

    acc = df[df["accepted"]]
    summary_rows = []
    for p in PARAMS_FOR_ABC:
        summary_rows.append({
            "parameter": p,
            "accepted_mean": acc[p].mean(),
            "accepted_median": acc[p].median(),
            "accepted_p05": acc[p].quantile(0.05),
            "accepted_p95": acc[p].quantile(0.95),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(abc_dir, "posterior_summary.csv"), index=False)

    pp_dir = os.path.join(args.out, "posterior_predictive")
    pred = posterior_predictive(df, base, pp_dir, n_examples=args.posterior_examples, seed=args.seed + 1000)
    plot_target_vs_posterior(target_feat, pred, os.path.join(args.out, "target_vs_posterior_predictive.png"))

    print(f"Saved full real-image inference workflow to: {args.out}")
    print("Target features:")
    print(pd.Series(target_feat).to_string())


if __name__ == "__main__":
    main()
