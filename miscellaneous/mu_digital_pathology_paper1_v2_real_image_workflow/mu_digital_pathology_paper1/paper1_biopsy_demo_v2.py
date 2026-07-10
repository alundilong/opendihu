#!/usr/bin/env python3
"""Demo driver for the three-phenotype motor-unit digital pathology model."""

from __future__ import annotations

import os
import pandas as pd

from mu_digital_pathology_v2 import RemodelParams, scenario_params, run_forward, extract_features
from biopsy_renderer_v2 import plot_snapshot_biopsy_panel, plot_scenario_biopsy_comparison


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    base_out = "/mnt/data/mu_digital_pathology_paper1_demo/biopsy_renderer_demo_v2"
    ensure_dir(base_out)

    base = RemodelParams(grid_n=42, n_mu=25, seed=7)
    params = scenario_params("aging_type2", base)
    final_state, snapshots, hist = run_forward(params)
    plot_snapshot_biopsy_panel(
        snapshots,
        os.path.join(base_out, "aging_type2_biopsy_snapshots_v2.png"),
        title="Aging type-II vulnerability (I / IIa / IIx): coarse map vs biopsy-like rendering",
    )
    hist.to_csv(os.path.join(base_out, "aging_type2_history_v2.csv"), index=False)
    pd.DataFrame([extract_features(final_state)]).to_csv(os.path.join(base_out, "aging_type2_final_features_v2.csv"), index=False)

    scenario_names = ["classic_like", "aging_type2", "exercise_rescued", "als_like_failed_rescue"]
    final_states = {}
    feature_rows = []
    for idx, name in enumerate(scenario_names):
        p = scenario_params(name, RemodelParams(grid_n=42, n_mu=25, seed=20 + idx))
        st, _, _ = run_forward(p)
        final_states[name] = st
        feat = extract_features(st)
        feat["scenario"] = name
        feature_rows.append(feat)

    plot_scenario_biopsy_comparison(
        final_states,
        os.path.join(base_out, "scenario_biopsy_comparison_v2.png"),
        title="Final-stage scenario comparison: synthetic biopsy-like rendering (I / IIa / IIx)",
    )
    pd.DataFrame(feature_rows).to_csv(os.path.join(base_out, "scenario_final_features_v2.csv"), index=False)
    print(f"Saved biopsy-renderer v2 demo outputs to: {base_out}")


if __name__ == "__main__":
    main()
