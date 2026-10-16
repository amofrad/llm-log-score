"""Generate SI Table S9 from the fixed-grid threshold-selection analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common

ROOT = Path(__file__).resolve().parents[1]
PAPER_TARGETS = (.20, .25, .30, .35, .40)


def one(frame, model, alpha, method):
    rows = frame[(frame.model_key == model) & (frame.representation == "raw")
                 & (frame.method == method) & np.isclose(frame.alpha, alpha, rtol=0, atol=1e-14)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {model}, {alpha}, {method}; found {len(rows)}")
    return rows.iloc[0]


def run(out_dir=None, recompute=False, primary_grader="openai", analysis_dir=None):
    out_dir = Path(out_dir) if out_dir else ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path(analysis_dir) if analysis_dir else ROOT / "results/intermediate"
    minimum_bounds = []
    for grader in ("openai", "gemini"):
        folder = analysis_dir / f"alpha_risk_control_{grader}"
        if recompute or not (folder / "primary_results.csv").exists():
            import alpha_risk_control as arc
            arc.run(grader, folder, repeats=100)
        # Minimum bounds quoted in the SI discussion, including nonqualifying cases.
        bounds = pd.read_csv(folder / "all_bounds.csv", float_precision="round_trip")
        primary_seed = pd.read_csv(folder / "primary_results.csv", usecols=["seed"]).seed.unique()
        assert len(primary_seed) == 1
        bounds = bounds[(bounds.seed == primary_seed[0]) & (bounds.n_cert_answered > 0)]
        for model, group in bounds.groupby("model_key", sort=False):
            best = group.sort_values(["risk_upper", "threshold"]).iloc[0]
            minimum_bounds.append({"grader": grader, "model_key": model,
                                   "minimum_supported_alpha": best.risk_upper,
                                   "threshold": best.threshold,
                                   "n_cert_answered": best.n_cert_answered,
                                   "k_cert_incorrect": best.k_cert_incorrect})
    pd.DataFrame(minimum_bounds).to_csv(out_dir / "threshold_selection_summary.csv", index=False)
    primary = pd.read_csv(analysis_dir / f"alpha_risk_control_{primary_grader}" / "primary_results.csv",
                          float_precision="round_trip")
    records = []
    for model in common.MODEL_ORDER:
        for alpha in PAPER_TARGETS:
            row = one(primary, model, alpha, "fixed_full")
            records.append(row.to_dict() | {"grader": primary_grader})
    pd.DataFrame(records).to_csv(out_dir / "TableS9_RiskControl.csv", index=False)
    print(f"Wrote SI Table S9 to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--grader", choices=("openai", "gemini"), default="openai",
                        help="Grader displayed in Table S9.")
    args = parser.parse_args()
    run(args.out_dir, args.recompute, args.grader)
