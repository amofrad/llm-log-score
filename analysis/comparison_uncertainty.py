"""Relative matched-abstention gains and uncertainty for top-p cross-tables.

Resample whole questions, preserving each report and all 50 EPP responses.
The relative accuracy gain is 100 * (RBD accuracy - EPP accuracy) / EPP
accuracy, with the RBD curve interpolated at that resample's EPP abstention.
Intervals are pointwise 95% percentile intervals, not simultaneous bands.
Run with --outdir to preview outputs without replacing published numerical results.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, MODEL_ORDER, load_run
from make_figures import (
    load_penalty_arm, _frontier_decision_matrices,
    _interp_frontier_on_abstention, _log_top_p_outcome,
    _penalty_sample_average_weights,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[1] / "results/figures")
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    relative, cells = [], []
    row_names = ["abstain", "correct", "incorrect"]
    col_names = ["coverage", "miscoverage_with_idk", "miscoverage_without_idk"]
    thresholds = np.linspace(0, 0.95, 96)
    for mi, key in enumerate(MODEL_ORDER):
        label = RUNS[key]["label"]
        reports = load_run(RUNS[key]["run"]).set_index("question_id", drop=False)
        assert reports.index.is_unique
        for level in (3, 6):
            epp = load_penalty_arm(key, level)
            ids = reports.index.intersection(epp.index)
            assert len(ids) == len(reports) == len(epp) == 4326
            log, pen = reports.loc[ids], epp.loc[ids]
            assert np.all(pen["n_samples_requested"].to_numpy() == 50)
            weights = np.array([[d[r] for r in row_names] for d in pen.apply(_penalty_sample_average_weights, axis=1)])
            np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
            _, correct, abstain = _frontier_decision_matrices(log, thresholds)
            n = len(ids)

            def relative_gain(w):
                epp_rates = w @ weights
                abs_grid = w @ abstain
                assert abs_grid.min() <= epp_rates[0] <= abs_grid.max()
                rbd = _interp_frontier_on_abstention(abs_grid, w @ correct, np.array([epp_rates[0]]))[0]
                gain = rbd - epp_rates[1]
                return 100 * gain / epp_rates[1], rbd, epp_rates, gain

            point, rbd, ep, gap = relative_gain(np.ones(n) / n)
            rng = np.random.default_rng(20261003 + 10 * mi + level)
            boot = []
            if level == 3:
                outcomes = log.candidates.map(_log_top_p_outcome).to_numpy()
                indicators = np.column_stack([outcomes == c for c in ["correct", "abstain", "incorrect"]])
                question_cells = weights[:, :, None] * indicators[:, None, :]
                totals = question_cells.sum(axis=0)
                fractions = totals / totals.sum(axis=1, keepdims=True)
                boot_cells = []
            for _ in range(args.reps):
                sample = rng.integers(0, n, size=n)
                w = np.bincount(sample, minlength=n).astype(float) / n
                boot.append(relative_gain(w)[0])
                if level == 3:
                    means = np.tensordot(w, question_cells, axes=(0, 0))
                    boot_cells.append(means / means.sum(axis=1, keepdims=True))
            lo, hi = np.quantile(boot, [0.025, 0.975])
            relative.append(dict(model=key, model_label=label, L=level, n_questions=n,
                                 epp_accuracy=ep[1], rbd_accuracy=rbd, abstention=ep[0],
                                 relative_accuracy_gain=point, ci_low=lo, ci_high=hi,
                                 bootstrap_reps=args.reps))
            if level == 3:
                ci = np.quantile(boot_cells, [0.025, 0.975], axis=0)
                for ri, row in enumerate(row_names):
                    for cj, col in enumerate(col_names):
                        cells.append(dict(model=key,model_label=label,epp_outcome=row,set_outcome=col,
                                          averaged_count=totals[ri,cj],row_percent=100*fractions[ri,cj],
                                          ci_low=100*ci[0,ri,cj],ci_high=100*ci[1,ri,cj],
                                          n_questions=n,bootstrap_reps=args.reps))
            print(f"{label}, L={level}: relative accuracy gain {point:.2f}% [{lo:.2f}, {hi:.2f}]", flush=True)
    rel = pd.DataFrame(relative)
    tab = pd.DataFrame(cells)
    rel.to_csv(args.outdir / "matched_relative_accuracy.csv",index=False)
    tab.to_csv(args.outdir / "set_outcome_intervals.csv",index=False)

    lines = ["% Table S2: pointwise 95% intervals; percentages within each EPP row."]
    for mi, key in enumerate(MODEL_ORDER):
        if mi:
            lines.append(r"\midrule")
        for ri, row in enumerate(row_names):
            values = []
            for col in col_names:
                r = tab[(tab.model == key) & (tab.epp_outcome == row) & (tab.set_outcome == col)].iloc[0]
                digits = 3 if 0 < r.row_percent < .01 else 2 if r.row_percent < .1 else 1
                values.append(f"${r.row_percent:.{digits}f}$ $[{r.ci_low:.{digits}f}, {r.ci_high:.{digits}f}]$")
            label = RUNS[key]["label"] if ri == 0 else ""
            lines.append(label + " & " + row.title() + " & " + " & ".join(values) + r"\\")
    (args.outdir / "TableS2_SetOutcomeIntervals_rows.tex").write_text("\n".join(lines) + "\n")
    return rel


if __name__ == "__main__":
    main()
