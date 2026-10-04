"""Checks for label-free decisions, error certificates, and held-out evaluation."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import threshold_selection as selection


class ThresholdSelectionTests(unittest.TestCase):
    def test_raw_selection_preserves_ties_and_excludes_idk(self):
        raw = json.dumps({"answers": [{"answer": "first", "points": 40},
                                      {"answer": "second", "points": 40},
                                      {"answer": "I don't know", "points": 20}]})
        entries, top, _ = selection.raw_selection(raw)
        self.assertEqual(entries[top]["answer"], "first")
        self.assertEqual(entries[top]["probability"], .4)
        self.assertIsNone(selection.raw_selection('{"answers":[{"answer":"IDK","points":100}]}')[1])

    def test_grade_recovery_does_not_replace_raw_selection(self):
        raw = json.dumps({"answers": [{"answer": "wrong", "points": 40},
                                      {"answer": "variant a", "points": 30},
                                      {"answer": "variant b", "points": 20},
                                      {"answer": "IDK", "points": 10}]})
        record = {"question_id": "one", "question": "Question", "gold_answer": "gold",
                  "raw_log_response": raw, "log_candidates_json": [
                      {"answer": "gold", "probability": .5, "grade": "correct", "source_answers": ["variant a", "variant b"]},
                      {"answer": "wrong", "probability": .4, "grade": "incorrect"},
                      {"answer": "IDK", "probability": .1, "grade": "not_attempted"}]}
        result = selection.prepare_records([record]).iloc[0]
        self.assertEqual((result.answer, result.u, result.z), ("wrong", .4, 0))

    def test_recovery_uses_group_totals_when_displays_collide(self):
        raw = json.dumps({"answers": [{"answer": "Cobra", "points": 85},
                                      {"answer": "The Cobra", "points": 8},
                                      {"answer": "IDK", "points": 7}]})
        record = {"question_id": "one", "question": "Question", "gold_answer": "King cobra",
                  "raw_log_response": raw, "log_candidates_json": [
                      {"answer": "King cobra", "probability": .85, "grade": "correct"},
                      {"answer": "Cobra", "probability": .08, "grade": "incorrect"},
                      {"answer": "IDK", "probability": .07, "grade": "not_attempted"}]}
        self.assertEqual(selection.prepare_records([record]).iloc[0].z, 1)
        record["log_candidates_json"][0]["probability"] = .84
        with self.assertRaises(ValueError):
            selection.prepare_records([record])

    def test_binomial_bound_endpoints_and_exact_coverage(self):
        bound = selection.upper_binomial(np.array([0, 0, 20]), np.array([0, 20, 20]), .05)
        np.testing.assert_allclose(bound, [1., 1 - .05**(1/20), 1.])
        n = 30
        k = np.arange(n + 1)
        bounds = selection.upper_binomial(k, n, .05)
        for p in [.01, .1, .25, .5, .9]:
            failure_probability = binom.pmf(k, n, p)[bounds < p].sum()
            self.assertLessEqual(failure_probability, .05 + 1e-12)

    def test_grid_bound_and_exact_threshold_inclusion(self):
        frame = pd.DataFrame({"u": [.95, .95, -1.], "z": [1, 0, 0]})
        bounds = selection.calibrate(frame, np.array([.9, .95, 1.]))
        np.testing.assert_equal(bounds.n_calibration.to_numpy(), [2, 2, 0])
        self.assertGreater(bounds.risk_upper.iloc[0], selection.upper_binomial(1, 2, .05))
        self.assertEqual(bounds.risk_upper.iloc[-1], 1.)

    def test_nonmonotone_bounds_and_no_supported_threshold(self):
        bounds = pd.DataFrame({"threshold": [.4, .5, .6], "n_calibration": [100, 80, 40],
                               "risk_upper": [.2, .3, .1]})
        self.assertEqual(selection.choose_threshold(bounds, .25).threshold, .4)
        self.assertEqual(selection.choose_threshold(bounds, .15).threshold, .6)
        self.assertIsNone(selection.choose_threshold(bounds, .05))
        result = selection.evaluate(pd.DataFrame({"u": [.9], "z": [1]}), None)
        self.assertEqual(result["answer_rate"], 0)
        self.assertTrue(np.isnan(result["observed_error"]))

    def test_split_order_invariance_and_test_labels_not_used_in_selection(self):
        frame = pd.DataFrame({"question_id": [f"q{i:04d}" for i in range(1000)],
                              "u": [.8]*1000, "z": [1]*1000})
        cal, test = selection.split_frames(frame, selection.SPLIT_SEED)
        cal_reordered, test_reordered = selection.split_frames(frame.iloc[::-1], selection.SPLIT_SEED)
        self.assertEqual(set(cal.question_id), set(cal_reordered.question_id))
        self.assertEqual(set(test.question_id), set(test_reordered.question_id))
        self.assertFalse(set(cal.question_id) & set(test.question_id))
        baseline, bounds = selection.analyze_split(frame, selection.SPLIT_SEED)
        changed = frame.copy()
        changed.loc[changed.question_id.isin(test.question_id), "z"] = 0
        result, new_bounds = selection.analyze_split(changed, selection.SPLIT_SEED)
        pd.testing.assert_frame_equal(bounds, new_bounds)
        np.testing.assert_equal(baseline.threshold.to_numpy(), result.threshold.to_numpy())
        self.assertTrue((baseline.observed_error.dropna() == 0).all())
        self.assertTrue((result.observed_error.dropna() == 1).all())


if __name__ == "__main__":
    unittest.main()
