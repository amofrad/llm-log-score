"""Checks for selection, denominators, and the question bootstrap shortcut."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import threshold_calibration as calibration


def record(i, entries):
    return {"question_id": str(i), "log_candidates_json": json.dumps([
        {"answer": str(j), "probability": p, "grade": grade}
        for j, (p, grade) in enumerate(entries)])}


class ThresholdCalibrationTests(unittest.TestCase):
    def test_decimal_grid_includes_probabilities_equal_to_cutoff(self):
        for probability in [.07, .14, .28, .56, .57, .58, .95]:
            groups, counts = calibration.grouped_reports([record(0, [(probability, "correct")])])
            threshold = calibration.THRESHOLDS[np.argmin(abs(calibration.THRESHOLDS - probability))]
            result = calibration.rates(groups, counts, np.array([threshold]))
            self.assertEqual(result["n_answered"][0], 1)

    def test_empty_reports_ties_and_exact_threshold(self):
        records = [record(0, [(.4, "incorrect"), (.4, "correct"), (.2, "not_attempted")]),
                   record(1, [(.9, "correct"), (.1, "not_attempted")]),
                   record(2, [(1., "not_attempted")])]
        groups, counts = calibration.grouped_reports(records)
        result = calibration.rates(groups, counts, np.array([0., .4, .8, 1.]))
        np.testing.assert_equal(result["n_answered"], [2, 2, 1, 0])
        np.testing.assert_allclose(result["answer_rate"], [2/3, 2/3, 1/3, 0])
        np.testing.assert_allclose(result["observed_error"][:3], [.5, .5, 0])
        np.testing.assert_allclose(result["reported_error"][:3], [.35, .35, .1])
        self.assertTrue(np.isnan(result["observed_error"][-1]))

    def test_grouped_bootstrap_matches_question_resampling(self):
        records = [record(0, [(.8, "correct")]), record(1, [(.8, "correct")]),
                   record(2, [(.8, "incorrect")]), record(3, [(1, "not_attempted")])]
        groups, _ = calibration.grouped_reports(records)
        # Drawing question indices [0, 0, 2, 3, 1, 2] gives these category counts.
        category_counts = {(-1., 0): 1, (.8, 0): 2, (.8, 1): 3}
        weights = np.array([category_counts[tuple(pair)] for pair in groups])
        result = calibration.rates(groups, weights, np.array([.8]))
        self.assertAlmostEqual(result["answer_rate"][0], 5/6)
        self.assertAlmostEqual(result["observed_error"][0], 2/5)
        self.assertAlmostEqual(result["reported_error"][0], .2)

    def test_perfect_empirical_calibration_identity(self):
        groups = np.array([[.5, 0], [.5, 1], [.8, 0], [.8, 1]])
        counts = np.array([5, 5, 2, 8])
        result = calibration.rates(groups, counts, np.array([0., .5, .8]))
        np.testing.assert_allclose(result["error_gap"], 0, atol=1e-15)
        self.assertTrue(np.all(result["observed_error"] <= 1 - np.array([0., .5, .8]) + 1e-15))

    def test_no_normalization_and_duplicate_question_rejected(self):
        records = [record(0, [(.6, "correct"), (.1, "not_attempted")])]
        groups, counts = calibration.grouped_reports(records)
        result = calibration.rates(groups, counts, np.array([.6, .75]))
        np.testing.assert_equal(result["n_answered"], [1, 0])
        self.assertAlmostEqual(result["reported_error"][0], .4)
        with self.assertRaises(ValueError):
            calibration.grouped_reports(records * 2)


if __name__ == "__main__":
    unittest.main()
