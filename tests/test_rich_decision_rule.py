"""Check report information boundaries, tied scores, and certification isolation."""
import itertools
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rich_decision_rule as rich


class RichDecisionRuleTests(unittest.TestCase):
    def test_features_keep_idk_separate_from_unallocated_probability(self):
        raw = json.dumps({"answers": [{"answer": "a", "points": 60},
                                      {"answer": "b", "points": 20},
                                      {"answer": "IDK", "points": 10}]})
        f = rich.report_features(raw)
        self.assertTrue(f["eligible"])
        self.assertAlmostEqual(f["u"], .6)
        self.assertAlmostEqual(f["idk"], .1)
        self.assertAlmostEqual(f["listed_total"], .9)
        self.assertAlmostEqual(f["margin"], .4)
        self.assertAlmostEqual(f["top_share"], .75)
        empty = rich.report_features('{"answers":[{"answer":"IDK","points":100}]}')
        self.assertFalse(empty["eligible"])
        self.assertEqual(empty["u"], -1)
        self.assertEqual(empty["normalized_entropy"], 0)

    def test_grades_change_labels_but_not_report_features(self):
        record = {"question_id": "a", "question": "Example", "gold_answer": "foo",
                  "raw_log_response": '{"answers":[{"answer":"foo","points":100}]}',
                  "log_candidates_json": [{"answer": "foo", "probability": 1., "grade": "correct"}]}
        correct = rich.prepare([record])
        record["log_candidates_json"][0]["grade"] = "incorrect"
        incorrect = rich.prepare([record])
        pd.testing.assert_frame_equal(correct[list(rich.FEATURES)], incorrect[list(rich.FEATURES)])
        self.assertEqual((correct.z.iloc[0], incorrect.z.iloc[0]), (1, 0))

    def test_sequence_stops_before_a_later_passing_test(self):
        score = np.r_[np.full(900, .3), np.full(30, .6), np.full(30, .9)]
        z = np.r_[np.ones(900), np.zeros(30), np.ones(30)]
        chosen, n, k, bounds, visited, passed = rich.fixed_sequence(
            score, z, [0, 1, 2], .2, .05, grid=np.array([.2, .5, .8]))
        self.assertEqual(chosen, 0)
        self.assertEqual(visited, [0, 1])
        self.assertEqual(passed, [0])
        self.assertLess(bounds[2], .2)
        result = rich.fixed_sequence(score, z, [1, 2, 0], .2, .05,
                                     grid=np.array([.2, .5, .8]))
        self.assertIsNone(result[0])
        self.assertEqual(result[4], [1])

    def test_aurc_handles_ties_without_label_order_bias(self):
        s = np.array([.9, .5, .5])
        y = np.array([1, 0, 1])
        expected = np.mean([np.mean(np.cumsum(1-y[list(order)]) / np.arange(1, 4))
                            for order in [(0, 1, 2), (0, 2, 1)]])
        self.assertAlmostEqual(rich.aurc(s, y), expected)
        self.assertAlmostEqual(rich.aurc([.5]*3, y), 1/3)
        for order in itertools.permutations(range(3)):
            self.assertAlmostEqual(rich.aurc(s[list(order)], y[list(order)]), expected)
        self.assertAlmostEqual(rich.coverage_error(s, y, 2/3), .25)

    def test_split_membership_is_disjoint_and_order_invariant(self):
        frame = pd.DataFrame({"question_id": [f"q{i:05d}" for i in range(4326)]})
        indices = rich.split_indices(frame, rich.SEED)
        self.assertEqual([len(x) for x in indices.values()], [2163, 1081, 1082])
        self.assertEqual(len(set(np.concatenate(list(indices.values())))), len(frame))
        reversed_frame = frame.iloc[::-1].reset_index(drop=True)
        reversed_indices = rich.split_indices(reversed_frame, rich.SEED)
        for name in indices:
            self.assertEqual(set(frame.iloc[indices[name]].question_id),
                             set(reversed_frame.iloc[reversed_indices[name]].question_id))

    def test_test_labels_cannot_change_scores_orders_or_certification(self):
        rng = np.random.default_rng(32)
        n = 500
        frame = pd.DataFrame(rng.uniform(size=(n, len(rich.FEATURES))), columns=rich.FEATURES)
        frame["eligible"] = True
        frame["z"] = rng.binomial(1, .8, n)
        frame["question_id"] = [f"q{i:04d}" for i in range(n)]
        frame.loc[0, ["u", "eligible"]] = [-1, False]
        a = rich.analyze(frame, rich.SEED)
        changed = frame.copy()
        changed.loc[a[4]["test"], "z"] = 1-changed.loc[a[4]["test"], "z"]
        b = rich.analyze(changed, rich.SEED)
        for method in rich.METHODS:
            np.testing.assert_array_equal(a[3][method], b[3][method])
            self.assertEqual(a[3][method][0], -1.)
        pd.testing.assert_frame_equal(a[2], b[2])
        for col in ["threshold", "supported", "calibration_upper"]:
            pd.testing.assert_series_equal(a[1][col], b[1][col])


if __name__ == "__main__":
    unittest.main()
