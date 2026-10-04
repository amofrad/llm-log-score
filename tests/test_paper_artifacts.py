"""Numerical consistency across the published paper artifacts."""
from decimal import Decimal
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[1] / 'results/figures'


class PaperArtifactsTests(unittest.TestCase):
    def test_decision_outcomes_and_display_rounding(self):
        table = pd.read_csv(FIGURES/'tables/Table1_DecisionOutcomes.csv', dtype={'display_abstain':str, 'display_incorrect':str, 'display_correct':str})
        self.assertEqual(len(table), 18)
        self.assertFalse(table.duplicated(['model','method','L']).any())
        np.testing.assert_allclose(table[['abstain','incorrect','correct']].sum(axis=1), 1, atol=1e-12)
        for _, row in table.iterrows():
            self.assertEqual(sum(Decimal(row[x]) for x in ['display_abstain','display_incorrect','display_correct']), Decimal('1.000'))

    def test_relative_gains_match_absolute_gains_at_the_same_abstention(self):
        relative = pd.read_csv(FIGURES/'matched_relative_accuracy.csv')
        absolute = pd.read_csv(FIGURES/'matched_abstention_gaps.csv')
        self.assertEqual(len(relative), 6)
        for _, r in relative.iterrows():
            a = absolute[(absolute.model == r.model_label) & (absolute.L == r.L)].iloc[0]
            self.assertAlmostEqual(r.abstention, a.pen_abstention, places=12)
            self.assertAlmostEqual(r.epp_accuracy, a.pen_accuracy, places=12)
            self.assertAlmostEqual(r.rbd_accuracy-r.epp_accuracy, a.gap_accuracy, places=12)
            self.assertAlmostEqual(r.relative_accuracy_gain, 100*a.gap_accuracy/a.pen_accuracy, places=10)
            self.assertLess(r.ci_low, r.ci_high)

    def test_set_outcome_counts_and_intervals_use_the_same_denominators(self):
        intervals = pd.read_csv(FIGURES/'set_outcome_intervals.csv')
        for name in ['F4_OutcomeTable_L3','FS1_OutcomeTable_L0','FS1_OutcomeTable_L6']:
            cells = pd.read_csv(FIGURES/(name+'_set_outcomes.csv'))
            self.assertEqual(len(cells), 27)
            for (_, _), group in cells.groupby(['model','penalty_outcome']):
                self.assertEqual(set(group.set_outcome), {'coverage','miscoverage_with_idk','miscoverage_without_idk'})
                self.assertAlmostEqual(group.row_percent.sum(), 100, places=9)
                np.testing.assert_allclose(group.row_percent, 100*group.count_or_expected_count/group.count_or_expected_count.sum(), atol=1e-9)
            np.testing.assert_allclose(cells.groupby('model').count_or_expected_count.sum(), 4326, atol=1e-8)
            if name.startswith('F4_'):
                merged=cells.merge(intervals,left_on=['model','penalty_outcome','set_outcome'],right_on=['model','epp_outcome','set_outcome'],validate='one_to_one')
                self.assertEqual(len(merged),27)
                np.testing.assert_allclose(merged.row_percent_x,merged.row_percent_y,atol=1e-9)
                np.testing.assert_allclose(merged.count_or_expected_count,merged.averaged_count,atol=1e-9)
                self.assertTrue(((merged.ci_low>=0)&(merged.ci_high<=100)&(merged.ci_low<=merged.ci_high)).all())

    def test_all_current_figures_and_tables_are_present(self):
        import json
        manifest=json.loads((FIGURES/'artifact_manifest.json').read_text())
        self.assertEqual(len(manifest['figures']),10)
        for figure in manifest['figures']:
            for name in figure['files']:
                self.assertTrue((FIGURES/name).is_file(),name)
        tables=json.loads((FIGURES/'tables/manifest.json').read_text())
        self.assertEqual(len(tables),13)
        for table in tables:
            self.assertTrue((FIGURES/'tables'/table['data']).is_file())
            tex=(FIGURES/'tables'/table['latex']).read_text()
            self.assertNotIn('\n@ROWS@\n',tex)


if __name__ == '__main__':
    unittest.main()
