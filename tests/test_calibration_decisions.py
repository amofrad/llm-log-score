"""Independent checks of calibration/score identities and threshold equivalence."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
import calibration_decisions as decisions


class CalibrationDecisionTests(unittest.TestCase):
    def test_penalty_nonanswers_reconcile_without_changing_denominator(self):
        record = dict(n_samples_requested=50, penalty_value=3,
                      penalty_correct_samples=10, penalty_incorrect_samples=20,
                      penalty_abstain_samples=15, penalty_not_attempted_samples=5,
                      penalty_accuracy_overall=.2, penalty_hallucination_rate=.4,
                      penalty_abstention_rate=.3)
        self.assertEqual(decisions.penalty_record_rates(record, 3), (.2, .4))
        record['penalty_not_attempted_samples'] = 4
        with self.assertRaises(ValueError):
            decisions.penalty_record_rates(record, 3)

    def test_isotonic_matches_independent_library_with_ties(self):
        rng=np.random.default_rng(8)
        for _ in range(30):
            u=rng.integers(0,11,100)/10
            z=rng.integers(0,2,100)
            fit=decisions.fit_isotonic(u,z)
            expected=IsotonicRegression().fit(u,z).predict(fit.x)
            np.testing.assert_allclose(fit.fitted,expected,atol=1e-14)

    def test_exhaustive_empirical_equivalence_and_penalty_order(self):
        # Enumerate outcomes rather than just checking two implementations agree.
        u=np.array([0.,.2,.2,.4,.6,.8])
        penalties=[0.,.1,.5,1.,2.,3.,6.,10.]
        candidates=np.unique(np.r_[0.,u,np.inf])
        for outcomes in itertools.product([0,1],repeat=len(u)):
            z=np.asarray(outcomes)
            fit=decisions.fit_isotonic(u,z)
            previous=-np.inf
            for penalty in penalties:
                actual=decisions.isotonic_threshold(fit,penalty)
                values=np.array([np.sum((u>=t)*(z-penalty*(1-z))) for t in candidates])
                optimum=candidates[np.flatnonzero(np.isclose(values,values.max(),atol=1e-12,rtol=0))[0]]
                self.assertEqual(actual,optimum)
                self.assertEqual(actual,decisions.direct_threshold(fit,penalty))
                self.assertGreaterEqual(actual,previous)
                previous=actual

    def test_step_extension_matches_raw_cutoff_between_training_scores(self):
        fit=decisions.fit_isotonic([.2,.4,.8,.9],[0,0,1,1])
        u=np.array([-1.,0.,.1,.2,.3,.4,.5,.79,.8,.85,.9,1.])
        p=decisions.isotonic_probability(fit,u)
        for penalty in [0.,1.,3.,6.]:
            selected=(u>=0)&(u>=decisions.direct_threshold(fit,penalty))
            by_probability=(u>=0)&(p>=penalty/(1+penalty))
            np.testing.assert_array_equal(selected,by_probability)

    def test_calibration_score_identity_and_empty_decisions(self):
        u=np.array([.9,.8,.75,.2,-1.]); z=np.array([1,0,1,0,0])
        m,scores=decisions.metrics(u,z,.75,3)
        np.testing.assert_array_equal(scores,[1,-3,1,0,0])
        self.assertAlmostEqual(m['mean_score'],-.2)
        self.assertAlmostEqual(m['score_overstatement'],4*m['answer_rate']*(m['conditional_error']-m['mean_raw_reported_error']))
        m,_=decisions.metrics(u,z,np.inf,3)
        self.assertEqual(m['mean_score'],0)
        self.assertTrue(np.isnan(m['conditional_error']))

    def test_beta_gradient_and_monotonicity(self):
        u=np.array([0.,.1,.3,.5,.8,.9,1.]); z=np.array([0,0,1,0,1,1,1])
        x=decisions.beta_design(u)
        coef=np.array([.6,1.2,-.4])
        error=check_grad(lambda c:decisions.beta_objective(c,x,z)[0],
                         lambda c:decisions.beta_objective(c,x,z)[1],coef)
        self.assertLess(error,1e-6)
        coef=decisions.fit_beta(u,z)
        p=decisions.beta_probability(coef,np.arange(1001)/1000)
        self.assertTrue((np.diff(p)>=0).all())
        self.assertTrue((coef[:2]>=0).all())

    def test_test_labels_cannot_change_fitting_or_selected_threshold(self):
        rng=np.random.default_rng(44); n=300
        frame=pd.DataFrame({'question_id':[f'q{i:04d}' for i in range(n)],
                            'u':rng.integers(1,101,n)/100,'z':rng.integers(0,2,n)})
        for penalty in [0,3,6]:
            frame[f'penalty_correct_{penalty}']=.3
            frame[f'penalty_incorrect_{penalty}']=.1
        a=decisions.analyze(frame,decisions.SEED)
        _,test=decisions.prior.split_frames(frame,decisions.SEED)
        changed=frame.copy()
        changed.loc[changed.question_id.isin(test.question_id),'z']=1-changed.loc[changed.question_id.isin(test.question_id),'z']
        b=decisions.analyze(changed,decisions.SEED)
        pd.testing.assert_frame_equal(a[1],b[1])
        np.testing.assert_array_equal(a[4].fitted,b[4].fitted)
        np.testing.assert_array_equal(a[5],b[5])


if __name__=='__main__':
    unittest.main()
