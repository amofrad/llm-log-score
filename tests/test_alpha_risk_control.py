"""Checks for simultaneous risk bounds and development/certification separation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import multinomial

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
import alpha_risk_control as risk
import calibration_decisions as decisions


class AlphaRiskControlTests(unittest.TestCase):
    def example(self,n=300):
        rng=np.random.default_rng(15)
        return pd.DataFrame({"question_id":[f'q{i:04d}' for i in range(n)],
                             "u":rng.integers(0,101,n)/100,"z":rng.integers(0,2,n)})

    def test_lookup_uses_smallest_supported_cutoff_without_monotone_risk(self):
        bounds=pd.DataFrame({"threshold":[.3,.5,.7,.9],"risk_upper":[.2,.4,.1,.05],
                             "n_cert_answered":[100,80,40,0],"k_cert_incorrect":[10,20,1,0]})
        self.assertIsNone(risk.lookup(bounds,.09))
        self.assertEqual(risk.lookup(bounds,.15).threshold,.7)
        self.assertEqual(risk.lookup(bounds,.25).threshold,.3)
        for alpha in (0,1,np.nan,-.1):
            with self.assertRaises(ValueError):
                risk.lookup(bounds,alpha)

    def test_exhaustive_simultaneous_bound_coverage_with_random_answer_counts(self):
        # Enumerate every six-observation sample composition over (U,Z).
        probabilities=np.array([.2,.3,.05,.45])
        u=np.array([.2,.2,.8,.8]); z=np.array([0,1,0,1])
        grid=np.array([0.,.7,.95]); delta=.1
        failure=0.; total=0.
        for a in range(7):
            for b in range(7-a):
                for c in range(7-a-b):
                    composition=[a,b,c,6-a-b-c]
                    mass=multinomial.pmf(composition,6,probabilities)
                    frame=pd.DataFrame({'u':np.repeat(u,composition),'z':np.repeat(z,composition)})
                    bound=risk.build_bounds(frame,grid,delta)
                    failure+=mass*bool(np.any(bound.risk_upper.to_numpy()[:2]<[.25,.1]))
                    total+=mass
        self.assertAlmostEqual(total,1.)
        self.assertLessEqual(failure,delta+1e-12)

    def test_development_grid_contains_all_fitted_penalty_cutoffs(self):
        frame=self.example()
        grid=risk.learned_grid(frame)
        fit=decisions.fit_isotonic(frame.u,frame.z)
        for L in np.r_[np.arange(201)/20,100]:
            threshold=decisions.isotonic_threshold(fit,L)
            self.assertTrue(np.isinf(threshold) or threshold in grid)
        empty=pd.DataFrame({'u':[-1.,-1.],'z':[0,0]})
        np.testing.assert_array_equal(risk.learned_grid(empty),[0.])
        self.assertIsNone(risk.lookup(risk.build_bounds(empty,[0.]),.99))

    def test_certification_cannot_change_grid_and_test_cannot_change_decisions(self):
        frame=self.example()
        _,dev,cert,test=risk.split(frame,risk.SEED)
        self.assertTrue(set(dev.question_id).isdisjoint(cert.question_id))
        self.assertTrue(set(dev.question_id).isdisjoint(test.question_id))
        self.assertTrue(set(cert.question_id).isdisjoint(test.question_id))
        changed=frame.copy()
        changed.loc[changed.question_id.isin(cert.question_id),'z']=1-changed.loc[changed.question_id.isin(cert.question_id),'z']
        np.testing.assert_array_equal(risk.learned_grid(dev),risk.learned_grid(risk.split(changed,risk.SEED)[1]))
        original=risk.analyze(frame,risk.SEED)
        changed=frame.copy()
        changed.loc[changed.question_id.isin(test.question_id),'z']=1-changed.loc[changed.question_id.isin(test.question_id),'z']
        updated=risk.analyze(changed,risk.SEED)
        pd.testing.assert_frame_equal(original[1],updated[1])
        cols=['method','alpha','supported','threshold','certification_upper']
        pd.testing.assert_frame_equal(original[0][cols],updated[0][cols])

    def test_reference_matches_existing_fixed_grid_experiment(self):
        frame=self.example()
        reference,_=risk.old.analyze_split(frame,risk.SEED)
        updated,_,_=risk.analyze(frame,risk.SEED)
        a=reference[reference.method=='selected'].sort_values('alpha').reset_index(drop=True)
        b=updated[(updated.method=='fixed_full')&updated.alpha.isin(risk.old.TARGETS)].sort_values('alpha').reset_index(drop=True)
        for col in ('threshold','supported','n_answered','n_incorrect','answer_rate','observed_error','error_lo','error_hi'):
            np.testing.assert_allclose(a[col].astype(float),b[col].astype(float),equal_nan=True)


if __name__=='__main__':
    unittest.main()
