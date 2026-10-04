"""Check rejection guarantees' implementation and partial-tie semantics."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
import threshold_efficiency as eff


class ThresholdEfficiencyTests(unittest.TestCase):
    def test_holm_and_fallback_retain_bonferroni(self):
        rng = np.random.default_rng(13)
        for _ in range(100):
            p = 10**rng.uniform(-6, 0, 102)
            base = p <= .05/len(p)
            for method in ('holm','fallback'):
                rejected, _ = eff.reject(p,method,np.linspace(0,1,102))
                self.assertTrue(np.all(rejected[base]))

    def test_fallback_transfers_only_after_rejection(self):
        # High threshold passes and its budget reaches the middle threshold.
        rejected, levels = eff.reject(np.array([.02,.025,.01]),'fallback',np.arange(3))
        self.assertTrue(rejected.all())
        np.testing.assert_allclose(levels,[.05,.05*2/3,.05/3])
        # A failure retains that node's budget; it cannot reach the next node.
        rejected, levels = eff.reject(np.array([.02,.50,.01]),'fallback',np.arange(3))
        np.testing.assert_array_equal(rejected,[False,False,True])
        self.assertAlmostEqual(levels[0],.05/3)

    def test_fixed_sequences_stop_at_failure(self):
        grid = np.arange(101)/100
        p = np.ones(101)
        p[90] = .001
        p[89] = .001
        p[88] = .1
        p[87] = .001
        rejected,_ = eff.reject(p,'multistart',grid)
        np.testing.assert_array_equal(np.flatnonzero(rejected),[89,90])

    def test_partial_ties_nested_empty_lists_and_baseline(self):
        grid = np.array([0.,.4,.6,1.])
        u = np.array([-1.,0.,.4,.4,.5,.6,1.])
        coins = np.array([.1,.2,.1,.6,.7,.9,.8])
        t,gamma = eff.candidate_family(grid,True)
        mask = eff.acceptance(u,t,gamma,coins)
        self.assertFalse(mask[0].any())
        self.assertTrue(np.all(mask[:,:-1] >= mask[:,1:]))
        np.testing.assert_array_equal(mask[:,::4],u[:,None]>=grid)
        # At t=.4, gamma=.5, only the first tied question is answered.
        np.testing.assert_array_equal(mask[:,6],[False,False,True,False,True,True,True])

    def test_exact_bound_matches_pvalue_and_zero_answers(self):
        n = np.array([0,10,10,100,100,500])
        k = np.array([0,0,10,15,40,120])
        upper = eff.upper_bound(k,n,.05/102)
        for alpha in [.2,.25,.3,.35,.4]:
            p = np.where(n>0,binom.cdf(k,n,alpha),1)
            np.testing.assert_array_equal(upper<=alpha,p<=.05/102)
        self.assertEqual(upper[0],1)
        with self.assertRaises(ValueError):eff.upper_bound([.5],[3],.05)

    def test_test_grades_cannot_change_selection(self):
        u=np.array([.4,.6,.9,.3,.8,.9]);z=np.array([0,1,1,0,1,1])
        grid=np.array([.3,.6,.9]);mask=eff.acceptance(u,grid,np.ones(3))
        n,k=eff.counts(mask,z,np.arange(3))
        changed=z.copy();changed[3:]=1-changed[3:]
        nn,kk=eff.counts(mask,changed,np.arange(3))
        np.testing.assert_array_equal(n,nn);np.testing.assert_array_equal(k,kk)
        for method in ('bonferroni','holm','fallback'):
            self.assertEqual(eff.select(n,k,.4,method,grid)[0],eff.select(nn,kk,.4,method,grid)[0])


if __name__ == '__main__':unittest.main()
