"""Check split nesting, test-label isolation, and finite-sample edge cases."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import threshold_split_sensitivity as sensitivity


class SplitSensitivityTests(unittest.TestCase):
    def test_nested_selection_and_common_unseen_test(self):
        a,ta,ca=sensitivity.split_indices(4326,20260912,5)
        b,tb,cb=sensitivity.split_indices(4326,20260912,7)
        self.assertEqual((len(a),len(ta),len(b),len(tb)),(2163,2163,3028,1298))
        self.assertTrue(set(a)<set(b))
        self.assertFalse(set(b)&set(cb))
        self.assertTrue(set(cb)<=set(ta))
        np.testing.assert_array_equal(ca,cb)
        np.testing.assert_array_equal(tb,cb)

    def test_test_labels_cannot_change_bounds_or_choices(self):
        u=np.full(1000,.8);z=np.ones(1000,dtype=int);grid=np.array([.5,.8,1.])
        for tenths in [5,7]:
            sel,test,common=sensitivity.split_indices(1000,91,tenths)
            n,k,upper=sensitivity.bounds_at(u[sel],z[sel],grid)
            index=sensitivity.choose(grid,n,upper,.2)
            changed=z.copy();changed[test]=0
            nn,kk,uu=sensitivity.bounds_at(u[sel],changed[sel],grid)
            np.testing.assert_array_equal(n,nn);np.testing.assert_array_equal(k,kk)
            np.testing.assert_array_equal(upper,uu)
            self.assertEqual(index,sensitivity.choose(grid,nn,uu,.2))
            self.assertEqual(sensitivity.evaluate(u[test],z[test],grid[index])['observed_error'],0.)
            self.assertEqual(sensitivity.evaluate(u[test],changed[test],grid[index])['observed_error'],1.)

    def test_empty_reports_do_not_count_as_answers_at_zero(self):
        u=np.array([-1.,0.,.9]);z=np.array([0,1,0]);grid=np.array([0.,1.])
        n,k,upper=sensitivity.bounds_at(u,z,grid)
        np.testing.assert_array_equal(n,[2,0]);np.testing.assert_array_equal(k,[1,0])
        self.assertEqual(upper[1],1.)
        self.assertIsNone(sensitivity.evaluate(u,z,None)['observed_error'])
        self.assertEqual(sensitivity.evaluate(u,z,0.)['n_answered'],2)

    def test_exact_bound_and_nonmonotone_threshold_support(self):
        grid=np.array([.5,1.]);u=np.full(20,.8);z=np.ones(20,dtype=int)
        _,_,upper=sensitivity.bounds_at(u,z,grid)
        self.assertAlmostEqual(upper[0],1-(sensitivity.DELTA/2)**(1/20))
        grid=np.array([.4,.5,.6]);n=np.array([100,80,40]);bounds=np.array([.2,.3,.1])
        self.assertEqual(sensitivity.choose(grid,n,bounds,.25),0)
        self.assertEqual(sensitivity.choose(grid,n,bounds,.15),2)
        self.assertIsNone(sensitivity.choose(grid,n,bounds,.05))


if __name__=='__main__':unittest.main()
