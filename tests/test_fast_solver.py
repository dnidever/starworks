"""Regression checks against the retained Python reference; run with unittest."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from starmodel import starmodel as reference
from fast_solver import starmodel as fast
from fast_grid import grid_summary
from model_runner import run_model,diagnostics

class FastSolverTests(unittest.TestCase):
    def test_profiles_and_flags_match_reference(self):
        parameters=[(1.,.86071,5500.2,.7,.008),(.8,.22456,3960.,.7,.008),(50.,97160.,47770.,.7,.008)]
        parameters += [(1.,float(l),float(t),.7,.008) for l in np.linspace(.7,1.1,8) for t in np.linspace(4900,6200,8)]
        with contextlib.redirect_stdout(io.StringIO()):
            for p in parameters:
                a=reference(*p,verbose=False);b=fast(*p)
                self.assertEqual(a[:3],b[:3],p)
                for name in a[3].dtype.names:
                    if name=='zone':np.testing.assert_array_equal(a[3][name],b[3][name])
                    else:np.testing.assert_allclose(a[3][name],b[3][name],rtol=1e-9,atol=1e-8,equal_nan=True)
    def test_grid_matches_full_model_diagnostics(self):
        rows=grid_summary(1.,.7,.008,np.linspace(.8,1.,4),np.linspace(5200,5800,4))
        for row in rows:
            result=run_model(1.,row[0],row[1],.7,.008)
            self.assertEqual(int(row[2]),result['flag'])
            self.assertEqual(int(row[3]),result['error'])
            np.testing.assert_allclose(row[4:],list(diagnostics(result).values()),rtol=1e-12,atol=1e-12,equal_nan=True)
if __name__=='__main__':unittest.main()
