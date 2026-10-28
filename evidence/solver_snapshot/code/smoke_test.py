"""Smoke test of the archived solver snapshot (run from solver_snapshot/code):  python smoke_test.py
Re-solves the OU example (H=0.75, T=2, lambda=sigma=1, X0=0, J=17, p=1) in both forms and compares the variance at
t = 0.5, 1, 2 with the stored round-2 values (reference_results/r2_algorithm_check.json, key ou_H0.75_J17)."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hfc, hwc
here = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(os.path.join(here, '..', 'reference_results', 'r2_algorithm_check.json')))['ou_H0.75_J17']
P, Eq = hwc.Poly, hwc.Equation
ou = Eq(0.0, P([[(-1.0, (1,))]]), P([[(1.0, (0,))]]), name='OU')
times = np.array([0.5, 1.0, 2.0])
ro = hwc.solve(ou, 0.75, 2.0, 17, 1, times)                         # orthonormal form
rc = hwc.solve(ou, 0.75, 2.0, 17, 1, times, form='correlated')      # correlated form
G = hfc.gram(0.75, 2.0, 'cos', 17); a = rc.coef[:, rc.basis.lin, 0]
vc = np.einsum('ti,ij,tj->t', a, G, a); vo = ro.variance()[:, 0]
print('orthonormal :', vo); print('correlated  :', vc); print('stored      :', ref['solver']); print('exact (ref) :', ref['exact'])
assert np.allclose(vo, ref['solver'], rtol=1e-9, atol=1e-12) and np.allclose(vc, vo, rtol=1e-10, atol=1e-12)
print('SMOKE TEST PASSED')
