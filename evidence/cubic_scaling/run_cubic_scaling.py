"""NEW computation for Paper 1 (27 Oct 2026; not part of V2.1): convergence in J and p and cost scaling of the
general solver for the cubic single well dX = (-X - X^3) dt + 0.5 dB_H, X(0) = 1, T = 4 (Example 'cubic').
Uses the archived solver snapshot (../solver_snapshot/code, hwc.solve) unchanged.
Runs the (J, p) pairs that are not stored in reference_results/v12_cubic.json, plus one stored pair
(H=0.75, J=17, p=2) as a reproduction check. Output: cubic_scaling.json next to this script."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.abspath(os.path.join(HERE, '..', 'solver_snapshot'))
sys.path.insert(0, os.path.join(SNAP, 'code'))
import hwc
P, Eq = hwc.Poly, hwc.Equation

def cubic(a, b, sig, x0):
    return Eq(x0, P([[(a, (1,)), (-b, (3,))]]), P([[(sig, (0,))]]), name='cubic drift')

times = np.array([0.5, 1.0, 2.0, 4.0])
runs = [(0.75, 17, 2)] + [(H, J, p) for (J, p) in [(5, 1), (9, 1), (5, 2), (5, 3), (5, 4), (17, 3)] for H in (0.6, 0.75, 0.9)]
OUT = {}
for H, J, p in runs:
    t0 = time.time(); r = hwc.solve(cubic(-1.0, 1.0, 0.5, 1.0), H, 4.0, J, p, times)
    OUT[f'{H}_{J}_{p}'] = dict(H=H, J=J, p=p, t=r.times.tolist(), mean=r.mean[:, 0].tolist(), var=r.variance()[:, 0].tolist(),
                              n=int(r.basis.n), seconds=float(r.info['seconds']), wall=time.time() - t0, engine=r.info['engine'])
    print(H, J, p, OUT[f'{H}_{J}_{p}']['n'], round(OUT[f'{H}_{J}_{p}']['wall'], 1), flush=True)
    json.dump(OUT, open(os.path.join(HERE, 'cubic_scaling.json'), 'w'), indent=1)
print('done')
