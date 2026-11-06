"""Round 3: grid refinement of the Caputo-memory benchmark (EEM22 Eq. (78)).

The FWHE first-order reference (EEM22 Eq. (57)) and the solver's memory branch both evaluate the Green function g
and the mean psi by Talbot inversion on a uniform grid and interpolate them with cubic splines.  This script refines
that grid (n = 401, 801, 1601) for the reference and (ngrid = 401, 1601) for the solver, so that the comparison is
not limited by a shared discretisation.  Deterministic; no random numbers.
Run from working/:  python scripts/round3_memory_reference.py
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, 'evidence', 'round3_fwhe'); sys.path.insert(0, os.path.join(EV, 'solver_snapshot_v1.1.1'))
import hfc, hwc, examples_v6 as E
P, Eq = hwc.Poly, hwc.Equation
a_, al, mu_m, s_m, TT = 0.5, 0.5, 0.0198, 0.02, 20.0
b = mu_m + s_m ** 2 / 2
mem = Eq(1.0, P([[(b, (1,))]]), P([[(s_m, (1,))]]), memory=dict(a=a_, alpha=al), name='memory')
tl = np.array([4.0, 8.0, 12.0, 16.0, 20.0]); Hs = [0.7, 0.95]
OUT = {'t': tl.tolist(), 'H': Hs, 'reference': {}, 'ours': {}}
for n in [401, 801, 1601]:
    t0 = time.perf_counter(); tm, g, u = E.memory_kernels(n=n); sk = time.perf_counter() - t0
    OUT['reference'][str(n)] = {'kernel_seconds': sk, **{str(H): [E.fwhe57(H, float(t), tm, g, u) for t in tl] for H in Hs}}
    print('reference', n, round(sk, 1), flush=True)
for ng in [401, 1601]:
    for J in [33, 129]:
        t0 = time.perf_counter()
        r = hwc.solve(mem, 0.7, TT, J, 1, tl, form='correlated', memory_opts={'ngrid': ng})
        s = time.perf_counter() - t0; a1 = r.coef[:, r.basis.lin, 0]
        rec = {'seconds_solve': s}
        for H in Hs:
            G = hfc.gram(H, TT, 'cos', J); rec[str(H)] = np.einsum('ti,ij,tj->t', a1, G, a1).tolist()
        OUT['ours'][f'{ng}_{J}'] = rec
        print('ours', ng, J, round(s, 1), flush=True)
json.dump(OUT, open(os.path.join(EV, 'memory_reference_refinement.json'), 'w'), indent=1)
print('written', flush=True)
