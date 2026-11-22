"""Problem 3: deterministic trajectory facts and refinement of the terminal reversal probability.

Writes evidence/P3_lotka_volterra/reversal_checks.json, read by make_p3_tables.py.
  * Deterministic system (lambda = 0): crossing time of x_1 and x_2, x(16) and the coexistence equilibrium.
  * P(X_1(16) > X_2(16)) from the CosChaos polynomial on common Gaussian draws:
      - the first 17 coordinates are the 2^20 draws of p3_lotka_volterra.py (seed 11), so the J = 17, p = 2
        values reproduce the published ones exactly; coordinates 17..64 are independent draws (seed 12);
      - degree two at J = 17, 33, 65 for every H; degree three at J = 9, 13, 17 for H = 0.95.
The script stops (exit status 1) if a stored value is not reproduced or a stated agreement fails.
Run from the scripts folder: python check_reversal.py   (about ten minutes on one core)
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import EVID  # noqa: E402

sys.path.insert(0, os.path.join(EVID, 'solver'))
import hfc  # noqa: E402
import hwc  # noqa: E402

P3 = os.path.join(EVID, 'P3_lotka_volterra')
R = json.load(open(os.path.join(P3, 'p3_results.json')))
PAR = R['params']
TIMES = np.array(R['times'], dtype=float)
NSAMP = 2 ** 20


def build_eq():
    return hwc.Equation(
        PAR['x0'],
        hwc.Poly([[(PAR['r1'], (1, 0)), (-PAR['a11'], (2, 0)), (-PAR['a12'], (1, 1))],
                  [(PAR['r2'], (0, 1)), (-PAR['a21'], (1, 1)), (-PAR['a22'], (0, 2))]]),
        hwc.Poly([[(PAR['lam'], (1, 0))], [(PAR['lam'], (0, 1))]]),
    )


def deterministic():
    def f(t, x):
        return [PAR['r1'] * x[0] - PAR['a11'] * x[0] ** 2 - PAR['a12'] * x[0] * x[1],
                PAR['r2'] * x[1] - PAR['a21'] * x[0] * x[1] - PAR['a22'] * x[1] ** 2]
    sol = solve_ivp(f, (0.0, PAR['T']), PAR['x0'], method='DOP853', rtol=1e-12, atol=1e-12, dense_output=True)
    g = lambda t: sol.sol(t)[0] - sol.sol(t)[1]
    grid = np.linspace(1e-6, PAR['T'], 16001)
    s = np.sign([g(t) for t in grid])
    k = int(np.where(np.diff(s) != 0)[0][0])
    cross = brentq(g, grid[k], grid[k + 1], xtol=1e-12)
    A = np.array([[PAR['a11'], PAR['a12']], [PAR['a21'], PAR['a22']]])
    eq = np.linalg.solve(A, [PAR['r1'], PAR['r2']])
    return dict(crossing_time=float(cross), x_T=[float(v) for v in sol.sol(PAR['T'])], equilibrium=eq.tolist(),
                crossings_on_grid=int(np.sum(np.diff(s) != 0)))


def p_dom(sol, xi, chunk):
    res_T = hwc.Result(sol.times[-1:], sol.coef[-1:], sol.basis, sol.noise, sol.form)
    Y = res_T.series(xi, chunk=chunk)[0]
    return float(np.mean(Y[:, 0] > Y[:, 1]))


def main():
    t_start = time.perf_counter()
    rng = np.random.default_rng(11)
    lead = rng.standard_normal((NSAMP, 17))          # identical to p3_lotka_volterra.py
    tail = np.random.default_rng(12).standard_normal((NSAMP, 48))
    xi = np.hstack((lead, tail))
    del lead, tail
    eq = build_eq()
    out = dict(samples=NSAMP, seeds=dict(leading_17=11, coordinates_17_to_64=12),
               deterministic=deterministic(), cases={})
    errors = []
    for H in ('0.55', '0.75', '0.95'):
        ref = R['cases'][H]['reference']
        stored = next(r['p_dom'] for r in R['cases'][H]['coschaos']['rows'] if r['J'] == 17 and r['p'] == 2)
        settings = [(17, 2), (33, 2), (65, 2)] + ([(9, 3), (13, 3), (17, 3)] if H == '0.95' else [])
        vals = {}
        for J, p in settings:
            hfc._rules.clear()
            sol = hwc.solve(eq, float(H), PAR['T'], J, p, TIMES, rtol=1e-8, atol=1e-10)
            vals[f'J{J}_p{p}'] = p_dom(sol, xi[:, :J], 65536 if p <= 2 else 8192)
            print(f'H={H} J={J} p={p}: P = {100 * vals[f"J{J}_p{p}"]:.4f}%', flush=True)
        ref_p, ref_se = ref['p_dom'], ref['p_dom_se']
        se = lambda q: math.sqrt(q * (1.0 - q) / NSAMP)
        z = {k: (v - ref_p) / math.sqrt(se(v) ** 2 + ref_se ** 2) for k, v in vals.items()}
        p2 = [vals[f'J{J}_p2'] for J in (17, 33, 65)]
        case = dict(reference=ref_p, reference_se=ref_se, stored_J17_p2=stored, values=vals, z_vs_reference=z,
                    spread_p2_over_J=max(p2) - min(p2))
        if H == '0.95':
            p3 = [vals[f'J{J}_p3'] for J in (9, 13, 17)]
            case['spread_p3_over_J'] = max(p3) - min(p3)
        out['cases'][H] = case
        # Checks behind the sentences of Section 4.3 and Appendix D
        if abs(vals['J17_p2'] - stored) > 3.0 / NSAMP:   # at most three of 2^20 near-ties may flip
            errors.append(f'H={H}: J=17, p=2 does not reproduce the stored value {stored}')
        if case['spread_p2_over_J'] > 5e-4:
            errors.append(f'H={H}: degree-two estimate moves by more than 0.05 points with J')
        if H in ('0.55', '0.75') and abs(z['J17_p2']) > 2.0:
            errors.append(f'H={H}: degree-two estimate differs from the reference by more than 2 SE')
        if H == '0.95':
            if abs(z['J17_p3']) > 2.0:
                errors.append('H=0.95: degree-three estimate differs from the reference by more than 2 SE')
            if case['spread_p3_over_J'] > 5e-4:
                errors.append('H=0.95: degree-three estimate moves by more than 0.05 points with J')
    out['seconds'] = time.perf_counter() - t_start
    out['checks_failed'] = errors
    json.dump(out, open(os.path.join(P3, 'reversal_checks.json'), 'w'), indent=1)
    print('wrote reversal_checks.json;', 'FAILED: ' + '; '.join(errors) if errors else 'all checks ok')
    return 1 if errors else 0


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        status = main()
    sys.exit(status)
