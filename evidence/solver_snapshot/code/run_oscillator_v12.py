"""Oscillator example with the general solver (vector state, correlated H-independent form).
The reference covariances (singular-kernel quadrature, Lyapunov at H=1/2) are copied from results/oscillator_v9.json.
Writes results/v12_oscillator.json.  python code/run_oscillator_v12.py"""
import sys, json, time
import numpy as np
sys.path.insert(0, 'v6/code')
import hwc
P, Eq = hwc.Poly, hwc.Equation
O = json.load(open('v6/results/oscillator_v9.json'))
w0, z, sg, Q0, V0, T = (O['params'][k] for k in ['omega', 'zeta', 'sigma', 'Q0', 'V0', 'T'])
osc = Eq([Q0, V0], P([[(1.0, (0, 1))], [(-w0 ** 2, (1, 0)), (-2 * z * w0, (0, 1))]]), P([[], [(sg, (0, 0))]]),
         name='forced oscillator')
K = np.diag([w0 ** 2, 1.0])
def energy(m, S):
    return 0.5 * (m @ K @ m + np.trace(K @ S)), m @ K @ S @ K @ m + 0.5 * np.trace(K @ S @ K @ S)
REF = {k: np.array(v) for k, v in O['ref'].items()}
Js = [17, 33, 65, 129, 257, 513]; tr = np.array([2.0, 5.0, 10.0, 15.0, 20.0])
R = {J: hwc.solve(osc, 0.5, T, J, 1, tr, form='correlated') for J in Js}
rows = []
for H in [0.5, 0.7, 0.9]:
    for t in [2.0, 10.0, 20.0]:
        i = list(tr).index(t); S = REF[f'{H}_{t}']; m = R[513].mean[i]
        e, ve = energy(m, S)
        err = lambda J, a: float(abs(R[J].covariance(H)[i][a, a] / S[a, a] - 1))
        rows.append(dict(H=H, t=t, varQ=S[0, 0], varV=S[1, 1], covQV=S[0, 1], EE=e, VarE=ve, ratio=w0 ** 2 * S[0, 0] / S[1, 1],
                         errQ={str(J): err(J, 0) for J in [65, 129, 257]}, errV={str(J): err(J, 1) for J in [65, 129, 257]}))
conv = {f'{H}_{t}': [float(abs(R[J].covariance(H)[list(tr).index(t)][1, 1] / REF[f'{H}_{t}'][1, 1] - 1)) for J in Js]
        for H in [0.5, 0.7, 0.9] for t in [10.0, 20.0]}
convQ = {f'{H}_{t}': [float(abs(R[J].covariance(H)[list(tr).index(t)][0, 0] / REF[f'{H}_{t}'][0, 0] - 1)) for J in Js]
         for H in [0.5, 0.7, 0.9] for t in [10.0, 20.0]}
t0 = time.time(); C257 = hwc.solve(osc, 0.5, T, 257, 1, np.array([20.0]), form='correlated'); tC = time.time() - t0
Hs = np.round(np.arange(0.5, 0.951, 0.025), 3); sweep = []; t0 = time.time()
for H in Hs:
    S = C257.covariance(float(H))[0]; e, ve = energy(C257.mean[0], S)
    sweep.append(dict(H=float(H), varQ=S[0, 0], varV=S[1, 1], EE=e, VarE=ve, ratio=S[0, 0] * w0 ** 2 / S[1, 1]))
tG = time.time() - t0
tgrid = np.linspace(0, 20, 81); C513 = hwc.solve(osc, 0.5, T, 513, 1, tgrid, form='correlated')
curves = {str(H): dict(varQ=C513.covariance(H)[:, 0, 0].tolist(), varV=C513.covariance(H)[:, 1, 1].tolist()) for H in [0.5, 0.7, 0.9]}
json.dump(dict(params=O['params'], rows=rows, sweep=sweep, curves=curves, tgrid=tgrid.tolist(), ref=O['ref'], conv=conv,
               convQ=convQ, Js=Js, timing=dict(coef_J257_s=tC, sweep_19H_s=tG)), open('v6/results/v12_oscillator.json', 'w'),
          default=float)
print('done', tC, tG, flush=True)
