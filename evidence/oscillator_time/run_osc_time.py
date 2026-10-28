"""NEW computation for V5 (27 Oct 2026): variance histories of the Duffing and pendulum oscillators (Example 21).
The Monte Carlo function and its settings are copied from code/run_new_v12.py (Davies-Harte fGn, Heun, n=1000 steps,
M=2^17 paths, chunk 4096, seeds 300+100H (Duffing) and 500+100H (pendulum)). The only change is the set of output
times (every 0.1 on [0,10]). The random stream does not depend on the output times, so the values at t=1,2,5,10 must
reproduce the stored v12_new.json entries; the maximum relative difference is recorded at the end.
Solver runs use hwc.solve unchanged at the same output times: Duffing p=1,2 (J=65) and p=3 (J=17); pendulum p=1 (J=65).
Output: osc_time.json next to this script."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.abspath(os.path.join(HERE, '..', 'solver_snapshot'))
sys.path.insert(0, os.path.join(SP, 'code'))
import hwc, pathfun as pf
P, Eq = hwc.Poly, hwc.Equation
w0, z, kap, sg, Q0, V0, T = 1.0, 0.3, 1.0, 0.4, 1.0, 0.0, 10.0
duff = Eq([Q0, V0], P([[(1.0, (0, 1))], [(-w0 ** 2, (1, 0)), (-2 * z * w0, (0, 1)), (-kap, (3, 0))]]),
          P([[], [(sg, (0, 0))]]), name='Duffing oscillator')
pend_F = lambda t, X: np.stack([X[:, 1], -w0 ** 2 * np.sin(X[:, 0]) - 2 * z * w0 * X[:, 1]], 1)
pend_D = lambda t, X: np.stack([np.zeros(len(X)), np.full(len(X), sg)], 1)
pend = Eq([Q0, V0], pend_F, pend_D, name='pendulum')
tq = np.round(np.arange(1, 101) * 0.1, 10)


def mc(H, force, n, M, seed, chunk=4096):
    sq = pf.dh_setup(H, n); rng = np.random.default_rng(seed); dt = T / n
    idx = np.rint(tq / dt).astype(int); assert np.allclose(idx * dt, tq)
    S = np.zeros((len(tq), 2, 5)); done = 0
    while done < M:
        m = min(chunk, M - done)
        dB = np.diff(pf.dh_paths(sq, n, H, m, rng), axis=1) * (sg * T ** H)
        Q = np.full(m, Q0); V = np.full(m, V0); j = 0
        for i in range(n):
            aQ, aV = V, force(Q) - 2 * z * w0 * V
            Qp, Vp = Q + aQ * dt, V + aV * dt + dB[:, i]
            bQ, bV = Vp, force(Qp) - 2 * z * w0 * Vp
            Q, V = Q + 0.5 * (aQ + bQ) * dt, V + 0.5 * (aV + bV) * dt + dB[:, i]
            while j < len(idx) and idx[j] == i + 1:
                for k, Xk in enumerate((Q, V)):
                    S[j, k] += [m, Xk.sum(), (Xk ** 2).sum(), (Xk ** 3).sum(), (Xk ** 4).sum()]
                j += 1
        done += m
    N = S[..., 0]; e1, e2, e3, e4 = (S[..., k] / N for k in range(1, 5))
    var = e2 - e1 ** 2; mu4 = e4 - 4 * e1 * e3 + 6 * e1 ** 2 * e2 - 3 * e1 ** 4
    return dict(mean=e1.tolist(), var=var.tolist(), se_var=np.sqrt((mu4 - var ** 2) / N).tolist(),
                se_mean=np.sqrt(var / N).tolist(), n=n, M=int(N[0, 0]))


def keep(r):
    return dict(mean=r.mean.tolist(), var=r.variance().tolist(), seconds=r.info['seconds'], engine=r.info['engine'], n=r.basis.n)

OUT = {'t': tq.tolist(), 'params': dict(omega=w0, zeta=z, kappa=kap, sigma=sg, Q0=Q0, V0=V0, T=T), 'duff': {}, 'pend': {}}
outp = os.path.join(HERE, 'osc_time.json'); t00 = time.time()
for H in [0.6, 0.75, 0.9]:
    OUT['duff'][f'mc_{H}_1000'] = mc(H, lambda q: -w0 ** 2 * q - kap * q ** 3, 1000, 2 ** 17, 300 + int(100 * H))
    OUT['pend'][f'mc_{H}_1000'] = mc(H, lambda q: -w0 ** 2 * np.sin(q), 1000, 2 ** 17, 500 + int(100 * H))
    print('mc', H, round(time.time() - t00), flush=True); json.dump(OUT, open(outp, 'w'))
for H in [0.6, 0.75, 0.9]:
    for J, p in [(65, 1), (65, 2), (17, 3)]:
        OUT['duff'][f'{H}_{J}_{p}'] = keep(hwc.solve(duff, H, T, J, p, tq))
    OUT['pend'][f'{H}_65_1'] = keep(hwc.solve(pend, H, T, 65, 1, tq))
    print('chaos', H, round(time.time() - t00), flush=True); json.dump(OUT, open(outp, 'w'))
ref = json.load(open(os.path.join(SP, 'reference_results', 'v12_new.json')))
ii = [int(np.argmin(abs(tq - t))) for t in ref['t']]; chk = {}
for sys_ in ('duff', 'pend'):
    for k, v in ref[sys_].items():
        if k in OUT[sys_]:
            a = np.array(v['var']); b = np.array(OUT[sys_][k]['var'])[ii]
            chk[f'{sys_}/{k}'] = float(np.max(np.abs(a - b) / np.abs(a)))
OUT['reproduction_max_rel_diff_at_stored_times'] = chk; OUT['seconds_total'] = time.time() - t00
json.dump(OUT, open(outp, 'w'), indent=0)
print('reproduction', json.dumps(chk), flush=True)
