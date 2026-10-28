"""New example: Duffing and pendulum oscillators driven by fBm, solved with the general solver, against
exact-grid Monte Carlo (Davies-Harte fGn, Heun scheme).  Writes results/v12_new.json.
python code/run_new_v12.py"""
import sys, json, time
import numpy as np
sys.path.insert(0, 'v6/code'); sys.path.insert(1, 'apps')
import hwc, pathfun as pf
P, Eq = hwc.Poly, hwc.Equation
w0, z, kap, sg, Q0, V0, T = 1.0, 0.3, 1.0, 0.4, 1.0, 0.0, 10.0
duff = Eq([Q0, V0], P([[(1.0, (0, 1))], [(-w0 ** 2, (1, 0)), (-2 * z * w0, (0, 1)), (-kap, (3, 0))]]),
          P([[], [(sg, (0, 0))]]), name='Duffing oscillator')
pend_F = lambda t, X: np.stack([X[:, 1], -w0 ** 2 * np.sin(X[:, 0]) - 2 * z * w0 * X[:, 1]], 1)
pend_D = lambda t, X: np.stack([np.zeros(len(X)), np.full(len(X), sg)], 1)
pend = Eq([Q0, V0], pend_F, pend_D, name='pendulum')
tq = np.array([1.0, 2.0, 5.0, 10.0])

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
    return dict(mean=r.mean.tolist(), var=r.variance().tolist(), seconds=r.info['seconds'], engine=r.info['engine'],
                n=r.basis.n)
OUT = {'t': tq.tolist(), 'params': dict(omega=w0, zeta=z, kappa=kap, sigma=sg, Q0=Q0, V0=V0, T=T), 'duff': {}, 'pend': {}}
t00 = time.time()
for H in [0.6, 0.75, 0.9]:
    for J, p in [(65, 1), (17, 2), (33, 2), (65, 2), (9, 3), (17, 3)]:
        OUT['duff'][f'{H}_{J}_{p}'] = keep(hwc.solve(duff, H, T, J, p, tq))
    print('duff chaos', H, round(time.time() - t00), flush=True)
    for J in [17, 65]:
        OUT['pend'][f'{H}_{J}_1'] = keep(hwc.solve(pend, H, T, J, 1, tq))
    OUT['pend'][f'{H}_5_2'] = keep(hwc.solve(pend, H, T, 5, 2, tq, quad_order=7))
    print('pend chaos', H, round(time.time() - t00), flush=True)
for H in [0.6, 0.75, 0.9]:
    OUT['duff'][f'mc_{H}_1000'] = mc(H, lambda q: -w0 ** 2 * q - kap * q ** 3, 1000, 2 ** 17, 300 + int(100 * H))
    OUT['pend'][f'mc_{H}_1000'] = mc(H, lambda q: -w0 ** 2 * np.sin(q), 1000, 2 ** 17, 500 + int(100 * H))
    print('mc', H, round(time.time() - t00), flush=True)
OUT['duff']['mc_0.75_2000'] = mc(0.75, lambda q: -w0 ** 2 * q - kap * q ** 3, 2000, 2 ** 17, 400 + 75)
OUT['seconds_total'] = time.time() - t00
json.dump(OUT, open('v6/results/v12_new.json', 'w'))
print('done', round(time.time() - t00), flush=True)
