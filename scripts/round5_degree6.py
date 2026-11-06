"""Round 5 (optional run): CosChaos J = 9, p = 6 for the cubic event P(X(4) > c), H = 0.55, 0.75, 0.95.

The exact product tensor of hwc (Phi_a Phi_g = sum_nu V Phi_nu) has 9.4e7 stored entries at J = 9, p = 6; hwc builds it
in Python lists (about 6 GB).  This script builds the SAME entries without lists and stores each unordered pair {a, g}
once (4.7e7 entries; indices int16/int16/int32, values float64, about 0.75 GB, memory-mapped from disk), and evaluates
the two products used by the cubic projection with compiled scatter loops (numba).  The arithmetic is unchanged
(float64, the same Hermite product coefficients); only storage and loop order differ.  hwc.product_tensor,
hwc._prod_full, hwc._prod_low and hwc._prod3_low are replaced for this run only; solver_v1.1.3 is not modified.
Check: at J = 9, p = 4 the low-memory engine reproduces the coefficients of the original engine (stored in the output).
Sampling: 10^6 xi, seed 20261105 + 300 + iH as in round3c P3 / round4 (same xi as the p <= 5 runs).
Output: evidence/round5/P5_cubic_event_p6.json (incremental).  Run from working/:  python scripts/round5_degree6.py
"""
import json, math, os, sys, time, itertools
import numpy as np
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'evidence', 'round3_fwhe', 'solver_v1.1.3'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solver'))   # release layout: solver/ next to scripts/
import hwc
import round4_probability as R4
OUTD = os.path.join(ROOT, 'evidence', 'round5'); os.makedirs(OUTD, exist_ok=True)
log = lambda *a: print(*a, flush=True)
C1 = np.zeros((13, 13, 13))
for m in range(13):
    for n in range(13):
        for r in range(min(m, n) + 1):
            C1[m, n, r] = hwc._c1(m, n, r)


def counts(B, J):
    A = np.zeros((B.n, J), np.int64)
    for i, a in enumerate(B.idx):
        for j in a: A[i, j] += 1
    return A


def build(J, p):
    d = os.path.join(OUTD, f'tensor_J{J}_p{p}')
    if os.path.exists(os.path.join(d, 'V.npy')):
        return {k: np.load(os.path.join(d, k + '.npy'), mmap_mode='r') for k in 'IKNV'}, json.load(open(os.path.join(d, 'meta.json')))
    os.makedirs(d, exist_ok=True); t0 = time.perf_counter()
    B = hwc.ChaosBasis(J, p); B2 = hwc.ChaosBasis(J, 2 * p); A = counts(B, J); A2 = counts(B2, J)
    base = 13 ** np.arange(J, dtype=np.int64); key2 = A2 @ base; o2 = np.argsort(key2); ks2 = key2[o2]
    tot = sum(int(np.prod(np.minimum(A[i], A[i:]) + 1, axis=1).sum()) for i in range(B.n))
    I = np.lib.format.open_memmap(os.path.join(d, 'I.npy'), 'w+', np.int16, (tot,)); K = np.lib.format.open_memmap(os.path.join(d, 'K.npy'), 'w+', np.int16, (tot,))
    N = np.lib.format.open_memmap(os.path.join(d, 'N.npy'), 'w+', np.int32, (tot,)); V = np.lib.format.open_memmap(os.path.join(d, 'V.npy'), 'w+', np.float64, (tot,))
    pos = 0
    for i in range(B.n):
        a = A[i]; Kc = A[i:]; M = np.minimum(a, Kc) + 1; tk = np.prod(M, axis=1); T = int(tk.sum())
        kk = np.repeat(np.arange(i, B.n), tk); t = np.arange(T) - np.repeat(np.cumsum(tk) - tk, tk)
        Mr = M[kk - i]; stride = np.cumprod(Mr, axis=1) // Mr; r = (t[:, None] // stride) % Mr
        b = A[kk]; nu = a + b - 2 * r
        val = np.prod(C1[np.broadcast_to(a, b.shape), b, r], axis=1)
        n_ = o2[np.searchsorted(ks2, nu @ base)]
        I[pos:pos + T] = i; K[pos:pos + T] = kk; N[pos:pos + T] = n_; V[pos:pos + T] = val; pos += T
    assert pos == tot
    for x in (I, K, N, V): x.flush()
    meta = dict(J=J, p=p, n=B.n, n2=B2.n, entries=tot, seconds=time.perf_counter() - t0)
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w')); del I, K, N, V
    return build(J, p)


@njit(cache=True)
def _full(I, K, N, V, u, v, n2):
    out = np.zeros(n2)
    for e in range(I.shape[0]):
        i = I[e]; k = K[e]
        if i == k: out[N[e]] += V[e] * u[i] * v[i]
        else: out[N[e]] += V[e] * (u[i] * v[k] + u[k] * v[i])
    return out


@njit(cache=True)
def _three(I, K, N, V, s, w, n):
    out = np.zeros(n)
    for e in range(I.shape[0]):
        i = I[e]; k = K[e]; vs = V[e] * s[N[e]]
        out[k] += vs * w[i]
        if i != k: out[i] += vs * w[k]
    return out


def install(T, meta):
    def product_tensor(J, p):
        assert (J, p) == (meta['J'], meta['p']); return dict(T=T, n2=meta['n2'], B=None)
    hwc.product_tensor = product_tensor
    hwc._prod_full = lambda Tt, u, v: _full(T['I'], T['K'], T['N'], T['V'], np.ascontiguousarray(u, float), np.ascontiguousarray(v, float), Tt['n2'])
    hwc._prod_low = lambda Tt, u, v: hwc._prod_full(Tt, u, v)[:Tt['B'].n]
    hwc._prod3_low = lambda Tt, s, w: _three(T['I'], T['K'], T['N'], T['V'], np.ascontiguousarray(s, float), np.ascontiguousarray(w, float), Tt['B'].n)


if __name__ == '__main__':
    path = os.path.join(OUTD, 'P5_cubic_event_p6.json'); R = json.load(open(path)) if os.path.exists(path) else {}
    R['meta'] = dict(times=R4.TIMES.tolist(), levels=R4.LEVELS.tolist(), M=R4.M, batches=R4.NB, engine='low-memory exact product tensor (this script)')
    if 'check_p4' not in R:                      # engine check against the original hwc tensor at J = 9, p = 4
        orig = (hwc.product_tensor, hwc._prod_full, hwc._prod_low, hwc._prod3_low)
        ref = hwc.solve(R4.cubic_eq(), 0.95, R4.T, 9, 4, R4.TIMES, rtol=1e-8, atol=1e-10).coef
        T4, m4 = build(9, 4); install(T4, m4)
        new = hwc.solve(R4.cubic_eq(), 0.95, R4.T, 9, 4, R4.TIMES, rtol=1e-8, atol=1e-10).coef
        R['check_p4'] = dict(max_abs_diff=float(np.abs(new - ref).max()), max_abs_coef=float(np.abs(ref).max()), entries=m4['entries'])
        hwc.product_tensor, hwc._prod_full, hwc._prod_low, hwc._prod3_low = orig
        json.dump(R, open(path, 'w'), indent=1); log('check p4', R['check_p4'])
    T6, m6 = build(9, 6); R['tensor_p6'] = m6; log('tensor p6', m6); install(T6, m6); json.dump(R, open(path, 'w'), indent=1)
    for H, iH in [(0.95, 2), (0.75, 1), (0.55, 0)]:
        if str(H) in R: continue
        R[str(H)] = {'chaos_9,6': R4.chaos(H, 9, 6, R4.SEED + 300 + iH)}
        log('chaos', H, round(R[str(H)]['chaos_9,6']['seconds_solve'], 1), round(R[str(H)]['chaos_9,6']['seconds_eval'], 1))
        json.dump(R, open(path, 'w'), indent=1)
    log('done')
