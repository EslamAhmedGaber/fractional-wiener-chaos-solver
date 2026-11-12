"""CosChaos on general downward-closed index sets for the cubic single well
dX = (aX - bX^3) dt + sigma dB_H (a = -1, b = 1, X0 = 1, sigma = 0.5, T = 4).

Same Galerkin coefficient law as hwc.solve (orthonormal form, exact Hermite products, DOP853, rtol 1e-8, atol 1e-10), but
the chaos basis Phi_alpha, |alpha| in S, may be any downward-closed set S of multi-indices (total degree TD(J, p), or
anisotropic sets with a degree budget per mode).  The cubic projection uses the exact product tensor of S (each unordered
pair once; values float64) written to a memory-mapped scratch folder (deleted at the end of the study).
Check: for S = TD(9, 4) the coefficients equal those of hwc.solve (stored in the output).
Samples: xi[:, :9] are the 10^6 coordinates of the earlier runs (seed 20261105 + 300 + iH, 20 batches of 50 000);
extra coordinates (j >= 9) come from SeedSequence([20261105, 5, 77, iH]).
Run from the repository root:  python scripts/round5_ladder.py <config-name> [<config-name> ...]
"""
import ctypes, json, math, os, shutil, sys, time
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'evidence', 'round3_fwhe', 'solver_v1.1.3'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solver'))   # release layout: solver/ next to scripts/
import hwc
OUT = os.path.join(ROOT, 'evidence', 'round5'); SCR = os.path.join(OUT, 'scratch'); os.makedirs(SCR, exist_ok=True)
RES = os.path.join(OUT, os.environ.get('LADDER_OUT', 'P5_ladder.json'))
HS = [0.55, 0.75, 0.95]; T = 4.0; SEED = 20261105; M = 1_000_000; NB = 20
LEV = np.round(np.arange(0.0, 1.2001, 0.01), 2); EDGES = np.linspace(-2.0, 2.0, 16001)
log = lambda *a: print(*a, flush=True)


def peak_mb():
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC); k32 = ctypes.windll.kernel32; ps = ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2 ** 20


# ---------------------------------------------------------------- index sets
def td_set(J, p):
    out = [()]
    def rec(prefix, start, left):
        for j in range(start, J):
            a = prefix + (j,); out.append(a)
            if left > 1: rec(a, j, left - 1)
    rec((), 0, p)
    A = np.zeros((len(out), J), np.int64)
    for i, a in enumerate(out):
        for j in a: A[i, j] += 1
    return A


def aniso_set(J, pj, ptot):
    """alpha with alpha_j <= pj[j], sum_j alpha_j / pj[j] <= 1 (weighted total degree) and |alpha| <= ptot."""
    pj = np.asarray(pj, float); rows = [np.zeros(J, np.int64)]
    def rec(a, j, budget, deg):
        for k in range(j, J):
            for c in range(1, int(pj[k]) + 1):
                nb = budget - c / pj[k]
                if nb < -1e-12 or deg + c > ptot: break
                b = a.copy(); b[k] = c; rows.append(b); rec(b, k + 1, nb, deg + c)
    rec(np.zeros(J, np.int64), 0, 1.0, 0)
    return np.array(rows)


def order(A):                                    # by total degree, then lexicographic: parents come first
    key = np.lexsort(tuple(A[:, j] for j in reversed(range(A.shape[1]))) + (A.sum(1),))
    return A[key]


# ---------------------------------------------------------------- exact product tensor of S (pairs once)
C1 = None
def c1_table(P):
    t = np.zeros((P + 1, P + 1, P + 1))
    for m in range(P + 1):
        for n in range(P + 1):
            for r in range(min(m, n) + 1): t[m, n, r] = hwc._c1(m, n, r)
    return t


def build_tensor(A, name):
    d = os.path.join(SCR, name)
    if os.path.exists(os.path.join(d, 'meta.json')):
        meta = json.load(open(os.path.join(d, 'meta.json')))
        return {k: np.load(os.path.join(d, k + '.npy'), mmap_mode='r') for k in 'IKNV'}, meta
    os.makedirs(d, exist_ok=True); t0 = time.perf_counter(); n, J = A.shape; P = int(A.max())
    C1t = c1_table(P); base = (2 * P + 1) ** np.arange(J, dtype=np.int64)
    tot = sum(int(np.prod(np.minimum(A[i], A[i:]) + 1, axis=1).sum()) for i in range(n))
    idt = np.int16 if n < 32767 else np.int32
    I = np.lib.format.open_memmap(os.path.join(d, 'I.npy'), 'w+', idt, (tot,)); K = np.lib.format.open_memmap(os.path.join(d, 'K.npy'), 'w+', idt, (tot,))
    KEY = np.lib.format.open_memmap(os.path.join(d, 'KEY.npy'), 'w+', np.int64, (tot,)); V = np.lib.format.open_memmap(os.path.join(d, 'V.npy'), 'w+', np.float64, (tot,))
    pos = 0; ukeys = []
    for i in range(n):
        a = A[i]; Kc = A[i:]; Mm = np.minimum(a, Kc) + 1; tk = np.prod(Mm, axis=1); Tn = int(tk.sum())
        kk = np.repeat(np.arange(i, n), tk); t = np.arange(Tn) - np.repeat(np.cumsum(tk) - tk, tk)
        Mr = Mm[kk - i]; stride = np.cumprod(Mr, axis=1) // Mr; r = (t[:, None] // stride) % Mr
        b = A[kk]; nu = a + b - 2 * r
        I[pos:pos + Tn] = i; K[pos:pos + Tn] = kk; KEY[pos:pos + Tn] = nu @ base
        V[pos:pos + Tn] = np.prod(C1t[np.broadcast_to(a, b.shape), b, r], axis=1); pos += Tn
        if i % 2000 == 0: ukeys = [np.unique(np.concatenate(ukeys + [KEY[max(0, pos - Tn):pos]]))]
        else: ukeys.append(np.unique(KEY[pos - Tn:pos]))
    q = np.unique(np.concatenate(ukeys)); n2 = len(q)
    N = np.lib.format.open_memmap(os.path.join(d, 'N.npy'), 'w+', np.int32, (tot,))
    for s in range(0, tot, 20_000_000):
        N[s:s + 20_000_000] = np.searchsorted(q, KEY[s:s + 20_000_000])
    # the constant and first-degree elements of S must sit at the front of Q in the same order (checked)
    keyS = A @ base; posS = np.searchsorted(q, keyS); assert np.all(q[posS] == keyS)
    for x in (I, K, N, V): x.flush()
    del KEY; os.remove(os.path.join(d, 'KEY.npy'))
    meta = dict(name=name, n=n, J=J, pmax=P, n2=n2, entries=tot, bytes=int(tot * (2 * np.dtype(idt).itemsize + 4 + 8)),
                seconds=time.perf_counter() - t0, posS=posS.tolist())
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'))
    return build_tensor(A, name)


@njit(cache=True)
def _sq(I, K, N, V, u, n2):
    out = np.zeros(n2)
    for e in range(I.shape[0]):
        i = I[e]; k = K[e]
        if i == k: out[N[e]] += V[e] * u[i] * u[i]
        else: out[N[e]] += 2.0 * V[e] * u[i] * u[k]
    return out


@njit(cache=True)
def _three(I, K, N, V, s, w, n):
    out = np.zeros(n)
    for e in range(I.shape[0]):
        i = I[e]; k = K[e]; vs = V[e] * s[N[e]]
        out[k] += vs * w[i]
        if i != k: out[i] += vs * w[k]
    return out


@njit(cache=True)
def _jac_proj(I, K, N, V, s, n):
    """M[k, i] = E[S Phi_i Phi_k] for S = sum_nu s_nu Phi_nu (the linearisation of the cube is 3 M(X^2))."""
    M = np.zeros((n, n))
    for e in range(I.shape[0]):
        i = I[e]; k = K[e]; v = V[e] * s[N[e]]
        M[k, i] += v
        if i != k: M[i, k] += v
    return M


def solve(A, Tt, meta, H, J, a=-1.0, b=1.0, x0=1.0, sig=0.5, times=(1.0, 4.0), method='DOP853'):
    noise = hwc.HarmonicNoise(H, T, J); n = A.shape[0]
    lin = np.array([int(np.flatnonzero((A.sum(1) == 1) & (A[:, j] == 1))[0]) for j in range(J)])
    I, K, N, V = Tt['I'], Tt['K'], Tt['N'], Tt['V']; n2 = meta['n2']; nf = [0]
    def rhs(t, c):
        nf[0] += 1; s = _sq(I, K, N, V, c, n2); f = a * c - b * _three(I, K, N, V, s, c, n)
        f[lin] += sig * noise.ell(t)[0]; return f
    c0 = np.zeros(n); c0[0] = x0; t0 = time.perf_counter()
    if method == 'DOP853':
        sol = solve_ivp(rhs, (0.0, T), c0, t_eval=np.array(times), method='DOP853', rtol=1e-8, atol=1e-10)
    else:                                       # implicit (stiff high-degree sets): Radau with the exact Jacobian
        jac = lambda t, c: a * np.eye(n) - 3.0 * b * _jac_proj(I, K, N, V, _sq(I, K, N, V, c, n2), n)
        sol = solve_ivp(rhs, (0.0, T), c0, t_eval=np.array(times), method='Radau', jac=jac, rtol=1e-8, atol=1e-10)
    assert sol.success, sol.message
    return sol.y.T, dict(nfev=nf[0], seconds=time.perf_counter() - t0, method=method, njev=int(getattr(sol, 'njev', 0) or 0))


def evaluate(A, C, xi):
    """X(xi) for coefficient rows C (nt, n): Phi_alpha by the parent recursion alpha = parent + c e_j (j = last nonzero)."""
    n, J = A.shape; P = int(A.max()); m = xi.shape[0]
    pos = {tuple(r): i for i, r in enumerate(A)}; par = np.zeros(n, int); cc = np.zeros(n, int); jj = np.zeros(n, int)
    for i in range(1, n):
        nz = np.flatnonzero(A[i]); j = nz[-1]; pa = A[i].copy(); pa[j] = 0; par[i] = pos[tuple(pa)]; cc[i] = A[i, j]; jj[i] = j
    hn = np.empty((P + 1, J, m)); hn[0] = 1.0; hn[1] = xi.T[:J]
    for k in range(1, P): hn[k + 1] = (xi.T[:J] * hn[k] - math.sqrt(k) * hn[k - 1]) / math.sqrt(k + 1)
    out = np.zeros((C.shape[0], m)); Phi = np.empty((n, m)); Phi[0] = 1.0
    for i in range(1, n): Phi[i] = Phi[par[i]] * hn[cc[i], jj[i]]
    return C @ Phi


def xi_batches(iH, Jmax):
    rng_old = np.random.default_rng(SEED + 300 + iH); rng_new = np.random.default_rng(np.random.SeedSequence([SEED, 5, 77, iH]))
    for b in range(NB):
        x = rng_old.standard_normal((M // NB, 9))
        extra = rng_new.standard_normal((M // NB, max(0, Jmax - 9)))
        yield np.hstack([x, extra]) if Jmax > 9 else x[:, :Jmax]


def law(A, C, iH, J, chunk=2000):
    t0 = time.perf_counter(); stats_b = []
    for xb in xi_batches(iH, J):
        xs = []
        for s in range(0, xb.shape[0], chunk): xs.append(evaluate(A, C, xb[s:s + chunk, :J]))
        x = np.concatenate(xs, axis=1)        # (nt, 50000)
        stats_b.append(dict(cnt4=(x[-1][:, None] > LEV).sum(0), cnt1=(x[0][:, None] > LEV).sum(0),
                            pw=np.array([[np.sum(x[r] ** k) for k in range(5)] for r in (0, 1)]),
                            hist=np.histogram(np.clip(x[-1], -2 + 1e-12, 2 - 1e-12), EDGES)[0]))
    return stats_b, time.perf_counter() - t0


CONFIGS = {
    'TD9_4': ('td', 9, 4), 'TD9_5': ('td', 9, 5), 'TD9_6': ('td', 9, 6), 'TD13_4': ('td', 13, 4), 'TD13_5': ('td', 13, 5),
    'TD17_4': ('td', 17, 4), 'TD11_5': ('td', 11, 5),
}


def make_set(cfg):
    if cfg[0] == 'td': return order(td_set(cfg[1], cfg[2]))
    return order(aniso_set(cfg[1], cfg[2], cfg[3]))


def run(name, cfg):
    R = json.load(open(RES)) if os.path.exists(RES) else {}
    if name in R and all(str(H) in R[name] for H in HS): return
    A = make_set(cfg); J = A.shape[1]; t0 = time.perf_counter(); Tt, meta = build_tensor(A, name)
    rec = R.get(name, dict(config=list(map(lambda v: v if not isinstance(v, np.ndarray) else v.tolist(), cfg)), n=int(A.shape[0]), J=J,
                           pmax=int(A.max()), entries=meta['entries'], tensor_bytes=meta['bytes'], tensor_seconds=meta['seconds']))
    for iH, H in enumerate(HS):
        if str(H) in rec: continue
        if os.environ.get('LADDER_H') and str(H) not in os.environ['LADDER_H'].split(','): continue
        C, info = solve(A, Tt, meta, H, J, method=os.environ.get('LADDER_METHOD', 'DOP853'))
        sb, se = law(A, C, iH, J)
        np.savez(os.path.join(OUT, f'ladder_{name}_H{H}.npz'), C=C, **{f'{k}_{b}': v for b, s in enumerate(sb) for k, v in s.items()})
        cnt = np.array([s['cnt4'] for s in sb]) / (M // NB)
        rec[str(H)] = dict(p4=cnt.mean(0).tolist(), ci4=(1.96 * cnt.std(0, ddof=1) / math.sqrt(NB)).tolist(),
                           seconds_solve=info['seconds'], nfev=info['nfev'], method=info['method'], njev=info['njev'], seconds_eval=se, peak_mb=peak_mb())
        R = json.load(open(RES)) if os.path.exists(RES) else {}; R[name] = rec; json.dump(R, open(RES, 'w'), indent=1)
        log(name, H, 'n', A.shape[0], 'entries', meta['entries'], 'solve', round(info['seconds']), 'nfev', info['nfev'], 'eval', round(se), 'peakMB', round(peak_mb()))


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        A = order(td_set(9, 4)); Tt, meta = build_tensor(A, 'TD9_4')
        C, _ = solve(A, Tt, meta, 0.95, 9)
        ref = hwc.solve(hwc.Equation(1.0, hwc.Poly([[(-1.0, (1,)), (-1.0, (3,))]]), hwc.Poly([[(0.5, (0,))]])), 0.95, T, 9, 4, np.array([1.0, 4.0]), rtol=1e-8, atol=1e-10)
        B = hwc.ChaosBasis(9, 4); mp = [B.pos[tuple(sorted(j for j in range(9) for _ in range(r[j])))] for r in A]
        diff = float(np.abs(C - ref.coef[:, mp, 0]).max())
        xb = next(xi_batches(2, 9))[:2000]; x1 = evaluate(A, C, xb)[-1]; x2 = (ref.series(xb)[:, :, 0])[-1]
        out = dict(coef_max_abs_diff=diff, eval_max_abs_diff=float(np.abs(x1 - x2).max()))
        json.dump(out, open(os.path.join(OUT, 'P5_ladder_check.json'), 'w'), indent=1); log('check', out)
    else:
        extra = json.load(open(os.path.join(OUT, 'ladder_configs.json'))) if os.path.exists(os.path.join(OUT, 'ladder_configs.json')) else {}
        for name in sys.argv[1:]:
            cfg = CONFIGS.get(name) or tuple(extra[name]); run(name, cfg)
        log('done', sys.argv[1:])
