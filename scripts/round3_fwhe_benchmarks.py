"""Round 3 benchmarks: cosine-coordinate solver versus the fractional Wiener-Hermite expansion (FWHE) of
El-Beltagy, Etman and Maged, Chaos Solitons Fractals 156 (2022) 111847 [EEM22].

Examples (EEM22 numbering):
  B1  Ornstein-Uhlenbeck, dX = -lam X dt + sig dB_H (EEM22 Example 1 with linear drift; lam = 0 is the
      constant-drift case solved in closed form in EEM22, Var = sig^2 t^{2H}).
  B2  Wick growth / fractional Malthusian model, dX = mu X dt + sig X <> dB_H (EEM22 Example 2, Eqs. (70)-(76)).
  B3  Caputo memory model, dX + a D^alpha X dt = (mu + sig^2/2) X dt + sig X <> dB_H (EEM22 Eq. (78)).

References, in order of authority: exact variance (B1, B2); for B3 the FWHE first-order variance EEM22 Eq. (57)
evaluated with Gauss quadrature and Laplace-inverted Green functions (a high-accuracy numerical reference).
FWHE values are computed in two ways:
  * FWHE-analytic: EEM22 Eq. (55)/(57) with the analytic kernels (order k = 1, and k <= 5 for B2);
  * FWHE-GL: the numerical algorithm of EEM22 Section 4.2 - first-order Grunwald-Letnikov (GL) fractional
    integrals/derivatives, explicit Euler for the deterministic kernel equations, trapezoidal outer integral -
    on N uniform time steps.  Its accuracy is checked against the closed form Var = sig^2 t^{2H} (lam = 0).
'ours' uses the default ODE tolerances (rtol 1e-11); 'ours_fast' uses rtol 1e-7, atol 1e-9.
The solver is the v1.1.1 snapshot (evidence/round3_fwhe/solver_snapshot_v1.1.1).  CPU times are wall-clock times of
this implementation on one machine.  No random numbers are used.
Run from working/:  python scripts/round3_fwhe_benchmarks.py
"""
import json, math, os, sys, time
import numpy as np
from math import gamma as Gm

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, 'evidence', 'round3_fwhe'); sys.path.insert(0, os.path.join(EV, 'solver_snapshot_v1.1.1'))
import hfc, hwc, examples_v6 as E
P, Eq = hwc.Poly, hwc.Equation
OUT = {'note': __doc__}
log = lambda *a: print(*a, flush=True)


# ---------------------------------------------------------------- FWHE numerical algorithm (EEM22 Sec. 4.2)
def gl_int_w(beta, n):
    """weights of the first-order GL fractional integral J^beta: w_0 = 1, w_j = w_{j-1} (j-1+beta)/j."""
    w = np.empty(n); w[0] = 1.0
    for j in range(1, n):
        w[j] = w[j - 1] * (j - 1 + beta) / j
    return w


def gl_der_w(alpha, n):
    """weights of the first-order GL fractional derivative D^alpha: g_0 = 1, g_j = g_{j-1} (1 - (alpha+1)/j)."""
    g = np.empty(n); g[0] = 1.0
    for j in range(1, n):
        g[j] = g[j - 1] * (1.0 - (alpha + 1.0) / j)
    return g


def fwhe_gl_var(H, dt, kernel, idx):
    """Var_1[X(t_i)] = 2H Gamma(2H) int_0^t k(t,v) J^{2H-1}_v k(t,.) dv  (EEM22 Eq. (57)), first-order GL for J,
    trapezoidal rule for the outer integral.  kernel(i) returns k(t_i, t_n), n = 0..i."""
    beta = 2 * H - 1; w = gl_int_w(beta, max(idx) + 1); out = []
    for i in idx:
        if i == 0:
            out.append(0.0); continue
        k = kernel(i)
        Jk = dt ** beta * np.convolve(w[:i + 1], k)[:i + 1]
        f = k * Jk; out.append(2 * H * Gm(2 * H) * dt * (f.sum() - 0.5 * (f[0] + f[-1])))
    return np.array(out)


def euler_caputo(b, a, alpha, dt, n, y0=1.0, caputo=True):
    """explicit Euler + first-order GL for y' + a D^alpha y = b y, y(0) = y0 (the deterministic FWHE equations).
    caputo=True: Caputo derivative (mean X^(0)); caputo=False: Riemann-Liouville derivative of the function that
    vanishes before the jump, i.e. the impulse response g with Laplace transform 1/(z + a z^alpha - b)."""
    g = gl_der_w(alpha, n + 1); y = np.empty(n + 1); y[0] = y0; c = y0 if caputo else 0.0
    for m in range(n):
        d = dt ** (-alpha) * np.dot(g[:m + 1], (y[m::-1] - c))
        y[m + 1] = y[m] + dt * (b * y[m] - a * d)
    return y


def timed(fun, rep=1):
    t0 = time.perf_counter()
    for _ in range(rep):
        r = fun()
    return r, (time.perf_counter() - t0) / rep


Ns = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
Js = [5, 9, 17, 33, 65, 129]

# ---------------------------------------------------------------- B1: OU (and the constant-drift case lam = 0)
T = 2.0; tq = np.round(np.arange(0.25, T + 1e-9, 0.25), 10)
OUT['B1'] = {'T': T, 't': tq.tolist(), 'sig': 1.0, 'x0': 0.0}
for lam in [1.0, 0.0]:
    key = f'lam={lam:g}'; OUT['B1'][key] = {}
    ou = Eq(0.0, P([[(-lam, (1,))]]), P([[(1.0, (0,))]]), name='OU')
    for H in [0.55, 0.75, 0.95]:
        if lam == 0:
            ex = tq ** (2 * H)
        else:
            ex = np.array([E.ou_exact_var(H, lam, 1.0, float(t)) for t in tq])
        rec = {'exact': ex.tolist(), 'ours': {}, 'fwhe_gl': {}}
        for J in Js:
            r, s = timed(lambda: hwc.solve(ou, H, T, J, 1, tq))
            v = r.variance()[:, 0]; rec['ours'][str(J)] = dict(var=v.tolist(), seconds=s,
                                                            max_rel=float(np.max(np.abs(v / ex - 1))))
            # matched-accuracy timing: ODE tolerance 1e-7 (well below the truncation error; max_rel recorded)
            r, s = timed(lambda: hwc.solve(ou, H, T, J, 1, tq, rtol=1e-7, atol=1e-9))
            v = r.variance()[:, 0]; rec.setdefault('ours_fast', {})[str(J)] = dict(var=v.tolist(), seconds=s,
                                                            max_rel=float(np.max(np.abs(v / ex - 1))))
        for N in Ns:
            dt = T / N; idx = [int(round(t / dt)) for t in tq]
            ker = lambda i: (1.0 - lam * dt) ** (i - np.arange(i + 1))       # Euler kernel, k(t_n, t_n) = sig = 1
            v, s = timed(lambda: fwhe_gl_var(H, dt, ker, idx))
            rec['fwhe_gl'][str(N)] = dict(var=v.tolist(), seconds=s, max_rel=float(np.max(np.abs(v / ex - 1))))
        OUT['B1'][key][str(H)] = rec
        log('B1', key, H, {J: f"{rec['ours'][J]['max_rel']:.1e}" for J in rec['ours']},
            {N: f"{rec['fwhe_gl'][N]['max_rel']:.1e}" for N in rec['fwhe_gl']})

# ---------------------------------------------------------------- B2: Wick growth (EEM22 Example 2)
mu, sg, X0 = 0.2, 0.5, 1.0
wg = Eq(X0, P([[(mu, (1,))]]), P([[(sg, (1,))]]), name='Wick growth')
OUT['B2'] = {'T': T, 't': tq.tolist(), 'mu': mu, 'sig': sg, 'x0': X0}
for H in [0.6, 0.75, 0.95]:
    s2 = sg ** 2 * tq ** (2 * H); base = X0 ** 2 * np.exp(2 * mu * tq)
    rec = {'exact_wick': (base * np.expm1(s2)).tolist(),
           'eem22_eq72': (base * (np.exp(-sg ** 2 * tq + 2 * s2) - 2 * np.exp(-sg ** 2 * tq / 2 + s2 / 2) + 1)).tolist(),
           'fwhe_order': {str(k): (base * sum(s2 ** n / math.factorial(n) for n in range(1, k + 1))).tolist() for k in range(1, 6)},
           'ours': {}, 'fwhe_gl_k1': {}}
    for J, pmax in [(5, 5), (17, 5), (65, 3)]:
        r, s = timed(lambda: hwc.solve(wg, H, T, J, pmax, tq))
        c2 = r.coef[:, :, 0] ** 2; deg = r.basis.deg
        rec['ours'][str(J)] = {str(p): c2[:, (deg >= 1) & (deg <= p)].sum(1).tolist() for p in range(1, pmax + 1)}
        rec['ours'][str(J)]['seconds_pmax'] = s; rec['ours'][str(J)]['pmax'] = pmax
        for p in range(1, min(pmax, 3) + 1):                                   # time of each lower degree
            _, sp = timed(lambda: hwc.solve(wg, H, T, J, p, tq)); rec['ours'][str(J)][f'seconds_p{p}'] = sp
        log('B2 ours', H, J, pmax, round(s, 1))
    for N in Ns[:6]:
        dt = T / N; idx = [int(round(t / dt)) for t in tq]
        def ker(i, dt=dt):
            n = np.arange(i + 1); m0 = X0 * (1 + mu * dt) ** n                 # Euler mean X^(0)
            return sg * m0 * (1 + mu * dt) ** (i - n)                          # Euler kernel, k(t_n,t_n) = sig X^(0)(t_n)
        v, s = timed(lambda: fwhe_gl_var(H, dt, ker, idx))
        rec['fwhe_gl_k1'][str(N)] = dict(var=v.tolist(), seconds=s)
    OUT['B2'][str(H)] = rec

# ---------------------------------------------------------------- B3: Caputo memory model (EEM22 Eq. (78))
a_, al, mu_m, s_m, TT = 0.5, 0.5, 0.0198, 0.02, 20.0
b = mu_m + s_m ** 2 / 2
mem = Eq(1.0, P([[(b, (1,))]]), P([[(s_m, (1,))]]), memory=dict(a=a_, alpha=al), name='memory')
tl = np.array([4.0, 8.0, 12.0, 16.0, 20.0])
t0 = time.perf_counter(); tm, g, u = E.memory_kernels(n=801)
OUT['B3'] = {'T': TT, 't': tl.tolist(), 'a': a_, 'alpha': al, 'mu': mu_m, 'sig': s_m, 'x0': 1.0,
             'reference_kernel_seconds': time.perf_counter() - t0}
for H in [0.7, 0.95]:
    ref, sref = timed(lambda: np.array([E.fwhe57(H, float(t), tm, g, u) for t in tl]))
    rec = {'fwhe57_reference': ref.tolist(), 'reference_seconds': sref, 'ours': {}, 'fwhe_gl': {}}
    for J in [9, 17, 33, 65, 129]:
        r, s = timed(lambda: hwc.solve(mem, H, TT, J, 1, tl))
        v = r.variance()[:, 0]
        rec['ours'][str(J)] = dict(var=v.tolist(), seconds=s, max_rel=float(np.max(np.abs(v / ref - 1))))
    for N in [100, 200, 400, 800, 1600]:
        dt = TT / N; idx = [int(round(t / dt)) for t in tl]
        def run(dt=dt, N=N, idx=idx):
            uu = euler_caputo(b, a_, al, dt, N, 1.0); gg = euler_caputo(b, a_, al, dt, N, 1.0, caputo=False)
            ker = lambda i: s_m * gg[i - np.arange(i + 1)] * uu[:i + 1]
            return fwhe_gl_var(H, dt, ker, idx)
        v, s = timed(run)
        rec['fwhe_gl'][str(N)] = dict(var=v.tolist(), seconds=s, max_rel=float(np.max(np.abs(v / ref - 1))))
    OUT['B3'][str(H)] = rec
    log('B3', H, {J: f"{rec['ours'][J]['max_rel']:.1e}" for J in rec['ours']},
        {N: f"{rec['fwhe_gl'][N]['max_rel']:.1e}" for N in rec['fwhe_gl']})

json.dump(OUT, open(os.path.join(EV, 'benchmarks.json'), 'w'), indent=1)
log('written', os.path.join(EV, 'benchmarks.json'))
