"""Round 3c: verification of the full-solution capability (pathwise solution and probability law).

The solver returns X_{J,p}(t) = sum_alpha c_alpha(t) Phi_alpha(xi), an explicit function of the Gaussian coordinates
xi = L^{-1} Z of the fBm path, Z_k = int_0^T h_k dB_H.  This script checks, for H = 0.55, 0.75, 0.95:
  P1  pathwise convergence on common fBm paths (Davies-Harte, piecewise-linear between grid nodes; Z_k is the exact
      integral of h_k against that path, so the same path gives the same xi for the chaos and for the exact solution):
      Wick growth  X = X0 exp(mu t + sig B_H(t) - sig^2 t^{2H}/2)   (mu = 0.2, sig = 0.5, X0 = 1, T = 2)
      OU           X = sig int_0^t e^{-lam(t-s)} dB_H(s)              (lam = sig = 1, X0 = 0, T = 2)
      cubic drift  dX = (aX - bX^3)dt + sig dB_H, Heun on the path grid (a = -1, b = 1, X0 = 1, sig = 0.5, T = 4)
      metric: root-mean-square error over the paths, relative to the RMS of the reference, at t = 1 and at t = T;
  P2  probability law of Wick growth against the exact lognormal law: mean, variance, skewness, excess kurtosis,
      quantiles (1, 5, 95, 99 %) and density, for degrees p = 1..8 (law of the surrogate by Gauss-Hermite quadrature
      and a fine eta grid, exact for the surrogate), plus 10^6 surrogate samples (J = 9, p = 5) evaluated with the solver's series;
  P3  probability law of the cubic model against Monte Carlo with 10^6 exact (Davies-Harte) fBm paths and the Heun
      scheme (512 and 256 steps), with 95 % confidence intervals (batch means, 20 batches), and the cost of the
      surrogate law (one solve + 10^6 evaluations) against the cost of the Monte Carlo law.
Seeds are fixed (numpy default_rng, seed 20261105 + offsets).  Solver: evidence/round3_fwhe/solver_v1.1.3.
Run from working/:  python scripts/round3c_fullsolution.py [P1|P2|P2b|P3]
"""
import json, math, os, sys, time
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'evidence', 'round3_fwhe', 'solver_v1.1.3'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solver'))   # release layout: solver/ next to scripts/
import hwc, examples_v6 as E
P, Eq = hwc.Poly, hwc.Equation
OUTD = os.path.join(ROOT, 'evidence', 'round3_full'); os.makedirs(OUTD, exist_ok=True)
log = lambda *a: print(*a, flush=True)
HS = [0.55, 0.75, 0.95]; SEED = 20261105


def fgn(H, n, M, dt, rng):
    """M exact fractional Gaussian noise samples of length n (Davies-Harte circulant embedding), scaled to step dt."""
    k = np.arange(n + 1); g = 0.5 * ((k + 1.) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1.) ** (2 * H)); g[0] = 1.0
    lam = np.fft.fft(np.concatenate([g, g[-2:0:-1]])).real; assert lam.min() > -1e-9; lam = np.clip(lam, 0, None)
    m = 2 * n; Mh = (M + 1) // 2
    W = np.fft.fft(np.sqrt(lam / m) * (rng.standard_normal((Mh, m)) + 1j * rng.standard_normal((Mh, m))), axis=1)[:, :n]
    return np.concatenate([W.real, W.imag])[:M] * dt ** H


def coords(noise, dB, dt):
    """xi = L^{-1} Z with Z_k = int h_k dB for the piecewise-linear path with increments dB on a uniform grid."""
    n = dB.shape[1]; tg = np.linspace(0, noise.T, n + 1); dm = np.diff(noise.m(tg), axis=0)
    Z = (dB / dt) @ dm
    return np.linalg.solve(noise.L, Z.T).T


def series(res, xi, chunk=500):
    out = []
    for s in range(0, xi.shape[0], chunk):
        out.append(res.series(xi[s:s + chunk])[:, :, 0])
    return np.concatenate(out, axis=1)          # (nt, m)


def wick_eq(): return Eq(1.0, P([[(0.2, (1,))]]), P([[(0.5, (1,))]]), name='Wick growth')
def ou_eq(): return Eq(0.0, P([[(-1.0, (1,))]]), P([[(1.0, (0,))]]), name='OU')
def cubic_eq(): return Eq(1.0, P([[(-1.0, (1,)), (-1.0, (3,))]]), P([[(0.5, (0,))]]), name='cubic')


def heun_cubic(dB, dt, x0=1.0, a=-1.0, b=1.0, s=0.5, keep=None):
    x = np.full(dB.shape[0], x0); f = lambda y: a * y - b * y ** 3; out = {}
    for i in range(dB.shape[1]):
        xt = x + f(x) * dt + s * dB[:, i]; x = x + 0.5 * (f(x) + f(xt)) * dt + s * dB[:, i]
        if keep is not None and (i + 1) in keep: out[i + 1] = x.copy()
    return out


def rel_rms(a, b): return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


# ------------------------------------------------------------------ P1 pathwise
def P1():
    R = {}; M = 2000
    for iH, H in enumerate(HS):
        rng = np.random.default_rng(SEED + iH)
        # Wick and OU on [0, 2]
        T = 2.0; n = 4096; dt = T / n; dB = fgn(H, n, M, dt, rng); B = np.cumsum(dB, 1)
        tq = np.array([1.0, 2.0]); iq = [int(round(t / dt)) - 1 for t in tq]
        Xw = 1.0 * np.exp(0.2 * tq[None, :] + 0.5 * B[:, iq] - 0.5 * 0.25 * tq[None, :] ** (2 * H))     # (M, 2)
        tg = np.linspace(0, T, n + 1)
        Xo = np.stack([((np.exp(-(t - tg[1:i + 2])) - np.exp(-(t - tg[:i + 1]))) * (dB[:, :i + 1] / dt)).sum(1)
                       for t, i in zip(tq, iq)], 1)
        rw, ro = {}, {}
        for J in [5, 9, 17, 33, 65, 129]:
            noise = hwc.HarmonicNoise(H, T, J); xi = coords(noise, dB, dt)
            for p in range(1, (5 if J <= 17 else 3 if J <= 33 else 2) + 1):
                t0 = time.perf_counter(); r = hwc.solve(wick_eq(), H, T, J, p, tq, rtol=1e-10, atol=1e-12); s = time.perf_counter() - t0
                X = series(r, xi, chunk=200 if J >= 33 else 1000)
                rw[f'{J},{p}'] = dict(J=J, p=p, t1=rel_rms(X[0], Xw[:, 0]), tT=rel_rms(X[1], Xw[:, 1]), seconds=s)
            r = hwc.solve(ou_eq(), H, T, J, 1, tq, rtol=1e-10, atol=1e-12); X = series(r, xi)
            ro[str(J)] = dict(t1=rel_rms(X[0], Xo[:, 0]), tT=rel_rms(X[1], Xo[:, 1]))
        for J in [257, 513]:
            noise = hwc.HarmonicNoise(H, T, J); xi = coords(noise, dB, dt)
            r = hwc.solve(ou_eq(), H, T, J, 1, tq, rtol=1e-10, atol=1e-12); X = series(r, xi)
            ro[str(J)] = dict(t1=rel_rms(X[0], Xo[:, 0]), tT=rel_rms(X[1], Xo[:, 1]))
        # five example paths for the figure (Wick, J = 33, p = 1..3) on a dense time grid
        td = np.linspace(0, T, 81)[1:]; idd = [int(round(t / dt)) - 1 for t in td]
        noise = hwc.HarmonicNoise(H, T, 33); xi5 = coords(noise, dB[:5], dt)
        paths = {'t': td.tolist(), 'exact': (np.exp(0.2 * td[None, :] + 0.5 * B[:5][:, idd] - 0.125 * td[None, :] ** (2 * H))).tolist()}
        for p in (1, 2, 3):
            r = hwc.solve(wick_eq(), H, T, 33, p, td, rtol=1e-10, atol=1e-12); paths[f'p{p}'] = series(r, xi5).T.tolist()
        # cubic on [0, 4]
        T4 = 4.0; n4 = 4096; dt4 = T4 / n4; dB4 = fgn(H, n4, M, dt4, rng)
        ref = heun_cubic(dB4, dt4, keep={1024, 4096}); Xc = np.stack([ref[1024], ref[4096]], 1)
        rc = {}
        for J, ps in [(5, [1, 2, 3, 4]), (9, [1, 2, 3, 4]), (17, [1, 2, 3]), (33, [1, 2]), (65, [1, 2])]:
            noise = hwc.HarmonicNoise(H, T4, J); xi = coords(noise, dB4, dt4)
            for p in ps:
                t0 = time.perf_counter(); r = hwc.solve(cubic_eq(), H, T4, J, p, np.array([1.0, 4.0]), rtol=1e-8, atol=1e-10); s = time.perf_counter() - t0
                X = series(r, xi, chunk=200 if J >= 33 else 1000)
                rc[f'{J},{p}'] = dict(J=J, p=p, t1=rel_rms(X[0], Xc[:, 0]), tT=rel_rms(X[1], Xc[:, 1]), seconds=s)
            log('P1 cubic', H, J, {k: f"{v['t1']:.1e}/{v['tT']:.1e}" for k, v in rc.items() if v['J'] == J})
        R[str(H)] = dict(wick=rw, ou=ro, cubic=rc, paths=paths, M=M, n_wick=n, n_cubic=n4)
        log('P1', H, 'wick', {k: f"{v['t1']:.1e}/{v['tT']:.1e}" for k, v in rw.items()}, 'ou', {k: f"{v['t1']:.1e}" for k, v in ro.items()})
        json.dump(R, open(os.path.join(OUTD, 'P1_pathwise.json'), 'w'), indent=1)


# ------------------------------------------------------------------ P2 law of Wick growth
def lognormal_law(t, H, mu=0.2, s=0.5, x0=1.0):
    v = s ** 2 * t ** (2 * H); m = x0 * math.exp(mu * t)
    w = math.exp(v)
    law = dict(mean=m, var=m ** 2 * (w - 1), skew=(w + 2) * math.sqrt(w - 1), exkurt=w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 6)
    for q in (0.01, 0.05, 0.95, 0.99):
        law[f'q{int(100 * q):02d}'] = m * math.exp(math.sqrt(v) * stats.norm.ppf(q) - v / 2)
    return law, (m, v)


def surrogate_law(t, H, VJ, p, mu=0.2, s=0.5, x0=1.0):
    """law of X_{J,p}(t) = x0 e^{mu t} sum_{n<=p} (s sqrt(VJ))^n He_n(eta)/n!, eta ~ N(0,1) (Eq. (wickcoef))."""
    m = x0 * math.exp(mu * t); a = s * math.sqrt(VJ)
    g = lambda e: m * sum(a ** n * np.polynomial.hermite_e.hermeval(e, [0] * n + [1]) / math.factorial(n) for n in range(p + 1))
    x, w = np.polynomial.hermite_e.hermegauss(200); w = w / w.sum(); y = g(x)
    mean = float(w @ y); c = y - mean; var = float(w @ c ** 2)
    law = dict(mean=mean, var=var, skew=float(w @ c ** 3) / var ** 1.5, exkurt=float(w @ c ** 4) / var ** 2 - 3)
    e = np.linspace(-9, 9, 2_000_001); pdf = stats.norm.pdf(e); ye = g(e); o = np.argsort(ye); cdf = np.cumsum(pdf[o]); cdf /= cdf[-1]
    for q in (0.01, 0.05, 0.95, 0.99):
        law[f'q{int(100 * q):02d}'] = float(ye[o][np.searchsorted(cdf, q)])
    return law, g


def P2():
    R = {}
    for iH, H in enumerate(HS):
        T = 2.0; rec = {}
        for t in (1.0, 2.0):
            ex, (m, v) = lognormal_law(t, H); rt = {'exact': ex}
            for J in (17, 65):
                noise = hwc.HarmonicNoise(H, T, J); mt = noise.m(np.array([t]))[0]; VJ = float(mt @ noise.G @ mt)
                rt[f'VJ_{J}'] = VJ
                for p in range(1, 9):
                    law, _ = surrogate_law(t, H, VJ, p)
                    rt[f'{J},{p}'] = {k: law[k] / ex[k] - 1 for k in ex}
            # density data for the figure (J = 65)
            noise = hwc.HarmonicNoise(H, T, 65); mt = noise.m(np.array([t]))[0]; VJ = float(mt @ noise.G @ mt)
            xs = np.linspace(0, m * math.exp(4 * math.sqrt(v)), 600)
            rt['density_x'] = xs.tolist(); rt['density_exact'] = stats.lognorm.pdf(xs, s=math.sqrt(v), scale=m * math.exp(-v / 2)).tolist()
            for p in (1, 2, 4):
                _, g = surrogate_law(t, H, VJ, p); e = np.linspace(-8, 8, 400001); y = g(e)
                hist, edges = np.histogram(y, bins=xs, weights=stats.norm.pdf(e) * (e[1] - e[0]))
                rt[f'density_p{p}'] = (hist / np.diff(edges)).tolist()
            rec[str(t)] = rt
        # 10^6 surrogate samples with the general series evaluation (J = 9, p = 5): validates the generic route
        r = hwc.solve(wick_eq(), H, T, 9, 5, np.array([1.0, 2.0]), rtol=1e-10, atol=1e-12)
        rng = np.random.default_rng(SEED + 100 + iH); t0 = time.perf_counter(); X = []
        for s_ in range(20):
            X.append(series(r, rng.standard_normal((50_000, 9)), chunk=1000))
        X = np.concatenate(X, 1); s_eval = time.perf_counter() - t0
        samp = {}
        for it, t in enumerate((1.0, 2.0)):
            x = X[it]; c = x - x.mean()
            samp[str(t)] = dict(mean=float(x.mean()), var=float(c @ c / len(x)), skew=float(np.mean(c ** 3) / np.mean(c ** 2) ** 1.5),
                                exkurt=float(np.mean(c ** 4) / np.mean(c ** 2) ** 2 - 3),
                                **{f'q{q:02d}': float(np.quantile(x, q / 100)) for q in (1, 5, 95, 99)})
        rec['samples_J9_p5'] = dict(n=int(X.shape[1]), seconds_eval=s_eval, law=samp)
        R[str(H)] = rec
        log('P2', H, {k: (round(100 * rec['2.0'][f'65,{k}']['exkurt'], 3), round(100 * rec['2.0'][f'65,{k}']['q99'], 3)) for k in range(1, 9)}, round(s_eval, 1))
        json.dump(R, open(os.path.join(OUTD, 'P2_wick_law.json'), 'w'), indent=1)


# ------------------------------------------------------------------ P3 law of the cubic model and cost
def batch_ci(x, fun, nb=20):
    full = fun(x); parts = np.array([fun(b) for b in np.array_split(x, nb)])
    return full, float(1.96 * parts.std(ddof=1) / math.sqrt(nb))


STATS = {'mean': np.mean, 'var': lambda x: float(np.var(x)), 'skew': lambda x: float(stats.skew(x)),
         'exkurt': lambda x: float(stats.kurtosis(x)), **{f'q{q:02d}': (lambda q: (lambda x: float(np.quantile(x, q / 100))))(q) for q in (1, 5, 95, 99)}}


def P3():
    R = {}; M = 1_000_000; T = 4.0
    for iH, H in enumerate(HS):
        rng = np.random.default_rng(SEED + 200 + iH); rec = {}
        for nst in (512, 256):
            dt = T / nst; keep = {nst // 4, nst}; X1, X4 = [], []; t0 = time.perf_counter()
            for b in range(20):
                out = heun_cubic(fgn(H, nst, M // 20, dt, rng), dt, keep=keep); X1.append(out[nst // 4]); X4.append(out[nst])
            s_mc = time.perf_counter() - t0
            X1, X4 = np.concatenate(X1), np.concatenate(X4)
            rec[f'mc_{nst}'] = dict(M=M, seconds=s_mc, **{str(t): {k: batch_ci(x, f) for k, f in STATS.items()} for t, x in ((1.0, X1), (4.0, X4))})
            if nst == 512:
                rec['mc_hist'] = {str(t): np.histogram(x, bins=np.linspace(-1.5, 2.5, 161), density=True)[0].tolist() for t, x in ((1.0, X1), (4.0, X4))}
            log('P3 MC', H, nst, round(s_mc, 1))
        for J, p in [(9, 1), (9, 2), (9, 3), (9, 4), (17, 3)]:
            t0 = time.perf_counter(); r = hwc.solve(cubic_eq(), H, T, J, p, np.array([1.0, 4.0]), rtol=1e-8, atol=1e-10); s_solve = time.perf_counter() - t0
            rng2 = np.random.default_rng(SEED + 300 + iH); t0 = time.perf_counter(); X = []
            for b in range(20):
                X.append(series(r, rng2.standard_normal((M // 20, J)), chunk=1000))
            X = np.concatenate(X, 1); s_eval = time.perf_counter() - t0
            rec[f'chaos_{J},{p}'] = dict(seconds_solve=s_solve, seconds_eval=s_eval, **{str(t): {k: batch_ci(X[i], f) for k, f in STATS.items()} for i, t in enumerate((1.0, 4.0))})
            if (J, p) in ((9, 1), (9, 4)):
                rec[f'hist_{J},{p}'] = {str(t): np.histogram(X[i], bins=np.linspace(-1.5, 2.5, 161), density=True)[0].tolist() for i, t in enumerate((1.0, 4.0))}
            log('P3 chaos', H, J, p, round(s_solve, 1), round(s_eval, 1))
        R[str(H)] = rec
        json.dump(R, open(os.path.join(OUTD, 'P3_cubic_law.json'), 'w'), indent=1)




def P2b():
    """10^6 general-series samples (J = 9, p = 5) with 95 % batch-means intervals, against the quadrature law of the
    same chaos solution (t = T, where J does not enter)."""
    R = {}; T = 2.0
    for iH, H in enumerate(HS):
        r = hwc.solve(wick_eq(), H, T, 9, 5, np.array([2.0]), rtol=1e-10, atol=1e-12)
        rng = np.random.default_rng(SEED + 100 + iH); t0 = time.perf_counter(); X = []
        for s_ in range(20):
            X.append(series(r, rng.standard_normal((50_000, 9)), chunk=1000)[0])
        s_eval = time.perf_counter() - t0; X = np.concatenate(X)
        noise = hwc.HarmonicNoise(H, T, 9); mt = noise.m(np.array([T]))[0]; VJ = float(mt @ noise.G @ mt)
        law, _ = surrogate_law(T, H, VJ, 5); ex, _ = lognormal_law(T, H)
        R[str(H)] = dict(seconds_eval=s_eval, n=len(X), quadrature=law, exact=ex, samples={k: batch_ci(X, f) for k, f in STATS.items()})
        log('P2b', H, round(s_eval, 1))
    json.dump(R, open(os.path.join(OUTD, 'P2b_samples.json'), 'w'), indent=1)

if __name__ == '__main__':
    for part in (sys.argv[1:] or ['P1', 'P2', 'P2b', 'P3']):
        log('start', part); globals()[part]()
