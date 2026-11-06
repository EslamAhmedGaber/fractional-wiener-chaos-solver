"""Round 4: probability of an event for the cubic model, dX = (aX - bX^3)dt + sig dB_H (a = -1, b = 1, X0 = 1,
sig = 0.5, T = 4), H = 0.55, 0.75, 0.95.

Event: X(t) > c, for a grid of levels c and times t = 0.5, 1, ..., 4.
  CosChaos  law of X_{J,p}(t) = sum_alpha c_alpha(t) Phi_alpha(xi), sampled with 10^6 standard Gaussian xi
            (J, p) = (9,1), (9,2), (9,3), (9,4), (17,3); solver evidence/round3_fwhe/solver_v1.1.3
  Monte Carlo  10^6 exact (Davies-Harte) fBm paths, Heun scheme with 512 and 256 steps on [0, 4]
  Gaussian  normal law with the Monte Carlo mean and variance (mean-variance estimate)
95 % intervals: batch means over 20 batches of 50 000.  Seeds: numpy default_rng(20261105 + 400 + iH) (Monte Carlo,
512 steps), + 410 + iH (256 steps), + 300 + iH (CosChaos coordinates, as in round3c P3).
Run from working/:  python scripts/round4_probability.py   ->  evidence/round4/P4_cubic_event.json
"""
import json, math, os, sys, time
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'evidence', 'round3_fwhe', 'solver_v1.1.3'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solver'))   # release layout: solver/ next to scripts/
import hwc
from round3c_fullsolution import fgn, heun_cubic, series, cubic_eq, HS, SEED
OUTD = os.path.join(ROOT, 'evidence', 'round4'); os.makedirs(OUTD, exist_ok=True)
log = lambda *a: print(*a, flush=True)
T = 4.0; TIMES = np.arange(1, 9) * 0.5; LEVELS = np.round(np.arange(0.0, 1.5001, 0.05), 2)
M = 1_000_000; NB = 20; SUB = 5            # 20 batches of 50 000, each generated in 5 pieces of 10 000 paths


def exceed(x):            # (nt, m) -> (nt, nlev)
    return np.stack([(x > c).mean(axis=-1) for c in LEVELS], axis=-1)


def summarise(batches):   # batches: list of dicts with 'p' (nt,nlev), 'mean' (nt,), 'm2' (nt,)
    P = np.array([b['p'] for b in batches]); mu = np.array([b['mean'] for b in batches]); m2 = np.array([b['m2'] for b in batches])
    p = P.mean(0); ci = 1.96 * P.std(0, ddof=1) / math.sqrt(len(batches))
    mean = mu.mean(0); var = m2.mean(0) - mean ** 2
    return dict(p=p.tolist(), ci=ci.tolist(), mean=mean.tolist(), var=var.tolist())


def mc(H, nst, seed):
    rng = np.random.default_rng(seed); dt = T / nst; keep = {int(round(t / dt)) for t in TIMES}; out = []; t0 = time.perf_counter()
    for b in range(NB):
        xs = []
        for s in range(SUB):
            o = heun_cubic(fgn(H, nst, M // NB // SUB, dt, rng), dt, keep=keep)
            xs.append(np.stack([o[int(round(t / dt))] for t in TIMES]))
        x = np.concatenate(xs, 1)
        out.append(dict(p=exceed(x), mean=x.mean(1), m2=(x ** 2).mean(1)))
    return dict(seconds=time.perf_counter() - t0, **summarise(out))


def chaos(H, J, p, seed):
    t0 = time.perf_counter(); r = hwc.solve(cubic_eq(), H, T, J, p, TIMES, rtol=1e-8, atol=1e-10); s_solve = time.perf_counter() - t0
    rng = np.random.default_rng(seed); t0 = time.perf_counter(); out = []
    for b in range(NB):
        x = series(r, rng.standard_normal((M // NB, J)), chunk=1000)
        out.append(dict(p=exceed(x), mean=x.mean(1), m2=(x ** 2).mean(1)))
    return dict(seconds_solve=s_solve, seconds_eval=time.perf_counter() - t0, **summarise(out))


if __name__ == '__main__':
    path = os.path.join(OUTD, 'P4_cubic_event.json')
    R = json.load(open(path)) if os.path.exists(path) else {}
    R['meta'] = dict(times=TIMES.tolist(), levels=LEVELS.tolist(), M=M, batches=NB, model='a=-1,b=1,X0=1,sig=0.5,T=4')
    for iH, H in enumerate(HS):
        rec = R.get(str(H), {})
        for J, p in [(9, 1), (9, 2), (9, 3), (9, 4), (17, 3)]:
            k = f'chaos_{J},{p}'
            if k not in rec:
                rec[k] = chaos(H, J, p, SEED + 300 + iH); log('chaos', H, J, p, round(rec[k]['seconds_solve'], 1), round(rec[k]['seconds_eval'], 1))
                R[str(H)] = rec; json.dump(R, open(path, 'w'), indent=1)
        for nst, off in ((512, 400), (256, 410)):
            k = f'mc_{nst}'
            if k not in rec:
                rec[k] = mc(H, nst, SEED + off + iH); log('mc', H, nst, round(rec[k]['seconds'], 1))
                R[str(H)] = rec; json.dump(R, open(path, 'w'), indent=1)
    log('done')
