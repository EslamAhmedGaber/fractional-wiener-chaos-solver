"""Round 5, Phase 1: sharp Monte Carlo reference for the cubic single well (a = -1, b = 1, X0 = 1, sigma = 0.5, T = 4).
Exact (Davies-Harte) fBm increments, Heun scheme, H = 0.55, 0.75, 0.95; 512 steps (2 x 10^7 paths) and 256 steps
(10^7 paths, step check). Batches of 50 000 paths (10 pieces of 5 000), independent seeds
SeedSequence([20261105, 5, iH, nsteps, batch]). Each batch is saved as one .npz (exceedance counts on a 0.01 level grid,
power sums, a 16 000-bin histogram on [-2, 2] at t = 4, and counts at t = 1), so a stop loses at most one batch per worker.
Run from working/:  python scripts/round5_mc_reference.py <worker> <n_workers>
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from round3c_fullsolution import fgn, heun_cubic
OUT = os.path.join(ROOT, 'evidence', 'round5', 'mc'); os.makedirs(OUT, exist_ok=True)
HS = [0.55, 0.75, 0.95]; T = 4.0; PIECE = 5000; PIECES = 10
LEV = np.round(np.arange(0.0, 1.2001, 0.01), 2); EDGES = np.linspace(-2.0, 2.0, 16001)
PLAN = [(2, 512, 400), (1, 512, 400), (0, 512, 400), (2, 256, 200), (1, 256, 200), (0, 256, 200)]


def batch(iH, nst, b):
    f = os.path.join(OUT, f'H{HS[iH]}_n{nst}_b{b:04d}.npz')
    if os.path.exists(f): return False
    rng = np.random.default_rng(np.random.SeedSequence([20261105, 5, iH, nst, b])); dt = T / nst; t0 = time.perf_counter()
    cnt4 = np.zeros(len(LEV), np.int64); cnt1 = np.zeros(len(LEV), np.int64); pw = np.zeros((2, 5)); hist = np.zeros(16000, np.int64)
    for _ in range(PIECES):
        o = heun_cubic(fgn(HS[iH], nst, PIECE, dt, rng), dt, keep={nst // 4, nst})
        for row, x in ((0, o[nst // 4]), (1, o[nst])):
            pw[row] += [np.sum(x ** k) for k in range(5)]
        x4 = o[nst]; cnt4 += (x4[:, None] > LEV).sum(0); cnt1 += (o[nst // 4][:, None] > LEV).sum(0)
        hist += np.histogram(np.clip(x4, -2 + 1e-12, 2 - 1e-12), EDGES)[0]
    np.savez(f + '.tmp.npz', cnt4=cnt4, cnt1=cnt1, pw=pw, hist=hist.astype(np.int32), n=PIECE * PIECES, seconds=time.perf_counter() - t0)
    os.replace(f + '.tmp.npz', f); return True


if __name__ == '__main__':
    w, nw = int(sys.argv[1]), int(sys.argv[2]); tasks = [(iH, nst, b) for iH, nst, nb in PLAN for b in range(nb)]
    for k, (iH, nst, b) in enumerate(tasks):
        if k % nw == w and batch(iH, nst, b): print('done', HS[iH], nst, b, flush=True)
    print('worker finished', flush=True)
