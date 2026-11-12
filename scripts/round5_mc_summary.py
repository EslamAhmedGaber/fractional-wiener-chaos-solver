"""summary: the sharp Monte Carlo reference from evidence/round5/mc/*.npz.
95 % intervals: batch means over the 50 000-path batches (1.96 sd / sqrt(nb)). Quantiles from the 16 000-bin histogram
(linear interpolation inside a bin, width 2.5e-4), intervals from the batch quantiles.
Writes evidence/round5/P5_mc_reference.json.  Run from the repository root:  python scripts/round5_mc_summary.py
"""
import glob, json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, 'evidence', 'round5')
LEV = np.round(np.arange(0.0, 1.2001, 0.01), 2); EDGES = np.linspace(-2.0, 2.0, 16001); QS = [0.01, 0.05, 0.95, 0.99]


def hq(h, q):
    c = np.cumsum(h) / h.sum(); i = int(np.searchsorted(c, q)); lo = c[i - 1] if i else 0.0
    return float(EDGES[i] + (q - lo) / (c[i] - lo) * (EDGES[i + 1] - EDGES[i]))


def mom(pw):                                         # pw: power sums 0..4 -> mean, var, skew, exkurt
    n = pw[0]; m1, m2, m3, m4 = (pw[k] / n for k in range(1, 5)); v = m2 - m1 ** 2
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return dict(mean=m1, var=v, skew=c3 / v ** 1.5, exkurt=c4 / v ** 2 - 3)


def summarise(files):
    B = [np.load(f) for f in files]; nb = len(B); n = int(sum(int(b['n']) for b in B))
    p = np.array([b['cnt4'] / b['n'] for b in B]); p1 = np.array([b['cnt1'] / b['n'] for b in B])
    hw = lambda x: (1.96 * x.std(0, ddof=1) / math.sqrt(nb))
    pw4 = sum(b['pw'][1] for b in B); full = mom(pw4); per = [mom(b['pw'][1]) for b in B]
    H_ = sum(b['hist'].astype(np.int64) for b in B)
    out = dict(paths=n, batches=nb, seconds=float(sum(float(b['seconds']) for b in B)), p4=p.mean(0).tolist(), ci4=hw(p).tolist(),
               p1=p1.mean(0).tolist(), ci1=hw(p1).tolist())
    for k in ('mean', 'var', 'skew', 'exkurt'):
        out[k] = full[k]; out[k + '_ci'] = float(hw(np.array([q[k] for q in per])))
    for q in QS:
        out[f'q{int(round(100 * q)):02d}'] = hq(H_, q); out[f'q{int(round(100 * q)):02d}_ci'] = float(hw(np.array([hq(b['hist'], q) for b in B])))
    return out


if __name__ == '__main__':
    R = {}
    for H in ('0.55', '0.75', '0.95'):
        R[H] = {f'mc_{n}': summarise(sorted(glob.glob(os.path.join(D, 'mc', f'H{H}_n{n}_b*.npz')))) for n in (512, 256)}
    R['levels'] = LEV.tolist()
    json.dump(R, open(os.path.join(D, 'P5_mc_reference.json'), 'w'), indent=1); print('ok')
