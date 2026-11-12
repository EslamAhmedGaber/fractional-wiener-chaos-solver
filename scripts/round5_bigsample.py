"""10^7 samples of xi for a converged CosChaos configuration (the first 10^6 are those of the ladder runs).
Run from the repository root:  python scripts/round5_bigsample.py <config-name> [nbatches=200]
Writes evidence/round5/big_<name>_H<H>.npz and evidence/round5/P5_bigsample.json.
"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import round5_ladder as L
OUT = L.OUT; RES = os.path.join(OUT, 'P5_bigsample.json')


def xi_batches(iH, Jmax, nb):
    ro = np.random.default_rng(L.SEED + 300 + iH); rn = np.random.default_rng(np.random.SeedSequence([L.SEED, 5, 77, iH]))
    for b in range(nb):
        x = ro.standard_normal((L.M // L.NB, 9)); e = rn.standard_normal((L.M // L.NB, max(0, Jmax - 9)))
        yield np.hstack([x, e]) if Jmax > 9 else x[:, :Jmax]


if __name__ == '__main__':
    name = sys.argv[1]; nb = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    extra = json.load(open(os.path.join(OUT, 'ladder_configs.json'))); cfg = L.CONFIGS.get(name) or tuple(extra[name]); A = L.make_set(cfg); J = A.shape[1]
    for iH, H in enumerate(L.HS):
        R = json.load(open(RES)) if os.path.exists(RES) else {}
        if name in R and str(H) in R[name]: continue
        if not os.path.exists(os.path.join(OUT, f'ladder_{name}_H{H}.npz')): continue
        C = np.load(os.path.join(OUT, f'ladder_{name}_H{H}.npz'))['C']; t0 = time.perf_counter(); cnt = []; pw = []; hist = []
        for xb in xi_batches(iH, J, nb):
            x = np.concatenate([L.evaluate(A, C[-1:], xb[s:s + 2000, :J]) for s in range(0, xb.shape[0], 2000)], axis=1)[0]
            cnt.append((x[:, None] > L.LEV).sum(0)); pw.append([np.sum(x ** k) for k in range(5)])
            hist.append(np.histogram(np.clip(x, -2 + 1e-12, 2 - 1e-12), L.EDGES)[0])
        cnt = np.array(cnt); p = cnt / (L.M // L.NB)
        np.savez(os.path.join(OUT, f'big_{name}_H{H}.npz'), cnt=cnt, pw=np.array(pw), hist=np.array(hist, np.int32))
        R.setdefault(name, {})[str(H)] = dict(samples=nb * (L.M // L.NB), p4=p.mean(0).tolist(), ci4=(1.96 * p.std(0, ddof=1) / math.sqrt(nb)).tolist(), seconds=time.perf_counter() - t0)
        json.dump(R, open(RES, 'w'), indent=1); L.log('big', name, H, round(time.perf_counter() - t0))
