"""Problem 2 analysis: errors against the control-variate exact WIS reference across all six (lam, H) cases.
Writes evidence/P2_population/p2_summary.json.

The variance eligibility margin includes reference uncertainty for every method.
"""
import json
import math
import os
import numpy as np
from style import EVID

P2 = os.path.join(EVID, 'P2_population')
CASES = [(lam, H) for lam in (0.02, 0.05) for H in (0.55, 0.75, 0.95)]


def load(task, lam, H):
    fn = os.path.join(P2, f'{task}_lam{lam}_H{H}.json')
    return json.load(open(fn)) if os.path.exists(fn) else None


def rel_errs(m, v, ref):
    rm, rv = np.array(ref['mean']), np.array(ref['var'])
    em = 100.0 * (np.array(m) / rm - 1.0)
    ev = 100.0 * (np.array(v) / rv - 1.0)
    return em, ev


summary = {}
for lam, H in CASES:
    ref = load('reference', lam, H)
    if ref is None:
        raise FileNotFoundError(f'Missing reference for lam={lam}, H={H}')
    rv = np.array(ref['var'])
    rse = 100.0 * np.array(ref['var_se']) / rv
    quad_pct = 100.0 * np.array(ref['quad_half_var_diff']) / rv
    key = f'{lam}_{H}'
    S = dict(
        lam=lam, H=H, ref_mean=ref['mean'], ref_var=rv.tolist(),
        ref_var_relse_pct=rse.tolist(), quad_var_pct=quad_pct.tolist(), rows=[],
    )
    print(f'== lam {lam} H {H}: ref max relSE={rse.max():.4f}% quad={np.abs(quad_pct).max():.4f}%')

    cc = load('coschaos', lam, H)
    for r in cc['rows']:
        em, ev = rel_errs(r['mean'], r['var'], ref)
        max_v = float(np.abs(ev).max())
        row = dict(
            method='CosChaos', deterministic=True, J=r['J'], p=r['p'], n_coef=r['n_coef'],
            seconds=r['seconds'], max_mean=float(np.abs(em).max()), max_var=max_v,
            var_err_by_t=ev.tolist(), criterion=float(np.max(np.abs(ev) + 2.0 * rse)),
        )
        S['rows'].append(row)

    fw = load('fwhe', lam, H)
    for r in fw['rows']:
        em, ev = rel_errs(r['mean'], r['var'], ref)
        max_v = float(np.abs(ev).max())
        row = dict(
            method='FWHE', deterministic=True, dt=r['dt'], seconds=r['seconds'],
            max_mean=float(np.abs(em).max()), max_var=max_v,
            var_err_by_t=ev.tolist(), criterion=float(np.max(np.abs(ev) + 2.0 * rse)),
        )
        S['rows'].append(row)

    gf = load('gfbm', lam, H)
    for r in gf['rows']:
        best = None
        for est, vkey, skey, mkey in (('plain', 'var', 'var_se', 'mean'),
                                      ('control variate', 'var_cv', 'var_cv_se', 'mean_cv')):
            em, ev = rel_errs(r[mkey], r[vkey], ref)
            se = 100.0 * np.array(r[skey]) / rv
            comb = np.sqrt(se ** 2 + rse ** 2)
            guard = float(np.max(np.abs(ev) + 2.0 * comb))
            cand = dict(
                method='GfBm EM', deterministic=False, estimator=est, n=r['n'], M=r['M'],
                seconds=r['seconds'], max_mean=float(np.abs(em).max()),
                max_var=float(np.abs(ev).max()), var_err_by_t=ev.tolist(),
                var_se_pct=se.tolist(), comb_se_max=float(comb.max()), criterion=guard,
            )
            if best is None or guard < best['criterion']:
                best = cand
        S['rows'].append(best)

    ex = load('exactmc', lam, H)
    for r in ex['rows']:
        best = None
        for est, vkey, skey in (('plain', 'var', 'var_se'), ('control variate', 'var_cv', 'var_cv_se')):
            if vkey not in r:
                continue
            em, ev = rel_errs(r['mean'], r[vkey], ref)
            se = 100.0 * np.array(r[skey]) / rv
            comb = np.sqrt(se ** 2 + rse ** 2)
            guard = float(np.max(np.abs(ev) + 2.0 * comb))
            cand = dict(
                method='Exact WIS + MC', deterministic=False, estimator=est, n=r['n'], M=r['M'],
                seconds=r['seconds'], max_mean=float(np.abs(em).max()),
                max_var=float(np.abs(ev).max()), var_err_by_t=ev.tolist(),
                var_se_pct=se.tolist(), comb_se_max=float(comb.max()), criterion=guard,
            )
            if best is None or guard < best['criterion']:
                best = cand
        S['rows'].append(best)

    summary[key] = S

out_path = os.path.join(P2, 'p2_summary.json')
json.dump(summary, open(out_path, 'w'), indent=1)
print('wrote', out_path)
