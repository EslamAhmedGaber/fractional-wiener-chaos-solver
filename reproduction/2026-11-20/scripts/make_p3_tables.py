"""Generate LaTeX macros (source/tables/numbers_p3.tex) and Table 6 (source/tables/t6_lv.tex)
for Problem 3 (2D coupled competitive Lotka-Volterra) from evidence/P3_lotka_volterra/p3_results.json."""
import json
import math
import os
import numpy as np
from style import EVID, ROOT

P3 = os.path.join(EVID, 'P3_lotka_volterra')
R = json.load(open(os.path.join(P3, 'p3_results.json')))
TAB = os.path.join(ROOT, 'source', 'tables')
os.makedirs(TAB, exist_ok=True)
TOL = 1.0


def sig(x, n=2):
    if x == 0 or not np.isfinite(x):
        return '0'
    d = max(0, n - 1 - int(math.floor(math.log10(abs(x)))))
    return f'{x:.{d}f}'


def eval_2d_errors(mean, cov, ref, cov_se=None):
    rm = np.array(ref['mean'])
    rc = np.array(ref['cov'])
    rse = np.array(ref['cov_se'])
    m = np.array(mean)
    c = np.array(cov)

    em = 100.0 * np.abs(m / rm - 1.0)
    ev1 = 100.0 * np.abs(c[:, 0, 0] / rc[:, 0, 0] - 1.0)
    ev2 = 100.0 * np.abs(c[:, 1, 1] / rc[:, 1, 1] - 1.0)
    scale12 = np.sqrt(rc[:, 0, 0] * rc[:, 1, 1])
    ec12 = 100.0 * np.abs(c[:, 0, 1] - rc[:, 0, 1]) / scale12
    max_mean = float(em.max())
    max_cov = float(max(ev1.max(), ev2.max(), ec12.max()))

    se = np.zeros_like(rc) if cov_se is None else np.asarray(cov_se)
    c1_se = 100.0 * np.sqrt((se[:, 0, 0] / rc[:, 0, 0]) ** 2 + (rse[:, 0, 0] / rc[:, 0, 0]) ** 2)
    c2_se = 100.0 * np.sqrt((se[:, 1, 1] / rc[:, 1, 1]) ** 2 + (rse[:, 1, 1] / rc[:, 1, 1]) ** 2)
    c12_se = 100.0 * np.sqrt((se[:, 0, 1] / scale12) ** 2 + (rse[:, 0, 1] / scale12) ** 2)
    guard = float(max(np.max(ev1 + 2.0 * c1_se), np.max(ev2 + 2.0 * c2_se), np.max(ec12 + 2.0 * c12_se)))
    return max_mean, max_cov, guard, float(ev1.max()), float(ev2.max()), float(ec12.max())


lines = [
    r'\begin{widetable}\centering\small',
    r'\caption{Problem~3 (2D competitive Lotka--Volterra, $\lambda=0.05$): cheapest configuration of each method meeting the $1\%$ tolerance on both means, both variances and normalised cross-covariance $|\Delta\Cov_{12}|/\sqrt{V_1V_2}$ at all eight times (\emph{not met}: most accurate tested). Single-thread wall-clock times include preparation. Bold: cheapest eligible winner.}\label{tab:lv}',
    r'\begin{tabular}{@{}clllrrr@{}}\toprule',
    r'$H$ & Method & Setting & & Mean (\%) & Cov.\ (\%) & Time (s)\\\midrule',
]

summary = {}
for H_str in ('0.55', '0.75', '0.95'):
    case = R['cases'][H_str]
    ref = case['reference']
    cc_evals = []
    for r in case['coschaos']['rows']:
        mm, mc, crit, e1, e2, e12 = eval_2d_errors(r['mean'], r['cov'], ref, cov_se=None)
        cc_evals.append(dict(method='CosChaos', J=r['J'], p=r['p'], seconds=r['seconds'],
                             max_mean=mm, max_cov=mc, criterion=crit, e1=e1, e2=e2, e12=e12,
                             p_dom=r.get('p_dom'), p_dom_eval_s=r.get('p_dom_eval_s')))
    rk_evals = []
    em_evals = []
    for r in case['mc']:
        mm, mc, crit, e1, e2, e12 = eval_2d_errors(r['mean'], r['cov'], ref, cov_se=r['cov_se'])
        item = dict(method=r['method'], n=r['n'], M=r['M'], seconds=r['seconds'],
                    max_mean=mm, max_cov=mc, criterion=crit, e1=e1, e2=e2, e12=e12)
        if 'RK4' in r['method']:
            rk_evals.append(item)
        else:
            em_evals.append(item)

    def pick(rows):
        ok = [x for x in rows if x['criterion'] <= TOL and x['max_mean'] <= TOL]
        if ok:
            return min(ok, key=lambda x: x['seconds']), True
        return min(rows, key=lambda x: x['criterion']), False

    p_cc, ok_cc = pick(cc_evals)
    p_rk, ok_rk = pick(rk_evals)
    p_em, ok_em = pick(em_evals)
    picks = [
        (r'\CC{}', f"$J={p_cc['J']}$, $p={p_cc['p']}$", p_cc, ok_cc),
        ('2D WIS-RK4 + CV + MC', rf"$n={p_rk['n']}$, $M=2^{{{int(round(math.log2(p_rk['M'])))}}}$", p_rk, ok_rk),
        ('2D GfBm EM + CV + MC', rf"$n={p_em['n']}$, $M=2^{{{int(round(math.log2(p_em['M'])))}}}$", p_em, ok_em),
    ]
    best = min((p for p in picks if p[3]), key=lambda p: p[2]['seconds'], default=None)
    summary[H_str] = dict(cc_evals=cc_evals, rk_evals=rk_evals, em_evals=em_evals,
                          pick_cc=p_cc, ok_cc=ok_cc, pick_rk=p_rk, ok_rk=ok_rk, pick_em=p_em, ok_em=ok_em)
    for i, (lab, stg, row, ok) in enumerate(picks):
        t = sig(row['seconds'], 2)
        if best is not None and lab == best[0]:
            t = rf'\textbf{{{t}}}'
        mark = '' if ok else r'\emph{not met}'
        stg_s = stg if ok else rf'\emph{{{stg}}}'
        cells = [H_str if i == 0 else '', lab, stg_s, mark,
                 sig(row['max_mean'], 2), sig(row['max_cov'], 2), t]
        lines.append(' & '.join(cells) + r'\\')
    lines.append(r'\addlinespace[2pt]')

lines[-1] = r'\bottomrule\end{tabular}'
lines.append(
    r'\par\smallskip{\footnotesize Cov.\ (\%) is the maximum of relative errors in $\Var X_1$, $\Var X_2$ and '
    r'$|\Delta\Cov(X_1,X_2)|/\sqrt{V_1V_2}$ across $t=2,\dots,16$. Eligibility uses the common pointwise rule in Section~\ref{sec:results}. '
    r'For the two reference variances, the maximum paired grid change is $\LVStepMax\%$ and the maximum relative SE is $\LVRefSEMax\%$ (rounded).\par}'
)
lines.append(r'\end{widetable}')
open(os.path.join(TAB, 't6_lv.tex'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

# ---- Macros for Problem 3
M = {}
M['LVRefSEMax'] = sig(max(R['cases'][h]['reference']['max_var_relse_pct'] for h in ('0.55', '0.75', '0.95')), 2)
M['LVStepMax'] = sig(max(R['cases'][h]['reference']['step_half_var_diff_pct'] for h in ('0.55', '0.75', '0.95')), 2)

# Peak Var(X2) amplification and correlation shift
v2_peaks = {h: max(c[1][1] for c in R['cases'][h]['reference']['cov']) for h in ('0.55', '0.75', '0.95')}
M['LVPeakLow'] = sig(v2_peaks['0.55'], 2)
M['LVPeakMid'] = sig(v2_peaks['0.75'], 2)
M['LVPeakHigh'] = sig(v2_peaks['0.95'], 3)
M['LVPeakRatio'] = f"{v2_peaks['0.95'] / v2_peaks['0.55']:.1f}"
M['LVCorrEnd'] = f"{R['cases']['0.95']['reference']['corr'][-1]:.2f}"

# Competitive reversal probability P(X1(T) > X2(T)) across H
pdom_ref = {h: 100.0 * R['cases'][h]['reference']['p_dom'] for h in ('0.55', '0.75', '0.95')}
pdom_cc95 = 100.0 * next(r['p_dom'] for r in R['cases']['0.95']['coschaos']['rows'] if r['J'] == 17 and r['p'] == 2)
M['LVDomLow'] = f"{pdom_ref['0.55']:.2f}"
M['LVDomMid'] = f"{pdom_ref['0.75']:.2f}"
M['LVDomHigh'] = f"{pdom_ref['0.95']:.2f}"
M['LVDomCCHigh'] = f"{pdom_cc95:.2f}"
M['LVDomRatio'] = f"{pdom_ref['0.95'] / pdom_ref['0.55']:.0f}"

law = next(r for r in R['cases']['0.95']['coschaos']['rows'] if r['J'] == 17 and r['p'] == 2)
M['LVLawSolve'] = sig(law['seconds'], 2)
M['LVLawEval'] = sig(law['p_dom_eval_s'], 2)

# Accuracy and speedup at H = 0.95 and H = 0.75
cc95_p1 = next(r for r in summary['0.95']['cc_evals'] if r['J'] == 65 and r['p'] == 1)
cc95_win = summary['0.95']['pick_cc']
rk95_win = summary['0.95']['pick_rk']
cc75_win = summary['0.75']['pick_cc']
rk75_win = summary['0.75']['pick_rk']

M['LVDegOneFloor'] = sig(cc95_p1['max_cov'], 2)
M['LVCCErrHigh'] = sig(cc95_win['max_cov'], 2)
M['LVCCSecHigh'] = sig(cc95_win['seconds'], 3)
M['LVRKSecHigh'] = sig(rk95_win['seconds'], 4)
M['LVSpeedup'] = f"{rk95_win['seconds'] / cc95_win['seconds']:.0f}"
M['LVCCErrMid'] = sig(cc75_win['max_cov'], 2)
M['LVCCSecMid'] = sig(cc75_win['seconds'], 2)
M['LVRKSecMid'] = sig(rk75_win['seconds'], 2)
M['LVSpeedupMid'] = f"{rk75_win['seconds'] / cc75_win['seconds']:.0f}"
M['LVRKSecLow'] = sig(summary['0.55']['pick_rk']['seconds'], 2)
M['LVCCErrLow'] = sig(summary['0.55']['pick_cc']['max_cov'], 2)

json.dump(summary, open(os.path.join(P3, 'p3_summary.json'), 'w'), indent=1)
with open(os.path.join(TAB, 'numbers_p3.tex'), 'w', encoding='utf-8') as fh:
    fh.write('% Generated by scripts/make_p3_tables.py from evidence/P3_lotka_volterra/p3_results.json\n')
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('numbers_p3.tex and t6_lv.tex written;', len(M), 'macros')
