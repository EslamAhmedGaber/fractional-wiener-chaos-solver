"""Generate LaTeX macros (source/tables/numbers_p2.tex) and Table 5 (source/tables/t5_compare.tex)
for Problem 2 (all six (lam, H) cases, fair deterministic vs sampled tolerance rule)."""
import json
import math
import os
import numpy as np
from style import EVID, ROOT

P2 = os.path.join(EVID, 'P2_population')
SUM = json.load(open(os.path.join(P2, 'p2_summary.json')))
TAB = os.path.join(ROOT, 'source', 'tables')
os.makedirs(TAB, exist_ok=True)
TOL = 1.0
CASES = [(lam, H) for lam in (0.02, 0.05) for H in (0.55, 0.75, 0.95)]
METHODS = ['CosChaos', 'FWHE', 'GfBm EM', 'Exact WIS + MC']
LABEL = {
    'CosChaos': r'\CC{}',
    'FWHE': 'FWHE, first order',
    'GfBm EM': 'GfBm EM + CV + MC',
    'Exact WIS + MC': 'exact WIS + CV + MC',
}


def sig(x, n=2):
    if x == 0 or not np.isfinite(x):
        return '0'
    d = max(0, n - 1 - int(math.floor(math.log10(abs(x)))))
    return f'{x:.{d}f}'


def setting(r):
    if r['method'] == 'CosChaos':
        return f"$J={r['J']}$, $p={r['p']}$"
    if r['method'] == 'FWHE':
        return rf"$\Delta t={r['dt']:g}$"
    return rf"$n={r['n']}$, $M=2^{{{int(round(math.log2(r['M'])))}}}$"


def meets(r):
    return r['criterion'] <= TOL and r['max_mean'] <= TOL


def pick(rows):
    ok = [r for r in rows if meets(r)]
    if ok:
        return min(ok, key=lambda r: r['seconds']), True
    return min(rows, key=lambda r: r['criterion']), False


lines = [
    r'\begin{widetable}\centering\small',
    r'\caption{Problem~2 (all six $(\lambda,H)$ cases): cheapest configuration of each method meeting the $1\%$ tolerance on mean and variance at all eight times (\emph{not met}: most accurate tested). Columns give observed maximum relative errors against the numerical WIS reference and single-thread times. Bold: cheapest eligible winner.}\label{tab:compare}',
    r'\begin{tabular}{@{}cclllrrr@{}}\toprule',
    r'$\lambda$ & $H$ & Method & Setting & & Mean (\%) & Var.\ (\%) & Time (s)\\\midrule',
]
chosen = {}
for lam, H in CASES:
    S = SUM[f'{lam}_{H}']
    picks = []
    for m in METHODS:
        rows = [r for r in S['rows'] if r['method'] == m]
        r, ok = pick(rows)
        picks.append((m, r, ok))
    best = min((p for p in picks if p[2]), key=lambda p: p[1]['seconds'], default=None)
    chosen[f'{lam}_{H}'] = {m: dict(row=r, ok=ok) for m, r, ok in picks}
    for i, (m, r, ok) in enumerate(picks):
        t = sig(r['seconds'], 2)
        if best is not None and m == best[0]:
            t = rf'\textbf{{{t}}}'
        mark = '' if ok else r'\emph{not met}'
        cells = [
            f'{lam:g}' if i == 0 else '',
            f'{H:g}' if i == 0 else '',
            LABEL[m], setting(r), mark,
            sig(r['max_mean'], 2), sig(r['max_var'], 2), t,
        ]
        if not ok:
            cells[3] = rf'\emph{{{cells[3]}}}'
        lines.append(' & '.join(cells) + r'\\')
    lines.append(r'\addlinespace[2pt]')
lines[-1] = r'\bottomrule\end{tabular}'
lines.append(
    r'\par\smallskip{\footnotesize Settings: $J$ modes, degree $p$ (\CC{}); time step $\Delta t$ (FWHE); '
    r'$n$ steps on $[0,T]$, $M$ sampled paths. Eligibility follows the common pointwise rule in Section~\ref{sec:results}. '
    r'Reference relative SE is at most $\RefSEmax\%$ everywhere.\par}'
)
lines.append(r'\end{widetable}')
open(os.path.join(TAB, 't5_compare.tex'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

# ---- Macros (strictly verified over ALL six cases!)
M = {}
M['RefQuadMax'] = sig(max(max(abs(x) for x in S['quad_var_pct']) for S in SUM.values()), 1)
M['RefSEmax'] = sig(max(max(S['ref_var_relse_pct']) for S in SUM.values()), 2)
se95 = max(max(SUM[k]['ref_var_relse_pct']) for k in ('0.02_0.95', '0.05_0.95'))
M['RefSEstrong'] = f'{math.ceil(se95 * 100) / 100:.2f}'


def cc(lam, H, J, p):
    return next(r for r in SUM[f'{lam}_{H}']['rows'] if r['method'] == 'CosChaos' and r['J'] == J and r['p'] == p)


def fw(lam, H, dt):
    return next(r for r in SUM[f'{lam}_{H}']['rows'] if r['method'] == 'FWHE' and abs(r['dt'] - dt) < 1e-12)


# Mean error over ALL six cases (no excluded H = 0.55!)
M['PopMeanErr'] = sig(max(cc(l, h, 33, 2)['max_mean'] for l, h in CASES), 2)
M['PopVarErr'] = sig(cc(0.02, 0.95, 17, 2)['max_var'], 2)
M['PopVarTime'] = sig(cc(0.02, 0.95, 17, 2)['seconds'], 2)
M['PopVarErrStrong'] = sig(cc(0.05, 0.95, 17, 2)['max_var'], 2)
M['PopVarTimeStrong'] = sig(cc(0.05, 0.95, 17, 2)['seconds'], 2)
M['PopVarErrMidStrong'] = sig(cc(0.05, 0.75, 65, 2)['max_var'], 2)
M['PopVarTimeMidStrong'] = sig(cc(0.05, 0.75, 65, 2)['seconds'], 2)
M['PopMidWinTime'] = sig(chosen['0.05_0.75']['CosChaos']['row']['seconds'], 2)
M['PopMidWeakWinTime'] = sig(chosen['0.02_0.75']['CosChaos']['row']['seconds'], 2)
M['PopDegOneFloor'] = sig(cc(0.02, 0.95, 65, 1)['max_var'], 2)
M['PopFwheFloor'] = sig(fw(0.05, 0.95, 0.003125)['max_var'], 2)
M['PopFwheFloorWeak'] = sig(fw(0.02, 0.95, 0.003125)['max_var'], 2)
M['PopFwhePubVar'] = sig(fw(0.02, 0.95, 0.05)['max_var'], 2)
M['PopFwhePubMean'] = sig(fw(0.02, 0.95, 0.05)['max_mean'], 2)
fl = chosen['0.02_0.55']['FWHE']['row']
M['PopFwheLowBest'] = sig(fl['max_var'], 2)
M['PopFwheLowTime'] = sig(fl['seconds'], 2)

g95 = [r for r in SUM['0.02_0.95']['rows'] if r['method'] == 'GfBm EM']
gbest = min(g95, key=lambda r: r['criterion'])
M['GfbmBestVar'] = sig(gbest['max_var'], 2)
M['GfbmBestTime'] = sig(gbest['seconds'], 2)
M['GfbmBestN'] = str(gbest['n'])

c95 = chosen['0.02_0.95']
c95s = chosen['0.05_0.95']
M['ExactTime'] = sig(c95['Exact WIS + MC']['row']['seconds'], 2)
M['CCOneTime'] = sig(c95['CosChaos']['row']['seconds'], 2)
M['PopSpeedup'] = f"{c95['Exact WIS + MC']['row']['seconds'] / c95['CosChaos']['row']['seconds']:.0f}"
M['PopSpeedupStrong'] = f"{c95s['Exact WIS + MC']['row']['seconds'] / c95s['CosChaos']['row']['seconds']:.0f}"

pk = {H: max(SUM[f'0.05_{H}']['ref_var']) for H in (0.55, 0.75, 0.95)}
M['PeakLow'] = sig(pk[0.55], 2)
M['PeakMid'] = sig(pk[0.75], 3)
M['PeakHigh'] = sig(pk[0.95], 3)
M['PeakRatio'] = f'{pk[0.95] / pk[0.55]:.1f}'

laws = [json.load(open(os.path.join(P2, f'law_lam{l}_H0.95.json'))) for l in (0.02, 0.05)]
M['LawErr'] = sig(max(max(abs(x) for x in L['rel_err_terminal_pct']) for L in laws), 2)
M['LawEvalSec'] = sig(max(L['eval_seconds'] for L in laws), 2)

with open(os.path.join(TAB, 'numbers_p2.tex'), 'w', encoding='utf-8') as fh:
    fh.write('% Generated by scripts/make_p2_tables.py from evidence/P2_population/p2_summary.json\n')
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('numbers_p2.tex and t5_compare.tex written;', len(M), 'macros')
