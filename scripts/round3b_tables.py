"""Round 3b tables from evidence/round3_fwhe/workprecision.json (written by round3b_workprecision.py).
Writes source/tables/tab_memory.tex (main text), tab_fwhe_work.tex and tab_fwhe_memory.tex (supplement), and
evidence/round3_fwhe/derived_numbers_3b.json with every number quoted in the text.
Run from working/:  python scripts/round3b_tables.py
"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, 'evidence', 'round3_fwhe'); TB = os.path.join(ROOT, 'source', 'tables')
W = json.load(open(os.path.join(EV, 'workprecision.json')))
D = {}


def sgn(x, d): return f'${x:+.{d}f}$'


def sci(x, d=4):
    e = int(np.floor(np.log10(abs(x)))); return f'${x / 10 ** e:.{d}f}\\times10^{{{e}}}$'


# --- main-text Table: memory model at t = 20
tm = W['memory']['tab_memory']; rows = []
for H in ['0.5', '0.7', '0.95']:
    r = tm[H]
    diag = r['diag_dev_pct']
    rows.append(f"{H} & {sci(r['reference_t20'], 6)} & " + ' & '.join(sgn(r['dev_pct'][J], 4) for J in ['9', '33', '129', '257'])
                + f" & {'0.00' if abs(diag) < 0.005 else sgn(diag, 2)} & {sci(r['v2_J65'], 2)} \\\\")
    D[f'mem_t20_H{H}'] = r
open(os.path.join(TB, 'tab_memory.tex'), 'w').write(
    '\\begin{tabular}{rccccccc}\\toprule\n & & \\multicolumn{4}{c}{deviation from the reference solution (\\%)} & & \\\\\\cmidrule(lr){3-6}\n'
    '$H$ & reference, $t=20$ & $J=9$ & $J=33$ & $J=129$ & $J=257$ & diag.\\ $G^H$ (\\%) & $v_2$ (absolute) \\\\\\midrule\n'
    + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')

# --- supplement: matched accuracy (OU)
def cheapest(rec, tol):
    ok = [(d['seconds'], k, d['max_rel']) for k, d in rec.items() if d['max_rel'] <= tol]
    return min(ok) if ok else None


rows = []
for H in ['0.55', '0.75', '0.95']:
    r = W['OU'][H]; cells = []
    for tol in (1e-2, 1e-3, 1e-4):
        for key, sym in (('ours', 'J'), ('fwhe_gl', 'N')):
            c = cheapest(r[key], tol)
            cells.append('not reached' if c is None else f'${sym}={int(c[1])}$, {c[0]:.3f}\\,s')
            D[f'OU_match_H{H}_{key}_{tol:g}'] = None if c is None else dict(setting=c[1], seconds=c[0], max_rel=c[2])
    rows.append(f'{H} & ' + ' & '.join(cells) + '\\\\')
open(os.path.join(TB, 'tab_fwhe_work.tex'), 'w').write(
    '\\begin{tabular}{lcccccc}\\toprule\n & \\multicolumn{2}{c}{max.\\ rel.\\ error $\\le10^{-2}$} & \\multicolumn{2}{c}{$\\le10^{-3}$} & \\multicolumn{2}{c}{$\\le10^{-4}$}\\\\'
    '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n$H$ & CosChaos & FWHE-GL & CosChaos & FWHE-GL & CosChaos & FWHE-GL\\\\\\midrule\n'
    + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')

# --- supplement: memory model, all output times
M = W['memory']; t = M['t']; rows = []
for H in ['0.55', '0.75', '0.95']:
    r = M[H]; ref = np.array(r['reference'])
    for i, ti in enumerate(t):
        cells = [sgn(100 * (r['ours'][J]['var'][i] / ref[i] - 1), 3) for J in ('33', '129')]
        cells += [sgn(100 * (r['fwhe_gl'][N]['var'][i] / ref[i] - 1), 3) for N in ('400', '1600', '12800')]
        rows.append(f"{H if i == 0 else ''} & {ti:g} & {sci(ref[i])} & " + ' & '.join(cells) + '\\\\')
    rows.append('\\midrule')
tt = [M[H]['ours']['129']['seconds'] for H in ('0.55', '0.75', '0.95')]; tc = [M[H]['ours_cached']['129']['seconds'] for H in ('0.55', '0.75', '0.95')]
tg = {N: [M[H]['fwhe_gl'][N]['seconds'] for H in ('0.55', '0.75', '0.95')] for N in ('400', '1600', '12800')}
rows.append(f"time (s) & & -- & \\multicolumn{{2}}{{c}}{{{min(tt):.0f}--{max(tt):.0f}; {min(tc):.2f}--{max(tc):.2f} with stored Green function}} & "
            + ' & '.join(f'{max(tg[N]):.2f}' for N in tg) + '\\\\')
open(os.path.join(TB, 'tab_fwhe_memory.tex'), 'w').write(
    '\\begin{tabular}{rrcccccc}\\toprule\n & & reference & \\multicolumn{2}{c}{CosChaos, $p=1$} & \\multicolumn{3}{c}{FWHE-GL, $k=1$}\\\\'
    '\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\n$H$ & $t$ & solution & $J=33$ & $J=129$ & $N=400$ & $N=1600$ & $N=12\\,800$\\\\\\midrule\n'
    + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')

# --- supplement: OU at H = 0.75, every output time (same run as the work-precision panels)
r = W['OU']['0.75']; ex = np.array(r['reference']); tq = W['OU']['t']; rows = []
for i, ti in enumerate(tq):
    if abs(ti * 4 - round(ti * 4)) > 1e-9 or ti not in (0.25, 0.5, 1.0, 1.5, 2.0): continue
    c = [sgn(100 * (r['ours'][J]['var'][i] / ex[i] - 1), d) for J, d in (('17', 2), ('65', 3))] + [sgn(100 * (r['fwhe_gl'][N]['var'][i] / ex[i] - 1), 2) for N in ('400', '1600')]
    rows.append(f'{ti:g} & {ex[i]:.6f} & ' + ' & '.join(c) + '\\\\')
rows.append('\\midrule time (s) & -- & ' + ' & '.join(f"{r[k][n]['seconds']:.3f}" for k, n in (('ours', '17'), ('ours', '65'), ('fwhe_gl', '400'), ('fwhe_gl', '1600'))) + '\\\\')
open(os.path.join(TB, 'tab_fwhe_ou.tex'), 'w').write(
    '\\begin{tabular}{rccccc}\\toprule\n & & \\multicolumn{2}{c}{CosChaos, $p=1$} & \\multicolumn{2}{c}{FWHE-GL, $k=1$}\\\\\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n'
    '$t$ & reference & $J=17$ & $J=65$ & $N=400$ & $N=1600$\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')
D['speedup'] = M['speedup']
D['mem_J129_t20'] = {H: 100 * (M[H]['ours']['129']['var'][-1] / M[H]['reference'][-1] - 1) for H in ('0.55', '0.75', '0.95')}
D['mem_maxrel'] = {H: {'ours129': M[H]['ours']['129']['max_rel'], 'gl1600': M[H]['fwhe_gl']['1600']['max_rel'], 'gl12800': M[H]['fwhe_gl']['12800']['max_rel']} for H in ('0.55', '0.75', '0.95')}
D['OU_points'] = {H: {'J17': W['OU'][H]['ours']['17'], 'N12800': W['OU'][H]['fwhe_gl']['12800']} for H in ('0.55', '0.75', '0.95')}
for H in D['OU_points']:
    for k in D['OU_points'][H]: D['OU_points'][H][k] = {kk: vv for kk, vv in D['OU_points'][H][k].items() if kk != 'var'}
json.dump(D, open(os.path.join(EV, 'derived_numbers_3b.json'), 'w'), indent=1)
print('tables written')
