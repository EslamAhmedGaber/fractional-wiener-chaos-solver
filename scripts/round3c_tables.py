"""tables and number macros (every number quoted in the text comes from here).
Reads evidence/round3_full/P1_pathwise.json, P2_wick_law.json, P3_cubic_law.json, reference_checks.json and
evidence/round3_fwhe/workprecision.json; writes source/tables/tab_pathou.tex, tab_wicklaw.tex (main text),
tab_cubiclaw.tex (main text), tab_cubicpath.tex, tab_wicksamples.tex, tab_cubiclawfull.tex (supplement) and
source/tables/round3_numbers.tex (LaTeX macros).  Run from the repository root:  python scripts/round3c_tables.py
"""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FD = os.path.join(ROOT, 'evidence', 'round3_full'); EV = os.path.join(ROOT, 'evidence', 'round3_fwhe'); TB = os.path.join(ROOT, 'source', 'tables')
P1 = json.load(open(os.path.join(FD, 'P1_pathwise.json'))); P2 = json.load(open(os.path.join(FD, 'P2_wick_law.json')))
P3 = json.load(open(os.path.join(FD, 'P3_cubic_law.json'))); RC = json.load(open(os.path.join(FD, 'reference_checks.json')))
W = json.load(open(os.path.join(EV, 'workprecision.json')))
HS = ['0.55', '0.75', '0.95']; MAC = {}


def sci(x, d=1):
    if x == 0: return '0'
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) >= 10: m /= 10; e += 1
    return f'{m:.{d}f}\\times10^{{{e}}}'


def pct(x, d=None):
    v = 100 * x; d = (2 if abs(round(v, 2)) < 10 else 1 if abs(round(v, 1)) < 100 else 0) if d is None else d
    return f'${v:+.{d}f}$'.replace('+', '+' if v >= 0 else '')


def w(name, s): open(os.path.join(TB, name + '.tex'), 'w', encoding='utf-8').write(s)


# ---------------- pathwise OU (main)
Js = ['5', '9', '17', '33', '65', '129', '257', '513']
rows = []
for k, J in enumerate(Js):
    cells = []
    for H in HS:
        e = P1[H]['ou'][J]['t1']; cells.append(f'{e:.4f}')
        if k == 0: cells.append('')
        else:
            e0 = P1[H]['ou'][Js[k - 1]]['t1']; cells.append(f'{math.log(e0 / e) / math.log(int(J) / int(Js[k - 1])):.2f}')
    rows.append(f'{J} & ' + ' & '.join(cells) + '\\\\')
w('tab_pathou', '\\begin{tabular}{rcccccc}\\toprule\n & \\multicolumn{2}{c}{$H=0.55$} & \\multicolumn{2}{c}{$H=0.75$} & \\multicolumn{2}{c}{$H=0.95$}\\\\'
  '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n$J$ & error & order & error & order & error & order\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')
MAC['OUpathNineFiveFiveThirteen'] = f"{100 * P1['0.95']['ou']['513']['t1']:.2f}\\%"
MAC['OUpathFiveFiveFiveThirteen'] = f"{100 * P1['0.55']['ou']['513']['t1']:.1f}\\%"

# ---------------- Wick law at t = T (main)
QN = [('var', 'variance'), ('skew', 'skewness'), ('exkurt', 'excess kurtosis'), ('q01', '$1\\%$ quantile'), ('q99', '$99\\%$ quantile')]
rows = []
for k, lab in QN:
    for i, H in enumerate(HS):
        d = P2[H]['2.0']
        rows.append(f"{lab if i == 0 else ''} & {H} & " + ' & '.join(pct(d[f'65,{p}'][k]) for p in range(1, 9)) + '\\\\')
    rows.append('\\midrule' if k != 'q99' else '')
w('tab_wicklaw', '\\begin{tabular}{llrrrrrrrr}\\toprule\nquantity & $H$ & $p=1$ & $2$ & $3$ & $4$ & $5$ & $6$ & $7$ & $8$\\\\\\midrule\n' + '\n'.join(r for r in rows if r) + '\n\\bottomrule\\end{tabular}\n')
pw = {H: [P1[H]['wick'][f'17,{p}']['tT'] for p in range(1, 6)] for H in HS}
MAC['WickPathTpOneLow'] = f"{100 * min(v[0] for v in pw.values()):.0f}"; MAC['WickPathTpOneHigh'] = f"{100 * max(v[0] for v in pw.values()):.0f}"
MAC['WickPathTpFiveLow'] = f"{100 * min(v[4] for v in pw.values()):.1f}"; MAC['WickPathTpFiveHigh'] = f"{100 * max(v[4] for v in pw.values()):.1f}"
MAC['WickVarPthreeHigh'] = f"{100 * max(abs(P2[H]['2.0']['65,3']['var']) for H in HS):.1f}"
MAC['WickQnineninePfiveHigh'] = f"{100 * max(abs(P2[H]['2.0']['65,5']['q99']) for H in HS):.2f}"
P2b = json.load(open(os.path.join(FD, 'P2b_samples.json')))
rows = []
for H in HS:
    b = P2b[H]; q = b['quadrature']
    rows.append(f'{H} & ' + ' & '.join(f"${b['samples'][k][0]:.4f}\\pm{b['samples'][k][1]:.2g}$ ({q[k]:.4f})" for k, _ in QN) + f" & {b['seconds_eval']:.0f}\\\\")
MAC['WickSampleSecLow'] = f"{min(P2b[H]['seconds_eval'] for H in HS):.0f}"; MAC['WickSampleSecHigh'] = f"{max(P2b[H]['seconds_eval'] for H in HS):.0f}"
MAC['WickSampleZmax'] = f"{max(abs(P2b[H]['samples'][k][0] - P2b[H]['quadrature'][k]) / P2b[H]['samples'][k][1] for H in HS for k, _ in QN):.1f}"
w('tab_wicksamples', '\\begin{tabular}{lcccccc}\\toprule\n$H$ & variance & skewness & excess kurtosis & $1\\%$ quantile & $99\\%$ quantile & time (s)\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')

# ---------------- cubic pathwise (supplement)
rows = []
for H in HS:
    c = P1[H]['cubic']
    for k in ['5,1', '5,2', '9,1', '9,2', '9,3', '9,4', '17,2', '17,3', '33,2', '65,1', '65,2']:
        J, p = k.split(',')
        rows.append(f"{H if k == '5,1' else ''} & {J} & {p} & {c[k]['t1']:.3f} & {c[k]['tT']:.3f} & {c[k]['seconds']:.1f}\\\\")
    rows.append('\\midrule')
w('tab_cubicpath', '\\begin{tabular}{rrrccc}\\toprule\n$H$ & $J$ & $p$ & error, $t=1$ & error, $t=4$ & solve (s)\\\\\\midrule\n' + '\n'.join(rows[:-1]) + '\n\\bottomrule\\end{tabular}\n')
MAC['CubicPathFour'] = {H: P1[H]['cubic']['9,3']['tT'] for H in HS}

# ---------------- cubic law against Monte Carlo
QC = [('mean', 'mean'), ('var', 'variance'), ('skew', 'skewness'), ('exkurt', 'excess kurtosis'), ('q01', '$1\\%$ q.'), ('q05', '$5\\%$ q.'), ('q95', '$95\\%$ q.'), ('q99', '$99\\%$ q.')]
def cell(v, ci, d=4): return f'{v:.{d}f}\\,$\\pm${ci:.{d}f}'
rows = []; full = []
best = {}
for H in HS:
    R = P3[H]; mc = R['mc_512']['4.0']; mc256 = R['mc_256']['4.0']
    for key, lab in QC:
        v, ci = mc[key]; row = [cell(v, ci)]
        for cf in ['chaos_9,1', 'chaos_9,2', 'chaos_9,3', 'chaos_9,4', 'chaos_17,3']:
            vv, cc = R[cf]['4.0'][key]; row.append(f'{vv:.4f}')
        full.append(f"{H if key == 'mean' else ''} & {lab} & " + ' & '.join(row) + f' & {mc256[key][0]:.4f}\\\\')
    full.append('\\midrule')
w('tab_cubiclawfull', '\\begin{tabular}{llccccccc}\\toprule\n & & MC, $512$ steps & \\multicolumn{4}{c}{CosChaos, $J=9$} & chaos & MC, $256$\\\\\\cmidrule(lr){4-7}\n'
  '$H$ & quantity ($t=4$) & ($95\\%$ CI) & $p=1$ & $p=2$ & $p=3$ & $p=4$ & $J=17$, $p=3$ & steps\\\\\\midrule\n' + '\n'.join(full[:-1]) + '\n\\bottomrule\\end{tabular}\n')
# main: deviations from Monte Carlo at t = 4 (relative % for variance and quantiles, absolute for skewness and kurtosis)
REL = ['var', 'q01', 'q05', 'q95', 'q99']; ABS = ['skew', 'exkurt']; rows = []; match = {}
for H in HS:
    R = P3[H]; mc = R['mc_512']['4.0']
    for cf, lab in [('chaos_9,1', '$J=9$, $p=1$'), ('chaos_9,3', '$J=9$, $p=3$'), ('chaos_9,4', '$J=9$, $p=4$'), ('chaos_17,3', '$J=17$, $p=3$')]:
        cells = [f"${100 * (R[cf]['4.0'][k][0] / mc[k][0] - 1):+.1f}$" for k in REL] + [f"${R[cf]['4.0'][k][0] - mc[k][0]:+.3f}$" for k in ABS]
        t = R[cf]['seconds_solve'] + R[cf]['seconds_eval']
        rows.append(f"{H if cf == 'chaos_9,1' else ''} & {lab} & " + ' & '.join(cells) + f' & {t:.0f}\\\\')
    cells = [f"$\\pm{100 * mc[k][1] / abs(mc[k][0]):.1f}$" for k in REL] + [f"$\\pm{mc[k][1]:.3f}$" for k in ABS]
    rows.append(f" & Monte Carlo & " + ' & '.join(cells) + f" & {R['mc_512']['seconds']:.0f}\\\\")
    rows.append('\\midrule')
    e = abs(R['chaos_9,3']['4.0']['q99'][0] / mc['q99'][0] - 1); c = mc['q99'][1] / abs(mc['q99'][0])
    M = 1e6 * (c / e) ** 2; match[H] = dict(M=M, seconds=R['mc_512']['seconds'] * M / 1e6, chaos=R['chaos_9,3']['seconds_solve'] + R['chaos_9,3']['seconds_eval'])
w('tab_cubiclaw', '\\begin{tabular}{llrrrrrrrr}\\toprule\n & & \\multicolumn{5}{c}{relative deviation (\\%)} & \\multicolumn{2}{c}{difference} & \\\\\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\n'
  '$H$ & CosChaos & var. & $q_{0.01}$ & $q_{0.05}$ & $q_{0.95}$ & $q_{0.99}$ & skew. & ex.\\ kurt. & time (s)\\\\\\midrule\n' + '\n'.join(rows[:-1]) + '\n\\bottomrule\\end{tabular}\n')
MAC['CubicMatchMLow'] = sci(min(v['M'] for v in match.values())); MAC['CubicMatchMHigh'] = sci(max(v['M'] for v in match.values()))
MAC['CubicMatchSecLow'] = f"{min(v['seconds'] for v in match.values()):.0f}"; MAC['CubicMatchSecHigh'] = f"{max(v['seconds'] for v in match.values()):.0f}"
MAC['CubicMatch'] = match
qq = [100 * abs(P3[H]['chaos_9,3']['4.0']['q99'][0] / P3[H]['mc_512']['4.0']['q99'][0] - 1) for H in HS]
MAC['CubicQnnNine'] = f'{min(qq):.1f}--{max(qq):.1f}'
qa = [abs(100 * (P3[H][cf]['4.0'][k][0] / P3[H]['mc_512']['4.0'][k][0] - 1)) for H in HS for cf in ('chaos_9,3', 'chaos_9,4', 'chaos_17,3') for k in ('q01', 'q05', 'q95', 'q99')]
MAC['CubicQall'] = f'{min(qa):.1f}--{max(qa):.1f}'
MAC['CubicQallMax'] = f'{max(qa):.1f}'
STATS_KEYS = ['mean', 'var', 'skew', 'exkurt', 'q01', 'q05', 'q95', 'q99']

names = {'0.55': 'FiveFive', '0.75': 'SevenFive', '0.95': 'NineFive'}
for H in HS:
    R = P3[H]; mc = R['mc_512']['4.0']
    for cf, tag in (('chaos_17,3', 'Seventeen'), ('chaos_9,3', 'Nine')):
        v = 100 * (R[cf]['4.0']['var'][0] / mc['var'][0] - 1); qs = [100 * (R[cf]['4.0'][k][0] / mc[k][0] - 1) for k in ('q01', 'q05', 'q95', 'q99')]
        MAC[f'CubicVar{tag}{names[H]}'] = f'{v:+.2f}\\%'; MAC[f'CubicQ{tag}{names[H]}'] = f'{min(qs):.1f}--{max(qs):.1f}\\%'
    MAC[f'CubicKurtMC{names[H]}'] = f"{mc['exkurt'][0]:.2f}"; MAC[f'CubicKurtDiff{names[H]}'] = f"{R['chaos_17,3']['4.0']['exkurt'][0] - mc['exkurt'][0]:+.2f}"
    m1 = R['mc_512']['1.0']; MAC[f'CubicVarTone{names[H]}'] = f"{100 * (R['chaos_17,3']['1.0']['var'][0] / m1['var'][0] - 1):+.2f}\\%"
dev = max(abs(P3[H]['mc_256'][t][k][0] - P3[H]['mc_512'][t][k][0]) / P3[H]['mc_512'][t][k][1] for H in HS for t in ('1.0', '4.0') for k in STATS_KEYS)
MAC['CubicStepDev'] = f'{dev:.1f}'

# ---------------- macros from the work-precision and reference checks
O = W['OU']; Wk = W['Wick']; Mm = W['memory']
MAC['OUnineJseventeenErr'] = sci(O['0.95']['ours']['17']['max_rel']); MAC['OUnineJseventeenSec'] = f"{O['0.95']['ours']['17']['seconds']:.2f}"
MAC['OUnineNtwelveSec'] = f"{O['0.95']['fwhe_gl']['12800']['seconds']:.2f}"
gl = [min(d['max_rel'] for d in Wk[H]['fwhe_gl_k1'].values()) for H in HS]
MAC['WickGLfloorLow'] = f"{100 * min(gl):.0f}\\%"; MAC['WickGLfloorHigh'] = f"{100 * max(gl):.0f}\\%"
MAC['WickOursOneCentSec'] = f"{max(min(d['seconds'] for d in Wk[H]['ours'].values() if d['max_rel'] <= 1e-2) for H in HS):.2f}"
MAC['MemNineFiveMaxrel'] = sci(Mm['0.95']['ours']['129']['max_rel']); MAC['MemNineFiveGLsixteen'] = sci(Mm['0.95']['fwhe_gl']['1600']['max_rel'])
MAC['MemNineFiveGLtwelve'] = sci(Mm['0.95']['fwhe_gl']['12800']['max_rel']); MAC['MemFiveFiveGLsixteen'] = sci(Mm['0.55']['fwhe_gl']['1600']['max_rel'])
MAC['MemFiveFiveOurs'] = sci(Mm['0.55']['ours']['129']['max_rel']); MAC['MemCachedSec'] = f"{min(Mm[H]['ours_cached']['9']['seconds'] for H in HS):.1f}--{max(Mm[H]['ours_cached']['257']['seconds'] for H in HS):.1f}"
MAC['MemOldSec'] = f"{Mm['speedup']['seconds_v111']:.0f}"; MAC['MemNewSec'] = f"{Mm['speedup']['seconds_v113']:.0f}"
MAC['MemNineFiveCachedSec'] = f"{Mm['0.95']['ours_cached']['129']['seconds']:.2f}"; MAC['MemNineFiveGLtwelveSec'] = f"{Mm['0.95']['fwhe_gl']['12800']['seconds']:.2f}"
MAC['RefOUzmax'] = f"{max(abs(RC[H]['z']) for H in RC):.2f}"; MAC['RefWickzmax'] = f"{max(abs(RC[H]['wick_z']) for H in RC):.2f}"
rr = []
for H in ['0.5', '0.55', '0.7', '0.75', '0.95']:
    r = Mm[f'ref_{H}']; rr.append((max(abs(np.array(r['richardson']) / np.array(r['r1601']) - 1)), abs(r['richardson'][-1] / r['r1601'][-1] - 1)))
MAC['RefMemRichMax'] = sci(max(a for a, b in rr)); MAC['RefMemRichTwenty'] = sci(max(b for a, b in rr))
# cubic cost
c = {H: P3[H] for H in HS}
MAC['CubicMCSec'] = f"{min(c[H]['mc_512']['seconds'] for H in HS):.0f}--{max(c[H]['mc_512']['seconds'] for H in HS):.0f}"
MAC['CubicChaosThreeSec'] = f"{min(c[H]['chaos_9,3']['seconds_solve'] + c[H]['chaos_9,3']['seconds_eval'] for H in HS):.0f}--{max(c[H]['chaos_9,3']['seconds_solve'] + c[H]['chaos_9,3']['seconds_eval'] for H in HS):.0f}"
MAC['CubicChaosFourSec'] = f"{min(c[H]['chaos_9,4']['seconds_solve'] + c[H]['chaos_9,4']['seconds_eval'] for H in HS):.0f}--{max(c[H]['chaos_9,4']['seconds_solve'] + c[H]['chaos_9,4']['seconds_eval'] for H in HS):.0f}"
out = ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in MAC.items() if isinstance(v, str))
open(os.path.join(TB, 'round3_numbers.tex'), 'w', encoding='utf-8').write('% generated by scripts/round3c_tables.py\n' + out)
json.dump(MAC, open(os.path.join(FD, 'derived_numbers_3c.json'), 'w'), indent=1)
print(out)
