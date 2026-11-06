"""Round 5 tables: sharp Monte Carlo reference, J-p ladder, refined law.
Run from working/:  python scripts/round5_tables.py
Inputs : evidence/round5/{P5_mc_reference.json, P5_ladder*.json, ladder_configs.json, ladder_*.npz, big_*.npz,
         P5_cubic_event_p6.json}, evidence/round4/P4_cubic_event.json
Outputs: evidence/round5/P5_summary.json, supplement/source/tables/{tab_mcsharp,tab_ladder,tab_ladcost,tab_lawrefined}.tex,
         source/tables/round5_numbers.tex
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import round5_ladder as L
import round5_mc_summary as MCS
W = os.path.dirname(HERE); E5 = os.path.join(W, 'evidence', 'round5')
HS = ('0.55', '0.75', '0.95'); LEV = np.round(np.arange(0, 1.2001, 0.01), 2); ix = lambda c: int(np.argmin(abs(LEV - c)))
MCR = json.load(open(os.path.join(E5, 'P5_mc_reference.json')))
P4 = json.load(open(os.path.join(W, 'evidence', 'round4', 'P4_cubic_event.json'))); LEV4 = np.round(np.arange(0, 1.5001, 0.05), 2)
P6 = json.load(open(os.path.join(E5, 'P5_cubic_event_p6.json'))); IT = 7
CFG = json.load(open(os.path.join(E5, 'ladder_configs.json')))
REC = {}
for f in ('P5_ladder.json', 'P5_ladder_radau.json', 'P5_ladder_deg.json'):
    p = os.path.join(E5, f)
    if os.path.exists(p):
        for k, v in json.load(open(p)).items():
            for H, r in v.items():
                if isinstance(r, dict) and 'p4' in r: REC.setdefault(k, {}).setdefault(H, {}).update(r)
# order of the ladder in the tables (paper setting first)
ORDER = ['TD9_5', 'TD9_6', 'TD11_5', 'TD13_4', 'TD17_4', 'AN13_8', 'AN13_10', 'AN13_12', 'AN13_16b', 'AN13_20b', 'AN13_24c', 'AN13_28c',
         'AN17_10', 'AN33_10h', 'AN65_10h', 'AN33_16s']
LEVELS_TAB = (0.7, 0.8, 0.9)


def cfg_of(n):
    return L.CONFIGS.get(n) or tuple(CFG[n])


def describe(n):
    c = cfg_of(n)
    if c[0] == 'td': return dict(kind='td', J=c[1], p=c[2], label=f'TD, $J={c[1]}$, $p={c[2]}$')
    J, pj, pt = c[1], list(c[2]), c[3]; k = max(i for i, v in enumerate(pj) if v != pj[-1]) + 1 if any(v != pj[-1] for v in pj) else 0
    head = ','.join(str(v) for v in pj[:k]); tail = pj[-1]
    return dict(kind='an', J=J, pj=pj, ptot=pt, label=f'$J={J}$; $({head},{tail},\\dots)$', head=pj[:k], tail=tail)


def counts(n, H, big):
    if n == 'TD9_6': return None
    p = os.path.join(E5, (f'big_{n}_H{H}.npz' if big else f'ladder_{n}_H{H}.npz'))
    if not os.path.exists(p): return None
    z = np.load(p)
    return z['cnt'] if big else np.array([z[f'cnt4_{b}'] for b in range(len([k for k in z.files if k.startswith('cnt4_')]))])


def deviation(n, H, c):
    ref = MCR[H]['mc_512']['p4'][ix(c)]
    if n == 'TD9_6':
        r = P6[H]['chaos_9,6']; k = int(np.argmin(abs(LEV4 - c))); return dict(dev=100 * (r['p'][IT][k] / ref - 1), ci=100 * r['ci'][IT][k] / ref, M=10 ** 6)
    for big in (True, False):
        C = counts(n, H, big)
        if C is not None:
            m = C.shape[0] * 50000; p = C[:, ix(c)] / 50000.
            return dict(dev=100 * (p.mean() / ref - 1), ci=100 * 1.96 * p.std(ddof=1) / math.sqrt(len(p)) / ref, M=m)
    return None


def law(n, H):
    p = os.path.join(E5, f'big_{n}_H{H}.npz')
    if not os.path.exists(p): return None
    z = np.load(p); pws = z['pw']; hs = z['hist'].astype(np.int64); nb = len(pws)
    full = MCS.mom(pws.sum(0)); per = [MCS.mom(q) for q in pws]; out = {}
    for k in ('var', 'skew', 'exkurt'):
        out[k] = full[k]; out[k + '_ci'] = 1.96 * np.std([q[k] for q in per], ddof=1) / math.sqrt(nb)
    for q in (0.01, 0.05, 0.95, 0.99):
        kq = f'q{int(round(100 * q)):02d}'; out[kq] = MCS.hq(hs.sum(0), q)
        g = [MCS.hq(hs[i:i + 10].sum(0), q) for i in range(0, nb, 10)]; out[kq + '_ci'] = 1.96 * np.std(g, ddof=1) / math.sqrt(len(g))
    r = MCR[H]['mc_512']; cmp = {}
    for k in ('var', 'q01', 'q05', 'q95', 'q99'): cmp[k] = 100 * (out[k] / r[k] - 1); cmp[k + '_ci'] = 100 * out[k + '_ci'] / abs(r[k])
    for k in ('skew', 'exkurt'): cmp[k] = out[k] - r[k]; cmp[k + '_ci'] = out[k + '_ci']
    return cmp


def bigsec(n, H):
    import re
    s = 0.0
    for f in os.listdir(E5):
        if f.startswith('ladder_run') and f.endswith('.log'):
            for line in open(os.path.join(E5, f)):
                m = re.match(rf'big {n} {H} (\d+)', line.strip())
                if m: s = float(m.group(1))
    return s or None


S = {}
for n in ORDER:
    d = describe(n); S[n] = dict(desc=d, H={})
    for H in HS:
        dv = {str(c): deviation(n, H, c) for c in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9)}
        if dv['0.8'] is None: continue
        r = REC.get(n, {}).get(H, {})
        if n == 'TD9_6': r = dict(seconds_solve=P6[H]['chaos_9,6']['seconds_solve'], seconds_eval=P6[H]['chaos_9,6']['seconds_eval'], method='DOP853 (low-memory engine)')
        S[n]['H'][H] = dict(dev=dv, solve_s=r.get('seconds_solve'), eval_s_1e6=r.get('seconds_eval'), nfev=r.get('nfev'), method=r.get('method', 'DOP853'),
                            peak_mb=r.get('peak_mb'), big_eval_s=bigsec(n, H), law=law(n, H))
    if n == 'TD9_6': S[n]['n'] = P6['tensor_p6']['n']; S[n]['entries'] = P6['tensor_p6']['entries']
    else:
        A = L.make_set(cfg_of(n)); S[n]['n'] = int(A.shape[0])
        sc = os.path.join(E5, 'scratch', n); S[n]['entries'] = None
        mp = os.path.join(E5, 'tensor_meta', n + '.json')   # copy of scratch/<name>/meta.json (scratch is deleted)
        if os.path.exists(mp):
            mm = json.load(open(mp)); S[n]['entries'] = mm.get('entries'); S[n]['tensor_mb'] = mm.get('bytes', 0) / 2 ** 20; S[n]['tensor_s'] = mm.get('seconds')


# ---------------------------------------------------------------- LaTeX tables
BEST = {'0.55': 'AN65_10h', '0.75': 'AN65_10h', '0.95': 'AN13_28c'}       # last refinement of each ladder (stopping rule)
UNIFORM = 'AN33_16s'
SUPT = os.path.join(W, 'supplement', 'source', 'tables'); MAINT = os.path.join(W, 'source', 'tables')
P3 = json.load(open(os.path.join(W, 'evidence', 'round3_full', 'P3_cubic_law.json')))
fs = lambda x, d=1: ('$' if round(abs(x), d) == 0 else ('$+' if x >= 0 else '$-')) + f'{abs(x):.{d}f}$'
def wr(p, s): open(p, 'w', encoding='utf-8', newline='\n').write(s)

# (1) sharp Monte Carlo reference
LV = (0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
rows = []; zmax_old = 0.0; zmax_step = 0.0
for H in HS:
    a = MCR[H]['mc_512']; b = MCR[H]['mc_256']; o = P4[H]['mc_512']
    P = [100 * a['p4'][ix(c)] for c in LV]; R = [100 * a['ci4'][ix(c)] / a['p4'][ix(c)] for c in LV]
    zs = [(a['p4'][ix(c)] - b['p4'][ix(c)]) / math.hypot(a['ci4'][ix(c)], b['ci4'][ix(c)]) for c in LV]
    k4 = [int(np.argmin(abs(LEV4 - c))) for c in LV]
    zo = [(o['p'][IT][k] - a['p4'][ix(c)]) / math.hypot(o['ci'][IT][k], a['ci4'][ix(c)]) for c, k in zip(LV, k4)]
    zmax_old = max(zmax_old, max(abs(z) for z in zo)); zmax_step = max(zmax_step, max(abs(z) for z in zs))
    rows.append(f'{H} & $P$ (\\%), $2\\times10^7$ paths & ' + ' & '.join(f'{p:.4f}' if p < 1 else f'{p:.3f}' for p in P) + '\\\\')
    rows.append(' & $\\pm95\\%$ (relative, \\%) & ' + ' & '.join(f'{r:.2f}' for r in R) + '\\\\')
    rows.append(' & $256$ steps, $10^7$ paths ($z$) & ' + ' & '.join(fs(z, 2) for z in zs) + '\\\\')
    rows.append(' & $10^6$ paths, Table~\\ref{tab:cubicevent} ($z$) & ' + ' & '.join(fs(z, 2) for z in zo) + '\\\\' + ('\\midrule' if H != '0.95' else ''))
wr(os.path.join(SUPT, 'tab_mcsharp.tex'), '\\begin{tabular}{ll' + 'r' * len(LV) + '}\\toprule\n$H$ & & ' + ' & '.join(f'$c={c}$' for c in LV) + '\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')

# (2) ladder: deviations and cost
def cell(n, H, c):
    d = S[n]['H'].get(H, {}).get('dev', {}).get(str(c))
    if d is None: return '--'
    s = fs(d['dev']) + '\\,{\\scriptsize(' + f'{d["ci"]:.1f}' + ')}'
    return s + ('$^{*}$' if d['M'] > 10 ** 6 else '')
rows = []; rows_c = []
for n in ORDER:
    d = S[n]['desc']
    lab = d['label']
    rows.append(f'{lab} & {S[n]["n"]} & ' + ' & '.join(cell(n, H, 0.8) for H in HS) + ' & ' + ' & '.join(cell(n, H, 0.9) for H in HS) + '\\\\')
    sol = [S[n]['H'][H]['solve_s'] for H in HS if H in S[n]['H'] and S[n]['H'][H].get('solve_s')]
    ev = [S[n]['H'][H]['eval_s_1e6'] for H in HS if H in S[n]['H'] and S[n]['H'][H].get('eval_s_1e6')]
    pk = [S[n]['H'][H]['peak_mb'] for H in HS if H in S[n]['H'] and S[n]['H'][H].get('peak_mb')]
    met = sorted({S[n]['H'][H]['method'].split(' ')[0] for H in S[n]['H']})
    ent = S[n].get('entries'); tmb = S[n].get('tensor_mb')
    rng = lambda v: (f'{min(v):.0f}' if round(min(v)) == round(max(v)) else f'{min(v):.0f}--{max(v):.0f}') if v else '--'
    rows_c.append(f'{lab} & {S[n]["n"]} & ' + (f'{ent / 1e6:.1f}' if ent else '--') + ' & ' + (f'{tmb:.0f}' if tmb else '--') + ' & ' + rng(pk) + ' & ' + '/'.join(met) + ' & ' + rng(sol) + ' & ' + rng(ev) + '\\\\')
hdr = '\\begin{tabular}{lrrrrrrr}\\toprule\n & & \\multicolumn{3}{c}{$c=0.8$} & \\multicolumn{3}{c}{$c=0.9$}\\\\\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nindex set & $n$ & $H=0.55$ & $0.75$ & $0.95$ & $0.55$ & $0.75$ & $0.95$\\\\\\midrule\n'
wr(os.path.join(SUPT, 'tab_ladder.tex'), hdr + '\n'.join(rows) + '\n\\midrule\nMonte Carlo, $\\pm95\\%$ & & ' + ' & '.join(f'$\\pm{100 * MCR[H]["mc_512"]["ci4"][ix(0.8)] / MCR[H]["mc_512"]["p4"][ix(0.8)]:.1f}$' for H in HS) + ' & ' + ' & '.join(f'$\\pm{100 * MCR[H]["mc_512"]["ci4"][ix(0.9)] / MCR[H]["mc_512"]["p4"][ix(0.9)]:.1f}$' for H in HS) + '\\\\\n\\bottomrule\\end{tabular}\n')
wr(os.path.join(SUPT, 'tab_ladcost.tex'), '\\begin{tabular}{lrrrrlrr}\\toprule\nindex set & $n$ & entries ($10^6$) & tensor (MB) & peak (MB) & integrator & solve (s) & $10^6$ samples (s)\\\\\\midrule\n' + '\n'.join(rows_c) + '\n\\bottomrule\\end{tabular}\n')

# (3) refined law against the sharp reference, and the paper setting for comparison
KEYS = ('var', 'q01', 'q05', 'q95', 'q99', 'skew', 'exkurt')
def lawrow(n, H, big=True):
    if big: c = S[n]['H'][H]['law']
    else:
        z = np.load(os.path.join(E5, f'ladder_{n}_H{H}.npz')); nb = len([k for k in z.files if k.startswith('pw_')])
        pws = [z[f'pw_{b}'][1] for b in range(nb)]; hs = [z[f'hist_{b}'].astype(np.int64) for b in range(nb)]
        full = MCS.mom(sum(pws)); per = [MCS.mom(p) for p in pws]; Hh = sum(hs); r = MCR[H]['mc_512']; c = {}
        for k in ('var', 'skew', 'exkurt'):
            v = full[k]; c[k] = 100 * (v / r[k] - 1) if k == 'var' else v - r[k]
            c[k + '_ci'] = 1.96 * np.std([q[k] for q in per], ddof=1) / math.sqrt(nb) * (100 / r[k] if k == 'var' else 1)
        for q in (0.01, 0.05, 0.95, 0.99):
            kq = f'q{int(round(100 * q)):02d}'; c[kq] = 100 * (MCS.hq(Hh, q) / r[kq] - 1)
            c[kq + '_ci'] = 100 * 1.96 * np.std([MCS.hq(h, q) for h in hs], ddof=1) / math.sqrt(nb) / abs(r[kq])
    return c
def fmt(c):
    return ' & '.join(fs(c[k], 2) if k in ('var', 'q01', 'q05', 'q95', 'q99') else fs(c[k], 3) for k in KEYS)
rows = []; LAW = {}
for H in HS:
    b = BEST[H]; cb = lawrow(b, H); c5 = lawrow('TD9_5', H, big=False); LAW[H] = dict(best=b, best_cmp=cb, td95_cmp=c5)
    t_b = (S[b]['H'][H]['solve_s'] or 0) + (S[b]['H'][H]['big_eval_s'] or 0)
    t_5 = (S['TD9_5']['H'][H]['solve_s'] or 0) + (S['TD9_5']['H'][H]['eval_s_1e6'] or 0)
    rows.append(f'{H} & TD, $J=9$, $p=5$ ($10^6$) & ' + fmt(c5) + f' & {t_5:.0f}\\\\')
    rows.append(f' & {S[b]["desc"]["label"]} ($10^7$) & ' + fmt(cb) + f' & {t_b:.0f}\\\\')
    if H in S.get(UNIFORM, {}).get('H', {}) and S[UNIFORM]['H'][H].get('law'):
        cu = S[UNIFORM]['H'][H]['law']; LAW[H]['uniform_cmp'] = cu
        t_u = (S[UNIFORM]['H'][H]['solve_s'] or 0) + (S[UNIFORM]['H'][H]['big_eval_s'] or 0)
        rows.append(f' & {S[UNIFORM]["desc"]["label"]} ($10^7$) & ' + fmt(cu) + f' & {t_u:.0f}\\\\')
    r = MCR[H]['mc_512']
    rows.append(' & Monte Carlo, $2\\times10^7$ ($\\pm95\\%$) & ' + ' & '.join((f'$\\pm{100 * r[k + "_ci"] / abs(r[k]):.2f}$' if k in ('var', 'q01', 'q05', 'q95', 'q99') else f'$\\pm{r[k + "_ci"]:.3f}$') for k in KEYS) + f' & {r["seconds"]:.0f}\\\\' + ('\\midrule' if H != '0.95' else ''))
wr(os.path.join(SUPT, 'tab_lawrefined.tex'), '\\begin{tabular}{llrrrrrrrr}\\toprule\n & & \\multicolumn{5}{c}{relative deviation (\\%)} & \\multicolumn{2}{c}{difference} & \\\\\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\n$H$ & CosChaos & var. & $q_{0.01}$ & $q_{0.05}$ & $q_{0.95}$ & $q_{0.99}$ & skew. & ex.\\ kurt. & time (s)\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')
json.dump(dict(summary=S, law=LAW, zmax_old=zmax_old, zmax_step=zmax_step), open(os.path.join(E5, 'P5_summary.json'), 'w'), indent=1, default=float)
print('tables written', round(zmax_old, 2), round(zmax_step, 2))


# (4) main-table rows (refined sets against the 2x10^7 reference) and macros
SH = json.load(open(os.path.join(E5, 'P5_mode_shares.json')))
main_rows = {}
for H in HS:
    b = BEST[H]; cb = LAW[H]['best_cmp']; r = MCR[H]['mc_512']
    t_b = (S[b]['H'][H]['solve_s'] or 0) + (S[b]['H'][H]['big_eval_s'] or 0)
    lab = ('$J=%d$, $p_0=%d$$^{\\dagger}$' % (S[b]['desc']['J'], S[b]['desc']['pj'][0]))
    main_rows[H] = (f' & {lab} & ' + fmt(cb) + f' & {t_b:.0f}\\\\\n' +
                    ' & Monte Carlo, $2\\times10^7$$^{\\dagger}$ & ' + ' & '.join((f'$\\pm{100 * r[k + "_ci"] / abs(r[k]):.2f}$' if k in ('var', 'q01', 'q05', 'q95', 'q99') else f'$\\pm{r[k + "_ci"]:.3f}$') for k in KEYS) + f' & {r["seconds"]:.0f}\\\\\n')
json.dump(main_rows, open(os.path.join(E5, 'P5_main_rows.json'), 'w'), indent=1)
dv = lambda c: [S[BEST[H]]['H'][H]['dev'][str(c)]['dev'] for H in HS]
allc = [abs(x) for c in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9) for x in dv(c)]
qmax = max(abs(LAW[H]['best_cmp'][k]) for H in HS for k in ('q01', 'q05', 'q95', 'q99'))
vmax = max(abs(LAW[H]['best_cmp']['var']) for H in HS)
smax = max(abs(LAW[H]['best_cmp'][k]) for H in HS for k in ('skew', 'exkurt'))
td_q = max(abs(LAW[H]['td95_cmp'][k]) for H in HS for k in ('q01', 'q05', 'q95', 'q99')); td_v = max(abs(LAW[H]['td95_cmp']['var']) for H in HS)
ci8 = [S[BEST[H]]['H'][H]['dev']['0.8']['ci'] for H in HS]
tb = [(S[BEST[H]]['H'][H]['solve_s'] or 0) + (S[BEST[H]]['H'][H]['big_eval_s'] or 0) for H in HS]; tm = [MCR[H]['mc_512']['seconds'] for H in HS]
def pchg(a, b, H='0.95', c=0.8):
    A = counts(a, H, True); B = counts(b, H, True); k = ix(c); d = (B[:, k] - A[:, k]).astype(float); m = A[:, k].mean()
    return 100 * d.mean() / m, 100 * 1.96 * d.std(ddof=1) / math.sqrt(len(d)) / m
ch1 = pchg('AN13_20b', 'AN13_24c'); ch2 = pchg('AN13_24c', 'AN13_28c'); chJ = pchg('AN33_10h', 'AN65_10h', '0.55')
M = dict(TailOldZ=f'{zmax_old:.2f}', TailStepZ=f'{zmax_step:.2f}',
         TailShareFF=f'{100 * SH["0.55"]["cum"]["1"]:.0f}', TailShareSF=f'{100 * SH["0.75"]["cum"]["1"]:.0f}', TailShareNF=f'{100 * SH["0.95"]["cum"]["1"]:.0f}',
         TailShareNineFF=f'{100 * SH["0.55"]["cum"]["9"]:.1f}',
         TailPzero=str(S[BEST['0.95']]['desc']['pj'][0]), TailJ=str(S[BEST['0.55']]['desc']['J']),
         TailDevEightMax=f'{max(abs(x) for x in dv(0.8)):.1f}', TailDevAllMax=f'{max(allc):.1f}',
         TailPmin=f'{min(100 * MCR[H]["mc_512"]["p4"][ix(0.9)] for H in HS):.2f}',
         TailCIEight=f'{min(ci8):.1f}--{max(ci8):.1f}', TailHWNF=f'{100 * MCR["0.95"]["mc_512"]["ci4"][ix(0.8)] / MCR["0.95"]["mc_512"]["p4"][ix(0.8)]:.2f}',
         TailChgOne=f'{abs(ch1[0]):.2f}', TailChgTwo=f'{abs(ch2[0]):.2f}', TailChgJ=f'{abs(chJ[0]):.1f}', TailChgJci=f'{chJ[1]:.1f}',
         LawRefVarMax=f'{vmax:.2f}', LawRefQMax=f'{qmax:.2f}', LawRefShapeMax=f'{smax:.3f}', LawTdQMax=f'{td_q:.1f}', LawTdVarMax=f'{td_v:.1f}',
         TailCostLow=f'{min(tb) / 1e3:.1f}', TailCostHigh=f'{max(tb) / 1e3:.1f}', TailMCLow=f'{min(tm) / 1e3:.1f}', TailMCHigh=f'{max(tm) / 1e3:.1f}')
d8 = lambda n, H='0.95': S[n]['H'][H]['dev']['0.8']['dev']
four = [d8(n, H) for n in ('TD13_4', 'TD17_4') for H in HS]; low = [abs(d8(n)) for n in ('TD9_6', 'TD11_5')]
M.update(TailTDFourLow=f'{min(four):.0f}', TailTDFourHigh=f'{max(four):.0f}', TailTDLowLow=f'{min(low):.0f}', TailTDLowHigh=f'{max(low):.0f}',
         TailJdevA=f'{d8("AN13_10", "0.55"):.1f}', TailJdevB=f'{d8("AN33_10h", "0.55"):.1f}', TailJdevC=f'{d8("AN65_10h", "0.55"):.1f}')
ci9 = [S[BEST[H]]['H'][H]['dev']['0.9']['ci'] for H in HS]
M.update(TailEffN=f'{S["AN13_16b"]["n"]:,}'.replace(',', '{,}'), TailEffDev=f'{abs(d8("AN13_16b")):.1f}',
         TailTDN=f'{S["TD9_6"]["n"]:,}'.replace(',', '{,}'), TailTDDev=f'{abs(d8("TD9_6")):.0f}',
         TailDevSevenMax=f'{max(abs(x) for x in dv(0.7)):.1f}', TailCINine=f'{max(ci9):.1f}')
if all(H in S.get(UNIFORM, {}).get('H', {}) for H in HS):
    M.update(TailUnifEightMax=f'{max(abs(d8(UNIFORM, H)) for H in HS):.1f}', TailUnifQMax=f'{max(abs(LAW[H]["uniform_cmp"][k]) for H in HS for k in ("q01", "q05", "q95", "q99")):.2f}')
wr(os.path.join(MAINT, 'round5_numbers.tex'), '% Round 5 numbers (generated by scripts/round5_tables.py; do not edit)\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in M.items()))
json.dump(dict(macros=M, chg_deg=[ch1, ch2], chg_J55=chJ), open(os.path.join(E5, 'P5_derived_numbers.json'), 'w'), indent=1)
print(M)

# (5) refined curve for main-text Fig. 9(d)
RC = {'levels': LEV.tolist()}
for H in HS:
    C = counts(BEST[H], H, True); p = C / 50000.
    RC[H] = dict(set=BEST[H], label=S[BEST[H]]['desc']['label'], samples=int(C.shape[0] * 50000), p=p.mean(0).tolist(), ci=(1.96 * p.std(0, ddof=1) / math.sqrt(len(p))).tolist())
json.dump(RC, open(os.path.join(E5, 'P5_refined_curve.json'), 'w'), indent=1)
