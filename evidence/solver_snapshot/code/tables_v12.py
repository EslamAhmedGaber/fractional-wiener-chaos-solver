# Generates the LaTeX tables of the solver examples from the result files (v12).
import json, os, sys, numpy as np
EXJ = json.load(open('v6/results/v12_examples.json')); OUI = EXJ['ou_interior']
CU = json.load(open('v6/results/v12_cubic.json')); C8 = json.load(open('v6/results/cubic_v8.json'))
OS = json.load(open('v6/results/v12_oscillator.json')); BE = json.load(open('v6/results/v12_bench.json'))
NW = json.load(open('v6/results/v12_new.json'))
def w(name, s): open(f'v6/paper/tables/{name}.tex', 'w', encoding='utf-8').write(s)
def e(x, d=2):
    m, ex = f'{x:.{d}e}'.split('e'); return f'${m}\\times10^{{{int(ex)}}}$'
def e1(x): return e(x, 1)
def sg(x, d=1): return f'${x:+.{d}f}$'.replace('+0.0$', '0.0$') if abs(x) >= 0.05 else '$0.0$'
# T2 OU
rows = []
for H in ['0.5', '0.55', '0.75', '0.95']:
    o = EXJ['ou'][H]; ex = o['exact'][-1]
    endp = [100 * (o[str(J)]['var'][-1] / ex - 1) for J in [5, 17, 65]]
    inter = [OUI[H][i] * 100 for i in [0, 2, 4]]
    rows.append(f"{H} & {ex:.6f} & " + ' & '.join(f'{x:+.4f}' for x in endp) + ' & ' + ' & '.join(f'{x:.3f}' for x in inter) + ' \\\\')
w('tab_ou', "\\begin{tabular}{rccccccc}\\toprule\n & & \\multicolumn{3}{c}{signed error at $t=T=2$ (\\%)} & \\multicolumn{3}{c}{$|$error$|$ at $t=1$ (\\%)}\\\\\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n$H$ & $\\operatorname{Var}X(2)$ exact & $J=5$ & $J=17$ & $J=65$ & $J=5$ & $J=17$ & $J=65$\\\\\\midrule\n"
  + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}")

# T3 memory
rows = []
for H in ['0.5', '0.7', '0.95']:
    m = EXJ['memory'][H]; fw = m['fwhe57']['20.0']; g = m['gauss']
    rows.append(f"{H} & {fw:.6e} & " + ' & '.join(f"{100*(g[str(J)]['20.0']['var1']/fw-1):+.4f}" for J in [9, 33, 129, 257])
                + f" & {100*(g['257']['20.0']['var1_diag']/fw-1):+.2f} & {m['65']['var2'][-1]:.2e} \\\\")
w('tab_memory', "\\begin{tabular}{rccccccc}\\toprule\n & & \\multicolumn{4}{c}{deviation of the solver from (FWHE) (\\%)} & & \\\\\\cmidrule(lr){3-6}\n$H$ & $\\sigma^2\\langle g,g\\rangle_H$, $t=20$ & $J=9$ & $J=33$ & $J=129$ & $J=257$ & diag.\\ $G$ (\\%) & 2nd chaos \\\\\\midrule\n"
  + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}")

# T4 population
prev = {'0.55': 1.3749, '0.65': 2.126, '0.75': 3.3356, '0.85': 5.2964, '0.95': 8.4957}
rows = []
for H in ['0.5', '0.55', '0.75', '0.95']:
    v = EXJ['popJ'][H]; vinf = v['129'] + (v['129'] - v['65']) / (2 ** (2 * float(H)) - 1)
    p2 = {'0.55': EXJ['pop']['0.55']['p2'], '0.75': EXJ['pop']['0.75']['p2'], '0.95': EXJ['pop']['0.95']['p2']}.get(H)
    p2s = f"{max(p2['var']):.4f}" if p2 else (f"{EXJ['popH05_p2']['33']:.4f}")
    rows.append(f"{H} & " + ' & '.join(f"{v[str(J)]:.4f}" for J in [17, 33, 65, 129]) + f" & {vinf:.4f} & {p2s} & {prev.get(H, '--')} \\\\")
w('tab_pop', "\\begin{tabular}{rccccccc}\\toprule\n & \\multicolumn{4}{c}{degree $p=1$, $\\max_t\\operatorname{Var}X(t)$} & & $p=2$ & previous \\\\\\cmidrule(lr){2-5}\n$H$ & $J=17$ & $J=33$ & $J=65$ & $J=129$ & extrap. & $J=33$ & RL--F $N=65$ \\\\\\midrule\n"
  + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}")

# T5 classical limit
lim = EXJ['limit']; rows = []
for i, H in enumerate(lim['H']):
    if H in (0.5, 0.501, 0.51, 0.55, 0.6, 0.7):
        rows.append(f"{H} & {lim['G_minus_I_max'][i]:.4f} & {lim['offdiag_corr'][i]:.4f} & {lim['ou_exact_T'][i]:.6f} & {lim['ou_var_T'][i]:.6f} & {lim['pop_maxvar_J33'][i]:.4f} \\\\")
w('tab_limit', "\\begin{tabular}{rccccc}\\toprule\n$H$ & $\\max|G-I|$ & max corr. & OU exact & OU, $J=65$ & population, $J=33$ \\\\\\midrule\n"
  + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}")

# cubic drift
rows = []
for sig, key in [('0.25', 'sigma0.25'), ('0.5', 'sigma0.5')]:
    for H in ['0.6', '0.75', '0.9']:
        mc = C8[key]['mc'][f'{H}_512']; tq = [0.5, 1.0, 2.0, 4.0]
        for t in [1.0, 4.0]:
            i = tq.index(t); v = mc['var'][i]
            def err(J, p):
                r = CU['sw'][f'{sig}_{H}_{J}_{p}']; j = [round(x, 6) for x in r['t']].index(t); return 100 * (r['var'][j] / v - 1)
            p4 = sg(100 * (CU['sw9'][f'{H}_9_4']['var'][i] / v - 1)) if sig == '0.5' else '--'
            r2 = CU['sw'][f'{sig}_{H}_65_2']; j = [round(x, 6) for x in r2['t']].index(t)
            rows.append(f"{sig} & {H} & {t:g} & {v:.4f} & {sg(err(65, 1))} & {sg(err(65, 2))} & {p4} & ${1e3*mc['m3'][i]:.3f}$ & ${1e3*r2['m3'][j]:.3f}$\\\\")
    rows.append('\\addlinespace')
w('tab_cubic', "\\begin{tabular}{rrrcccccc}\\toprule\n & & & MC & \\multicolumn{3}{c}{error of Var (\\%)} & \\multicolumn{2}{c}{third moment ($\\times10^3$)}\\\\\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n$\\sigma$ & $H$ & $t$ & Var $X(t)$ & $p=1$ & $p=2$ & $p=4$, $J=9$ & MC & $p=2$\\\\\\midrule\n"
  + '\n'.join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\n")

# oscillator
rows = []
for r in OS['rows']:
    if r['t'] not in (10.0, 20.0): continue
    rows.append(f"{r['H']:g} & {r['t']:g} & {r['varQ']:.6f} & {r['varV']:.6f} & {r['EE']:.6f} & {r['VarE']:.6f} & {r['ratio']:.4f} & {e1(r['errQ']['257'])} & {e1(r['errV']['257'])}\\\\")
w('tab_osc', "\\begin{tabular}{rrccccccc}\\toprule\n & & \\multicolumn{5}{c}{reference (quadrature; Lyapunov at $H=\\frac12$)} & \\multicolumn{2}{c}{solver error, $J=257$}\\\\\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\n$H$ & $t$ & Var $Q$ & Var $V$ & $\\E E$ & Var $E$ & $\\omega^2$Var$Q$/Var$V$ & Var $Q$ & Var $V$\\\\\\midrule\n"
  + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}\n")

# cost
rows = []; first = True; OULAB = 'OU, $H=0.75$, $t=0.5$'
for c in BE['ou']['chaos']:
    if c['J'] in (9, 33, 129):
        rows.append(f"{OULAB if first else ''} & solver, $J={c['J']}$ & {1e3*c['time']:.1f} & {e1(c['relerr'])} \\\\"); first = False
for m in BE['ou']['mc']:
    rows.append(f" & exact-grid MC, $M=2^{{{int(np.log2(m['M']))}}}$, 1024 steps & {1e3*m['time']:.0f} & {e1(m['rmse'])} \\\\")
rows.append('\\addlinespace'); first = True; POPLAB = 'population, $H=\\frac12$, $t=11.25$'; OULAB = 'OU, $H=0.75$, $t=0.5$'
for c in BE['pop']['chaos']:
    if c['J'] in (65, 129, 257):
        rows.append(f"{POPLAB if first else ''} & solver $p=1$, $J={c['J']}$ & {1e3*c['time']:.0f} & {e1(abs(c['relerr']))} \\\\"); first = False
R_ = BE['pop']['richardson_129_257']; m2 = BE['pop']['mc_2e5']
rows.append(f" & solver, extrapolated from $J=129,257$ & {1e3*R_['time']:.0f} & {e1(abs(R_['relerr']))} \\\\")
rows.append(f" & It\\^o MC, $2\\times10^5$ paths, $\\Delta t=0.005$ & {1e3*m2['time']:.0f} & s.e.\\ {e1(m2['se']/1.1361)} \\\\")
w('tab_cost', "\\begin{tabular}{llrr}\\toprule\nproblem & method & time (ms) & rel.\\ error\\\\\\midrule\n" + '\n'.join(rows) + "\n\\bottomrule\\end{tabular}\n")

# new example: Duffing and pendulum
rows = []
for sys_, lab in [('duff', 'Duffing'), ('pend', 'pendulum')]:
    for H in ['0.6', '0.75', '0.9']:
        mc = NW[sys_][f'mc_{H}_1000']
        for ti, t in enumerate(NW['t']):
            if t not in (2.0, 10.0): continue
            vQ = mc['var'][ti][0]; seQ = mc['se_var'][ti][0]
            if sys_ == 'duff':
                cols = [NW['duff'][f'{H}_{k}']['var'][ti][0] for k in ['65_1', '65_2', '17_3']]
            else:
                cols = [NW['pend'][f'{H}_65_1']['var'][ti][0], None, None]
            fz = lambda x: '0.0' if abs(x) < 0.05 else f'{x:+.1f}'
            errs = ' & '.join('--' if c is None else f"${fz(100*(c/vQ-1))}$ (${fz((c-vQ)/seQ)}$)" for c in cols)
            rows.append(f"{lab if (H=='0.6' and t==2.0) else ''} & {H} & {t:g} & {vQ:.4f} & {100*seQ/vQ:.2f} & {errs}\\\\")
    rows.append('\\addlinespace')
w('tab_new', "\\begin{tabular}{lrrccccc}\\toprule\n & & & \\multicolumn{2}{c}{Monte Carlo} & \\multicolumn{3}{c}{solver: error of Var $Q$ in \\% ($z$-score)}\\\\\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\nsystem & $H$ & $t$ & Var $Q$ & s.e.\\ (\\%) & $p=1$, $J=65$ & $p=2$ & $p=3$, $J=17$\\\\\\midrule\n"
  + '\n'.join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\n")
print('tables written')
