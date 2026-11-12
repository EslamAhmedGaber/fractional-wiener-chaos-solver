"""supplementary figure: refinement of P(X(4) > c) for the cubic single well against the 2x10^7-path reference.
Run after round5_tables.py:  python scripts/round5_figure.py  ->  figures/one/v8_fig_tailrefine.{pdf,png}
Visual key as in the paper: colour = H (viridis), CosChaos solid/filled; the grey band is the Monte Carlo 95% interval.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matplotlib; matplotlib.use('Agg')
import round3_final_figures as F
import matplotlib.pyplot as plt
W = os.path.dirname(HERE); E5 = os.path.join(W, 'evidence', 'round5')
S = json.load(open(os.path.join(E5, 'P5_summary.json')))['summary']
MCR = json.load(open(os.path.join(E5, 'P5_mc_reference.json'))); LEV = np.round(np.arange(0, 1.2001, 0.01), 2); ix = lambda c: int(np.argmin(abs(LEV - c)))
rel = lambda H, c: 100 * MCR[H]['mc_512']['ci4'][ix(c)] / MCR[H]['mc_512']['p4'][ix(c)]
fig, ax = F.frame('one', h=2.3)
# (a) degree in xi_0 at H = 0.95 (J = 13)
H = '0.95'; deg = [('AN13_10', 10), ('AN13_12', 12), ('AN13_16b', 16), ('AN13_20b', 20), ('AN13_24c', 24), ('AN13_28c', 28)]
x = [p for _, p in deg]; y = [S[n]['H'][H]['dev']['0.8']['dev'] for n, _ in deg]; e = [S[n]['H'][H]['dev']['0.8']['ci'] for n, _ in deg]
a = ax[0]; a.axhspan(-rel(H, 0.8), rel(H, 0.8), color='0.88', lw=0); a.axhline(0, color='0.5', lw=0.6)
a.errorbar(x, y, yerr=e, color=F.HC(H), **F.OURS, capsize=1.5, elinewidth=0.7)
d5 = S['TD9_5']['H'][H]['dev']['0.8']
a.errorbar([5], [d5['dev']], yerr=[d5['ci']], color=F.HC(H), ls='none', marker='o', ms=3.6, mfc='white', capsize=1.5, elinewidth=0.7)
a.annotate('$J=9$, $p=5$', (5, d5['dev']), xytext=(7.5, d5['dev'] + 1.0), fontsize=F.SMALL, va='center')
a.set_xlabel('degree $p_0$ in $\\xi_0$ ($J=13$)'); a.set_ylabel('deviation of $P(X(4)>0.8)$ (%)'); a.set_xlim(3, 30); a.set_ylim(-20, 4)
a.set_title('$H=0.95$', fontsize=F.SMALL); F.letter(a, '(a)')
# (b) number of modes at H = 0.55 and 0.75 (p_0 = 10)
a = ax[1]; jj = [('AN13_10', 13), ('AN17_10', 17), ('AN33_10h', 33), ('AN65_10h', 65)]
a.axhspan(-rel('0.55', 0.8), rel('0.55', 0.8), color='0.88', lw=0); a.axhline(0, color='0.5', lw=0.6)
for H in ('0.55', '0.75'):
    y = [S[n]['H'][H]['dev']['0.8']['dev'] for n, _ in jj]; e = [S[n]['H'][H]['dev']['0.8']['ci'] for n, _ in jj]
    a.errorbar([J for _, J in jj], y, yerr=e, color=F.HC(H), **F.OURS, capsize=1.5, elinewidth=0.7, label=f'$H={H}$')
a.set_xscale('log', base=2); a.set_xticks([13, 17, 33, 65]); a.set_xticklabels(['13', '17', '33', '65']); a.minorticks_off()
a.set_xlabel('number of modes $J$ ($p_0=10$)'); a.set_ylim(-3, 11); a.legend(loc='upper right'); F.letter(a, '(b)')
# (c) refined sets over the level c
a = ax[2]; BEST = {'0.55': 'AN65_10h', '0.75': 'AN65_10h', '0.95': 'AN13_28c'}; cs = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
a.axhline(0, color='0.5', lw=0.6)
for H in ('0.55', '0.75', '0.95'):
    n = BEST[H]; y = [S[n]['H'][H]['dev'][str(c)]['dev'] for c in cs]; e = [S[n]['H'][H]['dev'][str(c)]['ci'] for c in cs]
    a.fill_between(cs, [-rel(H, c) for c in cs], [rel(H, c) for c in cs], color=F.light(F.HC(H), 0.18), lw=0)
    off = {'0.55': -0.006, '0.75': 0.0, '0.95': 0.006}[H]
    a.errorbar([c + off for c in cs], y, yerr=e, color=F.HC(H), **F.OURS, capsize=1.5, elinewidth=0.7, label=f'$H={H}$')
a.set_xlabel('level $c$'); a.set_ylim(-4, 4); a.legend(loc='lower left', ncol=1); F.letter(a, '(c)')
F.save(fig, 'v8_fig_tailrefine', 'one')
print('ok')
