"""Round 3 final figures (one visual language with the v5 figures of the manuscript).

Visual key used in every figure of the paper: colour = Hurst index (viridis, dark = persistent; make_figures_v5.HC);
CosChaos = solid line, filled circles; FWHE = dashed line, open squares; reference solution = dark grey line;
Monte Carlo = open red circles.  Fonts and sizes follow make_figures_v5.py (STIX serif, 9 pt at print size).
Reads evidence/round3_fwhe/workprecision.json; the dense memory reference curve of panel (a) is evaluated here
(Gauss-Jacobi quadrature on the 1601-point Talbot Green functions; identical inversions to examples_v6.memory_kernels).
Also reads evidence/round4/P4_cubic_event.json (Fig. 9(d)), evidence/round3_full/P1_pathwise.json, P2_wick_law.json, P3_cubic_law.json, evidence/round3_fwhe/benchmarks.json
and evidence/data/v12_examples.json (population).  Writes figures/{one,two}/v7_fig_*.{pdf,png} and
journal_files/graphical_abstract.{pdf,png} (= Fig. 1).
Run from working/:  python scripts/round3b_figures.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, 'evidence', 'round3_fwhe'); sys.path.insert(0, os.path.join(EV, 'solver_v1.1.3'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solver'))   # release layout: solver/ next to scripts/
W = json.load(open(os.path.join(EV, 'workprecision.json'))); BM = json.load(open(os.path.join(EV, 'benchmarks.json')))
FD = os.path.join(ROOT, 'evidence', 'round3_full')
P1 = json.load(open(os.path.join(FD, 'P1_pathwise.json'))); P2 = json.load(open(os.path.join(FD, 'P2_wick_law.json')))
P3 = json.load(open(os.path.join(FD, 'P3_cubic_law.json'))) if os.path.exists(os.path.join(FD, 'P3_cubic_law.json')) else None
P4 = json.load(open(os.path.join(ROOT, 'evidence', 'round4', 'P4_cubic_event.json')))   # Round 4: event probability (round4_probability.py)
EV_LO, EV_HI, EV_LEVEL = 0.4, 0.9, 0.7   # Round 5: level axis extended to 0.9
EXd = json.load(open(os.path.join(ROOT, 'evidence', 'data', 'v12_examples.json'), encoding='utf-8'))
MC_COL = '#c0392b'
MCS = dict(ls='none', marker='o', ms=3.4, mfc='none', mec=MC_COL, mew=0.8)

BASE, SMALL, TICK = 9.0, 8.0, 8.6
mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': BASE, 'axes.labelsize': BASE, 'axes.titlesize': BASE, 'legend.fontsize': SMALL,
    'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'lines.linewidth': 1.3, 'lines.markersize': 3.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'legend.frameon': False, 'legend.handlelength': 1.9, 'legend.borderaxespad': 0.2, 'legend.labelspacing': 0.25,
    'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True})
_cm = mpl.colormaps['viridis']
def HC(H):
    H = float(H)
    return '0.15' if abs(H - 0.5) < 1e-9 else _cm(0.08 + 0.62 * (0.95 - H) / 0.45)
REF_COL = '0.35'
OURS = dict(ls='-', marker='o', ms=3.6, lw=1.3)
FWHE = dict(ls=(0, (4, 1.8)), marker='s', ms=3.6, lw=1.1, mfc='white', mew=0.9)
W_ONE, W_TWO = 468 / 72.27, 522 / 72.27
HS = ['0.55', '0.75', '0.95']


def light(c, f):
    c = np.array(mpl.colors.to_rgb(c)); return tuple(1 - f * (1 - c))


def pareto(pts):
    pts = sorted((s, e) for s, e in pts if e > 0); out, best = [], np.inf
    for s, e in pts:
        if e < best: out.append((s, e)); best = e
    return np.array(out)


def letter(ax, s):
    ax.text(-0.22, 1.03, s, transform=ax.transAxes, fontsize=BASE + 1, fontweight='bold', va='bottom', ha='left')


def frame(lay, h=2.35):
    fig, ax = plt.subplots(1, 3, figsize=(W_ONE if lay == 'one' else W_TWO, h), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.05, wspace=0.07)
    return fig, ax


def save(fig, name, lay):
    d = os.path.join(ROOT, 'figures', lay); os.makedirs(d, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(d, f'{name}.{ext}'), dpi=600 if ext == 'png' else None)
    plt.close(fig)


def method_legend(ax, loc, extra=(), **kw):
    h = [Line2D([], [], color='0.25', **OURS), Line2D([], [], color='0.25', **FWHE)] + list(extra)
    lab = ['CosChaos', 'FWHE'] + [e.get_label() for e in extra]
    ax.legend(h, lab, loc=loc, **kw)


def H_legend(ax, loc, **kw):
    ax.legend([Line2D([], [], color=HC(H), lw=2.2) for H in HS], [f'$H={H}$' for H in HS], loc=loc, **kw)


def curve(rec, key):
    """all runs of one method in refinement order (full curve)."""
    ks = sorted(rec[key], key=lambda k: int(k.split(',')[0]))
    return np.array([(rec[key][k]['seconds'], rec[key][k]['max_rel']) for k in ks])


def wp_panel(ax, rec, ours_key, fw_key, ylabel=True, ours_pareto=False, extra=None):
    for H in HS:
        r = rec[H]
        po = pareto([(d['seconds'], d['max_rel']) for d in r[ours_key].values()]) if ours_pareto else curve(r, ours_key)
        pf = curve(r, fw_key)
        ax.loglog(po[:, 0], po[:, 1], color=HC(H), **OURS)
        ax.loglog(pf[:, 0], pf[:, 1], color=HC(H), **FWHE)
        if extra:
            pe = curve(r, extra); ax.loglog(pe[:, 0], pe[:, 1], color=light(HC(H), 0.45), lw=0.8, marker='o', ms=2.4)
    ax.set_xlabel('wall-clock time (s)')
    if ylabel: ax.set_ylabel('max. relative variance error')


def fig_ou(lay):
    R = W['OU']; t = np.array(R['t']); fig, ax = frame(lay)
    tt = np.linspace(0, 2, 201)
    import examples_v6 as E
    for H in HS:
        r = R[H]
        ax[0].plot(tt, [E.ou_exact_var(float(H), 1.0, 1.0, float(s)) for s in tt], color=REF_COL, lw=0.9)
        ax[0].plot(t, r['ours']['17']['var'], color=HC(H), ls='none', marker='o', ms=3.6)
        ax[0].plot(t, r['fwhe_gl']['400']['var'], color=HC(H), ls='none', marker='s', ms=4.8, mfc='none', mew=0.9)
        ax[0].text(2.04, r['reference'][-1], f'{H}', color=HC(H), fontsize=SMALL, va='center')
    ax[0].set_xlim(0, 2.25); ax[0].set_ylim(0, None); ax[0].set_xlabel('$t$'); ax[0].set_ylabel(r'Var $X(t)$')
    ax[0].legend([Line2D([], [], color=REF_COL, lw=0.9), Line2D([], [], color='0.25', ls='none', marker='o', ms=3.6),
                  Line2D([], [], color='0.25', ls='none', marker='s', ms=4.8, mfc='none', mew=0.9)],
                 ['reference', 'CosChaos, $J=17$', 'FWHE, $N=400$'], loc='upper left')
    H = '0.75'; r = R[H]; ex = np.array(r['reference'])
    for J, f in [('17', 0.45), ('65', 1.0)]:
        ax[1].plot(t, 100 * (np.array(r['ours'][J]['var']) / ex - 1), color=light(HC(H), f), **OURS, label=f'CosChaos, $J={J}$')
    for N, f in [('400', 0.45), ('1600', 1.0)]:
        ax[1].plot(t, 100 * (np.array(r['fwhe_gl'][N]['var']) / ex - 1), color=light(HC(H), f), **FWHE, label=f'FWHE, $N={N}$')
    ax[1].axhline(0, color='0.6', lw=0.6); ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'relative error (%), $H=0.75$')
    ax[1].legend(loc='upper right', ncol=1)
    wp_panel(ax[2], R, 'ours', 'fwhe_gl'); method_legend(ax[2], 'lower left')
    ax2 = ax[2].twinx(); ax2.set_axis_off(); H_legend(ax2, 'upper right')
    for a, s in zip(ax, 'abc'): letter(a, s)
    save(fig, 'v7_fig_bench_ou', lay)


def fig_wick(lay):
    R = W['Wick']; fig, ax = frame(lay)
    t = np.array(R['t']); H = '0.75'; r = R[H]; mu, sg = R['mu'], R['sig']; tt = np.linspace(0, 2, 201)
    ax[0].plot(tt, np.exp(2 * mu * tt) * np.expm1(sg ** 2 * tt ** (2 * float(H))), color='0.6', lw=3.0, zorder=0, label='reference')
    ax[0].plot(t, r['fwhe_order']['1'], color=HC(H), **FWHE, label='FWHE, order 1')
    ax[0].plot(t, r['ours']['65,1']['var'], color=light(HC(H), 0.45), **OURS, label='CosChaos, $p=1$')
    ax[0].plot(t, r['ours']['65,3']['var'], color=HC(H), **OURS, label='CosChaos, $p=3$')
    ax[0].set_xlabel('$t$'); ax[0].set_ylabel(r'Var $X(t)$, $H=0.75$'); ax[0].legend(loc='upper left'); ax[0].set_xlim(0, 2.05)
    wp_panel(ax[1], R, 'ours', 'fwhe_gl_k1', ours_pareto=True); H_legend(ax[1], 'lower left')
    H = '0.75'; pw = P1[H]['wick']; Js = [5, 9, 17, 33, 65, 129]
    for p, f in zip(range(1, 6), [0.3, 0.45, 0.6, 0.8, 1.0]):
        xs = [J for J in Js if f'{J},{p}' in pw]; ys = [pw[f'{J},{p}']['t1'] for J in xs]
        ax[2].loglog(xs, ys, color=light(HC(H), f), label=f'$p={p}$', **OURS)
    ax[2].legend(loc='center right', ncol=2, fontsize=7.4, bbox_to_anchor=(1.0, 0.62), title='CosChaos', title_fontsize=7.4)
    ax[2].set_xlabel('modes $J$'); ax[2].set_ylabel('pathwise relative RMS error, $t=1$'); ax[2].set_xlim(4, 180)
    ax[2].set_xticks(Js); ax[2].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter()); ax[2].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    for a, s in zip(ax, 'abc'): letter(a, s)
    save(fig, 'v7_fig_bench_wick', lay)


def fig_fullsolution(lay, target=None, size=None):
    H = '0.75'; Pp = P1[H]['paths']; td = np.array(Pp['t']); fig, ax = (plt.subplots(1, 3, figsize=size, constrained_layout=True) if size else frame(lay))
    if size: fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.05)
    ex = np.array(Pp['exact'])
    for k in range(3):
        ax[0].plot(np.r_[0, td], np.r_[1, ex[k]], color=REF_COL, lw=1.6)
        ax[0].plot(np.r_[0, td], np.r_[1, np.array(Pp['p1'])[k]], color=light(HC(H), 0.45), lw=1.0, ls='-')
        ax[0].plot(np.r_[0, td], np.r_[1, np.array(Pp['p3'])[k]], color=HC(H), lw=1.1, ls='-')
    ax[0].legend([Line2D([], [], color=REF_COL, lw=1.6), Line2D([], [], color=light(HC(H), 0.45), lw=1.0), Line2D([], [], color=HC(H), lw=1.1)],
                 ['exact', 'CosChaos, $p=1$', 'CosChaos, $p=3$'], loc='upper left')
    ax[0].set_xlabel('$t$'); ax[0].set_ylabel('$X(t)$ on three fBm paths')
    d = P2[H]['2.0']; xs = np.array(d['density_x']); xc = 0.5 * (xs[1:] + xs[:-1])
    ax[1].plot(xs, d['density_exact'], color=REF_COL, lw=1.8, label='exact')
    import importlib.util
    sp = importlib.util.spec_from_file_location('r3c', os.path.join(HERE, 'round3c_fullsolution.py')); r3c = importlib.util.module_from_spec(sp); sp.loader.exec_module(r3c)
    VJ = d['VJ_65']; e = np.linspace(-8, 8, 400001)
    for p, f in [(1, 0.35), (3, 0.65), (5, 1.0)]:
        _, g = r3c.surrogate_law(2.0, float(H), VJ, p); y = g(e)
        hist, edges = np.histogram(y, bins=xs, weights=r3c.stats.norm.pdf(e) * (e[1] - e[0]))
        ax[1].plot(xc, hist / np.diff(edges), color=light(HC(H), f), lw=1.1, label=f'CosChaos, $p={p}$')
    ex = d['exact']; xg = np.linspace(-3.0, 6.5, 400)
    ax[1].plot(xg, r3c.stats.norm.pdf(xg, ex['mean'], np.sqrt(ex['var'])), color='0.1', ls=':', lw=1.3, label='Gaussian (same\nmean, variance)')
    ax[1].axvline(0, color='0.75', lw=0.5, zorder=0)
    ax[1].set_xlim(-2.6, 6.5); ax[1].set_xlabel('$x$'); ax[1].set_ylabel('density of $X(2)$')
    def legend_b(fs, **kw):
        hh, ll = ax[1].get_legend_handles_labels()
        lg1 = ax[1].legend(hh[:4], ll[:4], loc='upper right', fontsize=fs, handlelength=1.5, **kw); ax[1].add_artist(lg1)
        ax[1].legend(hh[4:], ll[4:], loc='center right', bbox_to_anchor=(1.02, 0.56), fontsize=fs, handlelength=1.2, **kw)
    legend_b(7.0)
    st = [('var', 'variance', 'o', '-'), ('skew', 'skewness', 's', '--'), ('exkurt', 'excess kurtosis', '^', '-.'), ('q01', '1% quantile', 'v', ':'), ('q99', '99% quantile', 'D', (0, (5, 1, 1, 1)))]
    for k, lab, mk, ls in st:
        e = [abs(d[f'65,{p}'][k]) * 100 for p in range(1, 9)]
        pp = [p for p, v in zip(range(1, 9), e) if v > 0]; ee = [v for v in e if v > 0]
        ax[2].semilogy(pp, ee, color=HC(H), marker=mk, ls=ls, ms=3.4, lw=1.0, mfc='white' if mk != 'o' else HC(H), label=lab)
    ax[2].set_xlabel('chaos degree $p$'); ax[2].set_ylabel('|relative error| (%), $t=2$'); ax[2].set_xticks(range(1, 9)); ax[2].legend(loc='lower left', fontsize=7.2, title='CosChaos', title_fontsize=7.2)
    if size:
        for a in ax:
            if a is ax[1]:
                for art in list(a.artists): art.remove()
                legend_b(5.6, frameon=True, framealpha=0.9, edgecolor='none', labelspacing=0.15, borderpad=0.25); continue
            lg = a.get_legend()
            if lg is not None:
                hh, ll = a.get_legend_handles_labels() if a is not ax[0] else (lg.legend_handles, [t.get_text() for t in lg.get_texts()])
                a.legend(hh, ll, fontsize=5.6, frameon=True, framealpha=0.9, edgecolor='none', loc=lg._loc, handlelength=1.6, labelspacing=0.15, borderpad=0.25)
    for a, s in zip(ax, 'abc'): letter(a, s)
    if target: fig.savefig(target + '.pdf'); fig.savefig(target + '.png', dpi=300); plt.close(fig)
    else: save(fig, 'v7_fig_fullsolution', lay)


def fig_population(lay):
    W_ = 252 / 72.27 if lay == 'two' else W_ONE * 0.62
    fig, a = plt.subplots(1, 1, figsize=(W_, 2.3), constrained_layout=True)
    po = EXd['pop']; tp = np.array(po['0.55']['p1']['t'])
    for H in ['0.55', '0.75', '0.95']:
        a.plot(tp, po[H]['p1']['var'], color=HC(H), lw=1.25, label=f'CosChaos, $H={H}$')
        t2 = np.array(po[H]['p2']['t']); v2 = np.array(po[H]['p2']['var']); s2 = slice(0, None, max(1, len(t2) // 12))
        a.plot(t2[s2], v2[s2], color=HC(H), ls='none', marker='o', ms=2.6)
    pc = EXd['pop_lim_curves']; a.plot(pc['t'], pc['0.5'], color=HC(0.5), lw=1.1, label=r'CosChaos, $H=\frac{1}{2}$')
    mc = EXd['pop_mc_H05']; sel = slice(0, None, 8)
    a.plot(np.array(mc['t'])[sel], np.array(mc['var'])[sel], label=r'Monte Carlo (Itô), $H=\frac{1}{2}$', ms=2.6, **{k: v for k, v in MCS.items() if k != 'ms'})
    a.plot([], [], color='0.3', ls='none', marker='o', ms=2.6, label='CosChaos, $p=2$'); a.set_xlabel('$t$'); a.set_ylabel(r'Var $X(t)$')
    a.set_ylim(-0.5, 12); a.legend(loc='upper right', ncol=1, fontsize=7.0)
    save(fig, 'v7_fig_population', lay)


def fig_cubiclaw(lay):
    fig, axs = plt.subplots(2, 2, figsize=(W_ONE if lay == 'one' else W_TWO, 4.45), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.07, hspace=0.06); ax = axs.ravel(); H = '0.75'; R = P3[H]
    edges = np.linspace(-1.5, 2.5, 161); xc = 0.5 * (edges[1:] + edges[:-1])
    hm = np.array(R['mc_hist']['4.0']); sel = slice(0, None, 3)
    ax[0].plot(xc[sel], hm[sel], label='Monte Carlo ($10^6$ paths)', **MCS)
    for cf, f, lab in [('hist_9,1', 0.4, 'CosChaos, $p=1$'), ('hist_9,4', 1.0, 'CosChaos, $p=4$')]:
        ax[0].plot(xc, R[cf]['4.0'], color=light(HC(H), f), lw=1.2, label=lab)
    ax[0].set_xlim(-1.3, 1.3); ax[0].set_ylim(0, 1.75); ax[0].set_xlabel('$x$'); ax[0].set_ylabel('density of $X(4)$, $H=0.75$'); ax[0].legend(loc='upper left', fontsize=7.2, ncol=1)
    hm1 = np.array(R['mc_hist']['1.0'])
    ax[1].plot(xc[sel], hm1[sel], **MCS)
    for cf, f in [('hist_9,1', 0.4), ('hist_9,4', 1.0)]:
        ax[1].plot(xc, R[cf]['1.0'], color=light(HC(H), f), lw=1.2)
    ax[1].set_xlim(-1.0, 1.4); ax[1].set_xlabel('$x$'); ax[1].set_ylabel('density of $X(1)$, $H=0.75$')
    Js = [5, 9, 17, 33, 65]
    for Hh in HS:
        c = P1[Hh]['cubic']
        ax[2].loglog(Js, [c[f'{J},2']['t1'] for J in Js], color=HC(Hh), **OURS, label=f'$H={Hh}$')
        ax[2].loglog(Js, [0.9 * c['5,2']['t1'] * (J / 5) ** (-float(Hh)) for J in Js], color='0.55', ls=':', lw=0.9)
    ax[2].set_xlabel('modes $J$'); ax[2].set_ylabel('pathwise relative RMS error, $t=1$'); ax[2].legend(loc='lower left', fontsize=7.2, title='CosChaos, $p=2$', title_fontsize=7.2); ax[2].set_xlim(4, 90)
    ax[2].set_xticks(Js); ax[2].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter()); ax[2].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # (d) tail probability P(X(4) > c): CosChaos (J=9, p=5) as baseline, refined CosChaos index sets (Round 5, stars),
    #     Monte Carlo (95% intervals), Gaussian with the Monte Carlo mean and variance
    lev = np.array(P4['meta']['levels']); it = P4['meta']['times'].index(4.0); m = (lev >= EV_LO - 1e-9) & (lev <= EV_HI + 1e-9)
    RC = json.load(open(os.path.join(ROOT, 'evidence', 'round5', 'P5_refined_curve.json')))
    lr = np.array(RC['levels']); star_c = [0.7, 0.75, 0.8, 0.85, 0.9]; ks = [int(np.argmin(abs(lr - c))) for c in star_c]
    from scipy.stats import norm
    for Hh in HS:
        r = P4[Hh]; pc = np.array(r['chaos_9,5']['p'])[it]; pm = np.array(r['mc_512']['p'])[it]; ci = np.array(r['mc_512']['ci'])[it]
        mu, vv = r['mc_512']['mean'][it], r['mc_512']['var'][it]
        ax[3].semilogy(lev[m], norm.sf((lev[m] - mu) / np.sqrt(vv)), color=HC(Hh), ls=':', lw=1.2)
        ax[3].semilogy(lev[m], pc[m], color=HC(Hh), **OURS)
        ax[3].errorbar(lev[m], pm[m], yerr=ci[m], fmt='none', ecolor=MC_COL, elinewidth=0.7, capsize=0)
        ax[3].plot(lev[m], pm[m], **{**MCS, 'ms': 4.6})
        ax[3].plot(star_c, [RC[Hh]['p'][k] for k in ks], ls='none', marker='*', ms=7.2, color=HC(Hh), mec='white', mew=0.4, zorder=5)
    ax[3].axvline(EV_LEVEL, color='0.7', lw=0.6, zorder=0)
    ax[3].set_xlabel('level $c$'); ax[3].set_ylabel(r'$P(X(4)>c)$'); ax[3].set_xlim(EV_LO - 0.02, EV_HI + 0.02)
    ax[3].legend([Line2D([], [], color='0.25', **OURS), Line2D([], [], ls='none', marker='*', ms=7.2, color='0.25', mec='white', mew=0.4),
                  Line2D([], [], **{**MCS, 'ms': 4.6}), Line2D([], [], color='0.25', ls=':', lw=1.2)],
                 ['CosChaos, $J=9$, $p=5$', 'CosChaos, refined', 'Monte Carlo ($10^6$ paths)', 'Gaussian (same mean, variance)'], loc='lower left', fontsize=7.0)
    ax3 = ax[3].twinx(); ax3.set_axis_off(); H_legend(ax3, 'upper right', fontsize=7.0)
    for a, s in zip(ax, 'abcd'): letter(a, s)
    save(fig, 'v7_fig_cubiclaw', lay)


def memory_dense():
    import hwc, examples_v6 as E
    R = W['memory']; b = R['mu'] + R['sig'] ** 2 / 2
    tm = np.linspace(0, R['T'], 1601); g, u, _ = hwc._talbot_green(R['a'], R['alpha'], b, tm, 30, need_w=False)
    tt = np.linspace(0.5, R['T'], 40)
    return tt, {H: np.array([E.fwhe57(float(H), float(s), tm, g, u) for s in tt]) for H in HS}


def fig_memory(lay, dense):
    R = W['memory']; t = np.array(R['t']); fig, ax = frame(lay); tt, dv = dense
    for H in HS:
        r = R[H]
        ax[0].semilogy(tt, dv[H], color=REF_COL, lw=0.9)
        ax[0].plot(t, r['ours']['33']['var'], color=HC(H), ls='none', marker='o', ms=3.6)
        ax[0].plot(t, r['fwhe_gl']['400']['var'], color=HC(H), ls='none', marker='s', ms=4.8, mfc='none', mew=0.9)
        ax[0].text(20.4, dv[H][-1], f'{H}', color=HC(H), fontsize=SMALL, va='center')
    ax[0].set_xlim(0, 22.5); ax[0].set_xlabel('$t$'); ax[0].set_ylabel('first-chaos variance')
    ax[0].legend([Line2D([], [], color=REF_COL, lw=0.9), Line2D([], [], color='0.25', ls='none', marker='o', ms=3.6),
                  Line2D([], [], color='0.25', ls='none', marker='s', ms=4.8, mfc='none', mew=0.9)],
                 ['reference', 'CosChaos, $J=33$', 'FWHE, $N=400$'], loc='lower right')
    H = '0.95'; r = R[H]; ex = np.array(r['reference'])
    for J, f in [('33', 0.45), ('129', 1.0)]:
        ax[1].plot(t, 100 * (np.array(r['ours'][J]['var']) / ex - 1), color=light(HC(H), f), **OURS, label=f'CosChaos, $J={J}$')
    for N, f in [('400', 0.45), ('1600', 1.0)]:
        ax[1].plot(t, 100 * (np.array(r['fwhe_gl'][N]['var']) / ex - 1), color=light(HC(H), f), **FWHE, label=f'FWHE, $N={N}$')
    ax[1].axhline(0, color='0.6', lw=0.6); ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'relative error (%), $H=0.95$')
    ax[1].legend(loc='upper right')
    wp_panel(ax[2], R, 'ours_cached', 'fwhe_gl', extra='ours'); H_legend(ax[2], 'lower left')
    for a, s in zip(ax, 'abc'): letter(a, s)
    save(fig, 'v7_fig_bench_memory', lay)


def graphical_abstract():
    """Graphical abstract = Fig. 1 at 13.3 cm x 5.3 cm (>= 1328 x 531 px at 300 dpi)."""
    fig_fullsolution('one', target=os.path.join(ROOT, 'journal_files', 'graphical_abstract'), size=(13.3 / 2.54, 5.3 / 2.54))


if __name__ == '__main__':
    dense = memory_dense()
    for lay in ('one', 'two'):
        fig_ou(lay); fig_wick(lay); fig_memory(lay, dense); fig_fullsolution(lay); fig_population(lay)
        if P3 is not None: fig_cubiclaw(lay)
    graphical_abstract()
    print('done')
