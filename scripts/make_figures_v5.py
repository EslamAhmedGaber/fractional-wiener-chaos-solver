"""V5 figures for the CSF manuscript (27 Oct 2026).

Reads only stored evidence (evidence/data/*.json, evidence/oscillator_time/osc_time.json,
evidence/cubic_scaling/cubic_scaling_analysis.json, evidence/cost_check.json), the V4 plotted-data
records (figure_design/legacy_extracts/*.csv) for panels whose raw arrays exist only there, and the closed-form
Gram matrix (evidence/verification_code/hfc.py) for the three Gram panels. Every data selection is checked
against the V4 plotted data or a number quoted in the manuscript (CHECKS, written to figure_design/V5_CHECKS.json).

Writes figures/one/<name>.{pdf,svg,png} (one-column layout, 468 pt text width) and figures/two/<name>.* (two-column:
'wide' figures at 522 pt, 'col' figures at 252 pt), figures/supplement/<name>.*, and figure_design/V5_PROVENANCE.json.
Run:  python scripts/make_figures_v5.py
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

V = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ED = os.path.join(V, 'evidence', 'data'); SD = os.path.join(V, 'figure_design', 'legacy_extracts')
sys.path.insert(0, os.path.join(V, 'evidence', 'verification_code'))
import hfc  # closed-form Gram matrix (Proposition 2)

J_ = lambda *p: json.load(open(os.path.join(*p), encoding='utf-8'))
TN, EX, OS, F8, CU, BE = (J_(ED, f) for f in ['theory_noise.json', 'v12_examples.json', 'v12_oscillator.json',
                                              'fig8_data.json', 'v12_cubic.json', 'v12_bench.json'])
C8=J_(ED,'cubic_v8.json'); R10=J_(ED,'v10_runs.json')
OT = J_(V, 'evidence', 'oscillator_time', 'osc_time.json')

csv = lambda name: pd.read_csv(os.path.join(SD, name + '.csv'), dtype={'series': str, 'axis': str})
CHECKS, PROV = [], {}


def check(label, ok, detail=''):
    CHECKS.append(dict(check=label, ok=bool(ok), detail=str(detail)[:200]))
    assert ok, (label, detail)


def series(df, axis, s):
    g = df[(df.axis == str(axis)) & (df.series == str(s))].sort_values('point', key=lambda c: c.astype(int))
    return g.x.to_numpy(float), g.y.to_numpy(float)

# ----------------------------------------------------------------- style
BASE, SMALL, TICK = 9.0, 8.6, 8.6
mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': BASE, 'axes.labelsize': BASE, 'axes.titlesize': BASE, 'legend.fontsize': SMALL,
    'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'lines.linewidth': 1.3, 'lines.markersize': 3.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'legend.frameon': False, 'legend.handlelength': 1.7, 'legend.borderaxespad': 0.2, 'legend.labelspacing': 0.25,
    'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': True})

# one colour per Hurst index, threaded through every figure (sequential, colour-blind safe, dark = persistent)
_cm = mpl.colormaps['viridis']
def HC(H):
    H = float(H)
    if abs(H - 0.5) < 1e-9: return '0.15'          # Brownian case: near-black
    return _cm(0.08 + 0.62 * (0.95 - H) / 0.45)    # Avoid pale yellow-green on white paper.
MC_COL = '#c0392b'                                  # Monte Carlo / reference marks (one alarm-free accent)
REF_COL = '0.35'
DEG_STY = {1: dict(ls=(0, (1.2, 1.4)), lw=1.3), 2: dict(ls='-', lw=1.4), 3: dict(ls=(0, (5, 1.6)), lw=1.3), 4: dict(ls='-.', lw=1.2)}

W_ONE, W_TWO, W_COL = 468 / 72.27, 522 / 72.27, 252 / 72.27
OUT = {lay: os.path.join(V, 'figures', lay) for lay in ['one', 'two', 'supplement']}


def letter(ax, s, dx=-0.20, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=BASE + 1, fontweight='bold', va='bottom', ha='left')


def layout(kind, lay, n, h_one=2.40, h_col=2.05):
    """kind 'wide': 1 x n in both layouts (full width).  kind 'col': 1 x n in one-column, n x 1 stacked in a two-column
    single column.  Returns fig, axes (flat)."""
    if kind == 'wide' or lay == 'one':
        W = W_ONE if lay == 'one' else W_TWO
        fig, axs = plt.subplots(1, n, figsize=(W, h_one), constrained_layout=True)
    else:
        fig, axs = plt.subplots(n, 1, figsize=(W_COL, h_col * n), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, hspace=0.06)
    return fig, np.atleast_1d(axs).ravel()


def save(fig, name, lay, prov):
    fig.get_layout_engine().set(w_pad=0.07,h_pad=0.09)
    os.makedirs(OUT[lay], exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(os.path.join(OUT[lay], f'{name}.{ext}'), dpi=600 if ext == 'png' else None)
    fig.canvas.draw()
    r=fig.canvas.get_renderer()
    # Candidate overlaps are advisory; final PDF review decides actual visible conflicts.
    texts=[(t,t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
           if t.get_visible() and t.get_text().strip()]
    fb=fig.bbox
    inside=[(t,bb) for t,bb in texts if fb.contains((bb.x0+bb.x1)/2,(bb.y0+bb.y1)/2)]
    outside=[t.get_text() for t,bb in inside if bb.x0<fb.x0-1 or bb.x1>fb.x1+1 or bb.y0<fb.y0-1 or bb.y1>fb.y1+1]
    ov=[(a.get_text(),b.get_text()) for i,(a,ba) in enumerate(inside) for b,bb in inside[i+1:]
        if ba.overlaps(bb)]
    PROV.setdefault(name,dict(prov,layouts={}))['layouts'][lay]=dict(
        size_in=[round(x,3) for x in fig.get_size_inches()],text_overlaps=ov,text_outside=outside)
    if lay!='two':
        rows=[]
        for ai,ax in enumerate(fig.axes):
            for li,line in enumerate(ax.lines):
                xy=np.asarray(line.get_xydata())
                rows.extend(dict(axis=ai,artist='line',series=li,point=k,x=float(x),y=float(y))
                            for k,(x,y) in enumerate(xy))
            for ci,coll in enumerate(ax.collections):
                for pi,path in enumerate(coll.get_paths()):
                    rows.extend(dict(axis=ai,artist='collection_path',series=f'{ci}:{pi}',point=k,x=float(x),y=float(y))
                                for k,(x,y) in enumerate(path.vertices))
        sd=os.path.join(V,'figure_design','source_data');os.makedirs(sd,exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(sd,name+'.csv'),index=False)
    plt.close(fig)


def logticks(ax, axis='y'):
    loc = mpl.ticker.LogLocator(base=10, numticks=12)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_locator(loc)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    (ax.yaxis if axis == 'y' else ax.xaxis).set_minor_formatter(mpl.ticker.NullFormatter())


def jticks(ax, Js):
    ax.set_xscale('log'); ax.set_xticks(Js); ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f'{v:g}'))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

# ======================================================================= Fig 1: Gram construction
def fig_gram(lay):
    fig, ax = layout('wide', lay, 3, h_one=2.75)
    k = np.arange(1, 513); df = csv('fig1_gram_structure')
    for i, H in enumerate(['0.55', '0.7', '0.8', '0.95']):
        d, a = np.array(TN['diag'][H]['d']), np.array(TN['diag'][H]['asym'])
        xs, ys = series(df, 1, 2 * i); check(f'F1a diag H={H} equals V4 plotted', np.allclose(ys, d, rtol=1e-12))
        ax[0].loglog(k, d, color=HC(H), lw=1.3, label=f'$H={H}$'); ax[0].loglog(k, a, color=HC(H), lw=0.9, ls='--')
        if H != '0.55': check(f'F1a ratio at k=512, H={H}', 0.9995 < d[-1] / a[-1] < 1.0001, d[-1] / a[-1])
    ax[0].set_xlabel('mode index $k$'); ax[0].set_ylabel(r'$T^{1-2H}G^H_{kk}$'); logticks(ax[0])
    ax[0].set_ylim(2e-4,2)
    # The caption defines dashed large-k asymptotes; keep H keys outside data.
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5,-0.28), ncol=2, columnspacing=0.7)
    G = hfc.gram(0.8, 1.0, 'cos', 10); C = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
    off = C - np.eye(10); vmax = 0.12
    check('F1b largest correlation 0.104 at (0,2)', abs(np.max(np.abs(off)) - 0.104) < 5e-4 and abs(off[0, 2]) == np.max(np.abs(off)), np.max(np.abs(off)))
    M = np.ma.masked_where(np.eye(10, dtype=bool), off)
    cmap = mpl.colormaps['RdBu_r'].copy(); cmap.set_bad('0.82')
    im = ax[1].imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, origin='upper')
    ax[1].set_xticks(range(0, 10, 3)); ax[1].set_yticks(range(0, 10, 3))
    ax[1].set_xlabel('mode index $l$'); ax[1].set_ylabel('mode index $k$')
    cb = fig.colorbar(im, ax=ax[1], shrink=0.85, pad=0.02, ticks=[-0.1, 0, 0.1]); cb.ax.tick_params(labelsize=TICK)
    cb.set_label(r'corr$(Z_k,Z_l)$, $k\neq l$', fontsize=SMALL)
    G17 = hfc.gram(0.8, 1.0, 'cos', 17); L = np.linalg.cholesky(G17); Li = np.linalg.inv(L)
    t = np.linspace(0, 1, 801); Hm = hfc.h('cos', 17, 1.0, t)         # (len t, J)
    dfo = csv('fig1b_orthonormal_limit')
    for i, kk in enumerate([2, 4, 8]):
        corr = L[kk, kk] * (Hm @ Li[kk]) - Hm[:, kk]
        xs, ys = series(dfo, 2, i); check(f'F1c correction k={kk} equals V4 plotted', np.allclose(ys, corr, atol=1e-10))
        ax[2].plot(t, corr, color=['#1b9e77', '#d95f02', '#7570b3'][i], lw=1.2, label=f'$k={kk}$')
    ax[2].axhline(0, color='0.6', lw=0.6); ax[2].set_xlabel('$t/T$'); ax[2].set_ylabel(r'$L_{kk}u_k-h_k$')
    ax[2].legend(loc='upper center', ncol=1)
    for a, s in zip(ax, 'abc'): letter(a, s)
    save(fig, 'v5_fig01_gram', lay, dict(panels={'a': 'theory_noise.json diag[H].d/asym (T=1)', 'b': 'hfc.gram(0.8,1,cos,10)', 'c': 'hfc.gram(0.8,1,cos,17), Cholesky'}, from_v4='1a,1b,4a'))

# ======================================================================= Fig 2: noise series
def fig_noise(lay):
    fig, ax = layout('wide', lay, 3, h_one=2.50)
    Js = np.array(TN['Js']); n8 = TN['noise']['0.8']; df = csv('fig2_noise_series')
    check('F2a cosine equals V4 plotted', np.allclose(series(df, 0, 0)[1], n8['cos']['classical']))
    wb = series(df, 0, 6)[1]
    sty = [('cosine', n8['cos']['classical'], dict(color='0.1', marker='o', lw=1.4, zorder=5)),
           ('optimal projection', n8['cos']['optimal'], dict(color='0.55', marker='.', lw=0.0, ms=5, zorder=6)),
           (r'sine $(k-\frac{1}{2})\pi$', n8['sin']['classical'], dict(color='#1b9e77', marker='^', lw=1.0)),
           ('discrete KL', n8['KL'], dict(color='#7570b3', marker='s', lw=1.0, ms=3)),
           ('RL--Fourier', n8['RLF'], dict(color='#d95f02', marker='D', lw=1.0, ms=3)),
           ('weighted bound', wb, dict(color='0.3', ls='--', lw=0.9)),
           ('Schur bound', n8['cos']['schur'], dict(color='0.3', ls=':', lw=1.0))]
    for lab, y, kw in sty: ax[0].loglog(Js, y, label=lab.replace('--', '–'), **kw)
    ax[0].set_ylim(3e-5, 0.4); ax[0].set_xlabel('modes $J$'); ax[0].set_ylabel('worst-case RMS error')
    jticks(ax[0], [5, 17, 65, 257]); logticks(ax[0]); ax[0].legend(loc='lower left', fontsize=SMALL, ncol=1, labelspacing=0.15)
    for H in ['0.5', '0.6', '0.7', '0.8', '0.9', '0.95']:
        ax[1].loglog(Js, TN['noise'][H]['cos']['classical'], color=HC(H), marker='o', ms=2.8, lw=1.2, label=f'$H={H}$')
    jticks(ax[1], [5, 17, 65, 257]); logticks(ax[1]); ax[1].set_xlabel('modes $J$'); ax[1].set_ylabel('worst-case RMS error')
    ax[1].set_ylim(4e-5,0.3); ax[1].legend(loc='lower left', ncol=2, columnspacing=0.5)
    t = np.linspace(0, 1, 1025)
    for H in ['0.5', '0.7', '0.95']:
        ax[2].plot(t, TN['noise'][H]['profile17'], color=HC(H), lw=1.1, label=f'$H={H}$')
    check('F2c profile endpoints vanish', abs(TN['noise']['0.7']['profile17'][0]) < 1e-12 and abs(TN['noise']['0.7']['profile17'][-1]) < 1e-6, TN['noise']['0.7']['profile17'][-1])
    ax[2].set_xlabel('$t/T$'); ax[2].set_ylabel(r'RMS error, $J=17$'); ax[2].set_ylim(0, 0.10); ax[2].legend(loc='center', ncol=1)
    for a, s in zip(ax, 'abc'): letter(a, s)
    save(fig, 'v5_fig02_noise', lay, dict(panels={'a': 'theory_noise.json noise[0.8]; weighted bound from V4 plotted data', 'b': 'theory_noise.json noise[H].cos.classical', 'c': 'theory_noise.json noise[H].profile17'}, from_v4='2a,2b,3a'))

# ======================================================================= Fig 3: worked example
def fig_worked(lay):
    fig, ax = layout('wide', lay, 2, h_one=2.40, h_col=1.95)
    Js = EX['ou_interior_J']; df = csv('fig3_linear_validation')
    for i, H in enumerate(['0.5', '0.55', '0.75', '0.95']):
        y = np.abs(EX['ou_interior'][H]); check(f'F3a OU H={H} equals V4 plotted', np.allclose(series(df, 0, i)[1], y))
        ax[0].loglog(Js, y, color=HC(H), marker='o', ms=2.8, label=f'$H={H}$')
    jticks(ax[0], Js); logticks(ax[0]); ax[0].set_xlabel('modes $J$'); ax[0].set_ylabel(r'rel. error of Var $X(1)$'); ax[0].legend(loc='lower left', ncol=2, columnspacing=0.8)
    for i, H in enumerate(['0.5', '0.75', '0.95']):
        w = EX['wick'][H]; it = w['t'].index(1.0); ex = w['exact'][it]
        e = [abs(w['33'][str(p)][it] / ex - 1) for p in range(1, 6)]
        check(f'F3b Wick H={H} equals V4 plotted', np.allclose(series(df, 1, i)[1], e))
        ax[1].semilogy(range(1, 6), e, color=HC(H), marker='o', ms=2.8, label=f'$H={H}$')
    ax[1].set_xticks(range(1, 6)); logticks(ax[1]); ax[1].set_xlabel('chaos degree $p$'); ax[1].set_ylabel(r'rel. error of Var $X(1)$'); ax[1].legend(loc='upper right')
    for a, s in zip(ax, 'ab'): letter(a, s)
    save(fig, 'v5_fig03_worked', lay, dict(panels={'a': 'v12_examples.json ou_interior (OU, t=1, T=2)', 'b': 'v12_examples.json wick[H][33] (t=1)'}, from_v4='6a,6b'))

# ======================================================================= Fig 4: Brownian limit
def fig_limit(lay):
    fig, ax = layout('wide', lay, 3, h_one=2.50)
    pc = EX['pop_lim_curves']; t = np.array(pc['t'])
    for H in ['0.5', '0.51', '0.55', '0.6']:
        ax[0].plot(t, pc[H], color={'0.5':'0.15','0.51':'#9ecae1','0.55':'#4292c6','0.6':'#084594'}[H], lw=1.2, label=f'$H={H}$')
    mc = EX['pop_mc_H05']; tm = np.array(mc['t']); sel = slice(0, None, 8)
    ax[0].plot(tm[sel], np.array(mc['var'])[sel], 'o', ms=2.4, mfc='none', mec=MC_COL, mew=0.8, label=r'Itô MC, $H=\frac{1}{2}$')
    ax[0].set_xlabel('$t$'); ax[0].set_ylabel(r'Var $X(t)$, $J=129$'); ax[0].legend(loc='lower right', ncol=1)
    L = EX['limit']; h = np.array(L['H']) - 0.5; m = h > 0
    ou = np.abs(np.array(L['ou_var_T']) - L['ou_var_T'][0]); po = np.abs(np.array(L['pop_maxvar_J33']) - L['pop_maxvar_J33'][0])
    df = csv('fig4_classical_limit'); xs, ys = series(df, 1, 0); check('F4b OU equals V4 plotted', np.allclose(np.sort(ys), np.sort(ou[m])), (ys[:3], ou[m][:3]))
    ax[1].loglog(h[m], ou[m], color='0.15', marker='o', ms=2.8, label=r'OU, Var $X(2)$')
    ax[1].loglog(h[m], po[m], color='#d95f02', marker='s', ms=2.8, label='population, peak Var')
    ax[1].loglog(h[m], h[m], color='0.55', ls=':', lw=0.9, label='slope 1')
    ax[1].set_xlabel(r'$H-\frac{1}{2}$'); ax[1].set_ylabel(r'$|m_H-m_{1/2}|$'); logticks(ax[1]); logticks(ax[1], 'x'); ax[1].legend(['OU variance', 'population peak', 'slope 1'], loc='upper left')
    for i, (Jv, c) in enumerate(zip([9, 33, 129], ['#9ecae1', '#4292c6', '#08306b'])):
        xs, ys = series(df, 2, i); ax[2].semilogx(xs, ys, color=c, marker='o', ms=2.6, label=f'$J={Jv}$')
        if i == 2: check('F4c J=129 approaches 11.2', abs(ys.max() - 11.2) < 0.06, ys.max())
    ax[2].set_xlabel(r'$H-\frac{1}{2}$'); ax[2].set_ylabel(r'$\|G^H-I\|_2/(H-1/2)$'); logticks(ax[2], 'x'); ax[2].set_ylim(0, 12.5); ax[2].legend(loc='lower left')
    for a, s in zip(ax, 'abc'): letter(a, s)
    save(fig, 'v5_fig04_limit', lay, dict(panels={'a': 'v12_examples.json pop_lim_curves, pop_mc_H05 (every 8th point)', 'b': 'v12_examples.json limit', 'c': 'V4 plotted data fig4_classical_limit axis 2'}, from_v4='5a,5b,5c'))

# ======================================================================= Fig 5: memory and population
def fig_mempop(lay):
    fig, ax = layout('wide', lay, 2, h_one=2.40, h_col=1.95)
    me = EX['memory']; t = np.array(me['t'])
    for H in ['0.5', '0.7', '0.95']:
        ax[0].semilogy(t[1:], np.array(me[H]['65']['var1'])[1:], color=HC(H), lw=1.2, label=f'$H={H}$')
        fw = me[H]['fwhe57']; tf = [float(k) for k in fw]; vf = [fw[k] for k in fw]
        ax[0].plot(tf, vf, 'o', mfc='white', mec=HC(H), ms=3.6, mew=0.9)
        i20 = int(np.argmin(abs(t - 20))); check(f'F5a memory H={H} deviation at t=20 < 2e-4', abs(me[H]['65']['var1'][i20] / fw['20.0'] - 1) < 2e-4, me[H]['65']['var1'][i20] / fw['20.0'] - 1)
    ax[0].plot([], [], 'o', mfc='white', mec='0.3', label='FWHE reference'); ax[0].set_xlabel('$t$'); ax[0].set_ylabel('first-chaos variance'); logticks(ax[0])
    ax[0].set_ylim(1e-5, 4e-2); ax[0].legend(loc='lower right', ncol=1)
    po = EX['pop']; tp = np.array(po['0.55']['p1']['t'])
    for H in ['0.55', '0.75', '0.95']:
        ax[1].plot(tp, po[H]['p1']['var'], color=HC(H), lw=1.25, label=f'$H={H}$')
        ax[1].plot(po[H]['p2']['t'], po[H]['p2']['var'], color=HC(H), lw=1.0, ls='--')
    pc = EX['pop_lim_curves']; ax[1].plot(pc['t'], pc['0.5'], color=HC(0.5), lw=1.1, label=r'$H=\frac{1}{2}$, $J=129$')
    mc = EX['pop_mc_H05']; sel = slice(0, None, 8); ax[1].plot(np.array(mc['t'])[sel], np.array(mc['var'])[sel], 'o', ms=2.2, mfc='none', mec=MC_COL, mew=0.7, label=r'Itô MC, $H=\frac{1}{2}$')
    ax[1].plot([], [], color='0.3', lw=1.0, ls='--', label='$p=2$, $J=33$'); ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'Var $X(t)$ ($p=1$, $J=65$)')
    ax[1].set_ylim(-0.5,12)
    ax[1].legend(loc='upper right', ncol=1, fontsize=SMALL)
    check('F5b peak H=0.95 near 8.5', abs(max(po['0.95']['p1']['var']) - 8.5) < 0.05, max(po['0.95']['p1']['var']))
    for a, s in zip(ax, 'ab'): letter(a, s)
    save(fig, 'v5_fig05_memory_population', lay, dict(panels={'a': 'v12_examples.json memory[H][65].var1 and fwhe57', 'b': 'v12_examples.json pop[H].p1/p2, pop_lim_curves[0.5], pop_mc_H05'}, from_v4='7a,8a'))

# ======================================================================= Fig 6: cubic drift
def fig_cubic(lay):
    W = W_ONE if lay == 'one' else W_TWO
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.65), constrained_layout=True); fig.get_layout_engine().set(w_pad=0.03, h_pad=0.04, wspace=0.05, hspace=0.07)
    ax = axs.ravel(); df = csv('fig8_single_well'); t = np.array(F8['t'])
    # (a) single-well variance, sigma=0.5, H=0.75
    xs, mcv = series(df, 0, 3)
    eb = [series(df, 0, f'0:{i}') for i in range(16)]; lo = np.array([e[1].min() for e in eb]); hi = np.array([e[1].max() for e in eb])
    check('F6a dense MC matches v10_runs, a separate stream from Table 8',np.allclose(mcv,R10['mono_mc']['var'],rtol=1e-13))
    check('F6a p1 and p2 match general solver, not legacy precursor',all(np.allclose(series(df,0,j)[1],CU['sw'][f'0.5_0.75_65_{j+1}']['var'],rtol=1e-12) for j in (0,1)))
    ax[0].plot(t, CU['sw']['0.5_0.75_65_1']['var'], color='0.45', **DEG_STY[1], label='$p=1$, $J=65$')
    ax[0].plot(t, CU['sw']['0.5_0.75_65_2']['var'], color='#0072B2', **DEG_STY[2], label='$p=2$, $J=65$')
    x4, y4 = series(df, 0, 2); ax[0].plot(x4, y4, 'D', color='#762a83', ms=3.4, label='$p=4$, $J=9$')
    ax[0].errorbar(xs, mcv, yerr=[mcv - lo, hi - mcv], fmt='o', ms=2.4, color=MC_COL, mfc='none', mew=0.8, elinewidth=0.8, capsize=0, label=r'MC $\pm2$ s.e.')
    ax[0].set_xlabel('$t$'); ax[0].set_ylabel(r'Var $X(t)$'); ax[0].legend(loc='lower right')
    # (b) third central moment
    xm, ym = series(df, 1, 3); x4m, y4m = series(df, 1, 1)
    ax[1].plot(t, 1e3 * np.array(CU['sw']['0.5_0.75_65_2']['m3']), color='#0072B2', **DEG_STY[2], label='$p=2$, $J=65$')
    ax[1].axhline(0, color='0.45', **DEG_STY[1], label='$p=1$ (zero)')
    ax[1].plot(x4m, y4m, 'D', color='#762a83', ms=3.4, label='$p=4$, $J=9$'); ax[1].plot(xm, ym, 'o', ms=2.4, color=MC_COL, mfc='none', mew=0.8, label='MC')
    ax[1].set_ylim(-10, 1.0); ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'$\mathrm{E}[(X-\mathrm{E}X)^3]\times10^{3}$'); ax[1].legend(loc='lower right', ncol=2, columnspacing=0.7)
    # (c) degree refinement at t = 4 (V4 data; the J=65, p=2 square of H=0.75 corrected, see FIGURE_AUDIT_REPORT 0.1)
    dd = csv('fig8_degree_doublewell')
    cu = CU['sw']['0.5_0.75_65_2']; tt = np.array(cu['t']); v4 = np.array(cu['var'])[np.argmin(abs(tt - 4.0))]
    ref75=C8['sigma0.5']['mc']['0.75_512']['var'][3]
    sq75 = 100 * (v4 / ref75 - 1); check('F6c corrected H=0.75 J=65 p=2 deviation = -4.6 % (Table 8)', abs(sq75 + 4.6) < 0.05, sq75)
    check('F6c V4 plotted value was the t=0.4 entry', abs(series(dd, 2, 3)[1][0] - 100 * (np.array(cu['var'])[3] / ref75 - 1)) < 1e-9)
    ax[2].axhspan(-0.8, 0.8, color='0.9', lw=0)
    for (s_line, s_sq, H) in [(0, 1, 0.6), (2, 3, 0.75), (4, 5, 0.9)]:
        xl, yl = series(dd, 2, s_line); ax[2].plot(xl, yl, color=HC(H), marker='o', ms=3, lw=1.1, label=f'$H={H}$, $J=9$')
        ysq = sq75 if s_sq == 3 else series(dd, 2, s_sq)[1][0]
        ax[2].plot([2], [ysq], 's', mfc='white', mec=HC(H), mew=1.0, ms=4.2)
    ax[2].plot([], [], 's', mfc='white', mec='0.3', label='$J=65$, $p=2$'); ax[2].axhline(0, color='0.6', lw=0.6)
    ax[2].set_xticks([1, 2, 3, 4]); ax[2].set_xlabel('chaos degree $p$'); ax[2].set_ylabel(r'Var deviation at $t=4$ (%)'); ax[2].set_ylim(-8.5, 3); ax[2].legend(loc='lower right', fontsize=SMALL)
    # (d) double well
    xa, ya = series(dd, 3, 0); xp, yp = series(dd, 3, 1)
    ax[3].plot(xa, ya, 'o', ms=2.6, color=MC_COL, label='MC, all paths'); ax[3].plot(xp, yp, 's', ms=2.6, mfc='none', mec=MC_COL, mew=0.8, label=r'MC, $X(t)>0$')
    for s, p in [(2, 2), (3, 3), (4, 4)]:
        x, y = series(dd, 3, s); ax[3].plot(x, y, color=['#0072B2', '#2166ac', '#762a83'][p - 2], **DEG_STY[p], label=f'$p={p}$, $J=9$')
    check('F6d MC all paths at t=4 = 0.0263', abs(ya[np.argmin(abs(xa - 4))] - 0.0263) < 1e-4, ya[np.argmin(abs(xa - 4))])
    ax[3].set_xlabel('$t$'); ax[3].set_ylabel(r'Var $X(t)$, double well'); ax[3].set_ylim(0, 0.03); ax[3].legend(loc='lower right', ncol=2, columnspacing=0.7, fontsize=SMALL)
    for a, ss in zip(ax, 'abcd'): letter(a, ss)
    save(fig,'v5_fig06_cubic',lay,dict(panels={'a,b':'v12_cubic.json general solver and v10_runs.json dense MC (legacy extraction verified)',
      'c':'legacy fig8_degree_doublewell, corrected H=.75 p=2 J=65 point from v12_cubic at t=4',
      'd':'legacy fig8_degree_doublewell axis 3'},from_v4='9a,9b,10a,10b'))

def fig_barrier():
    lay='supplement'
    fig,ax=plt.subplots(1,2,figsize=(W_ONE,2.45),constrained_layout=True)
    bb = csv('fig8_barrier')
    for k, (axi, H) in enumerate([(0, 0.65), (1, 0.8)]):
        a = ax[axi]; a.axhspan(-2, 2, color='0.9', lw=0); a.axhline(0, color='0.6', lw=0.6)
        for s, (lab, mk, c) in enumerate([('$J=9$, $p=2$', 'v', '#0072B2'), ('$J=17$, $p=3$', 's', '#2166ac'), ('$J=9$, $p=4$', 'D', '#762a83')]):
            x, y = series(bb, 4 + k, s); a.plot(x, y, marker=mk, color=c, ms=3.2, lw=1.0, label=lab)
        a.set_xlabel('$t$'); a.set_ylabel('$z$-score of Var'); a.set_ylim(-10, 3); a.set_xticks([0.25, 0.5, 0.75, 1.0])
        a.text(0.97, 0.95, f'$H={H}$', transform=a.transAxes, ha='right', va='top')
        if k == 0: a.legend(loc='lower left')
    x, y = series(bb, 4, 2); check('F6e H=0.65 J=9 p=4 |z|<=1.4 at t=1', abs(y[np.argmin(abs(x - 1))]) <= 1.4, y)
    for a,ss in zip(ax,'ab'):letter(a,ss)
    save(fig,'v5_figS5_barrier',lay,dict(panels={'a,b':'legacy fig8_barrier axes 4,5'},from_v4='11a,11b'))

# ======================================================================= Fig 7: linear oscillator
def fig_osc(lay):
    fig, ax = layout('wide', lay, 3, h_one=2.50)
    sw = pd.DataFrame(OS['sweep'])
    for col, lab, c, ls in [('varQ', r'Var $Q$', '#0072B2', '-'), ('varV', r'Var $V$', '#D55E00', '-'), ('EE', r'$\mathrm{E}[E]$', '#2166ac', '--')]:
        ax[0].plot(sw.H, sw[col], color=c, ls=ls, lw=1.3, label=lab)
    for H in ['0.5', '0.7', '0.9']:
        R = OS['ref'][f'{H}_20.0']; ax[0].plot([float(H)] * 2, [R[0][0], R[1][1]], 'o', mfc='white', mec='0.2', ms=3.4, mew=0.8, ls='none')
    ax[0].plot([], [], 'o', mfc='white', mec='0.2', label='quadrature')
    check('F7a Var V(20) 0.1333 -> 0.0323', abs(sw.varV.iloc[0] - 0.1333) < 1e-4 and abs(sw.varV.iloc[-1] - 0.0323) < 1e-4, (sw.varV.iloc[0], sw.varV.iloc[-1]))
    ax[0].set_xlabel('$H$'); ax[0].set_ylabel('moment at $t=20$'); ax[0].legend(loc='lower left', ncol=1)
    ax[1].plot(sw.H, sw.ratio, color='#542788', lw=1.4); ax[1].axhline(1, color='0.6', lw=0.6)
    check('F7b ratio 1.00 -> 5.30', abs(sw.ratio.iloc[-1] - 5.30) < 0.01, sw.ratio.iloc[-1])
    ax[1].set_xlabel('$H$'); ax[1].set_ylabel(r'$\omega^2\,$Var $Q$ / Var $V$ at $t=20$')
    tg = np.array(OS['tgrid'])
    for H, ls in [('0.5', (0, (1.2, 1.3))), ('0.7', (0, (5, 1.6))), ('0.9', '-')]:
        cv = OS['curves'][H]; ax[2].plot(tg, cv['varQ'], color='#0072B2', ls=ls, lw=1.1); ax[2].plot(tg, cv['varV'], color='#D55E00', ls=ls, lw=1.1)
        for tt in [2.0, 5.0, 10.0, 15.0, 20.0]:
            R = OS['ref'].get(f'{H}_{tt}')
            if R: ax[2].plot([tt, tt], [R[0][0], R[1][1]], 'o', mfc='white', mec='0.2', ms=2.8, mew=0.7, ls='none')
    hs = [Line2D([], [], color='#0072B2', lw=1.1), Line2D([], [], color='#D55E00', lw=1.1)] + [Line2D([], [], color='0.3', ls=ls, lw=1.1) for ls in [(0, (1.2, 1.3)), (0, (5, 1.6)), '-']]
    ax[2].legend(hs, [r'Var $Q$', r'Var $V$', r'$H=\frac{1}{2}$', '$H=0.7$', '$H=0.9$'], loc='upper right', ncol=2, columnspacing=0.8)
    ax[2].set_xlabel('$t$'); ax[2].set_ylabel('variance, $J=513$'); ax[2].set_ylim(0, 0.36)
    for a, s in zip(ax, 'abc'): letter(a, s)
    save(fig, 'v5_fig07_oscillator', lay, dict(panels={'a,b': 'v12_oscillator.json sweep (J=257), ref (quadrature)', 'c': 'v12_oscillator.json curves (J=513), ref'}, from_v4='12a,12b,13a'))

# ======================================================================= Fig 8: Duffing and pendulum histories (new computation)
def fig_nonlin_osc(lay):
    fig, axs = plt.subplots(2, 2, figsize=(W_ONE if lay == 'one' else W_TWO, 4.5), constrained_layout=True)
    ax = axs.ravel()
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, hspace=0.06)
    t = np.array(OT['t']); m = t >= 1.0
    chk = OT['reproduction_max_rel_diff_at_stored_times']; check('F8 MC rerun reproduces stored values', max(chk.values()) < 1e-12, max(chk.values()))
    H = '0.9'; mc = np.array(OT['duff'][f'mc_{H}_1000']['var'])[:, 0]; se = np.array(OT['duff'][f'mc_{H}_1000']['se_var'])[:, 0]
    ax[0].fill_between(t, mc - 2 * se, mc + 2 * se, color=MC_COL, alpha=0.25, lw=0)
    ax[0].plot(t, mc, color=MC_COL, lw=0.9, label=r'MC ($2^{17}$ paths)')
    cols = {1: '0.45', 2: '#0072B2', 3: '#2166ac'}
    for p, k in [(1, '65_1'), (2, '65_2'), (3, '17_3')]:
        v = np.array(OT['duff'][f'{H}_{k}']['var'])[:, 0]; ax[0].plot(t, v, color=cols[p], **DEG_STY[p], label=f'$p={p}$, $J={k.split("_")[0]}$')
    ax[0].set_xlabel('$t$'); ax[0].set_ylabel(r'Duffing Var $Q(t)$, $H=0.9$'); ax[0].legend(loc='lower right'); ax[0].set_xlim(0, 10)
    ax[1].fill_between(t[m], -200 * se[m] / mc[m], 200 * se[m] / mc[m], color=MC_COL, alpha=0.2, lw=0)
    for p, k in [(1, '65_1'), (2, '65_2'), (3, '17_3')]:
        v = np.array(OT['duff'][f'{H}_{k}']['var'])[:, 0]; ax[1].plot(t[m], 100 * (v[m] / mc[m] - 1), color=cols[p], **DEG_STY[p], label=f'$p={p}$')
    i10 = -1; check('F8b Duffing t=10 p=2 deviation -5.8 % (Table 10)', abs(100 * (np.array(OT['duff'][f'{H}_65_2']['var'])[i10, 0] / mc[i10] - 1) + 5.8) < 0.06)
    ax[1].axhline(0, color='0.6', lw=0.6); ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'deviation of Var $Q$ from MC (%)'); ax[1].legend(loc='lower right'); ax[1].set_xlim(0, 10)
    for H in ['0.6', '0.75', '0.9']:
        mcp = np.array(OT['pend'][f'mc_{H}_1000']['var'])[:, 0]; sep = np.array(OT['pend'][f'mc_{H}_1000']['se_var'])[:, 0]
        v = np.array(OT['pend'][f'{H}_65_1']['var'])[:, 0]
        ax[2].fill_between(t[m], 100 * (-2 * sep[m] / mcp[m]), 100 * (2 * sep[m] / mcp[m]), color=HC(H), alpha=0.15, lw=0)
        ax[2].plot(t[m], 100 * (v[m] / mcp[m] - 1), color=HC(H), lw=1.3, label=f'$H={H}$')
    check('F8c pendulum t=10 deviations -2.4/-18.5/-71.4 %', all(abs(100 * (np.array(OT['pend'][f'{H}_65_1']['var'])[-1, 0] / np.array(OT['pend'][f'mc_{H}_1000']['var'])[-1, 0] - 1) - d) < 0.06 for H, d in [('0.6', -2.4), ('0.75', -18.5), ('0.9', -71.4)]))
    ax[2].axhline(0, color='0.6', lw=0.6); ax[2].set_xlabel('$t$'); ax[2].set_ylabel(r'pendulum: deviation of Var $Q$ (%)'); ax[2].legend(loc='lower left'); ax[2].set_xlim(0, 10)
    degree = J_(V, 'evidence', 'pendulum_degree_refinement', 'final_diagnostics.json')['selected']
    ps = np.array([r['p'] for r in degree])
    ef = np.array([r['error_full'] for r in degree]); ej = np.array([r['error_same_J'] for r in degree])
    check('F8d fixed J=3 measured degrees', all(r['J'] == 3 for r in degree) and ps.tolist() == [1,2,4,6,8])
    check('F8d degree-eight displacement improvement', abs(ef[-1]+14.060528912354197)<1e-10)
    check('F8d nonmonotonic first two degrees retained', ef[1] < ef[0])
    ax[3].axhline(0, color='0.65', lw=0.7)
    ax[3].plot(ps, ef, 'o-', color='#0072B2', ms=3.5, label='full fBm reference')
    ax[3].plot(ps, ej, 's--', color='#D55E00', ms=3.2, label='same-mode reference')
    ax[3].set(xticks=ps, xlabel='chaos degree $p$', ylabel='pendulum: variance deviation (%)', ylim=(-82,12))
    ax[3].legend(loc='lower right')
    ax[3].text(0.04,0.96,r'$H=0.9$, $J=3$, $t=10$',transform=ax[3].transAxes,ha='left',va='top',fontsize=SMALL)
    for a, s in zip(ax, 'abcd'): letter(a, s, dx=0.0, dy=1.03)
    save(fig, 'v5_fig08_duffing_pendulum', lay, dict(panels={'a,b,c': 'evidence/oscillator_time/osc_time.json; historical arrays unchanged',
         'd':'evidence/pendulum_degree_refinement/final_diagnostics.json selected fixed-J rows; straight lines guide the eye'},
         from_v4='histories preserved; degree panel added'))

def fig_cost(lay):
    W = W_ONE if lay == 'one' else W_COL
    fig, ax = plt.subplots(figsize=(W, 2.6), constrained_layout=True)
    ou=BE['ou'];ch=ou['chaos'];mc=ou['mc']
    ax.loglog([c['time'] for c in ch],[c['relerr'] for c in ch],'-o',
              color='#0072B2',ms=4,label='solver: absolute relative error')
    ax.loglog([c['time'] for c in mc],[c['rmse'] for c in mc],'s',
              color=MC_COL,mfc='none',ms=4,label='MC: relative RMS, 6 repetitions')
    for c,offset in [(ch[1],(4,5)),(ch[-1],(4,5))]:
        ax.annotate(f"$J={c['J']}$",(c['time'],c['relerr']),xytext=offset,
                    textcoords='offset points',fontsize=SMALL)
    ax.set_xlabel('CPU time (s)');ax.set_ylabel('variance error, OU at $t=0.5$')
    ax.legend(loc='upper right',fontsize=SMALL)
    ax.set_xlim(.011,22);ax.set_ylim(5e-5,.3)
    check('OU final MC RMS uses M=65536, not preceding run',abs(mc[-1]['rmse']-.0084258)<1e-6)
    save(fig,'v5_fig09_cost',lay,dict(panels={'-':'v12_bench.json ou; six solver settings, four MC settings, six MC repetitions'},
         from_v4='Table 11, expanded plotted settings',note='No bias-floor line: empirical bias does not certify time-discretisation bias. Population SE and OU RMS are not mixed.'))
# ======================================================================= supplementary figures
def supp_figs():
    lay = 'supplement'
    fig, ax = plt.subplots(1, 1, figsize=(W_ONE * 0.55, 2.3), constrained_layout=True)
    df = csv('fig2_noise_paths')
    for i, (Jv, c) in enumerate(zip([5, 17, 65, 257], ['#c6dbef', '#6baed6', '#2171b5', '#08306b'])):
        x, y = series(df, 3, i); ax.plot(x, y, color=c, lw=0.9 + 0.2 * i, label=f'$J={Jv}$')
    ax.set_xlabel('$t/T$'); ax.set_ylabel(r'$B_H^J(t)$, one path, $H=0.8$'); ax.legend(loc='upper left', ncol=2)
    save(fig, 'v5_figS2_noise_path', lay, dict(panels={'-': 'V4 plotted data fig2_noise_paths axis 3'}, from_v4='3b'))
    fig, ax = plt.subplots(1, 1, figsize=(W_ONE * 0.55, 2.3), constrained_layout=True)
    for Jv, c in zip(['8', '32', '128'], ['#9ecae1', '#4292c6', '#08306b']):
        L = np.array(TN['limit'][Jv]); ax.loglog(L[:, 0], L[:, 2], color=c, marker='o', ms=2.6, label=f'max corr, $J={Jv}$')
        ax.loglog(L[:, 0], L[:, 1], color=c, marker='s', ms=2.6, ls='--', lw=0.9)
    ax.plot([], [], color='0.3', marker='s', ls='--', ms=2.6, label=r'max$|G^H_{kl}-\delta_{kl}|$'); ax.set_xlabel(r'$H-\frac{1}{2}$'); ax.set_ylabel(r'distance of $G^H$ from $I$'); ax.legend(loc='upper left', fontsize=SMALL)
    save(fig, 'v5_figS3_gram_limit_entrywise', lay, dict(panels={'-': 'theory_noise.json limit[J] (columns: H-1/2, max|G-I|, max corr)'}, from_v4='4b'))
    fig, ax = plt.subplots(1, 3, figsize=(W_ONE, 2.1), constrained_layout=True)
    dm = csv('fig3_memory')
    for i, H in enumerate(['0.5', '0.7', '0.95']):
        x, y = series(dm, 3, i); ax[0].loglog(x, y, color=HC(H), marker='o', ms=2.6, label=f'$H={H}$')
    ax[0].set_xlabel('$J$'); ax[0].set_ylabel('memory variance error'); ax[0].legend(loc='lower left')
    dp = csv('fig3_population')
    for i, H in enumerate(['0.5', '0.55', '0.75', '0.95']):
        x, y = series(dp, 5, i); ax[1].loglog(x, y, color=HC(H), marker='o', ms=2.6, label=f'$H={H}$')
    ax[1].set_xlabel('$J$'); ax[1].set_ylabel('population peak error'); ax[1].legend(loc='lower left')
    for H in ['0.5', '0.7', '0.9']:
        ax[2].loglog(OS['Js'], OS['conv'][f'{H}_10.0'], color=HC(H), marker='o', ms=2.6, label=f'$H={H}$')
    ax[2].set_xlabel('$J$'); ax[2].set_ylabel(r'oscillator variance error'); ax[2].legend(loc='lower left')
    ax[1].set_ylim(1e-6,0.1)
    jticks(ax[0],[9,33,129,257]);jticks(ax[1],[17,33,65]);jticks(ax[2],[9,33,129,513])
    for a, ss in zip(ax, 'abc'): letter(a, ss)
    save(fig, 'v5_figS4_rates_individual', lay, dict(panels={'a': 'V4 plotted data fig3_memory axis 3', 'b': 'V4 plotted data fig3_population axis 5 (reference: extrapolated value)', 'c': 'v12_oscillator.json conv'}, from_v4='7b, 8b, 13b'))



def fig_temperature():
    R=J_(ED,'new_apps_v7.json');d=np.loadtxt(os.path.join(ED,'nhemitemp_annual.csv'),delimiter=',',skiprows=1)
    fig=plt.figure(figsize=(W_ONE,3.9),constrained_layout=True)
    gs=fig.add_gridspec(2,1,height_ratios=[1.2,1.0]);ax=[fig.add_subplot(gs[i]) for i in (0,1)]
    years,y=d[:,0],d[:,1];te=years>=1960
    ax[0].plot(years[~te],y[~te],color='0.4',lw=.8,label='observed, fit period')
    ax[0].plot(years[te],y[te],'o',color='black',ms=2.8,label='observed, 1960–1989')
    for key,col,lab,ls in [('const_fOU','#0072B2',r'fOU, $\hat H=0.92$','-'),
                          ('const_OU','#D55E00',r'OU, $H=1/2$','--')]:
        f=R['forecast'][key];mu=np.array(f['mean']);sd=np.array(f['sd'])
        ax[0].fill_between(years[te],mu-1.96*sd,mu+1.96*sd,color=col,alpha=.14,lw=0)
        ax[0].plot(years[te],mu,color=col,ls=ls,label=lab)
    ax[0].set(xlabel='year',ylabel='anomaly (°C)',ylim=(-1.1,.5))
    ax[0].legend(loc='lower left',ncol=2)
    for key,col,lab in [('const_profile','#0072B2','constant mean'),('trend_profile','#D55E00','linear trend')]:
        h=np.array(R['fit'][key]['H']);v=np.array(R['fit'][key]['nll']);ix=np.argsort(h)
        ax[1].plot(h[ix],v[ix]-v.min(),'o-',color=col,ms=3,label=lab)
    ax[1].axhline(1.92,color='0.5',ls=':',label='nominal 95% cut-off')
    ax[1].set(xlabel='$H$',ylabel=r'profile $-\log L$ (relative)',ylim=(-.2,6))
    ax[1].legend(loc='upper right')
    for a,ss in zip(ax,'ab'):letter(a,ss,dx=-.08)
    save(fig,'v5_figS1_temperature','supplement',dict(panels={'a,b':'new_apps_v7.json and nhemitemp_annual.csv'},
      from_v4='Supplement S1(a,b)',note='Old cost panel removed; OU cost is main Figure 9; complete costs retained in Table 11.'))

if __name__ == '__main__':
    for lay in ['one', 'two']:
        for f in [fig_gram, fig_noise, fig_worked, fig_limit, fig_mempop, fig_cubic, fig_osc, fig_nonlin_osc, fig_cost]:
            f(lay)
    supp_figs()
    fig_barrier()
    fig_temperature()
    json.dump(CHECKS, open(os.path.join(V, 'figure_design', 'V5_CHECKS.json'), 'w'), indent=1)
    json.dump(PROV, open(os.path.join(V, 'figure_design', 'V5_PROVENANCE.json'), 'w'), indent=1)
    bad = {k: v['layouts'] for k, v in PROV.items() if any(l['text_overlaps'] or l['text_outside'] for l in v['layouts'].values())}
    print('checks', len(CHECKS), 'all ok' if all(c['ok'] for c in CHECKS) else 'FAIL', '| figures', len(PROV), '| layout issues', json.dumps(bad)[:1500])

