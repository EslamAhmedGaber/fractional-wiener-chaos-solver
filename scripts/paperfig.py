"""paperfig: publication figures for mathematical and numerical papers.

Deterministic Matplotlib helpers for figure families that recur in applied
mathematics papers. Each helper draws onto a given Axes, so panels combine
freely. Nothing here invents data: every function plots what it is given.
Adapt or replace any choice when a better encoding serves the content.

    import paperfig as pf
    pf.use_style()
    fig, axes = pf.figure(cols=2, width='double')
    pf.order_plot(axes[0], N, {'method A': errA}, slopes=[-2])
    pf.label_panels(axes); pf.save(fig, 'fig_convergence')
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Okabe-Ito: distinguishable under common colour-vision deficiencies.
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000', '#F0E442']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
LINES = ['-', '--', '-.', ':']
ACCENT = COLORS[0]        # the proposed method
REFERENCE = '#7a7a7a'     # exact values, reference slopes, theory lines
WIDTHS = {'single': 3.35, 'onehalf': 5.0, 'double': 6.9}   # inches; check the venue


def use_style(base_size: float = 8.5):
    """Readable at print size: 8-9 pt text, thin but visible lines, no chart junk."""
    mpl.rcParams.update({
        'font.size': base_size, 'axes.titlesize': base_size + 0.5, 'axes.labelsize': base_size,
        'xtick.labelsize': base_size - 1, 'ytick.labelsize': base_size - 1,
        'legend.fontsize': base_size - 1, 'legend.frameon': False,
        'font.family': 'serif', 'mathtext.fontset': 'cm',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'lines.markersize': 3.5,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.prop_cycle': mpl.cycler(color=COLORS),
    })


def figure(rows=1, cols=1, width='double', aspect=0.78, **kw):
    """Figure sized for the page; aspect = panel height / panel width."""
    w = WIDTHS.get(width, width) if isinstance(width, str) else width
    h = w / cols * aspect * rows
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), constrained_layout=True, squeeze=False, **kw)
    return fig, axes.ravel()


def label_panels(axes, start='a', fmt='({})', x=-0.02, y=1.02):
    """Panel letters in reading order, matching caption references."""
    for i, ax in enumerate(axes):
        ax.text(x, y, fmt.format(chr(ord(start) + i)), transform=ax.transAxes,
                ha='right', va='bottom', fontweight='bold')


def save(fig, stem, formats=('pdf', 'png'), out_dir='.'):
    """Vector PDF for the manuscript, PNG for previews; returns the paths."""
    paths = []
    for f in formats:
        p = Path(out_dir) / f'{stem}.{f}'
        fig.savefig(p)
        paths.append(p)
    return paths


def tex_sci(v):
    """1e-09 -> $10^{-9}$; 3e-4 -> $3\\times10^{-4}$. Never print 1e-09 on a figure."""
    m, e = f'{v:.0e}'.split('e')
    e = int(e)
    return f'$10^{{{e}}}$' if m == '1' else f'${m}\\times10^{{{e}}}$'


# --------------------------------------------------------------- MF08 order plot
def order_plot(ax, x, series: dict, slopes=(), slope_anchor=None, floor=None,
               xlabel='$N$', ylabel='error', fit_range=None, var=None, offset=3.0):
    """Error against resolution on log-log axes, with labelled reference orders.

    series: name -> errors (non-positive values are dropped, never replaced).
    slopes: reference orders, grey dotted, drawn `offset` times above the data
            so they never hide a series; labelled with `var` (default: xlabel).
    floor:  e.g. rounding level; dashed line, named at the right.
    fit_range: (xmin, xmax) to report a least-squares slope per series in the legend.
    """
    x = np.asarray(x, float)
    for i, (name, y) in enumerate(series.items()):
        y = np.asarray(y, float)
        ok = y > 0
        label = name
        if fit_range is not None:
            m = ok & (x >= fit_range[0]) & (x <= fit_range[1])
            if m.sum() >= 2:
                p = np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]
                label = f'{name} (slope {p:.2f})'
        ax.loglog(x[ok], y[ok], LINES[0], marker=MARKERS[i % 8], color=COLORS[i % 8], label=label)
    xa, xb = x.min(), x.max()
    v = var or xlabel.strip('$')
    for s in slopes:
        if slope_anchor is None:
            anchor = (xa, offset * max(np.asarray(y, float)[0] for y in series.values()))
        else:
            anchor = slope_anchor
        c = anchor[1] / anchor[0] ** s
        xs = np.array([xa, xb])
        ax.loglog(xs, c * xs ** s, color=REFERENCE, lw=0.8, ls=':')
        ax.annotate(f'${v}^{{{s:g}}}$', (xb, c * xb ** s), xytext=(2, 0), textcoords='offset points',
                    color=REFERENCE, ha='left', va='center', annotation_clip=False)
    if floor is not None:
        # Named in the legend: a free-floating label collides with data near the floor.
        ax.axhline(floor, color=REFERENCE, lw=0.7, ls='--', label='rounding level')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc='best')


# ---------------------------------------------------- MF17 work-precision front
def pareto_front(cost, err):
    """Indices of runs not dominated in (cost, error)."""
    order = np.argsort(cost)
    keep, best = [], np.inf
    for i in order:
        if err[i] < best:
            keep.append(i); best = err[i]
    return np.array(keep)


def work_precision(ax, methods: dict, target=None, xlabel='wall time (s)', ylabel='relative error',
                   accent=None):
    """methods: name -> (cost, error) of all runs; only each Pareto front is drawn.

    The proposed method (`accent`) is thicker; a target line answers
    "who reaches this accuracy first".
    """
    for i, (name, (c, e)) in enumerate(methods.items()):
        c, e = np.asarray(c, float), np.asarray(e, float)
        k = pareto_front(c, e)
        strong = name == accent
        ax.loglog(c[k], e[k], LINES[0 if strong else 1], marker=MARKERS[i % 8], color=COLORS[i % 8],
                  lw=1.8 if strong else 1.0, label=name, zorder=3 if strong else 2)
    if target is not None:
        ax.axhline(target, color=REFERENCE, lw=0.7, ls=':')
        ax.annotate(tex_sci(target), (1, target), xycoords=('axes fraction', 'data'), xytext=(2, 0),
                    textcoords='offset points', color=REFERENCE, va='center', ha='left', annotation_clip=False)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(loc='best')


# ------------------------------------------------------ certified bracket plot
def bracket_plot(ax, x, lower, upper, exact=None, xlabel='dimension $D$', ylabel='value', relative_to=None):
    """Deterministic bracket [lower, upper] against resolution (linear y, log2 x).

    relative_to: plot (bound - ref)/|ref| so both ends stay visible.
    """
    x = np.asarray(x, float); lo = np.asarray(lower, float); up = np.asarray(upper, float)
    ref = relative_to
    f = (lambda v: (np.asarray(v, float) - ref) / abs(ref)) if ref is not None else (lambda v: v)
    ax.fill_between(x, f(lo), f(up), color=ACCENT, alpha=0.18, lw=0, label='computed bracket')
    ax.plot(x, f(lo), color=ACCENT, lw=1.0, marker='o'); ax.plot(x, f(up), color=ACCENT, lw=1.0, marker='o')
    if exact is not None:
        ax.axhline(f(exact), color='k', lw=0.8, ls='--', label='exact value')
    ax.set_xscale('log', base=2)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel if ref is None else f'({ylabel} $-$ exact) / exact')
    ax.legend(loc='best')


def interval_width(ax, cost_det, width_det, cost_mc, width_mc, xlabel='wall time (s)',
                   ylabel='interval width', det_label='deterministic bracket', mc_label='Monte Carlo 95% CI'):
    """Two kinds of interval against cost: a bracket and a statistical CI.

    The thin continuation shows the M^(-1/2) law of the CI. The caption must say
    that the two intervals have different meanings.
    """
    ax.loglog(cost_det, width_det, '-s', color=ACCENT, label=det_label)
    ax.loglog(cost_mc, width_mc, 'o', mfc='white', color=COLORS[1], label=mc_label)
    c = np.asarray(cost_mc, float); w = np.asarray(width_mc, float)
    k = np.exp(np.mean(np.log(w) + 0.5 * np.log(c)))
    xs = np.logspace(np.log10(c.min()), np.log10(max(c.max(), np.max(cost_det)) * 100), 50)
    ax.loglog(xs, k * xs ** -0.5, color=COLORS[1], lw=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(loc='best')


# --------------------------------------------- theorem band (verification plot)
def band_plot(ax, x, curves: dict, band=(0.5, 1.0), band_label='proven band', xlabel='$n$',
              ylabel='normalized value', logx=True):
    """Normalized quantities against a proven interval: the theorem as a shaded region."""
    ax.axhspan(*band, color='#dfe8f3', zorder=0, label=band_label)
    for b in band:
        ax.axhline(b, color=REFERENCE, lw=0.6)
    for i, (name, y) in enumerate(curves.items()):
        ax.plot(x, y, LINES[i % 4], color=COLORS[i % 8], label=name)
    if logx:
        ax.set_xscale('log')
    lo, hi = band
    pad = 0.15 * (hi - lo)
    ax.set_ylim(min(lo - pad, min(np.min(v) for v in curves.values()) - pad),
                max(hi + pad, max(np.max(v) for v in curves.values()) + pad))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(loc='lower left', ncol=2)


# ------------------------------------------------- MF15/MF24 posterior band
def posterior_band(ax, t, mean, sd, truth=None, obs_t=None, level=1.96, xlabel='$t$', ylabel='$B_H(t)$'):
    """Latent path, posterior mean and pointwise band; observation locations as ticks."""
    ax.fill_between(t, mean - level * sd, mean + level * sd, color=ACCENT, alpha=0.2, lw=0,
                    label='95% pointwise band')
    ax.plot(t, mean, color=ACCENT, label='posterior mean')
    if truth is not None:
        ax.plot(t, truth, color='k', lw=0.8, label='latent path')
    if obs_t is not None:
        y0 = ax.get_ylim()[0]
        ax.plot(obs_t, np.full_like(obs_t, y0), '|', color=REFERENCE, ms=5, label='sensor centres')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2)


# ------------------------------------------------------------- MF16 heatmap
def param_heatmap(ax, x, y, Z, xlabel, ylabel, cbar_label, diverging=False, contour=None, fig=None,
                  log=False):
    """Parameter map. Diverging colormap only for signed quantities around a meaningful 0."""
    Zp = np.log10(Z) if log else Z
    if diverging:
        v = np.nanmax(np.abs(Zp)); cmap, vmin, vmax = 'RdBu_r', -v, v
    else:
        cmap, vmin, vmax = 'viridis', np.nanmin(Zp), np.nanmax(Zp)
    m = ax.pcolormesh(x, y, Zp, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    if contour is not None:
        cs = ax.contour(x, y, Zp, levels=[np.log10(contour) if log else contour], colors='w', linewidths=1.0)
        ax.clabel(cs, fmt=(lambda v: tex_sci(10 ** v)) if log else '%g', fontsize=7)
    cb = (fig or ax.figure).colorbar(m, ax=ax, pad=0.02)
    cb.set_label(('log$_{10}$ ' if log else '') + cbar_label); cb.outline.set_linewidth(0.4)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


# ----------------------------------------------------------- MF18 risk backtest
def risk_backtest(ax, t, pnl, var, alpha, label_var='VaR'):
    """Losses against the VaR forecast; exceedances marked; counts in the legend.

    Convention: pnl is profit (negative = loss); var is a positive loss level,
    so an exceedance is pnl < -var. Report the test statistic in the caption.
    """
    pnl, var = np.asarray(pnl, float), np.asarray(var, float)
    exc = pnl < -var
    ax.plot(t, pnl, color='#9a9a9a', lw=0.5, label='daily P&L')
    ax.plot(t, -var, color=ACCENT, lw=1.1, label=f'$-${label_var}$_{{{alpha:g}}}$ forecast')
    ax.plot(np.asarray(t)[exc], pnl[exc], 'v', color=COLORS[1], ms=3.5,
            label=f'exceedances: {exc.sum()} (expected {alpha * len(pnl):.1f})')
    ax.set_xlabel('day'); ax.set_ylabel('P&L')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    return int(exc.sum())


def kupiec_pof(n_exc, n, alpha):
    """Kupiec proportion-of-failures LR statistic and chi2(1) p-value."""
    from scipy.stats import chi2
    x = n_exc
    phat = x / n
    def ll(q):
        return (n - x) * math.log(1 - q) + (x * math.log(q) if x else 0.0)
    lr = -2 * (ll(alpha) - (ll(phat) if 0 < phat < 1 else 0.0))
    return lr, 1 - chi2.cdf(lr, 1)


# ------------------------------------------------- MF28 procedure diagram
def procedure_diagram(ax, columns, title=None):
    """Boxes in columns: e.g. 'once' preparation -> 'per input' steps -> uses.

    columns: list of (header, [box texts]). Boxes in a column are joined top to
    bottom; one straight arrow joins consecutive columns at mid-height.
    Keep box text short; mathtext is fine.
    """
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nc = len(columns); cw = 1 / nc; bw = cw * 0.74
    prev_right = None
    for j, (header, boxes) in enumerate(columns):
        cx = cw * (j + 0.5)
        ax.text(cx, 0.97, header, ha='center', va='top', fontweight='bold', color=COLORS[j % 8])
        nb = len(boxes); bh = min(0.17, 0.78 / nb - 0.04)
        ys = np.linspace(0.84 - bh / 2, 0.06 + bh / 2, nb) if nb > 1 else [0.45]
        for i, (txt, y) in enumerate(zip(boxes, ys)):
            ax.add_patch(FancyBboxPatch((cx - bw / 2, y - bh / 2), bw, bh,
                                        boxstyle='round,pad=0.008,rounding_size=0.015',
                                        fc='white', ec=COLORS[j % 8], lw=1.0))
            ax.text(cx, y, txt, ha='center', va='center')
            if i > 0:
                ax.add_patch(FancyArrowPatch((cx, ys[i - 1] - bh / 2), (cx, y + bh / 2), arrowstyle='-|>',
                                             mutation_scale=7, lw=0.8, color='k'))
        if prev_right is not None:
            ym = float(np.mean(ys))
            ax.add_patch(FancyArrowPatch((prev_right, ym), (cx - bw / 2, ym), arrowstyle='-|>',
                                         mutation_scale=9, lw=1.0, color='k'))
        prev_right = cx + bw / 2
    if title:
        ax.set_title(title)
