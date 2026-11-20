"""Figure 1: (a) one fBm path rebuilt from J cosine coordinates (exact at t = T);
(b) the CosChaos pipeline with clean margins (zero clipped box borders)."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.fft import fft
from style import setup, save, panel_label, clean, EVID, WIDTH, COS, REF

sys.path.insert(0, os.path.join(EVID, 'solver'))
import hfc


def fbm_path(H, n, T, seed):
    k = np.arange(n + 1, dtype=float)
    g = 0.5 * ((k + 1.0) ** (2.0 * H) - 2.0 * (k ** (2.0 * H)) + np.abs(k - 1.0) ** (2.0 * H))
    lam = np.clip(fft(np.r_[g, g[1:-1][::-1]]).real, 0.0, None)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)
    return np.r_[0.0, (fft(w * np.sqrt(lam / (2.0 * n))).real[:n] * ((T / n) ** H)).cumsum()]


def rebuild(B, T, J, t):
    Z = hfc.path_coords(B, T, J)
    return hfc.Mcoef(J, T, t) @ Z


def box(ax, x, y, w, h, title, body, fc, fs=(7.0, 6.6), dy=0.07):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.022',
        fc=fc, ec='#55607a', lw=0.75, clip_on=False,
    ))
    ax.text(x + w / 2, y + h - 0.025, title, ha='center', va='top',
            fontsize=fs[0], weight='bold', color='#1b2a44', linespacing=1.12)
    ax.text(x + w / 2, y + h / 2 - dy, body, ha='center', va='center',
            fontsize=fs[1], color='#1b2a44', linespacing=1.32)


def fig1(layout):
    setup()
    w = WIDTH[layout]
    one = (layout == 'one_column')
    H, T, n = 0.75, 1.0, 4096
    B = fbm_path(H, n, T, 20261114)
    t = np.linspace(0.0, T, 801)
    grid = np.linspace(0.0, T, n + 1)

    fig = plt.figure(figsize=(w, 2.55))
    ax = fig.add_axes([0.10, 0.16, 0.33, 0.75] if one else [0.09, 0.16, 0.34, 0.75])
    ax.plot(grid, B, color=REF, lw=0.55, label='fBm path ($H=0.75$)')
    for J, lw, al in [(3, 1.0, 0.45), (9, 1.25, 0.72), (33, 1.55, 1.0)]:
        ax.plot(t, rebuild(B, T, J, t), color=COS, lw=lw, alpha=al, label=f'$B_H^J$, $J={J}$')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$B_H(t)$ and $B_H^J(t)$')
    ax.set_xlim(0.0, T)
    ax.legend(loc='lower left', fontsize=6.4)
    clean(ax)
    panel_label(ax, 'a', x=-0.18)

    # Generous xlim/ylim padding (-0.025..1.025) so FancyBboxPatch pad=0.012 is NEVER clipped!
    bx = fig.add_axes([0.465, 0.04, 0.525, 0.92] if one else [0.465, 0.04, 0.525, 0.92])
    bx.set_axis_off()
    bx.set_xlim(-0.025, 1.025)
    bx.set_ylim(-0.01, 1.02)
    bx.text(-0.02, 0.96, '(b)', weight='bold', fontsize=9)

    fs = (6.4, 6.0) if one else (6.9, 6.5)
    bw, xs = (0.295, (0.005, 0.352, 0.700))
    box(bx, xs[0], 0.50, bw, 0.41, 'Noise\n(once per $H$, $J$)',
        '$G^H$ in closed form\n$G^H=LL^\\top$\n$\\ell(t)=L^\\top h(t)$', '#e8eef7', fs)
    box(bx, xs[1], 0.50, bw, 0.41, 'Coefficients\n(once per equation)',
        'master equation\nfor $c_\\alpha(t)$, $|\\alpha|\\leq p$\nODE integration', '#e8eef7', fs)
    box(bx, xs[2], 0.50, bw, 0.41, 'Outputs\n(no new solve)',
        'mean $c_{\\mathbf{0}}(t)$\ncovariance $\\sum c_\\alpha c_\\alpha^\\top$\nlaw: evaluate $X(t,\\xi)$', '#e8eef7', fs)
    for x0, x1 in ((xs[0] + bw + 0.014, xs[1] - 0.014), (xs[1] + bw + 0.014, xs[2] - 0.014)):
        bx.add_patch(FancyArrowPatch((x0, 0.705), (x1, 0.705), arrowstyle='-|>',
                                     mutation_scale=8.5, lw=0.95, color='#55607a'))
    first = ('Wick-affine data: one $H$-free solve,\none contraction with $G^H$ per Hurst index' if one
             else 'Wick-affine data: one $H$-free solve, one contraction with $G^H$ per Hurst index')
    box(bx, 0.005, 0.05, 0.99, 0.36, 'Reuse',
        first + '\nsame $(H,T,J,p)$: every equation reuses the noise stage\n'
        'every output time and threshold reuses the coefficients',
        '#f6f3e8', fs, dy=0.04 if one else 0.07)
    save(fig, 'fig1_method', layout)


if __name__ == '__main__':
    for lay in ('one_column', 'two_columns'):
        fig1(lay)
    print('fig1 written')
