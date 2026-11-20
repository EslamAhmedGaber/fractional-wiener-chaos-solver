"""Figure 5 (Problem 3, 2D competitive Lotka-Volterra) in both layouts:
(a) Continuous variance and cross-covariance trajectories at H = 0.95 from CosChaos (J=33, p=2) vs 2D WIS-RK4 ref;
(b) Memory-driven competitive reversal P(X1(T) > X2(T)) across H;
(c) 2D work-precision at H = 0.95: CosChaos (p=2, p=1) vs 2D WIS-RK4 + CV MC vs 2D GfBm EM + CV MC."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import setup, save, panel_label, clean, EVID, WIDTH, COS, COS2, REF, FWHE, WIS, EXACT_MC, BP, HSTYLE

P3 = os.path.join(EVID, 'P3_lotka_volterra')
R = json.load(open(os.path.join(P3, 'p3_results.json')))
SUM = json.load(open(os.path.join(P3, 'p3_summary.json')))


def fig5(layout):
    setup()
    w = WIDTH[layout]
    if layout == 'two_columns':
        fig, axs = plt.subplots(1, 3, figsize=(w, 2.90), gridspec_kw=dict(width_ratios=[1.12, 1.04, 1.08]))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.30, top=0.87, wspace=0.38)
        ax_a, ax_b, ax_c = axs
    else:
        fig, grid = plt.subplots(2, 2, figsize=(w, 4.45))
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.92, wspace=0.32, hspace=0.50)
        ax_a, ax_b, ax_c, lax = grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]
        lax.axis('off')

    # (a) Full 2x2 covariance components at H = 0.95: Var X2, Var X1, Cov(X1, X2)
    c95 = R['cases']['0.95']
    dt = np.array(c95['coschaos']['dense_times'])
    dcov = np.array(c95['coschaos']['dense_cov'])
    rt = np.array(c95['reference']['times'])
    rcov = np.array(c95['reference']['cov'])
    rse = np.array(c95['reference']['cov_se'])

    ax_a.plot(dt, dcov[:, 1, 1], color=COS, lw=1.7, label='$\\mathrm{Var}\\,X_2(t)$')
    ax_a.errorbar(rt, rcov[:, 1, 1], yerr=2.0 * rse[:, 1, 1], fmt='o', ms=3.4, color=REF, lw=0.6, capsize=1.2, zorder=5)
    ax_a.plot(dt, dcov[:, 0, 1], color=BP, lw=1.4, ls='--', label='$\\mathrm{Cov}(X_1,X_2)$')
    ax_a.errorbar(rt, rcov[:, 0, 1], yerr=2.0 * rse[:, 0, 1], fmt='s', ms=3.1, color=REF, mfc='white', lw=0.6, capsize=1.2, zorder=5)
    ax_a.plot(dt, dcov[:, 0, 0], color=COS2, lw=1.4, ls='-.', label='$\\mathrm{Var}\\,X_1(t)$')
    ax_a.errorbar(rt, rcov[:, 0, 0], yerr=2.0 * rse[:, 0, 0], fmt='^', ms=3.2, color=REF, mfc='white', lw=0.6, capsize=1.2, zorder=5)
    ax_a.set_xlabel('$t$')
    ax_a.set_ylabel('covariance entries')
    ax_a.set_xlim(0.0, 16.5)
    ax_a.set_xticks([0, 4, 8, 12, 16])
    ax_a.set_title('full $2\\times2$ covariance ($H=0.95$)', fontsize=7.5, pad=6)
    ax_a.legend(loc='upper left', fontsize=6.2)
    clean(ax_a)
    panel_label(ax_a, 'a', x=-0.18, y=1.06)

    # (b) Competitive reversal probability P(X1(T) > X2(T)) across H
    Hs = np.array([0.55, 0.75, 0.95])
    p_ref = [100.0 * R['cases'][f'{h:.2f}']['reference']['p_dom'] for h in Hs]
    p_se = [200.0 * R['cases'][f'{h:.2f}']['reference']['p_dom_se'] for h in Hs]
    p_cc = [100.0 * next(r['p_dom'] for r in R['cases'][f'{h:.2f}']['coschaos']['rows'] if r['J'] == 17 and r['p'] == 2) for h in Hs]
    x = np.arange(len(Hs))
    bw = 0.34
    ax_b.bar(x - bw / 2, p_cc, width=bw, color=COS, alpha=0.9, label='CosChaos ($J=17$, $p=2$)')
    ax_b.bar(x + bw / 2, p_ref, width=bw, color='#cccccc', edgecolor=REF, lw=0.75, yerr=p_se, capsize=2.0,
             label='2D WIS-RK4 ref.')
    for xi, val in zip(x, p_ref):
        ax_b.text(xi, max(val, p_cc[xi]) + 1.1, f'{p_cc[xi]:.2f} / {val:.2f}',
                  ha='center', va='bottom', fontsize=6.0, color='#1b2a44')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f'$H={h}$' for h in Hs])
    ax_b.set_ylabel('$\\mathbb{P}(X_1(T)>X_2(T))$ (%)')
    ax_b.set_ylim(0.0, 32.5)
    ax_b.set_title('competitive reversal', fontsize=7.5, pad=6)
    ax_b.legend(loc='upper left', fontsize=6.1)
    clean(ax_b)
    panel_label(ax_b, 'b', x=-0.20, y=1.06)

    # (c) Work-precision at H = 0.95
    S95 = SUM['0.95']
    cc2 = sorted([r for r in S95['cc_evals'] if r['p'] == 2], key=lambda r: r['J'])
    cc1 = sorted([r for r in S95['cc_evals'] if r['p'] == 1], key=lambda r: r['J'])
    rk = sorted(S95['rk_evals'], key=lambda r: r['seconds'])
    em = sorted(S95['em_evals'], key=lambda r: r['seconds'])

    ax_c.axhline(1.0, color='#bbbbbb', lw=0.6, ls=':')
    ax_c.loglog([r['seconds'] for r in cc2], [r['max_cov'] for r in cc2],
                color=COS, marker='o', lw=1.55, ms=3.8, label='CosChaos, $p=2$ ($J$ modes)', zorder=4)
    for r in cc2:
        if r['J'] in (5, 17, 65):
            off = (4, 3) if r['J'] == 5 else ((-6, -4) if r['J'] == 17 else (4, -3))
            ha = 'left' if r['J'] != 17 else 'right'
            va = 'bottom' if r['J'] == 5 else 'top'
            ax_c.annotate(f"$J={r['J']}$" if r['J'] == 5 else str(r['J']),
                          (r['seconds'], r['max_cov']), textcoords='offset points',
                          xytext=off, ha=ha, va=va, fontsize=6.0, color=COS)
    ax_c.loglog([r['seconds'] for r in cc1], [r['max_cov'] for r in cc1],
                color=FWHE, ls='--', marker='s', mfc='white', ms=3.2, lw=1.1,
                label='CosChaos, $p=1$ (Gaussian floor)', zorder=3)

    for rows, colour in ((cc2, COS), (cc1, FWHE)):
        errors = np.array([r['max_cov'] for r in rows])
        upper = np.array([r['criterion'] for r in rows])
        ax_c.errorbar([r['seconds'] for r in rows], errors,
                      yerr=[np.zeros_like(errors), np.maximum(upper-errors, 0.0)],
                      fmt='none', ecolor=colour, elinewidth=0.6, capsize=1.3)

    rx = np.array([r['seconds'] for r in rk])
    ry = np.array([r['max_cov'] for r in rk])
    rg = np.array([r['criterion'] for r in rk])
    ax_c.errorbar(rx, ry, yerr=[np.zeros_like(ry), np.maximum(rg - ry, 0.0)],
                  fmt='v', ms=3.6, mfc='none', mec=EXACT_MC, ecolor=EXACT_MC, elinewidth=0.75, capsize=1.6,
                  ls=':', lw=0.9, color=EXACT_MC, label='2D WIS-RK4 + CV + MC', zorder=3)

    ex = np.array([r['seconds'] for r in em])
    ey = np.array([r['max_cov'] for r in em])
    eg = np.array([r['criterion'] for r in em])
    ax_c.errorbar(ex, ey, yerr=[np.zeros_like(ey), np.maximum(eg - ey, 0.0)],
                  fmt='D', ms=3.4, mfc='none', mec=WIS, ecolor=WIS, elinewidth=0.7, capsize=1.5,
                  ls='-.', lw=0.85, color=WIS, label='2D GfBm EM + CV + MC', zorder=2)

    ax_c.set_xlabel('time (s)')
    ax_c.set_ylabel('max. cov. error (%)')
    ax_c.set_title('work\u2013precision ($H=0.95$)', fontsize=7.5, pad=6)
    ax_c.set_ylim(0.15, 35)
    clean(ax_c)
    panel_label(ax_c, 'c', x=-0.19, y=1.06)

    hs, ls = ax_c.get_legend_handles_labels()
    hs.append(Line2D([], [], marker='o', ls='none', color=REF, ms=3.5))
    ls.append('2D WIS-RK4 reference, panel (a)')
    if layout == 'two_columns':
        fig.legend(hs, ls, loc='lower center', bbox_to_anchor=(0.53, 0.01),
                   ncol=3, fontsize=6.4, columnspacing=1.1, frameon=False)
    else:
        lax.legend(hs, ls, loc='center', fontsize=6.8, frameon=False)
    save(fig, 'fig5_lv', layout)


if __name__ == '__main__':
    for lay in ('one_column', 'two_columns'):
        fig5(lay)
    print('fig5 written')
