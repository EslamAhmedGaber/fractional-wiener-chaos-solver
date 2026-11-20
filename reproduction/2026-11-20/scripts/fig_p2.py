"""Figure 4 (Problem 2, population model) in both layouts:
(a) Smooth continuous variance trajectories (201 points) vs control-variate exact reference;
(b),(c) Work-precision at H = 0.95 (lam = 0.02, 0.05) with explicit +2 SE error caps on MC methods;
(d) Terminal probability density at H = 0.95, lam = 0.05 from the single CosChaos solve vs exact reference."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import setup, save, panel_label, clean, EVID, WIDTH, COS, REF, FWHE, WIS, EXACT_MC, HSTYLE

P2 = os.path.join(EVID, 'P2_population')
SUM = json.load(open(os.path.join(P2, 'p2_summary.json')))


def J(name):
    return json.load(open(os.path.join(P2, name)))


def pareto(xs, ys):
    pts = sorted(zip(xs, ys))
    out = []
    best = np.inf
    for x, y in pts:
        if y < best:
            out.append((x, y))
            best = y
    return out


def wp_panel(ax, key, title, letter):
    S = SUM[key]
    cc = sorted([r for r in S['rows'] if r['method'] == 'CosChaos' and r['p'] == 2], key=lambda r: r['J'])
    fw = sorted([r for r in S['rows'] if r['method'] == 'FWHE'], key=lambda r: -r['dt'])
    gf = [r for r in S['rows'] if r['method'] == 'GfBm EM']
    ex = [r for r in S['rows'] if r['method'] == 'Exact WIS + MC']

    ax.axhline(1.0, color='#bbbbbb', lw=0.6, ls=':')
    ax.loglog([r['seconds'] for r in cc], [r['max_var'] for r in cc],
              color=COS, marker='o', lw=1.55, ms=3.8, label='CosChaos, $p=2$ ($J$ modes)', zorder=4)
    place = {5: ((4, 3), 'left', 'bottom'), 17: ((-4, -3), 'right', 'top'), 65: ((4, -3), 'left', 'top')}
    for r in cc:
        if r['J'] in place:
            off, ha, va = place[r['J']]
            ax.annotate(f"$J={r['J']}$" if r['J'] == 5 else str(r['J']),
                        (r['seconds'], r['max_var']), textcoords='offset points',
                        xytext=off, ha=ha, va=va, fontsize=6.0, color=COS)

    ax.loglog([r['seconds'] for r in fw], [r['max_var'] for r in fw],
              color=FWHE, ls='--', marker='s', mfc='white', ms=3.4, lw=1.15,
              label='FWHE, first order ($\\Delta t$)', zorder=3)
    pub = next((r for r in fw if abs(r['dt'] - 0.05) < 1e-12), None)
    if pub:
        ax.annotate('published $\\Delta t$', (pub['seconds'], pub['max_var']),
                    textcoords='offset points', xytext=(5, 3), fontsize=5.8, color=FWHE, ha='left', va='bottom')

    for rows, colour in ((cc, COS), (fw, FWHE)):
        errors = np.array([r['max_var'] for r in rows])
        upper = np.array([r['criterion'] for r in rows])
        ax.errorbar([r['seconds'] for r in rows], errors,
                    yerr=[np.zeros_like(errors), np.maximum(upper-errors, 0.0)],
                    fmt='none', ecolor=colour, elinewidth=0.6, capsize=1.3)

    # Sampled methods: plot |err| marker + vertical bar up to guard = |err| + 2 SE (no confusing floating line)
    if gf:
        xs = np.array([r['seconds'] for r in gf])
        ys = np.array([r['max_var'] for r in gf])
        gs = np.array([r['criterion'] for r in gf])
        ax.errorbar(xs, ys, yerr=[np.zeros_like(ys), np.maximum(gs - ys, 0.0)],
                    fmt='D', ms=3.3, mfc='none', mec=WIS, ecolor=WIS, elinewidth=0.65, capsize=1.5,
                    label='GfBm EM + CV + MC ($n$, $M$)', zorder=2)
    if ex:
        ex_s = sorted(ex, key=lambda r: r['seconds'])
        xs = np.array([r['seconds'] for r in ex_s])
        ys = np.array([r['max_var'] for r in ex_s])
        gs = np.array([r['criterion'] for r in ex_s])
        ax.errorbar(xs, ys, yerr=[np.zeros_like(ys), np.maximum(gs - ys, 0.0)],
                    fmt='v', ms=3.6, mfc='none', mec=EXACT_MC, ecolor=EXACT_MC, elinewidth=0.75, capsize=1.6,
                    ls=':', lw=0.85, color=EXACT_MC, label='exact WIS + CV + MC', zorder=3)

    ax.set_title(title, fontsize=7.6)
    ax.set_xlabel('time (s)')
    ax.set_ylim(0.008, 65)
    clean(ax)
    panel_label(ax, letter, x=-0.18)


def fig4(layout):
    setup()
    w = WIDTH[layout]
    fig, grid = plt.subplots(2, 2, figsize=(w, 4.55 if layout == 'two_columns' else 4.75))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.93, wspace=0.28, hspace=0.50)
    ax_a, ax_b, ax_c, ax_d = grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]

    # (a) Smooth 201-point variance trajectories from CosChaos coefficients vs exact reference
    lam = 0.05
    for H in (0.55, 0.75, 0.95):
        ref = J(f'reference_lam{lam}_H{H}.json')
        cc = J(f'coschaos_lam{lam}_H{H}.json')
        st = HSTYLE[H]
        ax_a.plot(cc['dense_times'], cc['dense_var'], color=COS, lw=st['lw'])
        ax_a.errorbar(ref['times'], ref['var'], yerr=2.0 * np.array(ref['var_se']),
                      fmt='o', ms=st['ms'] - 0.5, color=REF, fillstyle=st['fillstyle'],
                      lw=0.65, capsize=1.3, zorder=5)
        ax_a.annotate(f'$H={H}$', (cc['dense_times'][-1], cc['dense_var'][-1]),
                      textcoords='offset points', xytext=(4, 0), fontsize=6.4,
                      ha='left', va='center', color=COS)
    ax_a.set_xlim(0.0, 24.5)
    ax_a.set_xticks([0, 5, 10, 15, 20])
    ax_a.set_xlabel('$t$')
    ax_a.set_ylabel('$\\mathrm{Var}\\,y(t)$')
    ax_a.set_title('$\\lambda=0.05$, continuous coefficients ($J=33$, $p=2$)', fontsize=7.6)
    clean(ax_a)
    panel_label(ax_a, 'a', x=-0.18)

    # (b), (c) Work-precision
    wp_panel(ax_b, '0.02_0.95', 'published data, $\\lambda=0.02$, $H=0.95$', 'b')
    ax_b.set_ylabel('max. variance error (%)')
    wp_panel(ax_c, '0.05_0.95', 'stronger noise, $\\lambda=0.05$, $H=0.95$', 'c')
    ax_c.set_ylabel('max. variance error (%)')

    # (d) Terminal probability density y(T) at lam = 0.05, H = 0.95 from the same solve
    law = J('law_lam0.05_H0.95.json')
    edges = np.array(law['hist_edges'])
    mids = 0.5 * (edges[:-1] + edges[1:])
    ax_d.fill_between(mids, 0.0, law['hist_ref'], color='#d9d9d9', step='mid', label='WIS reference ($2^{20}$ paths)')
    ax_d.step(mids, law['hist_ref'], where='mid', color=REF, lw=0.95)
    ax_d.plot(mids, law['hist_cc'], color=COS, lw=1.55, label='CosChaos ($J=17$, $p=2$)')
    ax_d.set_xlabel('$y(T)$ at $T=20$')
    ax_d.set_ylabel('probability density')
    ax_d.set_title('terminal law ($\\lambda=0.05$, $H=0.95$, $2^{20}$ evals)', fontsize=7.6)
    ax_d.set_ylim(0.0, 0.114)
    ax_d.legend(loc='upper right', fontsize=6.2)
    clean(ax_d)
    panel_label(ax_d, 'd', x=-0.18)

    hs, ls = ax_b.get_legend_handles_labels()
    hs.append(Line2D([], [], marker='o', ls='none', color=REF, ms=3.8))
    ls.append('WIS reference, panel (a)')
    fig.legend(hs, ls, loc='lower center', bbox_to_anchor=(0.53, 0.01),
               ncol=3, fontsize=6.6, columnspacing=1.2, frameon=False)
    save(fig, 'fig4_population', layout)


if __name__ == '__main__':
    for lay in ('one_column', 'two_columns'):
        fig4(lay)
    print('fig4 written')
