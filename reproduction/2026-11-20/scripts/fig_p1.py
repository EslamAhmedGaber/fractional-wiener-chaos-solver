"""Figures 2 and 3 (Problem 1, fractional Malthusian growth) in both layouts."""
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from style import setup, save, panel_label, clean, EVID, WIDTH, COS, REF, FWHE, BP, HSTYLE

P1 = os.path.join(EVID, 'P1_malthusian')
R = json.load(open(os.path.join(P1, 'p1_results.json')))
MU, SIG = 0.2, 0.5
TIMES = np.array(R['params']['times'])


def fwhe_err(H, p, t):
    q = (SIG ** 2) * (t ** (2.0 * H))
    return 100.0 * (sum((q ** r) / math.factorial(r) for r in range(1, p + 1)) / np.expm1(q) - 1.0)


def fig2(layout):
    """(a) terminal identity with FWHE; (b)-(d) first-order variance work-precision."""
    setup()
    w = WIDTH[layout]
    if layout == 'two_columns':
        fig, axs = plt.subplots(1, 4, figsize=(w, 2.60), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.29, top=0.88, wspace=0.52)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(w, 4.95))
        axs = axs.ravel()
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.94, wspace=0.30, hspace=0.48)

    ax = axs[0]
    H = 0.75
    rows = {(r['J'], r['p']): r for r in R['identity'] if r['method'] == 'CosChaos' and abs(r['H'] - H) < 1e-9}
    ax.axhline(0.0, color=REF, lw=0.6)
    tt = np.linspace(0.25, 2.0, 200)
    ax.plot(tt, fwhe_err(H, 5, tt), color=FWHE, ls='--', lw=1.35, label='FWHE, order 5')
    for J, lw, al in [(5, 1.0, 0.55), (17, 1.6, 1.0)]:
        ax.plot(TIMES, rows[(J, 5)]['err_by_t'], color=COS, lw=lw, alpha=al, marker='o', ms=3.2, label=f'CosChaos, $J={J}$')
    ax.annotate('identical at $t=T$', xy=(2.0, rows[(5, 5)]['err_by_t'][-1]), xytext=(1.02, -2.6),
                fontsize=6.8, arrowprops=dict(arrowstyle='->', lw=0.65, color='#444444'))
    ax.set_xlabel('$t$')
    ax.set_ylabel('variance error (%)')
    ax.set_xlim(0.2, 2.08)
    ax.legend(loc='lower right', fontsize=6.4)
    clean(ax)
    panel_label(ax, 'a', x=-0.20)
    ax.set_title('$H=0.75$, from coefficients', fontsize=7.6)

    for k, H in enumerate((0.55, 0.75, 0.95)):
        ax = axs[k + 1]
        fg = [r for r in R['fwhegl'] if abs(r['H'] - H) < 1e-9 and r['method'] == 'FWHE-GL']
        cc = [r for r in R['fwhegl'] if abs(r['H'] - H) < 1e-9 and r['method'] == 'CosChaos']
        ax.loglog([r['seconds'] for r in fg], [r['max_abs_err'] for r in fg],
                  color=FWHE, ls='--', marker='s', mfc='white', ms=3.4, lw=1.15, label='FWHE-GL ($N$ steps)')
        ax.loglog([r['seconds'] for r in cc], [r['max_abs_err'] for r in cc],
                  color=COS, marker='o', ms=3.4, lw=1.45, label='CosChaos ($J$ modes)')
        for r in cc:
            if r['J'] in (5, 17, 65):
                txt = f"$J={r['J']}$" if r['J'] == 5 else f"{r['J']}"
                xy = (r['seconds'], r['max_abs_err'])
                if H == 0.95 and r['J'] == 5:
                    ax.annotate(txt, xy, textcoords='offset points', xytext=(6, 10), ha='left', va='bottom',
                                fontsize=6.0, color=COS,
                                arrowprops=dict(arrowstyle='-', lw=0.5, color=COS, shrinkA=0, shrinkB=2))
                elif H == 0.95 and r['J'] == 17:
                    ax.annotate(txt, xy, textcoords='offset points', xytext=(-5, -2), ha='right', va='top',
                                fontsize=6.0, color=COS)
                else:
                    ax.annotate(txt, xy, textcoords='offset points', xytext=(5, 0), ha='left', va='center',
                                fontsize=6.0, color=COS)
        fgs = sorted(fg, key=lambda r: r['N'])
        ax.annotate(f"$N={fgs[0]['N']}$", (fgs[0]['seconds'], fgs[0]['max_abs_err']),
                    textcoords='offset points', xytext=(3, 4), ha='left', va='bottom', fontsize=6.0, color=FWHE)
        if H == 0.95:
            last = dict(xytext=(3, 3), ha='left', va='bottom')
        elif H == 0.75:
            last = dict(xytext=(-8, -3), ha='right', va='top')
        else:
            last = dict(xytext=(2, -5), ha='right', va='top')
        ax.annotate(f"{fgs[-1]['N']:,}", (fgs[-1]['seconds'], fgs[-1]['max_abs_err']),
                    textcoords='offset points', fontsize=6.0, color=FWHE, **last)
        xs = [r['seconds'] for r in fg + cc]
        ax.set_xlim(min(xs) / 2.2, max(xs) * (10.0 if H == 0.95 else 4.0))
        ax.set_title(f'$H={H}$', fontsize=7.6)
        ax.set_xlabel('time (s)')
        if k == 0 or (layout == 'one_column' and k == 1):
            ax.set_ylabel('max. variance error (%)')
        ax.set_ylim(3e-4, 60)
        clean(ax)
        panel_label(ax, 'bcd'[k], x=-0.22)

    hs, ls = axs[1].get_legend_handles_labels()
    fig.legend(hs, ls, loc='lower center',
               bbox_to_anchor=(0.66 if layout == 'two_columns' else 0.54, 0.01),
               ncol=2, fontsize=6.8, frameon=False, columnspacing=1.8)
    save(fig, 'fig2_fwhe', layout)


def fig3(layout):
    """(a) discrete Wick vs CosChaos with distinct H fillstyles; (b) single H-free solve."""
    setup()
    w = WIDTH[layout]
    fig, axs = plt.subplots(1, 2, figsize=(w, 2.85 if layout == 'two_columns' else 3.05),
                            gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.33, top=0.91, wspace=0.30)

    ax = axs[0]
    W = R['discrete_wick']['rows']
    hd = {}
    for H in (0.55, 0.75, 0.95):
        st = HSTYLE[H]
        bp = sorted([r for r in W if abs(r['H'] - H) < 1e-9 and r['method'].startswith('BP')], key=lambda r: r['n'])
        cc = sorted([r for r in W if abs(r['H'] - H) < 1e-9 and r['method'].startswith('CosChaos')], key=lambda r: r['J'])
        hd['BP', H], = ax.loglog(
            [r['n'] for r in bp], [abs(r['max_var_err']) for r in bp],
            color=BP, ls=':', marker='^', lw=st['lw'], ms=st['ms'],
            fillstyle=st['fillstyle'], label=f'discrete Wick, $H={H}$',
        )
        hd['CC', H], = ax.loglog(
            [r['J'] for r in cc], [abs(r['max_var_err']) for r in cc],
            color=COS, marker='o', lw=st['lw'], ms=st['ms'],
            fillstyle=st['fillstyle'], label=f'CosChaos, $H={H}$',
        )
    ax.set_xlabel('modes $J$ (CosChaos) or cells $n$ (discrete Wick)')
    ax.set_ylabel('max. variance error (%)')
    order = [hd[m, H] for H in (0.55, 0.75, 0.95) for m in ('CC', 'BP')]
    labels = [h.get_label() for h in order]
    if layout == 'two_columns':
        ax.legend(order, labels, ncol=3, fontsize=6.4, loc='upper center',
                  bbox_to_anchor=(0.5, -0.25), columnspacing=1.0, frameon=False)
    else:
        fig.legend(order, labels, ncol=3, fontsize=6.4, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), columnspacing=1.0, frameon=False)
    ax.set_ylim(2e-3, 200)
    clean(ax)
    panel_label(ax, 'a', x=-0.13)

    ax = axs[1]
    sw = R['h_sweep']
    Hs = [r['H'] for r in sw['rows']]
    errs = [r['max_rel_err'] for r in sw['rows']]
    ax.semilogy(Hs, errs, color=COS, marker='o', ms=3.8, lw=1.5)
    ax.axhline(1.0, color='#888888', lw=0.65, ls='--')
    ax.text(0.553, 1.15, '1%', fontsize=6.6, color='#555555')
    ax.set_xlabel('Hurst index $H$')
    ax.set_ylabel('max. variance error (%)')
    ax.set_xticks([0.55, 0.65, 0.75, 0.85, 0.95])
    ax.set_title('one $H$-free solve ($J=65$, $p=7$)', fontsize=7.6)
    clean(ax)
    panel_label(ax, 'b', x=-0.20)
    save(fig, 'fig3_memory', layout)


if __name__ == '__main__':
    for lay in ('one_column', 'two_columns'):
        fig2(lay)
        fig3(lay)
    print('fig2, fig3 written')
