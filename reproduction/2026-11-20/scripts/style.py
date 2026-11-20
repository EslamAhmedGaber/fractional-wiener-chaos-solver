"""Unified publication-grade visual identity (Okabe-Ito, 300-dpi PNG + vector PDF)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVID = os.path.join(ROOT, 'evidence')
FIGS = {
    'one_column': os.path.join(ROOT, 'source', 'figures', 'one_column'),
    'two_columns': os.path.join(ROOT, 'source', 'figures', 'two_columns'),
}
WIDTH = {'one_column': 6.3, 'two_columns': 7.0}

COS = '#1F4E8C'      # CosChaos: deep blue, solid, circles
COS2 = '#56B4E9'     # secondary CosChaos accent (species 1 / light blue)
REF = '#111111'      # exact / reference: black
FWHE = '#D55E00'     # FWHE: vermilion, dashed, open squares
WIS = '#009E73'      # GfBm EM / WIS-RK4: bluish green, dash-dot, diamonds
EXACT_MC = '#555555' # Exact WIS + antithetic MC: dark grey, dotted, inverted triangles
BP = '#CC79A7'       # discrete Wick: reddish purple, dotted, triangles
GAUSS = '#7F7F7F'    # Gaussian closure: grey dotted

# Three distinct marker fill styles for H = 0.55, 0.75, 0.95 (print & greyscale legible)
HSTYLE = {
    0.55: dict(lw=0.95, ms=3.5, fillstyle='none'),
    0.75: dict(lw=1.35, ms=4.0, fillstyle='left'),
    0.95: dict(lw=1.85, ms=4.5, fillstyle='full'),
}


def setup():
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm',
        'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.3,
        'legend.fontsize': 7.1, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.3, 'lines.markersize': 4.0,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'legend.frameon': False, 'legend.handlelength': 2.0,
        'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, f'({s})', transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom')


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, name, layout):
    os.makedirs(FIGS[layout], exist_ok=True)
    fig.savefig(os.path.join(FIGS[layout], name + '.pdf'))
    fig.savefig(os.path.join(FIGS[layout], name + '.png'), dpi=300)
    plt.close(fig)
