"""CosChaos final manuscript: main-text Figures 2-5 (Figure 1 is made by flagship_figure/make_flagship.py).

Reads the stored evidence in ../evidence (copied unchanged from the validation workspace of
12 November 2026) and the closed-form Gram code evidence/verification_code/hfc.py.
Writes figures/{one,two}/fig2_coordinates, fig3_verification, fig4_oscillator_hurst, fig5_cubic (.pdf, .png)
and figures/figure_checks.json.  Run from any folder:  python scripts/make_figures.py
"""
import csv, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
WK = os.path.dirname(HERE)
FROZEN = WK
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WK, "figures")
sys.path.insert(0, os.path.join(FROZEN, "evidence", "verification_code"))
import hfc  # closed-form Gram matrix (Proposition 1)

J_ = lambda *p: json.load(open(os.path.join(FROZEN, "evidence", *p), encoding="utf-8"))
J12_ = lambda *p: json.load(open(os.path.join(WK, "evidence", "round12", *p), encoding="utf-8"))

W = J_("round3_fwhe", "workprecision.json")
P1 = J_("round3_full", "P1_pathwise.json")
P2 = J_("round3_full", "P2_wick_law.json")
P3 = J_("round3_full", "P3_cubic_law.json")
P4 = J_("round4", "P4_cubic_event.json")
RC = J_("round5", "P5_refined_curve.json")
MR = J_("round5", "P5_mc_reference.json")
TN = J_("data", "theory_noise.json")
EX = J_("data", "v12_examples.json")
OS = J_("data", "v12_oscillator.json")

SW_IS = J12_("single_well_is_matrix.json")
DW_IS = J12_("double_well_stretch.json")
HI = json.load(open(os.path.join(WK, "evidence", "round13", "hurst_inference.json"), encoding="utf-8"))
RAY = json.load(open(os.path.join(WK, "evidence", "round14", "selected_proposals.json"), encoding="utf-8"))
# pooled first run (M = 32768) and replication (M = 65536) of the 18 selected events
POOL = list(csv.DictReader(open(os.path.join(WK, "evidence", "rare_events_pooled", "RARE_EVENTS_ALL_RESULTS.csv"), encoding="utf-8")))


def pool_sel(model, H):
    return sorted([e for e in POOL if e["model"] == model and abs(float(e["H"]) - float(H)) < 1e-9], key=lambda e: -float(e["pooled_p"]))



def ray_sel(model, H):
    """Selected importance-sampling results (smaller corrected norm of the two candidate directions)."""
    return sorted([e for e in RAY if e["model"] == model and abs(e["H"] - float(H)) < 1e-9], key=lambda e: -e["p_ref"])

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(dict(check=label, ok=bool(ok), detail=str(detail)[:160]))
    assert ok, (label, detail)


# ----------------------------------------------------------------------------------------------- style
BASE, SMALL, TICK = 8.8, 7.6, 8.0
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": BASE, "axes.labelsize": BASE, "axes.titlesize": BASE, "legend.fontsize": SMALL,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "lines.linewidth": 1.3, "lines.markersize": 3.4,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.28,
    "legend.handletextpad": 0.5, "legend.columnspacing": 1.1,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True
})
_cm = mpl.colormaps["viridis"]


def HC(H):
    H = float(H)
    return "0.15" if abs(H - 0.5) < 1e-9 else _cm(0.08 + 0.62 * (0.95 - H) / 0.45)


def light(c, f):
    c = np.array(mpl.colors.to_rgb(c))
    return tuple(1 - f * (1 - c))


REF = dict(color="0.62", lw=2.6, solid_capstyle="round", zorder=1)
MC_COL = "#c0392b"
MCS = dict(ls="none", marker="o", ms=3.6, mfc="white", mec=MC_COL, mew=0.9, zorder=4)
GAU = dict(color="0.05", ls=(0, (1, 1.6)), lw=1.3)
OURS = dict(ls="-", marker="o", ms=3.4, lw=1.3)
FWHE = dict(ls=(0, (4, 1.8)), marker="s", ms=3.4, lw=1.0, mfc="white", mew=0.9)
GUIDE = dict(color="0.55", ls=(0, (1, 1.8)), lw=0.9, zorder=0)
W_ONE, W_TWO = 468 / 72.27, 522 / 72.27
HS = ["0.55", "0.75", "0.95"]


def frame(lay, n=3, h=2.30, top=0.0):
    fig, ax = plt.subplots(1, n, figsize=(W_ONE if lay == "one" else W_TWO, h + top), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, wspace=0.08, rect=(0, 0, 1, h / (h + top)))
    return fig, np.atleast_1d(ax)


def letter(ax, s, dx=-0.23):
    ax.text(dx, 1.04, "(%s)" % s, transform=ax.transAxes, fontsize=BASE + 0.4, fontweight="bold", va="bottom", ha="left")


def H_key(fig, Hs, y=0.985, x=0.995):
    hs = [Line2D([], [], color=HC(H), lw=2.4, solid_capstyle="butt") for H in Hs]
    labs = [r"$H=\frac{1}{2}$" if abs(float(H) - 0.5) < 1e-9 else "$H=%s$" % H for H in Hs]
    fig.legend(hs, labs, loc="upper right", bbox_to_anchor=(x, y), ncol=len(Hs), fontsize=SMALL + 0.4, handlelength=1.5, columnspacing=1.3, borderaxespad=0)


def jticks(ax, Js):
    ax.set_xscale("log")
    ax.set_xticks(Js)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: "%g" % v))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def logy(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def save(fig, name, lay):
    d = os.path.join(OUT, lay)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".pdf"))
    fig.savefig(os.path.join(d, name + ".png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- Fig. 1
def wick_surrogate(t, VJ, p, mu=0.2, s=0.5, x0=1.0):
    m = x0 * math.exp(mu * t)
    a = s * math.sqrt(VJ)
    return lambda e: m * sum(a ** n * np.polynomial.hermite_e.hermeval(e, [0] * n + [1]) / math.factorial(n) for n in range(p + 1))


def strip(fig, y0, y1):
    ax = fig.add_axes([0.0, y0, 1.0, y1 - y0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        (r"cosine basis on $[0,T]$", r"$h_0=T^{-1/2},\ \ h_k=\sqrt{2/T}\,\cos(k\pi t/T)$"),
        ("closed-form covariance", r"$G^H_{kl}=\langle h_k,h_l\rangle_H$ by ${}_1F_1$"),
        ("independent coordinates", r"$G^H=LL^{\top},\ \ \xi=L^{-1}Z$"),
        ("one master equation", r"for the coefficients $c_\alpha(t)$"),
        ("explicit polynomial solution", r"$X_{J,p}=\sum_\alpha c_\alpha(t)\,\Phi_\alpha(\xi)$"),
    ]
    n = len(steps)
    gap = 0.022
    x0, x1 = 0.012, 0.988
    w = (x1 - x0 - gap * (n - 1)) / n
    for i, (ttl, body) in enumerate(steps):
        x = x0 + i * (w + gap)
        last = i == n - 1
        ax.add_patch(FancyBboxPatch((x, 0.10), w, 0.80, boxstyle="round,pad=0.0,rounding_size=0.012",
                                    fc=light(HC(0.75), 0.16 if last else 0.07),
                                    ec=HC(0.75) if last else "0.55", lw=0.9 if last else 0.6,
                                    transform=ax.transAxes, clip_on=False))
        ax.text(x + w / 2, 0.66, ttl, ha="center", va="center", fontsize=SMALL + 0.2, fontweight="bold", color="0.1")
        ax.text(x + w / 2, 0.33, body, ha="center", va="center", fontsize=SMALL - 0.1, color="0.1")
        if not last:
            ax.add_patch(FancyArrowPatch((x + w + 0.002, 0.5), (x + w + gap - 0.002, 0.5),
                                         arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.3", transform=ax.transAxes))


def fig2(lay):
    Wf = W_ONE if lay == "one" else W_TWO
    Hf = 2.36
    sx = Wf / W_ONE
    b0, hh = 0.42, 1.70
    fig = plt.figure(figsize=(Wf, Hf))
    box = lambda x, w: fig.add_axes([x * sx / Wf, b0 / Hf, w * sx / Wf, hh / Hf])
    ax = [fig.add_axes([0.40 * sx / Wf, b0 / Hf, hh / Wf, hh / Hf]), box(3.32, 1.21), box(5.13, 1.21)]
    cax = fig.add_axes([(0.40 * sx + hh + 0.07) / Wf, (b0 + 0.12) / Hf, 0.075 / Wf, (hh - 0.24) / Hf])

    G = hfc.gram(0.8, 1.0, "cos", 10)
    C = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
    off = C - np.eye(10)
    check("F2a largest correlation 0.104 at (0,2)", abs(np.max(np.abs(off)) - 0.104) < 5e-4 and abs(off[0, 2]) == np.max(np.abs(off)), np.max(np.abs(off)))
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("0.85")
    im = ax[0].imshow(np.ma.masked_where(np.eye(10, dtype=bool), off), cmap=cmap, vmin=-0.12, vmax=0.12, origin="upper", aspect="auto")
    ax[0].set_xticks(range(0, 10, 3))
    ax[0].set_yticks(range(0, 10, 3))
    ax[0].set_xlabel("mode $l$")
    ax[0].set_ylabel("mode $k$")
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(True)
    cb = fig.colorbar(im, cax=cax, ticks=[-0.1, 0, 0.1])
    cb.ax.tick_params(labelsize=TICK - 0.4, length=2)
    cb.set_label(r"correlation of $Z_k$ and $Z_l$", fontsize=SMALL)
    cb.outline.set_linewidth(0.5)

    k = np.arange(1, 513)
    for H in ["0.55", "0.7", "0.8", "0.95"]:
        dg, asy = np.array(TN["diag"][H]["d"]), np.array(TN["diag"][H]["asym"])
        ax[1].loglog(k, asy, color="0.35", lw=0.8, ls=(0, (3, 1.6)), zorder=3)
        ax[1].loglog(k, dg, color=HC(H), lw=1.5)
        ax[1].text(640, dg[-1], H, color=HC(H), fontsize=SMALL, va="center", ha="left")
        check("F2b ratio at k=512, H=%s in [0.9995, 1.0001]" % H, 0.9995 < dg[-1] / asy[-1] < 1.0001, dg[-1] / asy[-1])
    ax[1].text(640, 2.3, "$H$", fontsize=SMALL, va="center", ha="left")
    ax[1].set_xlim(0.8, 2300)
    ax[1].set_ylim(6e-5, 4)
    ax[1].set_xlabel("mode $k$")
    ax[1].set_ylabel(r"$T^{1-2H}G^H_{kk}$")
    logy(ax[1])
    ax[1].set_xticks([1, 10, 100, 1000])
    ax[1].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax[1].legend([Line2D([], [], color="0.2", lw=1.5), Line2D([], [], color="0.35", lw=0.8, ls=(0, (3, 1.6)))],
                 ["closed form", "large-$k$ law"], loc="lower left")

    # (c) Coupled RMS error of truncated cosine noise
    Js = np.array(TN["Js"])
    for H in ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95"]:
        yv = np.array(TN["noise"][H]["cos"]["classical"])
        ax[2].loglog(Js, yv, color=HC(H), marker="o", ms=2.5, lw=1.15)
        ax[2].loglog(Js, yv[0] * (Js / Js[0]) ** (-float(H)), **GUIDE)
        ax[2].text(295, yv[-1], H, color=HC(H), fontsize=SMALL - 0.3, va="center", ha="left")
    ax[2].text(295, 0.062, "$H$", fontsize=SMALL, va="center", ha="left")
    jticks(ax[2], [5, 17, 65, 257])
    logy(ax[2])
    ax[2].set_xlim(4.2, 540)
    ax[2].set_ylim(2.5e-4, 0.25)
    ax[2].set_xlabel("modes $J$")
    ax[2].set_ylabel("worst-case RMS error of $B_H^J$")
    ax[2].legend([Line2D([], [], color="0.25", marker="o", lw=1.15),
        Line2D([], [], **GUIDE)], ["CosChaos", r"$J^{-H}$ guide"],loc="lower left",fontsize=SMALL)
    for a, s, dx in zip(ax, "abc", (-0.22, -0.36, -0.36)):
        letter(a, s, dx)
    save(fig, "fig2_coordinates", lay)


# ----------------------------------------------------------------------------------------------- Fig. 3
def curve(rec, key):
    ks = sorted(rec[key], key=lambda s: int(s.split(",")[0]))
    return np.array([(rec[key][s]["seconds"], rec[key][s]["max_rel"]) for s in ks])


def fig3(lay):
    fig, ax = frame(lay, h=2.30, top=0.16)
    Js = np.array(EX["ou_interior_J"])
    for H in ["0.5", "0.55", "0.75", "0.95"]:
        yv = np.abs(EX["ou_interior"][H])
        ax[0].loglog(Js, yv, color=HC(H), **OURS)
        ax[0].loglog(Js, yv[0] * (Js / Js[0]) ** (-2 * float(H)), **GUIDE)
    check("F3a OU H=0.75: 0.37 % at J=17", abs(100 * abs(EX["ou_interior"]["0.75"][2]) - 0.37) < 0.005, EX["ou_interior"]["0.75"][2])
    jticks(ax[0], [5, 17, 65])
    ax[0].set_xticks(list(Js))
    logy(ax[0])
    ax[0].set_xlim(4.2, 155)
    ax[0].set_xlabel("modes $J$")
    ax[0].set_ylabel(r"relative error of Var $X(1)$")
    ax[0].legend([Line2D([], [], **GUIDE)], ["slope $-2H$"], loc="lower left")

    Jp = [5, 9, 17, 33, 65, 129, 257, 513]
    for H in HS:
        yv = np.array([P1[H]["ou"][str(j)]["t1"] for j in Jp])
        ax[1].loglog(Jp, yv, color=HC(H), **OURS)
        ax[1].loglog(Jp, yv[0] * (np.array(Jp) / 5.0) ** (-float(H)), **GUIDE)
    check("F3b pathwise error 0.0006 at H=0.95, J=513", abs(P1["0.95"]["ou"]["513"]["t1"] - 0.0006) < 5e-5, P1["0.95"]["ou"]["513"]["t1"])
    jticks(ax[1], [5, 17, 65, 257])
    logy(ax[1])
    ax[1].set_xlim(4.2, 620)
    ax[1].set_xlabel("modes $J$")
    ax[1].set_ylabel(r"pathwise relative RMS error, $t=1$")
    ax[1].legend([Line2D([], [], **GUIDE)], ["slope $-H$"], loc="lower left")

    R = W["OU"]
    for H in HS:
        po, pf = curve(R[H], "ours"), curve(R[H], "fwhe_gl")
        ax[2].loglog(pf[:, 0], pf[:, 1], color=HC(H), **FWHE)
        ax[2].loglog(po[:, 0], po[:, 1], color=HC(H), **OURS)
    r = R["0.95"]
    check("F3c OU H=0.95: J=17 gives 8.0e-4 in 0.06 s; N=12800 gives 1.0e-3 in 0.30 s",
          abs(r["ours"]["17"]["max_rel"] - 8.0e-4) < 5e-6 and abs(r["ours"]["17"]["seconds"] - 0.06) < 0.005 and abs(r["fwhe_gl"]["12800"]["max_rel"] - 1.0e-3) < 5e-5)
    ax[2].set_xlabel("time of one solve (s)")
    ax[2].set_ylabel("max. relative variance error")
    logy(ax[2])
    ax[2].set_ylim(3e-7, 0.6)
    ax[2].xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=8))
    ax[2].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax[2].legend([Line2D([], [], color="0.25", **OURS), Line2D([], [], color="0.25", **FWHE)], ["CosChaos", "FWHE-GL"], loc="lower left")
    for a, s in zip(ax, "abc"):
        letter(a, s)
    H_key(fig, ["0.5", "0.55", "0.75", "0.95"])
    save(fig, "fig3_verification", lay)


# ----------------------------------------------------------------------------------------------- Fig. 4
def fig4(lay):
    fig, ax = frame(lay, h=2.36, top=0.16)
    # (a) density of X(4) at H = 0.75
    H = "0.75"
    R = P3[H]
    edges = np.linspace(-1.5, 2.5, 161)
    xc = 0.5 * (edges[1:] + edges[:-1])
    ax[0].plot(xc, R["hist_9,1"]["4.0"], color=light(HC(H), 0.38), lw=1.1, label="CosChaos, $p=1$")
    ax[0].plot(xc, R["hist_9,4"]["4.0"], color=HC(H), lw=1.3, label="CosChaos, $p=4$")
    ax[0].plot(xc[::3], np.array(R["mc_hist"]["4.0"])[::3], label="Monte Carlo", **MCS)
    ax[0].set_xlim(-1.25, 1.25)
    ax[0].set_ylim(0, 1.72)
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel(r"density of $X(4)$, $H=0.75$")
    ax[0].legend(loc="upper left", handlelength=1.5)

    # (b) tail probability P(X(4) > c): direct evaluation to c = 0.9, importance sampling to c = 1.1
    lev = np.array(P4["meta"]["levels"])
    it = P4["meta"]["times"].index(4.0)
    m = (lev >= 0.5 - 1e-9) & (lev <= 0.9 + 1e-9)
    lr = np.array(RC["levels"])
    cs = np.linspace(0.5, 1.12, 125)
    star_c = [0.7, 0.75, 0.8, 0.85, 0.9]
    is_c = [0.8, 0.9, 1.0, 1.1]
    for Hh in HS:
        r = P4[Hh]
        pc = np.array(r["chaos_9,5"]["p"])[it]
        pm = np.array(r["mc_512"]["p"])[it]
        ci = np.array(r["mc_512"]["ci"])[it]
        mu, vv = r["mc_512"]["mean"][it], r["mc_512"]["var"][it]
        ax[1].semilogy(cs, norm.sf((cs - mu) / math.sqrt(vv)), **{**GAU, "color": HC(Hh), "lw": 1.1})
        ax[1].semilogy(lev[m], pc[m], color=HC(Hh), lw=1.25)
        ax[1].plot(lev[m], pm[m], **{**MCS, "ms": 3.8, "mfc": "none"})
        # Also plot 2e7 MC reference points at c=1.0, 1.1
        ref_ext = [SW_IS[Hh][f"p1_c{c_v}"]["p_ref"] for c_v in [1.0, 1.1]]
        ax[1].plot([1.0, 1.1], ref_ext, **{**MCS, "ms": 3.8, "mfc": "none"})
        # importance-sampling estimates (selected guide, first run and replication pooled) at c = 0.8, 0.9, 1.0, 1.1
        ps = sorted(pool_sel("single", Hh), key=lambda e: float(e["threshold"]))
        check("F5b pooled thresholds %s" % Hh, [float(e["threshold"]) for e in ps] == is_c)
        ax[1].plot(is_c, [float(e["pooled_p"]) for e in ps], ls="none", marker="*", ms=5.2, color=HC(Hh), mec="0.1", mew=0.35, zorder=6)
    i7 = int(np.argmin(abs(lev - 0.7)))
    check("F4b MC P(X(4)>0.7) = 1.10, 1.34, 1.76 %", all(abs(100 * np.array(P4[Hh]["mc_512"]["p"])[it][i7] - v) < 0.006 for Hh, v in zip(HS, (1.10, 1.34, 1.76))))
    ax[1].plot([0.7, 0.7], [4e-7, 0.13], color="0.78", lw=0.6, zorder=0)
    ax[1].set_xlabel("level $c$")
    ax[1].set_ylabel(r"$P(X(4)>c)$")
    ax[1].set_xlim(0.48, 1.14)
    ax[1].set_ylim(4e-7, 0.13)
    logy(ax[1])
    ax[1].legend([
        Line2D([], [], color="0.25", lw=1.25),
        Line2D([], [], ls="none", marker="*", ms=5.2, color="0.25", mec="0.1", mew=0.35),
        Line2D([], [], **{**MCS, "ms": 3.8, "mfc": "none"}),
        Line2D([], [], **{**GAU, "lw": 1.1}),
    ], ["CosChaos, $J=9,p=5$", "importance sampling", "Monte Carlo ($2\\times10^7$)", "Gaussian"],
        loc="lower left", fontsize=SMALL - 0.85, handlelength=1.4)

    # (c) variance-reduction factor against the probability of the event: 12 single-well and 6 double-well events,
    #     first run (M = 32768) and replication (M = 65536) pooled, M = 98304 weighted paths per event
    M_IS = 98304
    pg = np.array([2e-7, 2e-2])
    ax[2].plot(pg, (1 - pg) / (M_IS * 0.02 ** 2 * pg), **GUIDE)
    for Hh in HS:
        e1 = pool_sel("single", Hh)
        e2 = pool_sel("double", Hh)
        ax[2].plot([float(e["pooled_p"]) for e in e1], [float(e["pooled_vrf"]) for e in e1], color=HC(Hh), ls="-", lw=1.2, marker="o", ms=3.8, zorder=4)
        ax[2].plot([float(e["pooled_p"]) for e in e2], [float(e["pooled_vrf"]) for e in e2], color=HC(Hh), ls=(0, (4, 1.8)), lw=1.0, marker="s", ms=3.8,
                   mfc="white", mew=1.0, zorder=5)
    v = [float(e["pooled_vrf"]) for e in POOL]
    check("F5c 18 pooled events", len(POOL) == 18)
    check("F5c pooled factors 74 to 45,597", abs(min(v) - 73.95) < 0.01 and abs(max(v) - 45597.27) < 0.01, (min(v), max(v)))
    check("F5c replication within 3.5 s.e. of the first run", max(float(e["replication_z_vs_original"]) for e in POOL) < 3.5)
    check("F5c factor grows as the single-well event becomes rarer",
          all(np.all(np.diff([float(e["pooled_vrf"]) for e in pool_sel("single", Hh)]) > 0) for Hh in HS))
    ax[2].set_xscale("log")
    logy(ax[2])
    ax[2].set_xlim(1.2e-2, 4e-7)
    ax[2].set_ylim(30, 1.5e5)
    ax[2].xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=8))
    ax[2].xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax[2].set_xlabel("probability of the event")
    ax[2].set_ylabel("variance-reduction factor")
    ax[2].legend([
        Line2D([], [], color="0.25", ls="-", lw=1.2, marker="o", ms=3.8),
        Line2D([], [], color="0.25", ls=(0, (4, 1.8)), lw=1.0, marker="s", ms=3.8, mfc="white", mew=1.0),
        Line2D([], [], **GUIDE),
    ], ["single well, $X(4)>c$", "double well, $X(2)<c$", r"$2\%$ relative s.e."], loc="upper left", fontsize=SMALL - 0.6, handlelength=2.2)
    for a, s in zip(ax, "abc"):
        letter(a, s)
    H_key(fig, HS)
    save(fig, "fig5_cubic", lay)


# ----------------------------------------------------------------------------------------------- Fig. 5
def fig5(lay):
    """Every Hurst index from one coefficient solve: sweep (a), likelihood inference from noisy records (b, c)."""
    fig, ax = frame(lay, h=2.36)
    # (a) linear oscillator: one coefficient solve, nineteen Hurst indices
    Hv = np.array([s["H"] for s in OS["sweep"]])
    vq = np.array([s["varQ"] for s in OS["sweep"]])
    vv = np.array([s["varV"] for s in OS["sweep"]])
    check("F5a Var V(20) 0.1333 -> 0.0323 and ratio 1.00 -> 5.30",
          abs(vv[0] - 0.1333) < 1e-4 and abs(vv[-1] - 0.0323) < 1e-4 and abs(OS["sweep"][-1]["ratio"] - 5.30) < 0.01)
    for yv, ls in [(vq, "-"), (vv, (0, (4, 1.6)))]:
        ax[0].plot(Hv, yv, color="0.45", lw=1.0, ls=ls, zorder=1)
        ax[0].scatter(Hv, yv, c=[HC(h) for h in Hv], s=11, zorder=3, linewidths=0)
    for Hh in ["0.5", "0.7", "0.9"]:
        Rf = OS["ref"]["%s_20.0" % Hh]
        ax[0].plot([float(Hh)] * 2, [Rf[0][0], Rf[1][1]], ls="none", marker="o", ms=6.2, mfc="none", mec="0.15", mew=0.8, zorder=2)
    ax[0].text(0.70, 0.193, r"Var $Q$", fontsize=SMALL + 0.2, ha="center")
    ax[0].text(0.63, 0.094, r"Var $V$", fontsize=SMALL + 0.2, ha="center")
    ax[0].set_xlabel("Hurst index $H$")
    ax[0].set_ylabel("variance at $t=20$")
    ax[0].set_ylim(0, 0.235)
    ax[0].set_xlim(0.48, 0.97)
    ax[0].legend([
        Line2D([], [], ls="none", marker="o", ms=3.4, color=HC(0.75)),
        Line2D([], [], ls="none", marker="o", ms=6.2, mfc="none", mec="0.15", mew=0.8),
    ], ["CosChaos, one solve", "quadrature reference"], loc="lower left", handlelength=1.2)

    # (b) Hurst estimates, data set by data set: CosChaos likelihood (J = 65) against the reference likelihood
    R = HI["results"]
    Ht = ["%.2f" % h for h in HI["setup"]["H_true"]]
    lo, hi = 0.50, 1.0
    ax[1].plot([lo, hi], [lo, hi], color="0.6", lw=0.7, zorder=1)
    for h in Ht:
        x = np.array(R[h]["ref"]["H_hat"])
        y = np.array(R[h]["J65"]["H_hat"])
        ax[1].scatter(x, y, s=5, color=HC(h), alpha=0.75, linewidths=0, zorder=3, label="$H=%s$" % h)
        ax[1].plot([float(h)], [float(h)], marker="+", ms=9, mew=1.1, color="0.05", zorder=4)
    check("F5b three true H, 200 data sets each", all(len(R[h]["ref"]["H_hat"]) == 200 for h in Ht))
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(lo, hi)
    ax[1].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax[1].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax[1].set_xlabel(r"$\hat H$, reference likelihood")
    ax[1].set_ylabel(r"$\hat H$, CosChaos likelihood ($J=65$)")
    ax[1].legend(loc="upper left", handlelength=0.8, markerscale=2.2, title="true value (+)", title_fontsize=SMALL)

    # (c) truncation against sampling: r.m.s. difference to the reference estimate, and the sampling s.d.
    Js = HI["setup"]["J"]
    for h in Ht:
        d = [R[h]["J%d" % J]["rms_diff_to_ref_estimate"] for J in Js]
        ax[2].plot(Js, d, color=HC(h), label="$H=%s$" % h, **OURS)
        ax[2].axhline(R[h]["ref"]["H_sd"], color=HC(h), ls=(0, (1, 1.6)), lw=1.0, zorder=1)
    big = "J%d" % Js[-1]
    check("F5c largest-J estimates within 2e-4 r.m.s. of the reference",
          max(R[h][big]["rms_diff_to_ref_estimate"] for h in Ht) < 2e-4)
    check("F5c sampling s.d. 0.021-0.030", all(0.020 < R[h]["ref"]["H_sd"] < 0.031 for h in Ht))
    jticks(ax[2], Js)
    logy(ax[2])
    ax[2].set_ylim(4e-6, 0.12)
    ax[2].text(Js[-1], 0.036, "sampling s.d. (dotted)", fontsize=SMALL - 0.4, color="0.25", va="bottom", ha="right")
    ax[2].set_xlabel("modes $J$")
    ax[2].set_ylabel(r"r.m.s. $|\hat H_J-\hat H_{\rm ref}|$")
    ax[2].legend(loc="lower left", handlelength=1.8)
    for a, s in zip(ax, "abc"):
        letter(a, s)
    save(fig, "fig4_oscillator_hurst", lay)


if __name__ == "__main__":
    for lay in ("one", "two"):
        fig2(lay)
        fig3(lay)
        fig4(lay)
        fig5(lay)
    os.makedirs(OUT, exist_ok=True)
    json.dump(CHECKS, open(os.path.join(OUT, "figure_checks.json"), "w"), indent=1)
    print("figures written to", OUT, "| checks", len(CHECKS), "all ok" if all(c["ok"] for c in CHECKS) else "FAIL")
