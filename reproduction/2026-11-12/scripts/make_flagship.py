"""Flagship figure: CosChaos, from fractional noise correlations to a reusable Wiener chaos solver.

Three bands. Top: the obstacle (computed correlation matrices of the cosine coordinates) and the assembled-once
construction. Middle: the solver (chaos basis, master equation, explicit polynomial) and the FWHE lane, joined by
the exact coefficient correspondence. Bottom: five outputs of one computed polynomial, each with its condition.
Solid frames: identities and proved estimates. Dashed frame: numerical demonstrations.

Data (all stored evidence, nothing recomputed except the Gram matrix from its closed form):
  correlations   hfc.gram (closed form, Proposition on the Gram matrix)
  (a) oscillator sweep          evidence/data/v12_oscillator.json  ['sweep'], ['ref']
  (b) Wick growth paths         evidence/round3_full/P1_pathwise.json  ['0.75']['paths']
  (c) Wick growth law at t = 2  evidence/round3_full/P2_wick_law.json  ['0.75']['2.0']
  (d) Hurst estimates           evidence/round13/hurst_inference.json  results[H]['ref'|'J65']['H_hat']
  (e) importance sampling       evidence/rare_events_pooled/RARE_EVENTS_ALL_RESULTS.csv  (pooled_p, pooled_vrf)
Run:  python make_flagship.py   (writes flagship.pdf, flagship.tif (600 dpi), flagship_preview.png, checks.json)
"""
import csv, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
WK = os.path.dirname(HERE)  # archive root (scripts/ and evidence/ side by side)
EV = os.path.join(WK, "evidence")
sys.path.insert(0, os.path.join(EV, "verification_code"))
import hfc  # closed-form Gram matrix

J_ = lambda *p: json.load(open(os.path.join(EV, *p), encoding="utf-8"))
OS = J_("data", "v12_oscillator.json")
P1 = J_("round3_full", "P1_pathwise.json")
P2 = J_("round3_full", "P2_wick_law.json")
HI = J_("round13", "hurst_inference.json")
RE = list(csv.DictReader(open(os.path.join(EV, "rare_events_pooled", "RARE_EVENTS_ALL_RESULTS.csv"), encoding="utf-8")))

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(dict(check=label, ok=bool(ok), detail=str(detail)[:160]))
    assert ok, (label, detail)


# ------------------------------------------------------------------ style (same language as the paper figures)
BASE, SMALL, TINY = 7.6, 6.9, 6.6
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": BASE, "axes.labelsize": SMALL, "xtick.labelsize": TINY, "ytick.labelsize": TINY,
    "legend.fontsize": TINY, "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5, "axes.labelpad": 1.5, "legend.frameon": False, "legend.handlelength": 1.3,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.2, "legend.handletextpad": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
_cm = mpl.colormaps["viridis"]


def HC(H):
    H = float(H)
    return "0.15" if abs(H - 0.5) < 1e-9 else _cm(0.08 + 0.62 * (0.95 - H) / 0.45)


REF = dict(color="0.62", lw=2.2, solid_capstyle="round", zorder=1)
GAU = dict(color="0.05", ls=(0, (1, 1.5)), lw=1.1)
INK = "0.12"
CC_FILL = "#eaf2f8"      # CosChaos boxes (identity colour of the earlier construction strip)
CC_EDGE = "#2c4a63"
FW_EDGE = "0.45"
W_IN = 468 / 72.27          # one-column text width; the two-column figure* is 522 pt (scale 1.12)
H_IN = 5.05

fig = plt.figure(figsize=(W_IN, H_IN))
# overlay axes for boxes, arrows and text, in figure fraction
ov = fig.add_axes([0, 0, 1, 1]); ov.set_axis_off(); ov.set_xlim(0, 1); ov.set_ylim(0, 1)


def box(x, y, w, h, lines, solid=True, fill=CC_FILL, edge=CC_EDGE, fs=SMALL, head=None, lw=0.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008", fc=fill, ec=edge, lw=lw,
                       ls="-" if solid else (0, (3, 2)), zorder=2)
    ov.add_patch(p)
    yy = y + h - 0.012
    if head:
        ov.text(x + 0.008, yy, head, fontsize=fs, fontweight="bold", va="top", ha="left", color=INK, zorder=3)
        yy -= 0.034
    n = len(lines)
    if n:
        step = (yy - y - 0.008) / n
        for i, s in enumerate(lines):
            ov.text(x + 0.008, yy - (i + 0.5) * step, s, fontsize=fs, va="center", ha="left", color=INK, zorder=3)


def arrow(x0, y0, x1, y1, color="0.35", lw=0.9):
    ov.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=7), zorder=4)


def tag(x, y, s, ha="left"):
    ov.text(x, y, s, fontsize=TINY, style="italic", color="0.35", ha=ha, va="bottom")


# ------------------------------------------------------------------ band 1: obstacle
yA0, yA1 = 0.745, 0.990
ov.text(0.006, yA1, "The obstacle", fontsize=BASE, fontweight="bold", va="top", ha="left", color=INK)
Jc = 10
for i, Hh in enumerate([0.5, 0.8]):
    G = np.eye(Jc) if Hh == 0.5 else hfc.gram(Hh, 1.0, "cos", Jc)
    d = np.sqrt(np.diag(G)); Cr = G / np.outer(d, d); Cm = Cr.copy(); np.fill_diagonal(Cm, np.nan)
    if Hh == 0.8:
        check("odd-parity entries vanish (k+l odd)", max(abs(Cr[k, l]) for k in range(Jc) for l in range(Jc) if (k + l) % 2) < 1e-12)
        check("H=0.8 correlations nonzero for k+l even", max(abs(Cr[k, l]) for k in range(Jc) for l in range(Jc) if (k + l) % 2 == 0 and k != l) > 0.05)
    ax = fig.add_axes([0.022 + i * 0.112, yA0 + 0.062, 0.092, 0.092 * W_IN / H_IN])
    cmap = mpl.colormaps["RdBu_r"].copy(); cmap.set_bad("0.82")
    im = ax.imshow(Cm, cmap=cmap, vmin=-0.15, vmax=0.15, interpolation="nearest")
    ax.set_xticks([0, 9]); ax.set_yticks([0, 9]); ax.tick_params(length=1.5, labelsize=TINY - 0.2, pad=1)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
    ax.set_title(r"$H=\frac{1}{2}$" if Hh == 0.5 else r"$H=0.8$", fontsize=SMALL, pad=2)
cax = fig.add_axes([0.243, yA0 + 0.068, 0.006, 0.11])
cb = fig.colorbar(im, cax=cax, ticks=[-0.1, 0, 0.1]); cb.ax.tick_params(labelsize=TINY - 0.4, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
ov.text(0.135, yA0 - 0.014, "corr$(Z_k,Z_l)$, $Z_k=\\int_0^T h_k\\,dB_H$, $k,l<10$:\nclassical coordinates are correlated for $H>\\frac{1}{2}$", fontsize=TINY, va="bottom", ha="center", color=INK, linespacing=1.1)

# ------------------------------------------------------------------ band 1: construction, assembled once
ov.text(0.300, yA1, "The construction", fontsize=BASE, fontweight="bold", va="top", ha="left", color=INK)
tag(0.995, yA1 - 0.028, "assembled once for $(H,T,J)$", ha="right")
bw, bh, by = 0.222, 0.150, yA0 + 0.030
bx = [0.300, 0.536, 0.772]
box(bx[0], by, bw, bh, [r"$h_0=T^{-1/2}$", r"$h_k=\sqrt{2/T}\,\cos(k\pi t/T)$", r"constant mode kept: $\xi_0=T^{-H}B_H(T)$"], head=r"1  Cosine basis on $[0,T]$")
box(bx[1], by, bw, bh, [r"$G^H_{kl}=\langle h_k,h_l\rangle_H$ from", r"$M_H(k\pi)={}_1F_1(2H-1;2H;ik\pi)$", r"$J$ evaluations; zero if $k+l$ odd"], head=r"2  Gram matrix, closed form")
box(bx[2], by, bw, bh, [r"$G^H=LL^\top,\ \ \xi=L^{-1}Z\sim N(0,I_J)$", r"nested in $J$; near-best within $\sqrt{2}$", r"squared noise error of order $J^{-2H}$"], head=r"3  Independent coordinates")
for i in range(2):
    arrow(bx[i] + bw + 0.002, by + bh / 2, bx[i + 1] - 0.004, by + bh / 2)
arrow(0.252, by + bh / 2, bx[0] - 0.004, by + bh / 2)

# ------------------------------------------------------------------ band 2: the solver and the FWHE lane
yB0, yB1 = 0.455, 0.700
ov.text(0.006, yB1, "The solver", fontsize=BASE, fontweight="bold", va="top", ha="left", color=INK)
tag(0.995, yB1 - 0.028, "one deterministic solve per equation", ha="right")
cw, ch, cy = 0.300, 0.125, yB0 + 0.072
cx = [0.006, 0.344, 0.690]
box(cx[0], cy, cw, ch, [r"$\Phi_\alpha(\xi)=\prod_j \mathrm{He}_{\alpha_j}(\xi_j)/\sqrt{\alpha_j!}$", r"$|\alpha|\leq p$: $\binom{J+p}{p}$ coefficients per state"], head=r"4  Chaos basis in $\xi$")
box(cx[1], cy, cw, ch, [r"$\dot c_\alpha=\mathcal{P}_\alpha[F]+\sum_j\sqrt{\alpha_j}\,\ell_j\,\mathcal{P}_{\alpha-e_j}[D]$", r"$\ell=L^\top h$; any admissible $F$, Wick $D$, degree $p$"], head=r"5  One master equation")
box(cx[2], cy, 0.304, ch, [r"$X_{J,p}(t,\xi)=\sum_{|\alpha|\leq p}c_\alpha(t)\,\Phi_\alpha(\xi)$", r"$\mathbb{E}X_{J,p}=c_0,\ \ \mathrm{Var}\,X_{J,p}=\sum_{\alpha\neq0}c_\alpha^2$"], head=r"6  Explicit polynomial")
arrow(cx[0] + cw + 0.002, cy + ch / 2, cx[1] - 0.004, cy + ch / 2)
arrow(cx[1] + cw + 0.002, cy + ch / 2, cx[2] - 0.004, cy + ch / 2)
# connector from box 3 down to box 4 (elbow)
xm0, xm1 = bx[2] + bw / 2, cx[0] + 0.15
ymid = 0.5 * (by + cy + ch)
ov.plot([xm0, xm0, xm1], [by - 0.004, ymid, ymid], color="0.35", lw=0.9, zorder=4, solid_capstyle="butt")
arrow(xm1, ymid, xm1, cy + ch + 0.004)
# FWHE lane: same random variable, different representation
fy, fh = yB0 - 0.002, 0.052
p = FancyBboxPatch((0.006, fy), 0.988, fh, boxstyle="round,pad=0.003,rounding_size=0.006", fc="white", ec=FW_EDGE,
                   lw=0.6, ls="-", zorder=2)
ov.add_patch(p)
ov.text(0.014, fy + fh / 2, r"FWHE:  $X=X^{(0)}+\sum_{n\geq1}I_n(f_n)$, kernels $f_n$ in $n$ time variables, one kernel equation per order, moments by multiple integrals",
        fontsize=TINY, va="center", ha="left", color="0.25", zorder=3)
ov.text(0.988, fy + fh / 2, r"same $X$:  $c^*_\alpha=\frac{n!}{\sqrt{\alpha!}}\langle f_n,U_\alpha\rangle$", fontsize=TINY, va="center", ha="right", color="0.25", zorder=3)
ov.plot([cx[2] + 0.152] * 2, [fy + fh + 0.002, cy - 0.004], color=FW_EDGE, lw=0.7, ls="-", zorder=2)
ov.text(cx[2] + 0.160, 0.5 * (fy + fh + cy), "$=$", fontsize=SMALL, va="center", ha="left", color="0.25")

# ------------------------------------------------------------------ band 3: what one computed polynomial gives
yC0, yC1 = 0.016, 0.425
fr = Rectangle((0.004, yC0 - 0.012), 0.992, yC1 - yC0 + 0.012, fc="none", ec="0.45", lw=0.6, ls=(0, (3, 2)), zorder=1)
ov.add_patch(fr)
ov.text(0.010, yC1 - 0.006, "What one computed polynomial gives", fontsize=BASE, fontweight="bold", va="top", ha="left", color=INK)
tag(0.992, yC1 - 0.034, "evaluated many times, no new solve", ha="right")
pw, ph, py = 0.142, 0.190, yC0 + 0.120
px = [0.050, 0.248, 0.446, 0.644, 0.842]
axs = [fig.add_axes([x, py, pw, ph]) for x in px]
conds = ["affine, $H$-independent data:\none solve, 19 values of $H$", "Wick growth, $H=0.75$:\npath by path", "Wick growth, $t=2$: law;\nGaussian fit misses tails",
         "linear, additive, $H$-independent:\nGaussian likelihood in $H$", "guide for sampling;\nexact weights, discrete model"]
heads = [r"(a) moments", "(b) solution on a path", "(c) non-Gaussian law", "(d) Hurst inference", "(e) rare events"]
for a, x, h_, c_ in zip(axs, px, heads, conds):
    a.text(-0.05, 1.07, h_, transform=a.transAxes, fontsize=SMALL, fontweight="bold", va="bottom", ha="left")
    ov.text(x + pw / 2 - 0.01, yC0 + 0.000, c_, fontsize=TINY, va="bottom", ha="center", color="0.25", linespacing=1.05)

# (a) oscillator: one coefficient solve, nineteen Hurst indices (variance of position and velocity at t = 20)
a = axs[0]
Hv = np.array([s["H"] for s in OS["sweep"]]); vq = np.array([s["varQ"] for s in OS["sweep"]]); vv = np.array([s["varV"] for s in OS["sweep"]])
check("19 Hurst indices from one solve", len(Hv) == 19)
check("Var V(20): 0.1333 at H=1/2, 0.0323 at H=0.95", abs(vv[0] - 0.1333) < 1e-4 and abs(vv[-1] - 0.0323) < 1e-4)
for yv, ls in [(vq, "-"), (vv, (0, (4, 1.6)))]:
    a.plot(Hv, yv, color="0.5", lw=0.8, ls=ls, zorder=1)
    a.scatter(Hv, yv, c=[HC(h) for h in Hv], s=5, zorder=3, linewidths=0)
for Hh in ["0.5", "0.7", "0.9"]:
    Rf = OS["ref"]["%s_20.0" % Hh]
    a.plot([float(Hh)] * 2, [Rf[0][0], Rf[1][1]], ls="none", marker="o", ms=4.2, mfc="none", mec="0.15", mew=0.6, zorder=2)
a.text(0.66, 0.19, r"Var $Q$", fontsize=TINY); a.text(0.56, 0.072, r"Var $V$", fontsize=TINY)
a.set_xlim(0.47, 0.98); a.set_ylim(0, 0.225); a.set_xticks([0.5, 0.75, 1.0]); a.set_xticklabels(["0.5", "0.75", "1"])
a.set_yticks([0, 0.1, 0.2]); a.set_xlabel("$H$"); a.set_ylabel("variance, $t=20$")

# (b) Wick growth: exact and CosChaos (p = 3) on three common fBm paths
a = axs[1]; col = HC("0.75")
Pp = P1["0.75"]["paths"]; td = np.r_[0, np.array(Pp["t"])]
for k in range(3):
    a.plot(td, np.r_[1, np.array(Pp["exact"])[k]], **REF)
    a.plot(td, np.r_[1, np.array(Pp["p3"])[k]], color=col, lw=0.9, zorder=3)
a.set_xlim(0, 2.02); a.set_ylim(0.4, 5.9); a.set_xticks([0, 1, 2]); a.set_xlabel("$t$"); a.set_ylabel("$X(t)$")
a.legend([Line2D([], [], **REF), Line2D([], [], color=col, lw=0.9)], ["exact", "$p=3$"], loc="upper left", handlelength=1.1)

# (c) Wick growth law at t = 2: exact, CosChaos p = 5, Gaussian with the exact mean and variance
a = axs[2]
d = P2["0.75"]["2.0"]; xs = np.array(d["density_x"]); xc = 0.5 * (xs[1:] + xs[:-1]); ex = d["exact"]
mu_, s_, x0_ = 0.2, 0.5, 1.0
e = np.linspace(-8, 8, 400001); aa = s_ * math.sqrt(d["VJ_65"])
y = x0_ * math.exp(mu_ * 2.0) * sum(aa ** n * np.polynomial.hermite_e.hermeval(e, [0] * n + [1]) / math.factorial(n) for n in range(6))
hist, edges = np.histogram(y, bins=xs, weights=norm.pdf(e) * (e[1] - e[0]))
a.plot(xs, d["density_exact"], **REF)
a.plot(xc, hist / np.diff(edges), color=col, lw=1.0, zorder=3)
sd = math.sqrt(ex["var"]); xg = np.linspace(-3.2, 8.4, 400)
a.plot(xg, norm.pdf(xg, ex["mean"], sd), **GAU)
gq = norm.ppf(0.99, ex["mean"], sd)
check("Gaussian 99% quantile 5.01, exact 7.41", abs(gq - 5.01) < 0.005 and abs(ex["q99"] - 7.41) < 0.005, (gq, ex["q99"]))
for q, c, ls, lw in [(ex["q99"], "0.5", "-", 1.6), (gq, "0.05", (0, (1, 1.2)), 1.1)]:
    a.plot([q, q], [0, 0.11], color=c, ls=ls, lw=lw, solid_capstyle="butt")
a.text(6.3, 0.15, "$q_{0.99}$", fontsize=TINY, ha="center")
a.set_xlim(-3.0, 8.3); a.set_ylim(0, 0.84); a.set_xticks([0, 4, 8]); a.set_yticks([0, 0.4, 0.8]); a.set_xlabel("$x$"); a.set_ylabel("density of $X(2)$")
a.legend([Line2D([], [], **REF), Line2D([], [], color=col, lw=1.0), Line2D([], [], **GAU)], ["exact", "$p=5$", "Gaussian"], loc="upper right", handlelength=1.1)

# (d) Hurst estimates: CosChaos likelihood (J = 65) against the reference likelihood, 200 data sets per H
a = axs[3]
Rr = HI["results"]; Ht = ["%.2f" % h for h in HI["setup"]["H_true"]]
a.plot([0.45, 1.0], [0.45, 1.0], color="0.6", lw=0.6, zorder=1)
for h in Ht:
    xh = np.array(Rr[h]["ref"]["H_hat"]); yh = np.array(Rr[h]["J65"]["H_hat"])
    a.scatter(xh, yh, s=2.2, color=HC(h), alpha=0.75, linewidths=0, zorder=3)
    a.plot([float(h)], [float(h)], marker="+", ms=6, mew=0.9, color="0.05", zorder=4)
check("200 data sets per true H", all(len(Rr[h]["ref"]["H_hat"]) == 200 for h in Ht))
check("largest-J estimates within 2e-4 r.m.s. of the reference", max(Rr[h]["J129"]["rms_diff_to_ref_estimate"] for h in Ht) < 2e-4)
a.set_xlim(0.5, 1.0); a.set_ylim(0.5, 1.0); a.set_xticks([0.5, 0.75, 1.0]); a.set_yticks([0.5, 0.75, 1.0])
a.set_xticklabels(["0.5", "0.75", "1"]); a.set_yticklabels(["0.5", "0.75", "1"])
a.set_xlabel(r"$\hat H$, reference"); a.set_ylabel(r"$\hat H$, CosChaos")

# (e) variance reduction against crude Monte Carlo at equal sample size, pooled over both independent runs
a = axs[4]
for Hh in ["0.55", "0.75", "0.95"]:
    for model, mk, ls in [("single", "o", "-"), ("double", "s", (0, (3, 1.5)))]:
        rr = sorted([r for r in RE if r["model"] == model and abs(float(r["H"]) - float(Hh)) < 1e-9], key=lambda r: -float(r["pooled_p"]))
        a.plot([float(r["pooled_p"]) for r in rr], [float(r["pooled_vrf"]) for r in rr], color=HC(Hh), ls=ls, lw=0.9, marker=mk, ms=2.8,
               mfc=HC(Hh) if model == "single" else "white", mew=0.7, zorder=3)
vr = [float(r["pooled_vrf"]) for r in RE]
check("18 events, pooled factors 74 to 45,597", len(RE) == 18 and round(min(vr)) == 74 and round(max(vr)) == 45597, (min(vr), max(vr)))
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e-2, 5e-7); a.set_ylim(30, 1.5e5)
a.set_xticks([1e-2, 1e-4, 1e-6]); a.xaxis.set_minor_locator(mpl.ticker.NullLocator())
a.set_yticks([1e2, 1e3, 1e4, 1e5]); a.yaxis.set_minor_locator(mpl.ticker.NullLocator())
a.set_xlabel("event probability"); a.set_ylabel("variance reduction")
a.legend([Line2D([], [], color="0.3", marker="o", ms=2.8, lw=0.9), Line2D([], [], color="0.3", marker="s", ms=2.8, mfc="white", ls=(0, (3, 1.5)), lw=0.9)],
         ["single well", "double well"], loc="upper left", handlelength=1.6)

# Hurst key shared by panels (a), (d), (e)
hs = [Line2D([], [], color=HC(H), lw=2.0) for H in ["0.5", "0.55", "0.6", "0.75", "0.9", "0.95"]]
fig.legend(hs, [r"$H=\frac{1}{2}$", "$0.55$", "$0.6$", "$0.75$", "$0.9$", "$0.95$"], loc="center", bbox_to_anchor=(0.60, yC1 - 0.018),
           ncol=6, fontsize=TINY, handlelength=1.2, columnspacing=0.9, borderaxespad=0, frameon=False)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WK, "figures", "flagship")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "flagship.pdf"))
fig.savefig(os.path.join(out, "flagship.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(out, "flagship_preview.png"), dpi=200)
json.dump(CHECKS, open(os.path.join(out, "checks.json"), "w"), indent=1)
print("ok", len(CHECKS))
