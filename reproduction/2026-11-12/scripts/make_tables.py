"""CosChaos final manuscript: generated LaTeX tables from stored evidence.

Writes source/tables/tab_is_pooled.tex (importance sampling, 18 events, first run and
replication pooled) and source/tables/tab_tanh.tex (selective projection, saturating drift).
Run:  python scripts/make_tables.py
"""
import csv, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
WK = os.path.dirname(HERE)
EV = os.path.join(WK, "evidence")
OUT = os.path.join(WK, "source", "tables")
os.makedirs(OUT, exist_ok=True)

POOL = list(csv.DictReader(open(os.path.join(EV, "rare_events_pooled", "RARE_EVENTS_ALL_RESULTS.csv"), encoding="utf-8")))
SEL = json.load(open(os.path.join(EV, "round14", "selected_proposals.json"), encoding="utf-8"))
DWR = {h: json.load(open(os.path.join(EV, "round14", "double_well_reference_H%s.json" % h), encoding="utf-8")) for h in ("0.55", "0.75", "0.95")}


def sci(x, d=2):
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, d - 1) >= 10:
        m, e = m / 10, e + 1
    return r"$%.*f\times10^{%d}$" % (d - 1, m, e)


def vrf(x):
    if x < 10:
        return "%.1f" % x
    v = int(round(x))
    return "{:,}".format(v).replace(",", "{,}")


rows, zs = [], []
for model in ("single", "double"):
    for H in ("0.55", "0.75", "0.95"):
        sel = [e for e in POOL if e["model"] == model and abs(float(e["H"]) - float(H)) < 1e-9]
        sel.sort(key=lambda e: -float(e["pooled_p"]))
        for e in sel:
            c = float(e["threshold"])
            s = [r for r in SEL if r["model"] == model and abs(r["H"] - float(H)) < 1e-9 and abs(r["c"] - c) < 1e-9][0]
            if model == "single":
                pr, sr, src = s["p_ref"], s["se_ref"], "a"
            else:
                L = DWR[H]["levels"]["%.1f" % c]
                pr, sr, src = L["p"], L["se"], "b"
            p, se = float(e["pooled_p"]), float(e["pooled_se"])
            z = (p - pr) / math.hypot(se, sr)
            zs.append(abs(z))
            ev = ("$X(4)>%.1f$" % c) if model == "single" else ("$X(2)<%s$" % ("0" if c == 0 else "%.1f" % c))
            rows.append((model, H, ev, int(e["guide_degree"]), s["ray_norm_sq"], p, 100 * se / p, pr, abs(z), float(e["pooled_vrf"])))

lines = [r"\begin{tabular}{llcrrrrrr}\toprule",
         r"well & $H$ & event & $p_*$ & $r_{p_*}^2$ & $\widehat P$ & rel.\ s.e.\ (\%) & reference & $|z|$ & VRF\\\midrule"]
lines[0] = r"\begin{tabular}{llcrrrrrrr}\toprule"
last = None
for (model, H, ev, g, r2, p, rse, pr, z, v) in rows:
    if last is not None and model != last:
        lines.append(r"\midrule")
    lines.append("%s & %s & %s & %d & %.1f & %s & %.1f & %s & %.2f & %s\\\\" % (
        model if model != last else "", H, ev, g, r2, sci(p), rse, sci(pr), z, vrf(v)))
    last = model
lines.append(r"\bottomrule\end{tabular}")
open(os.path.join(OUT, "tab_is_pooled.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
assert len(rows) == 18 and max(zs) < 3.0, max(zs)

# selective projection for the saturating drift (stored selective-projection results)
V = json.load(open(os.path.join(EV, "round12_1", "verification.json"), encoding="utf-8"))
open(os.path.join(OUT, "tab_tanh.tex"), "w", encoding="utf-8").write(
    r"""\begin{tabular}{rrrrrrrr}\toprule
$H$ & $J$ & $K$ & $p$ & coefficients & max.\ error (\%) & prepare (s) & evaluate (s)\\\midrule
0.75 & 65 & 1 & 3 & \TanhNA & \TanhErrA & \MomentTimeA & 0.161\\
0.95 & 65 & 2 & 5 & \TanhNB & \TanhErrB & \MomentTimeB & 0.171\\
\bottomrule\end{tabular}
""")
print("tables written; max |z| = %.2f" % max(zs))


# ---------------------------------------------------------------- supplementary rare-event tables
RAYP = json.load(open(os.path.join(EV, "round14", "ray_proposals.json"), encoding="utf-8"))
SOUT = os.path.join(WK, "supplement", "tables")
os.makedirs(SOUT, exist_ok=True)


def ev_label(model, c):
    return ("$X(4)>%.1f$" % c) if model == "single" else ("$X(2)<%s$" % ("0" if c == 0 else "%.1f" % c))


# (S-a) both candidate guides for every event (first run, 32,768 paths)
L = [r"\begin{tabular}{llcrrrrrrc}\toprule",
     r" & & & \multicolumn{3}{c}{degree one} & \multicolumn{3}{c}{degree three} & \\\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
     r"well & $H$ & event & $r_1^2$ & rel.\ s.e.\ (\%) & VRF & $r_3^2$ & rel.\ s.e.\ (\%) & VRF & selected\\\midrule"]
last = None
for model in ("single", "double"):
    for H in ("0.55", "0.75", "0.95"):
        cs = sorted({v["c"] for v in RAYP.values() if v["model"] == model and abs(v["H"] - float(H)) < 1e-9}, reverse=(model == "double"))
        if model == "single":
            cs = sorted(cs)
        for c in cs:
            k1 = "%s_H%s_c%s_p1" % (model, H, c)
            k3 = "%s_H%s_c%s_p3" % (model, H, c)
            a, b = RAYP[k1], RAYP[k3]
            sel = 1 if a["ray_norm_sq"] < b["ray_norm_sq"] else 3
            if last is not None and model != last:
                L.append(r"\midrule")
            L.append("%s & %s & %s & %.1f & %.1f & %s & %.1f & %.1f & %s & %d\\\\" % (
                model if model != last else "", H, ev_label(model, c), a["ray_norm_sq"], a["rel_se_percent"], vrf(a["vrf"]),
                b["ray_norm_sq"], b["rel_se_percent"], vrf(b["vrf"]), sel))
            last = model
L.append(r"\bottomrule\end{tabular}")
open(os.path.join(SOUT, "tab_S_candidates.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# (S-b) first run, replication and pooled estimate for every event
L = [r"\begin{tabular}{llcrrrrrrr}\toprule",
     r" & & & \multicolumn{2}{c}{first run ($M=32{,}768$)} & \multicolumn{2}{c}{replication ($M=65{,}536$)} & \multicolumn{2}{c}{pooled} & \\\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
     r"well & $H$ & event & $\widehat P$ & VRF & $\widehat P$ & VRF & $\widehat P$ & VRF & $|z_{\rm rep}|$\\\midrule"]
last = None
for model in ("single", "double"):
    for H in ("0.55", "0.75", "0.95"):
        sel = [e for e in POOL if e["model"] == model and abs(float(e["H"]) - float(H)) < 1e-9]
        sel.sort(key=lambda e: -float(e["pooled_p"]))
        for e in sel:
            if last is not None and model != last:
                L.append(r"\midrule")
            L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %.2f\\\\" % (
                model if model != last else "", H, ev_label(model, float(e["threshold"])),
                sci(float(e["original_p"]), 3), vrf(float(e["original_vrf"])), sci(float(e["replication_p"]), 3), vrf(float(e["replication_vrf"])),
                sci(float(e["pooled_p"]), 3), vrf(float(e["pooled_vrf"])), float(e["max_replication_z_vs_reference"])))
            last = model
L.append(r"\bottomrule\end{tabular}")
open(os.path.join(SOUT, "tab_S_replication.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# (S-c) paired 512/1024-step check
PT = json.load(open(os.path.join(EV, "rare_events_pooled", "paired_timestep.json"), encoding="utf-8"))
L = [r"\begin{tabular}{llcrrrr}\toprule",
     r"well & $H$ & event & $p_*$ & 512 steps & 1024 steps & change (\%)\\\midrule"]
for e in PT:
    L.append("%s & %.2f & %s & %d & %s & %s & %.3f\\\\" % (e["model"], e["H"], ev_label(e["model"], e["c"]), e["degree"],
                                                       sci(e["coarse"], 4), sci(e["fine"], 4), e["relative_difference_percent"]))
L.append(r"\bottomrule\end{tabular}")
open(os.path.join(SOUT, "tab_S_timestep.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
assert max(e["relative_difference_percent"] for e in PT) < 0.06
print("supplementary rare-event tables written")
