"""targeted rare-event experiments (cubic single and double well of the manuscript).

Reference simulation and proposal calibration:

(A) reference:  the double-well reference was a 131,072-path crude Monte Carlo run with as few as 7 hits.
    `python rare_events.py ref <iH>` runs 2^23 crude Monte Carlo paths for H = (0.55, 0.75, 0.95)[iH] on the
    same time-discrete target (exact fGn increments, Heun, 512 steps) with a fresh seed.

(B) proposal rule:  at H = 0.95 the degree-three design points for c >= 1 lie far out and the dedicated
    proposals degenerate; the reported rows reused another threshold's shift.  We test one rule that is fixed
    before any sampling and needs no such choice:
        direction  d = x*/|x*|  from the polynomial design point (degree 3, or degree 1 for comparison);
        magnitude  the smallest s >= 0 for which the time-discrete equation driven by the shift alone,
                   dB = C_d (s d), reaches the threshold:  X^Delta(T; s d) = c.
    This is the first-order reliability design point of the full discrete model restricted to the ray chosen
    by the polynomial; it costs a scalar root search with single-path solves.
    `python rare_events.py ray pilot|full`

Outputs: evidence/round14/double_well_reference_H*.json, evidence/round14/ray_proposals[_pilot].json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, time
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "solver_snapshot"))
sys.path.insert(0, os.path.join(HERE, "..", "round12_experiments"))
import is_core as ic  # noqa: E402

EV = os.path.join(HERE, "..", "..", "evidence")
OUT = os.path.join(EV, "round14")
os.makedirs(OUT, exist_ok=True)
H_LIST = (0.55, 0.75, 0.95)
J, N_STEPS, M_IS = 9, 512, 32768
MODELS = {
    "single": dict(T=4.0, a=-1.0, b=1.0, s=0.5, x0=1.0, sign=+1, levels=(0.8, 0.9, 1.0, 1.1), drift=lambda x: -x - x ** 3),
    "double": dict(T=2.0, a=1.0, b=1.0, s=0.30, x0=1.0, sign=-1, levels=(0.0, -0.5), drift=lambda x: x - x ** 3),
}


# ------------------------------------------------------------------ (A) crude Monte Carlo reference
def reference(iH, log2M=23, model="double"):
    """Crude Monte Carlo on the time-discrete target.  model="single" is an independent re-estimate of the
    stored 2x10^7-path single-well reference (used for H = 0.55 to examine one 3 s.e. discrepancy)."""
    H = H_LIST[iH]; m = MODELS[model]
    coords = ic.DiscreteCoords(H, m["T"], J, N_STEPS)
    rng = np.random.default_rng(np.random.SeedSequence([20261111, 14, iH] + ([] if model == "double" else [1])))
    M = 2 ** log2M
    t0 = time.perf_counter()
    est = ic.is_estimate(H, m["T"], N_STEPS, coords, np.zeros(J), m["levels"], M, rng, m["drift"], x0=m["x0"], s=m["s"],
                         chunk=8192, sign=m["sign"])
    rec = dict(model=model, H=H, M=M, n_steps=N_STEPS, seconds=time.perf_counter() - t0,
               levels={str(c): dict(p=float(est["p"][j]), se=float(est["se"][j]), hits=int(est["hits"][j]))
                       for j, c in enumerate(m["levels"])})
    json.dump(rec, open(os.path.join(OUT, "%s_well_reference_H%s.json" % (model, H)), "w"), indent=1)
    print(json.dumps(rec, indent=1), flush=True)


# ------------------------------------------------------------------ (B) ray-corrected proposals
def ray_scale(coords, d, c, m):
    """Smallest s >= 0 with sign*(X(T; shift s d) - c) = 0 for the time-discrete equation driven by the shift alone."""
    delta = coords.shift(d); dt = m["T"] / N_STEPS; n_eval = [0]

    def g(sc):
        n_eval[0] += 1
        x = ic.heun((sc * delta)[None, :], dt, m["drift"], m["x0"], m["s"])[0]
        return m["sign"] * (x - c)
    lo, step = 0.0, 0.25
    if g(0.0) >= 0:
        return 0.0, n_eval[0]
    hi = step
    while g(hi) < 0:
        lo, hi = hi, hi + step
        if hi > 30:
            raise RuntimeError("no crossing along the ray")
    return brentq(g, lo, hi, xtol=1e-10), n_eval[0]


def load_refs():
    sw = json.load(open(os.path.join(EV, "round12", "single_well_is_matrix.json")))
    dw = json.load(open(os.path.join(EV, "round12", "double_well_stretch.json")))
    ref = {"single": {}, "double": {}}
    for H in H_LIST:
        for c in MODELS["single"]["levels"]:
            e = sw[str(H)]["p1_c%s" % c]
            ref["single"][(H, c)] = (e["p_ref"], e["se_ref"], "2e7-path stored reference")
        new = os.path.join(OUT, "double_well_reference_H%s.json" % H)
        for c in MODELS["double"]["levels"]:
            if os.path.exists(new):
                r = json.load(open(new))["levels"][str(c)]
                ref["double"][(H, c)] = (r["p"], r["se"], "2^23-path reference")
            else:
                r = dw[str(H)]["crude_mc"]["levels"][str(c)]
                ref["double"][(H, c)] = (r["p"], r["se"], "131072-path stored reference")
    return ref


def ray(mode):
    ref = load_refs()
    out = {}
    cases = ([("single", 0.95), ("double", 0.75)] if mode == "pilot"
             else [(mod, H) for mod in ("single", "double") for H in H_LIST])
    for mod, H in cases:
        m = MODELS[mod]; iH = H_LIST.index(H)
        coords = ic.DiscreteCoords(H, m["T"], J, N_STEPS)
        solves = {}
        for p in (1, 3):
            solves[p] = ic.coschaos_cubic(H, m["T"], J, p, x0=m["x0"], a=m["a"], b=m["b"], s=m["s"])
        for ic_, c in enumerate(m["levels"]):
            for p in (1, 3):
                res, t_solve = solves[p]
                t0 = time.perf_counter()
                xs, _ = ic.design_point(res, c, sign=m["sign"])
                t_dp = time.perf_counter() - t0
                d = xs / np.linalg.norm(xs)
                t0 = time.perf_counter()
                sc, n_eval = ray_scale(coords, d, c, m)
                t_ray = time.perf_counter() - t0
                rng = np.random.default_rng(np.random.SeedSequence([20261111, 15, ("single", "double").index(mod), iH, ic_, p]))
                t0 = time.perf_counter()
                est = ic.is_estimate(H, m["T"], N_STEPS, coords, sc * d, [c], M_IS, rng, m["drift"], x0=m["x0"], s=m["s"],
                                     chunk=4096, sign=m["sign"])
                t_mc = time.perf_counter() - t0
                pi, se = float(est["p"][0]), float(est["se"][0])
                pr, ser, src = ref[mod][(H, c)]
                row = dict(model=mod, H=H, c=c, guide_degree=p, poly_norm_sq=float(xs @ xs), ray_norm_sq=float(sc ** 2),
                           ray_single_path_solves=n_eval, p_is=pi, se_is=se, rel_se_percent=100 * se / pi if pi > 0 else None,
                           hits=int(est["hits"][0]), w_mean=float(est["w_mean"]),
                           vrf=pi * (1 - pi) / (M_IS * se ** 2) if se > 0 else None,
                           p_ref=pr, se_ref=ser, ref_source=src, z=abs(pi - pr) / float(np.hypot(se, ser)),
                           t_solve=t_solve, t_design=t_dp, t_ray=t_ray, t_sampling=t_mc, M=M_IS, n_steps=N_STEPS)
                out["%s_H%s_c%s_p%d" % (mod, H, c, p)] = row
                print("%-6s H=%.2f c=%+.1f p=%d  |x*|^2 poly %6.2f ray %6.2f (%2d solves, %.3fs)  P=%.3e rel.se %5.2f%%  VRF %8.0f  z %.2f  [ref %.3e +- %.1e]"
                      % (mod, H, c, p, row["poly_norm_sq"], row["ray_norm_sq"], n_eval, t_dp + t_ray, pi, row["rel_se_percent"],
                         row["vrf"], row["z"], pr, ser), flush=True)
                json.dump(out, open(os.path.join(OUT, "ray_proposals%s.json" % ("_pilot" if mode == "pilot" else "")), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "ref":
        reference(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 23,
                  sys.argv[4] if len(sys.argv) > 4 else "double")
    else:
        ray(sys.argv[2] if len(sys.argv) > 2 else "pilot")
