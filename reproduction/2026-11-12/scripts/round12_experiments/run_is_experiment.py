"""Complete CosChaos-Guided Importance Sampling Experiment (Single-Well Matrix + Double-Well Stretch Test)."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[name] = '1'

import json, time, csv
from pathlib import Path
import numpy as np
import is_core as ic

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

REF_PATH = ROOT.parents[1] / 'evidence/round5/P5_mc_reference.json'
REF = json.loads(REF_PATH.read_text(encoding='utf-8'))
REF_LEVELS = REF['levels']


def get_ref(H, c):
    idx = int(round(c * 100))
    assert abs(REF_LEVELS[idx] - c) < 1e-9
    mc = REF[str(H)]['mc_512']
    p_ref = float(mc['p4'][idx])
    ci_ref = float(mc['ci4'][idx])
    se_ref = ci_ref / 1.96
    return dict(p_ref=p_ref, ci_ref=ci_ref, se_ref=se_ref, paths=mc['paths'], seconds=mc['seconds'])


def run_single_well_matrix():
    H_list = [0.55, 0.75, 0.95]
    c_list = [0.8, 0.9, 1.0, 1.1]
    J = 9
    n = 512
    T = 4.0
    M = 32768  # 2^15 paths
    drift = lambda x: -x - x ** 3

    out_path = OUT / 'single_well_is_matrix.json'
    results = json.loads(out_path.read_text(encoding='utf-8')) if out_path.exists() else {}

    for iH, H in enumerate(H_list):
        h_key = str(H)
        if h_key not in results:
            results[h_key] = {}
        t_coord0 = time.perf_counter()
        coords = ic.DiscreteCoords(H, T, J, n)
        t_coord = time.perf_counter() - t_coord0

        solves = {}
        for p in [1, 3]:
            res, t_solve = ic.coschaos_cubic(H, T, J, p, x0=1.0, a=-1.0, b=1.0, s=0.5)
            solves[p] = (res, t_solve)

        for ic_idx, c in enumerate(c_list):
            ref = get_ref(H, c)
            for p in [1, 3]:
                run_key = f"p{p}_c{c}"
                if run_key in results[h_key]:
                    continue
                res, t_solve = solves[p]
                t_dp0 = time.perf_counter()
                xs, p_val = ic.design_point(res, c, sign=+1)
                t_dp = time.perf_counter() - t_dp0

                rng = np.random.default_rng(np.random.SeedSequence([20261110, iH, ic_idx, p]))
                t_mc0 = time.perf_counter()
                est = ic.is_estimate(H, T, n, coords, xs, c_list, M, rng, drift, x0=1.0, s=0.5, chunk=4096, sign=+1)
                t_mc = time.perf_counter() - t_mc0

                p_is = float(est['p'][ic_idx])
                se_is = float(est['se'][ic_idx])
                ci_is = 1.96 * se_is
                w_mean = float(est['w_mean'])
                hits = int(est['hits'][ic_idx])

                # VRF = p*(1-p) / (M * SE_IS^2)
                p_ref = ref['p_ref']
                vrf_ref = float(p_ref * (1.0 - p_ref) / (M * (se_is ** 2))) if (p_ref > 0 and se_is > 0) else None
                vrf_self = float(p_is * (1.0 - p_is) / (M * (se_is ** 2))) if se_is > 0 else None

                # Z-score and combined 95% interval check against reference
                comb_se = float(np.sqrt(se_is ** 2 + ref['se_ref'] ** 2))
                z_score = float(abs(p_is - p_ref) / comb_se) if comb_se > 0 else 0.0
                inside_95 = bool(abs(p_is - p_ref) <= 1.96 * comb_se)

                # All levels estimated by this single proposal
                all_levels = {}
                for j, cj in enumerate(c_list):
                    ref_j = get_ref(H, cj)
                    pj = float(est['p'][j])
                    sej = float(est['se'][j])
                    vrf_j = float(pj * (1.0 - pj) / (M * (sej ** 2))) if sej > 0 else None
                    all_levels[str(cj)] = dict(
                        p_is=pj, se_is=sej, rel_se_percent=float(100.0 * sej / pj) if pj > 0 else None,
                        hits=int(est['hits'][j]), p_ref=ref_j['p_ref'], se_ref=ref_j['se_ref'], vrf_self=vrf_j
                    )

                row = dict(
                    H=H, c=c, p=p, J=J, n=n, M=M,
                    xi_norm_sq=float(xs @ xs), xi_norm=float(np.linalg.norm(xs)),
                    poly_at_design=float(p_val),
                    p_is=p_is, se_is=se_is, ci95_is=ci_is,
                    rel_ci95_percent=float(100.0 * ci_is / p_is) if p_is > 0 else None,
                    paths_for_1pct_ci95=float((1.96 * 100.0 * se_is / p_is) ** 2 * M) if p_is > 0 else None,
                    w_mean=w_mean, hits=hits, hit_rate=float(hits / M),
                    p_ref=p_ref, se_ref=ref['se_ref'], ci95_ref=ref['ci_ref'],
                    vrf_ref=vrf_ref, vrf_self=vrf_self,
                    z_score_vs_ref=z_score, inside_combined_95ci=inside_95,
                    t_coord_sec=t_coord, t_solve_sec=t_solve, t_dp_sec=t_dp, t_mc_sec=t_mc,
                    t_total_sec=float(t_solve + t_dp + t_mc),
                    all_levels=all_levels
                )
                results[h_key][run_key] = row
                out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
                print(f"[SingleWell] H={H} c={c} p={p}: P_IS={p_is:.4e} +- {se_is:.2e} (Ref={p_ref:.4e} +- {ref['se_ref']:.2e}), "
                      f"VRF_self={vrf_self:.1f}x, z={z_score:.2f}, t={row['t_total_sec']:.2f}s", flush=True)

    # Export clean CSV summary table
    csv_path = OUT / 'single_well_is_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'H', 'c', 'p', 'p_ref', 'se_ref', 'p_is', 'se_is', 'rel_ci95_pct',
            'vrf_ref', 'vrf_self', 'z_score', 'inside_95ci', 'w_mean', 'hit_rate', 't_solve_s', 't_total_s'
        ])
        for H in H_list:
            for c in c_list:
                for p in [1, 3]:
                    r = results[str(H)][f"p{p}_c{c}"]
                    writer.writerow([
                        H, c, p, r['p_ref'], r['se_ref'], r['p_is'], r['se_is'],
                        round(r['rel_ci95_percent'], 3),
                        round(r['vrf_ref'], 1) if r['vrf_ref'] else '',
                        round(r['vrf_self'], 1) if r['vrf_self'] else '',
                        round(r['z_score_vs_ref'], 3), r['inside_combined_95ci'],
                        round(r['w_mean'], 5), round(r['hit_rate'], 4),
                        round(r['t_solve_sec'], 4), round(r['t_total_sec'], 3)
                    ])


def run_double_well_stretch():
    """Stretch test: Double-well rare transition dX = (X - X^3) dt + sigma dB_H, X(0) = 1.0, T = 2.0, sigma = 0.30.
    Target: rare barrier crossing P(X(T) < 0.0) and transition into left well P(X(T) < -0.5)."""
    out_path = OUT / 'double_well_stretch.json'
    results = json.loads(out_path.read_text(encoding='utf-8')) if out_path.exists() else {}

    H_list = [0.55, 0.75, 0.95]
    c_list = [0.0, -0.5]  # barrier crossing X(T) < 0 and well transition X(T) < -0.5
    T = 2.0
    J = 9
    n = 512
    s = 0.30
    x0 = 1.0
    drift = lambda x: x - x ** 3
    M_is = 32768
    M_mc = 131072  # 2^17 crude MC reference paths

    for iH, H in enumerate(H_list):
        h_key = str(H)
        if h_key in results:
            continue
        results[h_key] = {}
        coords = ic.DiscreteCoords(H, T, J, n)

        # Crude MC reference with M_mc = 131,072 paths
        rng_mc = np.random.default_rng(np.random.SeedSequence([20261110, 99, iH]))
        t0 = time.perf_counter()
        mc_est = ic.is_estimate(H, T, n, coords, np.zeros(J), c_list, M_mc, rng_mc, drift, x0=x0, s=s, chunk=4096, sign=-1)
        t_mc_ref = time.perf_counter() - t0
        results[h_key]['crude_mc'] = dict(
            M=M_mc, seconds=t_mc_ref,
            levels={str(c_list[j]): dict(p=float(mc_est['p'][j]), se=float(mc_est['se'][j]), hits=int(mc_est['hits'][j]))
                    for j in range(len(c_list))}
        )
        print(f"[DoubleWell Crude MC] H={H}: P(X<0)={mc_est['p'][0]:.4e}+-{mc_est['se'][0]:.2e}, "
              f"P(X<-0.5)={mc_est['p'][1]:.4e}+-{mc_est['se'][1]:.2e} ({t_mc_ref:.2f}s)", flush=True)

        # Solve CosChaos for double-well a=+1, b=+1
        for p in [1, 3]:
            res, t_solve = ic.coschaos_cubic(H, T, J, p, x0=x0, a=1.0, b=1.0, s=s)
            for ic_idx, c in enumerate(c_list):
                t_dp0 = time.perf_counter()
                xs, p_val = ic.design_point(res, c, sign=-1)
                t_dp = time.perf_counter() - t_dp0

                rng = np.random.default_rng(np.random.SeedSequence([20261110, 88, iH, ic_idx, p]))
                t_is0 = time.perf_counter()
                est = ic.is_estimate(H, T, n, coords, xs, c_list, M_is, rng, drift, x0=x0, s=s, chunk=4096, sign=-1)
                t_is = time.perf_counter() - t_is0

                p_is = float(est['p'][ic_idx])
                se_is = float(est['se'][ic_idx])
                vrf_self = float(p_is * (1.0 - p_is) / (M_is * (se_is ** 2))) if se_is > 0 else None
                mc_p = results[h_key]['crude_mc']['levels'][str(c)]['p']
                mc_se = results[h_key]['crude_mc']['levels'][str(c)]['se']
                comb_se = float(np.sqrt(se_is ** 2 + mc_se ** 2))
                z_score = float(abs(p_is - mc_p) / comb_se) if comb_se > 0 else None

                results[h_key][f"p{p}_c{c}"] = dict(
                    H=H, c=c, p=p, M=M_is,
                    xi_norm_sq=float(xs @ xs), poly_at_design=float(p_val),
                    p_is=p_is, se_is=se_is, w_mean=float(est['w_mean']),
                    hits=int(est['hits'][ic_idx]), hit_rate=float(est['hits'][ic_idx] / M_is),
                    p_mc=mc_p, se_mc=mc_se, vrf_self=vrf_self, z_score_vs_mc=z_score,
                    t_solve_sec=t_solve, t_dp_sec=t_dp, t_is_sec=t_is, t_total_sec=float(t_solve + t_dp + t_is)
                )
                print(f"[DoubleWell IS] H={H} c={c} p={p}: P_IS={p_is:.4e}+-{se_is:.2e}, VRF={vrf_self:.1f}x, z={z_score:.2f}", flush=True)
        out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run_single_well_matrix()
    run_double_well_stretch()
