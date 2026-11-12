"""Complete the <1% all-time-steps anisotropic refinement benchmark and verification."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[name] = '1'

import json, time
from pathlib import Path
import numpy as np
import refine_solver as rs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


def run_fast_candidates():
    out = rs.read('fast_selective_candidates')
    # Explore compact (K, p, q, qh) at J=65 (H=0.75) and J=33, 65 (H=0.95)
    configs = [
        # H=0.75, J=65
        (0.75, 65, 1, 3, 8, 8),
        (0.75, 65, 1, 5, 10, 8),
        (0.75, 65, 2, 3, 8, 8),
        (0.75, 65, 2, 5, 10, 8),
        (0.75, 65, 3, 3, 8, 8),
        (0.75, 65, 3, 5, 10, 8),
        # H=0.95, J=33 and J=65
        (0.95, 33, 1, 3, 8, 8),
        (0.95, 33, 2, 3, 8, 8),
        (0.95, 33, 3, 3, 8, 8),
        (0.95, 65, 1, 3, 8, 8),
        (0.95, 65, 2, 3, 8, 8),
        (0.95, 65, 2, 5, 10, 8),
        (0.95, 65, 3, 3, 8, 8),
        (0.95, 65, 3, 5, 10, 8),
    ]
    for H, J, K, p, q, qh in configs:
        key = f"{H}_J{J}_K{K}_p{p}_q{q}_qh{qh}"
        if key not in out:
            m = rs.Selective(H, J, K, p, q, qh)
            row = dict(H=H, J=J, K=K, p=p, q=q, qh=qh, n=m.n, seconds=m.seconds, nfev=m.nfev, **rs.error(m.coef, H))
            out[key] = row
            rs.save('fast_selective_candidates', out)
        else:
            row = out[key]
        print(f"[FastCand] {key}: max_var_err={row['max_variance_error_percent']:.4f}% (at t={row['worst_time']}), "
              f"final_var_err={row['final_variance_error_percent']:.4f}%, setup={row['seconds']:.3f}s", flush=True)
    return out


def verify_and_benchmark_winners():
    out = rs.read('verified_benchmark')
    # Winners that achieve <1% across ALL 32 positive time steps with safety margin:
    # We test both ultra-fast K=1/K=2 and K=3 candidates, plus refinement checks (q/qh, rtol/atol, J=129)
    winners = [
        ('H0.75_fast_K1_p3', 0.75, 65, 1, 3, 8, 8),
        ('H0.75_fast_K2_p3', 0.75, 65, 2, 3, 8, 8),
        ('H0.75_fast_K2_p5', 0.75, 65, 2, 5, 10, 8),
        ('H0.75_fast_K3_p3', 0.75, 65, 3, 3, 8, 8),
        ('H0.95_fast_J33_K2_p3', 0.95, 33, 2, 3, 8, 8),
        ('H0.95_fast_J65_K1_p3', 0.95, 65, 1, 3, 8, 8),
        ('H0.95_fast_J65_K2_p3', 0.95, 65, 2, 3, 8, 8),
        ('H0.95_fast_J65_K2_p5', 0.95, 65, 2, 5, 10, 8),
    ]
    for label, H, J, K, p, q, qh in winners:
        if label in out:
            continue
        setup_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            m = rs.Selective(H, J, K, p, q, qh)
            setup_times.append(time.perf_counter() - t0)
        err = rs.error(m.coef, H)

        # Refinement check: higher quadrature (q+4, qh+4) and tighter ODE tolerances
        m_ref_q = rs.Selective(H, J, K, p, q + 4, qh + 4, rtol=2e-8, atol=2e-10)
        quad_ode_diff_pct = float(100.0 * np.max(np.abs(
            np.sum(m.coef[1:, 1:] ** 2, axis=1) / np.sum(m_ref_q.coef[1:, 1:] ** 2, axis=1) - 1.0
        )))

        # Evaluate 65,536 paths across 3 seeds (1101, 1102, 1103)
        eval_reps = []
        for seed in [1101, 1102, 1103]:
            t0 = time.perf_counter()
            paths = m.produce(65536, seed)
            elapsed = time.perf_counter() - t0
            met = rs.b.metrics(paths)
            eval_reps.append(dict(
                seed=seed, online_seconds=elapsed,
                tail=met['tail'], q95=met['q95'], q99=met['q99'],
                final_var=met['variance'][-1], final_mean=met['mean'][-1]
            ))

        median_setup = float(np.median(setup_times))
        median_online = float(np.median([r['online_seconds'] for r in eval_reps]))
        out[label] = dict(
            H=H, J=J, K=K, p=p, q=q, qh=qh, n_basis=m.n,
            setup_seconds_list=setup_times, median_setup_seconds=median_setup,
            median_online_65536_seconds=median_online,
            total_65536_seconds=float(median_setup + median_online),
            max_variance_error_percent=err['max_variance_error_percent'],
            final_variance_error_percent=err['final_variance_error_percent'],
            worst_time=err['worst_time'],
            max_mean_normalized_percent=err['max_mean_normalized_percent'],
            quad_ode_refinement_diff_percent=quad_ode_diff_pct,
            replicates_65536=eval_reps
        )
        rs.save('verified_benchmark', out)
        print(f"[Verified] {label}: max_err={err['max_variance_error_percent']:.4f}%, "
              f"final_err={err['final_variance_error_percent']:.4f}%, "
              f"setup={median_setup:.3f}s, online(65k)={median_online:.3f}s, total={median_setup+median_online:.3f}s, "
              f"quad_ode_diff={quad_ode_diff_pct:.5f}%", flush=True)


if __name__ == '__main__':
    run_fast_candidates()
    verify_and_benchmark_winners()
