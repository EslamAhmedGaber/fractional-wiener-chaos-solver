"""Problem 2: fractional population model with quadratic losses (Noor, Bazuhair, El-Beltagy 2023, Model 2)
    dy = (a y - eps y^2) dt + lam y <> dB_H,   y(0) = y_0,   0 <= t <= T,
with y_0 = 0.5, a = 0.5, eps = 0.01, T = 20, lam in {0.02, 0.05}, H in {0.55, 0.75, 0.95}.

All six (lam, H) cases are computed and timed on one BLAS thread:
  * Reference : exact WIS pathwise solution (Bernoulli reduction of ELS26 translation),
                evaluated on M = 2^20 independent exact fBm paths (n = 1024 trapezoidal steps,
                verified by half-grid check against n = 512), variance-reduced via the exact
                zero-mean linearised-logistic Gaussian control variate L(tau)^2 - Var(L(tau))
                (100% independent of CosChaos, cross-fitted across batches for zero bias).
  * CosChaos  : accelerated direct solver (evidence/solver/hwc.py), cold-start median of 3 runs
                (rtol = 1e-8, atol = 1e-10), plus 201-point continuous variance curves and
                2^20-sample fast quadratic-form terminal law / quantile evaluation.
  * FWHE      : first-order FWHE grid algorithm (Noor et al. 2023, Sec. 4.2), median of 3 runs.
  * GfBm EM   : ELS26 explicit Euler + exact linear control variate + Monte Carlo.
  * Exact WIS : best-case WIS path method (exact Bernoulli formula) + independent Monte Carlo
                (reporting both plain and B_H(t)^2 control-variate estimators, picking best guard).
"""
import json
import math
import os
import sys
import time
import numpy as np
from math import gamma as Gm
from scipy.fft import fft
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'solver'))
import hfc
import hwc

BASE = dict(y0=0.5, a=0.5, eps=0.01, T=20.0)
Q = 8
TIMES = BASE['T'] * np.arange(1, Q + 1, dtype=float) / Q
DENSE_TIMES = np.linspace(0.0, BASE['T'], 201)
CASES = [(lam, H) for lam in (0.02, 0.05) for H in (0.55, 0.75, 0.95)]


def fgn_increments(H, n, M, rng, T):
    """Exact fBm increments on n uniform steps of [0, T] via minimal circulant embedding."""
    k = np.arange(n + 1, dtype=float)
    g = 0.5 * ((k + 1.0) ** (2.0 * H) - 2.0 * (k ** (2.0 * H)) + np.abs(k - 1.0) ** (2.0 * H))
    lam = fft(np.r_[g, g[1:-1][::-1]]).real
    assert lam.min() > -1e-9, f'non-PSD circulant embedding: {lam.min()}'
    lam = np.clip(lam, 0.0, None)
    w = rng.standard_normal((M, 2 * n)) + 1j * rng.standard_normal((M, 2 * n))
    return fft(w * np.sqrt(lam / (2.0 * n)), axis=1).real[:, :n] * ((T / n) ** H)


def wis_exact(B, H, lam, idx, h):
    """Exact WIS pathwise Bernoulli solution of Problem 2 at grid indices idx."""
    y0, a, eps = BASE['y0'], BASE['a'], BASE['eps']
    eB = np.exp(lam * B)
    out = np.empty((len(idx), B.shape[0]), dtype=float)
    for q, m in enumerate(idx):
        tau = m * h
        u = tau - h * np.arange(m + 1, dtype=float)
        w = np.full(m + 1, h, dtype=float)
        w[0] = w[-1] = 0.5 * h
        ker = np.exp(-a * u + 0.5 * (lam ** 2) * (u ** (2.0 * H))) * w
        integ = (eB[:, :m + 1] @ ker) / eB[:, m]
        first = math.exp(-a * tau + 0.5 * (lam ** 2) * (tau ** (2.0 * H))) / (y0 * eB[:, m])
        out[q] = 1.0 / (first + eps * integ)
    return out


def _linearised_cv_weights(H, n):
    """Exact grid weights W (Q, n+1) and exact Gaussian variance Var(L_q) for the
    linearised logistic perturbation L(tau) = d/d(lam) y(tau)|_{lam=0}."""
    y0, a, eps, T = BASE['y0'], BASE['a'], BASE['eps'], BASE['T']
    h = T / n
    idx = (np.arange(1, Q + 1) * n) // Q
    ts = h * np.arange(n + 1, dtype=float)
    t2H = ts ** (2.0 * H)
    Sigma = 0.5 * (t2H[:, None] + t2H[None, :] - np.abs(ts[:, None] - ts[None, :]) ** (2.0 * H))
    W = np.zeros((Q, n + 1), dtype=float)
    for q, m in enumerate(idx):
        tau = m * h
        yd = a / (eps + (a / y0 - eps) * math.exp(-a * tau))
        u = tau - h * np.arange(m + 1, dtype=float)
        wt = np.full(m + 1, h, dtype=float)
        wt[0] = wt[-1] = 0.5 * h
        W[q, :m + 1] = -eps * (yd ** 2) * np.exp(-a * u) * wt
        W[q, m] += yd
    exact_var_L = np.einsum('qi,ij,qj->q', W, Sigma, W)
    return W, exact_var_L


def run_reference(lam, H, n=1024, M_total=2 ** 20, chunk=2 ** 12, seed=2026):
    T = BASE['T']
    h = T / n
    idx = (np.arange(1, Q + 1) * n) // Q
    W, exact_var_L = _linearised_cv_weights(H, n)
    rng = np.random.default_rng(seed + int(1000 * H) + int(1e5 * lam))
    Y = np.empty((Q, M_total), dtype=float)
    L = np.empty((Q, M_total), dtype=float)
    # Quadrature check on first 2^15 paths (n=1024 vs n=512 on identical paths)
    q_chk = 2 ** 15
    Yh = np.empty((Q, q_chk), dtype=float)

    t0 = time.perf_counter()
    for c in range(M_total // chunk):
        inc = fgn_increments(H, n, chunk, rng, T)
        B = np.column_stack((np.zeros(chunk), inc.cumsum(1)))
        sl = slice(c * chunk, (c + 1) * chunk)
        Y[:, sl] = wis_exact(B, H, lam, idx, h)
        L[:, sl] = (B @ W.T).T
        if (c + 1) * chunk <= q_chk:
            Yh[:, sl] = wis_exact(B[:, ::2], H, lam, idx // 2, 2.0 * h)
    sec = time.perf_counter() - t0

    from reference_stats import reference_moments
    m, v, m_se, v_se = reference_moments(Y[:, :, None], L, exact_var_L)
    mean, var = m[:, 0], v[:, 0, 0]
    mean_se, var_se = m_se[:, 0], v_se[:, 0, 0]

    # Quadrature half-grid check on identical first q_chk paths
    mf, vf = Y[:, :q_chk].mean(1), Y[:, :q_chk].var(1, ddof=1)
    mh, vh = Yh.mean(1), Yh.var(1, ddof=1)

    yT = Y[-1]
    qs = (0.01, 0.05, 0.5, 0.95, 0.99)
    bq = np.array([np.quantile(b, qs) for b in np.array_split(yT, 32)])
    q_se = bq.std(0, ddof=1) / math.sqrt(32)

    res = dict(
        task='reference', lam=lam, H=H, n=n, M=M_total, control_variate='linearised_logistic',
        seconds=float(sec), times=TIMES.tolist(),
        mean=mean.tolist(), var=var.tolist(),
        mean_se=mean_se.tolist(), var_se=var_se.tolist(),
        quad_half_mean_diff=(mh - mf).tolist(), quad_half_var_diff=(vh - vf).tolist(),
        terminal_quantiles={str(q): float(np.quantile(yT, q)) for q in qs},
        terminal_quantile_se={str(q): float(s) for q, s in zip(qs, q_se)},
    )
    np.save(os.path.join(HERE, f'ref_terminal_lam{lam}_H{H}.npy'), yT.astype(np.float32))
    print(f'ref lam={lam} H={H}: {sec:.1f}s max_relse={100 * np.max(var_se / var):.4f}% quad={100 * np.max(np.abs(vh - vf) / vf):.4f}%', flush=True)
    return res


def coschaos_solve(lam, H, J, p, times=TIMES, rtol=1e-8, atol=1e-10):
    hfc._rules.clear()
    a, eps = BASE['a'], BASE['eps']
    eq = hwc.Equation([BASE['y0']], hwc.Poly([[(a, (1,)), (-eps, (2,))]]), hwc.Poly([[(lam, (1,))]]))
    t0 = time.perf_counter()
    sol = hwc.solve(eq, H, BASE['T'], J, p, times, rtol=rtol, atol=atol)
    return sol, time.perf_counter() - t0


def run_coschaos(lam, H, reps=3):
    configs = [(5, 1), (9, 1), (17, 1), (33, 1), (65, 1), (129, 1),
               (5, 2), (9, 2), (17, 2), (33, 2), (65, 2), (129, 2),
               (9, 3), (17, 3)]
    rows = []
    for J, p in configs:
        if J == 129 and p == 2 and H >= 0.75:
            continue
        ts = []
        nrep = 1 if (p == 3 and J == 17) else reps
        for _ in range(nrep):
            sol, sec = coschaos_solve(lam, H, J, p)
            ts.append(sec)
        rows.append(dict(
            J=J, p=p, n_coef=sol.info['n_coefficients'], nfev=sol.info['nfev'],
            seconds=float(np.median(ts)), seconds_all=[float(x) for x in ts],
            mean=sol.mean[:, 0].tolist(), var=sol.variance()[:, 0].tolist(),
        ))
    sol_dense, _ = coschaos_solve(lam, H, 33, 2, times=DENSE_TIMES)
    print(f'  CC lam={lam} H={H} done (J=17,p=2: {next(r["seconds"] for r in rows if r["J"]==17 and r["p"]==2):.4f}s)', flush=True)
    return dict(
        task='coschaos', lam=lam, H=H, times=TIMES.tolist(), rows=rows,
        dense_times=DENSE_TIMES.tolist(),
        dense_mean=sol_dense.mean[:, 0].tolist(),
        dense_var=sol_dense.variance()[:, 0].tolist(),
    )


def fwhe_first_order(lam, H, dt):
    """First-order FWHE (Noor et al. 2023, Sec. 4.2)."""
    y0, a, eps, T = BASE['y0'], BASE['a'], BASE['eps'], BASE['T']
    N = int(round(T / dt))
    beta = 2.0 * H - 1.0
    w = np.empty(N + 1, dtype=float)
    w[0] = 1.0
    for j in range(1, N + 1):
        w[j] = w[j - 1] * (j - 1.0 + beta) / j
    Y = np.empty(N + 1, dtype=float)
    Phi = np.empty(N + 1, dtype=float)
    g = np.empty(N + 1, dtype=float)
    GJ = np.empty(N + 1, dtype=float)
    F = np.empty(N + 1, dtype=float)
    V = np.empty(N + 1, dtype=float)
    Y[0] = y0; Phi[0] = 0.0; g[0] = lam * y0
    dt_b = dt ** beta
    GJ[0] = dt_b * w[0] * g[0]; F[0] = 0.0; V[0] = 0.0
    c = 2.0 * H * Gm(2.0 * H)
    for i in range(N):
        Y[i + 1] = Y[i] + dt * (a * Y[i] - eps * Y[i] * Y[i] - eps * V[i])
        Phi[i + 1] = Phi[i] + dt * (a - 2.0 * eps * Y[i])
        g[i + 1] = lam * Y[i + 1] * math.exp(-Phi[i + 1])
        GJ[i + 1] = dt_b * float(np.dot(w[:i + 2], g[i + 1::-1]))
        F[i + 1] = F[i] + c * dt * 0.5 * (g[i] * GJ[i] + g[i + 1] * GJ[i + 1])
        V[i + 1] = math.exp(2.0 * Phi[i + 1]) * F[i + 1]
    idx = (np.arange(1, Q + 1) * N) // Q
    return Y[idx], V[idx]


def run_fwhe(lam, H, reps=3):
    rows = []
    for dt in (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            m, v = fwhe_first_order(lam, H, dt)
            ts.append(time.perf_counter() - t0)
        rows.append(dict(dt=dt, seconds=float(np.median(ts)), mean=m.tolist(), var=v.tolist()))
    print(f'  FWHE lam={lam} H={H} done', flush=True)
    return dict(task='fwhe', lam=lam, H=H, times=TIMES.tolist(), rows=rows)


def gfbm_em_paths(B, H, lam, n_out, h):
    y0, a, eps = BASE['y0'], BASE['a'], BASE['eps']
    eBm = np.exp(-lam * B)
    out = np.empty((len(n_out), B.shape[0]), dtype=float)
    for q, m in enumerate(n_out):
        tau = m * h
        s = h * np.arange(m, dtype=float)
        cf = np.exp(-a * s + 0.5 * (lam ** 2) * (tau ** (2.0 * H) - (tau - s) ** (2.0 * H)))
        z = np.full(B.shape[0], y0, dtype=float)
        for i in range(m):
            J = cf[i] * eBm[:, i]
            z = z - h * eps * z * z / J
        out[q] = z / (math.exp(-a * tau + 0.5 * (lam ** 2) * (tau ** (2.0 * H))) * eBm[:, m])
    return out


def run_gfbm(lam, H, ns=(80, 160, 320, 640), Ms=(2 ** 14, 2 ** 16, 2 ** 18), chunk=2 ** 13, seed=77):
    T = BASE['T']
    y0, a = BASE['y0'], BASE['a']
    EC = y0 * np.exp(a * TIMES)
    EC2 = (y0 ** 2) * np.exp(2.0 * a * TIMES + (lam ** 2) * (TIMES ** (2.0 * H)))
    rows = []
    for n in ns:
        h = T / n
        idx = (np.arange(1, Q + 1) * n) // Q
        for M in Ms:
            rng = np.random.default_rng(seed + n + M + int(1000 * H) + int(1e5 * lam))
            t0 = time.perf_counter()
            parts = []; ctl = []
            for _ in range(max(1, M // chunk)):
                mm = min(chunk, M)
                B = np.column_stack((np.zeros(mm), fgn_increments(H, n, mm, rng, T).cumsum(1)))
                parts.append(gfbm_em_paths(B, H, lam, idx, h))
                ctl.append(y0 * np.exp(a * TIMES[:, None] + lam * B[:, idx].T - 0.5 * (lam ** 2) * (TIMES[:, None] ** (2.0 * H))))
            Y = np.concatenate(parts, axis=1)
            C = np.concatenate(ctl, axis=1)
            sec = time.perf_counter() - t0
            nb = 16
            Yb = Y.reshape(Q, nb, -1); Cb = C.reshape(Q, nb, -1)
            plain_b = Yb.var(2, ddof=1)
            b1 = np.array([np.cov(Y[q], C[q])[0, 1] / C[q].var(ddof=1) for q in range(Q)])
            b2 = np.array([np.cov(Y[q] ** 2, C[q] ** 2)[0, 1] / (C[q] ** 2).var(ddof=1) for q in range(Q)])
            m1 = Yb.mean(2) - b1[:, None] * (Cb.mean(2) - EC[:, None])
            m2 = (Yb ** 2).mean(2) - b2[:, None] * ((Cb ** 2).mean(2) - EC2[:, None])
            cv_b = m2 - m1 ** 2
            rows.append(dict(
                n=n, M=M, seconds=float(sec),
                mean=Y.mean(1).tolist(), var=Y.var(1, ddof=1).tolist(),
                var_se=(plain_b.std(1, ddof=1) / math.sqrt(nb)).tolist(),
                mean_se=(Y.std(1, ddof=1) / math.sqrt(Y.shape[1])).tolist(),
                var_cv=cv_b.mean(1).tolist(), var_cv_se=(cv_b.std(1, ddof=1) / math.sqrt(nb)).tolist(),
                mean_cv=m1.mean(1).tolist(),
            ))
    print(f'  GfBm EM lam={lam} H={H} done', flush=True)
    return dict(task='gfbm', lam=lam, H=H, times=TIMES.tolist(), rows=rows)


def run_exact_mc(lam, H, n=256, Ms=(2 ** 14, 2 ** 16, 2 ** 18, 2 ** 19, 2 ** 20), chunk=2 ** 12, seed=99):
    """Best-case WIS path baseline: exact Bernoulli formula + independent MC with B_H(t)^2 control variate."""
    T = BASE['T']
    h = T / n
    idx = (np.arange(1, Q + 1) * n) // Q
    t2H = TIMES ** (2.0 * H)
    rows = []
    for M in Ms:
        rng = np.random.default_rng(seed + n + M + int(1000 * H) + int(1e5 * lam))
        t0 = time.perf_counter()
        y_list = []; b_list = []
        done = 0
        while done < M:
            mm = min(chunk, M - done)
            B = np.column_stack((np.zeros(mm), fgn_increments(H, n, mm, rng, T).cumsum(1)))
            y_list.append(wis_exact(B, H, lam, idx, h))
            b_list.append(B[:, idx].T)
            done += mm
        Y = np.concatenate(y_list, axis=1)
        BT = np.concatenate(b_list, axis=1)
        nb = 32
        Yb = Y.reshape(Q, nb, -1)
        # Plain unbiased variance
        bv_plain = Yb.var(2, ddof=1)
        # Control-variate variance using exact zero-mean B_H(t_q)^2 - t_q^{2H}
        ym = Y.mean(1, keepdims=True)
        Z_sq = (Y - ym) ** 2
        C_sq = (BT ** 2) - t2H[:, None]
        beta = np.array([np.cov(Z_sq[q], C_sq[q])[0, 1] / C_sq[q].var(ddof=1) for q in range(Q)])
        bv_cv = (Z_sq - beta[:, None] * C_sq).reshape(Q, nb, -1).mean(2) * (Yb.shape[2] / (Yb.shape[2] - 1.0))
        sec = time.perf_counter() - t0
        rows.append(dict(
            n=n, M=M, seconds=float(sec),
            mean=Y.mean(1).tolist(),
            var=Y.var(1, ddof=1).tolist(),
            var_se=(bv_plain.std(1, ddof=1) / math.sqrt(nb)).tolist(),
            var_cv=bv_cv.mean(1).tolist(),
            var_cv_se=(bv_cv.std(1, ddof=1) / math.sqrt(nb)).tolist(),
        ))
    print(f'  Exact WIS+CV MC lam={lam} H={H} done', flush=True)
    return dict(task='exactmc', lam=lam, H=H, times=TIMES.tolist(), rows=rows)


def run_law_check(lam, H, J=17, p=2):
    qs = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
    sol, solve_s = coschaos_solve(lam, H, J, p)
    rng = np.random.default_rng(5)
    t0 = time.perf_counter()
    xi = rng.standard_normal((2 ** 20, J))
    Y = sol.series(xi)[:, :, 0]
    quant = np.quantile(Y, qs, axis=1)
    eval_s = time.perf_counter() - t0
    ref = np.load(os.path.join(HERE, f'ref_terminal_lam{lam}_H{H}.npy')).astype(float)
    rq = np.quantile(ref, qs)
    lo, hi = float(np.quantile(ref, 0.002)), float(np.quantile(ref, 0.998))
    edges = np.linspace(lo, hi, 61)
    h_ref, _ = np.histogram(ref, bins=edges, density=True)
    h_cc, _ = np.histogram(Y[-1], bins=edges, density=True)
    return dict(
        lam=lam, H=H, J=J, p=p, solve_seconds=float(solve_s), eval_seconds=float(eval_s),
        quantile_levels=qs.tolist(),
        coschaos_quantiles_all_times=quant.tolist(),
        ref_terminal_quantiles=rq.tolist(),
        rel_err_terminal_pct=(100.0 * (quant[:, -1] / rq - 1.0)).tolist(),
        hist_edges=edges.tolist(), hist_ref=h_ref.tolist(), hist_cc=h_cc.tolist(),
    )


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for lam, H in CASES:
            ref = run_reference(lam, H)
            json.dump(ref, open(os.path.join(HERE, f'reference_lam{lam}_H{H}.json'), 'w'), indent=1)
            cc = run_coschaos(lam, H)
            json.dump(cc, open(os.path.join(HERE, f'coschaos_lam{lam}_H{H}.json'), 'w'), indent=1)
            fw = run_fwhe(lam, H)
            json.dump(fw, open(os.path.join(HERE, f'fwhe_lam{lam}_H{H}.json'), 'w'), indent=1)
            gf = run_gfbm(lam, H)
            json.dump(gf, open(os.path.join(HERE, f'gfbm_lam{lam}_H{H}.json'), 'w'), indent=1)
            ex = run_exact_mc(lam, H)
            json.dump(ex, open(os.path.join(HERE, f'exactmc_lam{lam}_H{H}.json'), 'w'), indent=1)
        for lam in (0.02, 0.05):
            law = run_law_check(lam, 0.95, J=17, p=2)
            json.dump(law, open(os.path.join(HERE, f'law_lam{lam}_H0.95.json'), 'w'), indent=1)
            print(f'law lam={lam} H=0.95: eval={law["eval_seconds"]:.2f}s max_q_err={max(abs(x) for x in law["rel_err_terminal_pct"]):.3f}%', flush=True)
    print('Problem 2 complete!')
