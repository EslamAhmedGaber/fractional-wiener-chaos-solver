"""Problem 3: 2D coupled competitive Lotka-Volterra ecosystem under shared multiplicative fBm
    dX_1 = (r_1 X_1 - a_11 X_1^2 - a_12 X_1 X_2) dt + lam X_1 <> dB_H,   X_1(0) = 4,
    dX_2 = (r_2 X_2 - a_21 X_1 X_2 - a_22 X_2^2) dt + lam X_2 <> dB_H,   X_2(0) = 4,
on 0 <= t <= T = 16, with r_1 = 0.6, r_2 = 0.4, a_11 = 0.014, a_12 = 0.008, a_21 = 0.004,
a_22 = 0.009, lam = 0.05, H in {0.55, 0.75, 0.95}.

Reference: 2D ELS26 WIS translation integrated by classical RK4 (n = 256 fBm steps, 128 RK4 steps)
across M = 2^20 independent exact fBm paths with B_H(t)^2 - t^{2H} variance-reduction control variate
(giving relative SE ~ 0.08-0.14%).
Comparators (all 1-thread): accelerated 2D CosChaos, 2D WIS-RK4 + CV MC (n = 128, M up to 2^20),
and 2D GfBm EM + CV MC (n = 256, 512).
"""
import json
import math
import os
import sys
import time
import numpy as np
from scipy.fft import fft
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'solver'))
import hfc
import hwc

PARAMS = dict(r1=0.6, r2=0.4, a11=0.014, a12=0.008, a21=0.004, a22=0.009, lam=0.05, x0=[4.0, 4.0], T=16.0)
Q = 8
TIMES = PARAMS['T'] * np.arange(1, Q + 1, dtype=float) / Q
DENSE_TIMES = np.linspace(0.0, PARAMS['T'], 161)
HS = (0.55, 0.75, 0.95)


def fgn_increments(H, n, M, rng, T):
    k = np.arange(n + 1, dtype=float)
    g = 0.5 * ((k + 1.0) ** (2.0 * H) - 2.0 * (k ** (2.0 * H)) + np.abs(k - 1.0) ** (2.0 * H))
    lam = fft(np.r_[g, g[1:-1][::-1]]).real
    assert lam.min() > -1e-9
    lam = np.clip(lam, 0.0, None)
    w = rng.standard_normal((M, 2 * n)) + 1j * rng.standard_normal((M, 2 * n))
    return fft(w * np.sqrt(lam / (2.0 * n)), axis=1).real[:, :n] * ((T / n) ** H)


def lv_wis_rk4(B, H, idx, h):
    r1, r2 = PARAMS['r1'], PARAMS['r2']
    a11, a12, a21, a22 = PARAMS['a11'], PARAMS['a12'], PARAMS['a21'], PARAMS['a22']
    lam = PARAMS['lam']
    x10, x20 = PARAMS['x0']
    eBm = np.exp(-lam * B)
    M = B.shape[0]
    out = np.empty((len(idx), M, 2), dtype=float)
    Hstep = 2.0 * h
    for q, m in enumerate(idx):
        tau = m * h
        s = h * np.arange(m + 1, dtype=float)
        cf = np.exp(0.5 * (lam ** 2) * (tau ** (2.0 * H) - (tau - s) ** (2.0 * H)))
        invJ = (1.0 / cf)[None, :] / eBm[:, :m + 1]
        z1 = np.full(M, x10, dtype=float)
        z2 = np.full(M, x20, dtype=float)
        for i in range(0, m, 2):
            i0 = invJ[:, i]; im = invJ[:, i + 1]; i1 = invJ[:, i + 2]
            k1_1 = r1 * z1 - (a11 * z1 * z1 + a12 * z1 * z2) * i0
            k1_2 = r2 * z2 - (a21 * z1 * z2 + a22 * z2 * z2) * i0
            u1 = z1 + 0.5 * Hstep * k1_1; u2 = z2 + 0.5 * Hstep * k1_2
            k2_1 = r1 * u1 - (a11 * u1 * u1 + a12 * u1 * u2) * im
            k2_2 = r2 * u2 - (a21 * u1 * u2 + a22 * u2 * u2) * im
            u1 = z1 + 0.5 * Hstep * k2_1; u2 = z2 + 0.5 * Hstep * k2_2
            k3_1 = r1 * u1 - (a11 * u1 * u1 + a12 * u1 * u2) * im
            k3_2 = r2 * u2 - (a21 * u1 * u2 + a22 * u2 * u2) * im
            u1 = z1 + Hstep * k3_1; u2 = z2 + Hstep * k3_2
            k4_1 = r1 * u1 - (a11 * u1 * u1 + a12 * u1 * u2) * i1
            k4_2 = r2 * u2 - (a21 * u1 * u2 + a22 * u2 * u2) * i1
            z1 += (Hstep / 6.0) * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
            z2 += (Hstep / 6.0) * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
        out[q, :, 0] = z1 * invJ[:, m]
        out[q, :, 1] = z2 * invJ[:, m]
    return out


def lv_wis_em(B, H, idx, h):
    r1, r2 = PARAMS['r1'], PARAMS['r2']
    a11, a12, a21, a22 = PARAMS['a11'], PARAMS['a12'], PARAMS['a21'], PARAMS['a22']
    lam = PARAMS['lam']
    x10, x20 = PARAMS['x0']
    eBm = np.exp(-lam * B)
    M = B.shape[0]
    out = np.empty((len(idx), M, 2), dtype=float)
    for q, m in enumerate(idx):
        tau = m * h
        s = h * np.arange(m + 1, dtype=float)
        cf = np.exp(0.5 * (lam ** 2) * (tau ** (2.0 * H) - (tau - s) ** (2.0 * H)))
        invJ = (1.0 / cf)[None, :] / eBm[:, :m + 1]
        z1 = np.full(M, x10, dtype=float)
        z2 = np.full(M, x20, dtype=float)
        for i in range(m):
            ii = invJ[:, i]
            dz1 = r1 * z1 - (a11 * z1 * z1 + a12 * z1 * z2) * ii
            dz2 = r2 * z2 - (a21 * z1 * z2 + a22 * z2 * z2) * ii
            z1 = z1 + h * dz1
            z2 = z2 + h * dz2
        out[q, :, 0] = z1 * invJ[:, m]
        out[q, :, 1] = z2 * invJ[:, m]
    return out


def stats_2d_cv(Y, BT, H, nb=32):
    """Independent-path 2D means, covariances and dominance prob with exact B_H(t) / B_H(t)^2-t^{2H}
    fitted control variates for the timed comparator."""
    # Y: (Q, M, 2), BT: (Q, M)
    Q_, M, _ = Y.shape
    t2H = TIMES ** (2.0 * H)
    half = M // 2
    Yb = Y.reshape(Q_, nb, -1, 2)
    Bb = BT.reshape(Q_, nb, -1)
    k = Yb.shape[2]

    # Control-variate mean using exact zero-mean B_H(t_q)
    bm = np.empty((Q_, nb, 2), dtype=float)
    for q in range(Q_):
        for d in range(2):
            b1 = np.cov(Y[q, half:, d], BT[q, half:])[0, 1] / t2H[q]
            b2 = np.cov(Y[q, :half, d], BT[q, :half])[0, 1] / t2H[q]
            bm[q, :nb // 2, d] = Yb[q, :nb // 2, :, d].mean(1) - b1 * Bb[q, :nb // 2].mean(1)
            bm[q, nb // 2:, d] = Yb[q, nb // 2:, :, d].mean(1) - b2 * Bb[q, nb // 2:].mean(1)
    mean = bm.mean(1)
    mean_se = bm.std(1, ddof=1) / math.sqrt(nb)

    # Control-variate covariance using exact zero-mean B_H(t_q)^2 - t_q^{2H}
    dY = Y - mean[:, None, :]
    C_sq = (BT ** 2) - t2H[:, None]
    Cb = C_sq.reshape(Q_, nb, -1).mean(2)
    bcov = np.empty((Q_, nb, 2, 2), dtype=float)
    corr_fac = k / (k - 1.0)
    for q in range(Q_):
        for i in range(2):
            for j in range(i, 2):
                Pij = dY[q, :, i] * dY[q, :, j]
                Pb = Pij.reshape(nb, -1).mean(1) * corr_fac
                b1 = np.cov(Pij[half:], C_sq[q, half:])[0, 1] / C_sq[q, half:].var(ddof=1)
                b2 = np.cov(Pij[:half], C_sq[q, :half])[0, 1] / C_sq[q, :half].var(ddof=1)
                vals = np.empty(nb, dtype=float)
                vals[:nb // 2] = Pb[:nb // 2] - b1 * Cb[q, :nb // 2]
                vals[nb // 2:] = Pb[nb // 2:] - b2 * Cb[q, nb // 2:]
                bcov[q, :, i, j] = vals
                bcov[q, :, j, i] = vals
    cov = bcov.mean(1)
    cov_se = bcov.std(1, ddof=1) / math.sqrt(nb)
    bdom = (Yb[-1, :, :, 0] > Yb[-1, :, :, 1]).mean(1)
    p_dom = float(bdom.mean())
    p_dom_se = float(bdom.std(ddof=1) / math.sqrt(nb))
    return mean, cov, mean_se, cov_se, p_dom, p_dom_se


def build_eq():
    r1, r2 = PARAMS['r1'], PARAMS['r2']
    a11, a12, a21, a22 = PARAMS['a11'], PARAMS['a12'], PARAMS['a21'], PARAMS['a22']
    lam = PARAMS['lam']
    return hwc.Equation(
        PARAMS['x0'],
        hwc.Poly([
            [(r1, (1, 0)), (-a11, (2, 0)), (-a12, (1, 1))],
            [(r2, (0, 1)), (-a21, (1, 1)), (-a22, (0, 2))],
        ]),
        hwc.Poly([[(lam, (1, 0))], [(lam, (0, 1))]]),
    )


def run_reference(H, n=256, M_total=2 ** 20, chunk=2 ** 12, seed=3026):
    T = PARAMS['T']
    h = T / n
    idx = (np.arange(1, Q + 1) * n) // Q
    rng = np.random.default_rng(seed + int(1000 * H))
    Y = np.empty((Q, M_total, 2), dtype=float)
    BT = np.empty((Q, M_total), dtype=float)
    chk = 2 ** 14
    Yc = np.empty((Q, chk, 2), dtype=float)

    t0 = time.perf_counter()
    for c in range(M_total // chunk):
        inc = fgn_increments(H, n, chunk, rng, T)
        B = np.column_stack((np.zeros(chunk), inc.cumsum(1)))
        sl = slice(c * chunk, (c + 1) * chunk)
        Y[:, sl, :] = lv_wis_rk4(B, H, idx, h)
        BT[:, sl] = B[:, idx].T
        if (c + 1) * chunk <= chk:
            Yc[:, sl, :] = lv_wis_rk4(B[:, ::2], H, idx // 2, 2.0 * h)
    sec = time.perf_counter() - t0

    from reference_stats import reference_moments
    m, cov, mse, covse = reference_moments(Y, BT, TIMES ** (2.0 * H))
    events = (Y[-1, :, 0] > Y[-1, :, 1]).reshape(64, -1).mean(1)
    pdom = float(events.mean())
    pdomse = float(events.std(ddof=1) / math.sqrt(len(events)))
    # Step-halving check on first chk paths
    df = Y[:, :chk, :] - Y[:, :chk, :].mean(1, keepdims=True)
    dc = Yc - Yc.mean(1, keepdims=True)
    covf = np.einsum('qmi,qmj->qij', df, df) / (chk - 1)
    covc = np.einsum('qmi,qmj->qij', dc, dc) / (chk - 1)
    step_diff_pct = float(100.0 * max(
        np.max(np.abs(covc[:, 0, 0] / covf[:, 0, 0] - 1.0)),
        np.max(np.abs(covc[:, 1, 1] / covf[:, 1, 1] - 1.0)),
    ))
    v1_relse = 100.0 * covse[:, 0, 0] / cov[:, 0, 0]
    v2_relse = 100.0 * covse[:, 1, 1] / cov[:, 1, 1]
    corr = cov[:, 0, 1] / np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
    print(f'P3 ref H={H}: {sec:.1f}s max_relse={max(v1_relse.max(), v2_relse.max()):.4f}% step_chk={step_diff_pct:.4f}% P(X1>X2)={100*pdom:.2f}%+/-{100*pdomse:.2f}%', flush=True)
    return dict(
        H=H, n=n, M=M_total, seconds=float(sec), times=TIMES.tolist(),
        mean=m.tolist(), cov=cov.tolist(), mean_se=mse.tolist(), cov_se=covse.tolist(),
        corr=corr.tolist(), p_dom=pdom, p_dom_se=pdomse,
        max_var_relse_pct=float(max(v1_relse.max(), v2_relse.max())),
        step_half_var_diff_pct=step_diff_pct,
    )


def run_coschaos(H, reps=3):
    eq = build_eq()
    configs = [(5, 1), (9, 1), (17, 1), (33, 1), (65, 1),
               (5, 2), (9, 2), (17, 2), (33, 2), (65, 2),
               (9, 3)]
    rows = []
    for J, p in configs:
        ts = []
        nrep = 1 if p == 3 else reps
        for _ in range(nrep):
            hfc._rules.clear()
            t0 = time.perf_counter()
            sol = hwc.solve(eq, H, PARAMS['T'], J, p, TIMES, rtol=1e-8, atol=1e-10)
            ts.append(time.perf_counter() - t0)
        cov = sol.covariance()
        corr = cov[:, 0, 1] / np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
        row = dict(
            J=J, p=p, n_coef=sol.info['n_coefficients'], nfev=sol.info['nfev'],
            seconds=float(np.median(ts)),
            mean=sol.mean.tolist(), cov=cov.tolist(), corr=corr.tolist(),
        )
        if J == 17 and p == 2:
            rng = np.random.default_rng(11)
            t_ev = time.perf_counter()
            xi = rng.standard_normal((2 ** 20, J))
            res_T = hwc.Result(sol.times[-1:], sol.coef[-1:], sol.basis, sol.noise, sol.form)
            YT = res_T.series(xi)[0]
            row['p_dom'] = float(np.mean(YT[:, 0] > YT[:, 1]))
            row['p_dom_eval_s'] = float(time.perf_counter() - t_ev)
        rows.append(row)
    hfc._rules.clear()
    sol_d = hwc.solve(eq, H, PARAMS['T'], 33, 2, DENSE_TIMES, rtol=1e-8, atol=1e-10)
    cov_d = sol_d.covariance()
    corr_d = np.where(
        (cov_d[:, 0, 0] > 1e-15) & (cov_d[:, 1, 1] > 1e-15),
        cov_d[:, 0, 1] / np.sqrt(np.maximum(cov_d[:, 0, 0] * cov_d[:, 1, 1], 1e-30)),
        1.0,
    )
    print(f'  P3 CC H={H} done (J=17,p=2: {next(r["seconds"] for r in rows if r["J"]==17 and r["p"]==2):.4f}s)', flush=True)
    return dict(
        H=H, rows=rows, dense_times=DENSE_TIMES.tolist(),
        dense_mean=sol_d.mean.tolist(), dense_cov=cov_d.tolist(), dense_corr=corr_d.tolist(),
    )


def run_mc_comparators(H, seed=4026):
    T = PARAMS['T']
    rows = []
    # 1. Strong fair baseline: 2D WIS-RK4 + B_H^2 CV MC (n = 128 -> 64 RK4 steps)
    n = 128
    h = T / n
    idx = (np.arange(1, Q + 1) * n) // Q
    chunk = 2 ** 12
    for M in (2 ** 14, 2 ** 16, 2 ** 18, 2 ** 19, 2 ** 20):
        rng = np.random.default_rng(seed + n + M + int(1000 * H))
        t0 = time.perf_counter()
        y_l = []; b_l = []
        done = 0
        while done < M:
            mm = min(chunk, M - done)
            B = np.column_stack((np.zeros(mm), fgn_increments(H, n, mm, rng, T).cumsum(1)))
            y_l.append(lv_wis_rk4(B, H, idx, h))
            b_l.append(B[:, idx].T)
            done += mm
        Y = np.concatenate(y_l, axis=1)
        BT = np.concatenate(b_l, axis=1)
        m, cov, mse, covse, pdom, pdomse = stats_2d_cv(Y, BT, H, nb=32)
        sec = time.perf_counter() - t0
        rows.append(dict(
            method='2D WIS-RK4 + CV MC', n=n, M=M, seconds=float(sec),
            mean=m.tolist(), cov=cov.tolist(), mean_se=mse.tolist(), cov_se=covse.tolist(),
            p_dom=pdom, p_dom_se=pdomse,
        ))
        print(f'  P3 WIS-RK4+CV H={H} n={n} M=2^{int(round(math.log2(M)))} {sec:.2f}s', flush=True)

    # 2. Published 2D GfBm EM + CV MC (n = 256, 512)
    for n_em, M in ((256, 2 ** 16), (512, 2 ** 18)):
        h = T / n_em
        idx = (np.arange(1, Q + 1) * n_em) // Q
        rng = np.random.default_rng(seed + 10000 + n_em + M + int(1000 * H))
        t0 = time.perf_counter()
        y_l = []; b_l = []
        done = 0
        while done < M:
            mm = min(chunk, M - done)
            B = np.column_stack((np.zeros(mm), fgn_increments(H, n_em, mm, rng, T).cumsum(1)))
            y_l.append(lv_wis_em(B, H, idx, h))
            b_l.append(B[:, idx].T)
            done += mm
        Y = np.concatenate(y_l, axis=1)
        BT = np.concatenate(b_l, axis=1)
        m, cov, mse, covse, pdom, pdomse = stats_2d_cv(Y, BT, H, nb=32)
        sec = time.perf_counter() - t0
        rows.append(dict(
            method='2D GfBm EM + CV MC', n=n_em, M=M, seconds=float(sec),
            mean=m.tolist(), cov=cov.tolist(), mean_se=mse.tolist(), cov_se=covse.tolist(),
            p_dom=pdom, p_dom_se=pdomse,
        ))
        print(f'  P3 GfBm-EM+CV H={H} n={n_em} M=2^{int(round(math.log2(M)))} {sec:.2f}s', flush=True)
    return rows


if __name__ == '__main__':
    out = dict(params=PARAMS, times=TIMES.tolist(), cases={})
    with threadpool_limits(limits=1):
        for H in HS:
            ref = run_reference(H)
            cc = run_coschaos(H)
            mc = run_mc_comparators(H)
            out['cases'][f'{H:.2f}'] = dict(reference=ref, coschaos=cc, mc=mc)
    path = os.path.join(HERE, 'p3_results.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path)
