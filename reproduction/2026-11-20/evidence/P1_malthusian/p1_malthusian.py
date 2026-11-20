"""Problem 1: fractional Malthusian (Wick) growth (El-Beltagy, Etman, Maged 2022, Example 2)
    dX = mu X dt + sigma X <> dB_H,   X(0) = 1,   mu = 0.2, sigma = 0.5, T = 2.
Exact law: X(t) = exp(mu t + sigma B_H(t) - sigma^2 t^{2H} / 2).

Tasks written to evidence/P1_malthusian/p1_results.json (1 BLAS thread, cold timings):
  worked        : worked numbers at H = 0.75, J = 3 (cross-checked with 40-digit mpmath)
  condition     : spectral condition numbers kappa_2(G^(J)) and Cholesky residuals
  identity      : CosChaos (J, p) vs analytic FWHE order p vs exact law at eight times
  check         : general solver hwc.solve vs closed-form coefficients
  fwhegl        : first-order variance work-precision: FWHE-GL (N steps) vs CosChaos (J modes)
  discrete_wick : Bender-Parczewski (2010) discrete Wick exponential vs CosChaos + first-cell loss
  h_sweep       : single H-free solve (J = 65, p = 7) across nine Hurst indices H = 0.55..0.95
"""
import itertools
import json
import math
import os
import sys
import time
import numpy as np
from math import gamma as Gm
from scipy.integrate import quad, solve_ivp
from scipy.special import gamma, hyp2f1, roots_legendre
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'solver'))
import hfc
import hwc

MU, SIG, T, X0 = 0.2, 0.5, 2.0, 1.0
TIMES = np.arange(1, 9, dtype=float) * 0.25
HS = (0.55, 0.75, 0.95)


def exact_var(t, H):
    return np.exp(2.0 * MU * t) * np.expm1((SIG ** 2) * (t ** (2.0 * H)))


def qJ(H, J, t):
    G = hfc.gram(H, T, J)
    m = hfc.Mcoef(J, T, np.atleast_1d(t))
    return (SIG ** 2) * np.einsum('ti,ij,tj->t', m, G, m)


def var_series(q, p, t):
    return np.exp(2.0 * MU * t) * sum((q ** r) / math.factorial(r) for r in range(1, p + 1))


def task_worked():
    import mpmath as mp
    mp.mp.dps = 40
    H, J = 0.75, 3
    G = hfc.gram(H, T, J)
    L = np.linalg.cholesky(G)
    M = hfc.M_H(H, np.array([math.pi, 2.0 * math.pi]))
    # Independent 40-digit mpmath check
    Hm, Tm = mp.mpf('0.75'), mp.mpf('2')
    Cm = 2 * Hm * Tm ** (2 * Hm - 1)
    m1 = mp.hyp1f1(2 * Hm - 1, 2 * Hm, 1j * mp.pi)
    m2 = mp.hyp1f1(2 * Hm - 1, 2 * Hm, 2j * mp.pi)
    G_mp = np.array([
        [float(Tm ** (2 * Hm - 1)), 0.0, float(-mp.sqrt(2) * Cm / mp.pi * m2.imag / 2)],
        [0.0, float(Cm * (m1.real - 2 * (1 - Hm) / mp.pi * m1.imag)), 0.0],
        [float(-mp.sqrt(2) * Cm / mp.pi * m2.imag / 2), 0.0, float(Cm * (m2.real - 2 * (1 - Hm) / (2 * mp.pi) * m2.imag))],
    ])
    mp_diff = float(np.max(np.abs(G - G_mp)))
    assert mp_diff < 1e-14, f'mpmath mismatch: {mp_diff}'

    out = dict(
        H=H, J=J, T=T, mu=MU, sigma=SIG,
        M_pi=[float(M[0].real), float(M[0].imag)],
        M_2pi=[float(M[1].real), float(M[1].imag)],
        G=G.tolist(), L=L.tolist(), mpmath_max_diff=mp_diff, rows=[],
    )
    for t in (1.0, 2.0):
        m = hfc.Mcoef(J, T, np.array([t]))[0]
        b = SIG * (L.T @ m)
        q = float(b @ b)
        qex = (SIG ** 2) * (t ** (2.0 * H))
        row = dict(t=t, m=m.tolist(), b=b.tolist(), qJ=q, q=qex, exact=float(exact_var(t, H)))
        for p in (1, 3, 5):
            row[f'cos_p{p}'] = float(var_series(q, p, t))
            row[f'fwhe_p{p}'] = float(var_series(qex, p, t))
        out['rows'].append(row)
    return out


def task_condition():
    rows = []
    for H in HS:
        for J in (5, 17, 65, 257):
            G = hfc.gram(H, 1.0, J)
            ev = np.linalg.eigvalsh(G)
            L = np.linalg.cholesky(G)
            chol_res = float(np.linalg.norm(L @ L.T - G, 'fro') / np.linalg.norm(G, 'fro'))
            rows.append(dict(
                H=H, J=J, lam_min=float(ev[0]), lam_max=float(ev[-1]),
                cond2=float(ev[-1] / ev[0]), chol_rel_res=chol_res,
            ))
    return rows


def task_identity():
    rows = []
    for H in HS:
        ex = exact_var(TIMES, H)
        q = (SIG ** 2) * (TIMES ** (2.0 * H))
        for p in (1, 3, 5, 7):
            fw = var_series(q, p, TIMES)
            rows.append(dict(
                H=H, method='FWHE analytic', p=p,
                err_T=float(100.0 * (fw[-1] / ex[-1] - 1.0)),
                max_abs_err=float(100.0 * np.max(np.abs(fw / ex - 1.0))),
            ))
            for J in (3, 5, 9, 17, 33, 65):
                cc = var_series(qJ(H, J, TIMES), p, TIMES)
                rows.append(dict(
                    H=H, method='CosChaos', J=J, p=p,
                    err_T=float(100.0 * (cc[-1] / ex[-1] - 1.0)),
                    max_abs_err=float(100.0 * np.max(np.abs(cc / ex - 1.0))),
                    terminal_equals_fwhe=float(abs(cc[-1] - fw[-1]) / fw[-1]),
                    err_by_t=(100.0 * (cc / ex - 1.0)).tolist(),
                ))
    return rows


def task_check():
    out = []
    eq = hwc.Equation([X0], hwc.Poly([[(MU, (1,))]]), hwc.Poly([[(SIG, (1,))]]))
    for H, J, p in ((0.95, 5, 5), (0.75, 9, 3), (0.55, 17, 2)):
        hfc._rules.clear()
        sol = hwc.solve(eq, H, T, J, p, TIMES, rtol=1e-11, atol=1e-13)
        v = sol.variance()[:, 0]
        vref = var_series(qJ(H, J, TIMES), p, TIMES)
        out.append(dict(
            H=H, J=J, p=p, seconds=float(sol.info['seconds']),
            max_rel_diff_to_closed_form=float(np.max(np.abs(v / vref - 1.0))),
        ))
    return out


def gl_weights(beta, n):
    w = np.empty(n, dtype=float)
    w[0] = 1.0
    for j in range(1, n):
        w[j] = w[j - 1] * (j - 1.0 + beta) / j
    return w


def fwhe_gl_first_order(H, N):
    """EEM22 Sec. 4.2 at first order: explicit Euler + first-order Grunwald-Letnikov + trapezoid."""
    dt = T / N
    beta = 2.0 * H - 1.0
    w = gl_weights(beta, N + 1)
    X0g = (1.0 + MU * dt) ** np.arange(N + 1)
    idx = np.rint(TIMES / dt).astype(int)
    out = np.empty(len(idx), dtype=float)
    pref = 2.0 * H * Gm(2.0 * H) * dt
    dt_b = dt ** beta
    for pos, i in enumerate(idx):
        k = SIG * X0g[:i + 1] * ((1.0 + MU * dt) ** (i - np.arange(i + 1)))
        Jk = dt_b * np.convolve(w[:i + 1], k)[:i + 1]
        f = k * Jk
        out[pos] = pref * (f.sum() - 0.5 * (f[0] + f[-1]))
    return out


def task_fwhegl(reps=3):
    rows = []
    eq = hwc.Equation([X0], hwc.Poly([[(MU, (1,))]]), hwc.Poly([[(SIG, (1,))]]))
    for H in HS:
        v1 = (SIG ** 2) * np.exp(2.0 * MU * TIMES) * (TIMES ** (2.0 * H))
        for N in (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                v = fwhe_gl_first_order(H, N)
                ts.append(time.perf_counter() - t0)
            rows.append(dict(
                H=H, method='FWHE-GL', N=N,
                seconds=float(np.median(ts)),
                max_abs_err=float(100.0 * np.max(np.abs(v / v1 - 1.0))),
            ))
        for J in (3, 5, 9, 17, 33, 65, 129, 257):
            ts = []
            for _ in range(reps):
                hfc._rules.clear()
                t0 = time.perf_counter()
                sol = hwc.solve(eq, H, T, J, 1, TIMES, rtol=1e-8, atol=1e-10)
                v = sol.variance()[:, 0]
                ts.append(time.perf_counter() - t0)
            v_closed = np.exp(2.0 * MU * TIMES) * qJ(H, J, TIMES)
            ode_diff = float(np.max(np.abs(v / v_closed - 1.0)))
            assert ode_diff < 1e-6, f'ODE error too large: {ode_diff}'
            rows.append(dict(
                H=H, method='CosChaos', J=J,
                seconds=float(np.median(ts)),
                max_abs_err=float(100.0 * np.max(np.abs(v / v1 - 1.0))),
                ode_rel_diff=ode_diff,
            ))
        print('fwhegl done H =', H, flush=True)
    return rows


def molchan_unit(s, h):
    a = h - 0.5
    c = np.sqrt(2.0 * h * gamma(1.0 - a) / (gamma(1.0 + a) * gamma(1.0 - 2.0 * a)))
    return c * (((1.0 - s) / s) ** a) * hyp2f1(-a, 1.0, 1.0 + a, 1.0 - s)


def cell_coeff(h, m, t, order=12):
    def integral(a, b):
        return quad(molchan_unit, a, b, args=(h,), epsabs=1e-12, epsrel=2e-12, limit=200)[0]
    vals = np.empty(m, dtype=float)
    vals[0] = integral(0.0, 1.0 / m)
    if m > 1:
        vals[-1] = integral(1.0 - 1.0 / m, 1.0)
    if m > 2:
        x, w = roots_legendre(order)
        s = (np.arange(1, m - 1)[:, None] + (1.0 + x) * 0.5) / m
        vals[1:-1] = (molchan_unit(s, h) @ w) / (2.0 * m)
    return SIG * (t ** h) * math.sqrt(m) * vals


def bp_var(h, n):
    weights = [cell_coeff(h, round(n * t / T), t) for t in TIMES]
    v = np.array([math.exp(2.0 * MU * t) * math.expm1(np.log1p(w * w).sum()) for t, w in zip(TIMES, weights)])
    return v, weights[-1]


def task_discrete_wick(reps=3):
    rows = []
    loss_rows = []
    eq = hwc.Equation([X0], hwc.Poly([[(MU, (1,))]]), hwc.Poly([[(SIG, (1,))]]))
    for H in HS:
        ex = exact_var(TIMES, H)
        # Validate discrete Wick formula against exact 2^8 enumeration
        w8 = cell_coeff(H, 8, T)
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=8)))
        y8 = math.exp(MU * T) * np.prod(1.0 + signs * w8, axis=1)
        v8, _ = bp_var(H, 8)
        assert abs(y8.var() - v8[-1]) < 1e-12

        for n in (8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                v, _ = bp_var(H, n)
                ts.append(time.perf_counter() - t0)
            err = 100.0 * (v / ex - 1.0)
            rows.append(dict(
                H=H, method='BP discrete Wick', n=n,
                seconds=float(np.median(ts)),
                max_var_err=float(np.max(np.abs(err))),
                terminal_err=float(err[-1]),
            ))
        for J, p in ((5, 5), (17, 5), (33, 7), (65, 7)):
            ts = []
            for _ in range(reps):
                hfc._rules.clear()
                t0 = time.perf_counter()
                v = var_series(qJ(H, J, TIMES), p, TIMES)
                ts.append(time.perf_counter() - t0)
            err = 100.0 * (v / ex - 1.0)
            row = dict(
                H=H, method='CosChaos', J=J, p=p,
                seconds=float(np.median(ts)),
                max_var_err=float(np.max(np.abs(err))),
                terminal_err=float(err[-1]),
            )
            if J == 5:
                tgs = []
                for _ in range(reps):
                    hfc._rules.clear()
                    t0 = time.perf_counter()
                    sol = hwc.solve(eq, H, T, J, p, TIMES, rtol=1e-8, atol=1e-10)
                    dv = sol.variance()[:, 0]
                    tgs.append(time.perf_counter() - t0)
                assert np.max(np.abs(dv - v)) < 1e-6
                row['general_solver_s'] = float(np.median(tgs))
            rows.append(row)

        # First-cell energy loss diagnostic at n = 16384
        a = H - 0.5
        C = math.sqrt(2.0 * H * gamma(1.0 - a) / (gamma(1.0 + a) * gamma(1.0 - 2.0 * a)))
        weights = cell_coeff(H, 16384, T)
        q = (SIG ** 2) * (T ** (2.0 * H))
        qn = float(weights @ weights)
        def smooth(s):
            return (C * ((1.0 - s) ** a) * hyp2f1(-a, 1.0, 1.0 + a, 1.0 - s)) ** 2
        first = quad(smooth, 0.0, 1.0 / 16384, weight='alg', wvar=(-2.0 * a, 0.0), epsabs=1e-12)[0] * q
        first_loss = first - float(weights[0] ** 2)
        driver_loss = q - qn
        loss_rows.append(dict(
            H=H, n=16384,
            driver_var_lost_pct=float(100.0 * driver_loss / q),
            first_cell_share_of_driver_loss=float(first_loss / driver_loss),
        ))
        print('discrete_wick done H =', H, flush=True)
    return dict(rows=rows, loss=loss_rows)


def task_h_sweep(reps=3):
    J, p = 65, 7
    omega = np.arange(J, dtype=float) * (math.pi / T)
    scale = np.full(J, math.sqrt(2.0 / T))
    scale[0] = 1.0 / math.sqrt(T)
    Hs = [round(float(h), 2) for h in np.linspace(0.55, 0.95, 9)]
    trials = []
    for _ in range(reps):
        hfc._rules.clear()
        t0 = time.perf_counter()
        def rhs(t, y):
            out = np.empty_like(y)
            out[0] = MU * y[0]
            out[1:] = SIG * scale * np.cos(omega * t)
            return out
        y0 = np.zeros(1 + J, dtype=float)
        y0[0] = 1.0
        sol = solve_ivp(rhs, (0.0, T), y0, t_eval=TIMES, method='DOP853', rtol=1e-10, atol=1e-12)
        assert sol.success
        amp = sol.y[0]
        z = sol.y[1:].T
        setup_s = time.perf_counter() - t0
        assert np.max(np.abs(amp - np.exp(MU * TIMES))) < 1e-8
        assert np.max(np.abs(z - SIG * hfc.Mcoef(J, T, TIMES))) < 1e-8
        qrows = []
        for H in Hs:
            tq = time.perf_counter()
            G = hfc.gram(H, T, J)
            q = np.einsum('ti,ij,tj->t', z, G, z)
            v = (amp ** 2) * sum((q ** k) / math.factorial(k) for k in range(1, p + 1))
            secs = time.perf_counter() - tq
            ex = exact_var(TIMES, H)
            qrows.append(dict(H=H, max_rel_err=float(100.0 * np.max(np.abs(v / ex - 1.0))), seconds=float(secs)))
        trials.append(dict(setup_s=float(setup_s), query_s=float(sum(r['seconds'] for r in qrows)), rows=qrows))
    return dict(
        J=J, p=p,
        setup_s=float(np.median([tr['setup_s'] for tr in trials])),
        query_s=float(np.median([tr['query_s'] for tr in trials])),
        rows=trials[0]['rows'],
    )


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        res = dict(
            params=dict(mu=MU, sigma=SIG, T=T, x0=X0, times=TIMES.tolist()),
            worked=task_worked(),
            condition=task_condition(),
            identity=task_identity(),
            check=task_check(),
            fwhegl=task_fwhegl(),
            discrete_wick=task_discrete_wick(),
            h_sweep=task_h_sweep(),
        )
    out_path = os.path.join(HERE, 'p1_results.json')
    json.dump(res, open(out_path, 'w'), indent=1)
    print('wrote', out_path)
