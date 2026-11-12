"""experiment: likelihood-based Hurst inference from noisy records after ONE coefficient solve.

Model (the linear oscillator of the manuscript):
    dQ = V dt,  dV = (-w^2 Q - 2 zeta w V) dt + sigma dB_H,   (Q,V)(0) = (1,0),  T = 20.
Data set: M_REC = 20 independent records (repeated experiments on the same oscillator), each with
    n = 80 noisy displacement observations  y_i = Q(t_i) + tau * eps_i,  t_i = i/4.
Unknowns: theta = (H, sigma, tau).  The mechanical parameters are known.

CosChaos side.  One correlated degree-one solve with unit sigma gives the H-independent response matrix
A_J (n x J) and the mean; for every H the signal covariance is  sigma^2 A_J G^H A_J^T.
No coefficient equation is solved again for any H, sigma, tau, record or data set.

Independent side (no cosine object is used).
  * Records: exact fractional Gaussian noise on N_DATA uniform steps by circulant embedding (Davies-Harte),
    passed through the exact response of the oscillator to a piecewise-linear driver.
  * Reference likelihood: covariance W S(H) W^T of the same discretisation on N_REF steps, with S(H) the
    exact fGn covariance: a direct discretisation of the kernel double integral.

The first pilots (single record; velocity observations) are kept in evidence/round13/pilots: one record
of this length identifies H only to about +-0.1 even with the reference likelihood, so the experiment
uses repeated records.  That design decision was taken before the full run.

Output: evidence/round13/hurst_inference.json.   Run:  python hurst_inference.py [pilot|full]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
import numpy as np
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "solver_snapshot"))
import hfc, hwc  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "evidence", "round13")
os.makedirs(OUT, exist_ok=True)

W0, ZETA, SIGMA, TAU, T = 1.0, 0.3, 0.4, 0.02, 20.0
OBS = os.environ.get("CC_OBS", "Q")              # observed component: "Q" displacement or "V" velocity
N_OBS = int(os.environ.get("CC_NOBS", "80"))
M_REC = int(os.environ.get("CC_MREC", "20"))     # independent records per data set
T_OBS = np.arange(1, N_OBS + 1) * (T / N_OBS)
A_, B_ = ZETA * W0, W0 * np.sqrt(1 - ZETA ** 2)
H_GRID = np.round(np.arange(0.505, 0.9851, 0.005), 3)
RHO_GRID = np.logspace(-5.5, 0.5, 121)          # rho = tau^2 / sigma^2
J_LIST = tuple(int(v) for v in os.environ.get("CC_J", "17,33,65,129").split(","))
N_REF, N_REF_CHECK, N_DATA, N_TRUTH = 4000, 8000, 8000, 16000
H_TRUE = (0.6, 0.75, 0.9)


# ------------------------------------------------------------------ exact linear response (no cosines)
def mean_obs(t):
    """Mean of the observed component for (Q,V)(0) = (1,0)."""
    if OBS == "Q":
        return np.exp(-A_ * t) * (np.cos(B_ * t) + (A_ / B_) * np.sin(B_ * t))
    return -np.exp(-A_ * t) * np.sin(B_ * t) * (A_ ** 2 + B_ ** 2) / B_


def K_int(u):
    """int_0^u k(s) ds for the unit impulse response of the observed component to the force:
    Q: k = exp(-a s) sin(b s)/b;   V: k = d/ds of that, so the integral is exp(-a u) sin(b u)/b."""
    u = np.maximum(u, 0.0)
    if OBS == "Q":
        return (B_ - np.exp(-A_ * u) * (A_ * np.sin(B_ * u) + B_ * np.cos(B_ * u))) / (B_ * (A_ ** 2 + B_ ** 2))
    return np.exp(-A_ * u) * np.sin(B_ * u) / B_


def response_matrix(N):
    """W[i,m]: weight of the increment on cell m in the observation at t_i (piecewise-linear driver)."""
    s = np.linspace(0.0, T, N + 1)
    d = T / N
    return (K_int(T_OBS[:, None] - s[None, :-1]) - K_int(T_OBS[:, None] - s[None, 1:])) / d


def fgn_acov(H, N):
    k = np.arange(N, dtype=float)
    d = T / N
    return 0.5 * d ** (2 * H) * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def cov_reference(H, W):
    """W S(H) W^T with the Toeplitz S applied by FFT convolution."""
    N = W.shape[1]
    g = fgn_acov(H, N)
    ker = np.concatenate([g[:0:-1], g])                       # symmetric kernel, length 2N-1
    WS = fftconvolve(W, ker[None, :], mode="full", axes=1)[:, N - 1:2 * N - 1]
    C = WS @ W.T
    return 0.5 * (C + C.T)


def davies_harte(H, N, size, rng):
    """Exact fGn (step T/N) by circulant embedding; returns (size, N)."""
    g = fgn_acov(H, N + 1)
    c = np.concatenate([g, g[-2:0:-1]])                       # length 2N
    lam = np.fft.fft(c).real
    if lam.min() < -1e-10 * lam.max():
        raise RuntimeError("circulant embedding not nonnegative")
    lam = np.clip(lam, 0.0, None)
    M = len(c)
    z = rng.standard_normal((size, M)) + 1j * rng.standard_normal((size, M))
    x = np.fft.fft(z * np.sqrt(lam / M), axis=1)
    return x.real[:, :N]


# ------------------------------------------------------------------ CosChaos: one solve, every H
def coschaos_response(J):
    eq = hwc.Equation(x0=[1.0, 0.0],
                      drift=hwc.Poly([[(1.0, (0, 1))], [(-W0 ** 2, (1, 0)), (-2 * ZETA * W0, (0, 1))]]),
                      diffusion=hwc.Poly([[(0.0, (0, 0))], [(1.0, (0, 0))]]), name="oscillator, unit sigma")
    t0 = time.perf_counter()
    res = hwc.solve(eq, 0.75, T, J, 1, T_OBS, form="correlated")
    sec = time.perf_counter() - t0
    comp = 0 if OBS == "Q" else 1
    return res.coef[:, res.basis.lin, comp], res.coef[:, 0, comp], sec   # (n, J) response to Z_k, mean


def cov_coschaos(H, A):
    G = hfc.gram(H, T, "cos", A.shape[1])
    C = A @ G @ A.T
    return 0.5 * (C + C.T)


# ------------------------------------------------------------------ profile likelihood on the grids
def refine_parabola(x, f, i):
    """Vertex of the parabola through three grid points around the minimum, value and curvature."""
    if i == 0 or i == len(x) - 1:
        return x[i], f[i], np.nan
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    f0, f1, f2 = f[i - 1], f[i], f[i + 1]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (f1 - f0) + x1 * (f0 - f2) + x0 * (f2 - f1)) / den
    b = (x2 ** 2 * (f0 - f1) + x1 ** 2 * (f2 - f0) + x0 ** 2 * (f1 - f2)) / den
    c = f1 - a * x1 ** 2 - b * x1
    if a <= 0:
        return x1, f1, np.nan
    xv = -b / (2 * a)
    return xv, a * xv ** 2 + b * xv + c, 2 * a


class Likelihood:
    """Eigen-decomposition of the unit-sigma signal covariance on the H grid; then every fit is algebra."""

    def __init__(self, covfun):
        self.lam, self.U = [], []
        for H in H_GRID:
            lam, U = np.linalg.eigh(covfun(H))
            self.lam.append(np.clip(lam, 0.0, None)); self.U.append(U)
        self.lam = np.array(self.lam)                         # (nH, n)

    def fit(self, R):
        """R: residuals (ndata, m, n0), m independent records per data set sharing (H, sigma, tau).
        sigma^2 is profiled analytically, rho = tau^2/sigma^2 and H on grids with parabolic refinement.
        Returns per data set (H, sigma, tau, se_H from the profile curvature, nll) and the H profiles."""
        ndata, m, n0 = R.shape
        n = m * n0
        nll = np.empty((ndata, len(H_GRID))); rho_hat = np.empty_like(nll); s2_hat = np.empty_like(nll)
        lr = np.log(RHO_GRID)
        for ih in range(len(H_GRID)):
            S = ((R @ self.U[ih]) ** 2).sum(1)                 # (ndata, n0), summed over the m records
            d = self.lam[ih][None, :] + RHO_GRID[:, None]      # (nrho, n0)
            s2 = S @ (1.0 / d).T / n                           # (ndata, nrho)
            f = 0.5 * (n * np.log(s2) + m * np.log(d).sum(1)[None, :] + n * (1 + np.log(2 * np.pi)))
            j = np.argmin(f, axis=1)
            for r in range(ndata):
                lv, fv, _ = refine_parabola(lr, f[r], j[r])
                nll[r, ih] = fv; rho_hat[r, ih] = np.exp(lv)
                s2_hat[r, ih] = np.interp(lv, lr, s2[r])
        out = np.empty((ndata, 5))
        for r in range(ndata):
            i = int(np.argmin(nll[r]))
            Hv, fv, curv = refine_parabola(H_GRID, nll[r], i)
            rho = np.interp(Hv, H_GRID, rho_hat[r]); s2 = np.interp(Hv, H_GRID, s2_hat[r])
            out[r] = (Hv, np.sqrt(s2), np.sqrt(rho * s2), 1 / np.sqrt(curv) if curv == curv else np.nan, fv)
        return out, nll


# ------------------------------------------------------------------ experiment
def main(mode):
    ndata = 40 if mode == "pilot" else 200
    rec = {"setup": dict(observed=OBS, w=W0, zeta=ZETA, sigma=SIGMA, tau=TAU, T=T, n_obs=N_OBS,
                         n_datasets=ndata, records_per_dataset=M_REC, H_true=list(H_TRUE),
                         H_grid=[float(H_GRID[0]), float(H_GRID[-1]), 0.005], J=list(J_LIST),
                         N_ref=N_REF, N_ref_check=N_REF_CHECK, N_data=N_DATA, N_truth=N_TRUTH, mode=mode,
                         seed=20261111, blas_threads=1)}

    Wr, Wc, Wd, Wt = (response_matrix(N) for N in (N_REF, N_REF_CHECK, N_DATA, N_TRUTH))
    chk = []
    for H in (0.55, 0.75, 0.95):
        C1, C2 = cov_reference(H, Wr), cov_reference(H, Wc)
        chk.append(dict(H=H, max_rel_diff=float(np.max(np.abs(C1 - C2)) / np.max(np.abs(C2)))))
    rec["reference_grid_check"] = chk
    print("reference N=%d vs N=%d:" % (N_REF, N_REF_CHECK), chk, flush=True)

    # CosChaos: one coefficient solve per J, reused for everything below
    resp = {}
    rec["coefficient_solve_seconds"] = {}
    for J in J_LIST:
        A, mu, sec = coschaos_response(J)
        resp[J] = A; rec["coefficient_solve_seconds"][str(J)] = sec
        rec["mean_max_abs_diff_vs_exact_J%d" % J] = float(np.max(np.abs(mu - mean_obs(T_OBS))))
    print("coefficient solves (s):", rec["coefficient_solve_seconds"], flush=True)

    # work-precision of both covariance routes against the finest grid
    wp = []
    for H in H_TRUE:
        Ctrue = cov_reference(H, Wt); sc = np.max(np.abs(Ctrue))
        for J in J_LIST:
            ts = []
            for _ in range(15):
                t0 = time.perf_counter(); CJ = cov_coschaos(H, resp[J]); ts.append(time.perf_counter() - t0)
            wp.append(dict(H=H, route="coschaos", size=J, ms=1e3 * float(np.median(ts)),
                           max_rel=float(np.max(np.abs(CJ - Ctrue)) / sc)))
        for N in (125, 250, 500, 1000, 2000, 4000):
            Wn = response_matrix(N); ts = []
            for _ in range(7):
                t0 = time.perf_counter(); Cn = cov_reference(H, Wn); ts.append(time.perf_counter() - t0)
            wp.append(dict(H=H, route="direct", size=N, ms=1e3 * float(np.median(ts)),
                           max_rel=float(np.max(np.abs(Cn - Ctrue)) / sc)))
    rec["covariance_work_precision"] = wp
    for r in wp:
        print("  wp H=%.2f %-8s size %5d  %8.3f ms  max rel %.2e" % (r["H"], r["route"], r["size"], r["ms"], r["max_rel"]),
              flush=True)

    # likelihood tables on the H grid: Gram matrices and contractions only
    liks, tl = {}, {}
    t0 = time.perf_counter(); liks["ref"] = Likelihood(lambda H: cov_reference(H, Wr)); tl["ref"] = time.perf_counter() - t0
    for J in J_LIST:
        t0 = time.perf_counter(); liks["J%d" % J] = Likelihood(lambda H, A=resp[J]: cov_coschaos(H, A))
        tl["J%d" % J] = time.perf_counter() - t0
    rec["likelihood_table_seconds"] = tl
    rec["likelihood_table_H_values"] = len(H_GRID)
    print("likelihood tables for %d H values (s):" % len(H_GRID), tl, flush=True)

    rng = np.random.default_rng(20261111)
    res = {}
    for Ht in H_TRUE:
        ntot = ndata * M_REC
        Qc = np.concatenate([SIGMA * davies_harte(Ht, N_DATA, min(400, ntot - a), rng) @ Wd.T
                             for a in range(0, ntot, 400)])            # centred noise-free records
        R = (Qc + TAU * rng.standard_normal(Qc.shape)).reshape(ndata, M_REC, N_OBS)
        emp = (Qc.T @ Qc) / ntot / SIGMA ** 2
        Cref = cov_reference(Ht, Wc)
        sd = np.sqrt((Cref ** 2 + np.outer(np.diag(Cref), np.diag(Cref))) / ntot)
        entry = {"records_vs_reference_cov_max_abs_z": float(np.max(np.abs(emp - Cref) / sd)),
                 "records_vs_reference_cov_rms_z": float(np.sqrt(np.mean(((emp - Cref) / sd) ** 2))),
                 "n_records_total": ntot}
        fits = {}
        for name, lk in liks.items():
            t0 = time.perf_counter(); est, nll = lk.fit(R); sec = time.perf_counter() - t0
            fits[name] = est
            Hh, se = est[:, 0], est[:, 3]
            ok = np.isfinite(se)
            z = (Hh[ok] - Ht) / se[ok]
            entry[name] = dict(H_mean=float(Hh.mean()), H_bias=float(Hh.mean() - Ht), H_sd=float(Hh.std(ddof=1)),
                               H_rmse=float(np.sqrt(np.mean((Hh - Ht) ** 2))),
                               bias_over_se_of_mean=float((Hh.mean() - Ht) / (Hh.std(ddof=1) / np.sqrt(ndata))),
                               mean_reported_se=float(np.nanmean(se)), coverage95=float(np.mean(np.abs(z) < 1.96)),
                               n_interior=int(ok.sum()),
                               sigma_mean=float(est[:, 1].mean()), sigma_sd=float(est[:, 1].std(ddof=1)),
                               tau_mean=float(est[:, 2].mean()), tau_sd=float(est[:, 2].std(ddof=1)),
                               fit_seconds_all_datasets=sec, H_hat=[float(v) for v in Hh],
                               H_se=[float(v) if v == v else None for v in se])
            if name in ("ref", "J17", "J65"):
                entry[name]["profile_first_dataset"] = [float(v) for v in nll[0] - nll[0].min()]
        for J in J_LIST:
            dlt = fits["J%d" % J][:, 0] - fits["ref"][:, 0]
            entry["J%d" % J]["rms_diff_to_ref_estimate"] = float(np.sqrt(np.mean(dlt ** 2)))
            entry["J%d" % J]["max_abs_diff_to_ref_estimate"] = float(np.max(np.abs(dlt)))
            entry["J%d" % J]["mean_diff_to_ref_estimate"] = float(dlt.mean())
        res["%.2f" % Ht] = entry
        print("\nH_true = %.2f   records vs reference covariance: max|z| = %.2f, rms z = %.2f (%d records)"
              % (Ht, entry["records_vs_reference_cov_max_abs_z"], entry["records_vs_reference_cov_rms_z"], ntot))
        for name in liks:
            e = entry[name]
            print("  %-5s bias %+.4f (%+.1f se) sd %.4f rmse %.4f rep.se %.4f cov95 %.2f sigma %.4f tau %.4f dRef rms %s"
                  % (name, e["H_bias"], e["bias_over_se_of_mean"], e["H_sd"], e["H_rmse"], e["mean_reported_se"],
                     e["coverage95"], e["sigma_mean"], e["tau_mean"],
                     ("%.5f" % e["rms_diff_to_ref_estimate"]) if "rms_diff_to_ref_estimate" in e else "-"), flush=True)
    rec["H_grid_values"] = [float(v) for v in H_GRID]
    rec["results"] = res
    name = ("pilots/hurst_inference_pilot_%s%d_m%d.json" % (OBS, N_OBS, M_REC)) if mode == "pilot" else "hurst_inference.json"
    os.makedirs(os.path.join(OUT, "pilots"), exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, name), "w"), indent=1)
    print("\nwritten", name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
