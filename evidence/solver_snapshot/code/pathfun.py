"""Path functionals of fBm from (i) RL-Fourier coordinates with certificates, (ii) Davies-Harte, (iii) discrete KL."""
import numpy as np
from math import sqrt, pi, log
from scipy.stats import qmc, norm
from scipy.special import erfcinv
from scipy.linalg import eigh
import rlf

SQ2PI = sqrt(2 / pi)


def coord_matrices(H, K, G, Ns):
    """Grid t_i = i/G; returns dict n -> (G+1) x (n+1) matrix [q_*, q_0..q_{n-1}] and the
    exact residual covariance quantities needed for certificates."""
    B = rlf.Basis(H, K)
    t = np.arange(G + 1) / G
    Qall = B.D(t) @ B.R
    out = {}
    for n in Ns:
        Qn = Qall[:, :n]
        if B.nu > 0:
            dn = B.d[:n]; rho = sqrt(1 - dn @ dn)
            qs = (B.kappa * t - Qn @ dn) / rho
        else:
            qs = np.zeros_like(t)
        M = np.column_stack([qs, Qn])
        # residual R = B_H - approx ; bridge residual Z(t) = R(t) - t R(1)
        v = t ** (2 * H) - np.sum(M ** 2, 1)
        cov1 = 0.5 * (t ** (2 * H) + 1 - np.abs(1 - t) ** (2 * H)) - M @ M[-1]
        vz = v - 2 * t * cov1 + t ** 2 * v[-1]
        out[n] = dict(M=M, v=np.maximum(v, 0), vz=np.maximum(vz, 0))
    return B, t, out


def range_of_bridge(P, t):
    Y = P - np.outer(P[:, -1], t)
    return Y.max(1) - Y.min(1)


def sobol_normals(dim, m_log2, seed):
    s = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(m_log2)
    s = np.clip(s, 1e-16, 1 - 1e-16)
    return norm.ppf(s)


def coord_ranges(M, t, m_log2, reps, seed0=0, batch=4096, ldp_shift=None):
    """RQMC samples of the grid range of the bridge built from coordinates. Returns reps x 2^m array."""
    d = M.shape[1]; res = []
    for r in range(reps):
        Z = sobol_normals(d, m_log2, seed0 + r)
        F = np.empty(len(Z))
        for a in range(0, len(Z), batch):
            P = Z[a:a + batch] @ M.T
            F[a:a + batch] = range_of_bridge(P, t)
        res.append(F)
    return np.array(res)


# ---------- certificates ----------
def eps_union(sds, delta):
    """Smallest eps with sum_i 2*Phi(-eps/sd_i) <= delta (bisection)."""
    sds = sds[sds > 0]
    if len(sds) == 0: return 0.0
    lo, hi = 0.0, sds.max() * 40
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.sum(2 * norm.sf(mid / sds)) > delta: lo = mid
        else: hi = mid
    return hi


def eps_cells(H, G, delta):
    """eps with P(max over G cells of sup_{s<=h}|B_H(t_i+s)-B_H(t_i)| > eps) <= delta, via
    stationary increments, self-similarity, Borell-TIS (sigma=1) and Sudakov-Fernique E sup B_H <= sqrt(2/pi)."""
    h = 1.0 / G
    x = SQ2PI + sqrt(2 * log(2 * G / delta))
    return x * h ** H


def eps_linear(G, delta):
    """eps with P(h |B_H(1)| > eps) <= delta."""
    return norm.isf(delta / 2) / G


# ---------- Davies-Harte ----------
def dh_setup(H, n):
    k = np.arange(n + 1)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.r_[g, g[-2:0:-1]]
    lam = np.fft.fft(c).real
    assert lam.min() > -1e-10, lam.min()
    return np.sqrt(np.maximum(lam, 0) / (2 * n))


def dh_paths(sq, n, H, m, rng):
    """m paths of fBm on grid i/n, i=0..n (exact Gaussian law on the grid)."""
    L = 2 * n
    W = rng.standard_normal((m, L)) + 1j * rng.standard_normal((m, L))
    X = np.fft.fft(W * sq, axis=1)[:, :n].real
    inc = X * n ** (-H)
    P = np.zeros((m, n + 1)); P[:, 1:] = np.cumsum(inc, 1)
    return P


def dh_ranges(H, n, total, seed=1, batch=1024):
    sq = dh_setup(H, n); rng = np.random.default_rng(seed); t = np.arange(n + 1) / n
    out = []
    for a in range(0, total, batch):
        P = dh_paths(sq, n, H, min(batch, total - a), rng)
        out.append(range_of_bridge(P, t))
    return np.concatenate(out)


# ---------- discrete KL ----------
def kl_matrix(H, G, nmax):
    t = np.arange(1, G + 1) / G
    C = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))
    w, V = eigh(C, subset_by_index=[G - nmax, G - 1])
    w, V = w[::-1], V[:, ::-1]
    M = np.vstack([np.zeros(nmax), V * np.sqrt(w)])
    return np.r_[0, t], M


# ---------- exact H = 1/2 ----------
def kuiper_sf(x, kmax=100):
    x = np.atleast_1d(x)[:, None]; k = np.arange(1, kmax + 1)[None, :]
    return np.clip(np.sum(2 * (4 * k ** 2 * x ** 2 - 1) * np.exp(-2 * k ** 2 * x ** 2), 1), 0, 1)
