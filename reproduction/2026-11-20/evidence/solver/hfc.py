"""Cosine coordinates of fractional Brownian motion (fBm) on [0, T] for 1/2 <= H < 1.

Implements Proposition 1 (closed-form cosine Gram matrix), Cholesky whitening,
and exact piecewise-linear path coordinates for the CosChaos solver.

Everything fractional enters through one confluent hypergeometric function:
    M_H(x) = 1F1(2H - 1; 2H; i x) = (2H - 1) int_0^1 u^{2H - 2} e^{i x u} du
           = e^{i x} - (i x / (2H)) N_H(x),
    N_H(x) = 1F1(2H; 2H + 1; i x) = 2H int_0^1 u^{2H - 1} e^{i x u} du.

Cosine basis on [0, T]:
    h_0(t) = T^{-1/2},
    h_k(t) = sqrt(2 / T) cos(k pi t / T),   k >= 1.
"""
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import roots_jacobi

_rules = {}


def _jacobi_rule(beta, n):
    key = (round(float(beta), 14), int(n))
    if key not in _rules:
        x, w = roots_jacobi(n, 0.0, beta)
        _rules[key] = ((x + 1.0) * 0.5, w / (2.0 ** (beta + 1.0)))
    return _rules[key]


def N_H(H, x, chunk=200000):
    """Evaluate N_H(x) = 1F1(2H; 2H+1; i x) via Gauss-Jacobi quadrature."""
    x = np.asarray(x, dtype=float)
    shp = x.shape
    xf = x.ravel()
    if xf.size == 0:
        return np.empty(shp, dtype=complex)
    n = int(max(80, 0.75 * float(np.max(np.abs(xf))) + 60))
    u, w = _jacobi_rule(2.0 * H - 1.0, n)
    out = np.empty(xf.size, dtype=complex)
    step = max(1, chunk // n)
    for a in range(0, xf.size, step):
        out[a:a + step] = 2.0 * H * (np.exp(1j * np.multiply.outer(xf[a:a + step], u)) @ w)
    return out.reshape(shp)


def M_H(H, x):
    """Evaluate M_H(x) = 1F1(2H-1; 2H; i x); equals 1 identically at H = 1/2."""
    x = np.asarray(x, dtype=float)
    if abs(H - 0.5) < 1e-15:
        return np.ones(x.shape, dtype=complex)
    return np.exp(1j * x) - 1j * x * N_H(H, x) / (2.0 * H)


def freqs(J):
    """Dimensionless cosine frequencies k * pi for k = 0, ..., J - 1."""
    return np.arange(J, dtype=float) * math.pi


def h(J, T, t):
    """Evaluate cosine basis functions h_0, ..., h_{J-1} at t."""
    t = np.asarray(t, dtype=float)
    omega = freqs(J) / T
    out = math.sqrt(2.0 / T) * np.cos(np.multiply.outer(t, omega))
    out[..., 0] = 1.0 / math.sqrt(T)
    return out


def Mcoef(J, T, t):
    """Integrated cosine basis m_k(t) = int_0^t h_k(s) ds (independent of H)."""
    t = np.asarray(t, dtype=float)
    omega = freqs(J) / T
    out = np.empty(t.shape + (J,), dtype=float)
    out[..., 0] = t / math.sqrt(T)
    if J > 1:
        out[..., 1:] = math.sqrt(2.0 / T) * np.sin(np.multiply.outer(t, omega[1:])) / omega[1:]
    return out


def gram(H, T, J):
    """Closed-form J x J cosine Gram matrix G^H of fBm on [0, T] (Proposition 1)."""
    if not (0.5 <= H < 1.0):
        raise ValueError('H must satisfy 1/2 <= H < 1')
    if abs(H - 0.5) < 1e-15:
        return np.eye(J, dtype=float)
    x = freqs(J)
    C = 2.0 * H * (T ** (2.0 * H - 1.0))
    m = M_H(H, x)
    p, r = m.real, m.imag
    G = np.zeros((J, J), dtype=float)
    if J > 1:
        k = np.arange(J, dtype=float)
        K, L = np.meshgrid(k, k, indexing='ij')
        rk, rl = np.meshgrid(r, r, indexing='ij')
        even = ((np.add.outer(np.arange(J), np.arange(J)) & 1) == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            off = (2.0 * C / math.pi) * (L * rl - K * rk) / (K * K - L * L)
        G = np.where(even & (K != L), off, 0.0)
        l_idx = np.arange(1, J)
        g0 = np.where((l_idx & 1) == 0, -(math.sqrt(2.0) * C / math.pi) * r[1:] / l_idx, 0.0)
        G[0, 1:] = g0
        G[1:, 0] = g0
        G[l_idx, l_idx] = C * (p[1:] - 2.0 * (1.0 - H) * r[1:] / (l_idx * math.pi))
    G[0, 0] = T ** (2.0 * H - 1.0)
    return G


def chol(G):
    """Lower-triangular Cholesky factor L with G = L L^T."""
    return np.linalg.cholesky(G)


def u_functions(L, Ht):
    """Orthonormal integrands u = L^{-1} h evaluated on time grid (rows: times)."""
    return solve_triangular(L, Ht.T, lower=True).T


def path_coords(B, T, J, L=None):
    """Exact cosine coordinates Z (and optionally whitened xi = L^{-1} Z) from a
    uniformly sampled piecewise-linear path B of shape (..., N + 1) on [0, T]."""
    B = np.asarray(B, dtype=float)
    n = B.shape[-1] - 1
    dt = T / n
    grid = np.linspace(0.0, T, n + 1)
    dm = np.diff(Mcoef(J, T, grid), axis=0) / dt  # (n, J)
    Z = np.diff(B, axis=-1) @ dm
    if L is None:
        return Z
    if Z.ndim == 1:
        return solve_triangular(L, Z, lower=True)
    return solve_triangular(L, Z.T, lower=True).T


def _E1(H, s, a):
    s = np.asarray(s, dtype=float)
    S = np.multiply.outer(s, np.ones_like(a))
    X = np.multiply.outer(s, a)
    return (S ** (2.0 * H)) * N_H(H, X) / (2.0 * H)


def Q(H, T, J, t, chunk_t=256):
    """Cross-covariance Q_k(t) = E[B_H(t) Z_k] in closed form."""
    t = np.asarray(t, dtype=float)
    if abs(H - 0.5) < 1e-15:
        return Mcoef(J, T, t)
    a = freqs(J) / T
    c = np.full(J, math.sqrt(2.0 / T), dtype=complex)
    c[0] = 1.0 / math.sqrt(T)
    E1T = _E1(H, np.array([T]), a)[0]
    out = np.empty(t.shape + (J,), dtype=float)
    tf = t.ravel()
    of = out.reshape(-1, J)
    for s in range(0, tf.size, chunk_t):
        tt = tf[s:s + chunk_t]
        Qa = H * (E1T[None, :] + np.exp(1j * np.multiply.outer(tt, a))
                  * (np.conj(_E1(H, tt, a)) - _E1(H, T - tt, a)))
        of[s:s + chunk_t] = np.real(c[None, :] * Qa)
    return out


def residual_var_classical(H, t, Mt, Qt, G):
    """Exact squared noise truncation error E|B_H(t) - B_H^J(t)|^2."""
    return (t ** (2.0 * H)) - 2.0 * np.einsum('tj,tj->t', Mt, Qt) + np.einsum('ti,ij,tj->t', Mt, G, Mt)
