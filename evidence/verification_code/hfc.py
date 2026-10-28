"""Harmonic fractional chaos (HFC): classical cosine/sine bases on [0,T], standard fBm, 1/2 <= H < 1.

Everything fractional is carried by one confluent function
    M_H(x) = 1F1(2H-1; 2H; i x) = (2H-1) int_0^1 u^{2H-2} e^{ixu} du = e^{ix} - (ix/2H) N_H(x),
    N_H(x) = 1F1(2H; 2H+1; i x) = 2H int_0^1 u^{2H-1} e^{ixu} du.
Kernel: phi_H(u,v) = H(2H-1)|u-v|^{2H-2}; <f,g>_H = int int f g phi_H; E[int f dB_H int g dB_H] = <f,g>_H.

Families (k = index):
  'cos': h_0 = T^{-1/2}, h_k = sqrt(2/T) cos(k pi t/T), k >= 1    (frequencies x_k = k pi)
  'sin': h_k = sqrt(2/T) sin((k-1/2) pi t/T), k >= 1               (frequencies x_k = (k-1/2) pi)
"""
import numpy as np
from scipy.special import roots_jacobi

_rules = {}


def _rule(beta, n):
    key = (round(beta, 14), n)
    if key not in _rules:
        x, w = roots_jacobi(n, 0.0, beta)
        _rules[key] = ((x + 1) / 2, w / 2 ** (beta + 1))
    return _rules[key]


def N_H(H, x, chunk=200000):
    x = np.asarray(x, float); shp = x.shape; xf = x.ravel()
    n = int(max(80, 0.75 * (np.max(np.abs(xf)) if xf.size else 0) + 60))
    u, w = _rule(2 * H - 1, n)
    out = np.empty(xf.size, complex)
    step = max(1, chunk // n)
    for a in range(0, xf.size, step):
        out[a:a + step] = 2 * H * (np.exp(1j * np.multiply.outer(xf[a:a + step], u)) @ w)
    return out.reshape(shp)


def M_H(H, x):
    x = np.asarray(x, float)
    if abs(H - 0.5) < 1e-15:
        return np.ones(x.shape, complex)
    return np.exp(1j * x) - 1j * x * N_H(H, x) / (2 * H)


# ---------------- basis ----------------
def freqs(kind, J):
    if kind == 'cos':
        return np.arange(J) * np.pi
    if kind == 'sin':
        return (np.arange(1, J + 1) - 0.5) * np.pi
    raise ValueError(kind)


def h(kind, J, T, t):
    t = np.asarray(t, float); x = freqs(kind, J) / T
    if kind == 'cos':
        H_ = np.sqrt(2 / T) * np.cos(np.multiply.outer(t, x)); H_[..., 0] = 1 / np.sqrt(T); return H_
    return np.sqrt(2 / T) * np.sin(np.multiply.outer(t, x))


def Mcoef(kind, J, T, t):
    """Classical coefficients M_j(t) = int_0^t h_j (H-independent)."""
    t = np.asarray(t, float); x = freqs(kind, J) / T
    if kind == 'cos':
        out = np.empty(t.shape + (J,))
        out[..., 0] = t / np.sqrt(T)
        out[..., 1:] = np.sqrt(2 / T) * np.sin(np.multiply.outer(t, x[1:])) / x[1:]
        return out
    return np.sqrt(2 / T) * (1 - np.cos(np.multiply.outer(t, x))) / x


# ---------------- closed-form correlation matrix (one sequence) ----------------
def gram(H, T, kind, J):
    x = freqs(kind, J)
    C = 2 * H * T ** (2 * H - 1)
    if abs(H - 0.5) < 1e-15:
        return np.eye(J)
    m = M_H(H, x); p, r = m.real, m.imag
    G = np.zeros((J, J))
    if kind == 'cos':
        k = np.arange(J, dtype=float)
        K, L = np.meshgrid(k, k, indexing='ij')
        rk, rl = np.meshgrid(r, r, indexing='ij')
        even = (np.add.outer(np.arange(J), np.arange(J)) % 2 == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            off = (2 * C / np.pi) * (L * rl - K * rk) / (K ** 2 - L ** 2)
        G = np.where(even & (K != L), off, 0.0)
        # row/column 0
        l = np.arange(1, J)
        g0 = np.where(l % 2 == 0, -(np.sqrt(2) * C / np.pi) * r[1:] / l, 0.0)
        G[0, 1:] = g0; G[1:, 0] = g0
        G[0, 0] = T ** (2 * H - 1)
        kk = np.arange(1, J)
        G[kk, kk] = C * (p[1:] - 2 * (1 - H) * r[1:] / (kk * np.pi))
        return G
    k = np.arange(1, J + 1, dtype=float)
    K, L = np.meshgrid(k, k, indexing='ij')
    rk, rl = np.meshgrid(r, r, indexing='ij')
    diff_even = ((np.subtract.outer(np.arange(J), np.arange(J)) % 2) == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        A = (C / np.pi) * (rl - rk) / (K - L)
        B = (C / np.pi) * (rk + rl) / (K + L - 1)
    G = np.where(diff_even, A, B)
    G[np.arange(J), np.arange(J)] = C * (p - (2 * H - 1) * ((-1.0) ** (k - 1) - r) / x)
    return G


# ---------------- covariance with the path: Q_j(t) = E[B_H(t) Z_j] ----------------
def _E1(H, s, a):
    S = np.multiply.outer(np.asarray(s, float), np.ones_like(a)); X = np.multiply.outer(np.asarray(s, float), a)
    return S ** (2 * H) * N_H(H, X) / (2 * H)


def Q(H, T, kind, J, t, chunk_t=256):
    """Q_j(t) = H int_0^T h_j(v)[v^{2H-1} + sgn(t-v)|t-v|^{2H-1}] dv (closed form)."""
    t = np.asarray(t, float); a = freqs(kind, J) / T
    if kind == 'cos':
        c = np.where(np.arange(J) == 0, 1 / np.sqrt(T), np.sqrt(2 / T)).astype(complex)
    else:
        c = -1j * np.sqrt(2 / T) * np.ones(J)
    if abs(H - 0.5) < 1e-15:
        return Mcoef(kind, J, T, t)
    E1T = _E1(H, np.array([T]), a)[0]
    out = np.empty(t.shape + (J,))
    tf = t.ravel(); of = out.reshape(-1, J)
    for s in range(0, tf.size, chunk_t):
        tt = tf[s:s + chunk_t]
        Qa = H * (E1T[None, :] + np.exp(1j * np.multiply.outer(tt, a)) * (np.conj(_E1(H, tt, a)) - _E1(H, T - tt, a)))
        of[s:s + chunk_t] = np.real(c[None, :] * Qa)
    return out


def dQ(H, T, kind, J, t):
    """Q_j'(t) = int_0^T h_j(v) phi_H(t,v) dv (the noise-driving 'fractional harmonic' of the optimal projection)."""
    t = np.asarray(t, float); a = freqs(kind, J) / T
    if abs(H - 0.5) < 1e-15:
        return h(kind, J, T, t)
    c = (np.where(np.arange(J) == 0, 1 / np.sqrt(T), np.sqrt(2 / T)).astype(complex) if kind == 'cos'
         else -1j * np.sqrt(2 / T) * np.ones(J))
    S1 = np.multiply.outer(t, np.ones_like(a)); S2 = np.multiply.outer(T - t, np.ones_like(a))
    Mt = S1 ** (2 * H - 1) * M_H(H, np.multiply.outer(t, a)); MTt = S2 ** (2 * H - 1) * M_H(H, np.multiply.outer(T - t, a))
    return np.real(c[None, :] * H * np.exp(1j * np.multiply.outer(t, a)) * (np.conj(Mt) + MTt))


# ---------------- orthonormal transformation ----------------
def chol(G):
    return np.linalg.cholesky(G)


def u_functions(L, Ht):
    """u = L^{-1} h (rows: times): fractional-orthonormal combinations of the harmonics."""
    from scipy.linalg import solve_triangular
    return solve_triangular(L, Ht.T, lower=True).T


def residual_var_classical(H, t, Mt, Qt, G):
    """E|B_H(t) - M(t).Z|^2 = t^{2H} - 2 M.Q + M^T G M."""
    return t ** (2 * H) - 2 * np.einsum('tj,tj->t', Mt, Qt) + np.einsum('ti,ij,tj->t', Mt, G, Mt)


def residual_var_optimal(H, t, Qt, G):
    return t ** (2 * H) - np.einsum('tj,tj->t', Qt, np.linalg.solve(G, Qt.T).T)
