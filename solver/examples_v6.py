"""Worked examples in the harmonic fractional chaos (cosine family on [0,T]).
Each example is solved in the correlated form (Z, G) and, independently, in the orthonormal form (xi = L^{-1} Z)."""
import numpy as np, mpmath as mp
from math import factorial, gamma as Gm
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre, roots_jacobi
from scipy.interpolate import CubicSpline
import hfc


def Rcov(H, u, v):
    return 0.5 * (u ** (2 * H) + v ** (2 * H) - np.abs(u - v) ** (2 * H))


# ---------- E2: fractional OU ----------
def ou_classical_coeffs(kind, J, T, t, lam, sig):
    """a_j(t) = sig int_0^t e^{-lam(t-s)} h_j(s) ds, closed form."""
    t = np.asarray(t, float); x = hfc.freqs(kind, J) / T
    E = (np.exp(1j * np.multiply.outer(t, x)) - np.exp(-lam * t)[:, None]) / (lam + 1j * x)[None, :]
    if kind == 'cos':
        A = np.sqrt(2 / T) * E.real; A[:, 0] = (1 - np.exp(-lam * t)) / lam / np.sqrt(T)
    else:
        A = np.sqrt(2 / T) * E.imag
    return sig * A


def ou_exact_var(H, lam, sig, t, n=300):
    if t == 0: return 0.0
    x, w = roots_legendre(n); s = (x + 1) / 2 * t; ws = w * t / 2; e = np.exp(-lam * (t - s))
    I1 = np.sum(ws * e * Rcov(H, t, s)); tot = 0.0
    for i in range(n):
        u = s[i]
        for lo, hi in [(0, u), (u, t)]:
            v = (x + 1) / 2 * (hi - lo) + lo; wv = w * (hi - lo) / 2
            tot += ws[i] * e[i] * np.sum(wv * np.exp(-lam * (t - v)) * Rcov(H, u, v))
    return sig ** 2 * (t ** (2 * H) - 2 * lam * I1 + lam ** 2 * tot)


# ---------- E4: memory model kernels (Caputo) ----------
def memory_kernels(a_=0.5, al=0.5, mu=0.0198, s=0.02, T=20.0, n=401):
    b = mu + s ** 2 / 2; mp.mp.dps = 30; tm = np.linspace(0, T, n)
    g = np.array([1.0] + [float(mp.invertlaplace(lambda z: 1 / (z + a_ * z ** al - b), float(t), method='talbot')) for t in tm[1:]])
    u = np.array([1.0] + [float(mp.invertlaplace(lambda z: (1 + a_ * z ** (al - 1)) / (z + a_ * z ** al - b), float(t), method='talbot')) for t in tm[1:]])
    return tm, g, u


def conv(gk, f, h):
    out = np.zeros_like(f)
    for i in range(1, len(gk)):
        w = np.full(i + 1, h); w[0] = w[-1] = h / 2
        out[i] = np.tensordot(w * gk[i::-1], f[:i + 1], axes=(0, 0))
    return out


def memory_chaos(H, J, tm, g, u, sig=0.02, X0=1.0):
    T = tm[-1]; G = hfc.gram(H, T, 'cos', J); hh = hfc.h('cos', J, T, tm); hstep = tm[1] - tm[0]
    a0 = X0 * u; a1 = sig * conv(g, a0[:, None] * hh, hstep)
    F2 = np.einsum('ti,tj->tij', hh, a1); F2 = 0.5 * (F2 + F2.transpose(0, 2, 1)); a2 = sig * conv(g, F2, hstep)
    L = np.linalg.cholesky(G)
    var1 = np.einsum('ti,ij,tj->t', a1, G, a1); var2 = 2 * np.einsum('tij,jk,tkl,li->t', a2, G, a2, G)
    c1 = a1 @ L; C2 = np.einsum('ki,tkl,lj->tij', L, a2, L)          # orthonormal coordinates
    return dict(mean=a0, var1=var1, var2=var2, var1_on=np.sum(c1 ** 2, 1), var2_on=2 * np.einsum('tij,tij->t', C2, C2),
                var1_diag=np.einsum('ti,i,ti->t', a1, np.diag(G), a1))


def fwhe57(H, t, tm, g, u, sig=0.02, X0=1.0):
    gS, uS = CubicSpline(tm, g), CubicSpline(tm, u); f = lambda v: sig * gS(t - v) * X0 * uS(v)
    x, w = roots_legendre(400); v = (x + 1) / 2 * t; wv = w * t / 2
    if H == 0.5: return float(np.sum(wv * f(v) ** 2))
    al = 2 * H - 1; y, wy = roots_jacobi(200, al - 1, 0); y = (y + 1) / 2; wy = wy / 2 ** al
    Jf = np.array([vv ** al / Gm(al) * np.sum(wy * f(vv * y)) for vv in v])
    return float(2 * H * Gm(2 * H) * np.sum(wv * f(v) * Jf))


# ---------- E5: population, correlated and orthonormal forms ----------
def population(H, J, p, form='correlated', pars=(0.5, 0.01, 0.02, 0.5, 30.0), tout=None, Gover=None):
    ag, eps, lmb, x0, TT = pars
    G = hfc.gram(H, TT, 'cos', J) if Gover is None else Gover
    L = np.linalg.cholesky(G)
    if form == 'correlated':
        M = G; drive = lambda t: hfc.h('cos', J, TT, np.array([t]))[0]
    else:  # orthonormal: moment matrix I, driving functions L^T h
        M = np.eye(J); drive = lambda t: L.T @ hfc.h('cos', J, TT, np.array([t]))[0]

    def rhs(t, y):
        a0 = y[0]; av = y[1:J + 1]; hv = drive(t)
        if p == 1:
            return np.r_[ag * a0 - eps * (a0 ** 2 + av @ M @ av), ag * av - eps * 2 * a0 * av + lmb * a0 * hv]
        A = y[J + 1:].reshape(J, J); AM = A @ M
        d0 = a0 ** 2 + av @ M @ av + 2 * np.trace(AM @ AM); d1 = 2 * a0 * av + 4 * A @ (M @ av)
        d2 = 2 * a0 * A + np.outer(av, av) + 4 * A @ M @ A; S2 = 0.5 * (np.outer(av, hv) + np.outer(hv, av))
        return np.r_[ag * a0 - eps * d0, ag * av - eps * d1 + lmb * a0 * hv, (ag * A - eps * d2 + lmb * S2).ravel()]
    y0 = np.zeros(1 + J + (J * J if p == 2 else 0)); y0[0] = x0
    sol = solve_ivp(rhs, (0, TT), y0, t_eval=tout, rtol=1e-10, atol=1e-12, method='DOP853')
    Y = sol.y.T; av = Y[:, 1:J + 1]; var = np.einsum('ti,ij,tj->t', av, M, av)
    if p == 2:
        A = Y[:, J + 1:].reshape(-1, J, J); var = var + 2 * np.einsum('tij,jk,tkl,li->t', A, M, A, M)
    return dict(t=sol.t, mean=Y[:, 0], var=var, a1=av)


def memory_var1_gauss(H, J, tlist, tm, g, u, sig=0.02, X0=1.0, nq=2000, T=None):
    """First-chaos variance with Gauss-Legendre coefficients (removes the grid error of conv)."""
    T = tm[-1] if T is None else T
    gS, uS = CubicSpline(tm, g), CubicSpline(tm, u); G = hfc.gram(H, T, 'cos', J); L = np.linalg.cholesky(G)
    x, w = roots_legendre(nq); out = {}
    for t in tlist:
        v = (x + 1) / 2 * t; wv = w * t / 2; a1 = (wv * sig * gS(t - v) * X0 * uS(v)) @ hfc.h('cos', J, T, v)
        out[float(t)] = dict(var1=float(a1 @ G @ a1), var1_on=float(np.sum((a1 @ L) ** 2)), var1_diag=float(np.sum(a1 ** 2 * np.diag(G))))
    return out
