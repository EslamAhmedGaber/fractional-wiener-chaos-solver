"""Independent engine for the RL-Fourier fBm coordinates q_m(t) (constant H, 1/2 <= H < 1).

Written from the manuscript formulas only (Theorems std:thm:q, bound), not from the manuscript package code.
Kummer functions y(x) = M(a; b; i x) are obtained by integrating Kummer's ODE
    x y'' + (b - i x) y' - i a y = 0
from a convergent series start, together with the two primitives needed:
    Psi1(X) = int_0^X M(1-nu; 1; i x) dx              (chi_H moments)
    Phi(X)  = int_0^X x^{2 nu} M(1-nu; 1+nu; i x) dx   (Q_omega(t) = c omega^{-2H} Phi(omega t)).
"""
import numpy as np
from math import gamma as G, sqrt, pi
from scipy.integrate import solve_ivp
from scipy.special import roots_jacobi


def jacint(n, a, b, lo=0.0, hi=1.0):
    """Nodes/weights for int_lo^hi f(t) (hi-t)^a (t-lo)^b dt."""
    x, w = roots_jacobi(n, a, b)
    L = hi - lo
    return lo + L * (x + 1) / 2, w * (L / 2) ** (a + b + 1)


class Kummer:
    def __init__(self, nu, xmax, x0=1.0, nser=80, rtol=1e-13, atol=1e-15):
        a1, b1, a2, b2 = 1 - nu, 1.0, 1 - nu, 1 + nu
        self.nu = nu

        def ser(a, b):
            c = np.empty(nser)
            c[0] = 1.0
            for n in range(1, nser):
                c[n] = c[n - 1] * (a + n - 1) / ((b + n - 1) * n)
            return c
        c1, c2 = ser(a1, b1), ser(a2, b2)
        n = np.arange(nser)
        ip = (1j) ** n
        y1 = np.sum(c1 * ip * x0 ** n); d1 = np.sum((c1 * ip * n)[1:] * x0 ** (n[1:] - 1))
        P1 = np.sum(c1 * ip * x0 ** (n + 1) / (n + 1))
        y2 = np.sum(c2 * ip * x0 ** n); d2 = np.sum((c2 * ip * n)[1:] * x0 ** (n[1:] - 1))
        F2 = np.sum(c2 * ip * x0 ** (n + 2 * nu + 1) / (n + 2 * nu + 1))
        # series pieces on [0, x0] for evaluation below x0
        self._c1, self._c2, self._x0 = c1 * ip, c2 * ip, x0

        def rhs(x, s):
            Y1, D1, _, Y2, D2, _ = s
            return [D1, (1j * a1 * Y1 - (b1 - 1j * x) * D1) / x, Y1,
                    D2, (1j * a2 * Y2 - (b2 - 1j * x) * D2) / x, x ** (2 * nu) * Y2]
        self.sol = solve_ivp(rhs, (x0, xmax), np.array([y1, d1, P1, y2, d2, F2], complex),
                             method='DOP853', rtol=rtol, atol=atol, dense_output=True)
        assert self.sol.success, self.sol.message

    def _eval(self, x, comp):
        x = np.asarray(x, float); out = np.empty(x.shape, complex)
        lo = x < self._x0; hi = ~lo
        if hi.any():
            out[hi] = self.sol.sol(x[hi].ravel())[comp].reshape(x[hi].shape) if x[hi].ndim else self.sol.sol(x[hi])[comp]
        if lo.any():
            xl = x[lo]; n = np.arange(len(self._c1)); nu = self.nu
            P = xl[..., None] ** n
            if comp == 0: out[lo] = P @ self._c1
            elif comp == 2: out[lo] = (P * xl[..., None]) @ (self._c1 / (n + 1))
            elif comp == 3: out[lo] = P @ self._c2
            elif comp == 5: out[lo] = (P * xl[..., None] ** (2 * nu + 1)) @ (self._c2 / (n + 2 * nu + 1))
        return out

    def M1(self, x): return self._eval(x, 0)
    def Psi1(self, x): return self._eval(x, 2)
    def M2(self, x): return self._eval(x, 3)
    def Phi(self, x): return self._eval(x, 5)


def constants(H):
    nu = H - 0.5
    aH = sqrt(2 * H * G(1 + nu) * G(1 - nu) / G(1 - 2 * nu))
    bH = aH * G(1 - nu)
    CH = 2 * H * G(1 + nu) * G(2 - 2 * nu) / G(1 - nu)
    return nu, aH, bH, CH


class Basis:
    """Ordered RL-Fourier basis with N = 2K+1 members at Hurst index H."""

    def __init__(self, H, K, nt=None, kummer=None):
        self.H, self.K, self.N = H, K, 2 * K + 1
        nu, aH, bH, CH = constants(H)
        self.nu, self.aH, self.bH, self.CH, self.kappa = nu, aH, bH, CH, sqrt(CH)
        self.om = 2 * pi * np.arange(1, K + 1)
        self.lam = (1.0 + np.arange(1, K + 1)) ** nu
        self.km = kummer or Kummer(nu, 2 * pi * K * 1.0001 + 1)
        nt = nt or max(240, int(1.25 * 2 * pi * K) + 120)
        t, w = jacint(nt, 0.0, 2 * nu)
        Pm = self._phi_over_tnu(t)                      # nt x N
        self.Gram = (Pm * w[:, None]).T @ Pm
        L = np.linalg.cholesky(self.Gram)
        self.R = np.linalg.inv(L).T                     # e_m = sum_j R_jm phi_j
        self.cond = np.linalg.cond(self.Gram)
        # chi_H moments
        draw = np.empty(self.N)
        draw[0] = sqrt(1 - 2 * nu) * bH
        Ps = self.km.Psi1(self.om) / self.om
        draw[1::2] = sqrt(1 - 2 * nu) * self.lam * sqrt(2) * bH * Ps.real
        draw[2::2] = sqrt(1 - 2 * nu) * self.lam * sqrt(2) * bH * Ps.imag
        self.d = self.R.T @ draw
        self.rho = sqrt(max(1 - np.sum(self.d ** 2), 0.0)) if nu > 0 else 0.0
        self.c = aH ** 2 * G(1 - nu) / G(1 + nu)

    def _phi_over_tnu(self, t):
        t = np.asarray(t, float)
        P = np.empty(t.shape + (self.N,))
        P[..., 0] = self.bH
        Mv = self.km.M1(np.multiply.outer(t, self.om))
        P[..., 1::2] = self.lam * sqrt(2) * self.bH * Mv.real
        P[..., 2::2] = self.lam * sqrt(2) * self.bH * Mv.imag
        return P

    def e(self, t):
        t = np.asarray(t, float)
        return (t ** self.nu)[..., None] * (self._phi_over_tnu(t) @ self.R)

    def D(self, t):
        t = np.asarray(t, float); H = self.H
        Dm = np.empty(t.shape + (self.N,))
        Dm[..., 0] = self.c * t ** (2 * H) / (2 * H)
        Q = self.c * self.om ** (-2 * H) * self.km.Phi(np.multiply.outer(t, self.om))
        Dm[..., 1::2] = self.lam * sqrt(2) * Q.real
        Dm[..., 2::2] = self.lam * sqrt(2) * Q.imag
        return Dm

    def q(self, t, n=None):
        """Columns q_0..q_{n-1}(t) and the augmentation q_*(t) for the prefix of length n."""
        n = n or self.N
        Qm = self.D(t)[..., : self.N] @ self.R
        Qn = Qm[..., :n]
        if self.nu > 0:
            dn = self.d[:n]; rho = sqrt(max(1 - np.sum(dn ** 2), 0))
            qs = (self.kappa * np.asarray(t, float) - Qn @ dn) / rho
        else:
            qs = np.zeros(np.shape(t))
        return Qn, qs

    def vplus(self, t, n=None):
        Qn, qs = self.q(t, n)
        return np.asarray(t, float) ** (2 * self.H) - np.sum(Qn ** 2, -1) - qs ** 2
