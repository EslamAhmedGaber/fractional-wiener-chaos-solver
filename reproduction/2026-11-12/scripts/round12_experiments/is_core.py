"""CosChaos-guided importance sampling for fBm-driven SDEs (exploratory; see PROTOCOL.md).

Discrete setting: increments dB ~ N(0, S) on n uniform steps (exact fGn covariance, Davies-Harte sampling).
Coordinates Z = A dB, A_ki = step average of the cosine function h_k; G_d = A S A^T = L_d L_d^T; xi = L_d^{-1} Z ~ N(0, I).
A shift xi -> xi + xi* is realised by dB' = dB + delta, delta = S A^T L_d^{-T} xi*, with likelihood ratio
w = exp(-xi*.xi - |xi*|^2/2) (xi = coordinates of the unshifted draw).
"""
import math, time
import numpy as np
from scipy.linalg import toeplitz, solve_triangular
from scipy.optimize import minimize
import hfc, hwc


def fgn_autocov(H, n):
    k = np.arange(n, dtype=float)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def fgn(H, n, M, dt, rng):
    """M exact fGn samples of length n (Davies-Harte), scaled to step dt (same construction as the paper's reference)."""
    k = np.arange(n + 1); g = 0.5 * ((k + 1.) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1.) ** (2 * H)); g[0] = 1.0
    lam = np.fft.fft(np.concatenate([g, g[-2:0:-1]])).real; assert lam.min() > -1e-9; lam = np.clip(lam, 0, None)
    m = 2 * n; Mh = (M + 1) // 2
    W = np.fft.fft(np.sqrt(lam / m) * (rng.standard_normal((Mh, m)) + 1j * rng.standard_normal((Mh, m))), axis=1)[:, :n]
    return np.concatenate([W.real, W.imag])[:M] * dt ** H


def heun(dB, dt, drift, x0, s):
    x = np.full(dB.shape[0], float(x0))
    for i in range(dB.shape[1]):
        xt = x + drift(x) * dt + s * dB[:, i]; x = x + 0.5 * (drift(x) + drift(xt)) * dt + s * dB[:, i]
    return x


class DiscreteCoords:
    """Exact discrete cosine coordinates of fBm increments on n steps of [0, T]."""

    def __init__(self, H, T, J, n):
        self.H, self.T, self.J, self.n = H, T, J, n; dt = T / n; self.dt = dt
        tg = np.linspace(0, T, n + 1)
        Mk = hfc.Mcoef('cos', J, T, tg)                       # (n+1, J): m_k(t) = int_0^t h_k
        self.A = (np.diff(Mk, axis=0) / dt).T                # (J, n): step averages of h_k
        self.S = toeplitz(fgn_autocov(H, n)) * dt ** (2 * H)  # exact increment covariance
        SA = self.S @ self.A.T                               # (n, J)
        self.Gd = self.A @ SA; self.Ld = np.linalg.cholesky(self.Gd)
        self.K = solve_triangular(self.Ld, SA.T, lower=True).T   # (n, J) = S A^T L_d^{-T}

    def xi(self, dB):
        return solve_triangular(self.Ld, self.A @ dB.T, lower=True).T

    def shift(self, xs):
        return self.K @ np.asarray(xs)


def coschaos_cubic(H, T, J, p, x0=1.0, a=-1.0, b=1.0, s=0.5):
    eq = hwc.Equation(x0=[x0], drift=hwc.Poly([[(a, (1,)), (-b, (3,))]]), diffusion=hwc.Poly([[(s, (0,))]]), name='cubic')
    t0 = time.perf_counter(); res = hwc.solve(eq, H, T, J, p, np.array([T]))
    return res, time.perf_counter() - t0


def design_point(res, c, sign=+1):
    """xi* = argmin |xi|^2 s.t. sign*(P(xi) - c) >= 0, P = CosChaos polynomial at the final time."""
    P = lambda x: float(res.series(np.atleast_2d(x))[-1, 0, 0])
    C = res.orthonormal()[-1, :, 0]; g = C[res.basis.lin]; c0 = C[0]
    x_lin = (c - c0) * g / (g @ g)
    if res.basis.p == 1:
        return x_lin, P(x_lin)
    cons = {'type': 'ineq', 'fun': lambda x: sign * (P(x) - c)}
    best = None
    for start in (x_lin, 1.2 * x_lin, 0.8 * x_lin):
        r = minimize(lambda x: 0.5 * x @ x, start, jac=lambda x: x, constraints=[cons], method='SLSQP',
                     options=dict(maxiter=500, ftol=1e-12))
        if sign * (P(r.x) - c) >= -1e-9 and (best is None or r.x @ r.x < best @ best):
            best = r.x
    if best is None:
        best = x_lin
    return best, P(best)


def is_estimate(H, T, n, coords, xs, levels, M, rng, drift, x0, s, chunk=4096, sign=+1):
    """IS estimates of P(sign*X(T) > sign*c) for each c in levels with one shift xs; also plain MC with the same rng stream
    when xs is zero. Returns sums for mean and variance."""
    xs = np.asarray(xs, float); delta = coords.shift(xs); half = 0.5 * xs @ xs
    L = np.asarray(levels, float); s1 = np.zeros(len(L)); s2 = np.zeros(len(L)); sw = 0.0; hits = np.zeros(len(L))
    done = 0; dt = T / n
    while done < M:
        m = min(chunk, M - done); dB = fgn(H, n, m, dt, rng)
        w = np.exp(-(coords.xi(dB) @ xs) - half)
        x = heun(dB + delta[None, :], dt, drift, x0, s)
        ind = (sign * x[:, None] > sign * L[None, :])
        v = ind * w[:, None]; s1 += v.sum(0); s2 += (v ** 2).sum(0); sw += w.sum(); hits += ind.sum(0); done += m
    mean = s1 / M; var = (s2 / M - mean ** 2) / (M - 1) * M / M
    return dict(p=mean, se=np.sqrt(np.maximum(s2 / M - mean ** 2, 0) / (M - 1)), w_mean=sw / M, hits=hits, M=M)
