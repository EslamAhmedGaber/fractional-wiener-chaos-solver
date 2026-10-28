"""General-degree Wiener chaos for dX = (a X - b X^3) dt + sig dB_H, X(0) = x0, on the cosine basis.

Orthonormal coordinates xi = L^{-1} Z (G = L L^T); normalized Hermite polynomials Phi_alpha(xi).
The cube is projected exactly onto degree <= p: the square X^2 is kept to degree 2p, then
E[Phi_eta X^3] = sum_{alpha,nu} c_alpha s_nu E[Phi_alpha Phi_eta Phi_nu].
One sparse tensor P[alpha, gamma -> nu] (Phi_alpha Phi_gamma = sum_nu P Phi_nu) serves both steps.
"""
import numpy as np
from math import factorial, comb, sqrt
import itertools
import hfc


def multi_indices(J, p):
    out = []
    def rec(prefix, left, d):
        if d == J:
            out.append(tuple(prefix)); return
        for k in range(left + 1):
            rec(prefix + [k], left - k, d + 1)
    rec([], p, 0)
    out.sort(key=lambda a: (sum(a), tuple(-x for x in a)))
    return out


def _coef1(m, n, r):
    return factorial(r) * comb(m, r) * comb(n, r) * sqrt(factorial(m + n - 2 * r) / (factorial(m) * factorial(n)))


def product_tensor(J, p):
    A = multi_indices(J, p); A2 = multi_indices(J, 2 * p)
    pos2 = {a: i for i, a in enumerate(A2)}
    I, K, N, V = [], [], [], []
    nz = [tuple(d for d in range(J) if a[d]) for a in A]
    for i, al in enumerate(A):
        for j in range(i, len(A)):
            ga = A[j]
            common = [d for d in nz[i] if ga[d]]
            ranges = [range(min(al[d], ga[d]) + 1) for d in common]
            base = [al[d] + ga[d] for d in range(J)]
            for rr in itertools.product(*ranges):
                nu = list(base); val = 1.0
                for d, r in zip(common, rr):
                    nu[d] -= 2 * r
                for d in range(J):
                    if al[d] or ga[d]:
                        r = rr[common.index(d)] if d in common else 0
                        val *= _coef1(al[d], ga[d], r)
                k = pos2[tuple(nu)]
                I.append(i); K.append(j); N.append(k); V.append(val)
                if i != j:
                    I.append(j); K.append(i); N.append(k); V.append(val)
    return A, A2, np.array(I), np.array(K), np.array(N), np.array(V)


def cube_projection(c, T_):
    A, A2, I, K, N, V = T_
    s = np.bincount(N, weights=V * c[I] * c[K], minlength=len(A2))
    return np.bincount(K, weights=V * c[I] * s[N], minlength=len(A))


def solve(H, J, p, pars, tout, T_=None, nsteps=None):
    """Fixed-step classical RK4 on the coefficient system; returns mean, variance, third central moment."""
    a, b, sig, x0, T = pars
    if T_ is None:
        T_ = product_tensor(J, p)
    A = T_[0]; nA = len(A)
    G = hfc.gram(H, T, 'cos', J); L = np.linalg.cholesky(G)
    lin = [A.index(tuple(int(d == j) for d in range(J))) for j in range(J)]
    def rhs(t, c):
        q = cube_projection(c, T_)
        dc = a * c - b * q
        dc[lin] += sig * (L.T @ hfc.h('cos', J, T, np.array([t]))[0])
        return dc
    if nsteps is None:
        nsteps = int(np.ceil(400 * T * max(1, J / 8)))
    # make every output time a grid point: nsteps must be a multiple of the common denominator of tout/T
    from fractions import Fraction
    from math import lcm
    den = 1
    for to in np.asarray(tout):
        den = lcm(den, Fraction(float(to) / T).limit_denominator(10000).denominator)
    nsteps = int(np.ceil(nsteps / den) * den)
    dt = T / nsteps
    c = np.zeros(nA); c[0] = x0
    tout = np.asarray(tout); outs = {}
    grid = np.linspace(0, T, nsteps + 1)
    want = {int(round(to / dt)): to for to in tout}
    assert all(abs(k * dt - to) < 1e-12 for k, to in want.items()), 'output time off the grid'
    if 0 in want: outs[want[0]] = c.copy()
    for n in range(nsteps):
        t = grid[n]
        k1 = rhs(t, c); k2 = rhs(t + dt / 2, c + dt / 2 * k1); k3 = rhs(t + dt / 2, c + dt / 2 * k2); k4 = rhs(t + dt, c + dt * k3)
        c = c + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if n + 1 in want: outs[want[n + 1]] = c.copy()
    C = np.array([outs[to] for to in tout])
    mean = C[:, 0]; var = np.sum(C[:, 1:] ** 2, 1)
    # third central moment: E[(X-m)^3] = sum_{alpha,gamma,eta != 0} c c c E[Phi Phi Phi]
    I, K, N, V = T_[2], T_[3], T_[4], T_[5]
    m3 = []
    nA2 = len(T_[1])
    for row in C:
        d = row.copy(); d[0] = 0.0
        s = np.bincount(N, weights=V * d[I] * d[K], minlength=nA2)   # (X-m)^2 in degree <= 2p
        m3.append(np.dot(s[:nA], d))                                     # E[(X-m)^2 (X-m)]
    return dict(t=tout, mean=mean, var=var, m3=np.array(m3), nA=nA, nsteps=nsteps)
