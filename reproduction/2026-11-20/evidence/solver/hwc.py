"""CosChaos: Cosine Wiener chaos solver for fBm-driven SDEs (1/2 <= H < 1).

Equation (state x in R^d, one scalar fBm B_H on [0, T]):
    dX = F(t, X) dt + D(t, X) <> dB_H(t),    X(0) = x_0,
where <> denotes the Wick product (Wick-Ito-Skorohod interpretation).

Algorithmic highlights:
  * Closed-form cosine Gram matrix G^H (Proposition 1) and nested Cholesky
    factorisation G^H = L L^T with precomputed driver Ls = diag(scale) L so
    that ell(t) = cos(omega t) @ Ls in one BLAS-2 step.
  * Direct vectorised engines for polynomial systems of degree <= 3:
      - p = 1: direct (c0, c1) Gaussian-chaos engine for any d >= 1 (no tensor).
      - p = 2: direct symmetric matrix-layout engine (c0, c1, C) for any d >= 1,
               avoiding per-step packing/unpacking and bincount overhead, with
               specialised symmetric squaring _sq2 and BLAS-3 evaluation.
      - p >= 3: exact sparse Hermite product tensor with reduceat contraction.
      - Correlated H-free engine for Wick-affine equations (Proposition 7).
  * Fast BLAS-3 quadratic-form law evaluator for p <= 2 (chunked, low memory).
  * Exact d-dimensional Gauss-Hermite reduction for callable F, D at p = 1, and
    tensor Gauss-Hermite quadrature for callable F, D at p >= 2.
"""
import itertools
import math
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
import hfc


# ------------------------------------------------------------------------------
# Noise coordinates
class HarmonicNoise:
    """Cosine coordinates of fBm on [0, T]: Z_k = int h_k dB_H, Cov(Z) = G^H = L L^T."""

    def __init__(self, H, T, J, gram=None):
        self.H, self.T, self.J = float(H), float(T), int(J)
        self.G = hfc.gram(self.H, self.T, self.J) if gram is None else np.asarray(gram, dtype=float)
        self.L = np.linalg.cholesky(self.G)
        # Precomputed vectors/matrices for O(J^2) allocation-free scalar-time evaluation
        self._omega = np.arange(self.J, dtype=float) * (math.pi / self.T)
        self._scale = np.full(self.J, math.sqrt(2.0 / self.T), dtype=float)
        self._scale[0] = 1.0 / math.sqrt(self.T)
        self._Ls = self._scale[:, None] * self.L  # h(t) @ L == cos(omega * t) @ Ls

    def h(self, t):
        return hfc.h(self.J, self.T, np.atleast_1d(np.asarray(t, dtype=float)))

    def h_scalar(self, t):
        return self._scale * np.cos(self._omega * t)

    def ell(self, t):
        return self.h(t) @ self.L

    def ell_scalar(self, t):
        return np.cos(self._omega * t) @ self._Ls

    def m(self, t):
        return hfc.Mcoef(self.J, self.T, np.atleast_1d(np.asarray(t, dtype=float)))

    def path(self, t, xi):
        return np.asarray(xi, dtype=float) @ (self.m(t) @ self.L).T


# ------------------------------------------------------------------------------
# Multi-indices and Hermite chaos basis
class ChaosBasis:
    """Total-degree chaos basis A_{J,p} = {alpha in N_0^J : |alpha| <= p}."""

    def __init__(self, J, p):
        self.J, self.p = int(J), int(p)
        self.idx = [a for n in range(p + 1) for a in itertools.combinations_with_replacement(range(J), n)]
        self.pos = {a: i for i, a in enumerate(self.idx)}
        self.n = len(self.idx)
        self.deg = np.array([len(a) for a in self.idx], dtype=np.int32)
        self.lin = np.array([self.pos[(j,)] for j in range(J)], dtype=np.int32) if p >= 1 else np.zeros(0, np.int32)
        T_, S_, Jj, Fa = [], [], [], []
        for i, a in enumerate(self.idx):
            for j in set(a):
                b = list(a)
                b.remove(j)
                T_.append(i)
                S_.append(self.pos[tuple(b)])
                Jj.append(j)
                Fa.append(math.sqrt(a.count(j)))
        self.r_tgt = np.array(T_, dtype=np.int32)
        self.r_src = np.array(S_, dtype=np.int32)
        self.r_j = np.array(Jj, dtype=np.int32)
        self.r_fac = np.array(Fa, dtype=float)
        # Pre-sort raising indices by target for fast reduceat in general p >= 3
        if self.r_tgt.size:
            o = np.argsort(self.r_tgt, kind='stable')
            self._rt_tgt = self.r_tgt[o]
            self._rt_src = self.r_src[o]
            self._rt_j = self.r_j[o]
            self._rt_fac = self.r_fac[o]
            self._rt_rows, self._rt_starts = np.unique(self._rt_tgt, return_index=True)
        else:
            self._rt_rows = np.zeros(0, np.int32)
            self._rt_starts = np.zeros(0, np.int32)

    def raise_noise(self, Dc, ell):
        """Projection of D <> (ell . xi) onto Phi_alpha; Dc has shape (n, d)."""
        out = np.zeros_like(Dc)
        if self._rt_rows.size == 0:
            return out
        w = (self._rt_fac * ell[self._rt_j])[:, None] * Dc[self._rt_src]
        out[self._rt_rows] = np.add.reduceat(w, self._rt_starts, axis=0)
        return out

    def evaluate(self, xi):
        """Evaluate Phi_alpha(xi) for xi of shape (m, J) -> (m, n)."""
        xi = np.atleast_2d(np.asarray(xi, dtype=float))
        m = xi.shape[0]
        if not hasattr(self, '_tree'):
            par = np.zeros(self.n, dtype=np.int32)
            cc = np.zeros(self.n, dtype=np.int32)
            jj = np.zeros(self.n, dtype=np.int32)
            for i, a in enumerate(self.idx):
                if not a:
                    continue
                j = a[-1]
                c = a.count(j)
                par[i] = self.pos[a[:-c]]
                cc[i] = c
                jj[i] = j
            self._tree = [
                (np.flatnonzero(self.deg == d), par[self.deg == d], cc[self.deg == d], jj[self.deg == d])
                for d in range(1, self.p + 1)
            ]
        hn = np.empty((self.p + 1, self.J, m), dtype=float)
        hn[0] = 1.0
        if self.p >= 1:
            hn[1] = xi.T
        for k in range(1, self.p):
            hn[k + 1] = (xi.T * hn[k] - math.sqrt(k) * hn[k - 1]) / math.sqrt(k + 1)
        V = np.empty((self.n, m), dtype=float)
        V[0] = 1.0
        for lev, par, cc, jj in self._tree:
            V[lev] = V[par] * hn[cc, jj]
        return V.T


def _c1(m, n, r):
    return math.factorial(r) * math.comb(m, r) * math.comb(n, r) * math.sqrt(
        math.factorial(m + n - 2 * r) / (math.factorial(m) * math.factorial(n))
    )


_TENSORS = {}


def product_tensor(J, p):
    """Exact Hermite linearisation tensor Phi_a Phi_g = sum_nu V Phi_nu (|a|,|g| <= p)."""
    key = (J, p)
    if key in _TENSORS:
        return _TENSORS[key]
    B = ChaosBasis(J, p)
    B2 = ChaosBasis(J, 2 * p)
    cnt = [{j: a.count(j) for j in set(a)} for a in B.idx]
    I, K, N, V = [], [], [], []
    for i in range(B.n):
        ci = cnt[i]
        for k in range(i, B.n):
            ck = cnt[k]
            common = [d for d in ci if d in ck]
            base = dict(ci)
            for d, v in ck.items():
                base[d] = base.get(d, 0) + v
            val0 = 1.0
            for d, v in ci.items():
                if d not in ck:
                    val0 *= _c1(v, 0, 0)
            for d, v in ck.items():
                if d not in ci:
                    val0 *= _c1(0, v, 0)
            for rr in itertools.product(*[range(min(ci[d], ck[d]) + 1) for d in common]):
                val = val0
                nu = dict(base)
                for d, r in zip(common, rr):
                    val *= _c1(ci[d], ck[d], r)
                    nu[d] -= 2 * r
                lab = tuple(sorted(itertools.chain.from_iterable([d] * v for d, v in nu.items() if v)))
                n_ = B2.pos[lab]
                I.append(i); K.append(k); N.append(n_); V.append(val)
                if i != k:
                    I.append(k); K.append(i); N.append(n_); V.append(val)
    out = dict(B=B, n2=B2.n, I=np.array(I, np.int32), K=np.array(K, np.int32),
               N=np.array(N, np.int32), V=np.array(V, dtype=float))
    for tag, keyarr in (('n', out['N']), ('k', out['K'])):
        o = np.argsort(keyarr, kind='stable')
        for f in 'IKNV':
            out[f + '_' + tag] = out[f][o]
        u_, st = np.unique(out[('N' if tag == 'n' else 'K') + '_' + tag], return_index=True)
        out['rows_' + tag], out['starts_' + tag] = u_, st
    nlow = int(np.searchsorted(out['N_n'], B.n))
    out['nlow'] = nlow
    m = out['rows_n'] < B.n
    out['rows_low'], out['starts_low'] = out['rows_n'][m], out['starts_n'][m]
    for f in 'IKNV':
        del out[f]
    _TENSORS[key] = out
    return out


def _prod_full(Tt, u, v):
    w = Tt['V_n'] * u[Tt['I_n']] * v[Tt['K_n']]
    out = np.zeros(Tt['n2'], dtype=float)
    out[Tt['rows_n']] = np.add.reduceat(w, Tt['starts_n'])
    return out


def _prod_low(Tt, u, v):
    k = Tt['nlow']
    w = Tt['V_n'][:k] * u[Tt['I_n'][:k]] * v[Tt['K_n'][:k]]
    out = np.zeros(Tt['B'].n, dtype=float)
    out[Tt['rows_low']] = np.add.reduceat(w, Tt['starts_low'])
    return out


def _prod3_low(Tt, s_full, w):
    x = Tt['V_k'] * w[Tt['I_k']] * s_full[Tt['N_k']]
    out = np.zeros(Tt['B'].n, dtype=float)
    out[Tt['rows_k']] = np.add.reduceat(x, Tt['starts_k'])
    return out


# ------------------------------------------------------------------------------
# Equation data
class Poly:
    """Vector polynomial field: comp i = sum coef(t) * prod_k x_k^{powers_k}, degree <= 3."""

    def __init__(self, terms):
        self.terms = [[(c, tuple(int(q) for q in pw)) for c, pw in comp] for comp in terms]
        self.d = len(self.terms)
        self.degree = max((sum(pw) for comp in self.terms for _, pw in comp), default=0)
        if self.degree > 3:
            raise ValueError('polynomial terms of degree > 3 are not implemented in the exact engine')
        # Pre-compile flat term list for low-overhead RHS evaluation
        self._compiled = []
        for comp, cterms in enumerate(self.terms):
            for coef, pw in cterms:
                fac = tuple(k for k, q in enumerate(pw) for _ in range(q))
                self._compiled.append((comp, coef, callable(coef), fac))

    def affine(self):
        return self.degree <= 1

    def __call__(self, t, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros((X.shape[0], self.d), dtype=float)
        for comp, coef, is_call, fac in self._compiled:
            cv = coef(t) if is_call else coef
            if cv == 0:
                continue
            if not fac:
                out[:, comp] += cv
            else:
                prod = np.full(X.shape[0], cv, dtype=float)
                for k in fac:
                    prod *= X[:, k]
                out[:, comp] += prod
        return out


@dataclass
class Equation:
    x0: object
    drift: object        # Poly or callable(t, X[m, d]) -> [m, d]
    diffusion: object    # Poly or callable (Wick)
    name: str = 'equation'


# ------------------------------------------------------------------------------
# Degree-1 and Degree-2 matrix-layout algebra (direct, zero pack/unpack in ODE)
def _p2_maps(B):
    if not hasattr(B, '_p2'):
        kk, ll, pos = [], [], []
        for i, a in enumerate(B.idx):
            if len(a) == 2:
                kk.append(a[0]); ll.append(a[1]); pos.append(i)
        B._p2 = (np.array(kk, dtype=np.int32), np.array(ll, dtype=np.int32), np.array(pos, dtype=np.int32))
    return B._p2


def _to_mat(B, col):
    kk, ll, pos = _p2_maps(B)
    J = B.J
    U = np.zeros((J, J), dtype=float)
    v = col[pos]
    diag = (kk == ll)
    U[kk[diag], kk[diag]] = v[diag] * (1.0 / math.sqrt(2.0))
    U[kk[~diag], ll[~diag]] = v[~diag] * 0.5
    U[ll[~diag], kk[~diag]] = v[~diag] * 0.5
    return float(col[0]), col[B.lin].copy(), U


def _from_mat(B, e0, e, E):
    kk, ll, pos = _p2_maps(B)
    out = np.zeros(B.n, dtype=float)
    out[0] = e0
    out[B.lin] = e
    diag = (kk == ll)
    out[pos] = np.where(diag, E[kk, ll] * math.sqrt(2.0), 2.0 * E[kk, ll])
    return out


def _sq2(u0, u, U):
    """Exact degree-2 projection of X^2 for X = u0 + u.xi + :xi^T U xi: (U symmetric)."""
    Uu = U @ u
    U2 = U @ U  # symmetric since U is symmetric
    p0 = u0 * u0 + float(u @ u) + 2.0 * float(np.sum(U * U))
    p1 = (2.0 * u0) * u + 4.0 * Uu
    Pm = (2.0 * u0) * U + np.outer(u, u) + 4.0 * U2
    return p0, p1, Pm


def _mul2(u0, u, U, v0, v, V):
    """Exact degree-2 projection of X Y for symmetric U, V."""
    p0 = u0 * v0 + float(u @ v) + 2.0 * float(np.sum(U * V))
    p1 = u0 * v + v0 * u + 2.0 * (U @ v + V @ u)
    UV = U @ V
    uv = np.outer(u, v)
    Pm = u0 * V + v0 * U + 0.5 * (uv + uv.T) + 2.0 * (UV + UV.T)
    return p0, p1, Pm


def _cube2(c0, c, C):
    """Exact degree-2 projection of X^3 for X = c0 + c.xi + :xi^T C xi: (C symmetric)."""
    e0, e1, E = _sq2(c0, c, C)
    q0, q1, Q = _mul2(e0, e1, E, c0, c, C)
    C2 = C @ C
    trC2 = float(np.trace(C2))
    Cc = C @ c
    q1 = q1 + (4.0 * trC2) * c + 8.0 * (C @ Cc)
    cCc = np.outer(c, Cc)
    Q = Q + 2.0 * (cCc + cCc.T) + (2.0 * float(c @ c) + 4.0 * trC2) * C + 8.0 * (C2 @ C)
    return q0, q1, Q


def _trilinear2(x, y, z):
    """Exact degree-2 projection of X Y Z via polarisation of the cube."""
    def lin(*terms):
        a0 = sum(w * t[0] for w, t in terms)
        a1 = sum(w * t[1] for w, t in terms)
        A = sum(w * t[2] for w, t in terms)
        return a0, a1, A
    s = lambda *w: _cube2(*lin(*w))
    terms = [
        (1.0, s((1.0, x), (1.0, y), (1.0, z))),
        (-1.0, s((1.0, x), (1.0, y))),
        (-1.0, s((1.0, x), (1.0, z))),
        (-1.0, s((1.0, y), (1.0, z))),
        (1.0, _cube2(*x)),
        (1.0, _cube2(*y)),
        (1.0, _cube2(*z)),
    ]
    return lin(*[(w / 6.0, t) for w, t in terms])


def _proj_poly_p1_direct(P, t, c0, c1, o0, o1):
    """Accumulate degree-1 projection of Poly P into (o0[d], o1[d, J])."""
    for comp, coef, is_call, fac in P._compiled:
        cv = coef(t) if is_call else coef
        if cv == 0:
            continue
        nf = len(fac)
        if nf == 0:
            o0[comp] += cv
        elif nf == 1:
            a = fac[0]
            o0[comp] += cv * c0[a]
            o1[comp] += cv * c1[a]
        elif nf == 2:
            a, b = fac
            o0[comp] += cv * (c0[a] * c0[b] + float(c1[a] @ c1[b]))
            o1[comp] += cv * (c0[a] * c1[b] + c0[b] * c1[a])
        else:
            a, b, c = fac
            ab = float(c1[a] @ c1[b])
            ac = float(c1[a] @ c1[c])
            bc = float(c1[b] @ c1[c])
            o0[comp] += cv * (c0[a] * c0[b] * c0[c] + c0[a] * bc + c0[b] * ac + c0[c] * ab)
            o1[comp] += cv * ((c0[b] * c0[c] + bc) * c1[a] + (c0[a] * c0[c] + ac) * c1[b] + (c0[a] * c0[b] + ab) * c1[c])


def _proj_poly_p2_direct(P, t, c0, c1, C, o0, o1, oC):
    """Accumulate degree-2 projection of Poly P directly in symmetric matrix layout."""
    for comp, coef, is_call, fac in P._compiled:
        cv = coef(t) if is_call else coef
        if cv == 0:
            continue
        nf = len(fac)
        if nf == 0:
            o0[comp] += cv
        elif nf == 1:
            a = fac[0]
            o0[comp] += cv * c0[a]
            o1[comp] += cv * c1[a]
            oC[comp] += cv * C[a]
        elif nf == 2:
            a, b = fac
            if a == b:
                r0, r1, rC = _sq2(c0[a], c1[a], C[a])
            else:
                r0, r1, rC = _mul2(c0[a], c1[a], C[a], c0[b], c1[b], C[b])
            o0[comp] += cv * r0
            o1[comp] += cv * r1
            oC[comp] += cv * rC
        else:
            a, b, c = fac
            if a == b == c:
                r0, r1, rC = _cube2(c0[a], c1[a], C[a])
            else:
                r0, r1, rC = _trilinear2((c0[a], c1[a], C[a]), (c0[b], c1[b], C[b]), (c0[c], c1[c], C[c]))
            o0[comp] += cv * r0
            o1[comp] += cv * r1
            oC[comp] += cv * rC


def _project_poly_tensor(P, t, c, B, Tt):
    """General degree-p exact projection using the Hermite linearisation tensor."""
    n, d = c.shape
    out = np.zeros((n, d), dtype=float)
    cache = {}

    def prodpair(i, j):
        key = (min(i, j), max(i, j))
        if key not in cache:
            cache[key] = _prod_full(Tt, c[:, i], c[:, j])
        return cache[key]

    for comp, coef, is_call, fac in P._compiled:
        cv = coef(t) if is_call else coef
        if cv == 0:
            continue
        nf = len(fac)
        if nf == 0:
            out[0, comp] += cv
        elif nf == 1:
            out[:, comp] += cv * c[:, fac[0]]
        elif nf == 2:
            key = (min(fac), max(fac))
            out[:, comp] += cv * (cache[key][:n] if key in cache else _prod_low(Tt, c[:, fac[0]], c[:, fac[1]]))
        else:
            out[:, comp] += cv * _prod3_low(Tt, prodpair(fac[0], fac[1]), c[:, fac[2]])
    return out


# ------------------------------------------------------------------------------
# Gauss-Hermite projections for non-polynomial callables
def _gh(order, dim):
    x, w = np.polynomial.hermite_e.hermegauss(order)
    w = w / math.sqrt(2.0 * math.pi)
    if dim == 0:
        return np.zeros((1, 0), dtype=float), np.ones(1, dtype=float)
    grids = np.meshgrid(*([np.arange(order)] * dim), indexing='ij')
    ii = np.stack([g.ravel() for g in grids], axis=1)
    return x[ii], np.prod(w[ii], axis=1)


def _project_callable_p1(Fun, t, c, order):
    """Exact d-dimensional Gaussian reduction for callable F at p = 1."""
    c0 = c[0]
    C = c[1:].T  # (d, J)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    r = int(np.sum(S > 1e-300 * max(1.0, float(S.max(initial=0.0)))))
    z, w = _gh(order, r)
    X = c0[None, :] + (z @ (U[:, :r] * S[:r]).T if r else 0.0)
    Fv = np.asarray(Fun(t, X), dtype=float)
    out = np.zeros_like(c)
    out[0] = w @ Fv
    if r:
        Ez = (z * w[:, None]).T @ Fv
        out[1:] = Vt[:r].T @ Ez
    return out


class _TensorGH:
    def __init__(self, B, order, max_nodes):
        if order ** B.J > max_nodes:
            raise NotImplementedError(
                f'tensor Gauss-Hermite requires {order}^{B.J} nodes (limit {max_nodes}); '
                'use polynomial F and D, p = 1, or a smaller J'
            )
        z, w = _gh(order, B.J)
        self.V = B.evaluate(z)
        self.PT = (self.V * w[:, None]).T

    def __call__(self, Fun, t, c):
        return self.PT @ np.asarray(Fun(t, self.V @ c), dtype=float)


# ------------------------------------------------------------------------------
# Result container and fast law evaluator
@dataclass
class Result:
    times: np.ndarray
    coef: np.ndarray   # (nt, n, d): orthonormal coefficients (or sqrt(alpha!)-normalised correlated if form=='correlated')
    basis: ChaosBasis
    noise: HarmonicNoise
    form: str
    info: dict = field(default_factory=dict)

    @property
    def mean(self):
        return self.coef[:, 0, :]

    def covariance(self, H=None):
        if self.form == 'orthonormal':
            if H is not None and abs(H - self.noise.H) > 0:
                raise ValueError('orthonormal coefficients belong to one H; solve again or use the correlated form')
            c = self.coef[:, 1:, :]
            return np.einsum('tas,tar->tsr', c, c)
        return correlated_covariance(self, H)

    def variance(self, H=None):
        return np.einsum('tss->ts', self.covariance(H))

    def orthonormal(self, H=None):
        if self.form == 'orthonormal':
            return self.coef
        return correlated_to_orthonormal(self, H)

    def series(self, xi, chunk=65536):
        """Evaluate X_{J,p}(t, xi) for xi of shape (m, J) -> (nt, m, d).

        For p <= 2 uses a chunked BLAS-3 quadratic-form evaluator with O(chunk * J)
        memory; for p >= 3 evaluates the general Hermite polynomial basis.
        """
        xi = np.atleast_2d(np.asarray(xi, dtype=float))
        c = self.orthonormal()
        nt, _, d = c.shape
        m = xi.shape[0]
        p = self.basis.p
        J = self.basis.J
        if p == 0:
            return np.broadcast_to(c[:, 0:1, :], (nt, m, d)).copy()
        if p == 1:
            c0 = c[:, 0, :]            # (nt, d)
            c1 = c[:, self.basis.lin, :]  # (nt, J, d)
            out = np.empty((nt, m, d), dtype=float)
            for s in range(0, m, chunk):
                x = xi[s:s + chunk]
                out[:, s:s + chunk, :] = c0[:, None, :] + np.einsum('mj,tjd->tmd', x, c1)
            return out
        if p == 2:
            c0 = np.empty((nt, d), dtype=float)
            c1 = np.empty((nt, d, J), dtype=float)
            Cs = np.empty((nt, d, J, J), dtype=float)
            shift = np.empty((nt, d), dtype=float)
            for ti in range(nt):
                for di in range(d):
                    a0, a1, A = _to_mat(self.basis, c[ti, :, di])
                    c0[ti, di] = a0
                    c1[ti, di] = a1
                    Cs[ti, di] = A
                    shift[ti, di] = a0 - float(np.trace(A))
            out = np.empty((nt, m, d), dtype=float)
            c1_flat = c1.reshape(nt * d, J).T  # (J, nt*d)
            for s in range(0, m, chunk):
                x = xi[s:s + chunk]
                mm = x.shape[0]
                lin = (x @ c1_flat).reshape(mm, nt, d).transpose(1, 0, 2)  # (nt, mm, d)
                for ti in range(nt):
                    for di in range(d):
                        out[ti, s:s + mm, di] = shift[ti, di] + lin[ti, :, di] + np.einsum('ij,ij->i', x @ Cs[ti, di], x)
            return out
        # General p >= 3
        out = np.empty((nt, m, d), dtype=float)
        for s in range(0, m, chunk):
            V = self.basis.evaluate(xi[s:s + chunk])
            out[:, s:s + chunk, :] = np.einsum('ma,tad->tmd', V, c)
        return out


def _gram_for(res, H):
    if H is None or abs(H - res.noise.H) == 0:
        return res.noise.G
    return hfc.gram(H, res.noise.T, res.noise.J)


def _sym2(coef, B):
    """Degree-2 sqrt(alpha!)-normalised Wick coefficients -> symmetric matrices A."""
    nt, _, d = coef.shape
    J = B.J
    A = np.zeros((nt, d, J, J), dtype=float)
    kk, ll, pos = _p2_maps(B)
    diag = (kk == ll)
    v = coef[:, pos, :].transpose(0, 2, 1)  # (nt, d, n2)
    A[:, :, kk[diag], kk[diag]] = v[:, :, diag] * (1.0 / math.sqrt(2.0))
    A[:, :, kk[~diag], ll[~diag]] = v[:, :, ~diag] * 0.5
    A[:, :, ll[~diag], kk[~diag]] = v[:, :, ~diag] * 0.5
    return A


def correlated_covariance(res, H=None):
    """Covariance of a correlated-form chaos of degree <= 2: a1^T G a1 + 2 tr(A G A^T G)."""
    B = res.basis
    G = _gram_for(res, H)
    if B.p > 2:
        raise NotImplementedError('correlated_covariance implemented for p <= 2')
    a1 = res.coef[:, B.lin, :]
    cov = np.einsum('tis,ij,tjr->tsr', a1, G, a1)
    if B.p == 2:
        A = _sym2(res.coef, B)
        cov = cov + 2.0 * np.einsum('tsij,jk,trkl,li->tsr', A, G, A, G)
    return cov


def correlated_to_orthonormal(res, H=None):
    """Transform sqrt(alpha!)-normalised correlated coefficients (p <= 2) to orthonormal ones."""
    B = res.basis
    G = _gram_for(res, H)
    L = np.linalg.cholesky(G)
    out = np.zeros_like(res.coef)
    out[:, 0] = res.coef[:, 0]
    if B.p >= 1:
        out[:, B.lin, :] = np.einsum('ij,tis->tjs', L, res.coef[:, B.lin, :])
    if B.p == 2:
        A = _sym2(res.coef, B)
        C = np.einsum('ki,tskl,lj->tsij', L, A, L)
        kk, ll, pos = _p2_maps(B)
        diag = (kk == ll)
        vals = np.where(diag[None, None, :], C[:, :, kk, ll] * math.sqrt(2.0), 2.0 * C[:, :, kk, ll])
        out[:, pos, :] = vals.transpose(0, 2, 1)
    elif B.p > 2:
        raise NotImplementedError('correlated_to_orthonormal implemented for p <= 2')
    return out


# ------------------------------------------------------------------------------
# Main solver entry point (Algorithm 1)
def solve(eq, H, T, J, p, times, form='orthonormal', rtol=1e-9, atol=1e-11,
          method='DOP853', quad_order=None, max_nodes=200000, max_coefficients=600000,
          engine='auto', gram=None):
    """Solve the truncated CosChaos coefficient system for `eq`; returns a Result."""
    t0 = time.perf_counter()
    x0 = np.atleast_1d(np.asarray(eq.x0, dtype=float))
    d = len(x0)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > T * (1.0 + 1e-14):
        raise ValueError('output times must strictly increase within [0, T]')
    if not (0.5 <= H < 1.0):
        raise ValueError('H must satisfy 1/2 <= H < 1')
    nB = math.comb(J + p, p)
    if nB > max_coefficients:
        raise MemoryError(f'{nB} coefficients per component exceed limit {max_coefficients}')

    noise = HarmonicNoise(H, T, J, gram=gram)
    B = ChaosBasis(J, p)
    polyF, polyD = isinstance(eq.drift, Poly), isinstance(eq.diffusion, Poly)
    if form == 'correlated':
        if not (polyF and polyD and eq.drift.affine() and eq.diffusion.affine()):
            raise ValueError('the correlated (H-independent) form requires F and D affine in x')
    elif form != 'orthonormal':
        raise ValueError(f'unknown form: {form}')

    drive = noise.h_scalar if form == 'correlated' else noise.ell_scalar

    # ---- Fast direct p = 1 engine for polynomial F and D (any d >= 1)
    if polyF and polyD and p == 1 and engine != 'tensor':
        y0 = np.zeros(d * (1 + J), dtype=float)
        y0[:d] = x0
        o0 = np.empty(d, dtype=float)
        o1 = np.empty((d, J), dtype=float)
        g0 = np.empty(d, dtype=float)
        g1 = np.empty((d, J), dtype=float)

        def rhs_p1(t, y):
            c0 = y[:d]
            c1 = y[d:].reshape(d, J)
            o0.fill(0.0); o1.fill(0.0)
            g0.fill(0.0); g1.fill(0.0)
            _proj_poly_p1_direct(eq.drift, t, c0, c1, o0, o1)
            _proj_poly_p1_direct(eq.diffusion, t, c0, c1, g0, g1)
            ell = drive(t)
            o1[:] += g0[:, None] * ell[None, :]
            out = np.empty_like(y)
            out[:d] = o0
            out[d:] = o1.ravel()
            return out

        sol = solve_ivp(rhs_p1, (0.0, float(times[-1])), y0, t_eval=times, method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        Y = sol.y.T  # (nt, d + d*J)
        C = np.zeros((len(times), B.n, d), dtype=float)
        C[:, 0, :] = Y[:, :d]
        C[:, B.lin, :] = Y[:, d:].reshape(len(times), d, J).transpose(0, 2, 1)
        return Result(times, C, B, noise, form, dict(
            nfev=sol.nfev, seconds=time.perf_counter() - t0,
            engine='direct degree-one polynomial engine', n_coefficients=B.n * d,
        ))

    # ---- Fast direct symmetric-matrix p = 2 engine for polynomial F and D (any d >= 1)
    if polyF and polyD and p == 2 and engine != 'tensor':
        s1 = d + d * J
        y0 = np.zeros(s1 + d * J * J, dtype=float)
        y0[:d] = x0
        o0 = np.empty(d, dtype=float)
        o1 = np.empty((d, J), dtype=float)
        oC = np.empty((d, J, J), dtype=float)
        g0 = np.empty(d, dtype=float)
        g1 = np.empty((d, J), dtype=float)
        gC = np.empty((d, J, J), dtype=float)

        def rhs_p2(t, y):
            c0 = y[:d]
            c1 = y[d:s1].reshape(d, J)
            Cmat = y[s1:].reshape(d, J, J)
            o0.fill(0.0); o1.fill(0.0); oC.fill(0.0)
            g0.fill(0.0); g1.fill(0.0); gC.fill(0.0)
            _proj_poly_p2_direct(eq.drift, t, c0, c1, Cmat, o0, o1, oC)
            _proj_poly_p2_direct(eq.diffusion, t, c0, c1, Cmat, g0, g1, gC)
            ell = drive(t)
            o1[:] += g0[:, None] * ell[None, :]
            # Wick raising of g1.xi by ell.xi in symmetric-matrix convention :xi^T C xi:
            ge = g1[:, :, None] * ell[None, None, :]
            oC[:] += 0.5 * (ge + ge.transpose(0, 2, 1))
            out = np.empty_like(y)
            out[:d] = o0
            out[d:s1] = o1.ravel()
            out[s1:] = oC.ravel()
            return out

        sol = solve_ivp(rhs_p2, (0.0, float(times[-1])), y0, t_eval=times, method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        Y = sol.y.T
        nt = len(times)
        C = np.zeros((nt, B.n, d), dtype=float)
        C[:, 0, :] = Y[:, :d]
        C[:, B.lin, :] = Y[:, d:s1].reshape(nt, d, J).transpose(0, 2, 1)
        Cmat_out = Y[:, s1:].reshape(nt, d, J, J)
        kk, ll, pos = _p2_maps(B)
        diag = (kk == ll)
        vals = np.where(diag[None, None, :], Cmat_out[:, :, kk, ll] * math.sqrt(2.0), 2.0 * Cmat_out[:, :, kk, ll])
        C[:, pos, :] = vals.transpose(0, 2, 1)
        return Result(times, C, B, noise, form, dict(
            nfev=sol.nfev, seconds=time.perf_counter() - t0,
            engine='direct degree-two matrix engine', n_coefficients=B.n * d,
        ))

    # ---- General engine (p >= 3 polynomial, or callable F / D)
    need_tensor = (polyF and eq.drift.degree >= 2) or (polyD and eq.diffusion.degree >= 2)
    Tt = product_tensor(J, p) if need_tensor else None
    if Tt is not None:
        Tt['B'] = B
    gh = None
    if (not polyF or not polyD) and p >= 2:
        gh = _TensorGH(B, quad_order or (2 * p + 3), max_nodes)
    qo = quad_order or 20

    def proj(Fun, isP, t, c):
        if isP:
            return _project_poly_tensor(Fun, t, c, B, Tt)
        if p == 1:
            return _project_callable_p1(Fun, t, c, qo)
        return gh(Fun, t, c)

    def rhs_gen(t, y):
        c = y.reshape(B.n, d)
        f = proj(eq.drift, polyF, t, c)
        g = proj(eq.diffusion, polyD, t, c)
        return (f + B.raise_noise(g, drive(t))).ravel()

    y0 = np.zeros((B.n, d), dtype=float)
    y0[0] = x0
    sol = solve_ivp(rhs_gen, (0.0, float(times[-1])), y0.ravel(), t_eval=times, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    C = sol.y.T.reshape(len(times), B.n, d)
    engname = ('exact polynomial tensor engine' if (polyF and polyD) else
               ('Gaussian reduction (p=1)' if p == 1 else 'tensor Gauss-Hermite'))
    return Result(times, C, B, noise, form, dict(
        nfev=sol.nfev, seconds=time.perf_counter() - t0, engine=engname, n_coefficients=B.n * d,
    ))


def third_central_moment(res):
    """Exact third central moment E[(X - E X)^3] for a scalar result."""
    B = res.basis
    if B.p == 1:
        return np.zeros(len(res.times), dtype=float)
    if B.p == 2:
        out = []
        for row in res.orthonormal()[:, :, 0]:
            _, c, C = _to_mat(B, row)
            out.append(float(6.0 * (c @ C @ c) + 8.0 * np.trace(C @ C @ C)))
        return np.array(out, dtype=float)
    Tt = product_tensor(B.J, B.p)
    Tt['B'] = B
    out = []
    for row in res.orthonormal()[:, :, 0]:
        dd = row.copy()
        dd[0] = 0.0
        s = _prod_full(Tt, dd, dd)
        out.append(float(np.dot(s[:B.n], dd)))
    return np.array(out, dtype=float)
