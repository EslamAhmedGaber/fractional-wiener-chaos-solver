"""Harmonic Wiener chaos solver for equations driven by one fractional Brownian motion, 1/2 <= H < 1.

Equation (state x in R^d, one scalar fBm B_H on [0,T]):
    dX = F(t, X) dt + D(t, X) <> dB_H,   X(0) = x0,           (<> : Wick product)
optionally with a Caputo memory term on the left (scalar, linear, constant coefficient):
    dX + a D^alpha_C X dt = F dt + D <> dB_H.

The user supplies x0, F, D (polynomials in x with constant or time-dependent coefficients, or callables),
and H, T, J, p.  No solution kernel is supplied.  The solver
  1. builds the cosine Gram matrix G^H, its Cholesky factor L, and the driving functions
     h(t) (correlated form) or l(t) = L^T h(t) (orthonormal form);
  2. writes X_{J,p}(t) = sum_{|alpha|<=p} c_alpha(t) Phi_alpha(xi)  (normalised Hermite polynomials,
     xi = L^{-1} Z ~ N(0, I_J));
  3. generates the coefficient equations by the general law
        dc_alpha/dt = E[F(t, X_{J,p}) Phi_alpha] + sum_{j: alpha_j>0} sqrt(alpha_j) l_j(t) E[D(t, X_{J,p}) Phi_{alpha-e_j}];
  4. integrates them (DOP853), or, with memory, solves them degree by degree with the Green function of the
     memory operator obtained from its Laplace symbol;
  5. returns the coefficients, the mean c_0, the covariance sum_{alpha != 0} c_alpha c_alpha^T, and the series.

Projection engines (chosen from the equation data):
  * polynomial F, D of degree <= 3: exact projection with the Hermite product tensor (any p; size limited by
    the number of multi-indices);
  * callable F, D and p = 1: exact reduction to a d-dimensional Gaussian expectation (Gauss-Hermite), any J;
  * callable F, D and p >= 2: tensor Gauss-Hermite in J dimensions, only if the node count is below a limit.
Correlated form (coefficients independent of H) is available when F and D are affine in x (Wick-linear class);
moments are then quadratic forms in G^H for any H.
"""
import itertools, math, time
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
import hfc


# ----------------------------------------------------------------------------------------------------------
# noise coordinates
class HarmonicNoise:
    """Cosine (or sine) coordinates of fBm on [0,T]: Z_k = int h_k dB_H, Cov(Z) = G^H = L L^T."""

    def __init__(self, H, T, J, kind='cos'):
        self.H, self.T, self.J, self.kind = float(H), float(T), int(J), kind
        self.G = hfc.gram(H, T, kind, J)
        self.L = np.linalg.cholesky(self.G)

    def h(self, t):
        return hfc.h(self.kind, self.J, self.T, np.atleast_1d(np.asarray(t, float)))

    def ell(self, t):                       # rows: l(t)^T = h(t)^T L
        return self.h(t) @ self.L

    def m(self, t):                         # m_k(t) = int_0^t h_k
        return hfc.Mcoef(self.kind, self.J, self.T, np.atleast_1d(np.asarray(t, float)))

    def path(self, t, xi):                  # B_H^J(t) = m(t)^T L xi ; xi shape (n, J)
        return np.asarray(xi) @ (self.m(t) @ self.L).T


# ----------------------------------------------------------------------------------------------------------
# multi-indices (sorted tuples of coordinate labels, e.g. (0, 0, 3) = 2 e_0 + e_3) and Hermite algebra
class ChaosBasis:
    def __init__(self, J, p):
        self.J, self.p = int(J), int(p)
        self.idx = [a for n in range(p + 1) for a in itertools.combinations_with_replacement(range(J), n)]
        self.pos = {a: i for i, a in enumerate(self.idx)}
        self.n = len(self.idx)
        self.deg = np.array([len(a) for a in self.idx])
        self.lin = np.array([self.pos[(j,)] for j in range(J)]) if p >= 1 else np.zeros(0, int)
        T_, S_, Jj, Fa = [], [], [], []
        for i, a in enumerate(self.idx):
            for j in set(a):
                b = list(a); b.remove(j)
                T_.append(i); S_.append(self.pos[tuple(b)]); Jj.append(j); Fa.append(math.sqrt(a.count(j)))
        self.r_tgt, self.r_src = np.array(T_, int), np.array(S_, int)
        self.r_j, self.r_fac = np.array(Jj, int), np.array(Fa, float)

    def raise_noise(self, Dc, ell):
        """Projection of D <> (l . xi) onto Phi_alpha: sum_j sqrt(alpha_j) l_j D_{alpha - e_j}.  Dc: (n, d)."""
        w = (self.r_fac * ell[self.r_j])[:, None] * Dc[self.r_src]
        out = np.zeros_like(Dc)
        for k in range(Dc.shape[1]):
            out[:, k] = np.bincount(self.r_tgt, weights=w[:, k], minlength=self.n)
        return out

    def evaluate(self, xi):
        """Phi_alpha(xi) for xi of shape (m, J) -> (m, n)."""
        xi = np.atleast_2d(xi); m = xi.shape[0]
        he = np.empty((self.p + 1, m, self.J)); he[0] = 1.0
        if self.p >= 1: he[1] = xi
        for k in range(1, self.p):
            he[k + 1] = (xi * he[k] - math.sqrt(k) * he[k - 1]) / math.sqrt(k + 1)
        V = np.ones((m, self.n))
        for i, a in enumerate(self.idx):
            for j in set(a):
                V[:, i] *= he[a.count(j), :, j]
        return V


def _c1(m, n, r):
    # Phi_m Phi_n = sum_r _c1(m,n,r) Phi_{m+n-2r}  (normalised probabilists' Hermite polynomials)
    return math.factorial(r) * math.comb(m, r) * math.comb(n, r) * math.sqrt(
        math.factorial(m + n - 2 * r) / (math.factorial(m) * math.factorial(n)))


_TENSORS = {}


def product_tensor(J, p):
    """Exact products Phi_a Phi_g = sum_nu V Phi_nu for |a|,|g| <= p, nu in the degree-2p set."""
    key = (J, p)
    if key in _TENSORS:
        return _TENSORS[key]
    B = ChaosBasis(J, p); B2 = ChaosBasis(J, 2 * p)
    cnt = [dict((j, a.count(j)) for j in set(a)) for a in B.idx]
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
                if d not in ck: val0 *= _c1(v, 0, 0)
            for d, v in ck.items():
                if d not in ci: val0 *= _c1(0, v, 0)
            for rr in itertools.product(*[range(min(ci[d], ck[d]) + 1) for d in common]):
                val = val0; nu = dict(base)
                for d, r in zip(common, rr):
                    val *= _c1(ci[d], ck[d], r); nu[d] -= 2 * r
                lab = tuple(sorted(itertools.chain.from_iterable([d] * v for d, v in nu.items() if v)))
                n_ = B2.pos[lab]
                I.append(i); K.append(k); N.append(n_); V.append(val)
                if i != k:
                    I.append(k); K.append(i); N.append(n_); V.append(val)
    out = dict(B=B, n2=B2.n, I=np.array(I, np.int32), K=np.array(K, np.int32), N=np.array(N, np.int32),
               V=np.array(V))
    # two sorted copies: by output nu (for products) and by eta (for the cube projection);
    # sorted segments are summed with np.add.reduceat, which is much faster than a random scatter
    for tag, keyarr in (('n', out['N']), ('k', out['K'])):
        o = np.argsort(keyarr, kind='stable')
        for f in 'IKNV':
            out[f + '_' + tag] = out[f][o]
        u_, st = np.unique(out[('N' if tag == 'n' else 'K') + '_' + tag], return_index=True)
        out['rows_' + tag], out['starts_' + tag] = u_, st
    nlow = int(np.searchsorted(out['N_n'], B.n))
    out['nlow'] = nlow; m = out['rows_n'] < B.n
    out['rows_low'], out['starts_low'] = out['rows_n'][m], out['starts_n'][m]
    for f in 'IKNV':
        del out[f]
    _TENSORS[key] = out
    return out


def _prod_full(Tt, u, v):          # coefficients of u*v in the degree-2p set (exact)
    w = Tt['V_n'] * u[Tt['I_n']] * v[Tt['K_n']]
    out = np.zeros(Tt['n2']); out[Tt['rows_n']] = np.add.reduceat(w, Tt['starts_n']); return out


def _prod_low(Tt, u, v):           # projection of u*v onto degree <= p
    k = Tt['nlow']; w = Tt['V_n'][:k] * u[Tt['I_n'][:k]] * v[Tt['K_n'][:k]]
    out = np.zeros(Tt['B'].n); out[Tt['rows_low']] = np.add.reduceat(w, Tt['starts_low']); return out


def _prod3_low(Tt, s_full, w):     # projection of s*w onto degree <= p, s given in the degree-2p set
    x = Tt['V_k'] * w[Tt['I_k']] * s_full[Tt['N_k']]
    out = np.zeros(Tt['B'].n); out[Tt['rows_k']] = np.add.reduceat(x, Tt['starts_k']); return out


# ----------------------------------------------------------------------------------------------------------
# equation data
class Poly:
    """Vector polynomial field: comp i = sum_terms coef(t) * prod_k x_k^{powers_k}; degree <= 3 per term.

    terms: list over components; each a list of (coef, powers) with coef a number or a function of t."""

    def __init__(self, terms):
        self.terms = [[(c, tuple(int(q) for q in pw)) for c, pw in comp] for comp in terms]
        self.d = len(self.terms)
        self.degree = max((sum(pw) for comp in self.terms for _, pw in comp), default=0)
        if self.degree > 3:
            raise ValueError('polynomial terms of degree > 3 are not implemented in the exact engine')

    def affine(self):
        return self.degree <= 1

    def __call__(self, t, X):               # evaluation at points X (m, d) -> (m, d)
        X = np.atleast_2d(X); out = np.zeros((X.shape[0], self.d))
        for i, comp in enumerate(self.terms):
            for c, pw in comp:
                cv = c(t) if callable(c) else c
                out[:, i] += cv * np.prod(X ** np.array(pw), axis=1)
        return out


@dataclass
class Equation:
    x0: object
    drift: object                    # Poly or callable(t, X[m,d]) -> [m,d]
    diffusion: object                # Poly or callable (Wick)
    memory: dict = None              # {'a': a, 'alpha': alpha}: Caputo term a D^alpha X on the left side
    name: str = 'equation'


# ----------------------------------------------------------------------------------------------------------
# projections of F and D onto the chaos basis
def _project_poly(P, t, c, B, Tt):
    n, d = c.shape; out = np.zeros((n, d)); cache = {}

    def prodpair(i, j):
        key = (min(i, j), max(i, j))
        if key not in cache:
            cache[key] = _prod_full(Tt, c[:, i], c[:, j])
        return cache[key]
    for comp, terms in enumerate(P.terms):
        for coef, pw in terms:
            cv = coef(t) if callable(coef) else coef
            if cv == 0: continue
            fac = [k for k, q in enumerate(pw) for _ in range(q)]
            if len(fac) == 0:
                out[0, comp] += cv
            elif len(fac) == 1:
                out[:, comp] += cv * c[:, fac[0]]
            elif len(fac) == 2:
                out[:, comp] += cv * (cache[(min(fac), max(fac))][:n] if (min(fac), max(fac)) in cache
                                      else _prod_low(Tt, c[:, fac[0]], c[:, fac[1]]))
            else:
                out[:, comp] += cv * _prod3_low(Tt, prodpair(fac[0], fac[1]), c[:, fac[2]])
    return out


# ---- degree-two engine (p = 2): exact products by the matrix formulas of the paper (Proposition on products up to
# degree two); elements are (u0, u, U) = u0 + u.xi + :xi^T U xi: with U symmetric.
def _p2_maps(B):
    if not hasattr(B, '_p2'):
        kk, ll, pos = [], [], []
        for i, a in enumerate(B.idx):
            if len(a) == 2:
                kk.append(a[0]); ll.append(a[1]); pos.append(i)
        B._p2 = (np.array(kk), np.array(ll), np.array(pos))
    return B._p2


def _to_mat(B, col):
    kk, ll, pos = _p2_maps(B); J = B.J
    U = np.zeros((J, J)); v = col[pos]
    diag = kk == ll
    U[kk[diag], kk[diag]] = v[diag] / math.sqrt(2)
    U[kk[~diag], ll[~diag]] = v[~diag] / 2; U[ll[~diag], kk[~diag]] = v[~diag] / 2
    return col[0], col[B.lin], U


def _from_mat(B, e0, e, E):
    kk, ll, pos = _p2_maps(B); out = np.zeros(B.n)
    out[0] = e0; out[B.lin] = e
    diag = kk == ll
    vals = np.where(diag, E[kk, ll] * math.sqrt(2), 2 * E[kk, ll])
    out[pos] = vals
    return out


def _mul2(x, y):
    u0, u, U = x; v0, v, V = y
    p0 = u0 * v0 + u @ v + 2 * np.sum(U * V)
    p = u0 * v + v0 * u + 2 * (U @ v + V @ u)
    UV = U @ V
    Pm = u0 * V + v0 * U + 0.5 * (np.outer(u, v) + np.outer(v, u)) + 2 * (UV + UV.T)
    return p0, p, Pm


def _cube2(x):
    c0, c, C = x
    e = _mul2(x, x); q0, q, Q = _mul2(e, x)
    C2 = C @ C; trC2 = np.trace(C2); Cc = C @ c
    q = q + 4 * trC2 * c + 8 * C @ Cc
    Q = Q + 2 * (np.outer(c, Cc) + np.outer(Cc, c)) + 2 * (c @ c) * C + 4 * trC2 * C + 8 * C2 @ C
    return q0, q, Q


def _lin(*terms):
    a0 = sum(w * t[0] for w, t in terms); a = sum(w * t[1] for w, t in terms); A = sum(w * t[2] for w, t in terms)
    return a0, a, A


def _trilinear2(x, y, z):          # symmetric trilinear form T(x,y,z) = Proj(xyz) by polarisation of the cube
    s = lambda *w: _cube2(_lin(*w))
    terms = [(1, s((1, x), (1, y), (1, z))), (-1, s((1, x), (1, y))), (-1, s((1, x), (1, z))),
             (-1, s((1, y), (1, z))), (1, _cube2(x)), (1, _cube2(y)), (1, _cube2(z))]
    return _lin(*[(w / 6, t) for w, t in terms])


def _project_poly_p2(Pn, t, c, B):
    n, d = c.shape; X = [_to_mat(B, c[:, i]) for i in range(d)]; out = np.zeros((n, d))
    for comp, terms in enumerate(Pn.terms):
        for coef, pw in terms:
            cv = coef(t) if callable(coef) else coef
            if cv == 0: continue
            fac = [k for k, q in enumerate(pw) for _ in range(q)]
            if len(fac) == 0:
                out[0, comp] += cv; continue
            if len(fac) == 1:
                out[:, comp] += cv * c[:, fac[0]]; continue
            if len(fac) == 2:
                r = _mul2(X[fac[0]], X[fac[1]])
            elif fac[0] == fac[1] == fac[2]:
                r = _cube2(X[fac[0]])
            else:
                r = _trilinear2(X[fac[0]], X[fac[1]], X[fac[2]])
            out[:, comp] += cv * _from_mat(B, *r)
    return out


def _gh(order, dim):
    x, w = np.polynomial.hermite_e.hermegauss(order); w = w / math.sqrt(2 * math.pi)
    if dim == 0:
        return np.zeros((1, 0)), np.ones(1)
    grids = np.meshgrid(*([np.arange(order)] * dim), indexing='ij')
    ii = np.stack([g.ravel() for g in grids], 1)
    return x[ii], np.prod(w[ii], 1)


def _project_callable_p1(Fun, t, c, order):
    """p = 1: X = c0 + C xi is Gaussian in R^d; project F onto 1 and xi_j exactly by a d-dim reduction."""
    c0 = c[0]; C = c[1:].T                                  # (d, J)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    r = int(np.sum(S > 1e-300 * max(1.0, S.max(initial=0.0))))
    z, w = _gh(order, r)
    X = c0[None, :] + (z @ (U[:, :r] * S[:r]).T if r else 0.0)
    Fv = np.asarray(Fun(t, X), float)
    out = np.zeros_like(c)
    out[0] = w @ Fv
    if r:
        Ez = (z * w[:, None]).T @ Fv                       # (r, d) : E[F zeta_k]
        out[1:] = Vt[:r].T @ Ez                            # E[F xi_j] = sum_k V_jk E[F zeta_k]
    return out


class _TensorGH:
    def __init__(self, B, order, max_nodes):
        if order ** B.J > max_nodes:
            raise NotImplementedError(
                f'tensor Gauss-Hermite needs {order}^{B.J} nodes (limit {max_nodes}); use a polynomial '
                f'representation of F and D, p = 1, or a smaller J')
        z, w = _gh(order, B.J); self.V = B.evaluate(z); self.w = w; self.PT = (self.V * w[:, None]).T

    def __call__(self, Fun, t, c):
        return self.PT @ np.asarray(Fun(t, self.V @ c), float)


# ----------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    times: np.ndarray
    coef: np.ndarray                   # (nt, n, d): orthonormal coefficients (or correlated ones if form=='correlated')
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
        """Orthonormal coefficients (degree <= 2) from correlated ones for the given H."""
        if self.form == 'orthonormal':
            return self.coef
        return correlated_to_orthonormal(self, H)

    def series(self, xi):
        """Evaluate X_{J,p}(t) at Gaussian coordinates xi (m, J) -> (nt, m, d)."""
        c = self.orthonormal()
        return np.einsum('ma,tas->tms', self.basis.evaluate(xi), c)


def _gram_for(res, H):
    if H is None or abs(H - res.noise.H) == 0:
        return res.noise.G
    return hfc.gram(H, res.noise.T, res.noise.kind, res.noise.J)


def correlated_covariance(res, H=None):
    """Covariance of a correlated-form chaos of degree <= 2: a1^T G a1 + 2 tr(A G A' G) (Theorem 5)."""
    B = res.basis; G = _gram_for(res, H)
    if B.p > 2:
        raise NotImplementedError('correlated moments implemented for p <= 2')
    a1 = res.coef[:, B.lin, :]                                       # (nt, J, d)
    cov = np.einsum('tis,ij,tjr->tsr', a1, G, a1)
    if B.p == 2:
        A = _sym2(res.coef, B)                                       # (nt, d, J, J)
        cov = cov + 2 * np.einsum('tsij,jk,trkl,li->tsr', A, G, A, G)
    return cov


def _sym2(coef, B):
    """Degree-2 normalised Wick coefficients -> symmetric matrices A with  sum a P = :Z^T A Z:."""
    nt, n, d = coef.shape; J = B.J
    A = np.zeros((nt, d, J, J))
    for i, a in enumerate(B.idx):
        if len(a) != 2: continue
        k, l = a
        if k == l:
            A[:, :, k, k] = coef[:, i, :] / math.sqrt(2)
        else:
            A[:, :, k, l] = A[:, :, l, k] = coef[:, i, :] / 2
    return A


def correlated_to_orthonormal(res, H=None):
    B = res.basis; G = _gram_for(res, H); L = np.linalg.cholesky(G)
    out = np.zeros_like(res.coef); out[:, 0] = res.coef[:, 0]
    out[:, B.lin, :] = np.einsum('ij,tis->tjs', L, res.coef[:, B.lin, :])        # c = L^T a
    if B.p == 2:
        A = _sym2(res.coef, B); C = np.einsum('ki,tskl,lj->tsij', L, A, L)          # C = L^T A L
        for i, a in enumerate(B.idx):
            if len(a) != 2: continue
            k, l = a
            out[:, i, :] = C[:, :, k, k] * math.sqrt(2) if k == l else 2 * C[:, :, k, l]
    elif B.p > 2:
        raise NotImplementedError
    return out


# ----------------------------------------------------------------------------------------------------------
def solve(eq, H, T, J, p, times, form='orthonormal', kind='cos', rtol=1e-11, atol=1e-13,
          quad_order=None, max_nodes=200000, max_coefficients=600000, memory_opts=None, engine='auto', gram=None):
    """Solve the truncated chaos system for `eq`; returns a Result."""
    t0 = time.perf_counter()
    x0 = np.atleast_1d(np.asarray(eq.x0, float)); d = len(x0)
    times = np.asarray(times, float)
    if np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > T * (1 + 1e-14):
        raise ValueError('output times must increase within [0, T]')
    if not (0.5 <= H < 1):
        raise ValueError('the construction covers 1/2 <= H < 1')
    nB = math.comb(J + p, p)
    if nB > max_coefficients:
        raise MemoryError(f'{nB} coefficients per state component exceed the limit {max_coefficients}')
    noise = HarmonicNoise(H, T, J, kind); B = ChaosBasis(J, p)
    if gram is not None:                      # e.g. the diagonal approximation of G^H
        noise.G = np.asarray(gram, float); noise.L = np.linalg.cholesky(noise.G)
    polyF, polyD = isinstance(eq.drift, Poly), isinstance(eq.diffusion, Poly)
    if form == 'correlated':
        if not (polyF and polyD and eq.drift.affine() and eq.diffusion.affine()):
            raise ValueError('the correlated (H-independent) form requires F and D affine in x')
    if eq.memory is not None:
        return _solve_memory(eq, noise, B, times, form, memory_opts or {}, t0)
    need_tensor = (polyF and eq.drift.degree >= 2) or (polyD and eq.diffusion.degree >= 2)
    use_p2 = need_tensor and p == 2 and engine != 'tensor'
    Tt = product_tensor(J, p) if (need_tensor and not use_p2) else None
    if Tt is not None:
        Tt['B'] = B
    gh = None
    if (not polyF or not polyD) and p >= 2:
        gh = _TensorGH(B, quad_order or (2 * p + 3), max_nodes)
    qo = quad_order or 20

    def proj(Fun, isP, t, c):
        if isP:
            if use_p2 and Fun.degree >= 2:
                return _project_poly_p2(Fun, t, c, B)
            return _project_poly(Fun, t, c, B, Tt)
        if p == 1:
            return _project_callable_p1(Fun, t, c, qo)
        return gh(Fun, t, c)
    drive = (lambda t: noise.h(t)[0]) if form == 'correlated' else (lambda t: noise.ell(t)[0])

    def rhs(t, y):
        c = y.reshape(B.n, d)
        f = proj(eq.drift, polyF, t, c)
        g = proj(eq.diffusion, polyD, t, c)
        return (f + B.raise_noise(g, drive(t))).ravel()
    y0 = np.zeros((B.n, d)); y0[0] = x0
    sol = solve_ivp(rhs, (0.0, float(times[-1])), y0.ravel(), t_eval=times, method='DOP853', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    C = sol.y.T.reshape(len(times), B.n, d)
    engname = (('degree-two matrix projection' if use_p2 else 'exact polynomial projection') if (polyF and polyD) else
              ('Gaussian reduction (p=1)' if p == 1 else 'tensor Gauss-Hermite'))
    return Result(times, C, B, noise, form, dict(nfev=sol.nfev, seconds=time.perf_counter() - t0, engine=engname,
                                                n_coefficients=B.n * d))


# ----------------------------------------------------------------------------------------------------------
# memory: X' + a D^alpha_C X = b X + f0 + (s X + g0) <> dB_H  (scalar, Wick-linear, constant coefficients)
def _talbot_green(a, alpha, b, tm, dps=30):
    import mpmath as mp
    mp.mp.dps = dps
    g = [1.0]; u = [1.0]; w = [0.0]
    for t in tm[1:]:
        g.append(float(mp.invertlaplace(lambda z: 1 / (z + a * z ** alpha - b), float(t), method='talbot')))
        u.append(float(mp.invertlaplace(lambda z: (1 + a * z ** (alpha - 1)) / (z + a * z ** alpha - b),
                                        float(t), method='talbot')))
        w.append(float(mp.invertlaplace(lambda z: 1 / (z * (z + a * z ** alpha - b)), float(t), method='talbot')))
    return np.array(g), np.array(u), np.array(w)


def _solve_memory(eq, noise, B, times, form, opt, t0):
    from scipy.interpolate import CubicSpline
    from scipy.special import roots_legendre
    if len(np.atleast_1d(eq.x0)) != 1 or B.p > 2:
        raise NotImplementedError('memory branch: scalar state and p <= 2')
    if not (isinstance(eq.drift, Poly) and isinstance(eq.diffusion, Poly) and eq.drift.affine()
            and eq.diffusion.affine()):
        raise NotImplementedError('memory branch: F and D affine in x with constant coefficients')

    def split(P):
        lin = sum(c for c, pw in P.terms[0] if sum(pw) == 1); cst = sum(c for c, pw in P.terms[0] if sum(pw) == 0)
        if callable(lin) or callable(cst): raise NotImplementedError('constant coefficients required')
        return float(lin), float(cst)
    b, f0 = split(eq.drift); s, g0 = split(eq.diffusion)
    a, alpha = eq.memory['a'], eq.memory['alpha']; x0 = float(np.atleast_1d(eq.x0)[0])
    T = noise.T; ngrid = opt.get('ngrid', 401); nq = opt.get('nq', 2000)
    tm = np.linspace(0, T, ngrid)
    g, u, w = _talbot_green(a, alpha, b, tm, opt.get('dps', 30))
    gS = CubicSpline(tm, g); a0S = CubicSpline(tm, x0 * u + f0 * w)
    drive = noise.h if form == 'correlated' else noise.ell
    xg, wg = roots_legendre(nq)
    C = np.zeros((len(times), B.n, 1))
    C[:, 0, 0] = a0S(times)
    for it, t in enumerate(times):                       # degree 1: c_j(t) = int_0^t g(t-v) l_j(v) [s c0(v) + g0] dv
        if t == 0: continue
        v = (xg + 1) / 2 * t; wv = wg * t / 2
        C[it, B.lin, 0] = (wv * gS(t - v) * (s * a0S(v) + g0)) @ drive(v)
    if B.p == 2:                                          # degree 2 by trapezoidal convolution on the grid
        hstep = tm[1] - tm[0]; dr = drive(tm)
        a1 = np.zeros((ngrid, noise.J))
        for i in range(1, ngrid):
            ww = np.full(i + 1, hstep); ww[0] = ww[-1] = hstep / 2
            a1[i] = (ww * g[i::-1] * (s * a0S(tm[:i + 1]) + g0)) @ dr[:i + 1]
        F2 = np.einsum('ti,tj->tij', dr, a1); F2 = 0.5 * (F2 + F2.transpose(0, 2, 1))   # symmetric forcing
        A2 = np.zeros_like(F2)
        for i in range(1, ngrid):
            ww = np.full(i + 1, hstep); ww[0] = ww[-1] = hstep / 2
            A2[i] = s * np.tensordot(ww * g[i::-1], F2[:i + 1], axes=(0, 0))
        idx = np.rint(times / hstep).astype(int)
        if np.max(np.abs(tm[idx] - times)) > 1e-9:
            raise ValueError('with p = 2 the output times must be memory-grid nodes')
        for i, al in enumerate(B.idx):                    # A = sum a P / normalisation of _sym2
            if len(al) != 2: continue
            k, l = al
            C[:, i, 0] = A2[idx, k, k] * math.sqrt(2) if k == l else 2 * A2[idx, k, l]
    return Result(times, C, B, noise, form, dict(seconds=time.perf_counter() - t0, engine='memory Green function',
                                                n_coefficients=B.n))


def third_central_moment(res):
    """E[(X - EX)^3] for a scalar orthonormal-form result (exact, uses the product tensor)."""
    B = res.basis
    if B.p == 1:
        return np.zeros(len(res.times))
    if B.p == 2:                                   # 6 c^T C c + 8 tr(C^3)
        out = []
        for row in res.orthonormal()[:, :, 0]:
            _, c, C = _to_mat(B, row); out.append(float(6 * c @ C @ c + 8 * np.trace(C @ C @ C)))
        return np.array(out)
    Tt = product_tensor(B.J, B.p); Tt['B'] = B
    out = []
    for row in res.orthonormal()[:, :, 0]:
        dd = row.copy(); dd[0] = 0.0
        s = _prod_full(Tt, dd, dd)
        out.append(float(np.dot(s[:B.n], dd)))
    return np.array(out)
