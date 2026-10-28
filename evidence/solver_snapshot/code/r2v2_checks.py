"""Checks of the formulas added in Round 2 Version 2 (Sections 2.4 and 2.5.3).
(1) Table 3: degree-three Hermite elements are orthonormal, and the correlated Wick monomials of degree
    three are orthogonal to all polynomials of degree <= 2 (exact Gauss-Hermite quadrature, 3 coordinates).
(2) relation c_{2e_k} = sqrt2 C_kk, c_{e_k+e_l} = 2 C_kl between :Z^T A Z: and the orthonormal basis.
(3) u_3 = (h_3 - (G31/G11) h_1)/sqrt(G33 - G31^2/G11) equals the Cholesky construction (cosine, H=0.75, T=1).
(4) Wick growth: closed-form coefficients (eq:wickcoef) solve the master equation (J=5, p=3, H=0.75),
    and Var = X0^2 e^{2 mu t} sum_n (sigma^2 V_J)^n/n!.
(5) counterexample in the correlated coordinates: G00 = sqrt2 for J=1, T=2, H=0.75.
Writes ../results/r2v2_checks.json
"""
import json, os, sys, itertools, math
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
V12 = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'code', 'v12'))
sys.path.insert(0, V12)
import hfc  # noqa: E402

res = {}
# (1) ------------------------------------------------------------------
x, w = hermegauss(12); w = w / w.sum()
X = np.array(list(itertools.product(x, x, x))); Wt = np.prod(np.array(list(itertools.product(w, w, w))), axis=1)
He = {0: lambda z: 1 + 0 * z, 1: lambda z: z, 2: lambda z: z**2 - 1, 3: lambda z: z**3 - 3 * z}
idx = [a for a in itertools.product(range(4), repeat=3) if sum(a) <= 3]
Phi = np.array([np.prod([He[a[i]](X[:, i]) for i in range(3)], axis=0) / math.sqrt(np.prod([math.factorial(ai) for ai in a])) for a in idx])
Gram = (Phi * Wt) @ Phi.T
res['orthonormal_degree_le3_maxdev'] = float(np.abs(Gram - np.eye(len(idx))).max())
rng = np.random.default_rng(1); B = rng.normal(size=(3, 3)); G = B @ B.T + 3 * np.eye(3); Lc = np.linalg.cholesky(G)
Z = X @ Lc.T  # Z = L xi
k, l, m = 0, 1, 2
wick3 = {'Zk^3-3GkkZk': Z[:, k]**3 - 3 * G[k, k] * Z[:, k],
         'Zk^2Zl-GkkZl-2GklZk': Z[:, k]**2 * Z[:, l] - G[k, k] * Z[:, l] - 2 * G[k, l] * Z[:, k],
         'ZkZlZm-...': Z[:, k] * Z[:, l] * Z[:, m] - G[k, l] * Z[:, m] - G[k, m] * Z[:, l] - G[l, m] * Z[:, k]}
low = Phi[[i for i, a in enumerate(idx) if sum(a) <= 2]]
res['wick3_orthogonal_to_degree_le2_maxabs'] = {n: float(np.abs((low * Wt) @ v).max()) for n, v in wick3.items()}
# (2) ------------------------------------------------------------------
A = rng.normal(size=(3, 3)); A = A + A.T
wickq = Z @ A @ Z.T if False else np.einsum('ni,ij,nj->n', Z, A, Z) - np.trace(A @ G)
Cc = Lc.T @ A @ Lc
coef = {a: float((Phi[i] * Wt) @ wickq) for i, a in enumerate(idx) if sum(a) == 2}
pred = {}
for a in coef:
    nz = [i for i in range(3) if a[i] > 0]
    pred[a] = math.sqrt(2) * Cc[nz[0], nz[0]] if len(nz) == 1 else 2 * Cc[nz[0], nz[1]]
res['C_to_c_relation_maxdev'] = float(max(abs(coef[a] - pred[a]) for a in coef))
# (3) ------------------------------------------------------------------
H, T, J = 0.75, 1.0, 6
Gm = hfc.gram(H, T, 'cos', J); L = np.linalg.cholesky(Gm); Li = np.linalg.inv(L)
u3_formula = np.zeros(J); u3_formula[3] = 1; u3_formula[1] = -Gm[3, 1] / Gm[1, 1]; u3_formula /= math.sqrt(Gm[3, 3] - Gm[3, 1]**2 / Gm[1, 1])
res['u3_formula_vs_cholesky_maxdev'] = float(np.abs(u3_formula - Li[3]).max())
# (4) ------------------------------------------------------------------
H, T, J, p, mu, sig, X0 = 0.75, 2.0, 5, 3, 0.2, 0.5, 1.0
Gm = hfc.gram(H, T, 'cos', J); L = np.linalg.cholesky(Gm)
def h(t):
    kk = np.arange(J); return np.where(kk == 0, 1 / math.sqrt(T), math.sqrt(2 / T) * np.cos(kk * math.pi * t / T))
def mvec(t):
    kk = np.arange(J); return np.where(kk == 0, t / math.sqrt(T), math.sqrt(2 * T) * np.sin(kk * math.pi * t / T) / (kk * math.pi + (kk == 0)))
alphas = [a for a in itertools.product(range(p + 1), repeat=J) if sum(a) <= p]
pos = {a: i for i, a in enumerate(alphas)}
def rhs(t, c):
    ell = L.T @ h(t); dc = mu * c.copy()
    for a, i in pos.items():
        for j in range(J):
            if a[j] >= 1:
                b = list(a); b[j] -= 1
                dc[i] += sig * math.sqrt(a[j]) * ell[j] * c[pos[tuple(b)]]
    return dc
c0 = np.zeros(len(alphas)); c0[pos[(0,) * J]] = X0
sol = solve_ivp(rhs, (0, 1.0), c0, method='DOP853', rtol=1e-12, atol=1e-14, t_eval=[1.0])
cnum = sol.y[:, -1]; beta = L.T @ mvec(1.0)
cform = np.array([X0 * math.exp(mu) * sig**sum(a) * np.prod(beta**np.array(a)) / math.sqrt(np.prod([math.factorial(ai) for ai in a])) for a in alphas])
res['wick_growth_coef_closed_form_vs_ode_maxdev'] = float(np.abs(cnum - cform).max())
V = float(mvec(1.0) @ Gm @ mvec(1.0))
res['wick_growth_var_formula_vs_sum_c2'] = float(abs(np.sum(cnum**2) - cnum[pos[(0,) * J]]**2 - X0**2 * math.exp(2 * mu) * sum((sig**2 * V)**n / math.factorial(n) for n in range(1, p + 1))))
res['wick_growth_N_coefficients'] = len(alphas)
# (5) ------------------------------------------------------------------
res['G00_J1_T2_H075'] = float(hfc.gram(0.75, 2.0, 'cos', 1)[0, 0]); res['sqrt2'] = math.sqrt(2)
os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
json.dump(res, open(os.path.join(HERE, '..', 'results', 'r2v2_checks.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
