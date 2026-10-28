"""Re-solves the linear, Wick-type, memory and population examples with the general solver (hwc.solve).
Reference values that do not use the solver (exact OU variance by quadrature, FWHE formula, Ito Monte Carlo)
are copied from results/examples.json.  Writes results/v12_examples.json.
Run from the package root:  python code/run_examples_v12.py"""
import sys, json, time, math
import numpy as np
sys.path.insert(0, 'v6/code')
import hfc, hwc
import examples_v6 as E          # used only for the independent references and the hand-derived propagator check

P, Eq = hwc.Poly, hwc.Equation
OLD = json.load(open('v6/results/examples.json'))
OUT = {}
log = lambda *a: print(*a, flush=True)
t00 = time.time()

def corr_var(res, G, deg=None):
    """variance of a correlated-form result with a given Gram matrix (degree-1 [+ degree-2] parts)."""
    B = res.basis; a1 = res.coef[:, B.lin, 0]
    v1 = np.einsum('ti,ij,tj->t', a1, G, a1)
    if B.p < 2 or deg == 1:
        return v1, np.zeros_like(v1)
    A = hwc._sym2(res.coef, B)[:, 0]
    return v1, 2 * np.einsum('tij,jk,tkl,li->t', A, G, A, G)

# ---------------- OU: dX = -lam X dt + sig dB_H, T = 2 ----------------
T = 2.0; tq = np.linspace(0, T, 201)
ou = Eq(0.0, P([[(-1.0, (1,))]]), P([[(1.0, (0,))]]), name='OU')
OUT['ou'] = {}
for J in [5, 17, 65]:
    rc = hwc.solve(ou, 0.75, T, J, 1, tq, form='correlated')          # one H-independent solve per J
    for H in [0.5, 0.55, 0.75, 0.95]:
        rec = OUT['ou'].setdefault(str(H), dict(exact=OLD['ou'][str(H)]['exact']))
        G = hfc.gram(H, T, 'cos', J)
        vc = corr_var(rc, G)[0]
        ro = hwc.solve(ou, H, T, J, 1, tq)                              # orthonormal form, H-dependent
        vd = corr_var(rc, np.diag(np.diag(G)))[0]
        rec[str(J)] = dict(var=vc.tolist(), eq_err=float(np.abs(vc - ro.variance()[:, 0]).max()), diag=vd.tolist())
log('ou done', round(time.time() - t00))
Jl = [5, 9, 17, 33, 65, 129]; OUT['ou_interior'] = {}
for H in [0.5, 0.55, 0.75, 0.95]:
    ex1 = E.ou_exact_var(H, 1.0, 1.0, 1.0)
    OUT['ou_interior'][str(H)] = [float(abs(hwc.solve(ou, H, T, J, 1, np.array([1.0]), form='correlated').variance()[0, 0] / ex1 - 1)) for J in Jl]
OUT['ou_interior_J'] = Jl
log('ou interior done', round(time.time() - t00))

# ---------------- Wick growth: dX = mu X dt + sig X <> dB_H ----------------
mu, sg, X0 = 0.2, 0.5, 1.0
wg = Eq(X0, P([[(mu, (1,))]]), P([[(sg, (1,))]]), name='Wick growth')
OUT['wick'] = {}
for H in [0.5, 0.75, 0.95]:
    tw = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    exact = X0 ** 2 * np.exp(2 * mu * tw) * np.expm1(sg ** 2 * tw ** (2 * H)); rec = dict(t=tw.tolist(), exact=exact.tolist())
    for J, pmax in [(5, 5), (17, 5), (33, 5), (65, 3)]:
        r = hwc.solve(wg, H, T, J, pmax, tw)
        c2 = r.coef[:, :, 0] ** 2; deg = r.basis.deg
        rec[str(J)] = {str(p): c2[:, (deg >= 1) & (deg <= p)].sum(1).tolist() for p in range(1, pmax + 1)}
        rec[str(J)]['seconds'] = r.info['seconds']; rec[str(J)]['n'] = r.basis.n
        log('wick', H, J, pmax, r.basis.n, round(r.info['seconds'], 1))
    OUT['wick'][str(H)] = rec

# ---------------- Caputo memory model (EEM22 Eq. 78) ----------------
a_, al, mu_m, s_m = 0.5, 0.5, 0.0198, 0.02
mem = Eq(1.0, P([[(mu_m + s_m ** 2 / 2, (1,))]]), P([[(s_m, (1,))]]), memory=dict(a=a_, alpha=al), name='memory')
tm = np.linspace(0, 20.0, 401)
OUT['memory'] = {'t': tm.tolist()}
for J in [9, 33, 65]:
    r = hwc.solve(mem, 0.7, 20.0, J, 2, tm, form='correlated')        # H-independent coefficients
    OUT['memory']['mean'] = r.mean[:, 0].tolist()
    for H in [0.5, 0.7, 0.95]:
        G = hfc.gram(H, 20.0, 'cos', J)
        v1, v2 = corr_var(r, G)
        on = r.orthonormal(H)[:, :, 0]; deg = r.basis.deg
        v1o = (on[:, deg == 1] ** 2).sum(1); v2o = (on[:, deg == 2] ** 2).sum(1)
        vd = corr_var(r, np.diag(np.diag(G)), deg=1)[0]
        OUT['memory'].setdefault(str(H), {})[str(J)] = dict(var1=v1.tolist(), var2=v2.tolist(), var1_on=v1o.tolist(),
            var2_on=v2o.tolist(), var1_diag=vd.tolist(), eq_err=float(max(np.abs(v1 - v1o).max(), np.abs(v2 - v2o).max())))
    log('memory p2', J, round(time.time() - t00))
tl = np.array([4.0, 8.0, 12.0, 16.0, 20.0])
for H in [0.5, 0.7, 0.95]:
    OUT['memory'][str(H)]['fwhe57'] = OLD['memory'][str(H)]['fwhe57']
    OUT['memory'][str(H)]['gauss'] = {}
for J in [9, 17, 33, 65, 129, 257]:
    r = hwc.solve(mem, 0.7, 20.0, J, 1, tl, form='correlated')
    for H in [0.5, 0.7, 0.95]:
        G = hfc.gram(H, 20.0, 'cos', J); v1 = corr_var(r, G)[0]; vd = corr_var(r, np.diag(np.diag(G)))[0]
        v1o = (r.orthonormal(H)[:, 1:, 0] ** 2).sum(1)
        OUT['memory'][str(H)]['gauss'][str(J)] = {str(t): dict(var1=float(v1[i]), var1_on=float(v1o[i]), var1_diag=float(vd[i]))
                                                  for i, t in enumerate(tl)}
log('memory done', round(time.time() - t00))

# ---------------- population (NBE23): dX = (rho X - eps X^2) dt + lam X <> dB_H ----------------
rho, eps, lam, x0, TT = 0.5, 0.01, 0.02, 0.5, 30.0
pop = Eq(x0, P([[(rho, (1,)), (-eps, (2,))]]), P([[(lam, (1,))]]), name='population')
tout = np.linspace(0, TT, 301); OUT['pop'] = {}
for H in [0.55, 0.65, 0.75, 0.85, 0.95]:
    r = hwc.solve(pop, H, TT, 65, 1, tout)
    hand = E.population(H, 65, 1, 'correlated', tout=tout)              # hand-derived propagator (check only)
    G = hfc.gram(H, TT, 'cos', 65)
    rd = hwc.solve(pop, H, TT, 65, 1, tout, gram=np.diag(np.diag(G)))
    rec = {'p1': dict(t=tout.tolist(), mean=r.mean[:, 0].tolist(), var=r.variance()[:, 0].tolist()),
           'p1_eq': float(max(np.abs(r.variance()[:, 0] - hand['var']).max(), np.abs(r.mean[:, 0] - hand['mean']).max())),
           'p1_diag_var': rd.variance()[:, 0].tolist()}
    if H in (0.55, 0.75, 0.95):
        r2 = hwc.solve(pop, H, TT, 33, 2, tout); hand2 = E.population(H, 33, 2, 'correlated', tout=tout)
        rec['p2'] = dict(t=tout.tolist(), mean=r2.mean[:, 0].tolist(), var=r2.variance()[:, 0].tolist())
        rec['p2_eq'] = float(np.abs(r2.variance()[:, 0] - hand2['var']).max())
    OUT['pop'][str(H)] = rec
log('pop done', round(time.time() - t00))
tf = np.linspace(0, TT, 1201); OUT['popJ'] = {}
for H in [0.5, 0.55, 0.75, 0.95]:
    OUT['popJ'][str(H)] = {str(J): float(hwc.solve(pop, H, TT, J, 1, tf).variance()[:, 0].max()) for J in [17, 33, 65, 129]}
OUT['popH05_p2'] = {str(J): float(hwc.solve(pop, 0.5, TT, J, 2, tf).variance()[:, 0].max()) for J in [17, 33, 65]}
tt = np.array([11.25, 15.0])
OUT['popH05_t'] = {f'{J}_{p}': hwc.solve(pop, 0.5, TT, J, p, tt).variance()[:, 0].tolist()
                   for J, p in [(129, 1), (257, 1), (17, 2), (33, 2), (65, 2)]}
OUT['pop_mc_H05'] = OLD['pop_mc_H05']
tf2 = np.linspace(0, TT, 601)
OUT['pop_lim_curves'] = {'t': tf2.tolist(), **{str(H): hwc.solve(pop, H, TT, 129, 1, tf2).variance()[:, 0].tolist()
                                               for H in [0.5, 0.51, 0.55, 0.6]}}
log('popJ done', round(time.time() - t00))

# ---------------- Brownian limit ----------------
Hs = [0.5, 0.5001, 0.501, 0.505, 0.51, 0.52, 0.55, 0.6, 0.7]
lim = {'H': Hs, 'G_minus_I_2': [], 'G_minus_I_max': [], 'offdiag_corr': [], 'pop_maxvar_J33': []}
rou = hwc.solve(ou, 0.5, 2.0, 65, 1, np.array([2.0]), form='correlated')
for H in Hs:
    G = hfc.gram(H, 1.0, 'cos', 33)
    lim['G_minus_I_2'].append(float(np.linalg.norm(G - np.eye(33), 2))); lim['G_minus_I_max'].append(float(np.abs(G - np.eye(33)).max()))
    dd = np.sqrt(np.diag(G)); lim['offdiag_corr'].append(float(np.abs(G / np.outer(dd, dd) - np.eye(33)).max()))
    lim['pop_maxvar_J33'].append(float(hwc.solve(pop, H, TT, 33, 1, np.linspace(0, 30, 601)).variance()[:, 0].max()))
lim['ou_exact_T'] = OLD['limit']['ou_exact_T']
lim['ou_var_T'] = [float(rou.variance(H)[0, 0]) for H in Hs]
OUT['limit'] = lim
OUT['seconds_total'] = time.time() - t00
json.dump(OUT, open('v6/results/v12_examples.json', 'w'))
log('all done', round(time.time() - t00))
