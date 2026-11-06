"""Round 3b: work-precision curves, reference solutions and the memory-branch speed-up.

For the three FWHE benchmark examples (OU = EEM22 Example 1 with linear drift, Wick growth = EEM22 Example 2,
Caputo memory model = EEM22 Eq. (78)) and H = 0.55, 0.75, 0.95 this script records, for the cosine solver and for
the FWHE numerical algorithm (first-order Grunwald-Letnikov, EEM22 Sec. 4.2, re-implemented in
round3_fwhe_benchmarks.py), the maximum relative error of the variance over the output times against the
reference solution, together with the wall-clock time.  Timings: best of `rep` repetitions on one machine.
Reference solutions (Appendix F of the manuscript):
  OU      exact variance (closed form, examples_v6.ou_exact_var);
  Wick    exact variance x0^2 e^{2 mu t}(e^{sig^2 t^{2H}} - 1);
  memory  first-chaos variance sig^2 <g(t-.)psi, g(t-.)psi>_H by Gauss-Jacobi quadrature on Talbot-inverted Green
          functions, Richardson-extrapolated from 801- and 1601-point grids (grid error O(h^{3/2})).
The solver is v1.1.3 (evidence/round3_fwhe/solver_v1.1.3): identical numerics to v1.1.1, faster memory branch.
No random numbers are used.  Run from working/:  python scripts/round3b_workprecision.py
"""
import json, math, os, sys, time, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, 'evidence', 'round3_fwhe')
sys.path.insert(0, os.path.join(EV, 'solver_v1.1.3')); sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solver'))   # release layout: solver/ next to scripts/
import hfc, hwc, examples_v6 as E
P, Eq = hwc.Poly, hwc.Equation
log = lambda *a: print(*a, flush=True)


def load_fwhe_helpers():
    """gl_int_w, fwhe_gl_var, euler_caputo from round3_fwhe_benchmarks.py (executed up to the B1 block only)."""
    src = open(os.path.join(HERE, 'round3_fwhe_benchmarks.py'), encoding='utf-8').read()
    src = src[:src.find('Ns = [')]
    ns = {'__file__': os.path.join(HERE, 'round3_fwhe_benchmarks.py')}
    src = src.replace("sys.path.insert(0, os.path.join(EV, 'solver_snapshot_v1.1.1'))", "")
    exec(compile(src, 'round3_fwhe_benchmarks.py', 'exec'), ns)
    return ns


def best(fun, rep):
    out, ts = None, []
    for _ in range(rep):
        t0 = time.perf_counter(); out = fun(); ts.append(time.perf_counter() - t0)
    return out, min(ts)


def main():
    F = load_fwhe_helpers(); fwhe_gl_var, euler_caputo = F['fwhe_gl_var'], F['euler_caputo']
    OUT = {'note': __doc__, 'H': [0.55, 0.75, 0.95]}
    Hs = OUT['H']; T = 2.0; tq = np.round(np.arange(0.25, T + 1e-9, 0.25), 10)

    # ------------------------------------------------------------ OU
    ou = Eq(0.0, P([[(-1.0, (1,))]]), P([[(1.0, (0,))]]), name='OU')
    OUT['OU'] = {'T': T, 't': tq.tolist(), 'lam': 1.0, 'sig': 1.0, 'x0': 0.0}
    for H in Hs:
        ex = np.array([E.ou_exact_var(H, 1.0, 1.0, float(t)) for t in tq])
        rec = {'reference': ex.tolist(), 'ours': {}, 'fwhe_gl': {}}
        Js = [5, 9, 17, 33, 65, 129, 257, 513, 1025] + ([2049] if H == 0.55 else [])
        for J in Js:
            r, s = best(lambda: hwc.solve(ou, H, T, J, 1, tq, rtol=1e-7, atol=1e-9), 3 if J <= 129 else 1)
            v = r.variance()[:, 0]
            rec['ours'][str(J)] = dict(var=v.tolist(), seconds=s, max_rel=float(np.max(np.abs(v / ex - 1))))
        for N in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]:
            dt = T / N; idx = [int(round(t / dt)) for t in tq]
            ker = lambda i, dt=dt: (1.0 - dt) ** (i - np.arange(i + 1))
            v, s = best(lambda: fwhe_gl_var(H, dt, ker, idx), 3 if N <= 3200 else 1)
            rec['fwhe_gl'][str(N)] = dict(var=v.tolist(), seconds=s, max_rel=float(np.max(np.abs(v / ex - 1))))
        OUT['OU'][str(H)] = rec
        log('OU', H, {J: f"{rec['ours'][J]['max_rel']:.1e}/{rec['ours'][J]['seconds']:.2f}" for J in rec['ours']})
        log('   GL', {N: f"{rec['fwhe_gl'][N]['max_rel']:.1e}/{rec['fwhe_gl'][N]['seconds']:.3f}" for N in rec['fwhe_gl']})

    json.dump(OUT, open(os.path.join(EV, 'workprecision.json'), 'w'), indent=1)
    # ------------------------------------------------------------ Wick growth
    mu, sg, X0 = 0.2, 0.5, 1.0
    wg = Eq(X0, P([[(mu, (1,))]]), P([[(sg, (1,))]]), name='Wick growth')
    OUT['Wick'] = {'T': T, 't': tq.tolist(), 'mu': mu, 'sig': sg, 'x0': X0}
    for H in Hs:
        s2 = sg ** 2 * tq ** (2 * H); base = X0 ** 2 * np.exp(2 * mu * tq); ex = base * np.expm1(s2)
        rec = {'reference': ex.tolist(), 'ours': {}, 'fwhe_gl_k1': {},
               'fwhe_order': {str(k): (base * sum(s2 ** n / math.factorial(n) for n in range(1, k + 1))).tolist()
                              for k in range(1, 6)}}
        for J in [5, 9, 17, 33, 65]:
            for p in range(1, (5 if J <= 17 else 3) + 1):
                r, s = best(lambda: hwc.solve(wg, H, T, J, p, tq, rtol=1e-7, atol=1e-9), 3 if J <= 17 else 1)
                v = r.variance()[:, 0]
                rec['ours'][f'{J},{p}'] = dict(J=J, p=p, var=v.tolist(), seconds=s,
                                               max_rel=float(np.max(np.abs(v[1:] / ex[1:] - 1))))
            log('Wick ours', H, J, {k: f"{d['max_rel']:.1e}/{d['seconds']:.2f}" for k, d in rec['ours'].items() if d['J'] == J})
        for N in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
            dt = T / N; idx = [int(round(t / dt)) for t in tq]
            def ker(i, dt=dt):
                n = np.arange(i + 1); m0 = X0 * (1 + mu * dt) ** n
                return sg * m0 * (1 + mu * dt) ** (i - n)
            v, s = best(lambda: fwhe_gl_var(H, dt, ker, idx), 3 if N <= 3200 else 1)
            rec['fwhe_gl_k1'][str(N)] = dict(var=v.tolist(), seconds=s, max_rel=float(np.max(np.abs(v / ex - 1))))
        OUT['Wick'][str(H)] = rec

    json.dump(OUT, open(os.path.join(EV, 'workprecision.json'), 'w'), indent=1)
    # ------------------------------------------------------------ Caputo memory model
    a_, al, mu_m, s_m, TT = 0.5, 0.5, 0.0198, 0.02, 20.0; b = mu_m + s_m ** 2 / 2
    mem = Eq(1.0, P([[(b, (1,))]]), P([[(s_m, (1,))]]), memory=dict(a=a_, alpha=al), name='memory')
    tl = np.array([4.0, 8.0, 12.0, 16.0, 20.0])
    OUT['memory'] = {'T': TT, 't': tl.tolist(), 'a': a_, 'alpha': al, 'mu': mu_m, 'sig': s_m, 'x0': 1.0}
    kern = {}
    for n in [801, 1601]:
        t0 = time.perf_counter(); kern[n] = E.memory_kernels(n=n); OUT['memory'][f'kernel_seconds_{n}'] = time.perf_counter() - t0
    q = 2 ** 1.5 - 1
    Hall = [0.5, 0.55, 0.7, 0.75, 0.95]
    ref = {}
    for H in Hall:
        r8 = np.array([E.fwhe57(H, float(t), *kern[801]) for t in tl]); r16 = np.array([E.fwhe57(H, float(t), *kern[1601]) for t in tl])
        ref[H] = r16 + (r16 - r8) / q
        OUT['memory'][f'ref_{H}'] = dict(r801=r8.tolist(), r1601=r16.tolist(), richardson=ref[H].tolist())
    # speed-up check: v1.1.1 versus v1.1.3, identical settings (H = 0.95, J = 129, ngrid = 1601)
    spec = importlib.util.spec_from_file_location('hwc111', os.path.join(EV, 'solver_snapshot_v1.1.1', 'hwc.py'))
    hwc111 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hwc111)
    mem111 = hwc111.Equation(1.0, hwc111.Poly([[(b, (1,))]]), hwc111.Poly([[(s_m, (1,))]]), memory=dict(a=a_, alpha=al))
    t0 = time.perf_counter(); r_old = hwc111.solve(mem111, 0.95, TT, 129, 1, tl, memory_opts={'ngrid': 1601}); s_old = time.perf_counter() - t0
    hwc._GREEN_CACHE.clear()
    t0 = time.perf_counter(); r_new = hwc.solve(mem, 0.95, TT, 129, 1, tl, memory_opts={'ngrid': 1601}); s_new = time.perf_counter() - t0
    t0 = time.perf_counter(); r_new2 = hwc.solve(mem, 0.95, TT, 129, 1, tl, memory_opts={'ngrid': 1601}); s_cached = time.perf_counter() - t0
    OUT['memory']['speedup'] = dict(H=0.95, J=129, ngrid=1601, seconds_v111=s_old, seconds_v113=s_new, seconds_v113_cached=s_cached,
        max_abs_diff_variance=float(np.max(np.abs(r_old.variance() - r_new.variance()))),
        max_abs_diff_coef=float(np.max(np.abs(r_old.coef - r_new.coef))), cpu_count=os.cpu_count())
    log('speedup', OUT['memory']['speedup'])
    json.dump(OUT, open(os.path.join(EV, 'workprecision.json'), 'w'), indent=1)
    # work-precision (Green functions recomputed for every timed solve: cache cleared)
    for H in Hall:
        rec = {'reference': ref[H].tolist(), 'ours': {}, 'fwhe_gl': {}}
        for J in [9, 17, 33, 65, 129, 257]:
            hwc._GREEN_CACHE.clear()
            t0 = time.perf_counter(); r = hwc.solve(mem, H, TT, J, 1, tl, memory_opts={'ngrid': 1601}); s = time.perf_counter() - t0
            v = r.variance()[:, 0]
            rec['ours'][str(J)] = dict(var=v.tolist(), seconds=s, max_rel=float(np.max(np.abs(v / ref[H] - 1))))
        if H in (0.55, 0.75, 0.95):
            for N in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
                dt = TT / N; idx = [int(round(t / dt)) for t in tl]
                def run(dt=dt, N=N, idx=idx):
                    uu = euler_caputo(b, a_, al, dt, N, 1.0); gg = euler_caputo(b, a_, al, dt, N, 1.0, caputo=False)
                    ker = lambda i: s_m * gg[i - np.arange(i + 1)] * uu[:i + 1]
                    return fwhe_gl_var(H, dt, ker, idx)
                v, s = best(run, 3 if N <= 1600 else 1)
                rec['fwhe_gl'][str(N)] = dict(var=v.tolist(), seconds=s, max_rel=float(np.max(np.abs(v / ref[H] - 1))))
        OUT['memory'][str(H)] = rec
        json.dump(OUT, open(os.path.join(EV, 'workprecision.json'), 'w'), indent=1)
        log('memory', H, {J: f"{rec['ours'][J]['max_rel']:.1e}/{rec['ours'][J]['seconds']:.1f}" for J in rec['ours']},
            {N: f"{rec['fwhe_gl'][N]['max_rel']:.1e}/{rec['fwhe_gl'][N]['seconds']:.2f}" for N in rec['fwhe_gl']})
    # Table tab:memory (example of Sec. 5): t = 20, H = 0.5, 0.7, 0.95; correlated form, ngrid 1601 (cached Green)
    tab = {}
    for H in [0.5, 0.7, 0.95]:
        row = {'reference_t20': float(ref[H][-1]), 'dev_pct': {}}
        for J in [9, 33, 129, 257]:
            r = hwc.solve(mem, H, TT, J, 1, tl, form='correlated', memory_opts={'ngrid': 1601})
            row['dev_pct'][str(J)] = float(100 * (r.variance()[-1, 0] / ref[H][-1] - 1))
            if J == 257:
                a1 = r.coef[-1, r.basis.lin, 0]; G = hfc.gram(H, TT, 'cos', J)
                row['check_aGa_minus_var'] = float(a1 @ G @ a1 - r.variance()[-1, 0])
                row['diag_dev_pct'] = float(100 * ((a1 ** 2) @ np.diag(G) / ref[H][-1] - 1))
        t0 = time.perf_counter()
        r2 = hwc.solve(mem, H, TT, 65, 2, tl, form='correlated', memory_opts={'ngrid': 1601})
        r1 = hwc.solve(mem, H, TT, 65, 1, tl, form='correlated', memory_opts={'ngrid': 1601})
        row['v2_J65'] = float(r2.variance()[-1, 0] - r1.variance()[-1, 0]); row['v2_seconds'] = time.perf_counter() - t0
        tab[str(H)] = row; log('tab_memory', H, row)
    OUT['memory']['tab_memory'] = tab
    json.dump(OUT, open(os.path.join(EV, 'workprecision.json'), 'w'), indent=1)
    log('written')


if __name__ == '__main__':
    main()
