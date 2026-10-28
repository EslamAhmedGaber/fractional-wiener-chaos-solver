"""Re-solves the cubic-drift examples with the general solver (hwc.solve). Monte Carlo references are not
recomputed (results/cubic_v8.json, v10_runs.json). Writes results/v12_cubic.json.  python code/run_cubic_v12.py"""
import sys, json, time
import numpy as np
sys.path.insert(0, 'v6/code')
import hwc
P, Eq = hwc.Poly, hwc.Equation
def cubic(a, b, sig, x0):
    return Eq(x0, P([[(a, (1,)), (-b, (3,))]]), P([[(sig, (0,))]]), name='cubic drift')
def keep(r):
    return dict(t=r.times.tolist(), mean=r.mean[:, 0].tolist(), var=r.variance()[:, 0].tolist(),
                m3=hwc.third_central_moment(r).tolist(), seconds=r.info['seconds'], engine=r.info['engine'], n=r.basis.n)
OUT = {'sw': {}, 'sw9': {}, 'dw': {}, 'gl_chaos': {}}; t00 = time.time()
tq = np.array([0.5, 1.0, 2.0, 4.0]); F8t = np.round(np.arange(1, 41) * 0.1, 10)
for sig, Js in [(0.5, [17, 33, 65]), (0.25, [33, 65])]:
    for H in [0.6, 0.75, 0.9]:
        for J in Js:
            for p in [1, 2]:
                times = F8t if (sig == 0.5 and H == 0.75 and J == 65) else tq
                OUT['sw'][f'{sig}_{H}_{J}_{p}'] = keep(hwc.solve(cubic(-1, 1, sig, 1.0), H, 4.0, J, p, times))
        print('sw', sig, H, round(time.time() - t00), flush=True)
for H in [0.6, 0.75, 0.9]:
    for p in [1, 2, 3, 4]:
        OUT['sw9'][f'{H}_9_{p}'] = keep(hwc.solve(cubic(-1, 1, 0.5, 1.0), H, 4.0, 9, p, tq))
print('sw9', round(time.time() - t00), flush=True)
tdw = np.round(np.arange(0.4, 4.01, 0.4), 10)
OUT['dw']['0.75_65_2'] = keep(hwc.solve(cubic(1, 1, 0.2, 0.5), 0.75, 4.0, 65, 2, F8t))
for p in [2, 3, 4]:
    OUT['dw'][f'0.75_9_{p}'] = keep(hwc.solve(cubic(1, 1, 0.2, 0.5), 0.75, 4.0, 9, p, tdw))
print('dw', round(time.time() - t00), flush=True)
tg = np.array([0.25, 0.5, 0.75, 1.0])
for H, J, p in [(0.65, 9, 4), (0.8, 9, 4), (0.95, 9, 4), (0.65, 9, 2), (0.65, 17, 2), (0.65, 17, 3),
                (0.8, 9, 2), (0.8, 17, 2), (0.8, 17, 3)]:
    OUT['gl_chaos'][f'{H}_{J}_{p}'] = keep(hwc.solve(cubic(1, 1, 0.4, 0.3), H, 1.0, J, p, tg))
print('gl', round(time.time() - t00), flush=True)
OUT['seconds_total'] = time.time() - t00
json.dump(OUT, open('v6/results/v12_cubic.json', 'w'))
print('done', flush=True)
