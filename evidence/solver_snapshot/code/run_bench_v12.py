"""CPU time and error of the solver for the cost table; Monte Carlo baselines are copied from results/new_apps_v7.json.
Writes results/v12_bench.json.   python code/run_bench_v12.py"""
import sys, json, time
import numpy as np
sys.path.insert(0, 'v6/code')
import hwc
P, Eq = hwc.Poly, hwc.Equation
NA = json.load(open('v6/results/new_apps_v7.json'))['bench']
ou = Eq(0.0, P([[(-1.0, (1,))]]), P([[(1.0, (0,))]]))
ex = NA['ou']['exact']; out = {'ou': {'exact': ex, 'mc': NA['ou']['mc'], 'chaos': []}, 'pop': {}}
hwc.solve(ou, 0.75, 1.0, 9, 1, np.array([0.5]))            # warm-up (imports, caches)
for J in [5, 9, 17, 33, 65, 129]:
    ts = []
    for rep in range(5):
        t0 = time.perf_counter(); r = hwc.solve(ou, 0.75, 1.0, J, 1, np.array([0.5])); v = r.variance()[0, 0]
        ts.append(time.perf_counter() - t0)
    out['ou']['chaos'].append(dict(J=J, time=float(np.median(ts)), relerr=float(abs(v / ex - 1))))
pop = Eq(0.5, P([[(0.5, (1,)), (-0.01, (2,))]]), P([[(0.02, (1,))]]))
ref = NA['pop']['mc_ref']; out['pop']['mc_ref'] = ref; out['pop']['mc_2e5'] = NA['pop']['mc_2e5']; out['pop']['chaos'] = []
for J in [33, 65, 129, 257]:
    t0 = time.perf_counter(); r = hwc.solve(pop, 0.5, 30.0, J, 1, np.array([11.25])); v = r.variance()[0, 0]
    out['pop']['chaos'].append(dict(J=J, p=1, time=time.perf_counter() - t0, var=float(v), relerr=float(v / ref['var'] - 1)))
c = out['pop']['chaos']; v129, v257 = c[-2]['var'], c[-1]['var']; vR = v257 + (v257 - v129) / (2 ** 1 - 1)
out['pop']['richardson_129_257'] = dict(var=vR, relerr=vR / ref['var'] - 1, time=c[-2]['time'] + c[-1]['time'])
json.dump(out, open('v6/results/v12_bench.json', 'w'))
print(json.dumps(out['ou']['chaos']), json.dumps(out['pop']['chaos']), out['pop']['richardson_129_257'])
