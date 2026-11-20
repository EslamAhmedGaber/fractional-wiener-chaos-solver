"""End-to-end validation at preselected configurations, with moment reduction inside both timers."""
import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"
for folder in ("P2_population", "P3_lotka_volterra", "solver"):
    sys.path.insert(0, str(EVID/folder))
import hfc
import hwc
import p2_population as pop
import p3_lotka_volterra as lv
from reference_stats import reference_moments


def run(lam, h, modes, paths):
    logistic = lam is not None
    model = pop if logistic else lv
    settings = pop.BASE if logistic else lv.PARAMS
    n = 256 if logistic else 128
    time_grid = model.TIMES
    idx = (np.arange(1, 9)*n)//8
    step = settings["T"]/n
    if logistic:
        ref = json.loads((EVID/"P2_population"/f"reference_lam{lam}_H{h}.json").read_text())
        ref_cov = np.array(ref["var"])[:,None,None]
        ref_se = np.array(ref["var_se"])[:,None,None]
    else:
        ref = json.loads((EVID/"P3_lotka_volterra/p3_results.json").read_text())["cases"][str(h)]["reference"]
        ref_cov, ref_se = np.array(ref["cov"]), np.array(ref["cov_se"])
    def assess(mean, cov, se):
        rm = np.asarray(ref["mean"]).reshape(mean.shape)
        scale = np.sqrt(np.diagonal(ref_cov,axis1=1,axis2=2))
        scale = scale[:,:,None]*scale[:,None,:]
        return dict(mean_pct=float(np.max(100*np.abs(mean/rm-1))),
                    cov_pct=float(np.max(100*np.abs(cov-ref_cov)/scale)),
                    guard_pct=float(np.max(100*(np.abs(cov-ref_cov)+2*np.hypot(se,ref_se))/scale)))
    times = []
    for _ in range(3):
        hfc._rules.clear()
        start = perf_counter()
        if logistic:
            eq = hwc.Equation([settings["y0"]], hwc.Poly([[(settings["a"],(1,)),(-settings["eps"],(2,))]]),
                              hwc.Poly([[(lam,(1,))]]))
        else:
            eq = lv.build_eq()
        solution = hwc.solve(eq,h,settings["T"],modes,2,time_grid,rtol=1e-8,atol=1e-10)
        mean, cov = solution.mean.copy(), solution.covariance()
        times.append(perf_counter()-start)
    det = assess(mean,cov,0)
    start = perf_counter()
    seed = (99 + int(1e5*lam) if logistic else 4026) + n + paths + int(1000*h)
    rng = np.random.default_rng(seed)
    pieces, controls = [], []
    for left in range(0, paths, 4096):
        k = min(4096,paths-left)
        inc = model.fgn_increments(h,n,k,rng,settings["T"])
        b = np.column_stack((np.zeros(k), inc.cumsum(1)))
        pieces.append(pop.wis_exact(b,h,lam,idx,step)[:,:,None] if logistic else lv.lv_wis_rk4(b,h,idx,step))
        controls.append(b[:,idx].T)
    y = np.concatenate(pieces, axis=1)
    x = np.concatenate(controls, axis=1)
    mean,cov,_,se = reference_moments(y,x,time_grid**(2*h),batches=32)
    elapsed = perf_counter()-start
    sampled = assess(mean,cov,se)
    return dict(model="logistic" if logistic else "competition",lam=lam,H=h,J=modes,p=2,
                comparator_n=n,comparator_M=paths,coschaos_trials_s=times,
                coschaos_s=float(np.median(times)),sampler_s=elapsed,ratio=elapsed/np.median(times),
                coschaos=det,sampler=sampled,
                both_eligible=all(c["mean_pct"]<=1 and c["guard_pct"]<=1 for c in (det,sampled)))


if __name__=="__main__":
    rows=[]
    configs=[(.02,.95,17,2**16),(.05,.95,17,2**16),(None,.75,65,2**19),(None,.95,33,2**18)]
    with threadpool_limits(1):
        for config in configs:
            row=run(*config)
            rows.append(row)
            print(json.dumps(row),flush=True)
    (EVID/"end_to_end.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")

