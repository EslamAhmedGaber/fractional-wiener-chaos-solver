"""Small-sample RQMC frontier, kept separate from the selective-solver checks."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'round12_experiments'))
import refine_solver as r
out={}
for H in [.75,.95]:
    rows=[];vref=np.array(r.REF[str(H)]['reference']['variance'])[1:]
    for N in [512,1024,2048,4096]:
        for seed in [71101,71102,71103]:
            t=time.perf_counter();m=r.b.FullRQMC(H);prep=time.perf_counter()-t
            t=time.perf_counter();x=m.produce(N,seed);sec=time.perf_counter()-t
            err=100*max(abs(x.var(axis=0)[1:]/vref-1))
            rows.append(dict(N=N,seed=seed,setup_seconds=prep,evaluation_seconds=sec,max_variance_error_percent=float(err)))
    out[str(H)]=rows
    print(H,[(N,max(v['max_variance_error_percent'] for v in rows if v['N']==N)) for N in [512,1024,2048,4096]],flush=True)
(ROOT.parent/'evidence/round12_1/rqmc_frontier.json').write_text(json.dumps(out,indent=2))
