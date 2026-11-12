"""Paired 512/1024 Heun check on four preselected challenging events."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WK=ROOT
sys.path.insert(0,str(WK/'scripts/round14_experiments'))
import rare_events as re
(ROOT/'reports').mkdir(exist_ok=True)
ic=re.ic
sel=json.loads((WK/'evidence/round14/selected_proposals.json').read_text())
cases=[('single',.55,.9),('single',.95,1.1),('double',.55,-.5),('double',.95,-.5)]
records=[]
for i,(mod,H,c) in enumerate(cases):
    m=re.MODELS[mod];row=next(x for x in sel if x['model']==mod and x['H']==H and x['c']==c)
    res,_=ic.coschaos_cubic(H,m['T'],9,row['guide_degree'],x0=m['x0'],a=m['a'],b=m['b'],s=m['s'])
    xs,_=ic.design_point(res,c,sign=m['sign']);d=xs/np.linalg.norm(xs)
    cc=ic.DiscreteCoords(H,m['T'],9,512);sc,_=re.ray_scale(cc,d,c,m);shift=sc*d
    cf=ic.DiscreteCoords(H,m['T'],9,1024);delta=cf.shift(shift);M=65536
    seed=[20261111,1502,i];rng=np.random.default_rng(np.random.SeedSequence(seed))
    sums=np.zeros(3);squares=np.zeros(3);t0=time.perf_counter()
    for a in range(0,M,4096):
        db=ic.fgn(H,1024,min(4096,M-a),m['T']/1024,rng)
        w=np.exp(-(cf.xi(db)@shift)-.5*(shift@shift))
        shifted=db+delta
        xf=ic.heun(shifted,m['T']/1024,m['drift'],m['x0'],m['s'])
        xc=ic.heun(shifted.reshape(-1,512,2).sum(axis=2),m['T']/512,m['drift'],m['x0'],m['s'])
        yf=w*(m['sign']*(xf-c)>0);yc=w*(m['sign']*(xc-c)>0)
        vals=np.stack([yc,yf,yf-yc],axis=1);sums+=vals.sum(axis=0);squares+=(vals**2).sum(axis=0)
    means=sums/M;ses=np.sqrt(np.maximum(squares/M-means**2,0)/(M-1))
    records.append(dict(model=mod,H=H,c=c,degree=row['guide_degree'],M=M,seed=seed,
                        coarse=float(means[0]),fine=float(means[1]),difference=float(means[2]),
                        coarse_se=float(ses[0]),fine_se=float(ses[1]),paired_difference_se=float(ses[2]),
                        relative_difference_percent=float(100*means[2]/means[1]),
                        z_difference=float(abs(means[2])/ses[2]) if ses[2]>0 else None,
                        seconds=time.perf_counter()-t0))
    (ROOT/'reports/paired_timestep.json').write_text(json.dumps(records,indent=2))
    print(mod,H,c,'relative change %',records[-1]['relative_difference_percent'],'paired SE',ses[2],flush=True)
print('DONE',flush=True)
