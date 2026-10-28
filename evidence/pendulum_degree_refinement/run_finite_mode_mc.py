"""Independent path integration for the SAME finite-J cosine Gaussian process."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"evidence/solver_snapshot/code"))
import hfc
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1000);parser.add_argument('--output',type=Path,default=HERE/'finite_mode_mc.json');args=parser.parse_args()
H=.9;T=10.;sigma=.4;steps=args.steps;M=2**17;chunk=1024;seed=202610281
args.output.parent.mkdir(parents=True,exist_ok=True)
Js=[3,5,9,17,65];times=np.array([1.,2.,5.,10.]);idx=np.rint(times/(T/steps)).astype(int)
grid=np.linspace(0,T,steps+1)
L=np.linalg.cholesky(hfc.gram(H,T,"cos",max(Js)))
modes=hfc.Mcoef("cos",max(Js),T,grid)
drives={J:np.diff(modes[:,:J]@L[:J,:J],axis=0)*sigma for J in Js}
S={J:np.zeros((len(times),2,5)) for J in Js}
rng=np.random.default_rng(seed);start=time.perf_counter()
with threadpool_limits(limits=2):
 for first in range(0,M,chunk):
  count=min(chunk,M-first);xi=rng.standard_normal((count,max(Js)))
  for J in Js:
   db=xi[:,:J]@drives[J].T;q=np.ones(count);v=np.zeros(count);j=0;dt=T/steps
   for k in range(steps):
    a=-np.sin(q)-.6*v;qp=q+dt*v;vp=v+dt*a+db[:,k]
    qn=q+.5*dt*(v+vp);v=v+.5*dt*(a-np.sin(qp)-.6*vp)+db[:,k];q=qn
    if j<len(idx) and k+1==idx[j]:
     for dim,x in enumerate([q,v]):S[J][j,dim]+=[count,x.sum(),(x*x).sum(),(x**3).sum(),(x**4).sum()]
     j+=1
  if first==0 or (first+count)%32768==0:print(first+count,round(time.perf_counter()-start,2),flush=True)
results={}
for J,s in S.items():
 n=s[:,:,0];e1,e2,e3,e4=(s[:,:,i]/n for i in range(1,5))
 var=e2-e1*e1;mu4=e4-4*e1*e3+6*e1*e1*e2-3*e1**4
 results[str(J)]=dict(mean=e1.tolist(),variance=var.tolist(),se_mean=np.sqrt(var/n).tolist(),
                     se_variance=np.sqrt(np.maximum(mu4-var*var,0)/n).tolist())
out=dict(H=H,T=T,sigma=sigma,steps=steps,paths=M,seed=seed,chunk=chunk,times=times.tolist(),
         description="Heun solution with exact finite-J cosine noise increments; common xi across J; distinct from full-fBm MC",
         results=results,seconds=time.perf_counter()-start)
args.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({k:v["variance"][-1][0] for k,v in results.items()}),flush=True)

