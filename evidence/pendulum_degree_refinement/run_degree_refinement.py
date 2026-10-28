"""Fixed-J pendulum degree refinement; unchanged general solver, checked quadrature."""
from pathlib import Path
import sys,json,time,argparse,platform
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"evidence/solver_snapshot/code"))
import hwc
parser=argparse.ArgumentParser()
parser.add_argument("--case",nargs=3,type=int,action="append",metavar=("J","P","Q"))
parser.add_argument("--threads",type=int,default=2)
parser.add_argument("--memory-lean",action="store_true")
parser.add_argument("--rtol",type=float,default=1e-11)
parser.add_argument("--output",type=Path,default=HERE/"degree_refinement.json")
args=parser.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
if args.memory_lean:
 class LeanTensorGH:
  def __init__(self,B,order,max_nodes):
   if order**B.J>max_nodes:raise NotImplementedError('quadrature node budget exceeded')
   z,self.w=hwc._gh(order,B.J)
   self.V=B.evaluate(z)
  def __call__(self,Fun,t,c):
   return self.V.T@(self.w[:,None]*np.asarray(Fun(t,self.V@c),float))
 hwc._TensorGH=LeanTensorGH
F=lambda t,X:np.stack([X[:,1],-np.sin(X[:,0])-.6*X[:,1]],axis=1)
# Exactly the same additive diffusion, represented as a polynomial to avoid
# evaluating a constant at every quadrature node. The core solver is unchanged.
D=hwc.Poly([[],[(.4,(0,0))]])
eq=hwc.Equation([1.,0.],F,D,name="pendulum")
times=np.array([1.,2.,5.,10.])
cases=args.case or [(3,1,20),(3,2,11),(3,3,11),(3,4,11),
                     (5,1,20),(5,2,11),(5,3,9),(5,3,11),(5,4,11)]
out=json.loads(args.output.read_text()) if args.output.exists() else dict(
    parameters=dict(H=.9,T=10.,omega=1.,zeta=.3,sigma=.4,Q0=1.,V0=0.),times=times.tolist(),
    note="Fixed-J diagnostic; general solver unchanged. Quadrature errors require separate checks.",
    environment=dict(python=platform.python_version(),numpy=np.__version__,blas_threads=args.threads),runs={})
for J,p,q in cases:
 key=f"J{J}_p{p}_q{q}"+('_lean' if args.memory_lean else '')+(f'_rtol{args.rtol}' if args.rtol!=1e-11 else '')
 if key in out["runs"]:continue
 print(json.dumps(dict(start=key,nodes=q**J if p>1 else q**2)),flush=True)
 start=time.perf_counter()
 try:
  with threadpool_limits(limits=args.threads):
   r=hwc.solve(eq,.9,10.,J,p,times,quad_order=q,max_nodes=2000000,max_coefficients=600000,rtol=args.rtol,atol=args.rtol*.01)
  record=dict(J=J,p=p,quadrature_order=q,nodes=q**J if p>1 else None,
              seconds=time.perf_counter()-start,mean=r.mean.tolist(),variance=r.variance().tolist(),
              covariance=r.covariance().tolist(),info=r.info,
              coefficients_per_component=r.basis.n,status="completed",memory_lean=args.memory_lean,rtol=args.rtol)
  # Retain coefficients locally so higher moments/distributions can be examined without re-solving.
  coefficient_dir=args.output.parent/"coefficients";coefficient_dir.mkdir(exist_ok=True)
  exponents=np.array([np.bincount(a,minlength=J) for a in r.basis.idx],dtype=np.int16)
  np.savez_compressed(coefficient_dir/(key+".npz"),coef=r.coef,times=times,exponents=exponents)
 except (MemoryError,NotImplementedError,ValueError,RuntimeError) as exc:
  record=dict(J=J,p=p,quadrature_order=q,seconds=time.perf_counter()-start,status="failed",error=str(exc))
 out["runs"][key]=record
 args.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
 print(json.dumps(dict(case=key,status=record["status"],seconds=round(record["seconds"],2),var_t10=record.get("variance",[[None]])[-1])),flush=True)

