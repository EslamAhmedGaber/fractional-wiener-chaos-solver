"""Verify coefficient/moment records and rerun a small general-solver pendulum case."""
from pathlib import Path
import sys,json,subprocess,numpy as np
R=Path(__file__).resolve().parents[1];P=R/"evidence/pendulum_degree_refinement"
sys.path.insert(0,str(R/"evidence/solver_snapshot/code"))
import hwc
from threadpoolctl import threadpool_limits
runs={}
for f in P.glob("*.json"):
 d=json.loads(f.read_text())
 if isinstance(d,dict) and "runs" in d:runs.update(d["runs"])
count=0
for f in (P/"coefficients").glob("*.npz"):
 r=runs[f.stem];d=np.load(f,allow_pickle=False);c=d["coef"]
 assert np.isfinite(c).all()
 assert np.allclose(c[:,0,:],r["mean"],rtol=1e-13,atol=2e-14)
 cov=np.einsum("tis,tir->tsr",c[:,1:,:],c[:,1:,:])
 assert np.allclose(cov,r["covariance"],rtol=1e-13,atol=2e-14)
 assert np.allclose(np.diagonal(cov,axis1=1,axis2=2),r["variance"],rtol=1e-13,atol=2e-14)
 assert d["exponents"].shape==(c.shape[1],r["J"])
 count+=1
assert count==len([r for r in runs.values() if r["status"]=="completed"])
print("PASS saved coefficients, means, covariances and variances:",count,flush=True)
F=lambda t,X:np.stack([X[:,1],-np.sin(X[:,0])-.6*X[:,1]],axis=1)
D=hwc.Poly([[],[(.4,(0,0))]])
with threadpool_limits(limits=2):
 result=hwc.solve(hwc.Equation([1.,0.],F,D),.9,10.,3,4,np.array([1.,2.,5.,10.]),
                  quad_order=11,max_nodes=2000000,rtol=1e-11,atol=1e-13)
assert np.allclose(result.coef,np.load(P/"coefficients/J3_p4_q11.npz")["coef"],rtol=1e-7,atol=1e-9)
print("PASS rerun of general callable-drift coefficient solver",flush=True)
for name in ["diffusion_equivalence_check","memory_lean_equivalence_check","ode_tolerance_check"]:
 assert json.loads((P/(name+".json")).read_text())["passed"]
a=runs["J3_p8_q55_lean_rtol1e-09"];b=runs["J3_p8_q65_lean_rtol1e-09"]
delta=100*(np.array(b["variance"])/np.array(a["variance"])-1)
assert np.max(abs(delta))<.06
bad=100*(runs["J5_p6_q15_lean_rtol1e-09"]["variance"][-1][0]/runs["J5_p6_q13_lean_rtol1e-09"]["variance"][-1][0]-1)
assert abs(bad)>5,"Retain the unresolved higher-dimensional quadrature discrepancy"
baseline=runs["J3_p1_q20"]["variance"][-1][0];p2=runs["J3_p2_q11"]["variance"][-1][0]
assert p2<baseline,"Do not assert monotonic refinement at every degree"
print("PASS quadrature diagnostics and retained negative findings",flush=True)
# Ensure published tables are exactly regenerated from the released records.
before={f:f.read_bytes() for f in P.glob("*.tex")}
subprocess.run([sys.executable,str(P/"generate_degree_report.py")],check=True,stdout=subprocess.DEVNULL)
assert all(f.read_bytes()==content for f,content in before.items())
print("PASS data-generated manuscript tables/macros",flush=True)
print("ALL PENDULUM REFINEMENT CHECKS PASSED")

