from pathlib import Path
import subprocess,sys,json,hashlib,numpy as np
R=Path(__file__).resolve().parents[1]
m=R/"MANIFEST_SHA256.json"
if not m.exists():raise SystemExit("Release manifest missing")
manifest=json.loads(m.read_text(encoding="utf-8"))
for rel,expected in manifest.items():
 assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==expected,rel
print("PASS release hashes:",len(manifest),flush=True)
code=R/"evidence/solver_snapshot/code"
subprocess.run([sys.executable,str(code/"smoke_test.py")],check=True,cwd=R)
subprocess.run([sys.executable,str(code/"r2v2_checks.py")],check=True,cwd=R)
d=json.loads((code.parent/"results/r2v2_checks.json").read_text())
errors={k:v for k,v in d.items() if "maxdev" in k or "var_formula_vs_sum" in k}
assert errors and max(abs(v) for v in errors.values())<1e-10,errors
print("PASS Hermite/Wick identities:",len(errors),flush=True)
sys.path.insert(0,str(code))
import hfc,hwc
G=hfc.gram(.5,2.,"cos",17);assert np.allclose(G,np.eye(17),atol=1e-13)
G=hfc.gram(.75,2.,"cos",17);assert np.linalg.eigvalsh(G).min()>0
i,j=np.indices(G.shape);assert np.max(abs(G[(i+j)%2==1]))<1e-13
print("PASS Brownian limit, positivity and cosine parity",flush=True)
P,Eq=hwc.Poly,hwc.Equation
equation=Eq(1.,P([[(-1.,(1,)),(-1.,(3,))]]),P([[(.5,(0,))]]))
reference=json.loads((R/"evidence/data/v12_cubic.json").read_text())
# A stored cubic benchmark tests the same general nonlinear projection engine.
result=hwc.solve(equation,.75,4.,17,2,np.array([.5,1.,2.,4.]))
assert np.isfinite(result.coef).all()
stored=reference['sw']['0.5_0.75_17_2']
assert np.allclose(result.mean[:,0],stored['mean'],rtol=1e-8,atol=1e-11)
assert np.allclose(result.variance()[:,0],stored['var'],rtol=1e-8,atol=1e-11)
print("PASS nonlinear cubic run, variance:",result.variance()[:,0],flush=True)
mc=json.loads((R/"evidence/pendulum_refinement/pendulum_refinement.json").read_text())
assert mc["paths"]==131072 and mc["levels"]==[1000,2000,4000]
assert np.isfinite(np.array(mc["variances"])).all()
for pair in mc["paired_comparisons"]:
 lo=mc["levels"].index(pair["coarse_steps"]);hi=mc["levels"].index(pair["fine_steps"])
 assert np.allclose(np.array(mc["variances"])[hi]-np.array(mc["variances"])[lo],pair["variance_difference"],atol=1e-15)
 assert (np.array(pair["variance_difference_se"])>=0).all()
print("PASS coupled refinement result consistency")
subprocess.run([sys.executable,str(R/"scripts/verify_pendulum_refinement.py")],check=True,cwd=R)
print("ALL RELEASE CHECKS PASSED")

