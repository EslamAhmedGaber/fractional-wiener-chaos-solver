import refine_solver as r
import numpy as np
from scipy.special import roots_hermitenorm
checks=[]
for p,q in [(3,12),(5,16)]:
    m=r.Selective(.75,3,3,p,q,12,rtol=2e-8,atol=2e-10)
    old=r.hwc.solve(r.b.bm.neural_equation(),.75,2.,3,p,r.TIMES,quad_order=q,rtol=2e-8,atol=2e-10)
    err=float(np.max(abs(m.coef-old.coef[:,:,0])))
    checks.append(dict(test=f'K=J agreement with published tensor solver p={p}',error=err,passed=err<1e-8))
# Independent four-dimensional projection versus conditional Gaussian formula.
K=2;J=4;p=3;q=24
basis=r.hwc.ChaosBasis(K,p)
x,w=roots_hermitenorm(q);w/=np.sqrt(2*np.pi)
z=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
wl=np.outer(w,w).ravel();V=basis.evaluate(z)
c=np.random.default_rng(24).normal(0,.015,basis.n);c[0]=.2;c[1]=.3;c[2]=-.2
high=np.array([.12,-.08]);a=V@c
y=np.tanh(a[:,None]+np.linalg.norm(high)*x)
low_cond=(V*wl[:,None]).T@(y@w)
high_cond=high*(wl@((1-y*y)@w))
zz=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
wh=np.outer(w,w).ravel();full=np.tanh(a[:,None]+zz@high)
low_full=(V*wl[:,None]).T@(full@wh)
high_full=(wl@full)@(zz*wh[:,None])
err=float(max(np.max(abs(low_full-low_cond)),np.max(abs(high_full-high_cond))))
checks.append(dict(test='conditional projection vs full 4D quadrature',error=err,passed=err<1e-8))
r.save('backend_validation',checks)
assert all(c['passed'] for c in checks),checks
print(checks,flush=True)
