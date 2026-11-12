"""Experimental selective Galerkin backend for the fixed additive tanh model."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,time,json
import numpy as np
from scipy.special import roots_hermitenorm
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'fixed_model_benchmark'))
import run_benchmark as b
hwc,hfc=b.hwc,b.hfc
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
TIMES=b.TIMES
REF=json.loads((b.OUT/'reference.json').read_text())
def save(name,data):
    (OUT/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def read(name):
    f=OUT/(name+'.json');return json.loads(f.read_text()) if f.exists() else {}
def error(c,H):
    v=np.sum(c[:,1:]**2,axis=1);mu=c[:,0]
    ref=REF[str(H)]['reference'];vref=np.array(ref['variance'])
    e=v[1:]/vref[1:]-1
    return dict(variance=v.tolist(),mean=mu.tolist(),max_variance_error_percent=float(100*max(abs(e))),
        final_variance_error_percent=float(100*e[-1]),worst_time=float(TIMES[1:][np.argmax(abs(e))]),
        max_mean_normalized_percent=float(100*max(abs(mu[1:]-np.array(ref['mean'])[1:])/np.sqrt(vref[1:]))))
class Selective:
    def __init__(self,H,J,K,p,q=16,qh=12,rtol=2e-7,atol=2e-9,times=TIMES):
        start=time.perf_counter();hfc._rules.clear()
        self.H,self.J,self.K,self.p=H,J,K,p
        self.noise=hwc.HarmonicNoise(H,2.,J);self.basis=hwc.ChaosBasis(K,p)
        self.nlow=self.basis.n;self.n=self.nlow+J-K
        x,w=roots_hermitenorm(q);w=w/np.sqrt(2*np.pi)
        mesh=np.meshgrid(*([x]*K),indexing='ij');z=np.stack(mesh,axis=-1).reshape(-1,K)
        weights=np.ones([q]*K)
        for k in range(K):
            shape=[1]*K;shape[k]=q;weights*=w.reshape(shape)
        weights=weights.ravel();V=self.basis.evaluate(z);PT=(V*weights[:,None]).T
        xh,wh=roots_hermitenorm(qh);wh=wh/np.sqrt(2*np.pi)
        self.build_projection_seconds=time.perf_counter()-start
        def rhs(t,c):
            low=c[:self.nlow];high=c[self.nlow:];sig=np.linalg.norm(high)
            a=V@low
            y=np.tanh(a[:,None]+sig*xh[None,:])
            ty=y@wh
            slope=weights@((1-y*y)@wh)
            dc=-1.5*c
            dc[:self.nlow]+=1.2*(PT@ty)
            dc[0]+=.2*np.cos(np.pi*t)
            dc[self.nlow:]+=1.2*slope*high
            ell=self.noise.ell(t)[0]
            dc[self.basis.lin]+=.7*ell[:K]
            dc[self.nlow:]+=.7*ell[K:]
            return dc
        y0=np.zeros(self.n);y0[0]=.2
        sol=solve_ivp(rhs,(0.,2.),y0,method='DOP853',t_eval=times,rtol=rtol,atol=atol)
        if not sol.success:raise RuntimeError(sol.message)
        self.coef=sol.y.T;self.nfev=sol.nfev;self.seconds=time.perf_counter()-start
    def produce(self,N,seed):
        rng=np.random.default_rng(seed);out=np.empty((N,len(self.coef)))
        for start in range(0,N,2048):
            xi=rng.standard_normal((min(2048,N-start),self.J))
            out[start:start+len(xi)]=self.basis.evaluate(xi[:,:self.K])@self.coef[:,:self.nlow].T+xi[:,self.K:]@self.coef[:,self.nlow:].T
        return out
def survey():
    out=read('p1_survey')
    for H in [.75,.95]:
        for J in [9,17,33,65,129]:
            key=f'{H}_{J}'
            if key in out:continue
            start=time.perf_counter()
            result=hwc.solve(b.bm.neural_equation(),H,2.,J,1,TIMES,quad_order=32,rtol=2e-7,atol=2e-9)
            out[key]=dict(H=H,J=J,p=1,seconds=time.perf_counter()-start,**error(result.coef[:,:,0],H))
            save('p1_survey',out)
            print('P1',H,J,round(out[key]['max_variance_error_percent'],3),'at',out[key]['worst_time'],'sec',round(out[key]['seconds'],2),flush=True)
def search():
    out=read('selective_search')
    for H in [.75,.95]:
        for J in [17,33,65]:
            for p in [3,5,7]:
                K=3;q=16;qh=12;key=f'{H}_J{J}_K{K}_p{p}_q{q}_qh{qh}'
                if key in out:continue
                model=Selective(H,J,K,p,q,qh)
                row=dict(H=H,J=J,K=K,p=p,q=q,qh=qh,n=model.n,seconds=model.seconds,nfev=model.nfev,**error(model.coef,H))
                out[key]=row;save('selective_search',out)
                np.savez_compressed(OUT/(key+'.npz'),coef=model.coef,times=TIMES)
                print('SELECT',key,'max%',round(row['max_variance_error_percent'],4),'at',row['worst_time'],'sec',round(model.seconds,3),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
