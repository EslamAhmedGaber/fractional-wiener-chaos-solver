"""Fixed-model amortization benchmark; all measured methods return full time-grid output."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,time,json,platform
import numpy as np
import scipy
from scipy.special import ndtri
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import benchmark as bm
hwc,hfc=bm.hwc,bm.hfc
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
TIMES=np.linspace(0,2,33);NSTEPS=512;BATCH=2048

def save(name,data):
    (OUT/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def read(name):
    f=OUT/(name+'.json');return json.loads(f.read_text()) if f.exists() else {}
def metrics(x):
    mu=x.mean(axis=0);cov=x.T@x/len(x)-np.outer(mu,mu)
    return dict(mean=mu.tolist(),variance=np.diag(cov).tolist(),covariance=cov.tolist(),
        tail=float(np.mean(x[:,-1]>.7)),q95=float(np.quantile(x[:,-1],.95)),
        q99=float(np.quantile(x[:,-1],.99)))
def heun(increments):
    nstep=increments.shape[1];dt=2/nstep;stride=nstep//32
    x=np.full(len(increments),.2);out=np.empty((len(x),33));out[:,0]=x
    noise=np.ascontiguousarray(increments.T)
    for k,inc in enumerate(noise):
        t=k*dt;f=bm.drift(t,x);xp=x+dt*f+.7*inc
        x+=.5*dt*(f+bm.drift(t+dt,xp))+.7*inc
        if (k+1)%stride==0:out[:,(k+1)//stride]=x
    return out
class FullMC:
    def __init__(self,H,steps=NSTEPS):
        self.steps=steps;self.dt=2/steps;self.H=H
        k=np.arange(steps+1);g=.5*((k+1.)**(2*H)-2*k**(2*H)+abs(k-1.)**(2*H))
        g[0]=1.;lam=np.fft.fft(np.r_[g,g[-2:0:-1]]).real
        assert lam.min()>-1e-9
        self.factor=np.sqrt(np.maximum(lam,0)/(2*steps))
    def increments(self,n,rng):
        z=rng.standard_normal((n//2,2*self.steps))+1j*rng.standard_normal((n//2,2*self.steps))
        w=np.fft.fft(self.factor*z,axis=1)[:,:self.steps]
        return np.r_[w.real,w.imag]*self.dt**self.H
    def produce(self,n,seed):
        rng=np.random.default_rng(seed);out=np.empty((n,33))
        for start in range(0,n,BATCH):
            size=min(BATCH,n-start);out[start:start+size]=heun(self.increments(size,rng))
        return out
class FullRQMC:
    def __init__(self,H,steps=NSTEPS):
        self.steps=steps
        t=np.linspace(0,2,steps+1)[1:]
        K=.5*(t[:,None]**(2*H)+t[None,:]**(2*H)-abs(t[:,None]-t[None,:])**(2*H))
        lam,U=np.linalg.eigh(K)
        self.factor=np.diff(np.vstack([np.zeros((1,steps)),U[:,::-1]*np.sqrt(np.maximum(lam[::-1],0))]),axis=0)
        self.reconstruction_error=float(np.max(abs(np.cumsum(self.factor,0)@np.cumsum(self.factor,0).T-K)))
    def produce(self,n,seed,coarse=False):
        sampler=qmc.Sobol(self.steps,scramble=True,seed=seed);out=np.empty((n,33))
        lower=np.empty_like(out) if coarse else None
        for start in range(0,n,BATCH):
            size=min(BATCH,n-start)
            z=ndtri(np.clip(sampler.random(size),1e-15,1-1e-15));inc=z@self.factor.T
            out[start:start+size]=heun(inc)
            if coarse:lower[start:start+size]=heun(inc.reshape(size,self.steps//2,2).sum(axis=2))
        return (out,lower) if coarse else out
class Chaos:
    def __init__(self,H,J=3,p=5,q=20):
        hfc._rules.clear();hwc._TENSORS.clear()
        self.J=J
        self.r=hwc.solve(bm.neural_equation(),H,2.,J,p,TIMES,quad_order=q,
            max_nodes=600000,rtol=2e-8,atol=2e-10)
        self.coef=self.r.coef[:,:,0]
    def produce(self,n,seed):
        rng=np.random.default_rng(seed);out=np.empty((n,33))
        for start in range(0,n,BATCH):
            size=min(BATCH,n-start);xi=rng.standard_normal((size,self.J))
            out[start:start+size]=self.r.basis.evaluate(xi)@self.coef.T
        return out
    def exact_metrics(self):
        c=self.coef;K=c[:,1:]@c[:,1:].T
        return dict(mean=c[:,0].tolist(),variance=np.diag(K).tolist(),covariance=K.tolist())
def make_reference():
    saved=read('reference')
    for H in [.75,.95]:
        key=str(H)
        if key in saved:continue
        start=time.perf_counter();m=FullRQMC(H);setup=time.perf_counter()-start
        fine=[];coarse=[];reps=[];coarsereps=[]
        for seed in range(8801,8809):
            x,xc=m.produce(8192,seed,coarse=True)
            fine.append(x);coarse.append(xc);reps.append(metrics(x));coarsereps.append(metrics(xc))
        f=metrics(np.concatenate(fine));c=metrics(np.concatenate(coarse))
        saved[key]=dict(H=H,setup_seconds=setup,total_verification_seconds=time.perf_counter()-start,
            reference=f,coarse_reference=c,replicates=reps,coarse_replicates=coarsereps,
            paths_per_replicate=8192,seeds=list(range(8801,8809)),steps=512,
            covariance_reconstruction_error=m.reconstruction_error)
        save('reference',saved)
        print('REFERENCE',H,'var(T)',f['variance'][-1],'sec',round(saved[key]['total_verification_seconds'],2),flush=True)
def benchmark():
    setups=read('setups');runs=read('runs')
    for H in [.75,.95]:
        for label,constructor in [('cos_J3_p5_q20',lambda:Chaos(H)),('full_MC',lambda:FullMC(H)),('full_RQMC',lambda:FullRQMC(H))]:
            key=f'{H}_{label}'
            if all(f'{key}_{n}' in runs for n in [1024,8192,65536]):continue
            duration=[]
            for _ in range(3):
                t=time.perf_counter();model=constructor();duration.append(time.perf_counter()-t)
            setups[key]=dict(seconds=duration,median_seconds=float(np.median(duration)))
            if isinstance(model,Chaos):
                setups[key]['exact_surrogate_moments']=model.exact_metrics()
                np.savez_compressed(OUT/(key+'_coefficients.npz'),times=TIMES,coef=model.coef)
            save('setups',setups)
            for n in [1024,8192,65536]:
                runkey=f'{key}_{n}'
                if runkey in runs:continue
                reps=[]
                for seed in [1101,1102,1103]:
                    t=time.perf_counter();x=model.produce(n,seed);elapsed=time.perf_counter()-t
                    reps.append(dict(seed=seed,online_seconds=elapsed,metrics=metrics(x)))
                runs[runkey]=dict(H=H,method=label,N=n,replicates=reps)
                save('runs',runs)
                print('RUN',key,n,'setup',round(np.median(duration),4),'online',round(np.median([r['online_seconds'] for r in reps]),4),flush=True)
def refinements():
    saved=read('refinements')
    for H,J,p,q in [(.75,3,5,32),(.95,3,5,32),(.75,5,3,12),(.75,5,3,14)]:
        key=f'{H}_J{J}_p{p}_q{q}'
        if key in saved:continue
        times=[]
        for _ in range(3 if q==12 else 1):
            start=time.perf_counter();m=Chaos(H,J,p,q);times.append(time.perf_counter()-start)
        np.savez_compressed(OUT/(key+'_coefficients.npz'),times=TIMES,coef=m.coef)
        samples={}
        if J==5 and q==12:
            for n in [1024,8192,65536]:
                reps=[]
                for seed in [1101,1102,1103]:
                    t=time.perf_counter();x=m.produce(n,seed);elapsed=time.perf_counter()-t
                    reps.append(dict(seed=seed,online_seconds=elapsed,metrics=metrics(x)))
                samples[str(n)]=reps
        saved[key]=dict(H=H,J=J,p=p,q=q,setup_seconds=times,
            exact_surrogate_moments=m.exact_metrics(),samples=samples)
        save('refinements',saved)
        print('REFINEMENT',key,'setup',np.round(times,3),'var(T)',m.exact_metrics()['variance'][-1],flush=True)
if __name__=='__main__':
    save('environment',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),processor=platform.processor(),threads=1,times=TIMES.tolist()))
    globals()[sys.argv[1]]()
