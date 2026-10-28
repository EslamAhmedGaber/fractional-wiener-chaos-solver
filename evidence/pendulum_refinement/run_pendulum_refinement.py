"""Coupled Heun refinement for the manuscript's large-angle pendulum.

One exact-grid Davies--Harte fBm sample supplies all three step sizes.
Sampling errors of differences use paired influence functions; they are
diagnostics of discretization stability, not rigorous bias bounds.
"""
from pathlib import Path
import argparse,json,time,platform,sys
import numpy as np

def dh_increments(H,n,m,rng):
    k=np.arange(n+1,dtype=float)
    cov=.5*(abs(k+1)**(2*H)-2*abs(k)**(2*H)+abs(k-1)**(2*H))
    lam=np.fft.fft(np.r_[cov,cov[-2:0:-1]]).real
    if lam.min() < -1e-9: raise ValueError("Circulant embedding not positive")
    z=rng.standard_normal((m,2*n))+1j*rng.standard_normal((m,2*n))
    return np.fft.fft(z*np.sqrt(np.maximum(lam,0)/(2*n)),axis=1)[:,:n].real*n**(-H)

def simulate(db,T,times):
    m,n=db.shape; dt=T/n
    q=np.ones(m); v=np.zeros(m)
    out=np.empty((m,len(times),2)); ids=np.rint(np.asarray(times)/dt).astype(int)
    assert np.allclose(ids*dt,times)
    j=0
    for i in range(n):
        av=-np.sin(q)-.6*v
        qp=q+dt*v; vp=v+dt*av+db[:,i]
        qn=q+.5*dt*(v+vp)
        v=v+.5*dt*(av-np.sin(qp)-.6*vp)+db[:,i]
        q=qn
        if j<len(ids) and i+1==ids[j]:
            out[:,j,0]=q;out[:,j,1]=v;j+=1
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--paths",type=int,default=2**17)
    ap.add_argument("--fine-steps",type=int,default=4000)
    ap.add_argument("--chunk",type=int,default=512)
    ap.add_argument("--seed",type=int,default=20261028)
    ap.add_argument("--output",type=Path,default=Path(__file__).with_name("pendulum_refinement.json"))
    a=ap.parse_args()
    assert a.fine_steps%4==0
    start=time.perf_counter(); H=.9; T=10.; sigma=.4; times=[1.,2.,5.,10.]
    levels=[a.fine_steps//4,a.fine_steps//2,a.fine_steps]
    values=np.empty((a.paths,3,4,2))
    rng=np.random.default_rng(a.seed)
    for first in range(0,a.paths,a.chunk):
        last=min(first+a.chunk,a.paths); m=last-first
        db=dh_increments(H,a.fine_steps,m,rng)*sigma*T**H
        for j,n in enumerate(levels):
            increments=db.reshape(m,n,a.fine_steps//n).sum(2)
            values[first:last,j]=simulate(increments,T,times)
        if first==0 or last%16384==0 or last==a.paths:
            print(json.dumps({"paths":last,"seconds":round(time.perf_counter()-start,2)}),flush=True)
    means=values.mean(0); centered=values-means
    variances=(centered**2).mean(0)
    ifvar=centered**2-variances
    mean_se=values.std(0,ddof=1)/np.sqrt(a.paths)
    var_se=ifvar.std(0,ddof=1)/np.sqrt(a.paths)
    pairs=[]
    for lo,hi in [(0,1),(1,2),(0,2)]:
        dmean=means[hi]-means[lo];dvar=variances[hi]-variances[lo]
        se_m=(values[:,hi]-values[:,lo]).std(0,ddof=1)/np.sqrt(a.paths)
        se_v=(ifvar[:,hi]-ifvar[:,lo]).std(0,ddof=1)/np.sqrt(a.paths)
        pairs.append(dict(coarse_steps=levels[lo],fine_steps=levels[hi],
            mean_difference=dmean.tolist(),mean_difference_se=se_m.tolist(),
            variance_difference=dvar.tolist(),variance_difference_se=se_v.tolist(),
            variance_relative_percent=(100*dvar/variances[hi]).tolist()))
    result=dict(parameters=dict(H=H,T=T,sigma=sigma,omega=1.,zeta=.3,Q0=1.,V0=0.),
        paths=a.paths,seed=a.seed,chunk=a.chunk,levels=levels,times=times,components=["Q","V"],
        means=means.tolist(),variances=variances.tolist(),mean_se=mean_se.tolist(),variance_se=var_se.tolist(),
        paired_comparisons=pairs,seconds=time.perf_counter()-start,
        method="Exact-grid Davies--Harte increments; Heun; coarse increments summed from the same fine path",
        uncertainty="Paired first-order influence-function standard errors; no rigorous time-bias bound",
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(a.output),"seconds":result["seconds"]}),flush=True)
if __name__=="__main__":main()

