"""Exploratory benchmarks, unmodified CosChaos v1.1.3 solver.
Usage: python benchmark.py affine | chaos | mc | samej
Single BLAS thread. Results and checkpointed rows are saved after each run.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import hashlib
import json
import math
import platform
import sys
import time
import numpy as np
import scipy
from scipy.special import roots_jacobi, roots_legendre
from scipy.stats import norm, t as student_t
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "solver_snapshot"))
import hwc
import hfc
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
H_VALUES = [0.5, 0.75, 0.95]
T_NEURAL = 2.0
SIGMA = 0.7
X0 = 0.2
MC_PATHS = 65536
BATCH = 2048
THRESHOLD = 0.7


def save(name, data):
    (OUT / (name + ".json")).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def read(name):
    p = OUT / (name + ".json")
    return json.loads(p.read_text()) if p.exists() else {}


def timed(call, repeats=3):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        values.append(time.perf_counter() - start)
    return result, values


def beta(t):
    return 0.1 + 19.9 * np.asarray(t)


def integrated_beta(t):
    return 0.1 * np.asarray(t) + 9.95 * np.asarray(t)**2


def kernel(t, s):
    return np.sqrt(beta(s)) * np.exp(-0.5 * (integrated_beta(t) - integrated_beta(s)))


def exact_variances(H, times, order):
    """Weighted quadrature on the two triangles; independent of Fourier/Gram."""
    if H == 0.5:
        return -np.expm1(-integrated_beta(times))
    power = 2*H - 1
    # outer z^power, inner (1-r)^(power-1), both on [0,1].
    zo, wo = roots_jacobi(order, 0, power)
    ri, wi = roots_jacobi(order, power-1, 0)
    zo = (zo+1)/2
    wo = wo / 2**(power+1)
    ri = (ri+1)/2
    wi = wi / 2**power
    values = []
    for t in times:
        s = t*zo
        inner = kernel(t, s[:, None]*ri) @ wi
        values.append(2*H*power*t**(power+1) * (wo @ (kernel(t, s)*inner)))
    return np.array(values)


def affine():
    times = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    eq = hwc.Equation([1.0], hwc.Poly([[(lambda t: -beta(t)/2, (1,))]]),
                      hwc.Poly([[(lambda t: np.sqrt(beta(t)), (0,))]]),
                      name="standard-fBm forward diffusion")
    out = {"times": times.tolist(), "reference": {}, "cases": []}
    for H in H_VALUES:
        ref, runtime = timed(lambda: exact_variances(H, times, 256))
        coarse, coarse_time = timed(lambda: exact_variances(H, times, 128))
        fast, fast_time = timed(lambda: exact_variances(H, times, 64))
        out["reference"][str(H)] = dict(variance=ref.tolist(), seconds=runtime,
            refinement_rel=float(np.max(abs(coarse/ref-1))),
            q64_rel=float(np.max(abs(fast/ref-1))), q64_seconds=fast_time,
            q128_seconds=coarse_time)
    for J in [5, 17, 33, 65, 129]:
        result, solves = timed(lambda: hwc.solve(eq, 0.5, 1., J, 1, times,
                                        form="correlated", rtol=1e-9, atol=1e-11))
        a = result.coef[:, 1:, 0]
        start = time.perf_counter()
        rows = []
        for H in H_VALUES:
            startg = time.perf_counter()
            G = hfc.gram(H, 1., "cos", J)
            gt = time.perf_counter() - startg
            startm = time.perf_counter()
            v = np.einsum("ti,ij,tj->t", a, G, a)
            mt = time.perf_counter()-startm
            ref = np.array(out["reference"][str(H)]["variance"])
            rows.append(dict(H=H, J=J, variance=v.tolist(), relative_variance_error=(v/ref-1).tolist(),
                max_abs_relative_error=float(np.max(abs(v/ref-1))),
                max_relative_score_slope_error=float(np.max(abs(ref/v-1))),
                gram_seconds=gt, contraction_seconds=mt))
        out["cases"].append(dict(J=J, one_solve_seconds=solves, all_H_post_seconds=time.perf_counter()-start,
            mean_abs_error=float(np.max(abs(result.mean[:,0]-np.exp(-0.5*integrated_beta(times))))),
            rows=rows))
        save("affine",out)
        print("AFFINE",J, [(r["H"], round(100*r["max_abs_relative_error"],5)) for r in rows],
              "solve",round(np.median(solves),4),flush=True)
    # Accuracy of the numerical Gram implementation checked independently on four entries.
    out["gram_check"] = {}
    order = 160
    for H in [0.75, 0.95]:
        power=2*H-1
        z,w = roots_jacobi(order,0,power)
        z=(z+1)/2
        w=w/2**(power+1)
        r,v = roots_jacobi(order,power-1,0)
        r=(r+1)/2
        v=v/2**power
        hz=hfc.h("cos",4,1.,z)
        hr=hfc.h("cos",4,1.,(z[:,None]*r).ravel()).reshape(order,order,4)
        integ=np.einsum("r,zrj->zj",v,hr)
        Gq=H*power*(hz.T@(w[:,None]*integ)+integ.T@(w[:,None]*hz))
        out["gram_check"][str(H)]=float(np.max(abs(Gq-hfc.gram(H,1.,"cos",4))))
    save("affine",out)


def drift(t,x):
    return -1.5*x+1.2*np.tanh(x)+0.2*np.cos(np.pi*t)


def neural_equation():
    return hwc.Equation([X0], drift, hwc.Poly([[(SIGMA, (0,))]]),name="contractive tanh recurrent unit")


def batch_stats(x):
    return np.array([x.mean(),(x*x).mean(),np.mean(x>THRESHOLD)])


def summarize(samples):
    x = np.concatenate(samples)
    b = np.array([batch_stats(a) for a in samples])
    mu = x.mean()
    var = np.mean(x*x)-mu*mu
    influence_var = b[:,1]-2*mu*b[:,0]
    se = b.std(axis=0,ddof=1)/math.sqrt(len(b))
    v_se = influence_var.std(ddof=1)/math.sqrt(len(b))
    return dict(N=len(x), batches=len(b), mean=float(mu), variance=float(var),
                mean_se=float(se[0]), variance_se=float(v_se),
                tail=float((x>THRESHOLD).mean()), tail_se=float(se[2]),
                q95=float(np.quantile(x,0.95)),
                ci_multiplier=float(student_t.ppf(.975,len(b)-1)))


def chaos():
    out = read("chaos")
    configs = [(3,1,20),(3,3,12),(3,3,20),(3,5,12),(3,5,20),
               (5,1,20),(17,1,20),(33,1,20),
               (5,3,6),(5,3,8),(5,3,10)]
    for H in H_VALUES:
        for J,p,q in configs:
            key=f"{H}_{J}_{p}_{q}"
            if key in out:
                continue
            result, timings = timed(lambda: hwc.solve(neural_equation(), H,T_NEURAL,J,p,[1.,2.],
                                          quad_order=q,rtol=2e-8,atol=2e-10), repeats=1)
            start=time.perf_counter()
            rng=np.random.default_rng(20261110+int(100*H)+J*100)
            samples=[]
            # Identical coordinate samples at a fixed H,J for all degree/quad choices.
            for _ in range(32):
                xi=rng.standard_normal((4096,J))
                samples.append(result.basis.evaluate(xi)@result.coef[-1,:,0])
            samples=np.array(samples)
            elapsed=time.perf_counter()-start
            out[key]=dict(H=H,J=J,p=p,q=q,solve_seconds=timings[0],
                mean=result.mean[:,0].tolist(),variance=result.variance()[:,0].tolist(),
                n_coefficients=result.basis.n, gh_nodes=q**J if p>1 else q,
                sampling_seconds=elapsed,law=summarize(list(samples)),
                nfev=result.info["nfev"])
            np.savez_compressed(OUT/("chaos_"+key+".npz"),coef=result.coef[-1,:,0],samples=samples)
            save("chaos",out)
            print("CHAOS",key,"sec",round(timings[0],3),"var",out[key]["variance"][-1],
                  "tail",out[key]["law"]["tail"],flush=True)


def fgn(H,n,M,dt,rng):
    """Same circulant embedding as the published round3c_fullsolution.py."""
    k=np.arange(n+1)
    g=0.5*((k+1.)**(2*H)-2*k**(2*H)+abs(k-1.)**(2*H))
    g[0]=1.
    lam=np.fft.fft(np.r_[g,g[-2:0:-1]]).real
    if lam.min() < -1e-9:
        raise ValueError("Circulant embedding is not positive")
    m=2*n
    z=(rng.standard_normal((M//2,m))+1j*rng.standard_normal((M//2,m)))
    w=np.fft.fft(np.sqrt(np.maximum(lam,0)/m)*z,axis=1)[:,:n]
    return np.r_[w.real,w.imag]*dt**H


def heun(dB,dt):
    noise=np.ascontiguousarray(dB.T)
    x=np.full(dB.shape[0],X0)
    for k,inc in enumerate(noise):
        t=k*dt
        f=drift(t,x)
        pred=x+dt*f+SIGMA*inc
        x+=0.5*dt*(f+drift(t+dt,pred))+SIGMA*inc
    return x


def mc():
    out=read("mc")
    for H in H_VALUES:
        if str(H) in out:
            continue
        fine=[]
        coarse=[]
        elapsed_gen=elapsed_fine=elapsed_coarse=0.
        rng=np.random.default_rng(20261110+int(100*H))
        starts=time.perf_counter()
        for batch in range(MC_PATHS//BATCH):
            t=time.perf_counter()
            db=fgn(H,1024,BATCH,T_NEURAL/1024,rng)
            elapsed_gen+=time.perf_counter()-t
            t=time.perf_counter()
            fine.append(heun(db,T_NEURAL/1024))
            elapsed_fine+=time.perf_counter()-t
            t=time.perf_counter()
            coarse.append(heun(db.reshape(BATCH,512,2).sum(axis=2),T_NEURAL/512))
            elapsed_coarse+=time.perf_counter()-t
        r=dict(fine=summarize(fine),coarse=summarize(coarse),
               generation_seconds=elapsed_gen,heun_seconds=elapsed_fine,
               extra_coarse_seconds=elapsed_coarse,seconds=elapsed_gen+elapsed_fine,
               steps=1024,coarse_steps=512)
        r["paired_mean_delta"]=float(np.mean(np.array(fine)-np.array(coarse)))
        r["paired_rms_delta"]=float(np.sqrt(np.mean((np.array(fine)-np.array(coarse))**2)))
        out[str(H)]=r
        np.savez_compressed(OUT/f"mc_{H}.npz",fine=np.array(fine),coarse=np.array(coarse))
        save("mc",out)
        print("MC",H,"sec",round(r["seconds"],3),"var",r["fine"]["variance"],
              "SE",r["fine"]["variance_se"],"step_delta",r["paired_rms_delta"],flush=True)


def samej():
    out=read("samej")
    for H in H_VALUES:
        for J in [3,5]:
            key=f"{H}_{J}"
            if key in out:
                continue
            start=time.perf_counter()
            noise=hwc.HarmonicNoise(H,T_NEURAL,J)
            tg=np.linspace(0,T_NEURAL,513)
            increments=np.diff(noise.m(tg)@noise.L,axis=0)
            rng=np.random.default_rng(20261110+int(100*H)+J*100)
            samples=[]
            coarse=[]
            for _ in range(32):
                xi=rng.standard_normal((4096,J))
                db=xi@increments.T
                samples.append(heun(db,T_NEURAL/512))
                coarse.append(heun(db.reshape(4096,256,2).sum(axis=2),T_NEURAL/256))
            duration=time.perf_counter()-start
            out[key]=dict(H=H,J=J,law=summarize(samples),seconds=duration,
                          coarse=summarize(coarse),
                          paired_rms_delta=float(np.sqrt(np.mean((np.array(samples)-np.array(coarse))**2))))
            np.savez_compressed(OUT/f"samej_{key}.npz",samples=np.array(samples),coarse=np.array(coarse))
            save("samej",out)
            print("SAMEJ",key,"sec",round(duration,3),"var",out[key]["law"]["variance"],flush=True)


if __name__=="__main__":
    save("environment",dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
         platform=platform.platform(),processor=platform.processor(),
         blas_threads=1,solver_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
         for f in (ROOT/"solver_snapshot").glob("*.py")}))
    globals()[sys.argv[1]]()
