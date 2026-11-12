"""Independent numerical checks and fresh-seed replication; never edits historical rounds."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,math
import numpy as np
from scipy.special import roots_jacobi,roots_legendre
from scipy.linalg import solve_triangular
ROOT=Path(__file__).resolve().parents[1]; WK=ROOT
sys.path.insert(0,str(WK/'scripts/round14_experiments'))
import rare_events as re
ic=re.ic; hfc=ic.hfc
OUT=ROOT/'reports'; OUT.mkdir(exist_ok=True); checks=[]; reps=[]
def check(name,value,tol):
    checks.append(dict(name=name,value=float(value),tolerance=tol,passed=bool(value<tol)))
def save():
    (OUT/'independent_validation.json').write_text(json.dumps(dict(checks=checks,replications=reps),indent=2))
def gram_quad(H,T,J,n=160):
    x,w=roots_jacobi(n,0,2*H-2); r=(x+1)/2; w=w/2**(2*H-1)
    s,ws=roots_legendre(n);s=(s+1)/2;ws=ws/2
    v=T*(1-r[:,None])*s[None,:];u=v+T*r[:,None]
    hu=hfc.h('cos',J,T,u);hv=hfc.h('cos',J,T,v)
    return H*(2*H-1)*T**(2*H)*(np.einsum('a,b,a,abi,abj->ij',w,ws,1-r,hu,hv)+np.einsum('a,b,a,abi,abj->ij',w,ws,1-r,hv,hu))
for H in (.55,.75,.95):
    G=hfc.gram(H,2.,'cos',9);Gq=gram_quad(H,2.,9)
    check('direct two-dimensional Gram quadrature H='+str(H),np.max(abs(G-Gq)),2e-9)
    L=np.linalg.cholesky(G)
    check('whitening H='+str(H),np.max(abs(np.linalg.solve(L,G)@np.linalg.inv(L.T)-np.eye(9))),1e-11)
    G17=hfc.gram(H,2.,'cos',17)
    check('nested Cholesky H='+str(H),np.max(abs(np.linalg.cholesky(G17)[:9,:9]-L)),1e-10)
    check('T scaling H='+str(H),np.max(abs(G-2**(2*H-1)*hfc.gram(H,1.,'cos',9))),1e-12)
    M=hfc.Mcoef('cos',9,2.,np.array([2.]))
    check('endpoint variance H='+str(H),abs(float((M@G@M.T)[0,0])-2**(2*H)),1e-12)
check('Brownian Gram identity',np.max(abs(hfc.gram(.5,2.,'cos',9)-np.eye(9))),1e-14)
# Independent Gaussian polynomial moment identity, full tensor Gauss-Hermite at small J.
from scipy.special import roots_hermitenorm
x,w=roots_hermitenorm(5);w=w/np.sqrt(2*np.pi)
xi=np.stack(np.meshgrid(x,x,x,indexing='ij'),axis=-1).reshape(-1,3)
ww=np.einsum('i,j,k->ijk',w,w,w).ravel()
G=hfc.gram(.75,2.,'cos',3);L=np.linalg.cholesky(G);Z=xi@L.T
rng=np.random.default_rng(20261115);a=rng.normal(size=3);A=rng.normal(size=(3,3));A=(A+A.T)/2
Y=Z@a+np.einsum('bi,ij,bj->b',Z,A,Z)-np.trace(A@G)
check('correlated/orthogonal quadratic variance',abs(float(ww@(Y*Y))-(a@G@a+2*np.trace(A@G@A@G))),1e-10)
check('centred quadratic mean',abs(float(ww@Y)),1e-12)
# References: counts, Bernoulli standard errors, and complete record sets.
for f in (WK/'evidence/round14').glob('*well_reference*.json'):
    d=json.loads(f.read_text());M=d['M']
    for c,e in d['levels'].items():
        check(f.name+' p=hits/M c='+c,abs(e['p']-e['hits']/M),1e-15)
        check(f.name+' binomial se c='+c,abs(e['se']-math.sqrt(e['p']*(1-e['p'])/(M-1))),1e-13)
rows=json.loads((WK/'evidence/round14/ray_proposals.json').read_text())
check('all 36 candidate records present',abs(len(rows)-36),.5)
save()
# Fresh seeds for all 18 events; degree chosen before drawing validation samples.
for mod in ('single','double'):
    m=re.MODELS[mod]
    for iH,H in enumerate(re.H_LIST):
        coords=ic.DiscreteCoords(H,m['T'],9,512)
        sols={p:ic.coschaos_cubic(H,m['T'],9,p,x0=m['x0'],a=m['a'],b=m['b'],s=m['s'])[0] for p in (1,3)}
        for ci,c in enumerate(m['levels']):
            t0=time.perf_counter();cand={}
            for p in (1,3):
                xs,pv=ic.design_point(sols[p],c,sign=m['sign']);d=xs/np.linalg.norm(xs)
                scale,nev=re.ray_scale(coords,d,c,m)
                cand[p]=(scale*d,scale**2)
                old=rows[f'{mod}_H{H}_c{c}_p{p}']
                check(f'ray norm reproduction {mod} {H} {c} p{p}',abs(scale**2-old['ray_norm_sq']),1e-5)
            p=min(cand,key=lambda p:cand[p][1]);shift=cand[p][0]
            seed=[20261111,1501,0 if mod=='single' else 1,iH,ci]
            est=ic.is_estimate(H,m['T'],512,coords,shift,[c],65536,np.random.default_rng(np.random.SeedSequence(seed)),m['drift'],m['x0'],m['s'],chunk=4096,sign=m['sign'])
            old=rows[f'{mod}_H{H}_c{c}_p{p}'];pi=float(est['p'][0]);se=float(est['se'][0])
            e=dict(model=mod,H=H,c=c,degree=p,M=65536,seed=seed,p=pi,se=se,
                   z_vs_original=abs(pi-old['p_is'])/math.hypot(se,old['se_is']),
                   vrf=pi*(1-pi)/(65536*se**2),seconds=time.perf_counter()-t0)
            refs=[]
            if mod=='double':
                rf=json.loads((WK/f'evidence/round14/double_well_reference_H{H}.json').read_text())['levels'][str(c)]
                refs.append(('new_crude',rf['p'],rf['se']))
            else:
                rf=json.loads((WK/'evidence/round12/single_well_is_matrix.json').read_text())[str(H)][f'p1_c{c}']
                refs.append(('stored_20million',rf['p_ref'],rf['se_ref']))
                if H==.55:
                    rf=json.loads((WK/'evidence/round14/single_well_reference_H0.55.json').read_text())['levels'][str(c)]
                    refs.append(('new_16million',rf['p'],rf['se']))
            e['references']=[dict(source=src,p=pr,se=sr,z=abs(pi-pr)/math.hypot(se,sr)) for src,pr,sr in refs]
            reps.append(e);save()
            print(mod,H,c,'p',p,'P',pi,'VRF',round(e['vrf']),'z old',round(e['z_vs_original'],2),'z refs', [round(v['z'],2) for v in e['references']],flush=True)
print('DONE checks',len(checks),'failures',[x for x in checks if not x['passed']],flush=True)
