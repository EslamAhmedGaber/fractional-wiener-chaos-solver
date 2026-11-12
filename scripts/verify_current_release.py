"""Check the current release and small numerical regressions without rerunning large simulations."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import ast, hashlib, json, subprocess, sys
import numpy as np
R=Path(__file__).resolve().parents[1]; D=R/'reproduction/2026-11-12'
m=R/'MANIFEST_SHA256.json'
if m.exists():
    manifest=json.loads(m.read_text(encoding='utf8'))
    for name,digest in manifest.items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
    print('PASS manifest:',len(manifest),'files',flush=True)
for f in R.rglob('*.py'):
    if '.git' not in f.parts: ast.parse(f.read_text(encoding='utf-8-sig'))
print('PASS Python syntax',flush=True)
subprocess.run([sys.executable,str(D/'scripts/make_tables.py')],check=True)
refs=list((D/'reference_tables').rglob('*.tex'))
assert len(refs)==5
for f in refs:
    target=D/f.relative_to(D/'reference_tables')
    assert f.read_text(encoding='utf8')==target.read_text(encoding='utf8'),str(target)
print('PASS regenerated tables:',len(refs),flush=True)
subprocess.run([sys.executable,str(D/'scripts/pool_rare_events.py')],check=True)
for f,n in ((D/'figures/figure_checks.json',40),(D/'figures/flagship/checks.json',8)):
    if f.exists():
        c=json.loads(f.read_text());assert len(c)==n and all(x['ok'] for x in c),f
        print('PASS figure data assertions:',n,flush=True)
    else:
        print('NOT RUN: generate figures to validate',f.name,flush=True)
sys.path.insert(0,str(R/'solver'))
import hfc,hwc
G=hfc.gram(.5,2.,'cos',17);assert np.allclose(G,np.eye(17),atol=1e-13)
G=hfc.gram(.75,2.,'cos',17);assert np.linalg.eigvalsh(G).min()>0
i,j=np.indices(G.shape);assert np.max(abs(G[(i+j)%2==1]))<1e-13
P,Eq=hwc.Poly,hwc.Equation
eq=Eq(1.,P([[(-1.,(1,)),(-1.,(3,))]]),P([[(.5,(0,))]]))
r=hwc.solve(eq,.75,4.,17,2,np.array([.5,1.,2.,4.]))
stored=json.loads((R/'evidence/data/v12_cubic.json').read_text())['sw']['0.5_0.75_17_2']
assert np.allclose(r.mean[:,0],stored['mean'],rtol=1e-8,atol=1e-11)
assert np.allclose(r.variance()[:,0],stored['var'],rtol=1e-8,atol=1e-11)
print('PASS current engine: Brownian limit, Gram positivity/parity, cubic mean/variance',flush=True)
subprocess.run([sys.executable,str(D/'scripts/round12_experiments/validate_backend.py')],check=True)
print('ALL REQUESTED CURRENT-RELEASE CHECKS PASSED',flush=True)
