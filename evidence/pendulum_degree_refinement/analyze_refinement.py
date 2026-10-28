from pathlib import Path
import json,csv,numpy as np
P=Path(__file__).resolve().parent
full=json.loads((P.parent/"pendulum_refinement/pendulum_refinement.json").read_text())
finite=json.loads((P/"finite_mode_mc_2000.json").read_text())
runs={}
for f in sorted(P.glob('*.json')):
 data=json.loads(f.read_text())
 if isinstance(data,dict) and 'runs' in data:runs.update(data['runs'])
rows=[]
for key,r in runs.items():
 if r["status"]!="completed":continue
 for k,t in enumerate(full["times"]):
  for dim,component in enumerate(["Q","V"]):
   v=r["variance"][k][dim];vr=full["variances"][-1][k][dim]
   fm=finite["results"].get(str(r["J"]))
   vf=fm["variance"][k][dim] if fm else None
   rows.append(dict(case=key,J=r["J"],p=r["p"],quadrature_order=str(r["quadrature_order"]),t=t,component=component,
      variance=v,full_fbm_reference=vr,relative_error_full_percent=100*(v/vr-1),
      finite_J_reference=vf,relative_error_finite_percent=(100*(v/vf-1) if vf else None),
      finite_J_z=((v-vf)/fm["se_variance"][k][dim] if fm else None),
      seconds=r["seconds"],nodes=r.get("nodes"),coefficient_count=r["coefficients_per_component"]))
with (P/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:
 if r["component"]=="Q" and r["t"]==10:print(r["case"],"var",round(r["variance"],6),"err full%",round(r["relative_error_full_percent"],2),"err sameJ%",round(r["relative_error_finite_percent"],2),"z",round(r["finite_J_z"],2),"s",round(r["seconds"],1))

