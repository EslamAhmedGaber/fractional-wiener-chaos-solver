"""Generate the focused-update numbers from numerical records, without editing prose."""
from pathlib import Path
import json,math,numpy as np
R=Path(__file__).resolve().parents[1]
d=json.loads((R/"evidence/pendulum_refinement/pendulum_refinement.json").read_text())
o=json.loads((R/"evidence/data/v12_oscillator.json").read_text())
old=json.loads((R/"evidence/solver_snapshot/reference_results/v12_new.json").read_text())
q=np.array(d["variances"])[:,-1,0];v=np.array(d["variances"])[:,-1,1];se=np.array(d["variance_se"])[:,-1,0]
pair=d["paired_comparisons"][-1]
maximum=float(np.max(abs(np.array(pair["variance_relative_percent"]))))
macros={"AuditEnergyRatioLow":f"{o['sweep'][0]['ratio']:.2f}","AuditEnergyRatioHigh":f"{o['sweep'][-1]['ratio']:.2f}",
"PendRefMaxChange":f"{math.ceil(maximum*1e4)/1e4:.4f}","PendRefQChange":f"{abs(pair['variance_relative_percent'][-1][0]):.4f}",
"PendRefDeficit":f"{100*(1-old['pend']['0.9_65_1']['var'][-1][0]/q[-1]):.1f}",
"PendRefQFine":f"{q[-1]:.6f}","PendRefPaths":f"{d['paths']:,}".replace(",",r"\,")}
text="% Generated from stored results and coupled refinement; do not edit numbers.\n"
text+="".join("\\newcommand{\\"+k+"}{"+val+"}\n" for k,val in macros.items())
public=(R/"reference_tables").is_dir()
folders=[R/"reference_tables/main",R/"reference_tables/supplement"] if public else [R/"source/tables",R/"supplement/source/tables"]
for folder in folders:folder.mkdir(parents=True,exist_ok=True);(folder/"audit_update_macros.tex").write_text(text,encoding="utf-8")
table=r"""\begin{tabular}{rrrr}\toprule
Heun steps & $\widehat{\Var}Q(10)$ & s.e.\ of $\widehat{\Var}Q(10)$ & $\widehat{\Var}V(10)$\\\midrule
"""
for i,n in enumerate(d["levels"]):table+=f"{n} & {q[i]:.6f} & {se[i]:.6f} & {v[i]:.6f}"+r"\\"+"\n"
table+=r"\bottomrule\end{tabular}"+"\n"
(folders[1]/"tab_pendulum_refinement.tex").write_text(table,encoding="utf-8")
print(json.dumps(macros,indent=2))

