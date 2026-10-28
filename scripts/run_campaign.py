from pathlib import Path
import argparse,shutil,subprocess,sys
R=Path(__file__).resolve().parents[1]
names={"examples":"run_examples_v12.py","cubic":"run_cubic_v12.py","oscillator":"run_oscillator_v12.py","nonlinear":"run_new_v12.py","bench":"run_bench_v12.py","tables":"tables_v12.py"}
p=argparse.ArgumentParser();p.add_argument("campaign",choices=names);p.add_argument("--prepare-only",action="store_true")
a=p.parse_args();w=R/"runs"/a.campaign
for d in ["v6/code","v6/results","v6/paper/tables","apps"]: (w/d).mkdir(parents=True,exist_ok=True)
for f in (R/"evidence/solver_snapshot/code").glob("*.py"):shutil.copy2(f,w/"v6/code"/f.name)
for src in [R/"evidence/data",R/"evidence/solver_snapshot/reference_results"]:
 for f in src.glob("*.json"):shutil.copy2(f,w/"v6/results"/f.name)
print("Working directory:",w,flush=True)
if not a.prepare_only:subprocess.run([sys.executable,str(w/"v6/code"/names[a.campaign])],cwd=w,check=True)
