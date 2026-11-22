"""Build manuscript numbers for end-to-end timings and eligibility sensitivity."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def read(filename):
    return json.loads((ROOT/"evidence"/filename).read_text(encoding="utf-8"))
sensitivity = {r["case"]:r for r in read("selection_sensitivity.json")}
timings = read("end_to_end.json")
if len(timings)!=4 or not all(r["both_eligible"] for r in timings):
    raise ValueError("Check the four end-to-end configurations against the common criterion")
row = next(r for r in timings if r["model"]=="competition" and r["H"]==0.95)
values = {"EndCCSec":format(row["coschaos_s"],".3g"),"EndWISSec":format(row["sampler_s"],".3g"),
          "EndRatio":format(row["ratio"],".1f")}
for lam, name in ((0.02, "EndPopWeak"), (0.05, "EndPopStrong")):
    row = next(r for r in timings if r["model"]=="logistic" and r["lam"]==lam)
    values[name] = format(row["ratio"],".1f")
for name,case in (("ObservedLV","competition_0.95"),("ObservedPopWeak","logistic_0.02_0.95"),("ObservedPopStrong","logistic_0.05_0.95")):
    values[name] = format(sensitivity[case]["observed"]["ratio"],".1f")
row = next(r for r in timings if r["model"]=="competition" and r["H"]==0.75)
values.update({"EndCCSecMid":format(row["coschaos_s"],".3g"),"EndWISSecMid":format(row["sampler_s"],".3g"),
               "EndRatioMid":format(row["ratio"],".1f")})
values["ObservedLVMid"] = format(sensitivity["competition_0.75"]["observed"]["ratio"],".1f")
# Sampler time projected to a 0.1% tolerance: Monte Carlo cost grows as eps^-2, i.e. 100 times the 1% run (minutes).
high = next(r for r in timings if r["model"]=="competition" and r["H"]==0.95)
values["SamplerTenthMin"] = format(100*high["sampler_s"]/60,".2g")
values["SamplerTenthMinMid"] = format(100*row["sampler_s"]/60,".2g")
(ROOT/"source/tables/validation_numbers.tex").write_text("".join(
    "\\newcommand{\\"+k+"}{"+v+"}\n" for k,v in values.items()),encoding="utf-8")

