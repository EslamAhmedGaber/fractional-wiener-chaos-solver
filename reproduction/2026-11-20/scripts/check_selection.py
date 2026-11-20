"""Compare eligibility under two pointwise rules on the same measured settings."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
EVID = BASE / "evidence"


def select(rows, key):
    eligible = [r for r in rows if r["max_mean"] <= 1 and r[key] <= 1]
    return min(eligible, key=lambda r:r["seconds"]) if eligible else None


def compare(tag, left, right, error):
    result = {"case":tag}
    for key, name in ((error, "observed"), ("criterion", "common_margin")):
        a, b = select(left, key), select(right, key)
        result[name] = None if a is None or b is None else {
            "coschaos_s":a["seconds"], "sampler_s":b["seconds"],
            "ratio":b["seconds"]/a["seconds"], "J":a["J"], "p":a["p"]}
    return result


if __name__ == "__main__":
    p2 = json.loads((EVID/"P2_population/p2_summary.json").read_text())
    p3 = json.loads((EVID/"P3_lotka_volterra/p3_summary.json").read_text())
    rows = []
    for lam in ("0.02", "0.05"):
        case = p2[lam+"_0.95"]["rows"]
        rows.append(compare("logistic_"+lam+"_0.95",
                            [r for r in case if r["method"]=="CosChaos"],
                            [r for r in case if r["method"]=="Exact WIS + MC"], "max_var"))
    for h, case in p3.items():
        rows.append(compare("competition_"+h, case["cc_evals"], case["rk_evals"], "max_cov"))
    (EVID/"selection_sensitivity.json").write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))

