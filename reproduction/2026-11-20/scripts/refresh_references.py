"""Recompute the nine full-size independent references while retaining measured comparator runs."""
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def compute(case):
    model, lam, hurst = case
    with threadpool_limits(1):
        if model == "population":
            directory = ROOT / "evidence/P2_population"
            sys.path.insert(0, str(directory))
            import p2_population
            result = p2_population.run_reference(lam, hurst)
            name = directory / f"reference_lam{lam}_H{hurst}.json"
        else:
            directory = ROOT / "evidence/P3_lotka_volterra"
            sys.path.insert(0, str(directory))
            import p3_lotka_volterra
            result = p3_lotka_volterra.run_reference(hurst)
            return model, hurst, result
        name.write_text(json.dumps(result, indent=1), encoding="utf-8")
        return model, hurst, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    hs = (0.55, 0.75, 0.95)
    cases = [(m, l, h) for h in hs for m, l in (("population", 0.02), ("population", 0.05), ("competition", None))]
    target = ROOT / "evidence/P3_lotka_volterra/p3_results.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for model, h, result in pool.map(compute, cases):
            if model == "competition":
                data["cases"][str(h)]["reference"] = result
    target.write_text(json.dumps(data, indent=1), encoding="utf-8")

