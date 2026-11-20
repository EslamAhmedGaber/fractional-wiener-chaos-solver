"""Paired check of cross-covariance and terminal ordering at two fBm grid sizes."""
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parents[1] / "evidence/P3_lotka_volterra"
sys.path.insert(0, str(BASE))
import p3_lotka_volterra as lv


def check(hurst):
    seed = 20261120 + round(100 * hurst)
    rng = np.random.default_rng(seed)
    idx = np.arange(1, 9) * 64
    fine, coarse = [], []
    for _ in range(8):
        increments = lv.fgn_increments(hurst, 512, 1024, rng, 16.)
        path = np.column_stack((np.zeros(1024), increments.cumsum(1)))
        fine.append(lv.lv_wis_rk4(path, hurst, idx, 16./512))
        coarse.append(lv.lv_wis_rk4(path[:, ::2], hurst, idx//2, 16./256))
    fine, coarse = np.concatenate(fine, axis=1), np.concatenate(coarse, axis=1)
    cov_f, cov_c = [np.array([np.cov(y.T) for y in paths]) for paths in (fine, coarse)]
    scale = np.sqrt(cov_f[:, 0, 0] * cov_f[:, 1, 1])
    switched = (fine[-1, :, 0] > fine[-1, :, 1]) != (coarse[-1, :, 0] > coarse[-1, :, 1])
    return dict(H=hurst, seed=seed, paths=len(switched),
                max_normalised_cross_change_pct=float(100*np.max(np.abs(cov_f[:, 0, 1]-cov_c[:, 0, 1])/scale)),
                terminal_order_switches=int(switched.sum()))


if __name__ == "__main__":
    with threadpool_limits(1):
        rows = [check(h) for h in lv.HS]
    (BASE/"paired_refinement.json").write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
    print(rows)

