"""variance share of each cosine coordinate xi_k in the degree-one CosChaos solution at t = 4
(cubic single well, J = 33).  Guides the anisotropic index sets of round5_ladder.py.
Run from the repository root:  python scripts/round5_modeshare.py   ->  evidence/round5/P5_mode_shares.json
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(W, 'evidence', 'round3_fwhe', 'solver_v1.1.3')); sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solver'))   # release layout: solver/ next to scripts/
import hwc as HW
from round3c_fullsolution import cubic_eq
out = {}
for H in (0.55, 0.75, 0.95):
    r = HW.solve(cubic_eq(), H, 4.0, 33, 1, np.array([4.0]), rtol=1e-8, atol=1e-10)
    c = r.coef[0, 1:, 0]; sh = c ** 2 / np.sum(c ** 2)
    out[str(H)] = dict(share=sh.tolist(), cum={k: float(sh[:k].sum()) for k in (1, 2, 3, 5, 9, 13, 17, 25, 33)})
json.dump(out, open(os.path.join(W, 'evidence', 'round5', 'P5_mode_shares.json'), 'w'), indent=1)
print({H: {k: round(v, 4) for k, v in o['cum'].items()} for H, o in out.items()})
