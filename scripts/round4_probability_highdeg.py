import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
import round4_probability as R4
path = os.path.join(R4.OUTD, 'P4_cubic_event.json')
for H, iH in [(0.95, 2), (0.75, 1), (0.55, 0)]:
    for J, p in [(9, 5)]:
        D = json.load(open(path)); rec = D[str(H)]; k = f'chaos_{J},{p}'
        if k in rec: continue
        rec[k] = R4.chaos(H, J, p, R4.SEED + 300 + iH); R4.log('chaos', H, J, p, round(rec[k]['seconds_solve'], 1), round(rec[k]['seconds_eval'], 1))
        D[str(H)] = rec; json.dump(D, open(path, 'w'), indent=1)
R4.log('done-hi')
