"""Pool independent importance-sampling estimates from their reconstructed second moments."""
from pathlib import Path
import csv, json, math
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'
old=json.loads((E/'round14/selected_proposals.json').read_text())
rep=json.loads((E/'rare_events_pooled/independent_validation.json').read_text())['replications']
rows=[]
for e in rep:
    o=next(x for x in old if (x['model'],x['H'],x['c'])==(e['model'],e['H'],e['c']))
    n1,n2=o['M'],e['M']; N=n1+n2
    mean=(n1*o['p_is']+n2*e['p'])/N
    second=(n1*((n1-1)*o['se_is']**2+o['p_is']**2)+n2*((n2-1)*e['se']**2+e['p']**2))/N
    se=math.sqrt(max(second-mean*mean,0)/(N-1))
    rows.append(dict(model=e['model'],H=e['H'],threshold=e['c'],guide_degree=e['degree'],
        original_M=n1,original_p=o['p_is'],original_se=o['se_is'],original_vrf=o['vrf'],
        replication_M=n2,replication_p=e['p'],replication_se=e['se'],replication_vrf=e['vrf'],
        pooled_M=N,pooled_p=mean,pooled_se=se,pooled_vrf=mean*(1-mean)/(N*se**2),
        max_replication_z_vs_reference=max(z['z'] for z in e['references']),
        replication_z_vs_original=e['z_vs_original']))
reference=json.loads((E/'rare_events_pooled/RARE_EVENTS_ALL_RESULTS.json').read_text())
assert rows==reference, 'Pooled numerical records differ from the published reference'
out=ROOT/'reports';out.mkdir(exist_ok=True)
(out/'RARE_EVENTS_ALL_RESULTS.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
with (out/'RARE_EVENTS_ALL_RESULTS.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('PASS exact pooled-estimator reproduction:',len(rows),'events')
