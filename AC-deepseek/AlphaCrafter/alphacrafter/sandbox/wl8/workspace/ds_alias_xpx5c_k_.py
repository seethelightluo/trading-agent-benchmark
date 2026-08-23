import json
lines=[json.loads(l) for l in open('factor_library_audit.jsonl')]
# find last cycle with kept>0
for i in range(len(lines)-1, -1, -1):
    if lines[i].get('kept',0)>0:
        print('last kept>0 at index', i, 'cycle', lines[i].get('cycle'))
        print(json.dumps(lines[i], indent=1)[:2500])
        break
else:
    print('no cycle ever kept>0')
# summary of kept/rejected/evicted over time
from collections import Counter
for k in ['kept','rejected','evicted','quarantined']:
    c=Counter()
    for l in lines:
        v=l.get(k)
        if isinstance(v,list):
            c[len(v)]+=1
        elif isinstance(v,int):
            c[v]+=1
    print(k, dict(c))