import json
lines=[json.loads(l) for l in open('factor_library_audit.jsonl')]
# find cycles where kept>0 recently (last 30)
for i in range(len(lines)-1, -1, -1):
    if lines[i].get('kept',0)>0 and lines[i].get('cycle',0)>=130:
        print('cycle', lines[i].get('cycle'), json.dumps(lines[i], indent=1)[:1800])
        print('---')