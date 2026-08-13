import json
lines = open('factor_library_audit.jsonl').read().strip().split('\n')
print('audit lines:', len(lines))
for l in lines[-5:]:
    try:
        d = json.loads(l)
        print(json.dumps(d, indent=1)[:800])
        print('---')
    except Exception as e:
        print('ERR', e)