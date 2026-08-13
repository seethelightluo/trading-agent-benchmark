import json
lines = open('factor_library_audit.jsonl').read().splitlines()
# find entries with details
for line in lines[-20:]:
    try:
        d = json.loads(line)
        print(json.dumps(d)[:400])
        print('---')
    except Exception as e:
        print("ERR", e)
