import json
lines = open('factor_library_audit.jsonl').read().strip().splitlines()
print("lines:", len(lines))
for ln in lines[-3:]:
    try:
        d = json.loads(ln)
        print(json.dumps(d, indent=1)[:800])
    except Exception as e:
        print("ERR", e)