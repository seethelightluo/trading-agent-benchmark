import json
lines = open('factor_library_audit.jsonl').read().splitlines()
print("total lines:", len(lines))
for line in lines[-15:]:
    try:
        d = json.loads(line)
        keys = list(d.keys())
        print({k: str(d[k])[:80] for k in keys[:8]})
    except Exception as e:
        print("ERR", e, line[:200])
