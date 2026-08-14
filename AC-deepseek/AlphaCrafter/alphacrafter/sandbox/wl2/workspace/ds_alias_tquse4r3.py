import json, os
# Look at audit for most recent miner entries
with open('factor_library_audit.jsonl') as f:
    lines = f.readlines()
print("audit lines:", len(lines))
for ln in lines[-5:]:
    try:
        d = json.loads(ln)
        print(json.dumps(d)[:400])
        print("---")
    except Exception as e:
        print("ERR", e)
