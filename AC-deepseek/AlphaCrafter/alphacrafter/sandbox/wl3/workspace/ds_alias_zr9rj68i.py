import os, json
# Look at recent memory entries related to miner
lines = open('memory.txt').read().splitlines()
print("Total lines:", len(lines))
for l in lines[-25:]:
    print(l[:300])
print()
print("=== FACTOR LIBRARY AUDIT (last 5) ===")
with open('factor_library_audit.jsonl') as f:
    aud = [json.loads(x) for x in f.readlines() if x.strip()]
for a in aud[-5:]:
    print(json.dumps(a, default=str)[:400])