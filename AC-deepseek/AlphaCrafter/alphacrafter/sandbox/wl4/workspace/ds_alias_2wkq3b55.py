import json
# Look at audit trail details - check what earlier cycles contained (rejected/evicted/quarantined events)
with open('factor_library_audit.jsonl') as f:
    lines = f.readlines()
events = []
for i, l in enumerate(lines):
    d = json.loads(l)
    if d.get('rejected') or d.get('evicted') or d.get('quarantined') or d.get('conflicts'):
        events.append((i+1, d))
print("cycles with events:", len(events))
for i, d in events[-8:]:
    print(i, json.dumps(d)[:300])
