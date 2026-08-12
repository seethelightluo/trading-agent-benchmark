import json
with open('factor_library_audit.jsonl') as f:
    lines = f.readlines()
# show cycles 90-99
for l in lines[-10:]:
    print(l.strip()[:400])
