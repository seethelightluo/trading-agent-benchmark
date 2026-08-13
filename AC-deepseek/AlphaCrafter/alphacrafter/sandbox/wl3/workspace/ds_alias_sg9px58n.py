import json
# Audit file structure
with open('factor_library_audit.jsonl') as f:
    aud = [json.loads(x) for x in f.readlines() if x.strip()]
print("audit entries:", len(aud))
print(json.dumps(aud[-1], default=str)[:1500])
print()
# List factor ids from audit
ids = sorted(set(a.get('factor_id') for a in aud))
print("factor ids:", ids)