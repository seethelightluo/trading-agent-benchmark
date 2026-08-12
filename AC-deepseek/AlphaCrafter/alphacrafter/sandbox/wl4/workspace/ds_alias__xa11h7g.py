import json
with open('factor_library_audit.jsonl') as f:
    lines = f.readlines()
print("total lines:", len(lines))
print("first line:", lines[0].strip()[:500])
print("last line:", lines[-1].strip()[:500])
print()
with open('factors/vol_adj_mom_accel_20x60.json') as f:
    d = json.load(f)
keys = [k for k in d.keys()]
print("factor keys:", keys)
print("id/name/version/last_validated:", d.get('factor_id'), d.get('factor_name'), d.get('version'), d.get('last_validated'))
print("validation status:", d.get('validation',{}).get('status'))
print("metrics:", json.dumps(d.get('validation',{}).get('metrics',{}))[:600])
