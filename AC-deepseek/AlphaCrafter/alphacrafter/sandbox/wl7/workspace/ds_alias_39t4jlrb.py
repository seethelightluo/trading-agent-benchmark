import json
d = json.load(open('factors/corr_ew_60.json'))
print('keys:', list(d.keys()))
print('artifact_provenance:', json.dumps(d.get('artifact_provenance'), indent=1)[:500])
print('benchmark_admission:', json.dumps(d.get('benchmark_admission'), indent=1)[:800])
v = d.get('validation', {})
print('signal_artifact type:', type(v.get('signal_artifact')))
sa = v.get('signal_artifact')
if isinstance(sa, dict):
    print('signal_artifact keys:', list(sa.keys()))
    for k in sa:
        val = sa[k]
        print(k, type(val), (str(val)[:100] if not isinstance(val, (dict, list)) else (len(val) if isinstance(val, list) else '')))
elif isinstance(sa, str):
    print('signal_artifact string len:', len(sa), sa[:120])
