
import json
d = json.load(open('factors/trend_r2_30_signed.json'))
print(json.dumps({k: (v if k != 'validation' else {kk: (vv if kk != 'signal_artifact' else 'ARTIFACT<len=%d>' % len(vv.get('data',''))) for kk, vv in v.items()}) for k, v in d.items()}, indent=1)[:3000])
