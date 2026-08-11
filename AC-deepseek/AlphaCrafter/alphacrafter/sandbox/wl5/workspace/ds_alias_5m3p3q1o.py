import json
d = json.load(open('factors/kurt_20.json'))
print(json.dumps({k: (v if k != 'validation' else {kk: (vv if kk != 'signal_artifact' else 'ARTIFACT_PRESENT') for kk, vv in v.items()}) for k, v in d.items()}, indent=1)[:3000])
print("---trend_r2---")
d2 = json.load(open('factors/trend_r2_30_signed.json'))
print(json.dumps({k: (v if k != 'validation' else {kk: (vv if kk != 'signal_artifact' else 'ARTIFACT_PRESENT') for kk, vv in v.items()}) for k, v in d2.items()}, indent=1)[:3000])
