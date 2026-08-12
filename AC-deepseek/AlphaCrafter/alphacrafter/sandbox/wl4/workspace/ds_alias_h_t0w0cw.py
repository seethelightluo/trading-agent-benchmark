import json
d = json.load(open('factors/vol_adj_mom_accel_20x60.json'))
# print keys and validation section only (file may be big due to signal artifacts)
def prune(o, depth=0):
    if isinstance(o, dict):
        return {k: prune(v, depth+1) for k,v in list(o.items())[:40]}
    if isinstance(o, list):
        return o[:5]
    return o
print(json.dumps(prune(d), indent=1)[:3000])
print("TOP KEYS:", list(d.keys()))
print("VALIDATION:", json.dumps(d.get('validation',{}), indent=1)[:2500])