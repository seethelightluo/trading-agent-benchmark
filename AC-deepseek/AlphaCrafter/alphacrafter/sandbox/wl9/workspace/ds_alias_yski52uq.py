import json
d=json.load(open('factors/beta_VIX_60.json'))
v=d['validation']
print(list(v.keys()))
print(json.dumps(v,indent=1)[:1200])