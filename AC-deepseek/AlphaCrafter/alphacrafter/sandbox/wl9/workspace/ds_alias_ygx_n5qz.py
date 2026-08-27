import json
d=json.load(open('factors/beta_VIX_60.json'))
print(type(d))
print(str(d)[:800] if isinstance(d,dict) else str(d)[:800])