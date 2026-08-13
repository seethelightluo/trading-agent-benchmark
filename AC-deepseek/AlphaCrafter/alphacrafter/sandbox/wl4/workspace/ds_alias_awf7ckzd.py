import json
d = json.load(open('factors/vol_adj_mom_accel_20x60.json'))
print(json.dumps(d, indent=1)[:4000])