import json
d = json.load(open('factors/mom_10d_skip5.json'))
print(json.dumps(d, indent=1, default=str)[:3500])