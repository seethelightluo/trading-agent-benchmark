import json
d = json.load(open('factors/max_consec_gain_20.json'))
print(json.dumps(d, indent=1)[:3000])
