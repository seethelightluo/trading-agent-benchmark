import json
d = json.load(open("factors/quarantine/mom_10d_skip5.json.reason.json"))
print(json.dumps(d, indent=1)[:1500])
