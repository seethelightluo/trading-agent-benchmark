import json
d = json.load(open('factors/skew_20d.json'))
sa = d['validation']['signal_artifact']
print(type(sa), str(sa)[:200])
print("---")
print("metrics:", d['validation']['metrics'])
print("---")
# check other keys
print(list(d.keys()))