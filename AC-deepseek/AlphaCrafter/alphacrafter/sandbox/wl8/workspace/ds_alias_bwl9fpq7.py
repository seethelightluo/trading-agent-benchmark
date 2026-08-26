import json
d = json.load(open('../persistent/date.json'))
print({k: d[k] for k in ['current_date','visible_through','online_start'] if k in d})
print(list(d.keys())[:20])