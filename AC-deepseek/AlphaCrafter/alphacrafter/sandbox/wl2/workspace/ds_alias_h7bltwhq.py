import json
d = json.load(open('../persistent/date.json'))
print({k: (v if not isinstance(v, list) else f"list len {len(v)}") for k, v in d.items()})
td = d['trading_days']
print('last 3 trading days:', td[-3:])
print('visible_through idx:', td.index(d['visible_through']))
