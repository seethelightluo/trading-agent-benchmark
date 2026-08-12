import json
d = json.load(open('../persistent/date.json'))
print(json.dumps(d, indent=1)[:800])
print('trading_days count:', len(d.get('trading_days', [])))
print('first:', d.get('trading_days', [])[:3])
print('last:', d.get('trading_days', [])[-3:])
