import json
d = json.load(open('../persistent/date.json'))
print({k: d[k] if k != 'trading_days' else (len(d[k]), d[k][-3:]) for k in d})
print('visible_through:', d.get('visible_through'))
print('current:', d.get('current_date'))