import json
d = json.load(open('../persistent/date.json'))
print('visible_through:', d.get('visible_through'))
td = d.get('trading_days', [])
print('n_trading_days:', len(td))
print('first:', td[:3])
print('last5:', td[-5:])
print('sim_current:', d.get('current_date'))
