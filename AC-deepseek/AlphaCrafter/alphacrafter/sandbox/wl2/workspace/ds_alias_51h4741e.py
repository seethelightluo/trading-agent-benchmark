import json
d = json.load(open('../persistent/date.json'))
print('keys:', list(d.keys()))
print('visible_through:', d.get('visible_through'))
td = d.get('trading_days', [])
print('n trading days:', len(td))
print('last 5:', td[-5:])
print('idx 2028-06-01:', td.index('2028-06-01') if '2028-06-01' in td else 'not found')
