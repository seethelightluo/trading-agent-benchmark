import json
d=json.load(open('../persistent/date.json'))
print({k: d[k] for k in ['current_date','visible_through','simulation_complete']})
td=d['trading_days']
print('trading_days count', len(td), 'first', td[0], 'last', td[-1])
print('days after 2033-08-31:', [x for x in td if x>'2033-08-31'][:10])