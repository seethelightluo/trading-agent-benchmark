import json
print(json.load(open('../persistent/date.json')))
print(json.load(open('../persistent/account.json')).keys())
acct = json.load(open('../persistent/account.json'))
for k in ['total_assets','net_assets','available_cash','market_value','gross_position_rate']:
    print(k, acct.get(k))
print('positions:', len(acct.get('positions', [])))
for p in acct.get('positions', [])[:20]:
    print(' ', p['symbol'], p['direction'], p['quantity'], p.get('cost_price'), p.get('current_price'))