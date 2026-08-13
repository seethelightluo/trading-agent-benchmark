import json
d=json.load(open('../persistent/date.json'))
print('date.json:', d)
a=json.load(open('../persistent/account.json'))
print('account keys:', list(a.keys())[:20])
for k in ['cash','net_liquidation','total_pnl','current_date','as_of','date']:
    if k in a: print(k, a[k])
