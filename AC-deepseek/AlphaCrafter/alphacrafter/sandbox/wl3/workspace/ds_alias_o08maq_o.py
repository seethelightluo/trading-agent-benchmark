
import json
acct = json.load(open('../persistent/account.json'))
hist = acct.get('decision_history', [])
print("num decisions:", len(hist))
for d in hist[-3:]:
    print(json.dumps({k: v for k, v in d.items() if k != 'forecast_returns'}, indent=1)[:1500])
