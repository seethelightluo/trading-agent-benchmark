import json
print(json.load(open('../persistent/date.json')))
acc = json.load(open('../persistent/account.json'))
print(json.dumps({k: acc[k] for k in list(acc.keys())[:20]}, indent=1)[:2000])
