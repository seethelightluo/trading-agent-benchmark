import json
print(json.load(open('../persistent/date.json')))
acc = json.load(open('../persistent/account.json'))
print(json.dumps({k: (v if not isinstance(v,(dict,list)) else type(v).__name__) for k,v in acc.items()}, indent=1))
