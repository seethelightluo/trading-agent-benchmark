import json
print("date.json:", open('../persistent/date.json').read())
acc = json.load(open('../persistent/account.json'))
print("account keys:", list(acc.keys()))
print("total_assets:", acc.get('total_assets'), "net:", acc.get('net_assets'), "cash:", acc.get('available_cash'))
