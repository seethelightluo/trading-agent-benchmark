import json
print("date.json:", open('../persistent/date.json').read())
acc = json.load(open('../persistent/account.json'))
print("account keys:", list(acc.keys()))
print("net_assets:", acc.get('net_assets'), "date:", acc.get('date') or acc.get('current_date'))