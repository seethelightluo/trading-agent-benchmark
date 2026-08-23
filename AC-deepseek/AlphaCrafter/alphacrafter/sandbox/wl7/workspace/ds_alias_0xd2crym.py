import json
d=json.load(open('../persistent/date.json'))
print(type(d))
print(str(d)[:1500])