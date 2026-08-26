import json
d=json.load(open('factors/kaufman_eff_20d.json'))
print(type(d))
if isinstance(d,dict):
    for k,v in d.items():
        if k in ('validation','calculation','parameters','dependencies','tags','factor_id','factor_name','version'):
            print(k, json.dumps(v)[:800])
else:
    print(d[0].keys() if isinstance(d[0],dict) else type(d[0]))
    for k,v in d[0].items():
        if k in ('validation','calculation','parameters','factor_id','factor_name'):
            print(k, json.dumps(v)[:600])