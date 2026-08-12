import json
with open('factors/factor_ensemble.json') as f:
    ens = json.load(f)
print(type(ens))
if isinstance(ens, dict):
    for k,v in ens.items():
        if k in ('selected_factors','factors','metadata','updated','created','last_updated'):
            print(k, ':', str(v)[:500])
elif isinstance(ens, list):
    print('list len', len(ens))
    for x in ens[:3]:
        print(str(x)[:400])