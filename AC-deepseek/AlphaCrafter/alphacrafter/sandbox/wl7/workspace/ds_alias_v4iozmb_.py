import json, os
for p in ["../persistent/date.json","../persistent/account.json","date.json","factor_ensemble.json"]:
    if os.path.exists(p):
        print("==",p,"==")
        d=json.load(open(p))
        if isinstance(d,dict):
            for k in list(d.keys())[:20]:
                print(" ",k,":",str(d[k])[:200])
        else:
            print(d)
    else:
        print("MISSING",p)