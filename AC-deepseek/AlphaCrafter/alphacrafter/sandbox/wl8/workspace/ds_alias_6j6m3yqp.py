import json
for f in ["factor_ensemble.json","flip_mom_20x10.json","usdcny_beta_60.json"]:
    try:
        d=json.load(open("factors/"+f))
        print("===",f,"===")
        print(json.dumps({k:(v if not isinstance(v,dict) else list(v.keys())) for k,v in d.items()},indent=1)[:1500])
    except Exception as e:
        print(f,"ERR",e)