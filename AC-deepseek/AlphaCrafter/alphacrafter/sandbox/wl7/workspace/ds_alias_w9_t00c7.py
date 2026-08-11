import json
d = json.load(open("factors/mom_10d_skip5.json"))
print(json.dumps(d, indent=1)[:2500])
print("...")
print("keys:", list(d.keys()))
for k,v in d.items():
    if isinstance(v, (str,int,float)):
        print(" ", k, "=", v)
