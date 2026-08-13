import json

acc = json.load(open("../persistent/account.json"))
print("top-level keys:", list(acc.keys()))
for k in acc.keys():
    v = acc[k]
    if isinstance(v, (dict, list)):
        if k in ("positions", "orders", "watch_list"):
            print(k, "len", len(v))
        else:
            print(k, "->", json.dumps(v)[:500])
    else:
        print(k, "=", str(v)[:200])
