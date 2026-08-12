import pathlib, json
# check last audit entries for gate info
lines = pathlib.Path('factor_library_audit.jsonl').read_text().splitlines()
print("audit lines:", len(lines))
for l in lines[-6:]:
    try:
        d = json.loads(l)
        print(json.dumps(d)[:400])
    except: print(l[:200])
